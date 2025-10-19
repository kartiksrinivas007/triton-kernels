"""
Copyright (c) 2025, Kartik Srinivas
"""

import triton
import triton.language as tl
import torch
import numpy as np
from typing import Optional


configs = [
    triton.Config({"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN})
    for BM in [32, 64, 128]
    for BN in [32, 64, 128]
]


# This filter is static, done before hand (compile-time)
def config_filter(conf):
    BLOCK_M = conf.kwargs["BLOCK_SIZE_M"]
    BLOCK_N = conf.kwargs["BLOCK_SIZE_N"]
    return BLOCK_M * BLOCK_N < 128 * 128


# this filter is dynamic, when stuff is actually passed
def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["seqlen"]
    # Filter out configs where BLOCK_M > the seqlen
    configs = [conf for conf in configs if conf.kwargs.get("BLOCK_SIZE_M", 0) <= N_CTX]
    return configs


@triton.jit
def make_tensor_descriptor(pointer, stride, shape, block_shape) -> tl.tensor_descriptor:
    if isinstance(pointer, tl.tensor_descriptor):
        return pointer
    else:
        return tl.make_tensor_descriptor(
            base=pointer, shape=shape, strides=stride, block_shape=block_shape
        )
    pass


@triton.autotune(
    configs=list(filter(config_filter, configs)),
    key=["HEAD_DIM", "seqlen", "batch_size", "num_heads"],
    prune_configs_by={"block_bigger_than_size_prune": prune_invalid_configs},
)
@triton.jit
def flash_attn_fwd_wrapper_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    scale,
    # strides,
    q_stride_b,
    q_stride_h,
    q_stride_n,
    q_stride_d,
    k_stride_b,
    k_stride_h,
    k_stride_n,
    k_stride_d,
    v_stride_b,
    v_stride_h,
    v_stride_n,
    v_stride_d,
    o_stride_b,
    o_stride_h,
    o_stride_n,
    o_stride_d,
    # SHAPES
    batch_size,
    seqlen,
    num_heads,
    # BLOCK SIZES and COMPILE TIME CONSTANTS
    BLOCK_SIZE_M: tl.constexpr,  # For Q
    BLOCK_SIZE_N: tl.constexpr,  # for K and V
    HEAD_DIM: tl.constexpr,
):

    pid_m = tl.program_id(0)  # which SEQLEN block of Q am I handling
    pid_bh = tl.program_id(1)  # which Batch and Head am I handling
    pid_batch = pid_bh // num_heads
    pid_head = pid_bh % num_heads

    # ---------------------------------------------------------------------
    # Triton > 2.3.0 supports Tensor Descriptor
    # We can use that to load tensors instead of doing offset + stride computation
    # for things WITHIN a block (we still need to identify the block itself)
    # ---------------------------------------------------------------------
    # obtain the offsets of the particular Q block of interest
    # find the ptrs to load using the strides that you have and the pid

    offset_batch = pid_batch
    offset_head = pid_head
    offset_m = pid_m

    q_2d_shape = [batch_size * seqlen * num_heads, HEAD_DIM]
    row_offset_o_block = (
        offset_batch * seqlen * num_heads
        + offset_head * seqlen
        + offset_m * BLOCK_SIZE_M
    )

    # load the Q block
    # NOTE: Tensor Descriptor only supports stride 1 in the last dimension
    # NOTE: This must be row-major else stride_n is not correct and may be diff
    desc_q = make_tensor_descriptor(
        q_ptr,
        stride=[HEAD_DIM, 1],  # strides of that 2d block,
        shape=q_2d_shape,
        block_shape=[BLOCK_SIZE_M, HEAD_DIM],
    )

    # load the O block
    desc_o = make_tensor_descriptor(
        o_ptr,
        stride=[HEAD_DIM, 1],
        shape=q_2d_shape,  # the shapes are the same
        block_shape=[BLOCK_SIZE_M, HEAD_DIM],
    )
    desc_k = make_tensor_descriptor(
        k_ptr,
        stride=[HEAD_DIM, 1],
        shape=q_2d_shape,
        block_shape=[BLOCK_SIZE_N, HEAD_DIM],
    )

    desc_v = make_tensor_descriptor(
        v_ptr,
        stride=[HEAD_DIM, 1],
        shape=q_2d_shape,
        block_shape=[BLOCK_SIZE_N, HEAD_DIM],
    )

    # load Q block
    q_block = desc_q.load([row_offset_o_block, 0])

    # desc_o.store([row_offset_o_block, 0], tl.abs(q_block))

    # -----------------------------------------------------------------
    # Flash Attention Kernel
    # -----------------------------------------------------------------
    # Steps
    # Init accumulator to all zeros shape = shape of O block, same dtype
    # 1. Loop over all values of K and V
    #   a. Load K, V Block, and multiply with Q (Linear Attention Kernel)
    #   b. Add the (QK^T)V to the accumulator
    # 2. Store the accumulated result to O
    flash_attn_inner_kernel(
        q_block,
        desc_k,
        desc_v,
        desc_o,
        scale,
        # offsets
        offset_batch,
        offset_head,
        offset_m,
        # sizes
        batch_size,
        num_heads,
        seqlen,
        HEAD_DIM,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )

    pass


@triton.jit
def flash_attn_inner_kernel(
    q_block,
    desc_k,
    desc_v,
    desc_o,
    scale,
    # offsets
    offset_batch,
    offset_head,
    offset_m,
    # sizes
    batch_size,
    num_heads,
    seqlen,
    HEAD_DIM: tl.constexpr,
    # block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):

    row_offset_o = (
        offset_batch * seqlen * num_heads
        + offset_head * seqlen
        + offset_m * BLOCK_SIZE_M
    )

    LOG2E = 1.4426950489

    acc = tl.zeros((BLOCK_SIZE_M, HEAD_DIM), dtype=tl.float32)
    running_max = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    running_exp_sum_neg_max = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)

    for kv_index in tl.range(0, seqlen, BLOCK_SIZE_N):  # type: ignore

        # The kv_index is the row that needs to be loaded
        row_offset_k = (
            offset_batch * seqlen * num_heads + offset_head * seqlen + kv_index
        )
        k_block = desc_k.load([row_offset_k, 0])
        v_block = desc_v.load([row_offset_k, 0])

        qk_t = tl.dot(q_block, k_block.T) * scale
        max_qk_t = tl.max(qk_t, axis=1, keep_dims=True)
        qk_t = (qk_t - max_qk_t).to(tl.float32)
        present_exp_sum_neg_max = tl.sum(
            tl.exp2(LOG2E * qk_t), axis=1, keep_dims=True
        ).to(tl.float32)

        # update the running variables

        running_max_updated = tl.maximum(max_qk_t, running_max)
        running_exp_sum_neg_max_updated = present_exp_sum_neg_max * tl.exp(
            max_qk_t - running_max_updated
        ) + running_exp_sum_neg_max * tl.exp(running_max - running_max_updated)

        # correct for exp subtraction
        qk_t = qk_t + max_qk_t - running_max_updated
        attn_kernel = (tl.exp2(LOG2E * qk_t) / running_exp_sum_neg_max_updated).to(
            tl.float32
        )
        attn_kernel = tl.dot(attn_kernel, v_block)
        acc = (
            attn_kernel
            + acc
            * tl.exp(running_max - running_max_updated)
            * running_exp_sum_neg_max
            / running_exp_sum_neg_max_updated
        )

        running_max = running_max_updated
        running_exp_sum_neg_max = running_exp_sum_neg_max_updated

    # store acc in appropriate block of o
    desc_o.store([row_offset_o, 0], acc)
    pass


@triton.jit
def flash_attn_bwd_kernel():
    pass


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v):

        # Descriptors need to be allocated in global memory (is this expensive?)
        def alloc_fn(size: int, alignment: int, stream: Optional[int]) -> torch.Tensor:
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        # create variables
        o = torch.empty_like(q)  # make it the same shape as q
        scale = 1 / np.sqrt(q.shape[-1])

        def stride_computer(tensor: torch.Tensor):
            return tuple(tensor.stride(index) for index in range(len(tensor.shape)))

        def grid(META):
            return (
                triton.cdiv(q.shape[2], META["BLOCK_SIZE_M"]),
                q.shape[0] * q.shape[1],
                1,
            )

        ctx.grid = grid

        assert q.is_contiguous()
        assert k.is_contiguous()
        assert v.is_contiguous()
        assert o.is_contiguous()

        flash_attn_fwd_wrapper_kernel[ctx.grid](
            q,
            k,
            v,
            o,
            scale,
            # strides,
            *stride_computer(q),
            *stride_computer(k),
            *stride_computer(v),
            *stride_computer(o),
            # shapes
            batch_size=q.shape[0],
            num_heads=q.shape[1],
            seqlen=q.shape[2],
            HEAD_DIM=q.shape[-1]
            # you need not specify block sizes since that
            # is handled by the triton autotuner
        )
        return o

    @staticmethod
    def backward(ctx, do):
        return None, None, None

    pass
