import math

from packaging import version

import torch
import triton
import triton.language as tl

from einops import rearrange

def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_K': 16}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
    ],
    key=['chunk_size', 'token_dim'],
)
@triton.jit
def _ema_chunk_scan_bwd_dc_kernel(
    dout_ptr, prev_states_ptr, dA_cumsum_ptr, ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, token_dim,
    batch, seqlen, nchunks,
    # Strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_token_dim,
    stride_prev_states_batch, stride_prev_states_chunk, stride_prev_states_token_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_csize,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_csize,
    # Meta-parameters
    HAS_DDA_CS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_c = tl.program_id(axis=2)

    chunk_start = pid_c * chunk_size
    chunk_size_limit = tl.minimum(chunk_size, seqlen - chunk_start)

    dout_base = (
        dout_ptr
        + pid_b * stride_dout_batch
        + chunk_start * stride_dout_seqlen
    )
    prev_states_base = (
        prev_states_ptr
        + pid_b * stride_prev_states_batch
        + pid_c * stride_prev_states_chunk
    )
    dA_cumsum_base = (
        dA_cumsum_ptr
        + pid_b * stride_dA_cs_batch
        + pid_c * stride_dA_cs_chunk
    )
    if HAS_DDA_CS:
        ddA_cumsum_base = (
            ddA_cumsum_ptr
            + pid_b * stride_ddA_cs_batch
            + pid_c * stride_ddA_cs_chunk
        )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # tl.max_contiguous(offs_m, BLOCK_SIZE_M)
    mask_m = offs_m < chunk_size_limit

    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # tl.max_contiguous(offs_k, BLOCK_SIZE_K)
    dout_ptrs = (
        dout_base
        + offs_m[:, None] * stride_dout_seqlen
        + offs_k[None, :] * stride_dout_token_dim
    )
    prev_states_ptrs = prev_states_base + offs_k * stride_prev_states_token_dim
    for k_start in range(0, token_dim, BLOCK_SIZE_K):
        k_mask = (k_start + offs_k) < token_dim

        dout = tl.load(
            dout_ptrs,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        prev_states = tl.load(
            prev_states_ptrs,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        # prev_states = prev_states.to(dout_ptrs.dtype.element_ty)
        acc += tl.sum(dout * prev_states[None, :], axis=1)

        dout_ptrs += BLOCK_SIZE_K * stride_dout_token_dim
        prev_states_ptrs += BLOCK_SIZE_K * stride_prev_states_token_dim

    dA_cumsum_ptrs = dA_cumsum_base + offs_m * stride_dA_cs_csize
    scale = tl.exp(tl.load(dA_cumsum_ptrs, mask=mask_m, other=0.0).to(tl.float32))
    acc = acc * scale

    if HAS_DDA_CS:
        ddA_cumsum_ptrs = ddA_cumsum_base + offs_m * stride_ddA_cs_csize
        tl.store(
            ddA_cumsum_ptrs,
            tl.where(mask_m, acc, 0.0).to(ddA_cumsum_ptr.dtype.element_ty),
            mask=offs_m < chunk_size,
        )



def _ema_chunk_scan_bwd_dC(prev_states, dA_cumsum, dout, seq_idx=None, C=None, ngroups=1):
    batch, nchunks, token_dim = prev_states.shape
    _, seqlen, _ = dout.shape
    _, _, chunk_size = dA_cumsum.shape
    assert prev_states.shape == (batch, nchunks, token_dim)
    assert dA_cumsum.shape == (batch, nchunks, chunk_size)
    assert dout.shape == (batch, seqlen, token_dim)
    ddA_cumsum_prev = torch.empty(batch, nchunks, chunk_size, device=dout.device, dtype=torch.float32)
    ddA_cumsum_prev_strides = (ddA_cumsum_prev.stride(0), ddA_cumsum_prev.stride(1), ddA_cumsum_prev.stride(2))

    # sm_count = torch.cuda.get_device_properties(dout.device).multi_processor_count
    # nheads_per_program = max(min(math.ceil(batch * nchunks * nheads / sm_count), nheads_ngroups_ratio), 1)
    # nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)
    # dC = torch.empty(batch, seqlen, nsplits, ngroups, dstate, device=dout.device, dtype=torch.float32)
    grid_dc = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), batch, nchunks)

    with torch.cuda.device(dout.device.index):
        _ema_chunk_scan_bwd_dc_kernel[grid_dc](
            dout, prev_states, dA_cumsum, ddA_cumsum_prev,
            chunk_size, token_dim,
            batch, seqlen, nchunks,
            dout.stride(0), dout.stride(1), dout.stride(2), 
            prev_states.stride(0), prev_states.stride(1), prev_states.stride(2),
            dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2),
            *ddA_cumsum_prev_strides,
            HAS_DDA_CS=ddA_cumsum_prev is not None,
            # BLOCK_SIZE_K=8,
        )
    return ddA_cumsum_prev
