import math
from packaging import version

import torch
import triton
import triton.language as tl

from einops import rearrange


TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")

def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
    ],
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _chunk_scan_chunk_state_bwd_dx_kernel(
    # Pointers to matrices
    x_ptr, dout_ptr, dA_cumsum_ptr, dstates_ptr, dx_ptr,
    # Matrix dimensions
    chunk_size, token_dim,
    batch, seqlen,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_token_dim,
    stride_dout_batch, stride_dout_seqlen, stride_dout_token_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_csize,
    stride_dstates_batch, stride_dstates_chunk, stride_dstates_token_dim,
    stride_dx_batch, stride_dx_seqlen, stride_dx_token_dim,
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # type:ignore
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr
    ):
    
    pid_bc = tl.program_id(axis=2)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    # num_pid_n = tl.cdiv(token_dim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1) 

    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_dstates_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)

    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    if not HAS_SEQ_IDX:
        # scale = tl.exp(dA_cs_last - dA_cs_m)
        scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0))
    else:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
        # scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(dA_cs_last - dA_cs_m), 0.0)
        scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0)), 0.0)
    # Might be faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    # However, we're getting error with the Triton compiler 2.1.0 for that code path:
    # Unexpected mma -> mma layout conversion
    # Triton 2.2.0 fixes this
    offs_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_dstate[None, :] * stride_b_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_dstates_hdim + offs_dstate[:, None] * stride_dstates_dstate)
    if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates) * scale[:, None]
    else:
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate - k), other=0.0)
            dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_dstates_dstate
        acc *= scale[:, None]

    # x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    # x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    # dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    # dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    # ddt = tl.sum(acc * x, axis=1) * dt_m
    # ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    # tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    dout_ptrs = dout_ptr + (offs_k[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = chunk_size_limit
    K_MIN = pid_m * BLOCK_SIZE_M
    cb_ptrs += K_MIN * stride_cb_csize_k
    dout_ptrs += K_MIN * stride_dout_seqlen
    dA_cumsum_ptrs += K_MIN * stride_dA_cs_csize
    for k in range(K_MIN, K_MAX, BLOCK_SIZE_K):
        k = tl.multiple_of(k, BLOCK_SIZE_K)
        # For some reason setting mask to (offs_m[:, None] < chunk_size_limit) is much slower
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < K_MAX - k), other=0.0)
        dout = tl.load(dout_ptrs, mask=(offs_k[:, None] < K_MAX - k) & (offs_n[None, :] < hdim), other=0.0)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < K_MAX - k, other=0.0).to(tl.float32)
        # cb *= tl.exp(dA_cs_k[None, :] - dA_cs_m[:, None])
        cb *= tl.exp(tl.minimum((dA_cs_k[None, :] - dA_cs_m[:, None]), 0.0))
        # If we don't have the (k + offs_k[None, :] < K_MAX) mask, for indices outside this range,
        # we might have dA_cs_m = 0.0 and dA_cs_k very negative, and tl.exp will return inf.
        # Multiplying with cb, which is 0.0 outside the range, will make the result NaN.
        # This will cause NaN in acc, and hence NaN in dx and ddt.
        mask = (k + offs_k[None, :] >= offs_m[:, None]) & (k + offs_k[None, :] < K_MAX)
        cb = tl.where(mask, cb, 0.0)
        cb = cb.to(dout_ptr.dtype.element_ty)
        acc += tl.dot(cb, dout)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    dx = acc * dt_m[:, None]
    dx_ptr += pid_b * stride_dx_batch + pid_c * chunk_size * stride_dx_seqlen + pid_h * stride_dx_head
    dx_ptrs = dx_ptr + (offs_m[:, None] * stride_dx_seqlen + offs_n[None, :] * stride_dx_hdim)
    if HAS_D:
        dout_res_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
        dout_res = tl.load(dout_res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        dx += dout_res * D
    tl.store(dx_ptrs, dx, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    if HAS_D:
        dD_ptr += pid_b * stride_dD_batch + pid_c * stride_dD_chunk + pid_h * stride_dD_head + pid_m * stride_dD_csize
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim
            dD = tl.sum(dout_res * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
        else:
            dD = tl.sum(dout_res * x)
            tl.store(dD_ptr, dD)
    ddt = tl.sum(acc * x, axis=1)
    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)


def _chunk_scan_chunk_state_bwd_dx(x, dA_cumsum, dout, dstates, D=None, seq_idx=None, dx=None):
    batch, seqlen, token_dim = x.shape
    _, nchunks, chunk_size = dA_cumsum.shape
    assert dstates.shape == (batch, nchunks, token_dim)
    assert dout.shape == x.shape
    if dx is None:
        dx = torch.empty_like(x)
    else:
        assert dx.shape == x.shape

    grid_dx = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), triton.cdiv(token_dim, META['BLOCK_SIZE_N']),
                        batch * nchunks)
    with torch.cuda.device(x.device.index):
        _chunk_scan_chunk_state_bwd_dx_kernel[grid_dx](
            x, dout, dA_cumsum, dstates, dx,
            chunk_size, token_dim,
            batch, seqlen,
            x.stride(0), x.stride(1), x.stride(2),
            dout.stride(0), dout.stride(1), dout.stride(2),
            dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2),
            dstates.stride(0), dstates.stride(1), dstates.stride(2),
            dx.stride(0), dx.stride(1), dx.stride(2),
            IS_TRITON_22=TRITON_22
        )
    # if D is not None:
    #     BLOCK_SIZE_actual = _chunk_scan_chunk_state_bwd_dx_kernel.best_config.kwargs["BLOCK_SIZE_M"]
    #     n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
    #     dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
    #     if D.dim() == 1:
    #         dD = rearrange(dD, "h 1 -> h")
    return dx