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
        triton.Config({'BLOCK_SIZE_M': 16}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        # triton.Config({'BLOCK_SIZE_M': 64}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        # triton.Config({'BLOCK_SIZE_M': 128}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        # triton.Config({'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=2, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        # triton.Config({'BLOCK_SIZE_M': 64}, num_stages=4, num_warps=2, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
    ],
    key=['chunk_size', 'token_dim'],
)
@triton.jit
def _ema_chunk_state_bwd_db_kernel(
    # Pointers to matrices
    x_ptr, dstates_ptr, dA_cumsum_ptr, ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, token_dim,
    batch, seqlen,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_token_dim,
    stride_dstates_batch, stride_dstates_chunk, stride_states_token_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_csize,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_csize,
    # Meta-parameters
    HAS_DDA_CS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_c = tl.program_id(axis=2)
    pid_b = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=0) 

    # compute in the same style as mamba_chunk_state_bwd_db, but we do not have a for loop over heads
    # instead we have a tiled matmul between x and dstates, and we use that to compute ddA_cs in the same style
    # we do not need a db, although if needed you can compute it as an intermediate variable, we do nto need dt scaling also
    # the inner product is in token_dim, but token_dim is broken into BLOCK_SIZE_K tiles, so its over BLOCK_SIZE_K
    # Matmul: dstates (1, token_dim) @ x (token_dim, chunk_size) = (1, chunk_size)
    # Tiled: dstates_tile (1, BLOCK_SIZE_K) @ x_tile (BLOCK_SIZE_K, BLOCK_SIZE_M) = (1, BLOCK_SIZE_M)

    # Initialize base pointers
    x_base = x_ptr + pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen
    dstates_base = dstates_ptr + pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk
    dA_cumsum_base = dA_cumsum_ptr + pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk
    if HAS_DDA_CS:
        ddA_cumsum_base = ddA_cumsum_ptr + pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk

    # Offsets for M dimension (chunk_size)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Offsets for K dimension (token_dim tiles)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    
    # Load dA_cumsum values (needed for scaling)
    dA_cs_last = tl.load(dA_cumsum_base + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_base + offs_m * stride_dA_cs_csize
    dA_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)

    # Compute scale factor (no dt scaling needed per comment)
    scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0))

    # Initialize accumulator for ddA_cs: (BLOCK_SIZE_M,)
    # We compute: for each position m, sum over k: dstates[k] * x[m, k] * scale[m]
    ddA_cs_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Tiled matmul: loop over token_dim tiles
    num_k_tiles = tl.cdiv(token_dim, BLOCK_SIZE_K)
    for k_tile in range(num_k_tiles):
        k_start = k_tile * BLOCK_SIZE_K
        k_mask = (k_start + offs_k) < token_dim

        # Load dstates tile: (BLOCK_SIZE_K,)
        dstates_ptrs = dstates_base + (k_start + offs_k) * stride_states_token_dim
        dstates_tile = tl.load(dstates_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Load x tile as (K, M) directly
        x_ptrs = x_base + (
            (k_start + offs_k)[:, None] * stride_x_token_dim
            + offs_m[None, :] * stride_x_seqlen
        )
        x_tile = tl.load(
            x_ptrs,
            mask=k_mask[:, None] & (offs_m[None, :] < chunk_size_limit),
            other=0.0,
        ).to(tl.float32)  # (K, M)

        # Elementwise multiply and reduce over K: result shape (M,)
        db_tile = tl.sum(dstates_tile[:, None] * x_tile, axis=0)
        db_tile = db_tile * scale

        if HAS_DDA_CS:
            ddA_cs_acc += db_tile

    # Atomic add ddA_cs to ddA_cumsum at position m+1 (exclusive reverse cumsum)
    # This is consistent with the mamba kernel pattern. The gradient is computed wrt (dA_cs_last - dA_cs_m),
    # which requires an exclusive reverse cumsum. The contribution from position m is accumulated into
    # position m+1, hence the offset. This matches mamba's: ddA_cumsum_ptrs + stride_ddA_cs_csize.

    #TODO(kartiksrinivas): Need to add to the blog to explain tha off-by-one storage clearly her: Need to add to the blog to explain tha off-by-one storrage clearly here.
    if HAS_DDA_CS:
        ddA_cumsum_ptrs = ddA_cumsum_base + (offs_m + 1) * stride_ddA_cs_csize
        tl.atomic_add(ddA_cumsum_ptrs, ddA_cs_acc, mask=(offs_m < chunk_size_limit - 1))
 



def _ema_chunk_state_bwd_db(x, dA_cumsum, dstates, seq_idx=None, B=None, ngroups=1, raw_scale_gradient=False):
    """
    The additional argument raw_gradient indicates whether you need to compute
    the raw gradient with respect to A_cs or you need the gradient with respect
    to the underlying A_factors.
    """
    batch, seqlen, token_dim = x.shape
    _, nchunks, chunk_size = dA_cumsum.shape
    assert dstates.shape == (batch, nchunks, token_dim)
    ddA_cumsum = torch.empty(batch, nchunks, chunk_size, device=x.device, dtype=torch.float32)

    grid_db = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), batch, nchunks)
    with torch.cuda.device(x.device.index):
        _ema_chunk_state_bwd_db_kernel[grid_db](
            x, dstates, dA_cumsum, ddA_cumsum,
            chunk_size, token_dim,
            batch, seqlen,
            x.stride(0), x.stride(1), x.stride(2),
            dstates.stride(0), dstates.stride(1), dstates.stride(2),
            dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2),
            ddA_cumsum.stride(0), ddA_cumsum.stride(1), ddA_cumsum.stride(2),
            HAS_DDA_CS=ddA_cumsum is not None,
            BLOCK_SIZE_K=16,  # Fixed tile size for token_dim
        )
    if not raw_scale_gradient:
        # The first element of ddA_cumsum is always zero, since that dA_cumsum does not contribute
        # to the state of the chunk.
        # torch.cumsum(ddA_cumsum[..., 1:], dim=-1, out=ddA_cumsum[..., 1:])
        # But it's easier to just do the cumsum for all elements, the result will be the same.
        torch.cumsum(ddA_cumsum, dim=-1, out=ddA_cumsum)
    return ddA_cumsum

