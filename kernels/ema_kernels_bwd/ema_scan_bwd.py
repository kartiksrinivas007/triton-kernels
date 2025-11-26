import torch 
import triton 
import math 
import triton.language as tl
from packaging import version

# OLD CONFIGS from Tri
# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
#     ],
#     key=['hdim', 'dstate', 'chunk_size'],
# )
@triton.autotune(
    configs=[
        # small/medium head dim
        # triton.Config({'BLOCK_SIZE_T': 64,  'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=2),
        # triton.Config({'BLOCK_SIZE_T': 64,  'BLOCK_SIZE_K': 64}, num_stages=2, num_warps=4),
        # triton.Config({'BLOCK_SIZE_T': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        # typical EMA shapes (e.g., head_dim≈512–1024, chunk_size=128)
        triton.Config({'BLOCK_SIZE_T': 16, 'BLOCK_SIZE_K': 16},  num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_T': 128, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_T': 256, 'BLOCK_SIZE_K': 64},  num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_T': 256, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=8),
        # very wide head dim
        # triton.Config({'BLOCK_SIZE_T': 512, 'BLOCK_SIZE_K': 64},  num_stages=4, num_warps=8),
    ],
    key=['token_dim', 'chunk_size'],
)
@triton.jit
def _ema_chunk_scan_bwd_dstates_kernel(
    # Pointers to matrices
    dout_ptr, dprev_states_ptr, dA_cumsum_ptr,
    # Matrix dimensions
    token_dim, chunk_size,
    batch, seqlen, nchunks,
    # Strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_token_dim,
    stride_dprev_states_batch, stride_dprev_states_chunk, stride_dprev_states_token_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_chunk_size,
    # Meta-parameters
    BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_t = tl.program_id(axis=2)

    # offsets for token-dim tile
    offs_t = pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    tl.max_contiguous(offs_t, BLOCK_SIZE_T)

    # each kernel instance handles one (batch, chunk, token-tile)
    chunk_start = pid_c * chunk_size
    chunk_size_limit = min(chunk_size, seqlen - chunk_start)

    # base pointers for this (batch, chunk, token-tile)
    dout_base = (
        dout_ptr
        + pid_b * stride_dout_batch
        + chunk_start * stride_dout_seqlen
        + offs_t[None, :] * stride_dout_token_dim
    )
    dA_cs_base = (
        dA_cumsum_ptr
        + pid_b * stride_dA_cs_batch
        + pid_c * stride_dA_cs_chunk
    )

    # accumulator for dstates: shape [BLOCK_SIZE_T]
    acc = tl.zeros((BLOCK_SIZE_T,), dtype=tl.float32)

    # loop over the chunk dimension in blocks of BLOCK_SIZE_K
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    tl.max_contiguous(offs_k, BLOCK_SIZE_K)
    dA_cs_ptrs = dA_cs_base + offs_k * stride_dA_cs_chunk_size
    dout_ptrs = dout_base + offs_k[:, None] * stride_dout_seqlen

    for k_start in range(0, chunk_size_limit, BLOCK_SIZE_K):
        k_mask = offs_k + k_start < chunk_size_limit

        # load dA_cumsum for this (batch, chunk, k) block -> [K]
        dA_cs = tl.load(
            dA_cs_ptrs,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        scale = tl.exp(dA_cs)  # [K]

        # load dout for this (batch, token_dim, k) block -> [K, T]
        dout = tl.load(
            dout_ptrs,
            mask=(
                (offs_t[None, :] < token_dim)
                & k_mask[:, None]
            ),
            other=0.0,
        ).to(tl.float32)

        # apply weights and sum over k dimension
        weighted = dout * scale[:, None]
        acc += tl.sum(weighted, axis=0)

        # advance pointers along sequence for next k block
        dA_cs_ptrs += BLOCK_SIZE_K * stride_dA_cs_chunk_size
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen

    # write back dstates: [T] tile for this (batch, chunk)
    dprev_states_ptrs = (
        dprev_states_ptr
        + pid_b * stride_dprev_states_batch
        + pid_c * stride_dprev_states_chunk
        + offs_t * stride_dprev_states_token_dim
    )
    tl.store(
        dprev_states_ptrs,
        acc.to(dprev_states_ptr.dtype.element_ty),
        mask=(offs_t < token_dim),
    )




def _ema_chunk_scan_bwd_dstates(dA_cumsum, dout, seq_idx=None, dtype=None):
    batch, seqlen, token_dim = dout.shape
    _, nchunks, chunk_size = dA_cumsum.shape
    assert dA_cumsum.shape == (batch, nchunks, chunk_size)

    dtype = dA_cumsum.dtype if dtype is None else dtype

    # allocate space for the output
    dprev_states = torch.empty(batch, nchunks, token_dim, device=dout.device, dtype=dtype)

    grid_dstates = lambda META: (batch, nchunks, triton.cdiv(token_dim, META['BLOCK_SIZE_T']),)
    with torch.cuda.device(dout.device.index):
        _ema_chunk_scan_bwd_dstates_kernel[grid_dstates](
            dout, dprev_states, dA_cumsum,
            token_dim, chunk_size,
            batch, seqlen, nchunks,
            dout.stride(0), dout.stride(1), dout.stride(2),
            dprev_states.stride(0), dprev_states.stride(1), dprev_states.stride(2),
            dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2),
        )
    return dprev_states
