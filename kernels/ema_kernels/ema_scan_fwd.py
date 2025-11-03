import torch 
import triton 
import math 
import triton.language as tl

from packaging import version

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
#     ],
#     key=['chunk_size', 'token_dim', 'IS_CAUSAL'],
# )
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8)
    ],
    key=['chunk_size', 'token_dim', 'IS_CAUSAL'],
)
@triton.jit
def _ema_scan_fwd_kernel(
    # Pointers to matrices
    x_ptr, out_ptr, A_cumsum_ptr, prev_states_ptr,
    # Matrix dimensions
    chunk_size, token_dim,
    batch, seqlen,
    # Strides
    stride_x_batch, stride_x_seqlen,stride_x_token_dim,
    stride_out_batch, stride_out_seqlen, stride_out_token_dim,
    stride_A_cs_batch, stride_A_cs_chunk, stride_A_cs_csize,
    stride_states_batch, stride_states_chunk, stride_states_token_dim, # even the dstate is taken away
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    """Compute internal state and add offset from correct start state
    
    Find base pointer for x, A, prev
    
    Build A_cs_j outside
    Build prev_state_j and its mul with A_cs_j outside (store somewhere)
    Build out block of out
    add to acc
    
    acc = acc
    for k in range(0, csize_limit, block_size_k):
        build A_cs_i 
        Load block of x
        build block of A
        mask out upper half of A!
        mul block of X and A
        add to acc
        pass

    store acc in out 
    """
    pid_bc = tl.program_id(axis=2)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_m = tl.program_id(axis=0) # the A_cs_dim
    pid_n = tl.program_id(axis=1) # token_dim

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) # offset within token_dim
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) # offset within a chunk 
    offs_k = tl.arange(0, BLOCK_SIZE_K) # tiling offset

    # Base pointers
    A_cumsum_ptr += pid_b * stride_A_cs_batch + pid_c * stride_A_cs_chunk
    prev_states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk

    # (BLOCK_SIZE_M, ) # this is the 'i' of the outer product
    A_cs_m = tl.load(A_cumsum_ptr + offs_m * stride_A_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    # (BLOCK_SIZE_N, ) # this is a section of the final state of that chunk, the 'j' of the outer product
    prev_state = tl.load(prev_states_ptr + offs_n * stride_states_token_dim, mask = offs_n < token_dim , other=0.0)
    
    #produce (BLOCK_SIZE_M, BLOCK_SIZE_N) size outer product / mul between the elements
    #TODO(kartiksrinivas): is this better to write as uv^T and use a matmul?
    acc_init = A_cs_m[:, None] * prev_state[None, :] # outer product
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc += acc_init.to(tl.float32)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    # # Assertion `!(srcMmaLayout && dstMmaLayout) && "Unexpected mma -> mm a layout conversion"' failed.
    # # With Triton 2.2.0, this works
    # if IS_TRITON_22 or pid_c > -1:
    #     # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    #     offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    #     C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate)
    #     prev_states_ptrs = prev_states_ptr + (offs_n[None, :] * stride_states_hdim + offs_k_dstate[:, None] * stride_states_dstate)
    #     if not HAS_SEQ_IDX:
    #         scale_m = tl.exp(dA_cs_m)
    #     else:
    #         scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
    #     if BLOCK_SIZE_DSTATE <= 128:
    #         C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate), other=0.0)
    #         prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
    #         prev_states = prev_states.to(C_ptr.dtype.element_ty)
    #         acc = tl.dot(C, prev_states) * scale_m[:, None]
    #     else:
    #         for k in range(0, dstate, BLOCK_SIZE_K):
    #             C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate - k), other=0.0)
    #             # C = (C * scale_m[:, None]).to(C_ptr.dtype.element_ty)
    #             prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
    #             prev_states = prev_states.to(C_ptr.dtype.element_ty)
    #             acc += tl.dot(C, prev_states)
    #             C_ptrs += BLOCK_SIZE_K
    #             prev_states_ptrs += BLOCK_SIZE_K
    #         acc *= scale_m[:, None]
    


    # the offset of x in the seqlen dimension # this needs contiguity in C, Q

    offs_seqlen = pid_c * chunk_size + offs_k[:, None]

    # initial pointers for X and A_cs_k
    x_ptrs = x_ptr + pid_b * stride_x_batch + offs_seqlen * stride_x_seqlen + offs_n[None, :] * stride_x_token_dim
    A_cumsum_ptrs = A_cumsum_ptr + offs_k * stride_A_cs_csize

    offs_seqlen_out = pid_c * chunk_size + offs_m[:, None]
    out_ptrs = out_ptr + pid_b * stride_out_batch + offs_seqlen_out * stride_out_seqlen + offs_n[None, :] * stride_out_token_dim

    # only a lower triangular mutiplication 
    # TODO(kartiksrinivas) : Faster algorithms for lower triangular mul?
    K_MAX = chunk_size_limit if not IS_CAUSAL else min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)

    for k in range(0, K_MAX, BLOCK_SIZE_K):

        # technically speaking I should not be loading past K_MAX and not chunk_size?
        # (BLOCK_SIZE_K, )
        A_cs_k = tl.load(A_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)

        # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        mat = tl.exp(tl.minimum((A_cs_m[:, None] - A_cs_k[None, :]), 0.0))
        # this should not cause a warp divergence
        if IS_CAUSAL:
            mask = offs_m[:, None] >= k + offs_k[None, :] # lower triangular mask of 1's
            mat = tl.where(mask, mat, 0.0) 
        mat = mat.to(x_ptr.dtype.element_ty)

        # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < token_dim), other=0.0)

        #(BLOCK_SIZE_M, BLOCK_SIZE_N)
        acc += tl.dot(mat, x)

        
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        A_cumsum_ptrs += BLOCK_SIZE_K * stride_A_cs_csize


    # if HAS_D:
    #     if D_HAS_HDIM:
    #         D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
    #     else:
    #         D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
    #     x_residual = tl.load(x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
    #                          mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    #     acc += x_residual * D

    # if HAS_Z:
    #     out_x_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
    #     out_x_ptrs = out_x_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :])
    #     tl.store(out_x_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))

    #     z_ptr += pid_b * stride_z_batch + pid_c * chunk_size * stride_z_seqlen + pid_h * stride_z_head
    #     z_ptrs = z_ptr + (stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :])
    #     z = tl.load(z_ptrs, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim), other=0.0).to(tl.float32)
    #     acc *= z * tl.sigmoid(z)

    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < token_dim))

def _ema_scan_fwd(x, A_cumsum, states):
    batch, seqlen, token_dim = x.shape
    _, nchunks, _ = states.shape
    _, _, chunk_size = A_cumsum.shape

    # Allocate output
    out = torch.empty(batch, seqlen, token_dim, device = x.device, dtype=x.dtype)
    # per piece of the chunk, per piece of the token_dim (I will do a matmul)
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), triton.cdiv(token_dim, META['BLOCK_SIZE_N']),
                    batch * nchunks)

    _ema_scan_fwd_kernel[grid](
        x, out, A_cumsum, states,
        chunk_size, token_dim,
        batch, seqlen,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2),
        states.stride(0), states.stride(1), states.stride(2),
        IS_CAUSAL=True,
        IS_TRITON_22=TRITON_22,
    )
    return out