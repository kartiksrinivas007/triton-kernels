import torch 
import triton 
import math 
import triton.language as tl
from packaging import version

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")


if TRITON3:
    @triton.jit
    def softplus(dt):
        return tl.math.log(tl.math.exp(dt) + 1)
else:
    @triton.jit
    def softplus(dt):
        return tl.math.log1p(tl.exp(dt)) # type:ignore

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 1}),
        triton.Config({'BLOCK_SIZE_B': 2}),
        triton.Config({'BLOCK_SIZE_B': 4}),
        # triton.Config({'BLOCK_SIZE_B': 8}),
        # triton.Config({'BLOCK_SIZE_B': 16}),
        # triton.Config({'BLOCK_SIZE_B': 32}),
        # triton.Config({'BLOCK_SIZE_B': 64}),
    ],
    key=['chunk_size', 'nheads'],
)
@triton.jit
def ema_chunk_cumsum_fwd_kernel(
    # Pointers to matrices
    A_ptr, dA_cumsum_ptr,
    # Matrix dimension
    batch, seqlen, chunk_size,
    stride_A_batch, stride_A_seqlen,
    stride_dA_cs_batch, stride_dA_cs_chunk,  stride_dA_cs_csize,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
):
    # shapes to load
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    # offset along chunk, offset along batch, and offset along seqlen
    # offs_seqlen is necessay because A is shaped (B, L), unlike A_cumsum
    offs_q  = tl.arange(0, BLOCK_SIZE_CHUNK)
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_c = pid_c
    offs_seqlen = pid_c * chunk_size + offs_q

    # (BLOCK_SIZE_BATCH, BLOCK_SIZE_CHUNK)
    A_ptrs = A_ptr + offs_b[:, None] * stride_A_batch + offs_seqlen[None, :] * stride_A_seqlen
    dA_cs_ptrs = dA_cumsum_ptr + offs_b[:, None] * stride_dA_cs_batch + offs_c * stride_dA_cs_chunk + offs_q[None, :] * stride_dA_cs_csize

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    # (BLOCK_SIZE_BATCH, BLOCK_SIZE_CHUNK)
    A_chunk = tl.load(A_ptrs, mask=(offs_b[:, None] < batch) & (offs_q[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    dA_cs = tl.cumsum(A_chunk, axis=1)

    tl.store(dA_cs_ptrs, dA_cs, mask=(offs_b[:, None] < batch) & (offs_q[None, :] < chunk_size))


def ema_chunk_cumsum_fwd(A:torch.Tensor, chunk_size: int):
    batch, seqlen = A.shape
    nchunks = math.ceil(seqlen / chunk_size)

    dA_cumsum = torch.empty(batch, nchunks, chunk_size, device=A.device, dtype=torch.float32)
    grid_chunk_cs = lambda META: (triton.cdiv(batch, META['BLOCK_SIZE_B']), nchunks)
    with torch.cuda.device(A.device.index):
        ema_chunk_cumsum_fwd_kernel[grid_chunk_cs](
            A, dA_cumsum,
            batch, seqlen, chunk_size,
            A.stride(0), A.stride(1),
            dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), 
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return dA_cumsum

