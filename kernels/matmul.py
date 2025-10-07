import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A,
    B,
    C,
    M: tl.int32,
    K: tl.int32,
    N: tl.int32,
    # strides
    stride_am: tl.int32,
    stride_ak: tl.int32,
    stride_bk: tl.int32,
    stride_bn: tl.int32,
    stride_cm: tl.int32,
    stride_cn: tl.int32,
    # tunable parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # compute offset per block for A and B
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):

        # the k dimension offset you are taking
        offset_k = k + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs_to_load = A + (
            offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
        )
        b_ptrs_to_load = B + (
            offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn
        )

        a_mask = (offset_m[:, None] < M) & (offset_k[None, :] < K)
        b_mask = (offset_k[:, None] < K) & (offset_n[None, :] < N)

        a_tile = tl.load(a_ptrs_to_load, mask=a_mask, other=0)
        b_tile = tl.load(b_ptrs_to_load, mask=b_mask, other=0)

        accumulator += tl.dot(a_tile, b_tile)

    c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    c_ptrs = C + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator.to(C.dtype.element_ty), mask=c_mask)
