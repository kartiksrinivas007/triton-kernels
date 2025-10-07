import triton
import triton.language as tl


@triton.jit
def add_constant_kernel(
    x_ptr,
    z_ptr,
    x_size,
    x_stride,
    z_stride,
    BLOCK_SIZE: tl.constexpr,
    constant,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x_ptrs_to_load = x_ptr + offset * x_stride
    mask_inside = offset < x_size
    x = tl.load(x_ptrs_to_load, mask=mask_inside)
    op = x + constant

    z_ptrs_to_store = z_ptr + offset * z_stride
    tl.store(z_ptrs_to_store, op, mask=mask_inside)
    pass
