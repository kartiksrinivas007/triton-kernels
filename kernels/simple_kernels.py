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


@triton.jit
def add_along_first_axis_kernel(
    x_ptr,
    z_ptr,
    x_size_m,
    x_size_t,
    x_stride_m,
    x_stride_t: tl.constexpr,
    z_stride_m,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    offset_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # need to define an accumulator here
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for offset_t_base in tl.static_range(0, x_size_t, BLOCK_SIZE_T):  # type: ignore
        offset_t = offset_t_base + tl.arange(0, BLOCK_SIZE_T)

        # the net offsets to load the x block are now known
        mask_x_inside = (offset_m[:, None] < x_size_m) & (offset_t[None, :] < x_size_t)
        x = tl.load(
            x_ptr + offset_m[:, None] * x_stride_m + offset_t[None, :] * x_stride_t,
            mask=mask_x_inside,
        )
        accumulator += tl.sum(x, axis=1)  # sum over the other axis

    mask_z_inside = (
        offset_m < x_size_m
    )  # the size of x and z match, and the offsets are the same
    # get the locations for z
    z_ptrs_to_store = z_ptr + offset_m * z_stride_m
    tl.store(z_ptrs_to_store, accumulator, mask=mask_z_inside)
    pass


# Scalar flash attention kernel


@triton.jit
def conv2d_kernel(
    x_ptr,
    k_ptr,
    z_ptr,
    # sizes
    x_size_b,
    x_size_h: tl.constexpr,
    x_size_w: tl.constexpr,
    k_size_h: tl.constexpr,
    k_size_w: tl.constexpr,
    z_size_h: tl.constexpr,
    z_size_w: tl.constexpr,
    # strides
    x_stride_b,
    x_stride_h,
    x_stride_w,
    k_stride_h,
    k_stride_w,
    z_stride_b,
    z_stride_h,
    z_stride_w,
    # tunable params (block sizes)
    BLOCK_SIZE: tl.constexpr,
):

    DEBUG = False

    pid = tl.program_id(axis=0)
    offset_b = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_b = offset_b < x_size_b

    offset_k_h = tl.arange(0, k_size_h)
    offset_k_w = tl.arange(0, k_size_w)
    k_ptrs = k_ptr + offset_k_h[:, None] * k_stride_h + offset_k_w[None, :] * k_stride_w
    k_full = tl.load(k_ptrs)
    # we do not need masks here because we know the full thing sits in memory, this means that the load
    # that is done must be a multiple of a power of 2, else this might cause issues

    for offset_base_h in tl.static_range(0, z_size_h):  # type: ignore
        for offset_base_w in tl.static_range(0, z_size_w):  # type: ignore

            offsets_x_h = offset_base_h + tl.arange(0, k_size_h)
            mask_x_h = offsets_x_h < x_size_h
            offsets_x_w = offset_base_w + tl.arange(0, k_size_w)
            mask_x_w = offsets_x_w < x_size_w

            # load these offsets into memory
            x_ptrs = (
                x_ptr
                + offset_b[:, None, None] * x_stride_b
                + offsets_x_h[None, :, None] * x_stride_h
                + offsets_x_w[None, None, :] * x_stride_w
            )
            mask_x_block = (
                mask_b[:, None, None]
                & mask_x_h[None, :, None]
                & mask_x_w[None, None, :]
            )
            x_block = tl.load(x_ptrs, mask_x_block)

            # broadcast and multiply the block with k
            xk = x_block * k_full[None, :, :]
            xk_sum = tl.sum(xk, axis=1)  # multi axis sum is not supported
            xk_sum = tl.sum(xk_sum, axis=1)

            # then store these in the appropriate location of z (single location per batch)
            # the other 2 dims are constants
            z_ptr_to_store = (
                z_ptr
                + (offset_b * z_stride_b)
                + (offset_base_h * z_stride_h)
                + (offset_base_w * z_stride_w)
            )

            # only need validity for something that could potentially go outside, the last two are true by definition
            mask_z_is_valid = offset_b < x_size_b

            tl.store(z_ptr_to_store, xk_sum, mask_z_is_valid)

            if pid == 1 and offset_base_h == 0 and offset_base_w == 0 and DEBUG:
                tl.device_print("X block = ", x_block.shape[0])
                tl.device_print("pid =  ", pid)
            pass
    pass
