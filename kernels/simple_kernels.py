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

    pid = tl.program_id(axis = 0)
    offset_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # need to define an accumulator here
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for offset_t_base in tl.static_range(0, x_size_t, BLOCK_SIZE_T): #type: ignore
        offset_t = offset_t_base + tl.arange(0, BLOCK_SIZE_T)

        # the net offsets to load the x block are now known
        mask_x_inside = (offset_m[:, None] < x_size_m) & (offset_t[None, :] < x_size_t)
        x = tl.load(x_ptr + offset_m[:, None] * x_stride_m + offset_t[None, :] * x_stride_t, mask=mask_x_inside)
        accumulator += tl.sum(x, axis = 1) # sum over the other axis

    mask_z_inside = offset_m < x_size_m # the size of x and z match, and the offsets are the same 
    # get the locations for z 
    z_ptrs_to_store = z_ptr + offset_m * z_stride_m
    tl.store(z_ptrs_to_store, accumulator, mask=mask_z_inside)
    pass

        

    