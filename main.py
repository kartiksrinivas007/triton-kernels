import numpy as np
import triton
import torch
import jaxtyping
import triton.language as tl

from kernels.simple_kernels import add_constant_kernel, add_along_first_axis_kernel



if __name__ == "__main__":
    
    assert torch.cuda.is_available(), "CUDA must be avialble to run triton kernels"

    size = 1000
    x =  torch.rand(size, device="cuda")
    y =  torch.rand((size, size), device="cuda")

    output = torch.empty_like(x)
    grid = lambda meta : (triton.cdiv(size, meta["BLOCK_SIZE"]), )

    BLOCK_SIZE = 16
    CONSTANT = 5

    # add_constant_kernel[grid](
    #     x, 
    #     output, 
    #     x.shape[0], 
    #     x.stride(0),
    #     output.stride(0),
    #     BLOCK_SIZE=tl.constexpr(BLOCK_SIZE),
    #     constant=CONSTANT
    # )

    # print(x)
    # print(output)


    grid_second = lambda meta : (triton.cdiv(size, meta["BLOCK_SIZE_M"]), )
    add_along_first_axis_kernel[grid_second](
        y, 
        output,
        size, 
        tl.constexpr(size), 
        y.stride(0),
        y.stride(1),
        output.stride(0),
        BLOCK_SIZE_M=tl.constexpr(BLOCK_SIZE),
        BLOCK_SIZE_T=tl.constexpr(BLOCK_SIZE),
    )
    print(y)
    print(output)
    print(output  - torch.sum(y, dim=1))

    pass