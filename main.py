import numpy as np
import triton
import torch
import jaxtyping
import triton.language as tl

from kernels.add import add_constant_kernel



if __name__ == "__main__":
    
    assert torch.cuda.is_available(), "CUDA must be avialble to run triton kernels"

    size = 1000
    x =  torch.rand(size, device="cuda")
    y =  torch.rand(size, device="cuda")

    output = torch.empty_like(x)
    grid = lambda meta : (triton.cdiv(size, meta["BLOCK_SIZE"]), )

    BLOCK_SIZE = 16
    CONSTANT = 5

    add_constant_kernel[grid](
        x, 
        output, 
        x.shape[0], 
        x.stride(0),
        output.stride(0),
        BLOCK_SIZE=tl.constexpr(BLOCK_SIZE),
        constant=CONSTANT
    )

    print(x)
    print(output)

    pass