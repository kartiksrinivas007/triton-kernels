import numpy as np
import triton
import torch
import jaxtyping
import triton.language as tl

from kernels.simple_kernels import (
    add_constant_kernel,
    add_along_first_axis_kernel,
    conv2d_kernel,
)


if __name__ == "__main__":

    assert torch.cuda.is_available(), "CUDA must be avialble to run triton kernels"

    size = 1000
    x = torch.rand(size, device="cuda")
    y = torch.rand((size, size), device="cuda")

    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)

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

    # grid_second = lambda meta : (triton.cdiv(size, meta["BLOCK_SIZE_M"]), )
    # add_along_first_axis_kernel[grid_second](
    #     y,
    #     output,
    #     size,
    #     tl.constexpr(size),
    #     y.stride(0),
    #     y.stride(1),
    #     output.stride(0),
    #     BLOCK_SIZE_M=tl.constexpr(BLOCK_SIZE),
    #     BLOCK_SIZE_T=tl.constexpr(BLOCK_SIZE),
    # )
    # print(y)
    # print(output)
    # print(output  - torch.sum(y, dim=1))

    B = 200
    H = 10
    W = 10
    KH = 2
    KW = 2
    x = torch.ones((B, H, W), device="cuda")
    k = torch.randn((KH, KW), device="cuda")
    z_real = torch.nn.functional.conv2d(
        x[:, None, ...], k[None, None, ...]
    ).squeeze(1)
    z = torch.zeros_like(z_real)
    print(z.shape)
    grid_third = lambda meta: (triton.cdiv(B, meta["BLOCK_SIZE"]),)
    
    # pass these to the tensor then
    conv2d_kernel[grid_third](
        x,
        k,
        z,
        # sizes
        B,
        tl.constexpr(H),
        tl.constexpr(W),
        tl.constexpr(KH),
        tl.constexpr(KW),
        tl.constexpr(H - KH + 1),
        tl.constexpr(W - KW + 1),
        # strides
        x.stride(0),
        x.stride(1),
        x.stride(2),
        k.stride(0),
        k.stride(1),
        z.stride(0),
        z.stride(1),
        z.stride(2),
        # BLOCKS
        BLOCK_SIZE=tl.constexpr(BLOCK_SIZE),
    )

    print(z == z_real)

    pass
