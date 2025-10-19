import torch
import triton
import triton.language as tl

from kernels.simple_kernels import *


class TestSimpleKernels:

    @classmethod
    def setup_class(cls):
        size = 1000
        cls.size = size
        cls.x = torch.rand(size, device="cuda")
        cls.y = torch.rand((size, size), device="cuda")
        cls.BLOCK_SIZE = 16

        pass

    def test_addition_kernel(self):
        assert torch.cuda.is_available(), "CUDA must be avialble to run triton kernels"

        x = self.x
        BLOCK_SIZE = self.BLOCK_SIZE

        output = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(self.size, meta["BLOCK_SIZE"]),)

        CONSTANT = 5

        add_constant_kernel[grid](
            x,
            output,
            x.shape[0],
            x.stride(0),
            output.stride(0),
            BLOCK_SIZE=tl.constexpr(BLOCK_SIZE),
            constant=CONSTANT,
        )

        triton.testing.assert_close(output, x + CONSTANT)

        pass

    def test_axis_one_kernel(self):

        y = self.y
        size = self.size

        real_sum = y.sum(1)
        output = torch.empty_like(real_sum)

        grid_second = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE_M"]),)
        add_along_first_axis_kernel[grid_second](
            y,
            output,
            size,
            tl.constexpr(size),
            y.stride(0),
            tl.constexpr(y.stride(1)),
            output.stride(0),
            BLOCK_SIZE_M=tl.constexpr(self.BLOCK_SIZE),
            BLOCK_SIZE_T=tl.constexpr(self.BLOCK_SIZE),
        )
        triton.testing.assert_close(output, torch.sum(y, dim=1))
        pass


class TestConv2dKernels:

    # BLOCK_SIZE = 16
    # B = 200
    # H = 10
    # W = 10
    # KH = 2
    # KW = 2
    # x = torch.ones((B, H, W), device="cuda")
    # k = torch.randn((KH, KW), device="cuda")
    # z_real = torch.nn.functional.conv2d(
    #     x[:, None, ...], k[None, None, ...]
    # ).squeeze(1)
    # z = torch.zeros_like(z_real)
    # print(z.shape)
    # grid_third = lambda meta: (triton.cdiv(B, meta["BLOCK_SIZE"]),)

    # # pass these to the tensor then
    # conv2d_kernel[grid_third](
    #     x,
    #     k,
    #     z,
    #     # sizes
    #     B,
    #     tl.constexpr(H),
    #     tl.constexpr(W),
    #     tl.constexpr(KH),
    #     tl.constexpr(KW),
    #     tl.constexpr(H - KH + 1),
    #     tl.constexpr(W - KW + 1),
    #     # strides
    #     x.stride(0),
    #     x.stride(1),
    #     x.stride(2),
    #     k.stride(0),
    #     k.stride(1),
    #     z.stride(0),
    #     z.stride(1),
    #     z.stride(2),
    #     # BLOCKS
    #     BLOCK_SIZE=tl.constexpr(BLOCK_SIZE),
    # )
    pass
