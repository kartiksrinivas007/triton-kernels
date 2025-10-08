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
            constant=CONSTANT
        )

        triton.testing.assert_close(output, x + CONSTANT)

        pass

    
    def test_axis_one_kernel(self):

        y = self.y
        size = self.size

        real_sum= y.sum(1)
        output = torch.empty_like(real_sum)

        grid_second = lambda meta : (triton.cdiv(size, meta["BLOCK_SIZE_M"]), )
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