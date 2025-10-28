import torch
import triton
import triton.language as tl
import numpy as np
from kernels.simple_kernels import *
from kernels.flash_attn import *
from kernels.linear_attn import *
import triton.runtime.driver as driver


def _get_gpu_specifications(DEVICE):

    assert torch.cuda.is_available(), "CUDA must be avialble to run triton kernels"

    properties = driver.active.utils.get_device_properties(DEVICE.index)  # type:ignore
    NUM_SM = properties["multiprocessor_count"]
    NUM_REGS = properties["max_num_regs"]
    SIZE_SMEM = properties["max_shared_mem"]
    WARP_SIZE = properties["warpSize"]

    def is_cuda():
        return (
            triton.runtime.driver.active.get_current_target().backend  # type: ignore
            == "cuda"
        )

    def supports_host_descriptor():
        return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

    print("=" * 100)
    print("Device = ", DEVICE)
    print("Num REGS per SM  = ", NUM_REGS)
    print("Num SM  = ", NUM_SM)
    print("Total Shared memory (bytes) = ", SIZE_SMEM)
    print("Warp size = ", WARP_SIZE)
    print("Supports Host Descriptor = ", supports_host_descriptor())
    print("=" * 100)

    return DEVICE, properties


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


class TestAttentionKernels:

    @classmethod
    def setup_class(cls):
        DEVICE = driver.active.get_active_torch_device()  # type: ignore
        _, properties = _get_gpu_specifications(DEVICE)
        target = triton.runtime.driver.active.get_current_target()
        kernels = {}
        torch.manual_seed(42)
        BATCH_SIZE = 3
        HEADS = 5
        # TODO(kartiksrinivas): Need to make kernel support for arbitrary seqlen
        SEQLEN = 1024
        HEAD_DIM = 128
        cls.SCALE = 1 / np.sqrt(HEAD_DIM)
        MUL = 1

        cls.Q = (
            torch.randn(
                (BATCH_SIZE, HEADS, SEQLEN, HEAD_DIM),
                device=DEVICE,
                dtype=torch.float32,
            )
            * MUL
        )
        cls.K = (
            torch.randn(
                (BATCH_SIZE, HEADS, SEQLEN, HEAD_DIM),
                device=DEVICE,
                dtype=torch.float32,
            )
            * MUL
        )
        cls.V = (
            torch.randn(
                (BATCH_SIZE, HEADS, SEQLEN, HEAD_DIM),
                device=DEVICE,
                dtype=torch.float32,
            )
            * MUL
        )
        cls.O = torch.empty_like(cls.Q)

    def test_linear_attn_kernel(self):
        def linear_attention_forward(Q, K, V, scale, causal=False):
            P = torch.matmul(Q, K.transpose(2, 3)) * scale
            SEQLEN = Q.shape[-2]
            M = torch.tril(torch.ones(SEQLEN, SEQLEN, device=P.device))
            if causal:
                P[:, :, M == 0] = float("-inf")
            O = torch.matmul(P, V)
            return O

        def triton_linear_attention_kernel_forward(Q, K, V) -> torch.Tensor:
            output = LinearAttention.apply(Q, K, V)
            return output  # type: ignore

        assert torch.allclose(
            triton_linear_attention_kernel_forward(self.Q, self.K, self.V),
            linear_attention_forward(self.Q, self.K, self.V, self.SCALE),
            rtol=1e-1,
            atol=1e-1,
        )

    def test_flash_attn_kernel(self):

        def flash_attention_kernel_forward(Q, K, V) -> torch.Tensor:
            output = FlashAttention.apply(Q, K, V)
            return output  # type:ignore

        def simple_attention_forward(Q, K, V, scale, causal=False):
            P = torch.matmul(Q, K.transpose(2, 3)) * scale
            SEQLEN = Q.shape[-2]
            M = torch.tril(torch.ones(SEQLEN, SEQLEN, device=P.device))
            if causal:
                P[:, :, M == 0] = float("-inf")
            P = torch.softmax(P.float(), dim=-1)
            O = torch.matmul(P, V)
            return O

        ffw = flash_attention_kernel_forward(self.Q, self.K, self.V)
        ssw = simple_attention_forward(self.Q, self.K, self.V, self.SCALE)

        print(
            torch.allclose(
                ffw,
                ssw,
                rtol=1e-1,
                atol=1e-1,
            )
        )
