import numpy as np
import triton
import torch
import jaxtyping
import triton.language as tl
import triton.runtime.driver as driver

from kernels.simple_kernels import *
from kernels.flash_attn import *
from kernels.linear_attn import *


def _get_gpu_specifications():

    assert torch.cuda.is_available(), "CUDA must be avialble to run triton kernels"
    DEVICE = driver.active.get_active_torch_device()  # type: ignore

    properties = driver.active.utils.get_device_properties(DEVICE.index)  # type:ignore
    NUM_SM = properties["multiprocessor_count"]
    NUM_REGS = properties["max_num_regs"]
    SIZE_SMEM = properties["max_shared_mem"]
    WARP_SIZE = properties["warpSize"]

    def is_cuda():
        return (
            triton.runtime.driver.active.get_current_target().backend == "cuda" # type:ignore
        )  # type:ignore

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


def print_kernel_usage(
    triton_kernel,
    properties,
    **kwargs,
):
    pass


if __name__ == "__main__":

    DEVICE, properties = _get_gpu_specifications()

    target = triton.runtime.driver.active.get_current_target()
    kernels = {}

    torch.manual_seed(42)

    BATCH_SIZE = 2
    HEADS = 4
    SEQLEN = 1024
    HEAD_DIM = 128
    SCALE = 1 / np.sqrt(HEAD_DIM)
    MUL = 1

    Q = (
        torch.randn(
            (BATCH_SIZE, HEADS, SEQLEN, HEAD_DIM), device=DEVICE, dtype=torch.float32
        )
        * MUL
    )
    K = (
        torch.randn(
            (BATCH_SIZE, HEADS, SEQLEN, HEAD_DIM), device=DEVICE, dtype=torch.float32
        )
        * MUL
    )
    V = (
        torch.randn(
            (BATCH_SIZE, HEADS, SEQLEN, HEAD_DIM), device=DEVICE, dtype=torch.float32
        )
        * MUL
    )
    O = torch.empty_like(Q)

    def simple_attention_forward(Q, K, V, scale, causal=False):
        P = torch.matmul(Q, K.transpose(2, 3)) * scale
        M = torch.tril(torch.ones(SEQLEN, SEQLEN, device=DEVICE))
        if causal:
            P[:, :, M == 0] = float("-inf")
        P = torch.softmax(P.float(), dim=-1)
        O = torch.matmul(P, V)
        return O

    def linear_attention_forward(Q, K, V, scale, causal=False):
        P = torch.matmul(Q, K.transpose(2, 3)) * scale
        M = torch.tril(torch.ones(SEQLEN, SEQLEN, device=DEVICE))
        if causal:
            P[:, :, M == 0] = float("-inf")
        O = torch.matmul(P, V)
        return O

    def flash_attention_kernel_forward(Q, K, V) -> torch.Tensor:
        output = FlashAttention.apply(Q, K, V)
        return output  # type:ignore

    def triton_linear_attention_kernel_forward(Q, K, V) -> torch.Tensor:
        output = LinearAttention.apply(Q, K, V)
        return output  # type: ignore

    print(
        torch.allclose(
            triton_linear_attention_kernel_forward(Q, K, V),
            linear_attention_forward(Q, K, V, SCALE),
            rtol=1e-1,
            atol=1e-1,
        )
    )
    ffw = flash_attention_kernel_forward(Q, K, V)
    ssw = simple_attention_forward(Q, K, V, SCALE)

    print(
        torch.allclose(
            ffw,
            ssw,
            rtol=1e-1,
            atol=1e-1,
        )
    )
    
    bench_configs = []
    for head_dim in [32, 64, 128]:
        for batch_size in [1, 2]:
                for causal in [False]: # causal is not supported yet
                    for mode in ["fwd"]: # only forward is supported
                        bench_configs.append(
                            triton.testing.Benchmark(
                                x_names= ["SEQLEN"],
                                x_vals= [2**i for i in range(7,12)],
                                line_arg ="provider",
                                line_vals = ["triton", "torch"],
                                line_names = ["Triton", "Torch"],
                                styles = [("red", '-'), ("blue", "-")],
                                ylabel="TFLOPS",
                                plot_name=f"attn_{batch_size}_{1}_{head_dim}.bench",
                                args= {
                                    "BATCH":batch_size,
                                    "H": 1,
                                    "HEAD_DIM": head_dim,
                                    "causal": False,
                                    "mode": mode,
                                }
                            )
                        )

    @triton.testing.perf_report(bench_configs)
    def bench_flash_attention(BATCH, H, SEQLEN, HEAD_DIM, causal, mode, provider, device=DEVICE):
        assert mode in ["fwd", "bwd"]
        dtype = torch.float32
        q = torch.randn((BATCH, H, SEQLEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, SEQLEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, SEQLEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        scale = 1/np.sqrt(HEAD_DIM)
        ms = 1.0

        if provider == "torch":
            fn = lambda: simple_attention_forward(q, k, v, scale, causal)
            ms = triton.testing.do_bench(fn)

        if provider == "triton":
            fn = lambda: flash_attention_kernel_forward(q, k, v)
            ms = triton.testing.do_bench(fn)

        flops_per_matmul = 2.0 * BATCH * H * SEQLEN * SEQLEN * HEAD_DIM
        total_flops = 2 * flops_per_matmul
        if causal:
            total_flops *= 0.5
        if mode == "bwd":
            total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
        return total_flops * 1e-12 / (ms * 1e-3) # type: ignore

    
    bench_flash_attention.run(save_path=".", print_data=True)
    pass
