
import triton 
import torch
import numpy as np
import triton.runtime.driver as driver
from kernels.flash_attn import FlashAttention



DEVICE = driver.active.get_active_torch_device()  # type: ignore


def flash_attention_kernel_forward(Q, K, V) -> torch.Tensor:
    output = FlashAttention.apply(Q, K, V)
    return output  # type:ignore

def simple_attention_forward(Q, K, V, scale, causal=False):
    P = torch.matmul(Q, K.transpose(2, 3)) * scale
    SEQLEN = Q.shape[-2]
    M = torch.tril(torch.ones(SEQLEN, SEQLEN, device=DEVICE))
    if causal:
        P[:, :, M == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1)
    O = torch.matmul(P, V)
    return O


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

if __name__ == "__main__":
    bench_flash_attention.run(save_path="./kernels/benchmarks/flash_attn", print_data=True)

