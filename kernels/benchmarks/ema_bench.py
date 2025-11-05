import torch
import triton
import triton.testing
import numpy as np
from kernels.ema import ema_scan_combined
from kernels.ema_combined import ema_chunk_scan_combined

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from einops import rearrange, repeat
from kernels.ema_kernels.ema_combined_fwd import ema_matmul_scan_combined


MAMBA_NUM_HEADS = 2 # 2 heads
MAMBA_CHUNK_SIZE = 128

def ema_simple(X, P):

    # log space implementation of EMA in torch (otherwise floating point issues and nan)
    alpha_clamped = 1 - P
    log_alpha = torch.log(alpha_clamped)
    logC = torch.cumsum(log_alpha, dim=1)
    invC = torch.exp(-logC)
    weighted = (P * X) * invC
    S = torch.cumsum(weighted, dim=1)
    Z = torch.exp(logC) * S

    return Z

def ema_loop(X, P):
    B, T, D = X.shape
    Z = torch.zeros_like(X)
    for b in range(B):
        z_prev = torch.zeros(D, device=X.device, dtype=X.dtype)
        for t in range(T):
            p = P[b, t, 0]
            x = X[b, t]
            z = (1.0 - p) * z_prev + p * x
            Z[b, t] = z
            z_prev = z
    return Z



bench_configs = []
for head_dim in [1024]:
    for batch_size in [16]:
         for chunk_size in [128]:
                bench_configs.append(
                    triton.testing.Benchmark(
                        x_names=["SEQLEN"],
                        x_vals=[2**i for i in range(16, 17)],  
                        line_arg="provider",
                        line_vals=["triton_matmul", "triton_prefix", "ema_mamba", "mamba"],
                        line_names=["Triton_matmul", "Triton Prefix", "Ema_mamba", "Mamba"],
                        styles=[("purple", "-"),("red", "-"), ("yellow", "-"), ("green", "--")],
                        ylabel="GB/s (approx)",
                        plot_name=f"ema_b{batch_size}_head_dim{head_dim}_nh{MAMBA_NUM_HEADS}_chunk_size{chunk_size}.bench",
                        args={
                            "BATCH": batch_size,
                            "HEAD_DIM": head_dim,
                            "MAMBA_HEAD_DIM": head_dim / MAMBA_NUM_HEADS, # use 2 heads
                            "MAMBA_CHUNK_SIZE": chunk_size,
                            "device": "cuda",
                            
                        },
                    )
                )


@triton.testing.perf_report(bench_configs)
def bench_ema(BATCH, SEQLEN, HEAD_DIM, MAMBA_HEAD_DIM, MAMBA_CHUNK_SIZE, provider, device):
    dtype = torch.float32
    x = torch.randn((BATCH, SEQLEN, HEAD_DIM), dtype=dtype, device=device)
    p = torch.sigmoid(torch.randn((BATCH, SEQLEN, 1), dtype=dtype, device=device))

    if provider == "triton_prefix":
        fn = lambda: ema_scan_combined(x, p)
        ms = triton.testing.do_bench_cudagraph(fn)
    elif provider == "ema_mamba":
        dt = -torch.log(1 - p).to(torch.float32).squeeze(-1)
        X_beta = x / dt[..., None]
        X_m = rearrange(X_beta, "b l (h p) -> b l h p", p=int(MAMBA_HEAD_DIM))
        dt = repeat(dt, "b l -> b l h", h=MAMBA_NUM_HEADS)
        A = -1 * torch.ones(MAMBA_NUM_HEADS, dtype=torch.float32, device=device)
        B_m = rearrange(p.to(torch.float32), "b l 1 -> b l 1 1")
        C_m = torch.ones_like(B_m)
        fn = lambda: ema_chunk_scan_combined(
            X_m, dt, A, B_m, C_m, chunk_size=MAMBA_CHUNK_SIZE, seq_idx=None)
        ms = triton.testing.do_bench_cudagraph(fn)
    elif provider == "mamba":
        dt = -torch.log(1 - p).to(torch.float32).squeeze(-1)
        X_beta = x / dt[..., None]
        X_m = rearrange(X_beta, "b l (h p) -> b l h p", p=int(MAMBA_HEAD_DIM))
        dt = repeat(dt, "b l -> b l h", h=MAMBA_NUM_HEADS)
        A = -1 * torch.ones(MAMBA_NUM_HEADS, dtype=torch.float32, device=device)
        B_m = rearrange(p.to(torch.float32), "b l 1 -> b l 1 1")
        C_m = torch.ones_like(B_m)
        fn = lambda: mamba_chunk_scan_combined(
            X_m, dt, A, B_m, C_m, chunk_size=MAMBA_CHUNK_SIZE, seq_idx=None)
        ms = triton.testing.do_bench(fn)
    elif provider == "triton_matmul":
        fn = lambda : ema_matmul_scan_combined(
            x, p, chunk_size=MAMBA_CHUNK_SIZE
        )
        ms = triton.testing.do_bench_cudagraph(fn)
        pass


    # Rough throughput: bytes processed per second
    bytes_moved = x.numel() * x.element_size()    # read X
    bytes_moved += x.numel() * x.element_size()   # write Z
    bytes_moved += p.numel() * p.element_size()  # read P
    gbps = bytes_moved / (ms * 1e-3) / 1e9 # type: ignore 

    plot_name = f"./bench_dump/raw_{str(True)}_time_{str(True)}_ema_b{BATCH}_chunk_size{chunk_size}_head_dim{head_dim}_nh{MAMBA_NUM_HEADS}.bench"
    with open(plot_name, "a+") as f :
        f.write("\n ========================== FINAL ============================= \n")
        f.write(f"[{provider}] ms={ms:.3f}  bytes_moved={bytes_moved/1e6:.3f} MB  GBps={gbps:.6f} \n")
        pass

    return gbps


if __name__ == "__main__":
    bench_ema.run(print_data=True, save_path="./kernels/benchmarks/outputs_ema_scan")

    
    def print_triton_autotune_configs(kernel, kernel_name="kernel"):
        """Print or dump the best autotune configs for a Triton kernel."""
        if not hasattr(kernel, "autotune"):
            print(f"[{kernel_name}] No autotuner attached.")
            return

        cache = getattr(kernel.autotune, "_cache", None)
        if not cache:
            print(f"[{kernel_name}] No autotune results cached yet (kernel not run?).")
            return

        print(f"\n[{kernel_name}] ---- Optimal Triton autotune configs ----")
        for key, cfg in cache.items():
            print(f"Key: {key}")
            print(f"  num_warps   = {cfg.num_warps}")
            print(f"  num_stages  = {cfg.num_stages}")
            print(f"  kwargs      = {cfg.kwargs}")
            print("----------------------------------------------------")
    
        print_triton_autotune_configs(ema_matmul_scan_combined, "ema_matmul_scan_combined")
        print_triton_autotune_configs(ema_chunk_scan_combined, "ema_chunk_scan_combined")


    
    