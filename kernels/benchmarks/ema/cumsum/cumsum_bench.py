import torch
import triton
import triton.testing
import numpy as np
from kernels.ema_kernels.ema_cumsum import ema_chunk_cumsum_fwd
from kernels.mamba_kernels.mamba_cumsum import _chunk_cumsum_fwd

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from einops import rearrange, repeat

BATCH_SIZE = 16
MAMBA_NUM_HEADS = 2
DTYPE = torch.float32


bench_configs = []
for mamba_chunk_size in [128]:
    for head_dim in [1024]:
        for raw in [True]:
            for plot_time in [False, True]:
                bench_configs.append(
                    triton.testing.Benchmark(
                        x_names=["SEQLEN"],
                        x_vals=[2**i for i in range(16, 17)],  # 128 -> 2048
                        line_arg="provider",
                        line_vals=["ema","mamba"],
                        line_names=["Ema", "Mamba"],
                        styles=[("red", "-"), ("blue", "--"), ("yellow", "-"), ("green", "--")],
                        ylabel="GB/s (approx)" if not plot_time else "1/time",
                        plot_name=f"raw_{str(raw)}_time_{str(plot_time)}_ema_b{BATCH_SIZE}_chunk_size{mamba_chunk_size}_head_dim{head_dim}_nh{MAMBA_NUM_HEADS}.bench",
                        args={
                            "BATCH": BATCH_SIZE,
                            "HEAD_DIM": head_dim,
                            "MAMBA_HEAD_DIM": head_dim // MAMBA_NUM_HEADS, # use 2 heads
                            "MAMBA_CHUNK_SIZE": mamba_chunk_size,
                            "device": "cuda",
                            "raw": raw,
                        },
                    )
                )


@triton.testing.perf_report(bench_configs)
def bench_ema(BATCH, SEQLEN, HEAD_DIM, MAMBA_HEAD_DIM, MAMBA_CHUNK_SIZE, provider, device, raw):
    dtype = torch.float32

    X = torch.randn((BATCH_SIZE, SEQLEN, HEAD_DIM), dtype=torch.float32, device=device)
    P = torch.rand((BATCH_SIZE, SEQLEN, 1), dtype=torch.float32, device=device)
    Z = torch.empty_like(X)  # same shape as X, but smoothed according to P

    NUM_CHUNKS = (SEQLEN + MAMBA_CHUNK_SIZE - 1) // MAMBA_CHUNK_SIZE

    MAMBA_CS_NUMEL = BATCH_SIZE * MAMBA_NUM_HEADS * MAMBA_CHUNK_SIZE * NUM_CHUNKS
    DT_OUT_NUMEL = MAMBA_CS_NUMEL 
    P_NUMEL = P.numel() 
    EMA_CS_NUMEL = BATCH_SIZE * MAMBA_CHUNK_SIZE * NUM_CHUNKS

    MAMBA_DATA_READ = P_NUMEL * P.element_size()
    EMA_DATA_READ = P_NUMEL * P.element_size()

    MAMBA_DATA_OUT = (MAMBA_CS_NUMEL + DT_OUT_NUMEL) * P.element_size()
    EMA_DATA_OUT = EMA_CS_NUMEL * P.element_size()
    
    plot_name = f"./bench_dump/raw_{str(raw)}_time_{str(plot_time)}_ema_b{BATCH_SIZE}_chunk_size{mamba_chunk_size}_head_dim{head_dim}_nh{MAMBA_NUM_HEADS}.bench"
    

    def mamba_fn():
        dt_mamba = -torch.log(1 - P).to(torch.float32).squeeze(-1)
        dt_mamba = repeat(dt_mamba, "b l -> b l h", h=MAMBA_NUM_HEADS)
        A_mamba = -torch.ones(MAMBA_NUM_HEADS, device=device)
        mamba_cs, mamba_dt_out = _chunk_cumsum_fwd(
            dt_mamba, A_mamba, chunk_size=MAMBA_CHUNK_SIZE
        )


    def ema_fn():
        A_ema = torch.log(1 - P).squeeze(-1) # the final dimension
        ema_cs = ema_chunk_cumsum_fwd(
            A_ema, chunk_size=MAMBA_CHUNK_SIZE
        )

    if raw:
        if provider == "ema":
            A_ema = torch.log(1 - P).squeeze(-1) # the final dimension
            raw_ema_fn = lambda: ema_chunk_cumsum_fwd(
                A_ema, chunk_size=MAMBA_CHUNK_SIZE
            )
            ms = triton.testing.do_bench_cudagraph(raw_ema_fn)
            bytes_moved = EMA_DATA_OUT + EMA_DATA_READ

        elif provider == "mamba":
            dt_mamba = -torch.log(1 - P).to(torch.float32).squeeze(-1)
            dt_mamba = repeat(dt_mamba, "b l -> b l h", h=MAMBA_NUM_HEADS)
            A_mamba = -torch.ones(MAMBA_NUM_HEADS, device=device)
            raw_mamba_fn = lambda : _chunk_cumsum_fwd( 
                dt_mamba, A_mamba, chunk_size=MAMBA_CHUNK_SIZE
            )
            ms = triton.testing.do_bench_cudagraph(raw_mamba_fn)
            bytes_moved = MAMBA_DATA_OUT + MAMBA_DATA_READ 
    else:
        if provider == "ema":
            ms = triton.testing.do_bench_cudagraph(ema_fn)
            bytes_moved = EMA_DATA_OUT + EMA_DATA_READ 
        elif provider == "mamba":
            ms = triton.testing.do_bench_cudagraph(mamba_fn)
            bytes_moved = MAMBA_DATA_OUT + MAMBA_DATA_READ 

    # Rough throughput: bytes processed per second
    gbps = bytes_moved / (ms * 1e-3) / 1e9 # type: ignore 
    with open(plot_name, "a+") as f :
        f.write("="* 50 + "CUMSUM" + "="* 50 + "\n")
        f.write(f"[{provider}] ms={ms:.3f}  bytes_moved={bytes_moved/1e6:.3f} MB  GBps={gbps:.6f} \n")
        pass

    return gbps if not plot_time else 1.0/ms # type:ignore


if __name__ == "__main__":
    bench_ema.run(print_data=True, save_path="./kernels/benchmarks/ema/cumsum/cumsum_outputs/")