import torch
import triton
import triton.testing
import numpy as np
from kernels.ema_kernels.ema_cumsum import ema_chunk_cumsum_fwd
from kernels.ema_kernels_bwd.ema_scan_bwd import _ema_chunk_scan_bwd_dstates

from kernels.mamba_kernels.mamba_cumsum import _chunk_cumsum_fwd
from kernels.mamba_kernels_bwd.mamba_scan_bwd import _chunk_scan_bwd_dstates

from einops import rearrange, repeat

BATCH_SIZE = 16
DTYPE = torch.float32
MAMBA_NUM_HEADS = 2 # 2 heads
MAMBA_CHUNK_SIZE = 128


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
                        line_vals=["ema_torch","mamba"],
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
                            "plot_time": plot_time
                        },
                    )
                )


@triton.testing.perf_report(bench_configs)
def bench_ema(BATCH, SEQLEN, HEAD_DIM, MAMBA_HEAD_DIM, MAMBA_CHUNK_SIZE, provider, device, raw, plot_time):
    dtype = torch.float32

    X = torch.randn((BATCH_SIZE, SEQLEN, HEAD_DIM), dtype=torch.float32, device=device)
    P = torch.rand((BATCH_SIZE, SEQLEN, 1), dtype=torch.float32, device=device)
    Z = torch.empty_like(X)  # same shape as X, but smoothed according to P
    dout = torch.randn_like(X)

    dt = -torch.log(1 - P).to(torch.float32).squeeze(-1)
    X_beta = X / dt[..., None]
    X_m = rearrange(X_beta, "b l (h p) -> b l h p", p=MAMBA_HEAD_DIM)


    dt = repeat(dt, "b l -> b l h", h=MAMBA_NUM_HEADS)
    A = -1 * torch.ones(MAMBA_NUM_HEADS, dtype=torch.float32, device=device)
    B_m = rearrange(P.to(torch.float32), "b l 1 -> b l 1 1")
    C_m = torch.ones_like(B_m)
    dout_m = rearrange(dout, "b l (h p) -> b l h p", p=MAMBA_HEAD_DIM)


    A_ema = torch.log(1 - P).squeeze(-1) # the final dimension
    X_ema = X * P # broadcast
    dout_ema = dout

    NUM_CHUNKS = (SEQLEN + MAMBA_CHUNK_SIZE - 1) // MAMBA_CHUNK_SIZE


    # Steps
    # Recall the kernel with the new data
    # Make new data for the raw kernel call
    # 3. Recompute the data read and the data written 
    # 4. Compute the throughput

    ### sizes
    size_computer = lambda *tensors: sum(t.numel() * t.element_size() for t in tensors)

    plot_name = f"./bench_dump/bwd_raw_{str(raw)}_time_{str(plot_time)}_ema_b{BATCH_SIZE}_chunk_size{mamba_chunk_size}_head_dim{head_dim}_nh{MAMBA_NUM_HEADS}.bench"

    if not raw:
        mamba_data_read = size_computer(C_m, X_m)
        ema_data_read = size_computer(X_m)

        mamba_cs_size = BATCH * NUM_CHUNKS * MAMBA_CHUNK_SIZE * MAMBA_NUM_HEADS * A.element_size()
        ema_cs_size = BATCH * NUM_CHUNKS * MAMBA_CHUNK_SIZE * A.element_size()

        mamba_data_read += mamba_cs_size
        ema_data_read += ema_cs_size
        
        ema_or_mamba_data_out = BATCH * NUM_CHUNKS * HEAD_DIM * 1 * A.element_size()

        # print("Mamba data read = ", mamba_data_read)
        # print("ema data read = ", ema_data_read)
        # print("ema data out = ", ema_or_mamba_data_out)

        bytes_moved_mamba = mamba_data_read + ema_or_mamba_data_out

        def mamba_fn():
            mamba_cs, mamba_dt_out = _chunk_cumsum_fwd(
                dt, A, chunk_size=MAMBA_CHUNK_SIZE
            )

            mamba_dstates = _chunk_scan_bwd_dstates(C_m, mamba_cs, 
                                                            dout_m)

        def ema_fn():
            ema_cs = ema_chunk_cumsum_fwd(
                A_ema, chunk_size=MAMBA_CHUNK_SIZE
            )
            # across heads the computation should be the same
            torch_dstates = torch.sum(torch.mul(rearrange(dout_ema, "b (c q) t -> b c q t", q=MAMBA_CHUNK_SIZE), 
                                                torch.exp(ema_cs[..., None])), dim = 2)

        if provider == "ema_torch":
            ms = triton.testing.do_bench_cudagraph(ema_fn)
            bytes_moved = ema_data_read + ema_or_mamba_data_out if not plot_time else 1e9
        elif provider == "mamba":
            ms = triton.testing.do_bench_cudagraph(mamba_fn)
            bytes_moved = mamba_data_read + ema_or_mamba_data_out if not plot_time else 1e9

        # Rough throughput: bytes processed per second
        gbps = bytes_moved / (ms * 1e-3) / 1e9 # type: ignore 

        print(f"[{provider}] ms={ms}  bytes_moved={bytes_moved} MB  GBps={gbps}")
        return gbps

    else:

        mamba_cs = torch.randn((BATCH, MAMBA_NUM_HEADS, NUM_CHUNKS, MAMBA_CHUNK_SIZE), device=device, dtype=torch.float32)
        # mamba_dt_out = torch.randn_like(mamba_cs)
        ema_cs = torch.randn((BATCH, NUM_CHUNKS, MAMBA_CHUNK_SIZE), device=device, dtype=torch.float32)

        # ema_args = A_ema, ema_cs
        # mamba_args = B_m, X_m, mamba_dt_out, mamba_cs


        # mamba_data_read = size_computer(*mamba_args)
        # ema_data_read  = size_computer(*ema_args)

        # ema_or_mamba_data_out = BATCH * NUM_CHUNKS * HEAD_DIM * 1 * ema_cs.element_size()
        # # same amount of information is written, just reshaped
        # # print("Mamba data read = ", mamba_data_read)
        # # print("ema data read = ", ema_data_read)
        # # print("ema data out = ", ema_or_mamba_data_out)
        mamba_data_read = size_computer(C_m, X_m)
        ema_data_read = size_computer(X_m)

        mamba_cs_size = BATCH * NUM_CHUNKS * MAMBA_CHUNK_SIZE * MAMBA_NUM_HEADS * A.element_size()
        ema_cs_size = BATCH * NUM_CHUNKS * MAMBA_CHUNK_SIZE * A.element_size()

        mamba_data_read += mamba_cs_size
        ema_data_read += ema_cs_size
        
        ema_or_mamba_data_out = BATCH * NUM_CHUNKS * HEAD_DIM * 1 * A.element_size()


        def mamba_raw_fn():
            mamba_dstates = _chunk_scan_bwd_dstates(C_m, mamba_cs, 
                                                            dout_m)

        def ema_raw_fn():
            torch_dstates = torch.sum(torch.mul(rearrange(dout_ema, "b (c q) t -> b c q t", q=MAMBA_CHUNK_SIZE), 
                                            torch.exp(ema_cs[..., None])), dim = 2)

        if provider == "ema_torch":
            ms = triton.testing.do_bench_cudagraph(ema_raw_fn)
            bytes_moved = ema_data_read + ema_or_mamba_data_out 
        elif provider == "mamba":
            ms = triton.testing.do_bench_cudagraph(mamba_raw_fn)
            bytes_moved = mamba_data_read + ema_or_mamba_data_out 

        # Rough throughput: bytes processed per second
        gbps = bytes_moved / (ms * 1e-3) / 1e9 # type: ignore 
        with open(plot_name, "a+") as f :
            f.write("="* 50 + "BWD_DSTATES" + "="* 50 + "\n")
            f.write(f"[{provider}] ms={ms:.3f}  bytes_moved={bytes_moved/1e6:.3f} MB  GBps={gbps:.6f} \n")
            pass
        return gbps if not plot_time else 1.0/ms #type:ignore

         


if __name__ == "__main__":
    bench_ema.run(print_data=True, save_path="./kernels/benchmarks/ema/state_fwd/state_fwd_outputs/")