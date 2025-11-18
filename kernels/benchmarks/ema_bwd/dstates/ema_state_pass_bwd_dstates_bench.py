import torch
import triton
import triton.testing
import numpy as np

from kernels.ema_kernels_bwd.ema_state_passing_bwd_dstates import _ema_state_passing_bwd
from kernels.mamba_kernels_bwd.mamba_state_passing_bwd_dstates import _state_passing_bwd

from einops import rearrange

BATCH_SIZE = 16
DTYPE = torch.float32
MAMBA_NUM_HEADS = 2  # 2 heads
MAMBA_CHUNK_SIZE = 128


bench_configs = []
for mamba_chunk_size in [128]:
    for head_dim in [1024]:
        for raw in [True]:
            for plot_time in [False, True]:
                bench_configs.append(
                    triton.testing.Benchmark(
                        x_names=["SEQLEN"],
                        x_vals=[2**i for i in range(16, 17)],  # 65_536
                        line_arg="provider",
                        line_vals=["ema", "mamba"],
                        line_names=["Ema", "Mamba"],
                        styles=[("red", "-"), ("blue", "--")],
                        ylabel="GB/s (approx)" if not plot_time else "1/time",
                        plot_name=f"bwd_state_pass_raw_{str(raw)}_time_{str(plot_time)}_ema_b{BATCH_SIZE}_chunk_size{mamba_chunk_size}_head_dim{head_dim}_nh{MAMBA_NUM_HEADS}.bench",
                        args={
                            "BATCH": BATCH_SIZE,
                            "HEAD_DIM": head_dim,
                            "MAMBA_HEAD_DIM": head_dim // MAMBA_NUM_HEADS,  # use 2 heads
                            "MAMBA_CHUNK_SIZE": mamba_chunk_size,
                            "device": "cuda",
                            "raw": raw,
                            "plot_time": plot_time,
                        },
                    )
                )


@triton.testing.perf_report(bench_configs)
def bench_ema(
    BATCH,
    SEQLEN,
    HEAD_DIM,
    MAMBA_HEAD_DIM,
    MAMBA_CHUNK_SIZE,
    provider,
    device,
    raw,
    plot_time,
):
    dtype = DTYPE

    # number of chunks implied by sequence length and chunk size
    NUM_CHUNKS = (SEQLEN + MAMBA_CHUNK_SIZE - 1) // MAMBA_CHUNK_SIZE

    # random chunk-level states and upstream gradients for Mamba
    states_mamba = torch.randn(
        (BATCH, NUM_CHUNKS, MAMBA_NUM_HEADS, MAMBA_HEAD_DIM),
        device=device,
        dtype=dtype,
    )
    dout_mamba = torch.randn_like(states_mamba)

    # use the same per-chunk decay across heads so that flattening (h, p) is valid
    dA_ema = torch.randn((BATCH, NUM_CHUNKS), device=device, dtype=dtype)
    dA_mamba = dA_ema[:, None, :].expand(BATCH, MAMBA_NUM_HEADS, NUM_CHUNKS).contiguous()

    # EMA version: flatten (head, dim) into token_dim
    states_ema = rearrange(states_mamba, "b c h p -> b c (h p)")
    dout_ema = rearrange(dout_mamba, "b c h p -> b c (h p)")

    size_computer = lambda *tensors: sum(t.numel() * t.element_size() for t in tensors)

    mamba_data_read = size_computer(states_mamba, dA_mamba, dout_mamba)
    ema_data_read = size_computer(states_ema, dA_ema, dout_ema)

    # both kernels write dstates of size (BATCH, NUM_CHUNKS, HEAD_DIM)
    bytes_out = BATCH * NUM_CHUNKS * HEAD_DIM * states_mamba.element_size()

    plot_name = (
        f"./bench_dump/bwd_state_pass_raw_{str(raw)}_time_{str(plot_time)}"
        f"_ema_b{BATCH_SIZE}_chunk_size{MAMBA_CHUNK_SIZE}_head_dim{HEAD_DIM}_nh{MAMBA_NUM_HEADS}.bench"
    )

    # raw=True: benchmark only the backward state-passing kernels
    def mamba_raw_fn():
        _state_passing_bwd(
            states_mamba,
            dA_mamba,
            dout_mamba,
            dfinal_states=None,
            seq_idx=None,
            has_initial_states=False,
            dstates_dtype=dout_mamba.dtype,
            states_dtype=states_mamba.dtype,
            chunk_size=MAMBA_CHUNK_SIZE,
        )

    def ema_raw_fn():
        _ema_state_passing_bwd(
            states_ema,
            dA_ema,
            dout_ema,
            dfinal_states=None,
            seq_idx=None,
            has_initial_states=False,
            dstates_dtype=dout_ema.dtype,
            states_dtype=states_ema.dtype,
            chunk_size=MAMBA_CHUNK_SIZE,
        )

    if provider == "ema":
        ms = triton.testing.do_bench_cudagraph(ema_raw_fn)
        bytes_moved = ema_data_read + bytes_out
    elif provider == "mamba":
        ms = triton.testing.do_bench_cudagraph(mamba_raw_fn)
        bytes_moved = mamba_data_read + bytes_out
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Rough throughput: bytes processed per second
    gbps = bytes_moved / (ms * 1e-3) / 1e9  # type: ignore

    with open(plot_name, "a+") as f:
        f.write("=" * 50 + "BWD_STATE_PASS" + "=" * 50 + "\n")
        f.write(
            f"[{provider}] ms={ms:.3f}  bytes_moved={bytes_moved/1e6:.3f} MB  GBps={gbps:.6f} \n"
        )

    return gbps if not plot_time else 1.0 / ms  # type: ignore


if __name__ == "__main__":
    bench_ema.run(
        print_data=True,
        save_path="./kernels/benchmarks/ema_bwd/dstates/dstates_outputs/",
    )

