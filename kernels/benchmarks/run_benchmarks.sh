
# Do not run this async, CUDA might run out of memory

python -m kernels.benchmarks.ema.cumsum.cumsum_bench
python -m kernels.benchmarks.ema.state_fwd.state_fwd_bench
python -m kernels.benchmarks.ema.state_pass.state_pass_bench
python -m kernels.benchmarks.ema.scan_fwd.scan_fwd_bench


python -m kernels.benchmarks.ema_bench