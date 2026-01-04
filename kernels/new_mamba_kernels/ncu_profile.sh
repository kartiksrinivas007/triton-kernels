ncu --set full --replay-mode=kernel --target-processes=all --kernel-name-base=demangled --kernel-name "ssd_fwd_kernel" --launch-skip 12 --launch-count 1 --export mamba3_fwd_matmul_nov12_937p.ncu-rep --force-overwrite python3 mamba_ssd_fwd.py


# ncu --import mamba3_fwd_matmul_nov12_937p.ncu-rep --list-pages
