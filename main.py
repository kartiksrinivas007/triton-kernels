import numpy as np
import triton
import torch
import jaxtyping
import triton.language as tl
import triton.runtime.driver as driver
from einops import rearrange, repeat

from kernels.simple_kernels import *
from kernels.flash_attn import *
from kernels.linear_attn import *
from kernels.ema import *
from kernels.ema_combined import ema_chunk_scan_combined


from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


# DEVICE = driver.active.get_active_torch_device()  # type: ignore


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


def print_kernel_usage(
    triton_kernel,
    properties,
    **kwargs,
):
    pass


if __name__ == "__main__":

    DEVICE = driver.active.get_active_torch_device()  # type: ignore
    _, properties = _get_gpu_specifications(DEVICE)

    target = triton.runtime.driver.active.get_current_target()
    kernels = {}

    torch.manual_seed(42)

    # define the sequence (Batch, Seqlen, Dimension)
    # Should map to another sequence (Batch, Seqlen, Dimension)
    # Proabaiblity is of the same shape but is scalar

    BATCH_SIZE = 2
    SEQLEN = 512
    HEAD_DIM = 64
    MAMBA_HEAD_DIM = 32
    N_HEADS = 2
    BLOCK_SIZE_M = 8
    NUM_CHUNKS = (SEQLEN + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M


    MODE = "mamba"

    X = torch.randn((BATCH_SIZE, SEQLEN, HEAD_DIM), dtype=torch.float32, device=DEVICE)
    P = torch.rand((BATCH_SIZE, SEQLEN, 1), dtype=torch.float32, device=DEVICE)
    Z = torch.empty_like(X)  # same shape as X, but smoothed according to P

    # X needs to be broken into a bunch of heads and the head_dim

    dt = -torch.log(1 - P).to(torch.float32).squeeze(-1)
    X_beta = X / dt[..., None]
    X_m = rearrange(X_beta, "b l (h p) -> b l h p", p=MAMBA_HEAD_DIM)
    dt = repeat(dt, "b l -> b l h", h=N_HEADS)
    A = -1 * torch.ones(N_HEADS, dtype=torch.float32, device=DEVICE)
    B_m = rearrange(P.to(torch.float32), "b l 1 -> b l 1 1")
    C_m = torch.ones_like(B_m)

    if MODE == "MAMBA":
        mamba_z = mamba_chunk_scan_combined(
            X_m, dt, A, B_m, C_m, chunk_size=BLOCK_SIZE_M, seq_idx=None
        )

        mamba_z = rearrange(mamba_z, "b l h p -> b l (h p)")

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

    simple_z = ema_simple(X, P)
    z_loop = ema_loop(X, P)
    ema_z = ema_scan_combined(X, P)
    ema_combined_z = ema_chunk_scan_combined(
        X_m, dt, A, B_m, C_m, chunk_size=BLOCK_SIZE_M, seq_idx=None
    )
    ema_combined_z = rearrange(ema_combined_z, "b l h p -> b l (h p)")

    print(ema_combined_z - z_loop)
    print("Max diff = ", (ema_combined_z - z_loop).max())
    assert (torch.allclose(ema_combined_z, z_loop, atol=1e-2))
    assert (torch.allclose(ema_z, z_loop, atol=1e-2))
