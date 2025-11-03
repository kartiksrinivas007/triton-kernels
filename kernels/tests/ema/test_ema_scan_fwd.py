import torch
import triton
import triton.language as tl
import numpy as np
from einops import rearrange, repeat
import pytest

from kernels.ema_kernels.ema_cumsum import ema_chunk_cumsum_fwd
from kernels.ema_kernels.ema_state_fwd import _ema_chunk_state_fwd
from kernels.ema_kernels.ema_state_pass import _ema_state_passing_fwd
from kernels.ema_kernels.ema_scan_fwd import _ema_scan_fwd

from kernels.mamba_kernels.mamba_cumsum import _chunk_cumsum_fwd
from kernels.mamba_kernels.mamba_state_fwd import _chunk_state_fwd
from kernels.mamba_kernels.mamba_state_pass import _state_passing_fwd
from kernels.mamba_kernels.mamba_scan_fwd import _chunk_scan_fwd
from kernels.mamba_kernels.mamba_bmm import _bmm_chunk_fwd

import triton.runtime.driver as driver
import math


def ema_loop(X, P):
    B, T, D = X.shape
    N  = math.ceil(T)
    Z = torch.zeros(B, N, D)
    for b in range(B):
        z_prev = torch.zeros(D, device=X.device, dtype=X.dtype)
        for t in range(T):
            p = P[b, t, 0]
            x = X[b, t]
            z = (1.0 - p) * z_prev + p * x
            z_prev = z
    return Z


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

class TestEmaStateFwdKernels:
    BATCH_SIZE = 8
    SEQLEN = 8192
    TOKEN_DIM = 512
    MAMBA_HEAD_DIM = 32
    N_HEADS = 16
    DTYPE = torch.float32
    MAMBA_CHUNK_SIZE = 128 # the chunking level? 
    NUM_CHUNKS = (SEQLEN + MAMBA_CHUNK_SIZE - 1) // MAMBA_CHUNK_SIZE


    @classmethod
    def setup_class(cls):

        DEVICE = driver.active.get_active_torch_device()  # type: ignore
        _, properties = _get_gpu_specifications(DEVICE)
        target = triton.runtime.driver.active.get_current_target()
        cls.X = torch.randn((cls.BATCH_SIZE, cls.SEQLEN, cls.TOKEN_DIM), dtype=torch.float32, device=DEVICE)
        cls.P = torch.rand((cls.BATCH_SIZE, cls.SEQLEN, 1), dtype=torch.float32, device=DEVICE)
        cls.Z = torch.empty_like(cls.X)  # same shape as X, but smoothed according to P


        # input variables for the mamba kernel
        cls.dt = -torch.log(1 - cls.P).to(torch.float32).squeeze(-1)
        cls.X_beta = cls.X / cls.dt[..., None]
        cls.X_m = rearrange(cls.X_beta, "b l (h p) -> b l h p", p=cls.MAMBA_HEAD_DIM)


        cls.dt = repeat(cls.dt, "b l -> b l h", h=cls.N_HEADS)
        cls.A = -1 * torch.ones(cls.N_HEADS, dtype=torch.float32, device=DEVICE)
        cls.B_m = rearrange(cls.P.to(torch.float32), "b l 1 -> b l 1 1")
        cls.C_m = torch.ones_like(cls.B_m)


        cls.A_ema = torch.log(1 - cls.P).squeeze(-1) # the final dimension
        cls.X_ema = cls.X * cls.P # broadcast

        # modified input variables for the ema kernel
        # Steps 
        # 1. Define the matrices needed for the kernel A and dt
        # 1.2 Call the main kernel and see if the main kernel call works
        # 2. construct the inputs for both the kernels
        # 3. Write ema kernel for ema input
        # 3.5 Test if the ema kernel works in the basic case
        # 4. Write code to benchmark both kernels
        
        
    
    def test_mamba_state_passing_kernel(self):
        mamba_cs, mamba_dt_out = _chunk_cumsum_fwd(
            self.dt, self.A, chunk_size=self.MAMBA_CHUNK_SIZE
        )
        # get the outputs
        mamba_states = _chunk_state_fwd(
            self.B_m,
            self.X_m,
            mamba_dt_out,
            mamba_cs,
            seq_idx=None,
            states=None,
            states_in_fp32=True,
        )

        mamba_states_updated, mamba_final_state = _state_passing_fwd(rearrange(mamba_states, "... p n -> ... (p n)"), mamba_cs[:, :, :, -1],
                                            initial_states= None,
                                            seq_idx=None, chunk_size=None, out_dtype=self.C_m.dtype)

        mamba_states, mamba_final_state = [rearrange(t, "... (p n) -> ... p n", n=1) for t in [mamba_states_updated, mamba_final_state]]

        CB = _bmm_chunk_fwd(self.C_m, self.B_m, self.MAMBA_CHUNK_SIZE, seq_idx=None, output_dtype=torch.float32)
        mamba_out, mamba_out_x = _chunk_scan_fwd(CB, self.X_m, mamba_dt_out, mamba_cs, self.C_m, mamba_states, D=None, z=None, seq_idx=None)

        states_ema_loop = ema_loop(self.X, self.P)

        ema_cs = ema_chunk_cumsum_fwd(
            self.A_ema, chunk_size=self.MAMBA_CHUNK_SIZE
        )
        # across heads the computation should be the same
        ema_states = _ema_chunk_state_fwd(
            self.X_ema,
            ema_cs,
            seq_idx=None,
            states=None,
            states_in_fp32=True
        )

        ema_states_updated, ema_final_state = _ema_state_passing_fwd(
            ema_states, 
            ema_cs[..., -1],
            initial_states=None,
            chunk_size=None,  # not needed strictly speaking for this algo
            out_dtype=self.C_m.dtype
        )

        ema_output = _ema_scan_fwd(self.X_ema, ema_cs, ema_states_updated)

        mamba_test = rearrange(mamba_out,"b s h d -> b s (h d)")

        breakpoint()


        # call state passing
        assert (torch.allclose(mamba_test, ema_output, atol=1e-2))

        



