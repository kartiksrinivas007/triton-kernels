import torch
import triton
import triton.language as tl
import numpy as np
from einops import rearrange, repeat

from kernels.ema_kernels.ema_cumsum import ema_chunk_cumsum_fwd
from kernels.ema_kernels.ema_state_fwd import _ema_chunk_state_fwd

from kernels.mamba_kernels.mamba_cumsum import _chunk_cumsum_fwd
from kernels.mamba_kernels.mamba_state_fwd import _chunk_state_fwd

import triton.runtime.driver as driver

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
    MAMBA_CHUNK_SIZE = 64 # the chunking level? 
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
        
        
    def test_mamba_cumsum_kernel(self):
        mamba_cs, mamba_dt_out = _chunk_cumsum_fwd(
            self.dt, self.A, chunk_size=self.MAMBA_CHUNK_SIZE
        )
        # get the outputs
        states = _chunk_state_fwd(
            self.B_m,
            self.X_m,
            mamba_dt_out,
            mamba_cs,
            seq_idx=None,
            states=None,
            states_in_fp32=True,
        )

        states_real = rearrange(states.squeeze(-1), " b c h d -> b c (h d)")

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

        breakpoint()

        assert torch.allclose(states_real, ema_states, atol=1e-3)
        


