import torch
import triton
import triton.language as tl
import numpy as np

from einops import rearrange, repeat

from kernels.ema_kernels.ema_cumsum import ema_chunk_cumsum_fwd
from kernels.ema_kernels.ema_state_fwd import _ema_chunk_state_fwd
from kernels.ema_kernels.ema_state_pass import _ema_state_passing_fwd
from kernels.ema_kernels.ema_scan_fwd import _ema_scan_fwd
from kernels.ema_kernels_bwd.ema_scan_bwd import _ema_chunk_scan_bwd_dstates
from kernels.ema_kernels_bwd.ema_state_passing_bwd_dstates import _ema_state_passing_bwd
from kernels.ema_kernels_bwd.ema_chunk_scan_chunk_state_bwd_dx import (
    _chunk_scan_chunk_state_bwd_dx as _ema_chunk_scan_chunk_state_bwd_dx,
)

import triton.runtime.driver as driver


def ema_loop(X, P):
    B, T, D = X.shape
    Z = torch.zeros_like(X)
    for b in range(B):
        z_prev = torch.zeros(D, device=X.device, dtype=X.dtype)
        for t in range(T):
            p = P[b, t, 0]
            x = X[b, t]
            z = (1.0 - p) * z_prev + x
            z_prev = z
            Z[b, t, :] = z
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


class TestEmaChunkScanChunkStateBwdDx:
    BATCH_SIZE = 4
    SEQLEN = 512
    HEAD_DIM = 64
    MAMBA_HEAD_DIM = 32
    N_HEADS = 2
    N_GROUPS = 1
    DSTATE = 1
    MAMBA_CHUNK_SIZE = 64
    NUM_CHUNKS = (SEQLEN + MAMBA_CHUNK_SIZE - 1) // MAMBA_CHUNK_SIZE

    @classmethod
    def setup_class(cls):
        DEVICE = driver.active.get_active_torch_device()  # type: ignore
        _, properties = _get_gpu_specifications(DEVICE)
        target = triton.runtime.driver.active.get_current_target()

        cls.x = torch.randn(
            cls.BATCH_SIZE, cls.SEQLEN, cls.HEAD_DIM, device=DEVICE, dtype=torch.float32
        )
        cls.dout = torch.randn_like(cls.x)
        cls.P = torch.rand(
            cls.BATCH_SIZE, cls.SEQLEN, 1, device=DEVICE, dtype=torch.float32
        ).clamp(1e-3, 1 - 1e-3)

        # EMA forward chain inputs
        cls.x_ema = cls.x
        cls.A_ema = torch.log(1 - cls.P).squeeze(-1)  # (B, L)
        cls.ema_cs = ema_chunk_cumsum_fwd(
            cls.A_ema, chunk_size=cls.MAMBA_CHUNK_SIZE
        )  # (B, C, Q)

        cls.states = _ema_chunk_state_fwd(
            cls.x_ema,
            cls.ema_cs,
            seq_idx=None,
            states=None,
            states_in_fp32=True,
        )

        cls.states_updated, cls.final_states = _ema_state_passing_fwd(
            cls.states,
            cls.ema_cs[..., -1],
            initial_states=None,
            chunk_size=None,
            out_dtype=cls.x_ema.dtype,
        )

    def test_ema_forward_chain_matches_ema_loop(self):
        # Full EMA forward via Triton kernels
        ema_out = _ema_scan_fwd(self.x_ema, self.ema_cs, self.states_updated)

        # Reference EMA forward on original X
        z_ref = ema_loop(self.x, self.P)

        assert torch.allclose(ema_out, z_ref, atol=1e-2, rtol=1e-2)

    def test_ema_chunk_scan_chunk_state_bwd_dx_matches_autograd(self):
        # Reference EMA in pure PyTorch on original X
        x_ref = self.x.detach().clone().requires_grad_(True)
        z_ref = ema_loop(x_ref, self.P)
        loss_ref = (z_ref * self.dout).sum()
        loss_ref.backward()
        dx_ref = x_ref.grad

        # Triton backward chain (EMA kernels) w.r.t. x_ema = X * P
        dstates_scan = _ema_chunk_scan_bwd_dstates(self.ema_cs, self.dout)
        dstates_pass, ddA_chunk, _ = _ema_state_passing_bwd(  # type: ignore
            self.states_updated,
            self.ema_cs[..., -1],
            dstates_scan,
            chunk_size=self.MAMBA_CHUNK_SIZE,
        )

        dx_ema = _ema_chunk_scan_chunk_state_bwd_dx(  # type: ignore
            self.x_ema,
            self.ema_cs,
            self.dout,
            dstates_pass,
            D=None,
            seq_idx=None,
            dx=None,
        )


        assert torch.allclose(dx_ema, dx_ref, atol=1e-2, rtol=1e-2)
