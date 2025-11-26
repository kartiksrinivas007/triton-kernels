import torch
import triton.runtime.driver as driver

from kernels.ema_kernels_bwd.ema_scan_da import _ema_chunk_scan_bwd_ddAcs_stable


def ema_chunk_scan_cumsum_fwd(x_chunks, A, chunk_size):
    A_cs = torch.cumsum(A, dim=-1)  # (B,C,Q)
    scale_logits = A_cs[..., :, None] - A_cs[..., None, :]  # (B,C,Q,Q)
    scale = torch.exp(torch.minimum(scale_logits, torch.tensor(0.0, device=x_chunks.device)))  # (B,C,Q,Q)
    output = torch.matmul(scale, x_chunks)  # (B, C, Q, T)
    return output   


class TestEmaChunkScanBwddA:
    def setup_class(cls):
        torch.manual_seed(0)
        cls.device = driver.active.get_active_torch_device()  # type: ignore
        cls.B = 2
        cls.CHUNK = 3
        cls.T = 8
        cls.NCHUNKS = 1
        cls.SEQLEN = cls.CHUNK * cls.NCHUNKS

        cls.x = torch.randn(cls.B, cls.SEQLEN, cls.T, device=cls.device, dtype=torch.float32)
        # Interpret dA_cumsum as (A_cs_last - A_cs_m); keep negative so exp(min(...,0)) stays in exp branch
        cls.A = -torch.rand(cls.B, cls.NCHUNKS, cls.CHUNK, device=cls.device, dtype=torch.float32)
        cls.dA_cumsum = torch.cumsum(
            cls.A,
            dim=-1,
        )
        cls.dout = torch.randn(cls.B, cls.SEQLEN, cls.T, device=cls.device, dtype=torch.float32)

    
    def test_grad_matches_autograd_dA(self):
        # Autograd reference

        B, L, T = self.x.shape
        _, C, Q = self.dA_cumsum.shape
        x_chunks = self.x.view(self.B, self.NCHUNKS, self.CHUNK, self.T)  # (B,C,Q,T)
        # scale_logits = self.dA_cumsum[..., -1, None] - self.dA_cumsum  # (B,C,Q)

        A_var = self.A.detach().clone().requires_grad_(True)
        states_ref = ema_chunk_scan_cumsum_fwd(x_chunks, A_var, self.CHUNK)
        loss = (states_ref * self.dout.view(self.B, self.NCHUNKS, self.CHUNK, self.T)).sum()
        loss.backward()
        dA = A_var.grad

        ddA_kernel  = _ema_chunk_scan_bwd_ddAcs_stable(
            self.x,
            self.dA_cumsum,
            self.dout,
        )

        # No shift adjustment needed
        expected = dA
        assert torch.allclose(ddA_kernel, expected, atol=1e-2, rtol=1e-2) # type:ignore

    def test_grad_matches_autograd_dA_multi_chunk(self):
        # Sanity-check multi-chunk accumulation path
        torch.manual_seed(0)
        device = driver.active.get_active_torch_device()  # type: ignore
        B = 2
        CHUNK = 4
        NCHUNKS = 2
        SEQLEN = CHUNK * NCHUNKS
        TOKEN_DIM = 6

        x = torch.randn(B, SEQLEN, TOKEN_DIM, device=device, dtype=torch.float32)
        A = -torch.rand(B, NCHUNKS, CHUNK, device=device, dtype=torch.float32)
        dA_cumsum = torch.cumsum(A, dim=-1)
        dout = torch.randn(B, SEQLEN, TOKEN_DIM, device=device, dtype=torch.float32)

        A_var = A.detach().clone().requires_grad_(True)
        states_ref = ema_chunk_scan_cumsum_fwd(
            x.view(B, NCHUNKS, CHUNK, TOKEN_DIM),
            A_var,
            CHUNK,
        )
        loss = (states_ref * dout.view(B, NCHUNKS, CHUNK, TOKEN_DIM)).sum()
        loss.backward()
        expected = A_var.grad

        ddA_kernel = _ema_chunk_scan_bwd_ddAcs_stable(
            x,
            dA_cumsum,
            dout,
        )

        assert torch.allclose(ddA_kernel, expected, atol=1e-2, rtol=1e-2) # type:ignore
