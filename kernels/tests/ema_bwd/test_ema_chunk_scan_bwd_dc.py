import torch
import triton.runtime.driver as driver

from kernels.ema_kernels_bwd.ema_chunk_scan_bwd_dc import _ema_chunk_scan_bwd_dC


def ema_chunk_scan_fwd_torch(state_chunks, scale_logits, chunk_size):
    scale = torch.exp(torch.minimum(scale_logits, torch.tensor(0.0, device=state_chunks.device)))  # (B,C,Q)
    # (B, C, Q, 1) @ (B, C, 1, T) -> (B, C, Q, T)
    output = torch.matmul(scale[..., None], state_chunks.unsqueeze(-2)) # (B, C, Q, T)
    # breakpoint()
    return output


class TestEmaChunkScanBwdDc:
    def setup_class(cls):
        torch.manual_seed(0)
        cls.device = driver.active.get_active_torch_device()  # type: ignore
        cls.B = 2
        cls.CHUNK = 4
        cls.TOKEN_DIM = 8
        cls.NCHUNKS = 3
        cls.SEQLEN = cls.CHUNK * cls.NCHUNKS

        cls.prev_states = torch.randn(cls.B, cls.NCHUNKS, cls.TOKEN_DIM, device=cls.device, dtype=torch.float32)
        # keep negative so exp(...) stays <= 1
        # this is also scale
        cls.dA_cumsum = -torch.rand(cls.B, cls.NCHUNKS, cls.CHUNK, device=cls.device, dtype=torch.float32)
        cls.dout = torch.randn(cls.B, cls.SEQLEN, cls.TOKEN_DIM, device=cls.device, dtype=torch.float32)

    def test_matches_torch_reference(self):
        ddA_kernel = _ema_chunk_scan_bwd_dC(
            self.prev_states,
            self.dA_cumsum,
            self.dout,
            seq_idx=None,
            C=None,
            ngroups=1,
        )

        scale_var = self.dA_cumsum.detach().clone().requires_grad_(True)
        output_ref = ema_chunk_scan_fwd_torch(self.prev_states, scale_var, self.CHUNK)
        self.dout = self.dout.view(self.B, self.NCHUNKS, self.CHUNK, self.TOKEN_DIM)
        loss = (output_ref * self.dout).sum() # VJP
        loss.backward()

        dscale = scale_var.grad

        assert torch.allclose(ddA_kernel, dscale, atol=1e-2, rtol=1e-2) # type:ignore
