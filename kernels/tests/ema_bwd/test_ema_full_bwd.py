import torch
import triton.runtime.driver as driver
import math
from einops import rearrange

from kernels.ema_kernels_bwd.ema_combined_bwd import _ema_chunk_scan_combined_bwd


def ema_loop(X, P):
    """Simple EMA recurrence used as the autograd reference."""
    B, T, D = X.shape
    Z = torch.zeros_like(X)
    for b in range(B):
        z_prev = torch.zeros(D, device=X.device, dtype=X.dtype)
        for t in range(T):
            p = P[b, t]
            z_prev = (1.0 - p) * z_prev + X[b, t]
            Z[b, t] = z_prev
    return Z


class TestEmaCombinedBwd:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)
        cls.device = driver.active.get_active_torch_device()  # type: ignore

    def test_matches_autograd_single_chunk(self):
        B, T, D = 2, 8, 4
        chunk_size = T // 2
        nchunks = math.ceil(T / chunk_size)

        A = -torch.rand(B, T, device=self.device, dtype=torch.float32)
        A.requires_grad_()  # keep A as a leaf
        X = torch.randn(B, T, D, device=self.device, dtype=torch.float32, requires_grad=True)
        dout = torch.randn_like(X)
        P = 1 - torch.exp(A)

        out = ema_loop(X, P)
        loss = (out * dout).sum()
        loss.backward()

        dx_ref = X.grad
        dA_ref = A.grad
        assert dx_ref is not None and dA_ref is not None

        with torch.no_grad():
            dx_kernel, dA_kernel = _ema_chunk_scan_combined_bwd(
                dout, X.detach(), A.detach(), out.detach(), chunk_size
            )
            dA_kernel_reshaped = rearrange(dA_kernel, 'b c s -> b (c s)' )

        assert dx_kernel.shape == dx_ref.shape
        assert dA_kernel.shape == dA_ref.shape

        breakpoint()
        assert torch.allclose(dx_kernel, dx_ref, atol=1e-2, rtol=1e-2)
        assert torch.allclose(dA_kernel_reshaped, dA_ref, atol=1e-2, rtol=1e-2)
