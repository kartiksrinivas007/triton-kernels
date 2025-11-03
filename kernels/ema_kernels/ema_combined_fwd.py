import torch



from kernels.ema_kernels.ema_cumsum import ema_chunk_cumsum_fwd
from kernels.ema_kernels.ema_state_fwd import _ema_chunk_state_fwd
from kernels.ema_kernels.ema_state_pass import _ema_state_passing_fwd
from kernels.ema_kernels.ema_scan_fwd import _ema_scan_fwd



def _ema_matmul_scan_combined_fwd(x, P, chunk_size, dtype=torch.float32):
    batch, seqlen, token_dim = x.shape
    assert P.shape == (batch, seqlen, 1)
    #TODO(kartiksrinivas): Need to check this for bugs, what needs to be contiguous exactly
    if P.stride(-1) != 1:
        P = P.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:  # Either M or K dimension should be contiguous
        x = x.contiguous()
    
    A_ema = torch.log(1 - P).squeeze(-1) # the final dimension
    X_ema = x * P # broadcast

    ema_cs = ema_chunk_cumsum_fwd(
        A_ema, chunk_size=chunk_size
    )
    # across heads the computation should be the same
    ema_states = _ema_chunk_state_fwd(
        X_ema,
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
        out_dtype=dtype
    )

    ema_output = _ema_scan_fwd(X_ema, ema_cs, ema_states_updated)

    return ema_output, ema_cs, ema_states_updated, ema_final_state





class EMAMatmulCombinedFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, P, chunk_size):

        out, dA_cumsum, states, final_states = _ema_matmul_scan_combined_fwd(x, P, chunk_size=chunk_size)
        ctx.save_for_backward(out, dA_cumsum, states, final_states, x, P)
        ctx.chunk_size = chunk_size

        return out

    @staticmethod
    def backward(ctx, dout, *args):

        # This is not implemented yet

        out, x, dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx = ctx.saved_tensors
        assert not ctx.return_varlen_states, "return_varlen_states is not supported in backward"
        dfinal_states = args[0] if ctx.return_final_states else None
        # dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = _mamba_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx, dt_softplus=ctx.dt_softplus, dt_limit=ctx.dt_limit)
        dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = (None, ) * 9
        return dx, ddt, dA, dB, dC, None, dD, dz, ddt_bias, dinitial_states, None, None, None, None, None, None



def ema_matmul_scan_combined(x, P, chunk_size):
    return EMAMatmulCombinedFn.apply(x, P, chunk_size)


