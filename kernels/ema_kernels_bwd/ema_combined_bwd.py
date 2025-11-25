import torch 
import triton 
import math 
import triton.language as tl
from packaging import version

from einops import rearrange
from kernels.ema_kernels import ema_cumsum, ema_scan_fwd, ema_state_pass, ema_state_fwd
from kernels.ema_kernels_bwd import ema_chunk_scan_bwd_dc, ema_chunk_scan_chunk_state_bwd_dx, ema_chunk_state_bwd_db, ema_scan_bwd, ema_scan_da, ema_state_passing_bwd_dstates



def _ema_chunk_scan_combined_bwd(dout, x, A, out, chunk_size, 
                                   dt_limit=(0.0, float("inf"))):
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch, seqlen, token_dim = x.shape
    nchunks = math.ceil(seqlen / chunk_size)

    assert dout.shape == (batch, seqlen, token_dim)
    assert A.shape == (batch, seqlen)
    assert out.shape == x.shape


    #####################################################################################
    #                                  FORWARD PASS    
    #####################################################################################
    ema_cs = ema_cumsum.ema_chunk_cumsum_fwd(
        A, chunk_size=chunk_size
    )
    # across heads the computation should be the same
    ema_states = ema_state_fwd._ema_chunk_state_fwd(
        x,
        ema_cs,
        seq_idx=None,
        states=None,
        states_in_fp32=True
    )

    ema_states_updated, ema_final_state = ema_state_pass._ema_state_passing_fwd(
        ema_states, 
        ema_cs[..., -1],
        initial_states=None,
        chunk_size=None,  # not needed strictly speaking for this algo
        out_dtype=ema_states.dtype
    )
    # maybe if you need recompute_output
    # ema_output = _ema_scan_fwd(X_ema, ema_cs, ema_states_updated)

    
    #####################################################################################
    #                                  BACKWARD PASS    
    #####################################################################################
        
    dstates = ema_scan_bwd._ema_chunk_scan_bwd_dstates(ema_cs, dout, seq_idx=None, dtype=ema_states.dtype)

    dstates, ddA_chunk_cumsum, _,= ema_state_passing_bwd_dstates._ema_state_passing_bwd(
        ema_states,
        ema_cs[..., -1],
        dstates,
    )

    dx = ema_chunk_scan_chunk_state_bwd_dx._chunk_scan_chunk_state_bwd_dx(
        x, ema_cs, dout, dstates, D=None, seq_idx=None, dx=None
    )
    ddA_next = ema_chunk_state_bwd_db._ema_chunk_state_bwd_db(x, ema_cs, dstates, raw_scale_gradient=False) # alraedy in pure A gradient state
    ddA_prev = ema_chunk_scan_bwd_dc._ema_chunk_scan_bwd_dC(ema_states_updated, ema_cs, dout, seq_idx=None)

    ddA_prev[..., -1] += ddA_chunk_cumsum # gradients flowing in from the "final state"
    ddA_prev = ddA_prev.flip([-1]).cumsum(dim=-1).flip([-1]) # forward gradient sum to bring into pure "A" gradient state

    ddA = ema_scan_da._ema_chunk_scan_bwd_ddAcs_stable(x, ema_cs, dout)
    ddA += ddA_next + ddA_prev


    return_vals = (dx, ddA)
    return return_vals