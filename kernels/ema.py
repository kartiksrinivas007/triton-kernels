"""
Copyright (c) 2025, Kartik Srinivas
"""

import triton
import triton.language as tl
import torch
import numpy as np
from typing import Optional


configs = [
    triton.Config({"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN})
    for BM in [32, 64, 128]
    for BN in [32, 64, 128]
]


# This filter is static, done before hand (compile-time)
def config_filter(conf):
    BLOCK_M = conf.kwargs["BLOCK_SIZE_M"]
    BLOCK_N = conf.kwargs["BLOCK_SIZE_N"]
    return BLOCK_M * BLOCK_N < 128 * 128


# this filter is dynamic, when stuff is actually passed
def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["seqlen"]
    # Filter out configs where BLOCK_M > the seqlen
    configs = [conf for conf in configs if conf.kwargs.get("BLOCK_SIZE_M", 0) <= N_CTX]
    return configs



@triton.jit
def ema_within_chunk_kernel(
    X_ptr, P_ptr, Z_ptr,
    prod_ptr, sum_ptr,
    stride_xB, stride_xT, stride_xD,
    stride_pB, stride_pT, stride_pD,
    stride_zB, stride_zT, stride_zD,
    stride_prodB, stride_prodC, stride_prodD,
    stride_sumB, stride_sumC, stride_sumD,
    T,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)       
    pid_chunk = tl.program_id(1)   

    start_t = pid_chunk * BLOCK_T

    # base pointers (per batch)
    x_base = X_ptr + pid_b * stride_xB
    p_base = P_ptr + pid_b * stride_pB
    z_base = Z_ptr + pid_b * stride_zB

    offs_d = tl.arange(0, D) # vector of D offsets
    offs_d1 = tl.arange(0, 1) # vector of length-1 for prod block pointer

    # state
    z = tl.zeros((D,), dtype=tl.float32) # vector across D
    decay_prod = tl.full((1,), 1.0, dtype=tl.float32) # scalar as length-1 vector

    for i in tl.static_range(BLOCK_T):
        curr_t = start_t + i
        in_range = curr_t < T   # python bool usable as mask

        # pointers for this timestep
        x_ptr = x_base + curr_t * stride_xT + offs_d * stride_xD   # (D,)
        z_ptr = z_base + curr_t * stride_zT + offs_d * stride_zD   # (D,)
        p_ptr = p_base + curr_t * stride_pT                        # scalar pointer (last dim = 1)

        # masked loads
        x = tl.load(x_ptr, mask=in_range, other=0.0) # (D,)
        p_scalar = tl.load(p_ptr, mask=in_range, other=0.0) # (1,)

        # updates (p_scalar broadcasts over D)
        #TODO(kartiksrinivas): Do this in log space
        new_z = (1.0 - p_scalar) * z + p_scalar * x
        new_decay = decay_prod * (1.0 - p_scalar)

        # conditional update
        z = tl.where(in_range, new_z, z)
        decay_prod = tl.where(in_range, new_decay, decay_prod)

        # masked store of z
        tl.store(z_ptr, z, mask=in_range)

    # compute pointers for outputs
    # make prod_out_ptr a length-1 block pointer so we can store the length-1 decay_prod directly
    prod_out_ptr = prod_ptr + pid_b * stride_prodB + pid_chunk * stride_prodC + offs_d1 * stride_prodD
    # sum_out_ptr remains a D-length block pointer
    sum_out_ptr = sum_ptr + pid_b * stride_sumB + pid_chunk * stride_sumC + offs_d * stride_sumD

    # store: decay_prod is length-1 vector, prod_out_ptr is a length-1 block pointer -> compatible
    tl.store(prod_out_ptr, decay_prod)

    # store the D-vector final z into sum_out_ptr
    tl.store(sum_out_ptr, z)



@triton.jit
def ema_state_passing_kernel(
    Z_ptr, prod_ptr, sum_ptr, 
    o_ptr,
    
    Z_strideB, Z_strideT, Z_strideD,
    prod_strideB, prod_strideC, prod_strideD,
    sum_strideB, sum_strideC, sum_strideD,
    o_strideB, o_strideC, o_strideD,


    T,
    D:tl.constexpr,
    BLOCK_T:tl.constexpr
):
    pid_b = tl.program_id(axis = 0)

    z_base = Z_ptr + pid_b * Z_strideB
    prod_base = prod_ptr + pid_b * prod_strideB
    sum_base = sum_ptr + pid_b * sum_strideB
    o_base = o_ptr + pid_b * o_strideB


    offset_d = tl.arange(0, D)
    # no mask needed, the head dim is assumed perfectly matched
    single_offset = tl.arange(0, 1)
    
    num_chunks = (T + BLOCK_T - 1) // BLOCK_T

    Z_prev = tl.zeros((D, ), dtype=tl.float32) # float32 tensor for now

    for chunk_index in tl.range(num_chunks): # type:ignore 
        
        offset_t = chunk_index * BLOCK_T
        mask_t = offset_t < T # this is a python boolean

        # load Z block  (shape (D, ))
        Z_ptrs = z_base + offset_t * Z_strideT + offset_d * Z_strideD # (D, )
        prod_ptrs = prod_base + chunk_index * prod_strideC + single_offset * prod_strideD # this is a scalar (1,)
        sum_ptrs = sum_base + chunk_index * sum_strideC +  offset_d * sum_strideD # (D, )
        o_ptrs = o_base + chunk_index * o_strideC + offset_d * o_strideD

        Z = tl.load(Z_ptrs, mask=mask_t, other=0.0)
        prod = tl.load(prod_ptrs, mask=mask_t, other=0.0)
        sum  = tl.load(sum_ptrs, mask=mask_t, other=0.0)

        



        pass
    pass


class EMAChunkScanCombinedFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, p, BLOCK_T=8):

        
        ctx.dt_dtype = x.dtype

        def stride_computer(tensor: torch.Tensor):
            return tuple(tensor.stride(index) for index in range(len(tensor.shape)))

        def grid(META):
            return (
                x.shape[0], # batch size,
                triton.cdiv(x.shape[1], META['BLOCK_T']),
            )
        
        batch_size = x.shape[0]
        seqlen = x.shape[1]
        head_dim = x.shape[2]
        num_chunks = (seqlen + BLOCK_T - 1) // (BLOCK_T)

        z = torch.zeros_like(x)
        prod = torch.zeros((batch_size, num_chunks, head_dim), dtype=x.dtype, device=x.device)
        sum = torch.zeros_like(prod)

        ctx.grid = grid


        ema_within_chunk_kernel[grid](
            x, p, z,
            prod, sum, 
            *stride_computer(x),
            *stride_computer(p),
            *stride_computer(z),
            *stride_computer(prod),
            *stride_computer(sum),
            seqlen,
            head_dim,
            BLOCK_T=BLOCK_T,
        )
        out = None
        return out
    @staticmethod
    def backward(ctx, dout, *args):
        return None, None, None



def ema_scan_combined(x, p, BLOCK_T):

    assert x.is_contiguous()
    assert p.is_contiguous()

    return EMAChunkScanCombinedFn.apply(x, p, BLOCK_T)

