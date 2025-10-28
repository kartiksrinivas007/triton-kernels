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
def ema_chunk_kernel(
    X_ptr, P_ptr, Z_ptr,
    prod_ptr, sum_ptr,
    # strides for X: (B, T, D)
    stride_xB, stride_xT, stride_xD,
    # strides for P: (B, T, 1)
    stride_pB, stride_pT, stride_pD,
    # strides for Z: (B, T, D)
    stride_zB, stride_zT, stride_zD,
    # strides for prod: (B, C, 1)
    stride_prodB, stride_prodC, stride_prodD,
    # strides for sum: (B, C, D)
    stride_sumB, stride_sumC, stride_sumD,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)       # batch index
    pid_chunk = tl.program_id(1)   # chunk index along time

    start_t = pid_chunk * BLOCK_T

    # base pointers (per batch)
    x_base = X_ptr + pid_b * stride_xB
    p_base = P_ptr + pid_b * stride_pB
    z_base = Z_ptr + pid_b * stride_zB

    offs_d = tl.arange(0, D)       # vector of D offsets
    offs_d1 = tl.arange(0, 1)      # vector of length-1 for prod block pointer

    # state
    z = tl.zeros((D,), dtype=tl.float32)                 # vector across D
    decay_prod = tl.full((1,), 1.0, dtype=tl.float32)    # scalar as length-1 vector

    # iterate BLOCK_T steps (masked)
    for i in range(BLOCK_T):
        curr_t = start_t + i
        in_range = curr_t < T   # python bool usable as mask

        # pointers for this timestep
        x_ptr = x_base + curr_t * stride_xT + offs_d * stride_xD   # (D,)
        z_ptr = z_base + curr_t * stride_zT + offs_d * stride_zD   # (D,)
        p_ptr = p_base + curr_t * stride_pT                        # scalar pointer (last dim = 1)

        # masked loads
        x = tl.load(x_ptr, mask=in_range, other=0.0)        # (D,)
        p_scalar = tl.load(p_ptr, mask=in_range, other=0.0) # (1,)

        # updates (p_scalar broadcasts over D)
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


        ema_chunk_kernel[grid](
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

