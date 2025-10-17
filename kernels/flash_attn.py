"""
Copyright (c) 2025, Kartik Srinivas
"""

import triton 
import triton.language as tl
import torch
from typing import Optional


@triton.jit
def make_tensor_descriptor(pointer, stride, shape, block_shape) -> tl.tensor_descriptor:
    if isinstance(pointer, tl.tensor_descriptor):
        return pointer
    else:
        return tl.make_tensor_descriptor(
            base=pointer, 
            shape=shape, 
            strides=stride, 
            block_shape=block_shape
        ) 
    pass


@triton.jit
def flash_attn_fwd_wrapper_kernel(
    q_ptr, 
    k_ptr,
    v_ptr,
    scale, 
     
    #strides, 
    q_stride_b, 
    q_stride_h, 
    q_stride_n,
    q_stride_d,
    
    k_stride_b, 
    k_stride_h, 
    k_stride_n, 
    k_stride_d, 
    
    v_stride_b, 
    v_stride_h, 
    v_stride_n,
    v_stride_d, 

    o_stride_b, 
    o_stride_h, 
    o_stride_n, 
    o_stride_d,

    # BLOCK SIZES
    BLOCK_SIZE_M : tl.constexpr, # For Q
    BLOCK_SIZE_N : tl.constexpr, # for K and V
    HEAD_DIM: tl.constexpr,
):

    pid_m = tl.program_id(0) # which SEQLEN block of Q am I handling
    pid_bh = tl.program_id(1) # which Batch and Head am I handling
    pid_batch = pid_bh // HEAD_DIM
    pid_head = pid_bh % HEAD_DIM


    # ---------------------------------------------------------------------
    # Triton > 2.3.0 supports Tensor Descriptor
    # We can use that to load tensors instead of doing offset + stride computation
    # for things WITHIN a block (we still need to identify the block itself) 
    # ---------------------------------------------------------------------

    # obtain the offsets of the particular Q block of interest
    # find the ptrs to load using the strides that you have and the pid

    

    # instead of doing that you can make a tensor descriptor and load that directly
    # that is much easier than doing the pointer arithmetic
    
    

    pass


@triton.jit
def flash_attn_inner_kernel():
    pass


@triton.jit
def flash_attn_bwd_kernel():
    pass



class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale):


        # Descriptors need to be allocated in global memory (is this expensive?)
        def alloc_fn(size: int, alignment: int, stream: Optional[int]) -> torch.Tensor:
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        output = torch.empty_like(q) # make it the same shape as q
        flash_attn_fwd_wrapper_kernel()
        pass

    
    @staticmethod
    def backward(ctx, do):
        return None, None, None

    pass

    