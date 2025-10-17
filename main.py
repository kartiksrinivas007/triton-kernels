import numpy as np
import triton
import torch
import jaxtyping
import triton.language as tl
import triton.runtime.driver as driver

from kernels.simple_kernels import *
from kernels.flash_attn import *

def _get_gpu_specifications():

    assert torch.cuda.is_available(), "CUDA must be avialble to run triton kernels"
    DEVICE = driver.active.get_active_torch_device() # type: ignore

    properties = driver.active.utils.get_device_properties(DEVICE.index) # type:ignore
    NUM_SM = properties["multiprocessor_count"]
    NUM_REGS = properties["max_num_regs"]
    SIZE_SMEM = properties["max_shared_mem"]
    WARP_SIZE = properties["warpSize"]

    print("Device = ", DEVICE)
    print("Num REGS per SM  = ", NUM_REGS)
    print("Num SM  = ", NUM_SM)
    print("Total Shared memory (bytes) = ", SIZE_SMEM)
    print("Warp size = ", WARP_SIZE)

    return DEVICE, properties


def print_kernel_usage(triton_kernel, properties, **kwargs,):
    pass


if __name__ == "__main__":

    DEVICE, properties = _get_gpu_specifications()

    target = triton.runtime.driver.active.get_current_target()
    kernels = {}

    BATCH_SIZE = 2
    HEADS = 4
    SEQLEN = 10240
    HEAD_DIM = 128

    Q = torch.randn((BATCH_SIZE, HEADS, SEQLEN, HEAD_DIM), device=DEVICE, dtype=torch.float32)
    K = torch.randn((BATCH_SIZE, HEADS, SEQLEN, HEAD_DIM), device=DEVICE, dtype=torch.float32)
    V = torch.randn((BATCH_SIZE, HEADS, SEQLEN, HEAD_DIM), device=DEVICE, dtype=torch.float32)
    O = torch.empty_like(Q)

    def simple_attention_forward(Q, K, V, scale, causal=False):  
        P = torch.matmul(Q, K.transpose(2, 3)) * scale
        M = torch.tril(torch.ones(SEQLEN, SEQLEN, device = DEVICE))
        if causal:
            P[:, :, M == 0] = float("-inf")
        P = torch.softmax(P.float(), dim=-1)
        O = torch.matmul(P, V)
        return O
    
    print(simple_attention_forward(Q, K, V, scale = 1/np.sqrt(HEAD_DIM)))


    pass
