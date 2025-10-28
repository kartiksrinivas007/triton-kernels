import triton
import torch
import triton.runtime.driver as driver


from kernels.linear_attn import LinearAttention


def linear_attention_forward(Q, K, V, scale, causal=False):
    P = torch.matmul(Q, K.transpose(2, 3)) * scale
    SEQLEN = Q.shape[-2]
    M = torch.tril(torch.ones(SEQLEN, SEQLEN, device=P.device))
    if causal:
        P[:, :, M == 0] = float("-inf")
    O = torch.matmul(P, V)
    return O


def triton_linear_attention_kernel_forward(Q, K, V) -> torch.Tensor:
    output = LinearAttention.apply(Q, K, V)
    return output  # type: ignore
