
import torch 
import triton 
import math 
import triton.language as tl


# TODO(kartiksrinivas): Add a pruner, for BLOCK_SIZE_T < token_dim
# TODO(kartiksrinivas): Check correctness with "python -m pytest -v kernels/tests/ema/test_ema_state_pass.py --count=5  --randomly-seed=400244919"
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_T': 64}),
        triton.Config({'BLOCK_SIZE_T': 128}),
        triton.Config({'BLOCK_SIZE_T': 256}),
        triton.Config({'BLOCK_SIZE_T': 512}),
        # triton.Config({'BLOCK_SIZE_T': 1024}), -- causes bugs! its bigger than token_dim
        # triton.Config({'BLOCK_SIZE_T': 2048}),
    ],
    key=['token_dim'],
)
@triton.jit
def _ema_state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr, out_ptr, final_states_ptr, A_cs_last_ptr, initstates_ptr,
    # Matrix dimensions
    token_dim, nchunks, batch,
    # Strides
    stride_states_batch, stride_states_chunk, stride_states_token_dim,
    stride_out_batch, stride_out_chunk, stride_out_token_dim,
    stride_final_states_batch, stride_final_states_token_dim,
    stride_A_cs_last_batch, stride_A_cs_last_chunk,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """Compute EMA over the states acorss chunks themselves.
    acc = all zeroes
    out.store(acc)
    for chunk in chunks:
        new_state = load internal final state of chunk 
        decay = load decay of present chunk
        acc(newer value of new chunk start) = new_state + decay * acc (older value of old chunk start)
        out.store(acc)
    """

    pid_b = tl.program_id(axis=1)
    pid_t = tl.program_id(axis=0)

    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_t = pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)

    # Compute base pointers (no chunk movement)
    states_ptrs = states_ptr + offs_b[:, None] * stride_states_batch + offs_t[None, :] * stride_states_token_dim
    A_cs_last_ptrs = A_cs_last_ptr + offs_b * stride_A_cs_last_batch
    out_ptrs = out_ptr + offs_b[:, None] * stride_out_batch + offs_t[None, :] * stride_out_token_dim

    # Compute final position pointer
    final_states_ptrs = final_states_ptr + offs_b[:, None] * stride_final_states_batch + offs_t[None, :] * stride_final_states_token_dim

    # main mask to check for the batch sizes and the token sizes, chunks are within limit by definition (for loop)
    bt_mask  = (offs_b[:, None] < batch) & (offs_t[None, :] < token_dim)



    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_T), dtype=tl.float32)

    # stores the initial state and moves forward
    tl.store(out_ptrs, acc , mask=bt_mask)
    # move a single chunk forward
    out_ptrs += stride_out_chunk

    # TODO(kartiksrinivas) : pipelining in the loop?
    for c in range(nchunks):
        final_states = tl.load(states_ptrs, mask=bt_mask, other=0.0).to(tl.float32)
        A_cs_last = tl.load(A_cs_last_ptrs, mask = offs_b < batch, other=0.0).to(tl.float32)
        scale = tl.exp(A_cs_last)
        acc = scale * acc + final_states
        if c < nchunks - 1:
            tl.store(out_ptrs, acc, mask=bt_mask)
        else:
            tl.store(final_states_ptrs, acc, mask=bt_mask)

        states_ptrs += stride_states_chunk
        A_cs_last_ptrs += stride_A_cs_last_chunk
        out_ptrs += stride_out_chunk



# Why flatten and go forward when you can load a block of each dimension in head-dim and d_state?
def _ema_state_passing_fwd(states : torch.Tensor, A_cs_last : torch.Tensor , initial_states=None, chunk_size=None,
                       out_dtype=None):
    batch, nchunks, token_dim = states.shape
    assert A_cs_last.shape == (batch, nchunks)
    if initial_states is not None:
        raise NotImplementedError("No support for init_states in EMA kernel.")

    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((batch, nchunks, token_dim), device=states.device, dtype=out_dtype) # one start state per chunk
    final_states = torch.empty((batch, token_dim), device=states.device, dtype=torch.float32) # one final state per batch example
    grid = lambda META: (triton.cdiv(token_dim, META['BLOCK_SIZE_T']), triton.cdiv(batch, META['BLOCK_SIZE_B']))

    with torch.cuda.device(states.device.index):
        _ema_state_passing_fwd_kernel[grid](
            states, out, final_states, A_cs_last, initial_states,
            token_dim, nchunks, batch ,
            states.stride(0), states.stride(1), states.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            final_states.stride(0), final_states.stride(1),
            A_cs_last.stride(0), A_cs_last.stride(1),
            BLOCK_SIZE_B = 1 # no blocking  over batch for now
        )
    return out, final_states


