

def ema_loop(X, P):
    B, T, D = X.shape
    N  = math.ceil(T)
    Z = torch.zeros(B, N, D)
    for b in range(B):
        z_prev = torch.zeros(D, device=X.device, dtype=X.dtype)
        for t in range(T):
            p = P[b, t, 0]
            x = X[b, t]
            z = (1.0 - p) * z_prev + x
            z_prev = z
            Z[b, t, :] = z
    return Z



