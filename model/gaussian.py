import torch

def gaussian_centers_torch(mu_min: float, mu_max: float, delta_mu: float, device=None, dtype=torch.float32) -> torch.Tensor:
    K = int((mu_max) // delta_mu)
    mus = mu_min + torch.arange(K + 1, dtype=dtype, device=device) * delta_mu
    return mus  # (C,)

def gaussian_expand_torch(distances: torch.Tensor,
                          mu_min: float,
                          mu_max: float,
                          delta_mu: float,
                          sigma: float) -> torch.Tensor:
    """
    distances: torch tensor of shape (...), e.g. (N,N) or (num_pairs,) or (B,N,N)
    returns: (..., C) torch tensor of Gaussian-expanded features (requires_grad preserved)
    """
    device = distances.device
    dtype = distances.dtype
    mus = gaussian_centers_torch(mu_min, mu_max, delta_mu, device=device, dtype=dtype)  # (C,)
    d = distances.unsqueeze(-1)              # (..., 1)
    diff = d - mus                           # (..., C) broadcasted
    feats = torch.exp(-0.5 * (diff ** 2) / (sigma ** 2))
    return feats.to(device)