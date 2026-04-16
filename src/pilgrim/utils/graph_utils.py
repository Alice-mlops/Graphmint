import torch


def identity(n: int) -> list[int]:
    """Return a list of integers from 0 to n-1."""
    return list(range(n))


def half_interleave(n: int) -> list[int]:
    m = n // 2
    out = []
    if n % 2 == 0:
        for i in range(m):
            out.append(m + i)
            out.append(i)
    else:
        for i in range(m):
            out.append(m + 1 + i)
            out.append(i)
        out.append(m)
    return out


def subsample_xy(X, y, cap, seed=123):
    if cap is None or X.shape[0] <= cap:
        return X, y
    g = torch.Generator(device=X.device)
    g.manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g, device=X.device)[:cap]
    return X[idx], y[idx]


def y_stats(y: torch.Tensor, sample=5000, seed: int | None = None):
    y0 = y.detach()
    y_min = float(y0.min().item())
    y_max = float(y0.max().item())
    y_std = float(y0.float().std().item())
    if sample is None or y0.shape[0] <= sample:
        ys = y0
    else:
        if seed is None:
            idx = torch.randperm(y0.shape[0], device=y0.device)[:sample]
        else:
            g = torch.Generator(device=y0.device)
            g.manual_seed(seed)
            idx = torch.randperm(y0.shape[0], generator=g, device=y0.device)[:sample]
        ys = y0[idx]
    ys = ys.detach().cpu()
    uniq = int(torch.unique(ys).numel())
    return y_min, y_max, y_std, uniq
