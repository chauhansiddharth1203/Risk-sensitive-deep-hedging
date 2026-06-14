import torch


def cvar(pnl: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """
    Compute CVaR (Expected Shortfall)

    Args:
        pnl   : Tensor of shape (N,) -- profit & loss samples
        alpha : confidence level (e.g. 0.95)

    Returns:
        CVaR value (scalar tensor)
    """

    if pnl.dim() != 1:
        raise ValueError("pnl must be a 1D tensor")

    # Sort PnL ascending (worst losses first)
    sorted_pnl, _ = torch.sort(pnl)

    # Index corresponding to VaR
    k = int((1 - alpha) * len(sorted_pnl))

    # Safety clamp
    k = max(k, 1)

    # CVaR = mean of worst (1-alpha)% outcomes
    return sorted_pnl[:k].mean()
