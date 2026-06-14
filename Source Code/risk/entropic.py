import torch

def entropic_risk(pnl, lam=1.0):
    """
    Numerically stable entropic risk using log-sum-exp trick
    """
    x = -lam * pnl
    x_max = torch.max(x)
    return (1 / lam) * (x_max + torch.log(torch.mean(torch.exp(x - x_max))))
