import torch

def european_call_payoff(S_T, K=100.0):
    return torch.clamp(S_T - K, min=0.0)
