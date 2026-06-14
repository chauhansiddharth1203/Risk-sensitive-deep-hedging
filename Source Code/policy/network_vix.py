"""
policy/network_vix.py
----------------------
Sprint 2b policy with tightened VIX action scale.

Identical to HedgingPolicyVarSwap except:
  - Stock action scale  : tanh x 2   (was x 5)
  - VIX   action scale  : tanh x 0.3 (was x 5) --- VIX moves are ~20x
                          larger than SPY moves in % terms, so the
                          position size must be correspondingly smaller.
"""

import torch
import torch.nn as nn


class VIXHedgingPolicy(nn.Module):
    def __init__(self, hidden_dim=64, cost_rate=0.0002, S0=100.0,
                 stock_scale=2.0, vix_scale=0.3):
        super().__init__()
        self.cost_rate = cost_rate
        self.S0 = S0
        self.VS0 = 100.0         # VIX normalised per-window to 100
        self.stock_scale = stock_scale
        self.vix_scale   = vix_scale
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        return self.net(state)
