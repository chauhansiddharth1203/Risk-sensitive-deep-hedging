"""
policy/network_vix_gated.py
----------------------------
Regime-gated VIX hedging policy (Option A).

Hypothesis under test: the calm-window win and the crisis-window loss
share the same mechanism -- VIX hedging adds value when the regime is
calm and predictable, but breaks down in crisis because the VIX signal
moves too violently relative to the underlying. So instead of asking
the network to learn this regime-dependent behaviour implicitly, we
build it in as a learnable gate:

    dV_effective = gate(VS_norm) * dV_network

where gate(.) is a sigmoid that is open (≈1) when normalised VIX is
low and closes (->0) when normalised VIX is high. Threshold and width
of the sigmoid are learnable parameters initialised to plausible
values so the network can decide how aggressive the regime gate should
be.

Everything else (state, stock action, action scales, network width) is
identical to VIXHedgingPolicy so the comparison with the plain-Adam
multi-seed baseline isolates exactly one change: the gate.
"""

import torch
import torch.nn as nn


class VIXGatedPolicy(nn.Module):
    def __init__(self,
                 hidden_dim=64, cost_rate=0.0002, S0=100.0,
                 stock_scale=2.0, vix_scale=0.3,
                 gate_threshold_init=1.5,    # close gate when VS_norm > 1.5
                 gate_scale_init=0.2,
                 freeze_gate=False):
        super().__init__()
        self.cost_rate = cost_rate
        self.S0 = S0
        self.VS0 = 100.0
        self.stock_scale = stock_scale
        self.vix_scale   = vix_scale
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Gate parameters: sigmoid centred at `gate_threshold`, width
        # `gate_scale`. The gate value at VS_norm = threshold is 0.5.
        # Below threshold -> gate->1 (open, VIX hedging on);
        # above threshold -> gate->0 (closed, VIX position suppressed).
        self.gate_threshold = nn.Parameter(
            torch.tensor(float(gate_threshold_init)),
            requires_grad=not freeze_gate)
        # Store raw scale; we clamp |scale| in gate() to avoid pathological
        # very-narrow sigmoids that would behave like a step function and
        # block gradient flow.
        self.gate_scale = nn.Parameter(
            torch.tensor(float(gate_scale_init)),
            requires_grad=not freeze_gate)

    def gate(self, vs_norm):
        """Sigmoid gate. Returns shape-matched scalar in (0, 1)."""
        width = torch.abs(self.gate_scale).clamp(min=0.02)
        return torch.sigmoid((self.gate_threshold - vs_norm) / width)

    def forward(self, state):
        """Returns the *raw* network output. Gating is applied in the
        rollout because the rollout has access to per-step VS_norm and
        the action scaling."""
        return self.net(state)
