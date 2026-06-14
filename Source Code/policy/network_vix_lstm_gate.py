"""
policy/network_vix_lstm_gate.py
---------------------------------
Step 4a: LSTM online gate for slow-drift crisis regimes.

MOTIVATION
----------
The frozen sigmoid gate at theta=3.0 uses a single scalar (normalised
VIX / window-start VIX) to decide whether to suppress the VIX leg.
This works well when crises arrive as spikes (COVID March 2020) because
VIX/VIX_0 quickly exceeds 3x. But for slow-drift regimes (2022 rate
shock, where VIX rose from ~17 to ~35 over 12 months), the relative
VIX ratio stays below 3x for most of the window even when the regime
has clearly changed.

FIX: replace the scalar sigmoid gate with a small LSTM that sees the
last K steps of (SPY_returns, VIX_changes) and outputs a gate value in
[0,1]. The LSTM can detect the *trajectory* of VIX rise, not just the
absolute level -- so it fires earlier in slow-drift crises.

ARCHITECTURE
------------
Same as VIXGatedPolicy (Step 1) plus an LSTM gate module:

  gate_value_t = sigmoid(LSTM_gate(history_{t-K:t}))

where history includes (S_t/S_0, VS_t/VS_0, prev_dS, prev_dV) for the
last K steps.

TWO-STAGE TRAINING (learned from Week 13)
------------------------------------------
Joint training of policy + LSTM gate fails because pooled-regime CVaR
over-weights crisis gradient pressure, causing the gate to close
everywhere. We use the same two-stage protocol:

  Stage 1: Train the hedging policy with the gate FROZEN at its
           initial value (0.5, i.e. 50% open). Runs for `epochs_policy`
           epochs.

  Stage 2: Freeze the policy weights. Train ONLY the LSTM gate
           parameters on the validation set (2022-2023) to minimize
           CVaR while the policy is fixed. Runs for `epochs_gate`
           epochs. Because the gate objective is evaluated on OOT
           validation windows, it learns to open in calm periods and
           close in high-vol periods without seeing the test set.

This separation ensures:
  - The policy learns the correct delta/vega hedge (Stage 1)
  - The gate learns regime detection (Stage 2)
  - Neither stage contaminates the other's learning signal
"""

import torch
import torch.nn as nn


class LSTMGate(nn.Module):
    """
    Small LSTM that predicts a gate value in (0,1) from recent history.

    Input at step t: (S_t/S_0, VS_t/VS_0, prev_dS, prev_dV) -- 4 features.
    Hidden state: h_dim = 16 (small, fast).
    Output: single scalar in (0,1) via sigmoid.

    The gate opens (value -> 1) in low-vol, predictable regimes.
    The gate closes (value -> 0) during VIX spikes / high volatility.
    """

    def __init__(self, input_dim=4, h_dim=16, n_layers=1):
        super().__init__()
        self.h_dim   = h_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, h_dim, n_layers, batch_first=True)
        self.head = nn.Linear(h_dim, 1)
        # Initialise head bias to 1 so gate starts open (value ~ 0.73)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, 1.0)

    def forward(self, history):
        """
        history : (N, t, 4) tensor of state features up to step t
        returns : (N,) gate values in (0,1)
        """
        out, (h, _) = self.lstm(history)
        gate_logit = self.head(h[-1]).squeeze(1)   # (N,)
        return torch.sigmoid(gate_logit)


class VIXLSTMGatedPolicy(nn.Module):
    """
    VIX hedging policy with LSTM-based online regime gate.

    Architecture
    ------------
    - Same MLP policy network as VIXGatedPolicy (state_dim=5, hidden=64)
    - LSTM gate replaces the scalar sigmoid gate
    - Gate receives last min(t, window_size) steps of state history

    Parameters
    ----------
    window_size : int
        How many past steps the LSTM gate can see (default 6 = full window).
    freeze_policy : bool
        If True, policy MLP weights are frozen (Stage 2 gate training).
    freeze_gate : bool
        If True, LSTM gate weights are frozen (Stage 1 policy training).
    """

    def __init__(self,
                 hidden_dim=64,
                 cost_rate=0.0002,
                 S0=100.0,
                 stock_scale=2.0,
                 vix_scale=0.3,
                 gate_h_dim=16,
                 window_size=6,
                 freeze_policy=False,
                 freeze_gate=True):
        super().__init__()
        self.cost_rate   = cost_rate
        self.S0          = S0
        self.VS0         = 100.0
        self.stock_scale = stock_scale
        self.vix_scale   = vix_scale
        self.window_size = window_size

        # Policy MLP (same as VIXGatedPolicy)
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # LSTM gate
        self.lstm_gate = LSTMGate(input_dim=4, h_dim=gate_h_dim)

        # Freeze control
        for p in self.net.parameters():
            p.requires_grad = not freeze_policy
        for p in self.lstm_gate.parameters():
            p.requires_grad = not freeze_gate

    def forward(self, state):
        """Standard forward for the policy MLP."""
        return self.net(state)

    def get_gate(self, history):
        """
        history : (N, t, 4) tensor [(S/S0, VS/VS0, prev_dS, prev_dV)]
        returns : (N,) gate values
        """
        return self.lstm_gate(history)
