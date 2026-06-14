"""
training/trainer_gated.py
--------------------------
Trainer for the regime-gated VIX policy (Option A).

Same loss (CVaR + mean penalty), same optimiser (Adam), same hyper-
parameters as the plain-Adam Week-12 baseline. The only differences
from main_vix_futures_v2.py are:
  1. The policy class is VIXGatedPolicy (has a learnable gate).
  2. The rollout applies the gate to the VIX action before taking the
     P&L step.

This isolates exactly one change vs the plain-Adam baseline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim

from policy.network_vix_gated import VIXGatedPolicy
from market.vix_bootstrap import payoff_call_atm
from main_backtest_training import cvar_loss


def rollout_batch_gated(policy, S, VS, payoff_fn):
    """Rollout with regime-gated VIX action.

    Identical to main_vix_futures_v2.rollout_batch_vix except for the
    line that multiplies dV by the gate value, which depends on the
    current normalised VIX level.
    """
    N, Tp1 = S.shape
    Tn = Tp1 - 1
    pnl = torch.zeros(N, device=S.device)
    prev_S = torch.zeros(N, device=S.device)
    prev_V = torch.zeros(N, device=S.device)
    for t in range(Tn):
        vs_norm = VS[:, t] / policy.VS0
        state = torch.stack([
            S[:, t] / policy.S0,
            vs_norm,
            torch.full((N,), t / Tn, device=S.device),
            prev_S,
            prev_V,
        ], dim=1)
        a = policy.forward(state)
        dS = torch.tanh(a[:, 0]) * policy.stock_scale
        # Apply regime gate to the VIX action.
        g = policy.gate(vs_norm)
        dV = torch.tanh(a[:, 1]) * policy.vix_scale * g

        gain_S = prev_S * (S[:, t + 1] - S[:, t])
        gain_V = prev_V * (VS[:, t + 1] - VS[:, t])
        tc = (policy.cost_rate * torch.abs(dS - prev_S) * (S[:, t] / policy.S0)
            + policy.cost_rate * torch.abs(dV - prev_V) * (VS[:, t] / policy.VS0))
        pnl = pnl + gain_S + gain_V - tc
        prev_S, prev_V = dS, dV
    pnl = pnl - payoff_fn(S[:, -1])
    return pnl


def train_gated(s_tr, s_te, epochs, N, lam, lr, seed,
                gate_threshold_init=1.5, gate_scale_init=0.2,
                freeze_gate=False, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = VIXGatedPolicy(
        gate_threshold_init=gate_threshold_init,
        gate_scale_init=gate_scale_init,
        freeze_gate=freeze_gate,
    ).to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)
    hist = {"epoch": [], "loss": [], "cvar_train": [], "cvar_val": [],
            "mean_train": [], "gate_threshold": [], "gate_scale": []}

    for ep in range(epochs):
        alpha = 0.80 + 0.15 * min(ep / max(epochs - 1, 1), 1.0)
        S, VS = s_tr.sample_batch(N, device=device)
        pnl = rollout_batch_gated(policy, S, VS, payoff_call_atm)
        c = cvar_loss(pnl, alpha)
        loss = c + lam * torch.abs(pnl.mean())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()

        if (ep + 1) % 50 == 0 or ep == 0:
            with torch.no_grad():
                Sv, VSv = s_te.sample_batch(512, device=device)
                pv = rollout_batch_gated(policy, Sv, VSv, payoff_call_atm)
                cv = cvar_loss(pv, 0.95).item()
            hist["epoch"].append(ep + 1)
            hist["loss"].append(loss.item())
            hist["cvar_train"].append(-c.item())
            hist["cvar_val"].append(-cv)
            hist["mean_train"].append(pnl.mean().item())
            hist["gate_threshold"].append(
                float(policy.gate_threshold.detach().cpu()))
            hist["gate_scale"].append(
                float(policy.gate_scale.detach().cpu()))
            print(f"ep {ep+1:4d}  loss={loss.item():+.3f}  "
                  f"CVaR_train={-c.item():+.3f}  "
                  f"CVaR_val={-cv:+.3f}  "
                  f"gate(thr={hist['gate_threshold'][-1]:.3f}, "
                  f"sc={hist['gate_scale'][-1]:.3f})")

    return policy, hist
