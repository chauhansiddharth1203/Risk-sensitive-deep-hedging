"""
training/trainer_mom.py
-----------------------
Median-of-Means (MoM) gradient training for the VIX-as-vega policy.

Hypothesis under test: the seed-instability observed on crisis windows
(COVID Delta = -8.05 +/- 4.96, std bigger than the mean) is caused by
heavy-tailed VIX gradient updates. If true, an estimator that is
sub-Gaussian-concentrated under heavy tails should both shrink the
seed-std and shift the mean toward zero.

Method (Lugosi & Mendelson, 2019):
    1. Split the mini-batch of N samples into k roughly-equal blocks.
    2. Compute the *per-block* gradient by running backward on each
       block separately.
    3. Take the component-wise median of those k gradient tensors and
       use it as the parameter gradient.
    4. Step the optimiser as usual.

The only thing that changes vs. plain Adam is *how* the gradient is
aggregated across the batch. Architecture, loss, learning rate,
clipping, action scales -- all untouched. This makes the comparison
clean: any change in the result is attributable to gradient
aggregation, not to anything else.

Drop-in replacement for the `train` function in main_vix_futures_v2.py.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim

from policy.network_vix import VIXHedgingPolicy
from market.vix_bootstrap import payoff_call_atm
from main_backtest_training import cvar_loss
from main_vix_futures_v2 import rollout_batch_vix


def _split_blocks(N: int, k: int):
    """Yield k roughly-equal index slices covering [0, N)."""
    base = N // k
    rem = N - base * k
    start = 0
    for j in range(k):
        size = base + (1 if j < rem else 0)
        yield slice(start, start + size)
        start += size


def mom_gradient_step(policy, opt, S_batch, VS_batch, alpha, lam, k,
                      clip_norm=1.0):
    """One median-of-means optimiser step.

    Returns (loss_mean, cvar_mean, pnl_mean) -- the per-block means,
    averaged for logging purposes.
    """
    N = S_batch.shape[0]
    # Stash per-block gradient tensors keyed by parameter name.
    grad_stash = {name: [] for name, _ in policy.named_parameters()}
    losses, cvars, pnl_means = [], [], []

    for sl in _split_blocks(N, k):
        S_b = S_batch[sl]
        VS_b = VS_batch[sl]
        if S_b.shape[0] < 2:
            # Degenerate block (block_size < 2 can't produce a useful
            # CVaR estimate). Skip rather than crash.
            continue
        opt.zero_grad(set_to_none=True)
        pnl = rollout_batch_vix(policy, S_b, VS_b, payoff_call_atm)
        c = cvar_loss(pnl, alpha)
        loss = c + lam * torch.abs(pnl.mean())
        loss.backward()
        losses.append(loss.item())
        cvars.append(-c.item())
        pnl_means.append(pnl.mean().item())
        for name, p in policy.named_parameters():
            if p.grad is not None:
                grad_stash[name].append(p.grad.detach().clone())

    # Take component-wise median across blocks and write back into
    # param.grad, then step.
    opt.zero_grad(set_to_none=True)
    for name, p in policy.named_parameters():
        gs = grad_stash[name]
        if not gs:
            continue
        stacked = torch.stack(gs, dim=0)  # (k, *param.shape)
        # torch.median returns the lower of the two middle values for
        # even k. That's fine -- it's still a valid robust estimator.
        med = stacked.median(dim=0).values
        p.grad = med

    if clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_norm)
    opt.step()

    return (float(np.mean(losses)) if losses else float("nan"),
            float(np.mean(cvars))  if cvars  else float("nan"),
            float(np.mean(pnl_means)) if pnl_means else float("nan"))


def train_mom(s_tr, s_te, epochs, N, lam, lr, seed, k=9, device=None):
    """MoM training loop. Signature mirrors main_vix_futures_v2.train()
    plus the extra block-count argument `k`.

    Curriculum on alpha and all other hyperparameters are kept identical
    to the baseline trainer so the comparison is clean.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = VIXHedgingPolicy().to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)
    hist = {"epoch": [], "loss": [], "cvar_train": [], "cvar_val": [],
            "mean_train": [], "k": k}

    for ep in range(epochs):
        alpha = 0.80 + 0.15 * min(ep / max(epochs - 1, 1), 1.0)
        S, VS = s_tr.sample_batch(N, device=device)
        loss_m, cvar_m, pnl_m = mom_gradient_step(
            policy, opt, S, VS, alpha, lam, k)

        if (ep + 1) % 50 == 0 or ep == 0:
            with torch.no_grad():
                Sv, VSv = s_te.sample_batch(512, device=device)
                pv = rollout_batch_vix(policy, Sv, VSv, payoff_call_atm)
                cv = cvar_loss(pv, 0.95).item()
            hist["epoch"].append(ep + 1)
            hist["loss"].append(loss_m)
            hist["cvar_train"].append(cvar_m)
            hist["cvar_val"].append(-cv)
            hist["mean_train"].append(pnl_m)
            print(f"ep {ep+1:4d}  [MoM k={k}]  loss={loss_m:+.3f}  "
                  f"CVaR_train={cvar_m:+.3f}  "
                  f"mean={pnl_m:+.3f}  CVaR_val={-cv:+.3f}")

    return policy, hist
