"""
market/historical_bootstrap.py
-------------------------------
Week 9: Stationary block-bootstrap simulator for training on real SPY
returns, replacing the Heston/Bates SDE simulator entirely.

Why block bootstrap?
    i.i.d. resampling of daily returns destroys volatility clustering
    (the core property that makes real returns *different* from Heston).
    Politis-Romano (1994) stationary bootstrap samples blocks of random
    geometric length, preserving short-range autocorrelation and
    clustering while still producing unlimited synthetic episodes.

Match to Week 5 simulator interface:
    simulate_historical(N, device) -> (S, VS, payoff_fn, meta)

Variance-swap proxy:
    Same EWMA(lambda=0.94) x 252 annualisation as Week 7 historical stress
    test, scaled VS = v · S0 / SIGMA_V_FIXED with SIGMA_V_FIXED=0.30.
    This keeps the state space identical to the simulated-training
    pipeline, so the policy architecture is unchanged.
"""

import math
import numpy as np
import torch

from data.spx_windows import load_train_test

# Constants matched to Week 5 / Week 7 pipeline
S0            = 100.0
K             = 100.0
T             = 30
SIGMA_V_FIXED = 0.30
EWMA_LAMBDA   = 0.94
MEAN_BLOCK    = 10     # Geometric mean block length (~2 weeks)


def ewma_variance(returns, lam=EWMA_LAMBDA):
    """Annualised EWMA variance series, aligned with `returns`."""
    v = np.zeros(len(returns))
    v[0] = returns[0] ** 2
    for i in range(1, len(returns)):
        v[i] = lam * v[i - 1] + (1 - lam) * returns[i] ** 2
    return v * 252.0


class HistoricalBootstrap:
    """
    Stationary block bootstrap over a pool of real log-returns.

    Each draw samples N episodes of length T consecutive days by
    stitching random blocks of geometric length with mean MEAN_BLOCK.
    """

    def __init__(self, pool_returns, mean_block=MEAN_BLOCK):
        self.rets = np.asarray(pool_returns, dtype=np.float64)
        self.v    = ewma_variance(self.rets)
        self.n    = len(self.rets)
        self.p    = 1.0 / mean_block     # geometric param for block end

    def sample_batch(self, N, horizon=T, device="cpu"):
        """Returns (S, VS) each of shape (N, horizon+1)."""
        rets_batch = np.empty((N, horizon), dtype=np.float64)
        vs_batch   = np.empty((N, horizon + 1), dtype=np.float64)

        for i in range(N):
            # Stationary bootstrap: sample blocks until we have `horizon`
            out_r = np.empty(horizon)
            out_v = np.empty(horizon + 1)
            pos   = 0

            # Initial variance = v at the first sampled index
            start = np.random.randint(0, self.n)
            out_v[0] = self.v[start]

            cur = start
            while pos < horizon:
                out_r[pos] = self.rets[cur]
                out_v[pos + 1] = self.v[cur]
                pos += 1
                # Decide: extend block or start new one?
                if np.random.random() < self.p:
                    cur = np.random.randint(0, self.n)
                else:
                    cur = (cur + 1) % self.n   # wrap around (circular)

            rets_batch[i] = out_r
            vs_batch[i]   = out_v

        # Build price paths: S[0] = 100, S[t+1] = S[t] * exp(r_t)
        log_S = np.concatenate(
            [np.zeros((N, 1)), np.cumsum(rets_batch, axis=1)], axis=1)
        S_np = S0 * np.exp(log_S)
        VS_np = vs_batch * S0 / SIGMA_V_FIXED

        S  = torch.tensor(S_np,  dtype=torch.float32, device=device)
        VS = torch.tensor(VS_np, dtype=torch.float32, device=device)
        return S, VS


def payoff_call_atm(S_T):
    return torch.clamp(S_T - K, min=0.0)


def get_train_sampler():
    train_ret, _, _, _ = load_train_test()
    return HistoricalBootstrap(train_ret)


def get_test_sampler():
    _, test_ret, _, _ = load_train_test()
    return HistoricalBootstrap(test_ret)


def simulate_historical(N=64, device="cpu", sampler=None):
    """Drop-in replacement for simulate_heston / simulate_bates."""
    if sampler is None:
        sampler = get_train_sampler()
    S, VS = sampler.sample_batch(N, horizon=T, device=device)
    return S, VS, payoff_call_atm, {"sampler": "historical_bootstrap"}


if __name__ == "__main__":
    s = get_train_sampler()
    S, VS = s.sample_batch(4)
    print("S  shape:", S.shape, "range:", S.min().item(), S.max().item())
    print("VS shape:", VS.shape, "range:", VS.min().item(), VS.max().item())
    print("First path log-range:", (S[0].log().max() - S[0].log().min()).item())
