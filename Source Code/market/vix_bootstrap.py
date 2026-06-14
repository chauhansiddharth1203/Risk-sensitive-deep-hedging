"""
market/vix_bootstrap.py
------------------------
Sprint 2: joint stationary block bootstrap over (SPY, VIX) weekly pairs.

Differences from the Sprint 1 / Week 9 bootstrap:
  - Observation frequency is weekly, not daily (T = 6 rebalances per
    30-day window).
  - The state's second channel is VIX itself, not an EWMA-variance
    proxy, so the policy sees a *forward-looking* vega instrument.
  - We resample (SPY_t, VIX_t) pairs JOINTLY so that the historical
    SPY-VIX co-movement (the "vol-of-vol leverage") is preserved.

Each sampled window is normalised so SPY[0] = 100 and VIX[0] = 100.
The actual VIX level is recoverable from sample meta if needed.
"""

import numpy as np
import torch

from data.vix_windows import load as load_spy_vix

S0            = 100.0
K             = 100.0
T             = 6              # 6 weekly rebalances in a ~30-day window
SIGMA_V_FIXED = 0.30            # unused here but kept for interface parity
MEAN_BLOCK    = 3               # ~3-week blocks (covers one vol cluster)


class VIXBootstrap:
    """
    Joint stationary block bootstrap of (SPY, VIX) weekly observations.
    Each draw returns (S, VS) of shape (N, T+1) with:
        S [:, 0] = 100  (SPY normalised to 100 per window)
        VS[:, 0] = 100  (VIX normalised to 100 per window)
    """

    def __init__(self, spy_series, vix_series, mean_block=MEAN_BLOCK):
        assert len(spy_series) == len(vix_series)
        self.spy = np.asarray(spy_series, dtype=np.float64)
        self.vix = np.asarray(vix_series, dtype=np.float64)
        self.n   = len(self.spy)
        self.p   = 1.0 / mean_block

    def sample_batch(self, N, horizon=T, device="cpu"):
        S_out  = np.empty((N, horizon + 1))
        VS_out = np.empty((N, horizon + 1))

        for i in range(N):
            # Stationary block bootstrap of *indices*
            idx = np.empty(horizon + 1, dtype=np.int64)
            cur = np.random.randint(0, self.n - horizon - 1)
            idx[0] = cur
            for t in range(1, horizon + 1):
                if np.random.random() < self.p:
                    # start new block anywhere valid
                    cur = np.random.randint(0, self.n - 1)
                else:
                    cur = min(cur + 1, self.n - 1)
                idx[t] = cur

            spy_path = self.spy[idx]
            vix_path = self.vix[idx]

            # Convert to log-returns then back, normalised to start at 100.
            spy_ret = np.diff(np.log(spy_path))
            spy_norm = np.concatenate([[S0], S0 * np.exp(np.cumsum(spy_ret))])

            # VIX is mean-reverting level series, not a return series.
            # We preserve the *relative* VIX path per window: rescale so
            # VIX[0] = 100. This keeps spikes proportional.
            vix_norm = 100.0 * vix_path / vix_path[0]

            S_out[i]  = spy_norm
            VS_out[i] = vix_norm

        S  = torch.tensor(S_out,  dtype=torch.float32, device=device)
        VS = torch.tensor(VS_out, dtype=torch.float32, device=device)
        return S, VS


def payoff_call_atm(S_T):
    return torch.clamp(S_T - K, min=0.0)


def get_train_sampler():
    tr_s, tr_v, _, _ = load_spy_vix()
    return VIXBootstrap(tr_s, tr_v)


def get_test_sampler():
    _, _, te_s, te_v = load_spy_vix()
    return VIXBootstrap(te_s, te_v)


if __name__ == "__main__":
    s = get_train_sampler()
    S, VS = s.sample_batch(4)
    print("S shape:", S.shape,
          "range:", S.min().item(), S.max().item())
    print("VS shape:", VS.shape,
          "range:", VS.min().item(), VS.max().item())
    print("first path S :", S[0].tolist())
    print("first path VS:", VS[0].tolist())
