"""
data/vix_windows.py
--------------------
Sprint 2: Joint SPY + VIX weekly data loader.

The Sprint 1 diagnosis isolated the remaining sim-to-real failure to
the interaction between daily rebalancing and a lagging EWMA variance
proxy. Sprint 2 replaces the EWMA-VS with the VIX index itself, used
as a tradable proxy for 30-day forward realised vol (c.f. VIX futures
SPVXSP; we use ^VIX index directly for simplicity --- this is an
approximation that ignores futures roll cost, and we document it as a
limitation).

Rebalancing moves from daily to weekly. A 30-calendar-day option
window therefore has T = 6 weekly rebalances (Friday close ticks).

Train: 2005-01-01 -> 2020-12-31   (as in Sprint 1, COVID in-training)
Test : 2021-01-01 -> 2024-12-31
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

CACHE_FILE = "results/spy_vix_weekly_cache.npz"

TRAIN_START = "2005-01-01"
TRAIN_END   = "2020-12-31"
TEST_START  = "2021-01-01"
TEST_END    = "2024-12-31"


def download_joint(start, end):
    """Download SPY and ^VIX daily closes, return aligned DataFrame."""
    spy = yf.download("SPY",  start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    vix = yf.download("^VIX", start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    df = pd.concat([spy, vix], axis=1).dropna()
    df.columns = ["SPY", "VIX"]
    # Resample to weekly (Friday close)
    wk = df.resample("W-FRI").last().dropna()
    return wk


def load(use_cache=True):
    if use_cache and os.path.exists(CACHE_FILE):
        z = np.load(CACHE_FILE)
        return (z["train_spy"], z["train_vix"],
                z["test_spy"],  z["test_vix"])
    os.makedirs("results", exist_ok=True)
    tr = download_joint(TRAIN_START, TRAIN_END)
    te = download_joint(TEST_START,  TEST_END)
    np.savez(CACHE_FILE,
             train_spy=tr["SPY"].values, train_vix=tr["VIX"].values,
             test_spy=te["SPY"].values,  test_vix=te["VIX"].values)
    return (tr["SPY"].values, tr["VIX"].values,
            te["SPY"].values, te["VIX"].values)


if __name__ == "__main__":
    tr_s, tr_v, te_s, te_v = load(use_cache=False)
    print(f"Train SPY: {len(tr_s)} weekly obs  range {tr_s.min():.1f}-{tr_s.max():.1f}")
    print(f"Train VIX: {len(tr_v)} weekly obs  range {tr_v.min():.1f}-{tr_v.max():.1f}")
    print(f"Test  SPY: {len(te_s)} weekly obs")
    print(f"Test  VIX: {len(te_v)} weekly obs")
    print(f"\nCOVID peak VIX in train pool: {tr_v.max():.1f}")
    print(f"Post-train peak VIX (test):    {te_v.max():.1f}")
