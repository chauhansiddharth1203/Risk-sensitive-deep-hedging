"""
data/spx_windows.py
--------------------
Week 9: Historical SPY data loader for backtest-based training.

Splits daily log-returns into a TRAIN pool and a TEST pool by calendar
date, with strict separation --- no point in the test pool is ever
used for training.

Train: 2005-01-01 -> 2017-12-31   (includes 2008 GFC, 2011 debt-ceiling,
                                  2015 China shock, 2017 calm)
Test : 2018-01-01 -> 2024-12-31   (includes 2018 vol-mageddon, 2020
                                  COVID, 2022 rate shock, 2023 SVB)

The 2008 GFC and 2017 Calm windows used in Week 7--8 are IN the training
period; the 2020 COVID window is OUT. We deliberately keep this
asymmetry so the final table reports both in-period and out-of-period
performance separately.
"""

import os
import numpy as np
import yfinance as yf

TRAIN_START = "2005-01-01"
TRAIN_END   = "2017-12-31"
TEST_START  = "2018-01-01"
TEST_END    = "2024-12-31"

CACHE_FILE = "results/spy_daily_cache.npz"


def download_spy(start, end):
    df = yf.download("SPY", start=start, end=end,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No SPY data between {start} and {end}")
    return df["Close"].values.ravel()


def load_train_test(use_cache=True):
    """Returns (train_returns, test_returns, train_prices, test_prices)."""
    if use_cache and os.path.exists(CACHE_FILE):
        z = np.load(CACHE_FILE)
        return z["train_ret"], z["test_ret"], z["train_px"], z["test_px"]

    os.makedirs("results", exist_ok=True)
    train_px = download_spy(TRAIN_START, TRAIN_END)
    test_px  = download_spy(TEST_START,  TEST_END)

    train_ret = np.diff(np.log(train_px))
    test_ret  = np.diff(np.log(test_px))

    np.savez(CACHE_FILE,
             train_ret=train_ret, test_ret=test_ret,
             train_px=train_px,   test_px=test_px)

    return train_ret, test_ret, train_px, test_px


if __name__ == "__main__":
    tr, te, trp, tep = load_train_test(use_cache=False)
    print(f"Train: {len(tr)} returns from {TRAIN_START} to {TRAIN_END}")
    print(f"  mean={tr.mean():.6f}  std={tr.std():.6f}")
    print(f"Test : {len(te)} returns from {TEST_START} to {TEST_END}")
    print(f"  mean={te.mean():.6f}  std={te.std():.6f}")
