"""
calibration/calibrate_spx.py
------------------------------
Download SPY options from Yahoo Finance and calibrate Heston parameters
to the observed implied volatility surface.

Calibration minimises the root-mean-squared IV error across strikes and
maturities using L-BFGS-B with sensible bounds and a textbook initial guess.

Returns:
    params : dict  {kappa, theta, sigma_v, rho, v0, S0, r, rmse}
"""

import numpy as np
import warnings
from datetime import datetime, timedelta
from scipy.optimize import minimize

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

from calibration.heston_pricer import implied_vol, heston_call_price


# ------------------------------------------------------------------ #
# Data download                                                       #
# ------------------------------------------------------------------ #
def download_spy_options(target_days=30):
    """
    Download SPY near-the-money call options for the expiry closest
    to `target_days` calendar days from today.

    Returns (S0, strikes, T_years, market_ivs, expiry_str).
    Returns None if download fails.
    """
    if not HAS_YFINANCE:
        print("  yfinance not installed. Falling back to published parameters.")
        return None

    try:
        spy  = yf.Ticker("SPY")
        hist = spy.history(period="5d")
        if hist.empty:
            return None

        S0   = float(hist["Close"].iloc[-1])
        r    = 0.05   # approximate risk-free rate

        # Pick the expiry closest to target_days
        today    = datetime.today()
        exp_strs = spy.options
        if not exp_strs:
            return None

        exp_dts  = [datetime.strptime(e, "%Y-%m-%d") for e in exp_strs]
        target   = today + timedelta(days=target_days)
        chosen   = min(exp_dts, key=lambda x: abs((x - today).days - target_days))
        exp_str  = chosen.strftime("%Y-%m-%d")
        T_years  = max((chosen - today).days / 365.0, 1 / 365.0)

        chain = spy.option_chain(exp_str)
        calls = chain.calls.copy()

        # Filter: positive bid, near the money (0.85 < K/S0 < 1.15)
        calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)]
        calls = calls[
            (calls["strike"] > 0.85 * S0) &
            (calls["strike"] < 1.15 * S0)
        ].copy()

        # Mid price; filter wide bid-ask
        calls["mid"] = (calls["bid"] + calls["ask"]) / 2.0
        calls = calls[
            (calls["ask"] - calls["bid"]) / calls["mid"] < 0.30
        ]

        if len(calls) < 4:
            print(f"  Too few options after filtering ({len(calls)}). Falling back.")
            return None

        # Compute implied vols
        ivs = []
        for _, row in calls.iterrows():
            iv = implied_vol(row["mid"], S0, row["strike"], T_years, r)
            ivs.append(iv)

        strikes   = calls["strike"].values
        market_iv = np.array(ivs)

        # Drop NaN IVs
        mask      = ~np.isnan(market_iv)
        strikes   = strikes[mask]
        market_iv = market_iv[mask]

        if len(strikes) < 4:
            return None

        print(f"  Downloaded {len(strikes)} options  "
              f"(SPY={S0:.2f}, expiry={exp_str}, T={T_years*365:.0f}d)")

        return S0, strikes, T_years, market_iv, r, exp_str

    except Exception as e:
        print(f"  Download failed: {e}")
        return None


# ------------------------------------------------------------------ #
# Calibration                                                         #
# ------------------------------------------------------------------ #
def calibrate_heston(S0, strikes, T, market_ivs, r=0.0,
                     n_restarts=3, verbose=True):
    """
    Fit Heston parameters to market implied vols.

    Minimises:  RMSE(model_iv - market_iv)  over all supplied (K, IV) pairs.

    Returns dict with best-fit parameters.
    """

    bounds = [
        (0.10, 10.0),    # kappa
        (0.01,  0.25),   # theta  -- floor at 1% variance (10% vol); ceiling at 25% variance
        (0.05,  2.00),   # sigma_v
        (-0.99, -0.01),  # rho  (force negative leverage)
        (0.001, 0.50),   # v0
    ]

    initial_guesses = [
        [2.0,  0.04, 0.30, -0.70, 0.04],   # textbook Heston
        [1.5,  0.05, 0.50, -0.75, 0.05],   # SPX-like
        [1.0,  0.06, 0.70, -0.60, 0.03],   # high vol-of-vol
    ]

    def objective(params):
        kappa, theta, sigma_v, rho, v0 = params
        errors = []
        for K, iv_mkt in zip(strikes, market_ivs):
            try:
                price    = heston_call_price(S0, K, T, r, v0, kappa, theta, sigma_v, rho)
                iv_model = implied_vol(price, S0, K, T, r)
                if not np.isnan(iv_model):
                    errors.append((iv_model - iv_mkt) ** 2)
            except Exception:
                errors.append(1.0)
        return np.sqrt(np.mean(errors)) if errors else 1.0

    best_result = None
    best_rmse   = np.inf

    for x0 in initial_guesses[:n_restarts]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                objective, x0, bounds=bounds, method="L-BFGS-B",
                options={"maxiter": 300, "ftol": 1e-10},
            )
        if res.fun < best_rmse:
            best_rmse   = res.fun
            best_result = res

    kappa, theta, sigma_v, rho, v0 = best_result.x

    if verbose:
        print(f"\n  Calibrated Heston parameters:")
        print(f"    kappa   = {kappa:.4f}")
        print(f"    theta   = {theta:.4f}")
        print(f"    sigma_v = {sigma_v:.4f}")
        print(f"    rho     = {rho:.4f}")
        print(f"    v0      = {v0:.4f}")
        print(f"    IV RMSE = {best_rmse*100:.2f}%")

    return {
        "kappa":   kappa,
        "theta":   theta,
        "sigma_v": sigma_v,
        "rho":     rho,
        "v0":      v0,
        "S0":      S0,
        "r":       r,
        "rmse":    best_rmse,
    }


# ------------------------------------------------------------------ #
# Fallback: published SPX calibration                                #
# ------------------------------------------------------------------ #
FALLBACK_SPX_PARAMS = {
    "kappa":   1.5,
    "theta":   0.04,
    "sigma_v": 0.50,
    "rho":     -0.75,
    "v0":      0.04,
    "S0":      100.0,
    "r":       0.0,
    "rmse":    None,
    "source":  "Gatheral & Jacquier (2011), typical pre-crisis SPX",
}


def get_calibrated_params(target_days=30, use_fallback_if_fail=True):
    """
    Try to download SPY options and calibrate.
    Falls back to published SPX parameters if download fails.
    """
    data = download_spy_options(target_days=target_days)

    if data is None:
        if use_fallback_if_fail:
            print("  Using published SPX fallback parameters.")
            return FALLBACK_SPX_PARAMS
        else:
            raise RuntimeError("Options download failed and fallback disabled.")

    S0, strikes, T, market_ivs, r, exp_str = data
    params = calibrate_heston(S0, strikes, T, market_ivs, r=r)
    params["expiry"] = exp_str
    return params
