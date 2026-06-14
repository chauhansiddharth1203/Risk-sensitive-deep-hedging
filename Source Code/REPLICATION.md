# Replication Guidelines

This document describes how to reproduce the experiments reported in
`report_weeks2to13_up1.tex` using this repository.

---

## 1. Prerequisites

1. **Python:** 3.10 or newer
2. **Install:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Working directory:** repository root (`Deep-Hedging-Cvar-Heston/`)
4. **Hardware:** CPU sufficient for evaluation; GPU recommended for training
5. **Network:** required for SPY/VIX download scripts (`yfinance`)

---

## 2. Repository modules

| Module | Files | Responsibility |
|--------|-------|----------------|
| `market/heston_with_var_swap.py` | Heston paths + VS scaling | Core simulator |
| `market/bates_with_var_swap.py` | Bates jump-diffusion | Jump extension |
| `market/multi_heston.py` | Two-asset correlated SV | Spread option |
| `market/historical_bootstrap.py` | SPY block bootstrap | Backtest training |
| `market/vix_bootstrap.py` | VIX weekly windows | Real vega instrument |
| `policy/network_varswap.py` | Stock + VS MLP | Default learned policy |
| `policy/network_vix.py` | Stock + VIX MLP | Real-market policy |
| `policy/network_vix_gated.py` | Regime-gated VIX | Crisis-robust policy |
| `training/trainer_cvar_varswap.py` | CVaR training loop | Simulated training |
| `training/trainer_gated.py` | Gated VIX trainer | Regime-gated training |
| `utils/bootstrap_ci.py` | Non-parametric CIs | All headline tables |

---

## 3. Phase A — Initial framework (simulation)

### A.1 Baseline deep hedging

```bash
python main.py
python main_varswap_cvar.py
python main_lstm_cvar.py          # optional LSTM comparison
```

**Validates:** Initial framework recap (stock-only vs stock+VS CVaR).

### A.2 Statistical robustness and jumps

```bash
python main_bootstrap.py
python main_bates.py
```

**Outputs:**
- `bootstrap_ci_plot.png`
- `bates_vs_heston.png`, `bates_comparison.png`

### A.3 Market-realistic parameters

```bash
python main_calibrated.py
python main_regime_switching.py
python main_spx_calibration.py
```

**Outputs:**
- `spx_calib_comparison.png`
- `regime_calm_vs_stress.png`, `regime_comparison.png`
- `spx_iv_fit.png`

---

## 4. Phase B — Benchmarking and generalisation

```bash
python main_heston_benchmark.py
python main_otm_sweep.py
python main_hedge_analysis.py
python main_portfolio_hedge.py
python main_learning_curve.py
python main_risk_comparison.py
python main_path_dependent.py
python main_misspecification.py
python main_robust_training.py
python main_greeks_decomposition.py
```

**Key claim checks:**
| Claim | Script | Expected direction |
|-------|--------|------------------|
| Beats analytical ceiling | `main_heston_benchmark.py` | Stock+VS CVaR > Heston Δ-ν |
| VS helps most at ATM | `main_otm_sweep.py` | Peak improvement near K=100 |
| Barrier needs VS | `main_path_dependent.py` | Stock-only fails on barrier |
| Crisis misspec failure | `main_misspecification.py` | VS worse on crisis Heston |
| DR closes crisis gap | `main_robust_training.py` | Robust VS ≪ naive VS on crisis |

---

## 5. Phase C — Multi-asset and ablations

```bash
python main_multi_asset.py
python main_ablation_mean_cvar.py
python main_ablation_design.py
python main_historical_stress.py
```

**Outputs:**
- `results/multi_asset_pnl.png`
- `results/multi_asset_cvar_by_corr.png`
- `results/ablation_mean_cvar_frontier.png`
- `results/ablation_design_*.png`
- `results/historical_stress_cvar.png`

**Note:** Historical stress is the first honest negative on real SPX data.

---

## 6. Phase D — Real-market programme

Run in this order to mirror the report narrative:

```bash
# Sim-to-real negative
python main_sim_to_real.py
python main_sim_to_real_earlystop.py   # optional follow-up

# Backtest training (partial positive)
python main_backtest_training.py --epochs 400 --lam 0.5 --N 256

# Expanded corpus (regime-coverage hypothesis fails)
python main_expanded_corpus.py

# VIX futures
python main_vix_futures.py             # naive (expected to fail)
python main_vix_futures_v2.py          # corrected leverage

# Multi-seed verification
python main_vix_multiseed.py

# Regime-gated architecture (capstone)
python main_vix_gated_multiseed.py --thr-init 3.0 --freeze-gate

# Optional post-report extensions
python main_vix_mom.py
python main_vix_mom_multiseed.py
python main_gate_threshold_validation.py
python main_vix_with_costs.py
python main_nsde_train.py
python main_nsde_hedge.py
python main_lstm_gate.py
```

### Real-market data splits

| Study | Training pool | Test / eval windows |
|-------|---------------|---------------------|
| Backtest training | SPY 2005–2017 | 2008, 2017, 2018, 2020, 2022 |
| Expanded corpus | SPY 2005–2020 | 2021–2024 + crisis windows |
| VIX studies | SPY+VIX 2005–2020 | Same seven-window panel |

Cached data is written under `data/` after first download.

---

## 7. Figure regeneration

```bash
python make_report_plots.py
python make_threshold_sweep_plot.py
```

Consolidated report figures live in `results/` and `results/report/`.
**Do not rename image files** — the LaTeX report references them directly.

---

## 8. Minimum replication (≈ 2–4 hours on CPU)

If you only need to verify the thesis claims:

```bash
pip install -r requirements.txt
python main_heston_benchmark.py
python main_misspecification.py
python main_robust_training.py
python main_backtest_training.py --epochs 200 --lam 0.5
python main_vix_futures_v2.py
python main_vix_gated_multiseed.py --thr-init 3.0 --freeze-gate --epochs 200
```

---

## 9. Troubleshooting

| Issue | Fix |
|-------|-----|
| `yfinance` download fails | Retry; check network; delete stale cache in `data/` |
| CUDA OOM | Add `--device cpu` where supported, or reduce `--N` |
| VIX training oscillates | Use `main_vix_futures_v2.py` (tighter action bounds, lower lr) |
| Different CVaR decimals | Expected across PyTorch versions; check ranking not exact match |
| Missing `results/` images | Run the corresponding `main_*.py` first |

---

## 10. Mapping to report sections

| Report section | Primary scripts |
|----------------|-----------------|
| Statistical Robustness and Jump-Diffusion | `main_bootstrap.py`, `main_bates.py` |
| Market-Realistic Parameterisation | `main_calibrated.py`, `main_regime_switching.py` |
| Benchmarking, Portfolios, Interpretability | `main_heston_benchmark.py`, `main_otm_sweep.py`, `main_hedge_analysis.py`, `main_portfolio_hedge.py` |
| Generalisation and Robust Training | `main_risk_comparison.py`, `main_path_dependent.py`, `main_misspecification.py`, `main_robust_training.py`, `main_greeks_decomposition.py` |
| Multi-Asset Spread Option | `main_multi_asset.py` |
| Ablations, Robustness, and Theory | `main_ablation_mean_cvar.py`, `main_ablation_design.py`, `main_historical_stress.py` |
| Sim-to-Real Transfer | `main_sim_to_real.py` |
| Backtest-Based Training | `main_backtest_training.py` |
| Expanded Real-Market Corpus | `main_expanded_corpus.py` |
| VIX Futures | `main_vix_futures_v2.py` |
| Multi-Seed Verification | `main_vix_multiseed.py` |
| Regime-Gated Architecture | `main_vix_gated_multiseed.py`, `main_gate_threshold_validation.py` |

---

## 11. Reporting results

Each `main_*.py` script prints CVaR tables with bootstrap confidence intervals to stdout and saves figures to `results/`. Compare against the tables in `report_weeks2to13_up1.tex`. Seed-level multi-seed studies should reproduce the same **ranking** of windows (calm win, crisis loss/closure) even if point estimates differ slightly.
