# Risk-Sensitive Deep Hedging under Stochastic Volatility with Variance Instruments

**Author:** Siddharth Chauhan (CH21B103)
**Institution:** Wadhwani School of Data Science and AI, Indian Institute of Technology Madras
**Project Guide:** Dr. Nandan Sudarsanam
**Project Coordinator:** Mr. Aakash Sarraf (Principal Project Officer)
**Submission:** Final DDP Report and Presentation, June 2026

---

## Project Overview

This project investigates whether a learned deep hedging policy can outperform analytical option hedging benchmarks under stochastic volatility, while remaining robust in real markets.

The central question is: can a CVaR-trained neural network policy, given access to a variance swap as a second hedging instrument, surpass the closed-form Heston delta-vega ceiling, and continue to perform reliably on real SPY data including crisis windows?

The project answers this in three parts. First, on Heston-simulated paths, the CVaR plus variance swap policy beats the analytical Heston delta-nu ceiling by +3.59 CVaR (a 24% improvement). Second, on real SPY data, simulation-only training fails on crisis windows, which is documented as an honest negative result. Third, a regime-gated architecture using normalised VIX as the gating signal preserves the calm-market win and closes the 2020 COVID gap (from -8.05 to -1.18), satisfying pre-registered decision rules across all five random seeds.

The repository contains the full source code, all experimental scripts, the final report PDF, the project presentation, and the recorded walkthrough videos.

---

## Features

* **CVaR-based deep hedging framework**: feed-forward and LSTM policies trained directly on the tail-risk objective with alpha-annealing from 0.80 to 0.95
* **Variance swap as a second instrument**: linear vega exposure that breaks the incompleteness of stock-only hedging under stochastic volatility
* **Analytical benchmarks**: Black-Scholes delta hedge and the Heston delta-vega analytical ceiling, both reproduced from first principles
* **Path-dependent payoffs**: barrier, straddle, strangle, and call portfolios
* **Multi-asset spread option**: two correlated Heston assets with four tradable instruments
* **Misspecification stress tests**: Crisis Heston, Bates jump diffusion, SPX calibrated parameters
* **Domain randomisation**: training over randomised Heston parameters to cure crisis misspecification
* **Greeks decomposition**: P&L attribution that diagnoses nu-error (vega mismatch) as the root cause of failure
* **Real market transfer**: sim-to-real, backtest-based training on resampled SPY returns, expanded corpus regime testing
* **VIX futures policies**: corrected leverage adaptation as an alternative tradable vega instrument
* **Multi-seed verification**: five-seed reproducibility checks with bootstrap confidence intervals
* **Regime-gated architecture**: sigmoid VIX-threshold gate that closes the COVID crisis gap with low cross-seed variance
* **Reproducibility**: deterministic seeds, configurable hyperparameters, complete replication guide

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10 or newer |
| Deep learning | PyTorch 2.x |
| Numerical | NumPy, SciPy, pandas |
| Visualisation | matplotlib |
| Market data | yfinance |
| Calibration | SciPy optimisation, custom Heston characteristic-function pricer |
| Reporting | python-pptx for the presentation |
| Version control | Git, GitHub |
| Notebook prototyping | Jupyter (optional) |

There is no frontend, no backend service, and no database for this project. It is a research codebase that produces figures, metrics, and trained model checkpoints.

---

## Repository Structure

```
Deep-Hedging-Cvar-Heston-Submission/
|
|-- README.md                                  This file
|-- SETUP.md                                   Installation and run instructions
|-- DDP Final Report CH21B103.pdf              Final submitted report (56 pages)
|-- DDP Final Presentation CH21B103.pptx       Project presentation (17 slides)
|-- .gitignore                                 Excludes large media from git push
|
|-- Videos/                                    Pointer to the Drive folder
|   `-- README.md                              Drive links for both walkthrough videos
|
`-- Source Code/                               All project source code
    |
    |-- requirements.txt                       Pinned Python dependencies
    |-- REPLICATION.md                         Detailed experiment-to-script map
    |
    |-- main_*.py                              One entry script per experiment
    |-- make_report_plots.py                   Regenerates all result figures
    |-- make_threshold_sweep_plot.py           Threshold sweep plot for regime gate
    |-- plot_*.py                              Standalone plotting utilities
    |-- compare_varswap.py                     Variance swap comparison utility
    |
    |-- analysis/                              Post-run analysis helpers
    |-- assets/                                Logos and report assets
    |-- baselines/                             Black-Scholes delta and Heston delta-vega
    |-- calibration/                           SPX implied-volatility surface calibration
    |-- data/                                  Cached SPY/VIX data, NIFTY pipeline
    |-- evaluation/                            CVaR, bootstrap CI, P&L attribution
    |-- experiments/                           Higher-level orchestration helpers
    |-- instruments/                           Variance swap and VIX futures payoff models
    |-- market/                                Simulators: Heston, Bates, multi-asset, VIX
    |-- policy/                                Neural hedging policies
    |-- results/                               Generated plots, metrics, checkpoints
    |-- risk/                                  CVaR and other risk functionals
    |-- training/                              Training loops
    `-- utils/                                 Bootstrap CIs and helpers
```

The submission package is organised so that the four department-required deliverables (Source Code, Project Report PDF, Project Presentation, Videos) plus the two required Markdown files (README.md, SETUP.md) are visible at the top level. All executable code lives under the `Source Code/` folder.

---

## Screenshots

Key result figures are stored under `Source Code/results/`. The most representative ones are:

| Figure | File | What it shows |
|--------|------|---------------|
| Core benchmark | `Source Code/results/heston_benchmark.png` | Stock+VS deep hedger beats the analytical Heston delta-vega ceiling by +3.59 CVaR |
| Path-dependent | `Source Code/results/path_dependent_cvar.png` | Barrier option: stock-only fails, stock+VS recovers |
| Misspecification | `Source Code/results/misspecification_cvar.png` | Crisis Heston: naive policy CVaR collapses to -382 |
| Greeks waterfall | `Source Code/results/greeks_decomposition_waterfall.png` | Nu-error (vega) attributed as sole cause of failure |
| Domain randomisation cure | `Source Code/results/robust_training_cvar.png` | Crisis CVaR recovers from -382 to -38 |
| Multi-asset | `Source Code/results/multi_asset_cvar_by_corr.png` | Stock+VS improves CVaR across all cross-asset correlations |
| Backtest training | `Source Code/results/backtest_training_cvar.png` | Backtest-based training across seven historical windows |

To regenerate every figure from scratch, see SETUP.md section 5.

---

## Project Status

**Completed.** Final report and presentation submitted June 2026. The repository is preserved as the deliverable.

---

## Quick Start

For a minimal first run that reproduces the headline simulated result:

```
git clone https://github.com/chauhansiddharth1203/Risk-sensitive-deep-hedging.git
cd Risk-sensitive-deep-hedging
cd "Source Code"
python -m venv venv
venv\Scripts\activate         (Windows)
# source venv/bin/activate    (macOS or Linux)
pip install -r requirements.txt
python main_heston_benchmark.py
```

Output `results/heston_benchmark.png` should show the Stock+VS bar above the analytical Heston ceiling.

Full installation and the complete experiment run order are in **SETUP.md** at the repository root. The full experiment-to-script map and recommended phased run sequence are in **Source Code/REPLICATION.md**.

---

## Key Experiments

Each scientific claim in the report has a dedicated entry script. All scripts live in `Source Code/` and must be run from inside that folder (`cd "Source Code"` first).

| Scientific claim | Script | Main output |
|------------------|--------|-------------|
| Stock+VS beats analytical Heston ceiling | `main_heston_benchmark.py` | `heston_benchmark.png` |
| Bootstrap significance (six configs, p < 0.05) | `main_bootstrap.py` | `bootstrap_ci_plot.png` |
| Bates jump-diffusion robustness | `main_bates.py` | `bates_vs_heston.png` |
| OTM strike sweep (K = 85 to 110) | `main_otm_sweep.py` | `otm_sweep_improvement.png` |
| Portfolio (call, straddle, strangle) | `main_portfolio_hedge.py` | `portfolio_cvar_comparison.png` |
| Path-dependent barrier | `main_path_dependent.py` | `path_dependent_cvar.png` |
| Misspecification failure | `main_misspecification.py` | `misspecification_cvar.png` |
| Greeks decomposition diagnosis | `main_greeks_decomposition.py` | `greeks_decomposition_waterfall.png` |
| Domain randomisation cure (10x recovery) | `main_robust_training.py` | `robust_training_cvar.png` |
| Multi-asset spread option | `main_multi_asset.py` | `results/multi_asset_*.png` |
| Historical stress (honest negative) | `main_historical_stress.py` | `historical_stress_cvar.png` |
| Backtest-based training (partial positive) | `main_backtest_training.py` | `backtest_training_*.png` |
| Expanded corpus falsification | `main_expanded_corpus.py` | `expanded_corpus_cvar.png` |
| VIX futures with corrected leverage | `main_vix_futures_v2.py` | `vix_futures_v2_cvar.png` |
| Multi-seed verification | `main_vix_multiseed.py` | `vix_multiseed_delta.png` |
| Regime-gated architecture (capstone) | `main_vix_gated_multiseed.py` | Stdout metrics and optional plots |

See **Source Code/REPLICATION.md** for the phased run order and per-phase outputs.

---

## Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| Transaction cost | 2 basis points (`0.0002`) |
| CVaR level | 95% |
| Alpha annealing | 0.80 to 0.95 |
| Optimiser | Adam, learning rate 3e-4 |
| Batch size | 256 (512 for some VIX runs) |
| Hedging horizon | 30 steps in simulation, 30 days in real market |
| Heston parameters (in-distribution) | kappa=2, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04 |
| Regime gate threshold | theta=3.0 (frozen), width=0.3 |

A full hyperparameter reference table is included in the report appendix.

---

## Citation

If you use this code, please cite the DDP report:

> Siddharth Chauhan (2026). *Risk-Sensitive Deep Hedging under Stochastic Volatility with Variance Instruments.* Dual Degree Project Report, IIT Madras.

Core reference for the deep hedging framework:

> Buehler, H., Gonon, L., Teichmann, J., and Wood, B. (2019). Deep hedging. *Quantitative Finance*, 19(8), 1271 to 1291.

---

## License and Contact

Academic research code associated with an IIT Madras Dual Degree Project.

**Contact:** ch21b103@smail.iitm.ac.in
