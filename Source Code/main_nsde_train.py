"""
main_nsde_train.py
-------------------
Step 3 of the post-Week-13 roadmap: train the Neural SDE generator.

WHAT THIS DOES
--------------
Trains a Quant-GAN on real weekly (SPY, VIX) data 2005-2020 (same
training window as Sprint 1 / Week 10 expanded corpus).

After training, the generator can produce unlimited synthetic 30-day
episodes that match the empirical distribution of SPY+VIX co-movement,
including:
  - Fat tails and return clustering (volatility clustering)
  - Negative SPY-VIX correlation (fear gauge)
  - Extreme spikes (COVID-like episodes via interpolation in latent space)

The trained generator is saved to results/nsde_generator.pth and can
be plugged into main_nsde_hedge.py.

DIAGNOSTIC OUTPUTS
------------------
  1. Training loss curve (Wasserstein distance over epochs)
  2. Return distribution comparison: real vs generated (QQ-plot + histogram)
  3. VIX path samples vs real VIX paths
  4. Autocorrelation of |returns| (vol clustering check)
  5. Tail statistics table (5th/1st percentile of weekly returns)

Usage
-----
    python main_nsde_train.py [--epochs N] [--batch-size N]
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from data.vix_windows import load as load_spy_vix
from market.neural_sde import (
    NSDEGenerator, NSDEBootstrap,
    build_real_sequences, normalise_sequences, denormalise, T_HORIZON,
)
from training.trainer_nsde_gan import train_nsde_gan

os.makedirs("results/nsde", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ #
# Diagnostics                                                          #
# ------------------------------------------------------------------ #

def plot_loss_curve(hist, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist["epoch"], hist["w_dist"], color="#1565C0")
    axes[0].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[0].set_xlabel("Generator epoch")
    axes[0].set_ylabel("Wasserstein distance estimate")
    axes[0].set_title("Wasserstein distance (higher = better separation)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(hist["epoch"], hist["loss_g"], color="#C62828")
    axes[1].set_xlabel("Generator epoch")
    axes[1].set_ylabel("Generator loss  -E[D(fake)]")
    axes[1].set_title("Generator loss (lower = better generator)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_return_distribution(real_seqs, fake_seqs, mu, std, path):
    """
    real_seqs, fake_seqs : (N, T, 2) normalised sequences
    """
    real_dn = denormalise(real_seqs, mu, std)
    fake_dn = denormalise(fake_seqs, mu, std)

    spy_real = real_dn[:, :, 0].flatten()
    spy_fake = fake_dn[:, :, 0].flatten()
    vix_real = real_dn[:, :, 1].flatten()
    vix_fake = fake_dn[:, :, 1].flatten()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # SPY returns histogram
    bins_spy = np.linspace(-0.15, 0.15, 60)
    axes[0, 0].hist(spy_real, bins=bins_spy, alpha=0.6,
                    color="#1565C0", label="Real", density=True)
    axes[0, 0].hist(spy_fake, bins=bins_spy, alpha=0.5,
                    color="#E53935", label="Generated", density=True)
    axes[0, 0].set_title("SPY weekly log-return distribution")
    axes[0, 0].set_xlabel("log-return")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # VIX changes histogram
    bins_vix = np.linspace(-0.5, 0.5, 60)
    axes[0, 1].hist(vix_real, bins=bins_vix, alpha=0.6,
                    color="#2E7D32", label="Real", density=True)
    axes[0, 1].hist(vix_fake, bins=bins_vix, alpha=0.5,
                    color="#F57F17", label="Generated", density=True)
    axes[0, 1].set_title("VIX weekly log-change distribution")
    axes[0, 1].set_xlabel("log-change")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # QQ-plot SPY returns
    real_sorted = np.sort(spy_real)
    fake_sorted = np.sort(spy_fake)
    # Align to same length for QQ
    n_min = min(len(real_sorted), len(fake_sorted))
    idx_r = np.linspace(0, len(real_sorted) - 1, n_min, dtype=int)
    idx_f = np.linspace(0, len(fake_sorted) - 1, n_min, dtype=int)
    axes[1, 0].scatter(real_sorted[idx_r], fake_sorted[idx_f],
                       s=3, color="#1565C0", alpha=0.4)
    lims = [min(real_sorted.min(), fake_sorted.min()),
            max(real_sorted.max(), fake_sorted.max())]
    axes[1, 0].plot(lims, lims, "r--", linewidth=1, label="45-degree line")
    axes[1, 0].set_xlabel("Real quantiles")
    axes[1, 0].set_ylabel("Generated quantiles")
    axes[1, 0].set_title("QQ-plot: SPY log-returns")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Autocorrelation of |returns| (vol clustering)
    max_lag = 10
    def autocorr(x, max_lag):
        x = x - x.mean()
        result = [1.0]
        for lag in range(1, max_lag + 1):
            c = np.corrcoef(x[lag:], x[:-lag])[0, 1]
            result.append(c if not np.isnan(c) else 0.0)
        return result

    ac_real = autocorr(np.abs(spy_real), max_lag)
    ac_fake = autocorr(np.abs(spy_fake), max_lag)
    lags = list(range(max_lag + 1))
    axes[1, 1].bar([l - 0.2 for l in lags], ac_real, 0.4,
                   color="#1565C0", alpha=0.7, label="Real |returns|")
    axes[1, 1].bar([l + 0.2 for l in lags], ac_fake, 0.4,
                   color="#E53935", alpha=0.7, label="Generated |returns|")
    axes[1, 1].set_xlabel("Lag (weeks)")
    axes[1, 1].set_ylabel("Autocorrelation")
    axes[1, 1].set_title("Autocorrelation of |returns| -- vol clustering")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def print_tail_stats(real_seqs, fake_seqs, mu, std):
    real_dn = denormalise(real_seqs, mu, std)
    fake_dn = denormalise(fake_seqs, mu, std)
    spy_real = real_dn[:, :, 0].flatten()
    spy_fake = fake_dn[:, :, 0].flatten()

    print("\n  Tail statistics (SPY weekly log-returns):")
    print(f"  {'Stat':<20} {'Real':>10} {'Generated':>12}")
    print("  " + "-" * 44)
    for pct in [1, 5, 25, 50, 75, 95, 99]:
        r = float(np.percentile(spy_real, pct))
        f = float(np.percentile(spy_fake, pct))
        print(f"  {str(pct)+'th pctile':<20} {r:>+10.4f} {f:>+12.4f}")
    print(f"  {'std':<20} {spy_real.std():>10.4f} {spy_fake.std():>12.4f}")
    print(f"  {'skewness':<20} "
          f"{float(((spy_real - spy_real.mean())**3).mean() / spy_real.std()**3):>+10.4f} "
          f"{float(((spy_fake - spy_fake.mean())**3).mean() / spy_fake.std()**3):>+12.4f}")
    print(f"  {'kurtosis':<20} "
          f"{float(((spy_real - spy_real.mean())**4).mean() / spy_real.std()**4):>+10.4f} "
          f"{float(((spy_fake - spy_fake.mean())**4).mean() / spy_fake.std()**4):>+12.4f}")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser(
        description="Train Neural SDE (Quant-GAN) on real SPY+VIX data")
    ap.add_argument("--epochs",     type=int,   default=3000,
                    help="Generator update steps (default 3000)")
    ap.add_argument("--batch-size", type=int,   default=256)
    ap.add_argument("--lr-g",       type=float, default=1e-4)
    ap.add_argument("--lr-d",       type=float, default=1e-4)
    ap.add_argument("--n-critic",   type=int,   default=5)
    ap.add_argument("--lambda-gp",  type=float, default=10.0)
    ap.add_argument("--log-every",  type=int,   default=200)
    ap.add_argument("--seed",       type=int,   default=0)
    args = ap.parse_args()

    # ---- Load training data (2005-2020 same as Sprint 1) ----
    tr_s, tr_v, te_s, te_v = load_spy_vix()
    print(f"Train obs: {len(tr_s)} weekly  "
          f"SPY range [{tr_s.min():.1f}, {tr_s.max():.1f}]  "
          f"VIX range [{tr_v.min():.1f}, {tr_v.max():.1f}]")

    # Verify COVID is in training window
    print(f"Max VIX in training pool: {tr_v.max():.1f}  "
          f"(COVID peak should be ~80)")

    # ---- Train GAN ----
    print("\n" + "=" * 72)
    print("TRAINING NEURAL SDE (QUANT-GAN)")
    print(f"epochs={args.epochs}  batch={args.batch_size}  "
          f"lr_g={args.lr_g}  lr_d={args.lr_d}  "
          f"n_critic={args.n_critic}  lambda_gp={args.lambda_gp}")
    print("=" * 72)

    generator, mu, std, hist = train_nsde_gan(
        tr_s, tr_v,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        n_critic=args.n_critic,
        lambda_gp=args.lambda_gp,
        log_every=args.log_every,
        device=device,
        seed=args.seed,
    )

    # ---- Save model ----
    save_path = "results/nsde/nsde_generator.pth"
    torch.save({
        "state_dict": generator.state_dict(),
        "mu": mu,
        "std": std,
        "hparams": {
            "noise_dim": generator.noise_dim,
            "hidden_dim": generator.hidden_dim,
            "n_layers":  generator.n_layers,
        },
    }, save_path)
    print(f"\nGenerator saved -> {save_path}")

    # ---- Diagnostics ----
    print("\n" + "=" * 72)
    print("DIAGNOSTICS")
    print("=" * 72)

    # Real sequences
    raw_seqs = build_real_sequences(tr_s, tr_v, T=T_HORIZON)
    norm_seqs, _, _ = normalise_sequences(raw_seqs)
    real_tensor = torch.tensor(norm_seqs, dtype=torch.float32, device=device)

    # Generated sequences
    N_eval = min(len(real_tensor), 2000)
    z_eval = torch.randn(N_eval, T_HORIZON,
                         generator.noise_dim, device=device)
    with torch.no_grad():
        generator.eval()
        fake_norm = generator(z_eval).cpu().numpy()
    real_norm = real_tensor[:N_eval].cpu().numpy()

    print_tail_stats(real_norm, fake_norm, mu, std)

    plot_loss_curve(hist, "results/nsde/training_loss.png")
    plot_return_distribution(
        real_norm, fake_norm, mu, std,
        "results/nsde/distribution_comparison.png")

    # ---- Quick sample paths ----
    nsde_boot = NSDEBootstrap(generator, mu, std, device=device)
    S_sample, VS_sample = nsde_boot.sample_batch(6, device=device)
    print("\nSample generated paths (first 6 windows):")
    print("  SPY paths (T+1 steps, normalised to 100):")
    for i in range(3):
        print(f"    path {i}: "
              + " ".join([f"{x:.1f}" for x in S_sample[i].cpu().tolist()]))
    print("  VIX paths:")
    for i in range(3):
        print(f"    path {i}: "
              + " ".join([f"{x:.1f}" for x in VS_sample[i].cpu().tolist()]))

    print("\nNeural SDE training complete.")
    print("Next step: python main_nsde_hedge.py")


if __name__ == "__main__":
    main()
