"""
market/neural_sde.py
---------------------
Quant-GAN style Neural SDE for joint (SPY, VIX) path generation.

Reference: Wiese et al. (2020) "Quant GANs: deep generation of financial
time series" -- Quantitative Finance 20(9).

Architecture
------------
Generator  : LSTM that maps noise z ~ N(0,I) to a sequence of
             (log-return, log-VIX-change) pairs of length T.
             We use a causal architecture so each step depends only on
             past states and new noise input.

Discriminator : LSTM-based critic that scores a full T-step sequence.
                We use Wasserstein loss with gradient penalty (WGAN-GP)
                for stable training without mode collapse.

The generator produces weekly (SPY_return, delta_log_VIX) sequences
of length T=6 (matching the existing VIXBootstrap interface).

After training, the generator can be used as a drop-in replacement for
VIXBootstrap.sample_batch() so the downstream hedging pipeline is
unchanged.

Training data interface
-----------------------
Expects joint weekly (spy_returns, log_vix) arrays of shape (N_obs,).
Returns synthetic episodes of shape (N, T+1) for both SPY and VIX.
"""

import numpy as np
import torch
import torch.nn as nn

T_HORIZON = 6        # weekly rebalances per 30-day window
NOISE_DIM = 16       # latent noise dimension per step
HIDDEN_DIM = 64      # LSTM hidden dimension
S0 = 100.0
VIX0 = 100.0         # VIX normalised to 100 at start of each window


# ------------------------------------------------------------------ #
# Generator                                                           #
# ------------------------------------------------------------------ #

class NSDEGenerator(nn.Module):
    """
    LSTM-based generator for joint (SPY, VIX) weekly sequences.

    At each step t the generator receives:
      - a fresh noise vector z_t ~ N(0, I_{noise_dim})
      - the LSTM hidden state from the previous step

    It outputs:
      - r_t  : SPY log return for step t  (before de-normalising)
      - dv_t : VIX log change for step t

    The output is then post-processed in sample_batch() to convert
    log-returns back to price levels with the same normalisation as
    VIXBootstrap.
    """

    def __init__(self, noise_dim=NOISE_DIM, hidden_dim=HIDDEN_DIM,
                 n_layers=2):
        super().__init__()
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(noise_dim, hidden_dim, n_layers,
                            batch_first=True)
        # Output head: 2 outputs per step (SPY_ret, VIX_change)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, z):
        """
        z : (N, T, noise_dim) noise tensor
        returns : (N, T, 2) raw outputs [spy_ret_scaled, vix_change_scaled]
        """
        out, _ = self.lstm(z)         # (N, T, hidden_dim)
        return self.head(out)         # (N, T, 2)

    def sample_sequences(self, N, T=T_HORIZON, device="cpu"):
        """
        Sample N raw sequences of length T.
        Returns: raw (N, T, 2) tensor [spy_ret_scaled, vix_change_scaled]
        """
        z = torch.randn(N, T, self.noise_dim, device=device)
        with torch.no_grad():
            self.eval()
            out = self.forward(z)
        return out


# ------------------------------------------------------------------ #
# Discriminator (WGAN critic)                                         #
# ------------------------------------------------------------------ #

class NSDECritic(nn.Module):
    """
    LSTM-based Wasserstein critic.
    Scores a full (N, T, 2) sequence as real or fake.
    No sigmoid -- outputs raw scalar score per sequence.
    """

    def __init__(self, input_dim=2, hidden_dim=HIDDEN_DIM, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x : (N, T, 2)
        returns : (N,) scalar scores
        """
        out, (h, _) = self.lstm(x)
        # Use the last hidden state of the top layer
        score = self.head(h[-1])      # (N, 1)
        return score.squeeze(1)       # (N,)


# ------------------------------------------------------------------ #
# Data utilities                                                       #
# ------------------------------------------------------------------ #

def build_real_sequences(spy_prices, vix_levels, T=T_HORIZON):
    """
    Convert raw weekly (spy_price, vix_level) arrays into overlapping
    T-step normalised sequences for GAN training.

    Each window:
      - SPY log returns: r_t = log(S_t / S_{t-1})
      - VIX log changes: v_t = log(VIX_t / VIX_{t-1})

    Returns: (N_windows, T, 2) numpy array.
    """
    N = len(spy_prices)
    sequences = []
    for start in range(0, N - T - 1, 1):
        spy_window = spy_prices[start: start + T + 1]
        vix_window = vix_levels[start: start + T + 1]
        spy_ret = np.diff(np.log(spy_window))       # (T,)
        vix_chg = np.diff(np.log(vix_window + 1e-6))  # (T,)
        sequences.append(np.stack([spy_ret, vix_chg], axis=1))
    return np.array(sequences, dtype=np.float32)    # (N_win, T, 2)


def normalise_sequences(seqs):
    """
    Standardise each channel across the dataset so generator
    outputs are on the same scale as N(0,1) noise.
    Returns (normalised_seqs, mean, std) where mean/std are (2,) arrays.
    """
    # seqs: (N, T, 2)
    flat = seqs.reshape(-1, 2)
    mu  = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8
    return (seqs - mu) / std, mu, std


def denormalise(seqs_norm, mu, std):
    """Invert normalise_sequences."""
    return seqs_norm * std + mu


# ------------------------------------------------------------------ #
# Bootstrap wrapper (VIXBootstrap-compatible interface)               #
# ------------------------------------------------------------------ #

class NSDEBootstrap:
    """
    Drop-in replacement for VIXBootstrap that generates (SPY, VIX)
    paths using a trained NSDEGenerator instead of block-resampling
    real data.

    After __init__, call .sample_batch(N, device) exactly as you would
    call VIXBootstrap.sample_batch().
    """

    def __init__(self, generator, mu, std, device="cpu"):
        """
        generator : trained NSDEGenerator
        mu, std   : (2,) normalisation statistics from build_real_sequences
        """
        self.generator = generator.to(device)
        self.generator.eval()
        self.mu  = torch.tensor(mu,  dtype=torch.float32, device=device)
        self.std = torch.tensor(std, dtype=torch.float32, device=device)
        self.device = device

    def sample_batch(self, N, horizon=T_HORIZON, device=None):
        """
        Returns (S, VS) each of shape (N, horizon+1) on device.
        S[:, 0] = S0 = 100, VS[:, 0] = VIX0 = 100 (normalised per window).
        """
        if device is None:
            device = self.device
        self.generator = self.generator.to(device)
        self.mu  = self.mu.to(device)
        self.std = self.std.to(device)

        z = torch.randn(N, horizon, self.generator.noise_dim, device=device)
        with torch.no_grad():
            raw = self.generator(z)              # (N, T, 2) normalised
        # Denormalise
        raw_dn = raw * self.std + self.mu        # (N, T, 2) log-ret / log-chg

        spy_ret = raw_dn[:, :, 0]               # (N, T)
        vix_chg = raw_dn[:, :, 1]               # (N, T)

        # Reconstruct SPY prices starting at S0 = 100
        log_spy = torch.cat([
            torch.zeros(N, 1, device=device),
            torch.cumsum(spy_ret, dim=1)
        ], dim=1)                                # (N, T+1)
        S = S0 * torch.exp(log_spy)             # (N, T+1)

        # Reconstruct VIX starting at VIX0 = 100
        log_vix = torch.cat([
            torch.zeros(N, 1, device=device),
            torch.cumsum(vix_chg, dim=1)
        ], dim=1)                                # (N, T+1)
        VS = VIX0 * torch.exp(log_vix)          # (N, T+1)

        return S, VS
