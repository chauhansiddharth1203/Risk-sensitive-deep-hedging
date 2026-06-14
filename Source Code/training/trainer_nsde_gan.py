"""
training/trainer_nsde_gan.py
-----------------------------
WGAN-GP training loop for the NSDEGenerator / NSDECritic pair.

Loss (Wasserstein with gradient penalty, Gulrajani et al. 2017):
  L_D = E[D(fake)] - E[D(real)] + lambda_gp * GP
  L_G = -E[D(fake)]

where GP = E[(||grad_D(interpolated)||_2 - 1)^2].

Training schedule:
  - n_critic steps of discriminator per 1 generator step
    (standard WGAN practice; default n_critic = 5)
  - Gradient clipping on generator only

Diagnostic outputs every `log_every` epochs:
  - Wasserstein distance estimate (D(real) - D(fake))
  - Generator loss
  - Sample statistics of generated sequences vs real data
"""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim

from market.neural_sde import (
    NSDEGenerator, NSDECritic,
    build_real_sequences, normalise_sequences, denormalise,
    T_HORIZON,
)


def gradient_penalty(critic, real, fake, device):
    """
    Compute WGAN-GP gradient penalty.
    real, fake : (N, T, 2)
    """
    N = real.size(0)
    alpha = torch.rand(N, 1, 1, device=device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    score = critic(interpolated)
    grad = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_norm = grad.reshape(N, -1).norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty


def train_nsde_gan(
    spy_prices, vix_levels,
    epochs=2000,
    batch_size=256,
    lr_g=1e-4,
    lr_d=1e-4,
    n_critic=5,
    lambda_gp=10.0,
    log_every=200,
    device=None,
    seed=0,
):
    """
    Train the Quant-GAN on real (spy_prices, vix_levels) weekly data.

    Parameters
    ----------
    spy_prices  : (N_obs,) numpy array of SPY weekly close prices
    vix_levels  : (N_obs,) numpy array of VIX weekly close levels
    epochs      : number of generator update steps
    batch_size  : real and fake samples per step
    lr_g, lr_d  : learning rates
    n_critic    : critic (discriminator) steps per generator step
    lambda_gp   : gradient penalty weight
    log_every   : print diagnostics every N generator steps

    Returns
    -------
    generator   : trained NSDEGenerator
    mu, std     : (2,) normalisation statistics for denormalisation
    hist        : dict with training history
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build real sequences
    raw_seqs = build_real_sequences(spy_prices, vix_levels, T=T_HORIZON)
    norm_seqs, mu, std = normalise_sequences(raw_seqs)
    real_tensor = torch.tensor(norm_seqs, dtype=torch.float32, device=device)
    N_real = len(real_tensor)
    print(f"Real sequences: {N_real}  shape {real_tensor.shape}  "
          f"mean={real_tensor.mean().item():.3f}  "
          f"std={real_tensor.std().item():.3f}")

    generator = NSDEGenerator().to(device)
    critic    = NSDECritic().to(device)

    opt_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
    opt_d = optim.Adam(critic.parameters(),    lr=lr_d, betas=(0.5, 0.9))

    hist = {"epoch": [], "w_dist": [], "loss_g": []}
    global_step = 0

    for epoch in range(epochs):
        # ---- n_critic steps for critic ----
        for _ in range(n_critic):
            # Real batch
            idx = np.random.randint(0, N_real, batch_size)
            real_batch = real_tensor[idx]

            # Fake batch
            z = torch.randn(batch_size, T_HORIZON,
                            generator.noise_dim, device=device)
            fake_batch = generator(z).detach()

            score_real = critic(real_batch)
            score_fake = critic(fake_batch)
            gp = gradient_penalty(critic, real_batch, fake_batch, device)

            loss_d = score_fake.mean() - score_real.mean() + lambda_gp * gp
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

        # ---- 1 generator step ----
        z = torch.randn(batch_size, T_HORIZON,
                        generator.noise_dim, device=device)
        fake_batch = generator(z)
        loss_g = -critic(fake_batch).mean()

        opt_g.zero_grad()
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_g.step()

        global_step += 1

        if (epoch + 1) % log_every == 0 or epoch == 0:
            with torch.no_grad():
                # Wasserstein distance estimate
                idx2 = np.random.randint(0, N_real, 1024)
                z2 = torch.randn(1024, T_HORIZON,
                                 generator.noise_dim, device=device)
                w_dist = (critic(real_tensor[idx2]).mean()
                          - critic(generator(z2)).mean()).item()

                # Sample statistics
                fake_sample = generator(z2)
                fake_dn = denormalise(
                    fake_sample.cpu().numpy(),
                    mu, std)
                real_dn = denormalise(
                    real_tensor[idx2].cpu().numpy(),
                    mu, std)
                spy_ret_real_std = real_dn[:, :, 0].std()
                spy_ret_fake_std = fake_dn[:, :, 0].std()
                vix_chg_real_std = real_dn[:, :, 1].std()
                vix_chg_fake_std = fake_dn[:, :, 1].std()

            hist["epoch"].append(epoch + 1)
            hist["w_dist"].append(w_dist)
            hist["loss_g"].append(loss_g.item())
            print(f"ep {epoch+1:5d}  W={w_dist:+.4f}  "
                  f"L_G={loss_g.item():+.4f}  "
                  f"spy_ret std real={spy_ret_real_std:.4f} "
                  f"fake={spy_ret_fake_std:.4f}  "
                  f"vix_chg std real={vix_chg_real_std:.4f} "
                  f"fake={vix_chg_fake_std:.4f}")

    return generator, mu, std, hist
