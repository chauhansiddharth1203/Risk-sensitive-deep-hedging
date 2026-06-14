"""
training/trainer_cvar_regime_varswap.py
-----------------------------------------
Train stock+VS CVaR policy on regime-switching Heston paths.
"""

import torch
from risk.cvar import cvar
from market.regime_switching_heston_varswap import simulate_regime_switching_heston_varswap


def train_cvar_regime_varswap(
    policy,
    optimizer,
    device="cpu",
    epochs=300,
    batch_size=64,
    print_every=10,
    alpha_start=0.80,
    alpha_end=0.95,
    save_path="results/regime_varswap_cvar.pth",
):
    policy.train()

    for epoch in range(epochs):
        alpha = alpha_start + (alpha_end - alpha_start) * epoch / (epochs - 1)

        S, VS, payoff_fn, _ = simulate_regime_switching_heston_varswap(
            N=batch_size, device=device
        )

        pnls = []
        for i in range(batch_size):
            pnls.append(policy.rollout(S[i], VS[i], payoff_fn))
        pnls = torch.stack(pnls)

        loss = -cvar(pnls, alpha)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % print_every == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}  alpha={alpha:.3f}  CVaR Loss {loss.item():.4f}")

    torch.save(policy.state_dict(), save_path)
    print(f"Regime-switching CVaR (stock+VS) training complete. Saved to {save_path}")
