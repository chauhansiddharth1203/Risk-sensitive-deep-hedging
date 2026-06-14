"""
training/trainer_cvar_bates.py
-------------------------------
Train stock-only hedging policy with CVaR annealing on Bates model paths.
"""

import torch
from risk.cvar import cvar
from market.bates import simulate_bates


def train_cvar_bates(
    policy,
    optimizer,
    device="cpu",
    epochs=300,
    batch_size=64,
    print_every=10,
    alpha_start=0.80,
    alpha_end=0.95,
    save_path="results/bates_cvar.pth",
):
    policy.train()

    for epoch in range(epochs):
        alpha = alpha_start + (alpha_end - alpha_start) * epoch / (epochs - 1)

        S, payoff_fn = simulate_bates(N=batch_size, device=device)

        pnls = []
        for i in range(batch_size):
            pnls.append(policy.rollout(S[i], payoff_fn))
        pnls = torch.stack(pnls)

        loss = -cvar(pnls, alpha)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % print_every == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}  alpha={alpha:.3f}  CVaR Loss {loss.item():.4f}")

    torch.save(policy.state_dict(), save_path)
    print(f"Bates CVaR (stock-only) training complete. Saved to {save_path}")
