import torch
from risk.cvar import cvar
from market.heston import simulate_heston


def train(
    policy,
    optimizer,
    device="cpu",
    epochs=300,
    batch_size=64,
    print_every=10,
    alpha_start=0.80,
    alpha_end=0.95,
):
    """
    Train deep hedging policy using CVaR loss
    """

    policy.train()

    for epoch in range(epochs):

        # ---- CVaR confidence annealing ----
        alpha = alpha_start + (alpha_end - alpha_start) * epoch / (epochs - 1)

        # ---- Simulate market paths ----
        S, payoff_function = simulate_heston(
            N=batch_size,
            device=device
        )

        # ---- Safety check ----
        assert callable(payoff_function), "payoff_function must be callable"

        pnls = []

        for i in range(batch_size):
            # IMPORTANT: positional arguments only
            pnl = policy.rollout(
                S[i],
                payoff_function
            )
            pnls.append(pnl)

        pnls = torch.stack(pnls)

        # ---- CVaR loss ----
        
        mean_pnl = pnls.mean()

        loss = -cvar(pnls, alpha) - 1.0 * mean_pnl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}, "
                f"alpha={alpha:.3f}, "
                f"CVaR Loss {loss.item():.4f}"
            )

    torch.save(
        policy.state_dict(),
        "results/deep_hedge_var_cvar_annealed.pth"
    )

    print("Training completed and model saved.")
