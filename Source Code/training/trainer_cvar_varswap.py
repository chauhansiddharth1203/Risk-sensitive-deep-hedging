import torch
from risk.cvar import cvar
from market.heston_with_var_swap import simulate_heston_with_var_swap


def train_cvar_varswap(
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
    Train stock+variance-swap hedging policy with CVaR annealing objective.

    We use pure CVaR loss (no mean_pnl term) because adding a mean reward
    with two tradeable instruments causes the network to exploit the variance
    swap speculatively (high mean, poor CVaR) rather than use it for hedging.
    """

    policy.train()

    for epoch in range(epochs):

        alpha = alpha_start + (alpha_end - alpha_start) * epoch / (epochs - 1)

        S, VS, payoff_fn = simulate_heston_with_var_swap(
            N=batch_size,
            device=device,
        )

        pnls = []
        for i in range(batch_size):
            pnl = policy.rollout(S[i], VS[i], payoff_fn)
            pnls.append(pnl)

        pnls = torch.stack(pnls)

        # Pure CVaR objective -- no mean term to prevent speculative VS positions
        loss = -cvar(pnls, alpha)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % print_every == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:3d}  alpha={alpha:.3f}  CVaR Loss {loss.item():.4f}"
            )

    torch.save(policy.state_dict(), "results/varswap_cvar.pth")
    print("VarSwap CVaR training complete.")
