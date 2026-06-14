import torch
from market.heston_with_var_swap import simulate_heston_with_var_swap


def train_variance_varswap(
    policy,
    optimizer,
    device="cpu",
    epochs=300,
    batch_size=64,
    print_every=10,
):
    """
    Train stock+variance-swap hedging policy with variance minimisation objective.
    """

    policy.train()

    for epoch in range(epochs):

        S, VS, payoff_fn = simulate_heston_with_var_swap(
            N=batch_size,
            device=device,
        )

        pnls = []
        for i in range(batch_size):
            pnl = policy.rollout(S[i], VS[i], payoff_fn)
            pnls.append(pnl)

        pnls = torch.stack(pnls)
        loss = pnls.var()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}  Variance Loss {loss.item():.4f}")

    torch.save(policy.state_dict(), "results/varswap_variance.pth")
    print("VarSwap Variance training complete.")
