import torch
from market.heston import simulate_heston


def train_variance(
    policy,
    optimizer,
    device="cpu",
    epochs=300,
    batch_size=64,
    print_every=10,
):
    """
    Train deep hedging policy using variance objective
    """

    policy.train()

    for epoch in range(epochs):

        S, payoff_fn = simulate_heston(
            N=batch_size,
            device=device
        )

        pnls = []

        for i in range(batch_size):
            pnl = policy.rollout(
                S[i],
                payoff_fn
            )
            pnls.append(pnl)

        pnls = torch.stack(pnls)

        # ---- Variance loss ----
        loss = pnls.var()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}, Variance Loss {loss.item():.4f}"
            )

    torch.save(
        policy.state_dict(),
        "results/deep_hedge_variance.pth"
    )

    print("Variance training completed.")
