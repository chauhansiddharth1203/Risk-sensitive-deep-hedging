import torch
from market.heston import simulate_heston


def train_entropic(
    policy,
    optimizer,
    device="cpu",
    epochs=300,
    batch_size=64,
    print_every=10,
    risk_aversion=1.0,
):
    """
    Train deep hedging policy using entropic risk objective
    """

    policy.train()

    for epoch in range(epochs):

        # ---- Simulate market paths ----
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

        # ---- Entropic risk loss ----
        # L = (1/lambda) * log(E[exp(-lambda * PnL)])
        loss = (1.0 / risk_aversion) * torch.log(
            torch.mean(torch.exp(-risk_aversion * pnls))
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}, "
                f"Entropic Loss {loss.item():.4f}"
            )

    torch.save(
        policy.state_dict(),
        "results/deep_hedge_entropic.pth"
    )

    print("Entropic risk training completed.")
