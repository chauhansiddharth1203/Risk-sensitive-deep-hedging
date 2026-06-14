import torch
from risk.cvar import cvar
from market.heston import simulate_heston


def train_cvar_lstm(
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
    Train LSTM hedging policy with CVaR annealing objective.
    Identical loss to trainer_cvar.py but uses the LSTM rollout.
    """

    policy.train()

    for epoch in range(epochs):

        alpha = alpha_start + (alpha_end - alpha_start) * epoch / (epochs - 1)

        S, payoff_fn = simulate_heston(N=batch_size, device=device)

        pnls = []
        for i in range(batch_size):
            pnl = policy.rollout(S[i], payoff_fn)
            pnls.append(pnl)

        pnls = torch.stack(pnls)
        loss = -cvar(pnls, alpha) - 1.0 * pnls.mean()

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping -- LSTM gradients can explode
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % print_every == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:3d}  alpha={alpha:.3f}  CVaR Loss {loss.item():.4f}"
            )

    torch.save(policy.state_dict(), "results/lstm_cvar.pth")
    print("LSTM CVaR training complete.")
