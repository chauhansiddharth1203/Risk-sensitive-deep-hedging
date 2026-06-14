import torch
import torch.nn as nn


class HedgingPolicy(nn.Module):
    """
    Neural network hedging policy with transaction costs
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        cost_rate=0.001,   # <-- unified name
    ):
        super().__init__()

        self.cost_rate = cost_rate

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Conservative initialization
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        return self.net(state)

    def rollout(self, S_path, payoff_fn, return_logs=False):

        device = S_path.device
        T = S_path.shape[0] - 1

        pnl = torch.zeros((), device=device)
        prev_delta = torch.zeros((), device=device)

        # ---- LOGS ----
        deltas = []
        prices = []
        pnl_path = []

        for t in range(T):
            state = torch.tensor(
                [S_path[t], t / T, prev_delta],
                device=device,
            )

            action = self.forward(state)
            delta = torch.tanh(action[0]) * 5

            # store logs
            deltas.append(delta.item())
            prices.append(S_path[t].item())

            pnl += prev_delta * (S_path[t + 1] - S_path[t])
            pnl -= self.cost_rate * torch.abs(delta - prev_delta)

            pnl_path.append(pnl.item())

            prev_delta = delta

        payoff_value = payoff_fn(S_path[-1])
        pnl -= payoff_value

        if return_logs:
            return pnl, prices, deltas, pnl_path

        return pnl