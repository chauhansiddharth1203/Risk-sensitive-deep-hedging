import torch
import torch.nn as nn


class HedgingPolicyLSTM(nn.Module):
    """
    LSTM-based hedging policy.

    Motivation: The Heston model has a *latent* stochastic volatility v_t
    that is not directly observable.  An LSTM can infer v_t from the
    history of stock prices (since v_t drives the vol of dS) and therefore
    make better hedging decisions than an MLP that only sees the current
    price and time.

    Input at each step (3-dim):
        [S_t / S0,   t / T,   prev_delta]

    The LSTM hidden state carries a learned summary of path history, acting
    as an implicit estimate of the latent variance.

    Action:
        delta = tanh(linear(h_t)) * 5,   delta in [-5, 5]

    The rollout keeps the LSTM hidden state (h, c) across timesteps.
    """

    def __init__(
        self,
        input_dim=3,
        hidden_dim=64,
        cost_rate=0.001,
        S0=100.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.cost_rate  = cost_rate
        self.S0         = S0

        self.lstm    = nn.LSTM(input_dim, hidden_dim, batch_first=False)
        self.fc_out  = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def _init_hidden(self, device):
        h = torch.zeros(1, 1, self.hidden_dim, device=device)
        c = torch.zeros(1, 1, self.hidden_dim, device=device)
        return h, c

    def rollout(self, S_path, payoff_fn):
        """
        S_path   : (T+1,)
        payoff_fn: callable
        """
        device = S_path.device
        T = S_path.shape[0] - 1

        pnl        = torch.zeros((), device=device)
        prev_delta = torch.zeros((), device=device)
        h, c       = self._init_hidden(device)

        for t in range(T):
            inp = torch.tensor(
                [S_path[t] / self.S0, t / T, prev_delta.detach()],
                device=device,
            ).view(1, 1, 3)   # (seq=1, batch=1, features=3)

            out, (h, c) = self.lstm(inp, (h, c))
            action  = self.fc_out(out.squeeze())   # shape (1,)
            delta   = torch.tanh(action[0]) * 5.0  # squeeze to scalar

            pnl += prev_delta * (S_path[t + 1] - S_path[t])
            pnl -= self.cost_rate * torch.abs(delta - prev_delta)

            prev_delta = delta

        pnl -= payoff_fn(S_path[-1])
        return pnl
