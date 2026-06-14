import torch
import torch.nn as nn


class HedgingPolicyVarSwap(nn.Module):
    """
    Neural network hedging policy that trades in BOTH stock and variance swap.

    Variance swap is scaled as VS_t = v_t * S0 / sigma_v so that both
    instruments have equal per-step P&L standard deviation (~3.65).
    This prevents the network from exploiting scale differences and keeps
    training stable.

    State (5-dim) at each step:
        [S_t / S0,   VS_t / VS0,   t / T,   prev_delta_S,   prev_delta_V]

    Actions (2-dim):
        delta_S = tanh(a[0]) * 5     -- stock position in [-5, 5]
        delta_V = tanh(a[1]) * 5     -- variance swap position in [-5, 5]

    P&L per step:
        gain_S  = prev_delta_S * (S_{t+1} - S_t)
        gain_V  = prev_delta_V * (VS_{t+1} - VS_t)
        tc_S    = cost_rate * |delta_S - prev_delta_S| * S_t / S0
        tc_V    = cost_rate * |delta_V - prev_delta_V| * VS_t / VS0
    """

    def __init__(
        self,
        hidden_dim=64,
        cost_rate=0.001,
        S0=100.0,
    ):
        super().__init__()

        self.cost_rate = cost_rate
        self.S0  = S0
        self.VS0 = 0.04 * S0 / 0.3   # v0 * S0 / sigma_v ≈ 13.33

        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        # Conservative initialisation: start near zero-hedge
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        return self.net(state)

    def rollout(self, S_path, VS_path, payoff_fn, return_logs=False):
        """
        S_path  : (T+1,)  stock price path
        VS_path : (T+1,)  variance swap price path
        payoff_fn: callable
        return_logs: if True, also return per-step trade log dict
        """
        device = S_path.device
        T = S_path.shape[0] - 1

        pnl          = torch.zeros((), device=device)
        prev_delta_S = torch.zeros((), device=device)
        prev_delta_V = torch.zeros((), device=device)

        log = {"t": [], "S": [], "VS": [], "delta_S": [], "delta_V": [],
               "gain_S": [], "gain_V": [], "tc": [], "cum_pnl": []}

        for t in range(T):
            state = torch.stack([
                S_path[t]  / self.S0,
                VS_path[t] / self.VS0,
                torch.tensor(t / T, device=device),
                prev_delta_S,
                prev_delta_V,
            ])

            action  = self.forward(state)
            delta_S = torch.tanh(action[0]) * 5.0
            delta_V = torch.tanh(action[1]) * 5.0

            gain_S = prev_delta_S * (S_path[t + 1] - S_path[t])
            gain_V = prev_delta_V * (VS_path[t + 1] - VS_path[t])
            tc     = (self.cost_rate * torch.abs(delta_S - prev_delta_S) * (S_path[t] / self.S0)
                    + self.cost_rate * torch.abs(delta_V - prev_delta_V) * (VS_path[t] / self.VS0))

            pnl += gain_S + gain_V - tc

            if return_logs:
                log["t"].append(t)
                log["S"].append(S_path[t].item())
                log["VS"].append(VS_path[t].item())
                log["delta_S"].append(delta_S.item())
                log["delta_V"].append(delta_V.item())
                log["gain_S"].append(gain_S.item())
                log["gain_V"].append(gain_V.item())
                log["tc"].append(tc.item())
                log["cum_pnl"].append(pnl.item())

            prev_delta_S = delta_S
            prev_delta_V = delta_V

        pnl -= payoff_fn(S_path[-1])

        if return_logs:
            log["option_payoff"] = payoff_fn(S_path[-1]).item()
            log["final_pnl"]     = pnl.item()
            return pnl, log

        return pnl
