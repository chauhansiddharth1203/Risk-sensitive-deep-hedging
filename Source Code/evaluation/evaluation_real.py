import torch
from data.load_market_data import load_stock_data
from policy.network import HedgingPolicy

device = "cpu"

def call_payoff(S_T, K=1.0):
    return torch.maximum(S_T - K, torch.tensor(0.0, device=S_T.device))


def evaluate_real(policy):
    S_path = load_stock_data()
    S_path = S_path.to(device)

    
    pnl, prices, deltas, pnl_path = policy.rollout(S_path, call_payoff, return_logs=True)

    initial_price = prices[0]

    pnl_value = pnl.item()
    return_pct = (pnl_value / initial_price) * 100

    print("Real Data PnL:", pnl_value)
    print(f"Return (%): {return_pct:.2f}%")
    print("\nSample Trades:")
    for i in range(min(50, len(deltas))):
        print(f"t={i}, Price={prices[i]:.3f}, Delta={deltas[i]:.3f}, PnL={pnl_path[i]:.3f}")


if __name__ == "__main__":
    policy = HedgingPolicy(3, 1).to(device)
    policy.load_state_dict(torch.load("results/deep_hedge_var_cvar_annealed.pth", map_location=device))
    policy.eval()

    evaluate_real(policy)
    '''import matplotlib.pyplot as plt
    plt.plot(prices, label="Stock Price")
    plt.plot(deltas, label="Delta")
    plt.legend()
    plt.show()
'''