import torch
import numpy as np


def bootstrap_cvar_difference(
    pnl_deep,
    pnl_baseline,
    alpha=0.95,
    n_bootstrap=1000,
    seed=42,
):
    """
    Bootstrap confidence interval for CVaR difference:
    CVaR(deep) - CVaR(baseline)
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    pnl_deep = pnl_deep.cpu().numpy()
    pnl_base = pnl_baseline.cpu().numpy()

    n = len(pnl_deep)
    diffs = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)

        deep_sample = pnl_deep[idx]
        base_sample = pnl_base[idx]

        deep_q = np.quantile(deep_sample, 1 - alpha)
        base_q = np.quantile(base_sample, 1 - alpha)

        deep_cvar = deep_sample[deep_sample <= deep_q].mean()
        base_cvar = base_sample[base_sample <= base_q].mean()

        diffs.append(deep_cvar - base_cvar)

    diffs = np.array(diffs)

    mean_diff = diffs.mean()
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)

    return mean_diff, ci_lower, ci_upper
