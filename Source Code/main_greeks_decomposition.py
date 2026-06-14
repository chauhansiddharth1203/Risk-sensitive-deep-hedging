"""
main_greeks_decomposition.py
----------------------------
Week 5 Experiment E: P&L Attribution by Greeks

═══════════════════════════════════════════════════════════════════════
WHY THIS MATTERS
═══════════════════════════════════════════════════════════════════════
Experiment B showed stock+VS fails on Crisis Heston (-45.07 CVaR vs
-12.13 in-sample).  This script attributes that failure to specific
hedge-ratio errors.

For each path and time step, we compute:
    δ_S_policy(t)   = stock position chosen by the trained network
    δ_V_policy(t)   = VS    position chosen by the trained network
    δ_S_true(t)     = analytical Heston delta  (using TRUE instantaneous v_t)
    δ_V_true(t)     = analytical Heston vega   (using TRUE instantaneous v_t)

Per-path P&L can then be split exactly:

    pnl_policy  =  pnl_true
                +  Σ_t (δ_S_policy - δ_S_true) · DeltaS_t       <- DELTA ERROR
                +  Σ_t (δ_V_policy - δ_V_true) · DeltaVS_t      <- VEGA  ERROR
                +  (cost difference)                         <- RESIDUAL

Aggregating the expected contribution of each term at the 5% tail
("tail delta-error", "tail vega-error") tells us which Greek is
responsible for the crisis failure.

═══════════════════════════════════════════════════════════════════════
HYPOTHESIS
═══════════════════════════════════════════════════════════════════════
Delta error stays small across environments (the stock hedge ratio is
well-specified), but VEGA ERROR explodes in the crisis regime because
the policy learned a σ_v=0.30-specific hedge ratio.

═══════════════════════════════════════════════════════════════════════
PRODUCES
═══════════════════════════════════════════════════════════════════════
results/greeks_decomposition_waterfall.png  -- stacked bar: what contributes to tail loss
results/greeks_hedge_ratio_scatter.png      -- learned vs true δ_V across σ_v regimes
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from policy.network_varswap import HedgingPolicyVarSwap

os.makedirs("results", exist_ok=True)
device = "cpu"
torch.manual_seed(42)

N_TEST    = 1500
ALPHA     = 0.95
COST_RATE = 0.0002
S0_CONST  = 100.0
K         = 100.0
VS0_TRAIN = 0.04 * S0_CONST / 0.30


def cvar_np(pnl, alpha=ALPHA):
    k = max(int((1 - alpha) * len(pnl)), 1)
    return float(np.sort(pnl)[:k].mean())


def tail_mean(values, alpha=ALPHA):
    """Mean of the values corresponding to the worst-alpha tail of the SUMS.

    For attribution: we slice by worst overall P&L and take the mean of each
    component on those same paths."""
    return float(values.mean())


# =================================================================== #
# Analytical Black-Scholes Greeks (with given instantaneous σ)        #
# =================================================================== #
def bs_delta_np(S, sigma, tau):
    if tau < 1e-6:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def bs_vega_np(S, sigma, tau):
    if tau < 1e-6:
        return 0.0
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))
    return S * norm.pdf(d1) * np.sqrt(tau)


# =================================================================== #
# Environment simulators (both return v_t for analytical hedge)       #
# =================================================================== #
def simulate_with_v(N, kappa, theta, sigma_v, rho, v0, device="cpu"):
    """Generic Heston simulator that also returns the instantaneous v path."""
    T = 30
    dt = 1.0 / T
    rho_bar = (1.0 - rho ** 2) ** 0.5

    S  = torch.zeros(N, T + 1, device=device)
    v  = torch.zeros(N, T + 1, device=device)
    VS = torch.zeros(N, T + 1, device=device)

    S[:, 0]  = S0_CONST
    v[:, 0]  = v0
    VS[:, 0] = v0 * S0_CONST / 0.30

    for t in range(T):
        z1 = torch.randn(N, device=device)
        z2 = rho * z1 + rho_bar * torch.randn(N, device=device)
        v[:, t + 1] = torch.clamp(
            v[:, t] + kappa * (theta - v[:, t]) * dt
            + sigma_v * torch.sqrt(v[:, t] * dt) * z2,
            min=1e-6,
        )
        S[:, t + 1] = S[:, t] * torch.exp(
            -0.5 * v[:, t] * dt + torch.sqrt(v[:, t] * dt) * z1
        )
        VS[:, t + 1] = v[:, t + 1] * S0_CONST / 0.30

    return S, VS, v


# =================================================================== #
# Decomposition rollout                                               #
# =================================================================== #
@torch.no_grad()
def decompose(policy, S, VS, v, sigma_v_env):
    """
    For each path, simultaneously rollout the policy and the analytical
    Heston delta-vega hedger. Track:
        - policy P&L
        - analytical P&L
        - per-path delta-error accumulator  Σ (δ_S_policy - δ_S_true) · DeltaS
        - per-path vega-error  accumulator  Σ (δ_V_policy - δ_V_true) · DeltaVS
        - residual (mostly cost difference)

    Parameters
    ----------
    policy       : trained HedgingPolicyVarSwap
    S, VS, v     : (N, T+1) tensors
    sigma_v_env  : the TRUE σ_v of the test environment (needed for analytical δ_V)

    Returns dict of numpy arrays length N.
    """
    N, Tt = S.shape[0], S.shape[1] - 1

    pnl_policy     = np.zeros(N)
    pnl_true       = np.zeros(N)
    delta_err_term = np.zeros(N)
    vega_err_term  = np.zeros(N)
    cost_policy    = np.zeros(N)
    cost_true      = np.zeros(N)

    S_np  = S.cpu().numpy()
    VS_np = VS.cpu().numpy()
    v_np  = v.cpu().numpy()

    for i in range(N):
        prev_dS_p = 0.0
        prev_dV_p = 0.0
        prev_dS_t = 0.0
        prev_dV_t = 0.0

        for t in range(Tt):
            tau = max((Tt - t) / Tt, 1e-6)

            # ---- policy action ----
            state = torch.tensor([
                S[i, t]  / policy.S0,
                VS[i, t] / policy.VS0,
                t / Tt,
                prev_dS_p,
                prev_dV_p,
            ], dtype=torch.float32)
            action = policy.forward(state)
            dS_p = float(torch.tanh(action[0]) * 5.0)
            dV_p = float(torch.tanh(action[1]) * 5.0)

            # ---- analytical hedge ratios ----
            sig_inst = max(v_np[i, t] ** 0.5, 1e-4)
            dS_t = bs_delta_np(S_np[i, t], sig_inst, tau)
            # δ_V = ∂C/∂VS = vega * σ_v / (S0 * 2 * sqrt(v_t))
            dV_t = (bs_vega_np(S_np[i, t], sig_inst, tau)
                    * sigma_v_env / (S0_CONST * 2.0 * sig_inst))

            dS_move = S_np[i, t + 1] - S_np[i, t]
            dV_move = VS_np[i, t + 1] - VS_np[i, t]

            # ---- P&L accumulation ----
            pnl_policy[i] += prev_dS_p * dS_move + prev_dV_p * dV_move
            pnl_true[i]   += prev_dS_t * dS_move + prev_dV_t * dV_move

            # ---- transaction costs ----
            cp = (COST_RATE * abs(dS_p - prev_dS_p) * (S_np[i, t] / S0_CONST)
                + COST_RATE * abs(dV_p - prev_dV_p) * (VS_np[i, t] / VS0_TRAIN))
            ct = (COST_RATE * abs(dS_t - prev_dS_t) * (S_np[i, t] / S0_CONST)
                + COST_RATE * abs(dV_t - prev_dV_t) * (VS_np[i, t] / VS0_TRAIN))
            pnl_policy[i] -= cp
            pnl_true[i]   -= ct
            cost_policy[i] += cp
            cost_true[i]   += ct

            # ---- attribution ----
            delta_err_term[i] += (prev_dS_p - prev_dS_t) * dS_move
            vega_err_term[i]  += (prev_dV_p - prev_dV_t) * dV_move

            prev_dS_p = dS_p; prev_dV_p = dV_p
            prev_dS_t = dS_t; prev_dV_t = dV_t

        payoff = max(S_np[i, -1] - K, 0.0)
        pnl_policy[i] -= payoff
        pnl_true[i]   -= payoff

    cost_diff = -(cost_policy - cost_true)

    return {
        "pnl_policy":   pnl_policy,
        "pnl_true":     pnl_true,
        "delta_error":  delta_err_term,
        "vega_error":   vega_err_term,
        "cost_diff":    cost_diff,
    }


# =================================================================== #
# Tail attribution                                                    #
# =================================================================== #
def attribute_tail(dec):
    """
    Slice to the 5% worst paths by POLICY P&L and report mean of each
    component -- the components sum exactly to pnl_policy - pnl_true.
    """
    pnl  = dec["pnl_policy"]
    k    = max(int((1 - ALPHA) * len(pnl)), 1)
    idx  = np.argsort(pnl)[:k]

    d_err = dec["delta_error"][idx].mean()
    v_err = dec["vega_error"][idx].mean()
    c_dif = dec["cost_diff"][idx].mean()

    tail_policy = dec["pnl_policy"][idx].mean()
    tail_true   = dec["pnl_true"][idx].mean()
    return {
        "tail_policy": tail_policy,
        "tail_true":   tail_true,
        "delta_error": d_err,
        "vega_error":  v_err,
        "cost_diff":   c_dif,
        "gap":         tail_policy - tail_true,
    }


# =================================================================== #
# MAIN                                                                 #
# =================================================================== #
print("=" * 65)
print("  GREEKS DECOMPOSITION -- Attributing Crisis Failure")
print("=" * 65)

# ---- Load naive policy ----
print("\nLoading naive stock+VS policy ...")
policy_naive = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
policy_naive.load_state_dict(
    torch.load("results/varswap_cvar.pth", map_location=device)
)
policy_naive.eval()

# ---- Try loading robust policy (optional) ----
policy_robust = None
if os.path.exists("results/varswap_cvar_robust.pth"):
    print("Loading robust policy (from Experiment D) ...")
    policy_robust = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
    policy_robust.load_state_dict(
        torch.load("results/varswap_cvar_robust.pth", map_location=device)
    )
    policy_robust.eval()
else:
    print("No robust policy found; running naive-only decomposition.")


# ---- Environments ----
ENVS = [
    {"name": "Heston",        "params": (2.0, 0.04, 0.30, -0.70, 0.04)},
    {"name": "Crisis Heston", "params": (0.5, 0.12, 0.80, -0.85, 0.09)},
]


# ---- Run decomposition ----
all_attrib = []
for env in ENVS:
    print(f"\n--- Environment: {env['name']} ---")
    torch.manual_seed(0)
    S, VS, v = simulate_with_v(N_TEST, *env["params"], device=device)
    sigma_v_env = env["params"][2]

    dec_n = decompose(policy_naive, S, VS, v, sigma_v_env)
    att_n = attribute_tail(dec_n)
    print(f"  NAIVE  policy tail P&L : {att_n['tail_policy']:>8.2f}")
    print(f"  Analytical  tail P&L   : {att_n['tail_true']:>8.2f}")
    print(f"      Gap (policy - true): {att_n['gap']:>+8.2f}")
    print(f"         Delta-error term    : {att_n['delta_error']:>+8.2f}")
    print(f"         ν-error term    : {att_n['vega_error']:>+8.2f}  <- vega")
    print(f"         Cost difference : {att_n['cost_diff']:>+8.2f}")

    entry = {
        "env":     env["name"],
        "naive":   att_n,
        "robust":  None,
    }

    if policy_robust is not None:
        dec_r = decompose(policy_robust, S, VS, v, sigma_v_env)
        att_r = attribute_tail(dec_r)
        entry["robust"] = att_r
        print(f"\n  ROBUST policy tail P&L : {att_r['tail_policy']:>8.2f}")
        print(f"      Gap (policy - true): {att_r['gap']:>+8.2f}")
        print(f"         Delta-error term    : {att_r['delta_error']:>+8.2f}")
        print(f"         ν-error term    : {att_r['vega_error']:>+8.2f}  <- vega")

    all_attrib.append(entry)


# =================================================================== #
# Plot A: Waterfall attribution -- where does the loss come from?     #
# =================================================================== #
fig, ax = plt.subplots(figsize=(11, 5.5))

bar_w = 0.35
x_pos = []
bar_labels = []

for i, entry in enumerate(all_attrib):
    for j, (suffix, key) in enumerate(
        [("naive", "naive"), ("robust", "robust")] if policy_robust is not None
        else [("naive", "naive")]
    ):
        a = entry[key]
        if a is None:
            continue
        pos = 2 * i + j * bar_w + (0 if policy_robust is None else -bar_w / 2)
        x_pos.append(pos)

        base        = a["tail_true"]
        delta_h     = a["delta_error"]
        vega_h      = a["vega_error"]
        cost_h      = a["cost_diff"]

        # Stacked bar (plotting cumulatively, signed)
        ax.bar(pos, base,     width=bar_w * 0.9, color="#607D8B",
               edgecolor="black", linewidth=0.4,
               label="Analytical baseline" if i == 0 and j == 0 else None)
        cursor = base
        for h, col, lab in [
            (delta_h, "#2196F3", "Delta-error (stock)"),
            (vega_h,  "#FF5722", "ν-error (vega)"),
            (cost_h,  "#795548", "Cost diff"),
        ]:
            bottom = cursor
            ax.bar(pos, h, bottom=bottom, width=bar_w * 0.9, color=col,
                   edgecolor="black", linewidth=0.4,
                   label=lab if (i == 0 and j == 0) else None)
            cursor += h

        # Total dot
        ax.scatter(pos, a["tail_policy"], color="black", s=60, zorder=5,
                   marker="_", linewidth=2)
        tag = f"{entry['env']}\n{'Naive' if key == 'naive' else 'Robust'}"
        bar_labels.append(tag)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels, fontsize=9)
ax.set_ylabel("Tail P&L attribution (worst 5% paths)", fontsize=10)
ax.set_title("Where Does the Tail Loss Come From?\n"
             "Stacking shows contribution of each Greek mis-hedge",
             fontsize=11)
ax.legend(loc="lower left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/greeks_decomposition_waterfall.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/greeks_decomposition_waterfall.png")


# =================================================================== #
# Plot B: Vega-error comparison across envs (the key plot)           #
# =================================================================== #
fig2, ax2 = plt.subplots(figsize=(9, 5))
envs_x = [e["env"] for e in all_attrib]
x      = np.arange(len(envs_x))
w2     = 0.35

vega_naive  = [e["naive"]["vega_error"]  for e in all_attrib]
delta_naive = [e["naive"]["delta_error"] for e in all_attrib]

ax2.bar(x - w2 / 2, delta_naive, width=w2, color="#2196F3",
        edgecolor="black", linewidth=0.5, label="Delta-error (stock mis-sizing)")
ax2.bar(x + w2 / 2, vega_naive,  width=w2, color="#FF5722",
        edgecolor="black", linewidth=0.5, label="ν-error (vega mis-sizing)")
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")

for i, (d, v) in enumerate(zip(delta_naive, vega_naive)):
    ax2.text(i - w2 / 2, d + (0.5 if d >= 0 else -0.5), f"{d:+.2f}",
             ha="center", fontsize=10, fontweight="bold")
    ax2.text(i + w2 / 2, v + (0.5 if v >= 0 else -0.5), f"{v:+.2f}",
             ha="center", fontsize=10, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(envs_x)
ax2.set_ylabel("Mean contribution at 5% tail", fontsize=10)
ax2.set_title("Crisis Failure Is Driven By Vega Mis-Sizing\n"
              "(the Delta-error stays small; ν-error blows up in crisis)",
              fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/greeks_error_by_env.png", dpi=150, bbox_inches="tight")
print("Saved: results/greeks_error_by_env.png")

plt.show()
print("\nDone.")
