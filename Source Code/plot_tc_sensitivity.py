"""
plot_tc_sensitivity.py
----------------------
Plot results from experiments/tc_sensitivity.py

Produces two figures:
  1. CVaR vs transaction cost rate (deep hedge objectives + delta hedge)
  2. CVaR improvement over delta hedge vs transaction cost rate

Run AFTER: python experiments/tc_sensitivity.py
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

CSV_PATH = "results/tc_sensitivity.csv"

COLOURS = {
    "CVaR":     "#E53935",   # red
    "Variance": "#1E88E5",   # blue
    "Entropic": "#43A047",   # green
}
MARKERS = {"CVaR": "o", "Variance": "s", "Entropic": "^"}


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["tc_rate"]    = float(row["tc_rate"])
            row["cvar_deep"]  = float(row["cvar_deep"])
            row["cvar_delta"] = float(row["cvar_delta"])
            row["mean_deep"]  = float(row["mean_deep"])
            row["std_deep"]   = float(row["std_deep"])
            rows.append(row)
    return rows


def group_by_objective(rows):
    data = {}
    for row in rows:
        obj = row["objective"]
        data.setdefault(obj, {"tc": [], "cvar_deep": [], "cvar_delta": [], "improvement": []})
        data[obj]["tc"].append(row["tc_rate"])
        data[obj]["cvar_deep"].append(row["cvar_deep"])
        data[obj]["cvar_delta"].append(row["cvar_delta"])
        data[obj]["improvement"].append(row["cvar_deep"] - row["cvar_delta"])
    return data


def main():
    rows = load_csv(CSV_PATH)
    data = group_by_objective(rows)

    # Use one objective's delta hedge results (they should be identical)
    # Replace tc=0 with a small positive value so log scale works
    ref_obj = list(data.keys())[0]
    tc_vals   = [tc if tc > 0 else 5e-6 for tc in data[ref_obj]["tc"]]
    delta_cvar = data[ref_obj]["cvar_delta"]
    for obj in data:
        data[obj]["tc"] = [tc if tc > 0 else 5e-6 for tc in data[obj]["tc"]]

    # ---- Figure 1: Absolute CVaR ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tc_vals, delta_cvar, color="black", linestyle="--",
            marker="x", label="Delta Hedge (BS)", linewidth=1.5)
    for obj, d in data.items():
        ax.plot(d["tc"], d["cvar_deep"],
                color=COLOURS[obj], marker=MARKERS[obj],
                label=f"Deep Hedge ({obj})", linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xlabel("Transaction Cost Rate")
    ax.set_ylabel(f"CVaR @ {95}%")
    ax.set_title("CVaR vs Transaction Cost Rate by Risk Objective")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/tc_sensitivity_absolute.png", dpi=150)
    print("Saved: results/tc_sensitivity_absolute.png")

    # ---- Figure 2: Improvement over delta ----
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for obj, d in data.items():
        ax2.plot(d["tc"], d["improvement"],
                 color=COLOURS[obj], marker=MARKERS[obj],
                 label=f"{obj}", linewidth=1.5)

    ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax2.set_xscale("symlog", linthresh=1e-5)
    ax2.set_xlabel("Transaction Cost Rate")
    ax2.set_ylabel("CVaR improvement over Delta Hedge")
    ax2.set_title("Deep Hedge Advantage vs Transaction Cost Rate\n(Higher = Better)")
    ax2.legend(title="Risk Objective")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/tc_sensitivity_improvement.png", dpi=150)
    print("Saved: results/tc_sensitivity_improvement.png")

    plt.show()


if __name__ == "__main__":
    main()
