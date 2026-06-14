"""Generate supporting plots for the supervisor-meeting PDF."""
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results/report", exist_ok=True)
plt.rcParams.update({"font.family": "Helvetica", "font.size": 10})

# ------------------------------------------------------------------
# Plot 1: Delta across windows for three major stages
# ------------------------------------------------------------------
windows = ["2008\nGFC", "2017\nCalm", "2018\nVolm.", "2020\nCOVID",
           "2022\nRates", "2023\nSVB", "2024\nOOT"]
w8  = [-25.1, -0.4, -6.3, -22.8, -3.5, -2.1, -0.8]
w9  = [-1.67, -0.06, -1.61, -4.99, -1.80, -0.76, -0.50]
w12 = [-7.79, +0.48, -3.02, -8.05, -2.20, +0.07, +1.16]
w12_err = [3.29, 0.14, 1.41, 4.96, 0.87, 0.27, 0.06]

x = np.arange(len(windows))
w = 0.27

fig, ax = plt.subplots(figsize=(10, 4.5))
b1 = ax.bar(x - w, w8,  w, label="Week 8: simulator-trained",   color="#EF9A9A", edgecolor="black", lw=0.5)
b2 = ax.bar(x,      w9,  w, label="Week 9: real-data trained",   color="#FFE082", edgecolor="black", lw=0.5)
b3 = ax.bar(x + w, w12, w, yerr=w12_err, capsize=4,
            label="Week 12: 5-seed mean +/- std", color="#A5D6A7", edgecolor="black", lw=0.5)
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_xticks(x); ax.set_xticklabels(windows, fontsize=9)
ax.set_ylabel("Delta = CVaR(Deep) - CVaR(BS)   [higher = better]")
ax.set_title("Progression of the deep hedger across three training regimes")
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax.grid(axis="y", alpha=0.25)
ax.set_ylim(-30, 4)
plt.tight_layout()
plt.savefig("results/report/progression.png", dpi=160, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------
# Plot 2: Seed stability -- 2024 vs COVID
# ------------------------------------------------------------------
seeds_2024  = [1.11, 1.15, 1.27, 1.14, 1.13]
seeds_covid = [-4.39, -15.33, -1.67, -11.86, -7.01]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
seeds_x = [0, 1, 2, 3, 4]

ax1.scatter(seeds_x, seeds_2024, s=120, color="#2E7D32", zorder=3, edgecolor="black")
ax1.axhline(np.mean(seeds_2024), color="#2E7D32", lw=1, ls="--",
            label=f"mean = {np.mean(seeds_2024):+.2f}")
ax1.fill_between([-0.5, 4.5],
                 np.mean(seeds_2024) - np.std(seeds_2024),
                 np.mean(seeds_2024) + np.std(seeds_2024),
                 color="#A5D6A7", alpha=0.4, label="+/-1 std")
ax1.axhline(0, color="black", lw=0.8)
ax1.set_xlim(-0.5, 4.5); ax1.set_xticks(seeds_x)
ax1.set_ylim(-0.5, 2.0)
ax1.set_xlabel("seed"); ax1.set_ylabel("Delta on 2024 OOT window")
ax1.set_title("2024 OOT -- robust win\nDelta = +1.16 +/- 0.06")
ax1.legend(loc="lower right", fontsize=9)
ax1.grid(alpha=0.25)

ax2.scatter(seeds_x, seeds_covid, s=120, color="#C62828", zorder=3, edgecolor="black")
ax2.axhline(np.mean(seeds_covid), color="#C62828", lw=1, ls="--",
            label=f"mean = {np.mean(seeds_covid):+.2f}")
ax2.fill_between([-0.5, 4.5],
                 np.mean(seeds_covid) - np.std(seeds_covid),
                 np.mean(seeds_covid) + np.std(seeds_covid),
                 color="#EF9A9A", alpha=0.4, label="+/-1 std")
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xlim(-0.5, 4.5); ax2.set_xticks(seeds_x)
ax2.set_xlabel("seed"); ax2.set_ylabel("Delta on 2020 COVID window")
ax2.set_title("COVID -- unstable loss\nDelta = -8.05 +/- 4.96")
ax2.legend(loc="lower right", fontsize=9)
ax2.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("results/report/seed_stability.png", dpi=160, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------
# Plot 3: Analytical ceiling comparison (Week 3-4)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))
labels = ["DH stock only\n(CVaR loss)", "BS delta", "Analytical\nHeston Delta-ν ceiling",
          "DH stock + VS\n(CVaR loss)"]
vals   = [-21.11, -15.43, -14.68, -11.09]
colors = ["#90CAF9", "#BDBDBD", "#7E57C2", "#66BB6A"]
bars = ax.barh(labels, vals, color=colors, edgecolor="black", lw=0.6)
for bar, v in zip(bars, vals):
    ax.text(v - 0.4, bar.get_y() + bar.get_height()/2,
            f"{v:+.2f}", va="center", ha="right", fontsize=10, fontweight="bold")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("CVaR(95)   [less negative = better]")
ax.set_title("Saturating the Heston Delta-ν ceiling, then surpassing it with a variance-swap leg\n(ATM call, 30 steps; source: weekly_report_weeks2to5.tex)")
ax.set_xlim(-25, 1)
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig("results/report/ceiling.png", dpi=160, bbox_inches="tight")
plt.close()

print("Plots saved in results/report/")
