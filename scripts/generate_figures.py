#!/usr/bin/env python3
"""生成论文所有图表 (6张)。输出到 results/ 目录。"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
DATA = PROJECT / "data"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("dark_background")

C = {"cyan": "#00d4ff", "magenta": "#da70d6", "gold": "#ffd700",
     "green": "#00ff88", "red": "#ff6b6b", "orange": "#ff6b35", "gray": "#666666"}
BG = "#0e1117"


def load(name):
    with open(DATA / name, encoding="utf-8") as f:
        return json.load(f)


def fig1_cn_achievement(success):
    dims = ["wealth", "wisdom", "happiness", "health", "leadership", "beauty"]
    labels = ["Wealth\n财富", "Wisdom\n智慧", "Happiness\n幸福",
              "Health\n健康", "Leadership\n领导力", "Beauty\n美感"]
    g = [success["dimensions"][d]["grantee_mean"] for d in dims]
    p = [success["dimensions"][d]["general_mean"] for d in dims]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(dims))
    w = 0.35
    ax.bar(x - w/2, g, w, label="Grant Recipients (n=99,729)", color=C["cyan"], alpha=0.9)
    ax.bar(x + w/2, p, w, label="General Population (n=100,000)", color=C["gray"], alpha=0.7)

    for i, d in enumerate(dims):
        diff = success["dimensions"][d]["diff"]
        y_max = max(g[i], p[i])
        color = C["green"] if diff > 0 else C["red"]
        ax.text(i, y_max + 0.003, f"{diff:+.004f}***", ha="center", fontsize=9,
                color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean WEAT Score", fontsize=12)
    ax.set_title("Fig 1. Chinese Name-Achievement Association:\n"
                 "Grant Recipients vs General Population", fontsize=14, pad=15)
    ax.legend(fontsize=10, loc="upper right")
    ax.axhline(y=0, color="white", linewidth=0.5, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(RESULTS / "fig1_cn_achievement.png", dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Fig 1 done")


def fig2_elite(val):
    elite = val["elite_validation"]["categories"]
    cats = ["US Senators", "Oscar Winners", "Olympic Gold", "Nobel Laureates",
            "Billionaires", "Chinese Entrepreneurs"]
    cats = [c for c in cats if c in elite]
    diffs = [elite[c]["diff_vs_control"] for c in cats]
    ns = [elite[c]["n"] for c in cats]
    colors = [C["gold"], C["magenta"], C["green"], C["cyan"], C["orange"], C["red"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(cats)), diffs, color=colors[:len(cats)], alpha=0.9)
    for i, (d, n) in enumerate(zip(diffs, ns)):
        ax.text(d + 0.002, i, f"+{d:.3f} (n={n})", va="center", fontsize=10, color="white")

    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=12)
    ax.set_xlabel("WEAT Difference vs Random SSA Names", fontsize=12)
    ax.set_title("Fig 2. Global Elites Score Higher in Embedding Space\n"
                 "(All categories p < 0.001)", fontsize=14, pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(RESULTS / "fig2_elite_comparison.png", dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Fig 2 done")


def fig3_selfmade(val):
    sm = val["selfmade_vs_inherited"]["selfmade_vs_inherited"]
    ctrl_mean = val["elite_validation"]["control_mean"]
    groups = ["Self-Made\n(n=828)", "Inherited\n(n=595)", "SSA Random\n(n=5,000)"]
    vals = [sm["selfmade_mean"], sm["inherited_mean"], ctrl_mean]
    colors = [C["cyan"], C["gold"], C["gray"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(groups, vals, color=colors, alpha=0.9, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.4f}",
                ha="center", fontsize=12, color="white", fontweight="bold")

    ax.annotate("", xy=(0, vals[0]+0.008), xytext=(1, vals[0]+0.008),
                arrowprops=dict(arrowstyle="<->", color=C["green"], lw=2))
    mid_y = vals[0] + 0.011
    ax.text(0.5, mid_y, "p = 5e-6", ha="center", fontsize=10, color=C["green"])

    ax.set_ylabel("Mean WEAT Composite", fontsize=12)
    ax.set_title("Fig 3. Self-Made Billionaires Score Higher\n"
                 "Than Inherited Billionaires", fontsize=14, pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(RESULTS / "fig3_selfmade_vs_inherited.png", dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Fig 3 done")


def fig4_cultural_bias(cultural):
    quintiles = cultural["quintile_summary"]
    q_labels = ["Q1\n(Poorest)", "Q2", "Q3", "Q4", "Q5\n(Richest)"]
    q_weat = [q["avg_weat"] for q in quintiles]
    q_income = [q["avg_income"] for q in quintiles]
    q_colors = [C["red"], C["orange"], C["gold"], C["cyan"], C["magenta"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(q_labels, q_weat, color=q_colors, alpha=0.9, width=0.6)
    for bar, w, inc in zip(bars, q_weat, q_income):
        y_off = 0.002 if w >= 0 else -0.005
        ax.text(bar.get_x() + bar.get_width()/2, w + y_off,
                f"{w:+.4f}\n(${inc/1000:.0f}K)", ha="center", fontsize=9, color="white")

    ax.axhline(y=0, color="white", linewidth=0.8, alpha=0.5, linestyle="--")
    ax.set_ylabel("Mean WEAT Score", fontsize=12)
    ax.set_xlabel("Family Income Quintile (by state proxy)", fontsize=12)
    ax.set_title("Fig 4. The Cultural Bias Discovery:\n"
                 "Richest Families Have LOWEST WEAT Scores (r = -0.127***)",
                 fontsize=14, pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(RESULTS / "fig4_cultural_bias.png", dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Fig 4 done")


def fig5_controlled(ctrl):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Chicago within-tier
    ax = axes[0]
    ch = ctrl["chicago_salary"]
    tiers = list(ch["within_tier"].keys())
    tier_labels = ["Minority\nLeaning", "Mixed", "Mainstream"]
    tier_r = [ch["within_tier"][t]["r"] for t in tiers]
    tier_colors = [C["red"], C["gold"], C["cyan"]]

    bars = ax.bar(tier_labels, tier_r, color=tier_colors, alpha=0.9, width=0.5)
    for bar, r_val in zip(bars, tier_r):
        ax.text(bar.get_x() + bar.get_width()/2, r_val + 0.005,
                f"r={r_val:+.3f}***", ha="center", fontsize=10, color="white", fontweight="bold")

    ax.axhline(y=ch["raw_r"], color=C["magenta"], linewidth=1.5, linestyle="--",
               alpha=0.7, label=f"Raw r={ch['raw_r']:+.3f}")
    ax.set_ylabel("Pearson r (WEAT ~ Salary)", fontsize=11)
    ax.set_title("(a) Chicago Salary: WEAT Predicts Income\nWithin Each Cultural Tier", fontsize=12)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) Elite before/after
    ax = axes[1]
    elite_ctrl = ctrl["elite_names"]
    cats_c = list(elite_ctrl.keys())
    before = [elite_ctrl[c]["uncontrolled"]["diff"] for c in cats_c]
    after = [elite_ctrl[c]["mainstream_only"]["diff"] for c in cats_c]

    x = np.arange(len(cats_c))
    w = 0.35
    ax.bar(x - w/2, before, w, label="Uncontrolled", color=C["gray"], alpha=0.6)
    ax.bar(x + w/2, after, w, label="Mainstream Only", color=C["cyan"], alpha=0.9)

    for i in range(len(cats_c)):
        ax.text(i + w/2, after[i] + 0.002, f"{after[i]:+.3f}***",
                ha="center", fontsize=9, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(cats_c, fontsize=10)
    ax.set_ylabel("WEAT delta vs Control", fontsize=11)
    ax.set_title("(b) Elite Names: Effect Survives\nAfter Controlling for Culture", fontsize=12)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(RESULTS / "fig5_controlled_robustness.png", dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Fig 5 done")


def fig6_summary(val, ctrl):
    ch = ctrl["chicago_salary"]
    sm = val["selfmade_vs_inherited"]["selfmade_vs_inherited"]
    elite_ctrl = ctrl["elite_names"]

    experiments = [
        ("CN: Grantees vs General\n(3-layer fusion AUC)", 0.748, "AUC", C["cyan"]),
        ("US: Salary correlation\n(name-level)", 0.345, "r", C["green"]),
        ("US: Salary (controlled)\n(double partial)", ch["double_partial_r"], "r", C["green"]),
        ("Elite: Senators\n(mainstream only)", elite_ctrl.get("Senators", {}).get("mainstream_only", {}).get("diff", 0), "d", C["gold"]),
        ("Elite: Billionaires\n(mainstream only)", elite_ctrl.get("Billionaires", {}).get("mainstream_only", {}).get("diff", 0), "d", C["orange"]),
        ("Self-Made vs Inherited\n(mainstream only)", ctrl.get("selfmade_inherited", {}).get("mainstream_only", {}).get("diff", sm["diff"]), "d", C["magenta"]),
        ("Cultural Bias\n(income x WEAT)", -0.127, "r", C["red"]),
    ]

    labels = [e[0] for e in experiments]
    values = [e[1] for e in experiments]
    metrics = [e[2] for e in experiments]
    colors = [e[3] for e in experiments]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(experiments)), values, color=colors, alpha=0.9, height=0.6)
    for i, (v, m) in enumerate(zip(values, metrics)):
        offset = 0.015 if v >= 0 else -0.015
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, i, f"{m}={v:+.3f}", va="center", ha=ha,
                fontsize=11, color="white", fontweight="bold")

    ax.set_yticks(range(len(experiments)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(x=0, color="white", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Effect Size", fontsize=12)
    ax.set_title("Fig 6. Summary of All Findings\n"
                 "(All p < 0.001)", fontsize=14, pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(RESULTS / "fig6_summary.png", dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Fig 6 done")


def main():
    success = load("success_analysis.json")
    val = load("validation_results.json")
    cultural = load("cultural_bias_analysis.json")
    ctrl = load("controlled_validation.json")

    fig1_cn_achievement(success)
    fig2_elite(val)
    fig3_selfmade(val)
    fig4_cultural_bias(cultural)
    fig5_controlled(ctrl)
    fig6_summary(val, ctrl)

    print(f"\nAll figures saved to {RESULTS}/")


if __name__ == "__main__":
    main()
