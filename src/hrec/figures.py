"""Three figures that carry the story: who wins, where the wins come from, and how
the ranker re-weights its signals for cold-start users."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
BLUE, ORANGE = "#2f6fdd", "#e0823d"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def make_figures(R: dict, out: Path):
    out = Path(out)
    rows = sorted(R["results"], key=lambda r: r["NDCG@10"])
    names = [r["model"] for r in rows]

    # 1. NDCG@10 with CIs, warm vs cold
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.2))
    y = np.arange(len(rows))
    vals = np.array([r["NDCG@10"] for r in rows])
    lo = vals - np.array([r["NDCG@10_ci"][0] for r in rows])
    hi = np.array([r["NDCG@10_ci"][1] for r in rows]) - vals
    colors = [BLUE if "hybrid" in n else "#9aa5b1" for n in names]
    ax.barh(y, vals, color=colors, height=0.6, xerr=[lo, hi], error_kw=dict(ecolor=INK, lw=0.8, capsize=2))
    ax.scatter([r["NDCG@10_cold"] for r in rows], y, color=ORANGE, zorder=3, s=22, label="cold-start users only")
    ax.set_yticks(y, names, color=INK, fontsize=9)
    ax.set_xlabel("NDCG@10 on held-out test item (bars: all users, 95% CI)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_models.png", dpi=180)
    plt.close(fig)

    # 2. ablations
    if R.get("ablations"):
        ab = sorted(R["ablations"].items(), key=lambda kv: kv[1]["delta_vs_full"]["diff"])
        fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(ab) + 1.2))
        y = np.arange(len(ab))
        d = np.array([v["delta_vs_full"]["diff"] for _, v in ab])
        lo = d - np.array([v["delta_vs_full"]["ci_low"] for _, v in ab])
        hi = np.array([v["delta_vs_full"]["ci_high"] for _, v in ab]) - d
        ax.barh(y, d, color=[ORANGE if x < 0 else "#9aa5b1" for x in d], height=0.6,
                xerr=[lo, hi], error_kw=dict(ecolor=INK, lw=0.8, capsize=2))
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(y, [f"without {k}" for k, _ in ab], color=INK, fontsize=9)
        ax.set_xlabel("Change in NDCG@10 vs full hybrid (95% CI)")
        _style(ax)
        fig.tight_layout()
        fig.savefig(out / "fig_ablations.png", dpi=180)
        plt.close(fig)

    # 3. SHAP group importance, warm vs cold (normalised to share of total attribution)
    w, c = R["shap_group_importance"]["warm"], R["shap_group_importance"]["cold"]
    groups = sorted(w, key=lambda g: w[g] + c.get(g, 0))
    ws, cs = sum(w.values()), sum(c.values())
    fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(groups) + 1.2))
    y = np.arange(len(groups))
    ax.barh(y + 0.18, [w[g] / ws for g in groups], height=0.36, color=BLUE, label="warm users")
    ax.barh(y - 0.18, [c.get(g, 0) / cs for g in groups], height=0.36, color=ORANGE, label="cold-start users")
    ax.set_yticks(y, groups, color=INK, fontsize=9)
    ax.set_xlabel("Share of score attribution (mean |blend term + TreeSHAP|, each user's top-20)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_shap_cold_vs_warm.png", dpi=180)
    plt.close(fig)
