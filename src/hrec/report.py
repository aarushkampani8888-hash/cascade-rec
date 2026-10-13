"""Write RESULTS.md and figures from a metrics.json-style dict."""
from __future__ import annotations

from pathlib import Path


def write_report(R: dict, out: Path):
    out = Path(out)
    L = [f"# Results — {R['data']['dataset']}", "",
         f"{R['data']['users']} users, {R['data']['items']} items, {R['data']['interactions']} interactions; "
         f"{R['data']['cold_users']} users in the simulated cold-start cohort (5 known interactions at tune time, 6 at test).", "",
         "Leave-last-out, full-catalog ranking (no sampled negatives), test item untouched until the end.",
         "95% CIs from 2,000 bootstrap resamples over users.", "",
         "| Model | Recall@10 | NDCG@10 [95% CI] | NDCG@10 warm | NDCG@10 cold | Recall@20 | MRR | Coverage@10 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in sorted(R["results"], key=lambda r: r["NDCG@10"]):
        lo, hi = r["NDCG@10_ci"]
        L.append(f"| {r['model']} | {r['Recall@10']:.4f} | {r['NDCG@10']:.4f} [{lo:.4f}, {hi:.4f}] | "
                 f"{r['NDCG@10_warm']:.4f} | {r['NDCG@10_cold']:.4f} | {r['Recall@20']:.4f} | {r['MRR']:.4f} | "
                 f"{r['Coverage@10']:.3f} |")
    L += ["", "## Is the hybrid actually better? (paired bootstrap, NDCG@10)", "",
          "| vs. | hybrid − model (all) | 95% CI | P(diff ≤ 0) | hybrid − model (cold) | 95% CI |",
          "|---|---|---|---|---|---|"]
    for r in R["results"]:
        if "hybrid_minus_this" in r:
            d, c = r["hybrid_minus_this"], r["hybrid_minus_this_cold"]
            L.append(f"| {r['model']} | {d['diff']:+.4f} | [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] | {d['p_le_0']:.3f} "
                     f"| {c['diff']:+.4f} | [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] |")
    c = R["candidates"]
    L += ["", "## Retrieval stage", "",
          f"Union of retrievers gives {c['mean_per_user']:.0f} candidates/user; the test item is in the set for "
          f"{c['recall_union']:.1%} of users ({c['recall_union_cold']:.1%} of cold users). That's the ceiling for the ranker.", "",
          "| Retriever | top-N | recall@N |", "|---|---|---|"]
    for n, v in c["recall_per_retriever"].items():
        L.append(f"| {n} | {R['config']['n_candidates'][n]} | {v:.3f} |")
    if R.get("ablations"):
        L += ["", "## Ablations (drop one source's candidates and features, retrain ranker)", "",
              "| Dropped | NDCG@10 | Δ vs full [95% CI] | NDCG@10 cold |", "|---|---|---|---|"]
        for k, v in R["ablations"].items():
            d = v["delta_vs_full"]
            L.append(f"| {k} | {v['NDCG@10']:.4f} | {d['diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] | {v['NDCG@10_cold']:.4f} |")
    L += ["", "## What the ranker leans on (mean |SHAP| over each user's top-20)", "",
          "| Feature group | warm users | cold users |", "|---|---|---|"]
    w, cd = R["shap_group_importance"]["warm"], R["shap_group_importance"]["cold"]
    for g in w:
        L.append(f"| {g} | {w[g]:.3f} | {cd.get(g, 0):.3f} |")
    if "tree_correction_importance" in R:
        tw, tc = R["tree_correction_importance"]["warm"], R["tree_correction_importance"]["cold"]
        L += ["", "Tree corrections only (what the boosted trees add on top of the linear blend):", "",
              "| Feature group | warm users | cold users |", "|---|---|---|"]
        for g in tw:
            L.append(f"| {g} | {tw[g]:.3f} | {tc.get(g, 0):.3f} |")
    L += ["", "## Example explanations", ""]
    for u, ex in R["examples"].items():
        L.append(f"**User {u}** ({'cold-start' if ex['cold'] else 'warm'}) — recently watched: {', '.join(ex['recent_history'])}")
        L.append("")
        for i, rec in enumerate(ex["recommendations"], 1):
            L.append(f"{i}. **{rec['title']}** — {'; '.join(rec['reasons'])}")
        L.append("")
    L += ["## Ranker selection (scored on the most recent training slice, never on test)", "",
          "| leaves | min leaf | raw scores | start from blend | trees | val NDCG@10 |", "|---|---|---|---|---|---|"]
    for g in sorted(R["ranker_grid"], key=lambda g: -g["val_ndcg10"]):
        L.append(f"| {g['num_leaves']} | {g['min_data_in_leaf']} | {g['raw_scores']} | {g['init_blend']} | {g['trees']} | {g['val_ndcg10']:.4f} |")
    L += ["", "## Tuned settings", "", f"```\nretrievers: {R['tuned']}\nranker: {R['ranker_settings']} ({R['ranker_trees']} trees)\nblend weights: {R['blend_weights']}\n```"]
    (out / "RESULTS.md").write_text("\n".join(L))
    try:
        from .figures import make_figures
        make_figures(R, out)
    except Exception as e:  # figures are nice-to-have
        print("figures skipped:", e)
