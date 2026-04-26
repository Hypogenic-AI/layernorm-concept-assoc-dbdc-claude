"""Final summary plots for the report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGS = REPO / "figures"


def load_dat_all():
    rows = []
    for f in sorted(RESULTS.glob("dat_*.json")):
        try:
            data = json.load(open(f))
        except Exception:
            continue
        rows.extend(data)
    return pd.DataFrame(rows)


def fig_main_dat_panel(df: pd.DataFrame):
    """Main figure: DAT vs intervention, faceted by model and decoding,
    with validity-rate as alpha overlay."""
    df = df.copy()
    df["cond"] = df.apply(
        lambda r: f"{r['recipe']}-{r['target']}-{r['alpha']:.2f}", axis=1
    )
    # condition order — fluency-preserving first, then broken
    cond_order = [
        "stock-all-0.00",
        "no_mean-all-0.00",
        "no_mean-early-0.00",
        "no_mean-mid-0.00",
        "no_mean-late-0.00",
        # broken (label as "broken" group)
        "no_affine-all-0.00",
        "weakened_a-all-0.75",
        "weakened_a-all-0.50",
        "weakened_a-all-0.25",
        "identity-all-0.00",
        "weakened_a-early-0.50",
        "weakened_a-mid-0.50",
        "weakened_a-late-0.50",
    ]

    panels = df.groupby(["model", "decoding"])
    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 3.0 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    for ax, ((m, d), g) in zip(axes, panels):
        means, stds, validity, conds, scored, total = [], [], [], [], [], []
        for c in cond_order:
            sub = g[g["cond"] == c]
            if sub.empty:
                continue
            dats = sub["dat"].dropna().values
            valids = sub["validity_rate"].values if "validity_rate" in sub.columns else np.array([1.0]*len(sub))
            conds.append(c)
            means.append(float(np.mean(dats)) if len(dats) else 0)
            stds.append(float(np.std(dats)) if len(dats) > 1 else 0)
            validity.append(float(np.mean(valids)) if len(valids) else 0)
            scored.append(len(dats))
            total.append(len(sub))
        x = np.arange(len(conds))
        # color by validity (faded for broken interventions)
        colors = []
        for c, v in zip(conds, validity):
            if c == "stock-all-0.00":
                colors.append("black")
            elif c.startswith("no_mean"):
                colors.append("steelblue")
            else:
                colors.append("lightcoral" if v >= 0.9 else "tan")
        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=3)
        # mark interventions where the model is "broken" (validity < 0.9)
        for i, (xi, vi, c) in enumerate(zip(x, validity, conds)):
            if vi < 0.9 and c != "stock-all-0.00":
                ax.text(xi, max(means[i] + stds[i] + 2, 5), f"v={vi:.2f}",
                        ha="center", va="bottom", fontsize=7, color="darkred")
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("DAT score")
        ax.set_title(f"{m} | {d}")
        if "stock-all-0.00" in conds:
            stock_idx = conds.index("stock-all-0.00")
            ax.axhline(means[stock_idx], color="black", linestyle="--", alpha=0.4, linewidth=1)
        ax.set_ylim(0, 105)
    plt.tight_layout()
    out = FIGS / "main_dat_panel.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_geometry_layerwise():
    """Per-layer pair-distance under stock vs key interventions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models = ["gpt2", "gpt2-medium", "EleutherAI_pythia-410m"]
    for ax, m in zip(axes, models):
        f = RESULTS / f"geometry_{m}.json"
        if not f.exists():
            continue
        data = json.load(open(f))
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "last_layer_pair_dists"} for r in data])
        # plot only fluent-preserving recipes
        keep = [
            ("stock", "all", 0.00, "black", "-"),
            ("no_mean", "all", 0.00, "steelblue", "-"),
            ("no_mean", "early", 0.00, "lightblue", "--"),
            ("no_mean", "mid", 0.00, "skyblue", "--"),
            ("no_mean", "late", 0.00, "navy", "--"),
            ("no_affine", "all", 0.00, "darkviolet", ":"),
        ]
        for recipe, target, alpha, color, ls in keep:
            sub = df[(df["recipe"] == recipe) & (df["target"] == target) & (df["alpha"] == alpha)]
            sub = sub.sort_values("layer")
            if sub.empty:
                continue
            ax.plot(sub["layer"], sub["mean_pair_dist"],
                    label=f"{recipe}-{target}", color=color, linestyle=ls, linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean pairwise cosine distance")
        ax.set_title(m)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGS / "geometry_layerwise_three_models.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_dat_vs_validity_panel(df: pd.DataFrame):
    """Pareto plot DAT vs validity, all models on one axis."""
    fig, ax = plt.subplots(figsize=(10, 7))
    color_map = {
        "stock": "black",
        "no_mean": "steelblue",
        "no_affine": "darkviolet",
        "weakened_a": "darkorange",
        "identity": "red",
    }
    marker_map = {
        ("gpt2", "topp"): "o",
        ("gpt2", "greedy"): "s",
        ("gpt2-medium", "topp"): "^",
        ("EleutherAI/pythia-410m", "topp"): "D",
    }
    seen = set()
    for (m, d, recipe, target, alpha), g in df.groupby(["model", "decoding", "recipe", "target", "alpha"]):
        dats = g["dat"].dropna().values
        valids = g["validity_rate"].values if "validity_rate" in g.columns else [1.0]*len(g)
        if len(dats) == 0:
            continue
        c = color_map.get(recipe, "gray")
        marker = marker_map.get((m, d), "x")
        label_recipe = f"{recipe}-{target}-α{alpha:.2f}"
        ax.errorbar(
            np.mean(valids), np.mean(dats),
            xerr=np.std(valids) if len(valids) > 1 else 0,
            yerr=np.std(dats) if len(dats) > 1 else 0,
            color=c, marker=marker, alpha=0.7, capsize=2, markersize=8,
            linestyle=""
        )
    # Build a combined legend
    from matplotlib.lines import Line2D
    legend_elems = [
        *[Line2D([0], [0], color=v, marker="o", linestyle="", markersize=8, label=k)
          for k, v in color_map.items()],
        *[Line2D([0], [0], color="gray", marker=v, linestyle="", markersize=8, label=f"{k[0]}|{k[1]}")
          for k, v in marker_map.items()],
    ]
    ax.legend(handles=legend_elems, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_xlabel("Validity rate (mean across runs)")
    ax.set_ylabel("DAT score (mean across runs)")
    ax.axvline(0.9, color="gray", linestyle=":", alpha=0.5,
               label="validity=0.9 (fluency floor)")
    ax.grid(alpha=0.3)
    ax.set_title("DAT vs validity (Pareto): only points with validity ≥ 0.9 are coherent measurements")
    plt.tight_layout()
    out = FIGS / "dat_vs_validity_panel.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_human_baseline_compare():
    """LLM stock-DAT distribution overlaid on human baseline."""
    human_p = REPO / "datasets" / "dat_human_baseline" / "study2.tsv"
    if not human_p.exists():
        return None
    human = pd.read_csv(human_p, sep="\t")
    human_dat = human["dat"].dropna().values

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(human_dat, ax=ax, fill=True, color="lightgray", label=f"Human (n={len(human_dat):,})")

    df = load_dat_all()
    stock = df[(df["recipe"] == "stock") & (df["dat"].notna())]
    colors = plt.cm.tab10(np.linspace(0, 1, len(stock["model"].unique()) * 2))
    for i, ((m, d), g) in enumerate(stock.groupby(["model", "decoding"])):
        dats = g["dat"].values
        if len(dats) >= 3:
            sns.kdeplot(dats, ax=ax, color=colors[i], label=f"{m}|{d} (n={len(dats)}, μ={np.mean(dats):.1f})")
    ax.set_xlabel("DAT score")
    ax.set_title("DAT score distributions — humans vs LLMs (stock LN)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = FIGS / "humans_vs_llms.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    df = load_dat_all()
    print(f"Loaded {len(df)} DAT rows from {df['model'].unique() if 'model' in df.columns else '?'}")
    fig_main_dat_panel(df)
    fig_geometry_layerwise()
    fig_dat_vs_validity_panel(df)
    fig_human_baseline_compare()
    print("Saved final figures to", FIGS)


if __name__ == "__main__":
    main()
