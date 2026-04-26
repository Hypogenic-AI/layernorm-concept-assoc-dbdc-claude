"""Analysis: statistical comparisons of DAT under LN interventions.

Reads results/dat_*.json and results/geometry_*.json, computes effect sizes
and p-values, and saves figures to figures/.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGS = REPO / "figures"
FIGS.mkdir(exist_ok=True)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if s == 0:
        return float("nan")
    return (a.mean() - b.mean()) / s


# --------------------------------------------------------------------------- #
# 1. DAT analysis
# --------------------------------------------------------------------------- #

def analyze_dat_model(path: Path) -> pd.DataFrame:
    """Per-condition summary + Mann-Whitney vs stock."""
    rows = json.load(open(path))
    df = pd.DataFrame(rows)
    df["cond"] = df.apply(
        lambda r: f"{r['recipe']}|{r['target']}|{r['alpha']:.2f}", axis=1
    )

    # group summaries
    summary = []
    for cond, g in df.groupby("cond"):
        dats = g["dat"].dropna().values
        valids = g["validity_rate"].values if "validity_rate" in g.columns else np.array([])
        summary.append({
            "cond": cond,
            "n": len(g),
            "n_scored": len(dats),
            "dat_mean": float(np.mean(dats)) if len(dats) else float("nan"),
            "dat_std": float(np.std(dats)) if len(dats) > 1 else 0.0,
            "dat_median": float(np.median(dats)) if len(dats) else float("nan"),
            "validity_mean": float(np.mean(valids)) if len(valids) else float("nan"),
        })
    sdf = pd.DataFrame(summary)

    # vs stock comparison
    stock = df[df["cond"].str.startswith("stock|")]["dat"].dropna().values
    comparisons = []
    for cond, g in df.groupby("cond"):
        if cond.startswith("stock|"):
            continue
        v = g["dat"].dropna().values
        if len(v) >= 3 and len(stock) >= 3:
            try:
                u, p = stats.mannwhitneyu(v, stock, alternative="two-sided")
                d = cohens_d(v, stock)
                ci_low = float(np.percentile(v, 2.5))
                ci_high = float(np.percentile(v, 97.5))
            except Exception:
                p, d, u, ci_low, ci_high = float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
        else:
            p, d, u, ci_low, ci_high = float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
        comparisons.append({
            "cond": cond,
            "n_scored": len(v),
            "delta_dat": (float(np.mean(v)) - float(np.mean(stock))) if len(v) and len(stock) else float("nan"),
            "cohens_d": d,
            "u_stat": float(u) if not np.isnan(u) else float("nan"),
            "p_value": float(p) if not np.isnan(p) else float("nan"),
        })
    cdf = pd.DataFrame(comparisons)

    return sdf, cdf, df


# --------------------------------------------------------------------------- #
# 2. Geometry analysis
# --------------------------------------------------------------------------- #

def analyze_geometry(path: Path) -> pd.DataFrame:
    rows = json.load(open(path))
    return pd.DataFrame([{
        k: v for k, v in r.items()
        if k != "last_layer_pair_dists"
    } for r in rows])


# --------------------------------------------------------------------------- #
# 3. Plots
# --------------------------------------------------------------------------- #

def plot_dat_grouped(df: pd.DataFrame, model_name: str):
    """Bar plot: DAT mean +/- std, grouped by recipe/target."""
    df = df.copy()
    df["cond_short"] = df.apply(
        lambda r: f"{r['recipe']}\n{r['target']} α={r['alpha']:.2f}", axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    # left: DAT by condition
    ax = axes[0]
    g = df.groupby("cond_short")["dat"].agg(["mean", "std", "count"]).reset_index()
    g = g.sort_values("mean", ascending=False, na_position="last")
    ax.barh(g["cond_short"], g["mean"], xerr=g["std"], color="steelblue")
    ax.set_xlabel("DAT score")
    ax.set_title(f"{model_name}: DAT by LN intervention")
    ax.axvline(g[g["cond_short"].str.contains("stock")]["mean"].iloc[0], color="red",
               linestyle="--", alpha=0.5, label="stock baseline")
    ax.legend()
    # right: Validity rate
    ax = axes[1]
    g = df.groupby("cond_short")["validity_rate"].mean().reset_index()
    g = g.sort_values("validity_rate", ascending=False)
    ax.barh(g["cond_short"], g["validity_rate"], color="forestgreen")
    ax.set_xlabel("Validity rate (frac. of 10 slots producing GloVe-valid noun)")
    ax.set_title(f"{model_name}: Generation validity by intervention")
    plt.tight_layout()
    out = FIGS / f"dat_{model_name.replace('/', '_')}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_dat_vs_validity(df: pd.DataFrame, model_name: str):
    """Scatter: DAT vs validity, per condition (one point per cond, errorbars)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for cond, g in df.groupby(["recipe", "target", "alpha"]):
        dat = g["dat"].dropna().values
        valid = g["validity_rate"].values
        if len(dat) == 0:
            continue
        recipe, target, alpha = cond
        marker = {"all": "o", "early": "^", "mid": "s", "late": "v"}.get(target, "x")
        color = {
            "stock": "black",
            "no_mean": "tab:blue",
            "no_affine": "tab:cyan",
            "weakened_a": "tab:orange",
            "no_scale": "tab:purple",
            "identity": "tab:red",
        }.get(recipe, "gray")
        ax.errorbar(
            np.mean(valid), np.mean(dat),
            xerr=np.std(valid), yerr=np.std(dat),
            marker=marker, color=color, alpha=0.7, capsize=2,
            label=f"{recipe} {target} α={alpha:.2f}",
            markersize=8,
        )
    ax.set_xlabel("Validity rate")
    ax.set_ylabel("DAT score")
    ax.set_title(f"{model_name}: DAT vs Validity (Pareto)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    out = FIGS / f"dat_vs_validity_{model_name.replace('/', '_')}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_geometry_layerwise(geom_df: pd.DataFrame, model_name: str):
    """Mean pairwise concept distance vs layer, one curve per intervention."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["mean_pair_dist", "norm_ratio", "uniform_angle_mean"]
    titles = ["Mean pairwise cosine distance", "Norm std/mean ratio",
              "Mean angle to uniform vector (deg)"]
    for ax, metric, title in zip(axes, metrics, titles):
        for cond, g in geom_df.groupby(["recipe", "target", "alpha"]):
            recipe, target, alpha = cond
            label = f"{recipe} {target} α={alpha:.2f}"
            g = g.sort_values("layer")
            vals = g[metric].values
            # mask infs/nans for plotting
            vals = np.where(np.isfinite(vals), vals, np.nan)
            color = {
                "stock": "black",
                "no_mean": "tab:blue",
                "no_affine": "tab:cyan",
                "weakened_a": "tab:orange",
                "no_scale": "tab:purple",
                "identity": "tab:red",
            }.get(recipe, "gray")
            ls = {"all": "-", "early": ":", "mid": "--", "late": "-."}.get(target, "-")
            ax.plot(g["layer"], vals, label=label, color=color, linestyle=ls, alpha=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel(title)
        ax.set_title(f"{model_name}: {title}")
        if metric == "mean_pair_dist":
            ax.set_ylim(-0.01, 0.5)
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    out = FIGS / f"geometry_{model_name.replace('/', '_')}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_human_baseline_compare(dat_dfs: dict, human_path: Path):
    """Overlay LLM DAT distributions on human baseline density."""
    human = pd.read_csv(human_path, sep="\t")
    human_dat = human["dat"].dropna().values

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(human_dat, ax=ax, fill=True, color="lightgray", label=f"Human (n={len(human_dat):,})")
    colors = plt.cm.tab10(np.linspace(0, 1, len(dat_dfs)))
    for (model, df), c in zip(dat_dfs.items(), colors):
        # only stock condition
        dats = df[df["recipe"] == "stock"]["dat"].dropna().values
        if len(dats) >= 3:
            sns.kdeplot(dats, ax=ax, color=c, label=f"{model} stock (n={len(dats)})")
    ax.set_xlabel("DAT score")
    ax.set_title("DAT score distributions — humans vs LLMs (stock LN)")
    ax.legend()
    plt.tight_layout()
    out = FIGS / "dat_humans_vs_llms.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    print("=" * 70)
    print("DAT analysis")
    print("=" * 70)
    dat_dfs = {}
    summaries = {}
    comparisons = {}
    for f in sorted(RESULTS.glob("dat_*.json")):
        model_decoding = f.stem.replace("dat_", "")
        print(f"\n--- {model_decoding} ---")
        sdf, cdf, df = analyze_dat_model(f)
        dat_dfs[model_decoding] = df
        summaries[model_decoding] = sdf
        comparisons[model_decoding] = cdf
        print(sdf.to_string(index=False))
        print()
        print("Comparison vs stock:")
        print(cdf.to_string(index=False))
        # plots
        plot_dat_grouped(df, model_decoding)
        plot_dat_vs_validity(df, model_decoding)

    # combined human-vs-LLM plot
    human_p = REPO / "datasets" / "dat_human_baseline" / "study2.tsv"
    if human_p.exists() and dat_dfs:
        plot_human_baseline_compare(dat_dfs, human_p)

    # save summaries
    for k, sdf in summaries.items():
        sdf.to_csv(RESULTS / f"summary_dat_{k}.csv", index=False)
    for k, cdf in comparisons.items():
        cdf.to_csv(RESULTS / f"comparison_dat_{k}.csv", index=False)

    print("\n" + "=" * 70)
    print("Geometry analysis")
    print("=" * 70)
    for f in sorted(RESULTS.glob("geometry_*.json")):
        model = f.stem.replace("geometry_", "")
        print(f"\n--- {model} ---")
        geom = analyze_geometry(f)
        plot_geometry_layerwise(geom, model)
        # final-layer summary
        last_layer = geom["layer"].max()
        ll = geom[geom["layer"] == last_layer].copy()
        print("Final-layer geometry per intervention:")
        print(ll[["recipe", "target", "alpha", "mean_pair_dist", "norm_mean", "norm_ratio", "uniform_angle_mean"]].to_string(index=False))
        ll.to_csv(RESULTS / f"summary_geometry_{model}.csv", index=False)


if __name__ == "__main__":
    main()
