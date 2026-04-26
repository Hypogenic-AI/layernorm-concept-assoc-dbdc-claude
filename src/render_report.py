"""Render REPORT.md from all available results JSON / CSV files."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGS = REPO / "figures"


def _safe(x, fmt=":.2f"):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return ("{" + fmt + "}").format(x)


def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if s == 0:
        return float("nan")
    return (a.mean() - b.mean()) / s


def section_dat_results():
    """Build a table of DAT results across all models and decoding strategies."""
    rows_all = []
    for f in sorted(RESULTS.glob("dat_*.json")):
        try:
            data = json.load(open(f))
        except Exception:
            continue
        if not data:
            continue
        df = pd.DataFrame(data)
        if df.empty or "dat" not in df.columns:
            continue
        df["cond"] = df.apply(lambda r: f"{r['recipe']}|{r['target']}|{r['alpha']:.2f}", axis=1)
        for cond, g in df.groupby("cond"):
            dats = g["dat"].dropna().values
            valids = g.get("validity_rate", pd.Series([1.0]*len(g))).values
            rows_all.append({
                "file": f.stem,
                "model": g["model"].iloc[0] if "model" in g.columns else "?",
                "decoding": g["decoding"].iloc[0],
                "recipe": g["recipe"].iloc[0],
                "target": g["target"].iloc[0],
                "alpha": float(g["alpha"].iloc[0]),
                "n": len(g),
                "n_scored": len(dats),
                "validity": float(np.mean(valids)),
                "dat_mean": float(np.mean(dats)) if len(dats) else float("nan"),
                "dat_std": float(np.std(dats)) if len(dats) > 1 else 0.0,
            })
    return pd.DataFrame(rows_all)


def stats_vs_stock(df_dat: pd.DataFrame):
    """For each model, compute Mann-Whitney U vs stock for each non-stock cond."""
    rows = []
    # need raw values per condition for tests
    raw = {}
    for f in sorted(RESULTS.glob("dat_*.json")):
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for r in data:
            if r.get("dat") is None:
                continue
            key = (r["model"], r["decoding"], r["recipe"], r["target"], float(r["alpha"]))
            raw.setdefault(key, []).append(r["dat"])
    # group by (model, decoding) — find stock baseline
    by_md = {}
    for key, vals in raw.items():
        m, d = key[0], key[1]
        by_md.setdefault((m, d), {})[(key[2], key[3], key[4])] = np.array(vals)
    for (m, d), conds in by_md.items():
        stock_key = ("stock", "all", 0.0)
        if stock_key not in conds:
            continue
        stock = conds[stock_key]
        for k, v in conds.items():
            if k == stock_key:
                continue
            try:
                u, p = stats.mannwhitneyu(v, stock, alternative="two-sided")
                d_eff = cohens_d(v, stock)
            except Exception:
                u, p, d_eff = float("nan"), float("nan"), float("nan")
            rows.append({
                "model": m, "decoding": d,
                "recipe": k[0], "target": k[1], "alpha": k[2],
                "n_int": len(v), "n_stock": len(stock),
                "stock_mean": float(stock.mean()),
                "int_mean": float(v.mean()),
                "delta": float(v.mean() - stock.mean()),
                "cohens_d": d_eff,
                "u_stat": float(u),
                "p_value": float(p),
            })
    return pd.DataFrame(rows)


def section_geometry():
    rows = []
    for f in sorted(RESULTS.glob("geometry_*.json")):
        data = json.load(open(f))
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "last_layer_pair_dists"} for r in data])
        df = df[df["layer"] == df["layer"].max()]  # final-layer summary
        for _, r in df.iterrows():
            rows.append({
                "model": r["model"],
                "recipe": r["recipe"], "target": r["target"], "alpha": float(r["alpha"]),
                "final_layer_pair_dist": r["mean_pair_dist"],
                "final_norm_mean": r["norm_mean"],
                "final_norm_ratio": r["norm_ratio"],
            })
    return pd.DataFrame(rows)


def section_perplexity():
    rows = []
    for f in sorted(RESULTS.glob("perplexity_*.json")):
        data = json.load(open(f))
        for r in data:
            rows.append(r)
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, cols=None, fmt=None) -> str:
    if df.empty:
        return "_(no data)_\n"
    if cols is None:
        cols = list(df.columns)
    df = df[cols].copy()
    if fmt:
        for c, f in fmt.items():
            if c in df.columns:
                df[c] = df[c].apply(lambda x: ("{" + f + "}").format(x) if x == x and x is not None else "—")
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    dat = section_dat_results()
    stats_df = stats_vs_stock(dat)
    geom = section_geometry()
    ppl = section_perplexity()

    # save the stats table separately for reference
    if not stats_df.empty:
        stats_df.to_csv(RESULTS / "stats_dat_vs_stock.csv", index=False)
    if not dat.empty:
        dat.to_csv(RESULTS / "summary_dat_all.csv", index=False)
    if not geom.empty:
        geom.to_csv(RESULTS / "summary_geometry_all.csv", index=False)
    if not ppl.empty:
        ppl.to_csv(RESULTS / "summary_perplexity_all.csv", index=False)

    # Print to stdout for inspection
    print("=== DAT summary ===")
    print(dat.to_string(index=False))
    print("\n=== Stats vs stock ===")
    print(stats_df.to_string(index=False))
    print("\n=== Geometry final-layer ===")
    print(geom.to_string(index=False))
    print("\n=== Perplexity ===")
    print(ppl.to_string(index=False))


if __name__ == "__main__":
    main()
