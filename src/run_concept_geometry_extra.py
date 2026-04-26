"""Per-layer concept geometry under interventions: PAIRED across conditions.

For the same concept bank, run the model under stock and under each
intervention; compute the per-layer pair-distance and per-pair distance
*change*.  This gives paired statistical tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from ln_interventions import LNIntervention
from geometry_probes import (
    DEFAULT_CONCEPT_BANK,
    _last_token_hidden,
    all_pair_cosine_distances,
)


REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        output_hidden_states=True,
    ).to(args.device)
    model.eval()

    # Get baseline (stock) hidden states for all layers
    stock_states = _last_token_hidden(model, tok, DEFAULT_CONCEPT_BANK, device=args.device)
    n_layers = stock_states.shape[0]

    grid = [
        ("no_mean", "all", 0.0),
        ("no_affine", "all", 0.0),
        ("weakened_a", "all", 0.75),
        ("no_mean", "early", 0.0),
        ("no_mean", "mid", 0.0),
        ("no_mean", "late", 0.0),
    ]

    rows = []
    for recipe, target, alpha in grid:
        with LNIntervention(model, recipe=recipe, target=target, alpha=alpha):
            int_states = _last_token_hidden(model, tok, DEFAULT_CONCEPT_BANK, device=args.device)
        # paired comparison per layer
        for layer in range(n_layers):
            stock_d = all_pair_cosine_distances(stock_states[layer]).numpy()
            int_d = all_pair_cosine_distances(int_states[layer]).numpy()
            # remove infs/nans
            stock_d = np.nan_to_num(stock_d, nan=0.0, posinf=0.0, neginf=0.0)
            int_d = np.nan_to_num(int_d, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                t, p = stats.wilcoxon(int_d, stock_d, zero_method="wilcox", alternative="two-sided")
                t = float(t); p = float(p)
            except Exception:
                t = float("nan"); p = float("nan")
            rows.append({
                "model": args.model,
                "recipe": recipe, "target": target, "alpha": alpha,
                "layer": layer,
                "stock_mean_dist": float(stock_d.mean()),
                "int_mean_dist": float(int_d.mean()),
                "delta_mean_dist": float(int_d.mean() - stock_d.mean()),
                "wilcoxon_stat": t,
                "wilcoxon_p": p,
                "n_pairs": len(stock_d),
            })
        last_int_d = all_pair_cosine_distances(int_states[-1]).numpy()
        last_stock_d = all_pair_cosine_distances(stock_states[-1]).numpy()
        print(f"[{args.model}] {recipe:>10s} target={target:<5s} α={alpha:.2f} "
              f"final-layer Δ mean_dist = {(last_int_d.mean() - last_stock_d.mean()):.4f}")

    out_path = args.out
    if out_path is None:
        safe = args.model.replace("/", "_")
        out_path = RESULTS / f"paired_geometry_{safe}.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
