"""Per-layer concept-association geometry under LN interventions.

Tests H1: LN compresses cross-concept embedding distance.

For each LN intervention recipe, we feed a fixed bank of "unrelated nouns"
through the model with output_hidden_states=True, capture per-layer last-token
hidden states, and compute pairwise cosine distances.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ln_interventions import LNIntervention
from geometry_probes import (
    DEFAULT_CONCEPT_BANK,
    all_pair_cosine_distances,
    angle_to_uniform,
    collect_geometry,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


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

    grid = [
        ("stock", "all", 0.0),
        ("no_mean", "all", 0.0),
        ("no_affine", "all", 0.0),
        ("weakened_a", "all", 0.75),
        ("weakened_a", "all", 0.50),
        ("weakened_a", "all", 0.25),
        ("identity", "all", 0.0),
        ("no_mean", "early", 0.0),
        ("no_mean", "mid", 0.0),
        ("no_mean", "late", 0.0),
    ]

    rows = []
    for recipe, target, alpha in grid:
        with LNIntervention(model, recipe=recipe, target=target, alpha=alpha):
            states, geom = collect_geometry(model, tok, DEFAULT_CONCEPT_BANK, device=args.device)
        # save per-layer summary
        for entry in geom:
            row = {
                "model": args.model, "recipe": recipe, "target": target, "alpha": alpha,
                **entry,
            }
            rows.append(row)
        # also save raw pairwise distances at the *last* hidden layer (penultimate is more informative)
        last_layer = states.shape[0] - 1
        s = states[last_layer]
        dists = all_pair_cosine_distances(s).numpy().tolist()
        rows[-1]["last_layer_pair_dists"] = dists  # attach to last row
        print(f"[{args.model}] {recipe:>11s} target={target:<5s} alpha={alpha:.2f} "
              f"final-layer pair_dist mean={np.mean(dists):.4f}")

    out_path = args.out
    if out_path is None:
        safe = args.model.replace("/", "_")
        out_path = RESULTS_DIR / f"geometry_{safe}.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
