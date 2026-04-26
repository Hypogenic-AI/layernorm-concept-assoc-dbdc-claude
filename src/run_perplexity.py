"""Perplexity sanity check: confirm that LN interventions don't trivially
'break' the model.  We measure NLL on a small held-out text sample under
each intervention recipe.  Under the user's hypothesis we *expect* small
LN modulations to cost a little perplexity but not catastrophically — that
would distinguish 'interesting' from 'broken' interventions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ln_interventions import LNIntervention


REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)


# A short, neutral text for perplexity calibration.  We don't need a huge
# dataset — we want a fast diagnostic that distinguishes 'small effect' from
# 'broken'.
CALIB_TEXT = (
    "The art of teaching is the art of assisting discovery. The whole purpose "
    "of education is to turn mirrors into windows. Education is not the "
    "filling of a pail, but the lighting of a fire. The function of education "
    "is to teach one to think intensively and to think critically. "
    "Intelligence plus character — that is the goal of true education. "
    "Tell me and I forget, teach me and I may remember, involve me and I "
    "learn. Education is the most powerful weapon which you can use to "
    "change the world. The roots of education are bitter, but the fruit is "
    "sweet. The mind is not a vessel to be filled, but a fire to be kindled. "
    "I have never let my schooling interfere with my education."
)


@torch.no_grad()
def compute_nll(model, tokenizer, text: str, device: str = "cuda") -> float:
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    out = model(ids, labels=ids)
    nll = out.loss.item()
    return nll  # average per-token NLL


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
        args.model, torch_dtype=torch.float32
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
        ("weakened_a", "early", 0.5),
        ("weakened_a", "mid", 0.5),
        ("weakened_a", "late", 0.5),
    ]

    rows = []
    for recipe, target, alpha in grid:
        with LNIntervention(model, recipe=recipe, target=target, alpha=alpha):
            nll = compute_nll(model, tok, CALIB_TEXT, device=args.device)
        ppl = float(torch.exp(torch.tensor(nll))) if nll == nll else float("inf")
        rows.append({
            "model": args.model, "recipe": recipe, "target": target, "alpha": alpha,
            "nll": nll, "ppl": ppl,
        })
        print(f"[{args.model}] {recipe:>11s} target={target:<5s} α={alpha:.2f} | NLL={nll:.3f} PPL={ppl:.2f}")

    out_path = args.out
    if out_path is None:
        safe = args.model.replace("/", "_")
        out_path = RESULTS / f"perplexity_{safe}.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
