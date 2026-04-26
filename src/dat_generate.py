"""Run DAT generation under different LN interventions for an LM.

Generates 10 unrelated nouns from a model under a given LN intervention,
parses them, scores them, and saves raw + scored results.

Two prompting strategies:
  - "instruct"  : full DAT instructions + Olson-style few-shot examples
                  (works for instruct-tuned LLMs only; base LMs copy examples).
  - "list"      : minimal "list completion" prompt that base LMs handle.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import LogitsProcessor, LogitsProcessorList

from ln_interventions import LNIntervention
from dat_scoring import DATScorer, parse_word_list


class SafeLogitsProcessor(LogitsProcessor):
    """Replaces NaN/Inf in logits with very negative values so sampling
    doesn't crash when an LN intervention produces unstable activations."""

    def __call__(self, input_ids, scores):
        scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e6, neginf=-1e9)
        return scores


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# Iterative list generation: we feed the model "1. <w1>\n2. <w2>\n...\nk." and
# ask it for one noun at a time.  Each response is parsed for the FIRST valid
# noun, which is appended to the list.  This is far more robust than asking the
# model to produce all 10 in a single generation, and tests precisely the
# right thing -- given a partial unrelated-list context, what concept does the
# model surface next?

LIST_HEADER = "Here is a list of ten random unrelated single-word nouns:\n"

# Seed words spanning multiple semantic categories.  We use a single seed to
# anchor the format and let the model produce the remaining nine.
SEED_WORDS = [
    "apple", "shoe", "river", "engine", "book",
    "wallet", "candle", "fence", "mirror", "rocket",
]


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, decoding: dict,
                 device: str = "cuda", max_new_tokens: int = 80) -> str:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attn = torch.ones_like(ids)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attn,
        logits_processor=LogitsProcessorList([SafeLogitsProcessor()]),
        **decoding,
    )
    try:
        out = model.generate(ids, **gen_kwargs)
    except RuntimeError as e:
        if "probability tensor" in str(e) or "nan" in str(e).lower():
            return ""
        raise
    text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return text


@torch.no_grad()
def generate_dat_list(
    model, tokenizer, scorer: DATScorer, decoding: dict,
    seed_word: str, device: str = "cuda",
    n_target: int = 10, max_attempts_per_slot: int = 4,
):
    """Iteratively build a 10-noun list.  At each step, we feed the current
    list as context and ask the model to continue.  Take the first GloVe-
    valid noun the parser extracts; if none after `max_attempts_per_slot`
    fresh samples, mark the slot as a failure (does not count as a word).

    Returns (words, n_failed_slots) -- words may be < 10 long.
    All returned words are in GloVe vocab (validated).
    """
    words = [seed_word]
    n_failed_slots = 0
    while len(words) < n_target:
        prompt_lines = [f"{i+1}. {w}" for i, w in enumerate(words)]
        prompt = LIST_HEADER + "\n".join(prompt_lines) + f"\n{len(words)+1}."
        success = False
        for attempt in range(max_attempts_per_slot):
            text = generate_one(
                model, tokenizer, prompt, decoding,
                device=device, max_new_tokens=12,
            )
            cand = parse_word_list(" " + text.split("\n")[0])
            for w in cand:
                if w not in words and scorer.validate(w) is not None:
                    words.append(w)
                    success = True
                    break
            if success:
                break
        if not success:
            n_failed_slots += 1
            # fill slot with a placeholder we will filter out for scoring,
            # so the next prompt doesn't get stuck repeating the same item.
            # Use a placeholder containing a digit so it never validates
            placeholder = f"__FAILSLOT{len(words)}__"
            words.append(placeholder)
        # safety: cap total iterations
        if len(words) >= n_target:
            break
    return words, n_failed_slots


def make_decoding(strategy: str) -> dict:
    # All strategies use repetition_penalty + no_repeat_ngram_size to avoid
    # degenerate single-token loops common in small base LMs.  The DAT task
    # demands distinct words, so banning bigram repeats is task-appropriate.
    base = {"repetition_penalty": 1.6, "no_repeat_ngram_size": 2}
    if strategy == "greedy":
        return {"do_sample": False, "num_beams": 1, **base}
    if strategy == "topp":
        return {"do_sample": True, "top_p": 0.9, "temperature": 0.7, **base}
    if strategy == "high_t":
        return {"do_sample": True, "temperature": 1.2, "top_p": 0.95, **base}
    raise ValueError(strategy)


def run_condition(
    model, tokenizer,
    recipe: str, target: str, alpha: float,
    decoding_strategy: str,
    n_runs: int,
    scorer: DATScorer,
    seed_words: List[str],
    device: str = "cuda",
    base_seed: int = 42,
    log_prefix: str = "",
) -> List[dict]:
    decoding = make_decoding(decoding_strategy)
    rows = []
    with LNIntervention(model, recipe=recipe, target=target, alpha=alpha):
        for i in range(n_runs):
            seed_w = seed_words[i % len(seed_words)]
            set_seeds(base_seed + i)
            t0 = time.time()
            try:
                words, n_failed = generate_dat_list(
                    model, tokenizer, scorer, decoding,
                    seed_word=seed_w, device=device,
                )
            except Exception as e:
                rows.append({
                    "run": i, "recipe": recipe, "target": target, "alpha": alpha,
                    "decoding": decoding_strategy, "seed_word": seed_w,
                    "error": str(e), "dat": None, "n_valid": 0,
                    "extracted_words": [], "valid_words": [],
                    "n_failed_slots": 10,
                })
                continue
            valid_words = [w for w in words if scorer.validate(w) is not None]
            # validity rate: fraction of 10 slots that yielded a GloVe-valid noun
            validity_rate = len(valid_words) / 10.0
            dat_score = scorer.dat(valid_words) if len(valid_words) >= 7 else None
            rows.append({
                "run": i, "recipe": recipe, "target": target, "alpha": alpha,
                "decoding": decoding_strategy,
                "seed_word": seed_w,
                "dat": dat_score,
                "n_valid": len(valid_words),
                "validity_rate": validity_rate,
                "n_failed_slots": n_failed,
                "n_extracted": len(words),
                "extracted_words": words,
                "valid_words": valid_words,
                "elapsed_s": round(time.time() - t0, 3),
            })
    n_scored = sum(1 for r in rows if r["dat"] is not None)
    valid_dats = [r["dat"] for r in rows if r["dat"] is not None]
    valids = [r.get("validity_rate", 0.0) for r in rows]
    print(f"  {log_prefix} {recipe:>11s} | tgt={target:<5s} | alpha={alpha:.2f} | "
          f"dec={decoding_strategy:<7s} | scored={n_scored:>3d}/{n_runs} | "
          f"valid={np.mean(valids):.2f} | "
          f"DAT={np.mean(valid_dats) if valid_dats else float('nan'):>5.2f}±"
          f"{np.std(valid_dats) if len(valid_dats)>1 else 0:>4.2f}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n_runs", type=int, default=20)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--decoding", type=str, default="topp",
                   choices=["greedy", "topp", "high_t"])
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    print(f"Loading model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).to(args.device)
    model.eval()

    print("Loading DAT scorer (GloVe)…")
    scorer = DATScorer()
    print(f"  GloVe vocab: {len(scorer.vectors):,}")

    if args.quick:
        grid = [
            ("stock", "all", 0.0),
            ("no_mean", "all", 0.0),
            ("weakened_a", "all", 0.5),
        ]
    else:
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

    all_rows = []
    for recipe, target, alpha in grid:
        prefix = f"[{args.model}|{args.decoding}]"
        rows = run_condition(
            model, tok, recipe, target, alpha,
            args.decoding, args.n_runs, scorer,
            SEED_WORDS, device=args.device, log_prefix=prefix,
        )
        for r in rows:
            r["model"] = args.model
        all_rows.extend(rows)

    out_path = args.out
    if out_path is None:
        safe = args.model.replace("/", "_")
        out_path = RESULTS_DIR / f"dat_{safe}_{args.decoding}.json"
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"Saved {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
