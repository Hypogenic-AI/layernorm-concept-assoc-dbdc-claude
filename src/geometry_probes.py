"""Geometric probes of concept-association in transformer hidden states.

Uses forward hooks to capture per-layer hidden states for a fixed bank of
concept words. Computes:

  * Pairwise cosine distances among concept embeddings (per layer).
  * Hidden-state L2-norm distribution std/mean (per layer).
  * Angle between hidden state and uniform vector 1/sqrt(d).

Used to test H1 (LN compresses cross-concept embedding distance).
"""
from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn


# A bank of "unrelated nouns" used as our concept probe set, taken
# from Olson 2021 / Chen & Ding 2023 example responses (not in any
# fixed test set, designed to span semantic categories).
DEFAULT_CONCEPT_BANK = [
    "hippopotamus", "violin", "tomato", "machinery", "ticket",
    "prickle", "jumper", "garden", "thunder", "library",
    "mountain", "ocean", "battery", "carpet", "mirror",
    "spoon", "rocket", "bread", "feather", "anchor",
]


@torch.no_grad()
def _last_token_hidden(model, tokenizer, words: List[str], device: str = "cuda"):
    """For each word, run the model on the standalone word and capture the
    hidden state at *every* layer at the last token position.

    Returns a tensor of shape (n_layers+1, n_words, d_model). The first
    entry is the embedding layer (input).
    """
    model.eval()
    states_per_word = []
    for w in words:
        # tokenize "word" -- use leading space so it's a noun token
        ids = tokenizer.encode(" " + w, return_tensors="pt").to(device)
        out = model(ids, output_hidden_states=True)
        # hidden_states: tuple of (n_layers+1) tensors, each [1, T, d]
        h = torch.stack([h[0, -1] for h in out.hidden_states], dim=0)  # (L+1, d)
        states_per_word.append(h.detach().cpu())
    return torch.stack(states_per_word, dim=1)  # (L+1, n_words, d)


def pairwise_cosine_distance(states: torch.Tensor) -> torch.Tensor:
    """Mean pairwise cosine *distance* (1 - cos_sim) over words.
    states: (n_words, d) for one layer.
    Returns: scalar tensor.
    """
    s = nn.functional.normalize(states, dim=-1)
    sim = s @ s.t()  # (n, n)
    n = s.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    pairs = sim[iu[0], iu[1]]
    return (1 - pairs).mean()


def all_pair_cosine_distances(states: torch.Tensor) -> torch.Tensor:
    """Return all (n*(n-1)/2,) pairwise distances for a layer."""
    s = nn.functional.normalize(states, dim=-1)
    sim = s @ s.t()
    n = s.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return (1 - sim[iu[0], iu[1]])


def norm_stats(states: torch.Tensor) -> dict:
    """std/mean of L2 norms across words."""
    norms = states.norm(dim=-1)
    return {
        "mean": norms.mean().item(),
        "std": norms.std().item(),
        "ratio": (norms.std() / norms.mean()).item(),
    }


def angle_to_uniform(states: torch.Tensor) -> dict:
    """Mean & std angle (deg) of hidden states to the uniform vector 1/sqrt(d)."""
    d = states.shape[-1]
    u = torch.ones(d, device=states.device) / (d ** 0.5)
    s_norm = nn.functional.normalize(states, dim=-1)
    cos = s_norm @ u
    angles = torch.rad2deg(torch.arccos(cos.clamp(-1, 1)))
    return {"mean_deg": angles.mean().item(), "std_deg": angles.std().item()}


def collect_geometry(model, tokenizer, words: List[str], device: str = "cuda"):
    """Run model on each word and return per-layer geometry dictionary."""
    states = _last_token_hidden(model, tokenizer, words, device=device)
    n_layers = states.shape[0]
    out = []
    for layer in range(n_layers):
        s = states[layer]  # (n_words, d)
        out.append({
            "layer": layer,
            "mean_pair_dist": pairwise_cosine_distance(s).item(),
            "norm_mean": norm_stats(s)["mean"],
            "norm_std": norm_stats(s)["std"],
            "norm_ratio": norm_stats(s)["ratio"],
            "uniform_angle_mean": angle_to_uniform(s)["mean_deg"],
        })
    return states, out
