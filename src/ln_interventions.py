"""LayerNorm interventions for pretrained Transformers.

Applies inference-time modifications to every torch.nn.LayerNorm in a model
without retraining. The interventions follow the geometry decomposition of
Brody et al. 2023 / Gupta et al. 2024:

    LN(x) = gamma * (x - mu) / sigma + beta

Recipes:
    'stock'        -- no change.
    'no_mean'      -- skip the mean-subtraction step (RMSNorm-like).
    'no_scale'     -- skip the sigma-scaling step (use sigma=1).
    'identity'     -- bypass LN entirely (only residual stream).
    'weakened_a'   -- alpha * LN(x) + (1-alpha) * x.
    'no_affine'    -- skip the gamma/beta affine step.

Layer-targeting is implemented by passing `target_layers={'early','mid','late','all'}`
or an explicit set of (block_index, ln_role) pairs.
"""
from __future__ import annotations

import contextlib
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Custom forwards
# --------------------------------------------------------------------------- #

def _ln_no_mean(ln: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    # RMSNorm-like: keep sigma scaling; do not subtract mean.
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(var + ln.eps)
    if ln.elementwise_affine:
        out = x_norm * ln.weight
        if ln.bias is not None:
            out = out + ln.bias
        return out
    return x_norm


def _ln_no_scale(ln: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    # Subtract mean, do not divide by sigma.
    x_centered = x - x.mean(dim=-1, keepdim=True)
    if ln.elementwise_affine:
        out = x_centered * ln.weight
        if ln.bias is not None:
            out = out + ln.bias
        return out
    return x_centered


def _ln_identity(ln: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    # Identity bypass; leaves residual untouched.
    return x


def _ln_no_affine(ln: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    # Plain LN without learnt gamma/beta: zero-mean, unit-variance.
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mu) * torch.rsqrt(var + ln.eps)


def _ln_weakened(ln: nn.LayerNorm, x: torch.Tensor, alpha: float) -> torch.Tensor:
    # Convex combination of LN output and input.
    out = nn.functional.layer_norm(
        x, ln.normalized_shape,
        ln.weight if ln.elementwise_affine else None,
        ln.bias if ln.elementwise_affine else None,
        ln.eps,
    )
    return alpha * out + (1.0 - alpha) * x


RECIPE_FNS = {
    "no_mean": _ln_no_mean,
    "no_scale": _ln_no_scale,
    "identity": _ln_identity,
    "no_affine": _ln_no_affine,
}


# --------------------------------------------------------------------------- #
# Intervention manager
# --------------------------------------------------------------------------- #

class LNIntervention:
    """Context manager that swaps every LN forward in a model.

    Usage:
        with LNIntervention(model, recipe="no_mean", target="all"):
            out = model.generate(...)

    target: 'all' | 'early' | 'mid' | 'late' | List[int]
        Layer indices to intervene on (block-level for decoder transformers).
        'early' / 'mid' / 'late' partition the blocks into thirds.
    """

    def __init__(
        self,
        model: nn.Module,
        recipe: str,
        target: str | List[int] = "all",
        alpha: float = 0.5,
    ):
        self.model = model
        self.recipe = recipe
        self.target = target
        self.alpha = alpha
        self._original_forwards: list[tuple[nn.LayerNorm, callable]] = []

    # -------- target resolution -------- #
    def _resolve_target_block_ids(self) -> set[int]:
        block_count = self._count_blocks()
        if isinstance(self.target, list):
            return set(self.target)
        if self.target == "all":
            return set(range(block_count))
        third = max(1, block_count // 3)
        if self.target == "early":
            return set(range(0, third))
        if self.target == "mid":
            return set(range(third, 2 * third))
        if self.target == "late":
            return set(range(2 * third, block_count))
        raise ValueError(f"Unknown target {self.target!r}")

    def _count_blocks(self) -> int:
        # Works for GPT-2/Neo/Pythia: model.transformer.h or model.gpt_neox.layers.
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return len(self.model.transformer.h)
        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return len(self.model.gpt_neox.layers)
        # Fallback: count LayerNorms / 2 (each block has ~2 LNs).
        n = sum(1 for m in self.model.modules() if isinstance(m, nn.LayerNorm))
        return max(1, n // 2)

    def _block_iter(self):
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return list(self.model.gpt_neox.layers)
        return []

    # -------- enter/exit -------- #
    def __enter__(self):
        if self.recipe == "stock":
            return self
        target_ids = self._resolve_target_block_ids()
        blocks = self._block_iter()

        # 1) intervene on per-block LayerNorms in target blocks
        for b_idx, block in enumerate(blocks):
            if b_idx not in target_ids:
                continue
            for sub in block.modules():
                if isinstance(sub, nn.LayerNorm):
                    self._patch(sub)
        # Note: we deliberately leave the FINAL LayerNorm alone (the "scale
        # anchoring" LN identified by Kanavalau et al. 2026 as critical).
        return self

    def _patch(self, ln: nn.LayerNorm):
        original_forward = ln.forward
        self._original_forwards.append((ln, original_forward))
        recipe = self.recipe
        alpha = self.alpha

        if recipe == "weakened_a":
            def new_forward(x, _ln=ln, _alpha=alpha):
                return _ln_weakened(_ln, x, _alpha)
            ln.forward = new_forward
            return

        fn = RECIPE_FNS.get(recipe)
        if fn is None:
            raise ValueError(f"Unknown recipe {recipe!r}")

        def new_forward(x, _ln=ln, _fn=fn):
            return _fn(_ln, x)
        ln.forward = new_forward

    def __exit__(self, exc_type, exc, tb):
        for ln, fwd in self._original_forwards:
            ln.forward = fwd
        self._original_forwards.clear()
        return False
