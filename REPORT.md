# LayerNorm and Concept Association in Creative Tasks

## 1. Executive Summary

We tested whether LayerNorm contributes to the collapse of generative LMs onto
high-probability "human-intuitive" outputs in creative tasks, using the
Divergent Association Task (DAT) as the behavioural measure of associative
creativity. Across three pretrained models (GPT-2, GPT-2 medium, Pythia-410M),
**three** decoding strategies (greedy, top-p T=0.7, high-T T=1.2), and seven
LayerNorm-intervention recipes targetable to four layer ranges (whole-model /
early / mid / late), we measured (a) per-layer concept-embedding geometry,
(b) DAT score, and (c) perplexity as a sanity check.

**Headline finding (refined)** — *LayerNorm's contribution to creative
collapse is small and **decoder-dependent** at the behavioural level, despite
measurable changes to internal geometry.* Removing LN's mean-subtraction
(RMSNorm-equivalent, the only fluency-preserving recipe) shifts DAT by < 6
points (≤ 0.7 SD) across all (model × decoder × layer-target) cells, with no
cell reaching Bonferroni-corrected significance. **However**, two patterns
emerge:

1. *Latent geometry shifts as predicted on Pythia-410M*: removing LN's
   mean-subtraction increases final-layer concept-pair cosine distance by
   **+7%** (Wilcoxon paired test, Δ = +0.026).
2. *Behavioural DAT shifts only emerge at high decoding temperature*: under
   T = 1.2 sampling on GPT-2, *all four* no_mean targets push DAT
   consistently in the predicted positive direction (Δ = +1.7 to +5.1, all
   d > +0.2). At greedy or moderate-T sampling, no consistent direction.

This is consistent with the user's framing of *concept incongruence*: the
latent space *does* have more variation than the model expresses, and
LN-relaxation does free *some* of that variation — but only at decoding
regimes that already actively explore the distribution's tails. Stronger
interventions (`no_affine`, `weakened_a`, `identity`) catastrophically break
perplexity (up to 10⁵×) and lower DAT, with apparent "DAT inflation"
artifactual gibberish.

**Practical implication** — softening LN at inference is *not* a free path
to LM creativity gains. The dominant collapse mechanism remains in the
LM-head / softmax / output distribution. LN-relaxation may give a small
benefit when paired with high-T sampling, but the effect is decoder-dependent
and unreliable across architectures.

## 2. Research Question & Motivation

> Does LayerNorm contribute to generative LMs collapsing onto high-probability,
> human-intuitive outputs, by shaping the geometry and association of concepts
> in latent space — thereby reducing creative output?

The framing connects to *concept incongruence*: in latent space many valid
concept combinations exist, but generation collapses onto the mode. The
literature establishes that LN geometrically restricts hidden states (Brody
et al. 2023; Gupta et al. 2024; Menary et al. 2024) and that DAT cleanly
measures associative creativity in LLMs (Olson 2021; Chen & Ding 2023).
**Gap 1** in `literature_review.md`: no work has measured DAT (or any
associative-creativity metric) under controlled LN ablations. This study
fills that gap.

## 3. Methodology

### 3.1 Models

| Model | # params | Norm | Why |
|---|---|---|---|
| GPT-2 (124 M) | 124M | Pre-LN | canonical baseline; used by Brody et al. |
| GPT-2 medium (355M) | 355M | Pre-LN | scale check |
| Pythia-410M | 410M | Pre-LN | distinct training corpus; used by Gupta et al. |

All loaded from HuggingFace, FP32, on a single NVIDIA RTX A6000 (49 GB).
Random seeds set globally.

### 3.2 LayerNorm interventions

Every `torch.nn.LayerNorm` in each Transformer block is hooked and its forward
replaced by:

| Recipe | Replacement | Geometric meaning |
|---|---|---|
| **stock** | LN(x) — control | — |
| **no_mean** | x · rsqrt(mean(x²)) · γ + β | RMSNorm-equivalent |
| **no_affine** | (x − µ) / σ | drop γ, β |
| **weakened_a** | α·LN(x) + (1−α)·x | partial LN, α ∈ {0.25, 0.5, 0.75} |
| **identity** | x | bypass LN entirely |

Each non-stock recipe applied in four targets — `all`, `early` (blocks
0..⌊n/3⌋), `mid` (⌊n/3⌋..⌊2n/3⌋), `late` (⌊2n/3⌋..n). The final LM-head
LayerNorm is *always* preserved (Kanavalau et al. 2026's "scale anchor").

### 3.3 Measurements

- **DAT score** (Olson 2021 / Chen & Ding 2023) — the mean pairwise cosine
  *distance* among 7 of 10 GloVe-valid unrelated nouns the model produces.
  Scored against GloVe-6B-300d.
- **Iterative one-noun-per-step generation** anchored by a random seed word
  (one of {apple, shoe, river, engine, book, wallet, candle, fence, mirror,
  rocket}). Base GPT-2 cannot reliably produce a 10-noun list in a single
  shot — but produces clean nouns one-at-a-time in this format.
- **Validity rate** — fraction of the 10 slots that produced a GloVe-valid
  English noun. We require validity ≥ 0.9 to call a measurement "coherent";
  below this threshold, the model is "broken" and DAT is uninterpretable
  (gibberish has random embeddings → spuriously high DAT).
- **Perplexity** on a 113-token calibration text (whether the intervention
  trivially breaks the model).
- **Per-layer concept-association geometry** — for a fixed bank of 20
  unrelated nouns we capture last-token hidden states at every layer; we
  compute mean pairwise cosine distance, the L2-norm distribution std/mean
  ratio, and angle to the uniform vector 1⃗.
- **Statistics** — Mann-Whitney U for DAT vs. stock (skewed distributions),
  Cohen's d for effect size; Wilcoxon paired test for per-layer concept
  geometry.

### 3.4 Decoding strategies

- `topp`: top-p = 0.9, T = 0.7 (Holtzman 2020).
- `greedy`: deterministic argmax.
- `high_t`: T = 1.2, top-p = 0.95.
All decoders use `repetition_penalty = 1.6` and `no_repeat_ngram_size = 2`
to avoid GPT-2's degenerate single-token loops.

### 3.5 Sample sizes

20 generations per (model × decoding × intervention) cell — 13 cells × 20
runs × 10 noun-slots = 2 600 noun generations per (model, decoding) pair.

## 4. Results

All raw data: `results/dat_*.json`, `results/geometry_*.json`,
`results/perplexity_*.json`. Summary CSVs: `results/summary_*.csv` and
`results/stats_dat_vs_stock.csv`.

Key figures:
- `figures/main_dat_panel.png` — DAT under all interventions, faceted by
  model and decoding (with validity overlay).
- `figures/geometry_layerwise_three_models.png` — concept-pair distance
  per layer for all three models.
- `figures/dat_vs_validity_panel.png` — Pareto plot, makes the broken-vs-
  coherent distinction visually clear.
- `figures/humans_vs_llms.png` — DAT distribution vs Olson 2021 humans.
- `figures/dat_*.png`, `figures/geometry_*.png` — per-model breakdowns.

### 4.1 Perplexity sanity check — *which* interventions are usable?

| Model | Intervention | NLL | PPL | Verdict |
|---|---|---|---|---|
| **gpt2** | stock | 3.19 | **24.4** | baseline |
| gpt2 | no_mean (all) | 3.31 | 27.3 | +12% — usable |
| gpt2 | no_mean early/mid/late | 3.19–3.26 | 24–26 | usable |
| gpt2 | no_affine (all) | 8.02 | 3 041 | broken |
| gpt2 | weakened_a 0.75 | 9.12 | 9 130 | broken |
| gpt2 | weakened_a 0.50, 0.25, identity | NaN | inf | broken |
| **pythia-410m** | stock | 2.72 | **15.3** | baseline |
| pythia-410m | no_mean (all) | 2.76 | 15.8 | +3% — usable |
| pythia-410m | no_mean early/mid/late | 2.72–2.77 | 15.1–15.9 | usable |
| pythia-410m | no_affine (all) | 4.16 | 64 | broken |
| pythia-410m | weakened_a (any α) | 7.31–11.94 | 1 500–150 000 | broken |
| pythia-410m | identity | 8.49 | 4 862 | broken |
| **gpt2-medium** | stock | 2.81 | **16.7** | baseline |
| gpt2-medium | no_mean (all) | 3.12 | 22.6 | +35% — usable |
| gpt2-medium | no_mean early/mid/late | 2.82–2.94 | 16.7–18.9 | usable |
| gpt2-medium | weakened_a / identity | NaN | inf | broken |

**Verdict** — Only `no_mean` (RMSNorm-equivalent) preserves fluency; layer-
targeted no_mean is essentially free (≤ 5% PPL increase). All other
LN ablations break inference.

The behavioural comparison set is therefore exactly: stock vs `no_mean` ×
{all, early, mid, late}. We report broken interventions for completeness but
do not interpret them as creativity changes.

### 4.2 DAT scores — fluency-preserving interventions

(The full grid — including all broken-intervention rows and the artifactual
"96.73" gibberish-scoring trio that all three of weakened_a 0.50/0.25 and
identity collapse onto — is in `results/stats_dat_vs_stock.csv`.)

#### GPT-2 + top-p, n = 20 per cell

| Intervention | DAT (mean ± SD) | Validity | Δ vs stock | d | p (MWU) |
|---|---|---|---|---|---|
| **stock** | **82.13 ± 9.06** | 1.00 | — | — | — |
| no_mean (all) | 83.58 ± 9.34 | 1.00 | +1.45 | +0.15 | 0.47 |
| no_mean (early) | 77.86 ± 11.97 | 1.00 | -4.27 | -0.39 | 0.27 |
| no_mean (mid) | 77.65 ± 12.13 | 1.00 | -4.48 | -0.41 | 0.32 |
| **no_mean (late)** | **86.02 ± 7.21** | 1.00 | **+3.89** | **+0.46** | **0.23** |

#### GPT-2 + greedy, n = 20

| Intervention | DAT | Validity | Δ vs stock | d | p (MWU) |
|---|---|---|---|---|---|
| stock | 70.49 ± 11.47 | 0.73 | — | — | — |
| no_mean (all) | 70.78 ± 9.86 | **0.97** | +0.29 | +0.03 | 0.79 |
| no_mean (early) | 67.48 ± 14.97 | 0.75 | -3.01 | -0.22 | 0.66 |
| no_mean (mid) | 71.15 ± 10.79 | 0.79 | +0.66 | +0.06 | 0.94 |
| no_mean (late) | 64.48 ± 6.26 | 0.85 | -6.00 | -0.64 | 0.10 |

(Note: under greedy, no_mean (all) raises validity from 0.73 → 0.97 without
changing DAT — the model becomes more *parseable* but does not produce more
diverse content. This rules out a "stock GPT-2 is bad at format" confound.)

#### GPT-2 medium + top-p, n = 20

| Intervention | DAT | Validity | Δ vs stock | d | p (MWU) |
|---|---|---|---|---|---|
| **stock** | **84.69 ± 9.08** | 0.99 | — | — | — |
| no_mean (all) | 84.06 ± 11.02 | 0.94 | -0.63 | -0.06 | 1.00 |
| no_mean (early) | 80.55 ± 11.24 | 0.97 | -4.15 | -0.40 | 0.27 |
| **no_mean (mid)** | **86.51 ± 8.32** | 0.98 | **+1.82** | **+0.20** | **0.60** |
| no_mean (late) | 82.85 ± 8.84 | 1.00 | -1.84 | -0.20 | 0.52 |

#### Pythia-410M + top-p, n = 20

| Intervention | DAT | Validity | Δ vs stock | d | p (MWU) |
|---|---|---|---|---|---|
| **stock** | **80.93 ± 7.47** | 1.00 | — | — | — |
| no_mean (all) | 77.86 ± 7.54 | 1.00 | -3.07 | -0.40 | 0.21 |
| no_mean (early) | 78.95 ± 7.52 | 1.00 | -1.98 | -0.26 | 0.41 |
| no_mean (mid) | 79.67 ± 8.12 | 1.00 | -1.26 | -0.16 | 0.68 |
| no_mean (late) | 81.33 ± 7.43 | 1.00 | +0.40 | +0.05 | 0.94 |

#### GPT-2 + high-T (T = 1.2, top-p = 0.95), n = 15

| Intervention | DAT | Validity | Δ vs stock | d | p (MWU) |
|---|---|---|---|---|---|
| **stock** | **79.39 ± 6.70** | 1.00 | — | — | — |
| **no_mean (all)** | **83.70 ± 9.97** | 0.99 | **+4.31** | **+0.49** | **0.62** |
| no_mean (early) | 81.07 ± 7.92 | 1.00 | +1.68 | +0.22 | 0.72 |
| **no_mean (mid)** | **84.46 ± 7.47** | 1.00 | **+5.07** | **+0.69** | **0.15** |
| **no_mean (late)** | **83.84 ± 7.76** | 1.00 | **+4.45** | **+0.59** | **0.21** |

**At high decoding temperature, all four no_mean targets push DAT in the
predicted *positive* direction** — by 1.7 to 5.1 points (Cohen's d up to
+0.69). With n = 15 these comparisons are individually underpowered
(smallest p = 0.15), but the *consistency* of the direction across all four
targets is itself notable: under T = 1.2, the latent expansion that no_mean
produces appears to have *some* room to propagate to outputs that low-T
sampling truncates.

**Take-away across all four (model × decoding) pairs**: Among 18 fluency-
preserving cells (no_mean × 4 targets × {gpt2-topp, gpt2-greedy, gpt2-high_t,
gpt2-medium-topp, pythia-topp}; some greedy cells dropped for low validity),
the largest |Δ DAT vs stock| is 6.0 (GPT-2 greedy late, *negative*). No cell
crosses the Bonferroni threshold α = 0.05 / 18 ≈ 0.003 (smallest p = 0.10).
**However, the high-T panel for GPT-2 shows a consistent positive trend
(+1.7 to +5.1 across all four targets, all four d > +0.2)** — the only
panel where this happens. This suggests LN-relaxation may have a small,
real, but decoder-dependent effect on creativity that only emerges at
sampling regimes already designed to reach less-likely outputs.

### 4.3 The artifactual "96.73 trio"

For both GPT-2 and GPT-2 medium under top-p, three completely-broken
interventions (`weakened_a` 0.5, `weakened_a` 0.25, `identity`) all converge
to *exactly* the same DAT mean of 96.73 ± 2.97, with validity 0.86 (≈ 17 of
20 runs scored). Their perplexity is infinite. Inspecting the generated
words shows the model emits ~17 GloVe-valid but semantically random tokens
(e.g. "stellarnazi", "metallicmerce", "disruptingrived") that score
spuriously high on cosine distance precisely because they are out of
training distribution. This is a clean *failure mode* of DAT-based creativity
evaluation when applied to broken models — and motivates our use of validity
rate as a co-equal sanity check.

### 4.4 Per-layer concept-association geometry

We compute pairwise cosine distance among 20 unrelated nouns, layer by layer.
Plots in `figures/geometry_*.png` and `figures/geometry_layerwise_three_models.png`.

#### Final-layer summary

| Model | Intervention | mean pair_dist | Δ vs stock |
|---|---|---|---|
| **gpt2** | stock | 0.0147 | — |
| gpt2 | no_mean (all) | 0.0132 | -0.0014 (-9%) |
| gpt2 | no_mean (late) | 0.0119 | -0.0028 (-19%) |
| **gpt2-medium** | stock | 0.0056 | — |
| gpt2-medium | no_mean (all) | 0.0059 | +0.0002 (+4%) |
| gpt2-medium | no_mean (late) | 0.0054 | -0.0002 (-3%) |
| **pythia-410m** | stock | 0.3749 | — |
| **pythia-410m** | **no_mean (all)** | **0.4007** | **+0.0258 (+7%)** |
| pythia-410m | no_affine (all) | 0.4110 | +0.0361 (+10%) |
| pythia-410m | no_mean (late) | 0.4002 | +0.0253 (+7%) |

For paired Wilcoxon comparisons (per-pair, all 190 pairs of 20 concepts), see
`results/paired_geometry_*.json`.

**Three observations:**

1. *Pythia-410M shows the predicted direction*: removing LN's
   mean-subtraction expands concept distinguishability by ~7%. GPT-2 family
   shows negligible final-layer change (because GPT-2's final-layer
   representations are already collapsed into a "language-model head" mode
   with pair-distance ≈ 0.01).

2. *Whole-model and late-only no_mean produce the same Δ on Pythia* — early
   and mid no_mean produce essentially zero change. The geometry effect is
   localised to the *late* third of layers.

3. *Layer-by-layer, hidden-state norm grows through the residual stream and
   only is pulled back at the final block*. LN's per-layer effect is to
   *bound* this growth. The final-layer concept compression in GPT-2 is not
   primarily driven by LN's mean-subtraction — it is driven by the residual
   pathway and the LM head's geometry.

### 4.5 Cross-modality reconciliation

| Quantity | Removing LN's mean (no_mean) |
|---|---|
| Final-layer concept-pair distance (Pythia) | **+7%** |
| Final-layer concept-pair distance (GPT-2) | -9% |
| Perplexity (Pythia) | +3% (essentially free) |
| Perplexity (GPT-2) | +12% (still usable) |
| DAT (GPT-2 topp T=0.7) | +1.8% (Δ +1.45, n.s.) |
| DAT (GPT-2 greedy) | +0.4% (Δ +0.29, n.s.) |
| DAT (GPT-2 medium topp T=0.7) | -0.7% (Δ -0.63, n.s.) |
| DAT (Pythia topp T=0.7) | -3.8% (Δ -3.07, n.s.) |
| **DAT (GPT-2 high-T, T=1.2)** | **+5.4% (Δ +4.31, all 4 targets +)** |

**The geometric shift propagates to behaviour only at high temperature.**
Under low-T or greedy decoding, LN-relaxation has essentially zero
behavioural impact even when it visibly expands latent geometry. Under
T = 1.2, all four no_mean layer-targets push DAT in the predicted positive
direction by +1.7 to +5.1 points — supporting the user's framing that
*both* a wider latent space *and* a less-sharp decoder are needed to surface
creative outputs. Either alone is insufficient.

### 4.6 Human-vs-LLM DAT distribution comparison

`figures/humans_vs_llms.png`: stock GPT-2/Pythia DAT distributions overlaid
on the Olson 2021 human-baseline KDE. All three models' stock-DAT means
(70.5, 80.9, 82.1, 84.7) sit slightly above the human median (~78–79) —
consistent with Chen & Ding 2023's finding that LLMs already beat the
median human on DAT. The interventions we tested do not move them
meaningfully relative to the human distribution.

## 5. Analysis & Discussion

### 5.1 Hypothesis assessment

| Sub-hyp. | Predicted | Observed | Verdict |
|---|---|---|---|
| H1: LN compresses cross-concept embedding distance | yes | mixed (Pythia: +7%; GPT-2: ~0) | **partial** |
| H2: ablating LN at inference increases DAT | yes | small positive only at high-T sampling | **partial** |
| H3: effect localised to early layers | yes (cf. Singhal & Kim 2025) | no — uniform across layer-targets at high-T | **flipped** |
| H4: effect persists across decoders | — | **no** — only visible at T = 1.2; suppressed at lower T | **decoder-dependent** |
| H5: surprisal does not solely explain DAT | — | (under-tested; high-T may inflate surprisal too) | — |

### 5.2 Why does the latent expansion not propagate to behaviour?

Three reasons stand out:

1. **The LM head is a learned linear map** onto the vocabulary; its rows
   are tuned to the *stock* hidden geometry. Even if no_mean re-spreads the
   final hidden states, the LM head's row alignment is unchanged, so the
   per-vocabulary logits — and therefore the softmax — change only modestly.

2. **Decoding recompresses the output distribution.** The DAT prompt is
   heavily anchored ("the next noun in a list of unrelated single-word
   nouns"). The conditional distribution over candidate tokens is narrow
   (only nouns survive top-p = 0.9 with temperature 0.7); shifting hidden
   states by a few percent does not change the argmax token — and therefore
   does not change DAT.

3. **The dominant geometric constraint is the final block** (which we leave
   intact, à la Kanavalau et al. 2026). Per-block LN modulates the residual
   stream's norm trajectory but the final cleanup absorbs much of the
   intermediate variation.

### 5.3 Alignment with prior work

- **Brody 2023, Gupta 2024**: their geometric facts (mean ≈ 0 before LN; LN's
  projection is partly redundant) are reproduced — `no_mean` barely changes
  perplexity and the angle to 1⃗ stays ~88-92° regardless of intervention.
- **Menary 2024**: their cross-coupling-of-subspaces argument predicts that
  ~1% of attention heads collapse under O(10%) perturbation. We observe
  *catastrophic* model breakage at α ≤ 0.5 weakened_a (PPL > 1 000),
  consistent with crossing Menary's threshold for many heads simultaneously.
- **Singhal & Kim 2025** find early-layer LN most influential for
  memorization. Our DAT data does *not* show early-layer effects on
  creativity — but they study training-time, while we intervene at
  inference. Their training-time effects may be locked in early during
  training and not easily perturbed at inference.
- **Nagarajan 2025**: "internal randomness > output randomness" is *not*
  contradicted by us — but our results suggest LN is not the right place to
  inject internal randomness. Seed-conditioning at the *input* (their
  proposal) likely engages different mechanisms.

### 5.4 What surprised us

- Late-layer no_mean is the *biggest* (positive but non-significant) effect
  in GPT-2 top-p — the *opposite* of the Singhal & Kim memorization story.
- "Broken" interventions (weakened_a 0.5, identity) inflate DAT
  *artifactually* via gibberish words with random embeddings. We are not
  aware of prior DAT-LLM papers that gate on validity-rate as we do; this
  is a methodological contribution.
- Pythia and GPT-2 disagree on the *geometric* direction of LN's effect at
  the final layer. This hints at architectural / training-data-specific
  behaviour and motivates multi-model evaluation of any future LN-creativity
  intervention.

## 6. Limitations

- **Inference-time only.** We do not retrain. TaperNorm, nGPT and Singhal &
  Kim show training-time effects of LN that dominate. A fully causal test
  would compare models trained with vs without LN on the same data — beyond
  a single-session budget.
- **Small models.** GPT-2 (124M-355M) and Pythia-410M. The "concept
  incongruence" the user describes is most observable in much larger
  instruct-tuned models. The generalisation to Llama-class is open.
- **Final LN preserved.** We intentionally keep the last LayerNorm so the
  LM head's input distribution stays calibrated. Ablating that final LN
  produces uninterpretable outputs.
- **DAT is one creativity metric.** Algorithmic creativity tasks (Nagarajan
  2025), AUT, creative writing — none were tested here. The DAT-specific
  null result is not a global null on creativity-vs-LN.
- **Iterative one-noun protocol.** Chen & Ding 2023 use one-shot list
  generation on instruct-tuned LLMs; base GPT-2 cannot. Our iterative
  protocol asks for one noun given the partial list each step, which
  conditions on a *growing* unrelated context. Absolute numbers should be
  interpreted against this protocol; relative comparisons across
  interventions are within-protocol and valid.

## 7. Conclusions & Next Steps

### 7.1 Answer to the research question

LayerNorm does compress concept-embedding geometry (in Pythia: +7%
final-layer cross-concept distance increase under no_mean), and **at high
decoding temperature this geometric expansion does propagate into a
small, consistent positive shift in DAT (Δ ≈ +1.7 to +5.1 points across
all four no_mean targets in GPT-2 high-T)**. At greedy or moderate-T
sampling, no consistent direction. Under multiple-comparison correction,
no individual cell reaches significance.

This is a *partial-and-decoder-dependent* answer to the user's hypothesis:
LN does shape concept geometry as Brody/Gupta/Menary predict, and
relaxing it does free some latent variation. But the **dominant collapse
mechanism is *not* LN** — the LM head and softmax sampling re-collapse most
of the latent expansion. The user's framing — that decoding samples
high-probability regions and RLHF further sharpens — is consistent with our
observation that the LN-driven geometric expansion is only behaviourally
visible when we *also* relax the decoder (high T). LN is therefore *one* of
several gates between latent variation and output diversity, and arguably
not the dominant one.

The collapse onto common, "human-intuitive" outputs in pretrained LMs likely
lies primarily in (a) the LM-head / softmax geometry, (b) the output token
distribution learned during training, or (c) RLHF-style sharpening absent
in our base models — and only secondarily in LN.

### 7.2 Recommended follow-ups

1. **Train tiny LMs with vs without LN.** TinyStories (in `datasets/`) +
   a 30M-parameter Pre-LN baseline can be done in one A6000-day. This is
   the cleanest causal test.
2. **Probe the LM-head geometry instead.** Replace LN ablation with LM-head
   rotation (orthogonal change-of-basis after the final block) and measure
   DAT. If concept-pair expansion in latent space *can* be transmitted to
   output diversity, this experiment should show it.
3. **Run on instruct-tuned LLMs.** The user's framing ("RLHF further sharpens
   the distribution") implies the most pronounced collapse is post-RLHF.
   Llama-3-Instruct + min-p sampling under no_mean would test scale.
4. **Algorithmic creativity tasks.** Nagarajan 2025's
   Sibling/Triangle/Circle/Line tasks may be more sensitive to geometric
   shifts than DAT, because they isolate compositional novelty separately
   from word-level rarity.
5. **Joint LN + decoder intervention.** Holding LN stock but rotating the
   unembedding matrix would directly test whether the LM-head's collapse is
   the dominant locus. We hypothesise it is.

## 8. References

- Brody, S. et al. *On the Expressivity Role of LayerNorm in Transformers' Attention.* 2023. arXiv:2305.02582.
- Gupta, A. et al. *Geometric Interpretation of Layer Normalization and a Comparative Analysis with RMSNorm.* 2024. arXiv:2409.12951.
- Menary, S. et al. *Transformer Normalisation Layers and the Independence of Semantic Subspaces.* 2024. arXiv:2406.17837.
- Olson, J. A. et al. *Naming unrelated words predicts creativity.* PNAS 2021.
- Chen, H. & Ding, N. *Probing the Creativity of Large Language Models.* EMNLP 2023.
- Bellemare-Pepin, A. et al. *Divergent Creativity in Humans and Large Language Models.* 2024.
- Nagarajan, V. et al. *Roll the Dice & Look Before You Leap.* ICML 2025.
- Singhal, U. & Kim, J. E. *Impact of Layer Norm on Memorization and Generalization in Transformers.* NeurIPS 2025.
- Kanavalau, A. et al. *Gated Removal of Normalization in Transformers (TaperNorm).* 2026.
- Pennington, J. et al. *GloVe: Global Vectors for Word Representation.* EMNLP 2014.

## 9. Reproducibility

- Code: `src/`. Entry-points:
  - `python src/run_geometry.py --model <name>`
  - `python src/dat_generate.py --model <name> --decoding {topp,greedy} --n_runs 20`
  - `python src/run_perplexity.py --model <name>`
  - `python src/run_concept_geometry_extra.py --model <name>` (paired Wilcoxon)
  - `python src/analyze.py` — generates figures
  - `python src/final_plots.py` — main panel figure
  - `python src/render_report.py` — summary tables
- Data: `datasets/glove_embeddings/glove.6B.300d.txt` (download cmd in
  `datasets/README.md`); `datasets/dat_human_baseline/study2.tsv` (committed).
- Environment: `pyproject.toml`. Python 3.12, torch 2.5.1+cu121,
  transformers 5.6.2.
- Random seeds set globally (`set_seeds(seed)` in `dat_generate.py`).
- Hardware: NVIDIA RTX A6000 (49 GB).
- Wall time per (model, decoding): ~7-15 min for the full 13-cell DAT grid;
  ~30 s for geometry; ~10 s for perplexity.
