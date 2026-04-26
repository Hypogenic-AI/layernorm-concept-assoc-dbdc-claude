# Planning Document — LayerNorm and Concept Association in Creative Tasks

## Motivation & Novelty Assessment

### Why This Research Matters

Modern language models routinely generate fluent text but reliably collapse onto
common, "human-intuitive" outputs. The user-provided framing connects this to
*concept incongruence*: in latent space many valid combinations exist, but the
generation process collapses to the mode (a) because decoding samples
high-probability regions and (b) because RLHF further sharpens the distribution.
A complementary, less-explored hypothesis is that LayerNorm — applied at every
block of every Transformer — itself participates in this collapse, by stripping
representational degrees of freedom (Brody 2023, Gupta 2024) and coupling
semantic subspaces (Menary 2024). If true, this is a *mechanistic* lever for
creativity that is independent of decoder/RLHF interventions.

### Gap in Existing Work (from `literature_review.md`)

- Brody, Gupta, Menary, and Chu (2023–2024) describe LN's geometry: it forces
  hidden states onto a fixed-radius sphere and couples concept subspaces.
- Olson 2021, Chen & Ding 2023, Bellemare-Pepin 2024, and Nagarajan 2025
  measure LLM creativity (DAT, AUT, algorithmic-creativity tasks).
- **No work has directly measured creativity (DAT) under controlled LayerNorm
  ablations.** This is Gap 1 in our literature review and the contribution of
  this study.

### Our Novel Contribution

We test, on a panel of pretrained Transformer LMs (GPT-2 / GPT-Neo / Pythia),
whether intervening on LayerNorm at inference time changes:

1. **Concept-association geometry** — pairwise cosine distance among "unrelated
   nouns" embeddings across layers, hidden-state norm distribution.
2. **Creative output** — DAT scores when the model is prompted for divergent
   association; controlling for surprisal, decoding strategy, and base
   perplexity.

We use *causal interventions* (hook the model and modify LN) rather than
training new architectures. This makes the study fast (single-session, no
training run) yet causal: changes in DAT can be attributed to the intervention.

### Experiment Justification

- **Experiment 1 — LN-geometry baseline.** Confirm Gupta 2024 / Brody 2023
  geometric facts on our chosen models (norm distribution, mean ≈ 0, angle vs
  uniform vector). Without this, downstream interventions are uninterpretable.

- **Experiment 2 — Concept-association geometry per layer.** For a fixed bank
  of nouns, extract per-layer hidden states *with* and *without* LN, and
  measure pairwise cosine distances. Tests whether LN compresses concept
  separation.

- **Experiment 3 — DAT under LN interventions.** Prompt the model for 10
  unrelated nouns and score with GloVe (Olson scorer). Compare:
  (a) Stock LN (Pre-LN), (b) RMSNorm-style (no mean subtraction),
  (c) γ-frozen LN (no learned re-scaling), (d) "weakened" LN
  (attenuated normalization, partial re-introduction of magnitude).
  Critical: this is the first direct DAT-vs-LN measurement.

- **Experiment 4 — Decoding-strategy interaction.** Repeat (3) under greedy,
  top-p (0.9, T=0.7), and high-temperature sampling. Tests whether LN's role
  is decoder-confounded.

- **Experiment 5 — Surprisal / sanity controls.** Measure perplexity on a
  calibration sample and surprisal of the DAT-generated words. Confirms
  interventions don't merely "break" the model, and disentangles
  rare-word generation from divergent association.

---

## Research Question

> Does LayerNorm contribute to the collapse of generative LMs onto
> high-probability, human-intuitive outputs, and does it shape the geometry
> and association of concepts in the latent space — thereby reducing creative
> output?

We operationalise this as three sub-questions:

- **Q1 (geometric):** Does LN reduce pairwise cosine distance among "unrelated"
  concept embeddings, layer by layer?
- **Q2 (behavioural):** Does ablating / weakening LN at inference *increase*
  DAT scores, beyond what decoder changes achieve?
- **Q3 (interaction):** Is the effect localised to specific layers
  (per Singhal & Kim 2025: early-layer LN is most influential)?

## Hypothesis Decomposition

| Sub-hyp. | Statement | Predicted direction |
|---|---|---|
| H1 | LN compresses cross-concept embedding distance vs raw activations. | cosine(LN) < cosine(no-LN) at most layers |
| H2 | Ablating LN at inference increases DAT score. | DAT(ablated) > DAT(stock) |
| H3 | Effect is layer-specific; early layers dominate. | Early-layer ablation > late-layer ablation |
| H4 | Effect persists across decoders. | Effect not subsumed by min-p / high-T |
| H5 | Surprisal does not solely explain the DAT change. | Partial-correlation residual ≠ 0 |

## Methodology

### Models

- **GPT-2 small (124M)** — canonical Pre-LN baseline.
- **GPT-2 medium (355M)** — scale check.
- **Pythia-410M** — Pre-LN, decoder-only, distinct training corpus, used by
  Gupta 2024.
- **GPT-Neo-125M** *(stretch)* — Pre-LN, additional architecture.

All on RTX A6000 GPU, batched inference (no training).

### LN Intervention Recipes

We hook every `nn.LayerNorm` in each Transformer block and replace its forward
with one of:

1. **Stock** (no change) — control.
2. **No-mean** (RMSNorm-like) — drop µ subtraction; keep σ scaling and γ, β.
3. **No-scale** — drop σ scaling (use σ=1); keep mean subtraction and affine.
4. **Identity** — bypass normalization entirely (only the residual stream).
5. **Weakened-α** — interpolate: `α·LN(x) + (1-α)·x`, for α ∈ {0.25, 0.5, 0.75}.
6. **Layer-targeted ablation** — only ablate layers in {early third, mid third,
   late third}, leave the rest stock.

Note: identity and no-scale will likely fail catastrophically without
fine-tuning; we report what happens but expect generation degradation. The
*scientifically interesting* recipes are weakened-α and layer-targeted, which
preserve generation quality while modulating LN strength.

### Tasks

- **DAT prompt** (canonical, from Chen & Ding 2023):
  > "Please write 10 nouns in English that are as different from each other
  > as possible, in all meanings and uses of the words. Rules: Only single
  > words. Only nouns (e.g. things, objects, concepts). No proper nouns
  > (e.g. no specific people or places). No specialized vocabulary
  > (e.g. no technical terms). Think of the words on your own
  > (e.g. do not just look at objects in your surroundings)."
- **Decoding strategies**: greedy; top-p=0.9, T=0.7; T=1.2; T=1.5.
- **N runs per condition**: 30 — enough for non-parametric tests.

### Evaluation Metrics

| Metric | Source | What it tells us |
|---|---|---|
| **DAT score** | Olson scorer + GloVe-840B-300d | primary creativity outcome (0–100) |
| **Validity rate** | fraction of outputs producing ≥7 valid nouns | sanity / generation-quality control |
| **Surprisal** of generated nouns | model log-prob | confound check |
| **Perplexity on calibration text** | model NLL | confirms model isn't broken |
| **Layer-wise pairwise cosine** of noun-token hidden states | hooks | concept-association geometry |
| **Embedding L2-norm spread (std/mean)** | hooks | norm-collapse evidence (Gupta 2024) |
| **Angle to uniform vector `1⃗`** | per-layer | geometric sanity check |

### Statistical Plan

- **Primary** — Mann–Whitney U test on DAT scores between stock and each
  intervention condition (non-parametric; DAT distributions are skewed).
- **Effect size** — Cohen's d on DAT means, plus rank-biserial correlation.
- **Multiple comparisons** — Bonferroni or BH-FDR over the intervention grid.
- **Confound controls** — partial correlation of DAT with intervention strength
  controlling for mean surprisal.
- **Geometry tests** — paired t-test on per-layer cosine distance (paired by
  noun pair).

### Baselines

- **Stock model + greedy** — the lowest-creativity reference.
- **Stock model + top-p (0.9, T=0.7)** — Holtzman 2020.
- **Stock model + T=1.5** — high-T reference.
- **Random WordNet 7 nouns** — implicit upper bound for DAT (no model).

### Reproducibility

- Random seeds set globally (random, numpy, torch.manual_seed,
  torch.cuda.manual_seed_all).
- All model checkpoints fetched from HuggingFace at fixed revisions.
- All prompts, decoding params, and intervention configs serialized to JSON.
- All raw model outputs (10-noun strings) saved to `results/raw_outputs/`.

## Expected Outcomes

- **Strongest H2 result we'd accept**: weakened-α ∈ [0.5, 0.75] increases DAT
  by ≥3 points (≈¼ stdev of the human-baseline) on at least one model, with
  validity rate ≥ 0.8.
- **Geometric prediction (H1)**: layer-wise cosine distance among the noun-bank
  embeddings should *increase* under no-mean / weakened-α (mean subtraction
  removes a degree of freedom).
- **Null result is informative**: if no LN intervention shifts DAT (and
  generation remains valid), this *contradicts* the literature's prediction
  that LN constrains creative output — and is itself worth reporting.

## Timeline

| Phase | Time |
|---|---|
| Planning + env (this) | 30 min |
| Acquire GloVe (~5GB) | 15 min |
| Implement intervention framework | 45 min |
| Implement DAT + geometry probes | 45 min |
| Run experiments | 60 min |
| Analysis + visualization | 30 min |
| REPORT.md | 30 min |

## Potential Challenges & Mitigations

- **GloVe download time** → use 6B-300d as fallback (smaller).
- **Identity-LN catastrophically breaks generation** → expected; treat as
  boundary condition, not main result.
- **GPT-2's DAT prompting may yield gibberish** → use few-shot prompt with
  Olson 2021 examples; fallback to extracting first-N nouns from any noun-rich
  output via spaCy/regex.
- **Compute budget** → 30 runs × 3 decoding × 6 conditions × 4 models = 2160
  generations × ~1s each = manageable on a single A6000.
- **GPT-2 may not follow instructions well** → augment with few-shot examples
  in prompt and use stop tokens.

## Success Criteria

- All five experiments executed end-to-end with results saved.
- At least one statistical comparison reported with effect size and CI.
- REPORT.md contains a clear answer to Q1, Q2, Q3 supported by data.
- Code is reproducible from `pyproject.toml` + `src/`.
