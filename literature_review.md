# Literature Review

## Research hypothesis

> Layer normalization may contribute to the collapse of generative models onto
> high-probability, human-intuitive outputs, reducing the expression of creative or
> incongruent concept combinations. Investigating whether LayerNorm affects the
> geometry and association of concepts in latent space could reveal mechanisms
> underlying reduced creative output in language models.

The hypothesis straddles two literatures: (1) the geometry and computational role of
LayerNorm in Transformer attention, and (2) creativity and divergent semantic
association in LLMs. We surveyed 35 papers across both, with deep reads of 6 highly
relevant works.

---

## 1. Research area overview

### 1.1 LayerNorm geometry — what LN actually does

LayerNorm `LN(x) = γ ⊙ (x − µ)/σ + β` admits a clean geometric interpretation,
established by **Brody et al. 2023** and refined by **Gupta et al. 2024**:

1. **Mean subtraction** = projection of `x` onto the hyperplane orthogonal to the
   uniform vector `1⃗ = [1,…,1]`.
2. **Division by σ** = scaling of the projected vector to norm exactly √d.
3. **`γ, β` (learnable)** = element-wise stretch and translate.

Steps 1 and 2 force every hidden state onto a fixed-radius sphere intersected with the
hyperplane `H ⊥ 1⃗`. RMSNorm (Zhang & Sennrich 2019) drops step 1 and keeps only the
scaling — this matters less than expected because hidden states empirically have near-
zero mean *before* LN as well (Gupta et al. 2024 across 7 LLMs incl. GPT-J, Pythia,
Llama).

**Why this matters for creativity**:

- Magnitude is removed as a representation degree of freedom. If magnitude encodes
  *salience* or *confidence* of a concept, LN strips it.
- Repeated layer-by-layer LN is *irreversible* (Gupta 2024) — only `γ, β` (2d scalars)
  are learnable, but `T·d` per-token statistics get discarded.

### 1.2 Computational consequences for attention

**Brody et al. 2023** show that LN's two components have independent attention-related
roles:

- **Projection** lets attention queries be built parallel to `1⃗`, yielding *uniform*
  attention over keys (because all keys are orthogonal to `1⃗`). This makes "majority"
  computations easy — and converges 3× faster than without projection.
- **Scaling** prevents the *unselectable-key* problem (Demeter et al. 2020): a key
  inside the convex hull of others can never get max attention. After LN, all keys are
  on a sphere (every key is on the convex hull). At `d=8`, removing scaling left
  **32–51% of keys unselectable per layer**.

**Menary et al. 2024** extend this to the *composition of concepts*. Define a
*semantic subspace* `S_α ⊂ S^N` as one that can fully determine an attention
distribution. Pre-Norm (the standard) replaces `P_α x = x_α` with
`x_α / ||Σ_β x_β||`, **cross-coupling subspaces** through a shared denominator. The
only structure that avoids interference is one in which each subspace is itself a
sphere `S^{N_α-1}` and all spheres are mutually orthogonal — a strict, low-d.o.f.
constraint.

Their empirical claim: under O(10%) L2-norm perturbation, ~1% of sparse attention
heads exhibit *circuit collapse* (the head jumps to a different attended token). This
matters because **out-of-distribution generalization is exactly when interference is
not learned away** — and creative composition is fundamentally OOD relative to
training data.

### 1.3 Memorization, generalization, and LN

**Singhal & Kim (NeurIPS 2025)** show divergent LN roles in Pre-LN vs Post-LN:

- Pre-LN: removing LN destabilizes learning AND increases memorization.
- Post-LN: removing LN suppresses memorization without harming learning.
- *Early-layer* LN is the most influential.

This connects to creativity: a model that memorizes training distributions will
regurgitate, not associate divergently. The intervention space is rich.

### 1.4 Norm-removal architectures

A growing literature *removes* per-token normalization:

- **nGPT** (Loshchilov et al. ICLR 2025): every vector unit-norm; hidden state walks
  the hypersphere. 4–20× faster training, but no published creativity eval.
- **TaperNorm** (Kanavalau et al. 2026): gated cosine-decay from RMSNorm to a fixed
  linear map; identifies "scale anchoring" via the *final* normalization as the
  critical role (preventing "logit chasing" in cross-entropy).
- **DyT** (Zhu et al. 2025, cited in TaperNorm): dynamic element-wise nonlinearities
  replacing LN.
- **Baroni et al. 2025** (cited in TaperNorm): staged LN-removal in GPT-2 with small
  accuracy loss.

These provide *concrete recipes* to ablate LN in trained models — perfect intervention
substrate for our study.

### 1.5 Creativity in LLMs — measurement

**Chen & Ding (EMNLP 2023)** establish DAT (Divergent Association Task) as a clean,
objective measure of "associative creativity":

- Generate 10 unrelated nouns; take the first 7 valid ones; compute average pairwise
  cosine distance via GLoVe-300d (Olson et al. 2021):
  `DAT = (100 / n(n-1)) Σ_{i≠j} (1 − cos(v_i, v_j))`.
- GPT-4 (greedy) DAT = 89.1 — beats 96% of 8,572 humans. GPT-3.5-turbo > avg human.
- Smaller models lower; ~proportional to scale.
- Stochastic decoding raises DAT for non-GPT-4 but introduces invalid outputs.
- Correlates with surprisal (rare words = farther embeddings); controlling for
  surprisal, GPT-4's lead shrinks.

**Bellemare-Pepin et al. 2024** confirm with 100,000 humans across DAT and creative
writing (haiku, story, flash fiction). LLMs > avg human, < top humans.

**Nagarajan et al. (ICML 2025 spotlight)** introduce *algorithmic-creativity tasks*:
Sibling/Triangle Discovery, Circle/Line Construction. Findings:

- Multi-token (teacherless / diffusion) > next-token prediction for diversity.
- *Seed-conditioning* (input randomness) ≥ temperature sampling for diversity.
  Implies: "creativity at the input/internal-state level" beats "creativity at the
  output/decoding level" — directly aligned with our hypothesis.

**He et al. 2025** (Shakespearean Sparks) probe layer-wise via Layer-Skip: optimal
hallucination/creativity balance is in *early layers* of larger models.

### 1.6 Concept geometry in latent space

A parallel literature on the *linear-representation hypothesis* shows concepts are
encoded as linear directions: **Marks & Tegmark 2023** (truth), **Park et al. 2024**
(linear representations), **Geva et al. 2022** (FFN layers as concept promoters),
**Belrose et al. 2023** (Tuned Lens). These give us probes for concept-association
geometry under LN ablations.

---

## 2. Common methodologies

| Method | Used in | Notes |
|---|---|---|
| **DAT scoring** | Chen & Ding 2023, Bellemare-Pepin 2024, Olson 2021 | Cosine distance via GLoVe; canonical creativity metric. |
| **Algorithmic-creativity tasks** | Nagarajan 2025 | Sibling/Triangle/Circle/Line; coherent ∧ unique ∧ original. |
| **Layer-Skip / early exit** | He et al. 2025 | Per-layer probing during inference. |
| **Tuned Lens / Logit Lens** | Belrose 2023 | Per-layer prediction extraction. |
| **Linear probing** | Marks 2023, Park 2024 | Test linearity of concepts. |
| **L2-norm perturbation** | Menary 2024 | Test circuit stability. |
| **LN ablation** | Brody 2023, Singhal & Kim 2025, Baroni 2025, TaperNorm 2026 | Replace LN with identity / linear / RMSNorm / QKV-Norm / TaperNorm. |
| **Convex-hull analysis** | Brody 2023 (via Demeter 2020) | Identify unselectable keys. |
| **Hidden-state collection** | Gupta 2024 | 1M Wikipedia tokens × 7 LLMs; ~2-4 TB. |

---

## 3. Standard baselines

For LayerNorm ablations:
- **Stock Pre-LN GPT-2** — most common comparator in interpretability literature.
- **RMSNorm** (Llama family) — drop-in for LN, slightly cheaper.
- **No-Norm** — fails at scale without scale anchoring (Kanavalau 2026).
- **QKV-Norm** — minor variant, Pre-LN-equivalent for separability.
- **nGPT** — extreme "fully-normalized" baseline.

For creativity:
- **Greedy decoding** baseline (lowest entropy reference).
- **Top-p (p=0.9, T=0.7)** — Holtzman 2020.
- **Temperature sweep**.
- **Min-p sampling** (Nguyen ICLR 2025) — current state-of-the-art creative decoder.
- **Random WordNet sampling** — upper-bound DAT (no language constraint).

---

## 4. Evaluation metrics

| Metric | Range | What it measures | When to use |
|---|---|---|---|
| **DAT score** | 0–100 | Avg cosine distance among 7 nouns | Concept-association divergence, single-prompt |
| **Algorithmic creativity** | 0–1 | Coherent ∧ unique ∧ original fraction | Algorithmic tasks (Sibling/Triangle/Circle/Line) |
| **HCB score** (He 2025) | 0–1 | `w_c·S_c + w_h·(1−S_h)` (creative correctness) | Layer-wise tradeoff vs hallucination |
| **Surprisal** | 0–∞ | `−log P(token)` | Confound for DAT (rarer words → larger distances) |
| **Embedding L2-norm spread** | std/mean | Cross-token variance of `||x||` | Predictor of subspace interference (Menary 2024) |
| **Rank of `WQK`, `WV`** | int | Concept-subspace dimensionality | Mechanistic interpretability |
| **Cosine of hidden ⟂ 1⃗** | degrees | Evidence that LN's projection step is redundant (Gupta 2024) | Sanity-check our LN-modification target |

For our hypothesis, **DAT** is the primary outcome metric, with algorithmic-creativity
as a secondary check on causal mechanisms.

---

## 5. Datasets in the literature

| Dataset | Used for | Source |
|---|---|---|
| Olson 2021 DAT human baseline (~8.5K humans) | Reference distribution | OSF: vjazn / kbeq6 |
| GLoVe 840B 300d | DAT scoring | Stanford NLP / HF |
| Wikipedia (en, 20220301) | Internal-state probing | HF `wikipedia` |
| TinyStories | From-scratch LM training under LN ablations | HF `roneneldan/TinyStories` |
| WritingPrompts preferences | Diversity-aware preference tuning | HF `euclaise/WritingPrompts_preferences` |
| Algorithmic creativity tasks | Sibling/Triangle/Circle/Line | Procedurally generated |
| AUT (Alternative Uses Test) | Alternative creativity probe | Olson lab; sample-dependent |

---

## 6. Gaps and opportunities

The literature has done parts of the puzzle but no one has bolted them together:

**Gap 1 — No work measures DAT (or any associative-creativity metric) under
controlled LayerNorm ablations.** Brody, Gupta, and Menary establish what LN does
geometrically; Chen & Ding establish DAT — but no one has crossed the streams.

**Gap 2 — The "internal-randomness > output-randomness" finding (Nagarajan 2025) is
suggestive of an internal-geometry mechanism, but not localized to LN.** The role of
LN versus other architectural choices in the cognitive-overload phenomenon is unknown.

**Gap 3 — Layer-wise creativity (He et al. 2025) is observed but not mechanized.** We
don't know *why* early layers govern the creativity-hallucination tradeoff. Their
HCB-optimal layer is also where Singhal & Kim 2025 find LN-memorization effects
strongest.

**Gap 4 — nGPT, RMSNorm, and TaperNorm have all been shown to preserve general
performance but their effects on creative-divergence metrics are unmeasured.** It is
possible — and this is the hypothesis — that they actually *worsen* creative
divergence even as they preserve perplexity / accuracy.

**Gap 5 — No concept-association *geometry* analysis exists at the level of
divergent-pair retrieval.** Linear-concept work (Marks, Park, Geva) studies *single*
concepts. Concept-*association* (creative combination) requires interaction geometry.

---

## 7. Recommendations for our experiment

Given the gaps and the resources we have gathered, the cleanest design is:

### 7.1 Recommended datasets (in priority order)

1. **DAT human baseline** (`datasets/dat_human_baseline/study2.tsv`) — already
   pre-staged. 8,572 humans, our reference distribution.
2. **GLoVe 840B 300d** — required for DAT scoring; download via
   `datasets/README.md` instructions.
3. **TinyStories** — for from-scratch LM training under LN ablations. Used by
   TaperNorm paper as a clean pre-training corpus.
4. **Algorithmic-creativity task data** (procedurally generated via
   `code/algorithmic_creativity/{task}/{*.ipynb}`) — secondary creativity benchmark
   with exact, automatic scoring.
5. **Wikipedia (1M tokens)** — for representation probing of pretrained LLMs (norm
   distributions, subspace interference).
6. *(Optional)* **WritingPrompts preferences** for a writing-quality eval if needed
   for higher-stakes creative tasks.

### 7.2 Recommended baselines (Transformer normalization configurations)

For from-scratch experiments on TinyStories or the algorithmic-creativity tasks:

| Config | Description | Purpose |
|---|---|---|
| **Pre-LN** | Standard GPT-2 setup | Default baseline |
| **RMSNorm** | Skip mean-subtraction | Test mean-subtraction's role |
| **QKV-Norm** | LN after WQ/WK/WV (Menary 2024) | Test subspace-interference hypothesis |
| **No-LN + final-LN anchor** (à la Baroni 2025) | Removes per-token internal LN | Test scale-anchoring vs full LN |
| **TaperNorm** | Schedule from LN to fixed scaling | Inference-time clean LN removal |
| **nGPT** | All-vector hypersphere | Extreme normalization |
| **No-LN, no anchor + fixed-target scale loss** (Kanavalau 2026 §3.4) | Fully norm-free | Tests "is LN strictly necessary?" |

### 7.3 Recommended metrics

- **Primary**: DAT score (vs human-baseline percentile).
- **Primary**: algorithmic-creativity (coherent ∧ unique ∧ original) on Sibling /
  Triangle / Circle / Line.
- **Mechanistic**: per-layer norm distribution std (Gupta 2024 §3.1), embedding-vs-
  uniform-vector angle distribution (Gupta 2024 §4), L2-norm spread of `q,k,v` vectors
  (Menary 2024).
- **Concept-association probes**: cosine distance among concept directions (e.g.,
  10 unrelated nouns) at every layer pre-/post-LN.
- **Confounds**: surprisal, perplexity, downstream task accuracy (sanity that LN
  ablations haven't broken the model).

### 7.4 Methodological considerations

- **Model scale matters.** Brody 2023 explicitly notes that LN's scaling effect is
  most evident at small d (their experiments use d=8). Gupta 2024 study models up to
  8B. We should run ablations at multiple scales (e.g., 10M, 100M, 1B) — small models
  show clear LN effects; larger models test whether they survive.
- **Decoding strategy is a huge confound.** Following Chen & Ding 2023, evaluate at
  least with greedy and top-p (p=0.9, T=0.7), and ideally min-p (Nguyen 2025).
- **Avoid surprisal confound.** When comparing creativity, control for surprisal of
  generated words — otherwise a model that simply emits rare words will appear more
  creative.
- **Layer-wise probing is essential.** He et al. 2025 and Singhal & Kim 2025 both find
  early-layer LN is most influential. Our ablations should include layer-specific LN
  removal, not just whole-model ablation.
- **Mechanistic story before causal story.** First show LN ablations *change concept-
  association geometry*; then show those changes *change DAT/algorithmic creativity*.
  Mediation analysis is the cleanest framing.

---

## 8. Open theoretical questions

- Brody 2023 ends with: "what would happen if we force keys orthogonal to multiple
  normal vectors?" This relates to Menary's orthogonal-spheres requirement. Our
  experiments could empirically answer this.
- Is the "irreversibility" of LN (Gupta 2024) actually *cumulative* across depth? At
  layer L, what fraction of the original semantic information is recoverable?
- The *chaotic-attractor* dynamics of repeated LN (Chu 2024) may explain observed
  representational collapse — does this collapse predict reduced DAT?
- Does seed-conditioning's advantage over temperature (Nagarajan 2025) localize to
  specific layers? Is it the *bypassed* LNs in early layers that allow the seed's
  variance to propagate?

---

## 9. Summary

The literature provides three converging streams of evidence supportive of the
hypothesis:

1. **LayerNorm geometrically restricts representations** (Brody, Gupta, Menary, Chu).
2. **Creativity and divergent semantic association can be measured cleanly via DAT**
   (Olson 2021, Chen & Ding 2023, Bellemare-Pepin 2024) and via algorithmic-creativity
   tasks (Nagarajan 2025).
3. **Internal randomness/looseness matters more than output sampling** for divergent
   generation (Nagarajan 2025, He et al. 2025).

What is missing is an experimental link from (1) to (2) — measuring creativity under
controlled LN interventions — and a mechanistic account of (3) localized to specific
layers and normalization placements. Our experiment is positioned to fill exactly this
gap.
