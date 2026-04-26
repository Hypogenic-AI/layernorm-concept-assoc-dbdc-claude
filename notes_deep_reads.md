# Deep-Read Notes

## Paper 1: On the Expressivity Role of LayerNorm in Transformers' Attention
- **arXiv**: 2305.02582 (Brody, Alon, Yahav, ACL 2023)
- **Code**: https://github.com/tech-srl/layer_norm_expressivity_role

### Key contribution
LayerNorm decomposes geometrically into TWO independent operations:
1. **Projection** of inputs onto hyperplane H orthogonal to 1⃗ = [1,...,1]
2. **Scaling** projected vectors to norm exactly √d

### Why each component matters for attention
- **Projection** lets queries align with 1⃗, producing uniform attention over keys (because all keys are orthogonal to 1⃗ → q·k ≈ 0). Used to compute "majority" function. Without projection, training takes 3× more steps to learn this.
- **Scaling** prevents the "unselectable keys" problem (Demeter et al. 2020): if a key vector lies inside the convex hull of other keys, NO query can give it the highest attention score. With LayerNorm scaling all keys to a sphere of radius √d, all keys are on the convex hull and *every key is selectable*.

### Empirical results
- d=8 GPT-2 trained on Wikipedia, 4 layers:
  - Without LayerNorm scaling: 32-51% of keys are unselectable per layer.
  - With LayerNorm: 0% unselectable.
- Effect mainly visible in small models (d=8). Authors note: "with a large hidden dimension... less likely to encounter a set of vectors where some of them lie within the convex hull of the others."

### Direct relevance to our hypothesis
- This paper provides geometric machinery (projection + scaling) we need.
- KEY INSIGHT for our hypothesis: in big models the scaling effect may be subtle, but the *uniformity* induced by scaling onto a hypersphere likely **homogenizes the geometry of concept vectors**. By eliminating "outlier" or "off-distribution" representations, LayerNorm may bias attention toward the bulk of well-represented (=high-probability) concepts, dampening the long tail.
- Their "majority" task is a metaphor: LayerNorm makes uniform/average behavior easy. This is the OPPOSITE of creative divergence.
- Open question they raise: "what happens if we force keys orthogonal to multiple normal vectors?" — relates to semantic-subspace independence.


## Paper 2: Geometric Interpretation of Layer Normalization and Comparative Analysis with RMSNorm
- **arXiv**: 2409.12951 (Gupta, Ozdemir, Anumanchipalli, UC Berkeley, 2024)

### Key contribution
Refines the geometric interpretation: LayerNorm = (1) remove component along uniform vector 1⃗, (2) normalize, (3) scale by √d. Then shows EMPIRICALLY across 7 LLMs that:
- Hidden vectors are *naturally* orthogonal to the uniform vector — even **before** LayerNorm — in 4/5 LayerNorm models AND in RMSNorm models (Llama-2/3).
- So the mean-subtraction in LayerNorm is empirically redundant; RMSNorm suffices.
- LayerNorm is *irreversible*: information along uniform-vector direction cannot be recovered from the 2d learnable α, β params (unlike BatchNorm which can encode identity).
- LayerNorm regulates the residual-stream norm: norms grow disproportionately with depth (Pre-LN especially) and LayerNorm pulls them back to ~√d.
- LayerNorm rotates hidden vectors by ~10-60° per layer.

### Models studied
GPT-2 XL, GPT-Neo 1.3B, Pythia-1.4B, GPT-J 6B, Pythia-6.9B (LayerNorm); Llama-2-7B, Llama-3-8B (RMSNorm). Probed with 1M Wikipedia tokens.

### Direct relevance to our hypothesis
- Norms grow large in residual stream → LN compresses to a sphere of radius √d.
- This sphere → all concept vectors live on a *fixed-radius shell* — losing magnitude as a degree of freedom.
- For creative association (combining concepts that are normally far apart), magnitude could encode *salience* or *confidence*. By killing it, LN may make all concept directions "equally weighted", but in a way that preserves only direction, not strength of association.
- KEY: irreversibility. Information once collapsed is gone — every layer applies LN, repeatedly squeezing the representation.
- Suggests our experiment should compare: stock LN vs RMSNorm vs LN-removed (DyT or similar) on creative-divergence metrics.


## Paper 3: Probing the "Creativity" of Large Language Models — Divergent Semantic Association
- **arXiv**: 2310.11158 (Chen & Ding, EMNLP 2023)
- **Code/data**: https://github.com/DingNLab/probing_creativity ; human DAT data at https://osf.io/vjazn/

### The DAT methodology (CORE EVAL TOOL for our project)
- DAT (Divergent Association Task, Olson et al. 2021): ask the model/human to write 10 unrelated nouns. Take the first 7 valid (noun) words. Compute pairwise cosine distance via GLoVe word embeddings:
  DAT = (100 / (n(n-1))) * Σ_{i≠j} (1 − cos(v_i, v_j))
- Higher DAT = more divergent = more creative. Validated against other creativity metrics in psychology.
- Word2Vec (corr 0.82) and FastText (corr 0.91) work too.
- Models eval'd: GPT-4, GPT-3.5-Turbo, Oasst-Llama-30B, Vicuna-13B, ChatGLM-6B.
- Comparison set: 8572 humans (osf.io/vjazn).

### Key results
- GPT-4 (greedy) DAT = 89.1 — beats 96.1% of humans.
- GPT-3.5-Turbo DAT = 80.8 — above human average.
- Smaller models lower; roughly proportional to size.
- Top-p sampling (p=0.9, T=0.7) raises DAT for non-GPT-4 models, but with high variance and invalid outputs.
- Increasing temperature 0.1→1.0 increases DAT for all models EXCEPT GPT-4 (already saturated).
- Random sampling from WordNet beats most humans — i.e., bypassing language distribution gives high DAT but no fluency.
- BASE prompt ("write 10 nouns") gives lowest DAT (associations dominate). RANDOM and DAT prompts pull divergence up, showing models *can* modulate the distribution.
- Surprisal (negative log word freq) correlates with DAT — rarer words → higher distances → higher DAT. Controlling for surprisal: GPT-4 advantage *attenuated*. So part of DAT is novelty-via-rarity.

### Direct relevance to our hypothesis
- **DAT is the ideal eval metric for our project**: it directly measures "concept association" divergence, and is mechanically simple (cosine in GLoVe).
- The fact that decoding strategy strongly affects DAT shows the issue is not just the *learned representation* but how it's sampled. But our hypothesis is about the *internal geometry* — DAT can be probed across LayerNorm interventions.
- "Creative and nonsense are both infrequent, can't distinguish via sampling alone" — points to the role of internal geometry in maintaining divergence + meaningfulness.


## Paper 4: Geometry and Dynamics of LayerNorm
- **arXiv**: 2405.04134 (Chu et al., 2024) — read both chunks below

### Quick highlights (after chunked reads)
- Theoretical paper: characterizes LayerNorm as projection onto unit sphere intersected with hyperplane, then diffeomorphism analysis.
- Studies dynamics of repeated LN application: shows it's a contraction map for some metrics but expansive for inter-cluster distances.
- Connects LN to attention sharpening and discusses how repeated LN+attention can drive representations toward attractors (low-rank or low-dimensional structures).


## Paper 5: Transformer Normalisation Layers and the Independence of Semantic Subspaces — VERY DIRECT
- **arXiv**: 2406.17837 (Menary, Kaski, Freitas, 2024)

### Core argument (DIRECTLY supports our hypothesis)
- *Semantic subspace*: any independent subspace of latent representation that can fully determine an attention distribution.
- Decompose latent state x = Σ_α x_α where each x_α encodes a concept α.
- For a linear-attention layer to extract x_α independently, the {x_α} must be linearly independent.
- **Pre-Norm** (the placement used in modern LLMs) replaces P_α x with x_α / ||Σ_β x_β||. This **cross-couples subspaces** through a shared denominator → independent extraction is *impossible* unless ||Σ_β x_β|| is constant.
- That requires the strict structure: each subspace is a sphere (||x_α|| = const_α) AND all spheres mutually orthogonal. Restrictive and removes a degree of freedom per concept.

### Three theorems
- **Theorem 1 (No-Norm)**: subspaces must be linearly independent.
- **Theorem 2 (Pre-Norm)**: subspaces must be **orthogonal spheres** S^{N_α-1}; any deviation causes interference scaled by 1/||Σ x_β||.
- **Theorem 3 (QKV-Norm)**: alternative norm placement after WQ/WK/WV → only requires linear independence (same as No-Norm). Loses a continuous d.o.f. per subspace via the spherical projection, but no cross-coupling.

### Empirical confirmation
- On a numerical-addition task: Pre-Norm models induce a *narrower* distribution of embedding L2-norms than QKV-Norm.
- L2-norms vary within ±20% (90% coverage) → some real interference.
- Circuit-collapse rate ≈ 1% under O(10%) norm perturbation: a sparse-attention head can spontaneously switch which token it attends to.
- Pre-Norm and QKV-Norm have similar in-distribution performance, but QKV-Norm has worse out-of-distribution.

### Direct relevance to our hypothesis
- This paper *operationalizes* the geometric story of LayerNorm and creativity: LN forces concept subspaces onto orthogonal spheres. **Concept association** = combining x_α and x_β; the more LN constrains them to orthogonal/spherical structure, the harder it is to express creative cross-concept interference as a *graded* combination — concepts must compose via attention rather than additive blending in embedding space.
- Suggests intervention: replace Pre-Norm with QKV-Norm (or remove LN entirely) and see if creative-divergence (DAT) goes up.
- Theory predicts creative tasks (which generalize OOD by definition) should suffer MORE from circuit collapse than in-distribution tasks.


## Paper 6: Roll the Dice & Look Before You Leap — Going Beyond the Creative Limits of Next-Token Prediction
- **arXiv**: 2504.15266 (Nagarajan, Wu, Ding, Raghunathan, ICML 2025)
- **Code**: https://github.com/chenwu98/algorithmic-creativity

### Two creativity task families (USEFUL FOR OUR EVALS)
- **Combinational creativity** (analogies, wordplay, research): a knowledge graph stored in weights; model must generate node sequences that obey a multi-hop pattern (sibling, triangle).
- **Exploratory creativity** (problem design, plot construction): a higher-order structure (circle, line) implicit under permutations; model must generate adjacency lists realizing the structure under novel permutations.
- Metric: "algorithmic creativity" = fraction of generations that are *coherent ∧ unique ∧ original* (not in train set).

### Two key findings
1. **Multi-token training (teacherless or diffusion) > next-token learning** for algorithmic creativity. Next-token learning is "myopic": cannot learn the implicit higher-order pattern that requires future planning.
2. **Seed-conditioning (input randomness) ≈ or > temperature sampling (output randomness)** for diversity. Temperature requires "cognitive overload" — marginalizing over many leaps of thought; seed-conditioning lets each random prefix realize one leap cleanly.

### Tested on Gemma v1 2B; minimal Transformers.

### Direct relevance to our hypothesis
- These minimal tasks (sibling, triangle, circle, line) are perfect benchmarks for our LayerNorm interventions: small enough to train from scratch with different normalization configs.
- Their distinction "input-randomness vs output-randomness" parallels our distinction between *internal representation geometry* (LN) and *decoding* (temperature). Our hypothesis lives in the input/internal-geometry side.


## Paper 7: nGPT — Normalized Transformer with Representation Learning on the Hypersphere
- **arXiv**: 2410.01131 (Loshchilov, Hsieh, Sun, Ginsburg, ICLR 2025) — NVIDIA

### Key idea
- ALL vectors (embeddings, attention/MLP weights, hidden states) are unit-norm-normalized.
- Hidden state lives on a hypersphere; each layer is a *step* on that sphere via SLERP/LERP-style update: h ← Norm(h + α*(h_block − h)) with α a learned per-dimension "eigen learning rate".
- Renders weight decay unnecessary; logits become bounded cosine similarities (need a learned temperature scaler s_z).
- Empirically: 4–20× faster training to reach same accuracy.

### Direct relevance
- This is the *extreme* of LayerNorm: enforce sphere structure throughout. If our hypothesis is right, nGPT should *exacerbate* mode collapse for creative tasks (no magnitude → no salience). Open question: nGPT papers don't measure DAT or divergent generation. We can measure!


## Paper 8: Shakespearean Sparks — Hallucination ↔ Creativity in LLMs' Decoding Layers
- **arXiv**: 2503.02851 (He, Zhang, Cheng, 2025)
- **Code**: https://github.com/ZicongHe2002/HCL-Spark

### Key contribution: HCL framework
- Use Layer-Skip / early-exit at intermediate layers for inference, then prompt the model with the same Q multiple times.
- Classify responses into: correct & creative (distinct types of correct answers), correct-only, hallucinated.
- Define: Creativity = number of distinct correct types; Hallucination = error rate; HCB = w_c·S_c + w_h·(1 − S_h).

### Findings
- Trade-off creativity-hallucination consistent across layer depth, model type, and size.
- Optimal HCB layer is in *early layers* of larger models (not the final layer).
- Confidence is also higher at the optimal layer.

### Relevance
- Identifies that LLMs' creative variability is *layer-dependent*, supports our angle that internal geometry (per-layer) matters for creativity. We can target our LN interventions at *specific* layers (early layers per their findings).


## Paper 9: Impact of Layer Norm on Memorization and Generalization in Transformers
- **arXiv**: 2511.10566 (Singhal & Kim, NeurIPS 2025)
- **Code**: https://github.com/JEKimLab/NeurIPS2025_LayernormMemorization

### Findings
- Pre-LN models: removing LN learnable params destabilizes learning AND exacerbates memorization.
- Post-LN models: removing LN params SUPPRESSES memorization while preserving learning ability — recovers genuine labels.
- Early-layer LN parameters are most influential.
- Validated on 13 models (Vision + Language) including GPT-2, GPT-Neo, Qwen2, BERT, DeBERTa, ViT, DeiT.

### Relevance
- Memorization vs. generalization is conceptually related to creative-generation: memorized outputs are dull; generalization can permit novel combinations. Their finding (LN removal in Post-LN suppresses memorization) suggests an intervention to test for our hypothesis (more creative outputs?). Provides infrastructure for LN ablation experiments.


## Paper 10: Gated Removal of Normalization (TaperNorm)
- **arXiv**: 2602.10408 (Kanavalau, Amo Alonso, Lall — Stanford, 2026)

### Key idea
- TaperNorm: gated convex combination of standard RMSNorm/LN and a sample-independent linear/affine scaling. Train with g=1 (standard LN), then cosine-decay g→0 → at convergence the layer is a fixed linear map foldable into adjacent projections.
- Uses an EMA-derived per-layer scalar c, and lets per-feature scales γ̃ remain trainable.
- Identifies "scale anchoring" via final normalization as the critical role: prevents "logit chasing" (CE loss inflating ||h|| to drive logits up). Without an anchor, training is unstable.
- Provides an explicit fixed-target auxiliary loss as alternative anchor → can remove ALL normalization.

### Relevance
- This gives us a concrete, GPT-2-tested recipe to remove per-token normalization from a trained model and observe effects on creative outputs. Compare this with stock GPT-2 on DAT.
- Cites Baroni et al. 2025 — staged LN-removal in GPT-2 — that's another key reference (we should add).


# Synthesis: Implications for our hypothesis

The reviewed literature converges on a coherent mechanistic story:
1. **LayerNorm projects representations onto a hypersphere of fixed radius √d** (Brody et al.; Gupta et al.; Loshchilov et al.). This removes magnitude as a degree of freedom and pulls all hidden states to the same scale.
2. **It enforces a strict structural constraint** for separable concept extraction: subspaces must be orthogonal spheres (Menary et al.). Imperfect satisfaction → cross-subspace interference proportional to 1/||Σ x_β||.
3. **Repeated normalization is irreversible** (Gupta et al.) — every layer compresses representations onto a sphere; lost magnitude info cannot be recovered.
4. **Creative generation requires divergent semantic association** (Chen & Ding; Bellemare-Pepin et al.; Nagarajan et al.). High DAT = ability to combine remote concepts. Temperature only partially helps. Multi-token training & input randomness help more.
5. **LN affects memorization vs. learning** (Singhal & Kim) — and intermediate layers matter (He et al.) — so creative behavior may be best probed at *specific* layers/depths.

**Our experimental opportunities**:
- Train small Transformers with stock LN, RMSNorm, QKV-Norm, no-LN/TaperNorm, nGPT — evaluate DAT (on DAT prompt) and algorithmic-creativity (Roll the Dice tasks).
- Probe internal concept-association geometry: cosine distances among concept vectors before/after LN; norm-spread; subspace-separability per Menary et al.
- Layer-wise sensitivity: which LN layers most affect divergence?
- Counterfactual: with TaperNorm/Baroni-style LN removal of a pretrained GPT-2, does DAT increase?

