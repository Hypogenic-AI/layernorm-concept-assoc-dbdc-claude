# Cloned Code Repositories

Each repository below was cloned with `--depth 1`. Sub-folders contain detailed READMEs;
this top-level README summarizes purpose and how each maps to the research hypothesis
*"LayerNorm contributes to collapse onto high-probability outputs, reducing creative or
incongruent concept combinations."*

---

## 1. `layer_norm_expressivity_role/` — Brody, Alon, Yahav (ACL 2023)
- **URL**: https://github.com/tech-srl/layer_norm_expressivity_role
- **Purpose**: official code for *"On the Expressivity Role of LayerNorm in Transformers'
  Attention"* (arXiv 2305.02582). Decomposes LayerNorm into **projection** (orthogonal to
  1⃗) + **scaling** (to norm √d) and shows each component's role.
- **Key files**:
  - `majority/` — toy task showing projection helps compute "majority".
  - `unselectable/` — measures the fraction of "unselectable" key vectors when scaling is
    removed.
  - `requirements.txt` — torch, transformers, datasets.
- **Use for our research**: starting point for ablating projection vs scaling on small
  transformers.

## 2. `probing_creativity/` — Chen & Ding (EMNLP 2023) — **PRIMARY EVAL HARNESS**
- **URL**: https://github.com/DingNLab/probing_creativity
- **Purpose**: official code for *"Probing the 'Creativity' of LLMs: Can Models Produce
  Divergent Semantic Association?"* (arXiv 2310.11158). Implements DAT scoring against
  human baselines and across decoding strategies.
- **Key files**:
  - `dat_score.py` — DAT scoring class (GloVe / word2vec / fasttext).
  - `dataset.py` — DAT prompt-and-collect harness (uses FastChat for non-OpenAI models).
  - `human-level/study2.tsv` — 8,572-human DAT baseline (also copied to `datasets/`).
  - `greedy_search/`, `Top_p/`, `temperature/` — pre-collected model outputs across
    decoding strategies.
  - `DAT_analysis.ipynb` — full result reproduction.
- **Use for our research**: drop-in DAT eval pipeline for any HF model. We add LN-config
  ablations as new model-loading paths.

## 3. `divergent_association_task/` — Olson et al. (PNAS 2021) — **CANONICAL DAT SCORER**
- **URL**: https://github.com/jayolson/divergent-association-task
- **Purpose**: the original DAT scoring code (psychology paper). 90 lines of code,
  includes a curated `words.txt` dictionary that filters proper nouns and rare forms.
- **Key files**:
  - `dat.py` — `Model("glove.840B.300d.txt", "words.txt")` then `.dat(words)`.
  - `examples.py` — figure-1 reproductions (low/avg/high human scores).
- **Use for our research**: the *reference* DAT scorer. Use this so our scores are
  apples-to-apples comparable with the human baseline and Chen & Ding's LLM scores.

## 4. `algorithmic_creativity/` — Nagarajan et al. (ICML 2025 spotlight) — **TASK SUITE**
- **URL**: https://github.com/chenwu98/algorithmic-creativity
- **Purpose**: official code for *"Roll the dice & look before you leap: Going beyond the
  creative limits of next-token prediction"* (arXiv 2504.15266). Provides Sibling
  Discovery, Triangle Discovery, Circle Construction, Line Construction.
- **Key files**:
  - `simpletransformers/` — wrapper for HF transformers used in NTP / teacherless
    training.
  - `{sibling,triangle}-discovery/` and `{circle,line}-construction/` —
    per-task `ntp/`, `teacherless/`, `diffusion/` subdirs with `train.sh`, `eval.sh`,
    Jupyter notebooks for data generation.
  - Note: only GPT-2/SEDD pipeline is in the repo; the Gemma 2B experiments aren't.
- **Use for our research**: minimal controllable benchmarks where we can swap
  normalization and measure algorithmic-creativity directly. Train from scratch on a
  single A6000.

## 5. `hcl_spark/` — He, Zhang, Cheng (2025)
- **URL**: https://github.com/ZicongHe2002/HCL-Spark
- **Purpose**: official code for *"Shakespearean Sparks: The Dance of Hallucination and
  Creativity in LLMs' Decoding Layers"* (arXiv 2503.02851). Implements the HCL framework
  for **layer-wise** creativity vs hallucination tradeoff using Layer-Skip early-exit.
- **Use for our research**: gives us the layer-wise probe machinery — we can pinpoint
  *which* LN layer most affects creative divergence.

## 6. `layernorm_memorization/` — Singhal & Kim (NeurIPS 2025)
- **URL**: https://github.com/JEKimLab/NeurIPS2025_LayernormMemorization
- **Purpose**: official code for *"Impact of Layer Norm on Memorization and
  Generalization in Transformers"* (arXiv 2511.10566). Demonstrates Pre-LN vs Post-LN LN
  removal effects across 13 vision/language models.
- **Use for our research**: ready-made LN-removal hooks for BERT/DeBERTa/GPT-2/GPT-Neo/
  Qwen2/ViT — we can plug DAT/algorithmic-creativity evals into their LN-ablation
  scripts. Tells us early-layer LNs are most influential.

## 7. `DiversityTuning/` — Chung et al. (2025)
- **URL**: https://github.com/mj-storytelling/DiversityTuning
- **Purpose**: official code for *"Modifying LLM Post-Training for Diverse Creative
  Writing"* (arXiv 2503.17126). Diversified DPO/ORPO with deviation-score reward.
- **Use for our research**: alternative *training-time* mechanism for diversity, useful
  as a baseline against architectural (LN) interventions.

## 8. `minp_paper/` — Nguyen et al. (ICLR 2025 oral)
- **URL**: https://github.com/menhguin/minp_paper
- **Purpose**: official code/logs for *"Turning Up the Heat: Min-p Sampling for Creative
  and Coherent LLM Outputs"* (arXiv 2407.01082).
- **Note**: Min-p is already merged into HF Transformers (`min_p` arg in `.generate`)
  and into vLLM. Repo mostly contains evaluation logs and replication notebooks.
- **Use for our research**: Min-p is the de facto creative decoder; useful as a baseline
  decoding strategy when comparing different LN configurations.

## 9. `top_h_decoding/` — Baghaei et al. (2025)
- **URL**: https://github.com/ErfanBaghaei/Top-H-Decoding
- **Purpose**: official PyTorch implementation of *"Top-H Decoding: Adapting the
  Creativity and Coherence with Bounded Entropy in Text Generation"* (arXiv 2509.02510).
- **Use for our research**: an entropy-aware sampler that adapts to the input — another
  decoding-side baseline.

---

## Repositories we expect to install but did NOT clone

- **HuggingFace `transformers` and `datasets`** — already installed via uv.
- **vLLM** — for fast inference; install in experiment runner if needed.
- **Score-Entropy-Discrete-Diffusion (SEDD)** — required for algorithmic_creativity's
  diffusion baselines, but each task already vendors a self-contained copy under
  `{task}/diffusion/`.
- **TaperNorm** (Kanavalau et al. 2026) — no public release yet (preprint is Feb 2026).
  We may need to re-implement the gating scheme (Eq. 3 in the paper) ourselves.
- **nGPT** (Loshchilov et al. 2025) — no official open-source release found; community
  reimplementations exist (e.g.,
  https://github.com/lucidrains/nGPT-pytorch — to be cloned when needed).

---

## Dependencies summary

Cloning these repos brings in a Python ecosystem: torch, transformers, datasets, gensim,
numpy, scipy, scikit-learn, vllm (optional), wandb (optional), simpletransformers, and
matplotlib. Install incrementally per-experiment via `uv pip install`.
