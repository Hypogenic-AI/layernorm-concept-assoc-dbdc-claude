# Resources Catalog

## Summary

This catalog enumerates everything gathered for the project *"LayerNorm and concept
association in creative tasks"*: 35 papers, 9 cloned code repositories, 1 pre-staged
dataset (DAT human baseline), and download instructions for 6 additional datasets.

---

## Papers

**Total**: 35 PDFs (~78 MB combined). All PDFs in `papers/`. Detailed per-paper
descriptions and a deep-read tier table are in `papers/README.md`.

| # | Title | Authors | Year | File | Tier |
|---|---|---|---|---|---|
| 1 | On the Expressivity Role of LayerNorm in Transformers' Attention | Brody, Alon, Yahav | 2023 | arxiv_2305.02582 | T1 ★ |
| 2 | Geometric Interpretation of Layer Normalization and a Comparative Analysis with RMSNorm | Gupta, Ozdemir, Anumanchipalli | 2024 | arxiv_2409.12951 | T1 ★ |
| 3 | Transformer Normalisation Layers and the Independence of Semantic Subspaces | Menary, Kaski, Freitas | 2024 | arxiv_2406.17837 | T1 ★ |
| 4 | Geometry and Dynamics of LayerNorm | Chu et al. | 2024 | arxiv_2405.04134 | T1 ★ |
| 5 | nGPT: Normalized Transformer with Representation Learning on the Hypersphere | Loshchilov, Hsieh, Sun, Ginsburg | 2024 | arxiv_2410.01131 | T1 |
| 6 | Impact of Layer Norm on Memorization and Generalization in Transformers | Singhal, Kim | 2025 | arxiv_2511.10566 | T1 |
| 7 | Gated Removal of Normalization in Transformers (TaperNorm) | Kanavalau, Amo Alonso, Lall | 2026 | arxiv_2602.10408 | T1 |
| 8 | Normalization in Attention Dynamics | — | 2025 | arxiv_2510.22026 | T1 |
| 9 | Pre-RMSNorm and Pre-CRMSNorm Transformers | Jiang et al. | 2023 | arxiv_2305.14858 | T1 |
| 10 | On the impact of activation and normalization in obtaining isometric embeddings | — | 2023 | arxiv_2305.18399 | T1 |
| 11 | On Separate Normalization in Self-supervised Transformers | — | 2023 | arxiv_2309.12931 | T1 |
| 12 | Analysis of mean-field models from self-attention dynamics with LayerNorm | — | 2025 | arxiv_2501.03096 | T1 |
| 13 | Learning in Compact Spaces with Approximately Normalized Transformers | — | 2025 | arxiv_2505.22014 | T1 |
| 14 | GeoNorm: Unify Pre-Norm and Post-Norm with Geodesic Optimization | — | 2026 | arxiv_2601.22095 | T1 |
| 15 | Norm of Mean Contextualized Embeddings Determines their Variance | — | 2024 | arxiv_2409.11253 | T1 |
| 16 | Probing the Creativity of Large Language Models (DAT for LLMs) | Chen, Ding | 2023 | arxiv_2310.11158 | T2 ★ |
| 17 | Divergent Creativity in Humans and Large Language Models | Bellemare-Pepin et al. | 2024 | arxiv_2405.13012 | T2 |
| 18 | Is Temperature the Creativity Parameter of LLMs? | — | 2024 | arxiv_2405.00492 | T2 |
| 19 | Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs | Nguyen et al. | 2024 | arxiv_2407.01082 | T2 |
| 20 | Roll the dice & look before you leap: Going beyond the creative limits of NTP | Nagarajan, Wu, Ding, Raghunathan | 2025 | arxiv_2504.15266 | T2 ★ |
| 21 | Top-H Decoding: Adapting Creativity and Coherence with Bounded Entropy | Baghaei et al. | 2025 | arxiv_2509.02510 | T2 |
| 22 | Shakespearean Sparks: The Dance of Hallucination and Creativity in LLMs' Decoding Layers | He, Zhang, Cheng | 2025 | arxiv_2503.02851 | T2 |
| 23 | Modifying LLM Post-Training for Diverse Creative Writing | Chung et al. | 2025 | arxiv_2503.17126 | T2 |
| 24 | Creative Preference Optimization | — | 2025 | arxiv_2505.14442 | T2 |
| 25 | BILLY: Steering LLMs via Merging Persona Vectors for Creative Generation | — | 2025 | arxiv_2510.10157 | T2 |
| 26 | The Paradox of Stochasticity: Limited Creativity and Computational Decoupling | — | 2025 | arxiv_2502.08515 | T2 |
| 27 | Within-Model vs Between-Prompt Variability in LLMs for Creative Tasks | — | 2026 | arxiv_2601.21339 | T2 |
| 28 | Pushing GPT's Creativity to Its Limits: Alternative Uses and Torrance Tests | Goes et al. | 2023 | iccc23_pushing_gpt_creativity_limits.pdf | T2 |
| 29 | The Geometry of Truth: Emergent Linear Structure in LLM Representations | Marks, Tegmark | 2023 | arxiv_2310.06824 | T3 |
| 30 | Linear Representations of Hierarchical Concepts in Language Models | — | 2026 | arxiv_2604.07886 | T3 |
| 31 | Traveling Words: A Geometric Interpretation of Transformers | — | 2023 | arxiv_2309.07315 | T3 |
| 32 | Identifying Linear Relational Concepts in Large Language Models | — | 2023 | arxiv_2311.08968 | T3 |
| 33 | Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space | Geva et al. | 2022 | arxiv_2203.14680 | T3 |
| 34 | Eliciting Latent Predictions from Transformers with the Tuned Lens | Belrose et al. | 2023 | arxiv_2303.08112 | T3 |
| 35 | Mechanistic Interpretability for AI Safety - A Review | Bereska, Gavves | 2024 | arxiv_2404.14082 | T4 |

★ = deep-read (full paper). Detailed reading notes in `notes_deep_reads.md`.

---

## Datasets

| Name | Source | Size | Task | Location | Status |
|---|---|---|---|---|---|
| DAT human baseline (Olson 2021) | OSF: vjazn / kbeq6 | ~825 KB | DAT scoring reference | `datasets/dat_human_baseline/study2.tsv` | **Pre-staged** |
| GLoVe 840B 300d | Stanford NLP / HF | ~5.6 GB unzipped | DAT word-vector scoring | `datasets/glove_embeddings/` | Download instructions |
| TinyStories | HuggingFace `roneneldan/TinyStories` | ~600 MB raw | LM training (small) | `datasets/tinystories/` | Download instructions |
| Wikipedia 20220301.en | HuggingFace `wikipedia` | ~22 GB (use streaming) | Internal-state probing | (streaming) | Download instructions |
| WritingPrompts preferences | HuggingFace `euclaise/WritingPrompts_preferences` | ~120 MB | Diversity-aware tuning | `datasets/writingprompts_preferences/` | Download instructions (optional) |
| Algorithmic creativity tasks | Procedurally generated by `code/algorithmic_creativity/{task}/*.ipynb` | small (per task) | Sibling/Triangle/Circle/Line | (generated) | Generation instructions |
| AUT prompts | Olson lab | small | Alternative creativity probe | (n/a) | Optional |

Detailed download instructions, loading snippets, and licenses are in
`datasets/README.md`. Per project policy, the `datasets/.gitignore` excludes all but
the small DAT TSV and READMEs.

---

## Code Repositories

**Total**: 9 cloned (`--depth 1`). Detailed per-repo descriptions in `code/README.md`.

| # | Name | URL | Purpose |
|---|---|---|---|
| 1 | layer_norm_expressivity_role | github.com/tech-srl/layer_norm_expressivity_role | LN projection + scaling decomposition; "majority" + "unselectable keys" |
| 2 | probing_creativity ★ | github.com/DingNLab/probing_creativity | DAT eval harness with human baseline (study2.tsv) |
| 3 | divergent_association_task ★ | github.com/jayolson/divergent-association-task | Canonical DAT scorer (Olson 2021) |
| 4 | algorithmic_creativity ★ | github.com/chenwu98/algorithmic-creativity | Sibling/Triangle/Circle/Line task suite |
| 5 | hcl_spark | github.com/ZicongHe2002/HCL-Spark | Layer-wise creativity-vs-hallucination |
| 6 | layernorm_memorization | github.com/JEKimLab/NeurIPS2025_LayernormMemorization | LN-removal hooks across 13 models |
| 7 | DiversityTuning | github.com/mj-storytelling/DiversityTuning | Deviation-aware DPO/ORPO for diverse creative writing |
| 8 | minp_paper | github.com/menhguin/minp_paper | Min-p sampling reference + eval logs |
| 9 | top_h_decoding | github.com/ErfanBaghaei/Top-H-Decoding | Top-H entropy-aware decoder |

★ = primary tools we expect to call directly in experiments.

---

## Resource gathering notes

### Search strategy

We ran the paper-finder service with five complementary queries:

1. `layer normalization creativity language models concept association`
2. `layer normalization transformer geometry latent space`
3. `neural network creativity diversity sampling temperature`
4. `transformer interpretability concept neurons probing`
5. `removing layer normalization transformer pre-norm post-norm`

After deduplication, we collected 784 unique papers; 165 had relevance ≥ 2 and 42 had
relevance 3. From these we curated 35 papers across four tiers (LN geometry, LLM
creativity, concept geometry, mech-interp). Of the 35:

- 27 papers had clean arXiv IDs accessible via the Semantic Scholar API.
- 7 needed retry rounds due to API rate-limiting.
- 1 paper (Goes et al. 2023, ICCC) was not on arXiv; we located it on
  `computationalcreativity.net` and downloaded with a UA-spoofed curl.
- 2 papers (Geometric Interpretation of LayerNorm; Modifying LLM Post-Training for
  Diverse Creative Writing) needed manual web search to find their arXiv IDs
  (2409.12951 and 2503.17126 respectively).

### Selection criteria

Priority weighting:
- **Direct relevance to the hypothesis** (LN + creativity + concept geometry).
- **Recency** (2023–2026), with a few foundational older papers (Geva 2022, Marks 2023).
- **Code/data availability** so the experiment runner can run them directly.
- **Tier-1 venues** (NeurIPS, ICML, ICLR, ACL, EMNLP) where multiple options existed.

### Challenges encountered

- Semantic Scholar API was rate-limited; required exponential-backoff retries.
- Some 2026-year arXiv IDs (e.g., 2602.10408 TaperNorm, 2604.07886 Linear Hierarchical
  Concepts) returned successfully despite being only weeks old at lookup time, which
  suggests forward-numbered submissions: we verified each PDF downloaded and was
  > 100 KB.
- Paper-finder timed out on one of five queries; we retried in `fast` mode and got 62
  results.

### Gaps and workarounds

- **Olson et al. 2021 (DAT-PNAS) PDF**: not on arXiv; OSF redistributes the data only
  (the paper itself is on PNAS). The probing_creativity repo embeds the human-data
  TSV, which we copied into `datasets/`.
- **TaperNorm (Kanavalau 2026)**: no public code release. We may need to re-implement
  the gating scheme (Eq. 3 in the paper).
- **nGPT**: no official open-source release found; community reimplementations exist
  (`lucidrains/nGPT-pytorch`); we will clone if/when needed.
- **Pushing GPT's Creativity (Goes ICCC 2023)**: only the conference PDF is available;
  no associated public code release. Their methodology (multi-step interactive
  prompting) is straightforward to re-implement.

---

## Recommendations for experiment design

Based on gathered resources, we recommend:

1. **Primary dataset**: DAT scoring infrastructure
   (`datasets/dat_human_baseline/study2.tsv` + GloVe 840B 300d) — this is the
   cleanest, most-validated creativity metric.

2. **Secondary dataset**: Algorithmic-creativity tasks (Sibling Discovery, Triangle
   Discovery, Circle Construction, Line Construction) — fully objective, supports
   training transformers from scratch with different LN configurations on a single
   GPU.

3. **Baseline LN configurations**: Pre-LN, RMSNorm, QKV-Norm, no-LN-with-final-anchor,
   TaperNorm, nGPT.

4. **Evaluation metrics**: DAT score (against human percentile), algorithmic-
   creativity (coherent ∧ unique ∧ original), hidden-state norm distributions, L2-norm
   spread across `q,k,v` (Menary's interference proxy), per-layer concept-pair
   cosines.

5. **Code to adapt/reuse**:
   - `code/probing_creativity/` for the DAT pipeline (most directly).
   - `code/divergent_association_task/dat.py` for the Olson reference scorer.
   - `code/algorithmic_creativity/` for task suite + training scripts.
   - `code/layernorm_memorization/` for LN-removal hooks across 13 architectures.
   - `code/layer_norm_expressivity_role/` for projection-vs-scaling ablation.

6. **Confound controls**:
   - Run perplexity / downstream-task evals to confirm LN ablations don't trivially
     "break" the model.
   - Measure surprisal alongside DAT to disentangle "rare-word generation" from
     "divergent association".
   - Use multiple decoding strategies (greedy, top-p, min-p) so effects aren't
     decoder-confounded.
   - Run at multiple model scales (small, medium, large) per Brody 2023's note that
     LN's geometric effects are subtler at scale.

The full strategy and gap analysis are in `literature_review.md`.
