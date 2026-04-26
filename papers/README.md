# Downloaded Papers

35 PDFs covering LayerNorm geometry, transformer normalization variants, creativity in
LLMs, divergent semantic association, and concept geometry in latent space. Organized
by tier of relevance to the hypothesis.

---

## Tier 1 — LayerNorm geometry, role, and modifications (15 papers)

These directly examine LayerNorm's effect on representations.

| File | Authors / Year | Why relevant |
|---|---|---|
| `arxiv_2305.02582_on_the_expressivity_role_of_layernorm_in_transformers_attent.pdf` | Brody, Alon, Yahav (ACL 2023) | **DEEP-READ.** Decomposes LN into projection + scaling; scaling prevents "unselectable keys"; projection enables uniform-attention queries. |
| `arxiv_2409.12951_geometric_interpretation_layernorm_rmsnorm.pdf` | Gupta, Ozdemir, Anumanchipalli (2024) | **DEEP-READ.** 7-LLM empirical study: hidden vectors *naturally* orthogonal to 1⃗ even before LN → mean-subtraction redundant. LN regulates residual-stream norms (10–100× std reduction) and rotates by 10–60°/layer. Argues for RMSNorm. |
| `arxiv_2406.17837_transformer_normalisation_layers_and_the_independence_of_sem.pdf` | Menary, Kaski, Freitas (2024) | **DEEP-READ.** Pre-Norm forces semantic subspaces onto orthogonal spheres; QKV-Norm only requires linear independence. Predicts circuit-collapse phenomenon under L2-norm noise. |
| `arxiv_2405.04134_geometry_and_dynamics_of_layernorm.pdf` | Chu et al. (2024) | **DEEP-READ.** Geometric/dynamical-systems analysis of repeated LN application: contraction map driving representations toward attractors. |
| `arxiv_2410.01131_ngpt_normalized_transformer_with_representation_learning_on_.pdf` | Loshchilov, Hsieh, Sun, Ginsburg (NVIDIA, ICLR 2025) | **SKIMMED.** All vectors unit-norm; hidden state walks on a hypersphere; 4–20× faster training. Extreme of LN. |
| `arxiv_2511.10566_impact_of_layer_norm_on_memorization_and_generalization_in_t.pdf` | Singhal, Kim (NeurIPS 2025) | **SKIMMED.** Pre-LN: removing LN destabilizes learning; Post-LN: removing LN suppresses memorization. Early LN layers most influential. |
| `arxiv_2602.10408_gated_removal_of_normalization_in_transformers_enables_stabl.pdf` | Kanavalau, Amo Alonso, Lall (Stanford 2026) | **SKIMMED.** TaperNorm: gated transition from RMSNorm to fixed linear map. Identifies "scale anchoring" via final norm as the critical role. |
| `arxiv_2510.22026_normalization_in_attention_dynamics.pdf` | (2025) | Mean-field analysis of self-attention with LN. |
| `arxiv_2305.14858_pre_rmsnorm_and_pre_crmsnorm_transformers_equivalent_and_eff.pdf` | Jiang et al. (NeurIPS 2024) | Pre-RMSNorm equivalent to Pre-LN under specific conditions; CRMSNorm (compact). |
| `arxiv_2305.18399_on_the_impact_of_activation_and_normalization_in_obtaining_i.pdf` | (2023) | Activation × normalization for isometric embeddings at init. |
| `arxiv_2309.12931_on_separate_normalization_in_self_supervised_transformers.pdf` | (2023) | Separate normalization for [CLS] token vs others in SSL. |
| `arxiv_2501.03096_analysis_of_mean_field_models_arising_from_self_attention_dy.pdf` | (2025) | Mean-field models of self-attention dynamics with LN. |
| `arxiv_2505.22014_learning_in_compact_spaces_with_approximately_normalized_tra.pdf` | (2025) | Approximately-normalized transformers — middle ground between standard and nGPT. |
| `arxiv_2601.22095_geonorm_unify_pre_norm_and_post_norm_with_geodesic_optimizat.pdf` | (2026) | GeoNorm: unifies Pre-Norm and Post-Norm via geodesic optimization. |
| `arxiv_2409.11253_norm_of_mean_contextualized_embeddings_determines_their_vari.pdf` | (2024) | Connection between embedding norm and variance, pre-LN. |

## Tier 2 — Creativity in LLMs (13 papers)

| File | Authors / Year | Why relevant |
|---|---|---|
| `arxiv_2310.11158_probing_the_creativity_of_large_language_models_can_models_p.pdf` | Chen, Ding (EMNLP 2023) | **DEEP-READ.** Defines DAT methodology for LLMs. GPT-4 beats 96% of humans (greedy DAT 89.1). Temperature helps non-GPT-4 but trades off stability. |
| `arxiv_2405.13012_divergent_creativity_in_humans_and_large_language_models.pdf` | Bellemare-Pepin et al. (2024) | DAT and creative-writing benchmarking against 100,000 humans. Top LLMs > avg human but not > top humans. Shows prompt+temperature can boost semantic divergence. |
| `arxiv_2405.00492_is_temperature_the_creativity_parameter_of_large_language_mo.pdf` | (2024) | "Temperature ≠ creativity" — shows temperature minimally affects evaluative-creativity scoring. |
| `arxiv_2407.01082_turning_up_the_heat_min_p_sampling_for_creative_and_coherent.pdf` | Nguyen et al. (ICLR 2025) | Min-p sampling: dynamic threshold proportional to top-token prob. Improves diversity without losing coherence. |
| `arxiv_2504.15266_roll_the_dice_look_before_you_leap_going_beyond_the_creative.pdf` | Nagarajan, Wu, Ding, Raghunathan (ICML 2025) | **DEEP-READ.** Algorithmic-creativity benchmarks (Sibling, Triangle, Circle, Line). Multi-token training and seed-conditioning > NTP and temperature. |
| `arxiv_2509.02510_top_h_decoding_adapting_the_creativity_and_coherence_with_bo.pdf` | Baghaei et al. (2025) | Top-H entropy-aware decoding. |
| `arxiv_2503.02851_shakespearean_sparks_the_dance_of_hallucination_and_creativi.pdf` | He, Zhang, Cheng (2025) | **SKIMMED.** Layer-wise creativity vs hallucination via Layer-Skip; optimal HCB layer is in early layers of large models. |
| `arxiv_2503.17126_modifying_llm_post_training_diverse_creative_writing.pdf` | Chung et al. (2025) | DiversityTuning: deviation-aware DPO/ORPO. |
| `arxiv_2505.14442_creative_preference_optimization.pdf` | (2025) | Creative-preference optimization for LMs. |
| `arxiv_2510.10157_billy_steering_large_language_models_via_merging_persona_vec.pdf` | (2025) | BILLY: persona-vector steering for creative gen. |
| `arxiv_2502.08515_the_paradox_of_stochasticity_limited_creativity_and_computat.pdf` | (2025) | Limited creativity from temperature alone in structured fictional data. |
| `arxiv_2601.21339_within_model_vs_between_prompt_variability_in_large_language.pdf` | (2026) | Within-model vs between-prompt variability — separates sources of creative variance. |
| `iccc23_pushing_gpt_creativity_limits.pdf` | Goes et al. (ICCC 2023) | GPT-4 on Alternative Uses Test and Torrance test, with multi-step prompting. |

## Tier 3 — Concept geometry, linear concepts, and interpretability (6 papers)

| File | Authors / Year | Why relevant |
|---|---|---|
| `arxiv_2310.06824_the_geometry_of_truth_emergent_linear_structure_in_large_lan.pdf` | Marks, Tegmark (2023) | Linear structure of true/false concepts in LLM activations. |
| `arxiv_2604.07886_linear_representations_of_hierarchical_concepts_in_language_.pdf` | (2026) | Linear representations of hierarchical concepts. |
| `arxiv_2309.07315_traveling_words_a_geometric_interpretation_of_transformers.pdf` | (2023) | Geometric "traveling words" view of transformer layers. |
| `arxiv_2311.08968_identifying_linear_relational_concepts_in_large_language_mod.pdf` | (2023) | Linear relational concepts (LRCs). |
| `arxiv_2203.14680_transformer_feed_forward_layers_build_predictions_by_promoti.pdf` | Geva et al. (EMNLP 2022) | FF layers promote vocabulary-space concepts; key-value memory analogy. |
| `arxiv_2303.08112_eliciting_latent_predictions_from_transformers_with_the_tune.pdf` | Belrose et al. (2023) | Tuned-Lens: learn affine probes per layer to read predictions. Useful for layer-wise concept geometry analysis. |

## Tier 4 — Mechanistic interpretability surveys (1 paper)

| File | Authors / Year | Why relevant |
|---|---|---|
| `arxiv_2404.14082_mechanistic_interpretability_for_ai_safety_a_review.pdf` | Bereska, Gavves (2024) | 380-citation survey covering circuit discovery, sparse autoencoders, concept extraction. Background reading. |

---

## Reading depth summary

- **Deep-read (full paper, all chunks)**: 6 papers
  - Brody 2023, Gupta 2024, Menary 2024, Chen & Ding 2023, Nagarajan 2025, Chu 2024.
- **Skimmed (intro + key sections)**: 4 papers
  - Loshchilov 2024 (nGPT), Singhal & Kim 2025, He et al. 2025 (Shakespearean Sparks),
    Kanavalau 2026 (TaperNorm).
- **Abstract-level only**: remaining 25 papers (their abstracts are sufficient for
  contextual citation). All can be deep-read on demand by chunking via the
  `pdf_chunker.py` script:
  ```bash
  python .claude/skills/paper-finder/scripts/pdf_chunker.py papers/<file>.pdf --pages-per-chunk 3
  ```

Detailed reading notes are in `notes_deep_reads.md` at the workspace root.

## Coverage tracker

- 27 papers obtained via Semantic Scholar / arXiv (with arXiv IDs from S2 metadata).
- 7 papers had non-trivial download paths (no public PDF or future-numbered arXiv IDs);
  resolved via web search and fallback to arXiv abs HTML.
- 1 paper from ICCC 2023 (Goes et al.) downloaded via the conference site
  (computationalcreativity.net) since it is not on arXiv.
