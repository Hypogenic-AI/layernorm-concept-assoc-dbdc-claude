# LayerNorm and Concept Association in Creative Tasks

We test whether LayerNorm contributes to the collapse of generative LMs onto
common, "human-intuitive" outputs by intervening on LN at inference time on
GPT-2, GPT-2 medium, and Pythia-410M, and measuring the Divergent Association
Task (DAT) score, latent-space concept geometry, and perplexity.

## Key findings

- **Partial, decoder-dependent support for the hypothesis.** Removing LN's
  mean-subtraction (`no_mean`, RMSNorm-equivalent) shifts DAT in the
  predicted positive direction *only at high decoding temperature*
  (T=1.2): all four no_mean targets give Δ DAT ≈ +1.7 to +5.1 on GPT-2.
  Under greedy or moderate-T sampling, no consistent direction.
- **Latent geometry shifts as predicted on Pythia-410M.** Removing LN's
  mean-subtraction increases final-layer concept-pair cosine distance by
  ~7% (paired Wilcoxon, Δ = +0.026) — but this geometric expansion only
  reaches DAT under high-T sampling.
- **Strong LN ablations break the model.** `no_affine`, `weakened_a`, and
  `identity` interventions catastrophically increase perplexity (50× to
  10⁵×) and lower DAT. Apparent "DAT inflation" under these recipes is
  artifactual — random gibberish tokens produce GloVe-orthogonal embeddings.
  We gate on *validity rate* to avoid this trap.
- **The dominant collapse mechanism is not LN, but LN matters at the
  margins.** The LM-head softmax re-collapses most LN-induced geometric
  expansion. LN-relaxation gives a small benefit only when paired with
  high-T sampling; it is one of several gates between latent variation and
  output diversity, not the dominant one.

See [`REPORT.md`](REPORT.md) for full methodology, results tables, and
discussion.

## Reproducing

```bash
# 1) install env
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt   # or: uv add (see pyproject.toml)
# torch needs CUDA 12.1 build:
uv pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 2) download GloVe (~2 GB)
mkdir -p datasets/glove_embeddings
curl -L -o datasets/glove_embeddings/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
python -c "import zipfile; zipfile.ZipFile('datasets/glove_embeddings/glove.6B.zip').extract('glove.6B.300d.txt','datasets/glove_embeddings/')"

# 3) run experiments
python src/run_geometry.py     --model gpt2
python src/run_geometry.py     --model EleutherAI/pythia-410m
python src/run_perplexity.py   --model gpt2
python src/dat_generate.py     --model gpt2 --decoding topp --n_runs 20
python src/dat_generate.py     --model EleutherAI/pythia-410m --decoding topp --n_runs 20
python src/run_concept_geometry_extra.py --model gpt2

# 4) analysis + figures
python src/analyze.py
python src/render_report.py
```

## File map

```
src/
  ln_interventions.py            -- LayerNorm hook framework (recipes & targets)
  geometry_probes.py             -- per-layer concept geometry helpers
  dat_scoring.py                 -- GloVe-based DAT scorer + parser
  dat_generate.py                -- iterative 10-noun DAT generation under LN intervention
  run_geometry.py                -- per-layer hidden-state geometry
  run_perplexity.py              -- perplexity sanity check
  run_concept_geometry_extra.py  -- paired (stock vs intervention) per-layer geometry + Wilcoxon
  analyze.py                     -- summary tables + figures
  render_report.py               -- pulls all results into a single summary table

results/                         -- raw JSON + summary CSVs
figures/                         -- plots
papers/, code/, datasets/        -- pre-gathered resources
literature_review.md             -- 35-paper synthesis
resources.md                     -- catalog of resources
planning.md                      -- pre-experiment plan & motivation
REPORT.md                        -- full report (PRIMARY DELIVERABLE)
```

## Hardware / environment

- Python 3.12, torch 2.5.1+cu121, transformers 5.6.2
- NVIDIA RTX A6000 (49 GB VRAM)
- Per-model wall time:
  - geometry: ~30 seconds
  - perplexity: ~10 seconds
  - DAT (n=20): ~7-12 minutes
