# Datasets

This directory contains datasets used to evaluate **LayerNorm and concept association in
creative tasks**. Per project policy, large data files are NOT committed to git; only the
small DAT human-baseline TSV is checked in. Follow the download instructions below to
reconstruct any dataset.

The experiment runner should reproduce all downloads by following this README.

---

## Dataset 1 — DAT human baseline (Olson et al. 2021)

- **Purpose**: human reference distribution for the Divergent Association Task (DAT).
  Used to compare LLM creativity scores against ~8,500 humans across 98 countries.
- **Source**: [OSF: vjazn](https://osf.io/vjazn/) (also
  [OSF: kbeq6](https://osf.io/kbeq6/) for the open release of 8,900 participants).
- **Size**: ~825 KB.
- **Format**: TSV, one row per participant, columns include `id, age, gender, country,
  multilingual, dat, word.1...word.10`.
- **License**: Public (CC0 / open data per OSF).
- **Pre-staged**: `datasets/dat_human_baseline/study2.tsv` is included in this repository
  for convenience (it is small and was redistributed by Chen & Ding 2023 from OSF).

### Download instructions (alternative)
```bash
# Already redistributed via the probing_creativity repo:
cp code/probing_creativity/human-level/study2.tsv datasets/dat_human_baseline/study2.tsv
```

### Loading
```python
import pandas as pd
df = pd.read_csv("datasets/dat_human_baseline/study2.tsv", sep="\t")
# 8572 rows. df['dat'] is the DAT score; word.1..word.10 are the unrelated nouns.
```

---

## Dataset 2 — GloVe word embeddings (for DAT scoring)

- **Purpose**: cosine-distance scoring of generated unrelated nouns (the canonical DAT
  scoring metric uses `glove.840B.300d`).
- **Source**: [Stanford NLP GloVe](https://nlp.stanford.edu/projects/glove/),
  also mirrored on [HuggingFace](https://huggingface.co/stanfordnlp/glove).
- **Size**:
  - `glove.6B.zip` ≈ 822 MB (50d/100d/200d/300d combined). Download if memory-constrained.
  - `glove.840B.300d.zip` ≈ 2.03 GB (matches Olson et al.'s scoring; preferred).
- **License**: PDDL / public domain.

### Download instructions
```bash
mkdir -p datasets/glove_embeddings && cd datasets/glove_embeddings
# Recommended (for canonical DAT scoring):
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip   # produces glove.840B.300d.txt (~5.6 GB unzipped)
# Lightweight alternative:
# wget https://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip
cd -
```

### Loading via gensim (recommended for fast IO)
```python
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec("datasets/glove_embeddings/glove.840B.300d.txt",
               "datasets/glove_embeddings/glove.840B.300d.w2v.txt")
from gensim.models import KeyedVectors
kv = KeyedVectors.load_word2vec_format(
    "datasets/glove_embeddings/glove.840B.300d.w2v.txt", binary=False)
```

### Loading via the official DAT scoring script
```python
# code/divergent_association_task/dat.py provides Model("glove.840B.300d.txt", "words.txt")
from code.divergent_association_task import dat
m = dat.Model("datasets/glove_embeddings/glove.840B.300d.txt",
              "code/divergent_association_task/words.txt")
m.dat(["arm","eyes","feet","hand","head","leg","body"])  # 50 (low)
m.dat(["hippo","jumper","machinery","prickle","tickets","tomato","violin"])  # 95 (high)
```

---

## Dataset 3 — TinyStories (for from-scratch transformer training)

- **Purpose**: small, high-quality corpus to train sub-100M-parameter transformers from
  scratch under different LayerNorm configurations (Pre-LN, Post-LN, RMSNorm, QKV-Norm,
  no-Norm/TaperNorm, nGPT). Used by the TaperNorm paper (Kanavalau et al. 2026) as a
  pretraining benchmark for normalization ablations.
- **Source**: [HuggingFace `roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories).
- **Size**: ~600 MB raw text (~2 GB tokenized).
- **Format**: HuggingFace Datasets; columns `text` (story).
- **License**: CDLA-Sharing-1.0.

### Download instructions
```python
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories")
ds.save_to_disk("datasets/tinystories")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/tinystories")
print(ds["train"][0])
```

---

## Dataset 4 — Wikipedia subset (for representation probing)

- **Purpose**: corpus to probe internal representations (norm distributions, subspace
  geometry, concept-association cosines) of pretrained LLMs. Used by Brody et al. 2023,
  Gupta et al. 2024, and Menary et al. 2024.
- **Source**: [HuggingFace `wikipedia` (20220301.en)](https://huggingface.co/datasets/wikipedia).
- **Size**: ~22 GB (English, full); we recommend `streaming=True` to load 1M tokens.
- **License**: CC-BY-SA / GFDL.

### Download instructions (streaming, ~1M tokens)
```python
from datasets import load_dataset
ds = load_dataset("wikipedia", "20220301.en", streaming=True, split="train")
# Take ~1M tokens for representation analysis.
```

---

## Dataset 5 — WritingPrompts preferences (for creative-writing eval)

- **Purpose**: preference pairs over writing-prompt completions. Used by Chung et al.
  2025 (DiversityTuning) for diversity-aware DPO/ORPO.
- **Source**: [HuggingFace `euclaise/WritingPrompts_preferences`](https://huggingface.co/datasets/euclaise/WritingPrompts_preferences).
- **Size**: ~120 MB.
- **License**: see HuggingFace dataset card.

### Download instructions
```python
from datasets import load_dataset
ds = load_dataset("euclaise/WritingPrompts_preferences")
ds.save_to_disk("datasets/writingprompts_preferences")
```

---

## Dataset 6 — Algorithmic creativity tasks (procedurally generated)

- **Purpose**: minimal, controllable creativity benchmarks: Sibling Discovery, Triangle
  Discovery, Circle Construction, Line Construction (Nagarajan et al. 2025).
  Quantifies "algorithmic creativity" = fraction of generations that are coherent,
  unique, and original (not in train).
- **Source**: generated by Jupyter notebooks in
  `code/algorithmic_creativity/{sibling,triangle,circle,line}-{discovery,construction}/{ntp,teacherless,diffusion}/{*.ipynb}`.
- **HuggingFace mirror of examples**:
  [ChenWu98/algorithmic-creativity](https://huggingface.co/collections/ChenWu98/algorithmic-creativity-6834c1b60ee58242df9da20c).

### Generation instructions
```bash
cd code/algorithmic_creativity/sibling-discovery
jupyter nbconvert --execute --to notebook sibling.ipynb
# Then run train.sh and eval.sh; data is staged under data/ inside the task folder.
```

---

## Dataset 7 — AUT (Alternative Uses Test) prompts (optional)

- **Purpose**: alternative creativity benchmark (e.g., "Name unusual uses for a brick").
  Used by Pushing GPT's Creativity (Goes et al. 2023) and Bellemare-Pepin et al. 2024.
- **Source**: prompts and human responses available via Olson lab and the Beaty lab
  (e.g., `osf.io/ath2s` for the DSI tutorial code).
- **Note**: less ideal than DAT for our purposes — AUT is sample-dependent and requires
  human ratings, whereas DAT is fully objective and instance-free.

---

## Pre-staged sample data

| Path                                    | Contents                              |
|-----------------------------------------|---------------------------------------|
| `dat_human_baseline/study2.tsv`         | 8,572 humans × 10 unrelated words     |

Everything else must be downloaded as documented above.
