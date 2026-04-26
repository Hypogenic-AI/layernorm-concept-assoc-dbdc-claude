"""DAT scoring (Olson 2021): cosine-distance-based creativity metric.

Wraps the official scorer (`code/divergent_association_task/dat.py`) for
reuse, plus utilities for parsing 10-noun outputs from LLMs.
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from scipy.spatial.distance import cosine


REPO_ROOT = Path(__file__).resolve().parent.parent
GLOVE_PATH = REPO_ROOT / "datasets" / "glove_embeddings" / "glove.6B.300d.txt"
WORDS_PATH = REPO_ROOT / "code" / "divergent_association_task" / "words.txt"


class DATScorer:
    """Glove-300d-based DAT scorer."""

    def __init__(
        self,
        glove_path: Path = GLOVE_PATH,
        words_path: Path = WORDS_PATH,
        pattern: str = r"^[a-z][a-z-]*[a-z]$",
    ):
        # Load English noun-eligible vocabulary
        words: set[str] = set()
        with open(words_path, "r", encoding="utf8") as f:
            for line in f:
                line = line.rstrip("\n")
                if re.match(pattern, line):
                    words.add(line)
        # Load matching GloVe vectors only (saves ~5x memory)
        vectors: dict[str, np.ndarray] = {}
        with open(glove_path, "r", encoding="utf8") as f:
            for line in f:
                tokens = line.split(" ")
                w = tokens[0]
                if w in words:
                    vectors[w] = np.asarray(tokens[1:], dtype=np.float32)
        self.vectors = vectors

    # ------------- scoring -------------
    def validate(self, word: str) -> Optional[str]:
        clean = re.sub(r"[^a-zA-Z- ]+", "", word).strip().lower()
        if len(clean) <= 1:
            return None
        candidates = []
        if " " in clean:
            candidates.append(re.sub(r" +", "-", clean))
            candidates.append(re.sub(r" +", "", clean))
        else:
            candidates.append(clean)
            if "-" in clean:
                candidates.append(re.sub(r"-+", "", clean))
        for c in candidates:
            if c in self.vectors:
                return c
        return None

    def distance(self, w1: str, w2: str) -> float:
        return float(cosine(self.vectors[w1], self.vectors[w2]))

    def dat(self, words: Iterable[str], minimum: int = 7) -> Optional[float]:
        uniques: list[str] = []
        for w in words:
            v = self.validate(w)
            if v and v not in uniques:
                uniques.append(v)
        if len(uniques) < minimum:
            return None
        subset = uniques[:minimum]
        ds = [self.distance(a, b) for a, b in itertools.combinations(subset, 2)]
        return (sum(ds) / len(ds)) * 100.0

    def coverage(self, words: Iterable[str]) -> float:
        words = list(words)
        if not words:
            return 0.0
        valid = sum(1 for w in words if self.validate(w))
        return valid / len(words)


# --------------------------------------------------------------------------- #
# Parsing LLM outputs
# --------------------------------------------------------------------------- #

# Match items like "1. apple", "1) apple", "- apple", "* apple", "apple\n",
# bullet/numbered lists.
_NUMBER_PREFIX = re.compile(r"^\s*(?:\d+[\.\)]\s*|[-*•]\s*)?")
_TRAILING = re.compile(r"[\W_]+$")
_SPLIT = re.compile(r"[\n,;]+")


def parse_word_list(text: str) -> List[str]:
    """Extract candidate nouns from a generated text, preserving order."""
    items = _SPLIT.split(text)
    out: list[str] = []
    for item in items:
        s = _NUMBER_PREFIX.sub("", item).strip()
        s = _TRAILING.sub("", s)
        # take only the first token if multi-word, but keep hyphens
        s = s.split()[0] if s else ""
        if s and s.isascii() and re.match(r"^[A-Za-z][A-Za-z-]*[A-Za-z]$", s):
            out.append(s.lower())
    # de-duplicate but keep order
    seen: set[str] = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq
