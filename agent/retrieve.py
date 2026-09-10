"""Step 2: find how this brand has actually handled messages like this one before.

The whole retrieval engine, because at 20k rows that is all it needs to be:

    scores = index_vectors @ query_vector      # both unit-normalised -> cosine
    top    = argpartition(scores)[-k:]

Vectors are unit-normalised at build time, so the dot product IS cosine similarity and
there is no division anywhere. Cosine measures the angle between two vectors, which is
what we want -- a two-word complaint and a rambling one about the same problem should
count as similar, and angle ignores length.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


@dataclass
class Evidence:
    """One historical case: what a customer said, and what the brand really replied."""
    idx: int
    tweet_id: str
    customer_text: str
    brand_reply: str
    similarity: float


class Retriever:
    def __init__(self, path: Path | None = None):
        path = path or config.INDEX_NPZ
        if not path.exists():
            raise SystemExit(f"{path.name} not found -- run data/build_index.py")
        blob = np.load(path, allow_pickle=False)
        self.vectors: np.ndarray = blob["vectors"]
        self.tweet_ids = blob["tweet_ids"]
        self.customer_text = blob["customer_text"]
        self.brand_reply = blob["brand_reply"]
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(config.EMBED_MODEL)
        return self._model

    def search(self, text: str, k: int = config.TOP_K,
               exclude_id: str | None = None) -> list[Evidence]:
        """Nearest k historical cases.

        exclude_id drops the query's own row. Every golden example was drawn from the
        same corpus the index was built from, so without this the retriever finds the
        message itself at similarity 1.0 and hands the drafter the exact reply it is
        about to be graded against. That is not retrieval, it is the answer key leaking
        through the front door, and it would make every number in the report meaningless.
        """
        q = self.model.encode([text], normalize_embeddings=True).astype(np.float32)[0]
        scores = self.vectors @ q

        if exclude_id is not None:
            scores = np.where(self.tweet_ids == str(exclude_id), -np.inf, scores)

        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return [Evidence(idx=int(i),
                         tweet_id=str(self.tweet_ids[i]),
                         customer_text=str(self.customer_text[i]),
                         brand_reply=str(self.brand_reply[i]),
                         similarity=float(scores[i]))
                for i in top if np.isfinite(scores[i])]


    def by_ids(self, ids: list[str]) -> list[Evidence]:
        """Rebuild the evidence a cached run saw, from the ids it recorded.

        The judge has to check groundedness against the same five examples the drafter
        was given, and the result cache stores only their ids. Looking them back up
        keeps the cache small and, more importantly, keeps a single source of truth: if
        the evidence text were copied into the cache it could drift from the index, and
        the judge would be grading against something the drafter never saw.
        """
        pos = {str(t): i for i, t in enumerate(self.tweet_ids)}
        out = []
        for tid in ids:
            i = pos.get(str(tid))
            if i is not None:
                out.append(Evidence(idx=i, tweet_id=str(self.tweet_ids[i]),
                                    customer_text=str(self.customer_text[i]),
                                    brand_reply=str(self.brand_reply[i]),
                                    similarity=float("nan")))
        return out


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """One index, loaded once. A 200-item eval must not reload 20k vectors 200 times."""
    return Retriever()


def format_evidence(items: list[Evidence]) -> str:
    """Rendered for the prompt. Numbered so the drafter can cite which ones it used,
    which is what makes the judge's groundedness check checkable rather than a vibe."""
    if not items:
        return "(no similar past cases found)"
    return "\n\n".join(
        f"[{n}] similarity {e.similarity:.2f}\n"
        f"    customer: {e.customer_text}\n"
        f"    brand replied: {e.brand_reply}"
        for n, e in enumerate(items))


if __name__ == "__main__":
    demo = sys.argv[1] if len(sys.argv) > 1 else "my downloads keep disappearing offline"
    r = get_retriever()
    hits = r.search(demo)
    print(f"  query: {demo}\n")
    print(format_evidence(hits))
