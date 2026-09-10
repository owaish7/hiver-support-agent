"""Embed every historical customer message so we can find similar past cases.

An embedding turns a sentence into 384 numbers positioned so that sentences with
similar meaning land near each other. "Find the most similar past complaint" is then
one matrix multiply, not a keyword search.

Two choices to defend:

  * A LOCAL model (all-MiniLM-L6-v2, 22M params, CPU) rather than an embedding API.
    No rate limits during a 200-item eval, byte-identical results on every run, and the
    reviewer can rebuild this index with no API key -- which is what actually makes the
    "reproduce in under 15 minutes" promise hold.

  * A NumPy array rather than a vector database. At 20k rows the entire retrieval
    engine is one `@` and an argsort. FAISS or Chroma would add a dependency, a build
    step and a failure mode in exchange for nothing measurable at this size.

Only rows whose brand reply is SUBSTANTIVE go into the index. A "please DM us" reply is
not a resolution and cannot ground anything; indexing them would let the drafter
retrieve five punts and confidently produce a sixth.

Usage:
    python data/build_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def main() -> None:
    if not config.SAMPLE_CSV.exists():
        raise SystemExit(f"{config.SAMPLE_CSV.name} not found -- run data/make_sample.py")

    df = pd.read_csv(config.SAMPLE_CSV)
    total = len(df)

    usable = df[df["is_substantive"]].reset_index(drop=True)
    if usable.empty:
        raise SystemExit("no substantive replies -- check SUBSTANTIVE_MIN_WORDS / brand")

    print(f"  {total:,} rows, {len(usable):,} with a substantive reply ({len(usable)/total:.0%})")
    print(f"  the other {total - len(usable):,} are DM-punts and are NOT indexed:")
    print("  a punt is not a resolution and cannot ground a reply")

    from sentence_transformers import SentenceTransformer

    print(f"  loading {config.EMBED_MODEL} (first run downloads ~90MB)")
    model = SentenceTransformer(config.EMBED_MODEL)

    texts = usable["customer_text"].astype(str).tolist()
    print(f"  embedding {len(texts):,} customer messages")
    vecs = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        # Unit-normalised, so cosine similarity is a plain dot product later.
        normalize_embeddings=True,
    ).astype(np.float32)

    np.savez_compressed(
        config.INDEX_NPZ,
        vectors=vecs,
        tweet_ids=usable["customer_tweet_id"].astype(str).to_numpy(),
        customer_text=usable["customer_text"].astype(str).to_numpy(),
        brand_reply=usable["brand_reply_text"].astype(str).to_numpy(),
    )
    size_mb = config.INDEX_NPZ.stat().st_size / 1e6
    print(f"  wrote {config.INDEX_NPZ.name}  {vecs.shape}  {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
