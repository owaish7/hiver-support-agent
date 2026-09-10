"""Cut the chosen brand down to a committed sample.

Two reasons this file is committed to the repo rather than downloaded by the reviewer:

  1. The brief says results must reproduce in under 15 minutes. A 500MB Kaggle
     download behind a login does not fit in 15 minutes.
  2. The brief says a subsample is expected and encouraged.

Every number in the report comes from this file, so it is the actual dataset of
record. ~20k rows lands around 4MB, which is fine for git.

Usage:
    python data/make_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

HERE = Path(__file__).resolve().parent
N_SAMPLE = 20_000


def main() -> None:
    src = HERE / f"pairs_{config.BRAND}.csv"
    if not src.exists():
        raise SystemExit(f"{src.name} not found -- run data/build_threads.py first")

    df = pd.read_csv(src)
    before = len(df)

    # Drop rows we could never use as evidence or as an item.
    df = df.dropna(subset=["customer_text", "brand_reply_text"])
    df = df[df["customer_text"].str.strip().str.len() > 0]

    if len(df) > N_SAMPLE:
        # Random, not "the first 20k". The file is roughly time-ordered, so head()
        # would sample one period of the brand's history -- one product version, one
        # set of outages, one support policy -- and every intent frequency in the
        # report would be an artefact of that window.
        df = df.sample(N_SAMPLE, random_state=config.SEED)

    df = df.reset_index(drop=True)
    df.to_csv(config.SAMPLE_CSV, index=False)

    sub = int(df["is_substantive"].sum())
    size_mb = config.SAMPLE_CSV.stat().st_size / 1e6
    print(f"  {config.BRAND}: {before:,} pairs -> {len(df):,} sampled")
    print(f"  substantive: {sub:,} ({sub/len(df):.0%}) "
          f"<- this is the ceiling on grounded replies")
    print(f"  wrote {config.SAMPLE_CSV.name} ({size_mb:.1f} MB) -- commit this")


if __name__ == "__main__":
    main()
