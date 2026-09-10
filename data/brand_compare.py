"""Decision #1: which brand do we build on? Answered with data, not a hunch.

The trap in this dataset is that many support accounts reply "please DM us" to
almost everything. Build on one of those and "draft a reply grounded in how this
brand historically resolved similar issues" quietly degrades into "learn to say DM
us" -- the retrieval works, the eval passes, and the system has learned nothing.

So before committing, measure it. The output table goes straight into the report.

Usage:
    python data/build_threads.py     # first, produces pairs_*.csv
    python data/brand_compare.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

HERE = Path(__file__).resolve().parent


def main() -> None:
    rows = []
    frames: dict[str, pd.DataFrame] = {}

    for brand in config.CANDIDATE_BRANDS:
        path = HERE / f"pairs_{brand}.csv"
        if not path.exists():
            print(f"  missing {path.name} -- run data/build_threads.py first")
            continue
        df = pd.read_csv(path)
        frames[brand] = df
        n = len(df)
        sub = int(df["is_substantive"].sum())
        med = int(df["brand_reply_text"].fillna("").str.split().str.len().median())
        rows.append({
            "brand": brand,
            "pairs": n,
            "substantive": sub,
            "substantive_rate": sub / n if n else 0.0,
            "median_reply_words": med,
        })

    if not rows:
        raise SystemExit("no pairs_*.csv found")

    table = pd.DataFrame(rows).sort_values("substantive_rate", ascending=False)

    print("\n" + "=" * 78)
    print("BRAND CHOICE -- substantive reply rate")
    print("=" * 78)
    print(f"{'brand':<16}{'pairs':>9}{'substantive':>13}{'rate':>8}{'med words':>11}")
    print("-" * 78)
    for r in table.itertuples(index=False):
        print(f"{r.brand:<16}{r.pairs:>9,}{r.substantive:>13,}"
              f"{r.substantive_rate:>7.0%}{r.median_reply_words:>11}")
    print("-" * 78)
    print(f"  substantive = reply is >= {config.SUBSTANTIVE_MIN_WORDS} words (excluding")
    print("  @mentions and links) AND does not redirect to DM. See is_substantive()")
    print("  in build_threads.py -- it is crude on purpose and visible on purpose.")

    eligible = table[table["pairs"] >= 15_000]
    if eligible.empty:
        print("\n  WARNING: no candidate has >=15k pairs. Widen CANDIDATE_BRANDS.")
        eligible = table
    winner = eligible.iloc[0]
    print(f"\n  -> pick: {winner['brand']}  "
          f"({winner['substantive_rate']:.0%} substantive, {winner['pairs']:,} pairs)")
    print(f"     set BRAND = \"{winner['brand']}\" in config.py, then:")
    print(f"     python data/make_sample.py\n")

    # The rate alone can be gamed by a threshold, so print real replies. The one
    # check no script can do is whether these are actually useful as evidence.
    print("=" * 78)
    print("EYEBALL CHECK -- 3 replies counted substantive, per brand")
    print("=" * 78)
    for brand, df in frames.items():
        print(f"\n[{brand}]")
        sub = df[df["is_substantive"]]
        if sub.empty:
            print("  (none)")
            continue
        for row in sub.sample(min(3, len(sub)), random_state=config.SEED).itertuples(index=False):
            print(textwrap.fill(str(row.brand_reply_text), 74,
                                initial_indent="  - ", subsequent_indent="    "))


if __name__ == "__main__":
    main()
