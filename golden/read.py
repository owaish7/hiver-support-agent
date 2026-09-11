"""Read real messages before defining anything. No API, no labelling, just reading.

This is the step that cannot be automated and the one the whole taxonomy rests on. The
intent list in taxonomy/intents.py is currently a guess made from a skim; it is meant to
be replaced by whatever you actually see here.

What to do while reading, on paper or in a scratch file:

  * Note what the customer WANTS, in your own words. Not the topic they mention -- the
    thing they would be angry about if it went unanswered.
  * When two of your notes start looking like the same bucket, name the bucket.
  * When a message could go in two buckets, write down the tie-breaker you used. Those
    tie-breakers are the boundary rules, and they matter more than the bucket names.
    They are also what you will be asked about in an interview.
  * Note anything that fits nowhere. If you see the same misfit three times, that is a
    missing category, not an `other`.

Aim for 100. It takes about 40 minutes and everything downstream depends on it.

    python golden/read.py                 # 100 random first-messages
    python golden/read.py --n 30
    python golden/read.py --with-replies   # show what the brand actually said
    python golden/read.py --grep refund    # messages matching a word
"""

from __future__ import annotations

import argparse
import random
import sys
import textwrap
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

config.use_utf8_stdout()

WRAP = 78


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--with-replies", action="store_true",
                    help="also print the brand's real reply")
    ap.add_argument("--grep", type=str, help="only messages containing this text")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    if not config.SAMPLE_CSV.exists():
        raise SystemExit(f"{config.SAMPLE_CSV.name} not found -- run data/make_sample.py")

    df = pd.read_csv(config.SAMPLE_CSV)
    df = df[df["customer_text"].astype(str).str.strip().str.len() > 0]

    if args.grep:
        df = df[df["customer_text"].astype(str).str.contains(args.grep, case=False,
                                                             regex=False)]
        if df.empty:
            raise SystemExit(f"nothing matching {args.grep!r}")

    # A different seed each time would mean re-reading the same messages after a break,
    # or never being able to point at "number 47" again. Fixed and stated.
    rows = df.sample(min(args.n, len(df)), random_state=args.seed)

    print(f"\n{'=' * WRAP}")
    print(f"{len(rows)} messages to @{config.BRAND}  (seed {args.seed}, so this is the "
          f"same set every time)")
    print(f"{'=' * WRAP}")

    for i, row in enumerate(rows.itertuples(index=False), 1):
        print(f"\n{i:>3}. ", end="")
        print(textwrap.fill(str(row.customer_text), WRAP - 5,
                            subsequent_indent="     ").lstrip())
        if args.with_replies:
            tag = "" if row.is_substantive else "   [punt, not usable as evidence]"
            print(textwrap.fill(f"-> {row.brand_reply_text}{tag}", WRAP - 5,
                                initial_indent="     ", subsequent_indent="        "))

    sub = int(df["is_substantive"].sum())
    print(f"\n{'=' * WRAP}")
    print(f"  corpus: {len(df):,} messages, {sub:,} ({sub/len(df):.0%}) have a reply that")
    print("  actually answers rather than redirecting to DM.")
    print("\n  When you have your buckets and your tie-breakers, edit")
    print("  taxonomy/intents.py -- DEFINITIONS and BOUNDARY_RULES. Then:")
    print("    python golden/sample_golden.py")
    print("    python golden/label.py")


if __name__ == "__main__":
    main()
