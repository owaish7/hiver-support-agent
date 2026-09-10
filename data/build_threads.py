"""Rebuild conversation threads from the flat Kaggle CSV.

The dataset is one row per tweet with two link columns, `in_response_to_tweet_id` and
`response_tweet_id`. What we actually want is pairs:

    (the first thing a customer said)  ->  (what the brand replied to it)

The left side is what we classify. The right side is the historical evidence the
drafter retrieves over -- "how has this brand actually resolved this before".

Why only the FIRST customer message of a thread: later turns ("ok thanks", "still
broken", "?") are not independent intents. They inherit the intent of the message
above them, and including them would pad the test set with easy near-duplicates and
make accuracy look better than it is.

Memory: twcs.csv is ~500MB / ~2.8M rows and we need to resolve parent ids that may
live anywhere in the file, so this is two streaming passes rather than one load.

    pass 1: collect brand replies, remember which parent ids they point at
    pass 2: collect those parents

Usage:
    python data/build_threads.py                 # all candidate brands
    python data/build_threads.py --brand SpotifyCares
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

CHUNK = 250_000
COLS = ["tweet_id", "author_id", "inbound", "created_at", "text",
        "in_response_to_tweet_id"]


def _norm_id(value) -> str | None:
    """Ids arrive as int, float ('1234.0') or str depending on the column and chunk.
    Everything becomes a string so the two passes can actually match."""
    if pd.isna(value):
        return None
    if isinstance(value, float):
        return str(int(value))
    return str(value).strip()


def is_substantive(text: str) -> bool:
    """Does this reply tell the customer something, or does it punt to a DM?

    This single predicate decides which brand we build on, so it is a decision and
    not a detail. It is deliberately crude and deliberately visible: a reply is
    substantive if it is long enough to contain an instruction and does not redirect
    to a private channel.

    It will be wrong on some rows -- a long reply can still be a polite punt. That
    imprecision is reported rather than hidden, because the DM-punt rate sets a hard
    ceiling on how good "grounded in past resolutions" can possibly be.
    """
    if not isinstance(text, str):
        return False
    low = text.lower()
    if any(marker in low for marker in config.PUNT_MARKERS):
        return False
    words = [w for w in low.split() if not w.startswith("@") and not w.startswith("http")]
    return len(words) >= config.SUBSTANTIVE_MIN_WORDS


def extract_pairs(csv_path: Path, brands: list[str]) -> dict[str, pd.DataFrame]:
    if not csv_path.exists():
        raise SystemExit(
            f"\n  {csv_path} not found.\n"
            f"  Download it from "
            f"https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter\n"
            f"  and unzip twcs.csv into data/\n")

    brandset = {b.lower() for b in brands}

    # ---- pass 1: every reply written by a candidate brand -------------------
    replies: dict[str, list[dict]] = {b: [] for b in brands}
    wanted_parents: set[str] = set()

    print(f"pass 1/2  scanning {csv_path.name} for replies by {', '.join(brands)}")
    rows_seen = 0
    for chunk in pd.read_csv(csv_path, usecols=COLS, chunksize=CHUNK,
                             dtype={"author_id": str, "text": str}):
        rows_seen += len(chunk)
        hit = chunk[chunk["author_id"].str.lower().isin(brandset)]
        for row in hit.itertuples(index=False):
            parent = _norm_id(row.in_response_to_tweet_id)
            if parent is None:
                continue  # a brand tweet that starts a thread is marketing, not support
            wanted_parents.add(parent)
            replies[row.author_id].append({
                "brand_reply_id": _norm_id(row.tweet_id),
                "brand_reply_text": row.text,
                "customer_tweet_id": parent,
            })
        print(f"\r  {rows_seen:,} rows", end="", flush=True)
    print(f"\r  {rows_seen:,} rows -> {len(wanted_parents):,} parent ids to resolve")

    # ---- pass 2: the customer messages those replies point at ---------------
    print("pass 2/2  resolving parent messages")
    parents: dict[str, dict] = {}
    for chunk in pd.read_csv(csv_path, usecols=COLS, chunksize=CHUNK,
                             dtype={"author_id": str, "text": str}):
        chunk["_tid"] = chunk["tweet_id"].map(_norm_id)
        hit = chunk[chunk["_tid"].isin(wanted_parents)]
        for row in hit.itertuples(index=False):
            parents[row._tid] = {
                "customer_author_id": row.author_id,
                "customer_text": row.text,
                "created_at": row.created_at,
                "inbound": bool(row.inbound),
                # NaN here means nothing came before it -> this is the thread root,
                # i.e. genuinely the first thing this customer said.
                "is_root": _norm_id(row.in_response_to_tweet_id) is None,
            }
        print(f"\r  {len(parents):,} resolved", end="", flush=True)
    print()

    out: dict[str, pd.DataFrame] = {}
    for brand, rows in replies.items():
        merged = []
        for r in rows:
            p = parents.get(r["customer_tweet_id"])
            if p is None or not p["inbound"] or not p["is_root"]:
                continue
            merged.append({**r, **{k: v for k, v in p.items()
                                   if k not in ("inbound", "is_root")}})
        df = pd.DataFrame(merged)
        if not df.empty:
            # One brand reply per customer message. Some threads have several brand
            # tweets against one parent; keep the first so evidence stays 1:1.
            df = df.drop_duplicates(subset="customer_tweet_id", keep="first")
            df["is_substantive"] = df["brand_reply_text"].map(is_substantive)
        out[brand] = df
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", action="append",
                    help="brand to extract (repeatable). default: all candidates")
    ap.add_argument("--csv", type=Path, default=config.RAW_CSV)
    args = ap.parse_args()

    brands = args.brand or config.CANDIDATE_BRANDS
    frames = extract_pairs(args.csv, brands)

    for brand, df in frames.items():
        path = Path(__file__).resolve().parent / f"pairs_{brand}.csv"
        df.to_csv(path, index=False)
        sub = int(df["is_substantive"].sum()) if not df.empty else 0
        print(f"  {brand:<16} {len(df):>7,} pairs  {sub:>7,} substantive  -> {path.name}")


if __name__ == "__main__":
    main()
