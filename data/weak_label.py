"""Cheap machine labels for a pool of messages. Two uses, both narrow.

  1. STRATIFICATION. To top rare intents up in the golden set we need to know roughly
     which messages are rare before a human reads them. Weak labels decide *which rows
     a human looks at*. They never decide what the answer is.

  2. TRAINING baseline B2. The TF-IDF classifier needs more than 200 rows to train on,
     and it must not train on the golden set it is tested against.

The honest cost, which belongs in the report: if the weak labeller systematically
misses a kind of message, that kind never gets surfaced for hand-labelling, and the
golden set inherits the blind spot. Sampling from weak labels bounds what the golden
set can discover. This is why the ~15 hard cases in D6 are drawn by keyword search and
hand-picking instead, as a partial counterweight.

Usage:
    python data/weak_label.py                 # 2000 messages
    python data/weak_label.py --n 200         # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from agent.classify import classify
from llm import LLMError

OUT = Path(__file__).resolve().parent / "weak_labels.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not config.SAMPLE_CSV.exists():
        raise SystemExit(f"{config.SAMPLE_CSV.name} not found -- run data/make_sample.py")

    df = pd.read_csv(config.SAMPLE_CSV)
    df = df.sample(min(args.n, len(df)), random_state=config.SEED)

    # Resume: a rate limit at row 1800 should not throw away 1800 real API calls.
    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
        print(f"  resuming, {len(done)} already labelled")

    todo = [r for r in df.itertuples(index=False)
            if str(r.customer_tweet_id) not in done]
    print(f"  weak-labelling {len(todo)} messages with {config.GEN_MODEL}")

    written = errors = 0
    with args.out.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(todo, 1):
            try:
                res = classify(str(row.customer_text))
                rec = {
                    "id": str(row.customer_tweet_id),
                    "text": str(row.customer_text),
                    "weak_intent": res.parsed.intent.value,
                    "weak_confidence": res.parsed.confidence,
                }
                written += 1
            except LLMError as exc:
                rec = {"id": str(row.customer_tweet_id),
                       "text": str(row.customer_text),
                       "weak_intent": None, "error": str(exc)[:200]}
                errors += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"\r  {i}/{len(todo)}  errors={errors}", end="", flush=True)
            time.sleep(0.05)  # gentle on the free tier
    print(f"\r  done: {written} labelled, {errors} errors -> {args.out.name}")

    rows = [json.loads(l) for l in args.out.read_text(encoding="utf-8").splitlines() if l.strip()]
    counts = pd.Series([r.get("weak_intent") for r in rows]).value_counts(dropna=False)
    print("\n  weak label distribution (NOT ground truth, sampling aid only):")
    for name, n in counts.items():
        print(f"    {str(name):<24} {n:>5}  {n/len(rows):>5.1%}")


if __name__ == "__main__":
    main()
