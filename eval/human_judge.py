"""You grade replies by hand, blind to the LLM judge. This is the calibration set.

The brief asks for evidence of how well the judge agrees with a human. That requires a
human's answers on the same items, produced without seeing the judge's.

So: same four dimensions, same definitions, same evidence, no access to the machine
verdict. Ordering is shuffled with a fixed seed so you do not grade the system's early
replies while fresh and its late ones while tired -- that would correlate your ratings
with position in the file, and the agreement number would partly measure your attention
span.

60 replies at roughly 45 seconds each is about 50 minutes. That is the price of being
able to quote a reply-quality number at all.

Usage:
    python eval/human_judge.py               # grade
    python eval/human_judge.py --config canned
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

config.use_utf8_stdout()

from agent.retrieve import get_retriever

RESULTS = Path(__file__).resolve().parent / "results"
N_JUDGE = 60
WRAP = 78

DIMS = [
    ("grounded",
     "Is every factual claim traceable to one of the past cases shown?",
     "A reply that makes no factual claims (pure acknowledgement) counts as yes."),
    ("addresses_ask",
     "Does it respond to what THIS customer actually asked?",
     "So generic it would fit any message = no."),
    ("no_overpromise",
     "Does it avoid promising refunds, timeframes or fixes the brand does not promise?",
     "Check against the past cases, not against what seems reasonable."),
    ("tone_ok",
     "Does the tone match the brand and suit how upset they are?",
     "The soft one. Reported separately, excluded from `acceptable`."),
]


def ask(name: str, question: str, hint: str) -> bool:
    print(f"\n  {name}")
    print(f"    {question}")
    print(f"    ({hint})")
    while True:
        a = input("    [y/n]> ").strip().lower()
        if a in ("y", "yes"):
            return True
        if a in ("n", "no"):
            return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="system")
    ap.add_argument("--n", type=int, default=N_JUDGE,
                    help="how many to rate. 60 is the target; 20 still yields a usable "
                         "kappa with a wider interval, and a wide real number beats a "
                         "narrow fabricated one.")
    args = ap.parse_args()

    src = RESULTS / f"{args.config}.json"
    if not src.exists():
        raise SystemExit(f"{src.name} not found -- run eval/run_eval.py first")
    runs = json.loads(src.read_text(encoding="utf-8"))

    items = [(k, v) for k, v in runs.items() if v.get("reply")]
    rng = random.Random(config.SEED + 2)
    rng.shuffle(items)
    items = items[:args.n]

    out_path = RESULTS / f"human_{args.config}.json"
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    todo = [(k, v) for k, v in items if k not in done]

    if not todo:
        print(f"  all {len(done)} graded -- python eval/agreement.py")
        return

    print("\n" + "=" * WRAP)
    print(f"HUMAN GRADING  {len(done)}/{len(items)} done")
    print("  You are grading blind. The LLM judge's verdicts are not shown and must")
    print("  not be looked at until eval/agreement.py compares them.")
    print("=" * WRAP)

    retriever = get_retriever()

    for iid, run in todo:
        print("\n" + "-" * WRAP)
        print(f"  {len(done) + 1}/{len(items)}")
        print("-" * WRAP)
        print("  CUSTOMER:")
        print(textwrap.fill(run["text"], WRAP - 4, initial_indent="    ",
                            subsequent_indent="    "))
        print("\n  DRAFT REPLY:")
        print(textwrap.fill(run["reply"], WRAP - 4, initial_indent="    ",
                            subsequent_indent="    "))
        print("\n  PAST CASES THE DRAFTER SAW:")
        for n, e in enumerate(retriever.by_ids(run.get("retrieved_ids", []))):
            print(f"    [{n}] customer: {e.customer_text[:100]}")
            print(f"        replied:  {e.brand_reply[:100]}")

        rec = {name: ask(name, q, h) for name, q, h in DIMS}
        rec["acceptable"] = (rec["grounded"] and rec["addresses_ask"]
                             and rec["no_overpromise"])
        note = input("\n  note (optional)> ").strip()
        if note:
            rec["note"] = note

        done[iid] = rec
        out_path.write_text(json.dumps(done, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"  saved -> acceptable={rec['acceptable']}")

    print(f"\n  {len(done)}/{len(items)} graded. Now: python eval/agreement.py")


if __name__ == "__main__":
    main()
