"""Blind re-labelling of your own examples, to measure the ceiling on every other number.

The brief asks how well the LLM judge agrees with a human. There is one human here, so
the obvious follow-up is: how well does that human agree with *themselves*?

You relabel 50 of your own rows, at least a day later, in a different order, without
seeing your first answer. The agreement between the two passes is intra-annotator
reliability, and it bounds everything else in the report:

    if you disagree with yourself 12% of the time, the task is only ~88% well-defined,
    and a classifier scoring 88% is already at the noise floor of the answer key.

A model cannot be shown to beat a ceiling that is itself uncertain. This number is what
turns "my system got 87%" into a claim with a stated limit, and it costs 40 minutes.

Why the delay is enforced: relabelling an hour later measures short-term memory, not the
stability of your definitions. The point is to have forgotten the specific message and be
re-applying the written boundary rules cold.

Usage:
    python golden/relabel.py              # do the second pass
    python golden/relabel.py --report     # just the numbers
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

config.use_utf8_stdout()
from eval.stats import cohen_kappa, raw_agreement
from taxonomy.intents import DEFINITIONS, INTENTS

HERE = Path(__file__).resolve().parent
PASS2 = HERE / "relabel_pass2.jsonl"
N_RELABEL = 50
MIN_HOURS = 20
WRAP = 76


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def hours_since(stamp: str | None) -> float:
    if not stamp:
        return 1e9
    try:
        return (datetime.now() - datetime.fromisoformat(stamp)).total_seconds() / 3600
    except ValueError:
        return 1e9


def pick(rows: list[dict]) -> list[dict]:
    """Same 50 every time, so an interrupted session resumes on the same subset."""
    rng = random.Random(config.SEED + 1)
    chosen = rng.sample(rows, min(N_RELABEL, len(rows)))
    rng.shuffle(chosen)  # different order from pass 1: order itself is a memory cue
    return chosen


def report(pass1: dict[str, dict], pass2: list[dict]) -> None:
    paired = [(pass1[r["id"]], r) for r in pass2 if r["id"] in pass1]
    if not paired:
        print("  no overlapping rows yet")
        return

    a_int = [p1["intent"] for p1, _ in paired]
    b_int = [p2["intent"] for _, p2 in paired]
    a_esc = [bool(p1["should_escalate"]) for p1, _ in paired]
    b_esc = [bool(p2["should_escalate"]) for _, p2 in paired]

    print("\n" + "=" * WRAP)
    print(f"INTRA-ANNOTATOR AGREEMENT  n={len(paired)}")
    print("=" * WRAP)
    print(f"  intent    raw {raw_agreement(a_int, b_int):>6.1%}   "
          f"kappa {cohen_kappa(a_int, b_int):>+.3f}")
    print(f"  escalate  raw {raw_agreement(a_esc, b_esc):>6.1%}   "
          f"kappa {cohen_kappa(a_esc, b_esc):>+.3f}")
    print("\n  kappa: <0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate,")
    print("         0.61-0.80 substantial, 0.81+ almost perfect")
    print("  raw agreement alone would be misleading here -- with one class dominating,")
    print("  two people guessing the majority class agree most of the time. Kappa")
    print("  subtracts that. Report both, always.")

    flips = [(p1, p2) for p1, p2 in paired if p1["intent"] != p2["intent"]]
    if flips:
        print(f"\n  {len(flips)} intent disagreements -- these ARE the ambiguous boundary,")
        print("  and each one belongs in the failure analysis:")
        for p1, p2 in flips[:12]:
            print(f"\n    {p1['intent']}  ->  {p2['intent']}")
            print(textwrap.fill(p1["text"], WRAP - 6,
                                initial_indent="      ", subsequent_indent="      "))
        if len(flips) > 12:
            print(f"\n    ... and {len(flips) - 12} more")

    pairs = {}
    for p1, p2 in flips:
        key = tuple(sorted([p1["intent"], p2["intent"]]))
        pairs[key] = pairs.get(key, 0) + 1
    if pairs:
        print("\n  which boundaries are actually unstable:")
        for (x, y), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
            print(f"    {n:>2}x  {x} <-> {y}")
        print("\n  a pair appearing repeatedly means the boundary rule for it is not")
        print("  written clearly enough. That is a fixable defect, not bad luck.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="skip the 20h wait (invalidates the measurement, say so)")
    args = ap.parse_args()

    rows = load_jsonl(config.GOLDEN_JSONL)
    if not rows:
        raise SystemExit("golden/golden.jsonl is empty -- run golden/label.py first")
    pass1 = {r["id"]: r for r in rows}
    done = {r["id"]: r for r in load_jsonl(PASS2)}

    if args.report:
        report(pass1, list(done.values()))
        return

    subset = pick(rows)
    todo = [r for r in subset if r["id"] not in done]

    if not args.force and todo:
        youngest = min(hours_since(r.get("labelled_at")) for r in todo)
        if youngest < MIN_HOURS:
            raise SystemExit(
                f"\n  Only {youngest:.1f}h since some of these were first labelled.\n"
                f"  Wait until {MIN_HOURS}h have passed -- relabelling from memory\n"
                f"  measures recall, not whether your definitions are stable, and the\n"
                f"  resulting kappa would be an overestimate you would have to defend.\n"
                f"  --force overrides, but then say so in the report.\n")

    if not todo:
        report(pass1, list(done.values()))
        return

    print("\n" + "=" * WRAP)
    print("BLIND SECOND PASS -- your first answers are hidden on purpose")
    print("=" * WRAP)
    for n, name in enumerate(INTENTS, 1):
        print(f"  {n}  {name}")
    print("  ? = definitions   s = skip   q = quit")

    started = time.time()
    n_done = 0
    for row in todo:
        print("\n" + "-" * WRAP)
        print(f"  {len(done) + 1}/{len(subset)}")
        print("-" * WRAP)
        print(textwrap.fill(row["text"], WRAP, initial_indent="  ",
                            subsequent_indent="  "))
        print()

        intent = None
        while intent is None:
            c = input("  intent> ").strip().lower()
            if c == "?":
                for n, name in enumerate(INTENTS, 1):
                    print(f"  {n}  {name}: {DEFINITIONS[name][:70]}")
                continue
            if c == "q":
                report(pass1, list(done.values()))
                return
            if c == "s":
                intent = "skip"
                break
            if c.isdigit() and 1 <= int(c) <= len(INTENTS):
                intent = INTENTS[int(c) - 1]
        if intent == "skip":
            continue

        esc = None
        while esc is None:
            a = input("  escalate? [y/n]> ").strip().lower()
            if a in ("y", "yes"):
                esc = True
            elif a in ("n", "no"):
                esc = False

        rec = {"id": row["id"], "text": row["text"], "intent": intent,
               "should_escalate": esc,
               "labelled_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        done[row["id"]] = rec
        with PASS2.open("w", encoding="utf-8") as fh:
            for r in done.values():
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        n_done += 1
        rate = (time.time() - started) / n_done
        print(f"  saved. ~{rate * (len(todo) - n_done) / 60:.0f} min left")

    report(pass1, list(done.values()))


if __name__ == "__main__":
    main()
