"""Interactive hand-labelling. Target ~90 seconds per message.

Three deliberate choices about what this tool does NOT show you:

  1. The weak (machine) label is hidden. If you saw the model's guess you would agree
     with it more often than you should, and every later comparison between your labels
     and the model would be contaminated by that anchoring.

  2. The dev/test split is hidden, so you cannot label the test half differently -- even
     unconsciously, even by getting more tired.

  3. The brand's actual reply is hidden behind a keypress. It is useful context, but the
     classifier never sees it, and labelling with information the system cannot have
     builds a ceiling it can never reach. Revealing it auto-tags the row as hard with a
     note, so the report can say exactly how many labels needed it.

The escalation policy check runs AFTER you answer, never before. You commit, then see
what the written rules say, then decide whether to stand by it. Showing the rules first
would turn hand-labelling into rubber-stamping a regex.

Usage:
    python golden/label.py              # continue where you left off
    python golden/label.py --review     # re-open rows you marked for review
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

config.use_utf8_stdout()
from taxonomy.escalation import GROUND_TRUTH_REASONS, apply_policy, explain
from taxonomy.intents import DEFINITIONS, INTENTS

HERE = Path(__file__).resolve().parent
POOL = HERE / "pool.jsonl"
OUT = config.GOLDEN_JSONL

WRAP = 76


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def banner() -> None:
    print("\n" + "=" * WRAP)
    print("INTENTS                                    (number = pick, ? = definitions)")
    print("=" * WRAP)
    for n, name in enumerate(INTENTS, 1):
        print(f"  {n}  {name}")
    print("\n  r = reveal the brand's reply (tags this row hard)")
    print("  s = skip   q = save and quit")
    print("=" * WRAP)


def show_definitions() -> None:
    print()
    for n, name in enumerate(INTENTS, 1):
        print(f"  {n}  {name}")
        print(textwrap.fill(DEFINITIONS[name], WRAP - 6,
                            initial_indent="     ", subsequent_indent="     "))
    print()


def ask_intent(row: dict) -> tuple[str | None, bool, str]:
    """Returns (intent, revealed_reply, extra_note)."""
    revealed = False
    note = ""
    while True:
        choice = input("  intent> ").strip().lower()
        if choice == "?":
            show_definitions()
            continue
        if choice == "r":
            if not revealed:
                revealed = True
                note = "needed the brand's reply to decide"
                print(textwrap.fill(f"BRAND REPLIED: {row.get('brand_reply', '')}", WRAP,
                                    initial_indent="  > ", subsequent_indent="    "))
            continue
        if choice in ("s", "q"):
            return choice, revealed, note
        if choice.isdigit() and 1 <= int(choice) <= len(INTENTS):
            return INTENTS[int(choice) - 1], revealed, note
        print(f"  1-{len(INTENTS)}, or ? r s q")


def ask_escalation(text: str) -> tuple[bool, str | None]:
    while True:
        ans = input("  escalate to a human? [y/n]> ").strip().lower()
        if ans in ("y", "yes"):
            break
        if ans in ("n", "no"):
            return False, None
        print("  y or n")

    print("     1 security_risk        2 money_movement       3 legal_or_reputational")
    print("     4 safety               5 repeat_contact")
    while True:
        pick = input("  reason> ").strip()
        if pick.isdigit() and 1 <= int(pick) <= 5:
            return True, GROUND_TRUTH_REASONS[int(pick) - 1]
        print("  1-5")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", action="store_true",
                    help="re-open rows previously marked for review")
    args = ap.parse_args()

    pool = load_jsonl(POOL)
    if not pool:
        raise SystemExit("golden/pool.jsonl not found -- run golden/sample_golden.py")

    existing = {r["id"]: r for r in load_jsonl(OUT)}
    if args.review:
        todo = [r for r in pool if existing.get(r["id"], {}).get("needs_review")]
    else:
        todo = [r for r in pool if r["id"] not in existing]

    if not todo:
        print(f"  nothing to do -- {len(existing)}/{len(pool)} labelled")
        return

    banner()
    print(f"\n  {len(existing)}/{len(pool)} done, {len(todo)} to go\n")

    started = time.time()
    n_done = 0
    agreed = disagreed = 0

    for row in todo:
        print("\n" + "-" * WRAP)
        # `existing` already grew by n_done as rows were saved, so adding it again
        # made the counter skip numbers and overshoot the pool size.
        pos = len(existing) + 1
        hard_tag = f"   [flagged: {row['hard_reason']}]" if row.get("hard_reason") else ""
        print(f"  {pos}/{len(pool)}{hard_tag}")
        print("-" * WRAP)
        print(textwrap.fill(row["text"], WRAP, initial_indent="  ",
                            subsequent_indent="  "))
        print()

        intent, revealed, note = ask_intent(row)
        if intent == "q":
            break
        if intent == "s":
            continue

        should_esc, reason = ask_escalation(row["text"])

        # Only now show what the written policy says.
        verdict = apply_policy(row["text"])
        needs_review = False
        if verdict.should_escalate != should_esc or (
                should_esc and reason and verdict.reason != reason):
            disagreed += 1
            print("\n  ! your call differs from the written policy:")
            print(textwrap.indent(explain(row["text"]), "  "))
            note_in = input("  why does your call stand? (enter = mark for review)> ").strip()
            if note_in:
                note = f"{note}; {note_in}" if note else note_in
            else:
                needs_review = True
        else:
            agreed += 1

        extra = input("  note (optional, enter to skip)> ").strip()
        if extra:
            note = f"{note}; {extra}" if note else extra

        record = {
            "id": row["id"],
            "text": row["text"],
            "intent": intent,
            "should_escalate": should_esc,
            "escalation_reason": reason,
            "is_hard": bool(row.get("is_hard")) or revealed,
            "notes": note,
            "needs_review": needs_review,
            "split": row["split"],
            "stratum": row["stratum"],
            "prior_messages": 0,
            "labelled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        existing[row["id"]] = record

        # Rewrite whole file: --review edits rows in place, and appending would
        # leave two records with the same id and a silently ambiguous answer key.
        with OUT.open("w", encoding="utf-8") as fh:
            for r in existing.values():
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        n_done += 1
        rate = (time.time() - started) / n_done
        print(f"  saved. {rate:.0f}s/item, ~{rate * (len(todo) - n_done) / 60:.0f} min left")

    print(f"\n  {len(existing)}/{len(pool)} labelled this session: {n_done}")
    if agreed + disagreed:
        print(f"  agreed with the written policy on {agreed}/{agreed + disagreed}")
    flagged = sum(1 for r in existing.values() if r.get("needs_review"))
    if flagged:
        print(f"  {flagged} marked for review -- python golden/label.py --review")


if __name__ == "__main__":
    main()
