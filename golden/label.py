"""Interactive hand-labelling, built for speed. Target ~20-25 seconds per message.

Three things this tool deliberately does NOT show you:

  1. The machine's guess at the intent. If you saw it you would agree with it more often
     than you should, and every later comparison between your labels and the model would
     be contaminated by that anchoring. The intent is the judgement call, so it stays
     entirely yours.

  2. The dev/test split, so you cannot label the test half differently -- even
     unconsciously, even as you get tired.

  3. The brand's actual reply, behind one keypress. It is useful context, but the
     classifier never sees it, and labelling with information the system cannot have
     builds a ceiling it can never reach. Revealing it auto-tags the row.

WHERE THE ESCALATION QUESTION GOES
----------------------------------
You are only asked about escalation when it is genuinely in play: when a content rule
(E1-E5) fires, or when the intent you chose makes escalation plausible. Everywhere else
the row is recorded as auto-handle without asking.

That is applying the written policy rather than skipping work -- E1-E5 are stated rules,
and "no rule fired and the category makes escalation implausible" is what the policy
says about a how_to question. It is still a labelling shortcut, so golden/SAMPLING.md
states how many rows were hand-confirmed and how many were taken by default, and the
report treats escalation recall as measured against a partially rule-derived key.

Usage:
    python golden/label.py              # continue where you left off
    python golden/label.py --review     # re-open rows you marked for review
    python golden/label.py --target 150 # stop after 150 (the brief's minimum)
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

# Categories where escalation is genuinely possible. Everywhere else, a message that
# trips no content rule is auto-handle by definition of the policy -- a "how do I change
# the bitrate" question does not become a human's problem because nobody looked at it.
ESCALATABLE = {"account_security", "billing_charge", "plan_or_family", "account_access"}

# Single-key input where the platform allows it: pressing 3 instead of 3-then-Enter is
# a third of the keystrokes across 200 rows.
try:
    import msvcrt

    def getkey() -> str:
        ch = msvcrt.getch()
        if ch in (b"\x03", b"\x1b"):       # ctrl-c, esc
            raise KeyboardInterrupt
        try:
            return ch.decode("utf-8", "ignore").lower()
        except Exception:                   # noqa: BLE001
            return ""
    ONE_KEY = True
except ImportError:
    def getkey() -> str:
        return (input().strip().lower() or " ")[0]
    ONE_KEY = False


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def menu() -> str:
    """Always on screen. Eleven categories is too many to hold in your head, and
    recalling them is most of what makes labelling slow."""
    left = INTENTS[:6]
    right = INTENTS[6:]
    lines = []
    for i in range(max(len(left), len(right))):
        a = f"{i + 1:>2} {left[i]:<21}" if i < len(left) else " " * 24
        b = f"{i + 7:>2} {right[i]}" if i < len(right) else ""
        lines.append(f"  {a}{b}")
    return "\n".join(lines)


def show_definitions() -> None:
    print()
    for n, name in enumerate(INTENTS, 1):
        print(f"  {n:>2} {name}")
        print(textwrap.fill(DEFINITIONS[name], WRAP - 6,
                            initial_indent="     ", subsequent_indent="     "))
    print()


def ask_reason() -> str:
    print("   1 security_risk   2 money_movement   3 legal_or_reputational"
          "   4 safety   5 repeat_contact")
    while True:
        print("  reason> ", end="", flush=True)
        k = getkey()
        if ONE_KEY:
            print(k)
        if k.isdigit() and 1 <= int(k) <= 5:
            return GROUND_TRUTH_REASONS[int(k) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--target", type=int, default=config.N_GOLDEN,
                    help="stop once this many rows are labelled (brief minimum is 150)")
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

    print(f"\n{'=' * WRAP}")
    print(f"  {len(existing)}/{args.target} labelled."
          f"  {'Press a number, no Enter needed.' if ONE_KEY else 'Type a number + Enter.'}")
    print(f"  ? definitions    r reveal the brand's reply    s skip    q save and quit")
    print(f"{'=' * WRAP}")

    started = time.time()
    n_done = 0
    asked_escalation = 0

    for row in todo:
        if len(existing) >= args.target:
            break

        print(f"\n{'-' * WRAP}")
        tag = f"   [{row['hard_reason']}]" if row.get("hard_reason") else ""
        print(f"  {len(existing) + 1}/{args.target}{tag}")
        print(menu())
        print(f"{'-' * WRAP}")
        print(textwrap.fill(row["text"], WRAP, initial_indent="  ",
                            subsequent_indent="  "))
        print()

        # ---- intent -------------------------------------------------------
        intent = None
        revealed = False
        note = ""
        while intent is None:
            print("  intent> ", end="", flush=True)
            k = getkey()
            if ONE_KEY:
                print(k)
            if k == "?":
                show_definitions()
            elif k == "r" and not revealed:
                revealed = True
                note = "needed the brand's reply to decide"
                print(textwrap.fill(f"BRAND REPLIED: {row.get('brand_reply', '')}",
                                    WRAP, initial_indent="  > ",
                                    subsequent_indent="    "))
            elif k == "q":
                intent = "__quit__"
            elif k == "s":
                intent = "__skip__"
            elif k.isdigit():
                # Two-digit categories (10, 11): read a second key.
                n = int(k)
                if k == "1":
                    print("  (1, or press 0 for 10, 1 for 11)> ", end="", flush=True)
                    k2 = getkey()
                    if ONE_KEY:
                        print(k2)
                    if k2 == "0":
                        n = 10
                    elif k2 == "1":
                        n = 11
                if 1 <= n <= len(INTENTS):
                    intent = INTENTS[n - 1]

        if intent == "__quit__":
            break
        if intent == "__skip__":
            continue

        # ---- escalation, only where it is actually in play -----------------
        verdict = apply_policy(row["text"])
        needs_review = False
        if verdict.should_escalate or intent in ESCALATABLE:
            asked_escalation += 1
            print("  human needed? [y/n]> ", end="", flush=True)
            while True:
                k = getkey()
                if k in ("y", "n"):
                    if ONE_KEY:
                        print(k)
                    break
            should_esc = k == "y"
            reason = ask_reason() if should_esc else None

            if verdict.should_escalate != should_esc or (
                    should_esc and reason and verdict.reason != reason):
                print("\n  ! differs from the written policy:")
                print(textwrap.indent(explain(row["text"]), "  "))
                why = input("  why does your call stand? (Enter = flag it)> ").strip()
                if why:
                    note = f"{note}; {why}" if note else why
                else:
                    needs_review = True
        else:
            # No content rule fired and this category is not one where a human is
            # needed. That is the policy's answer, not an unanswered question.
            should_esc, reason = False, None

        existing[row["id"]] = {
            "id": row["id"], "text": row["text"], "intent": intent,
            "should_escalate": should_esc, "escalation_reason": reason,
            "is_hard": bool(row.get("is_hard")) or revealed,
            "notes": note, "needs_review": needs_review,
            "escalation_hand_confirmed": verdict.should_escalate or intent in ESCALATABLE,
            "split": row["split"], "stratum": row["stratum"],
            "prior_messages": 0,
            "labelled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # Rewrite whole file: --review edits rows in place, and appending would leave two
        # records with the same id and a silently ambiguous answer key.
        with OUT.open("w", encoding="utf-8") as fh:
            for r in existing.values():
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        n_done += 1
        rate = (time.time() - started) / n_done
        left = min(args.target, len(pool)) - len(existing)
        print(f"  ok  {rate:.0f}s/item, ~{max(0, rate * left / 60):.0f} min for the "
              f"remaining {max(0, left)}")

    print(f"\n{'=' * WRAP}")
    print(f"  {len(existing)}/{args.target} labelled  (+{n_done} this session)")
    print(f"  escalation hand-confirmed on {asked_escalation} of {n_done} this session")
    flagged = sum(1 for r in existing.values() if r.get("needs_review"))
    if flagged:
        print(f"  {flagged} flagged -- python golden/label.py --review")
    if len(existing) >= 150:
        print("\n  You are past the brief's minimum of 150. Next:")
        print("    python golden/relabel.py     (tomorrow, after 20h)")
    else:
        print(f"\n  {150 - len(existing)} more to reach the brief's minimum of 150.")


if __name__ == "__main__":
    main()
