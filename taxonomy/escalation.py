"""The escalation policy as code -- the same rules as ESCALATION.md, mechanically.

WHAT THIS IS NOT
----------------
This is not an oracle that generates the answer key. It cannot be: E1-E4 need someone
to read a sentence and decide whether "see you in court" is a real legal threat or a
figure of speech, and no keyword list settles that. The hand labels are the ground
truth, exactly as the brief requires.

WHAT THIS IS
------------
A consistency checker over those hand labels. It applies the policy mechanically and
reports every row where the label disagrees with it. Each disagreement is then one of:

  * a labelling slip     -> fix the label
  * a deliberate call    -> keep the label, record why in `notes`

Both outcomes are useful. The count of deliberate exceptions is a direct measure of how
much of this policy actually needs human judgement, and that number goes in the report.

Recall is favoured over precision on purpose: the patterns over-fire so that a human
looks at borderline rows, rather than under-firing and silently blessing a wrong label.

Usage:
    python taxonomy/escalation.py --validate golden/golden.jsonl
    python taxonomy/escalation.py --explain "you charged me twice, refund me now"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REASONS = ["security_risk", "money_movement", "legal_or_reputational",
           "safety", "repeat_contact", "low_confidence", "no_precedent"]

# Reasons an answer key is allowed to contain. low_confidence and no_precedent are
# facts about our program, so they can never be the correct answer -- see ESCALATION.md.
GROUND_TRUTH_REASONS = REASONS[:5]


def _rx(*patterns: str) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE)


# Rule order is fixed and first match wins, so that two people applying this policy to
# a message matching several rules still reach the same answer.
RULES: list[tuple[str, str, re.Pattern]] = [
    ("E1", "security_risk", _rx(
        r"\bhack(ed|ing)?\b", r"\bcompromis(ed|e)\b", r"\bstolen\b",
        r"someone (else )?(is |has )?(us(ing|ed)|logged|access)",
        r"\bunauthori[sz]ed\b", r"\bfraud(ulent)?\b", r"\bphish(ing)?\b",
        r"not me\b.{0,30}\b(login|log in|sign in|charge)",
    )),
    ("E2", "money_movement", _rx(
        r"charg(ed|ing|e) me (twice|again|\d+ times)", r"double[- ]?charg",
        r"\bcharged\b.{0,40}\b(after|despite|even though).{0,20}cancel",
        r"\brefund\b", r"\bmoney back\b", r"\breimburse", r"\bchargeback\b",
        r"\bdispute\b.{0,20}\bcharge\b", r"billed .{0,20}(twice|again|wrong)",
        r"took .{0,15}(money|payment) .{0,15}(twice|again)",
    )),
    ("E3", "legal_or_reputational", _rx(
        r"\blawyer\b", r"\bsolicitor\b", r"\battorney\b", r"\blegal action\b",
        r"\bsue\b", r"\bsuing\b", r"\bcourt\b", r"\bombudsman\b",
        r"\btrading standards\b", r"\bsmall claims\b", r"\bregulator\b",
        r"\b(bbc|watchdog|the press|the media|journalist)\b",
        r"\bgdpr\b", r"\bdata protection\b",
    )),
    ("E4", "safety", _rx(
        r"\bkill (my ?self|me)\b", r"\bsuicid", r"\bend (my|it) (life|all)\b",
        r"\bself[- ]harm\b", r"\bwant to die\b",
        r"\bi.?ll (kill|hurt|find) you\b", r"\bthreaten",
    )),
    # E5 is not textual -- it is a fact about the thread, supplied by the caller.
]


@dataclass
class Verdict:
    should_escalate: bool
    reason: str | None
    rule: str | None


def apply_policy(text: str, *, prior_messages: int = 0) -> Verdict:
    """Mechanical reading of ESCALATION.md. First matching rule wins.

    prior_messages: how many earlier messages this customer sent about this issue.
    2 or more means the current one is at least the third -> E5.
    """
    if not isinstance(text, str):
        text = ""

    for rule_id, reason, pattern in RULES:
        if pattern.search(text):
            return Verdict(True, reason, rule_id)

    if prior_messages >= 2:
        return Verdict(True, "repeat_contact", "E5")

    return Verdict(False, None, None)


def explain(text: str, *, prior_messages: int = 0) -> str:
    """Every rule that fires, not just the winner. Used when hand-labelling a
    borderline row, where knowing E2 *and* E3 matched is the whole question."""
    hits = []
    for rid, reason, pattern in RULES:
        m = pattern.search(text)
        if m:
            hits.append(f"  {rid} {reason:<22} matched {m.group(0)!r}")
    if prior_messages >= 2:
        hits.append(f"  E5 repeat_contact         {prior_messages} prior messages")

    v = apply_policy(text, prior_messages=prior_messages)
    head = (f"-> escalate ({v.reason}, {v.rule})" if v.should_escalate
            else "-> auto_handle (no rule fired)")
    return "\n".join(hits + [head]) if hits else head


# ------------------------------------------------------------------ validation

def validate(path: Path) -> int:
    if not path.exists():
        print(f"  {path} does not exist yet -- nothing to validate")
        return 0

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        print("  golden set is empty")
        return 0

    schema_errors, disagreements, deliberate = [], [], 0

    for r in rows:
        rid = r.get("id", "?")

        # Schema checks. These are real errors, not judgement calls.
        if r.get("should_escalate") and not r.get("escalation_reason"):
            schema_errors.append((rid, "escalate with no reason"))
        if not r.get("should_escalate") and r.get("escalation_reason"):
            schema_errors.append((rid, "reason given but not escalating"))
        reason = r.get("escalation_reason")
        if reason and reason not in GROUND_TRUTH_REASONS:
            schema_errors.append(
                (rid, f"reason {reason!r} is not allowed in an answer key "
                      f"(runtime-only or unknown)"))

        v = apply_policy(r.get("text", ""), prior_messages=r.get("prior_messages", 0))
        if v.should_escalate != bool(r.get("should_escalate")):
            if r.get("notes"):
                deliberate += 1
            else:
                disagreements.append(
                    (rid, f"policy says {'escalate' if v.should_escalate else 'auto'}"
                          f" ({v.rule or '-'}), label says "
                          f"{'escalate' if r.get('should_escalate') else 'auto'}",
                     r.get("text", "")[:90]))
        elif v.should_escalate and reason and v.reason != reason:
            if r.get("notes"):
                deliberate += 1
            else:
                disagreements.append(
                    (rid, f"policy reason {v.reason} ({v.rule}), label reason {reason}",
                     r.get("text", "")[:90]))

    print(f"\n  {len(rows)} labelled rows")
    print(f"  {deliberate} deliberate exceptions (disagree with policy, notes written)")

    if schema_errors:
        print(f"\n  {len(schema_errors)} SCHEMA ERRORS -- these are bugs, fix them:")
        for rid, why in schema_errors:
            print(f"    {rid}  {why}")

    if disagreements:
        print(f"\n  {len(disagreements)} unexplained disagreements -- review each and"
              f" either fix the label or write a note saying why it stands:")
        for rid, why, text in disagreements[:30]:
            print(f"    {rid}  {why}")
            print(f"         {text}")
        if len(disagreements) > 30:
            print(f"    ... and {len(disagreements) - 30} more")

    if not schema_errors and not disagreements:
        print("  no unexplained disagreements")

    return len(schema_errors) + len(disagreements)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=Path)
    ap.add_argument("--explain", type=str)
    ap.add_argument("--prior", type=int, default=0)
    args = ap.parse_args()

    if args.explain:
        print(explain(args.explain, prior_messages=args.prior))
    elif args.validate:
        sys.exit(1 if validate(args.validate) else 0)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
