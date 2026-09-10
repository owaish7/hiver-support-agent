"""B0 -- the floor. Reads nothing, decides nothing, answers the most common thing.

Every eval needs a number that a system with zero understanding achieves, otherwise
"87% accurate" is unreadable: 87% is excellent if guessing scores 20% and worthless if
guessing scores 84%. This row exists so the headline number can be interpreted at all.

Two escalation variants, because they fail in opposite directions and reporting only one
would be a choice about which failure to hide:

    always_auto      never escalates. Perfect precision on auto-handling, zero recall on
                     the messages that actually need a human -- including the hacked
                     accounts.
    always_escalate  escalates everything. Perfect escalation recall, and not a product.

Whenever a system claims high escalation recall, the question is what it cost in
auto-handle rate, and always_escalate is the row that makes that question concrete.

The majority intent is computed from the WEAK-LABELLED pool, never from the golden set.
Taking it from the answer key would make this baseline an oracle -- it would be reading
the test set's own class distribution, and a "trivial" baseline that peeks is not a
floor, it is a lie about how hard the task is.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.schemas import PipelineResult

WEAK = Path(__file__).resolve().parent.parent / "data" / "weak_labels.jsonl"


def majority_intent() -> str:
    if not WEAK.exists():
        return "playback_issue"  # documented guess; rerun after data/weak_label.py
    counts = Counter()
    for line in WEAK.read_text(encoding="utf-8").splitlines():
        if line.strip():
            lbl = json.loads(line).get("weak_intent")
            if lbl:
                counts[lbl] += 1
    return counts.most_common(1)[0][0] if counts else "playback_issue"


class TrivialBaseline:
    def __init__(self, escalation: str = "always_auto"):
        assert escalation in ("always_auto", "always_escalate")
        self.escalation = escalation
        self.name = f"trivial_{escalation}"
        self.intent = majority_intent()

    def handle(self, text: str, *, item_id: str = "", prior_messages: int = 0):
        esc = self.escalation == "always_escalate"
        return PipelineResult(
            id=item_id,
            text=text,
            intent=self.intent,
            intent_confidence=1.0,   # it is never in doubt, and always uninformative
            reply=None,              # makes no reply at all; scored as unacceptable
            should_escalate=esc,
            escalation_reason="repeat_contact" if esc else None,
            escalation_source="B0",
        ), []


if __name__ == "__main__":
    b = TrivialBaseline()
    print(f"  majority intent from weak labels: {b.intent}")
    r, _ = b.handle("anything at all")
    print(f"  -> {r.intent}, escalate={r.should_escalate}, reply={r.reply}")
