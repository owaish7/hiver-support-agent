"""B1 -- send the brand's single most common reply to every message.

This is the reply-quality floor, and on this dataset it is not a joke baseline. A large
share of real support replies are variations on "sorry about that, please DM us so we
can look into it", which is:

    grounded          yes, it is literally what the brand says
    addresses the ask arguably, for anything that needs account access
    no overpromise    yes, it promises nothing at all
    right tone        usually

So a canned punt can score well on the judge's rubric while resolving nothing. If it
comes close to the real system, the honest reading is not "the system is bad" but "this
rubric rewards safe non-answers", and the report has to say so rather than quietly
dropping the row.

That is the whole reason this baseline exists: it is the one that can invalidate the
headline reply-quality number, so it is measured rather than assumed away.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from agent.schemas import PipelineResult
from baselines.trivial import majority_intent


def normalise_reply(text: str) -> str:
    """Strip the parts that vary per customer, so near-identical templates collapse
    into one and the true most-common reply is visible rather than split across
    thousands of one-off variants."""
    t = re.sub(r"@\w+", "", str(text))
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[/^-]\s*[A-Z]{1,3}\s*$", "", t.strip())  # agent sign-offs like "/JB"
    t = re.sub(r"\d+", "", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def most_common_reply() -> str:
    import pandas as pd

    if not config.SAMPLE_CSV.exists():
        return "Hi! Sorry to hear that. Please DM us so we can look into this for you."
    df = pd.read_csv(config.SAMPLE_CSV)
    groups: dict[str, list[str]] = {}
    for raw in df["brand_reply_text"].dropna().astype(str):
        groups.setdefault(normalise_reply(raw), []).append(raw)
    if not groups:
        return "Hi! Sorry to hear that. Please DM us so we can look into this for you."
    key = Counter({k: len(v) for k, v in groups.items()}).most_common(1)[0][0]
    return groups[key][0]  # an actual reply the brand sent, not the normalised form


class CannedBaseline:
    name = "canned"

    def __init__(self):
        self.reply = most_common_reply()
        self.intent = majority_intent()

    def handle(self, text: str, *, item_id: str = "", prior_messages: int = 0):
        return PipelineResult(
            id=item_id,
            text=text,
            intent=self.intent,
            intent_confidence=1.0,
            reply=self.reply,
            should_escalate=False,
            escalation_source="B1",
        ), []


if __name__ == "__main__":
    b = CannedBaseline()
    print(f"  most common reply in the corpus:\n    {b.reply}")
