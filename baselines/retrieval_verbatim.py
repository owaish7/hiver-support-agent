"""B3 -- find the most similar past complaint, send the brand's real reply unchanged.

No generation at all. The retrieval is identical to the full system's; the difference is
that nothing writes anything afterwards.

This is the sharpest baseline in the set, because it isolates exactly one thing: what
the generation step is worth. The system and this row see the same evidence. If the
system only beats copy-paste by a couple of points, then the LLM is an expensive
paraphraser and the retrieval was doing the work -- and the honest move is to report
that, not to bury the row.

It also cannot be beaten on groundedness by construction: a reply the brand actually
sent is, by definition, something the brand actually says. Which is a useful reminder
that "grounded" is a weaker property than "correct" -- the retrieved reply may be
perfectly grounded and completely wrong for this customer.

exclude_id is passed through for the same reason as in the main pipeline: without it the
nearest neighbour of a golden item is the item itself, and this baseline would score
100% by handing back the answer key.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.retrieve import get_retriever
from agent.schemas import PipelineResult
from baselines.trivial import majority_intent


class RetrievalVerbatimBaseline:
    name = "retrieval_verbatim"

    def __init__(self):
        self.intent = majority_intent()

    def handle(self, text: str, *, item_id: str = "", prior_messages: int = 0):
        hits = get_retriever().search(text, exclude_id=item_id or None)
        out = PipelineResult(
            id=item_id,
            text=text,
            # Intent is not this baseline's job; it is filled with the majority class so
            # the intent column stays comparable with B0 rather than looking empty.
            intent=self.intent,
            intent_confidence=None,
            retrieved_ids=[h.tweet_id for h in hits],
            top_similarity=hits[0].similarity if hits else 0.0,
            reply=hits[0].brand_reply if hits else None,
            evidence_ids=[0] if hits else [],
            should_escalate=None,
            escalation_source="B3",
        )
        return out, hits


if __name__ == "__main__":
    b = RetrievalVerbatimBaseline()
    for demo in ["my downloads keep disappearing", "you charged me twice"]:
        r, hits = b.handle(demo)
        print(f"\n  query: {demo}")
        print(f"  sim {r.top_similarity:.2f} -> {r.reply}")
