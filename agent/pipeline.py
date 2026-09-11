"""The whole system, one message in, one PipelineResult out.

    classify -> retrieve -> draft -> escalate

Escalation runs LAST even though an escalated message is not sent to the customer. Two
reasons, and the second is the one that matters:

  1. R1 and R2 need the classifier's confidence and the retriever's best similarity, so
     they cannot be evaluated before those steps run.

  2. We want the draft even for escalated messages. A human picking up an escalated
     ticket is better off with a proposed reply they can edit than a blank box, and the
     draft is scored either way -- so reply quality is measured on the whole test set
     rather than only on the easy half. Escalating everything would otherwise be a way
     to make the reply-quality number disappear.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import escalate as escalate_mod
from agent.classify import classify
from agent.draft import draft
from agent.retrieve import Evidence, get_retriever
from agent.schemas import PipelineResult
from llm import LLMError


class SupportAgent:
    """config name: `system`. The thing the baselines are compared against."""

    name = "system"

    def __init__(self, *, ask_llm_escalation: bool = False):
        """ask_llm_escalation asks the model for its own reading of the escalation
        policy. It is OFF by default because it decides nothing -- the deterministic
        rules in taxonomy/escalation.py always win, and the model's opinion is only
        recorded as `llm_disagreed` for analysis.

        It costs about 930 tokens per item, roughly a third of the per-item budget, on a
        free tier capped at 200,000 tokens per day. Paying a third of the eval budget for
        a column that changes no outcome is the wrong trade. Turn it on deliberately when
        the question is "is this policy written clearly enough for a model to apply it",
        which is worth asking once rather than on every run.
        """
        self.ask_llm_escalation = ask_llm_escalation

    def handle(self, text: str, *, item_id: str = "", prior_messages: int = 0
               ) -> tuple[PipelineResult, list[Evidence]]:
        started = time.perf_counter()
        out = PipelineResult(id=item_id, text=text)
        evidence: list[Evidence] = []

        try:
            # 1. intent
            cls = classify(text)
            out.intent = cls.parsed.intent.value
            out.intent_confidence = cls.parsed.confidence
            out.runner_up = cls.parsed.runner_up.value if cls.parsed.runner_up else None
            out.input_tokens += cls.input_tokens
            out.output_tokens += cls.output_tokens

            # 2. evidence. exclude_id keeps the item's own row out of its own evidence.
            evidence = get_retriever().search(text, exclude_id=item_id or None)
            out.retrieved_ids = [e.tweet_id for e in evidence]
            out.top_similarity = evidence[0].similarity if evidence else 0.0

            # 3. reply
            dr = draft(text, evidence)
            out.reply = dr.parsed.reply
            out.evidence_ids = dr.parsed.evidence_ids
            out.input_tokens += dr.input_tokens
            out.output_tokens += dr.output_tokens

            # 4. human or not
            decision = escalate_mod.decide(
                text,
                confidence=out.intent_confidence,
                top_similarity=out.top_similarity,
                prior_messages=prior_messages,
                ask_llm=self.ask_llm_escalation,
            )
            out.should_escalate = decision.should_escalate
            out.escalation_reason = decision.reason
            out.escalation_source = decision.source

        except LLMError as exc:
            out.error = str(exc)[:300]
        except Exception as exc:  # noqa: BLE001 - one bad row must not kill a paid run
            out.error = f"{type(exc).__name__}: {exc}"[:300]

        out.latency_s = round(time.perf_counter() - started, 2)
        return out, evidence


if __name__ == "__main__":
    demo = sys.argv[1] if len(sys.argv) > 1 else \
        "you charged me twice for premium this month and it still says free"
    result, ev = SupportAgent().handle(demo)
    print(f"  message    {result.text}\n")
    print(f"  intent     {result.intent} (conf {result.intent_confidence})")
    print(f"  top sim    {result.top_similarity:.2f}" if result.top_similarity else "")
    print(f"  reply      {result.reply}")
    print(f"  evidence   {result.evidence_ids}")
    print(f"  escalate   {result.should_escalate} "
          f"({result.escalation_reason}, {result.escalation_source})")
    print(f"  tokens     {result.input_tokens} in / {result.output_tokens} out "
          f"in {result.latency_s}s")
    if result.error:
        print(f"  ERROR      {result.error}")
