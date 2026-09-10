"""Every model output is one of these. Nothing in this project parses prose.

Two design points worth defending:

1. `reasoning` fields come BEFORE the verdict field in every schema. Structured output
   is generated left to right, so a verdict emitted first is a guess the reasoning is
   then written to justify. Putting the reasoning first makes it an input to the
   verdict rather than a decoration on it.

2. `confidence` is self-reported and is NOT trusted as a probability. It is used only
   as a threshold input (R1 in ESCALATION.md) and its calibration is an open question
   flagged in the report, not a claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxonomy.intents import Intent


class Classification(BaseModel):
    """Step 1: what does this customer want?"""
    reasoning: str = Field(
        description="One sentence: what the customer is asking for, and which boundary "
                    "rule decided it if two categories were close.")
    intent: Intent
    runner_up: Intent | None = Field(
        default=None,
        description="The second-most-likely category if it was genuinely close, else "
                    "null. Used for failure analysis, never scored.")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="0-1. Be honest: use below 0.5 when two categories are equally "
                    "defensible.")


class DraftedReply(BaseModel):
    """Step 3: the reply, plus which retrieved evidence it leaned on."""
    reasoning: str = Field(
        description="One sentence: which retrieved example you are following and why "
                    "it applies here.")
    evidence_ids: list[int] = Field(
        default_factory=list,
        description="Indices of the retrieved examples actually used. Empty means the "
                    "reply is not grounded in anything, which is itself a finding.")
    reply: str = Field(
        description="The reply to send, as the brand would write it. Under 280 "
                    "characters.")


class EscalationDecision(BaseModel):
    """Step 4: human or not, and why. The reason is not optional."""
    reasoning: str = Field(description="One sentence citing the rule that applies.")
    should_escalate: bool
    reason: str | None = Field(
        default=None,
        description="Required when escalating: one of security_risk, money_movement, "
                    "legal_or_reputational, safety, repeat_contact. Null otherwise.")


class JudgeVerdict(BaseModel):
    """The LLM judge's grade for one drafted reply.

    Four binary dimensions rather than a 1-5 quality score. A 1-5 scale is not
    calibrated -- nobody can say what separates a 3 from a 4 -- so averaging it produces
    a number that cannot be acted on and cannot be compared against a human rating.
    A yes/no per named failure mode can be both.

    Each verdict is preceded by its own rationale for the ordering reason above.
    """
    grounded_rationale: str
    grounded: bool = Field(
        description="Is every factual claim in the reply traceable to one of the "
                    "retrieved historical examples?")

    addresses_ask_rationale: str
    addresses_ask: bool = Field(
        description="Does the reply respond to what this customer actually asked?")

    no_overpromise_rationale: str
    no_overpromise: bool = Field(
        description="Does the reply avoid promising refunds, timelines, or fixes that "
                    "the historical examples do not show the brand promising?")

    tone_ok_rationale: str
    tone_ok: bool = Field(
        description="Does the tone match the brand's voice and suit how upset the "
                    "customer is?")

    @property
    def acceptable(self) -> bool:
        """Tone is deliberately excluded. It is the softest dimension and the one a
        judge is least likely to agree with a human on, so letting it gate the headline
        number would import that noise into every other result. It is still reported
        separately."""
        return self.grounded and self.addresses_ask and self.no_overpromise


class PipelineResult(BaseModel):
    """One message, all the way through. This is what gets cached and graded."""
    id: str
    text: str

    intent: str | None = None
    intent_confidence: float | None = None
    runner_up: str | None = None

    retrieved_ids: list[str] = Field(default_factory=list)
    top_similarity: float | None = None

    reply: str | None = None
    evidence_ids: list[int] = Field(default_factory=list)

    should_escalate: bool | None = None
    escalation_reason: str | None = None
    escalation_source: str | None = Field(
        default=None,
        description="Which rule caused it: E1-E5 from message content, or R1/R2 from "
                    "runtime signals. Kept separate so the report can price what "
                    "caution costs.")

    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None
