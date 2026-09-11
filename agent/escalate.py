"""Step 4: auto-handle or hand to a human, with a stated reason.

Two sources, kept apart on purpose (see taxonomy/ESCALATION.md):

    E1-E5  properties of the customer's message.  Decide the correct answer.
    R1-R2  properties of our program.             Never decide the correct answer.

The order below matters. Content rules are checked first, so a message that is BOTH a
security risk and low-confidence escalates as `security_risk` -- the real cause, not the
symptom. Escalating a hacked-account report as "the classifier was unsure" would be
technically an escalation and useless to the human who picks it up.

The LLM is asked for its own read of the policy too, and where it disagrees with the
deterministic rules the deterministic rules win. The model's opinion is recorded rather
than obeyed: `llm_disagreed` is a column in the results, because a policy an LLM
consistently reads differently from the code is a badly written policy.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from agent.schemas import EscalationDecision
from llm import LLMError, complete
from taxonomy.escalation import apply_policy

POLICY_MD = Path(__file__).resolve().parent.parent / "taxonomy" / "ESCALATION.md"

SYSTEM = f"""\
You apply an escalation policy to a customer support message. You decide one thing:
should a human handle this, and under which rule?

{POLICY_MD.read_text(encoding="utf-8") if POLICY_MD.exists() else ""}

Apply only the content rules E1-E5. You cannot observe R1 or R2. Rules are checked in
order and the first match wins. If no rule fires, do not escalate.
"""


@dataclass
class Decision:
    should_escalate: bool
    reason: str | None
    source: str | None      # E1-E5, R1, R2
    llm_disagreed: bool = False


def decide(text: str, *, confidence: float | None, top_similarity: float | None,
           prior_messages: int = 0, ask_llm: bool = True) -> Decision:
    # --- content rules first: the real cause beats the symptom ---------------
    verdict = apply_policy(text, prior_messages=prior_messages)

    llm_disagreed = False
    if ask_llm:
        try:
            res = complete(SYSTEM, f"Message:\n{text}", EscalationDecision,
                           provider=config.GEN_PROVIDER, model=config.GEN_MODEL)
            llm_disagreed = res.parsed.should_escalate != verdict.should_escalate
        except LLMError:
            pass  # the deterministic rules stand alone; the model is a second opinion

    if verdict.should_escalate:
        return Decision(True, verdict.reason, verdict.rule, llm_disagreed)

    # --- runtime signals: allowed to escalate, never to define correctness ---
    if confidence is not None and confidence < config.TAU:
        return Decision(True, "low_confidence", "R1", llm_disagreed)
    if top_similarity is not None and top_similarity < config.SIGMA:
        return Decision(True, "no_precedent", "R2", llm_disagreed)

    return Decision(False, None, None, llm_disagreed)


if __name__ == "__main__":
    for msg, conf, sim in [
        ("someone hacked my account and changed my email", 0.95, 0.8),
        ("how do I make a collaborative playlist", 0.92, 0.7),
        ("uhh idk whats happening lol", 0.30, 0.6),
        ("something about the thing on the app", 0.85, 0.10),
    ]:
        d = decide(msg, confidence=conf, top_similarity=sim, ask_llm=False)
        state = f"ESCALATE ({d.reason}, {d.source})" if d.should_escalate else "auto_handle"
        print(f"  conf={conf:.2f} sim={sim:.2f}  {state:<38} {msg}")
