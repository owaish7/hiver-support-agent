"""LLM-as-judge for reply quality, on four binary dimensions.

Reply quality cannot be scored by a formula, so a model scores it. That model is then an
instrument, and an instrument nobody has calibrated is not evidence -- which is what
eval/agreement.py exists to fix. This file only produces the readings.

Three choices to defend:

  * A DIFFERENT MODEL FAMILY from the generator. Gemini writes, Llama grades. Models
    score their own family's output more generously, so self-grading would inflate the
    headline by an amount nobody can measure.

  * BINARY DIMENSIONS, not a 1-5 score. Nobody can say what separates a 3 from a 4, so
    averaging 1-5 ratings produces a number that cannot be acted on and cannot be
    compared against a human rating. A yes/no against a named failure mode can be both.

  * THE JUDGE SEES THE EVIDENCE. Groundedness is "traceable to the retrieved examples",
    so a judge without those examples would be rating plausibility instead, which is the
    one thing a fluent model is guaranteed to score highly on.

Usage:
    python eval/judge.py                     # judge the system's cached replies
    python eval/judge.py --config canned     # judge a baseline
    python eval/judge.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

config.use_utf8_stdout()

from agent.retrieve import format_evidence, get_retriever
from agent.schemas import JudgeVerdict
from llm import LLMError, complete

RESULTS = Path(__file__).resolve().parent / "results"

SYSTEM = """\
You grade a draft reply written by a customer support agent for a brand on Twitter.

You are given the customer's message, the draft reply, and the real past cases the
drafter was shown. Judge the draft ONLY against those past cases -- they define what this
brand actually says and does. You have no other knowledge of this brand's policies.

For each of the four dimensions, write a short rationale FIRST and then the verdict.

grounded: is every factual claim in the reply traceable to one of the past cases? A
reply that makes no factual claims at all (pure acknowledgement) counts as grounded.

addresses_ask: does the reply respond to what this customer actually asked? A reply that
answers a different question, or is so generic it would fit any message, is not
addressing the ask.

no_overpromise: does the reply avoid promising refunds, compensation, timeframes, or
fixes that the past cases do not show this brand promising?

tone_ok: does the tone match the brand's voice in the past replies, and suit how upset
this customer is?

Judge only what is written. Do not reward length, politeness formulae, or confidence.
"""


def judge_one(customer: str, reply: str, evidence_ids: list[str]) -> JudgeVerdict:
    evidence = get_retriever().by_ids(evidence_ids)
    user = (f"Customer message:\n{customer}\n\n"
            f"Draft reply:\n{reply}\n\n"
            f"Past cases the drafter was shown:\n{format_evidence(evidence)}")
    return complete(SYSTEM, user, JudgeVerdict,
                    provider="groq", model=config.JUDGE_MODEL).parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="system")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = RESULTS / f"{args.config}.json"
    if not src.exists():
        raise SystemExit(f"{src.name} not found -- run eval/run_eval.py first")
    runs = json.loads(src.read_text(encoding="utf-8"))

    out_path = RESULTS / f"judge_{args.config}.json"
    done = {} if args.force else (
        json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {})

    todo = [(k, v) for k, v in runs.items()
            if v.get("reply") and k not in done]
    if args.limit:
        todo = todo[: args.limit]

    skipped = sum(1 for v in runs.values() if not v.get("reply"))
    print(f"[judge:{args.config}] {len(todo)} to grade, {len(done)} cached, "
          f"{skipped} had no reply to grade")
    print(f"  judge model: {config.JUDGE_MODEL} (deliberately not {config.GEN_MODEL})")

    errors = 0
    for i, (iid, run) in enumerate(todo, 1):
        try:
            v = judge_one(run["text"], run["reply"], run.get("retrieved_ids", []))
            rec = v.model_dump()
            rec["acceptable"] = v.acceptable
        except LLMError as exc:
            rec = {"error": str(exc)[:200]}
            errors += 1
        done[iid] = rec
        out_path.write_text(json.dumps(done, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        if i % 10 == 0 or i == len(todo):
            print(f"\r  {i}/{len(todo)}  errors={errors}", end="", flush=True)
        time.sleep(0.05)
    print()

    graded = [v for v in done.values() if "error" not in v]
    if not graded:
        return

    n = len(graded)
    print(f"\n  {n} replies graded")
    for dim in ("grounded", "addresses_ask", "no_overpromise", "tone_ok"):
        passed = sum(1 for v in graded if v.get(dim))
        print(f"    {dim:<18} {passed:>4}/{n}  {passed/n:>5.0%}")
    acc = sum(1 for v in graded if v.get("acceptable"))
    print(f"    {'ACCEPTABLE':<18} {acc:>4}/{n}  {acc/n:>5.0%}"
          f"   (grounded AND addresses_ask AND no_overpromise)")
    print("\n  tone is excluded from `acceptable` on purpose: it is the softest")
    print("  dimension and the one a judge is least likely to agree with a human on,")
    print("  so letting it gate the headline would import that noise into everything.")
    print("\n  None of this is evidence until eval/agreement.py says the judge agrees")
    print("  with a human. Run that before quoting any number above.")


if __name__ == "__main__":
    main()
