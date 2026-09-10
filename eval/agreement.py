"""Does the LLM judge agree with a human? The brief asks for this by name.

Until this file has been run, every reply-quality number in the project is a reading
from an uncalibrated instrument, and the honest thing is to say so rather than to quote
it. This is the file that turns "the system writes acceptable replies 78% of the time"
into a claim with a stated error bar on the measuring device itself.

What it prints, and why each part is needed:

  raw agreement    how often the judge and the human said the same thing. Necessary but
                   not sufficient: if 85% of replies are acceptable, two raters who both
                   just say "yes" agree 85% of the time while knowing nothing.
  Cohen's kappa    the same number with chance agreement subtracted out.
  the 2x2 table    which DIRECTION the judge is wrong in. A judge that is too generous
                   and one that is too harsh both produce the same kappa and imply
                   completely different fixes.
  per dimension    which of the four questions the disagreement lives in. "The judge is
                   unreliable" is not actionable; "the judge and I disagree about
                   groundedness specifically, and agree on the other three" is.
  disagreements    the actual replies, quoted, so the report can show rather than assert.

Usage:
    python eval/agreement.py
    python eval/agreement.py --config canned
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

config.use_utf8_stdout()

from eval.stats import cohen_kappa, raw_agreement

RESULTS = Path(__file__).resolve().parent / "results"
DIMS = ["grounded", "addresses_ask", "no_overpromise", "tone_ok", "acceptable"]
WRAP = 78


def band(k: float) -> str:
    if k != k:
        return "undefined"
    for cut, name in ((0.20, "slight"), (0.40, "fair"), (0.60, "moderate"),
                      (0.80, "substantial"), (1.01, "almost perfect")):
        if k <= cut:
            return name
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="system")
    args = ap.parse_args()

    hp = RESULTS / f"human_{args.config}.json"
    jp = RESULTS / f"judge_{args.config}.json"
    rp = RESULTS / f"{args.config}.json"
    for p, how in ((hp, "eval/human_judge.py"), (jp, "eval/judge.py")):
        if not p.exists():
            raise SystemExit(f"{p.name} not found -- run {how} first")

    human = json.loads(hp.read_text(encoding="utf-8"))
    judge = json.loads(jp.read_text(encoding="utf-8"))
    runs = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}

    ids = [i for i in human if i in judge and "error" not in judge[i]]
    if not ids:
        raise SystemExit("no overlapping graded items")

    print("\n" + "=" * WRAP)
    print(f"JUDGE vs HUMAN   config={args.config}   n={len(ids)}")
    print(f"  judge: {config.JUDGE_MODEL}   generator: {config.GEN_MODEL}")
    print("=" * WRAP)
    print(f"  {'dimension':<18}{'raw':>7}{'kappa':>9}   {'reading':<16}"
          f"{'judge yes':>11}{'human yes':>11}")
    print("-" * WRAP)

    for dim in DIMS:
        h = [bool(human[i].get(dim)) for i in ids]
        j = [bool(judge[i].get(dim)) for i in ids]
        k = cohen_kappa(h, j)
        kstr = "   n/a" if k != k else f"{k:>+.3f}"
        marker = "  <-- headline" if dim == "acceptable" else ""
        print(f"  {dim:<18}{raw_agreement(h, j):>7.0%}{kstr:>9}   {band(k):<16}"
              f"{sum(j)/len(j):>10.0%}{sum(h)/len(h):>11.0%}{marker}")

    print("-" * WRAP)
    print("  kappa: <0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate,")
    print("         0.61-0.80 substantial, 0.81+ almost perfect")
    print("  Read raw and kappa together. When one answer dominates, chance agreement")
    print("  is already high, so kappa can look poor at 90% raw agreement. That is the")
    print("  kappa paradox, not a broken calculation -- and the 'judge yes' vs")
    print("  'human yes' columns are what tell you which way the judge leans.")

    # 2x2 on the headline dimension: which direction is the judge wrong in?
    h = [bool(human[i]["acceptable"]) for i in ids]
    j = [bool(judge[i]["acceptable"]) for i in ids]
    tp = sum(1 for a, b in zip(h, j) if a and b)
    fn = sum(1 for a, b in zip(h, j) if a and not b)
    fp = sum(1 for a, b in zip(h, j) if not a and b)
    tn = sum(1 for a, b in zip(h, j) if not a and not b)
    print(f"\n  acceptable -- 2x2")
    print(f"                       judge yes   judge no")
    print(f"    human yes          {tp:>9}  {fn:>9}")
    print(f"    human no           {fp:>9}  {tn:>9}")
    if fp > fn:
        print(f"\n  The judge passes {fp} replies a human rejected and rejects {fn} a")
        print("  human passed: it is TOO GENEROUS, so the reply-quality number is an")
        print("  overestimate and should be reported as an upper bound.")
    elif fn > fp:
        print(f"\n  The judge rejects {fn} replies a human passed and passes {fp} a human")
        print("  rejected: it is TOO HARSH, so the reply-quality number is conservative.")
    else:
        print("\n  Errors are symmetric: the judge is noisy rather than biased.")

    # The actual disagreements, quoted.
    diffs = [i for i in ids if bool(human[i]["acceptable"]) != bool(judge[i]["acceptable"])]
    if diffs:
        print(f"\n  {len(diffs)} disagreements on `acceptable` -- quote 2-3 of these in")
        print("  the report, with which of you you think was right:")
        for iid in diffs[:5]:
            run = runs.get(iid, {})
            print("\n  " + "-" * (WRAP - 2))
            print(f"  human={human[iid]['acceptable']}  judge={judge[iid]['acceptable']}")
            if run.get("text"):
                print(textwrap.fill(f"customer: {run['text']}", WRAP - 4,
                                    initial_indent="    ", subsequent_indent="              "))
            if run.get("reply"):
                print(textwrap.fill(f"reply:    {run['reply']}", WRAP - 4,
                                    initial_indent="    ", subsequent_indent="              "))
            for dim in DIMS[:4]:
                if bool(human[iid].get(dim)) != bool(judge[iid].get(dim)):
                    why = judge[iid].get(f"{dim}_rationale", "")
                    print(f"      split on {dim}: judge said "
                          f"{judge[iid].get(dim)} -- {why[:110]}")
            if human[iid].get("note"):
                print(f"      your note: {human[iid]['note']}")

    k = cohen_kappa(h, j)
    print("\n" + "=" * WRAP)
    if k != k or k < 0.4:
        print("  VERDICT: agreement is weak. Reply-quality numbers from this judge are")
        print("  not yet evidence. Say so plainly in the report, show the disagreements,")
        print("  and name the rubric wording you would change first.")
    elif k < 0.6:
        print("  VERDICT: moderate agreement. Reply-quality numbers are indicative, not")
        print("  precise. Quote them with this kappa attached, every time.")
    else:
        print("  VERDICT: substantial agreement. Reply-quality numbers can be quoted,")
        print("  still with the kappa and the sample size beside them.")
    print("=" * WRAP)


if __name__ == "__main__":
    main()
