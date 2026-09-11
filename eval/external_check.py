"""External validity: does the classifier work against labels I did NOT write?

The brief offers Banking77 as an optional secondary dataset "for intent work only".
It is not used to define this project's taxonomy -- 77 retail-banking intents have
nothing to do with a music-streaming brand, and 77 classes against a 200-row answer key
would be 2.6 examples per class. That use was considered and rejected.

This is the other use, and it addresses the single biggest hole in the evaluation.

THE PROBLEM IT SOLVES
---------------------
One person wrote this project's labels AND its prompts. Those two things came out of the
same head, so they are correlated in a way a second annotator's would not be. A reviewer
can fairly ask how much of the reported accuracy is the classifier being good, and how
much is the answer key being shaped like the classifier.

Banking77 is 3,080 test queries labelled by PolyAI, on a public benchmark, with published
reference numbers. Running the same prompting approach against it produces a claim that
does not depend on my answer key at all.

WHAT IT DOES NOT SHOW
---------------------
Banking77 queries are clean, short and well-formed. Real support tweets are typo-ridden,
sarcastic, emoji-laden, and frequently barely parseable. So this validates the METHOD, not
robustness to noise -- and a strong score here beside a weak score on tweets would itself
be the finding.

It is also not like-for-like against the published state of the art: ~94% comes from a
fine-tuned encoder that saw 10,003 training examples, while this is zero-shot. The
comparison is a reference point, not a competition.

Two rows are produced:

    tfidf_lr   trained on the 10,003 train split, tested on all 3,080. No API, instant.
    llm        the same prompting approach as agent/classify.py, on a stratified sample.

Usage:
    python eval/external_check.py --tfidf          # free, no key, ~20 seconds
    python eval/external_check.py --llm --limit 20 # smoke test
    python eval/external_check.py --llm            # 154 calls, 2 per intent
    python eval/external_check.py --report         # offline, from cache
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

config.use_utf8_stdout()

from eval.metrics import accuracy, macro_f1, per_class
from eval.stats import fmt_ci
from llm import LLMError, complete

# Source: PolyAI-LDN/task-specific-datasets, CC-BY-4.0.
# Casanueva et al. 2020, "Efficient Intent Detection with Dual Sentence Encoders"
# (https://arxiv.org/abs/2003.04807). Downloaded on demand rather than committed, so
# this repo does not redistribute someone else's dataset.
BASE = ("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets"
        "/master/banking_data")

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = Path(__file__).resolve().parent / "results"
CACHE = RESULTS / "banking77.json"

# A fine-tuned ModernBERT-base reported 0.9399 accuracy / 0.9401 macro-F1 on this split.
# Quoted as a reference point, not as a target: that model saw 10,003 labelled training
# examples and this run sees none.
PUBLISHED_REFERENCE = 0.94
# 2 x 77 = 154 calls. The test split is balanced at 40 per intent, so stratifying
# preserves its distribution rather than distorting it. Two rather than three purely to
# fit the measured free-tier budget in config.py.
PER_INTENT_SAMPLE = 2


def load_split(split: str) -> pd.DataFrame:
    local = DATA / f"banking77_{split}.csv"
    if local.exists():
        return pd.read_csv(local)
    print(f"  downloading banking77 {split} split")
    df = pd.read_csv(f"{BASE}/{split}.csv")
    DATA.mkdir(exist_ok=True)
    df.to_csv(local, index=False)
    return df


def build_prompt(intents: list[str]) -> str:
    return (
        "You classify customer queries sent to an online bank into exactly one intent.\n\n"
        "Intents:\n" + "\n".join(f"- {i}" for i in sorted(intents)) + "\n\n"
        "Several intents are deliberately close together (for example card_arrival vs "
        "card_delivery_estimate, or top_up_failed vs top_up_reverted). Pick the one that "
        "matches what the customer is asking for, not merely what they mention.\n\n"
        "Reply with the intent name exactly as written above."
    )


def run_tfidf(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    print(f"  training TF-IDF + logistic regression on {len(train):,} examples")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                  strip_accents="unicode", lowercase=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(train["text"], train["category"])
    pred = list(pipe.predict(test["text"]))
    truth = list(test["category"])
    return {"y_true": truth, "y_pred": pred, "n": len(truth)}


def run_llm(test: pd.DataFrame, limit: int | None, force: bool) -> dict:
    from pydantic import BaseModel, Field

    intents = sorted(test["category"].unique())

    class BankingIntent(BaseModel):
        reasoning: str = Field(description="One short sentence: what is being asked for.")
        intent: str = Field(description="Exactly one intent name from the list.")

    # Stratified: every one of the 77 intents gets represented. A random 154 would miss
    # roughly a third of the classes outright, and per-class numbers on the rest would be
    # computed from whatever happened to be drawn.
    # Plain loop rather than groupby().apply(): the apply form needs an `include_groups`
    # argument whose accepted values changed across pandas versions, and this has to work
    # on whatever the reviewer has installed.
    parts = [g.sample(min(PER_INTENT_SAMPLE, len(g)), random_state=config.SEED)
             for _, g in test.groupby("category")]
    sample = pd.concat(parts).reset_index(drop=True)
    if limit:
        sample = sample.head(limit)

    cached = {}
    if CACHE.exists() and not force:
        blob = json.loads(CACHE.read_text(encoding="utf-8"))
        cached = blob.get("llm_items", {})

    system = build_prompt(intents)
    valid = set(intents)
    # Retry anything that failed. Errors are cached so the reason stays visible, but a
    # cached error is not a result -- without this, a run that failed because the key was
    # missing would be silently skipped once the key was added, and the table would be
    # computed from whatever fraction happened to succeed.
    todo = [r for r in sample.itertuples(index=False)
            if r.text not in cached or cached[r.text].get("pred") is None]
    print(f"  {len(todo)} queries to classify, {len(cached)} cached  "
          f"(model: {config.GEN_MODEL})")

    errors = invalid = 0
    for i, row in enumerate(todo, 1):
        try:
            res = complete(system, f"Query:\n{row.text}", BankingIntent,
                           provider=config.GEN_PROVIDER, model=config.GEN_MODEL)
            pred = res.parsed.intent.strip()
            if pred not in valid:
                # A label outside the taxonomy is a wrong answer, not a crash. Counting
                # it as an error instead would quietly remove the model's worst failures
                # from the accuracy figure.
                invalid += 1
            cached[row.text] = {"truth": row.category, "pred": pred}
        except LLMError as exc:
            errors += 1
            cached[row.text] = {"truth": row.category, "pred": None,
                                "error": str(exc)[:150]}
        if i % 10 == 0 or i == len(todo):
            RESULTS.mkdir(parents=True, exist_ok=True)
            blob = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
            blob["llm_items"] = cached
            CACHE.write_text(json.dumps(blob, indent=2, ensure_ascii=False),
                             encoding="utf-8")
            print(f"\r  {i}/{len(todo)}  errors={errors} off-taxonomy={invalid}",
                  end="", flush=True)
        time.sleep(0.05)
    print()

    usable = [v for v in cached.values() if v.get("pred") is not None]
    return {"y_true": [v["truth"] for v in usable],
            "y_pred": [v["pred"] for v in usable],
            "n": len(usable), "errors": errors, "off_taxonomy": invalid}


def save(key: str, result: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    blob = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    blob[key] = {k: v for k, v in result.items() if k != "llm_items"}
    CACHE.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")


def report() -> None:
    if not CACHE.exists():
        raise SystemExit("nothing cached -- run with --tfidf and/or --llm first")
    blob = json.loads(CACHE.read_text(encoding="utf-8"))

    print("\n" + "=" * 84)
    print("EXTERNAL CHECK -- Banking77 (PolyAI), 77 intents, labels written by someone else")
    print("=" * 84)
    print(f"{'approach':<34}{'accuracy':>13}{'macro-F1':>11}{'n':>8}{'classes':>9}")
    print("-" * 84)

    for key, label in (("tfidf", "TF-IDF + logistic regression"),
                       ("llm", f"LLM zero-shot ({config.GEN_MODEL})")):
        r = blob.get(key)
        if not r or not r.get("y_true"):
            continue
        labels = sorted(set(r["y_true"]) | set(r["y_pred"]))
        acc = accuracy(r["y_true"], r["y_pred"])
        mf1 = macro_f1(per_class(r["y_true"], r["y_pred"], labels))
        correct = sum(1 for a, b in zip(r["y_true"], r["y_pred"]) if a == b)
        print(f"{label:<34}{fmt_ci(correct, r['n']):>13}{mf1:>11.3f}"
              f"{r['n']:>8}{len(set(r['y_true'])):>9}")

    print(f"{'fine-tuned ModernBERT (published)':<34}{PUBLISHED_REFERENCE:>12.0%}"
          f"{'0.940':>11}{3080:>8}{77:>9}")
    print("-" * 84)
    print("  Why this table is here: every other number in this project is measured")
    print("  against labels I wrote myself, using prompts I also wrote. These labels are")
    print("  PolyAI's. If the method holds up here, the main results are not purely an")
    print("  artefact of my own answer key.")
    print()
    print("  What it does NOT show:")
    print("   - Banking77 queries are clean and well-formed; tweets are not. This tests")
    print("     the method, not robustness to noise.")
    print("   - The published row is a fine-tuned encoder that saw 10,003 training")
    print("     examples. The LLM row is zero-shot. Reference point, not a competition.")
    print("   - Different domain entirely. Retail banking, not music streaming.")

    llm = blob.get("llm")
    if llm and llm.get("off_taxonomy"):
        print(f"\n  {llm['off_taxonomy']} predictions were not valid intent names and are")
        print("  counted as wrong, not dropped. Dropping them would quietly remove the")
        print("  model's worst failures from the accuracy figure.")

    for key, label in (("tfidf", "TF-IDF"), ("llm", "LLM")):
        r = blob.get(key)
        if not r or not r.get("y_true"):
            continue
        pairs: dict[tuple[str, str], int] = {}
        for t, p in zip(r["y_true"], r["y_pred"]):
            if t != p:
                pairs[(t, p)] = pairs.get((t, p), 0) + 1
        if pairs:
            print(f"\n  [{label}] most confused pairs (actual -> predicted):")
            for (t, p), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:8]:
                print(f"    {n:>3}x  {t}  ->  {p}")
            print("  Banking77 contains genuinely near-identical intents, so a human")
            print("  ceiling here is well below 100% too.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfidf", action="store_true")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.tfidf:
        save("tfidf", run_tfidf(load_split("train"), load_split("test")))
    if args.llm:
        save("llm", run_llm(load_split("test"), args.limit, args.force))
    if not (args.tfidf or args.llm) and not args.report:
        ap.print_help()
        return
    report()


if __name__ == "__main__":
    main()
