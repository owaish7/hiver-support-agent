"""B2 -- the simple baseline the brief asks for. No LLM anywhere at inference.

TF-IDF turns each message into a sparse vector of word weights: a word scores high if it
is frequent in THIS message and rare across the corpus, so "refund" carries signal and
"the" does not. Logistic regression then learns one weight per word per class. The model
has no idea what any word means. That is precisely why it is the right thing to beat:
whatever the LLM is worth, it is worth the gap between these two rows.

Trained on the 60 hand-labelled DEV rows and tested on the 90 TEST rows. The two splits
are disjoint and were fixed before any label existed, so there is no leakage.

The original plan was to train on LLM weak labels instead, to get more training data. That
plan died on contact: the weak labels were generated against an earlier version of the
taxonomy and name classes that no longer exist, and regenerating 400 of them would cost
roughly 376k tokens against a 200k daily cap.

Training on 60 rows across 11 classes is genuinely thin -- about 5 examples per class --
so this baseline is weaker here than a fairer budget would make it. That understates the
"do you even need an LLM" comparison, and the report says so rather than presenting the
gap as if it were free of that handicap.

What this row actually measures: how much of the LLM's accuracy survives distillation
into something that costs nothing and answers in about three milliseconds. If the gap is
small, that is a real engineering finding and it belongs in the report.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from agent.schemas import PipelineResult

HERE = Path(__file__).resolve().parent
WEAK = HERE.parent / "data" / "weak_labels.jsonl"
MODEL_PKL = HERE / "tfidf_lr.pkl"


def _load(split: str = "dev") -> tuple[list[str], list[str]]:
    """Rows from one split only. Never reads test."""
    if not config.GOLDEN_JSONL.exists():
        raise SystemExit("golden/golden.jsonl not found -- run golden/label.py")
    X, y = [], []
    for line in config.GOLDEN_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("split") == split and r.get("intent"):
            X.append(r["text"])
            y.append(r["intent"])
    return X, y


def train() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    X, y = _load("dev")
    if len(set(y)) < 2:
        raise SystemExit("dev split has fewer than 2 classes -- nothing to learn")

    print(f"  training on {len(X)} hand-labelled dev rows, {len(set(y))} classes "
          f"(test split never read)")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # bigrams catch "log in", "charged twice"
            min_df=1,             # 60 training rows: min_df=2 discards most of the vocabulary
            sublinear_tf=True,    # 10 mentions of "refund" is not 10x one mention
            strip_accents="unicode",
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            # Without this the classifier learns to answer the majority class and stops.
            # It would look accurate and be useless on exactly the rare intents the
            # golden set was topped up to measure.
            class_weight="balanced",
        )),
    ])
    pipe.fit(X, y)
    MODEL_PKL.write_bytes(pickle.dumps(pipe))
    print(f"  wrote {MODEL_PKL.name}")

    from collections import Counter
    print("  training distribution:")
    for k, n in Counter(y).most_common():
        print(f"    {k:<24} {n:>5}")


class TfidfBaseline:
    name = "tfidf_lr"

    def __init__(self):
        if not MODEL_PKL.exists():
            train()
        self.pipe = pickle.loads(MODEL_PKL.read_bytes())

    def handle(self, text: str, *, item_id: str = "", prior_messages: int = 0):
        probs = self.pipe.predict_proba([text])[0]
        best = int(probs.argmax())
        return PipelineResult(
            id=item_id,
            text=text,
            intent=str(self.pipe.classes_[best]),
            # A real calibrated probability, unlike the LLM's self-reported number.
            intent_confidence=float(probs[best]),
            reply=None,          # this baseline classifies only; it writes nothing
            should_escalate=None,
            escalation_source="B2",
        ), []


if __name__ == "__main__":
    if "--train" in sys.argv:
        train()
    else:
        b = TfidfBaseline()
        for demo in ["you charged me twice for premium",
                     "cant log into my account",
                     "my songs keep skipping"]:
            r, _ = b.handle(demo)
            print(f"  {r.intent:<22} {r.intent_confidence:.2f}  {demo}")
