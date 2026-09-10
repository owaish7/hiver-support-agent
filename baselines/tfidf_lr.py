"""B2 -- the simple baseline the brief asks for. No LLM anywhere at inference.

TF-IDF turns each message into a sparse vector of word weights: a word scores high if it
is frequent in THIS message and rare across the corpus, so "refund" carries signal and
"the" does not. Logistic regression then learns one weight per word per class. The model
has no idea what any word means. That is precisely why it is the right thing to beat:
whatever the LLM is worth, it is worth the gap between these two rows.

Trained on the LLM's weak labels rather than on hand labels, for two reasons:

  * 200 hand labels split into dev/test leaves ~60 to train on, which is not enough for
    8 classes and would understate this baseline unfairly.
  * training on the golden set and testing on it would be circular.

Golden ids are excluded from training explicitly (see `_load`), because the weak-label
pool and the golden pool are drawn from the same 20k rows and overlap by construction.
Without that exclusion this baseline would be tested on rows it trained on and the
comparison would be meaningless in the flattering direction.

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


def _load(exclude_ids: set[str]) -> tuple[list[str], list[str]]:
    if not WEAK.exists():
        raise SystemExit("data/weak_labels.jsonl not found -- run data/weak_label.py")
    X, y = [], []
    for line in WEAK.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("weak_intent") and r["id"] not in exclude_ids:
            X.append(r["text"])
            y.append(r["weak_intent"])
    return X, y


def golden_ids() -> set[str]:
    if not config.GOLDEN_JSONL.exists():
        return set()
    return {json.loads(l)["id"]
            for l in config.GOLDEN_JSONL.read_text(encoding="utf-8").splitlines()
            if l.strip()}


def train() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    excluded = golden_ids()
    X, y = _load(excluded)
    if len(set(y)) < 2:
        raise SystemExit("weak labels contain fewer than 2 classes -- nothing to learn")

    print(f"  training on {len(X)} weak-labelled messages "
          f"({len(excluded)} golden ids held out)")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # bigrams catch "log in", "charged twice"
            min_df=2,             # a word seen once cannot generalise, only memorise
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
