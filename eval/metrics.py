"""Classification metrics, written out rather than imported from sklearn.

They are quoted in the report and asked about in interviews, so they are here in a form
that can be read line by line. `--check` verifies every one against sklearn.

The definitions, in the terms the report uses:

    precision(c) = of the messages I called c, how many really were c
    recall(c)    = of the messages that really were c, how many did I find
    f1(c)        = harmonic mean of the two. Harmonic, not arithmetic, so that scoring
                   99% on one and 1% on the other gives ~2% rather than 50%.

    macro-F1     = plain average of the per-class F1s. Every class counts the same, so a
                   class with 8 examples matters as much as one with 40.
    weighted-F1  = average weighted by how common each class is. Common classes dominate.

Both are reported, always, and they answer different questions. Macro-F1 asks "does this
work for every intent"; weighted-F1 asks "what would a random customer experience". The
golden set was deliberately topped up on rare classes, so macro-F1 is measured on a
distribution that does not exist in the wild -- which is why printing it alone would be
the single most misleading thing this project could do.
"""

from __future__ import annotations

from collections import Counter


def confusion(y_true: list[str], y_pred: list[str], labels: list[str]
              ) -> dict[tuple[str, str], int]:
    """(actual, predicted) -> count. The off-diagonal cells are the failure analysis:
    a big cell says which two classes the boundary rule failed to separate."""
    cells = Counter(zip(y_true, y_pred))
    return {(a, p): cells.get((a, p), 0) for a in labels for p in labels}


def per_class(y_true: list[str], y_pred: list[str], labels: list[str]
              ) -> dict[str, dict[str, float]]:
    out = {}
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1,
                  "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}
    return out


def macro_f1(stats: dict[str, dict[str, float]]) -> float:
    """sklearn averages over every label passed in, including ones with zero support.
    Matching that matters: silently dropping empty classes would inflate the score by
    excluding exactly the classes the model never managed to predict."""
    if not stats:
        return 0.0
    return sum(s["f1"] for s in stats.values()) / len(stats)


def weighted_f1(stats: dict[str, dict[str, float]]) -> float:
    total = sum(s["support"] for s in stats.values())
    if not total:
        return 0.0
    return sum(s["f1"] * s["support"] for s in stats.values()) / total


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0


def binary_prf(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    """For the escalation decision, where the positive class is "needs a human".

    Recall is the safety number: of the messages that genuinely needed a human, how many
    did we route there. Precision is the cost: of the ones we sent, how many needed it.

    Recall alone is trivially gamed by escalating everything, which is why the results
    table always prints auto_handle_rate beside it. A system with 100% recall and a 4%
    auto-handle rate has not automated support, it has added a queue.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def format_confusion(y_true: list[str], y_pred: list[str], labels: list[str]) -> str:
    """Compact square grid. Rows are truth, columns are prediction, so anything off the
    diagonal reads as 'this actual class got called that instead'."""
    cells = confusion(y_true, y_pred, labels)
    short = {c: c[:6] for c in labels}
    width = 7
    lines = [" " * 22 + "".join(f"{short[c]:>{width}}" for c in labels)]
    for a in labels:
        row = "".join(
            f"{cells[(a, p)] or '.':>{width}}" for p in labels)
        support = sum(cells[(a, p)] for p in labels)
        lines.append(f"  {a:<20}{row}   ({support})")
    lines.append("  rows = actual, columns = predicted, '.' = zero")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        try:
            from sklearn.metrics import f1_score, precision_recall_fscore_support
        except ImportError:
            print("  sklearn not installed, skipping"); sys.exit(0)

        import random
        random.seed(3)
        labels = ["a", "b", "c", "d"]
        yt = [random.choice(labels) for _ in range(300)]
        yp = [random.choice(labels) if random.random() < 0.4 else t for t in yt]

        stats = per_class(yt, yp, labels)
        ok = True
        p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
        for i, c in enumerate(labels):
            for name, mine, theirs in (("precision", stats[c]["precision"], p[i]),
                                       ("recall", stats[c]["recall"], r[i]),
                                       ("f1", stats[c]["f1"], f[i])):
                match = abs(mine - theirs) < 1e-9
                ok &= match
                if not match:
                    print(f"  MISMATCH {c} {name}: {mine} vs {theirs}")
        for name, mine, theirs in (
                ("macro_f1", macro_f1(stats), f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
                ("weighted_f1", weighted_f1(stats), f1_score(yt, yp, labels=labels, average="weighted", zero_division=0))):
            match = abs(mine - theirs) < 1e-9
            ok &= match
            print(f"  {name:<12} mine={mine:.6f} sklearn={theirs:.6f} "
                  f"{'OK' if match else 'MISMATCH'}")
        print(f"  per-class precision/recall/f1 across {len(labels)} classes: "
              f"{'all OK' if ok else 'MISMATCH'}")
        sys.exit(0 if ok else 1)

    yt = ["a", "a", "b", "b", "c"]
    yp = ["a", "b", "b", "b", "a"]
    print(format_confusion(yt, yp, ["a", "b", "c"]))
