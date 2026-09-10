"""Build the pool of messages to hand-label. Three strata, deliberately unequal.

    140  random               natural prevalence, the honest half
     45  rare-class top-up    so a rare intent is not scored on 3 examples
     15  hard cases           picked by pattern, not by luck
    ---
    200

The top-up is the biggest single distortion in the whole project, so it is stated
plainly here and again in the report: after topping up, the class mix in the golden set
is NOT the class mix in the wild. Macro-F1 computed on it flatters rare classes. The
report therefore prints macro-F1, prevalence-weighted F1, and the true distribution
measured from the weak labels, and says which one production would feel.

Why top up at all: a class with 3 examples has a standard error near 30 points. Any
per-class number computed on it is noise, and "the model is bad at content_missing"
would be an unfalsifiable claim. Precision on rare classes is worth a stated bias.

Why the hard cases are picked by pattern rather than sampled: a random draw of 140
contains almost no non-English, emoji-only, or multi-intent messages, so the failure
analysis would have nothing real to analyse. They are tagged is_hard so every headline
number can be printed with and without them.

The dev/test split is assigned HERE, before any label exists, and is hidden by the
labelling tool. Assigning it afterwards would let the split be chosen -- even
unconsciously -- with knowledge of which items turned out awkward.

Usage:
    python golden/sample_golden.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

config.use_utf8_stdout()
from taxonomy.intents import INTENTS

HERE = Path(__file__).resolve().parent
POOL = HERE / "pool.jsonl"
WEAK = Path(__file__).resolve().parent.parent / "data" / "weak_labels.jsonl"

N_HARD = config.N_HARD
N_RANDOM = config.N_RANDOM
N_TOTAL = config.N_GOLDEN

EMOJI_ONLY = re.compile(r"^[\W\d_]+$", re.UNICODE)
INJECTION = re.compile(
    r"ignore (all )?(previous|prior|above)|system ?(notice|prompt|message)|"
    r"you are now|disregard (your|all)", re.IGNORECASE)

# Rough per-intent keyword groups. Used ONLY to spot messages that trip several groups
# at once, i.e. plausible multi-intent. Never used as a label.
SIGNALS = {
    "playback": r"\b(play|playing|skip|buffer|stutter|offline|download)\b",
    "access": r"\b(log ?in|login|password|locked|sign ?in|account)\b",
    "billing": r"\b(charg|paid|payment|card|bill|invoice|refund)\b",
    "subscription": r"\b(cancel|premium|subscri|upgrade|family|student|plan)\b",
    "content": r"\b(song|album|artist|podcast|playlist|removed|missing)\b",
    "app": r"\b(crash|update|app|install|version|freeze|bug)\b",
}


def strip_handles(text: str) -> str:
    return re.sub(r"@\w+", "", str(text)).strip()


def hardness(text: str) -> tuple[int, str, str]:
    """Score how awkward a message is. Returns (score, primary_kind, full_reason).

    `primary_kind` exists because the hard stratum is only 15 rows and one failure kind
    can easily flood it -- a brand's mentions contain thousands of bare "?" tweets, and
    taking the top 15 by score alone returns fifteen of them. Fifteen examples of one
    thing is one example. Selection below spreads the budget across kinds.
    """
    body = strip_handles(text)
    words = body.split()
    scored: list[tuple[int, str, str]] = []  # (weight, kind, reason)

    if INJECTION.search(body):
        scored.append((5, "injection", "looks like prompt injection"))
    if EMOJI_ONLY.match(body) and body:
        scored.append((4, "no_words", "no words at all"))
    if len(words) <= 3:
        scored.append((3, "too_short", "almost no content"))
    non_ascii = sum(1 for ch in body if ord(ch) > 127)
    if body and non_ascii / len(body) > 0.3:
        scored.append((3, "non_english", "largely non-English or emoji"))
    hits = [name for name, pat in SIGNALS.items() if re.search(pat, body, re.I)]
    if len(hits) >= 2:
        scored.append((3 if len(hits) >= 3 else 2, "multi_intent",
                       f"signals for {', '.join(hits)}"))
    if len(words) > 55:
        scored.append((2, "very_long", "very long"))
    if re.search(r"\b(thanks a lot|great job|wonderful|love how)\b.*\b(not|never|still)\b",
                 body, re.I):
        scored.append((2, "sarcasm", "possible sarcasm"))

    if not scored:
        return 0, "", ""
    total = sum(w for w, _, _ in scored)
    scored.sort(key=lambda t: -t[0])
    return total, scored[0][1], "; ".join(r for _, _, r in scored)


def normalise(text: str) -> str:
    """Collapse near-duplicates. Support mentions are full of messages that differ only
    by a username or an order number, and two of those are not two test cases."""
    body = strip_handles(text).lower()
    body = re.sub(r"https?://\S+", "", body)
    body = re.sub(r"\d+", "#", body)
    return re.sub(r"\s+", " ", body).strip()


def load_weak() -> dict[str, str]:
    if not WEAK.exists():
        print("  no weak_labels.jsonl -- skipping the rare-class top-up.")
        print("  run data/weak_label.py first, or accept a purely random 155.")
        return {}
    out = {}
    for line in WEAK.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("weak_intent"):
            out[rec["id"]] = rec["weak_intent"]
    return out


def main() -> None:
    if not config.SAMPLE_CSV.exists():
        raise SystemExit(f"{config.SAMPLE_CSV.name} not found -- run data/make_sample.py")

    df = pd.read_csv(config.SAMPLE_CSV)
    df["customer_tweet_id"] = df["customer_tweet_id"].astype(str)
    df = df[df["customer_text"].astype(str).str.strip().str.len() > 0]

    rng = random.Random(config.SEED)
    weak = load_weak()
    chosen: dict[str, dict] = {}

    # ---- stratum 1: random at natural prevalence ---------------------------
    ids = df["customer_tweet_id"].tolist()
    rng.shuffle(ids)
    by_id = df.set_index("customer_tweet_id")
    for tid in ids:
        if len(chosen) >= N_RANDOM:
            break
        chosen[tid] = {"stratum": "random", "hard_reason": ""}

    # ---- stratum 2: hard cases, spread across kinds -------------------------
    # Round-robin over failure kinds instead of taking the global top 15, and skip
    # near-duplicates of anything already taken. Fifteen bare "?" tweets would be one
    # test case wearing fifteen hats.
    # Deduping is scoped to the hard picks only. Seeding it from the random stratum
    # looked tidier but silently removed an entire failure kind whenever one example of
    # it happened to land in the random draw -- and the kinds most likely to land there
    # are exactly the common ones worth testing.
    seen_norm: set[str] = set()
    buckets: dict[str, list[tuple[int, str, str]]] = {}
    for tid in ids:
        if tid in chosen:
            continue
        s, kind, why = hardness(by_id.loc[tid, "customer_text"])
        if s > 0:
            buckets.setdefault(kind, []).append((s, tid, why))
    for lst in buckets.values():
        lst.sort(key=lambda t: -t[0])

    order = sorted(buckets, key=lambda k: -max(s for s, _, _ in buckets[k]))
    while len([c for c in chosen.values() if c["stratum"] == "hard"]) < N_HARD and order:
        progressed = False
        for kind in list(order):
            if len([c for c in chosen.values() if c["stratum"] == "hard"]) >= N_HARD:
                break
            while buckets[kind]:
                s, tid, why = buckets[kind].pop(0)
                norm = normalise(by_id.loc[tid, "customer_text"])
                if norm in seen_norm:
                    continue
                seen_norm.add(norm)
                chosen[tid] = {"stratum": "hard", "hard_reason": why}
                progressed = True
                break
            else:
                order.remove(kind)
        if not progressed:
            break

    # ---- stratum 3: top up the rarest classes ------------------------------
    budget = N_TOTAL - len(chosen)
    if weak and budget > 0:
        have = {i: 0 for i in INTENTS}
        for tid in chosen:
            lbl = weak.get(tid)
            if lbl in have:
                have[lbl] += 1
        pool_by_intent: dict[str, list[str]] = {i: [] for i in INTENTS}
        for tid, lbl in weak.items():
            if tid not in chosen and lbl in pool_by_intent and tid in by_id.index:
                pool_by_intent[lbl].append(tid)
        for lst in pool_by_intent.values():
            rng.shuffle(lst)

        # Rarest first, so the scarcest classes get the budget.
        for intent in sorted(INTENTS, key=lambda i: have[i]):
            while (have[intent] < config.N_PER_RARE_CLASS and budget > 0
                   and pool_by_intent[intent]):
                tid = pool_by_intent[intent].pop()
                chosen[tid] = {"stratum": "topup", "hard_reason": ""}
                have[intent] += 1
                budget -= 1

    # ---- dev/test split, fixed now, hidden from the labeller ---------------
    tids = sorted(chosen)
    rng.shuffle(tids)
    dev = set(tids[:config.DEV_SIZE])

    with POOL.open("w", encoding="utf-8") as fh:
        for n, tid in enumerate(tids, 1):
            row = by_id.loc[tid]
            fh.write(json.dumps({
                "id": tid,
                "seq": n,
                "text": str(row["customer_text"]),
                "brand_reply": str(row["brand_reply_text"]),
                "split": "dev" if tid in dev else "test",
                "stratum": chosen[tid]["stratum"],
                "is_hard": chosen[tid]["stratum"] == "hard",
                "hard_reason": chosen[tid]["hard_reason"],
                "weak_intent": weak.get(tid),
            }, ensure_ascii=False) + "\n")

    counts = pd.Series([c["stratum"] for c in chosen.values()]).value_counts()
    print(f"\n  wrote {POOL.name}: {len(chosen)} messages to label")
    for name, n in counts.items():
        print(f"    {name:<10} {n:>4}")
    print(f"    {'dev':<10} {len(dev):>4}   (tuning happens here)")
    print(f"    {'test':<10} {len(chosen) - len(dev):>4}   (frozen, touched once)")
    print(f"\n  next:  python golden/label.py\n")


if __name__ == "__main__":
    main()
