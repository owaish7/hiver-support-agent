"""The eval harness. This is the point of the project.

Runs every config over the golden set, grades against the hand labels, prints the table.

Results are cached per (config, item). A rerun costs nothing for work already done, an
interrupted run resumes instead of re-billing, and `--report` reproduces the entire
headline table with zero API calls -- which is what makes "reproduce our results in under
15 minutes" true for a reviewer who has no keys.

    python eval/run_eval.py --report              # offline, from cache, seconds
    python eval/run_eval.py --split dev           # tune here
    python eval/run_eval.py --split test          # touch once
    python eval/run_eval.py --config system --limit 5   # smoke test before spending
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

from eval.metrics import (accuracy, binary_prf, format_confusion, macro_f1, per_class,
                          weighted_f1)
from eval.stats import fmt_ci, wilson
from taxonomy.intents import INTENTS

RESULTS = Path(__file__).resolve().parent / "results"

CONFIGS = ["trivial_always_auto", "trivial_always_escalate", "canned",
           "tfidf_lr", "retrieval_verbatim", "system"]

BLURB = {
    "trivial_always_auto": "B0 majority intent, never escalates",
    "trivial_always_escalate": "B0 majority intent, escalates everything",
    "canned": "B1 the brand's most common reply, to everything",
    "tfidf_lr": "B2 TF-IDF + logistic regression, no LLM",
    "retrieval_verbatim": "B3 nearest past case, brand's real reply verbatim",
    "system": "   classify -> retrieve -> draft -> escalate",
}


def build(name: str):
    if name.startswith("trivial_"):
        from baselines.trivial import TrivialBaseline
        return TrivialBaseline(name.replace("trivial_", ""))
    if name == "canned":
        from baselines.canned import CannedBaseline
        return CannedBaseline()
    if name == "tfidf_lr":
        from baselines.tfidf_lr import TfidfBaseline
        return TfidfBaseline()
    if name == "retrieval_verbatim":
        from baselines.retrieval_verbatim import RetrievalVerbatimBaseline
        return RetrievalVerbatimBaseline()
    if name == "system":
        from agent.pipeline import SupportAgent
        return SupportAgent()
    raise SystemExit(f"unknown config {name!r}")


# ------------------------------------------------------------------ cache

def cache_path(name: str) -> Path:
    return RESULTS / f"{name}.json"


def load_cache(name: str) -> dict:
    p = cache_path(name)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_cache(name: str, cache: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cache_path(name).write_text(json.dumps(cache, indent=2, ensure_ascii=False),
                                encoding="utf-8")


def load_golden(split: str) -> list[dict]:
    if not config.GOLDEN_JSONL.exists():
        raise SystemExit("golden/golden.jsonl not found -- run golden/label.py")
    rows = [json.loads(l) for l in
            config.GOLDEN_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    if split != "all":
        rows = [r for r in rows if r.get("split") == split]
    return rows


# ------------------------------------------------------------------ running

def run_config(name: str, rows: list[dict], force: bool) -> dict:
    cache = {} if force else load_cache(name)
    # Retry anything that errored. Errors stay in the cache so the reason is visible, but
    # a cached error is not a result: on a rate-limited free tier transient failures are
    # normal, and treating them as done would silently shrink the eval to whatever
    # fraction happened to succeed -- while still printing a confident table.
    todo = [r for r in rows
            if r["id"] not in cache or cache[r["id"]].get("error")]
    if not todo:
        print(f"[{name}] all {len(cache)} cached")
        return cache

    system = build(name)
    print(f"\n[{name}] {len(todo)} to run, {len(cache)} cached")
    for i, row in enumerate(todo, 1):
        started = time.perf_counter()
        result, _ = system.handle(row["text"], item_id=row["id"],
                                  prior_messages=row.get("prior_messages", 0))
        result.latency_s = round(time.perf_counter() - started, 2)
        cache[row["id"]] = result.model_dump()
        # Save every item: a crash at 190 must not discard 189 paid-for calls.
        save_cache(name, cache)
        flag = " ERR" if result.error else ""
        print(f"  {i:>3}/{len(todo)}  {result.intent or '-':<22} "
              f"{result.latency_s:>5.1f}s{flag}")
    return cache


# ------------------------------------------------------------------ grading

def grade(name: str, rows: list[dict], cache: dict) -> dict:
    by_id = {r["id"]: r for r in rows}
    m = {"config": name, "n": 0, "errors": 0,
         "y_true": [], "y_pred": [], "esc_true": [], "esc_pred": [],
         "replies": 0, "in_tok": 0, "out_tok": 0, "latency": [],
         "hard_true": [], "hard_pred": [], "failures": []}

    for iid, entry in cache.items():
        row = by_id.get(iid)
        if row is None:
            continue  # cached from another split
        m["n"] += 1
        m["in_tok"] += entry.get("input_tokens", 0)
        m["out_tok"] += entry.get("output_tokens", 0)
        m["latency"].append(entry.get("latency_s", 0.0))
        if entry.get("error"):
            m["errors"] += 1

        if entry.get("intent") is not None:
            m["y_true"].append(row["intent"])
            m["y_pred"].append(entry["intent"])
            if row.get("is_hard"):
                m["hard_true"].append(row["intent"])
                m["hard_pred"].append(entry["intent"])
            if row["intent"] != entry["intent"]:
                m["failures"].append(
                    (iid, f"intent {entry['intent']} want {row['intent']}",
                     row["text"][:80]))

        if entry.get("should_escalate") is not None:
            m["esc_true"].append(bool(row["should_escalate"]))
            m["esc_pred"].append(bool(entry["should_escalate"]))
            if bool(row["should_escalate"]) and not entry["should_escalate"]:
                m["failures"].append(
                    (iid, f"MISSED escalation ({row.get('escalation_reason')})",
                     row["text"][:80]))

        if entry.get("reply"):
            m["replies"] += 1
    return m


def summarise(m: dict) -> dict:
    s = {"config": m["config"], "n": m["n"], "errors": m["errors"]}
    if m["y_true"]:
        stats = per_class(m["y_true"], m["y_pred"], INTENTS)
        s["acc"] = accuracy(m["y_true"], m["y_pred"])
        s["acc_correct"] = sum(1 for a, b in zip(m["y_true"], m["y_pred"]) if a == b)
        s["acc_n"] = len(m["y_true"])
        s["macro_f1"] = macro_f1(stats)
        s["weighted_f1"] = weighted_f1(stats)
        s["per_class"] = stats
        if m["hard_true"]:
            s["acc_hard"] = accuracy(m["hard_true"], m["hard_pred"])
            s["n_hard"] = len(m["hard_true"])
    if m["esc_true"]:
        e = binary_prf(m["esc_true"], m["esc_pred"])
        s["esc_precision"] = e["precision"]
        s["esc_recall"] = e["recall"]
        s["esc_missed"] = e["fn"]
        s["auto_rate"] = sum(1 for p in m["esc_pred"] if not p) / len(m["esc_pred"])
    s["reply_rate"] = m["replies"] / m["n"] if m["n"] else 0.0
    s["tok"] = (m["in_tok"] + m["out_tok"]) / m["n"] if m["n"] else 0
    s["lat"] = sum(m["latency"]) / len(m["latency"]) if m["latency"] else 0.0
    return s


# ------------------------------------------------------------------ reporting

def report(summaries: list[dict], metrics: list[dict], rows: list[dict],
           split: str) -> None:
    print("\n" + "=" * 100)
    print(f"RESULTS   brand={config.BRAND}   split={split}   n={len(rows)}   "
          f"gen={config.GEN_MODEL}")
    print("=" * 100)
    print(f"{'config':<26}{'intent acc':>13}{'macroF1':>9}{'wtdF1':>7}"
          f"{'escRec':>8}{'escPrec':>9}{'auto':>7}{'reply':>7}{'tok':>7}{'s':>6}")
    print("-" * 100)
    def pct(s: dict, key: str, width: int) -> str:
        """A config that does not make a given kind of decision prints n/a rather than
        a number. B2 classifies but never escalates; showing it as 0% escalation recall
        would read as a failure at something it was never asked to do."""
        v = s.get(key)
        return f"{'n/a':>{width}}" if v is None else f"{v:>{width}.0%}"

    def num(s: dict, key: str, width: int) -> str:
        v = s.get(key)
        return f"{'n/a':>{width}}" if v is None else f"{v:>{width}.2f}"

    for s in summaries:
        if not s["n"]:
            continue
        acc = fmt_ci(s.get("acc_correct", 0), s.get("acc_n", 0)) if "acc" in s else "   n/a"
        print(
            f"{s['config']:<26}{acc:>13}"
            f"{num(s, 'macro_f1', 9)}{num(s, 'weighted_f1', 7)}"
            f"{pct(s, 'esc_recall', 8)}{pct(s, 'esc_precision', 9)}"
            f"{pct(s, 'auto_rate', 7)}"
            f"{s['reply_rate']:>7.0%}{s['tok']:>7.0f}{s['lat']:>6.1f}"
        )
    print("-" * 100)
    bad = [s for s in summaries if s.get("errors")]
    if bad:
        print("  ERRORS -- these items were retried and still failed. Re-run to retry;")
        print("  any number below is computed on the rows that succeeded:")
        for s in bad:
            print(f"    {s['config']:<26}{s['errors']} of {s['n']}")
        print("-" * 100)
    for name in CONFIGS:
        if any(s["config"] == name and s["n"] for s in summaries):
            print(f"  {name:<26}{BLURB[name]}")
    print()
    print("  intent acc  = accuracy with a 95% Wilson interval. Gaps smaller than the")
    print("                interval are noise, not findings.")
    print("  macroF1     = every intent weighted equally. The golden set was topped up")
    print("                on rare intents, so this is measured on a distribution that")
    print("                does not exist in the wild. wtdF1 is closer to what a random")
    print("                customer would experience. Read both.")
    print("  escRec      = of the messages that genuinely needed a human, how many were")
    print("                routed there. Trivially gamed by escalating everything --")
    print("                which is why `auto` is printed right beside it.")
    print("  auto        = share handled without a human. escRec is only meaningful")
    print("                relative to this.")
    print("  reply       = share of messages that produced any reply at all.")

    for s, m in zip(summaries, metrics):
        if not m["y_true"]:
            continue
        if "acc_hard" in s:
            print(f"\n[{s['config']}] hard subset: {s['acc_hard']:.0%} "
                  f"on {s['n_hard']} deliberately awkward messages "
                  f"(vs {s['acc']:.0%} overall)")

    for s, m in zip(summaries, metrics):
        if s["config"] != "system" or not m["y_true"]:
            continue
        print(f"\n[{s['config']}] per-intent")
        print(f"  {'intent':<24}{'prec':>7}{'rec':>7}{'f1':>7}{'n':>5}")
        for c, st in s["per_class"].items():
            if st["support"]:
                print(f"  {c:<24}{st['precision']:>7.2f}{st['recall']:>7.2f}"
                      f"{st['f1']:>7.2f}{st['support']:>5}")
        print(f"\n[{s['config']}] confusion matrix")
        print(format_confusion(m["y_true"], m["y_pred"], INTENTS))

    for s, m in zip(summaries, metrics):
        if m["failures"] and s["config"] == "system":
            print(f"\n[{s['config']}] failures ({len(m['failures'])}) -- read these, "
                  f"they are the failure analysis:")
            for iid, why, text in m["failures"][:25]:
                print(f"  {iid:<14} {why}")
                print(f"                 {text}")
            if len(m["failures"]) > 25:
                print(f"  ... and {len(m['failures']) - 25} more")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", choices=CONFIGS)
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="print cached results only, call nothing")
    args = ap.parse_args()

    rows = load_golden(args.split)
    if not rows:
        raise SystemExit(f"no golden rows in split {args.split!r}")
    if args.limit:
        rows = rows[: args.limit]

    names = args.config or CONFIGS
    summaries, metrics = [], []
    for name in names:
        cache = load_cache(name) if args.report else run_config(name, rows, args.force)
        if args.report and not cache:
            print(f"[{name}] nothing cached")
        m = grade(name, rows, cache)
        metrics.append(m)
        summaries.append(summarise(m))

    report(summaries, metrics, rows, args.split)

    if args.split == "test" and not args.report:
        print("\n  NOTE: that was the frozen test split. Every prompt and threshold")
        print("  change from here on has to be declared in the report, because the")
        print("  moment you tune against these numbers they stop being held out.")


if __name__ == "__main__":
    main()
