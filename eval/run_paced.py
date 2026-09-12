"""Run an eval config to completion against a rolling token budget.

The free tier allows 200,000 tokens per rolling 24 hours, and a full test run costs more
than that. The limit is not a calendar-day reset -- tokens age out continuously -- so the
practical approach is to keep asking, back off when refused, and let the cache accumulate.

This exists because the alternative was worse. Shrinking the test split to whatever fits
in one window would mean the eval size was chosen by a billing limit rather than by what
makes the numbers meaningful, and the confidence intervals would widen accordingly.

run_eval.py already retries cached errors on each invocation, so this is just a loop with
a sleep and a stopping condition.

    python eval/run_paced.py --config system --split test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

config.use_utf8_stdout()

RESULTS = ROOT / "eval" / "results"


def state(name: str, wanted: set[str]) -> tuple[int, int]:
    """(done, errored) among the rows we actually care about."""
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return 0, 0
    cache = json.loads(path.read_text(encoding="utf-8"))
    rows = {k: v for k, v in cache.items() if k in wanted}
    done = sum(1 for v in rows.values() if not v.get("error"))
    err = sum(1 for v in rows.values() if v.get("error"))
    return done, err


def wanted_ids(split: str) -> set[str]:
    rows = [json.loads(l) for l in
            config.GOLDEN_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["id"] for r in rows if split == "all" or r.get("split") == split}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="system")
    ap.add_argument("--split", default="test")
    ap.add_argument("--wait", type=int, default=420,
                    help="seconds to wait after a pass that made no progress")
    ap.add_argument("--max-hours", type=float, default=8.0)
    args = ap.parse_args()

    wanted = wanted_ids(args.split)
    target = len(wanted)
    deadline = time.time() + args.max_hours * 3600
    last_done = -1
    stalls = 0

    while time.time() < deadline:
        done, err = state(args.config, wanted)
        print(f"\n[paced] {done}/{target} done, {err} errored", flush=True)
        if done >= target:
            print("[paced] complete", flush=True)
            return

        subprocess.call([sys.executable, "eval/run_eval.py",
                         "--split", args.split, "--config", args.config],
                        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        done_after, _ = state(args.config, wanted)
        gained = done_after - done
        print(f"[paced] +{gained} this pass ({done_after}/{target})", flush=True)

        if done_after >= target:
            print("[paced] complete", flush=True)
            return

        # No progress twice running means the window is genuinely empty rather than
        # briefly tight, so wait longer instead of hammering a closed door.
        stalls = stalls + 1 if done_after == last_done else 0
        last_done = done_after
        wait = args.wait * (2 if stalls >= 2 else 1)
        print(f"[paced] sleeping {wait}s", flush=True)
        time.sleep(wait)

    print("[paced] hit --max-hours; results so far are cached and usable", flush=True)


if __name__ == "__main__":
    main()
