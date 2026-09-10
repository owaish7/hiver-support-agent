"""One entry point, so the README is short and the order is not guesswork.

    python run.py report     offline. No API key, no dataset. Reproduces the table.
    python run.py setup      one-time: threads -> brand choice -> sample -> index
    python run.py labels     sampling + the interactive labelling session
    python run.py eval       run every config, then the judge
    python run.py check      the offline test suite + answer-key validation

`report` is deliberately first and deliberately free. A reviewer with fifteen minutes
and no API key should be able to see every headline number before deciding whether to
install anything.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES: dict[str, list[list[str]]] = {
    "setup": [
        ["data/build_threads.py"],
        ["data/brand_compare.py"],
        ["data/make_sample.py"],
        ["data/build_index.py"],
    ],
    "labels": [
        ["data/weak_label.py"],
        ["golden/sample_golden.py"],
        ["golden/label.py"],
        ["taxonomy/escalation.py", "--validate", "golden/golden.jsonl"],
    ],
    "eval": [
        ["eval/run_eval.py", "--split", "test"],
        ["eval/judge.py"],
    ],
    "report": [
        ["eval/run_eval.py", "--report", "--split", "test"],
    ],
    "check": [
        ["tests/test_offline.py"],
        ["eval/stats.py", "--check"],
        ["eval/metrics.py", "--check"],
        ["taxonomy/escalation.py", "--validate", "golden/golden.jsonl"],
    ],
}


def run(stage: str) -> int:
    if stage not in STAGES:
        print(f"unknown stage {stage!r}. one of: {', '.join(STAGES)}")
        return 2
    for cmd in STAGES[stage]:
        printable = " ".join(["python", *cmd])
        # flush: without it these headers are buffered while the child's output goes
        # straight to the terminal, so every banner appears after the stage it labels.
        print(f"\n{'=' * 78}\n$ {printable}\n{'=' * 78}", flush=True)
        code = subprocess.call([sys.executable, *cmd], cwd=ROOT)
        if code != 0:
            # escalation --validate exits non-zero when labels need review. That is
            # information, not a broken pipeline, so it does not stop the run.
            if cmd[0].endswith("escalation.py"):
                print(f"\n  ({printable} reported unresolved labels -- continuing)")
                continue
            print(f"\n  stopped: {printable} exited {code}")
            return code
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else "report"))
