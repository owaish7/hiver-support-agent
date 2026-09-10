"""Every tunable constant in the project lives here.

Deliberate: when someone asks "change the escalation threshold and re-run", the answer
is one file and one line, not a grep across the codebase.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- data

RAW_CSV = ROOT / "data" / "twcs.csv"          # the 500MB Kaggle download, gitignored
BRAND: str = "SpotifyCares"                    # set by data/brand_compare.py on day 1
SAMPLE_CSV = ROOT / "data" / "brand_sample.csv"  # ~20k rows, COMMITTED
INDEX_NPZ = ROOT / "data" / "index.npz"        # rebuilt in ~60s, gitignored

# Brands compared in data/brand_compare.py before committing to one.
CANDIDATE_BRANDS = ["SpotifyCares", "AppleSupport", "Delta"]

# A brand reply is "substantive" if it actually tells the customer something rather
# than punting to a private channel. This threshold decides the brand choice, so it
# is a decision, not a detail.
SUBSTANTIVE_MIN_WORDS = 15
PUNT_MARKERS = ["dm us", "dm me", "direct message", "send us a dm", "shoot us a dm",
                "in a dm", "via dm", "pm us", "send us a private"]

# ---------------------------------------------------------------- retrieval

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local, CPU, no API key
TOP_K = 5

# ---------------------------------------------------------------- thresholds
# BOTH tuned on the dev split ONLY (golden/dev). The test split is touched once.

TAU = 0.55    # intent-classifier confidence below this -> escalate(low_confidence)
SIGMA = 0.45  # best retrieval cosine below this -> escalate(no_precedent)

# ---------------------------------------------------------------- models

GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------- golden set

GOLDEN_JSONL = ROOT / "golden" / "golden.jsonl"
N_GOLDEN = 200
N_RANDOM = 140        # drawn at natural prevalence
N_PER_RARE_CLASS = 20  # rare intents topped up to this - THIS IS THE SAMPLING BIAS
N_HARD = 15           # deliberately nasty cases, tagged is_hard
DEV_SIZE = 60         # tuning happens here
# test = the remaining 140. Frozen.

SEED = 20260910       # every random draw in the project uses this


def use_utf8_stdout() -> None:
    """Windows consoles default to cp1252, which raises UnicodeEncodeError on the
    first emoji or non-Latin character. Real support tweets are full of both, and the
    hard stratum deliberately over-samples them -- so without this the labelling tool
    dies partway through a session on exactly the rows that matter most.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
