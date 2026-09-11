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

# Any reference to a private channel means the reply is not a self-contained public
# resolution, however it is phrased.
#
# This started as a list of substrings ("dm us", "send us a dm", ...) and that was
# measurably wrong. Sampling real replies found dozens of phrasings it missed -- "let's
# hop into DM", "follow/DM your confirmation number", "us in DM", "info over DM" -- and
# the miss rate was not equal across brands: 6.8% of Spotify's "substantive" replies were
# really punts against 26.5% of Apple's. That inflated Apple's score by 15 points, enough
# to move it from second place to third. A crude rule applied unevenly is worse than a
# crude rule applied evenly, so this matches the channel reference itself rather than
# trying to enumerate the ways of asking.
PUNT_PATTERN = (
    r"\bD\.?M\.?s?\b|direct message|private message|privately|"
    r"\be-?mail us\b|\bcall us\b|\bgive us a call\b|\bphone us\b|"
    r"\blive chat\b|\bchat with us\b|\bcontact us at\b"
)

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
