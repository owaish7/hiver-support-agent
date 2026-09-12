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
TOP_K = 4
# Each evidence pair is truncated before it reaches the prompt. Whole tweets are mostly
# greeting and sign-off ("Hey there! ... /KM"); the instruction is in the middle. Cutting
# them costs little signal and is measured, not assumed -- see the token budget below.
EVIDENCE_CHARS = 220

# ---------------------------------------------------------------- thresholds
# BOTH tuned on the dev split ONLY (golden/dev). The test split is touched once.

TAU = 0.55    # intent-classifier confidence below this -> escalate(low_confidence)
SIGMA = 0.45  # best retrieval cosine below this -> escalate(no_precedent)

# ---------------------------------------------------------------- models

# The generator and the judge are DIFFERENT MODEL FAMILIES on purpose. Models score
# their own family's output more generously (self-preference bias), so self-grading would
# inflate the headline reply-quality number by an amount nobody can measure. gpt-oss and
# Qwen have separate training lineages, which is what the argument actually rests on --
# not that they run on different hosts.
#
# Both are on Groq rather than Gemini for a measured reason: the Gemini free tier turned
# out to allow 20 requests PER DAY for gemini-2.5-flash, and this eval needs roughly a
# thousand. Groq's free tier allows 1,000 per day per model and answers in ~0.5s rather
# than ~2.4s. The Gemini path in llm.py still works and is one line away if a paid key
# ever appears; it is simply unusable at 20/day.
GEN_PROVIDER = os.getenv("GEN_PROVIDER", "groq")
GEN_MODEL = os.getenv("GEN_MODEL", "openai/gpt-oss-20b")

JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "groq")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen3.8-27b")

# Free-tier budget. The binding limit is NOT requests, which was the first guess -- it is
# TOKENS PER DAY. Measured from an actual 429:
#
#   openai/gpt-oss-20b   1,000 requests/day   200,000 tokens/day   8,000 tokens/minute
#
# Measured cost per item:
#   classify   ~940 tokens   (848 of that is the taxonomy + boundary rules, fixed)
#   draft    ~1,140 tokens   (291 system + ~365 evidence + output)
#   ---------------------
#   ~2,080 tokens per item with the escalation second-opinion call disabled
#   ~3,010 tokens per item with it enabled -- which is why it now defaults off
#
# So a 140-item test run costs ~291k tokens and does NOT fit in one day. It is meant to
# be run on a different day from dev tuning, which is the intended workflow anyway: tune
# on dev, then touch test once. Stated here so the schedule is a decision, not a surprise.
WEAK_LABEL_N = 400

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
