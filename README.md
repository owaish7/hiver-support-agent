# AI support agent for a single brand, and the evidence it works

> **Data attribution.** `data/brand_sample.csv` is a 20,000-row derived subsample of
> [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
> (thoughtvector, Kaggle), licensed **CC-BY-NC-SA-4.0** and redistributed here under the
> same licence for non-commercial evaluation. Banking77 is PolyAI's, CC-BY-4.0
> ([Casanueva et al. 2020](https://arxiv.org/abs/2003.04807)), downloaded on demand
> rather than redistributed.


Real customer-support conversations from Twitter ([Kaggle: thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)).
One brand. For every inbound message the system classifies the intent, drafts a reply
grounded in how that brand has actually resolved similar issues, and decides whether a
human should take it instead — with a stated reason.

The system is the small half of this repo. The evidence is the large half.

---

## Results

| config | intent acc | macro‑F1 | wtd‑F1 | esc recall | esc prec | auto‑handled | replies |
|---|---|---|---|---|---|---|---|
| B0 trivial (always auto) | 29% ±9 | 0.04 | 0.13 | 0% | 0% | 100% | 0% |
| B0 trivial (always escalate) | 29% ±9 | 0.04 | 0.13 | 100% | 8% | 0% | 0% |
| B1 canned reply | 29% ±9 | 0.04 | 0.13 | 0% | 0% | 100% | 100% |
| B2 TF‑IDF + logistic regression | 46% ±10 | 0.32 | 0.43 | n/a | n/a | n/a | 0% |
| B3 retrieval verbatim | 29% ±9 | 0.04 | 0.13 | n/a | n/a | n/a | 100% |
| **system** | **76% ±9** | **0.77** | **0.76** | **43%** | **60%** | **94%** | **100%** |

Test split n=90, hand-labelled. Generator `openai/gpt-oss-20b`, judge `qwen/qwen3.8-27b`.

**Read these two numbers before the headline:**

| | value | what it bounds |
|---|---|---|
| **Intra-annotator κ** (intent) | **+0.838** | the answer key is ~86% self-consistent, so 76% sits below the ceiling and the remaining gap is real |
| **Judge-vs-human κ** (acceptable) | **+0.599** | the judge is *too generous* — it passes 5 replies a human rejected and rejects 1 it should have passed. Reply quality is an upper bound: judge 86%, human 78% |

**The honest weakness: escalation recall is 43%.** It caught 3 of 7 messages that needed a
human. Every miss has one cause — the escalation rules are English keyword patterns that
under-fire on real phrasing, including *"my email was changed and it wasn't me"* (an
account takeover, auto-handled) and a Dutch security complaint the rules cannot read at
all. Full analysis in [`REPORT.md`](REPORT.md) §7.

Intent accuracy carries a 95% Wilson interval. At n=90 that is roughly ±9 points, so
**a gap smaller than the interval is not a result**. The 30-point gap over B2 survives it
easily; nothing smaller than ~18 points between two configs here is a finding.

### External check — against labels I did not write

| approach | accuracy | macro-F1 | n | classes |
|---|---|---|---|---|
| TF-IDF + logistic regression on **Banking77** | **87% ±1** | 0.874 | 3,080 | 77 |
| fine-tuned ModernBERT (published reference) | 94% | 0.940 | 3,080 | 77 |

The same classical method scores 46% on my 60 training rows and 87% on Banking77's 10,003.
That is evidence the 46% is a **data-budget artefact, not a method failure** — which
matters, because it is the number the LLM is being compared against.

[Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) is 3,080 queries across
77 intents labelled by PolyAI. It is **not** used to define this project's taxonomy — 77
banking intents have nothing to do with a music brand. It is used to show the method
survives contact with ground truth I did not write, which is the strongest objection
anyone can raise against a single-annotator project. Runs free and offline:
`python eval/external_check.py --tfidf`.

Its most-confused pairs (`why_verify_identity → verify_my_identity`, `card_arrival →
card_delivery_estimate`) are outside evidence for the argument this project makes about
its own categories: **boundaries are the hard part, not the class list.**

---

## Reproduce the headline in under 15 minutes

**No API key, no Kaggle download, no model weights.** Results are cached per item, so the
whole table renders from disk:

```bash
pip install -r requirements.txt
python run.py report
```

Prove the machinery rather than taking it on trust — also offline, also seconds:

```bash
python run.py check
```

That runs 24 tests including the four leakage guards described below, and verifies every
statistic in the report against `sklearn` and `statsmodels`.

### Running it live

```bash
cp .env.example .env          # GROQ_API_KEY (both generator and judge)
python run.py setup           # threads -> brand choice -> 20k sample -> embedding index
python eval/run_eval.py --config system --limit 5    # smoke test before spending quota
python run.py eval
```

`setup` needs `data/twcs.csv` from Kaggle. Everything else works from the committed
`data/brand_sample.csv`.

---

## How it works

```
message
   |
   +-- classify        structured output, 11 intents derived from reading 100 messages
   |
   +-- retrieve        top-4 similar past cases, local MiniLM embeddings + numpy
   |
   +-- draft           reply using ONLY those 4 cases as evidence
   |
   +-- escalate        content rules first, runtime signals second, reason always stated
```

**Retrieval is a NumPy array and one `@`.** At 11,728 indexed rows a vector database
would add a dependency, a build step and a failure mode in exchange for nothing
measurable. Knowing when not to reach for infrastructure is part of the answer.

**Embeddings are local** (`all-MiniLM-L6-v2`, CPU). No rate limits during an eval run,
identical results every time, and the index rebuilds with no API key — which is what
makes the 15-minute promise hold.

**The generator and the judge are different model families.** `gpt-oss-20b` writes,
`qwen3.8-27b` grades. Models score their own family's output more generously, so
self-grading would inflate the headline by an amount nobody can measure. What matters is
the separate training lineage, not the host — both happen to run on Groq, chosen after the
Gemini free tier turned out to allow 20 requests per day.

---

## The escalation rule that matters

The policy is written twice: as prose in [`taxonomy/ESCALATION.md`](taxonomy/ESCALATION.md)
which goes into the prompt, and as code in [`taxonomy/escalation.py`](taxonomy/escalation.py)
which checks the hand labels for consistency.

| | source | may decide the correct answer? |
|---|---|---|
| **E1–E5** | the customer's message — hacked account, disputed charge, legal threat, safety, repeat contact | **yes** |
| **R1–R2** | our program — classifier confidence below τ, retrieval similarity below σ | **no** |

If the model's own confidence were allowed into the answer key, the system could never
be wrong: hesitating would be correct by definition, and the eval would print a high
number while measuring nothing. R1/R2 still trigger escalations at run time, and those
extras land as a **drop in escalation precision** — which is the honest way to price
caution rather than hiding it inside the definition of correct.

The code is explicitly **not** an oracle. No keyword list can decide whether "see you in
court" is a real legal threat, so the hand labels stay the ground truth and the code only
reports disagreements for review.

---

## Four leakage guards, all tested

Each of these would have inflated every number in the report while leaving the code
looking correct.

1. **Retrieval excludes the query's own row.** Every golden item was drawn from the same
   corpus the index is built from. Without `exclude_id`, the nearest neighbour of a
   golden item is itself at similarity 1.0, and the drafter is handed the exact reply it
   is about to be graded against.
2. **B2 trains on dev only and never reads test.** The two splits were fixed before any
   label existed; the test fixture asserts a test-split label cannot reach training.
3. **The trivial baseline reads its majority class from dev, not test.** A "trivial"
   baseline that peeks at the answer key's own class distribution is not a floor, it is a
   lie about how hard the task is.
4. **The judge's evidence is rebuilt from ids, not cached alongside the reply**, so it
   cannot drift from what the drafter actually saw.

All four are asserted in [`tests/test_offline.py`](tests/test_offline.py).

---

## The golden set

200 hand-labelled messages: 140 random at natural prevalence, ~45 topping up rare
intents, 15 deliberately awkward (multi-intent, sarcasm, non-English, emoji-only,
prompt-injection-shaped) selected by pattern and spread across failure kinds rather than
taken from the top of one list.

Split 60 dev / 140 test, assigned **before any label exists** and hidden by the labelling
tool. Tuning happens on dev; test is touched once.

The labelling tool also hides the machine's guess (anchoring would contaminate every
later human-vs-model comparison) and the brand's actual reply (the classifier never sees
it, so labelling with it builds a ceiling the system cannot reach). Revealing the reply
is one keypress and auto-tags the row, so the report can say how many labels needed it.

Sampling and its declared biases: [`golden/SAMPLING.md`](golden/SAMPLING.md).

---

## Layout

```
run.py              one entry point: report | setup | labels | eval | check
config.py           every threshold, model id and seed. Nothing tunable lives elsewhere
llm.py              both providers behind one call, structured output only
data/               thread reconstruction, brand choice, 20k sample, embedding index
taxonomy/           intents + escalation policy, as prose and as code
golden/             the answer key, the labelling tool, the blind relabel
agent/              classify, retrieve, draft, escalate, pipeline
baselines/          trivial, canned, tfidf_lr, retrieval_verbatim
eval/               harness, judge, human judge, agreement, metrics, stats
tests/              19 offline tests, no key required
```

## Not here, on purpose

No conversation memory — every message is cold, and multi-turn threads are excluded from
the item set on purpose (D2). No UI. No vector database. No fine-tuning. The intent
taxonomy is 8 classes rather than 20 because a class with 3 examples in a 200-row answer
key produces numbers with a standard error near 30 points, and an unfalsifiable per-class
claim is worse than no claim.

Full reasoning, results, failure analysis and the mandatory "what is misleading about my
headline number" section: [`REPORT.md`](REPORT.md).
Every non-obvious decision: [`DECISIONS.md`](DECISIONS.md).
