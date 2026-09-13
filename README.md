# AI support agent for @SpotifyCares

Real customer-support conversations from Twitter. For every inbound message the system
classifies the intent, drafts a reply grounded in how the brand actually resolved similar
issues, and decides whether a human should take it instead — with a stated reason.

The system is the small half of this repo. The evidence is the large half.

---

## Results

Test split n=90, hand-labelled. Generator `openai/gpt-oss-20b`, judge `qwen/qwen3.8-27b`.

| config | intent acc | macro‑F1 | esc recall | esc prec | auto‑handled | replies |
|---|---|---|---|---|---|---|
| B0 trivial (always auto) | 29% ±9 | 0.04 | 0% | 0% | 100% | 0% |
| B0 trivial (always escalate) | 29% ±9 | 0.04 | 100% | 8% | 0% | 0% |
| B1 canned reply | 29% ±9 | 0.04 | 0% | 0% | 100% | 100% |
| B2 TF‑IDF + logistic regression | 46% ±10 | 0.32 | n/a | n/a | n/a | 0% |
| B3 retrieval verbatim | 29% ±9 | 0.04 | n/a | n/a | n/a | 100% |
| **system** | **76% ±9** | **0.77** | **43%** | **60%** | **94%** | **100%** |

Intervals are 95% Wilson. At n=90 that is ±9 points, so the 30-point gap over B2 is a
result and anything under ~18 points between two configs is not.

### Read these two before the headline

| | value | what it bounds |
|---|---|---|
| **Intra-annotator κ** (intent) | **+0.838** | the answer key is ~86% self-consistent. 76% sits *below* that ceiling, so the remaining gap is real rather than answer-key noise |
| **Judge-vs-human κ** (acceptable) | **+0.599** | the judge is **too generous** — it passes 5 replies a human rejected. Reply quality is an upper bound: judge 86%, human 78% |

The first is 50 rows relabelled blind, 20 hours later, first answer hidden. The second is
60 replies rated by hand against the same rubric the LLM judge uses.

### The honest weakness

**Escalation recall is 43%** — 3 of 7 messages needing a human were routed there. Every
miss has one cause: the escalation rules are English keyword patterns that under-fire on
real phrasing. *"my email was changed and it wasn't me"* is an account takeover that got
auto-handled; a Dutch security complaint the rules cannot read at all.

Full analysis: [`REPORT.md`](REPORT.md) §7.

---

## Reproduce it

**No API key, no dataset download.** Results are cached per item:

```bash
pip install -r requirements.txt
python run.py report     # the full table, offline, ~2 seconds
python run.py check      # 24 tests + every statistic verified against sklearn/statsmodels
```

Running it live needs `GROQ_API_KEY` in `.env` and `data/twcs.csv` from
[Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter):

```bash
python run.py setup      # threads -> brand choice -> 20k sample -> embedding index
python run.py eval
```

---

## How it works

```
message -> classify -> retrieve -> draft -> escalate
```

- **classify** — structured output, 11 intents derived from reading 100 real messages
- **retrieve** — top-4 similar past cases, local MiniLM embeddings, NumPy matrix multiply
- **draft** — a reply using only those 4 cases as evidence
- **escalate** — content rules first, runtime signals second, reason always stated

**The generator and the judge are different model families** (`gpt-oss` writes, `qwen`
grades). Models score their own family's output more generously, so self-grading would
inflate the headline by an amount nobody can measure.

**Embeddings are local.** No rate limits, identical results every run, and the index
rebuilds with no API key — which is what makes the offline reproduction work.

---

## The escalation rule that matters

| | source | may decide the correct answer? |
|---|---|---|
| **E1–E5** | the customer's message — hacked account, disputed charge, legal threat, safety, repeat contact | **yes** |
| **R1–R2** | our program — classifier confidence below τ, retrieval similarity below σ | **no** |

If the model's own confidence could define correctness, the system could never be wrong:
hesitating would be correct by definition, and the eval would print a confident number
while measuring nothing. R1/R2 still trigger escalations at run time, and those extras
land as a drop in escalation **precision** — pricing caution visibly instead of hiding it.

The policy is written twice: as prose in [`ESCALATION.md`](taxonomy/ESCALATION.md) which
goes into the prompt, and as code in [`escalation.py`](taxonomy/escalation.py) which checks
the hand labels. The code is explicitly **not** an oracle — no keyword list can decide
whether "see you in court" is a real legal threat, so the hand labels stay ground truth.

---

## Four leakage guards, all tested

Each would have inflated every number while leaving the code looking correct.

1. **Retrieval excludes the query's own row.** Every golden item lives in the index it
   retrieves from — without this, its nearest neighbour is itself at similarity 1.0 and
   the drafter gets the exact reply it is about to be graded against.
2. **B2 trains on dev only and never reads test.**
3. **The trivial baseline takes its majority class from dev, not test.** A baseline that
   peeks at the answer key is not a floor.
4. **The judge's evidence is rebuilt from ids**, so it cannot drift from what the drafter
   saw.

All asserted in [`tests/test_offline.py`](tests/test_offline.py).

---

## Layout

```
run.py         report | setup | labels | eval | check
config.py      every threshold, model id and seed
llm.py         both providers behind one call, structured output only
data/          thread reconstruction, brand choice, 20k sample, embedding index
taxonomy/      11 intents + escalation policy, as prose and as code
golden/        150 hand labels, the labelling tool, the blind relabel
agent/         classify, retrieve, draft, escalate, pipeline
baselines/     trivial, canned, tfidf_lr, retrieval_verbatim
eval/          harness, judge, human judge, agreement, metrics, stats
tests/         24 offline tests, no key required
```

**Full reasoning, failure analysis, and the mandatory "what is misleading about my
headline number" section: [`REPORT.md`](REPORT.md).**
Every non-obvious decision: [`DECISIONS.md`](DECISIONS.md).
How the golden set was built: [`golden/SAMPLING.md`](golden/SAMPLING.md).

---

<sub>`data/brand_sample.csv` is a 20,000-row subsample of
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
(CC-BY-NC-SA-4.0), redistributed under the same licence for non-commercial evaluation.
Banking77 is PolyAI's, CC-BY-4.0 ([Casanueva et al. 2020](https://arxiv.org/abs/2003.04807)),
downloaded on demand rather than redistributed.</sub>
