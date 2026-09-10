# Report

> **Status: awaiting the first real run.** Structure, framing and method are final;
> every `[TBD]` is a number that gets filled by `python run.py eval`. Nothing here is
> estimated or predicted.

---

## 1. Problem framing: what "good" means for this brand

The task as given is three things — classify, draft, escalate. Only one of them has an
unambiguous notion of correct, and being clear about that shaped everything else.

**Intent** has a right answer once the taxonomy is written, so it is scored against hand
labels with per-class precision/recall/F1.

**Escalation** has a right answer once the policy is written, so it is scored the same
way — but only against rules that read the *customer's message* (E1–E5). Rules that read
*our own program's confidence* (R1–R2) are excluded from ground truth entirely. See §6.

**Reply quality has no right answer**, and pretending otherwise is the main way a project
like this produces a confident number that means nothing. There is no reference reply:
the brand's historical reply is one option among many, and on this dataset it is often
"please DM us". So reply quality is judged by an LLM against four binary criteria, and
that judge is then itself measured against a human. Until §5 has a number in it, every
reply-quality figure in this report is a reading from an uncalibrated instrument.

### What "good" specifically means here

Not resolution rate. A large share of this brand's real replies move the customer to a
private channel, so a system that maximises "resolved in public" would be optimising for
something the brand itself does not do. Good means:

1. the intent is right, including on rare intents, hence macro-F1 alongside accuracy;
2. nothing that needs a human is auto-handled — escalation **recall** on E1–E5;
3. that safety is not bought by escalating everything — hence **auto-handle rate** is
   printed beside recall, always;
4. drafted replies say only what this brand has actually said.

### What I chose not to build

- **No multi-turn handling.** Items are first-inbound messages only. Later turns inherit
  the intent above them and would pad the test set with easy near-duplicates. The cost is
  that nothing about conversation state is measured at all.
- **No vector database.** At 20k rows retrieval is one matrix multiply.
- **No fine-tuning.** B2 (TF-IDF + logistic regression) already answers "is the LLM
  needed"; a fine-tuned encoder would cost a day to make the same point.
- **No Banking77.** 77 intents against a 200-row answer key is 2.6 examples per class.
- **No sentiment-based escalation.** Angry customers with ordinary problems are ordinary
  problems; "escalate if annoyed" is a queue, not a product.
- **No UI, no webhook service.** Neither is evidence.

---

## 2. Data and brand choice

`[TBD]` — table from `data/brand_compare.py`: pairs, substantive replies, rate, median
reply length for each candidate brand.

The trap this measurement exists to avoid: if most of a brand's replies are "please DM
us", then "draft a reply grounded in how the brand resolved this" degrades into "learn to
punt", and the retrieval, the judge and the headline number would all still look healthy.

Chosen brand: `[TBD]`. Substantive-reply rate: `[TBD]`. **That rate is a hard ceiling on
grounded reply quality**, and it is quoted again in §7.

Corpus: `[TBD]` (customer message → brand reply) pairs, subsampled to 20,000 and committed
to the repo. Only pairs with a substantive reply enter the retrieval index (`[TBD]` rows).

---

## 3. The golden set

200 hand-labelled messages. Method, strata and declared biases: [`golden/SAMPLING.md`](golden/SAMPLING.md).

- 140 random at natural prevalence, ~45 topping up rare intents, 15 deliberately awkward.
- 60 dev / 140 test, split assigned before any label existed and hidden while labelling.
- Class distribution: `[TBD]`, against the true distribution: `[TBD]`.

### Intra-annotator agreement — the ceiling on everything below

Blind relabel of 50 rows, 24+ hours later, different order, first answer hidden:

| | raw agreement | Cohen's κ |
|---|---|---|
| intent | `[TBD]` | `[TBD]` |
| escalate | `[TBD]` | `[TBD]` |

Most unstable boundaries: `[TBD]`.

This is the number to read first. A classifier cannot be shown to exceed the consistency
of its own answer key, so any accuracy at or above this level is at the noise floor and
the difference is not measurable with this set.

---

## 4. Results

`[TBD]` — the table from `python run.py report`, all six configs, intent accuracy with
95% Wilson intervals, macro-F1, weighted-F1, escalation recall and precision, auto-handle
rate, tokens and latency per message.

### Reading it

- **vs B0 (trivial)** — the floor. A config near it has learned nothing.
- **vs B2 (TF-IDF + logistic regression)** — what the LLM is worth. If the gap is small,
  the honest conclusion is that a classical classifier at ~3ms and zero cost does most of
  this job.
- **vs B3 (retrieval verbatim)** — what *generation* is worth. B3 sees identical
  evidence and does no writing at all, so the gap is the value added by the model
  composing a reply rather than copying one.
- **vs B1 (canned)** — whether the reply rubric is measuring anything. A single canned
  punt sent to every message can score well on grounded/no-overpromise/tone, because it
  is a real brand reply that promises nothing. If B1 comes close, the finding is about
  the rubric, not about the system.

At n=140 the 95% interval is roughly ±6 points. **Any gap smaller than that is not a
result**, and is not described as one below.

Per-intent breakdown, confusion matrix and hard-subset accuracy: `[TBD]`.

---

## 5. Does the judge agree with a human?

The brief asks for this by name, and it gates §4's reply column.

60 replies graded by hand on the same four binary dimensions, blind to the judge, in
shuffled order.

| dimension | raw agreement | Cohen's κ | judge says yes | human says yes |
|---|---|---|---|---|
| grounded | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| addresses_ask | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| no_overpromise | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| tone_ok | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| **acceptable** | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

2×2 on `acceptable`: `[TBD]`. Direction of error: `[TBD]`.

Both raw agreement and κ are reported because they answer different questions, and κ can
look poor at high raw agreement when one verdict dominates (the kappa paradox). The
per-dimension split matters more than the aggregate: "the judge is unreliable" is not
actionable, "the judge and I disagree specifically about groundedness" names the rubric
line to rewrite.

Three real disagreements, quoted, with my view on who was right: `[TBD]`.

---

## 6. The escalation design, and the one thing it refuses to do

Ground truth for escalation comes only from rules that read the customer's message:
hacked account, disputed charge, legal threat, safety, repeat contact.

Rules that read our own program — classifier confidence below τ, retrieval similarity
below σ — are allowed to *trigger* an escalation at run time but are **never** allowed to
define the correct answer.

If they were, the system could not be wrong. Any time it hesitated, hesitating would be
correct by definition; the eval would print a high number and measure nothing. Instead,
those extra escalations show up as a **drop in escalation precision**, which prices
caution visibly rather than hiding it inside the definition of correct.

τ = `[TBD]`, σ = `[TBD]`, both tuned on dev only.
Escalations by source (E1–E5 vs R1/R2): `[TBD]`.

---

## 7. Failure analysis

`[TBD]` — top 5 failure modes, each with real quoted examples and a hypothesis.

Expected candidates, to be confirmed or dropped against actual failures:

1. **The DM-punt ceiling.** Where the brand's real answer was "DM us", a grounded reply
   is a well-phrased punt. Groundedness scores well; nothing is resolved.
2. **Multi-intent messages.** Two genuine asks, one label. The confusion matrix should
   show this concentrated in specific pairs.
3. **Boundary pairs that flipped in the blind relabel.** Where the human was unstable,
   the model has no stable target to hit — these are answer-key defects, not model errors.
4. **Retrieval finding topically similar but situationally wrong precedents.** Cosine
   similarity does not know that "cancel because I'm moving country" and "cancel because
   you charged me twice" need different replies.
5. **Non-English and near-empty messages.** Over-represented in the hard stratum by design.

---

## 8. What is misleading about my headline number

Mandatory section, and the one I would read first if I were reviewing this.

1. **The golden set's class mix is not the real one.** Rare intents were topped up to ~20
   examples, so macro-F1 is computed on a distribution that does not exist in the wild
   and flatters rare classes. True distribution: `[TBD]`. Weighted-F1 (`[TBD]`) is closer
   to what a random customer experiences.

2. **One person wrote the labels *and* the prompts.** My idea of where `billing_payment`
   ends and the model's came from the same head, so they are correlated in a way a second
   annotator's would not be. My own blind self-agreement was κ=`[TBD]` — that is the
   measurement ceiling, and accuracy at or above it is unverifiable with this answer key.

3. **"Grounded" is graded against replies that are frequently "DM us."** `[TBD]`% of this
   brand's replies are non-substantive. A high groundedness score therefore partly
   measures how well the agent learned to punt to a human, which is not resolution.

4. **n=140 means ±6 points.** Every comparison in §4 smaller than its confidence interval
   is noise. I have not described any such gap as an improvement.

5. **An LLM wrote the replies and an LLM graded them.** The judge agrees with a human at
   κ=`[TBD]`, and errs in the `[TBD]` direction — so the reply-quality figure is a
   `[TBD]` (upper/lower) bound, not a point estimate.

6. **Escalation recall is trivially gameable.** Escalating everything scores 100% and is
   useless, which is why `trivial_always_escalate` is a row in the table and auto-handle
   rate sits beside recall everywhere.

7. **Rows were surfaced for labelling by machine weak-labels.** Whatever kind of message
   the weak labeller systematically misses was never offered for hand-labelling, so this
   answer key has a blind spot whose size I cannot measure from inside it.

8. **Test-set exposure.** Tuning happened on dev; test was run `[TBD]` time(s). `[TBD]`

---

## 9. What I would do next with one more week

1. **A second annotator on 100 rows.** Intra-annotator κ bounds the answer key, but only
   inter-annotator κ shows whether the taxonomy is communicable to anyone else. That is
   the single highest-value missing measurement.
2. **Calibrate confidence.** The classifier's self-reported confidence gates R1 and is
   currently trusted as a threshold input without evidence. Bucket predictions by
   confidence and plot actual accuracy per bucket; the expectation is it claims ~95%
   while being right ~75%, which would mean τ rests on a number the model invented.
   Replace it with agreement across 3 samples and re-tune.
3. **Rewrite the rubric line the per-dimension κ identifies as weakest, and re-measure.**
   Judge validation is only useful if it changes the judge.
4. **Multi-turn.** The biggest scoped-out piece. Escalation rule E5 (repeat contact) is
   currently the only thread-aware rule and cannot fire on single-message items.
5. **Retrieval quality as its own metric.** Every reply-quality number currently confounds
   retrieval and generation. Hand-label whether the top-5 evidence was *usable* for 100
   items and report retrieval precision separately.
