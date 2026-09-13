# AI support agent for @SpotifyCares — results and what they are worth

Dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (CC-BY-NC-SA-4.0).
Generator `openai/gpt-oss-20b`, judge `qwen/qwen3.8-27b`, embeddings `all-MiniLM-L6-v2` (local).
Test split n=90, hand-labelled. Every number below is measured; none is estimated.

---

## 1. Problem framing: what "good" means here

The task is three things — classify, draft, escalate — and only two of them have an
unambiguous notion of correct.

**Intent** has a right answer once the taxonomy is written, so it is scored against hand
labels. **Escalation** has a right answer once the policy is written — but only from rules
that read the *customer's message*, never from our own model's confidence (§6).

**Reply quality has no right answer.** There is no reference reply: the brand's historical
reply is one option among many, and 41% of this brand's replies are "please DM us". So
reply quality is judged by an LLM against four binary criteria, and that judge is then
measured against a human (§5). Until §5 has a number, reply figures are readings from an
uncalibrated instrument.

### What "good" specifically means

Not resolution rate. 41% of this brand's real replies move the customer to a private
channel, so optimising "resolved in public" would optimise for something the brand does
not do. Good means: the intent is right including on rare intents (hence macro-F1); nothing
needing a human is auto-handled (escalation **recall**); that safety is not bought by
escalating everything (hence **auto-handle rate** printed beside it); and drafted replies
say only what this brand has actually said.

### What I chose not to build

- **No multi-turn handling.** Items are first-inbound messages only; later turns inherit
  the intent above them. Cost: nothing about conversation state is measured.
- **No vector database.** At 11,728 indexed rows retrieval is one matrix multiply.
- **No fine-tuning.** B2 already answers "is the LLM needed".
- **No Banking77 taxonomy** — but it is used as an external check on the method (§4b).
- **No sentiment-based escalation.** Angry customers with ordinary problems are ordinary
  problems; "escalate if annoyed" is a queue, not a product.
- **No UI, no webhook service.** Neither is evidence.

---

## 2. Brand choice, decided by measurement

The trap: many support accounts reply "please DM us" to almost everything. Build on one of
those and "grounded in how the brand resolved this" degrades into "learn to punt", while
retrieval, the judge and the headline all still look healthy.

| brand | pairs | substantive | rate | median reply words |
|---|---|---|---|---|
| **SpotifyCares** | 26,068 | 15,310 | **59%** | 22 |
| Delta | 24,603 | 10,987 | 45% | 17 |
| AppleSupport | 74,613 | 30,406 | 41% | 22 |

Apple has 2.9× the volume and the lowest substantive rate. **59% is a hard ceiling on
grounded reply quality** and is quoted again in §8.

This table changed once during the project, which is the reason it exists. The first
version of the punt detector was a substring list and it missed phrasings like *"let's hop
into DM"* and *"follow/DM your confirmation number"* — **unevenly**: 6.8% of Spotify's
"substantive" replies were really punts against 26.5% of Apple's. Apple was inflated by 15
points and sat in second place. Sensitivity check: the ranking holds for word thresholds
of 12 and above; at 8 words Delta overtakes Spotify, so the choice is **not**
threshold-invariant and this report does not claim it is.

---

## 3. The golden set and its ceiling

150 hand-labelled messages: 140 random at natural prevalence, 45 topping up rare intents,
15 deliberately awkward. Split 60 dev / 90 test, **fixed before any label existed** and
hidden from the labelling tool. Method and declared biases: [`golden/SAMPLING.md`](golden/SAMPLING.md).

The taxonomy is 11 classes derived from reading 100 real messages. Three of them
(`account_security`, `how_to`, `dm_followup`) were absent from the first draft written
from a skim — that draft's failure is itself the argument for reading data first.

### Intra-annotator agreement — read this before any other number

50 rows relabelled blind, 20+ hours later, shuffled, first answer hidden:

| | raw agreement | Cohen's κ | reading |
|---|---|---|---|
| intent | 86.0% | **+0.838** | almost perfect |
| escalate | 96.0% | +0.728 | substantial |

**The answer key is ~86% self-consistent.** A classifier scoring near 86% is at the noise
floor of its own ground truth and the difference is not measurable with this set. The
system scores 76%, which sits *below* that ceiling — so the gap to perfect is real and
measurable rather than an artefact.

The 7 disagreements were not noise. Four are the same pair:

```
4x  content_unavailable <-> product_feedback
1x  app_bug <-> product_feedback
1x  dm_followup <-> other
1x  how_to <-> plan_or_family
```

All four of the first pair are **catalogue metadata errors** — *"they tagged the wrong
Logic"*, *"the song is named incorrectly"*, *"this album should be placed here"*. The
content exists and is mislabelled, which is neither "unavailable" nor "feedback". That is
a missing twelfth class the blind relabel discovered, and it is the top item in §9.

---

## 4. Results

Test split, n=90. Intent accuracy carries a 95% Wilson interval.

| config | intent acc | macro‑F1 | wtd‑F1 | esc recall | esc prec | auto‑handled | replies | tok/msg | s/msg |
|---|---|---|---|---|---|---|---|---|---|
| B0 trivial (always auto) | 29% ±9 | 0.04 | 0.13 | 0% | 0% | 100% | 0% | 0 | 0.0 |
| B0 trivial (always escalate) | 29% ±9 | 0.04 | 0.13 | 100% | 8% | 0% | 0% | 0 | 0.0 |
| B1 canned reply | 29% ±9 | 0.04 | 0.13 | 0% | 0% | 100% | 100% | 0 | 0.0 |
| B2 TF‑IDF + logistic regression | 46% ±10 | 0.32 | 0.43 | n/a | n/a | n/a | 0% | 0 | 0.0 |
| B3 retrieval verbatim | 29% ±9 | 0.04 | 0.13 | n/a | n/a | n/a | 100% | 0 | 0.3 |
| **system** | **76% ±9** | **0.77** | **0.76** | **43%** | **60%** | **94%** | **100%** | 3,464 | 33.9 |

**The intent result is real.** 76% vs 46% (B2) and vs 29% (B0) are gaps far wider than the
±9 interval. The LLM is worth roughly 30 points over a classical classifier here.

**B2 is handicapped and the gap is overstated because of it.** It trains on the 60 dev
rows — about 5 examples per class across 11 classes — because the LLM weak labels were
generated against the earlier taxonomy and regenerating 400 would have cost ~376k tokens
against a 200k/day cap. A fairer budget would narrow this gap; how much is unmeasured.

**B1 and B3 tie with B0 on intent** because neither classifies — they inherit the majority
class by construction. Their purpose is the reply column, judged in §5.

### Per-intent (system)

| intent | precision | recall | F1 | n |
|---|---|---|---|---|
| account_security | 1.00 | 1.00 | 1.00 | 4 |
| content_unavailable | 0.89 | 0.80 | 0.84 | 10 |
| product_feedback | 0.95 | 0.73 | 0.83 | 26 |
| playback_issue | 0.70 | 1.00 | 0.82 | 7 |
| billing_charge | 1.00 | 0.67 | 0.80 | 6 |
| plan_or_family | 0.80 | 0.80 | 0.80 | 5 |
| other | 0.75 | 1.00 | 0.86 | 3 |
| dm_followup | 1.00 | 0.50 | 0.67 | 2 |
| account_access | 0.50 | 0.80 | 0.62 | 5 |
| app_bug | 0.64 | 0.58 | 0.61 | 12 |
| how_to | 0.50 | 0.70 | 0.58 | 10 |

`account_security` — a class the first taxonomy did not have — scores perfectly on 4
examples. Four examples is too few to claim much, and the interval on it is enormous.

`app_bug` and `how_to` are the weak classes, and the confusion matrix says where: app_bug
leaks into playback_issue (3) and account_access (2); product_feedback leaks into how_to
(4). These are the same boundaries the blind relabel found unstable in a human.

---

## 4b. External check: does this work against labels I did not write?

Every number above is graded against labels I wrote, using prompts I also wrote. Banking77
([PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets), CC-BY-4.0;
[Casanueva et al. 2020](https://arxiv.org/abs/2003.04807)) is 3,080 test queries across 77
intents labelled by someone else.

| approach | accuracy | macro‑F1 | n | classes |
|---|---|---|---|---|
| TF‑IDF + logistic regression | **87% ±1** | 0.874 | 3,080 | 77 |
| fine-tuned ModernBERT (published) | 94% | 0.940 | 3,080 | 77 |

The same classical method that scores 46% on my 60 training rows scores 87% on Banking77's
10,003. That is direct evidence the 46% is a **data-budget artefact rather than a method
failure**, which matters because it is the number the LLM is being compared against.

This does **not** show robustness to noise: Banking77 queries are clean and well-formed;
tweets are not. And the published row is a fine-tuned encoder — a reference point, not a
competition.

**One finding transfers.** Banking77's most-confused pairs are `why_verify_identity →
verify_my_identity`, `unable_to_verify_identity → verify_my_identity`, `card_arrival →
card_delivery_estimate`. Near-synonymous intents inside a professionally built benchmark —
independent evidence for §3's claim that *boundaries* are the difficulty, and that a human
ceiling on fine-grained intent work sits well below 100%.

---

## 5. Does the judge agree with a human?

60 replies rated by hand on the same four binary dimensions, blind to the judge, in
shuffled order. 51 overlap with judged items (the judge reached 76 of 90 before hitting
its own daily token cap).

| dimension | raw | Cohen's κ | reading | judge says yes | human says yes |
|---|---|---|---|---|---|
| grounded | 96% | +0.778 | substantial | 90% | 90% |
| addresses_ask | 88% | +0.346 | fair | 94% | 86% |
| no_overpromise | 98% | +0.847 | almost perfect | 94% | 92% |
| tone_ok | 96% | **+0.000** | slight | 100% | 96% |
| **acceptable** | 88% | **+0.599** | moderate | 86% | 78% |

**`tone_ok` is a live demonstration of the kappa paradox**: 96% raw agreement and κ=0.000,
because both raters say yes essentially always, so chance agreement is already ~96%. Raw
agreement alone would have reported this dimension as near-perfect; it carries no
information at all. This is exactly why both columns are printed everywhere in this repo.

2×2 on `acceptable`:

| | judge yes | judge no |
|---|---|---|
| **human yes** | 39 | 1 |
| **human no** | 5 | 6 |

**The judge is too generous** — it passes 5 replies a human rejected and rejects 1 a human
passed. So **the reply-quality figure is an upper bound**, and the judge's 86% should be
read against the human's 78%.

`addresses_ask` at κ=0.346 is the weakest dimension and carries almost all the
disagreement. That is actionable in a way "the judge is unreliable" is not: it names the
one rubric line to rewrite.

### The disagreements, and what they found

All six splits on `acceptable` share a shape: **the system answers a nearby question
instead of the one asked, and the judge accepts it.**

> **customer:** *"Hari ini ngga bisa login di handphone, kenapa yah padahal email dan password benar"*
> **reply:** *"We can help out in English via Twitter, but we also have Indonesian support via email at …"*
> **human: no** — language deflection; the login problem is never touched. **Judge: yes.**

> **customer:** *"Email associated with fb account is no longer active but is on file with my subscription. How do I fix?"*
> **reply:** *"It can take 24-48 hours for the email to update. If it still doesn't change by then, let us know."*
> **human: no** — assumes they just changed their email and should wait; they said the address on file is dead. **Judge: yes.**

> **customer:** *"Followed migration steps, but when activating 60 days premium trial, says its not available in my area."*
> **reply:** *"Could you try signing up at … to see if the 60-day offer appears?"*
> **human: no** — ignores the stated blocker and says try again. **Judge: yes.**

I think the human is right in all three. Each reply is fluent, on-topic and grounded, and
answers a question the customer did not ask. That is the failure a judge optimising for
plausibility is least equipped to catch.

---

## 6. The escalation design, and the one thing it refuses to do

Ground truth comes only from rules reading the customer's message (E1–E5: hacked account,
disputed charge, legal threat, safety, repeat contact). Rules reading *our own program* —
classifier confidence below τ=0.55, retrieval similarity below σ=0.45 — may **trigger** an
escalation at run time but never define the correct answer.

If they did, the system could not be wrong: any time it hesitated, hesitating would be
correct by definition. The eval would print a confident number and measure nothing.
Instead those extra escalations show up as a **drop in escalation precision**, pricing
caution visibly.

Escalations by source on test: E1 ×2, E2 ×1, R1 ×1, R2 ×1. So 2 of 5 escalations came from
runtime signals, and precision is 60% — the two R-triggered ones are the cost.

**τ and σ were never tuned.** The token budget did not allow a dev sweep, so the defaults
in `config.py` were used and the test split was touched once. That removes any risk of test
contamination and also means these thresholds are almost certainly not optimal.

---

## 7. Failure analysis — top 5

**1. Escalation rules under-fire on real language. (recall 43%, 4 of 7 missed)**
Every miss has the same cause: E1–E5 are English keyword patterns.

- *"The email associated with my account was changed and it wasn't me!"* — E1 requires
  "not me" near *login* or *charge*, not near *email*. **Account takeover, auto-handled.**
- *"you have charged me more for my subscription when I have the student account"* — E2
  matches "charged me twice/again", not "charged me more".
- A Dutch security complaint — **the rules cannot fire on non-English text at all.**
- *"tried for months to cancel, no option to cancel in app"* — arguably a correct non-fire;
  no charge is disputed. I kept the human label and flagged the disagreement.

This is the same failure class as the DM-punt detector in §2: a keyword rule that looks
complete and under-fires on phrasing it did not anticipate. It is the single most important
defect in the system, because the cost is asymmetric — a missed security escalation is not
symmetric with an unnecessary one.

**2. Language deflection.** Non-English messages get routed to language support instead of
answered (§5). The judge passes these. Two of 60 human-rated replies; more in the corpus.

**3. Answering a nearby question.** The reply is fluent, grounded and on-topic, and
addresses something the customer did not ask — the dead-email and not-available-in-my-area
cases in §5. `addresses_ask` is where human and judge disagree most (κ=0.346).

**4. Retrieval finds topically similar, situationally wrong precedent.** *"you charged me
twice for premium and it still says free"* retrieved playback troubleshooting and the
drafter produced *"does logging out and back in help?"*. Cosine similarity does not know
that a billing complaint and a playback complaint need different replies.

**5. The unstable boundaries are unstable for the model too.** `app_bug`↔`playback_issue`
(3), `product_feedback`↔`how_to` (4), `app_bug`↔`account_access` (2) — the same pairs the
human flip-flopped on in §3. These are answer-key defects as much as model errors, which is
why intra-annotator agreement is reported first.

---

## 8. What is misleading about my headline number

1. **The golden set's class mix is not the real one.** Rare intents were topped up, so
   macro-F1 (0.77) is computed on a distribution that does not exist in the wild.
   Weighted-F1 (0.76) is closer to what a random customer experiences. They happen to be
   almost identical here, which is luck, not design.

2. **One person wrote the labels and the prompts.** My sense of where `billing_charge` ends
   and the model's came from the same head. Blind self-agreement was κ=0.838 — that is the
   measurement ceiling, and 76% is below it, so there is real room. §4b is a partial
   counterweight, not a fix.

3. **"Grounded" is graded against replies that are 41% DM punts.** A high groundedness
   score partly measures how well the agent learned to punt, which is not resolution.

4. **n=90 means ±9 points.** The system-vs-B2 gap (30 points) survives that easily; nothing
   smaller than ~18 points between two configs here is a finding.

5. **An LLM wrote the replies and an LLM graded them**, and the judge agrees with a human
   at κ=0.599 while being **too generous**. Reply quality is an upper bound: judge 86%,
   human 78% on the same 51 items. The judge covers 76 of 90 replies, not all of them,
   because it hit its own daily token cap.

6. **Escalation recall (43%) is measured on 7 positive examples.** The interval on that is
   enormous. The direction is certain — rules under-fire — but the magnitude is not.

7. **Escalation labels are partly rule-derived.** Hand-confirmation was only asked where a
   rule fired or the intent made a human plausible; 36 of 150 rows were hand-confirmed and
   the rest taken as auto-handle by the policy's definition.

8. **Near-duplicate messages make retrieval look easier.** Retrieval excludes each item's
   own row by id, but thousands of customers type near-identical complaints.

9. **Nothing was tuned.** τ, σ and every prompt are untuned defaults. No overfitting risk,
   and no optimisation either — these numbers are a floor for this design, not its best.

---

## 9. What I would do next with one more week

1. **Add `catalogue_error` as a twelfth intent.** The blind relabel found it: four of seven
   self-disagreements are metadata corrections that fit neither existing class. This is the
   highest-value change and it came from measurement, not intuition.
2. **Replace the escalation keyword rules with a classifier.** §7's root cause is that
   regexes under-fire on unanticipated phrasing and cannot read non-English at all. An
   escalation classifier trained on the E-rule definitions, validated against the same hand
   labels, is the direct fix.
3. **A second annotator on 50 rows.** Intra-annotator κ bounds the answer key; only
   inter-annotator κ shows the taxonomy is communicable to anyone else.
4. **Rewrite the `addresses_ask` rubric line and re-measure.** κ=0.346 is where judge and
   human diverge, and every quoted disagreement in §5 splits on it. Judge validation is
   only useful if it changes the judge.
5. **Calibrate confidence.** The classifier self-reports 1.0 routinely and R1 gates on it.
   Bucket by confidence, plot real accuracy per bucket, and replace it with agreement across
   3 samples if it is as miscalibrated as it looks.
