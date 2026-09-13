# Decision log

Fifteen non-obvious decisions and why. Every number here is measured.

Smaller choices live next to the code they affect — `config.py` carries the token budget
and threshold reasoning, and each module's docstring explains its own trade-offs.

---

**1. The brand was chosen by measurement, and the first measurement was wrong.**
Many support accounts reply "please DM us" to almost everything. Build on one of those and
"grounded in how the brand resolved this" quietly becomes "learn to punt" — retrieval
works, the eval passes, the system has learned nothing. So `data/brand_compare.py` measures
the substantive-reply rate first: **SpotifyCares 59%**, Delta 45%, AppleSupport 41%.

The first version of the punt detector was a substring list, and sampling real replies
showed it missed phrasings like *"let's hop into DM"* and *"follow/DM your confirmation
number"* — **unevenly**: 6.8% of Spotify's "substantive" replies were really punts against
26.5% of Apple's. Apple was inflated by 15 points and sat in second place. A crude rule
applied unevenly is worse than one applied evenly. Sensitivity check: the ranking holds
from a 12-word threshold upward; at 8 words Delta overtakes, so the choice is not
threshold-invariant and the report says so.

**2. One item = the first customer message of a thread.**
Later turns ("ok thanks", "still broken") inherit the intent above them. Including them
would pad the test set with easy near-duplicates and make accuracy look better than it is.
Cost: no multi-turn behaviour is measured at all.

**3. Banking77 was rejected as a taxonomy and adopted as an external check.**
*Rejected* as a category list — 77 retail-banking intents against a 150-row answer key is
2 examples per class. *Adopted* to answer the strongest objection available against this
project: one person wrote both the labels and the prompts, so they are correlated.
Banking77's labels are PolyAI's. TF-IDF scores **87% ±1** there against a published
fine-tuned reference of 94% — and the same method scores 46% on my 60 training rows, which
shows the low in-project figure is a data-budget artefact, not a method failure.

Its most-confused pairs (`why_verify_identity → verify_my_identity`, `card_arrival →
card_delivery_estimate`) are near-synonymous intents inside a professional benchmark —
outside evidence that **boundaries are the hard part, not the class list**.

**4. Eleven intents, and the first eight — written from a skim — were wrong.**
The first list missed three real categories: `account_security` (an account takeover is not
a forgotten password), `how_to` (a question is not a fault, and it is the best auto-handle
case in the corpus), and `dm_followup` ("check your DMs" is not a support request, and it
is 3% of traffic). It also pooled feature requests with praise, hiding the largest cluster.
The weak-label distribution showed the damage: **22% landed in `other`**. Reading 100
messages fixed it. That failure is the argument for reading data before defining anything.

Eleven rather than twenty because a class with three examples has a standard error near 30
points. Two classes still ended up at n≤4 in test and their numbers are quoted with that
caveat.

**5. Ground truth for escalation uses content rules only (E1–E5). Runtime signals (R1, R2)
never define correctness.**
The load-bearing decision. If "the classifier was unsure" could make an escalation correct,
the system could never be wrong — hesitating would be correct by definition, the eval would
print a confident number and measure nothing. R1/R2 still fire at run time; those extra
escalations land as a **drop in escalation precision** (60%), pricing caution visibly
instead of hiding it inside the definition of correct.

**6. Escalation rules are ordered and first match wins.**
*"You charged me twice and I'm calling my lawyer"* is `money_movement`, not
`legal_or_reputational`. The order is arbitrary but must be fixed, or two people applying
the same written policy produce different answer keys on multi-rule messages. Asserted in
`tests/test_offline.py`.

**7. Content rules outrank runtime signals at inference too.**
A hacked account escalated as "low confidence" is technically an escalation and useless to
whoever picks it up. The reason field has to name the real cause.

**8. No sentiment or anger rule.**
Angry customers with ordinary problems are still ordinary problems. "Escalate if annoyed"
would route a large share of all support tweets to a human, which is a queue, not a product.

**9. Only substantive replies enter the retrieval index.**
A "please DM us" reply is not a resolution and cannot ground anything; indexing punts would
let the drafter retrieve four and confidently produce a fifth. This also makes the punt
rate (41%) a hard ceiling on grounded reply quality, which is reported rather than worked
around.

**10. Retrieval excludes the query's own row.**
Every golden item lives in the index it retrieves from. Without `exclude_id` its nearest
neighbour is itself at similarity 1.0 and the drafter is handed the exact reply it is about
to be graded against. Found by writing the test, not by reading the code.

Excluding by id is right but not complete: if the same text appears twice under two ids,
the second copy survives. Measured duplicate rate 0.7%. Not patched, because matching two
different customers with the same complaint is the system working correctly.

**11. The golden set is stratified, and the bias is declared rather than corrected.**
140 random at natural prevalence, 45 topping up rare intents, 15 deliberately awkward
picked by pattern. The top-up means the class mix is not the mix in the wild, so macro-F1
flatters rare classes — hence macro and prevalence-weighted F1 are always printed together.

**12. The dev/test split is fixed before any label exists, and the labelling tool hides
three things.**
Assigning the split afterwards would let it be chosen, even unconsciously, with knowledge
of which rows turned out awkward. The tool hides the machine's guess (seeing it produces
agreement with it, contaminating every later human-vs-model comparison), the split itself,
and the brand's real reply (the classifier never sees it, so labelling with it builds an
unreachable ceiling). The written policy is checked **after** each answer, never before —
showing the rules first turns labelling into rubber-stamping a regex. Result: 9 deliberate
exceptions where the human overrode the policy with a written reason.

**13. Intra-annotator agreement is measured, blind, 20+ hours later.**
With one annotator this is the only honest ceiling available. 50 rows relabelled with the
first answer hidden and the order shuffled: **intent κ=+0.838** (raw 86.0%), **escalate
κ=+0.728** (raw 96.0%). The system scores 76%, below that ceiling, so the remaining gap is
real rather than answer-key noise.

The disagreements found a defect, not noise: four of seven are the same pair,
`content_unavailable ↔ product_feedback`, and all four are catalogue **metadata** errors —
the content exists and is mislabelled, which is neither category. A missing twelfth class,
discovered by measurement.

**14. The judge is a different model family, answers yes/no, and is assumed invalid until
proven otherwise.**
`gpt-oss` writes, `qwen` grades — models score their own family's output more generously,
and what matters is the separate training lineage, not the host. Four **binary** dimensions
rather than a 1–5 score, because nobody can say what separates a 3 from a 4, so averaging
1–5 ratings gives a number that can neither be acted on nor compared against a human.

Validated against 60 hand ratings: **κ=+0.599**, and the judge is **too generous** — it
passes 5 replies a human rejected. Reply quality is therefore an upper bound: judge 86%,
human 78%. `tone_ok` came out at 96% raw agreement and **κ=0.000**, a live demonstration of
the kappa paradox — raw agreement alone would have called it near-perfect.

**15. Nothing was tuned, and the weakest result is the headline of the failure analysis.**
τ and σ are untuned defaults; the token budget did not allow a dev sweep and test was run
once. No contamination risk, and no optimisation either — these are a floor for this design.

**Escalation recall is 43%**, 3 of 7. All four misses share one cause: E1–E5 are English
keyword patterns that under-fire on unanticipated phrasing — *"my email was changed and it
wasn't me"* (E1 wants "not me" near *login* or *charge*), *"charged me more"* (E2 matches
"charged me twice"), and a Dutch complaint the rules cannot read at all. Same failure class
as decision #1. The cost is asymmetric: a missed security escalation is not the mirror image
of an unnecessary one.
