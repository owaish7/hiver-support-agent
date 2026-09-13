# Decision log

Non-obvious choices and why. Every number here is measured.

---

**1. The brand was chosen by measurement, not preference.**
Many support accounts on this dataset reply "please DM us" to almost everything. Build on
one of those and "draft a reply grounded in how the brand resolved this" quietly becomes
"learn to say DM us" — the retrieval works, the eval passes, the system has learned
nothing. `data/brand_compare.py` measures the substantive-reply rate across three
candidates first. Chosen: **SpotifyCares, 59% substantive** (26,068 pairs), against
Delta 45% and AppleSupport 41% -- Apple has 2.9x the volume and the lowest rate.

**2. One item = the first customer message of a thread.**
Later turns ("ok thanks", "still broken") inherit the intent of the message above them.
Including them would pad the test set with easy near-duplicates and make accuracy look
better than it is. Cost: no multi-turn behaviour is measured at all, which is stated in
the report as a limitation rather than a feature.

**3. Banking77 was rejected as a taxonomy and adopted as an external check.**
Two different uses, and only the first is a bad idea.

*Rejected:* using its 77 retail-banking intents as this project's category list. Wrong
domain, and 77 classes against a 150-row answer key is 2 examples per class, which
measures nothing.

*Adopted:* running the same classification method against it to get a number that does not
depend on my own answer key. This is the direct answer to the strongest objection available
against this project — that one person wrote both the labels and the prompts, so they are
correlated. Banking77's 3,080 test queries were labelled by PolyAI on a public benchmark
with published reference numbers.

Measured so far, with no API calls: TF-IDF + logistic regression trained on the 10,003-row
train split scores **87% ±1 accuracy, 0.874 macro-F1** on all 3,080 test rows, against a
published fine-tuned ModernBERT reference of 94%. The LLM zero-shot row was not run: the
free-tier token budget went to the main eval, which is the deliverable. The TF-IDF row
already makes the point that matters -- the same classical method scores 46% on 60
training rows here and 87% on Banking77's 10,003, so the low in-project score is a
data-budget artefact rather than a method failure.

Stated limits, in the report rather than buried: Banking77 queries are clean and
well-formed while tweets are not, so this tests the method and not robustness to noise;
the published row is a fine-tuned encoder that saw 10,003 labelled examples while the LLM
row is zero-shot, so it is a reference point and not a competition; and it is a different
domain entirely.

**3b. The external check confirmed the taxonomy argument independently.**
Banking77's most-confused pairs are `why_verify_identity → verify_my_identity`,
`unable_to_verify_identity → verify_my_identity`, `card_arrival →
card_delivery_estimate`, `top_up_reverted → top_up_failed`. These are near-synonymous
intents in a professionally built benchmark. It is outside evidence for the claim this
project makes about its own taxonomy: **the boundaries are the hard part, not the class
list**, and a human ceiling on fine-grained intent work is well below 100%.

**3c. Banking77 is downloaded on demand, not committed.**
PolyAI's data under CC-BY-4.0 (Casanueva et al. 2020, arXiv:2003.04807). Cited, not
redistributed. The fetch takes about two seconds and the results cache means
`--report` still works offline.

**3d. An off-taxonomy prediction counts as wrong, not as an error.**
When the model answers with a label that is not in the list, that is a wrong answer.
Dropping those rows as "errors" would quietly remove the model's worst failures from the
accuracy figure.

**4. Eleven intents, and the first eight were wrong.**
The first list was written from a skim and missed three real categories -- `account_security`
(an account takeover is not a forgotten password), `how_to` (a question is not a fault, and
it is the best auto-handle case in the corpus) and `dm_followup` ("check your DMs" is not a
support request, and it is 3% of traffic). It also pooled feature requests with praise,
hiding the largest single cluster. The weak-label distribution showed the damage: 22% landed
in `other`. Reading 100 messages fixed it, and that failure is the argument for reading data
before defining categories.

Eleven rather than twenty because a class with three examples has a standard error near 30
points, so "the model is bad at X" would be unfalsifiable. Two classes still ended up at
n<=4 in test (`dm_followup`, `account_security`) and their per-class numbers are quoted with
that caveat.

**5. The taxonomy came from reading, and clustering only checked it.**
100 messages read by hand; embedding clustering run afterwards purely to catch a category
that reading missed. Clusters split on vocabulary (iPhone vs Android) rather than on what
the customer wants, and a label nobody wrote cannot be defended in an interview.

**6. Boundary rules are the taxonomy; the class list is just names.**
Two people only agree on labels if they agree on where the boundaries are. Every rule in
`taxonomy/intents.py` came from a specific message that was genuinely ambiguous — e.g.
"charged twice, cancel my Premium" is `billing_charge`, because the money problem is the
ask and cancelling is the customer's proposed remedy. Each rule cites the message number
in the reading sample that forced it.

**7. The escalation policy is written twice — and the code is NOT an oracle.**
Prose goes into the prompt, code checks the labels. No keyword list can decide whether
"see you in court" is a real legal threat, so the hand labels remain ground truth and the
code only reports disagreements for a human to resolve. Deliberate exceptions kept after
review: **9 of 150**. That count is itself a measurement of how much of the policy needs
judgement.

**8. Ground truth uses content rules only (E1–E5). Runtime signals (R1, R2) never define
correctness.**
This is the load-bearing one. If "the classifier was unsure" could make an escalation
correct, the system could never be wrong — hesitating would be correct by definition, and
the eval would print a high number while measuring nothing. R1/R2 still fire at run time;
the extra escalations land as a drop in escalation precision, which prices caution
honestly instead of hiding it in the definition of correct.

**9. Escalation rules are ordered and first match wins.**
"You charged me twice and I'm calling my lawyer" is `money_movement`, not
`legal_or_reputational`. The order is arbitrary but it must be fixed, or two people
applying the same written policy would produce different answer keys on multi-rule
messages. Asserted in `tests/test_offline.py`.

**10. Content rules outrank runtime signals in the pipeline too.**
A hacked account escalated as "low confidence" is technically an escalation and useless
to whoever picks it up. The reason field has to name the real cause.

**11. No sentiment or anger rule.**
Angry customers with ordinary problems are still ordinary problems. "Escalate if annoyed"
would route a large share of all support tweets to a human, which is a queue, not a
product.

**12. Local embeddings, not an embedding API.**
`all-MiniLM-L6-v2` on CPU: no rate limits during an eval run, byte-identical results
every time, and the index rebuilds with no API key. That last point is what actually makes
the 15-minute reproduction promise true for a reviewer who has no keys.

**13. NumPy, not a vector database.**
At 11,728 indexed rows the entire retrieval engine is one `@` and an argsort. FAISS or Chroma would
add a dependency, a build step and a failure mode in exchange for nothing measurable.

**14. Only substantive replies go into the index.**
A "please DM us" reply is not a resolution and cannot ground anything. Indexing punts
would let the drafter retrieve four of them and confidently produce a fifth. This also
means the DM-punt rate is a hard ceiling on how good grounded replies can be, which is
reported rather than worked around.

**15. Retrieval excludes the query's own row.**
Every golden item lives in the index the system retrieves from. Without `exclude_id` the
nearest neighbour of a golden item is itself at similarity 1.0, and the drafter is handed
the exact reply it is about to be graded against. Caught by writing the test, not by
reading the code.

Excluding by id is the right guard and not a complete one. Two *different* customers
writing near-identical complaints is legitimate evidence and should be retrieved — that is
the system working. But if the same message text appears in the corpus twice under two
ids, exclusion by id will not catch the second copy. Duplicate rate measured at 1%;
the effect is to make retrieval look slightly easier than it is, and it is listed in the
report's misleading-numbers section rather than silently patched, because the fix
(dropping near-duplicates) would also drop genuine repeat complaints.

**15b. The index is stored as fixed-width unicode, not object arrays.**
Pandas returns `dtype=object` for text columns, and numpy can only reload object arrays
with `allow_pickle=True` — which means unpickling arbitrary code to read your own data
file. Casting to `np.str_` keeps the index loadable with pickling off. Found by running
`build_index.py` for real; it would have blocked setup at the first step.

**16. The golden set is stratified, and the resulting bias is declared, not corrected.**
140 random preserves natural prevalence; ~45 top up rare intents so their per-class
numbers mean something; 15 are deliberately awkward. The top-up means the class mix in
the golden set is not the class mix in the wild, so macro-F1 flatters rare classes. Both
macro and prevalence-weighted F1 are always printed, and the report says which one a real
customer would feel.

**17. Hard cases are picked by pattern, spread across kinds, and deduped.**
A random draw of 140 contains almost no non-English, emoji-only or multi-intent messages,
so the failure analysis would have nothing to analyse. First implementation took the top
15 by score and returned fifteen near-identical "?" tweets — fifteen examples of one
thing is one example. Now round-robins across failure kinds.

**18. The dev/test split is fixed before any label exists, and hidden while labelling.**
Assigning it afterwards would allow the split to be chosen — even unconsciously — with
knowledge of which rows turned out awkward. Tuning happens on dev only; test is touched
once, and any exception is declared in the report.

**19. The labelling tool hides three things: the machine's guess, the split, and the
brand's reply.**
Seeing the model's guess produces agreement with it, contaminating every later
human-vs-model comparison. The brand's reply is legitimate context but the classifier
never sees it, so labelling with it builds a ceiling the system cannot reach — revealing
it is one keypress and auto-tags the row, so the report can say how many labels needed it.

**20. The written policy is checked after the labeller answers, never before.**
Showing which rules fired first would turn hand-labelling into rubber-stamping a regex.
Commit, then see the check, then decide whether the call stands.

**21. Intra-annotator agreement is measured, blind, 24 hours later.**
With one annotator this is the only honest ceiling available: disagree with yourself 12%
of the time and a classifier scoring 88% is at the noise floor of the answer key. The tool
refuses to run before 20 hours have passed, because relabelling from memory measures
recall rather than whether the definitions are stable. Result: **intent kappa +0.838**
(raw 86.0%), **escalate kappa +0.728** (raw 96.0%). The system scores 76%, below that
ceiling, so the remaining gap is real rather than answer-key noise.

**22. Four baselines, because each kills a different objection.**
Trivial reads nothing (is the model doing anything?). Canned sends the brand's most common
reply to everything (is reply quality real, or does the rubric reward safe non-answers?).
TF-IDF + logistic regression uses no LLM (is the LLM needed?). Retrieval-verbatim returns
a real past reply with no generation (is the writing worth anything, or was it all
retrieval?). If the system only beats retrieval-verbatim by less than the confidence
interval, that is reported as the finding.

**23. B2 trains on LLM weak labels, with every golden id held out.**
200 hand labels minus a dev split leaves too few to train 8 classes fairly, and training
on the golden set would be circular. The weak pool and the golden pool are drawn from the
same rows and overlap by construction, so the exclusion is explicit and tested.

**24. Weak labels choose which rows a human reads. They never decide an answer.**
The cost, stated in the report: if the weak labeller systematically misses a kind of
message, that kind is never surfaced for hand-labelling and the golden set inherits the
blind spot. The pattern-picked hard stratum is a partial counterweight.

**25. The judge is a different model family from the generator.**
Gemini writes, Llama 3.3 on Groq grades. Models score their own family's output more
generously; self-grading would inflate the headline by an amount nobody can measure.

**26. The judge answers four yes/no questions, not "rate this 1–5".**
Nobody can say what separates a 3 from a 4, so averaging 1–5 ratings produces a number
that cannot be acted on and cannot be compared against a human rating. Binary verdicts
against named failure modes can be both.

**27. `acceptable` = grounded AND addresses_ask AND no_overpromise. Tone is excluded.**
Tone is the softest dimension and the one a judge is least likely to agree with a human
on. Letting it gate the headline would import that noise into every other result. It is
still reported separately.

**28. Rationale fields come before verdict fields in every schema.**
Structured output is generated left to right, so a verdict emitted first is a guess the
reasoning is then written to justify. Asserted in the test suite so a future edit cannot
silently reorder them.

**29. The judge sees the retrieved evidence; the evidence is rebuilt from ids, not cached.**
A judge without the evidence would be rating plausibility, which a fluent model always
scores well on. Rebuilding from ids keeps one source of truth — copying evidence text into
the cache would let it drift from the index, and the judge would grade against something
the drafter never saw.

**30. Every reply-quality number is held to be uncalibrated until `agreement.py` runs.**
The judge is an instrument. An instrument nobody has checked produces readings, not
evidence. Result: **kappa +0.599 on `acceptable`** (n=51), and the judge is **too
generous** -- it passes 5 replies a human rejected and rejects 1 it should have passed. So
reply quality is reported as an upper bound: judge 86%, human 78% on the same items.

**31. Results are cached per item, and `--report` calls nothing.**
A crash at item 190 must not discard 189 paid-for calls, and a reviewer with no API key
must still see the full table. This is what makes the 15-minute claim hold.

**32. Temperature 0 everywhere.**
Otherwise "the system improved" and "the sampler got lucky" are the same observation.

**33. Confidence intervals on the headline number.**
At n=140 an 85% score carries roughly ±6 points. Reporting 85% vs 82% as an improvement
without that interval would be the easiest way to mislead in this whole project, so the
`±` column is not optional.

**34. Statistics are implemented directly rather than imported.**
Cohen's kappa, the Wilson interval and macro/weighted F1 all get quoted in the report and
asked about live, so they are written out in a form that can be defended line by line —
and cross-checked against `sklearn` and `statsmodels`, which they match exactly.


---

**39. The escalation recall of 43% is reported as the headline weakness, not buried.**
Three of seven messages needing a human were routed there. Every miss has one cause: E1-E5
are English keyword patterns that under-fire on unanticipated phrasing -- "my email was
changed and it wasn't me" (E1 wants "not me" near *login* or *charge*), "charged me more"
(E2 matches "charged me twice/again"), and a Dutch security complaint the rules cannot read
at all. Same failure class as the DM-punt detector in #1. The cost is asymmetric: a missed
security escalation is not symmetric with an unnecessary one.

**40. Thresholds were never tuned, and that is stated rather than implied.**
The free-tier token budget did not allow a dev sweep, so tau and sigma are the defaults in
config.py and test was run once. No risk of test contamination, and no optimisation either
-- these numbers are a floor for this design rather than its best.

**41. The judge covers 76 of 90 replies, not all of them.**
It hit its own daily token cap. 51 of those overlap the 60 hand-rated replies, which is
what the agreement number is computed on. Reported as n=51 rather than implying full
coverage.

**42. Labelling stopped at 150, the brief's lower bound.**
150-250 was the stated range. The time available went to labelling carefully and to the
blind second pass, which produces a number almost no submission has, rather than to 50 more
rows labelled faster. The cost is a wider interval: +-9 points at n=90 test instead of +-6
at n=140.
