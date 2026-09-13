# How the golden set was sampled and labelled

150 messages, hand-labelled by one person, used as the answer key for everything.
The brief asks for 150-250; this is the lower end, chosen because labelling is the one
part of this project that cannot be delegated and the time went to doing it carefully.

## Sampling

Three strata, deliberately unequal. Built by `golden/sample_golden.py` with a fixed seed.

| n | stratum | why |
|---|---|---|
| 140 | **random** | natural prevalence, drawn from the full 20k sample. The honest half. |
| 45 | **rare-class top-up** | rare intents brought toward ~20 examples each, chosen at random *within* each class from the weak-labelled pool. |
| 15 | **hard** | multi-intent, sarcasm, non-English, emoji-only, very long, prompt-injection-shaped. Selected by pattern, spread across failure kinds, near-duplicates removed. |

**The top-up is the largest distortion in this project.** A class with three examples has
a standard error near 30 points, so any per-class claim about it would be unfalsifiable —
but the price is that the class mix in this set is not the class mix in the wild. Macro-F1
computed here flatters rare classes. The report therefore always prints macro-F1,
prevalence-weighted F1, and says which one a real customer would experience. They came out
nearly identical here (0.77 vs 0.76), which is luck rather than design.

**The hard stratum is picked, not sampled.** A random draw of 140 contains almost no
non-English or multi-intent messages, so the failure analysis would have nothing real in
it. Every hard row is tagged `is_hard`, and headline numbers are reported with and
without them.

**Rows are surfaced by weak labels, and that bounds what this set can discover.** The
rare-class top-up uses cheap machine labels to decide *which rows a human reads* — never
what the answer is. If the weak labeller systematically misses a kind of message, that
kind is never surfaced and this set inherits the blind spot. The pattern-picked hard
stratum is a partial counterweight, not a fix.

## Split

**60 dev / 90 test**, assigned by seeded shuffle *before any label existed* and hidden by
the labelling tool.

Assigning the split after labelling would allow it to be chosen — even unconsciously —
with knowledge of which rows turned out awkward. Hiding it during labelling stops the two
halves being labelled differently as attention drifts.

Neither threshold was tuned: the free-tier token budget did not allow a dev sweep, so
the defaults in `config.py` were used and test was run **once**. That removes any risk of
test contamination and also means the thresholds are certainly not optimal.

## Escalation labels are partly rule-derived

The escalation question was asked only where it was genuinely in play: where a content rule
(E1-E5) fired, or where the chosen intent made a human plausible. **36 of 150 rows were
hand-confirmed**; the rest were recorded as auto-handle because no rule fired and the
category makes escalation implausible.

That is applying the written policy rather than skipping work -- a `how_to` question that
trips no rule is auto-handle by the policy's own definition. It is still a shortcut, so the
report treats escalation recall as measured against a partially rule-derived key.

## Labelling procedure

Each row gets `intent`, `should_escalate`, `escalation_reason`, `is_hard`, `notes`.
Roughly 90 seconds each; anything taking substantially longer was tagged `is_hard` with a
note explaining what made it hard. Those notes feed the failure analysis directly.

The tool (`golden/label.py`) deliberately withholds three things:

1. **The machine's guess.** Seeing it produces agreement with it, which would contaminate
   every later comparison between human labels and model output.
2. **The dev/test split**, as above.
3. **The brand's actual reply.** It is useful context, but the classifier never sees it —
   labelling with information the system cannot have builds a ceiling it can never reach.
   Revealing it is one keypress and auto-tags the row. Rows that needed it: **0** of 150.

**The written escalation policy is checked *after* each answer, never before.** Showing
which rules fired first would turn hand-labelling into rubber-stamping a regex. On
disagreement the labeller either changed the label or wrote why it stands; unresolved rows
are flagged and revisited via `--review`.

Deliberate exceptions kept after review: **9** of 150. That number measures how much of this
policy genuinely needs human judgement rather than pattern matching.

## Measuring the ceiling

50 rows were relabelled **blind, at least 24 hours later**, in a different order, with the
first answer hidden (`golden/relabel.py`).

- intent: raw **86.0%**, kappa **+0.838** (almost perfect)
- escalate: raw **96.0%**, kappa **+0.728** (substantial)

This bounds everything else in the report. If the labeller disagrees with themselves 12%
of the time, the task is ~88% well-defined and a classifier scoring 88% is already at the
noise floor of its own answer key. The tool refuses to run before 20 hours have passed,
because relabelling from memory measures recall rather than whether the definitions are
stable.

The intent pairs that flipped most often are listed in the report — a pair that recurs
means that boundary rule is not written clearly enough, which is a fixable defect rather
than bad luck.

## Reply quality

60 of the system's drafted replies were graded by hand on the same four binary dimensions
the LLM judge uses, blind to the judge's verdicts, in a shuffled order
(`eval/human_judge.py`). That set is what `eval/agreement.py` measures the judge against.
