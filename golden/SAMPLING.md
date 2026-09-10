# How the golden set was sampled and labelled

200 messages, hand-labelled by one person, used as the answer key for everything.

## Sampling

Three strata, deliberately unequal. Built by `golden/sample_golden.py` with a fixed seed.

| n | stratum | why |
|---|---|---|
| 140 | **random** | natural prevalence, drawn from the full 20k sample. The honest half. |
| ~45 | **rare-class top-up** | rare intents brought to ~20 examples each, chosen at random *within* each class from the weak-labelled pool. |
| 15 | **hard** | multi-intent, sarcasm, non-English, emoji-only, very long, prompt-injection-shaped. Selected by pattern, spread across failure kinds, near-duplicates removed. |

**The top-up is the largest distortion in this project.** A class with three examples has
a standard error near 30 points, so any per-class claim about it would be unfalsifiable —
but the price is that the class mix in this set is not the class mix in the wild. Macro-F1
computed here flatters rare classes. The report therefore always prints macro-F1,
prevalence-weighted F1, and the true distribution measured from the weak labels, and says
which one a real customer would experience.

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

**60 dev / 140 test**, assigned by seeded shuffle *before any label existed* and hidden by
the labelling tool.

Assigning the split after labelling would allow it to be chosen — even unconsciously —
with knowledge of which rows turned out awkward. Hiding it during labelling stops the two
halves being labelled differently as attention drifts.

All prompt wording and both thresholds (τ, σ) were tuned on **dev only**. Test was run
`[TBD]` time(s). Any additional look is declared in the report, because the moment the
test set is tuned against it stops being held out.

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
   Revealing it is one keypress and auto-tags the row. Rows that needed it: `[TBD]`.

**The written escalation policy is checked *after* each answer, never before.** Showing
which rules fired first would turn hand-labelling into rubber-stamping a regex. On
disagreement the labeller either changed the label or wrote why it stands; unresolved rows
are flagged and revisited via `--review`.

Deliberate exceptions kept after review: `[TBD]`. That number measures how much of this
policy genuinely needs human judgement rather than pattern matching.

## Measuring the ceiling

50 rows were relabelled **blind, at least 24 hours later**, in a different order, with the
first answer hidden (`golden/relabel.py`).

- intent: raw `[TBD]`, kappa `[TBD]`
- escalate: raw `[TBD]`, kappa `[TBD]`

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
