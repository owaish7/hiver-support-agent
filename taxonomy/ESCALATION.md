# Escalation policy

The agent must decide, for every message: **handle it automatically, or hand it to a
human?** Not a bare yes/no -- it must state a reason from a fixed list, because
"escalate" with no cause is not reviewable and cannot be measured.

## Content rules (E1-E5)

These are properties of **the customer's message**. They decide the correct answer.

| id | rule | reason |
|----|------|--------|
| **E1** | The customer says their account is compromised, hacked, accessed by someone else, or reports fraudulent activity. | `security_risk` |
| **E2** | The customer disputes a specific charge, says they were charged more than once or after cancelling, or demands a refund. | `money_movement` |
| **E3** | The customer threatens legal action, names a regulator or ombudsman, mentions a lawyer, or threatens to go to the press. | `legal_or_reputational` |
| **E4** | The message contains self-harm, threats of violence, or abuse directed at a person. | `safety` |
| **E5** | This is the customer's third or later message about the same unresolved issue. | `repeat_contact` |

If none of E1-E5 fires, the correct answer is **auto-handle**.

Rules are checked in order and the **first** match wins. So "you charged me twice and
I'm calling my lawyer" is `money_movement`, not `legal_or_reputational`. The order is
arbitrary but it must be fixed, otherwise two labellers reading the same policy would
disagree on multi-rule messages and the answer key would be inconsistent with itself.

## Runtime signals (R1-R2)

These are properties of **our program**, not of the message.

| id | rule | reason |
|----|------|--------|
| **R1** | Intent classifier confidence below `TAU`. | `low_confidence` |
| **R2** | Best retrieved historical match below cosine `SIGMA`. | `no_precedent` |

At inference these may also trigger an escalation. **They never decide the correct
answer.**

### Why that separation matters

If the answer key were allowed to say "escalate because the model was unsure", then
the model's uncertainty would be both the prediction and the target. The system could
never be wrong: any time it hesitated, hesitating would be correct by definition. The
eval would measure nothing and would still print a high number.

So ground truth comes only from E1-E5, which a second person could apply to the same
message and reach the same answer. R1 and R2 are allowed to cause extra escalations at
run time, and those extras then show up as a **drop in escalation precision** -- which
is the honest way to display what caution costs, rather than hiding it inside the
definition of correct.

## What this policy deliberately does not do

- **No sentiment or anger rule.** Angry customers with ordinary problems are still
  ordinary problems, and "escalate if annoyed" would route a large fraction of all
  support tweets to a human, which is not a product.
- **No monetary threshold.** These are tweets; there is no reliable amount to read.
- **No per-intent blanket escalation.** `billing_payment` is not automatically
  escalated -- "how do I change my card" is answerable. E2 fires on a *disputed*
  charge, which is the part a bot must not touch.
