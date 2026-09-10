"""The intent list, in one place, imported by everything else.

PROVISIONAL until the read-100 session on day 1. This is a straw-man derived from a
first skim, not from reading, and the whole point of D3 is that the taxonomy comes out
of actually reading messages. Expect names to change and at least one class to merge or
split. When that happens, change it here and nowhere else.

The definitions and boundary rules below are the load-bearing part. A list of class
names is not a taxonomy; two people only agree on labels if they agree on where the
boundaries are, and the boundaries are where all the interesting disagreement lives.
"""

from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    PLAYBACK_ISSUE = "playback_issue"
    ACCOUNT_ACCESS = "account_access"
    BILLING_PAYMENT = "billing_payment"
    SUBSCRIPTION_CHANGE = "subscription_change"
    CONTENT_MISSING = "content_missing"
    APP_BUG_CRASH = "app_bug_crash"
    PRAISE_OR_FEEDBACK = "praise_or_feedback"
    OTHER = "other"


INTENTS: list[str] = [i.value for i in Intent]

# One line each. These strings go into the classifier prompt verbatim, so the model and
# the human labeller are working from exactly the same definitions -- if they drift, the
# eval is measuring prompt drift rather than model capability.
DEFINITIONS: dict[str, str] = {
    "playback_issue":
        "Audio will not play, keeps stopping, skips, buffers, or downloads/offline "
        "mode is not working. The app itself runs; the music does not.",
    "account_access":
        "Cannot log in, locked out, password reset not arriving, or the account is "
        "linked to the wrong email or third-party login.",
    "billing_payment":
        "Something is wrong with money already paid or a payment that failed: charged "
        "unexpectedly, charged twice, card declined, invoice query.",
    "subscription_change":
        "Wants to start, stop, upgrade, downgrade, or restructure a plan, including "
        "family/student plan membership changes. Nothing has gone wrong yet.",
    "content_missing":
        "A specific song, album, artist, podcast or playlist is absent, removed, "
        "greyed out, or unavailable in their country.",
    "app_bug_crash":
        "The application itself misbehaves: crashes, will not open, UI broken, update "
        "failed, or a device/platform integration is broken.",
    "praise_or_feedback":
        "Compliments, thanks, complaints about product direction, or feature requests. "
        "No specific problem to resolve for this customer.",
    "other":
        "Anything else, including jobs, press, partnerships, spam, and messages with no "
        "discernible request.",
}

# The rules that decide the cases that actually get argued about. Every one of these
# came from a real message that was genuinely ambiguous on first read.
BOUNDARY_RULES: list[str] = [
    "Classify by what the customer WANTS, not by what they mention. 'My app keeps "
    "crashing so I want to cancel' is subscription_change: the cancellation is the "
    "ask, the crash is the justification.",

    "billing_payment vs subscription_change: if money has already moved wrongly, it is "
    "billing_payment. If they are asking to change a plan going forward and nothing has "
    "gone wrong, it is subscription_change. 'Charged twice, cancel my Premium' is "
    "billing_payment -- the money problem is the real ask.",

    "playback_issue vs app_bug_crash: if the app runs and the audio fails, it is "
    "playback_issue. If the app itself will not run, it is app_bug_crash.",

    "playback_issue vs content_missing: 'this song will not play' is playback_issue; "
    "'this song is not on Spotify any more' is content_missing. The test is whether "
    "they believe the content exists and is broken, or believe it is gone.",

    "account_access vs billing_payment: 'I paid but it still says free' is "
    "billing_payment, because the payment is the thing that did not take effect. "
    "'I cannot get into my account at all' is account_access.",

    "praise_or_feedback vs other: praise_or_feedback is about the product. A message "
    "about a job application or a partnership enquiry is other.",

    "A message with two genuine asks takes the one the customer would be angriest "
    "about if it went unanswered, and gets tagged is_hard with a note naming the "
    "runner-up. Multi-intent is a real failure mode and hiding it in a clean label "
    "would make the failure analysis dishonest.",
]


def prompt_block() -> str:
    """The taxonomy rendered for a prompt. Built from the same dicts the labelling
    guide renders from, so the model and the human cannot silently diverge."""
    lines = ["Intent categories:"]
    for name in INTENTS:
        lines.append(f"- {name}: {DEFINITIONS[name]}")
    lines.append("")
    lines.append("Boundary rules:")
    for rule in BOUNDARY_RULES:
        lines.append(f"- {rule}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(prompt_block())
