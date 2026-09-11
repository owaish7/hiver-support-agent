"""The intent list, in one place, imported by everything else.

Derived from reading 100 real @SpotifyCares messages (golden/read.py, seed 20260910).
Every definition and every boundary rule below traces to a specific message in that
sample, cited by its number, so each one can be defended by pointing at the tweet that
forced it.

Three categories exist because the data insisted on them, and their absence from the
first draft is the clearest evidence that guessing a taxonomy does not work:

  account_security  #75 "have been hacked by users!! My family has been kicked off"
                    is not the same problem as "I forgot my password", and a system that
                    cannot tell them apart will auto-handle an account takeover.

  how_to            #82 "what's the lowest bitrate I can stream on Android" is a
                    question, not a fault. It is also the single best auto-handle case
                    in the corpus, so burying it inside a fault category wastes it.

  dm_followup       #15 "pls check your DMs" is not a support request at all. Five of
                    100 messages are someone chasing an existing private conversation.

The first draft also hid the largest cluster: feature requests (~16 of 100) were pooled
with praise under one "feedback" label, even though a feature request goes to the product
team and praise is closed on the spot.

The boundary rules matter more than the names. A list of class names is not a taxonomy;
two people only agree on labels if they agree on where the edges are, and the edges are
where every real disagreement lives.
"""

from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    PLAYBACK_ISSUE = "playback_issue"
    CONTENT_UNAVAILABLE = "content_unavailable"
    BILLING_CHARGE = "billing_charge"
    PLAN_OR_FAMILY = "plan_or_family"
    ACCOUNT_ACCESS = "account_access"
    ACCOUNT_SECURITY = "account_security"
    APP_BUG = "app_bug"
    PRODUCT_FEEDBACK = "product_feedback"
    HOW_TO = "how_to"
    DM_FOLLOWUP = "dm_followup"
    OTHER = "other"


INTENTS: list[str] = [i.value for i in Intent]

# These strings go into the classifier prompt verbatim, so the model and the human
# labeller work from identical definitions. If they drifted apart, every disagreement
# would be unattributable -- is the model wrong, or right about a different question?
DEFINITIONS: dict[str, str] = {
    "playback_issue":
        "Audio will not play, stops, skips, pauses by itself, or downloaded/offline "
        "music has vanished. The app runs; the music does not.",
    "content_unavailable":
        "A specific song, album, artist or podcast is missing, was removed, is not "
        "released yet, or is unavailable in their country.",
    "billing_charge":
        "Money has already moved and something is wrong with it, or they are asking "
        "about how payment works: charged twice, charged after cancelling, paid but "
        "still on free, wants a refund.",
    "plan_or_family":
        "Starting, stopping, switching or restructuring a plan, including student "
        "discounts and adding or removing family-plan members. Nothing has gone wrong "
        "with money yet.",
    "account_access":
        "Cannot get into their account: forgotten password, locked out, login broken, "
        "or the account is tied to an email or Facebook login they no longer have.",
    "account_security":
        "They believe someone else has access, or that their data is exposed: hacked, "
        "account playing on an unknown device, family members kicked off, unauthorised "
        "charges, credentials shown in the open.",
    "app_bug":
        "The application itself misbehaves: crashes, will not open, search is broken, "
        "the web player is down, a feature errors out.",
    "product_feedback":
        "An opinion about the product rather than a fault to fix: feature requests, "
        "complaints about how an existing feature behaves, gripes about "
        "recommendations, and praise. Nothing to resolve for this customer.",
    "how_to":
        "A question with an answer, where nothing is broken. Limits, settings, how a "
        "feature works, when something will happen.",
    "dm_followup":
        "Chasing an existing private conversation -- 'check your DMs', 'please reply to "
        "my message'. Contains no support request of its own.",
    "other":
        "Anything else: job applications, press, jokes, spam, and messages with no "
        "discernible request.",
}

# Every rule below came from a message that was genuinely ambiguous on first read. The
# citation is the message number in the 100-message reading sample.
BOUNDARY_RULES: list[str] = [
    "Classify by what the customer WANTS, not by what they mention. Most of the hard "
    "cases mention two things and want one.",

    "#25 'Why do I pay for Spotify if it's just going to delete my downloaded albums' "
    "is playback_issue, not billing_charge. The vanished downloads are the fault; the "
    "payment is rhetorical. No charge is actually being disputed.",

    "#33 'why is y'all's app $13 a month and it only works half the time' is "
    "product_feedback, not billing_charge. There is no specific fault to fix and no "
    "charge in dispute -- it is a complaint about value.",

    "account_security vs account_access: the test is whether they believe someone ELSE "
    "is involved. #75 'have been hacked by users, my family has been kicked off' is "
    "account_security. #47 'reactivated my account which has a security check I can't "
    "get past' is account_access -- they are locked out, not compromised, despite the "
    "word 'security' appearing.",

    "#89 'noticed some weird behaviour on my app today, reset my account password just "
    "in case' is account_security. They suspect compromise, and suspicion is the test; "
    "waiting for proof would mean only classifying it correctly after the damage.",

    "billing_charge vs plan_or_family: if money has already moved wrongly it is "
    "billing_charge. If they are arranging a plan going forward and nothing has gone "
    "wrong, it is plan_or_family. #83 'I signed up for a student account but I was "
    "charged $9.99' is billing_charge -- the wrong amount already left their account.",

    "playback_issue vs app_bug: if the app runs and the audio fails, playback_issue. If "
    "the app itself fails, app_bug. #28 'my laptop crashes every time I try to open the "
    "application' is app_bug; #67 'keeps pausing and skipping songs' is playback_issue.",

    "playback_issue vs content_unavailable: the test is whether they think the content "
    "exists and is broken, or think it is gone. #7 'I can't play any of The Weeknd's "
    "songs on my phone, I can via my iPad' is playback_issue -- it plays elsewhere. #73 "
    "'why was take care removed from spotify' is content_unavailable.",

    "how_to vs product_feedback: a question with an answer is how_to; a wish is "
    "product_feedback. #48 'why is there a limit to how many songs I can have in my "
    "library' is how_to -- there is a documented limit and the answer resolves it. #61 "
    "'should have an option to hide individual artists' is product_feedback.",

    "dm_followup only applies when the message carries no request of its own. #3 'hi i "
    "sent you a dm, please take a minute to check it' is dm_followup. A message that "
    "states a problem AND mentions a DM is classified by the problem.",

    "product_feedback covers praise as well as complaints and feature requests. All "
    "three end the same way -- acknowledge, route to the product team if relevant, "
    "nothing to fix for this customer. Splitting praise out would create a class of "
    "about three examples in 100, which is too few to measure.",

    "A message with two genuine asks takes the one the customer would be angriest about "
    "if it went unanswered, and gets tagged is_hard with a note naming the runner-up. "
    "Multi-intent is a real failure mode and hiding it behind a clean label would make "
    "the failure analysis dishonest.",
]


def prompt_block() -> str:
    """The taxonomy rendered for a prompt. Built from the same dicts the labelling guide
    renders from, so the model and the human cannot silently diverge."""
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
