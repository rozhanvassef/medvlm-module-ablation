"""
Prompt templates.

WHY THERE ARE TWO PROMPTS AND NOT ONE
-------------------------------------
This is the most important design decision in Step 1, so read this carefully.

You could run the base model with a plain question and call the resulting score
"the zero-shot baseline". But that number would be MISLEADING, because it
conflates two very different deficits:

    (a) the model does not know medicine
    (b) the model knows the answer but replies in the wrong FORMAT

Deficit (b) is trivially fixable with a better prompt. If your baseline uses a
naive prompt and your fine-tuned model has learned the format, you will claim a
+30 point improvement of which maybe half is just "I told it to be brief".

A reviewer will spot this immediately. So we measure both:

    NAIVE       -- just the question. Shows raw out-of-the-box behaviour.
    CONSTRAINED -- explicitly asks for a short answer, and for yes/no questions
                   restricts the output space. This is the HONEST baseline:
                   the best the base model can do without any weight updates.

Your headline claim must be:  fine-tuned  vs  CONSTRAINED baseline.
The naive number goes in the report as a secondary row, and the gap between
naive and constrained is itself a result worth one sentence: it quantifies how
much of the "medical VLM gap" is really a formatting gap.

This design directly addresses the failure mode found in the SynthVision work,
where a model got better at medicine but scored worse because its answer style
drifted away from what the benchmark expected.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_CONSTRAINED = (
    "You are a medical imaging assistant answering questions about medical "
    "images. Answer with the shortest possible phrase. Do not explain, do not "
    "add caveats, do not use full sentences. For yes/no questions answer "
    "exactly 'yes' or 'no'."
)

SYSTEM_NAIVE = None  # no system prompt at all


# ---------------------------------------------------------------------------
# User-turn templates
# ---------------------------------------------------------------------------

def _naive_user_text(question: str, answer_type: str) -> str:
    """Raw question, no formatting guidance. Baseline for 'out of the box'."""
    return question


def _constrained_user_text(question: str, answer_type: str) -> str:
    """
    Question plus a terse format instruction.

    Note we branch on answer_type. This is legitimate and standard: the
    evaluation protocol already knows whether a question is closed or open (it
    has to, in order to score it), so telling the model the same thing is not
    leaking the answer -- only the answer *space*. Published med-VQA baselines
    are computed this way.
    """
    if answer_type == "closed":
        return f"{question}\nAnswer with one word: yes or no."
    return f"{question}\nAnswer with a single word or short phrase."


PROMPT_STYLES = {
    "naive": (SYSTEM_NAIVE, _naive_user_text),
    "constrained": (SYSTEM_CONSTRAINED, _constrained_user_text),
}


def build_messages(
    image,
    question: str,
    answer_type: str,
    style: str = "constrained",
    blind: bool = False,
) -> list[dict]:
    """
    Build a Qwen-style chat message list.

    Parameters
    ----------
    blind
        If True, the image is OMITTED entirely. This gives you the
        language-prior baseline: how well can the model answer medical
        questions with no image at all?

        This is one of the highest-value experiments in the whole project and
        it costs nothing extra to run now. Medical VQA benchmarks are known to
        be partly answerable from question phrasing alone ("Is the image a CT
        scan?" is guessable; yes/no questions have a majority class). If your
        blind model scores 55% on closed questions, then a sighted model
        scoring 67% is only really contributing 12 points of vision.

        Report this. Almost nobody does, and it makes your evaluation look far
        more careful than the average submission.
    """
    if style not in PROMPT_STYLES:
        raise KeyError(f"Unknown prompt style '{style}'. Options: {list(PROMPT_STYLES)}")

    system_prompt, user_text_fn = PROMPT_STYLES[style]
    text = user_text_fn(question, answer_type)

    content: list[dict] = []
    if not blind:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": text})

    messages: list[dict] = []
    if system_prompt is not None:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        })
    messages.append({"role": "user", "content": content})

    return messages
