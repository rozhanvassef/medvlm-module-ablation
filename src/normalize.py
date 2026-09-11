"""
Answer normalisation.

WHY THIS FILE MATTERS MORE THAN ANY OTHER IN STEP 1
---------------------------------------------------
A general-purpose VLM asked "Is there a fracture?" replies with something like:

    "Based on the radiograph provided, there does not appear to be an
     acute fracture. However, clinical correlation is recommended."

The ground-truth answer is "no".

A naive string comparison scores that as WRONG. If you evaluate that way your
"zero-shot baseline" collapses toward 0%, you fine-tune, the number jumps to
80%, and you have measured nothing except the model learning to stop hedging.
That is a formatting artefact, not a research finding.

The published baselines we validate against (~67% closed / ~51% open on SLAKE
for Qwen2.5-VL) were produced with normalisation like the below.

DESIGN RULES
------------
1. Normalisation is applied identically to prediction and ground truth.
2. Every heuristic here is a documented choice that affects your published
   numbers.

DOCUMENTED CHOICES (copy these into your methods section):
  * Only the first sentence of a prediction is scored.
  * Articles are stripped; punctuation removed except intra-word hyphens.
  * For questions whose GOLD answer is yes/no, predictions are mapped to
    yes/no using negation cues. This is applied only when the gold answer is
    yes/no, so it can never turn an open-ended answer into a spurious match.
  * For open-ended token overlap, trailing plural 's' is stripped so that
    "lungs" matches "lung".
"""

from __future__ import annotations

import re
import string

# ---------------------------------------------------------------------------
# Canonical yes / no surface forms
# ---------------------------------------------------------------------------
_YES_FORMS = {
    "yes", "yeah", "yep", "y", "true", "correct", "affirmative",
    "yes it is", "yes there is", "yes there are", "it is", "there is",
    "present", "positive", "yes it does", "it does",
}

_NO_FORMS = {
    "no", "nope", "n", "false", "incorrect", "negative",
    "no it is not", "no there is not", "no there are not", "it is not",
    "there is not", "there is no", "absent", "not present", "no it does not",
    "it does not", "none",
}

# ---------------------------------------------------------------------------
# Yes/no extraction from hedged clinical prose
# ---------------------------------------------------------------------------
# Applied ONLY when the gold answer is yes/no. Clinical language expresses
# negation in a small, stereotyped set of ways; these cover most of them.
#
# Order matters: negation is checked FIRST, because "yes, there is no evidence
# of ..." must resolve to NO. Hedged negatives are the dominant failure mode
# for un-finetuned models on radiology questions.
_NEGATION_PATTERNS = [
    r"\bdoes not\b", r"\bdo not\b", r"\bdid not\b", r"\bdoesn't\b", r"\bdon't\b",
    r"\bis not\b", r"\bare not\b", r"\bisn't\b", r"\baren't\b",
    r"\bcannot\b", r"\bcan not\b", r"\bcan't\b",
    r"\bno\b", r"\bnot\b", r"\bnone\b", r"\bnegative\b",
    r"\babsent\b", r"\bwithout\b", r"\bfree of\b", r"\bunremarkable\b",
    r"\bno evidence\b", r"\bno sign\b",
]

_AFFIRMATION_PATTERNS = [
    r"^yes\b", r"\bthere is\b", r"\bthere are\b", r"\bis present\b",
    r"\bare present\b", r"\bshows\b", r"\bdemonstrates\b", r"\bpositive\b",
    r"\bevidence of\b", r"\bconsistent with\b", r"\bappears to be\b",
    r"\bindicates\b", r"\bvisible\b",
]

_NEG_RE = re.compile("|".join(_NEGATION_PATTERNS), flags=re.IGNORECASE)
_AFF_RE = re.compile("|".join(_AFFIRMATION_PATTERNS), flags=re.IGNORECASE)

# Leading filler models emit before the real answer.
_LEADING_FILLER = re.compile(
    r"^(the\s+)?(answer\s+is|answer:|it\s+is|this\s+is|i\s+see|"
    r"based\s+on\s+the\s+(image|radiograph|scan|x-ray)[,\s]*|"
    r"in\s+the\s+image[,\s]*|the\s+image\s+shows|"
    r"the\s+(image|scan)\s+is)\s+",
    flags=re.IGNORECASE,
)

_ARTICLES = {"a", "an", "the"}


def _strip_punctuation(text: str) -> str:
    """Remove punctuation but keep intra-word hyphens ('x-ray', 't2-weighted')."""
    keep = {"-"}
    return "".join(ch for ch in text if ch not in set(string.punctuation) - keep)


def _first_sentence(text: str) -> str:
    """First line, then first sentence. Medical VQA answers are short."""
    t = text.split("\n")[0]
    return re.split(r"(?<=[.!?])\s", t)[0]


def normalize_answer(text: str, force_yes_no: bool = False) -> str:
    """
    Canonicalise a free-text answer.

    Parameters
    ----------
    force_yes_no
        Set True ONLY when the gold answer is yes/no. Enables negation-cue
        extraction, which resolves hedged clinical phrasing such as
        "there does not appear to be an acute fracture" -> "no".

        This flag is what makes the un-finetuned baseline a fair comparison.
        Never enable it for open-ended questions: it would collapse legitimate
        answers like "no abnormality" into a bare "no".
    """
    if text is None:
        return ""

    t = re.sub(r"\s+", " ", str(text).strip().lower())
    t = _first_sentence(t)

    # Keep the pre-filler text: cue words we need for yes/no extraction
    # ("the image SHOWS a mass") live inside the filler we are about to strip.
    t_full = t

    for _ in range(2):  # two passes: "the answer is it is yes"
        t = _LEADING_FILLER.sub("", t).strip()

    cleaned = _strip_punctuation(t)
    tokens = [w for w in cleaned.split() if w not in _ARTICLES]
    result = " ".join(tokens).strip()

    # Exact surface-form match first: cheap and unambiguous.
    if result in _YES_FORMS:
        return "yes"
    if result in _NO_FORMS:
        return "no"

    if force_yes_no:
        # Match against the FULL first sentence, not the filler-stripped one:
        # "the image shows a mass" loses its 'shows' cue after stripping.
        # Negation is checked before affirmation -- see note above.
        if _NEG_RE.search(t_full):
            return "no"
        if _AFF_RE.search(t_full):
            return "yes"
        # Unresolvable -> return cleaned text, which will score as wrong.
        # That is the honest outcome: the model did not answer the question.

    return result


def _singularize(token: str) -> str:
    """
    Crude plural stripping, for token-overlap scoring only.

    Handles the common case ('lungs' -> 'lung') while protecting short words
    and the -ss ending ('mass' must not become 'mas').
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize_for_f1(text: str) -> list[str]:
    """Token list for open-ended overlap scoring, with plurals folded."""
    return [_singularize(w) for w in normalize_answer(text).split()]


def is_yes_no(answer: str) -> bool:
    """True if the GROUND-TRUTH answer is yes/no. Drives the closed/open split."""
    return normalize_answer(answer) in {"yes", "no"}
