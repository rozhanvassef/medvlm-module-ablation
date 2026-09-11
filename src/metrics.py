"""
Metrics for medical VQA.

THE STANDARD PROTOCOL
---------------------
Medical VQA papers report two numbers, never one:

  * CLOSED-ended accuracy  -- questions with a small fixed answer set
                              (overwhelmingly yes/no). Scored by exact match
                              after normalisation.

  * OPEN-ended recall/F1   -- free-form answers ("liver", "chest x-ray",
                              "pleural effusion"). Scored by token overlap,
                              because exact match is unreasonably harsh when
                              the gold answer is "lung" and the model says
                              "the lungs".

If you report a single blended accuracy you cannot compare against published
numbers, and you lose the ability to see the most interesting effect in this
project: fine-tuning typically moves closed and open accuracy by very
different amounts.

WHY RECALL AND NOT JUST F1
--------------------------
Several med-VQA papers report *recall* for open-ended questions (did the model
say the gold tokens?) because a verbose-but-correct answer should not be
punished. F1 punishes verbosity. We compute BOTH and report both -- the gap
between them is itself a measure of how chatty the model is, which is exactly
the "format collapse" effect we are trying to control for.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from .normalize import normalize_answer, tokenize_for_f1


# ---------------------------------------------------------------------------
# Per-example scoring
# ---------------------------------------------------------------------------

def exact_match(pred: str, gold: str) -> float:
    """
    1.0 if normalised strings are identical, else 0.0.

    If the GOLD answer is yes/no we enable negation-cue extraction on the
    prediction (force_yes_no=True). This is what lets a hedged clinical reply
    ("there does not appear to be an acute fracture") score correctly against
    gold "no". It is gated on the gold answer, so it can never fire on an
    open-ended question and manufacture a false match.
    """
    gold_norm = normalize_answer(gold)
    force = gold_norm in {"yes", "no"}
    return float(normalize_answer(pred, force_yes_no=force) == gold_norm)


def token_scores(pred: str, gold: str) -> tuple[float, float, float]:
    """
    Token-overlap precision, recall and F1 (SQuAD-style).

    Returns (precision, recall, f1).
    """
    pred_tokens = tokenize_for_f1(pred)
    gold_tokens = tokenize_for_f1(gold)

    # Degenerate cases: if either side is empty, score 1.0 only if both are.
    if not pred_tokens or not gold_tokens:
        agree = float(pred_tokens == gold_tokens)
        return agree, agree, agree

    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(overlap.values())
    if n_same == 0:
        return 0.0, 0.0, 0.0

    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def contains_gold(pred: str, gold: str) -> float:
    """
    Lenient credit: 1.0 if the normalised gold answer appears as a WHOLE-WORD
    phrase inside the normalised prediction.

    DIAGNOSTIC ONLY -- never report as accuracy. If `contains_gold` is high
    while `exact_match` is low, the model knows the answer but buries it in
    prose: a formatting gap, not a knowledge gap.

    TWO DELIBERATE CHOICES, BOTH FIXES OF EARLIER BUGS
    --------------------------------------------------
    1. WHOLE-WORD MATCHING. A bare `in` test credits "no" inside "NOdule",
       "NOrmal", "aNOmaly", "unkNOwn". On a radiology benchmark where "no" is a
       common gold answer and the model constantly says "normal" and "no
       nodule", that inflates the diagnostic badly.

    2. NO force_yes_no HERE. If we resolved the prediction to a bare yes/no
       first, this function would return exactly what exact_match returns and
       the diagnostic would carry zero information. We match against the
       *unresolved* prediction, so the number means "did the literal gold token
       actually appear".

    CONSEQUENCE: for closed yes/no questions this is now a weaker signal than
    exact_match, by design. The format-gap signal for closed questions is
    `closed_extraction_rate` in the summary below -- use that one.

    NOTE ON PLURALS: matching is done on TOKEN lists, not raw strings, so the
    same plural folding used by token_scores applies here ("the lungs" contains
    "lung"). Doing it on strings would make this metric inconsistent with the
    open-ended scorer sitting next to it.
    """
    gold_tokens = tokenize_for_f1(gold)          # NOT forced to yes/no
    pred_tokens = tokenize_for_f1(pred)
    if not gold_tokens:
        return 0.0
    n = len(gold_tokens)
    for i in range(len(pred_tokens) - n + 1):
        if pred_tokens[i:i + n] == gold_tokens:
            return 1.0
    return 0.0


def needed_yes_no_extraction(pred: str, gold: str) -> float | None:
    """
    For a closed (yes/no gold) question: did the raw answer FAIL to be a bare
    yes/no, forcing the negation-cue extractor to rescue it?

    Returns 1.0 if extraction was needed, 0.0 if the model already answered
    cleanly, and None if the gold answer is not yes/no (question not closed).

    THIS IS THE FORMAT-GAP NUMBER FOR CLOSED QUESTIONS. A high rate means the
    base model is hedging in clinical prose rather than answering. It is the
    single most quotable diagnostic in Step 1, because it separates "the model
    doesn't know" from "the model won't say it plainly" -- and it predicts how
    much of your fine-tuning gain will be format compliance rather than
    medical knowledge.
    """
    gold_norm = normalize_answer(gold)
    if gold_norm not in {"yes", "no"}:
        return None
    plain = normalize_answer(pred)                      # no cue extraction
    forced = normalize_answer(pred, force_yes_no=True)  # with cue extraction
    if plain in {"yes", "no"}:
        return 0.0
    # Extraction changed the outcome into a usable yes/no.
    return 1.0 if forced in {"yes", "no"} else 1.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class MetricAccumulator:
    """Accumulates per-example scores, split by closed/open question type."""

    closed_em: list[float] = field(default_factory=list)
    closed_contains: list[float] = field(default_factory=list)
    open_recall: list[float] = field(default_factory=list)
    open_f1: list[float] = field(default_factory=list)
    open_em: list[float] = field(default_factory=list)
    open_contains: list[float] = field(default_factory=list)

    # Diagnostics that are cheap to collect and very informative.
    pred_lengths: list[int] = field(default_factory=list)
    # Fraction of closed answers that were NOT already a bare yes/no.
    closed_extraction: list[float] = field(default_factory=list)

    def add(self, pred: str, gold: str, answer_type: str) -> None:
        """
        answer_type must be 'closed' or 'open'. See data.py for how this is
        determined -- SLAKE ships an explicit label, VQA-RAD does not and we
        infer it from whether the gold answer is yes/no.
        """
        self.pred_lengths.append(len(str(pred).split()))

        if answer_type == "closed":
            self.closed_em.append(exact_match(pred, gold))
            self.closed_contains.append(contains_gold(pred, gold))
            ext = needed_yes_no_extraction(pred, gold)
            if ext is not None:
                self.closed_extraction.append(ext)
        else:
            _, recall, f1 = token_scores(pred, gold)
            self.open_recall.append(recall)
            self.open_f1.append(f1)
            self.open_em.append(exact_match(pred, gold))
            self.open_contains.append(contains_gold(pred, gold))

    @staticmethod
    def _mean(xs: Iterable[float]) -> float:
        xs = list(xs)
        return round(100.0 * sum(xs) / len(xs), 2) if xs else float("nan")

    def summary(self) -> dict:
        """Percentages, rounded, ready to print or dump to JSON."""
        return {
            "n_closed": len(self.closed_em),
            "n_open": len(self.open_recall),
            # --- headline numbers: these are what you compare to the paper ---
            "closed_accuracy": self._mean(self.closed_em),
            "open_recall": self._mean(self.open_recall),
            # --- secondary ---
            "open_f1": self._mean(self.open_f1),
            "open_exact_match": self._mean(self.open_em),
            # --- diagnostics (do not report as accuracy) ---
            "closed_contains_gold": self._mean(self.closed_contains),
            "open_contains_gold": self._mean(self.open_contains),
            # THE format-gap signal for closed questions. High = model hedges.
            "closed_extraction_rate": self._mean(self.closed_extraction),
            "mean_pred_words": round(
                sum(self.pred_lengths) / len(self.pred_lengths), 2
            ) if self.pred_lengths else float("nan"),
        }


def score_predictions(records: list[dict]) -> dict:
    """
    Score a list of prediction records.

    Each record must have keys: 'prediction', 'answer', 'answer_type'.
    This is the function you call on a saved .jsonl file, so you can re-score
    old runs after changing the normaliser without re-running the GPU.
    """
    acc = MetricAccumulator()
    for r in records:
        acc.add(r["prediction"], r["answer"], r["answer_type"])
    return acc.summary()
