"""
Unit tests for the normaliser and metrics. NO GPU, NO DOWNLOADS.

Run this locally on your laptop before every push:
    python tests/test_normalize_metrics.py

These tests encode the exact failure mode that ruins medical VQA evaluations:
a base model that answers correctly in hedged clinical prose being scored as
wrong. If these pass, your baseline number can be trusted.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.normalize import normalize_answer, is_yes_no
from src.metrics import (exact_match, token_scores, contains_gold,
                         needed_yes_no_extraction, score_predictions)

print("=== THE CRITICAL CASES (hedged clinical prose vs yes/no gold) ===")
crit = [
    ("Based on the radiograph provided, there does not appear to be an acute fracture.", "no"),
    ("Yes, there is evidence of pneumonia in the right lower lobe.", "yes"),
    ("The lungs appear clear and unremarkable.", "no"),
    ("There is no evidence of pleural effusion.", "no"),
    ("The image shows a large mass in the left hemisphere.", "yes"),
    ("I cannot determine this from the image.", "no"),
    ("Yes.", "yes"),
    ("No.", "no"),
]
ok = True
for raw, gold in crit:
    got = exact_match(raw, gold)
    if got != 1.0: ok = False
    print(f"  [{'OK ' if got==1.0 else 'FAIL'}] gold={gold:3s} <- {raw[:62]}")

print("\n=== open-ended must NOT be collapsed to yes/no ===")
opens = [
    ("no abnormality", "no abnormality", 1.0),
    ("the lungs",      "lung",           None),
    ("liver",          "kidney",         0.0),
]
for pred, gold, want in opens:
    em = exact_match(pred, gold)
    _, r, f = token_scores(pred, gold)
    print(f"  pred={pred!r:18s} gold={gold!r:16s} EM={em}  recall={r:.2f} f1={f:.2f}")
    if want is not None and em != want: ok = False

print("\n=== contains_gold must use WHOLE WORDS ===")
# Regression test. A bare substring check credits "no" inside "NOdule",
# "NOrmal", "aNOmaly", "unkNOwn" -- all extremely common radiology outputs.
# That silently inflated the knowledge-vs-format diagnostic.
for pred, gold, want in [
    ("nodule",                   "no",     0.0),
    ("normal",                   "no",     0.0),
    ("anomaly",                  "no",     0.0),
    ("the diagnosis is unknown", "no",     0.0),
    ("livermore",                "liver",  0.0),
    ("no",                       "no",     1.0),
    ("it shows the liver",       "liver",  1.0),
]:
    got = contains_gold(pred, gold)
    if got != want: ok = False
    print(f"  [{'OK ' if got==want else 'FAIL'}] contains_gold({pred!r:26s},{gold!r:8s}) = {got}  (want {want})")

print("\n=== hedging rate: did the answer need cue extraction? ===")
# 0.0 = model already answered cleanly. 1.0 = model hedged and had to be
# rescued. The mean of this over closed questions IS the format-gap number.
for pred, gold, want in [
    ("Yes.",                                     "yes", 0.0),
    ("No.",                                      "no",  0.0),
    ("There does not appear to be a fracture.",  "no",  1.0),
    ("The lungs are clear and unremarkable.",    "no",  1.0),
    ("liver",                                    "liver", None),   # not closed
]:
    got = needed_yes_no_extraction(pred, gold)
    if got != want: ok = False
    print(f"  [{'OK ' if got==want else 'FAIL'}] {pred[:44]!r:46s} -> {got}  (want {want})")

print("\n=== plural folding ===")
_, r, f = token_scores("the lungs are clear", "lung")
print(f"  'the lungs are clear' vs 'lung' -> recall={r:.2f} f1={f:.2f}  (recall must be 1.00)")
if r != 1.0: ok = False

print("\n=== closed/open classification of GOLD answers ===")
for g in ["yes", "no", "liver", "chest x-ray", "no abnormality"]:
    print(f"  gold={g!r:18s} -> {'closed' if is_yes_no(g) else 'open'}")

print("\n=== aggregation smoke test ===")
recs = [
    {"prediction": "Based on the image, there does not appear to be a fracture.", "answer": "no", "answer_type": "closed"},
    {"prediction": "Yes, clearly present.", "answer": "yes", "answer_type": "closed"},
    {"prediction": "The lungs.", "answer": "lung", "answer_type": "open"},
    {"prediction": "I think it is the heart.", "answer": "liver", "answer_type": "open"},
]
for k, v in score_predictions(recs).items():
    print(f"  {k:24s}: {v}")

print("\n" + ("ALL TESTS PASSED" if ok else "*** FAILURES REMAIN ***"))
