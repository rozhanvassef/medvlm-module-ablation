"""
Dataset loading for medical VQA.

We support three benchmarks. All are small enough to run entirely on Colab.

  VQA-RAD   ~2.2k QA pairs   radiology (chest/head/abdomen X-ray, CT, MRI)
  SLAKE     ~14k QA pairs    multi-modality, has explicit answer_type labels
  PathVQA   ~32k QA pairs    histopathology slides

A NOTE ON DATASET IDs
---------------------
HuggingFace dataset paths for these benchmarks are community-uploaded and do
occasionally move or get renamed. Each entry below therefore lists SEVERAL
candidate ids and we try them in order. If all candidates fail, the error
message tells you to search the Hub rather than leaving you with a stack trace.

Verify the loaded splits with `python -m src.data --inspect vqa_rad` before you
trust any numbers. Do that once, at the start, not after you've run 40
GPU-hours of experiments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from datasets import load_dataset

from .normalize import is_yes_no


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """Describes how to load and interpret one benchmark."""

    name: str
    candidate_ids: list[str]          # HF hub ids, tried in order
    test_split: str                   # name of the evaluation split
    image_key: str                    # column holding a PIL image
    question_key: str
    answer_key: str
    # Column holding an explicit OPEN/CLOSED label, if the dataset provides one.
    answer_type_key: str | None = None
    # Optional filter, e.g. SLAKE contains both English and Chinese questions.
    # `filter_column` names the single column the filter reads. Naming it lets
    # us pass input_columns= to ds.filter, so HuggingFace never decodes the
    # image column during filtering. Without it, filtering SLAKE decodes ~1000
    # JPEGs to read a two-letter language tag.
    row_filter: Callable[..., bool] | None = None
    filter_column: str | None = None


REGISTRY: dict[str, DatasetSpec] = {
    # -----------------------------------------------------------------------
    # VQA-RAD. Recommended starting point: smallest test split (~450 items),
    # images embedded as PIL objects, no separate archive to download.
    # -----------------------------------------------------------------------
    "vqa_rad": DatasetSpec(
        name="vqa_rad",
        candidate_ids=["flaviagiammarino/vqa-rad"],
        test_split="test",
        image_key="image",
        question_key="question",
        answer_key="answer",
        answer_type_key=None,  # not provided -> inferred from yes/no gold answer
    ),

    # -----------------------------------------------------------------------
    # SLAKE (English subset). This is the benchmark whose published Qwen2.5-VL
    # zero-shot numbers we are checking against: ~51% open / ~67% closed.
    #
    # The canonical release (BoKelvin/SLAKE) ships images in a separate zip and
    # is awkward to load. The mirrors below embed images directly. If none
    # resolve, fall back to the canonical repo and unzip imgs.zip yourself.
    # -----------------------------------------------------------------------
    "slake": DatasetSpec(
        name="slake",
        candidate_ids=[
            "mdwiratathya/SLAKE-vqa-english",
            "BoKelvin/SLAKE",
        ],
        test_split="test",
        image_key="image",
        question_key="question",
        answer_key="answer",
        answer_type_key="answer_type",  # 'OPEN' / 'CLOSED'
        # Some mirrors include Chinese questions under a q_lang column;
        # others (the English-only mirrors) omit it entirely. Applied only if
        # the column is actually present -- see load_benchmark.
        row_filter=lambda q_lang: str(q_lang).lower() == "en",
        filter_column="q_lang",
    ),

    # -----------------------------------------------------------------------
    # PathVQA. Largest of the three; useful later as an out-of-distribution
    # test set (train on VQA-RAD+SLAKE, evaluate here) for the specialisation
    # cost study. For Step 1 you can subsample it.
    # -----------------------------------------------------------------------
    "path_vqa": DatasetSpec(
        name="path_vqa",
        candidate_ids=["flaviagiammarino/path-vqa"],
        test_split="test",
        image_key="image",
        question_key="question",
        answer_key="answer",
        answer_type_key=None,
    ),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_benchmark(name: str, split: str | None = None, limit: int | None = None):
    """
    Load one benchmark and return (hf_dataset, spec).

    `limit` subsamples the FIRST n rows. Use it for smoke tests only -- never
    for a reported number, because the first n rows are not a random sample.
    For a genuine subsample use `.shuffle(seed=...).select(range(n))`.
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Options: {list(REGISTRY)}")

    spec = REGISTRY[name]
    split = split or spec.test_split

    errors = []
    ds = None
    for hub_id in spec.candidate_ids:
        try:
            ds = load_dataset(hub_id, split=split)
            print(f"[data] loaded '{hub_id}' split='{split}' ({len(ds)} rows)")
            break
        except Exception as exc:  # noqa: BLE001 - we want the message, not the type
            errors.append(f"  {hub_id}: {type(exc).__name__}: {exc}")

    if ds is None:
        raise RuntimeError(
            f"Could not load '{name}' from any candidate id.\n"
            + "\n".join(errors)
            + "\n\nSearch https://huggingface.co/datasets for a working mirror "
              "and add its id to REGISTRY in src/data.py."
        )

    if spec.row_filter is not None and spec.filter_column:
        if spec.filter_column in ds.column_names:
            before = len(ds)
            ds = ds.filter(spec.row_filter, input_columns=[spec.filter_column])
            print(f"[data] filter on '{spec.filter_column}' kept "
                  f"{len(ds)}/{before} rows")
        else:
            print(f"[data] no '{spec.filter_column}' column -> "
                  f"filter skipped (mirror is already single-language)")

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"[data] SUBSAMPLED to first {len(ds)} rows (smoke test only!)")

    return ds, spec


def classify_answer_type(row: dict, spec: DatasetSpec) -> str:
    """
    Return 'closed' or 'open' for one row.

    Rule:
      * If the dataset ships an explicit label, trust it.
      * Otherwise infer: a yes/no gold answer means the question was closed.

    The inference rule is the standard one used in the literature for VQA-RAD
    and PathVQA. It is not perfect -- a closed question with answer set
    {left, right} will be misclassified as open -- so STATE THE RULE IN YOUR
    REPORT. Consistency matters more than perfection here: as long as you apply
    the same rule to the baseline and to every fine-tuned model, the comparison
    is valid.
    """
    if spec.answer_type_key and spec.answer_type_key in row:
        label = str(row[spec.answer_type_key]).strip().lower()
        if label in {"closed", "close"}:
            return "closed"
        if label in {"open"}:
            return "open"
        # Unrecognised label -> fall through to inference.

    return "closed" if is_yes_no(row[spec.answer_key]) else "open"


def iter_examples(ds, spec: DatasetSpec):
    """
    Yield normalised example dicts:
        {'idx', 'image', 'question', 'answer', 'answer_type'}

    Keeping this shape uniform across datasets means evaluate.py never needs a
    per-dataset branch.
    """
    for i in range(len(ds)):
        row = ds[i]
        yield {
            "idx": i,
            "image": row[spec.image_key],
            "question": str(row[spec.question_key]).strip(),
            "answer": str(row[spec.answer_key]).strip(),
            "answer_type": classify_answer_type(row, spec),
        }


# ---------------------------------------------------------------------------
# CLI: inspect a dataset before trusting it
# ---------------------------------------------------------------------------

def _inspect(name: str, n: int = 5) -> None:
    """Print schema, class balance and a few examples. RUN THIS FIRST."""
    ds, spec = load_benchmark(name)

    print(f"\n=== {name} ===")
    print(f"rows    : {len(ds)}")
    print(f"columns : {ds.column_names}")

    # Read ONLY the text columns. Iterating ds[i] directly would decode every
    # image in the split just to count yes/no answers -- minutes wasted, and
    # enough memory pressure to kill a free Colab runtime on PathVQA.
    text_cols = [spec.answer_key]
    if spec.answer_type_key and spec.answer_type_key in ds.column_names:
        text_cols.append(spec.answer_type_key)
    meta = ds.select_columns(text_cols)

    types = [classify_answer_type(row, spec) for row in meta]
    n_closed = types.count("closed")
    print(f"closed  : {n_closed} ({100*n_closed/len(types):.1f}%)")
    print(f"open    : {len(types)-n_closed} ({100*(len(types)-n_closed)/len(types):.1f}%)")

    # The majority-class baseline for closed questions. If your model scores
    # below this on closed accuracy, it is worse than always answering "yes".
    closed_answers = [
        str(row[spec.answer_key]).strip().lower()
        for row, t in zip(meta, types) if t == "closed"
    ]
    if closed_answers:
        from collections import Counter
        top, count = Counter(closed_answers).most_common(1)[0]
        print(f"majority-class baseline (closed): '{top}' = "
              f"{100*count/len(closed_answers):.1f}%")

    print(f"\n--- first {n} examples ---")
    # Only NOW do we touch images, and only for n rows.
    for i, ex in enumerate(iter_examples(ds.select(range(min(n, len(ds)))), spec)):
        img = ex["image"]
        size = getattr(img, "size", "?")
        print(f"[{i}] ({ex['answer_type']:6s}) img={size}  "
              f"Q: {ex['question']}\n            A: {ex['answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a medical VQA dataset")
    parser.add_argument("--inspect", required=True, choices=list(REGISTRY))
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    _inspect(args.inspect, args.n)
