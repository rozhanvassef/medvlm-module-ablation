"""
Dataset exploration: figures and statistics.

WHY THIS IS A MODULE AND NOT JUST NOTEBOOK CELLS
------------------------------------------------
The figures here go in your report. A figure you can only regenerate by
re-running cells in the right order, in a notebook whose state you have since
lost, is a figure you will eventually have to redo by hand at 2am. Every plot
below is a function that takes a dataset name and writes a PNG:

    python -m src.explore --dataset vqa_rad --dataset slake --out figures/

One command, every figure, reproducible from a clean checkout. The notebook
(notebooks/00_dataset_exploration.ipynb) just calls these and shows the result
inline.

WHAT YOU ARE LOOKING FOR
------------------------
This is not decoration. Four things here change decisions you make later:

1. MAJORITY-CLASS BASELINE. If 60% of closed answers are "yes", a model that
   blindly says "yes" scores 60%. That, not zero, is the floor your fine-tuned
   model must clear. Reporting 63% accuracy against a 60% majority class is
   reporting almost nothing.

2. ANSWER CONCENTRATION. If the top 20 open-ended answers cover most of the
   data, the task is closer to classification than to generation, and your
   open-ended recall numbers will look better than the model deserves.

3. IMAGE SIZE SPREAD. Qwen-VL uses dynamic resolution. A wide spread means the
   min_pixels/max_pixels cap in model.py is doing real work, and the cap is a
   scientific control -- every model you compare must see the same budget.

4. WHAT THE IMAGES ACTUALLY LOOK LIKE. Grayscale radiographs are visually
   nothing like the natural photographs SigLIP was pretrained on. That gap is
   the entire premise of the vision-side adaptation hypothesis. Seeing it is
   worth one figure in the report.

NOTE ON SAMPLING
----------------
Text statistics use the FULL split (cheap -- text columns only, no image
decoding). Image statistics use a random SAMPLE, because decoding every image
in PathVQA to measure its width is minutes of wasted compute. Sample size is
reported on every image figure so the reader knows.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import ImageOps

import matplotlib.pyplot as plt

from .data import REGISTRY, classify_answer_type, load_benchmark
from .normalize import normalize_answer

# A restrained palette. Distinguishable in greyscale print, which matters if
# anyone prints your report.
C_CLOSED = "#3b6ea5"
C_OPEN = "#c17f3f"
C_NEUTRAL = "#6b7280"


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# Text statistics -- full split, no image decoding
# ---------------------------------------------------------------------------

def text_stats(name: str, split: str | None = None) -> dict:
    """
    Compute every text-only statistic in a single pass.

    Reads ONLY the question/answer/answer_type columns via select_columns, so
    no images are decoded. On PathVQA this is the difference between two
    seconds and several minutes.
    """
    ds, spec = load_benchmark(name, split=split)

    cols = [spec.question_key, spec.answer_key]
    if spec.answer_type_key and spec.answer_type_key in ds.column_names:
        cols.append(spec.answer_type_key)
    meta = ds.select_columns(cols)

    questions, answers, types = [], [], []
    for row in meta:
        questions.append(str(row[spec.question_key]).strip())
        answers.append(str(row[spec.answer_key]).strip())
        types.append(classify_answer_type(row, spec))

    closed = [a for a, t in zip(answers, types) if t == "closed"]
    openish = [a for a, t in zip(answers, types) if t == "open"]

    # Majority-class baseline: the score a model gets by always guessing the
    # single most common closed answer. THE number to compare against.
    majority_answer, majority_n = ("-", 0)
    if closed:
        majority_answer, majority_n = Counter(
            normalize_answer(a) for a in closed
        ).most_common(1)[0]

    return {
        "name": name,
        "n": len(answers),
        "n_closed": len(closed),
        "n_open": len(openish),
        "pct_closed": 100 * len(closed) / len(answers) if answers else 0.0,
        "majority_answer": majority_answer,
        "majority_baseline": 100 * majority_n / len(closed) if closed else float("nan"),
        "answer_words": [len(a.split()) for a in answers],
        "open_answer_words": [len(a.split()) for a in openish],
        "question_words": [len(q.split()) for q in questions],
        "top_open": Counter(normalize_answer(a) for a in openish).most_common(15),
        "top_all": Counter(normalize_answer(a) for a in answers).most_common(15),
        "first_words": Counter(q.split()[0].lower() for q in questions if q.split()).most_common(12),
        "open_unique": len({normalize_answer(a) for a in openish}),
    }


def print_stats(s: dict) -> None:
    """Human-readable summary. Copy these numbers into your README."""
    print(f"\n{'=' * 66}")
    print(f"  {s['name'].upper()}")
    print("=" * 66)
    print(f"  examples              : {s['n']}")
    print(f"  closed / open         : {s['n_closed']} / {s['n_open']} "
          f"({s['pct_closed']:.1f}% closed)")
    print(f"  unique open answers   : {s['open_unique']}")
    print(f"  mean answer length    : "
          f"{sum(s['answer_words'])/max(len(s['answer_words']),1):.2f} words")
    print(f"  mean question length  : "
          f"{sum(s['question_words'])/max(len(s['question_words']),1):.2f} words")
    print()
    print(f"  >>> MAJORITY-CLASS BASELINE (closed): "
          f"always answer '{s['majority_answer']}' = {s['majority_baseline']:.1f}%")
    print(f"      Your fine-tuned model must beat THIS, not zero.")
    print()
    print("  most common open-ended answers:")
    for ans, cnt in s["top_open"][:8]:
        print(f"      {cnt:5d}  {ans}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_composition(stats: list[dict], out: Path | None = None):
    """Closed/open balance plus the majority-class baseline, per dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))

    names = [s["name"] for s in stats]
    closed = [s["n_closed"] for s in stats]
    openish = [s["n_open"] for s in stats]

    axes[0].barh(names, closed, color=C_CLOSED, label="closed")
    axes[0].barh(names, openish, left=closed, color=C_OPEN, label="open")
    _style(axes[0], "Question type composition", "examples")
    axes[0].legend(fontsize=8, frameon=False)

    base = [s["majority_baseline"] for s in stats]
    axes[1].barh(names, base, color=C_NEUTRAL)
    for i, (b, s) in enumerate(zip(base, stats)):
        axes[1].text(b + 1, i, f"'{s['majority_answer']}'  {b:.1f}%",
                     va="center", fontsize=8)
    axes[1].set_xlim(0, 100)
    _style(axes[1], "Majority-class baseline (closed questions)", "% accuracy")

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[fig] {out}")
    return fig


def fig_answer_lengths(stats: list[dict], out: Path | None = None):
    """
    Answer length distribution.

    This is the figure that justifies the constrained prompt. Gold answers are
    one or two words. A base model replying in 30-word clinical prose is not
    wrong about medicine -- it is wrong about format, and that distinction is
    the spine of this project.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for s, colour in zip(stats, [C_CLOSED, C_OPEN, C_NEUTRAL]):
        lens = [min(w, 15) for w in s["answer_words"]]
        ax.hist(lens, bins=range(0, 17), alpha=0.6, label=s["name"],
                color=colour, edgecolor="white", linewidth=0.6)
    _style(ax, "Gold answer length (capped at 15 words)", "words", "count")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[fig] {out}")
    return fig


def fig_top_answers(stat: dict, out: Path | None = None):
    """Top open-ended answers. Shows how concentrated the label space really is."""
    top = stat["top_open"][:15][::-1]
    if not top:
        return None
    labels = [t[0][:28] for t in top]
    counts = [t[1] for t in top]

    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.28 * len(top) + 1)))
    ax.barh(labels, counts, color=C_OPEN)
    total_open = max(stat["n_open"], 1)
    covered = 100 * sum(counts) / total_open
    _style(ax,
           f"{stat['name']}: top open-ended answers "
           f"(these {len(top)} cover {covered:.0f}% of open questions)",
           "count")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[fig] {out}")
    return fig


def fig_question_openers(stat: dict, out: Path | None = None):
    """First word of each question -- a cheap proxy for question type mix."""
    top = stat["first_words"][:12][::-1]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.barh([t[0] for t in top], [t[1] for t in top], color=C_CLOSED)
    _style(ax, f"{stat['name']}: question opening words", "count")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[fig] {out}")
    return fig


# ---------------------------------------------------------------------------
# Image figures -- these DO decode images, so they sample
# ---------------------------------------------------------------------------

def image_sample(name: str, n: int = 12, seed: int = 0):
    """Return n random (image, question, answer, answer_type) tuples."""
    ds, spec = load_benchmark(name)
    idx = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    out = []
    for row in idx:
        out.append((
            row[spec.image_key],
            str(row[spec.question_key]).strip(),
            str(row[spec.answer_key]).strip(),
            classify_answer_type(row, spec),
        ))
    return out


def fig_image_grid(name: str, n: int = 8, seed: int = 0, out: Path | None = None):
    """
    A grid of real examples with their questions.

    Put this figure in your report. It communicates in one glance what three
    paragraphs of prose cannot: these are grayscale radiographs, they look
    nothing like the natural photographs the vision encoder was pretrained on,
    and the questions are short and clinical.
    """
    samples = image_sample(name, n=n, seed=seed)
    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.4 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, (img, q, a, t) in zip(axes, samples):
        # Letterbox every image into the same square box. Medical images vary
        # wildly in aspect ratio, and without this the grid comes out ragged
        # with titles floating at different heights -- unusable in a report.
        # The grey border is deliberate: radiographs are mostly black, so a
        # black pad would hide where the image actually ends.
        thumb = ImageOps.pad(img.convert("RGB"), (320, 320),
                             color=(228, 228, 232))
        ax.imshow(thumb)
        ax.axis("off")
        q_short = q if len(q) <= 44 else q[:42] + "..."
        colour = C_CLOSED if t == "closed" else C_OPEN
        ax.set_title(f"{q_short}\n[{t}] -> {a[:26]}", fontsize=7.5, pad=4,
                     color=colour)
    for ax in axes[len(samples):]:
        ax.axis("off")

    fig.suptitle(f"{name}: random examples (seed={seed})", fontsize=11)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[fig] {out}")
    return fig


def fig_image_sizes(name: str, n: int = 250, seed: int = 0, out: Path | None = None):
    """
    Image dimensions, from a random sample.

    Relevant because Qwen-VL resolution is dynamic: a large image silently
    expands to more tokens, and attention cost is quadratic in that. A wide
    spread here means the min_pixels/max_pixels cap in model.py is actively
    reshaping most of your inputs -- worth one sentence in the methods section.
    """
    ds, spec = load_benchmark(name)
    sample = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    sizes = [row[spec.image_key].size for row in sample]
    widths = [w for w, _ in sizes]
    heights = [h for _, h in sizes]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].scatter(widths, heights, s=12, alpha=0.5, color=C_CLOSED,
                    edgecolors="none")
    _style(axes[0], f"{name}: image dimensions (n={len(sizes)})",
           "width (px)", "height (px)")

    pixels = [w * h / 1000 for w, h in sizes]
    axes[1].hist(pixels, bins=30, color=C_NEUTRAL, edgecolor="white",
                 linewidth=0.6)
    # 768 image tokens x 28x28 px per patch = the cap set in model.py.
    cap_kpx = 768 * 28 * 28 / 1000
    axes[1].axvline(cap_kpx, color="#b3382c", linestyle="--", linewidth=1.4)
    # Place the label on whichever side has room, so it never clips the frame.
    xmax = axes[1].get_xlim()[1]
    on_right = cap_kpx < xmax * 0.75
    axes[1].text(cap_kpx, axes[1].get_ylim()[1] * 0.92,
                 " max_pixels cap " if on_right else " max_pixels cap ",
                 fontsize=8, color="#b3382c",
                 ha="left" if on_right else "right")
    _style(axes[1], "Total pixels vs the token budget", "kilopixels", "count")

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[fig] {out}")
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_all(names: list[str], out_dir: Path, n_grid: int = 8,
            n_sizes: int = 250, seed: int = 0) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for name in names:
        s = text_stats(name)
        print_stats(s)
        stats.append(s)
        fig_top_answers(s, out_dir / f"{name}_top_answers.png")
        fig_question_openers(s, out_dir / f"{name}_question_openers.png")
        fig_image_grid(name, n=n_grid, seed=seed,
                       out=out_dir / f"{name}_examples.png")
        fig_image_sizes(name, n=n_sizes, seed=seed,
                        out=out_dir / f"{name}_image_sizes.png")
        plt.close("all")

    fig_composition(stats, out_dir / "composition.png")
    fig_answer_lengths(stats, out_dir / "answer_lengths.png")
    plt.close("all")

    print(f"\n[explore] all figures written to {out_dir}/")
    print("[explore] REMEMBER: the majority-class baseline above is the number "
          "your fine-tuned model has to beat.")
    return stats


def main() -> None:
    # Force a headless backend, but ONLY for CLI use. Doing this at import time
    # would clobber a notebook's inline backend -- `from src import explore`
    # would silently stop figures from displaying, and they would go to disk
    # only. Backend selection belongs to the entry point, not the library.
    import matplotlib
    matplotlib.use("Agg", force=True)

    p = argparse.ArgumentParser(description="Explore a medical VQA dataset")
    p.add_argument("--dataset", action="append", choices=list(REGISTRY),
                   help="repeatable, e.g. --dataset vqa_rad --dataset slake")
    p.add_argument("--out", default="figures")
    p.add_argument("--n_grid", type=int, default=8)
    p.add_argument("--n_sizes", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    names = args.dataset or ["vqa_rad", "slake"]
    run_all(names, Path(args.out), args.n_grid, args.n_sizes, args.seed)


if __name__ == "__main__":
    main()
