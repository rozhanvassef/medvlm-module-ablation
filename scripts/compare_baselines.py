"""
Collect every baseline run into one table and apply the go/no-go rule.

This is the script that answers the actual Step 1 question:
"is there room to improve, and is my harness trustworthy?"

NO THIRD-PARTY DEPENDENCIES. Standard library only, deliberately: this script
reads .meta.json files, so you can run it on a laptop with no GPU, no torch and
no pip installs at all. Download the outputs from Drive, run this locally.

Usage:
    python scripts/compare_baselines.py --output_dir outputs/baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

# Published Qwen2.5-VL zero-shot reference points. These are the numbers your
# harness is being validated against. If your SLAKE constrained run lands far
# from these, the harness is wrong -- fix it before training anything.
PUBLISHED_REFERENCE = {
    "slake": {"closed_accuracy": 66.9, "open_recall": 51.2},
}

# Supervised specialist ceilings from the literature, for context on headroom.
# NOTE FOR YOUR REPORT: these come from models with a CLOSED answer vocabulary
# (they select from a fixed list). A generative model is playing a harder game,
# so treat these as an orientation point, not a target you have promised.
SUPERVISED_CEILING = {
    "slake": {"closed_accuracy": 89.7},    # BiomedCLIP
    "vqa_rad": {"closed_accuracy": 84.2},  # LLaVA-Med
}

COLUMNS = [
    ("dataset", "dataset", 10),
    ("prompt", "prompt", 12),
    ("blind", "blind", 6),
    ("n", "n", 6),
    ("closed_acc", "closed_acc", 11),
    ("open_recall", "open_rec", 9),
    ("open_f1", "open_f1", 8),
    ("closed_contains", "cont_gold", 10),
    ("extraction_rate", "hedge%", 8),
    ("mean_words", "words", 7),
]


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.1f}"
    return str(v)


def collect(output_dir: Path) -> list[dict]:
    rows = []
    for meta_file in sorted(output_dir.glob("*.meta.json")):
        meta = json.loads(meta_file.read_text())
        cfg, m = meta["config"], meta["metrics"]
        rows.append({
            "dataset": cfg["dataset"],
            "prompt": cfg["prompt_style"],
            "blind": bool(cfg["blind"]),
            "n": meta["n_examples"],
            "closed_acc": m.get("closed_accuracy"),
            "open_recall": m.get("open_recall"),
            "open_f1": m.get("open_f1"),
            "closed_contains": m.get("closed_contains_gold"),
            # Older runs predate this metric; tolerate its absence.
            "extraction_rate": m.get("closed_extraction_rate"),
            "mean_words": m.get("mean_pred_words"),
        })
    if not rows:
        raise SystemExit(
            f"No .meta.json files found in {output_dir}.\n"
            "Run src.evaluate first, or point --output_dir somewhere else."
        )
    rows.sort(key=lambda r: (r["dataset"], r["blind"], r["prompt"]))
    return rows


def print_table(rows: list[dict]) -> None:
    header = "  ".join(f"{title:>{w}}" for _, title, w in COLUMNS)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(f"{_fmt(r[key]):>{w}}" for key, _, w in COLUMNS))


def interpret(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("  STEP 1 DECISION REPORT")
    print("=" * 78)

    for dataset in dict.fromkeys(r["dataset"] for r in rows):
        sub = [r for r in rows if r["dataset"] == dataset]
        print(f"\n--- {dataset} ---")

        def pick(prompt, blind=False):
            hits = [r for r in sub if r["prompt"] == prompt and r["blind"] is blind]
            return hits[0] if hits else None

        naive = pick("naive")
        cons = pick("constrained")
        blind = pick("constrained", blind=True)

        if cons is None:
            print("  (no constrained run found -- that is the headline baseline, "
                  "run it before interpreting anything else)")
            continue

        # --- CHECK 1: does the harness reproduce published numbers? ---
        ref = PUBLISHED_REFERENCE.get(dataset)
        if ref:
            d_closed = cons["closed_acc"] - ref["closed_accuracy"]
            d_open = cons["open_recall"] - ref["open_recall"]
            print(f"  [harness check] closed {cons['closed_acc']:.1f} vs published "
                  f"{ref['closed_accuracy']:.1f}  (diff {d_closed:+.1f})")
            print(f"  [harness check] open   {cons['open_recall']:.1f} vs published "
                  f"{ref['open_recall']:.1f}  (diff {d_open:+.1f})")
            if abs(d_closed) <= 10 and abs(d_open) <= 10:
                print("  --> PASS. Harness reproduces the literature. Proceed.")
            else:
                print("  --> INVESTIGATE. >10pt gap. Check normalisation, prompt, "
                      "and the closed/open split before training.")
        else:
            print("  [harness check] no published reference for this dataset. "
                  "SLAKE is the one that validates the harness.")

        # --- CHECK 2: how much of the gap is formatting, not knowledge? ---
        if naive is not None:
            gap = cons["closed_acc"] - naive["closed_acc"]
            print(f"  [format gap]  naive {naive['closed_acc']:.1f} -> constrained "
                  f"{cons['closed_acc']:.1f}  ({gap:+.1f} pts from prompting alone)")
            print(f"                mean answer length {naive['mean_words']:.1f} -> "
                  f"{cons['mean_words']:.1f} words")
            print("                Report the CONSTRAINED number as your baseline. "
                  "Beating the naive one is not a result.")

        # --- CHECK 3: how much is vision actually contributing? ---
        if blind is not None:
            vision_gain = cons["closed_acc"] - blind["closed_acc"]
            print(f"  [vision test] blind {blind['closed_acc']:.1f} -> sighted "
                  f"{cons['closed_acc']:.1f}  (vision contributes {vision_gain:+.1f} pts)")
            if vision_gain < 5:
                print("                WARNING: the model barely uses the image. "
                      "Language priors dominate this benchmark. Report this "
                      "prominently -- it is a real finding.")

        # --- CHECK 4: is there headroom worth chasing? ---
        ceil = SUPERVISED_CEILING.get(dataset)
        if ceil:
            headroom = ceil["closed_accuracy"] - cons["closed_acc"]
            print(f"  [headroom]    {cons['closed_acc']:.1f} now vs "
                  f"~{ceil['closed_accuracy']:.1f} supervised ceiling "
                  f"= {headroom:.1f} pts available")
            if headroom < 5:
                print("                --> STOP. Not enough headroom. Pick a "
                      "different dataset or a smaller base model.")
            else:
                print("                --> GO. Meaningful gap to close.")

        # --- CHECK 5: knowledge gap or formatting gap? ---
        # For CLOSED questions the signal is the hedging rate: how often the raw
        # answer was not already a bare yes/no. contains-gold is NOT usable here
        # (it is weaker than exact match by construction once cue extraction
        # runs), which is why this check reads extraction_rate instead.
        ext = cons.get("extraction_rate")
        if ext is not None and not (isinstance(ext, float) and math.isnan(ext)):
            print(f"  [format diag] {ext:.1f}% of closed answers were NOT a bare "
                  f"yes/no and needed cue extraction")
            if ext > 50:
                print("                The base model hedges on most closed "
                      "questions. Expect a large share of your fine-tuning gain "
                      "to be format compliance, not new medical knowledge. Say "
                      "so explicitly in the report -- it is the honest reading.")
            elif ext < 15:
                print("                The base model already answers concisely, "
                      "so gains will be mostly genuine knowledge. Good news.")

        # For OPEN questions contains-gold is meaningful (whole-word phrase match).
        if cons["open_recall"] is not None and cons.get("open_f1") is not None:
            spread = cons["open_recall"] - cons["open_f1"]
            print(f"  [verbosity]   open recall {cons['open_recall']:.1f} vs "
                  f"F1 {cons['open_f1']:.1f}  (spread {spread:+.1f})")
            if spread > 15:
                print("                Large spread = correct but wordy answers. "
                      "Same format story, on the open-ended side.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs/baseline")
    args = p.parse_args()

    out = Path(args.output_dir)
    rows = collect(out)

    print("\n=== ALL BASELINE RUNS ===")
    print_table(rows)
    interpret(rows)

    csv_path = out / "baseline_summary.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
