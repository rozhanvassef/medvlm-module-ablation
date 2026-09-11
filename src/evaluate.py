"""
Zero-shot evaluation loop.

DESIGN REQUIREMENTS THIS FILE SATISFIES
---------------------------------------
1. RESUMABLE. Colab sessions die without warning. Predictions are appended to a
   .jsonl file as they are produced, and a re-run skips indices already present.
   You will need this. Assume every long run gets interrupted at least once.

2. DETERMINISTIC. Greedy decoding (do_sample=False) plus a fixed seed. Two runs
   of the same config must produce identical predictions, otherwise you cannot
   attribute a difference to your intervention.

3. SEPARATED FROM SCORING. The loop writes raw model outputs; metrics.py scores
   them afterwards. This means you can change the normaliser and re-score in
   two seconds without touching the GPU. Do not fuse generation and scoring.

4. LOGS ITS OWN CONFIG. Every output file carries the exact settings that
   produced it. Six weeks from now you will not remember whether run 14 used
   768 or 1024 image tokens.

Usage
-----
    python -m src.evaluate --config configs/baseline_vqarad_constrained.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from .data import iter_examples, load_benchmark
from .metrics import score_predictions
from .model import describe_hardware, load_model_and_processor
from .prompts import build_messages


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix every RNG we touch. Greedy decoding makes this mostly belt-and-braces,
    but dataset shuffling and any future sampling depend on it."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_done_indices(path: Path) -> set[int]:
    """Read an existing .jsonl and return the example indices already scored."""
    if not path.exists():
        return set()
    done = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["idx"])
            except (json.JSONDecodeError, KeyError):
                # A truncated final line is expected if the session was killed
                # mid-write. Skip it; that index will simply be redone.
                continue
    if done:
        print(f"[resume] found {len(done)} completed examples in {path.name}")
    return done


def read_records(path: Path) -> list[dict]:
    """Read all valid records back for scoring."""
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batch(model, processor, batch: list[dict], cfg: dict) -> list[str]:
    """
    Run one batch through the model and return decoded answer strings.

    The processing pipeline is:
        messages -> chat template text -> processor(text + images) -> generate

    `process_vision_info` from qwen_vl_utils is what applies the
    min_pixels/max_pixels budget set on the processor. If you build the image
    tensors yourself you will silently bypass that cap.
    """
    from qwen_vl_utils import process_vision_info

    all_messages = [
        build_messages(
            image=ex["image"],
            question=ex["question"],
            answer_type=ex["answer_type"],
            style=cfg["prompt_style"],
            blind=cfg["blind"],
        )
        for ex in batch
    ]

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in all_messages
    ]

    if cfg["blind"]:
        image_inputs = None
    else:
        image_inputs, _ = process_vision_info(all_messages)

    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated = model.generate(
        **inputs,
        max_new_tokens=cfg["max_new_tokens"],
        do_sample=False,          # greedy -> deterministic
        temperature=None,         # explicitly unset; avoids a HF warning
        top_p=None,
        top_k=None,
    )

    # generate() returns prompt + completion. Strip the prompt off each row.
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated)
    ]

    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: dict) -> dict:
    """Execute one evaluation config and return the metric summary."""
    set_seed(cfg["seed"])

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filename encodes the config so runs never overwrite each other.
    tag = (
        f"{cfg['dataset']}"
        f"_{cfg['model_id'].split('/')[-1]}"
        f"_{cfg['prompt_style']}"
        f"{'_blind' if cfg['blind'] else ''}"
        f"_seed{cfg['seed']}"
    )
    pred_path = out_dir / f"{tag}.jsonl"
    meta_path = out_dir / f"{tag}.meta.json"

    hw = describe_hardware()

    ds, spec = load_benchmark(
        cfg["dataset"], split=cfg.get("split"), limit=cfg.get("limit")
    )
    examples = list(iter_examples(ds, spec))

    done = load_done_indices(pred_path)
    todo = [ex for ex in examples if ex["idx"] not in done]
    print(f"[eval] {len(todo)} examples to run ({len(done)} already done)")

    if todo:
        model, processor = load_model_and_processor(
            model_id=cfg["model_id"],
            load_in_4bit=cfg["load_in_4bit"],
            min_pixels=cfg["min_pixels_tokens"] * 28 * 28,
            max_pixels=cfg["max_pixels_tokens"] * 28 * 28,
        )

        bs = cfg["batch_size"]
        start = time.time()

        # Append mode: safe to call repeatedly after a disconnect.
        with pred_path.open("a") as fh:
            for i in tqdm(range(0, len(todo), bs), desc=tag):
                batch = todo[i:i + bs]
                try:
                    outputs = generate_batch(model, processor, batch, cfg)
                except torch.cuda.OutOfMemoryError:
                    # Most likely cause: one unusually large image in the batch.
                    # Retry the batch one example at a time rather than dying.
                    torch.cuda.empty_cache()
                    print("\n[eval] OOM -> retrying batch at batch_size=1")
                    outputs = []
                    for ex in batch:
                        outputs.extend(generate_batch(model, processor, [ex], cfg))

                for ex, pred in zip(batch, outputs):
                    fh.write(json.dumps({
                        "idx": ex["idx"],
                        "question": ex["question"],
                        "answer": ex["answer"],
                        "answer_type": ex["answer_type"],
                        "prediction": pred.strip(),
                    }) + "\n")
                fh.flush()      # survive a hard kill
                os.fsync(fh.fileno())

        elapsed = time.time() - start
        print(f"[eval] generated {len(todo)} answers in {elapsed/60:.1f} min "
              f"({elapsed/max(len(todo),1):.2f} s/example)")

    # ---- score ----
    records = read_records(pred_path)
    summary = score_predictions(records)

    meta = {
        "config": cfg,
        "hardware": hw,
        "n_examples": len(records),
        "metrics": summary,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n{'='*60}\n  {tag}\n{'='*60}")
    for k, v in summary.items():
        print(f"  {k:24s}: {v}")
    print(f"\n[eval] predictions -> {pred_path}")
    print(f"[eval] metrics     -> {meta_path}")

    return summary


DEFAULTS = {
    "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "dataset": "vqa_rad",
    "split": None,
    "limit": None,
    "prompt_style": "constrained",
    "blind": False,
    "load_in_4bit": True,
    "min_pixels_tokens": 256,
    "max_pixels_tokens": 768,
    "batch_size": 4,
    "max_new_tokens": 32,
    "seed": 0,
    "output_dir": "outputs/baseline",
}


def load_config(path: str | None, overrides: dict) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        with open(path) as fh:
            cfg.update(yaml.safe_load(fh) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Zero-shot medical VQA baseline")
    p.add_argument("--config", type=str, default=None)
    # CLI overrides so you can sweep without writing a YAML for every variant.
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--model_id", type=str, default=None)
    p.add_argument("--prompt_style", type=str, default=None,
                   choices=["naive", "constrained"])
    p.add_argument("--blind", action="store_true", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config, vars(args))
    cfg.pop("config", None)
    run(cfg)


if __name__ == "__main__":
    main()
