# Medical VQA: Module-Wise Adaptation Study

Fine-tuning a small vision-language model into a medical imaging specialist, and
determining **which part of the network actually needs to change**.

**Status:** Step 1 — zero-shot baseline (no training yet)

---

## Research questions

1. **Does domain fine-tuning work?** Zero-shot vs fine-tuned on VQA-RAD, SLAKE, PathVQA.
2. **Which module carries the domain knowledge?** Vision encoder vs projector vs LLM.
3. **What does specialisation cost?** Out-of-domain and general-benchmark degradation.
4. **Does the model know when it doesn't know?** Calibration and abstention.

## Step 1: the go/no-go test

Before training anything, establish that a real gap exists and that the
evaluation harness is trustworthy.

**Reference points** (Qwen2.5-VL-3B, SLAKE, from the literature):

| | zero-shot | supervised specialist |
|---|---|---|
| closed-ended accuracy | ~66.9% | ~89.7% (BiomedCLIP) |
| open-ended recall | ~51.2% | — |

Landing near the zero-shot column validates the harness. The distance to the
right-hand column is the headroom this project sets out to close.

## Hardware

Model evaluation requires a **CUDA GPU** — `bitsandbytes` 4-bit has no usable
CPU or Intel-iGPU path, and `src/model.py` fails loudly rather than limping.
A free Colab T4 is sufficient for every Step 1 run.

Everything else runs anywhere with zero installs:

```bash
python tests/test_normalize_metrics.py                    # stdlib only
python scripts/compare_baselines.py --output_dir outputs/ # stdlib only
```

That split is deliberate: predictions are saved as JSONL, so you can change the
normaliser and re-score months-old runs on a laptop without touching a GPU.

## Repository layout

```
src/
  normalize.py   Answer canonicalisation. The most important file here.
  metrics.py     Closed-ended accuracy + open-ended recall/F1 + diagnostics.
  data.py        Dataset registry, closed/open classification, inspection CLI.
  model.py       Hardware-aware loading (T4-safe dtype and attention).
  prompts.py     Naive / constrained / blind prompt construction.
  evaluate.py    Batched, resumable, deterministic evaluation loop.
  explore.py     Dataset statistics and report figures (CLI-driven).
configs/         One YAML per run. This is the experiment log.
                 Six Step-1 runs: {vqarad,slake} x {naive,constrained,blind}.
tests/           CPU-only unit tests. Run before every push.
scripts/         Results aggregation and the go/no-go decision report.
notebooks/       Thin Colab drivers:
                 00_dataset_exploration.ipynb  (no GPU)
                 01_zero_shot_baseline.ipynb   (T4)
```

## Quickstart

```bash
python tests/test_normalize_metrics.py            # no GPU needed
python -m src.data --inspect vqa_rad              # verify the data
python -m src.explore --dataset vqa_rad --dataset slake --out figures/
python -m src.evaluate --config configs/baseline_vqarad_constrained.yaml --limit 20
python -m src.evaluate --config configs/baseline_slake_constrained.yaml
python scripts/compare_baselines.py
```

Or open `notebooks/01_zero_shot_baseline.ipynb` in Colab.

## Evaluation protocol

Closed-ended and open-ended questions are scored separately, as is standard in
the medical VQA literature. Reporting a single blended accuracy is not
comparable to published work.

- **Closed-ended:** exact match after normalisation.
- **Open-ended:** token-level recall (headline) and F1 (secondary).
- **Diagnostics** (never reported as accuracy):
  - *contains-gold* — whole-word, plural-folded phrase match against the
    unresolved prediction. Meaningful for open-ended questions: a large gap
    versus exact match means the model knows the answer but buries it in prose.
  - *closed-extraction rate* — share of closed answers that were **not** already
    a bare yes/no and had to be resolved by negation-cue extraction. This is the
    format-gap signal for closed questions. (Contains-gold cannot serve that
    role there: once cue extraction runs it is weaker than exact match by
    construction.)

### Three baselines, not one

| Baseline | Prompt | Image | Purpose |
|---|---|---|---|
| naive | bare question | yes | raw out-of-the-box behaviour |
| **constrained** | short-answer instruction | yes | **the number to beat** |
| blind | short-answer instruction | no | language-prior floor |

The naive→constrained gap quantifies how much of the apparent "medical gap" is
really a formatting gap. Comparing a fine-tuned model against the *naive*
baseline would inflate the reported gain, so the constrained run is the
headline baseline.

The blind run measures how much of the benchmark is answerable with no image at
all. Medical VQA benchmarks are partly solvable from question phrasing and
majority-class priors; subtracting this floor is what makes an accuracy claim
meaningful.

### Documented normalisation choices

These affect every number reported and belong in the methods section:

- Only the first sentence of a prediction is scored.
- Articles stripped; punctuation removed except intra-word hyphens.
- When the **gold** answer is yes/no, predictions are resolved via negation
  cues, so hedged clinical prose ("there does not appear to be an acute
  fracture") maps to `no`. Gated on the gold answer, so it can never fire on an
  open-ended question.
- Open-ended token overlap folds trailing plural `s` ("lungs" matches "lung").
- Closed/open split: explicit dataset label where available (SLAKE), otherwise
  inferred from whether the gold answer is yes/no (VQA-RAD, PathVQA).

## Reproducibility

- Greedy decoding, fixed seeds, all RNGs seeded.
- Every output file carries the config and hardware that produced it.
- Predictions saved as JSONL, so scoring changes can be re-applied without a GPU.
- Image token budget pinned via `min_pixels`/`max_pixels` and held constant
  across all runs.

## References

- ChartQA/medical VQA fine-tuning cookbooks: HuggingFace TRL, `2U1/Qwen-VL-Series-Finetune`
- `openmed-labs/synthvision` — LoRA medical VQA with base/fine-tuned comparison
- `ubc-tea/MedVLMBench` — medical VLM evaluation harness
