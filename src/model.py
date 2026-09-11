"""
Model loading, with hardware-aware defaults.

THE THREE THINGS THAT BREAK VLM CODE ON COLAB
---------------------------------------------
1. bfloat16 on a T4. The free-tier T4 is Turing architecture and has NO bf16
   support. Half the tutorials online hardcode bf16 and simply crash. We detect
   the GPU capability and pick fp16 or bf16 accordingly.

2. FlashAttention-2 on a T4. Same reason -- requires Ampere (SM 8.0+). We fall
   back to PyTorch's built-in SDPA attention, which works everywhere.

3. Uncapped image resolution. Qwen-VL models use *dynamic* resolution: a large
   image can silently expand to 2000+ tokens. Since attention is quadratic,
   this is the difference between a 2-hour run and a 10-hour run. We set
   min_pixels/max_pixels explicitly and log the resulting token budget.

Point 3 is not just a speed issue -- it is a SCIENTIFIC one. Your zero-shot
baseline and every fine-tuned model must use the SAME token budget, or you are
comparing models that literally saw different amounts of the image.
"""

from __future__ import annotations

import torch
from transformers import AutoProcessor, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Default is Qwen2.5-VL-3B rather than the newer Qwen3-VL, for one specific
# reason: the published zero-shot SLAKE numbers we are validating against
# (~51% open / ~67% closed) were measured on Qwen2.5-VL. Using the same model
# means a mismatch tells us our *harness* is wrong, which is the entire point
# of Step 1. Switch to Qwen3-VL for the headline experiments once the harness
# is trusted.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def _dtype_kwarg() -> str:
    """
    Return the correct precision keyword for the installed transformers.

    transformers < 4.56 uses `torch_dtype`; 4.56+ uses `dtype`. The old name
    still works in new versions but emits a deprecation warning, so we select
    rather than always using one.
    """
    import transformers

    try:
        from packaging.version import parse as _parse
        return "dtype" if _parse(transformers.__version__) >= _parse("4.56.0") \
            else "torch_dtype"
    except Exception:  # noqa: BLE001 - never let a version check break loading
        return "torch_dtype"


def _pick_dtype() -> torch.dtype:
    """bf16 on Ampere and newer (L4, A100), fp16 on Turing (T4)."""
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def _pick_attention() -> str:
    """FlashAttention-2 needs Ampere+; SDPA is the safe universal choice."""
    if not torch.cuda.is_available():
        return "eager"
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            pass
    return "sdpa"


def describe_hardware() -> dict:
    """Print and return a hardware summary. Log this with every run."""
    if not torch.cuda.is_available():
        info = {"gpu": "none (CPU)", "capability": None,
                "dtype": "float32", "attention": "eager"}
    else:
        major, minor = torch.cuda.get_device_capability()
        info = {
            "gpu": torch.cuda.get_device_name(0),
            "capability": f"{major}.{minor}",
            "vram_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            ),
            "dtype": str(_pick_dtype()).replace("torch.", ""),
            "attention": _pick_attention(),
        }
    for k, v in info.items():
        print(f"[hw] {k:12s}: {v}")
    return info


def load_model_and_processor(
    model_id: str = DEFAULT_MODEL_ID,
    load_in_4bit: bool = True,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 768 * 28 * 28,
):
    """
    Load a Qwen-VL model and its processor.

    Parameters
    ----------
    load_in_4bit
        4-bit NF4 quantisation. Cuts VRAM roughly 4x at a modest speed cost.
        Turn it OFF (False) if you have an A100 and want maximum speed.

        IMPORTANT FOR YOUR REPORT: quantisation slightly changes model outputs.
        Whatever you choose, use the SAME setting for the baseline and for
        every fine-tuned evaluation. Mixing them invalidates the comparison.

    min_pixels / max_pixels
        Image token budget. Qwen packs pixels into 28x28 patches, so
        `max_pixels = N * 28 * 28` means roughly N image tokens.

        768 tokens is a reasonable default for medical images: enough to
        resolve anatomy, cheap enough to run the full ablation grid. Treat this
        as a hyperparameter worth an ablation of its own later -- an
        accuracy-vs-token-budget curve is a genuinely useful figure.
    """
    dtype = _pick_dtype()
    attn = _pick_attention()

    # 4-bit quantisation requires a CUDA GPU. bitsandbytes has no usable CPU or
    # Intel-iGPU path, and the failure mode is a confusing kernel error deep in
    # the stack rather than a clear message. Fail loudly and early instead.
    if load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError(
            "load_in_4bit=True requires a CUDA GPU, and none was detected.\n"
            "  -> On Colab: Runtime > Change runtime type > T4 GPU.\n"
            "  -> On a CPU-only machine: this model will not run at usable "
            "speed. Use Colab. (You can still run tests/, src.data --inspect, "
            "and scripts/compare_baselines.py locally without a GPU.)"
        )

    print(f"[model] loading {model_id}")
    print(f"[model] dtype={dtype}  attention={attn}  4bit={load_in_4bit}")
    print(f"[model] image tokens: ~{min_pixels // (28*28)}-{max_pixels // (28*28)}")

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NF4 as used by QLoRA
            bnb_4bit_use_double_quant=True,     # quantise the quant constants too
            bnb_4bit_compute_dtype=dtype,       # matmuls happen in fp16/bf16
        )

    # Qwen2.5-VL and Qwen3-VL use different model classes. Resolving the class
    # from the config keeps this file working across both.
    model_cls = _resolve_model_class(model_id)

    # WHICH KEYWORD SETS THE PRECISION?
    # transformers renamed `torch_dtype` -> `dtype` in 4.56. Passing the wrong
    # one does NOT raise: it is swallowed into the config kwargs, the model
    # loads in float32, and you get an out-of-memory error on a T4 that looks
    # like it has nothing to do with precision. Pick the right key explicitly.
    model = model_cls.from_pretrained(
        model_id,
        attn_implementation=attn,
        quantization_config=quant_config,
        device_map="auto",
        **{_dtype_kwarg(): dtype},
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    # Left padding is REQUIRED for batched generation with decoder-only models.
    # With right padding the model generates from pad tokens and returns junk.
    # This is a classic silent bug: no error, just bad numbers.
    processor.tokenizer.padding_side = "left"

    return model, processor


def _resolve_model_class(model_id: str):
    """
    Pick the right AutoModel class for the checkpoint.

    We try, in order:
      1. Qwen3-VL class      (newest)
      2. Qwen2.5-VL class
      3. Generic AutoModelForImageTextToText  (works for most modern VLMs)

    If you hit 'Unrecognized configuration class', your `transformers` is too
    old for the checkpoint. Upgrade it -- do not downgrade the model.
    """
    import transformers

    lowered = model_id.lower()

    if "qwen3-vl" in lowered and hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        return transformers.Qwen3VLForConditionalGeneration
    if "qwen2.5-vl" in lowered and hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
        return transformers.Qwen2_5_VLForConditionalGeneration
    if "qwen2-vl" in lowered and hasattr(transformers, "Qwen2VLForConditionalGeneration"):
        return transformers.Qwen2VLForConditionalGeneration

    from transformers import AutoModelForImageTextToText
    return AutoModelForImageTextToText
