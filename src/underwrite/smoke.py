"""Load the quantized vision model and extract one document.

This is the Phase 0 smoke test: it proves the model loads in 4-bit, runs on a single
receipt, and reports peak VRAM and wall-clock seconds per document. Heavy imports
(``torch``/``transformers``) are deferred so the rest of the package stays importable
without a GPU toolchain installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from underwrite.config import Config
from underwrite.extract.parse import ParseError, extract_json_object
from underwrite.extract.prompts import SYSTEM_INSTRUCTION, build_extraction_prompt

_SAMPLE_LINES = (
    "SUNRISE TRADERS",
    "42 MG Road, Bengaluru 560001",
    "GSTIN 29ABCDE1234F1Z5",
    "",
    "Item             Qty    Amount",
    "Filter Coffee      2     180.00",
    "Notebook           1     120.00",
    "",
    "Subtotal                 300.00",
    "CGST 9%                   27.00",
    "SGST 9%                   27.00",
    "",
    "TOTAL                    354.00",
    "",
    "Date: 2026-10-17",
)


@dataclass(frozen=True)
class SmokeResult:
    """Outcome of a single-document smoke run."""

    model_id: str
    text: str
    fields: dict[str, Any]
    seconds: float
    peak_vram_mb: float


def _receipt_font(size: int) -> Any:
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_sample_receipt(width: int = 520, height: int = 660) -> Any:
    """Render a synthetic receipt so the smoke test needs no external dataset."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _receipt_font(18)
    y = 24
    for line in _SAMPLE_LINES:
        draw.text((24, y), line, fill="black", font=font)
        y += 28
    return image


def _load_processor_and_model(
    model_id: str,
    revision: str | None,
    compute_dtype: str,
    load_in_4bit: bool,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import AutoModelForImageTextToText as model_class
    except ImportError:  # pragma: no cover - older transformers releases
        from transformers import AutoModelForVision2Seq as model_class

    dtype = getattr(torch, compute_dtype)
    model_kwargs: dict[str, Any] = {"device_map": "auto"}
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = dtype

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = model_class.from_pretrained(model_id, revision=revision, **model_kwargs)
    model.eval()
    return processor, model


def run_smoke_test(
    config: Config,
    image_path: str | Path | None = None,
    max_new_tokens: int | None = None,
) -> SmokeResult:
    """Load the model and extract fields from a single receipt image."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            'PyTorch is not installed. Install the ML extras first: pip install -e ".[ml]"'
        ) from exc

    model_id = config.model.id
    budget = max_new_tokens or config.model.max_new_tokens

    if image_path is None:
        image = render_sample_receipt()
    else:
        from PIL import Image

        image = Image.open(Path(image_path)).convert("RGB")

    processor, model = _load_processor_and_model(
        model_id,
        config.model.revision,
        config.model.compute_dtype,
        config.model.quantization == "nf4",
    )

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_extraction_prompt(config.signals.fields)},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=budget, do_sample=config.model.do_sample)
    seconds = perf_counter() - started

    new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    )

    try:
        fields = extract_json_object(text)
    except ParseError:
        fields = {}

    return SmokeResult(
        model_id=model_id,
        text=text,
        fields=fields,
        seconds=seconds,
        peak_vram_mb=peak_vram_mb,
    )
