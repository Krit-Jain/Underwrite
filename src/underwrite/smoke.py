"""Load the quantized vision model and extract one document.

This is the Phase 0 smoke test: it proves the model loads in 4-bit, runs on a single
receipt, and reports peak VRAM and wall-clock seconds per document. Heavy imports
are deferred, so the rest of the package stays importable without a GPU toolchain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from underwrite.config import Config
from underwrite.extract.parse import ParseError, extract_json_object
from underwrite.extract.vlm import build_messages, generate, load_vision_model

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

    if image_path is None:
        image = render_sample_receipt()
    else:
        from PIL import Image

        image = Image.open(Path(image_path)).convert("RGB")

    processor, model = load_vision_model(config)
    messages = build_messages(image, config.signals.fields)
    budget = max_new_tokens or config.model.max_new_tokens

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = perf_counter()
    text = generate(processor, model, messages, budget, config.model.do_sample)
    seconds = perf_counter() - started

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    )

    try:
        fields = extract_json_object(text)
    except ParseError:
        fields = {}

    return SmokeResult(
        model_id=config.model.id,
        text=text,
        fields=fields,
        seconds=seconds,
        peak_vram_mb=peak_vram_mb,
    )
