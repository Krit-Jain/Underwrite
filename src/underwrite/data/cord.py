"""CORD receipt loader and ground-truth mapping.

CORD ground truth is a JSON string shaped ``{"gt_parse": {...}, "meta": {...}}``.
Only the money fields are annotated:
``gt_parse.total.total_price``, ``gt_parse.sub_total.subtotal_price`` and
``gt_parse.sub_total.tax_price``. There is no vendor or date, so those fields are
never scored on this corpus.

Amounts are Indonesian rupiah: both ``.`` and ``,`` act as thousands separators and
there is no minor unit, so they are parsed with ``decimal_sep=""``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterator

from underwrite.data.normalize import parse_money

DATASET_ID = "naver-clova-ix/cord-v2"
CORD_SPLITS = ("train", "validation", "test")

# Only the fields CORD actually annotates, as paths into ``gt_parse``.
MONEY_PATHS = {
    "total": ("total", "total_price"),
    "subtotal": ("sub_total", "subtotal_price"),
    "tax": ("sub_total", "tax_price"),
}


@dataclass(frozen=True)
class Document:
    doc_id: str
    image: Any
    gold: dict[str, str]
    source: str


def _nested(node: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _canonical_money(raw: Any) -> str | None:
    value = parse_money(raw, decimal_sep="")
    if value is None:
        return None
    return str(value.quantize(Decimal("0.01")))


def extract_gold(ground_truth: str) -> dict[str, str]:
    """Map a CORD ground-truth string to canonical money fields."""
    parsed = json.loads(ground_truth)
    gt_parse = parsed.get("gt_parse", {}) if isinstance(parsed, dict) else {}

    gold: dict[str, str] = {}
    for field, path in MONEY_PATHS.items():
        value = _canonical_money(_nested(gt_parse, *path))
        if value is not None:
            gold[field] = value
    return gold


def load_cord(splits: tuple[str, ...] = CORD_SPLITS) -> dict[str, Any]:
    """Load the CORD dataset splits (downloads and caches on first use)."""
    from datasets import load_dataset

    return {split: load_dataset(DATASET_ID, split=split) for split in splits}


def iter_cord_documents(splits: tuple[str, ...] = CORD_SPLITS) -> Iterator[Document]:
    """Yield documents with stable ids, images and canonical gold fields."""
    dataset = load_cord(splits)
    for split in splits:
        for index, row in enumerate(dataset[split]):
            gold = extract_gold(row["ground_truth"])
            if not gold:
                continue
            yield Document(
                doc_id=f"cord_{split}_{index:04d}",
                image=row["image"],
                gold=gold,
                source="cord",
            )
