#!/usr/bin/env python
"""Build the flat feature table used by the probe and the baselines.

Joins the extraction traces with document-level splits and the corpus OCR text,
adds source-grounding support, and writes one row per (document, field) with its
correctness label. Hidden-state vectors are *not* flattened here -- the probe reads
them straight from ``data/features/<doc_id>.npz`` so PCA can be fitted on the
probe-train split only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from underwrite.config import load_config  # noqa: E402
from underwrite.data.cord import CORD_SPLITS, load_cord  # noqa: E402
from underwrite.data.normalize import is_correct  # noqa: E402
from underwrite.data.splits import load_splits  # noqa: E402
from underwrite.signals.grounding import document_grounding, page_tokens  # noqa: E402
from underwrite.signals.teacher_forced import FEATURE_NAMES  # noqa: E402
from underwrite.signals.validators import CHECK_NAMES  # noqa: E402


def corpus_tokens(splits: tuple[str, ...] = CORD_SPLITS) -> dict[str, list[str]]:
    """OCR word tokens per document id, matching :func:`iter_cord_documents` ids."""
    dataset = load_cord(splits)
    tokens: dict[str, list[str]] = {}
    for split in splits:
        for index, row in enumerate(dataset[split]):
            tokens[f"cord_{split}_{index:04d}"] = page_tokens(row["ground_truth"])
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", default="reports/traces/extraction.jsonl")
    parser.add_argument("--out", default="data/features_table.csv")
    args = parser.parse_args()

    config = load_config()
    split_of = load_splits(config.data.splits_file)
    tokens_by_doc = corpus_tokens()

    records = [json.loads(line) for line in open(args.traces, encoding="utf-8")]
    rows: list[dict[str, object]] = []
    missing_ocr = 0

    for record in records:
        doc_id = record["doc_id"]
        extracted = record["extracted"]
        gold = record["gold"]
        tokens = tokens_by_doc.get(doc_id)
        if tokens is None:
            missing_ocr += 1
            tokens = []

        grounding = document_grounding(extracted, tokens, config.signals.fields, decimal_sep="")
        verbalized = record.get("verbalized", {})
        checks = record.get("validators", {})

        for field, truth in gold.items():
            row: dict[str, object] = {
                "doc_id": doc_id,
                "field": field,
                "split": split_of.get(doc_id, "unassigned"),
                "correct": int(is_correct(field, extracted.get(field), truth)),
                "grounding": grounding.get(field, 0.0),
                "verbalized": verbalized.get(field),
            }
            for name in FEATURE_NAMES:
                row[name] = record["fields"].get(field, {}).get(name)
            for check in CHECK_NAMES:
                row[check] = checks.get(check, 0.0)
            rows.append(row)

    frame = pd.DataFrame(rows)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)

    print(f"rows: {len(frame)}  documents: {frame['doc_id'].nunique()}")
    if missing_ocr:
        print(f"WARNING: {missing_ocr} documents had no OCR text")
    print("\nrows per split:")
    print(frame["split"].value_counts().to_string())
    print("\nbase accuracy per field and split:")
    summary = frame.groupby(["split", "field"])["correct"].agg(["mean", "size"])
    print(summary.to_string())
    print(f"\nwritten -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
