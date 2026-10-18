#!/usr/bin/env python
"""Build the flat feature table used by the probe and the baselines.

Joins the extraction traces with document-level splits and the corpus page text,
adds source-grounding support, and writes one row per (document, field) with its
correctness label. Hidden-state vectors are *not* flattened here -- the probe reads
them straight from ``data/features*/<doc_id>.npz`` so PCA can be fitted on the
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
from underwrite.data.synth_invoices import generate_specs, page_tokens  # noqa: E402
from underwrite.signals.grounding import document_grounding  # noqa: E402
from underwrite.signals.teacher_forced import FEATURE_NAMES  # noqa: E402
from underwrite.signals.validators import CHECK_NAMES  # noqa: E402


def cord_tokens(splits: tuple[str, ...] = CORD_SPLITS) -> dict[str, list[str]]:
    """OCR word tokens per document id, matching :func:`iter_cord_documents` ids."""
    from underwrite.signals.grounding import page_tokens as ocr_tokens

    dataset = load_cord(splits)
    tokens: dict[str, list[str]] = {}
    for split in splits:
        for index, row in enumerate(dataset[split]):
            tokens[f"cord_{split}_{index:04d}"] = ocr_tokens(row["ground_truth"])
    return tokens


def synthetic_tokens(count: int, seed: int) -> dict[str, list[str]]:
    """Oracle page text for each generated invoice, keyed by document id."""
    return {spec.doc_id: page_tokens(spec) for spec in generate_specs(count, seed=seed)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("cord", "synthetic"), default="cord")
    parser.add_argument("--traces", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()

    traces_path = Path(args.traces or f"reports/traces/extraction_{args.source}.jsonl")
    out_path = Path(args.out or f"data/features_table_{args.source}.csv")

    config = load_config()
    records = [json.loads(line) for line in open(traces_path, encoding="utf-8")]

    if args.source == "cord":
        split_of = load_splits(config.data.splits_file)
        tokens_by_doc = cord_tokens()
    else:
        # The synthetic set is the shift set: every document is out-of-distribution.
        split_of = {}
        tokens_by_doc = synthetic_tokens(len(records), args.seed)

    rows: list[dict[str, object]] = []
    # The money convention must match the corpus. CORD receipts are Indonesian rupiah:
    # '.' and ',' are thousands separators and there is no minor unit. The synthetic
    # Indian invoices carry paise, so their '.' really is a decimal point. Using the
    # rupiah convention on them turned a matched amount into a mismatch whenever the
    # model returned a JSON number instead of a string.
    decimal_sep = "" if args.source == "cord" else "."

    for record in records:
        doc_id = record["doc_id"]
        extracted = record["extracted"]
        gold = record["gold"]
        tokens = tokens_by_doc.get(doc_id, [])

        grounding = document_grounding(
            extracted, tokens, config.signals.fields, decimal_sep=decimal_sep
        )
        verbalized = record.get("verbalized", {})
        checks = record.get("validators", {})

        for field, truth in gold.items():
            row: dict[str, object] = {
                "doc_id": doc_id,
                "field": field,
                "split": split_of.get(doc_id, "shift"),
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)

    print(f"rows: {len(frame)}  documents: {frame['doc_id'].nunique()}  source: {args.source}")
    print("\nbase accuracy per field:")
    print(frame.groupby("field")["correct"].agg(["mean", "size"]).to_string())
    print(f"\nwritten -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
