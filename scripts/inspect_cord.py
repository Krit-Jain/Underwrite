#!/usr/bin/env python
"""Summarize the CORD corpus: split sizes and per-field ground-truth coverage."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from underwrite.data.cord import CORD_SPLITS, DATASET_ID, extract_gold, load_cord  # noqa: E402


def main() -> int:
    dataset = load_cord(CORD_SPLITS)

    print(f"dataset: {DATASET_ID}")
    for split in CORD_SPLITS:
        print(f"  {split:<11} {len(dataset[split]):>5} documents")

    coverage: Counter[str] = Counter()
    total = 0
    for split in CORD_SPLITS:
        for row in dataset[split]:
            total += 1
            for field in extract_gold(row["ground_truth"]):
                coverage[field] += 1

    print(f"\nscorable documents: {total}")
    print("ground-truth field coverage:")
    for field, count in coverage.most_common():
        print(f"  {field:<9} {count:>5}  ({100 * count / total:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
