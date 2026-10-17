#!/usr/bin/env python
"""Write the document-level split assignment for the CORD corpus."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from underwrite.config import load_config  # noqa: E402
from underwrite.data.cord import CORD_SPLITS, extract_gold, load_cord  # noqa: E402
from underwrite.data.splits import assign_splits, summarize, write_splits  # noqa: E402


def main() -> int:
    config = load_config()
    dataset = load_cord(CORD_SPLITS)

    doc_ids = [
        f"cord_{split}_{index:04d}"
        for split in CORD_SPLITS
        for index, row in enumerate(dataset[split])
        if extract_gold(row["ground_truth"])
    ]

    assignment = assign_splits(
        doc_ids,
        probe_train=config.splits.probe_train,
        calibration=config.splits.calibration,
        test=config.splits.test,
        seed=config.seed,
    )
    path = write_splits(config.data.splits_file, assignment)

    print(f"documents: {len(doc_ids)}")
    for name, count in summarize(assignment).items():
        print(f"  {name:<11} {count:>5}")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
