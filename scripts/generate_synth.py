#!/usr/bin/env python
"""Generate synthetic Indian GST invoices with their ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from underwrite.data.synth_invoices import generate_batch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    truth = generate_batch(
        args.count,
        args.out,
        seed=args.seed,
        augment_images=not args.no_augment,
    )
    print(f"generated {args.count} invoices -> {Path(args.out) / 'images'}")
    print(f"ground truth -> {truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
