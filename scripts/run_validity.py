#!/usr/bin/env python
"""Resplit validity of the certified threshold, comparing the ablations.

Fits the scorers on probe_train once, then repeatedly splits the pooled
calibration+test documents to check that the certificate is not an artefact of one
lucky split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from underwrite.eval.validity import evaluate_methods  # noqa: E402
from underwrite.probe.features import load_features  # noqa: E402
from underwrite.probe.train import score_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="data/features_table_v2.csv")
    parser.add_argument("--features-dir", default="data/features_promptfix")
    parser.add_argument("--field", default="total")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["S3 black-box fusion", "S4 probe", "S5 probe + fusion"],
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.10, 0.05])
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--splits", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--out", default="reports/tables/validity_total.csv")
    args = parser.parse_args()

    frame = pd.read_csv(args.table)
    bundles = sorted(Path(args.features_dir).glob("*.npz"))
    if not bundles:
        raise SystemExit(f"no feature bundles in {args.features_dir}")
    _, meta = load_features(bundles[0])
    layers = list(meta["layers"])

    scored, choice = score_all(frame, args.field, args.features_dir, layers, seed=args.seed)
    pooled = scored[scored["split"].isin(("calibration", "test"))]
    print(
        f"field={args.field}  probe layer={choice.layer} pca={choice.n_components} "
        f"C={choice.C}  pooled rows={len(pooled)}  base error={1 - pooled['correct'].mean():.3f}"
    )

    table = evaluate_methods(
        scored, args.methods, args.alphas, args.delta, n_splits=args.splits, seed=args.seed
    )
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False)
    print(f"\nwritten -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
