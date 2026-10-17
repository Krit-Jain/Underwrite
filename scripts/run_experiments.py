#!/usr/bin/env python
"""Run the S0..S5 ablation and report risk-coverage metrics.

Each method is fitted on probe_train only and scored once on test. The probe's
layer, PCA width and regularisation are chosen by cross-validation inside
probe_train.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from underwrite.probe.features import load_features  # noqa: E402
from underwrite.probe.train import run_field  # noqa: E402


def recorded_layers(features_dir: Path) -> list[int]:
    """Read the probe layers straight from a captured bundle's metadata."""
    bundles = sorted(features_dir.glob("*.npz"))
    if not bundles:
        raise SystemExit(f"no feature bundles in {features_dir}")
    _, meta = load_features(bundles[0])
    return list(meta["layers"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="data/features_table.csv")
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--fields", nargs="+", default=["total"])
    parser.add_argument("--out", default="reports/tables/ablation.csv")
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()

    frame = pd.read_csv(args.table)
    layers = recorded_layers(Path(args.features_dir))
    print(f"probe layers available: {layers}")

    tables = []
    for field in args.fields:
        table = run_field(frame, field, args.features_dir, layers, seed=args.seed)
        choice = table.attrs["probe_choice"]
        print(
            f"\n=== {field} ===  probe: layer={choice.layer} "
            f"pca={choice.n_components} C={choice.C} cv_auroc={choice.cv_auroc:.3f}"
        )
        print(table.drop(columns=["field"]).to_string(index=False))
        tables.append(table)

    combined = pd.concat(tables, ignore_index=True)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target, index=False)
    print(f"\nwritten -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
