#!/usr/bin/env python
"""Distribution shift: a CORD-calibrated gate deployed on synthetic Indian invoices.

The scorers are fitted on CORD probe_train only, then applied unchanged to the
synthetic corpus. Two questions are answered: does the certificate survive the
shift, and how many labelled target documents repair it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from underwrite.eval.shift import repair_curve, shift_table  # noqa: E402
from underwrite.probe.features import load_features  # noqa: E402
from underwrite.probe.train import TRAIN_SPLIT, apply_scorers, fit_scorers  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-table", default="data/features_table_v2.csv")
    parser.add_argument("--target-table", default="data/features_table_v2_synthetic.csv")
    parser.add_argument("--source-features", default="data/features_promptfix")
    parser.add_argument("--target-features", default="data/features_v2_synth")
    parser.add_argument("--field", default="total")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["S3 black-box fusion", "S5 probe + fusion"],
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.10, 0.05])
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--ks", nargs="+", type=int, default=[25, 50, 100, 150])
    parser.add_argument(
        "--include-source",
        action="store_true",
        help="pool the source calibration with the k labelled target documents "
        "(default: calibrate on the k target documents alone)",
    )
    parser.add_argument("--n-draws", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--out", default="reports/tables/shift.csv")
    parser.add_argument("--repair-out", default="reports/tables/repair.csv")
    args = parser.parse_args()

    source = pd.read_csv(args.source_table)
    target = pd.read_csv(args.target_table)

    source_field = source[source["field"] == args.field].reset_index(drop=True)
    train = source_field[source_field["split"] == TRAIN_SPLIT].reset_index(drop=True)
    if train.empty:
        raise SystemExit("no CORD probe_train rows for this field")

    bundles = sorted(Path(args.source_features).glob("*.npz"))
    _, meta = load_features(bundles[0])
    layers = list(meta["layers"])

    scorers = fit_scorers(train, args.field, args.source_features, layers, seed=args.seed)
    print(
        f"fitted on {len(train)} CORD probe_train rows; probe layer={scorers.choice.layer} "
        f"pca={scorers.choice.n_components} C={scorers.choice.C}"
    )

    source_scored = apply_scorers(scorers, source_field, args.source_features)
    source_calibration = source_scored[source_scored["split"] == "calibration"].reset_index(drop=True)

    target_field = target[target["field"] == args.field].reset_index(drop=True)
    target_scored = apply_scorers(scorers, target_field, args.target_features)

    print(
        f"source calibration rows: {len(source_calibration)} "
        f"(error {1 - source_calibration['correct'].mean():.3f})   "
        f"target rows: {len(target_scored)} (error {1 - target_scored['correct'].mean():.3f})"
    )

    table = shift_table(
        source_calibration, target_scored, args.methods, args.alphas, args.delta
    )
    print("\n=== shift: CORD-calibrated -> synthetic Indian invoices ===")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    repairs = []
    for method in args.methods:
        for alpha in args.alphas:
            curve = repair_curve(
                source_calibration[method].to_numpy(dtype=float),
                1 - source_calibration["correct"].to_numpy(dtype=int),
                target_scored[method].to_numpy(dtype=float),
                1 - target_scored["correct"].to_numpy(dtype=int),
                args.ks,
                alpha,
                args.delta,
                n_draws=args.n_draws,
                seed=args.seed,
                include_source=args.include_source,
            )
            curve["method"] = method
            repairs.append(curve)
    repair_table = pd.concat(repairs, ignore_index=True)
    print("\n=== repair: labelled target documents needed ===")
    print(repair_table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    for path, frame in ((args.out, table), (args.repair_out, repair_table)):
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target_path, index=False)
        print(f"written -> {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
