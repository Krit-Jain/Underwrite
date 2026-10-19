#!/usr/bin/env python
"""Audit-lane demonstration: real accepted items, then an injected template change.

The clean segment uses the *actual* errors among auto-accepted CORD documents at the
certified threshold. The drifted segment simulates a new client template, where the
error rate among accepted documents jumps. The audit lane should stay quiet through
the clean segment and alarm shortly after the switch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from underwrite.conformal.audit import simulate_audit  # noqa: E402
from underwrite.conformal.ltt import certify  # noqa: E402
from underwrite.probe.features import load_features  # noqa: E402
from underwrite.probe.train import TRAIN_SPLIT, apply_scorers, fit_scorers  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="data/features_table_v2.csv")
    parser.add_argument("--features-dir", default="data/features_promptfix")
    parser.add_argument("--field", default="total")
    parser.add_argument("--method", default="S5 probe + fusion")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="error budget; when omitted the smallest certifiable one is used",
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.30])
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--audit-rate", type=float, default=0.10)
    parser.add_argument("--accepted-fraction", type=float, default=0.30)
    parser.add_argument(
        "--repeats",
        type=int,
        default=7,
        help="repeat the accepted set to lengthen the stream; the correct alarm rule "
        "needs more audits than one test split provides. The default brings the clean "
        "segment to roughly the 600 documents used per segment in run_audit_detection.py, "
        "so the two are comparable",
    )
    parser.add_argument("--drift-error", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--out", default="reports/tables/audit_trajectory.csv")
    args = parser.parse_args()

    frame = pd.read_csv(args.table)
    field_frame = frame[frame["field"] == args.field].reset_index(drop=True)
    train = field_frame[field_frame["split"] == TRAIN_SPLIT].reset_index(drop=True)

    bundles = sorted(Path(args.features_dir).glob("*.npz"))
    _, meta = load_features(bundles[0])
    scorers = fit_scorers(train, args.field, args.features_dir, list(meta["layers"]), seed=args.seed)
    scored = apply_scorers(scorers, field_frame, args.features_dir)

    calibration = scored[scored["split"] == "calibration"]
    candidates = [args.alpha] if args.alpha else args.alphas
    chosen_alpha = None
    result = None
    for candidate in candidates:
        attempt = certify(
            calibration[args.method].to_numpy(dtype=float),
            1 - calibration["correct"].to_numpy(dtype=int),
            candidate,
            args.delta,
        )
        if attempt.certified:
            chosen_alpha, result = candidate, attempt
            break
    if result is None:
        raise SystemExit(f"cannot certify any of {candidates} on this calibration set")
    print(
        f"certified alpha={chosen_alpha}  threshold={result.threshold:.3f} "
        f"coverage={result.coverage:.3f}"
    )

    test = scored[scored["split"] == "test"].reset_index(drop=True)
    # Illustrative operating point: accept the top fraction by score. A *certified*
    # threshold at this calibration size accepts only a handful of documents, far too
    # few to demonstrate a monitoring lane, so the certified coverage is reported
    # alongside rather than used to build the stream.
    cutoff = float(np.quantile(test[args.method], 1.0 - args.accepted_fraction))
    accepted = test[test[args.method] >= cutoff]
    print(
        f"illustrative operating point: accept top {args.accepted_fraction:.0%} "
        f"(certified coverage would be {result.coverage:.3f})"
    )
    clean_errors = np.tile((1 - accepted["correct"].to_numpy(dtype=int)).astype(int), args.repeats)
    clean_rate = float(clean_errors.mean()) if clean_errors.size else 0.0
    print(f"clean segment: {clean_errors.size} accepted (repeated x{args.repeats}), error {clean_rate:.3f}")

    rng = np.random.default_rng(args.seed)
    drifted = (rng.random(clean_errors.size) < args.drift_error).astype(int)
    stream = np.concatenate([clean_errors, drifted])
    switch_at = clean_errors.size

    audit = simulate_audit(
        stream,
        audit_rate=args.audit_rate,
        alpha=chosen_alpha,
        check_every=10,
        seed=args.seed,
    )
    alarm = audit.first_alarm(chosen_alpha)
    print(f"template change injected at position {switch_at}")
    print(f"first alarm at position {alarm}" + ("" if alarm is None else f" (+{alarm - switch_at} after the switch)"))

    trajectory = audit.as_frame()
    trajectory["switch_at"] = switch_at
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(target, index=False)
    print(f"written -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
