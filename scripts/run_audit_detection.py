#!/usr/bin/env python
"""Evaluate the audit lane: false-alarm rate and detection delay.

Answers the two questions that decide whether the alarm is trustworthy:

* on a clean stream whose true error rate is *at or below* alpha, how often does it
  alarm anyway?
* once a template change pushes the error rate up, how many accepted documents pass
  before it notices?

Both are measured over many seeds, and the numbers come straight from the script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from underwrite.conformal.audit import injected_drift, simulate_audit  # noqa: E402

# The audit selection must not replay the stream generator's random draws.
AUDIT_SEED_OFFSET = 1_000_003


def clean_stream_alarms(alpha, error_rate, seeds, length, audit_rate, budget):
    alarms = 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        stream = (rng.random(length) < error_rate).astype(int)
        result = simulate_audit(
            stream,
            audit_rate=audit_rate,
            alpha=alpha,
            false_alarm_budget=budget,
            check_every=10,
            seed=seed + AUDIT_SEED_OFFSET,
        )
        alarms += result.first_alarm(alpha) is not None
    return alarms / len(seeds)


def detection_delays(alpha, seeds, clean_length, drift_length, drift_error, audit_rate, budget):
    delays = []
    missed = 0
    for seed in range(seeds):
        stream = injected_drift(clean_length, drift_length, 0.01, drift_error, seed=seed)
        result = simulate_audit(
            stream,
            audit_rate=audit_rate,
            alpha=alpha,
            false_alarm_budget=budget,
            check_every=10,
            seed=seed + AUDIT_SEED_OFFSET,
        )
        alarm = result.first_alarm(alpha)
        if alarm is None:
            missed += 1
        else:
            delays.append(max(0, alarm - clean_length))
    if not delays:
        return float("nan"), float("nan"), 1.0
    return float(np.median(delays)), float(np.percentile(delays, 90)), missed / seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seeds", type=int, default=500)
    parser.add_argument("--length", type=int, default=600)
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--out", default="reports/tables/audit_detection.csv")
    args = parser.parse_args()

    seeds = range(args.seeds)
    rows = []

    for error_rate in (0.02, 0.04, 0.06, 0.08):
        rate = clean_stream_alarms(
            args.alpha, error_rate, seeds, args.length, 0.10, args.budget
        )
        rows.append(
            {
                "scenario": "clean",
                "true_error": error_rate,
                "audit_rate": 0.10,
                "false_alarm_rate": rate,
                "median_delay": float("nan"),
                "p90_delay": float("nan"),
                "missed_rate": float("nan"),
                "seeds": args.seeds,
            }
        )

    for audit_rate in (0.05, 0.10, 0.20):
        median, p90, missed = detection_delays(
            args.alpha, args.seeds, 500, 500, 0.40, audit_rate, args.budget
        )
        rows.append(
            {
                "scenario": "drift",
                "true_error": 0.40,
                "audit_rate": audit_rate,
                "false_alarm_rate": float("nan"),
                "median_delay": median,
                "p90_delay": p90,
                "missed_rate": missed,
                "seeds": args.seeds,
            }
        )

    table = pd.DataFrame(rows)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nwritten -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
