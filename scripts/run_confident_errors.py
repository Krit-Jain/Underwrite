#!/usr/bin/env python
"""How often is the model confidently wrong?

The README's headline claim is that most errors arrive with a high token
log-probability. That needs a definition, so this script fixes one: a value is
"near-certain" when its mean token log-probability is above a stated cutoff, and the
share reported is *of the errors*, not of all values.

Sweeps several cutoffs so any number quoted in the post is traceable to a row here,
and prints one concrete example so the illustration in the README is a real document
rather than a constructed one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

CUTOFFS = (-0.01, -0.05, -0.10, -0.50, -1.00)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="data/features_table_v2.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--field", default="total")
    parser.add_argument("--out", default="reports/tables/confident_errors.csv")
    parser.add_argument(
        "--features-dir",
        default="data/features_promptfix",
        help="used to recover the extracted and gold values for the example",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.table)
    subset = frame[(frame["split"] == args.split) & (frame["field"] == args.field)]
    subset = subset.dropna(subset=["mean_logprob"])
    if subset.empty:
        raise SystemExit(f"no rows for split={args.split!r} field={args.field!r}")

    errors = subset[~subset["correct"].astype(bool)]
    print(
        f"{args.split} split, field={args.field}: {len(subset)} scored values, "
        f"{len(errors)} errors ({len(errors) / len(subset):.3f})"
    )

    rows = []
    for cutoff in CUTOFFS:
        confident = errors["mean_logprob"] > cutoff
        rows.append(
            {
                "split": args.split,
                "field": args.field,
                "cutoff": cutoff,
                "n_values": len(subset),
                "n_errors": len(errors),
                "n_confident_errors": int(confident.sum()),
                "share_of_errors": float(confident.mean()) if len(errors) else float("nan"),
            }
        )

    table = pd.DataFrame(rows)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    worst = errors.sort_values("mean_logprob", ascending=False).head(1)
    if not worst.empty:
        row = worst.iloc[0]
        bundle = Path(args.features_dir) / f"{row['doc_id']}.npz"
        extracted = gold = None
        if bundle.is_file():
            from underwrite.probe.features import load_features

            _, meta = load_features(bundle)
            extracted = (meta.get("extracted") or {}).get(args.field)
            gold = (meta.get("gold") or {}).get(args.field)
        print(
            f"\nmost confident error: {row['doc_id']}  field={args.field}  "
            f"extracted={extracted!r}  gold={gold!r}  "
            f"mean_logprob={row['mean_logprob']:.6f}"
        )

    print(f"\nwritten -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
