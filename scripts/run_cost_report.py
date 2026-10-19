#!/usr/bin/env python
"""What a run actually cost, measured from the extraction traces.

The traces record per-document wall-clock time, so seconds per document and the total
runtime are measurements. Electricity is *not* measured -- a laptop has no reliable
power telemetry -- so it is an estimate from a stated power draw, and the assumption is
written into the output rather than buried in prose.

    python scripts/run_cost_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

DEFAULT_TRACES = {
    "cord (promptfix)": "reports/traces/promptfix.jsonl",
    "synthetic indian gst": "reports/traces/extraction_v2_synthetic.jsonl",
}


def read_trace(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", default=None,
                        help="NAME=PATH pairs; defaults to the two canonical runs")
    parser.add_argument("--gpu-watts", type=float, default=90.0,
                        help="assumed average board power draw while extracting")
    parser.add_argument("--system-watts", type=float, default=40.0,
                        help="assumed draw of the rest of the laptop")
    parser.add_argument("--price-per-kwh", type=float, default=None,
                        help="optional, to convert kWh into a currency figure")
    parser.add_argument("--out", default="reports/tables/extraction_cost.csv")
    args = parser.parse_args()

    pairs = dict(DEFAULT_TRACES)
    if args.traces:
        pairs = {}
        for item in args.traces:
            name, _, path = item.partition("=")
            pairs[name] = path

    watts = args.gpu_watts + args.system_watts
    rows = []
    for name, path_str in pairs.items():
        path = Path(path_str)
        if not path.is_file():
            print(f"skipping {name}: no trace at {path}")
            continue
        records = read_trace(path)
        if not records:
            print(f"skipping {name}: trace is empty")
            continue

        seconds = [float(r["total_seconds"]) for r in records if r.get("total_seconds") is not None]
        tokens = [int(r["n_response_tokens"]) for r in records if r.get("n_response_tokens")]
        total_seconds = sum(seconds)
        hours = total_seconds / 3600.0
        kwh = watts * hours / 1000.0

        row = {
            "corpus": name,
            "documents": len(records),
            "total_seconds": total_seconds,
            "hours": hours,
            "seconds_per_document": total_seconds / len(records),
            "mean_response_tokens": sum(tokens) / len(tokens) if tokens else float("nan"),
            "assumed_watts": watts,
            "kwh": kwh,
        }
        if args.price_per_kwh is not None:
            row["estimated_cost"] = kwh * args.price_per_kwh
        rows.append(row)

    if not rows:
        raise SystemExit("no traces found; run the extraction first")

    table = pd.DataFrame(rows)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False)

    print(table.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print(
        f"\nseconds and hours are measured from the traces. Electricity assumes "
        f"{watts:.0f} W total draw ({args.gpu_watts:.0f} W board + {args.system_watts:.0f} W "
        f"system), stated as an assumption, not measured."
    )
    print(f"written -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
