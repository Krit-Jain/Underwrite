#!/usr/bin/env python
"""Smoke test: load the 4-bit vision model and extract fields from one receipt.

Examples
--------
    python scripts/smoke_test.py
    python scripts/smoke_test.py --image path/to/receipt.jpg
    python scripts/smoke_test.py --model google/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

# Allow running the script straight from a checkout without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from underwrite.config import load_config  # noqa: E402
from underwrite.smoke import run_smoke_test  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="config YAML (default: configs/default.yaml)")
    parser.add_argument("--image", default=None, help="receipt image; a synthetic one is used if omitted")
    parser.add_argument("--model", default=None, help="override the model id from the config")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.model:
        config = replace(config, model=replace(config.model, id=args.model))

    try:
        result = run_smoke_test(config, image_path=args.image, max_new_tokens=args.max_new_tokens)
    except RuntimeError as exc:
        print(f"smoke test cannot run: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result.fields, indent=2, ensure_ascii=False))
    print(
        f"model={result.model_id} seconds={result.seconds:.2f} "
        f"peak_vram_mb={result.peak_vram_mb:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
