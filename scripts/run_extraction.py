#!/usr/bin/env python
"""Run extraction and signal capture over a corpus, resumably.

One ``.npz`` feature bundle is written per document, so an interrupted run resumes
where it stopped: documents whose bundle already exists are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from underwrite.config import load_config  # noqa: E402
from underwrite.data.cord import iter_cord_documents  # noqa: E402
from underwrite.extract.vlm import load_vision_model, pick_layers  # noqa: E402
from underwrite.probe.capture import capture_document  # noqa: E402
from underwrite.probe.features import save_features  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=None, help="stop after this many documents")
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--traces", default="reports/traces/extraction.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--verbalized", action="store_true", help="also run the self-rating call")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    config = load_config()
    fields = config.signals.fields
    processor, model = load_vision_model(config)

    hidden_count = model.config.text_config.num_hidden_layers + 1
    layers = pick_layers(hidden_count, config.signals.hidden_layer_fractions)
    print(f"model={config.model.id} fields={list(fields)} layers={layers}", flush=True)

    features_dir = Path(args.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    traces_path = Path(args.traces)
    traces_path.parent.mkdir(parents=True, exist_ok=True)

    documents = iter_cord_documents(tuple(args.splits))
    if args.limit is not None:
        documents = islice(documents, args.limit)

    captured_count = 0
    skipped = 0
    timings: list[float] = []

    with traces_path.open("a", encoding="utf-8") as traces:
        for document in documents:
            target = features_dir / f"{document.doc_id}.npz"
            if target.exists():
                skipped += 1
                continue

            started = perf_counter()
            captured = capture_document(
                processor,
                model,
                document.doc_id,
                document.image,
                document.gold,
                fields,
                layers,
                max_new_tokens=args.max_new_tokens,
                with_verbalized=args.verbalized,
            )
            elapsed = perf_counter() - started

            save_features(target, captured.features, fields, layers)
            record = captured.trace_record("cord")
            record["total_seconds"] = elapsed
            traces.write(json.dumps(record) + "\n")
            traces.flush()

            captured_count += 1
            timings.append(elapsed)
            if args.progress_every and captured_count % args.progress_every == 0:
                mean = sum(timings) / len(timings)
                print(
                    f"  captured={captured_count} skipped={skipped} "
                    f"mean={mean:.2f}s/doc aligned={captured.offsets_aligned}",
                    flush=True,
                )
            elif not captured.offsets_aligned:
                print(f"  WARNING: offsets not aligned for {document.doc_id}", flush=True)

    print(f"captured {captured_count}, skipped {skipped}")
    if timings:
        mean = sum(timings) / len(timings)
        print(f"mean {mean:.2f}s/doc  ->  {mean * len(timings) / 3600:.2f} h for this batch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
