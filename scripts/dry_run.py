#!/usr/bin/env python
"""Dry run: extract a handful of documents and print per-field signals.

Exercises the whole capture path -- greedy extraction, the teacher-forced second
pass, token-level signals, and the validators -- and prints one row per field so the
signals can be inspected by eye before committing to a full run.
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
from underwrite.data.normalize import is_correct  # noqa: E402
from underwrite.data.synth_invoices import generate_specs  # noqa: E402
from underwrite.extract.parse import ParseError, extract_json_object  # noqa: E402
from underwrite.extract.vlm import (  # noqa: E402
    build_messages,
    generate,
    load_vision_model,
    pick_layers,
    teacher_forced,
)
from underwrite.signals.teacher_forced import summarise_fields  # noqa: E402
from underwrite.signals.validators import validate_document  # noqa: E402


def synthetic_documents(count: int, seed: int):
    from pathlib import Path as _Path

    from PIL import Image

    root = _Path("data/synthetic")
    truth = json.loads((root / "ground_truth.json").read_text(encoding="utf-8"))
    for spec in islice(generate_specs(count, seed=seed), count):
        image = Image.open(root / "images" / f"{spec.doc_id}.png").convert("RGB")
        yield spec.doc_id, image, truth[spec.doc_id]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--source", choices=("cord", "synthetic"), default="cord")
    parser.add_argument("--split", default="test", help="CORD split to draw from")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--traces", default="reports/traces/dry_run.jsonl")
    args = parser.parse_args()

    config = load_config()
    fields = config.signals.fields
    processor, model = load_vision_model(config)

    hidden_count = model.config.text_config.num_hidden_layers + 1
    layers = pick_layers(hidden_count, config.signals.hidden_layer_fractions)
    print(f"model={config.model.id} hidden_states={hidden_count} probe_layers={layers}")

    if args.source == "cord":
        documents = (
            (document.doc_id, document.image, document.gold)
            for document in islice(iter_cord_documents((args.split,)), args.count)
        )
    else:
        documents = synthetic_documents(args.count, args.seed)

    traces_path = Path(args.traces)
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    traces = traces_path.open("w", encoding="utf-8")

    header = (
        f"{'document':<20} {'field':<9} {'predicted':<14} {'truth':<14} "
        f"{'ok':<5} {'mean_lp':>8} {'first_H':>8} {'ntok':>5}  validators"
    )
    print(header)
    print("-" * len(header))

    for doc_id, image, gold in documents:
        messages = build_messages(image, fields)

        started = perf_counter()
        text = generate(processor, model, messages, args.max_new_tokens)
        generation_seconds = perf_counter() - started

        try:
            extracted = extract_json_object(text)
        except ParseError:
            extracted = {}

        signals = teacher_forced(processor, model, messages, text, layers)
        statistics = summarise_fields(
            signals.text,
            signals.offsets,
            signals.logprobs,
            signals.entropies,
            signals.margins,
            fields,
        )
        checks = validate_document(extracted)
        checks_text = " ".join(f"{name}={value:.0f}" for name, value in checks.items()) or "-"

        for field in fields:
            predicted = extracted.get(field)
            truth = gold.get(field)
            if truth is None:
                verdict = "-"
            else:
                verdict = "ok" if is_correct(field, predicted, truth) else "WRONG"

            summary = statistics.get(field)
            mean_lp = f"{summary.mean_logprob:8.3f}" if summary else "       -"
            first_h = f"{summary.first_token_entropy:8.3f}" if summary else "       -"
            n_tokens = f"{summary.n_tokens:5d}" if summary else "    -"

            print(
                f"{doc_id:<20} {field:<9} {str(predicted):<14.14} {str(truth):<14.14} "
                f"{verdict:<5} {mean_lp} {first_h} {n_tokens}  {checks_text}"
            )

        traces.write(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "source": args.source,
                    "generation_seconds": generation_seconds,
                    "n_prompt_tokens": signals.n_prompt_tokens,
                    "n_response_tokens": len(signals.token_ids),
                    "offsets_aligned": signals.offsets_aligned,
                    "extracted": extracted,
                    "gold": gold,
                    "validators": checks,
                    "fields": {
                        name: summary.as_features() for name, summary in statistics.items()
                    },
                    "hidden_shapes": {
                        str(layer): list(array.shape)
                        for layer, array in signals.hidden_states.items()
                    },
                }
            )
            + "\n"
        )
        traces.flush()

    traces.close()
    print(f"\ntraces -> {traces_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
