#!/usr/bin/env python
"""Build the precomputed demo batch the review UI serves.

Runs the real pipeline on the CORD test split, then freezes the result into one JSON
file with downscaled images embedded, so the hosted demo needs no GPU and no network.

Only the `total` field is gated: `total` is the flagship guarantee (one independent
sample per document). The other fields are shown but always routed to review, because
they are not certified.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from underwrite.app.certificate import (  # noqa: E402
    DECISION_AUTO,
    DECISION_REVIEW,
    build_certificate,
)
from underwrite.conformal.ltt import certify  # noqa: E402
from underwrite.config import load_config  # noqa: E402
from underwrite.data.cord import CORD_SPLITS, load_cord  # noqa: E402
from underwrite.probe.features import load_features  # noqa: E402
from underwrite.probe.train import TRAIN_SPLIT, apply_scorers, fit_scorers  # noqa: E402

CERTIFIED_FIELD = "total"


def images_by_doc() -> dict[str, object]:
    dataset = load_cord(CORD_SPLITS)
    images: dict[str, object] = {}
    for split in CORD_SPLITS:
        for index, row in enumerate(dataset[split]):
            images[f"cord_{split}_{index:04d}"] = row["image"]
    return images


def encode(image, width: int = 520, quality: int = 80) -> str:
    from PIL import Image

    picture = image.convert("RGB")
    scale = width / picture.width
    picture = picture.resize((width, round(picture.height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def document_meta(features_dir: Path, doc_id: str) -> dict:
    path = features_dir / f"{doc_id}.npz"
    if not path.is_file():
        return {}
    _, meta = load_features(path)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="data/features_table_cord.csv")
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--field", default=CERTIFIED_FIELD)
    parser.add_argument("--method", default="S5 probe + fusion")
    parser.add_argument("--documents", type=int, default=20)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.30])
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--audit-rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--out", default="reports/demo_batch.json")
    args = parser.parse_args()

    config = load_config()
    features_dir = Path(args.features_dir)
    frame = pd.read_csv(args.table)
    field_frame = frame[frame["field"] == args.field].reset_index(drop=True)
    train = field_frame[field_frame["split"] == TRAIN_SPLIT].reset_index(drop=True)

    bundles = sorted(features_dir.glob("*.npz"))
    if not bundles:
        raise SystemExit(f"no feature bundles in {features_dir}")
    _, meta = load_features(bundles[0])
    layers = list(meta["layers"])

    scorers = fit_scorers(train, args.field, features_dir, layers, seed=args.seed)
    scored = apply_scorers(scorers, field_frame, features_dir)

    calibration = scored[scored["split"] == "calibration"]
    alpha = None
    certification = None
    for candidate in args.alphas:
        attempt = certify(
            calibration[args.method].to_numpy(dtype=float),
            1 - calibration["correct"].to_numpy(dtype=int),
            candidate,
            args.delta,
        )
        if attempt.certified:
            alpha, certification = candidate, attempt
            break

    threshold = certification.threshold if certification and certification.certified else None
    print(f"alpha={alpha} threshold={threshold} certified={threshold is not None}")

    test = scored[scored["split"] == "test"].head(args.documents)
    library = images_by_doc()
    rng = np.random.default_rng(args.seed)

    documents = []
    audit_count = 0
    for _, row in test.iterrows():
        doc_id = row["doc_id"]
        score = float(row[args.method])
        decision = (
            DECISION_AUTO if threshold is not None and score >= threshold else DECISION_REVIEW
        )
        audit = decision == DECISION_AUTO and rng.random() < args.audit_rate
        audit_count += int(audit)

        meta = document_meta(features_dir, doc_id)
        extracted = meta.get("extracted", {})
        gold = meta.get("gold", {})

        fields = []
        for name in config.signals.fields:
            if name == args.field:
                field_decision = decision
            else:
                field_decision = DECISION_REVIEW
            fields.append(
                {
                    "name": name,
                    "value": extracted.get(name),
                    "truth": gold.get(name),
                    "decision": field_decision,
                    "confidence": round(score, 4) if name == args.field else None,
                    "certified": name == args.field,
                    "audit": audit and name == args.field,
                }
            )

        image = library.get(doc_id)
        documents.append(
            {
                "doc_id": doc_id,
                "image": encode(image) if image is not None else None,
                "fields": fields,
            }
        )

    decisions = [field["decision"] for document in documents for field in document["fields"]]
    certificate = build_certificate(
        decisions,
        alpha=alpha or args.alphas[-1],
        delta=args.delta,
        threshold=threshold,
        model_id=config.model.id,
        calibration_documents=len(calibration),
        audit_sample=audit_count,
    )

    payload = {
        "certificate": certificate.as_dict(),
        "certified_field": args.field,
        "method": args.method,
        "documents": documents,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    print(
        f"documents: {len(documents)}  auto: {certificate.auto_fields}  "
        f"review: {certificate.review_fields}  audited: {audit_count}"
    )
    print(f"written -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
