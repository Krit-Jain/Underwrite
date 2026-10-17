# Underwrite

Local, open-weight document extraction that ships a **measured error guarantee** instead of a
vague accuracy claim.

Every batch comes with a certificate: *N documents, per-field error rate ≤ α with confidence
1 − δ, X% auto-processed, the rest routed to a human.*

> **Status:** early scaffolding. The model harness and configuration layer exist; the probe,
> conformal gate, and review UI are in progress. Numbers in the write-up will be produced by the
> scripts in this repository — never estimated.

## The problem

Modern document parsers read clean pages at 96%+ benchmark accuracy and still emit fluent,
plausible, silently-wrong values, with no reliable signal telling you *which* values to trust.
A bookkeeper cannot hand a client "about 96% accurate". They need to know what to double-check.

Underwrite reads invoices and receipts into structured fields and, for each field, decides whether
to **auto-accept** it or **route it for review** — with the accept threshold chosen so the error
rate *among accepted fields* is bounded by a target α at confidence 1 − δ.

## Research question

At the same **certified** error rate, do white-box signals (hidden-state probes, only possible with
open weights) let us auto-process more documents than black-box signals (token log-probabilities,
verbalized confidence, validators, source grounding)?

## Why open weights

- **Hidden states are the point.** A closed API does not expose them, so the probe cannot exist.
- **A calibration is only valid for a frozen model.** Pinned open weights keep the certificate
  valid; a silently updated hosted model can break it.
- **Financial documents stay on the operator's machine.** No client data leaves the laptop.
- **Marginal cost is electricity**, which we measure rather than guess.

## How it works

```
 image ─► [VLM, 4-bit] ─► JSON fields ──┐
             │                          ├─► validators (sum, format, GSTIN, dates)
             └─► teacher-forced pass ───┤
                 logprobs, entropy,     └─► [probe + fusion] ─► p(correct)
                 margin, hidden states                          │
                                       [Learn-Then-Test gate: threshold τ, α, δ]
                                                                │
                       ┌────────────────┬─────────────────────┬─┘
                       ▼                ▼                     ▼
                  AUTO-ACCEPT      REVIEW (flagged)      AUDIT sample
                       └────────────────┴─────────────────────┘
                                        ▼
                            certificate + CSV/JSON export
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # core + tests

# For the model harness (GPU):
pip install -e ".[ml]"
```

PyTorch is best installed from the official selector for your CUDA build; see
<https://pytorch.org>. Gemma is gated on Hugging Face — accept the license and export a read token
(`HF_TOKEN`) before running the smoke test. Never commit the token.

## Quickstart

```bash
# Load the 4-bit model and extract one receipt (a synthetic one is generated if --image is omitted).
python scripts/smoke_test.py
```

It prints the extracted JSON, then the model id, wall-clock seconds, and peak VRAM in MB.

## Repository layout

```
configs/default.yaml     model id, seeds, splits, alpha grid, delta, paths
data/                    raw/processed/cache — local only, never committed
src/underwrite/
  config.py              configuration loading and validation
  extract/               prompts, output parsing, model harness
  signals/               token statistics, validators, source grounding
  probe/                 hidden-state probe and fusion model
  conformal/             Learn-Then-Test gate and audit-lane sampler
  eval/                  risk-coverage, validity, and shift evaluation
  app/                   review UI, batch pipeline, certificate
scripts/                 runnable entry points
tests/                   unit tests
reports/                 figures, tables, and traces produced by experiments
```

## Development

```bash
pytest
```

Tests are CPU-only and do not require model weights; the GPU smoke test is a separate script.

## Correctness definition

- Money fields: `Decimal` equality at 2 decimal places after stripping symbols and separators.
- Date: parsed ISO 8601 equality.
- Vendor: `rapidfuzz` ratio ≥ 90 after normalization.
- A parse failure counts as an error.

## Credits and prior work

This project builds directly on open work and does **not** claim their ideas as new:

- **Gemma** (Google DeepMind), open-weight vision-language models — the extractor.
- **CORD** (Park et al.), a consolidated receipt dataset for post-OCR parsing — the primary
  real-receipt corpus.
- **VerifyDoc** (Gurram, Apache-2.0, 2026) — per-field calibrated confidence, source grounding,
  and accept/review abstention at a target selective risk.
- **“Valid Per-Field Selective Risk Control for Document Extraction”** (arXiv:2608.14639, 2026) —
  shows the naive per-field accept/review procedure can silently violate its own guarantee, which
  is why the certified guarantee here is restricted to the per-document `total` field.
- **Learn-Then-Test** (Angelopoulos et al., arXiv:2110.01052) — the fixed-sequence threshold
  calibration the gate implements.

## Limitations

- The guarantee assumes new documents are exchangeable with the calibration documents. The audit
  lane exists because that assumption can fail.
- Calibration sets are small; only error budgets the data supports are certified.
- 4-bit quantization and a single small model bound achievable coverage.

## License

MIT — see [LICENSE](LICENSE).
