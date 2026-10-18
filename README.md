# Underwrite

Local, open-weight document extraction that ships a **measured error guarantee** instead of a
vague accuracy claim.

Every batch comes with a certificate: *N documents, per-field error rate ≤ α with confidence
1 − δ, X% auto-processed, the rest routed to a human.*

> **Status:** working end to end. Extraction, signal capture, the hidden-state probe, the
> conformal gate, resplit validity, shift/repair and the audit lane all run; the review UI
> serves a precomputed demo batch. Every number below is produced by a script in this
> repository and committed under `reports/`.
>
> **Headline** (CORD, `total`, 972 labelled documents): the hidden-state probe reaches AUROC
> **0.952** against **0.838** for the strongest black-box stack, and auto-processes **49.7%**
> of totals at a 5% error budget versus 14.7%. Over 200 random resplits it certifies in every
> split at 48.1% mean coverage (α=0.10) and 29.4% (α=0.05), against 39.5% and 2.0% for the
> black-box baseline.

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

## Results

Every table below is a committed script output in `reports/tables/`; every figure is in
`reports/figures/`.

**Extraction.** CORD, 983 documents, `google/gemma-4-E2B-it` in 4-bit NF4 on a 6 GB laptop
GPU: **9.06 s/document**, 2.47 h for the corpus, 6573 MB peak VRAM, 983/983 value spans
aligned to tokens, 2 unparseable responses. `total` is correct on **58.6%** of documents, and
**84%** of the errors arrive with near-certain token probabilities.

**Ablation** (`total`, n=292 held-out, base error 38.0%):

| method | AUROC | coverage @2% | @5% | @10% |
|---|---|---|---|---|
| S0 log-probability | 0.632 | 1.0% | 1.0% | 1.0% |
| S2 validators | 0.525 | 1.4% | 1.4% | 1.4% |
| S2g grounding | 0.725 | 0.3% | 0.3% | 0.3% |
| S3 black-box fusion | 0.838 | 8.2% | 14.7% | 46.2% |
| S4 probe only | 0.914 | 17.5% | 30.5% | 51.7% |
| **S5 probe + fusion** | **0.952** | **41.4%** | **49.7%** | **62.0%** |

**Without grounding** (`--no-grounding`, the oracle page text removed): S3 falls to 0.729 and
S5 holds 0.915. The probe's advantage does not depend on perfect OCR — if anything it grows.

**Validity** (200 random resplits of the pooled calibration+test documents):

| α | method | certified | mean coverage | violations |
|---|---|---|---|---|
| 0.10 | S3 black-box | 39.5% | 10.9% | 8.0% |
| 0.10 | S5 probe+fusion | 100% | 48.1% | 11.5% |
| 0.05 | S3 black-box | 2.0% | — | 2.0% |
| 0.05 | S5 probe+fusion | 100% | 29.4% | 6.0% |

**Shift and repair.** A CORD-calibrated gate deployed on synthetic Indian GST invoices keeps
its promise but loses almost all coverage (42.9% → 2.5%), because the confidence distribution
shifts even though the target corpus is easier. Recalibrating on labelled target documents
repairs it: **k=50 gives 100% certification at ~93% coverage for α=0.10; k=100 gives ~92% at
α=0.05.**

**Audit lane** (500 seeds): false alarms on clean streams run at **0.0–0.4%** against a 5%
budget. A 40% drift is caught with a median delay of ~406 accepted documents at a 10% audit
rate — but **46% of drifts are missed** in a 500-document window at that rate.

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # core + tests

# For the model harness (GPU):
pip install -e ".[ml]"
```

PyTorch is best installed from the official selector for your CUDA build; see
<https://pytorch.org>. Gemma 4 is Apache-2.0 and **not gated** — no license acceptance or
token is required. A read token is still worth setting for rate limits; never commit it.
The model revision is pinned in `configs/default.yaml`, because a calibration is only valid
against frozen weights.

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

- **The certificate covers `total` only.** `total` has one value per document, so calibration
  samples are independent. A per-field guarantee on the other fields would be invalid — a
  document's fields are correlated, which is exactly the failure mode described in
  arXiv:2608.14639.
- **Grounding uses oracle page text.** CORD's ground-truth OCR words and the synthetic
  renderer's own text stand in for a real OCR pass, so absolute coverage is optimistic. The
  `--no-grounding` ablation shows the S3-vs-S5 comparison does not depend on it.
- **The verbalized baseline is absent from the final numbers.** The last extraction run
  omitted the self-rating call to halve runtime, so S1 is a constant there. It was measured on
  the earlier run at AUROC 0.603 for `total`.
- **~290 calibration documents is thin.** α=0.05 sits at the edge of feasibility, and the
  α=0.10 violation rate over resplits (11.5%) is slightly above δ=10%.
- **The audit lane misses most drift in a short window.** At a 10% audit rate, 46% of 40%
  drifts go undetected within 500 documents. False alarms stay well under the 5% budget, so
  the failure mode is a late alarm rather than a noisy one.
- **A 2B model on a 6 GB card sets the accuracy ceiling** — 38% error on `total`. A larger
  model would raise coverage; it would not change the machinery.
- **Exchangeability is assumed**, and the audit lane exists because that assumption can fail.
- CORD is Indonesian (CC-BY-4.0) and annotates no vendor or date, so the five-field story is
  necessarily a three-field one on this corpus.

## License

MIT — see [LICENSE](LICENSE).
