#!/usr/bin/env python
"""Render the figures for the write-up from the recorded result tables.

Every number plotted comes from a CSV produced by an experiment script; nothing here
computes a result of its own except the risk-coverage curves, which are derived from
cached per-field scores (also produced by a script).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from underwrite.eval.risk_coverage import risk_coverage_curve  # noqa: E402
from underwrite.probe.features import load_features  # noqa: E402
from underwrite.probe.train import METHOD_COLUMNS, TRAIN_SPLIT, apply_scorers, fit_scorers  # noqa: E402

INK = "#16181d"
MUTED = "#6b7280"
ACCENT = "#1f3a5f"
PALETTE = ["#9ca3af", "#c026d3", "#0d9488", "#2563eb", "#f59e0b", "#dc2626", "#0f7b4f"]


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#e5e7eb")
    ax.grid(True, color="#f0f1f3", linewidth=0.8)
    ax.set_axisbelow(True)


def cached_scores(table: str, features_dir: str, field: str, cache: str) -> pd.DataFrame:
    """Per-field scores for every method, computed once and reused."""
    path = Path(cache)
    if path.is_file():
        return pd.read_csv(path)

    frame = pd.read_csv(table)
    field_frame = frame[frame["field"] == field].reset_index(drop=True)
    train = field_frame[field_frame["split"] == TRAIN_SPLIT].reset_index(drop=True)
    bundles = sorted(Path(features_dir).glob("*.npz"))
    _, meta = load_features(bundles[0])
    scorers = fit_scorers(train, field, features_dir, list(meta["layers"]))
    scored = apply_scorers(scorers, field_frame, features_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(path, index=False)
    return scored


def figure_risk_coverage(scored: pd.DataFrame, out: Path, field: str = "total") -> None:
    train = scored[scored["split"] == TRAIN_SPLIT]
    test = scored[scored["split"] == "test"]
    errors = 1 - test["correct"].to_numpy(dtype=int)

    figure, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    for index, method in enumerate(METHOD_COLUMNS):
        curve = risk_coverage_curve(test[method].to_numpy(dtype=float), errors)
        ax.plot(curve.coverage, curve.risk, label=method, color=PALETTE[index % len(PALETTE)],
                linewidth=2.0 if method.startswith("S5") else 1.4)
    ax.axhline(errors.mean(), color=MUTED, linestyle=":", linewidth=1)
    ax.text(1.0, errors.mean(), " accept everything", va="bottom", ha="right", fontsize=7, color=MUTED)
    for alpha in (0.10, 0.05):
        ax.axhline(alpha, color="#cbd5e1", linewidth=0.8)
    style(ax, f"Risk-coverage on `{field}` (test split, n={len(test)})",
          "Coverage (fraction auto-accepted)", "Error rate among accepted")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, min(1.0, errors.mean() * 1.15))
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", labelcolor=MUTED)
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def figure_auroc(ablation: pd.DataFrame, out: Path) -> None:
    table = ablation[ablation["field"] == "total"].iloc[::-1]
    figure, ax = plt.subplots(figsize=(6.4, 3.4), dpi=160)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(table))]
    ax.barh(table["method"], table["auroc"], color=colors, height=0.62)
    for y, value in enumerate(table["auroc"]):
        ax.text(value + 0.008, y, f"{value:.3f}", va="center", fontsize=8, color=MUTED)
    ax.axvline(0.5, color="#cbd5e1", linewidth=1, linestyle="--")
    style(ax, "How well each signal ranks correct above incorrect (`total`)",
          "AUROC (0.5 = chance)", "")
    ax.set_xlim(0.4, 1.02)
    ax.grid(axis="y", visible=False)
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def figure_validity(validity: pd.DataFrame, out: Path) -> None:
    methods = validity["method"].unique()
    alphas = sorted(validity["alpha"].unique())
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), dpi=160)

    width = 0.36
    positions = np.arange(len(methods))
    for offset, alpha in enumerate(alphas):
        subset = validity[validity["alpha"] == alpha].set_index("method").loc[methods]
        shift = (offset - (len(alphas) - 1) / 2) * width
        axes[0].bar(positions + shift, subset["certified_rate"], width=width,
                    label=f"α={alpha:.2f}", color=[ACCENT, "#0d9488"][offset % 2])
        axes[1].bar(positions + shift, subset["violation_rate"], width=width,
                    label=f"α={alpha:.2f}", color=[ACCENT, "#0d9488"][offset % 2])
        axes[2].bar(positions + shift, subset["mean_coverage"], width=width,
                    label=f"α={alpha:.2f}", color=[ACCENT, "#0d9488"][offset % 2])

    for ax, title in zip(axes, ("Certified (fraction of resplits)", "Violations (realised risk > α)",
                                "Mean coverage when certified")):
        ax.set_xticks(positions)
        ax.set_xticklabels([m.replace(" black-box fusion", "\nblack-box").replace(" probe + fusion", "\nprobe+fusion")
                            .replace(" probe", "\nprobe") for m in methods], fontsize=7.5)
        style(ax, title, "", "")
        ax.grid(axis="x", visible=False)
    axes[1].axhline(0.10, color="#dc2626", linestyle="--", linewidth=1)
    axes[1].text(len(methods) - 0.5, 0.105, "δ = 0.1", ha="right", fontsize=7, color="#dc2626")
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=MUTED)
    figure.suptitle("Resplit validity of the certified threshold (`total`, 200 random splits)",
                    color=INK, fontsize=11, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def figure_repair(repair: pd.DataFrame, out: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), dpi=160)
    for index, (method, group) in enumerate(repair.groupby("method")):
        color = PALETTE[index % len(PALETTE)]
        for alpha, sub in group.groupby("alpha"):
            label = f"{method} α={alpha:.2f}"
            axes[0].plot(sub["k"], sub["certified_rate"], marker="o", color=color,
                         linestyle="-" if alpha >= 0.2 else "--", label=label, linewidth=1.6)
            axes[1].plot(sub["k"], sub["mean_coverage"], marker="o", color=color,
                         linestyle="-" if alpha >= 0.2 else "--", label=label, linewidth=1.6)
    style(axes[0], "Certification after adding labelled target docs", "k labelled target documents", "certified rate")
    style(axes[1], "Coverage after repair", "k labelled target documents", "mean coverage")
    axes[1].legend(frameon=False, fontsize=6.5, labelcolor=MUTED, loc="upper left")
    figure.suptitle("Repairing a CORD-calibrated gate on Indian GST invoices",
                    color=INK, fontsize=11, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def figure_audit(audit: pd.DataFrame, out: Path) -> None:
    switch_at = int(audit["switch_at"].iloc[0])
    figure, ax = plt.subplots(figsize=(6.8, 3.3), dpi=160)
    ax.plot(audit["position"], audit["upper_bound"], color=ACCENT, linewidth=1.8)
    ax.axvline(switch_at, color="#dc2626", linestyle="--", linewidth=1.2)
    ax.text(switch_at, ax.get_ylim()[1] * 0.92, " new template", color="#dc2626", fontsize=8)
    ax.axhline(0.10, color="#cbd5e1", linestyle=":", linewidth=1)
    ax.text(audit["position"].iloc[-1], 0.104, "α = 0.10 ", ha="right", fontsize=7, color=MUTED)
    style(ax, "Audit lane: Clopper-Pearson upper bound on accepted-error rate",
          "Accepted documents in the stream", "Upper bound on error rate")
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", default="reports/tables")
    parser.add_argument("--out", default="reports/figures")
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--table", default="data/features_table_cord.csv")
    args = parser.parse_args()

    tables = Path(args.tables)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ablation = pd.read_csv(tables / "ablation.csv")
    figure_auroc(ablation, out / "auroc_by_method.png")
    print(f"wrote {out / 'auroc_by_method.png'}")

    scored = cached_scores(args.table, args.features_dir, "total", "data/scores_total.csv")
    figure_risk_coverage(scored, out / "risk_coverage.png")
    print(f"wrote {out / 'risk_coverage.png'}")

    validity_path = tables / "validity_total.csv"
    if validity_path.is_file():
        figure_validity(pd.read_csv(validity_path), out / "validity.png")
        print(f"wrote {out / 'validity.png'}")

    repair_path = tables / "repair.csv"
    if repair_path.is_file():
        figure_repair(pd.read_csv(repair_path), out / "repair.png")
        print(f"wrote {out / 'repair.png'}")

    audit_path = tables / "audit_trajectory.csv"
    if audit_path.is_file():
        figure_audit(pd.read_csv(audit_path), out / "audit.png")
        print(f"wrote {out / 'audit.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
