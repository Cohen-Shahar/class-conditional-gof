#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from sim_score_study.config import get_config

# StudyConfig is only needed for typing when loading config.json.
from sim_score_study.config import StudyConfig
from sim_score_study.plotting import (
    plot_metric_vs_n,
    plot_paired_gains,
    plot_paired_gains_combined,
    plot_performance_vs_n_2x2,
    plot_score_diagnostics,
    plot_misspecification_comparison_n10000,
)
from sim_score_study.reporting import load_all_results, summarize_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build simulation-study figures.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument(
        "--figure",
        default="all",
        choices=[
            "all",
            "sim-score-diagnostics",
            "sim-performance-vs-n",
            "sim-auroc-vs-n",
            "sim-tnr-vs-n",
            "sim-paired-gains",
            "sim-paired-gains-tnr",
            "sim-paired-gains-combined",
            "sim-misspecified",
        ],
    )
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def _load_config(results_root: Path, config_name: str | None):
    """Load StudyConfig.

    Backward compatibility note:
    Older result folders may contain a config.json missing newer fields. To
    support regenerating figures/tables from those runs, we merge the saved
    payload on top of the current default config for that name.
    """
    import json

    if config_name is not None:
        return get_config(config_name)

    cfg_path = results_root / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError("Could not find config.json under results-root. Pass --config explicitly.")
    with cfg_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # Start from the current defaults for this config name (fills any newly-added fields).
    base = get_config(payload.get("name", "paper"))
    merged = base.to_dict()
    merged.update(payload)
    return StudyConfig(**merged)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    figs_dir = results_root / "figures"
    config = _load_config(results_root, args.config)

    sigma_levels = sorted(config.sigma_x_levels)
    rep_sigma = sigma_levels[len(sigma_levels) // 2]
    lambda_low = min(config.lambda_levels)
    lambda_high = max(config.lambda_levels)

    include_score_diagnostics = bool(getattr(config, "run_with_pooled_scores", False))
    need_scores = args.figure == "sim-score-diagnostics" or (args.figure == "all" and include_score_diagnostics)

    metrics, _, pooled_scores = load_all_results(results_root, load_pooled_scores=need_scores)
    summary = summarize_metrics(metrics)

    if args.figure == "sim-score-diagnostics" or (args.figure == "all" and include_score_diagnostics):
        if pooled_scores is None:
            raise FileNotFoundError(
                "No pooled score payloads were found under results-root/cells. "
                "Re-run simulations with run_with_pooled_scores=true in config (e.g., paper_pooled_scores or smoke_pooled_scores)."
            )
        plot_score_diagnostics(
            pooled_scores=pooled_scores,
            output_path=figs_dir / "fig_sim_score_diagnostics.pdf",
            representative_sigma=rep_sigma,
            lambda_low=lambda_low,
            lambda_high=lambda_high,
        )

    include_misspec = bool(getattr(config, "expert_misspecification", False))
    should_plot_misspec = args.figure == "sim-misspecified" or (args.figure == "all" and include_misspec)

    if should_plot_misspec:
        plot_misspecification_comparison_n10000(
            metrics,
            output_path=figs_dir / "fig_sim_misspecification_comparison_n10000.pdf",
            n_train=10_000,
        )

    if args.figure in {"all", "sim-performance-vs-n"}:
        plot_performance_vs_n_2x2(
            summary,
            output_path=figs_dir / "fig_sim_performance_vs_n.pdf",
            lambda_low=lambda_low,
            lambda_high=lambda_high,
        )

    if args.figure in {"all", "sim-auroc-vs-n"}:
        plot_metric_vs_n(summary, metric_name="AUROC", output_path=figs_dir / "fig_sim_auroc_vs_n.pdf")

    if args.figure in {"all", "sim-tnr-vs-n"}:
        plot_metric_vs_n(summary, metric_name="TNR@TPR95", output_path=figs_dir / "fig_sim_tnr_vs_n.pdf")

    if args.figure in {"all", "sim-paired-gains"}:
        plot_paired_gains(metrics, metric_name="AUROC", output_path=figs_dir / "fig_sim_paired_gains.pdf")

    if args.figure in {"all", "sim-paired-gains-tnr"}:
        plot_paired_gains(metrics, metric_name="TNR@TPR95", output_path=figs_dir / "fig_sim_paired_gains_tnr.pdf")

    if args.figure in {"all", "sim-paired-gains-combined"}:
        plot_paired_gains_combined(metrics, output_path=figs_dir / "fig_sim_paired_gains_combined.pdf")


if __name__ == "__main__":
    main()

