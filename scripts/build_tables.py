#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from pathlib import Path

from sim_score_study.config import StudyConfig, get_config
from sim_score_study.reporting import (
    build_coef_stability_table,
    build_main_discrimination_table,
    build_main_probability_table,
    build_sim_settings_table,
    build_misspec_robustness_tables,
    export_table,
    load_all_results,
    summarize_coef_stability,
    summarize_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build simulation-study tables.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument(
        "--table",
        default="all",
        choices=[
            "all",
            "sim-settings",
            "sim-main-discrimination",
            "sim-main-probability",
            "sim-coef-stability",
            "sim-misspec-robustness",
        ],
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config name. If omitted, inferred from results-root/config.json is not required for tables using saved config values.",
    )
    return parser.parse_args()


def _load_config(results_root: Path, config_name: str | None):
    """Load StudyConfig.

    Backward compatibility note:
    Older result folders may contain a config.json missing newer fields. To
    support regenerating outputs from those runs, we merge the saved payload on
    top of the current default config for that name.
    """
    import json

    if config_name is not None:
        return get_config(config_name)

    cfg_path = results_root / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError("Could not find config.json under results-root. Pass --config explicitly.")
    with cfg_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    base = get_config(payload.get("name", "paper"))
    merged = base.to_dict()
    merged.update(payload)
    return StudyConfig(**merged)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    tables_dir = results_root / "tables"
    config = _load_config(results_root, args.config)

    metrics = None
    coefficients = None
    summary = None

    if args.table in {"all", "sim-main-discrimination", "sim-main-probability", "sim-coef-stability"}:
        metrics, coefficients, _ = load_all_results(results_root, load_pooled_scores=False)
        summary = summarize_metrics(metrics)

    if args.table in {"all", "sim-settings"}:
        df = build_sim_settings_table(config)
        export_table(
            df,
            tables_dir / "tab_sim_settings.csv",
            tables_dir / "tab_sim_settings.tex",
            caption="Simulation settings and scenario grid.",
            label="tab:sim-settings",
        )

    if args.table in {"all", "sim-main-discrimination"}:
        df = build_main_discrimination_table(summary, config)
        export_table(
            df,
            tables_dir / "tab_sim_main_discrimination.csv",
            tables_dir / "tab_sim_main_discrimination.tex",
            caption="Main discrimination and operating-point performance.",
            label="tab:sim-main-discrimination",
        )

    if args.table in {"all", "sim-main-probability"}:
        df = build_main_probability_table(summary, config)
        export_table(
            df,
            tables_dir / "tab_sim_main_probability.csv",
            tables_dir / "tab_sim_main_probability.tex",
            caption="Probability-quality metrics.",
            label="tab:sim-main-probability",
        )

    if args.table in {"all", "sim-coef-stability"}:
        coef_summary = summarize_coef_stability(coefficients)
        df = build_coef_stability_table(coef_summary, config)
        export_table(
            df,
            tables_dir / "tab_sim_coef_stability.csv",
            tables_dir / "tab_sim_coef_stability.tex",
            caption="Coefficient stability for LR-Decomp.",
            label="tab:sim-coef-stability",
        )
        coef_summary.to_csv(tables_dir / "tab_sim_coef_stability_full_tidy.csv", index=False)

    should_build_misspec = (
        args.table == "sim-misspec-robustness"
        or (args.table == "all" and bool(getattr(config, "expert_misspecification", False)))
    )
    if should_build_misspec:
        misspec_metrics, _coef, _ = load_all_results(results_root, load_pooled_scores=False)

        tab_auroc, tab_other = build_misspec_robustness_tables(
            metrics=misspec_metrics,
            config=config,
        )
        export_table(
            tab_auroc,
            tables_dir / "tab_sim_misspec_robustness.csv",
            tables_dir / "tab_sim_misspec_robustness.tex",
            caption=(
                "Robustness to expert-model misspecification. In each Monte Carlo replicate, an expert model is created by "
                "independently perturbing each expert parameter by ±p, but the data are generated from the true model. "
                "Cells report AUROC under misspecification; parentheses report paired AUROC differences vs the well-specified expert model "
                "(same replicate and scenario)."
            ),
            label="tab:sim-misspec-robustness",
        )
        export_table(
            tab_other,
            tables_dir / "tab_sim_misspec_robustness_other_metrics.csv",
            tables_dir / "tab_sim_misspec_robustness_other_metrics.tex",
            caption=(
                "Robustness to expert-model misspecification: additional metrics (TNR@TPR95, AUPRC, Brier score, Log loss). "
                "Cells report the metric under misspecification; parentheses report paired differences vs the well-specified expert model "
                "(same replicate and scenario)."
            ),
            label="tab:sim-misspec-robustness-other",
        )


if __name__ == "__main__":
    main()