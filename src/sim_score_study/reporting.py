from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import StudyConfig
from .utils import ensure_dir, format_mean_se


METHOD_ORDER = [
    "LR-Decomp",
    "LR-Total",
    "LR-Obs",
    "LR-Baseline",
    "RF-Raw",
    "RF-Raw+Features",
]

# Methods that should appear in manuscript tables (keep LR-Total in cell outputs).
PAPER_METHOD_ORDER = [
    "LR-Decomp",
    "LR-Obs",
    "LR-Baseline",
    "RF-Raw",
    "RF-Raw+Features",
]


def _cell_files(results_root: str | Path) -> list[Path]:
    return sorted((Path(results_root) / "cells").glob("*.pkl"))


def load_all_results(
    results_root: str | Path,
    *,
    load_pooled_scores: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    metric_frames = []
    coef_frames = []
    score_frames = []
    for path in _cell_files(results_root):
        payload = pd.read_pickle(path)
        metric_frames.append(payload["metrics"])
        coef_frames.append(payload["coefficients"])
        if load_pooled_scores:
            pooled = payload.get("pooled_scores")
            if pooled is not None:
                score_frames.append(pooled)
    if not metric_frames:
        raise FileNotFoundError(f"No cell result files found under {(Path(results_root) / 'cells')}")
    metrics = pd.concat(metric_frames, ignore_index=True)
    coefficients = pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame()
    if load_pooled_scores:
        pooled_scores = pd.concat(score_frames, ignore_index=True) if score_frames else None
    else:
        pooled_scores = None
    return metrics, coefficients, pooled_scores


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["AUROC", "AUPRC", "Brier", "LogLoss", "TNR@TPR95"]
    rows = []

    group_cols = ["n_train", "lambda", "method"]

    for keys, grp in metrics.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        for metric in metric_cols:
            vals = grp[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_se"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    return out


def summarize_paired_differences(metrics: pd.DataFrame, metric_name: str = "AUROC") -> pd.DataFrame:
    comparisons = [
        ("LR-Decomp", "LR-Total", "LR-Decomp-LR-Total"),
        ("LR-Decomp", "LR-Obs", "LR-Decomp-LR-Obs"),
        ("RF-Raw+Features", "RF-Raw", "RF-Raw+Features-RF-Raw"),
    ]
    rows = []

    unique_cells = metrics[["n_train", "lambda"]].drop_duplicates()
    for n_train, lambda_value in unique_cells.itertuples(index=False):
        subset = metrics[(metrics["n_train"] == n_train) & (metrics["lambda"] == lambda_value)]
        pivot = subset.pivot(index="replicate", columns="method", values=metric_name)
        for left, right, label in comparisons:
            diffs = pivot[left] - pivot[right]
            rows.append(
                {
                    "comparison": label,
                    "n_train": n_train,
                    "lambda": lambda_value,
                    f"{metric_name}_diff_mean": float(diffs.mean()),
                    f"{metric_name}_diff_se": float(diffs.std(ddof=1) / np.sqrt(diffs.notna().sum()))
                    if diffs.notna().sum() > 1
                    else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["comparison", "lambda", "n_train"]).reset_index(drop=True)


def summarize_coef_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n_train, lambda_value, feature), grp in coefficients.groupby(["n_train", "lambda", "feature"], sort=True):
        vals = grp["coef_standardized"].to_numpy(dtype=float)
        signs = np.sign(vals)
        unique, counts = np.unique(signs, return_counts=True)
        modal_sign = unique[np.argmax(counts)]
        sign_stability = counts.max() / counts.sum()
        rows.append(
            {
                "n_train": n_train,
                "lambda": lambda_value,
                "feature": feature,
                "coef_median": float(np.median(vals)),
                "sign_stability": float(sign_stability),
                "modal_sign": int(modal_sign),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_train", "feature", "lambda"]).reset_index(drop=True)


def build_sim_settings_table(config: StudyConfig) -> pd.DataFrame:
    rows = [
        ("Number of sources S", "50, fixed across instances"),
        ("Source locations r_s", "Equally spaced on [0,1], fixed across instances"),
        ("Class prevalence π1 = P(Y=1)", f"{config.prevalence:g}"),
        ("Latent prior for L", "U[0,1]"),
        ("Latent prior for M", f"N({config.latent_M_mean:g}, {config.latent_M_sd:g}^2)"),
        (
            "Detection-model parameters (alpha0_levels, alpha_M, alpha_d)",
            f"([{', '.join(f'{x:g}' for x in config.alpha0_levels)}], {config.alpha_M:g}, {config.alpha_d:g})",
        ),
        ("Source-specific noise parameters alpha0s", "U[0,1], fixed across instances"),
        (
            "Observation-model parameters (beta0, beta_M, beta_d)",
            f"({config.beta0:g}, {config.beta_M:g}, {config.beta_d:g})",
        ),
        ("Missingness informativeness levels lambda", ", ".join(f"{x:g}" for x in config.lambda_levels)),
        ("Observation-noise levels sigma_x", ", ".join(f"{x:g}" for x in config.sigma_x_levels)),
        (
            "Composite-event mixing parameter gamma",
            f"{config.gamma:g}",
        ),
        ("Training sample size", "; ".join(f"{n:,}" for n in config.training_sizes)),
        ("Test sample size", f"{config.test_size:,}"),
        ("Monte Carlo replicates R", f"{config.replicates:,}"),
    ]
    return pd.DataFrame(rows, columns=["Quantity", "Value(s)"])


def _lambda_label(lambda_value: float, config: StudyConfig) -> str:
    lows = min(config.lambda_levels)
    highs = max(config.lambda_levels)
    if float(lambda_value) == float(lows):
        return r"\lambda_{\text{low}}"
    if float(lambda_value) == float(highs):
        return r"\lambda_{\text{high}}"
    return f"{lambda_value:g}"


def _scenario_label(lambda_value: float, sigma_x: float, config: StudyConfig | None = None) -> str:
    # sigma_x is a legacy parameter; keep it in the signature for backwards compatibility.
    if config is not None:
        return f"({ _lambda_label(lambda_value, config) })"
    return f"({lambda_value:g})"


def _panel_label(n_train: int, idx: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return f"Panel {alphabet[idx]}: training size n={n_train:,}"


def build_main_discrimination_table(summary: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    rows = []
    for idx, n_train in enumerate(config.training_sizes):
        panel_header = {"Scenario": _panel_label(n_train, idx)}
        panel_header.update({method: "" for method in PAPER_METHOD_ORDER})
        rows.append(panel_header)

        subset = summary[summary["n_train"] == n_train]
        for lambda_value in config.lambda_levels:
            scen = subset[subset["lambda"] == lambda_value]
            # If sigma_x accidentally appears from older runs, collapse by taking the first row per method.
            row1 = {"Scenario": _scenario_label(lambda_value, sigma_x=0.0, config=config)}
            row2 = {"Scenario": ""}
            for method in PAPER_METHOD_ORDER:
                cell = scen[scen["method"] == method].iloc[0]
                row1[method] = format_mean_se(cell["AUROC_mean"], cell["AUROC_se"])
                row2[method] = f"({format_mean_se(cell['TNR@TPR95_mean'], cell['TNR@TPR95_se'])})"
            rows.append(row1)
            rows.append(row2)
    return pd.DataFrame(rows)


def build_main_probability_table(summary: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    rows = []
    metrics = [
        ("AUPRC", "AUPRC"),
        ("BS", "Brier"),
        ("LL", "LogLoss"),
    ]

    for idx, n_train in enumerate(config.training_sizes):
        panel_header = {"Scenario": _panel_label(n_train, idx), "Metric": ""}
        panel_header.update({method: "" for method in PAPER_METHOD_ORDER})
        rows.append(panel_header)

        subset = summary[summary["n_train"] == n_train]
        for lambda_value in config.lambda_levels:
            scen = subset[subset["lambda"] == lambda_value]
            for metric_label, metric_name in metrics:
                row = {
                    "Scenario": _scenario_label(lambda_value, sigma_x=0.0, config=config),
                    "Metric": metric_label,
                }
                for method in PAPER_METHOD_ORDER:
                    cell = scen[scen["method"] == method].iloc[0]
                    row[method] = format_mean_se(cell[f"{metric_name}_mean"], cell[f"{metric_name}_se"])
                rows.append(row)
    return pd.DataFrame(rows)


def build_coef_stability_table(coef_summary: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    rows = []
    feature_order = ["u_det", "u_nondet", "u_obs", "m_detect", "M_hat", "resid_mean", "resid_sd"]
    for idx, n_train in enumerate(config.training_sizes):
        header = {
            "Coefficient": _panel_label(n_train, idx),
            "Scenario": "",
            "Median standardized coefficient": "",
            "Sign stability": "",
        }
        rows.append(header)
        subset = coef_summary[coef_summary["n_train"] == n_train]
        for feature in feature_order:
            sub2 = subset[subset["feature"] == feature]
            for lambda_value in config.lambda_levels:
                # Sigma_x is fixed in the paper; if present in coef_summary from older runs, pick first.
                cell = sub2[sub2["lambda"] == lambda_value]
                if cell.empty:
                    continue
                cell = cell.iloc[0]
                rows.append(
                    {
                        "Coefficient": feature,
                        "Scenario": _scenario_label(lambda_value, sigma_x=0.0, config=config),
                        "Median standardized coefficient": f"{cell['coef_median']:.3f}",
                        "Sign stability": f"{cell['sign_stability']:.3f}",
                    }
                )
    return pd.DataFrame(rows)


def build_misspec_robustness_tables(
    *,
    metrics: pd.DataFrame,
    config: StudyConfig,
    misspec_suffix: str = "__misspec",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build robustness-to-misspecification tables from a *single* simulation run.

    This expects that when expert misspecification is enabled, each evaluated
    method appears twice in `metrics`:
      - baseline: e.g. "LR-Decomp"
      - misspec: e.g. "LR-Decomp__misspec"

    Pairing is done within each (replicate, n_train, lambda).

    Returns
    -------
    tab_auroc, tab_other
        - tab_auroc matches Table `tab:sim-misspec-robustness` specification.
        - tab_other reports the same paired diffs for the remaining metrics.
    """

    required_metrics = ["AUROC", "AUPRC", "Brier", "LogLoss", "TNR@TPR95"]
    key_cols = ["replicate", "n_train", "lambda", "method"]
    df = metrics[key_cols + required_metrics].copy()

    misspec_methods = {m for m in df["method"].unique() if str(m).endswith(misspec_suffix)}
    if not misspec_methods:
        raise ValueError(
            f"No misspecified methods found (suffix='{misspec_suffix}'). "
            "Set expert_misspecification=true in config.py and rerun simulations."
        )

    miss = df[df["method"].astype(str).str.endswith(misspec_suffix)].copy()
    base = df[~df["method"].astype(str).str.endswith(misspec_suffix)].copy()

    miss["method_base"] = miss["method"].astype(str).str.replace(f"{misspec_suffix}$", "", regex=True)

    # Build a minimal RHS with only the baseline metric values, and rename them to *_base
    # so we don't end up with duplicate column labels after the merge.
    base_merge_cols = ["replicate", "n_train", "lambda", "method"] + required_metrics
    base_for_merge = base[base_merge_cols].copy()
    base_for_merge = base_for_merge.rename(columns={m: f"{m}_base" for m in required_metrics})

    merged = miss.merge(
        base_for_merge,
        left_on=["replicate", "n_train", "lambda", "method_base"],
        right_on=["replicate", "n_train", "lambda", "method"],
        how="inner",
        validate="many_to_one",
    )

    # In `merged`, misspecified metric columns keep their original names (e.g. "AUROC"),
    # while baseline metrics are named "AUROC_base", etc.

    miss_only_methods = ["LR-Decomp", "LR-Obs", "RF-Raw+Features"]
    ref_method = "RF-Raw"

    def _summ_mean_se_arr(arr: np.ndarray) -> tuple[float, float]:
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return np.nan, np.nan
        mean = float(np.mean(arr))
        se = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else np.nan
        return mean, se

    # Table 1: AUROC with paired diff.
    rows_auroc: list[dict[str, object]] = []
    for idx, n_train in enumerate(config.training_sizes):
        header = {
            "Scenario": f"\\textit{{{_panel_label(n_train, idx)}}}",
            "LR-Decomp": "",
            "LR-Obs": "",
            "RF-Raw+Features": "",
            "RF-Raw": "",
        }
        rows_auroc.append(header)

        for lambda_value in config.lambda_levels:
            scen_label = _scenario_label(lambda_value, sigma_x=0.0, config=config)
            row = {"Scenario": scen_label}

            for method in miss_only_methods:
                sub = merged[
                    (merged["n_train"] == n_train)
                    & (merged["lambda"] == lambda_value)
                    & (merged["method_base"] == method)
                ]
                auroc_m = sub["AUROC"].to_numpy(dtype=float)
                auroc_b = sub["AUROC_base"].to_numpy(dtype=float)
                auroc_mean, auroc_se = _summ_mean_se_arr(auroc_m)
                diff_mean, diff_se = _summ_mean_se_arr(auroc_m - auroc_b)
                row[method] = f"{format_mean_se(auroc_mean, auroc_se)} ({format_mean_se(diff_mean, diff_se)})"

            sub_rf = base[
                (base["n_train"] == n_train)
                & (base["lambda"] == lambda_value)
                & (base["method"] == ref_method)
            ]
            auroc_mean, auroc_se = _summ_mean_se_arr(sub_rf["AUROC"].to_numpy(dtype=float))
            row[ref_method] = format_mean_se(auroc_mean, auroc_se)
            rows_auroc.append(row)

    tab_auroc = pd.DataFrame(rows_auroc)

    # Table 2: remaining metrics + paired diffs.
    other_metrics = [
        ("TNR@TPR95", "TNR@TPR95"),
        ("AUPRC", "AUPRC"),
        ("BS", "Brier"),
        ("LL", "LogLoss"),
    ]
    rows_other: list[dict[str, object]] = []
    for idx, n_train in enumerate(config.training_sizes):
        header = {
            "Panel": f"{_panel_label(n_train, idx)}",
            "Scenario": "",
            "Metric": "",
            "LR-Decomp": "",
            "LR-Obs": "",
            "RF-Raw+Features": "",
            "RF-Raw": "",
        }
        rows_other.append(header)

        for lambda_value in config.lambda_levels:
            scen_label = _scenario_label(lambda_value, sigma_x=0.0, config=config)
            for metric_label, metric_name in other_metrics:
                row = {
                    "Panel": "",
                    "Scenario": scen_label,
                    "Metric": metric_label,
                }
                for method in miss_only_methods:
                    sub = merged[
                        (merged["n_train"] == n_train)
                        & (merged["lambda"] == lambda_value)
                        & (merged["method_base"] == method)
                    ]
                    m_m = sub[f"{metric_name}"].to_numpy(dtype=float)
                    m_b = sub[f"{metric_name}_base"].to_numpy(dtype=float)
                    m_mean, m_se = _summ_mean_se_arr(m_m)
                    diff_mean, diff_se = _summ_mean_se_arr(m_m - m_b)
                    row[method] = f"{format_mean_se(m_mean, m_se)} ({format_mean_se(diff_mean, diff_se)})"

                sub_rf = base[
                    (base["n_train"] == n_train)
                    & (base["lambda"] == lambda_value)
                    & (base["method"] == ref_method)
                ]
                m_mean, m_se = _summ_mean_se_arr(sub_rf[metric_name].to_numpy(dtype=float))
                row[ref_method] = format_mean_se(m_mean, m_se)
                rows_other.append(row)

    tab_other = pd.DataFrame(rows_other)
    return tab_auroc, tab_other


def export_table(df: pd.DataFrame, out_csv: str | Path, out_tex: str | Path, caption: str, label: str) -> None:
    out_csv = Path(out_csv)
    out_tex = Path(out_tex)
    ensure_dir(out_csv.parent)
    ensure_dir(out_tex.parent)
    df.to_csv(out_csv, index=False)

    tex = df.to_latex(index=False, escape=False, caption=caption, label=label, longtable=False)
    out_tex.write_text(tex, encoding="utf-8")
