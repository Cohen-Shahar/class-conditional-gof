from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import StudyConfig
from .dgp import SourceDesign, generate_dataset
from .features import compute_feature_bundle
from .fitting import fit_latent_states
from .models import evaluate_methods
from .utils import ensure_dir, save_json, sort_metric_frame


def _cell_stem(replicate: int, n_train: int, lambda_value: float) -> str:
    return f"rep_{replicate:03d}__n_{n_train}__lam_{lambda_value:g}"


def _run_expert_pipeline(
    *,
    train: dict,
    test: dict,
    lambda_value: float,
    sigma_x: float,
    config_for_features: StudyConfig,
    design: SourceDesign,
    n_jobs: int | None,
    eval_config: StudyConfig,
    base_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Fit latents + compute engineered features under `config_for_features` and evaluate methods.

    Returns (metrics_df, coef_df, train_converged_rate, test_converged_rate).
    """
    train_fit = fit_latent_states(
        train["X"],
        train["D"],
        lambda_value,
        sigma_x,
        config_for_features,
        design,
        n_jobs=n_jobs,
    )
    test_fit = fit_latent_states(
        test["X"],
        test["D"],
        lambda_value,
        sigma_x,
        config_for_features,
        design,
        n_jobs=n_jobs,
    )

    train_bundle = compute_feature_bundle(train, train_fit, lambda_value, sigma_x, config_for_features, design)
    test_bundle = compute_feature_bundle(test, test_fit, lambda_value, sigma_x, config_for_features, design)

    metrics_df, coef_df = evaluate_methods(
        train_bundle=train_bundle,
        test_bundle=test_bundle,
        config=eval_config,
        random_state=base_seed + 123,
    )
    return metrics_df, coef_df, float(train_fit.converged.mean()), float(test_fit.converged.mean())


def run_single_cell(
    replicate: int,
    n_train: int,
    lambda_value: float,
    sigma_x: float,
    config: StudyConfig,
    design: SourceDesign,
    output_root: str | Path,
    overwrite: bool = False,
    n_jobs: int | None = None,
    *,
    save_pooled_scores: bool = False,
    save_models: bool = False,
) -> Path:
    output_root = Path(output_root)
    cell_dir = ensure_dir(output_root / "cells")
    out_path = cell_dir / f"{_cell_stem(replicate, n_train, lambda_value)}.pkl"
    if out_path.exists() and not overwrite:
        return out_path

    import numpy as np

    base_seed = config.random_seed + 10_000 * replicate + 97 * n_train
    rng_train = np.random.default_rng(base_seed + 1 + int(100 * lambda_value) + int(1000 * sigma_x))
    rng_test = np.random.default_rng(base_seed + 2 + int(100 * lambda_value) + int(1000 * sigma_x))

    # Data are always generated from the "true" config.
    train = generate_dataset(n_train, lambda_value, sigma_x, config, design, rng_train)
    test = generate_dataset(config.test_size, lambda_value, sigma_x, config, design, rng_test)

    # --- Baseline expert pipeline (well-specified) ---
    metrics_df, coef_df, train_conv, test_conv = _run_expert_pipeline(
        train=train,
        test=test,
        lambda_value=lambda_value,
        sigma_x=sigma_x,
        config_for_features=config,
        design=design,
        n_jobs=n_jobs,
        eval_config=config,
        base_seed=base_seed,
    )

    misspec_metrics_df = None
    misspec_coef_df = None
    misspec_train_conv = None
    misspec_test_conv = None
    misspec_factors = None

    # --- Optional misspecified expert pipeline (same data, different expert model for features) ---
    if bool(getattr(config, "expert_misspecification", False)):
        from .misspecification import build_misspecified_expert_config

        expert_config, misspec_factors = build_misspecified_expert_config(
            config,
            replicate=replicate,
            pct=float(getattr(config, "expert_misspecification_pct", 0.1)),
        )

        misspec_metrics_df, misspec_coef_df, misspec_train_conv, misspec_test_conv = _run_expert_pipeline(
            train=train,
            test=test,
            lambda_value=lambda_value,
            sigma_x=sigma_x,
            config_for_features=expert_config,
            design=design,
            n_jobs=n_jobs,
            # Evaluate the same restricted method set when configured.
            eval_config=config,
            base_seed=base_seed,
        )

        # Tag misspecified rows so they can co-exist with baseline in the same run.
        misspec_metrics_df = misspec_metrics_df.copy()
        misspec_metrics_df["method"] = misspec_metrics_df["method"].astype(str) + "__misspec"

        misspec_coef_df = misspec_coef_df.copy()
        misspec_coef_df["method"] = misspec_coef_df["method"].astype(str) + "__misspec"

    # Attach cell identifiers.
    metrics_df["replicate"] = replicate
    metrics_df["n_train"] = n_train
    metrics_df["lambda"] = lambda_value
    metrics_df = sort_metric_frame(metrics_df)

    coef_df["replicate"] = replicate
    coef_df["n_train"] = n_train
    coef_df["lambda"] = lambda_value

    if misspec_metrics_df is not None:
        misspec_metrics_df["replicate"] = replicate
        misspec_metrics_df["n_train"] = n_train
        misspec_metrics_df["lambda"] = lambda_value
        misspec_metrics_df = sort_metric_frame(misspec_metrics_df)

        misspec_coef_df["replicate"] = replicate
        misspec_coef_df["n_train"] = n_train
        misspec_coef_df["lambda"] = lambda_value

        metrics_out = pd.concat([metrics_df, misspec_metrics_df], ignore_index=True)
        coef_out = pd.concat([coef_df, misspec_coef_df], ignore_index=True)
    else:
        metrics_out = metrics_df
        coef_out = coef_df

    # NOTE: pooled scores are only saved for the baseline expert model.
    if save_pooled_scores:
        # Recompute baseline bundles for pooled export only (keeps payload small elsewhere).
        # (We could thread the bundles through, but we intentionally keep payload minimal.)
        train_fit = fit_latent_states(train["X"], train["D"], lambda_value, sigma_x, config, design, n_jobs=n_jobs)
        test_fit = fit_latent_states(test["X"], test["D"], lambda_value, sigma_x, config, design, n_jobs=n_jobs)
        test_bundle = compute_feature_bundle(test, test_fit, lambda_value, sigma_x, config, design)

        pooled_scores = test_bundle.features.copy()
        pooled_scores["y"] = test["y"]
        pooled_scores["replicate"] = replicate
        pooled_scores["n_train"] = n_train
        pooled_scores["lambda"] = lambda_value
        pooled_scores["sigma_x"] = sigma_x
    else:
        pooled_scores = None

    # TODO: save_models currently only supports baseline. (Kept as-is; rarely used and avoids doubling pickles.)
    models = None
    if save_models:
        from .models import fit_predict_logistic, fit_predict_random_forest
        from .features import feature_columns_for_method

        # baseline bundles for model saving
        train_fit = fit_latent_states(train["X"], train["D"], lambda_value, sigma_x, config, design, n_jobs=n_jobs)
        test_fit = fit_latent_states(test["X"], test["D"], lambda_value, sigma_x, config, design, n_jobs=n_jobs)
        train_bundle = compute_feature_bundle(train, train_fit, lambda_value, sigma_x, config, design)
        test_bundle = compute_feature_bundle(test, test_fit, lambda_value, sigma_x, config, design)

        train_full = pd.concat(
            [train_bundle.raw.reset_index(drop=True), train_bundle.features.reset_index(drop=True)],
            axis=1,
        )
        test_full = pd.concat(
            [test_bundle.raw.reset_index(drop=True), test_bundle.features.reset_index(drop=True)],
            axis=1,
        )
        feature_map = feature_columns_for_method(config)
        models = {}
        for method, feature_names in feature_map.items():
            if len(feature_names) == 0:
                continue
            if method.startswith("LR-"):
                _probs, _coefs, model_obj = fit_predict_logistic(
                    train_full,
                    train_bundle.labels,
                    test_full,
                    feature_names,
                    config,
                    random_state=base_seed + 123,
                )
                models[method] = model_obj
            else:
                _probs, model_obj = fit_predict_random_forest(
                    train_full,
                    train_bundle.labels,
                    test_full,
                    feature_names,
                    config,
                    random_state=base_seed + 123,
                )
                models[method] = model_obj

    payload = {
        "metrics": metrics_out,
        "coefficients": coef_out,
        "pooled_scores": pooled_scores,
        "models": models,
        "metadata": {
            "replicate": replicate,
            "n_train": n_train,
            "lambda": lambda_value,
            "train_fit_converged_rate": float(train_conv),
            "test_fit_converged_rate": float(test_conv),
            "expert_misspecification": bool(getattr(config, "expert_misspecification", False)),
            "expert_misspecification_pct": float(getattr(config, "expert_misspecification_pct", 0.0)),
            "expert_misspecification_factors": misspec_factors,
            "train_fit_converged_rate__misspec": None if misspec_train_conv is None else float(misspec_train_conv),
            "test_fit_converged_rate__misspec": None if misspec_test_conv is None else float(misspec_test_conv),
        },
    }
    pd.to_pickle(payload, out_path)
    return out_path


def write_run_manifest(
    config: StudyConfig,
    output_root: str | Path,
    *,
    design: SourceDesign | None = None,
    replicate: int | None = None,
) -> None:
    output_root = Path(output_root)
    ensure_dir(output_root)

    # Always write config once (idempotent).
    save_json(output_root / "config.json", config.to_dict())

    if design is None:
        return

    # Backward-compatible default path when replicate isn't provided.
    if replicate is None:
        out = output_root / "source_design.json"
    else:
        # Store per-replicate designs in a dedicated subfolder to keep the
        # results root tidy.
        design_root = output_root / "source_designs"
        ensure_dir(design_root)
        out = design_root / f"source_design__rep_{int(replicate):03d}.json"

    save_json(out, design.to_dict())
