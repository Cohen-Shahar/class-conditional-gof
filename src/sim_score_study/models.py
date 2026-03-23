from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import StudyConfig
from .features import FeatureBundle, feature_columns_for_method
from .metrics import compute_metrics


@dataclass
class ModelResult:
    metrics: dict[str, float]
    probabilities: np.ndarray
    standardized_coefficients: pd.Series | None


def fit_predict_logistic(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    feature_names: list[str],
    config: StudyConfig,
    *,
    random_state: int,
) -> tuple[np.ndarray, pd.Series, object]:
    lr_kwargs: dict[str, object] = {
        "C": config.lr_C,
        "fit_intercept": config.lr_fit_intercept,
        "solver": "lbfgs",
        "max_iter": config.lr_max_iter,
        "random_state": int(random_state),
    }

    penalty = str(getattr(config, "lr_penalty", "l2")).strip().lower()
    if penalty in {"", "l2"}:
        # Keep sklearn defaults for L2 to avoid deprecation warnings.
        pass
    elif penalty in {"none", "null"}:
        lr_kwargs["C"] = np.inf
    else:
        raise ValueError(
            f"Unsupported lr_penalty={config.lr_penalty!r} with solver='lbfgs'. "
            "Use 'l2' or 'none'."
        )

    pipe = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(**lr_kwargs),
            ),
        ]
    )
    pipe.fit(X_train[feature_names], y_train)
    probs = pipe.predict_proba(X_test[feature_names])[:, 1]
    coefs = pd.Series(pipe.named_steps["clf"].coef_[0], index=feature_names, dtype=float)
    return probs, coefs, pipe


def fit_predict_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    feature_names: list[str],
    config: StudyConfig,
    random_state: int,
) -> tuple[np.ndarray, object]:
    clf = RandomForestClassifier(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        min_samples_leaf=config.rf_min_samples_leaf,
        max_features=config.rf_max_features,
        n_jobs=int(getattr(config, "rf_n_jobs", 1)),
        random_state=int(random_state),
    )
    clf.fit(X_train[feature_names], y_train)
    return clf.predict_proba(X_test[feature_names])[:, 1], clf


def evaluate_methods(
    train_bundle: FeatureBundle,
    test_bundle: FeatureBundle,
    config: StudyConfig,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit each method on the training bundle and evaluate on the test bundle.

    Returns
    -------
    metrics_df : pd.DataFrame
        Per-method metrics (AUROC, etc.) from `compute_metrics`.
    coef_df : pd.DataFrame
        Standardized coefficients for LR methods when requested; empty otherwise.
    """

    train_full = pd.concat(
        [train_bundle.raw.reset_index(drop=True), train_bundle.features.reset_index(drop=True)],
        axis=1,
    )
    test_full = pd.concat(
        [test_bundle.raw.reset_index(drop=True), test_bundle.features.reset_index(drop=True)],
        axis=1,
    )

    metrics_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []

    feature_map = feature_columns_for_method(config)

    for method, feature_names in feature_map.items():
        if len(feature_names) == 0:
            continue

        missing = [c for c in feature_names if c not in train_full.columns or c not in test_full.columns]
        if missing:
            raise KeyError(f"Missing required columns for method {method}: {missing}")

        if method.startswith("LR-"):
            probs, coef_series, _model_obj = fit_predict_logistic(
                train_full,
                train_bundle.labels,
                test_full,
                feature_names,
                config,
                random_state=random_state,
            )
            standardized_coefficients = coef_series if method == "LR-Decomp" else None
        else:
            probs, _model_obj = fit_predict_random_forest(
                train_full,
                train_bundle.labels,
                test_full,
                feature_names,
                config,
                random_state=random_state,
            )
            standardized_coefficients = None

        metric_dict = compute_metrics(test_bundle.labels, probs)
        metric_dict["method"] = method
        metrics_rows.append(metric_dict)

        if standardized_coefficients is not None:
            for feature_name, coef in standardized_coefficients.items():
                coef_rows.append(
                    {
                        "method": method,
                        "feature": feature_name,
                        "coef_standardized": float(coef),
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    coef_df = pd.DataFrame(coef_rows)
    return metrics_df, coef_df
