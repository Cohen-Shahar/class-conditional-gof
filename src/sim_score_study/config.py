from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import numpy as np


@dataclass
class StudyConfig:
    name: str
    random_seed: int
    S: int
    prevalence: float
    latent_L_low: float
    latent_L_high: float
    latent_M_mean: float
    latent_M_sd: float
    # Base/intercept term in the detection model.
    # This study uses per-lambda calibration; alpha0 is selected elementwise from alpha0_levels.
    alpha0_levels: List[float]
    alpha_M: float
    alpha_d: float
    beta0: float
    beta_M: float
    beta_d: float
    lambda_levels: List[float]
    # Observation noise levels. This codebase supports varying sigma_x, but the paper protocol
    # fixes sigma_x to a single value.
    sigma_x_levels: List[float]
    gamma: float
    # Which Y=0 (invalid) data-generating mechanism to use.
    # - "composite": current protocol mechanism mixing evidence from two pseudo-events.
    # - "valid_latent_mcar_detection": sample (L,M) like a valid event and sample X similarly,
    #   but make detections MCAR with a fixed per-station detection probability.
    invalid_dgp: str
    invalid_mcar_detection_prob: float
    # Hybrid invalid (Y=0) mechanism: with probability p_mal_mix, use the
    # composite pseudo-event mechanism; with probability 1-p_mal_mix, use the
    # valid-latent + MCAR-detection mechanism.
    invalid_p_mal_mix: float
    training_sizes: List[int]
    test_size: int
    replicates: int
    min_detecting_sources: int
    M_bounds: List[float]
    fit_n_jobs: int
    rf_n_estimators: int
    rf_max_depth: int | None
    rf_min_samples_leaf: int
    rf_max_features: str
    rf_n_jobs: int
    lr_penalty: str
    lr_C: float
    lr_fit_intercept: bool
    lr_max_iter: int
    optimizer_maxiter: int
    optimizer_multistart_L: List[float]
    optimizer_reuse_empirical_init: bool

    # Optional robustness experiment: use a misspecified expert model for fitting/features.
    # Data are still generated from the "true" config; only the expert model (used to compute
    # L_hat/M_hat and all engineered features) is perturbed.
    expert_misspecification: bool = True
    # Multiplicative perturbation size: each expert parameter is multiplied by (1±p).
    expert_misspecification_pct: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def alpha0_for_lambda(self, lambda_value: float) -> float:
        """Return the calibrated alpha0 for a given lambda_value.

        alpha0_levels is paired elementwise with lambda_levels.
        """
        if len(self.alpha0_levels) != len(self.lambda_levels):
            raise ValueError(
                "alpha0_levels must have the same length as lambda_levels "
                f"(got {len(self.alpha0_levels)} vs {len(self.lambda_levels)})"
            )
        try:
            idx = list(self.lambda_levels).index(float(lambda_value))
        except ValueError as e:
            raise ValueError(
                f"lambda_value={lambda_value:g} not found in config.lambda_levels; cannot select calibrated alpha0"
            ) from e
        return float(self.alpha0_levels[idx])


def _base_config(name: str) -> Dict[str, Any]:
    # NOTE:
    # The protocol leaves several entries as [fill in]. The values below are
    # reasonable defaults chosen so that:
    # 1) detection is neither trivial nor degenerate;
    # 2) informative missingness increases with lambda; and
    # 3) the observation signal degrades as sigma_x grows.
    #
    # If the paper finalizes different calibration values, change only these
    # defaults and rerun the full pipeline. All figures/tables will update.
    return {
        "name": name,
        "random_seed": 20260301,
        "S": 50,
        "prevalence": 0.5,
        "latent_L_low": 0.0,
        "latent_L_high": 1.0,
        "latent_M_mean": 10.0,
        "latent_M_sd": 2.0,
        # Calibrated alpha0 per lambda (paired elementwise with lambda_levels).
        "alpha0_levels": [-2.2, -2.82],
        "alpha_M": 0.16,
        "alpha_d": 12.0,
        "beta0": 0.0,
        "beta_M": 1.0,
        "beta_d": 4.0,
        "lambda_levels": [1.0, 2.0],
        "sigma_x_levels": [1.0],
        "gamma": 0.5,
        "invalid_dgp": "composite",
        "invalid_mcar_detection_prob": 0.1,
        "invalid_p_mal_mix": 0.5,
        "training_sizes": [100, 1_000, 10_000],
        "test_size": 5_000,
        "replicates": 500,
        "min_detecting_sources": 2,
        "M_bounds": [0.0, 20.0],
        "fit_n_jobs": 1,
        "rf_n_estimators": 500,
        "rf_max_depth": None,
        "rf_min_samples_leaf": 1,
        "rf_max_features": "sqrt",
        "rf_n_jobs": 1,
        "lr_penalty": "l2",
        "lr_C": 1e6,
        "lr_fit_intercept": True,
        "lr_max_iter": 5000,
        "optimizer_maxiter": 250,
        "optimizer_multistart_L": [0.2, 0.5, 0.8],
        "optimizer_reuse_empirical_init": True,

        # Robustness experiment defaults (enabled by default).
        "expert_misspecification_pct": 0.25,
        "expert_misspecification": True,
    }


_CONFIGS: Dict[str, Dict[str, Any]] = {
    "paper": _base_config("paper"),

    "paper_light": {
        **_base_config("paper_light"),
        "replicates": 30,
        "test_size": 5000,
        "rf_n_estimators": 300,
    },

    # Quick end-to-end run to inspect the *actual* paper-format plots.
    # - All training sizes
    # - Two lambda scenarios (low/high)
    # - Only 3 Monte Carlo replicates
    "smoke": {
        **_base_config("smoke"),
        "replicates": 3,
        "test_size": 500,
        "rf_n_estimators": 200,
        "optimizer_maxiter": 80,
    },
}


def list_configs() -> List[str]:
    return sorted(_CONFIGS)


def get_config(name: str) -> StudyConfig:
    if name not in _CONFIGS:
        raise KeyError(f"Unknown config '{name}'. Available: {list_configs()}")
    payload = deepcopy(_CONFIGS[name])
    # Backwards compatibility: older runs/configs may include legacy keys.
    payload.pop("representative_sigma_strategy", None)
    payload.pop("gamma_label", None)
    payload.pop("alpha0", None)
    return StudyConfig(**payload)


def build_source_locations(S: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, S)


def sample_fixed_source_noise(S: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=S)
