from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StudyConfig
from .dgp import SourceDesign
from .fitting import FittedLatents


SCORE_COLUMNS = ["u_det", "u_nondet", "u_obs", "u_tot"]
AUX_COLUMNS = ["m_detect", "M_hat", "resid_mean", "resid_sd"]
ENGINEERED_COLUMNS = SCORE_COLUMNS + AUX_COLUMNS


@dataclass
class FeatureBundle:
    features: pd.DataFrame
    raw: pd.DataFrame
    labels: np.ndarray


def _gaussian_logpdf(x: np.ndarray, mu: np.ndarray, sigma_x: float) -> np.ndarray:
    return -0.5 * np.log(2.0 * np.pi * sigma_x**2) - 0.5 * ((x - mu) ** 2) / (sigma_x**2)


def compute_feature_bundle(
    dataset: dict,
    fitted: FittedLatents,
    lambda_value: float,
    sigma_x: float,
    config: StudyConfig,
    design: SourceDesign,
) -> FeatureBundle:
    X = dataset["X"]
    D = dataset["D"].astype(int)
    y = dataset["y"].astype(int)

    alpha0 = config.alpha0_for_lambda(lambda_value)

    mu = (
        config.beta0
        + config.beta_M * fitted.M_hat.reshape(-1, 1)
        - config.beta_d * np.abs(fitted.L_hat.reshape(-1, 1) - design.locations.reshape(1, -1))
    )
    eta = (
        alpha0
        + design.alpha0s.reshape(1, -1)
        + lambda_value
        * (
            config.alpha_M * fitted.M_hat.reshape(-1, 1)
            - config.alpha_d * np.abs(fitted.L_hat.reshape(-1, 1) - design.locations.reshape(1, -1))
        )
    )
    log_p = -np.logaddexp(0.0, -eta)
    log_1mp = -np.logaddexp(0.0, eta)

    obs_logpdf = np.zeros_like(mu, dtype=float)
    obs_mask = D.astype(bool)
    obs_logpdf[obs_mask] = _gaussian_logpdf(X[obs_mask], mu[obs_mask], sigma_x=sigma_x)

    m_detect = D.sum(axis=1).astype(float)

    det_denom = np.maximum(m_detect, 1.0)
    nondet_denom = np.maximum((float(config.S) - m_detect), 1.0)

    u_det = (D * log_p).sum(axis=1) / det_denom
    u_nondet = ((1 - D) * log_1mp).sum(axis=1) / nondet_denom
    u_obs = obs_logpdf.sum(axis=1) / det_denom
    u_tot = ((D * log_p) + ((1 - D) * log_1mp) + obs_logpdf).sum(axis=1) / float(config.S)

    residuals = np.where(obs_mask, X - mu, np.nan)
    resid_mean = np.nanmean(residuals, axis=1)
    resid_mean = np.where(np.isnan(resid_mean), 0.0, resid_mean)

    resid_sd = np.nanstd(residuals, axis=1, ddof=1)
    resid_sd = np.where(np.isnan(resid_sd), 0.0, resid_sd)

    features = pd.DataFrame(
        {
            "u_det": u_det,
            "u_nondet": u_nondet,
            "u_obs": u_obs,
            "u_tot": u_tot,
            "m_detect": m_detect,
            "M_hat": fitted.M_hat,
            "resid_mean": resid_mean,
            "resid_sd": resid_sd,
        }
    )

    raw_x0 = np.where(obs_mask, X, 0.0)
    raw_cols = {f"x0_{s+1:02d}": raw_x0[:, s] for s in range(config.S)}
    raw_cols.update({f"d_{s+1:02d}": D[:, s] for s in range(config.S)})
    raw = pd.DataFrame(raw_cols)

    return FeatureBundle(features=features, raw=raw, labels=y)


def feature_columns_for_method(config: StudyConfig) -> dict[str, list[str]]:
    mapping = {
        "LR-Decomp": ["u_det", "u_nondet", "u_obs", *AUX_COLUMNS],
        # Decision trees are intentionally excluded from the main protocol; see scripts/run_dt.py.
        "LR-Total": ["u_tot", *AUX_COLUMNS],
        "LR-Obs": ["u_obs", *AUX_COLUMNS],
        "LR-Baseline": AUX_COLUMNS.copy(),
        "RF-Raw": [*(f"x0_{s+1:02d}" for s in range(config.S)), *(f"d_{s+1:02d}" for s in range(config.S))],
        "RF-Raw+Features": [
            *(f"x0_{s+1:02d}" for s in range(config.S)),
            *(f"d_{s+1:02d}" for s in range(config.S)),
            # RF-Raw+Features uses all engineered columns *except* u_tot.
            "u_det",
            "u_nondet",
            "u_obs",
            *AUX_COLUMNS,
        ],
    }

    allowed = getattr(config, "methods", None)
    if allowed is None:
        return mapping
    allowed_set = set(allowed)
    return {k: v for k, v in mapping.items() if k in allowed_set}
