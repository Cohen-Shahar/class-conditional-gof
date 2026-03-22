from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Iterable

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import logsumexp

from .config import StudyConfig
from .dgp import SourceDesign


@dataclass
class FittedLatents:
    L_hat: np.ndarray
    M_hat: np.ndarray
    loglik: np.ndarray
    converged: np.ndarray


def _safe_log_p_and_log1m_p(eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # log(sigmoid(eta)) and log(1 - sigmoid(eta))
    log_p = -np.logaddexp(0.0, -eta)
    log_1mp = -np.logaddexp(0.0, eta)
    return log_p, log_1mp


def observed_loglik_instance(
    params: np.ndarray,
    x_row: np.ndarray,
    d_row: np.ndarray,
    lambda_value: float,
    sigma_x: float,
    config: StudyConfig,
    design: SourceDesign,
) -> float:
    L, M = float(params[0]), float(params[1])

    alpha0 = config.alpha0_for_lambda(lambda_value)

    eta = (
        alpha0
        + design.alpha0s
        + lambda_value * (config.alpha_M * M - config.alpha_d * np.abs(L - design.locations))
    )
    log_p, log_1mp = _safe_log_p_and_log1m_p(eta)
    ll_det = np.sum(d_row * log_p + (1 - d_row) * log_1mp)
    mu = config.beta0 + config.beta_M * M - config.beta_d * np.abs(L - design.locations)
    obs_idx = d_row.astype(bool)
    if np.any(obs_idx):
        resid = x_row[obs_idx] - mu[obs_idx]
        ll_obs = np.sum(-0.5 * np.log(2.0 * np.pi * sigma_x**2) - 0.5 * (resid**2) / (sigma_x**2))
    else:
        ll_obs = 0.0
    return float(ll_det + ll_obs)


def observed_loglik_gradient(
    params: np.ndarray,
    x_row: np.ndarray,
    d_row: np.ndarray,
    lambda_value: float,
    sigma_x: float,
    config: StudyConfig,
    design: SourceDesign,
) -> np.ndarray:
    L, M = float(params[0]), float(params[1])
    diff = L - design.locations
    abs_diff = np.abs(diff)
    sign_diff = np.sign(diff)

    alpha0 = config.alpha0_for_lambda(lambda_value)

    eta = alpha0 + design.alpha0s + lambda_value * (config.alpha_M * M - config.alpha_d * abs_diff)
    p = 1.0 / (1.0 + np.exp(-eta))
    det_resid = d_row - p

    grad_det_L = np.sum(det_resid * (-lambda_value * config.alpha_d * sign_diff))
    grad_det_M = np.sum(det_resid * (lambda_value * config.alpha_M))

    mu = config.beta0 + config.beta_M * M - config.beta_d * abs_diff
    obs_idx = d_row.astype(bool)
    if np.any(obs_idx):
        resid = x_row[obs_idx] - mu[obs_idx]
        common = resid / (sigma_x**2)
        grad_obs_L = np.sum(common * (-config.beta_d * sign_diff[obs_idx]))
        grad_obs_M = np.sum(common * config.beta_M)
    else:
        grad_obs_L = 0.0
        grad_obs_M = 0.0

    return np.array([grad_det_L + grad_obs_L, grad_det_M + grad_obs_M], dtype=float)


def _empirical_init(
    x_row: np.ndarray,
    d_row: np.ndarray,
    config: StudyConfig,
    design: SourceDesign,
) -> tuple[float, float]:
    det_idx = d_row.astype(bool)
    if np.any(det_idx):
        L0 = float(np.clip(np.mean(design.locations[det_idx]), 0.0, 1.0))
        corrected = np.nanmean(x_row[det_idx] + config.beta_d * np.abs(L0 - design.locations[det_idx]))
        M0 = (corrected - config.beta0) / max(config.beta_M, 1e-8)
    else:
        L0 = 0.5
        M0 = config.latent_M_mean
    low, high = config.M_bounds
    return L0, float(np.clip(M0, low, high))


def fit_single_event(
    x_row: np.ndarray,
    d_row: np.ndarray,
    lambda_value: float,
    sigma_x: float,
    config: StudyConfig,
    design: SourceDesign,
) -> tuple[float, float, float, bool]:
    starts: list[tuple[float, float]] = []
    if config.optimizer_reuse_empirical_init:
        starts.append(_empirical_init(x_row, d_row, config, design))
    starts.extend((L0, config.latent_M_mean) for L0 in config.optimizer_multistart_L)

    best_result = None
    for L0, M0 in starts:
        objective = lambda p: -observed_loglik_instance(p, x_row, d_row, lambda_value, sigma_x, config, design)
        gradient = lambda p: -observed_loglik_gradient(p, x_row, d_row, lambda_value, sigma_x, config, design)
        res = minimize(
            objective,
            x0=np.array([L0, M0], dtype=float),
            jac=gradient,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0), tuple(config.M_bounds)],
            options={"maxiter": config.optimizer_maxiter},
        )
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    assert best_result is not None
    ll = -float(best_result.fun)
    return (
        float(best_result.x[0]),
        float(best_result.x[1]),
        ll,
        bool(best_result.success),
    )


def fit_latent_states(
    X: np.ndarray,
    D: np.ndarray,
    lambda_value: float,
    sigma_x: float,
    config: StudyConfig,
    design: SourceDesign,
    n_jobs: int | None = None,
) -> FittedLatents:
    n_jobs = config.fit_n_jobs if n_jobs is None else n_jobs
    if n_jobs == 1:
        rows = [
            fit_single_event(X[i], D[i], lambda_value, sigma_x, config, design)
            for i in range(X.shape[0])
        ]
    else:
        worker = delayed(fit_single_event)
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(
            worker(X[i], D[i], lambda_value, sigma_x, config, design)
            for i in range(X.shape[0])
        )
    arr = np.asarray(rows, dtype=float)
    converged = np.asarray([row[3] for row in rows], dtype=bool)
    return FittedLatents(
        L_hat=arr[:, 0],
        M_hat=arr[:, 1],
        loglik=arr[:, 2],
        converged=converged,
    )
