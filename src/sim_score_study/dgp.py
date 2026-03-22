from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.special import expit

from .config import StudyConfig, sample_fixed_source_noise


@dataclass
class SourceDesign:
    locations: np.ndarray
    alpha0s: np.ndarray
    # Seed used for this design draw (for reproducibility / manifests).
    seed: int | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "locations": self.locations.tolist(),
            "alpha0s": self.alpha0s.tolist(),
            "seed": None if self.seed is None else int(self.seed),
        }


def build_source_design(
    config: StudyConfig,
    *,
    replicate: int | None = None,
    seed_offset: int = 0,
) -> SourceDesign:
    """Build (and optionally re-sample) the per-source design parameters.

    Historically, the study held locations/alpha-shifts fixed across all replicates.
    Passing a replicate index makes these parameters re-sampled deterministically
    for each replicate.

    Seeding rule:
      seed = config.random_seed + seed_offset + 10_000 * replicate + 17
      (when replicate is None, we use replicate=0)
    """
    rep = 0 if replicate is None else int(replicate)
    design_seed = int(config.random_seed) + int(seed_offset) + 10_000 * rep + 17

    # Locations and detection shifts are drawn from the distributions specified
    # in config.py helpers. Using the same seed makes the joint draw reproducible.
    rng = np.random.default_rng(design_seed)

    # Station locations: re-sample i.i.d. U[0,1] each replicate.
    # - This replaces the older "fixed grid + permutation" design.
    # - Deterministic within replicate (via design_seed), different across replicates.
    locations = rng.uniform(0.0, 1.0, size=config.S)

    # Keep using the helper distribution, but tie it to this replicate's seed.
    alpha0s = sample_fixed_source_noise(config.S, seed=design_seed)
    return SourceDesign(locations=locations, alpha0s=alpha0s, seed=design_seed)


def sample_latent_states(n: int, config: StudyConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    L = rng.uniform(config.latent_L_low, config.latent_L_high, size=n)
    M = rng.normal(config.latent_M_mean, config.latent_M_sd, size=n)
    return L, M


def detection_probabilities(
    L: np.ndarray,
    M: np.ndarray,
    lambda_value: float,
    config: StudyConfig,
    design: SourceDesign,
) -> np.ndarray:
    L = np.asarray(L).reshape(-1, 1)
    M = np.asarray(M).reshape(-1, 1)

    alpha0 = config.alpha0_for_lambda(lambda_value)

    eta = (
        alpha0
        + design.alpha0s.reshape(1, -1)
        + lambda_value
        * (config.alpha_M * M - config.alpha_d * np.abs(L - design.locations.reshape(1, -1)))
    )
    return expit(eta)


def observation_means(
    L: np.ndarray,
    M: np.ndarray,
    config: StudyConfig,
    design: SourceDesign,
) -> np.ndarray:
    L = np.asarray(L).reshape(-1, 1)
    M = np.asarray(M).reshape(-1, 1)
    return config.beta0 + config.beta_M * M - config.beta_d * np.abs(L - design.locations.reshape(1, -1))


def _finalize_batch(
    D: np.ndarray,
    X: np.ndarray,
    extra: dict,
    min_detecting_sources: int,
) -> dict:
    keep = D.sum(axis=1) >= min_detecting_sources
    return {
        "D": D[keep],
        "X": X[keep],
        **{k: v[keep] for k, v in extra.items()},
    }


def generate_valid_events(
    n: int,
    lambda_value: float,
    sigma_x: float,
    config: StudyConfig,
    design: SourceDesign,
    rng: np.random.Generator,
) -> dict:
    if n == 0:
        return {"D": np.empty((0, config.S), dtype=np.int8), "X": np.empty((0, config.S), dtype=float), "L_true": np.array([], dtype=float), "M_true": np.array([], dtype=float)}
    collected = {"D": [], "X": [], "L_true": [], "M_true": []}
    remaining = n
    while remaining > 0:
        batch_n = max(remaining * 2, 256)
        L, M = sample_latent_states(batch_n, config, rng)
        p = detection_probabilities(L, M, lambda_value, config, design)
        D = rng.binomial(1, p).astype(np.int8)
        mu = observation_means(L, M, config, design)
        X = mu + rng.normal(0.0, sigma_x, size=mu.shape)
        X[D == 0] = np.nan
        batch = _finalize_batch(D, X, {"L_true": L, "M_true": M}, config.min_detecting_sources)
        take = min(remaining, batch["D"].shape[0])
        for key in collected:
            collected[key].append(batch[key][:take])
        remaining -= take
    return {k: np.concatenate(v, axis=0) for k, v in collected.items()}


def generate_invalid_events(
    n: int,
    lambda_value: float,
    sigma_x: float,
    gamma: float,
    config: StudyConfig,
    design: SourceDesign,
    rng: np.random.Generator,
) -> dict:
    if n == 0:
        empty = np.empty((0, config.S), dtype=np.int8)
        return {"D": empty, "X": np.empty((0, config.S), dtype=float), "L_A": np.array([], dtype=float), "M_A": np.array([], dtype=float), "L_B": np.array([], dtype=float), "M_B": np.array([], dtype=float)}
    collected = {"D": [], "X": [], "L_A": [], "M_A": [], "L_B": [], "M_B": []}
    remaining = n
    while remaining > 0:
        batch_n = max(remaining * 2, 256)
        L_A, M_A = sample_latent_states(batch_n, config, rng)
        L_B, M_B = sample_latent_states(batch_n, config, rng)
        selectors = rng.binomial(1, gamma, size=(batch_n, config.S)).astype(bool)
        L_chosen = np.where(selectors, L_A.reshape(-1, 1), L_B.reshape(-1, 1))
        M_chosen = np.where(selectors, M_A.reshape(-1, 1), M_B.reshape(-1, 1))

        alpha0 = config.alpha0_for_lambda(lambda_value)

        eta = (
            alpha0
            + design.alpha0s.reshape(1, -1)
            + lambda_value * (config.alpha_M * M_chosen - config.alpha_d * np.abs(L_chosen - design.locations.reshape(1, -1)))
        )
        p = expit(eta)
        D = rng.binomial(1, p).astype(np.int8)
        mu = config.beta0 + config.beta_M * M_chosen - config.beta_d * np.abs(L_chosen - design.locations.reshape(1, -1))
        X = mu + rng.normal(0.0, sigma_x, size=mu.shape)
        X[D == 0] = np.nan
        batch = _finalize_batch(
            D,
            X,
            {"L_A": L_A, "M_A": M_A, "L_B": L_B, "M_B": M_B},
            config.min_detecting_sources,
        )
        take = min(remaining, batch["D"].shape[0])
        for key in collected:
            collected[key].append(batch[key][:take])
        remaining -= take
    return {k: np.concatenate(v, axis=0) for k, v in collected.items()}


def generate_invalid_events_valid_latent_mcar_detection(
    n: int,
    sigma_x: float,
    detection_prob: float,
    config: StudyConfig,
    design: SourceDesign,
    rng: np.random.Generator,
) -> dict:
    """Alternate invalid (Y=0) DGP.

    - Sample (L, M) exactly like valid.
    - Generate would-be observations X like valid, using the observation model.
    - Then sample detections MCAR: each station detects with fixed probability,
      ignoring L/M/location/X.
    """
    if n == 0:
        empty = np.empty((0, config.S), dtype=np.int8)
        return {
            "D": empty,
            "X": np.empty((0, config.S), dtype=float),
            "L_true": np.array([], dtype=float),
            "M_true": np.array([], dtype=float),
        }

    detection_prob = float(detection_prob)
    if not (0.0 <= detection_prob <= 1.0):
        raise ValueError(f"invalid_mcar_detection_prob must be in [0,1], got {detection_prob}")

    collected = {"D": [], "X": [], "L_true": [], "M_true": []}
    remaining = n
    while remaining > 0:
        batch_n = max(remaining * 2, 256)
        L, M = sample_latent_states(batch_n, config, rng)

        # MCAR detection for Y=0.
        D = rng.binomial(1, detection_prob, size=(batch_n, config.S)).astype(np.int8)

        # Observations follow the same generative model as valid, but are only observed when D=1.
        mu = observation_means(L, M, config, design)
        X = mu + rng.normal(0.0, sigma_x, size=mu.shape)
        X[D == 0] = np.nan

        batch = _finalize_batch(D, X, {"L_true": L, "M_true": M}, config.min_detecting_sources)
        take = min(remaining, batch["D"].shape[0])
        for key in collected:
            collected[key].append(batch[key][:take])
        remaining -= take

    return {k: np.concatenate(v, axis=0) for k, v in collected.items()}


def generate_invalid_events_hybrid_mal_mix(
    n: int,
    lambda_value: float,
    sigma_x: float,
    gamma: float,
    p_mal_mix: float,
    mcar_detection_prob: float,
    config: StudyConfig,
    design: SourceDesign,
    rng: np.random.Generator,
) -> dict:
    """Hybrid malformed-event mechanism.

    For each invalid instance i:
      Z_i ~ Bernoulli(p_mal_mix).
      - If Z_i=1: use the composite pseudo-event mechanism (two latent states, per-source selector ~ Bernoulli(gamma)).
      - If Z_i=0: sample (L,M) like valid, generate X like valid, but make detections MCAR with Bernoulli(p_mal) per source.

    As with other mechanisms, instances with < min_detecting_sources detections are rejected and resampled.
    """
    if n == 0:
        empty = np.empty((0, config.S), dtype=np.int8)
        return {"D": empty, "X": np.empty((0, config.S), dtype=float)}

    p_mal_mix = float(p_mal_mix)
    if not (0.0 <= p_mal_mix <= 1.0):
        raise ValueError(f"invalid_p_mal_mix must be in [0,1], got {p_mal_mix}")

    mcar_detection_prob = float(mcar_detection_prob)
    if not (0.0 <= mcar_detection_prob <= 1.0):
        raise ValueError(f"invalid_mcar_detection_prob must be in [0,1], got {mcar_detection_prob}")

    collected = {"D": [], "X": []}
    remaining = n
    while remaining > 0:
        batch_n = max(remaining * 2, 256)

        # Decide which sub-mechanism each invalid instance uses.
        Z = rng.binomial(1, p_mal_mix, size=batch_n).astype(bool)

        D = np.zeros((batch_n, config.S), dtype=np.int8)
        X = np.full((batch_n, config.S), np.nan, dtype=float)

        # Branch 1: composite pseudo-events.
        idx1 = np.where(Z)[0]
        if idx1.size > 0:
            L_A, M_A = sample_latent_states(idx1.size, config, rng)
            L_B, M_B = sample_latent_states(idx1.size, config, rng)
            selectors = rng.binomial(1, gamma, size=(idx1.size, config.S)).astype(bool)
            L_chosen = np.where(selectors, L_A.reshape(-1, 1), L_B.reshape(-1, 1))
            M_chosen = np.where(selectors, M_A.reshape(-1, 1), M_B.reshape(-1, 1))
            eta = (
                config.alpha0_for_lambda(lambda_value)
                + design.alpha0s.reshape(1, -1)
                + lambda_value
                * (config.alpha_M * M_chosen - config.alpha_d * np.abs(L_chosen - design.locations.reshape(1, -1)))
            )
            p = expit(eta)
            D1 = rng.binomial(1, p).astype(np.int8)
            mu = config.beta0 + config.beta_M * M_chosen - config.beta_d * np.abs(L_chosen - design.locations.reshape(1, -1))
            X1 = mu + rng.normal(0.0, sigma_x, size=mu.shape)
            X1[D1 == 0] = np.nan
            D[idx1] = D1
            X[idx1] = X1

        # Branch 0: valid-latent + MCAR detection.
        idx0 = np.where(~Z)[0]
        if idx0.size > 0:
            L, M = sample_latent_states(idx0.size, config, rng)
            D0 = rng.binomial(1, mcar_detection_prob, size=(idx0.size, config.S)).astype(np.int8)
            mu = observation_means(L, M, config, design)
            X0 = mu + rng.normal(0.0, sigma_x, size=mu.shape)
            X0[D0 == 0] = np.nan
            D[idx0] = D0
            X[idx0] = X0

        batch = _finalize_batch(D, X, extra={}, min_detecting_sources=config.min_detecting_sources)
        take = min(remaining, batch["D"].shape[0])
        collected["D"].append(batch["D"][:take])
        collected["X"].append(batch["X"][:take])
        remaining -= take

    return {k: np.concatenate(v, axis=0) for k, v in collected.items()}


def generate_dataset(
    n: int,
    lambda_value: float,
    sigma_x: float,
    config: StudyConfig,
    design: SourceDesign,
    rng: np.random.Generator,
) -> dict:
    n1 = int(round(n * config.prevalence))
    n0 = n - n1
    valid = generate_valid_events(n1, lambda_value, sigma_x, config, design, rng)
    if config.invalid_dgp == "composite":
        invalid = generate_invalid_events(n0, lambda_value, sigma_x, config.gamma, config, design, rng)
    elif config.invalid_dgp == "valid_latent_mcar_detection":
        invalid = generate_invalid_events_valid_latent_mcar_detection(
            n0,
            sigma_x=sigma_x,
            detection_prob=config.invalid_mcar_detection_prob,
            config=config,
            design=design,
            rng=rng,
        )
    elif config.invalid_dgp == "hybrid_mal_mix":
        invalid = generate_invalid_events_hybrid_mal_mix(
            n0,
            lambda_value=lambda_value,
            sigma_x=sigma_x,
            gamma=config.gamma,
            p_mal_mix=config.invalid_p_mal_mix,
            mcar_detection_prob=config.invalid_mcar_detection_prob,
            config=config,
            design=design,
            rng=rng,
        )
    else:
        raise ValueError(
            f"Unknown invalid_dgp='{config.invalid_dgp}'. Expected one of: "
            "'composite', 'valid_latent_mcar_detection', 'hybrid_mal_mix'."
        )
    D = np.concatenate([valid["D"], invalid["D"]], axis=0)
    X = np.concatenate([valid["X"], invalid["X"]], axis=0)
    y = np.concatenate([np.ones(n1, dtype=np.int8), np.zeros(n0, dtype=np.int8)], axis=0)
    cls = np.concatenate([np.repeat("valid", n1), np.repeat("invalid", n0)])
    perm = rng.permutation(n)
    return {
        "X": X[perm],
        "D": D[perm],
        "y": y[perm],
        "class_label": cls[perm],
        "n": n,
        "lambda": lambda_value,
        "sigma_x": sigma_x,
    }
