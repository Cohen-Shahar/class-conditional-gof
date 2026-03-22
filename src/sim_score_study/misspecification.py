from __future__ import annotations

import numpy as np

from .config import StudyConfig


_EXPERT_PARAM_FIELDS = [
    # Detection model (per-lambda calibration is in alpha0_levels)
    "alpha0_levels",
    "alpha_M",
    "alpha_d",
    # Observation model
    "beta0",
    "beta_M",
    "beta_d",
    # Observation noise
    "sigma_x_levels",
]


def sample_expert_misspecification_factors(
    *, rng: np.random.Generator, pct: float, size: int
) -> np.ndarray:
    """Sample multiplicative factors in {(1-pct), (1+pct)}."""
    pct = float(pct)
    if pct < 0:
        raise ValueError(f"expert_misspecification_pct must be >= 0, got {pct}")
    # Bernoulli determines +pct vs -pct.
    signs = rng.integers(0, 2, size=size)
    return np.where(signs == 1, 1.0 + pct, 1.0 - pct).astype(float)


def build_misspecified_expert_config(
    true_config: StudyConfig,
    *,
    replicate: int,
    pct: float | None = None,
) -> tuple[StudyConfig, dict[str, float]]:
    """Return a config to be used as the *expert model* under misspecification.

    Data are still generated with `true_config`. This function constructs a copy
    of the config whose expert parameters are independently perturbed by ±pct.

    The perturbation is replicate-specific and deterministic.

    Returns
    -------
    expert_config : StudyConfig
        Copy of config with perturbed expert parameters.
    factor_manifest : dict
        Field->factor mapping, stored for reproducibility.
    """
    pct = float(true_config.expert_misspecification_pct if pct is None else pct)

    # Deterministic seed per replicate; keep independent from data RNG streams.
    seed = int(true_config.random_seed) + 10_000 * int(replicate) + 9_901
    rng = np.random.default_rng(seed)

    factors = sample_expert_misspecification_factors(rng=rng, pct=pct, size=len(_EXPERT_PARAM_FIELDS))
    factor_manifest = {field: float(factors[i]) for i, field in enumerate(_EXPERT_PARAM_FIELDS)}

    expert_payload = true_config.to_dict()
    expert_payload["expert_misspecification"] = True
    expert_payload["expert_misspecification_pct"] = pct

    # Apply factors.
    for field, factor in factor_manifest.items():
        val = expert_payload[field]
        if isinstance(val, list):
            expert_payload[field] = [float(x) * factor for x in val]
        else:
            expert_payload[field] = float(val) * factor

    expert_config = StudyConfig(**expert_payload)

    # Keep study design (grid sizes etc.) identical to true config.
    # (Fields above already copied from true config.)
    return expert_config, factor_manifest
