from __future__ import annotations

import numpy as np

from sim_score_study.config import get_config
from sim_score_study.dgp import build_source_design


def test_source_design_resampled_by_replicate_and_deterministic() -> None:
    cfg = get_config("smoke")

    d1a = build_source_design(cfg, replicate=1)
    d1b = build_source_design(cfg, replicate=1)
    d2 = build_source_design(cfg, replicate=2)

    # Deterministic within replicate.
    assert d1a.seed == d1b.seed
    assert np.array_equal(d1a.locations, d1b.locations)
    assert np.array_equal(d1a.alpha0s, d1b.alpha0s)

    # Different across replicates.
    assert d1a.seed != d2.seed
    assert not np.array_equal(d1a.locations, d2.locations)
    assert not np.array_equal(d1a.alpha0s, d2.alpha0s)

