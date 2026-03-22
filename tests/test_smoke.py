from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim_score_study.config import get_config
from sim_score_study.features import feature_columns_for_method

EXPECTED_METHODS = {
    "LR-Decomp",
    "LR-Total",
    "LR-Obs",
    "LR-Baseline",
    "RF-Raw",
    "RF-Raw+Features",
}


def test_smoke_config_exists():
    cfg = get_config("smoke")
    assert cfg.S == 50
    assert len(cfg.training_sizes) == 1


def test_print_model_feature_sets(capsys):
    """Print the feature list for each model (for quick inspection in CI/logs).

    Run with: pytest -s -k test_print_model_feature_sets
    """
    cfg = get_config("smoke")
    fmap = feature_columns_for_method(cfg)

    # DT-Decomp is no longer part of the main pipeline.
    assert set(fmap) == EXPECTED_METHODS

    for method in sorted(fmap):
        print(f"{method}: {fmap[method]}")

    # Ensure something was printed (guards against accidental no-op).
    captured = capsys.readouterr()
    assert "LR-Decomp" in captured.out
