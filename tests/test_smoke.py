from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim_score_study.config import get_config, list_configs
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
    assert cfg.replicates == 3
    assert cfg.run_with_pooled_scores is False


def test_pooled_score_configs_exist_and_match_requested_replicates():
    names = set(list_configs())
    assert "paper_pooled_scores" in names
    assert "smoke_pooled_scores" in names

    paper_cfg = get_config("paper_pooled_scores")
    smoke_cfg = get_config("smoke_pooled_scores")

    assert paper_cfg.replicates == 20
    assert smoke_cfg.replicates == 1
    assert paper_cfg.run_with_pooled_scores is True
    assert smoke_cfg.run_with_pooled_scores is True


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
