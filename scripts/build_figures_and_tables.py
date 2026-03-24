#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
import subprocess

from sim_score_study.config import StudyConfig, get_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build all simulation figures and tables from an existing results directory. "
            "Inclusion of score diagnostics and misspecification outputs is inferred from results-root/config.json."
        )
    )
    parser.add_argument("--results-root", required=True)
    return parser.parse_args()


def _load_config(results_root: Path) -> StudyConfig:
    cfg_path = results_root / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find config.json under {results_root}")

    with cfg_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    base = get_config(payload.get("name", "paper"))
    merged = base.to_dict()
    merged.update(payload)
    return StudyConfig(**merged)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    config = _load_config(results_root)

    include_score_diagnostics = bool(getattr(config, "run_with_pooled_scores", False))
    include_misspec = bool(getattr(config, "expert_misspecification", False))

    print(
        f"[build_figures_and_tables] score_diagnostics={'enabled' if include_score_diagnostics else 'disabled'} "
        f"(run_with_pooled_scores={include_score_diagnostics})"
    )
    print(
        f"[build_figures_and_tables] misspecification={'enabled' if include_misspec else 'disabled'} "
        f"(expert_misspecification={include_misspec})"
    )

    this_dir = Path(__file__).resolve().parent

    cmd_tables = [
        sys.executable,
        str(this_dir / "build_tables.py"),
        "--results-root",
        str(results_root),
        "--table",
        "all",
    ]
    subprocess.check_call(cmd_tables)

    cmd_figures = [
        sys.executable,
        str(this_dir / "build_figures.py"),
        "--results-root",
        str(results_root),
        "--figure",
        "all",
    ]
    subprocess.check_call(cmd_figures)


if __name__ == "__main__":
    main()

