#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from pathlib import Path

from sim_score_study.config import get_config
from sim_score_study.dgp import build_source_design
from sim_score_study.experiment import run_single_cell, write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the simulation study cells.")
    parser.add_argument("--config", default="paper", help="Named config from sim_score_study.config")
    parser.add_argument("--output-root", required=True, help="Directory for all results")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cell files")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel jobs for latent-state fitting")
    parser.add_argument(
        "--save-models",
        action="store_true",
        help=(
            "Also store fitted sklearn model objects in each cell .pkl for optional diagnostics. "
            "Off by default because it can make pickles much larger and more brittle across sklearn versions."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config(args.config)

    output_root = Path(args.output_root)

    if len(config.sigma_x_levels) != 1:
        raise ValueError(
            "This study protocol expects sigma_x to be fixed (one value in sigma_x_levels). "
            f"Got sigma_x_levels={config.sigma_x_levels}."
        )

    # Write run-level manifest (config only). Source design is now replicate-specific.
    write_run_manifest(config, output_root)

    for replicate in range(1, config.replicates + 1):
        design = build_source_design(config, replicate=replicate)
        # Persist the replicate-specific design for reproducibility.
        write_run_manifest(config, output_root, design=design, replicate=replicate)

        sigma_x = float(config.sigma_x_levels[0])
        for n_train in config.training_sizes:
            for lambda_value in config.lambda_levels:
                print(
                    f"Running replicate={replicate} n_train={n_train} lambda={lambda_value:g}",
                    flush=True,
                )
                run_single_cell(
                    replicate=replicate,
                    n_train=n_train,
                    lambda_value=lambda_value,
                    sigma_x=sigma_x,
                    config=config,
                    design=design,
                    output_root=output_root,
                    overwrite=args.overwrite,
                    n_jobs=args.n_jobs,
                    save_pooled_scores=bool(getattr(config, "run_with_pooled_scores", False)),
                    save_models=args.save_models,
                )


if __name__ == "__main__":
    main()