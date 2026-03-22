from __future__ import annotations

import argparse

import numpy as np

from sim_score_study.config import get_config
from sim_score_study.dgp import build_source_design
from sim_score_study.dgp import generate_dataset
from sim_score_study.fitting import fit_latent_states


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run many simulated datasets and report latent-state optimizer failure rate.")
    p.add_argument("--config", type=str, default="smoke", help="Config name from sim_score_study.config.list_configs().")
    p.add_argument("--n_samples", type=int, default=1000, help="How many independent fits to run.")
    p.add_argument(
        "--n_events",
        type=int,
        default=1,
        help="Number of events per dataset. Use 1 to interpret n_samples as number of event-fits.",
    )
    p.add_argument("--lambda_value", type=float, default=None, help="Override lambda (default: first level in config).")
    p.add_argument("--sigma_x", type=float, default=None, help="Override sigma_x (default: first level in config).")
    p.add_argument("--seed", type=int, default=0, help="Extra seed offset for reproducibility.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config(args.config)

    lambda_value = float(args.lambda_value) if args.lambda_value is not None else float(config.lambda_levels[0])
    sigma_x = float(args.sigma_x) if args.sigma_x is not None else float(config.sigma_x_levels[0])

    # Make fitting deterministic for this report.
    config.fit_n_jobs = 1

    rng = np.random.default_rng(config.random_seed + args.seed)

    n_total = int(args.n_samples) * int(args.n_events)
    n_fail = 0

    # We generate one dataset per "sample" and fit latent state(s) for the events inside.
    for i in range(int(args.n_samples)):
        # Treat each dataset as its own replicate for source-design re-sampling.
        design = build_source_design(config, replicate=i + 1, seed_offset=int(args.seed))

        data = generate_dataset(int(args.n_events), lambda_value, sigma_x, config, design, rng)
        fit = fit_latent_states(data["X"], data["D"], lambda_value, sigma_x, config, design, n_jobs=1)
        n_fail += int((~fit.converged).sum())

    fail_rate = n_fail / max(n_total, 1)
    print(
        "Convergence report\n"
        f"  config       : {args.config}\n"
        f"  lambda        : {lambda_value}\n"
        f"  sigma_x       : {sigma_x}\n"
        f"  datasets      : {int(args.n_samples)}\n"
        f"  events/dataset: {int(args.n_events)}\n"
        f"  total fits    : {n_total}\n"
        f"  failures      : {n_fail}\n"
        f"  failure rate  : {fail_rate:.4f}\n"
    )


if __name__ == "__main__":
    main()

