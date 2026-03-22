"""Plot station detection probability heatmaps and MC detection summary.

This script produces:
1) Heatmaps of the station detection probability p_det(d, M) for a *single* station
   with alpha0s=0, as a function of:
   - d = |L - r_s|  (distance between event location and station location)
   - M             (event magnitude)

   Detection model (as implemented in the study DGP):

       eta = alpha0 + alpha0s + lambda * (alpha_M * M - alpha_d * d)
       p_det = sigmoid(eta)

2) A Monte Carlo estimate of the *average number of detecting stations* per event:
   - sample alpha0s for S=50 stations i.i.d. Uniform[0,1]
   - draw N events with (L, M) from the config priors
   - sample detections D_{i,s} ~ Bernoulli(p_det(L_i, M_i, r_s, alpha0s))
   - compute mean_i sum_s D_{i,s}

We do this for every (lambda, alpha0) scenario you request.

Outputs (written under this file's directory):
- detection_heatmaps/heat__lam_{lambda}__a0_{alpha0}.pdf (and .png)
- detection_summary.csv

Run:
    PYTHONPATH=src python tests/plot_detection_heatmap.py

To change the scenario grid without editing code, pass flags:
    PYTHONPATH=src python tests/plot_detection_heatmap.py \
        --lambdas 0.5 1 2 3 \
        --alpha0s -3 -2 -1.75 -1 0 1 2

"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="paper", help="Config name to load default parameters from.")
    p.add_argument(
        "--alpha-M",
        type=float,
        nargs="+",
        default=[0.16],
        help="Override alpha_M in the detection model (default: take from config).",
    )
    p.add_argument(
        "--alpha-d",
        type=float,
        nargs="+",
        default=[12.0],
        help="Override alpha_d in the detection model (default: take from config).",
    )
    p.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[1.0,2.0],
        help="Lambda values to plot/simulate.",
    )
    p.add_argument(
        "--alpha0s",
        type=float,
        nargs="+",
        default= [-2.23,-2.73], ####[-2.64,-3.6],
        help="alpha0 values to plot/simulate.",
    )
    p.add_argument("--n-events", type=int, default=10_000, help="Number of events for MC detection summary.")
    p.add_argument("--stations", type=int, default=50, help="Number of stations for MC detection summary.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for MC detection summary.")
    p.add_argument("--d-max", type=float, default=1.0, help="Max distance for heatmap (min is 0).")
    p.add_argument("--grid-size", type=int, default=301, help="Resolution for heatmap grid per axis.")
    return p.parse_args()


def _simulate_avg_detections(
    *,
    rng: np.random.Generator,
    n_events: int,
    S: int,
    alpha0: float,
    lambda_value: float,
    alpha_M: float,
    alpha_d: float,
    latent_L_low: float,
    latent_L_high: float,
    latent_M_mean: float,
    latent_M_sd: float,
) -> float:
    """Monte Carlo average of station detections per event."""

    # Station setup
    r_s = np.linspace(0.0, 1.0, S)
    alpha0s = rng.uniform(0.0, 1.0, size=S)

    # Events
    L = rng.uniform(latent_L_low, latent_L_high, size=n_events)
    M = rng.normal(latent_M_mean, latent_M_sd, size=n_events)

    # Broadcast to event x station
    D = np.abs(L[:, None] - r_s[None, :])
    eta = alpha0 + alpha0s[None, :] + lambda_value * (alpha_M * M[:, None] - alpha_d * D)
    p = sigmoid(eta)

    det = rng.binomial(n=1, p=p)
    return float(det.sum(axis=1).mean())


def main() -> None:
    args = parse_args()

    # Local import so this file can run with PYTHONPATH=src
    from sim_score_study.config import get_config

    cfg = get_config(args.config)

    # Grids. If the user passes an empty list (unlikely), fall back to config.
    alpha_M_grid = list(args.alpha_M) if args.alpha_M else [float(cfg.alpha_M)]
    alpha_d_grid = list(args.alpha_d) if args.alpha_d else [float(cfg.alpha_d)]

    out_dir = Path(__file__).resolve().parent
    heat_dir = out_dir / "detection_heatmaps"
    heat_dir.mkdir(parents=True, exist_ok=True)

    # Heatmap grid
    d_grid = np.linspace(0.0, float(args.d_max), int(args.grid_size))
    m_min = float(cfg.M_bounds[0])
    m_max = float(cfg.M_bounds[1])
    m_grid = np.linspace(m_min, m_max, int(args.grid_size))
    D, M = np.meshgrid(d_grid, m_grid)

    # For heatmaps we fix alpha0s=0 per request
    alpha0s_fixed = 0.0

    rng = np.random.default_rng(int(args.seed))

    rows: list[dict[str, object]] = []

    for lambda_value in list(args.lambdas):
        for alpha0 in list(args.alpha0s):
            for alpha_M in alpha_M_grid:
                for alpha_d in alpha_d_grid:
                    eta = (
                        float(alpha0)
                        + alpha0s_fixed
                        + float(lambda_value) * (float(alpha_M) * M - float(alpha_d) * D)
                    )
                    P = sigmoid(eta)

                    avg_det = _simulate_avg_detections(
                        rng=rng,
                        n_events=int(args.n_events),
                        S=int(args.stations),
                        alpha0=float(alpha0),
                        lambda_value=float(lambda_value),
                        alpha_M=float(alpha_M),
                        alpha_d=float(alpha_d),
                        latent_L_low=float(cfg.latent_L_low),
                        latent_L_high=float(cfg.latent_L_high),
                        latent_M_mean=float(cfg.latent_M_mean),
                        latent_M_sd=float(cfg.latent_M_sd),
                    )

                    rows.append(
                        {
                            "lambda": float(lambda_value),
                            "alpha0": float(alpha0),
                            "alpha_M": float(alpha_M),
                            "alpha_d": float(alpha_d),
                            "avg_num_detections": float(avg_det),
                            "n_events": int(args.n_events),
                            "stations": int(args.stations),
                            "seed": int(args.seed),
                        }
                    )

                    # Plot heatmap
                    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
                    im = ax.imshow(
                        P,
                        origin="lower",
                        aspect="auto",
                        extent=(float(d_grid.min()), float(d_grid.max()), float(m_grid.min()), float(m_grid.max())),
                        vmin=0.0,
                        vmax=1.0,
                        cmap="viridis",
                    )
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.set_label("Detection probability")

                    ax.set_xlabel(r"Distance $|L - r_s|$")
                    ax.set_ylabel(r"Magnitude $M$")
                    ax.set_title(
                        "Station detection probability heatmap (alpha0s=0)\n"
                        f"lambda={float(lambda_value):g}, alpha0={float(alpha0):g}, alpha_M={float(alpha_M):g}, alpha_d={float(alpha_d):g}; "
                        f"MC avg detections (S={int(args.stations)}, N={int(args.n_events)}): {avg_det:.2f}"
                    )

                    stem = (
                        f"heat__lam_{float(lambda_value):g}__a0_{float(alpha0):g}"
                        f"__aM_{float(alpha_M):g}__ad_{float(alpha_d):g}"
                    )
                    fig.savefig(heat_dir / f"{stem}.pdf", bbox_inches="tight")
                    fig.savefig(heat_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
                    plt.close(fig)

    summary = pd.DataFrame(rows).sort_values(["lambda", "alpha0", "alpha_M", "alpha_d"]).reset_index(drop=True)
    out_csv = out_dir / "detection_summary.csv"
    summary.to_csv(out_csv, index=False)

    print(f"Wrote heatmaps under: {heat_dir}")
    print(f"Wrote summary table: {out_csv}")


if __name__ == "__main__":
    main()
