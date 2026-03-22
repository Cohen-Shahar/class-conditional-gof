"""Diagnostics: detection patterns vs magnitude (M) only.

This script is a companion to tests/plot_detection_heatmap.py.

It simulates N events, samples detections for S stations, and then summarizes:
1) The expected (average) number of detecting stations as a function of M.
2) The proportion of events with fewer than 2 detections as a function of M.

We keep the same detection model used throughout the project:

    eta_{i,s} = alpha0 + alpha0s + lambda * (alpha_M * M_i - alpha_d * |L_i - r_s|)
    D_{i,s} ~ Bernoulli(sigmoid(eta_{i,s}))

Defaults (matching the *current* plot_detection_heatmap.py intent):
- lambdas default = [1.0, 2.0]
- alpha0s default = [-2.23, -2.73]
- alpha_d default = 12.0
- alpha_M default = 0.16

Outputs are written under this file's directory:
- detection_vs_magnitude/plot__lam_{lambda}__a0_{alpha0}__aM_{alpha_M}__ad_{alpha_d}.pdf/.png
- detection_vs_magnitude/summary.csv

Run:
    PYTHONPATH=src python tests/plot_detection_vs_magnitude.py

Override scenario parameters:
    PYTHONPATH=src python tests/plot_detection_vs_magnitude.py --lambdas 1 2 --alpha0s -2.23 --alpha-d 12 --alpha-M 0.16
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
    p.add_argument("--config", default="paper", help="Config name (for priors on L and M and M_bounds).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-events", type=int, default=10_000)
    p.add_argument("--stations", type=int, default=50)

    # Scenario grid (defaults matching the current heatmap file)
    p.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--alpha0s", type=float, nargs="+", default=[-2.23, -2.73])
    p.add_argument("--alpha-M", type=float, nargs="+", default=[0.16])
    p.add_argument("--alpha-d", type=float, nargs="+", default=[12.0])

    # Binning for M
    p.add_argument("--m-bins", type=int, default=25, help="Number of magnitude bins.")
    return p.parse_args()


def _simulate(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays (M, num_detect, lt2_detect) for n_events."""

    # Station setup
    r_s = np.linspace(0.0, 1.0, S)
    alpha0s = rng.uniform(0.0, 1.0, size=S)

    # Events: draw (L, M)
    L = rng.uniform(latent_L_low, latent_L_high, size=n_events)
    M = rng.normal(latent_M_mean, latent_M_sd, size=n_events)

    D = np.abs(L[:, None] - r_s[None, :])
    eta = alpha0 + alpha0s[None, :] + lambda_value * (alpha_M * M[:, None] - alpha_d * D)
    p = sigmoid(eta)
    det = rng.binomial(n=1, p=p)

    num_detect = det.sum(axis=1)
    lt2 = (num_detect < 2).astype(int)
    return M, num_detect.astype(float), lt2.astype(float)


def _binned_curve(x: np.ndarray, y: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(x, qs)
    # Ensure strictly increasing edges
    edges = np.unique(edges)
    if edges.size < 3:
        # Degenerate fallback
        return np.array([float(np.mean(x))]), np.array([float(np.mean(y))])

    idx = np.digitize(x, edges[1:-1], right=True)
    centers = []
    means = []
    for b in range(edges.size - 1):
        mask = idx == b
        if not np.any(mask):
            continue
        centers.append(float(np.mean(x[mask])))
        means.append(float(np.mean(y[mask])))
    return np.asarray(centers), np.asarray(means)


def main() -> None:
    args = parse_args()

    from sim_score_study.config import get_config

    cfg = get_config(args.config)

    out_dir = Path(__file__).resolve().parent / "detection_vs_magnitude"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    # One RNG stream for reproducibility across scenarios
    base_rng = np.random.default_rng(int(args.seed))

    # Use independent substreams per scenario so ordering does not affect results.
    scenario_idx = 0

    for lambda_value in list(args.lambdas):
        for alpha0 in list(args.alpha0s):
            for alpha_M in list(args.alpha_M):
                for alpha_d in list(args.alpha_d):
                    rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1, dtype=np.uint32))
                    scenario_idx += 1

                    M, num_detect, lt2 = _simulate(
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

                    # Bin by M
                    m_centers_1, avg_detect = _binned_curve(M, num_detect, bins=int(args.m_bins))
                    m_centers_2, prop_lt2 = _binned_curve(M, lt2, bins=int(args.m_bins))

                    # Record summary stats (overall)
                    rows.append(
                        {
                            "lambda": float(lambda_value),
                            "alpha0": float(alpha0),
                            "alpha_M": float(alpha_M),
                            "alpha_d": float(alpha_d),
                            "n_events": int(args.n_events),
                            "stations": int(args.stations),
                            "seed": int(args.seed),
                            "avg_num_detect_overall": float(np.mean(num_detect)),
                            "prop_lt2_overall": float(np.mean(lt2)),
                        }
                    )

                    # Plot
                    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8.5, 7.0), sharex=True, constrained_layout=True)

                    axes[0].plot(m_centers_1, avg_detect, marker="o", linewidth=1.5)
                    axes[0].set_ylabel("Avg # detecting stations")
                    axes[0].grid(alpha=0.2)

                    axes[1].plot(m_centers_2, prop_lt2, marker="o", linewidth=1.5, color="tab:red")
                    axes[1].set_ylabel("P(#detect < 2)")
                    axes[1].set_xlabel("Magnitude M")
                    axes[1].set_ylim(0.0, 1.0)
                    axes[1].grid(alpha=0.2)

                    fig.suptitle(
                        "Detection vs Magnitude (M)\n"
                        f"lambda={float(lambda_value):g}, alpha0={float(alpha0):g}, alpha_M={float(alpha_M):g}, alpha_d={float(alpha_d):g}"
                    )

                    stem = (
                        f"plot__lam_{float(lambda_value):g}__a0_{float(alpha0):g}"
                        f"__aM_{float(alpha_M):g}__ad_{float(alpha_d):g}"
                    )
                    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
                    fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
                    plt.close(fig)

    summary = pd.DataFrame(rows).sort_values(["lambda", "alpha0", "alpha_M", "alpha_d"]).reset_index(drop=True)
    summary.to_csv(out_dir / "summary.csv", index=False)

    print(f"Wrote plots under: {out_dir}")
    print(f"Wrote summary table: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

