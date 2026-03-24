#!/usr/bin/env python
from __future__ import annotations

"""Simple one-off diagnostics for the sim_score_study simulation.

What this script does
---------------------
- Samples ONE dataset with n_valid=10_000 and n_invalid=10_000.
- Uses scenario parameters: lambda=2, sigma_x=1.
- Prints class-conditional diagnostics about:
  * detections vs non-detections per instance
  * score feature summaries (u_det/u_nondet/u_obs/u_tot)
  * auxiliary feature summaries (m_detect/M_hat/residual stats)
- Writes a couple of quick figures under ``./tmp_diagnostics/``.

This is meant as a lightweight sanity check / explainability aid.
"""

from dataclasses import dataclass
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sim_score_study.config import get_config
from sim_score_study.dgp import build_source_design, generate_dataset
from sim_score_study.features import compute_feature_bundle
from sim_score_study.fitting import fit_latent_states
from sim_score_study.utils import ensure_dir


@dataclass(frozen=True)
class Scenario:
    lambda_value: float = 2.0
    sigma_x: float = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-off diagnostics for a single (lambda, sigma_x) scenario")
    p.add_argument("--config", default="paper", help="Named config from sim_score_study.config")
    p.add_argument("--lambda", dest="lambda_value", type=float, default=2.0)
    p.add_argument("--sigma", dest="sigma_x", type=float, default=1.0)
    p.add_argument("--n-per-class", type=int, default=10_000, help="Sample size per class (valid and invalid)")
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def _summarize_by_class(values: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    out = []
    for cls in [0, 1]:
        v = values[y == cls]
        out.append(
            {
                "class": cls,
                "mean": float(np.mean(v)),
                "sd": float(np.std(v, ddof=1)),
                "p05": float(np.quantile(v, 0.05)),
                "p50": float(np.quantile(v, 0.50)),
                "p95": float(np.quantile(v, 0.95)),
            }
        )
    return pd.DataFrame(out)


def _print_feature_summary(df: pd.DataFrame, y: np.ndarray, cols: list[str], title: str) -> None:
    print(f"\n== {title} ==")
    for col in cols:
        summ = _summarize_by_class(df[col].to_numpy(float), y)
        print(f"\n[{col}]")
        print(summ.to_string(index=False))


def _save_hist_two_classes(
    x0: np.ndarray,
    x1: np.ndarray,
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 60,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.hist(x0, bins=bins, alpha=0.55, density=True, label="Y=0 (invalid)")
    ax.hist(x1, bins=bins, alpha=0.55, density=True, label="Y=1 (valid)")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    cfg = get_config(args.config)
    cfg.prevalence = 0.5

    scen = Scenario(lambda_value=float(args.lambda_value), sigma_x=float(args.sigma_x))
    n_total = int(2 * args.n_per_class)

    print(
        f"Scenario: lambda={scen.lambda_value:g} sigma_x={scen.sigma_x:g} "
        f"n_per_class={args.n_per_class:,}",
        flush=True,
    )

    # Sample a fresh source design (locations + detection shifts) for this run.
    # We map the CLI seed to a deterministic "replicate" index.
    design = build_source_design(cfg, replicate=int(args.seed))

    rng = np.random.default_rng(int(args.seed))

    print("Sampling dataset...", flush=True)
    dataset = generate_dataset(
        n=n_total,
        lambda_value=scen.lambda_value,
        sigma_x=scen.sigma_x,
        config=cfg,
        design=design,
        rng=rng,
    )

    y = dataset["y"].astype(int)
    n_valid = int(np.sum(y == 1))
    n_invalid = int(np.sum(y == 0))
    print(f"Sampled n_total={n_total:,} with n_valid={n_valid:,} and n_invalid={n_invalid:,}")

    X = dataset["X"]
    D = dataset["D"].astype(int)

    m_detect = D.sum(axis=1)

    print("\n== Detection diagnostics (per-instance counts) ==")
    print("m_detect:")
    print(_summarize_by_class(m_detect.astype(float), y).to_string(index=False))

    print("\n== Extra diagnostics you’ll usually care about ==")
    # fraction with very few detections (can stress the fitter)
    for k in [0, 1, 2, 5, 10]:
        frac0 = float(np.mean((m_detect[y == 0] <= k)))
        frac1 = float(np.mean((m_detect[y == 1] <= k)))
        print(f"P(m_detect <= {k})  invalid(Y=0)={frac0:.4f}  valid(Y=1)={frac1:.4f}")

    print("\nFitting latent states (this can take a bit for n=20k)...", flush=True)
    fitted = fit_latent_states(X, D, scen.lambda_value, scen.sigma_x, cfg, design, n_jobs=None)
    print(f"Latent fit converged rate: {float(np.mean(fitted.converged)):.3f}")

    bundle = compute_feature_bundle(dataset, fitted, scen.lambda_value, scen.sigma_x, cfg, design)

    score_cols = ["u_det", "u_nondet", "u_obs", "u_tot"]
    aux_cols = ["m_detect", "M_hat", "resid_mean", "resid_sd"]

    # Add counts to feature df for convenience
    features = bundle.features.copy()
    features["m_detect"] = m_detect

    _print_feature_summary(features, y, score_cols, title="Score features (class-conditional summaries)")
    _print_feature_summary(features, y, aux_cols, title="Aux features (class-conditional summaries)")

    # Quick plots
    out_dir = ensure_dir(Path("tmp_diagnostics"))

    _save_hist_two_classes(
        x0=features.loc[y == 0, "u_tot"].to_numpy(float),
        x1=features.loc[y == 1, "u_tot"].to_numpy(float),
        title="Total score feature (u_tot)",
        xlabel="u_tot",
        out_path=out_dir / "hist_u_tot.pdf",
        bins=60,
    )

    # Scatter: detectiveness vs M_hat (often reveals separation / weirdness)
    fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
    ax.scatter(
        features.loc[y == 0, "m_detect"].to_numpy(float),
        features.loc[y == 0, "M_hat"].to_numpy(float),
        s=6,
        alpha=0.25,
        label="Y=0 (invalid)",
    )
    ax.scatter(
        features.loc[y == 1, "m_detect"].to_numpy(float),
        features.loc[y == 1, "M_hat"].to_numpy(float),
        s=6,
        alpha=0.25,
        label="Y=1 (valid)",
    )
    ax.set_title("m_detect vs M_hat")
    ax.set_xlabel("m_detect")
    ax.set_ylabel("M_hat")
    ax.grid(alpha=0.2)
    ax.legend(markerscale=2)
    fig.savefig(out_dir / "scatter_m_detect_vs_M_hat.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
