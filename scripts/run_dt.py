#!/usr/bin/env python
from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from sim_score_study.config import get_config
from sim_score_study.dgp import build_source_design, generate_dataset
from sim_score_study.features import AUX_COLUMNS
from sim_score_study.features import compute_feature_bundle
from sim_score_study.plotting import plot_tree_structure
from sim_score_study.fitting import fit_latent_states
from sim_score_study.metrics import compute_metrics
from sim_score_study.utils import ensure_dir, save_json


# ---------------------------
# In-file configuration block
# ---------------------------
# Edit this dict if you prefer changing the DT grid in the file instead of the CLI.
# Any key set to None keeps the existing CLI default behavior.
DT_DEFAULTS: dict[str, object] = {
    # I/O
    "base_config": "paper",
    "output_root": None,  # e.g., str(ROOT / "results" / "dt")

    # Replication
    "replicates": 1,
    "seed_offset": 0,

    # DGP grid (comma-separated strings or Python lists)
    "invalid_dgp": "composite",
    "gamma": 0.5,  # e.g., "0.3,0.5" or [0.3, 0.5]
    "invalid_mcar_detection_prob": 0.1,
    "invalid_p_mal_mix": 0.5,

    # Scenario grid
    "lambda_values": [2.0],
    "sigma_x_values": 1.0,
    "n_train_values": "100000",
    "test_size": 5000,

    # DT hyperparameter grid
    "max_depth": [3],  # use "None" for unlimited depth
    "min_samples_leaf": "50",
    "criterion": "gini",

    # Outputs
    "save_tree": True,
    "save_metrics_per_replicate": True,
}


def _maybe_override(args: argparse.Namespace, defaults: dict[str, object]) -> argparse.Namespace:
    """Override parsed CLI args with in-file defaults when provided.

    Rule: if a key exists in defaults and is not None, it replaces the attribute on args.
    """
    for k, v in defaults.items():
        if v is None:
            continue
        if hasattr(args, k):
            setattr(args, k, v)
    return args


@dataclass(frozen=True)
class DTSpec:
    invalid_dgp: str
    invalid_mcar_detection_prob: float
    invalid_p_mal_mix: float
    gamma: float
    lambda_value: float
    sigma_x: float
    n_train: int
    test_size: int
    random_seed: int
    max_depth: int | None
    min_samples_leaf: int
    criterion: str


def _parse_list(arg: str | list | float | int | None, cast):
    """Parse a grid argument.

    Accepts:
      - None (meaning: caller decides defaults)
      - list (already a grid)
      - scalar (float/int/str): treated as a single value unless it contains commas
      - comma-separated string
    """
    if arg is None:
        return None
    if isinstance(arg, list):
        return [cast(x) for x in arg]

    # Scalars: treat as singletons unless they're strings containing commas.
    if isinstance(arg, (float, int)):
        return [cast(arg)]

    s = str(arg).strip()
    if s == "":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [cast(p) for p in parts]


def _as_depth(x: str | int | None) -> int | None:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if s.lower() == "none":
        return None
    return int(s)


def _cell_id(spec: DTSpec, rep: int) -> str:
    return (
        f"rep_{rep:03d}__inv_{spec.invalid_dgp}__gam_{spec.gamma:g}__pmix_{spec.invalid_p_mal_mix:g}__"
        f"pmcar_{spec.invalid_mcar_detection_prob:g}__n_{spec.n_train}__lam_{spec.lambda_value:g}__sig_{spec.sigma_x:g}__"
        f"depth_{'None' if spec.max_depth is None else spec.max_depth}__leaf_{spec.min_samples_leaf}__crit_{spec.criterion}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run decision-tree grids (DT on decomposed engineered features) and write figures/metrics under results/dt. "
            "This is intentionally separate from the main simulation protocol.\n\n"
            "You can configure the grid either via CLI flags or by editing DT_DEFAULTS in this file."
        )
    )
    # Accept both --base-config (legacy) and --config (alias used by other scripts).
    p.add_argument("--base-config", "--config", dest="base_config", default="paper")
    p.add_argument("--output-root", dest="output_root", default=str(ROOT / "results" / "dt"))

    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--seed-offset", dest="seed_offset", type=int, default=0)

    # DGP grid
    p.add_argument("--invalid-dgp", dest="invalid_dgp", default="composite")
    p.add_argument("--gamma", dest="gamma", default=None)
    p.add_argument("--invalid-mcar-detection-prob", dest="invalid_mcar_detection_prob", default=None)
    p.add_argument("--invalid-p-mal-mix", dest="invalid_p_mal_mix", default=None)

    # Scenario grid
    p.add_argument("--lambda", dest="lambda_values", default=None)
    p.add_argument("--sigma-x", dest="sigma_x_values", default=None)
    p.add_argument("--n-train", dest="n_train_values", default="1000,10000")
    p.add_argument("--test-size", dest="test_size", type=int, default=1000)

    # DT hyperparameter grid
    p.add_argument("--max-depth", dest="max_depth", default="2,3,4,5")
    p.add_argument("--min-samples-leaf", dest="min_samples_leaf", default="50")
    p.add_argument("--criterion", dest="criterion", default="gini")

    p.add_argument("--save-tree", dest="save_tree", action="store_true")
    p.add_argument(
        "--save-metrics-per-replicate",
        dest="save_metrics_per_replicate",
        action="store_true",
        help="Write tables/dt_metrics_per_replicate.csv (enabled by default via DT_DEFAULTS).",
    )
    p.add_argument(
        "--no-save-metrics-per-replicate",
        dest="save_metrics_per_replicate",
        action="store_false",
        help="Disable writing per-replicate metrics CSV.",
    )
    p.set_defaults(save_metrics_per_replicate=False)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args = _maybe_override(args, DT_DEFAULTS)

    # If output_root is left as None in DT_DEFAULTS, keep the CLI default.
    if args.output_root is None:
        args.output_root = str(ROOT / "results" / "dt")

    base = get_config(args.base_config)
    base.fit_n_jobs = 1

    invalid_dgps = _parse_list(args.invalid_dgp, str) or ["composite"]
    gammas = _parse_list(args.gamma, float)
    pmcars = _parse_list(getattr(args, "invalid_mcar_detection_prob", None), float)
    pmixes = _parse_list(getattr(args, "invalid_p_mal_mix", None), float)

    lambda_values = _parse_list(args.lambda_values, float) or list(base.lambda_levels)
    sigma_x_values = _parse_list(args.sigma_x_values, float) or list(base.sigma_x_levels)
    n_train_values = _parse_list(args.n_train_values, int) or [1000, 10000]

    max_depth_values = _parse_list(args.max_depth, _as_depth) or [2, 3, 4, 5]
    min_samples_leaf_values = _parse_list(args.min_samples_leaf, int) or [50]
    criteria = _parse_list(args.criterion, str) or ["gini"]

    # Defaults for DGP params when not relevant.
    if gammas is None:
        gammas = [float(base.gamma)]
    if pmcars is None:
        pmcars = [float(base.invalid_mcar_detection_prob)]
    if pmixes is None:
        pmixes = [float(base.invalid_p_mal_mix)]

    out_root = ensure_dir(Path(args.output_root))
    figs_dir = ensure_dir(out_root / "figures")
    tables_dir = ensure_dir(out_root / "tables")

    # Save a manifest for reproducibility.
    manifest = {
        "base_config": args.base_config,
        "replicates": int(args.replicates),
        "seed_offset": int(args.seed_offset),
        "grid": {
            "invalid_dgp": invalid_dgps,
            "gamma": gammas,
            "invalid_mcar_detection_prob": pmcars,
            "invalid_p_mal_mix": pmixes,
            "lambda": lambda_values,
            "sigma_x": sigma_x_values,
            "n_train": n_train_values,
            "test_size": int(args.test_size),
            "dt": {
                "max_depth": max_depth_values,
                "min_samples_leaf": min_samples_leaf_values,
                "criterion": criteria,
            },
        },
    }
    save_json(out_root / "manifest.json", manifest)

    # NOTE: Source design is now replicate-specific; build it inside the rep loop.

    feature_names = ["u_det", "u_nondet", "u_obs", *AUX_COLUMNS]

    per_rep_rows: list[dict[str, object]] = []

    grid_iter = product(
        invalid_dgps,
        gammas,
        pmcars,
        pmixes,
        lambda_values,
        sigma_x_values,
        n_train_values,
        max_depth_values,
        min_samples_leaf_values,
        criteria,
    )

    for (
        invalid_dgp,
        gamma,
        invalid_mcar_detection_prob,
        invalid_p_mal_mix,
        lambda_value,
        sigma_x,
        n_train,
        max_depth,
        min_samples_leaf,
        criterion,
    ) in grid_iter:
        # Configure the DGP for this cell.
        cfg = get_config(args.base_config)
        cfg.fit_n_jobs = 1
        cfg.test_size = int(args.test_size)
        cfg.invalid_dgp = invalid_dgp
        cfg.gamma = float(gamma)
        cfg.invalid_mcar_detection_prob = float(invalid_mcar_detection_prob)
        cfg.invalid_p_mal_mix = float(invalid_p_mal_mix)

        spec = DTSpec(
            invalid_dgp=cfg.invalid_dgp,
            invalid_mcar_detection_prob=float(cfg.invalid_mcar_detection_prob),
            invalid_p_mal_mix=float(cfg.invalid_p_mal_mix),
            gamma=float(cfg.gamma),
            lambda_value=float(lambda_value),
            sigma_x=float(sigma_x),
            n_train=int(n_train),
            test_size=int(cfg.test_size),
            random_seed=int(cfg.random_seed) + int(args.seed_offset),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            criterion=str(criterion),
        )

        for rep in range(1, int(args.replicates) + 1):
            # Re-sample source design for this replicate.
            design = build_source_design(base, replicate=rep, seed_offset=int(args.seed_offset))

            base_seed = spec.random_seed + 10_000 * rep + 97 * spec.n_train
            rng_train = np.random.default_rng(base_seed + 1 + int(100 * spec.lambda_value) + int(1000 * spec.sigma_x))
            rng_test = np.random.default_rng(base_seed + 2 + int(100 * spec.lambda_value) + int(1000 * spec.sigma_x))

            train = generate_dataset(spec.n_train, spec.lambda_value, spec.sigma_x, cfg, design, rng_train)
            test = generate_dataset(spec.test_size, spec.lambda_value, spec.sigma_x, cfg, design, rng_test)

            train_fit = fit_latent_states(train["X"], train["D"], spec.lambda_value, spec.sigma_x, cfg, design, n_jobs=1)
            test_fit = fit_latent_states(test["X"], test["D"], spec.lambda_value, spec.sigma_x, cfg, design, n_jobs=1)

            train_bundle = compute_feature_bundle(train, train_fit, spec.lambda_value, spec.sigma_x, cfg, design)
            test_bundle = compute_feature_bundle(test, test_fit, spec.lambda_value, spec.sigma_x, cfg, design)

            X_train = train_bundle.features[feature_names]
            y_train = train_bundle.labels
            X_test = test_bundle.features[feature_names]
            y_test = test_bundle.labels

            clf = DecisionTreeClassifier(
                max_depth=spec.max_depth,
                min_samples_leaf=spec.min_samples_leaf,
                criterion=spec.criterion,
                random_state=base_seed + 123,
            )
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            mdict = compute_metrics(y_test, probs)

            row = {
                "replicate": rep,
                "invalid_dgp": spec.invalid_dgp,
                "gamma": spec.gamma,
                "invalid_mcar_detection_prob": spec.invalid_mcar_detection_prob,
                "invalid_p_mal_mix": spec.invalid_p_mal_mix,
                "n_train": spec.n_train,
                "test_size": spec.test_size,
                "lambda": spec.lambda_value,
                "sigma_x": spec.sigma_x,
                "max_depth": spec.max_depth,
                "min_samples_leaf": spec.min_samples_leaf,
                "criterion": spec.criterion,
                "train_fit_converged_rate": float(train_fit.converged.mean()),
                "test_fit_converged_rate": float(test_fit.converged.mean()),
                **{k: float(v) for k, v in mdict.items()},
            }
            per_rep_rows.append(row)

            if args.save_tree:
                out_path = figs_dir / f"tree__{_cell_id(spec, rep)}.pdf"
                plot_tree_structure(
                    clf,
                    feature_names=feature_names,
                    output_path=out_path,
                    max_depth=spec.max_depth,
                )

    per_rep_df = pd.DataFrame(per_rep_rows)

    if args.save_metrics_per_replicate:
        per_rep_df.to_csv(tables_dir / "dt_metrics_per_replicate.csv", index=False)

    # Summary: mean and standard error by grid cell.
    group_cols = [
        "invalid_dgp",
        "gamma",
        "invalid_mcar_detection_prob",
        "invalid_p_mal_mix",
        "n_train",
        "test_size",
        "lambda",
        "sigma_x",
        "max_depth",
        "min_samples_leaf",
        "criterion",
    ]

    metric_cols = ["AUROC", "AUPRC", "Brier", "LogLoss", "TNR@TPR95", "train_fit_converged_rate", "test_fit_converged_rate"]

    rows = []
    for keys, grp in per_rep_df.groupby(group_cols, sort=True):
        out = dict(zip(group_cols, keys))
        for m in metric_cols:
            vals = grp[m].to_numpy(dtype=float)
            out[f"{m}_mean"] = float(np.mean(vals))
            out[f"{m}_se"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
        rows.append(out)

    summary_df = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    summary_df.to_csv(tables_dir / "dt_metrics_summary.csv", index=False)

    # Convenience: print the best AUROC row.
    if not summary_df.empty:
        best = summary_df.sort_values("AUROC_mean", ascending=False).iloc[0]
        print("Best DT spec by AUROC_mean:")
        print(best.to_string())


if __name__ == "__main__":
    main()
