from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .reporting import PAPER_METHOD_ORDER, summarize_paired_differences
from .utils import ensure_dir


def _lambda_text(lambda_value: float) -> str:
    """Return a mathtext label for lambda (shared style across all figures)."""
    return rf"$\lambda={lambda_value:g}$"


def _metric_label(metric_name: str) -> str:
    return {
        "AUROC": "AUROC",
        "TNR@TPR95": "TNR at TPR=0.95",
    }.get(metric_name, metric_name)


def _set_log_n_axis(ax: plt.Axes, n_values: np.ndarray) -> None:
    n_values = np.asarray(n_values, dtype=float)
    n_values = n_values[np.isfinite(n_values)]
    if n_values.size == 0:
        return
    n_min = float(np.min(n_values))
    n_max = float(np.max(n_values))
    ax.set_xscale("log")
    ax.set_xlim(n_min * 0.9, n_max * 1.1)
    # Restrict ticks to the actually-used sample sizes (avoids awkward 10^1 ticks when data start at 10^2).
    uniq = sorted({int(x) for x in n_values})
    ax.set_xticks(uniq, labels=[f"{u:,}" for u in uniq])


def plot_score_diagnostics(
    pooled_scores: pd.DataFrame,
    output_path: str | Path,
    representative_sigma: float,
    lambda_low: float,
    lambda_high: float,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    score_cols = ["u_det", "u_nondet", "u_obs", "u_tot"]
    lambdas = [lambda_low, lambda_high]
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 14), constrained_layout=True)

    for col_idx, lambda_value in enumerate(lambdas):
        dat = pooled_scores[pooled_scores["lambda"] == lambda_value]
        # Legacy support: older runs stored sigma_x in pooled_scores.
        if "sigma_x" in dat.columns:
            dat = dat[dat["sigma_x"] == representative_sigma]

        for row_idx, score_col in enumerate(score_cols):
            ax = axes[row_idx, col_idx]
            y1 = dat.loc[dat["y"] == 1, score_col].to_numpy(dtype=float)
            y0 = dat.loc[dat["y"] == 0, score_col].to_numpy(dtype=float)
            parts = ax.violinplot([y1, y0], positions=[1, 2], showmeans=True, showextrema=False)
            for body in parts["bodies"]:
                body.set_alpha(0.5)
            ax.set_xticks([1, 2], labels=["Y=1", "Y=0"])
            if col_idx == 0:
                ax.set_ylabel(score_col)
            if row_idx == 0:
                ax.set_title(_lambda_text(lambda_value))
            ax.grid(alpha=0.2, axis="y")

    fig.suptitle("Class-conditional score distributions in representative scenarios", fontsize=14)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_n(
    summary: pd.DataFrame,
    metric_name: str,
    output_path: str | Path,
    *,
    methods: list[str] | None = None,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    methods = PAPER_METHOD_ORDER if methods is None else methods

    lambda_levels = sorted(summary["lambda"].unique())

    # Legacy support: if sigma_x exists in old results, collapse to a single representative level.
    if "sigma_x" in summary.columns:
        sigma_levels = sorted(summary["sigma_x"].unique())
        rep_sigma = sigma_levels[0] if sigma_levels else None
        if rep_sigma is not None:
            summary = summary[summary["sigma_x"] == rep_sigma]

    fig, axes = plt.subplots(
        nrows=len(lambda_levels),
        ncols=1,
        figsize=(4.8, 3.8 * len(lambda_levels)),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )

    if len(lambda_levels) == 1:
        axes = np.array([axes])

    # Track y-lims across all panels so they share a common scale.
    y_min, y_max = np.inf, -np.inf

    for i, lambda_value in enumerate(lambda_levels):
        ax = axes[i]
        scen = summary[summary["lambda"] == lambda_value]

        all_n = []
        for method in methods:
            sub = scen[scen["method"] == method].sort_values("n_train")
            x = sub["n_train"].to_numpy(dtype=float)
            y = sub[f"{metric_name}_mean"].to_numpy(dtype=float)
            yerr = sub[f"{metric_name}_se"].to_numpy(dtype=float)
            all_n.append(x)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linewidth=1.5,
                capsize=2.5,
                label=method,
            )
            if y.size:
                y_min = min(y_min, float(np.nanmin(y - yerr)))
                y_max = max(y_max, float(np.nanmax(y + yerr)))

        if all_n:
            _set_log_n_axis(ax, np.concatenate(all_n))

        ax.set_title(_lambda_text(lambda_value))
        ax.grid(alpha=0.2)
        ax.set_ylabel(_metric_label(metric_name))
        if i == len(lambda_levels) - 1:
            ax.set_xlabel("Training sample size")

    # Apply common y-lims across panels.
    if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
        pad = 0.03 * (y_max - y_min)
        for ax in axes.ravel():
            ax.set_ylim(y_min - pad, y_max + pad)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_performance_vs_n_2x2(
    summary: pd.DataFrame,
    output_path: str | Path,
    *,
    lambda_low: float,
    lambda_high: float,
    methods: list[str] | None = None,
) -> None:
    """Paper figure: a 2x2 grid (rows=metrics AUROC/TNR@TPR95, cols=lambda low/high).

    Assumes sigma_x is fixed (if multiple are present, uses the first level).
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    methods = PAPER_METHOD_ORDER if methods is None else methods

    # If legacy summary includes sigma_x, take a fixed representative value.
    if "sigma_x" in summary.columns:
        sigma_levels = sorted(summary["sigma_x"].unique())
        rep_sigma = sigma_levels[0] if sigma_levels else None
        if rep_sigma is not None:
            dat = summary[summary["sigma_x"] == rep_sigma]
        else:
            dat = summary
    else:
        dat = summary

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=True, constrained_layout=True)

    panel_specs = [
        (0, 0, "AUROC", lambda_low),
        (0, 1, "AUROC", lambda_high),
        (1, 0, "TNR@TPR95", lambda_low),
        (1, 1, "TNR@TPR95", lambda_high),
    ]

    ylims: dict[str, list[float]] = {"AUROC": [np.inf, -np.inf], "TNR@TPR95": [np.inf, -np.inf]}

    for r, c, metric, lam in panel_specs:
        ax = axes[r, c]
        scen = dat[dat["lambda"] == lam]
        all_n = []
        for method in methods:
            sub = scen[scen["method"] == method].sort_values("n_train")
            x = sub["n_train"].to_numpy(dtype=float)
            y = sub[f"{metric}_mean"].to_numpy(dtype=float)
            yerr = sub[f"{metric}_se"].to_numpy(dtype=float)
            all_n.append(x)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linewidth=1.5,
                capsize=2.5,
                label=method,
            )
            if y.size:
                ylims[metric][0] = min(ylims[metric][0], float(np.nanmin(y - yerr)))
                ylims[metric][1] = max(ylims[metric][1], float(np.nanmax(y + yerr)))

        if all_n:
            _set_log_n_axis(ax, np.concatenate(all_n))
        ax.grid(alpha=0.2)
        ax.set_ylabel(_metric_label(metric))
        ax.set_title(rf"$\lambda={lam:g}$")

    # Enforce the same y-scale within each row (metric) across the two lambda panels.
    for metric, (ymin, ymax) in ylims.items():
        if not (np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax):
            continue
        pad = 0.03 * (ymax - ymin)
        if metric == "AUROC":
            for ax in axes[0, :]:
                ax.set_ylim(ymin - pad, ymax + pad)
        else:
            for ax in axes[1, :]:
                ax.set_ylim(ymin - pad, ymax + pad)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_paired_gains(
    metrics: pd.DataFrame,
    metric_name: str,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    paired = summarize_paired_differences(metrics, metric_name=metric_name)
    comparisons = [
        "LR-Decomp-LR-Obs",
        "RF-Raw+Features-RF-Raw",
    ]
    comparison_titles = {
        "LR-Decomp-LR-Obs": "(LR-Decomp) − (LR-Obs)",
        "RF-Raw+Features-RF-Raw": "(RF-Raw+Features) − (RF-Raw)",
    }

    # Track y-lims across all panels.
    y_min, y_max = np.inf, -np.inf

    fig, axes = plt.subplots(
        nrows=len(comparisons),
        ncols=1,
        figsize=(4.8, 3.8 * len(comparisons)),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )

    if len(comparisons) == 1:
        axes = np.array([axes])

    lambda_levels = sorted(paired["lambda"].unique())

    for i, comparison in enumerate(comparisons):
        ax = axes[i]
        sub = paired[paired["comparison"] == comparison]
        all_n = []
        for lambda_value in lambda_levels:
            ss = sub[sub["lambda"] == lambda_value].sort_values("n_train")
            x = ss["n_train"].to_numpy(dtype=float)
            y = ss[f"{metric_name}_diff_mean"].to_numpy(dtype=float)
            yerr = ss[f"{metric_name}_diff_se"].to_numpy(dtype=float)
            all_n.append(x)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linewidth=1.5,
                capsize=2.5,
                label=_lambda_text(lambda_value),
            )
            if y.size:
                y_min = min(y_min, float(np.nanmin(y - yerr)))
                y_max = max(y_max, float(np.nanmax(y + yerr)))

        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        if all_n:
            _set_log_n_axis(ax, np.concatenate(all_n))
        ax.grid(alpha=0.2)
        nice = comparison_titles.get(comparison, comparison)
        ax.set_ylabel(f"{nice}\npaired Δ {_metric_label(metric_name)}")
        if i == len(comparisons) - 1:
            ax.set_xlabel("Training sample size")

    # Apply common y-lims across panels.
    if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
        pad = 0.05 * (y_max - y_min)
        for ax in axes.ravel():
            ax.set_ylim(y_min - pad, y_max + pad)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(lambda_levels)), bbox_to_anchor=(0.5, 1.08))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_paired_gains_combined(
    metrics: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Combined paired-gains figure with both AUROC and TNR@TPR95.

    Produces a 2x2 grid:
      rows = {AUROC, TNR@TPR95}
      cols = {LR-Decomp - LR-Obs, RF-Raw+Features - RF-Raw}

    Each panel shows paired differences with error bars across n, with separate
    lines for each lambda value.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    comparisons = [
        "LR-Decomp-LR-Obs",
        "RF-Raw+Features-RF-Raw",
    ]
    comparison_titles = {
        "LR-Decomp-LR-Obs": "(LR-Decomp) − (LR-Obs)",
        "RF-Raw+Features-RF-Raw": "(RF-Raw+Features) − (RF-Raw)",
    }
    metrics_to_plot = [
        "AUROC",
        "TNR@TPR95",
    ]

    fig, axes = plt.subplots(
        nrows=len(metrics_to_plot),
        ncols=len(comparisons),
        figsize=(12, 7.5),
        sharex=True,
        constrained_layout=True,
    )
    if len(metrics_to_plot) == 1:
        axes = np.array([axes])

    for r, metric_name in enumerate(metrics_to_plot):
        paired = summarize_paired_differences(metrics, metric_name=metric_name)
        lambda_levels = sorted(paired["lambda"].unique())

        for c, comparison in enumerate(comparisons):
            ax = axes[r, c]
            sub = paired[paired["comparison"] == comparison]

            all_n = []
            y_min, y_max = np.inf, -np.inf
            for lambda_value in lambda_levels:
                ss = sub[sub["lambda"] == lambda_value].sort_values("n_train")
                x = ss["n_train"].to_numpy(dtype=float)
                y = ss[f"{metric_name}_diff_mean"].to_numpy(dtype=float)
                yerr = ss[f"{metric_name}_diff_se"].to_numpy(dtype=float)
                all_n.append(x)
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    marker="o",
                    linewidth=1.5,
                    capsize=2.5,
                    label=_lambda_text(lambda_value),
                )
                if y.size:
                    lo = y - yerr
                    hi = y + yerr
                    if np.any(np.isfinite(lo)):
                        y_min = min(y_min, float(np.nanmin(lo)))
                    if np.any(np.isfinite(hi)):
                        y_max = max(y_max, float(np.nanmax(hi)))

            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
            if all_n:
                _set_log_n_axis(ax, np.concatenate(all_n))
            ax.grid(alpha=0.2)

            if r == 0:
                ax.set_title(comparison_titles.get(comparison, comparison))
            if c == 0:
                ax.set_ylabel(f"paired Δ {_metric_label(metric_name)}")
            if r == len(metrics_to_plot) - 1:
                ax.set_xlabel("Training sample size")

            if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
                pad = 0.05 * (y_max - y_min)
                ax.set_ylim(y_min - pad, y_max + pad)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), bbox_to_anchor=(0.5, 1.05))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_tree_structure(
    tree_clf,
    feature_names: list[str],
    output_path: str | Path,
    max_depth: int | None = None,
    *,
    prune_same_prediction: bool = False,
) -> None:
    """Plot a fitted decision tree classifier for explainability.

    If prune_same_prediction=True, we collapse (visually) any split where both children
    end up predicting the same class (i.e., the argmax class is equal in both
    subtrees). This creates a simpler *view* of the tree without refitting.
    """
    from sklearn import tree

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    if not prune_same_prediction:
        fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
        tree.plot_tree(
            tree_clf,
            feature_names=feature_names,
            class_names=["Y=0", "Y=1"],
            filled=True,
            rounded=True,
            impurity=False,
            max_depth=max_depth,
            ax=ax,
        )
        ax.set_title("Decision tree: fitted structure")
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    # Pruned view: collapse same-prediction splits and render with matplotlib.
    # This is a *view* only (no refit / no sklearn pruning), and it never requires Graphviz.
    import numpy as _np

    sk_tree = tree_clf.tree_

    def _is_leaf(node_id: int) -> bool:
        return sk_tree.children_left[node_id] == sk_tree.children_right[node_id]

    def _pred_class(node_id: int) -> int:
        return int(_np.argmax(sk_tree.value[node_id][0]))

    def _collapse(node_id: int, depth: int) -> int:
        """Return a representative node after collapsing same-prediction splits."""
        if max_depth is not None and depth >= max_depth:
            return node_id
        if _is_leaf(node_id):
            return node_id
        left = int(sk_tree.children_left[node_id])
        right = int(sk_tree.children_right[node_id])
        left_rep = _collapse(left, depth + 1)
        right_rep = _collapse(right, depth + 1)
        if _pred_class(left_rep) == _pred_class(right_rep):
            return left_rep
        return node_id

    # Build collapsed adjacency plus depth (for layout).
    reps: set[int] = set()
    edges: list[tuple[int, int]] = []

    def _walk(node_id: int, depth: int) -> int:
        rep = _collapse(node_id, depth)
        reps.add(rep)
        if max_depth is not None and depth >= max_depth:
            return rep
        if _is_leaf(rep):
            return rep
        left = int(sk_tree.children_left[rep])
        right = int(sk_tree.children_right[rep])
        lrep = _walk(left, depth + 1)
        rrep = _walk(right, depth + 1)
        if lrep != rrep:
            edges.append((rep, lrep))
            edges.append((rep, rrep))
        return rep

    root = _walk(0, 0)

    children: dict[int, list[int]] = {nid: [] for nid in reps}
    for a, b in edges:
        if a in children and b in reps:
            children[a].append(b)

    # Simple tidy layout: x-position by in-order traversal, y-position by depth.
    depths: dict[int, int] = {}

    def _assign_depths(nid: int, d: int) -> None:
        if nid in depths and depths[nid] <= d:
            return
        depths[nid] = d
        for ch in children.get(nid, []):
            _assign_depths(ch, d + 1)

    _assign_depths(root, 0)

    x_pos: dict[int, float] = {}
    _x_counter = 0

    def _leaf_like(nid: int) -> bool:
        return len(children.get(nid, [])) == 0 or _is_leaf(nid)

    def _assign_x(nid: int) -> float:
        nonlocal _x_counter
        chs = children.get(nid, [])
        if len(chs) == 0:
            _x_counter += 1
            x_pos[nid] = float(_x_counter)
            return x_pos[nid]
        xs = [_assign_x(c) for c in chs]
        x_pos[nid] = float(sum(xs) / len(xs))
        return x_pos[nid]

    _assign_x(root)

    max_depth_used = max(depths.values()) if depths else 0

    def _node_label(node_id: int) -> str:
        feat_idx = int(sk_tree.feature[node_id])
        thr = float(sk_tree.threshold[node_id])
        counts = sk_tree.value[node_id][0]
        pred = int(_np.argmax(counts))
        n = int(counts.sum())
        if _is_leaf(node_id) or feat_idx < 0 or _leaf_like(node_id):
            return f"pred=Y{pred}\nsamples={n}\nvalue={counts.astype(int).tolist()}"
        fname = feature_names[feat_idx] if 0 <= feat_idx < len(feature_names) else f"f{feat_idx}"
        return f"{fname} ≤ {thr:.3g}\npred=Y{pred}\nsamples={n}\nvalue={counts.astype(int).tolist()}"

    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    ax.axis("off")

    # Normalize coordinates to [0, 1].
    x_vals = list(x_pos.values()) or [1.0]
    min_x, max_x = min(x_vals), max(x_vals)
    span_x = max(1e-9, max_x - min_x)

    coords: dict[int, tuple[float, float]] = {}
    for nid in reps:
        x = (x_pos.get(nid, 1.0) - min_x) / span_x
        y = 1.0 - (depths.get(nid, 0) / max(1, max_depth_used + 1))
        coords[nid] = (x, y)

    for parent, childs in children.items():
        for ch in childs:
            x1, y1 = coords[parent]
            x2, y2 = coords[ch]
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.0, alpha=0.7)

    for nid in sorted(reps):
        x, y = coords[nid]
        ax.text(
            x,
            y,
            _node_label(nid),
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#eeeeee", ec="#444444", lw=0.8),
        )

    ax.set_title("Decision tree: pruned view (merged same-prediction splits)")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return


def plot_misspecification_comparison_n10000(
    metrics: pd.DataFrame,
    output_path: str | Path,
    *,
    n_train: int = 10_000,
    misspec_suffix: str = "__misspec",
    methods: list[str] | None = None,
) -> None:
    """Plot correct-vs-misspecified expert-model performance at a fixed training size.

    This figure is *optional* and will only work if the simulation run included
    misspecified expert outputs (methods with suffix `misspec_suffix`).

    The plot uses only rows with n_train==`n_train` and aggregates over all
    replicates and lambda scenarios.

    Produces a 1x2 panel figure:
      - left: AUROC
      - right: TNR@TPR95

    X-axis: {"Correct", "Misspecified"}
    Y-axis: metric value

    For each method, we plot the Monte Carlo mean in each condition and connect
    the two points with a line.
    """

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    methods = ["LR-Decomp", "LR-Obs", "RF-Raw+Features", "RF-Raw"] if methods is None else methods

    # Work on a minimal copy to avoid any surprises from duplicate labels / alignment.
    cols = ["replicate", "n_train", "lambda", "method", "AUROC", "TNR@TPR95"]
    dat = metrics.loc[:, cols].copy().reset_index(drop=True)

    dat = dat[dat["n_train"] == int(n_train)].copy().reset_index(drop=True)
    if dat.empty:
        raise ValueError(f"No metric rows found for n_train={n_train}")

    # Average within replicate across lambda scenarios so each replicate contributes equally.
    rep_avg = (
        dat.groupby(["replicate", "method"], as_index=False)[["AUROC", "TNR@TPR95"]]
        .mean()
        .reset_index(drop=True)
    )

    miss = rep_avg[rep_avg["method"].astype(str).str.endswith(misspec_suffix)].copy().reset_index(drop=True)
    base = rep_avg[~rep_avg["method"].astype(str).str.endswith(misspec_suffix)].copy().reset_index(drop=True)

    miss["method_base"] = miss["method"].astype(str).str.replace(f"{misspec_suffix}$", "", regex=True)

    miss = miss[miss["method_base"].isin(methods)].copy().reset_index(drop=True)
    base = base[base["method"].isin(methods)].copy().reset_index(drop=True)

    if miss.empty:
        raise ValueError(
            f"No misspecified method rows found for n_train={n_train} (suffix='{misspec_suffix}'). "
            "Did you run simulations with --expert-misspecification?"
        )

    # Prepare minimal RHS and rename baseline metrics to avoid duplicate column labels.
    base2 = base[["replicate", "method", "AUROC", "TNR@TPR95"]].copy()
    base2 = base2.rename(columns={"AUROC": "AUROC_base", "TNR@TPR95": "TNR@TPR95_base"})

    miss2 = miss[["replicate", "method_base", "AUROC", "TNR@TPR95"]].copy()
    miss2 = miss2.rename(columns={"AUROC": "AUROC_miss", "TNR@TPR95": "TNR@TPR95_miss"})

    merged = miss2.merge(
        base2,
        left_on=["replicate", "method_base"],
        right_on=["replicate", "method"],
        how="inner",
        validate="many_to_one",
    ).reset_index(drop=True)

    # Aggregate across replicates -> plot mean only.
    rows = []
    for m in methods:
        sub = merged[merged["method_base"] == m]
        if sub.empty:
            continue
        rows.append(
            {
                "method": m,
                "AUROC_correct": float(sub["AUROC_base"].mean()),
                "AUROC_miss": float(sub["AUROC_miss"].mean()),
                "TNR_correct": float(sub["TNR@TPR95_base"].mean()),
                "TNR_miss": float(sub["TNR@TPR95_miss"].mean()),
            }
        )
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9.5, 4.2), constrained_layout=True)

    specs = [
        (axes[0], "AUROC", "AUROC_correct", "AUROC_miss"),
        (axes[1], "TNR at TPR=0.95", "TNR_correct", "TNR_miss"),
    ]

    x = np.array([0, 1], dtype=float)
    xticks = ["Correct", "Misspecified"]

    for ax, title, col_c, col_m in specs:
        y_min, y_max = np.inf, -np.inf
        for i, row in df.iterrows():
            y = np.array([row[col_c], row[col_m]], dtype=float)
            ax.plot(x, y, marker="o", linewidth=2.0, label=row["method"])
            if np.all(np.isfinite(y)):
                y_min = min(y_min, float(np.min(y)))
                y_max = max(y_max, float(np.max(y)))

        ax.set_xticks(x, labels=xticks)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")

        if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
            pad = 0.05 * (y_max - y_min)
            ax.set_ylim(y_min - pad, y_max + pad)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
    #fig.suptitle(f"Correct vs misspecified expert model (n={n_train:,})", fontsize=12)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

