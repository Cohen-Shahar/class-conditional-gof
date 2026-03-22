from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def monte_carlo_mean_se(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), np.nan
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def format_mean_se(mean: float, se: float, digits: int = 3) -> str:
    if np.isnan(mean):
        return "NA"
    if np.isnan(se):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {se:.{digits}f}"


def scenario_key(lambda_value: float, sigma_x: float) -> str:
    return f"lambda={lambda_value:g}|sigma_x={sigma_x:g}"


def sort_metric_frame(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["n_train", "lambda", "sigma_x", "method", "replicate"] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)
