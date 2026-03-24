#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run simulations, then build all tables and figures. "
            "Misspecification and score-diagnostics outputs are inferred from the selected config."
        )
    )
    parser.add_argument("--config", default="paper")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Also store fitted model objects in each cell pickle during simulation (off by default; increases disk usage).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    this_dir = Path(__file__).resolve().parent

    cmd1 = [
        sys.executable,
        str(this_dir / "run_simulations.py"),
        "--config",
        args.config,
        "--output-root",
        args.output_root,
    ]
    if args.overwrite:
        cmd1.append("--overwrite")
    if args.n_jobs is not None:
        cmd1.extend(["--n-jobs", str(args.n_jobs)])
    if args.save_models:
        cmd1.append("--save-models")
    subprocess.check_call(cmd1)

    cmd2 = [sys.executable, str(this_dir / "build_tables.py"), "--results-root", args.output_root, "--table", "all"]
    subprocess.check_call(cmd2)

    cmd3 = [
        sys.executable,
        str(this_dir / "build_figures.py"),
        "--results-root",
        args.output_root,
        "--figure",
        "all",
    ]
    subprocess.check_call(cmd3)


if __name__ == "__main__":
    main()