# Expert-Guided Class-Conditional Goodness-of-Fit Scores for Interpretable Classification with Informative Missingness: An Application to Seismic Monitoring

This repository contains the simulation-study code used in the paper above.

## Citation

If you use this codebase, please cite:

```text
[Paper citation placeholder]
Author(s), "Expert-Guided Class-Conditional Goodness-of-Fit Scores for Interpretable Classification with Informative Missingness: An Application to Seismic Monitoring", [Venue], [Year].
```

You can also use `CITATION.cff` (update with final publication metadata as needed).

## How to use

### 1) Clone

```bash
git clone https://github.com/Cohen-Shahar/class-conditional-gof.git
cd class-conditional-gof
```

### 2) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Alternative non-editable install:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

### 3) Run the full pipeline

Run simulation + table/figure post-processing in one command:

```bash
python scripts/run_pipeline.py --config paper --output-root results/paper
```
This runs the full pipeline for the main paper results, including misspecification robustness, and saves outputs under `results/paper/`.

If you want to test a quick run with fewer replicates, use the `smoke` config:
```bash
python scripts/run_pipeline.py --config smoke --output-root results/smoke
```

if you want a run similar to the main paper but light on replicates, use the `paper_light` config:
```bash
python scripts/run_pipeline.py --config paper_light --output-root results/paper_light
```

# Additional results:
## Pooled scores and score diagnostics
This implementation uses config-driven behavior (from `src/sim_score_study/config.py`, that is saved `results/.../config.json`):
- `run_with_pooled_scores` controls whether pooled per-example scores are saved.
By default, the main pipeline does not save pooled per-example scores, which are needed for score diagnostics. This is to save memory and disk space.
If you also want score diagnostics in the same run, use a pooled-score config:

```bash
python scripts/run_pipeline.py --config paper_pooled_scores --output-root results/paper_pooled_scores
```

## Misspecification

This implementation uses config-driven behavior (from `src/sim_score_study/config.py`, that is saved `results/.../config.json`):
- `expert_misspecification` controls whether misspecified-expert robustness outputs are generated and saved.
misspecification is enabled by default.

Important memory/disk warning:

- pooled scores add per-example payloads to each cell file and can substantially increase memory pressure and disk usage.


## Generate a single decision tree (Interpretability results)

`DT-Decomp` is intentionally separate from the main pipeline. To generate a single tree setup:

```bash
python scripts/run_dt.py \
  --config paper \
  --output-root results/dt_single \
  --replicates 1 \
  --lambda 2.0 \
  --n-train 100000 \
  --max-depth 3 \
  --min-samples-leaf 50 \
  --criterion gini \
  --save-tree
```

This writes a manifest, metrics, and tree figures under `results/dt_single/`.

## Generate tables and figures separately
You can run simulations only (without post-processing) with:

```bash
python scripts/run_simulations.py --config paper --output-root results/paper
```

Then generate tables and figures from that existing results directory:

```bash
python scripts/build_tables.py --results-root results/paper --table all
python scripts/build_figures.py --results-root results/paper --figure all
```

Notes:

- with `--table all`, misspecification robustness tables are included automatically when `expert_misspecification=true` in `results-root/config.json`.
- with `--figure all`, standard figures are always generated, and misspecification/score-diagnostics figures are included automatically based on `results-root/config.json`.

You can also generate single tables or figures by specifying the desired subset of tables or figures (see `scripts/build_tables.py --help` and `scripts/build_figures.py --help` for details).

## Additional Information

### Repository layout

```text
sim_score_study/
├── pyproject.toml
├── requirements.txt
├── README.md
├── scripts/
│   ├── run_simulations.py
│   ├── build_tables.py
│   ├── build_figures.py
│   ├── build_figures_and_tables.py
│   ├── run_pipeline.py
│   └── run_dt.py
└── src/sim_score_study/
    ├── __init__.py
    ├── config.py
    ├── dgp.py
    ├── fitting.py
    ├── features.py
    ├── metrics.py
    ├── models.py
    ├── experiment.py
    ├── reporting.py
    └── plotting.py
```

### What is implemented from the protocol

#### Data-generating process
- Binary labels `Y in {0,1}` with `Y=1` = valid and `Y=0` = invalid.
- `S=50` sources with equally spaced fixed source locations on `[0,1]`.
- Source-specific offsets `alpha0s ~ U[0,1]` are drawn once at the start of each replicate and held fixed within that replicate.
- Valid events are generated from a single latent state `(L, M)`.
- Invalid events are generated using a **hybrid malformed-event mechanism** controlled by `invalid_p_mal_mix`:

  - With probability `invalid_p_mal_mix` ("composite" malformed events):
    evidence is mixed across sources from **two** pseudo-events `(L^A, M^A)` and `(L^B, M^B)`.
    For each source `s`, a selector `C^(s) ~ Bernoulli(gamma)` chooses which pseudo-event generates
    that source's detection indicator and, when detected, its observed value.

  - With probability `1 - invalid_p_mal_mix` ("irregular missingness" malformed events):
    a single latent state `(L, M)` is sampled as for valid events, but detections are generated
    **MCAR**, independently across sources, with probability `invalid_mcar_detection_prob`.
    When a source detects, its observed value is still generated from the observation model.

  This mixed malformed-event mechanism is the default simulation used in the paper.

#### Expert-guided fitting
- The modeled class is the valid class (`Y=1`).
- During fitting, all class-1 structural quantities are treated as known except the per-instance latent state `(L_i, M_i)`.
- The code maximizes the class-1 observed-data log-likelihood for each instance using bounded L-BFGS-B with multiple starts.

#### Scenario grid
The main paper protocol varies:
- the missingness informativeness parameter `lambda` (two levels: low/high), and
- the training sample size.

When changing `lambda`, `alpha0` is calibrated per-scenario so that the average number of detecting sources remains comparable across scenarios.
This is configured via elementwise pairing:

- `lambda_levels = [lambda_low, lambda_high]`
- `alpha0_levels = [alpha0_low, alpha0_high]`

#### Score features
The code computes all requested features:
- `u_det`
- `u_nondet`
- `u_obs`
- `u_tot`

and the auxiliary feature vector:
- `m_detect`
- `M_hat`
- `resid_mean`
- `resid_sd`

#### Methods compared
The code computes metrics for the following methods:
- `LR-Decomp`
- `LR-Total` (**computed and stored**, but **not used** in manuscript tables and figures)
- `LR-Obs`
- `LR-Baseline`
- `RF-Raw`
- `RF-Raw+Features`

#### DT-Decomp (implemented separately)
DT-Decomp is implemented separately from the main pipeline.
Use `scripts/run_dt.py` to fit DT-Decomp on a grid of settings and export representative tree figures under `results/dt/`.

#### Metrics
The code computes:
- AUROC
- AUPRC
- Brier score
- Log loss
- TNR at TPR = 0.95

#### Paired Monte Carlo design
Within each replicate and scenario, every method is evaluated on the exact same simulated training and test data.

### Reproducibility

- Per-replicate source designs are saved under `source_designs/source_design__rep_XXX.json`.
- Train/test datasets differ by seeded generators derived from the replicate, training size, and scenario.
- Random forests are seeded per cell.
- The simulation is exactly reproducible given the config and package versions.

### License and authorship

This project is released under the MIT License (see `LICENSE`).
