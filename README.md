# Simulation study codebase: score-based representations under source-level informative missingness

This repository implements the simulation protocol described in the paper "Expert-Guided Class-Conditional Goodness-of-Fit Scores for Interpretable Classification with Informative Missingness: An Application to Seismic Monitoring", including:

- the binary valid-vs-invalid data-generating process,
- source-level informative missingness,
- per-instance latent-state fitting under the modeled class,
- decomposed score extraction,
- transparent logistic-regression baselines,
- random-forest raw-data benchmarks,
- all figures and tables,
- Monte Carlo aggregation, paired comparisons, and coefficient-stability summaries.


## Repository layout

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

## What is implemented from the protocol

### Data-generating process
- Binary labels `Y in {0,1}` with `Y=1` = valid and `Y=0` = invalid.
- `S=50` sources with equally spaced fixed source locations on `[0,1]`.
- Source-specific offsets `alpha0s ~ U[0,1]` are drawn once at the start of each replicate and held fixed within that replicate.
- Valid events are generated from a single latent state `(L, M)`.
- Invalid events are generated using a **hybrid malformed-event mechanism** controlled by `invalid_p_mal_mix`:

  - With probability `invalid_p_mal_mix` ("composite" malformed events):
    evidence is mixed across sources from **two** pseudo-events `(L^A, M^A)` and `(L^B, M^B)`.
    For each source `s`, a selector `C^(s) ~ Bernoulli(gamma)` chooses which pseudo-event generates
    that source’s detection indicator and, when detected, its observed value.

  - With probability `1 - invalid_p_mal_mix` ("irregular missingness" malformed events):
    a single latent state `(L, M)` is sampled as for valid events, but detections are generated
    **MCAR**, independently across sources, with probability `invalid_mcar_detection_prob`.
    When a source detects, its observed value is still generated from the observation model.

  This mixed malformed-event mechanism is the default simulation used in the paper.

### Expert-guided fitting
- The modeled class is the valid class (`Y=1`).
- During fitting, all class-1 structural quantities are treated as known except the per-instance latent state `(L_i, M_i)`.
- The code maximizes the class-1 observed-data log-likelihood for each instance using bounded L-BFGS-B with multiple starts.

### Scenario grid
The main paper protocol varies:
- the missingness informativeness parameter `lambda` (two levels: low/high), and
- the training sample size.

When changing `lambda`, `alpha0` is calibrated per-scenario so that the average number of detecting sources remains comparable across scenarios.
This is configured via elementwise pairing:

- `lambda_levels = [lambda_low, lambda_high]`
- `alpha0_levels = [alpha0_low, alpha0_high]`

### Score features
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

### Methods compared
The code computes metrics for the following methods:
- `LR-Decomp`
- `LR-Total` (**computed and stored**, but **not used** in manuscript tables and figures)
- `LR-Obs`
- `LR-Baseline`
- `RF-Raw`
- `RF-Raw+Features`

### DT-Decomp (implemented separately)
DT-Decomp is implemented separately from the main pipeline.
Use `scripts/run_dt.py` to fit DT-Decomp on a grid of settings and export representative tree figures under `results/dt/`.

### Metrics
The code computes:
- AUROC
- AUPRC
- Brier score
- Log loss
- TNR at TPR = 0.95

### Paired Monte Carlo design
Within each replicate and scenario, every method is evaluated on the exact same simulated training and test data.

## Default parameter values

All simulation, model, and grid parameters are defined in `src/sim_score_study/config.py`.

### Key configuration fields (guide)

The `StudyConfig` fields correspond directly to the protocol.
Common fields you may want to change:

**Core study design**
- `random_seed`: global seed used to derive all replicate/cell RNGs.
- `S`: number of sources.
- `prevalence`: fraction of valid events (`Y=1`) in each dataset.
- `training_sizes`: list of training-set sizes.
- `test_size`: test-set size.
- `replicates`: number of Monte Carlo replicates.
- `min_detecting_sources`: rejection threshold (instances with fewer detections are resampled).

**Latent-state prior**
- `latent_L_low`, `latent_L_high`: bounds of `L ~ U[latent_Low, latent_L_high]`.
- `latent_M_mean`, `latent_M_sd`: parameters of `M ~ N(latent_M_mean, latent_M_sd^2)`.

**Detection model parameters** (Eq. in paper: detection probability)
- `alpha0_levels`: per-scenario calibrated intercepts, paired elementwise with `lambda_levels`.
- `alpha_M`: coefficient multiplying magnitude `M` in the detection linear predictor.
- `alpha_d`: coefficient multiplying distance `|L-r_s|` in the detection linear predictor.
- `lambda_levels`: informativeness levels (two levels in the main paper study).

**Observation model parameters** (Eq. in paper: observed values)
- `beta0`: observation intercept.
- `beta_M`: coefficient multiplying magnitude `M`.
- `beta_d`: coefficient multiplying distance `|L-r_s|`.
- `sigma_x_levels`: observation noise standard deviation(s) (a single value in the paper protocol).

**Malformed/invalid (Y=0) mechanism**
- `invalid_dgp`: invalid-class mechanism (default is `"composite"`; alternative supported value is `"valid_latent_mcar_detection"`).
- `invalid_p_mal_mix`: mixing probability between composite malformed vs irregular-missingness malformed.
- `gamma`: per-source selector probability for the composite malformed mechanism.
- `invalid_mcar_detection_prob`: MCAR detection probability used in the irregular-missingness malformed mechanism.

**Model-fitting knobs**
- `M_bounds`: bounds used by the optimizer for latent `M`.
- `fit_n_jobs`: parallelism used inside single-cell fitting.
- `optimizer_maxiter`, `optimizer_multistart_L`, `optimizer_reuse_empirical_init`: per-instance latent-state fitting controls.
- `run_with_pooled_scores`: whether to persist pooled per-example score payloads in cell files (default: `false`).
- `expert_misspecification`: whether to also run the misspecified-expert pipeline (default: `true`).
- `expert_misspecification_pct`: misspecification size `p` for multiplicative perturbations `(1±p)`.

**Predictive models**
- `rf_*`: random-forest hyperparameters.
- `lr_*`: logistic regression hyperparameters.

If you need to change any protocol values, edit `config.py` and rerun the pipeline.

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/Cohen-Shahar/class-conditional-gof.git
cd class-conditional-gof
```

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

If you do not want an editable install:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

## Available configs

The package ships with the following named configs:

- `smoke`: small end-to-end run (3 replicates) for quick pipeline checks
- `smoke_pooled_scores`: smoke-like run with pooled scores enabled (`replicates=1`)
- `paper_light`: reduced-cost development run
- `paper`: protocol-faithful run
- `paper_pooled_scores`: paper-like run with pooled scores enabled (`replicates=20`)

List them from Python:

```bash
python - <<'PY'
from sim_score_study import list_configs
print(list_configs())
PY
```

## Run simulations

Assume the output root is `results/paper`.

By default, the simulation saves per-cell metric/coefficient summaries and uses misspecification settings from config (`expert_misspecification`, `expert_misspecification_pct`).

```bash
python scripts/run_simulations.py --config paper --output-root results/paper
```

If you want score-diagnostics-ready outputs, use a pooled-score config:

```bash
python scripts/run_simulations.py --config paper_pooled_scores --output-root results/paper_pooled_scores
python scripts/run_simulations.py --config smoke_pooled_scores --output-root results/smoke_pooled_scores
```

Optional model-object export remains available:

```bash
python scripts/run_simulations.py --config paper --output-root results/paper --save-models
```

## Build all figures and tables from a results directory

After simulations finish, generate all tables and figures with one command:

```bash
python scripts/build_figures_and_tables.py --results-root "Directory_name"
```

Example:

```bash
python scripts/build_figures_and_tables.py --results-root results/paper
python scripts/build_figures_and_tables.py --results-root results/paper_pooled_scores
```

The script reads `Directory_name/config.json` and prints whether:

- misspecification outputs are enabled (from `expert_misspecification`), and
- score-diagnostics output is enabled (from `run_with_pooled_scores`).

Behavior:

- It always builds the main tables/figures.
- It includes misspecification tables/figures only when `expert_misspecification=true`.
- It includes `fig_sim_score_diagnostics.pdf` only when `run_with_pooled_scores=true`.

Warning on memory/disk usage:

- pooled scores store per-example payloads inside each cell pickle and can substantially increase disk usage and memory pressure.

### DT-Decomp (run separately)
Run and export DT-Decomp candidates:

```bash
python scripts/run_dt.py --config paper --output-root results/dt
```

The DT script saves candidate trees and a manifest under `results/dt/`.

### One-command full pipeline
If you want simulation + post-processing in one call:

```bash
python scripts/run_pipeline.py --config paper --output-root results/paper
```

To include score diagnostics in one call, use a pooled-score config and request diagnostics:

```bash
python scripts/run_pipeline.py --config paper_pooled_scores --output-root results/paper_pooled_scores --with-score-diagnostics
```

## Output conventions

After a full run, the output tree is:

```text
results/paper/
├── config.json
├── source_designs/
│   ├── source_design__rep_001.json
│   ├── source_design__rep_002.json
│   └── ...
├── cells/
├── tables/
└── figures/
```

The `cells/` directory contains one serialized result file per `(replicate, n_train, lambda)` cell.

## Reproducibility

- Per-replicate source designs are saved under `source_designs/source_design__rep_XXX.json`.
- Train/test datasets differ by seeded generators derived from the replicate, training size, and scenario.
- Random forests are seeded per cell.
- The simulation is exactly reproducible given the config and package versions.

## Minimal validation run

To verify the code path end to end and inspect the actual figures quickly:

```bash
python scripts/run_pipeline.py --config smoke --output-root results/smoke --n-jobs 1
python scripts/run_pipeline.py --config smoke_pooled_scores --output-root results/smoke_pooled_scores --n-jobs 1 --with-score-diagnostics
```

Then inspect:
- `results/smoke/tables/`
- `results/smoke/figures/`
- `results/smoke_pooled_scores/figures/fig_sim_score_diagnostics.pdf`

## Notes on manuscript integration

The scripts produce both `.csv` and `.tex` table outputs. The `.tex` files can be included directly in a manuscript. Figures are written as PDF files suitable for LaTeX inclusion with `\includegraphics`.

## License and authorship

This project is released under the MIT License (see `LICENSE`).

If you use this repository in academic work, please cite it using `CITATION.cff`.

### Model misspecification robustness

This repository can evaluate robustness to **expert-model misspecification**.
In this experiment:

- Training/test data are generated from the *true* model (the selected `--config`).
- **Within each Monte Carlo replicate**, an *expert* model is created by independently multiplying each
  expert parameter
  \((\alpha_0,\alpha_M,\alpha_d,\beta_0,\beta_M,\beta_d,\sigma_x)\)
  by either \((1-p)\) or \((1+p)\) with equal probability.
- The simulation then runs **two expert pipelines side-by-side on the exact same generated datasets**:
  1) baseline (well-specified expert model), and
  2) misspecified expert model.
- Latent-state fitting \((\u005chat L_i, \u005chat M_i)\) and engineered features are computed under each expert model.
- Predictive models are trained/evaluated as usual. Results from the misspecified expert model are stored
  with a `__misspec` suffix on the method name (e.g., `LR-Decomp__misspec`).

This mode is **enabled by default** and controlled in `src/sim_score_study/config.py`.

#### Configure the misspecification-robustness experiment

Set these fields in your selected config payload:

- `expert_misspecification: true`
- `expert_misspecification_pct: 0.25` for random ±25% multiplicative perturbations (current default).

Then run simulations normally:

```bash
python scripts/run_simulations.py \
  --config paper \
  --output-root results/paper_with_misspec_p25
```

Notes:
- The replicate-specific perturbation factors are stored in each cell pickle under `payload["metadata"]["expert_misspecification_factors"]`.

#### Post-processing misspecification outputs

Use the unified post-processing command for the run directory:

```bash
python scripts/build_figures_and_tables.py --results-root results/paper_with_misspec_p25
```

When `expert_misspecification=true`, this command includes the misspecification robustness tables and the misspecification comparison figure automatically.
