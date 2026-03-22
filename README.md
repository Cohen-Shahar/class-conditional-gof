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
- `latent_L_low`, `latent_L_high`: bounds of `L ~ U[latent_L_low, latent_L_high]`.
- `latent_M_mean`, `latent_M_sd`: parameters of `M ~ N(latent_M_mean, latent_M_sd^2)`.

**Detection model parameters** (Eq. in paper: detection probability)
- `alpha0`: detection intercept (used unless you provide `alpha0_levels`).
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
- `invalid_dgp`: invalid-class mechanism (paper default is `"hybrid_mal_mix"`).
- `invalid_p_mal_mix`: mixing probability between composite malformed vs irregular-missingness malformed.
- `gamma`: per-source selector probability for the composite malformed mechanism.
- `invalid_mcar_detection_prob`: MCAR detection probability used in the irregular-missingness malformed mechanism.

**Model-fitting knobs**
- `M_bounds`: bounds used by the optimizer for latent `M`.
- `fit_n_jobs`: parallelism used inside single-cell fitting.
- `optimizer_maxiter`, `optimizer_multistart_L`, `optimizer_reuse_empirical_init`: per-instance latent-state fitting controls.

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

- `smoke`: tiny test run for debugging the full pipeline
- `smoke_plots`: small run (3 replicates) that still generates the full set of paper-style plots/tables
- `paper_light`: reduced-cost development run
- `paper`: protocol-faithful run

List them from Python:

```bash
python - <<'PY'
from sim_score_study import list_configs
print(list_configs())
PY
```

## Exact commands to generate every requested figure and table by name

Assume the output root is `results/paper`.

### Step 1: run the simulation cells

By default, the simulation saves only the per-cell metric and coefficient summaries. This keeps the
`cells/*.pkl` files small.

```bash
python scripts/run_simulations.py --config paper --output-root results/paper
```

If you also want to generate the score-diagnostics figure (`fig_sim_score_diagnostics.pdf`), re-run
(or run) the simulations with pooled scores enabled:

```bash
python scripts/run_simulations.py --config paper --output-root results/paper --pooled-scores
```

Note: pooled scores add a large per-example payload to each cell file and can significantly increase
disk usage.

Optional: You can also store the fitted sklearn model objects (logistic-regression / random-forest)
inside each cell pickle for ad-hoc diagnostics. This is **off by default** because it can make the
`.pkl` files much larger (and can be brittle across sklearn version changes).

```bash
python scripts/run_simulations.py --config paper --output-root results/paper --save-models
```

This creates the per-cell Monte Carlo outputs under:

```text
results/paper/cells/
```

### Step 2: generate tables

#### Table `tab:sim-settings`
Outputs:
- `results/paper/tables/tab_sim_settings.csv`
- `results/paper/tables/tab_sim_settings.tex`

Command:
```bash
python scripts/build_tables.py --results-root results/paper --table sim-settings
```

#### Table `tab:sim-main-discrimination`
Outputs:
- `results/paper/tables/tab_sim_main_discrimination.csv`
- `results/paper/tables/tab_sim_main_discrimination.tex`

Command:
```bash
python scripts/build_tables.py --results-root results/paper --table sim-main-discrimination
```

#### Table `tab:sim-main-probability`
Outputs:
- `results/paper/tables/tab_sim_main_probability.csv`
- `results/paper/tables/tab_sim_main_probability.tex`

Command:
```bash
python scripts/build_tables.py --results-root results/paper --table sim-main-probability
```

#### Table `tab:sim-coef-stability`
Outputs:
- `results/paper/tables/tab_sim_coef_stability.csv`
- `results/paper/tables/tab_sim_coef_stability.tex`
- `results/paper/tables/tab_sim_coef_stability_full_tidy.csv`

Command:
```bash
python scripts/build_tables.py --results-root results/paper --table sim-coef-stability
```

### Step 3: generate figures

#### Figure `fig_sim_score_diagnostics.pdf`
Output:
- `results/paper/figures/fig_sim_score_diagnostics.pdf`

Command:
```bash
python scripts/build_figures.py --results-root results/paper --figure sim-score-diagnostics
```

Implementation details:
- pools test-set extracted score values across replicates,
- uses the smallest and largest configured `lambda`,
- produces a 4 x 2 panel figure with rows for `u_det`, `u_nondet`, `u_obs`, `u_tot`.

#### Figure `fig_sim_performance_vs_n.pdf`
Output:
- `results/paper/figures/fig_sim_performance_vs_n.pdf`

Command:
```bash
python scripts/build_figures.py --results-root results/paper --figure sim-performance-vs-n
```

This is the paper-style 2×2 grid: rows = (AUROC, TNR at TPR=0.95), columns = (λ_low, λ_high).

#### Figure `fig_sim_auroc_vs_n.pdf`
Output:
- `results/paper/figures/fig_sim_auroc_vs_n.pdf`

Command:
```bash
python scripts/build_figures.py --results-root results/paper --figure sim-auroc-vs-n
```

#### Figure `fig_sim_tnr_vs_n.pdf`
Output:
- `results/paper/figures/fig_sim_tnr_vs_n.pdf`

Command:
```bash
python scripts/build_figures.py --results-root results/paper --figure sim-tnr-vs-n
```

#### Figure `fig_sim_paired_gains.pdf`
Output:
- `results/paper/figures/fig_sim_paired_gains.pdf`

Command:
```bash
python scripts/build_figures.py --results-root results/paper --figure sim-paired-gains
```

This figure reports paired gains for:
- `LR-Decomp - LR-Obs`
- `RF-Raw+Features - RF-Raw`

#### Optional TNR version of paired gains
Output:
- `results/paper/figures/fig_sim_paired_gains_tnr.pdf`

Command:
```bash
python scripts/build_figures.py --results-root results/paper --figure sim-paired-gains-tnr
```

#### Combined paired gains (AUROC + TNR@TPR95)
Output:
- `results/paper/figures/fig_sim_paired_gains_combined.pdf`

Command:
```bash
python scripts/build_figures.py --results-root results/paper --figure sim-paired-gains-combined
```

This is a single 2×2 grid (rows = AUROC / TNR@TPR95, cols = the two paired comparisons).

#### Optional: misspecification comparison figure (n = 10,000)

This figure compares the *same four methods* under a correctly specified vs misspecified expert model
(using the misspecification experiment described above). It uses only the `n_train = 10,000` cells,
aggregates across replicates (and averages within replicate across lambda scenarios), and produces a
2-panel plot:

- left: AUROC
- right: TNR at TPR = 0.95

Output:
- `results/paper_with_misspec_p10/figures/fig_sim_misspecification_comparison_n10000.pdf`

Command:
```bash
python scripts/build_figures.py \
  --results-root results/paper_with_misspec_p10 \
  --figure sim-misspecified
```

#### Generate all figures *except* score diagnostics and misspecification

This mode is the default and works with the default simulation outputs (no pooled scores stored, and
no misspecification generated):

```bash
python scripts/build_figures.py --results-root results/paper --figure all-except-score-diagnostics-and-misspecified
```

(You can still request all standard paper figures—excluding score diagnostics—via
`--figure all-except-score-diagnostics`.)

### DT-Decomp (run separately)
Run and export DT-Decomp candidates:

```bash
python scripts/run_dt.py --config paper --output-root results/dt
```

The DT script saves candidate trees and a manifest under `results/dt/`.

### One-command full pipeline
If you want the whole study in one call:

```bash
python scripts/run_pipeline.py --config paper --output-root results/paper
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
python scripts/run_pipeline.py --config smoke_plots --output-root results/smoke_plots --n-jobs 1
```

Then inspect:
- `results/smoke_plots/tables/`
- `results/smoke_plots/figures/`

## Notes on manuscript integration

The scripts produce both `.csv` and `.tex` table outputs. The `.tex` files can be included directly in a manuscript. Figures are written as PDF files suitable for LaTeX inclusion with `\includegraphics`.

## License and authorship

This project is released under the MIT License (see `LICENSE`).

If you use this repository in academic work, please cite it using `CITATION.cff`.

### Model misspecification robustness (optional)

This repository can optionally evaluate robustness to **expert-model misspecification**.
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

This mode is **off by default**.

#### Run the misspecification-robustness experiment

```bash
python scripts/run_simulations.py \
  --config paper \
  --output-root results/paper_with_misspec_p10 \
  --expert-misspecification \
  --expert-misspecification-pct 0.1
```

Notes:
- `--expert-misspecification-pct 0.1` corresponds to a random \(\pm 10\%\) multiplicative perturbation.
- The replicate-specific perturbation factors are stored in each cell pickle under `payload["metadata"]["expert_misspecification_factors"]`.

#### Build the robustness tables

This produces **two tables** under `results/paper_with_misspec_p10/tables/`:

- `tab_sim_misspec_robustness.(csv|tex)`:
  AUROC under misspecification, with paired AUROC differences vs the baseline (same replicate/scenario).
- `tab_sim_misspec_robustness_other_metrics.(csv|tex)`:
  the same paired comparisons for `TNR@TPR95`, `AUPRC`, `Brier`, and `LogLoss`.

```bash
python scripts/build_tables.py \
  --results-root results/paper_with_misspec_p10 \
  --table sim-misspec-robustness
```
