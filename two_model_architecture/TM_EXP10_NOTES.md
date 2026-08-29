# TM-EXP-10: Scheduled Sampling on Factorized LA

## Purpose

Test an objective-level change for the hard `-->-` holdout:
- reduce train/eval mismatch by progressively replacing teacher-forced `mid` with model-predicted `mid` during LA training.

## Script

- `two_model_architecture/tm_exp10_scheduled_sampling.py`

## Method

Compared to TM-EXP-08, architecture is unchanged.  
Only LA training changes:
- `mid` prediction loss unchanged.
- `out` prediction is trained with mixed `mid` input:
  - ground-truth `mid` with probability `1 - p`
  - model `mid` with probability `p`
- `p` follows a predefined schedule (`start -> end`, warmup epochs).

Text-AR baseline model remains unchanged.

## Important Setup Correction (Rigor)

Early TM-EXP-10 runs had RNG coupling risk between LA-side sampling and text baseline stochasticity.

Confounded exploratory runs (excluded from interpretation):
- `two_model_architecture/results/tm_exp_10_20260227_192407/`
- `two_model_architecture/results/tm_exp_10_20260227_192950/`

Correction added in script:
- explicit epoch-level RNG stream isolation,
- dedicated LA sampling generator,
- RNG control logged in prereg plan.

## Valid Runs (after correction)

Common setup:
- holdout `-->-`, coverage-adjusted
- `d_model=192`, `num_layers=3`, `dim_ff=384`, `epochs=60`
- seeds `0,1,2`
- acceptance unchanged (`gain_min=0.03`, causal thresholds `0.20`)

### Run A: Control (`p=0.0` always)

- Run dir: `two_model_architecture/results/tm_exp_10_20260227_193655/`
- Aggregate:
  - LA intact = `0.6340`
  - Text-AR intact = `0.6213`
  - gain = `+0.0128`
  - LA deltas: shuffle = `0.5645`, drop = `0.5950`
- Verdict: not green (`gain_min` not met).

### Run B: Scheduled sampling (`0.0 -> 0.5`, warmup 30)

- Run dir: `two_model_architecture/results/tm_exp_10_20260227_194236/`
- Aggregate:
  - LA intact = `0.6323`
  - Text-AR intact = `0.6213`
  - gain = `+0.0110`
  - LA deltas: shuffle = `0.5674`, drop = `0.6128`
- Verdict: not green.

## Setup Validity

For both valid runs:
- `setup_after_adjustment.valid = true`
- `test_n = 940`
- expected schedule behavior present in train logs (`train_la_model_mid_prob`, `train_la_model_mid_frac`).

## Interpretation

1. Scheduled sampling does not improve margin on `-->-` in this regime.
2. Mild scheduled sampling (`0 -> 0.5`) is slightly worse than clean control.
3. Objective-level change tested here is not sufficient to cross the preregistered gain threshold.
