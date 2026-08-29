# TM-EXP-07: LA vs Stronger Text-AR Baseline

## Purpose

Upgrade the text baseline from pooled classification heads (TM-EXP-04) to an autoregressive reasoning-trace decoder (`mid -> out`) and re-test hard OOD pairs.

## Script

- `two_model_architecture/tm_exp07_compare_textar_local.py`

## Fairness Check (params, default config)

For `d_model=128`, holdout `+->*` train split:
- LA baseline params: `717,996`
- Text-AR baseline params: `778,646`

The text-AR model is somewhat larger but same order of magnitude.

## Run A: `+->*` (coverage-adjusted)

- Run dir: `two_model_architecture/results/tm_exp_07_20260227_182400/`
- Aggregate (3 seeds):
  - LA intact mean = `0.3196`
  - Text-AR intact mean = `0.8920`
  - gain (LA - Text-AR) = `-0.5724`
  - LA deltas: shuffle = `0.2253`, drop = `0.3077`
- Verdict: not green (LA does not outperform stronger baseline).

## Run B: `-->-` (coverage-adjusted)

- Run dir: `two_model_architecture/results/tm_exp_07_20260227_182731/`
- Aggregate (3 seeds):
  - LA intact mean = `0.0933`
  - Text-AR intact mean = `0.6238`
  - gain (LA - Text-AR) = `-0.5305`
  - LA deltas: shuffle = `0.0500`, drop = `0.0911`
- Verdict: not green.

## Interpretation

With a stronger text-AR baseline, the original LA reasoner loses clearly on both tested hard pairs. This indicates prior LA wins (TM-EXP-04) were not robust to baseline strength.

## Follow-up

TM-EXP-08 introduces a factorized LA reasoner with explicit two-step structure (`mid = f(a,op1,b)`, `out = g(mid,op2,c)`) to address this gap.
