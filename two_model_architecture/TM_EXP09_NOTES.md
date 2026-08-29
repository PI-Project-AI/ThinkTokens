# TM-EXP-09: Subtraction-Focused Curriculum on Factorized LA

## Purpose

Test whether a subtraction-focused LA training curriculum can improve the hardest OOD holdout (`-->-`) against the stronger text-AR baseline.

## Script

- `two_model_architecture/tm_exp09_subtraction_curriculum.py`

## Method Change vs TM-EXP-08

- Keep factorized LA architecture unchanged.
- Keep text-AR baseline unchanged.
- Change only LA training loss:
  - upweight examples containing `-` (on `op1`, `op2`, or both via `--la_sub_focus`)
  - weight schedule controlled by:
    - `--la_sub_weight_start`
    - `--la_sub_weight_end`
    - `--la_sub_weight_warmup_epochs`

This isolates curriculum effect without confounding model-class changes.

## Common Setup

- holdout: `op1='-'`, `op2='-'`
- coverage adjustment enabled
- config: `d_model=192`, `num_layers=3`, `dim_ff=384`, `epochs=60`
- seeds: `0,1,2`
- acceptance unchanged:
  - `gain_min=0.03`
  - `la_delta_shuffle_min=0.20`
  - `la_delta_drop_min=0.20`

## Run A (`sub_focus=both`, `1.0 -> 3.0`, warmup 20)

- Run dir: `two_model_architecture/results/tm_exp_09_20260227_185443/`
- Aggregate:
  - LA intact = `0.6330`
  - Text-AR intact = `0.6255`
  - gain = `+0.0074`
  - LA deltas: shuffle = `0.5631`, drop = `0.6135`
- Verdict: not green (`gain_min` not met).

## Run B (`sub_focus=out`, `1.0 -> 4.0`, warmup 20)

- Run dir: `two_model_architecture/results/tm_exp_09_20260227_190037/`
- Aggregate:
  - LA intact = `0.6330`
  - Text-AR intact = `0.6255`
  - gain = `+0.0074`
  - LA deltas: shuffle = `0.5635`, drop = `0.6135`
- Verdict: not green.

## Run C (stress: `sub_focus=out`, `4.0 -> 8.0`, warmup 1)

- Run dir: `two_model_architecture/results/tm_exp_09_20260227_190624/`
- Aggregate:
  - LA intact = `0.6252`
  - Text-AR intact = `0.6255`
  - gain = `-0.0004`
  - LA deltas: shuffle = `0.5571`, drop = `0.6057`
- Verdict: not green.

## Setup Validity Check

All three runs:
- `setup_after_adjustment.valid = true`
- `test_n = 940`
- no setup errors detected

Curriculum was actually applied:
- per-epoch `train_la_curriculum_sub_weight` changes as configured in training logs.

## Interpretation

1. Subtraction curriculum, as tested here, does not improve `-->-` margin over text-AR.
2. Mild/medium curriculum gives near-parity, same as TM-EXP-08 high-budget.
3. Extreme weighting degrades margin slightly.

Conclusion: bottleneck is likely architectural/objective-level beyond simple subtraction reweighting.
