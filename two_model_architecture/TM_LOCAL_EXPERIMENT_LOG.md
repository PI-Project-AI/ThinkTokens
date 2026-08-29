# Two-Model Local Experiment Log

## Date

2026-02-27

## Invalidated Run Artifacts

- `two_model_architecture/results/tm_exp_04_20260227_174131/`

Reason:
- two TM-EXP-04 runs were launched in parallel within the same timestamp second,
  causing potential output-directory collision.
- Marked invalid and excluded from interpretation.

## TM-EXP-01 (IID random split, compositional arithmetic)

- Script: `two_model_architecture/tm_exp01_reasoner_local.py`
- Final run: `two_model_architecture/results/tm_exp01_20260227_162025/`
- Setup validation:
  - split overlap = 0
  - parser exact match = 1.0
  - target exact match = 1.0
- Main result (3 seeds):
  - intact mean = 0.9225
  - shuffle mean = 0.1129
  - drop mean = 0.0702
  - random_b mean = 0.0105
  - intact std = 0.0335
  - verdict = green

Notes:
- Earlier variant (independent `B` heads) failed intact threshold (~0.864) despite strong causal deltas.
- Revised `B` with `out` conditioned on `mid` crossed criteria.
- Hardest op pair under IID remained `*->+` (mean ~0.74 across seeds).

## TM-EXP-02 (OOD holdout high digits)

- Run: `two_model_architecture/results/tm_exp_02_20260227_162430/`
- Pre-registered split: train/val on lower digits, test includes higher digits.
- Main result (3 seeds):
  - intact mean = 0.0641
  - shuffle mean = 0.0430
  - drop mean = 0.0330
  - verdict = not green

Setup-first diagnostic (before interpretation):
- train/test input token coverage mismatch:
  - unseen input LA tokens in test: `8`, `9`
- unseen output tokens in test also present.

Conclusion:
- This run is confounded for reasoning claims because unseen LA symbols break fair OOD interpretation.

## TM-EXP-03 (OOD holdout op pair, token-coverage-safe input)

- Run: `two_model_architecture/results/tm_exp_03_20260227_162637/`
- Split mode: hold out `op1='*', op2='+'` in test.
- Setup validation:
  - no split overlap
  - parser/target exact = 1.0
  - no missing test input tokens in train
  - 2 missing test solution tokens (`29`, `31`) in train
- Main result (3 seeds):
  - intact mean = 0.5120
  - shuffle mean = 0.0944
  - drop mean = 0.0250
  - delta shuffle mean = 0.4176
  - delta drop mean = 0.4870
  - verdict = not green (intact below 0.55 threshold)

Conclusion:
- Model retains causal dependency on LA under unseen op-composition.
- Generalization to unseen `*->+` composition remains limited (~0.51).
- The failure is informative, not random collapse.

## TM-EXP-04 (LA vs text-only proxy baseline)

- Script: `two_model_architecture/tm_exp04_compare_local.py`
- Notes: `two_model_architecture/TM_EXP04_NOTES.md`

### OOD holdout op pair (`*->+`) (coverage-adjusted)
- Run: `two_model_architecture/results/tm_exp_04_20260227_165400/`
- Setup:
  - split/parser/target checks all valid
  - strict test coverage adjustment applied:
    - removed 4/360 test rows with unseen solution tokens
    - missing test solution tokens after adjustment = 0
- Aggregate (3 seeds):
  - LA intact mean = 0.4448
  - Text intact mean = 0.1582
  - gain mean = 0.2865
  - LA deltas: shuffle = 0.3436, drop = 0.4036
- Verdict: green (for preregistered local criteria)

### IID random split
- Run: `two_model_architecture/results/tm_exp_04_20260227_164608/`
- Setup:
  - split/parser/target checks valid
  - no missing test input/solution tokens in train
- Aggregate (3 seeds):
  - LA intact mean = 0.9417
  - Text intact mean = 0.6156
  - gain mean = 0.3261
  - LA deltas: shuffle = 0.8357, drop = 0.8856
- Verdict: green (for preregistered local criteria)

Interpretation:
- Under this local matched-proxy baseline, LA currently outperforms text-only.
- This is still pre-benchmark evidence and not yet a definitive claim vs classic external CoT LLMs.

## TM-EXP-05 (OOD op-pair sweep)

- Script: `two_model_architecture/tm_exp05_op_pair_sweep_local.py`
- Notes: `two_model_architecture/TM_EXP05_NOTES.md`
- Run: `two_model_architecture/results/tm_exp_05_20260227_171759/`

Setup:
- 9 held-out op pairs (`op1->op2`)
- coverage adjustment per pair
- 3 seeds per pair
- 30 epochs per seed

Result:
- pairs green: `5/9`
- mean gain (LA - Text): `+0.192`
- gain positive on all pairs, but causal+gain criteria fail on 4 pairs

Interpretation:
- LA advantage is broad but not uniformly robust.
- Hardest regimes concentrate around subtraction-led compositions.

## TM-EXP-06 (targeted hard-pair stronger budget)

Purpose:
- test whether weak TM-EXP-05 pairs are optimization-limited.
- stronger matched config: `d_model=192`, `layers=3`, `dim_ff=384`, `epochs=60`.

### TM-EXP-06A (`minus->minus`)

- Run: `two_model_architecture/results/tm_exp_06_minusminus/tm_exp_04_20260227_174501/`
- Aggregate:
  - LA intact = 0.1663
  - Text intact = 0.1053
  - gain = +0.0610
  - LA deltas: shuffle = 0.1255, drop = 0.1660
- Verdict: not green

Delta vs TM-EXP-05 baseline on same pair:
- gain: `+0.026 -> +0.061` (improved)
- LA causal deltas increased but remain below threshold.

### TM-EXP-06B (`plus->mul`)

- Run: `two_model_architecture/results/tm_exp_06_plusmul/tm_exp_04_20260227_175507/`
- Aggregate:
  - LA intact = 0.6592
  - Text intact = 0.1729
  - gain = +0.4863
  - LA deltas: shuffle = 0.5649, drop = 0.6386
- Verdict: green

Delta vs TM-EXP-05 baseline on same pair:
- gain: `+0.082 -> +0.486` (large improvement)

Interpretation:
- Some weak pairs were training-budget-limited (`plus->mul`).
- `minus->minus` remains structurally difficult under current recipe.

## TM-EXP-07 (stronger text-AR baseline)

- Script: `two_model_architecture/tm_exp07_compare_textar_local.py`
- Notes: `two_model_architecture/TM_EXP07_NOTES.md`

Runs:
1. `two_model_architecture/results/tm_exp_07_20260227_182400/` (`+->*`)
2. `two_model_architecture/results/tm_exp_07_20260227_182731/` (`-->-`)

Result:
- LA loses clearly on both hard pairs against text-AR.
- This invalidates any broad outperform claim based only on TM-EXP-04 baseline strength.

## TM-EXP-08 (factorized LA recovery)

- Script: `two_model_architecture/tm_exp08_compare_factorized_la.py`
- Notes: `two_model_architecture/TM_EXP08_NOTES.md`

Runs:
1. `two_model_architecture/results/tm_exp_08_20260227_183245/` (`+->*`)
2. `two_model_architecture/results/tm_exp_08_20260227_183517/` (`-->-`)
3. `two_model_architecture/results/tm_exp_08_minusminus_hi/tm_exp_08_20260227_183740/` (`-->-`, higher budget)

Result:
- `+->*`: LA-factorized regains lead over text-AR (green).
- `-->-`: LA-factorized reaches near parity/slight positive gain, but below preregistered margin threshold.
- High-budget `-->-` confirms near parity but not robust margin.

Interpretation:
- Factorized LA architecture is a meaningful improvement over original LA.
- Remaining bottleneck is robust margin on subtraction-heavy compositions.

## TM-EXP-09 (subtraction-focused curriculum on factorized LA)

- Script: `two_model_architecture/tm_exp09_subtraction_curriculum.py`
- Notes: `two_model_architecture/TM_EXP09_NOTES.md`
- Holdout: `-->-` (coverage-adjusted, 3 seeds, high-budget 192/3/384, 60 epochs)

Runs:
1. `two_model_architecture/results/tm_exp_09_20260227_185443/` (`sub_focus=both`, `1.0->3.0`, warmup=20)
2. `two_model_architecture/results/tm_exp_09_20260227_190037/` (`sub_focus=out`, `1.0->4.0`, warmup=20)
3. `two_model_architecture/results/tm_exp_09_20260227_190624/` (`sub_focus=out`, `4.0->8.0`, warmup=1)

Setup-first checks:
- all runs have `setup_after_adjustment.valid = true`
- all runs have test size `n=940`
- curriculum fields logged per epoch (`train_la_curriculum_sub_weight`)

Result summary:
- Run 1 gain: `+0.0074`
- Run 2 gain: `+0.0074`
- Run 3 gain: `-0.0004`
- all runs keep LA causal deltas above thresholds, but all fail `gain_min=0.03`

Interpretation:
- No evidence that subtraction loss reweighting (as tested) solves the `-->-` margin gap.
- Mild curricula recover near parity only (same level as TM-EXP-08 high-budget).
- Extreme reweighting slightly harms final gain.

## TM-EXP-10 (scheduled sampling on factorized LA)

- Script: `two_model_architecture/tm_exp10_scheduled_sampling.py`
- Notes: `two_model_architecture/TM_EXP10_NOTES.md`
- Holdout: `-->-` (coverage-adjusted, 3 seeds, high-budget 192/3/384, 60 epochs)

Invalidated exploratory runs (excluded):
- `two_model_architecture/results/tm_exp_10_20260227_192407/`
- `two_model_architecture/results/tm_exp_10_20260227_192950/`

Reason:
- early TM-EXP-10 version allowed RNG coupling between LA-side scheduled-sampling randomness and text baseline stochasticity.
- setup code was corrected before final interpretation (epoch-level RNG stream isolation + dedicated LA sampling generator).

Valid runs:
1. `two_model_architecture/results/tm_exp_10_20260227_193655/` (control: `p=0.0`)
2. `two_model_architecture/results/tm_exp_10_20260227_194236/` (scheduled sampling: `0.0->0.5`, warmup=30)

Setup-first checks (valid runs):
- `setup_after_adjustment.valid = true`
- test size `n=940`
- scheduled-sampling fields logged (`train_la_model_mid_prob`, `train_la_model_mid_frac`)

Results:
- Control gain: `+0.0128` (not green)
- Scheduled-sampling gain: `+0.0110` (not green)
- both runs satisfy LA causal thresholds, but fail `gain_min=0.03`

Interpretation:
- Scheduled sampling (as tested) does not improve `-->-` margin.
- Mild schedule slightly underperforms clean control.
