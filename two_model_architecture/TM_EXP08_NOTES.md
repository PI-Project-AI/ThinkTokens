# TM-EXP-08: Factorized LA vs Text-AR

## Purpose

Test whether a more structured LA reasoner can recover performance against the stronger text-AR baseline from TM-EXP-07.

## Script

- `two_model_architecture/tm_exp08_compare_factorized_la.py`

## LA Change

LA reasoner is replaced by a factorized two-step architecture:
- `mid = f(a, op1, b)`
- `out = g(mid, op2, c)`

This introduces explicit compositional inductive bias and parameter sharing across operations.

## Run A: `+->*` (coverage-adjusted, 40 epochs)

- Run dir: `two_model_architecture/results/tm_exp_08_20260227_183245/`
- Aggregate (3 seeds):
  - LA-factorized intact mean = `0.8920`
  - Text-AR intact mean = `0.8564`
  - gain = `+0.0356`
  - LA deltas: shuffle = `0.7853`, drop = `0.7497`
- Verdict: green.

Delta vs TM-EXP-07 on same pair:
- gain `-0.5724 -> +0.0356` (large reversal).

## Run B: `-->-` (coverage-adjusted, 40 epochs)

- Run dir: `two_model_architecture/results/tm_exp_08_20260227_183517/`
- Aggregate (3 seeds):
  - LA-factorized intact mean = `0.6344`
  - Text-AR intact mean = `0.6227`
  - gain = `+0.0117`
  - LA deltas: shuffle = `0.5631`, drop = `0.5954`
- Verdict: not green (`gain_min=0.03` not met).

Delta vs TM-EXP-07 on same pair:
- gain `-0.5305 -> +0.0117`.

## Run C: `-->-` high-budget check (coverage-adjusted, 60 epochs, 192/3/384)

- Run dir: `two_model_architecture/results/tm_exp_08_minusminus_hi/tm_exp_08_20260227_183740/`
- Aggregate (3 seeds):
  - LA-factorized intact mean = `0.6330`
  - Text-AR intact mean = `0.6255`
  - gain = `+0.0074`
  - LA deltas: shuffle = `0.5716`, drop = `0.6230`
- Verdict: not green (margin still below threshold).

## Interpretation

1. Strong positive: factorized LA closes the gap and can beat text-AR on at least one hard pair (`+->*`).
2. Remaining issue: `-->-` reaches near parity but not robust positive margin.
3. Practical conclusion: architecture choice and inductive bias are now the main lever, not just training time.
