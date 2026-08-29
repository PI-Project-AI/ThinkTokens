# TM-EXP-05: OOD Op-Pair Sweep (LA vs Text Proxy)

## Goal

Measure whether LA advantage is broad across unseen operation compositions, not only on one cherry-picked holdout pair.

## Script

- `two_model_architecture/tm_exp05_op_pair_sweep_local.py`

## Main Sweep Run

- Run dir: `two_model_architecture/results/tm_exp_05_20260227_171759/`
- Setup:
  - 9 held-out op pairs (`op1->op2`) evaluated.
  - Coverage adjustment applied per pair (remove test samples with unseen solution tokens).
  - 3 seeds per pair.
  - Epochs: 30.

## Aggregate Outcome

- Pairs total: 9
- Pairs valid: 9
- Pairs green: 5
- Green fraction: 0.556
- Mean gain over pairs (LA - Text): `+0.192`
- Gain std over pairs: `0.109`

## Per-Pair Summary (gain = LA - Text)

- `plus->minus`: gain `+0.376`, green
- `mul->mul`: gain `+0.303`, green
- `mul->plus`: gain `+0.291`, green
- `plus->plus`: gain `+0.213`, green
- `mul->minus`: gain `+0.144`, green
- `minus->plus`: gain `+0.187`, not green (causal deltas slightly below thresholds)
- `minus->mul`: gain `+0.105`, not green
- `plus->mul`: gain `+0.082`, not green
- `minus->minus`: gain `+0.026`, not green

## Interpretation

1. LA advantage is not isolated: positive gain appears across all 9 pairs in this sweep.
2. Robustness is mixed: only ~56% of pairs satisfy all preregistered causal+gain criteria.
3. Weakest zones are subtraction-led pairs, especially `minus->minus`.

## Follow-up (TM-EXP-06)

Two hard pairs were rerun with stronger matched budget:
- `minus->minus`
- `plus->mul`

See `two_model_architecture/TM_LOCAL_EXPERIMENT_LOG.md` for those results.
