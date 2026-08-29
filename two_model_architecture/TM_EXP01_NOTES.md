# TM-EXP-01: LA Reasoner Causality (Local)

## Hypothesis

In a modular `A(text->LA) -> B(LA->LA) -> C(LA->text)` pipeline, a learned `B` can solve compositional arithmetic through LA, and perturbing LA should causally degrade performance.

## Constraints

- Local-only experiment.
- Lightweight artifacts only (JSON/JSONL summaries, no heavy checkpoints).
- Disjoint expression split between train/val/test.
- Deterministic translators for `A` and `C` to isolate `B`.

## Acceptance Criteria (Pre-registered)

- Intact test answer accuracy >= 0.90.
- Causal controls: intact minus shuffle >= 0.40 and intact minus drop >= 0.40.
- Seed stability: std of intact accuracy across seeds <= 0.05.

## Dataset

- Synthetic expressions: `((a op1 b) op2 c)`.
- `a,b,c` in `[0, 9]`.
- `op1,op2` in `{+, -, *}` with multiplication cap to keep bounded output.
- Disjoint split by expression key.

## Setup Validation Checklist

- Split overlap checks (train/val/test) must be zero.
- Parser `A` exact-match on all rows must be 1.0.
- Target LA trace generation exact-match must be 1.0.

## Run Command

```bash
python two_model_architecture/tm_exp01_reasoner_local.py
```

## Results

### Run A (initial factorization: independent heads)

- Output folder: `two_model_architecture/results/tm_exp01_20260227_161631/`
- Setup validation: `valid=true` (disjoint split + parser/target checks all passed).
- Aggregate (3 seeds):
  - `intact_mean=0.8636`
  - `shuffle_mean=0.1509`
  - `drop_mean=0.0655`
  - `delta_shuffle_mean=0.7126`
  - `delta_drop_mean=0.7980`
  - `intact_std=0.0155`
- Verdict: **not green** (missed intact accuracy threshold).

Interpretation:
- Causal signal was strong (large degradation under shuffle/drop), so setup and controls were meaningful.
- Failure was likely model design (final state head independent from intermediate state), not setup corruption.

### Setup-first debugging step (mandatory rule)

Before changing interpretation, setup was rechecked:
- split overlap = 0 for train/val/test,
- parser exact match = 1.0,
- target generation exact match = 1.0.

So we changed model design, not data/eval integrity.

### Run B (revised B: final-state head conditioned on intermediate LA state)

- Output folder: `two_model_architecture/results/tm_exp01_20260227_162025/`
- Config diff vs Run A:
  - `out_condition_on_mid: false -> true` (factorized reasoning path)
  - epochs kept at 40 for stable convergence.
- Aggregate (3 seeds):
  - `intact_mean=0.9225`
  - `shuffle_mean=0.1129`
  - `drop_mean=0.0702`
  - `random_b_mean=0.0105`
  - `delta_shuffle_mean=0.8096`
  - `delta_drop_mean=0.8523`
  - `intact_std=0.0335`
- Verdict: **green** (all preregistered criteria met).

### What We Learned

1. A modular LA pipeline can pass causal criteria locally on non-trivial compositional arithmetic.
2. The internal design of `B` matters: forcing `out` to depend on `mid` improved intact performance from ~0.864 to ~0.923 mean.
3. Error concentration is uneven by operation pattern (Run B intact `acc_out_by_op_pair` mean across seeds):
   - hardest: `*->+` (~0.74 mean),
   - strongest: `+->+`, `+->*`, `-->-` (roughly ~0.95 mean).

This suggests the next local experiments should target weak LA transformation regimes (`*->+`) with either curriculum or loss reweighting.
