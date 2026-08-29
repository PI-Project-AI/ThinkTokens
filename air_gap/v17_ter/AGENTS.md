# Repository Guidelines

## Scientific Purpose
- `v17_ter` is the predictive Phase-1 variant of the v17 line.
- It is used to test whether objective design (prediction vs reconstruction) changes IR usefulness at small scale.

## Scientific Core Rules (Mandatory)
- Define a clear, testable hypothesis and constraints before each run.
- Predefine acceptance criteria and causal controls before claiming success.
- Record dataset path/version, config diff, and seed(s) for every run.
- "Green" requires both metric gain over baseline and causal degradation under IR perturbation.
- If results look wrong, validate setup first (data split, eval code, checkpoint/vocab compatibility).
- Never delete artifacts or snapshots without explicit approval.

## Experiment Boundaries
- Keep tokenizer/data choices aligned with `air_gap/v17/` for fair comparison unless divergence is intentional and documented.
- Keep seed and evaluation settings aligned with `v17` for objective comparison.

## Local Run Checklist
- Run Phase 1 before Phase 2 and verify checkpoint compatibility.
- Log metrics/samples in `results_phase1/` and `results_phase2/`.
- Record hypothesis, config diff, and seeds in notes.

## Acceptance Criteria
- Must beat shuffle/random baselines and reduce `<unk>` dominance.
- IR perturbation tests must degrade performance to support causal claims.
- Stop early if collapse is persistent and causal controls show no meaningful IR signal.
