# Repository Guidelines

## Scientific Purpose
- `v17` is the small-scale air-gap baseline on TinyStories + Math.
- Core question: can a ~26.5M-parameter model maintain reasoning through discrete IR without bypass.
- This variant is the reconstruction-oriented baseline used for comparison with `v17_ter`.

## Scientific Core Rules (Mandatory)
- Define a clear, testable hypothesis and constraints before each run.
- Predefine acceptance criteria and causal controls before claiming success.
- Record dataset path/version, config diff, and seed(s) for every run.
- "Green" requires both metric gain over baseline and causal degradation under IR perturbation.
- If results look wrong, validate setup first (data split, eval code, checkpoint/vocab compatibility).
- Never delete artifacts or snapshots without explicit approval.

## Experiment Boundaries
- Keep tokenizer and data handling aligned with `air_gap/v17_ter/` when comparison is intended.
- Track any intentional divergence explicitly in run notes.
- Keep seed and eval protocol aligned with `v17_ter` for fair comparisons.

## Local Run Checklist
- Verify seed, vocab setup, and checkpoint compatibility before launch.
- Use phase-separated runs (`train_phase1.py`, then `train_phase2.py`).
- Log metrics and sample outputs in `results_phase1/` and `results_phase2/`.

## Acceptance Criteria
- Metrics must beat shuffle/random baselines.
- IR causal checks (shuffle/drop) must degrade performance.
- No "success" claim without reproducible config + logs.
- Stop early if outputs collapse to `<unk>` with flat metrics for multiple evals.
