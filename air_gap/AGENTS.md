# Repository Guidelines

## Scope
- This folder is the canonical air-gap research line (`v10` to `v18`).
- Air-gap rule is strict: no residual bypass from encoder states to decoder outputs; reasoning must pass through discrete IR tokens.

## Scientific Core Rules (Mandatory)
- Define a clear, testable hypothesis and constraints before each run.
- Predefine acceptance criteria and causal controls before claiming success.
- Record dataset path/version, config diff, and seed(s) for every run.
- "Green" requires both metric gain over baseline and causal degradation under IR perturbation.
- If results look wrong, validate setup first (data split, eval code, checkpoint/vocab compatibility).
- Never delete artifacts or snapshots without explicit approval.

## Folder Rules
- Keep each version standalone (`air_gap/vXX/` owns its scripts, checkpoints, and notes).
- Store big-run exports under the owning version (example: `air_gap/v18/h100_snapshot/`).
- Do not delete artifacts or snapshots without explicit approval.

## Run Policy
- Local runs are expected for small/medium versions (`v17`, `v17_ter`).
- `v18` is prepare-only unless explicit approval is given for a new large run.

## Mandatory Run Outputs
- Store per-epoch metrics in JSONL under the version folder (`results_*/*.jsonl`).
- Keep a short run note in the same version folder with hypothesis, config diff, seeds, and go/no-go criteria.
- Record causal ablation outcomes (shuffle/drop/scramble) in the run note before any success claim.

## Scientific Validity
- Every run note must include hypothesis, config diff, seed(s), dataset path/version, and acceptance criteria.
- "Green" requires causal evidence (IR shuffle/drop harms results) and task metrics above trivial baselines.

## Practical Conventions
- Prefer JSONL logs in `results_*` directories.
- Keep changes scoped per version; avoid cross-version edits unless required for fairness.
