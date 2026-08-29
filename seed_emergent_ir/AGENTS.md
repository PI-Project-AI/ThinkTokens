# Repository Guidelines

## Scope
- This folder is the seed/emergent-IR architecture line.
- It is separate from `air_gap/` and should keep its own assumptions and metrics.

## Scientific Core Rules (Mandatory)
- Define a clear, testable hypothesis and constraints before each run.
- Predefine acceptance criteria and causal controls before claiming success.
- Record dataset path/version, config diff, and seed(s) for every run.
- "Green" requires both metric gain over baseline and causal degradation under IR perturbation.
- If results look wrong, validate setup first (data split, eval code, checkpoint/vocab compatibility).
- Never delete artifacts or snapshots without explicit approval.

## Working Rules
- Keep training/eval scripts, notes, and configs versioned inside this folder.
- Document hypothesis, seed(s), dataset path/version, and acceptance criteria per run.
- Keep large artifacts local unless explicitly requested for tracking.

## Scientific Rule
- No success claim without baseline comparison and at least one causal perturbation test.
