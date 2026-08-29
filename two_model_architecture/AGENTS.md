# Repository Guidelines

## Scope
- This folder is the "two-model / Langage Abstrait (LA) pivot" architecture line.
- It is distinct from air-gap versions and should be evaluated independently.
- Target design is modular: `A (text->LA)`, `B (LA->LA reasoning)`, `C (LA->text)`.

## Scientific Core Rules (Mandatory)
- Define a clear, testable hypothesis and constraints before each run.
- Predefine acceptance criteria and causal controls before claiming success.
- Record dataset path/version, config diff, and seed(s) for every run.
- "Green" requires both metric gain over baseline and causal degradation under IR perturbation.
- If results look wrong, validate setup first (data split, eval code, checkpoint/vocab compatibility).
- Never delete artifacts or snapshots without explicit approval.

## Current Status
- Early baseline/prototype stage.
- Keep scripts and notes lightweight and reproducible.

## Working Rules
- Track source files and README notes.
- Keep generated outputs in `results/` local-only unless explicitly requested.
- For each experiment, record objective, dataset slice, seeds, and key metrics.
- Keep module boundaries explicit: avoid coupling shortcuts that bypass LA between A/B/C.

## Scientific Rule
- Claims must include a causal control or baseline comparison, not only raw score improvements.
- If results look wrong, first validate setup (data split, seed, eval code, checkpoint mapping) before interpreting.
- Hypothesis and constraints must be explicit and testable.
- Rigor means reproducible reruns with stable metrics, not one-off gains.
