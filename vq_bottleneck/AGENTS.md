# Repository Guidelines

## Scope
- This folder contains early VQ bottleneck architecture experiments.
- Treat it as historical baseline work unless a new explicit revival plan is defined.

## Scientific Core Rules (Mandatory)
- Define a clear, testable hypothesis and constraints before each run.
- Predefine acceptance criteria and causal controls before claiming success.
- Record dataset path/version, config diff, and seed(s) for every run.
- "Green" requires both metric gain over baseline and causal degradation under IR perturbation.
- If results look wrong, validate setup first (data split, eval code, checkpoint/vocab compatibility).
- Never delete artifacts or snapshots without explicit approval.

## Working Rules
- Preserve legacy code for reproducibility.
- Add new tests/prototypes in clearly named subfolders, not by rewriting baseline files.
- Keep heavy checkpoints/results local by default.

## Scientific Rule
- Any revived experiment must state what hypothesis it tests beyond existing air-gap results.
