# Repository Guidelines

## Scope
- This folder is the "IR + CoT parallel reasoning" architecture line.
- It focuses on coupling human-readable CoT and machine-native IR tracks.

## Scientific Core Rules (Mandatory)
- Define a clear, testable hypothesis and constraints before each run.
- Predefine acceptance criteria and causal controls before claiming success.
- Record dataset path/version, config diff, and seed(s) for every run.
- "Green" requires both metric gain over baseline and causal degradation under IR perturbation.
- If results look wrong, validate setup first (data split, eval code, checkpoint/vocab compatibility).
- Never delete artifacts or snapshots without explicit approval.

## Current Status
- Concept and planning stage; no stable training pipeline yet.

## Working Rules
- Keep conceptual docs and small prototypes here.
- Any prototype must define coupling mechanism, baseline, and failure criteria.
- Store heavy artifacts locally; do not commit large checkpoints/results by default.

## Scientific Rule
- Require consistency/causality checks (mask IR, mask CoT, cross-attention ablations) before interpreting gains.
