# Repository Guidelines

## Scientific Purpose
- `v18` is the scale-up air-gap test (~190M params) for real-language + reasoning through IR.
- Target is not just training completion; target is causal, non-trivial reasoning with valid logs.

## Scientific Core Rules (Mandatory)
- Define a clear, testable hypothesis and constraints before each run.
- Predefine acceptance criteria and causal controls before claiming success.
- Record dataset path/version, config diff, and seed(s) for every run.
- "Green" requires both metric gain over baseline and causal degradation under IR perturbation.
- If results look wrong, validate setup first (data split, eval code, checkpoint/vocab compatibility).
- Never delete artifacts or snapshots without explicit approval.

## Current Status
- H100 snapshot exists under `h100_snapshot/` as an archival export.
- That run is not scientifically exploitable as-is (Phase 2 collapse + incomplete eval logging).

## Work Policy
- Prepare and validate configs locally.
- Do not launch new full V18 training without explicit approval.

## Pre-Run Requirements
- Confirm metric logging (`results_phase1/eval_metrics.jsonl`, `results_phase2/eval_metrics.jsonl`).
- Confirm checkpoint/vocab compatibility checks before Phase 2.
- Record hypothesis, config diff, seeds, and acceptance criteria in run notes.

## Go/No-Go Gates (Before Cloud Spend)
- No-go if local smoke tests already show `<unk>` collapse or broken logging.
- Go only if Phase 1 and short Phase 2 produce stable, interpretable metrics and saved artifacts.
- Any cloud run plan must include explicit abort criteria and checkpoint cadence.
