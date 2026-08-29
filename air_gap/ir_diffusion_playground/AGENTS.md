# Repository Guidelines

## Scope
- This folder is for low-cost IR diffusion prototypes, not production-scale runs.
- Goal: test whether diffusion-style correction/search improves IR usefulness before scaling.

## Scientific Core Rules (Mandatory)
- Define a clear, testable hypothesis and constraints before each run.
- Predefine acceptance criteria and causal controls before claiming success.
- Record dataset path/version, config diff, and seed(s) for every run.
- "Green" requires both metric gain over baseline and causal degradation under IR perturbation.
- If results look wrong, validate setup first (data split, eval code, checkpoint/vocab compatibility).
- Never delete artifacts or snapshots without explicit approval.

## Working Rules
- Keep experiments small and fast (local GPU friendly).
- Always compare against a non-diffusion baseline from the same data/config.
- Log metrics in structured files (`.jsonl` preferred) under `results/`.

## Scientific Rule
- Require IR causal controls (shuffle/drop/scramble) for any positive claim.
- Treat improvements as provisional until replicated with fixed seeds and config notes.
