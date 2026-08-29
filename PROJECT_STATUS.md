# Project Status (Index)

This file is an index. It does not track per-run metrics.

## Why this format

Root-level status files became outdated when architecture lines evolved independently.  
Current status, run policy, and acceptance criteria are now maintained inside each architecture folder.

## Source of truth by architecture

1. `air_gap/`
   - Read: `air_gap/AGENTS.md`
   - Then read version-level notes: `air_gap/v17/AGENTS.md`, `air_gap/v17_ter/AGENTS.md`, `air_gap/v18/AGENTS.md`
2. `two_model_architecture/`
   - Read: `two_model_architecture/README.md`, `two_model_architecture/AGENTS.md`
3. `hybrid_parallel_reasoning/`
   - Read: `hybrid_parallel_reasoning/README.md`, `hybrid_parallel_reasoning/AGENTS.md`

Supporting lines:
- `seed_emergent_ir/README.md`, `seed_emergent_ir/AGENTS.md`
- `vq_bottleneck/README.md`, `vq_bottleneck/AGENTS.md`

## Global repository references

- `PROJECT_STRUCTURE.md`: canonical map of folders and ownership.
- `docs/README.md`: research intent, hypotheses, and method narrative.
- `docs/TiDAR/`: diffusion and IR concept notes.

## Reproducibility rules (applies to all lines)

- Record hypothesis, dataset path/version, config diff, seeds, and acceptance criteria for each run.
- Keep causal checks explicit (e.g., IR shuffle/drop) before claiming success.
- Store logs/metrics in versioned `results_*` folders.
- Do not delete artifacts or snapshots without explicit approval.
