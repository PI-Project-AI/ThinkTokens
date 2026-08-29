# Repository Guidelines

> **Historical note:** these are the working conventions used while the program was active. The program concluded in July 2026; the rules are kept for the record and for AI-assisted exploration of the archive.

## Project Structure & Module Organization
- `air_gap/`: canonical air-gap lineage (v10–v18). Each version is standalone with its own scripts and results/checkpoints folders. H100 exports live under `air_gap/v18/h100_snapshot/`.
- `seed_emergent_ir/`, `vq_bottleneck/`, `two_model_architecture/`, `hybrid_parallel_reasoning/`: separate architecture lines.
- `docs/`: research notes and filings (e.g., `docs/TiDAR/`, `docs/eSoleau_INPI/`), plus `docs/archive/` for older material.
- `legacy_root/`: archived root scripts and old guides (see `legacy_root/scripts/`).

## Key Concepts (Research Intent)
- **Air‑gap**: no residual bypass; all information must pass through discrete IR tokens.
- **IR tokens**: discrete internal symbols (VQ codebook) used for reasoning, not human language.
- **Causal tests**: IR shuffle/drop must degrade performance to count as real reasoning.
- **Two‑phase training**: IR learned first (compression/prediction), then specialized for reasoning.
- **Architecture separation**: each top-level architecture is an independent hypothesis track.

## Scientific Core Rules (Mandatory)
- Define a clear, testable hypothesis and constraints before each run.
- Predefine acceptance criteria and causal controls before claiming success.
- Record dataset path/version, config diff, and seed(s) for every run.
- "Green" requires both metric gain over baseline and causal degradation under IR perturbation.
- If results look wrong, validate setup first (data split, eval code, checkpoint/vocab compatibility).
- Never delete artifacts or snapshots without explicit approval.

## Coding Style & Naming Conventions
- Python, 4-space indentation, snake_case for modules/functions.
- Versioned experiments live under `air_gap/vXX/`; scripts typically named `train_phase*.py`, `eval_*.py`, `run_v*.sh`.
- Keep filenames ASCII; keep logs as `.jsonl` when possible.

## Testing Guidelines
- No unit-test framework; validation is via evaluation scripts and logged metrics.
- Prefer explicit causal checks (e.g., IR shuffle/drop comparisons in air-gap evaluations).
- Store metrics in versioned `results_*` directories; summarize acceptance criteria in run notes.

## Research Rigor & Reproducibility
- Every run must record: hypothesis, dataset version/path, config diff, seed(s), and acceptance criteria.
- "Green" results require meeting predefined criteria and causal checks.
- Record short run notes alongside the experiment (e.g., `air_gap/v17/notes.md`).
- Never delete artifacts or snapshots without explicit approval.

## Multi-Agent Operating Rules
- Assign one architecture per agent conversation to avoid context drift.
- Use one branch per agent scope, e.g. `agent/air-gap-v17` or `agent/two-model-baseline`.
- Avoid cross-architecture edits in one commit unless explicitly requested.
- Merge only after the run note and metric artifacts are present and reviewable.

## Key Files & Sources of Truth
- `PROJECT_STRUCTURE.md`: current repo map.
- `docs/README.md` and `docs/TiDAR/`: research notes and diffusion/IR concepts.
- `air_gap/v17/` and `air_gap/v17_ter/`: small‑scale air‑gap baselines to complete locally.
- `air_gap/v18/`: large‑scale air‑gap recipe (prepare only; do not launch without approval).

## Search Hygiene
- Default search scope should exclude archived material to avoid false context:
  - `rg "pattern" --glob '!docs/archive/**' --glob '!legacy_root/**'`
- Only include `docs/archive/` or `legacy_root/` when the task explicitly asks for historical references.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and scoped (e.g., “Fix V17 configs”, “Add IR diffusion notes”).
- Keep commits focused to one experiment or doc change.
- Avoid committing large artifacts (checkpoints/results); keep them local unless explicitly requested.
- PRs should include rationale, config/seed details, and a pointer to relevant logs/metrics.
