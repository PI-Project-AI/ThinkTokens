# ThinkTokens

**Can a language model reason in a discrete internal language of its own, and how do you prove the internal tokens actually carry the reasoning?**

This repository is the public research archive of ThinkTokens, a small-scale, self-funded research program (October 2025 to July 2026) on machine-native discrete reasoning. The program is concluded. What it leaves behind is one mechanism proof, three characterized negative results, and the evaluation discipline that separated them.

> **Status: research concluded · July 2026 · archive release, no active development.**
>
> The readable version of this work lives in three research notes:
> - [Machine-native reasoning: the question and the design space](https://pi-project.ai/en/research-and-development/machine-native-reasoning/)
> - [A causal-necessity proof at small scale (V17)](https://pi-project.ai/en/research-and-development/latent-reasoning/)
> - [Three silent failures, and the instrument that caught them](https://pi-project.ai/en/research-and-development/latent-reasoning-failures/)

## The evaluation standard

The program's acceptance bar, written into the repository guidelines before the runs:

1. **Performance:** the model must beat its baseline on the task.
2. **Necessity:** performance must collapse when the internal reasoning tokens are perturbed (shuffled, dropped, or replaced with random codes).

A discrete "reasoning" channel that survives perturbation is not reasoning through that channel. It is decoration. One of the three negative results below passed every conventional dashboard gate and was caught only by this counterfactual test; results that lack recorded controls (V16) are marked unverified rather than claimed. That asymmetry between dashboards and counterfactuals is the main finding of the program.

## Headline result (air_gap/v17)

A 26.5M-parameter encoder→decoder pair (2 × 6-layer transformers, d=384) forced to communicate solely through 64 discrete IR slots drawn from a hard-VQ codebook of 1024 × 384-d vectors, with no residual bypass. Trained on TinyStories plus synthetic arithmetic. The arithmetic task computes answers through the channel; the story task is exact reconstruction through the bottleneck, which tests channel capacity on real language rather than reasoning.

| Task | Intact IR | Shuffled IR |
|---|---|---|
| Arithmetic accuracy | 1.00 | 0.06 |
| Story reconstruction, token F1 | 0.633 | 0.109 |

The collapse under shuffle is the point: the channel is causally load-bearing, not decorative. Caveats stated plainly: single training run per configuration, unseeded shuffle, so the floors carry run-to-run noise of a few points. Primary records: `air_gap/v17/results_phase2/eval_metrics.jsonl` (per-epoch intact and shuffled metrics) and `docs/air_gap/v17s_full_run_analysis_2026-01-04.md`. A sibling variant with a predictive Phase-1 objective (v17_ter) did **not** clear the bar on the story channel (0.262 vs 0.187 shuffled) and is recorded as a non-certified control.

## The negatives (kept on purpose)

| Line | Setup | Outcome |
|---|---|---|
| `vq_bottleneck/` | VQ codebook retrofitted onto Pythia 410M and 1.4B | 0% task accuracy at both scales; diagnosis: the model compressed information into the codes, then routed around them (report: `docs/results/EXPERIMENT_REPORT.md`) |
| `seed_emergent_ir/` | Two-pass structured IR buffer on Pythia-70M + LoRA | Tokens structurally load-bearing but semantically inert: intact = shuffled = random ≈ 6.0%; only the counterfactual saw it |
| `air_gap/v18/` | The V17 recipe scaled ~7× (~190M params, 4096 × 768-d codebook, 64 IR slots, H100, ~€300) | Phase-2 collapse to `<unk>`, 0% arithmetic; the run predated the JSONL metric logging, so no metric files were persisted (details and eval-bug disclosure: `air_gap/v18/REPORT.md`); archived as a forensic snapshot |

The scale-up failures are unsolved optimization problems, not refutations of the mechanism. They are documented in full because a positional placeholder can pass every dashboard gate, and the field's incentives run against publishing that.

## Design-space map

| Family | Folder | Status |
|---|---|---|
| VQ bottleneck retrofitted onto a pretrained LM | `vq_bottleneck/` | run · negative |
| Two-pass architecture, structured IR buffer | `seed_emergent_ir/` | run · negative |
| Air-gapped encoder and decoder, from scratch | `air_gap/` (v10 to v17) | run · mechanism proof |
| Air gap scaled ~7× | `air_gap/v18/` | run · collapsed |
| Modular language pivot (translator + reasoner) | `two_model_architecture/` | explored · archived |
| Parallel tracks: readable CoT + dense code | `hybrid_parallel_reasoning/` | designed · never run |
| Continuous reasoning, discrete checkpoints | (concept only) | parked direction |

## Related work and positioning

The core mechanism explored here is public territory, carried to scale by better-resourced teams. This archive makes no priority claim on it.

- **Token Assorted** (Meta, Feb 2025, [arXiv 2502.03275](https://arxiv.org/abs/2502.03275)): VQ-VAE discrete latent tokens mixed with text at scale, with reported gains. The discrete-latent mechanism, shipped, predating this repository.
- **Thinking Without Words / Abstract-CoT** (IBM, Apr 2026, [arXiv 2604.22709](https://arxiv.org/abs/2604.22709)): a reserved abstract vocabulary replacing verbal CoT via post-training on pretrained models, up to 11.6× fewer reasoning tokens. The closest shipped analogue to an IR channel. It reports permutation and truncation sensitivity analyses; it makes no per-token causal-necessity claim of the kind this archive's protocol requires (collapse measured against a no-reasoning baseline under intervention).
- **Do Latent Tokens Think?** (Dec 2025, [arXiv 2512.21711](https://arxiv.org/abs/2512.21711)): latent reasoning tokens often function as uninterpretable placeholders. Independent confirmation of the failure mode documented in `seed_emergent_ir/`.

What this archive contributes is complementary: a necessity protocol that any latent-token result can be graded against, one small-scale proof that clears it, and three documented ways to fail it silently.

## Repository map

- `air_gap/`: the canonical line, v10 to v18. Each version folder is standalone (model, training scripts, run notes). Start with `air_gap/v17/`.
- `docs/README.md`: research intent, hypotheses, and the method narrative.
- `docs/results/`: the vq_bottleneck experiment report and figures.
- `docs/air_gap/v17s_full_run_analysis_2026-01-04.md`: the verified V17 run record.
- `docs/eSoleau_INPI/`: the December 2025 INPI e-Soleau filing (dated design documents, French).
- `docs/archive/`, `legacy_root/`: historical material, kept as-is and not maintained.
- `PROJECT_STRUCTURE.md`, `PROJECT_STATUS.md`, `AGENTS.md`: repo conventions and per-folder sources of truth.

## Reproducibility

The archive contains code, configs, run reports, and the primary metric records for the headline runs (`air_gap/v17/results_*/eval_metrics.jsonl`, `air_gap/v17_ter/results_phase2/eval_metrics.jsonl`). It does not contain checkpoints, per-epoch sample dumps, bulk logs, or datasets (TinyStories is public; synthetic generators are included), and some run notes reference local `results_*` directories that stayed untracked; those artifacts are available on request. Historical reports are preserved with their original wording; where early parameter estimates were later corrected from source, the current-status docs carry the corrected figures (V17: 26.5M params; V18: ~190M params, 64 IR slots). Figures quoted above are single-run measurements; treat them as evidence of mechanism, not benchmark claims.

## Provenance

Development history was kept in a private repository with a dated commit trail from October 2025, alongside an INPI e-Soleau deposit dated December 8, 2025. This public tree is a cleaned snapshot with a compacted history; the full trail is available on request. Research and engineering were AI-assisted (Claude); every run, metric, and decision recorded here was reviewed by the author.

## License and citation

Apache License 2.0 (see `LICENSE`). Third-party components and datasets are credited in `THIRD_PARTY_NOTICES.md`. If you use the protocol or the run records, cite via `CITATION.cff`.

Paul Provost · [PI Project](https://pi-project.ai) · contact via the site or the email in `CITATION.cff`.
