# v17s Full Runs Analysis (2026-01-04)

## Scope
This report summarizes the completed full runs for v17 and v17_ter after the
vocab alignment fixes. It uses the last contiguous epoch 1-20 block from each
`eval_metrics.jsonl` as the authoritative run record. Earlier entries in those
files belong to prior runs.

## Data sources
- v17 metrics: `air_gap/v17/results_phase2/eval_metrics.jsonl`
- v17_ter metrics: `air_gap/v17_ter/results_phase2/eval_metrics.jsonl`
- v17 logs: `air_gap/v17/run_full_phase1_20260104_121135.log`,
  `air_gap/v17/run_full_phase2_20260104_121135.log`
- v17_ter logs: `air_gap/v17_ter/run_full_phase1_20251229_155736.log`,
  `air_gap/v17_ter/run_full_phase2_20251229_184542.log`

## Key results (last epoch)

| Run | Task head | Math acc | Math shuffle | Story F1 | Story shuffle | Delta |
| --- | --- | --- | --- | --- | --- | --- |
| v17 | `story` | 1.00 | 0.06 | 0.633 | 0.109 | +0.524 |
| v17_ter | `story_pred` | 1.00 | 0.19 | 0.262 | 0.188 | +0.074 |

Notes:
- v17 best story F1 is 0.637 at epoch 13 and remains stable through epoch 20.
- v17_ter best story_pred F1 is 0.275 at epoch 1 and drifts down to 0.262 by
  epoch 20.
- v17 and v17_ter use different task heads (`story` vs `story_pred`), so the
  story F1 values are not directly comparable across runs.

## Interpretation
- **Math task**: both runs reach 100% accuracy with clear separation from the
  shuffle baseline, indicating the math channel is learned reliably.
- **Story task**:
  - v17 shows a strong gap over shuffle and a stable plateau, suggesting the
    model learns a robust story signal rather than overfitting.
  - v17_ter shows only a small margin over shuffle and no improvement after the
    earliest epochs. This pattern suggests the story_pred objective or data
    composition may be weak, mismatched, or dominated by noise.
- **Shuffle baselines**: v17_ter has a higher shuffle baseline on math (0.19 vs
  0.06), which likely reflects a different sample or task distribution. That
  baseline difference is another reason to avoid direct cross-run ranking.

## Evidence checks
- Sample files exist for every epoch: `air_gap/v17/results_phase2/samples_epoch*.jsonl`
  and `air_gap/v17_ter/results_phase2/samples_epoch*.jsonl`.
- `<UNK>` occurrences in epoch 20 samples are low (v17: 3, v17_ter: 1), which
  aligns with the vocabulary fix and suggests the eval pipeline is now valid.

## Implications for next steps
- v17 is a valid baseline for the story task; further work should focus on
  controlled variants, not revalidating fundamentals.
- v17_ter should be investigated for why story_pred does not improve. Likely
  candidates are target construction, weighting, or task formulation rather
  than model capacity.

## Dataset diagnostics (story_pred, v17_ter test split)
Computed on the phase 2 test dataset (2,000 samples, shared vocab with train).
These numbers describe the *story_pred* subset only.

- Story_pred sample count: 1,106
- Input length (tokens): mean 23.2, median 20
- Target length (tokens): mean 21.8, median 19
- Target length > max_new_tokens (192): 0%
- Token overlap F1 (input vs target): mean 0.213, median 0.207

Interpretation:
- Targets fit within the generation budget, so truncation is not limiting F1.
- The lexical overlap between input and target is ~0.21, which is close to the
  shuffle baseline F1. This suggests the story_pred metric may be dominated by
  generic token overlap rather than sequence-level prediction quality.

## Story_pred diagnostics (v17_ter)
Diagnostics were run with `air_gap/v17_ter/diagnostics_story_pred.py` using the
full test split (2,000 samples) and the trained `model_final.pt`.

### Teacher-forced losses (per token)
- math: speaker loss mean 2.14e-05 (median 7.75e-07)
- story_pred: speaker loss mean 6.07 (median 5.84)
- vq loss stays low for both tasks (math mean 9.07e-04, story_pred mean 2.75e-03)

Interpretation:
- The model fits math almost perfectly under teacher forcing.
- The story_pred channel remains high-loss under supervision, which explains the
  weak F1 plateau in generation. This points to a data/target or task definition
  issue rather than insufficient capacity.

### IR ablation (generation)
| IR mode | Math acc | Story_pred F1 |
| --- | --- | --- |
| intact | 1.00 | 0.262 |
| shuffle | 0.228 | 0.187 |
| zero | 0.00 | 0.182 |

Prefix F1 for story_pred (intact / shuffle / zero):
- 5 tokens: 0.215 / 0.084 / 0.101
- 10 tokens: 0.241 / 0.148 / 0.092
- 20 tokens: 0.263 / 0.182 / 0.176

Interpretation:
- IR carries real signal for both tasks, but the story_pred gains are modest.
- Prefix F1 rises slightly with more tokens, yet remains close to overlap-level
  baselines, reinforcing that story_pred is not being learned robustly.
