# TM-EXP-04: LA vs Text-Only Baseline (Local Proxy)

## Purpose

Provide an early, controlled comparison between:
- `LA-pivot`: deterministic `A` + learned `B(LA->LA)` + deterministic `C`
- `Text-only baseline`: matched-capacity encoder model predicting the same reasoning trace outputs directly from text

This is a **local proxy** comparison, not a final benchmark against an external production CoT LLM.

## Protocol

- Shared synthetic dataset and split for both models.
- Same seed list and epoch budget.
- Setup validation before interpretation.
- Multi-seed reporting.
- Causal controls required for LA (`shuffle`, `drop`).

## Runs

### 1. OOD holdout op pair (`*->+`) with confound reduction

- Run dir: `two_model_architecture/results/tm_exp_04_20260227_165400/`
- Setup:
  - split overlap = 0
  - parser exact match = 1.0
  - target exact match = 1.0
  - no missing input tokens in train
  - strict test-coverage adjustment applied:
    - removed 4/360 test rows (1.11%) with unseen solution tokens
    - missing test solution tokens after adjustment = 0
- Aggregate (3 seeds):
  - LA intact mean = 0.4448
  - Text intact mean = 0.1582
  - gain (LA - Text) mean = 0.2865
  - LA delta shuffle mean = 0.3436
  - LA delta drop mean = 0.4036
- Verdict under prereg criteria: green

Interpretation:
- LA outperforms the matched text-only proxy on this OOD composition split.
- Absolute performance remains moderate; this is not yet strong generalization.
- Coverage confound is now explicitly controlled for this OOD report.

### 2. IID random split

- Run dir: `two_model_architecture/results/tm_exp_04_20260227_164608/`
- Setup:
  - split overlap = 0
  - parser exact match = 1.0
  - target exact match = 1.0
  - no missing input or solution tokens in train for test set
- Aggregate (3 seeds):
  - LA intact mean = 0.9417
  - Text intact mean = 0.6156
  - gain (LA - Text) mean = 0.3261
  - LA delta shuffle mean = 0.8357
  - LA delta drop mean = 0.8856
- Verdict under prereg criteria: green

Interpretation:
- Strong local signal that LA architecture can outperform this matched text-only proxy in IID.

## Limitations

1. Baseline is a local matched proxy, not an external classic CoT LLM.
2. Task is synthetic arithmetic composition, not broad natural-language reasoning.
3. External validity remains unproven until a benchmark suite with standardized CoT baselines is run.

## Takeaway

TM-EXP-04 is a meaningful early result: under controlled local conditions, LA outperforms the current matched text-only proxy while preserving causal dependence checks. It is evidence for continued investment, not final proof against real CoT LLM baselines.
