# V18 Training Status and Next Steps

## What we have
- **Phase 1:** Completed. Checkpoint saved at `phase1_ae.pt`.
- **Phase 2:** Ran through ~8/15 epochs (batch=64, lr=1e-4). Metrics stayed flat:
  - Math accuracy: 0%.
  - Story_pred token F1: ~0.06–0.07.
  - Outputs dominated by `<unk>`.
- **Artifacts:** `results_phase1/` and `results_phase2/` are empty; the scripts print evals but do not write JSON. Only `phase1_ae.pt` is useful.
- **Runtime / cost:** ~3 days on H100-80/90GB (GPU mem ~68GB), estimated ~€300. Run incomplete (stopped during Phase 2).

## Conclusions from the run
- Phase 2 collapsed into UNK and did not recover over multiple epochs; with current settings it is unlikely to improve.
- Phase 1 model exists and can be reused, but it is unvalidated (no saved metrics); quick re-validation is recommended before further use.
- No usable Phase 2 model or eval logs were produced (eval bug also meant only the last item per batch was counted in metrics).

## Proposed changes for next Phase 2 run
- **Reduce batch / LR:** Try batch=16–32 and lr=5e-5 (AdamW) to avoid UNK collapse.
- **Log metrics to JSONL (now implemented):** Per-epoch metrics only, to keep files tiny.
  - Phase 1 now writes `results_phase1/eval_metrics.jsonl` (predictive reconstruction acc).
  - Phase 2 now writes `results_phase2/eval_metrics.jsonl` (task metrics) and a handful of examples per epoch to `eval_samples_epoch{N}.jsonl` (<=8 lines).
- **Save checkpoint:** After Phase 2, save `results_phase2/model_final.pt` (or per-epoch checkpoints if needed).

## Perspective (what an ML team would say)
- Phase 1 checkpoint is likely usable as an initialization/IR encoder, but without metrics its quality is unknown; re-run a quick validation with the new logging to confirm reconstruction/IR stability.
- The UNK collapse points to brittleness in optimization or the IR bottleneck; try gentler hyperparams (batch/LR), shorter generations, and ensure logging for fast feedback.
- If collapse persists, revisit the IR/VQ specification or add regularization/baselines to isolate whether the architecture or training setup is at fault.

## Minimal logging snippet to add in `evaluate`
```python
# after computing metrics per epoch
log_entry = {
    "epoch": epoch + 1,
    "math_acc": acc_math,
    "story_prec": prec_story,
    "story_rec": rec_story,
    "story_f1": f1_story,
}
os.makedirs(save_dir, exist_ok=True)
with open(os.path.join(save_dir, "eval_metrics.jsonl"), "a") as f:
    f.write(json.dumps(log_entry) + "\n")

# optional: collect up to N samples per epoch and dump to a JSONL file
```

## Next actions
- Patch `train_phase2.py` with smaller batch/LR and JSONL logging.
- Restart Phase 2 using the existing `phase1_ae.pt`.
- If UNK collapse persists, consider further capacity/regularization tweaks or fall back to a baseline for comparison.
