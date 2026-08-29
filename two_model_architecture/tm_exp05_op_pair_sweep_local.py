"""
TM-EXP-05 (local): OOD op-pair sweep for LA vs text-only comparison.

Runs TM-EXP-04-style comparisons over every held-out (op1, op2) pair and
summarizes whether LA gains are broad or narrow.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from statistics import mean, pstdev
from typing import Dict, List

try:
    from two_model_architecture.tm_exp01_reasoner_local import (
        Config as LAConfig,
        OPS,
        build_expression_space,
        build_samples,
        evaluate_setup,
        make_splits,
        parse_seeds,
    )
    from two_model_architecture.tm_exp04_compare_local import (
        aggregate,
        enforce_test_solution_token_coverage,
        run_seed,
    )
except ModuleNotFoundError:
    from tm_exp01_reasoner_local import (  # type: ignore
        Config as LAConfig,
        OPS,
        build_expression_space,
        build_samples,
        evaluate_setup,
        make_splits,
        parse_seeds,
    )
    from tm_exp04_compare_local import (  # type: ignore
        aggregate,
        enforce_test_solution_token_coverage,
        run_seed,
    )


def pair_tag(op1: str, op2: str) -> str:
    repl = {"+": "plus", "-": "minus", "*": "mul"}
    return f"{repl[op1]}_to_{repl[op2]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="TM-EXP-05 op-pair sweep")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_min", type=int, default=0)
    parser.add_argument("--num_max", type=int, default=9)
    parser.add_argument("--mul_cap", type=int, default=5)
    parser.add_argument("--split_seed", type=int, default=20260227)
    parser.add_argument("--template_seed", type=int, default=1337)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--output_root", type=str, default="two_model_architecture/results")

    # Sweep acceptance criteria (same style as TM-EXP-04)
    parser.add_argument("--gain_min", type=float, default=0.03)
    parser.add_argument("--la_delta_shuffle_min", type=float, default=0.20)
    parser.add_argument("--la_delta_drop_min", type=float, default=0.20)
    args = parser.parse_args()

    cfg = LAConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_min=args.num_min,
        num_max=args.num_max,
        mul_cap=args.mul_cap,
        split_seed=args.split_seed,
        template_seed=args.template_seed,
        seeds=args.seeds,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, f"tm_exp_05_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    prereg = {
        "experiment_id": "TM-EXP-05",
        "purpose": "Assess breadth of LA-vs-text gain over all held-out op pairs",
        "criteria_per_pair": {
            "gain_min": args.gain_min,
            "la_delta_shuffle_min": args.la_delta_shuffle_min,
            "la_delta_drop_min": args.la_delta_drop_min,
        },
        "config": asdict(cfg),
        "timestamp": ts,
    }
    with open(os.path.join(run_dir, "preregistered_plan.json"), "w", encoding="utf-8") as f:
        json.dump(prereg, f, indent=2)

    exprs = build_expression_space(cfg.num_min, cfg.num_max, cfg.mul_cap)
    seeds = parse_seeds(cfg.seeds)
    pair_results: List[Dict[str, object]] = []

    for op1 in OPS:
        for op2 in OPS:
            tag = pair_tag(op1, op2)
            pair_dir = os.path.join(run_dir, f"holdout_{tag}")
            os.makedirs(pair_dir, exist_ok=True)

            splits = make_splits(
                exprs,
                cfg.split_seed,
                cfg.train_ratio,
                cfg.val_ratio,
                split_mode="holdout_op_pair",
                num_max=cfg.num_max,
                holdout_op1=op1,
                holdout_op2=op2,
            )
            rows_by_split = {
                split_name: build_samples(split_exprs, cfg.template_seed + i)
                for i, (split_name, split_exprs) in enumerate(splits.items())
            }

            setup_before = evaluate_setup(rows_by_split)
            coverage_adjustment = enforce_test_solution_token_coverage(rows_by_split)
            setup_after = evaluate_setup(rows_by_split)

            with open(os.path.join(pair_dir, "setup_validation.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "setup_before_adjustment": setup_before,
                        "coverage_adjustment": coverage_adjustment,
                        "setup_after_adjustment": setup_after,
                    },
                    f,
                    indent=2,
                )
            if not setup_after["valid"] or len(rows_by_split["test"]) == 0:
                pair_results.append(
                    {
                        "op1": op1,
                        "op2": op2,
                        "tag": tag,
                        "status": "invalid_setup",
                        "test_after": len(rows_by_split["test"]),
                    }
                )
                continue

            seed_summaries = [run_seed(rows_by_split, cfg, seed, pair_dir) for seed in seeds]
            agg = aggregate(seed_summaries)
            verdict = {
                "gain_ok": agg["gain_mean"] >= args.gain_min,
                "la_causal_shuffle_ok": agg["la_delta_shuffle_mean"] >= args.la_delta_shuffle_min,
                "la_causal_drop_ok": agg["la_delta_drop_mean"] >= args.la_delta_drop_min,
            }
            verdict["green"] = all(verdict.values())

            with open(os.path.join(pair_dir, "seed_summaries.json"), "w", encoding="utf-8") as f:
                json.dump(seed_summaries, f, indent=2)
            with open(os.path.join(pair_dir, "aggregate_summary.json"), "w", encoding="utf-8") as f:
                json.dump({"aggregate": agg, "verdict": verdict}, f, indent=2)

            pair_row = {
                "op1": op1,
                "op2": op2,
                "tag": tag,
                "status": "ok",
                "coverage_removed": coverage_adjustment["removed_count"],
                "coverage_removed_fraction": coverage_adjustment["removed_fraction"],
                "aggregate": agg,
                "verdict": verdict,
            }
            pair_results.append(pair_row)
            with open(os.path.join(run_dir, "pair_results.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(pair_row) + "\n")

    ok_pairs = [r for r in pair_results if r.get("status") == "ok"]
    green_pairs = [r for r in ok_pairs if r["verdict"]["green"]]
    gains = [r["aggregate"]["gain_mean"] for r in ok_pairs]
    summary = {
        "pairs_total": len(pair_results),
        "pairs_ok": len(ok_pairs),
        "pairs_green": len(green_pairs),
        "pairs_green_fraction": (len(green_pairs) / len(ok_pairs)) if ok_pairs else 0.0,
        "gain_mean_over_pairs": mean(gains) if gains else 0.0,
        "gain_std_over_pairs": pstdev(gains) if len(gains) > 1 else 0.0,
        "best_pair_by_gain": None,
        "worst_pair_by_gain": None,
        "run_dir": run_dir,
    }
    if ok_pairs:
        best = max(ok_pairs, key=lambda r: r["aggregate"]["gain_mean"])
        worst = min(ok_pairs, key=lambda r: r["aggregate"]["gain_mean"])
        summary["best_pair_by_gain"] = {
            "tag": best["tag"],
            "gain_mean": best["aggregate"]["gain_mean"],
            "la_intact_mean": best["aggregate"]["la_intact_mean"],
            "text_intact_mean": best["aggregate"]["text_intact_mean"],
        }
        summary["worst_pair_by_gain"] = {
            "tag": worst["tag"],
            "gain_mean": worst["aggregate"]["gain_mean"],
            "la_intact_mean": worst["aggregate"]["la_intact_mean"],
            "text_intact_mean": worst["aggregate"]["text_intact_mean"],
        }

    with open(os.path.join(run_dir, "sweep_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
