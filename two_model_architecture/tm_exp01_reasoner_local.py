"""
TM-EXP-01 (local): Two-model LA pivot with rigorous controls.

Purpose:
- Module A (deterministic): text -> LA(problem)
- Module B (learned):      LA(problem) -> LA(solution trace)
- Module C (deterministic):LA(solution trace) -> text answer

This experiment is intentionally local and lightweight. It emphasizes:
1) pre-registered hypothesis and acceptance criteria,
2) setup validation before interpretation,
3) causal controls (shuffle/drop/random-B),
4) reproducibility over multiple seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OPS = ["+", "-", "*"]
NUMBER_MIN = -200
NUMBER_MAX = 200
NUMBER_TOKENS = [str(i) for i in range(NUMBER_MIN, NUMBER_MAX + 1)]
SPECIALS = ["<pad>", "<mask>"]
VOCAB = SPECIALS + OPS + NUMBER_TOKENS
TOK2ID = {tok: i for i, tok in enumerate(VOCAB)}
ID2TOK = {i: tok for tok, i in TOK2ID.items()}
PAD_ID = TOK2ID["<pad>"]
MASK_ID = TOK2ID["<mask>"]

TEXT_TEMPLATES = [
    "compute {a} {op1} {b} then {op2} {c}",
    "what is {a} {op1} {b} then {op2} {c}",
    "start with {a}, apply {op1} {b}, then {op2} {c}",
    "evaluate {a} {op1} {b} and then {op2} {c}",
]


def apply_op(lhs: int, op: str, rhs: int) -> int:
    if op == "+":
        return lhs + rhs
    if op == "-":
        return lhs - rhs
    if op == "*":
        return lhs * rhs
    raise ValueError(f"Unknown op: {op}")


def expression_key(a: int, op1: str, b: int, op2: str, c: int) -> str:
    return f"{a}|{op1}|{b}|{op2}|{c}"


def render_text(a: int, op1: str, b: int, op2: str, c: int, template_idx: int) -> str:
    template = TEXT_TEMPLATES[template_idx % len(TEXT_TEMPLATES)]
    return template.format(a=a, op1=op1, b=b, op2=op2, c=c)


def parse_text_to_la_problem(text: str) -> List[int]:
    # A deterministic parser for the synthetic grammar.
    pieces = re.findall(r"-?\d+|[+\-*]", text)
    if len(pieces) != 5:
        raise ValueError(f"Could not parse LA problem from text: {text}")
    n1, op1, n2, op2, n3 = pieces
    return [TOK2ID[n1], TOK2ID[op1], TOK2ID[n2], TOK2ID[op2], TOK2ID[n3]]


def la_solution_to_text(la_solution_ids: Sequence[int]) -> str:
    # C only uses final state token as the human-facing answer.
    ans_tok = ID2TOK[int(la_solution_ids[-1])]
    return ans_tok


def build_expression_space(num_min: int, num_max: int, mul_cap: int) -> List[Tuple[int, str, int, str, int]]:
    exprs: List[Tuple[int, str, int, str, int]] = []
    for a in range(num_min, num_max + 1):
        for b in range(num_min, num_max + 1):
            for c in range(num_min, num_max + 1):
                for op1 in OPS:
                    for op2 in OPS:
                        if (op1 == "*" and (abs(a) > mul_cap or abs(b) > mul_cap)) or (
                            op2 == "*" and (abs(c) > mul_cap)
                        ):
                            continue
                        mid = apply_op(a, op1, b)
                        out = apply_op(mid, op2, c)
                        if NUMBER_MIN <= mid <= NUMBER_MAX and NUMBER_MIN <= out <= NUMBER_MAX:
                            exprs.append((a, op1, b, op2, c))
    return exprs


def make_splits(
    exprs: Sequence[Tuple[int, str, int, str, int]],
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    split_mode: str = "random",
    holdout_high_digits: int = 0,
    num_max: int = 9,
    holdout_op1: str = "*",
    holdout_op2: str = "+",
) -> Dict[str, List[Tuple[int, str, int, str, int]]]:
    rng = random.Random(split_seed)
    exprs = list(exprs)
    if split_mode == "random":
        rng.shuffle(exprs)
        n = len(exprs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return {
            "train": exprs[:n_train],
            "val": exprs[n_train : n_train + n_val],
            "test": exprs[n_train + n_val :],
        }
    if split_mode == "holdout_high_digits":
        if holdout_high_digits <= 0:
            raise ValueError("holdout_high_digits must be > 0 when split_mode=holdout_high_digits")
        cutoff = num_max - holdout_high_digits
        in_dist = [e for e in exprs if e[0] <= cutoff and e[2] <= cutoff and e[4] <= cutoff]
        ood = [e for e in exprs if not (e[0] <= cutoff and e[2] <= cutoff and e[4] <= cutoff)]
        if not in_dist or not ood:
            raise ValueError("Invalid holdout split: empty in-distribution or OOD set")
        rng.shuffle(in_dist)
        rng.shuffle(ood)
        n = len(in_dist)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return {
            "train": in_dist[:n_train],
            "val": in_dist[n_train : n_train + n_val],
            "test": ood,
        }
    if split_mode == "holdout_op_pair":
        if holdout_op1 not in OPS or holdout_op2 not in OPS:
            raise ValueError(f"holdout_op1/holdout_op2 must be in {OPS}")
        train_val = [e for e in exprs if not (e[1] == holdout_op1 and e[3] == holdout_op2)]
        test = [e for e in exprs if e[1] == holdout_op1 and e[3] == holdout_op2]
        if not train_val or not test:
            raise ValueError("Invalid holdout_op_pair split: empty train_val or test set")
        rng.shuffle(train_val)
        n = len(train_val)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return {
            "train": train_val[:n_train],
            "val": train_val[n_train : n_train + n_val],
            "test": test,
        }
    raise ValueError(f"Unknown split_mode: {split_mode}")


def build_samples(
    exprs: Sequence[Tuple[int, str, int, str, int]],
    template_seed: int,
) -> List[Dict[str, object]]:
    rng = random.Random(template_seed)
    rows: List[Dict[str, object]] = []
    for (a, op1, b, op2, c) in exprs:
        mid = apply_op(a, op1, b)
        out = apply_op(mid, op2, c)
        template_idx = rng.randint(0, len(TEXT_TEMPLATES) - 1)
        text = render_text(a, op1, b, op2, c, template_idx)
        problem_ids = [TOK2ID[str(a)], TOK2ID[op1], TOK2ID[str(b)], TOK2ID[op2], TOK2ID[str(c)]]
        solution_ids = [TOK2ID[str(mid)], TOK2ID[str(out)]]
        rows.append(
            {
                "key": expression_key(a, op1, b, op2, c),
                "text": text,
                "problem_ids": problem_ids,
                "solution_ids": solution_ids,
                "answer_text": str(out),
            }
        )
    return rows


class LADataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, object]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.rows[idx]


def collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    problem = torch.tensor([b["problem_ids"] for b in batch], dtype=torch.long)
    solution = torch.tensor([b["solution_ids"] for b in batch], dtype=torch.long)
    return {
        "problem_ids": problem,
        "solution_ids": solution,
        "text": [b["text"] for b in batch],
        "answer_text": [b["answer_text"] for b in batch],
        "key": [b["key"] for b in batch],
    }


@dataclass
class Config:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 256
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 30
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    num_min: int = 0
    num_max: int = 9
    mul_cap: int = 5
    split_seed: int = 20260227
    template_seed: int = 1337
    seeds: str = "0,1,2"
    output_root: str = "two_model_architecture/results"
    out_condition_on_mid: bool = True
    split_mode: str = "random"  # random | holdout_high_digits | holdout_op_pair
    holdout_high_digits: int = 0
    holdout_op1: str = "*"
    holdout_op2: str = "+"
    experiment_id: str = "TM-EXP-01"
    intact_acc_out_min: float = 0.90
    delta_shuffle_out_min: float = 0.40
    delta_drop_out_min: float = 0.40
    seed_stability_std_max: float = 0.05


class ReasonerB(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.pos = nn.Parameter(torch.randn(1, 5, cfg.d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.proj = nn.Sequential(
            nn.Linear(5 * cfg.d_model, cfg.dim_ff),
            nn.ReLU(),
            nn.Linear(cfg.dim_ff, cfg.d_model),
            nn.ReLU(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.dim_ff),
            nn.ReLU(),
            nn.Linear(cfg.dim_ff, cfg.d_model),
            nn.ReLU(),
        )
        self.head_mid = nn.Linear(cfg.d_model, vocab_size)
        self.head_out = nn.Linear(cfg.d_model, vocab_size)

    def forward(
        self,
        problem_ids: torch.Tensor,
        mid_tokens_for_out: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.embed(problem_ids) + self.pos
        h = self.encoder(h)
        flat = h.reshape(h.size(0), -1)
        z = self.proj(flat)
        mid_logits = self.head_mid(z)

        if self.cfg.out_condition_on_mid:
            if mid_tokens_for_out is None:
                mid_tokens_for_out = torch.argmax(mid_logits, dim=-1)
            mid_emb = self.embed(mid_tokens_for_out)
            z_out = self.out_proj(torch.cat([z, mid_emb], dim=-1))
            out_logits = self.head_out(z_out)
        else:
            out_logits = self.head_out(z)
        return mid_logits, out_logits


def evaluate_setup(rows_by_split: Dict[str, List[Dict[str, object]]]) -> Dict[str, object]:
    # Data split disjointness and parser compatibility checks.
    split_keys = {name: {r["key"] for r in rows} for name, rows in rows_by_split.items()}
    overlap_train_val = len(split_keys["train"].intersection(split_keys["val"]))
    overlap_train_test = len(split_keys["train"].intersection(split_keys["test"]))
    overlap_val_test = len(split_keys["val"].intersection(split_keys["test"]))

    parser_ok = 0
    target_ok = 0
    total = 0
    for rows in rows_by_split.values():
        for r in rows:
            total += 1
            parsed = parse_text_to_la_problem(r["text"])
            parser_ok += int(parsed == r["problem_ids"])
            p = r["problem_ids"]
            a = int(ID2TOK[p[0]])
            op1 = ID2TOK[p[1]]
            b = int(ID2TOK[p[2]])
            op2 = ID2TOK[p[3]]
            c = int(ID2TOK[p[4]])
            mid = apply_op(a, op1, b)
            out = apply_op(mid, op2, c)
            expected = [TOK2ID[str(mid)], TOK2ID[str(out)]]
            target_ok += int(expected == r["solution_ids"])

    train_problem_tokens = {tok for r in rows_by_split["train"] for tok in r["problem_ids"]}
    test_problem_tokens = {tok for r in rows_by_split["test"] for tok in r["problem_ids"]}
    train_solution_tokens = {tok for r in rows_by_split["train"] for tok in r["solution_ids"]}
    test_solution_tokens = {tok for r in rows_by_split["test"] for tok in r["solution_ids"]}
    missing_problem_tokens = sorted(test_problem_tokens - train_problem_tokens)
    missing_solution_tokens = sorted(test_solution_tokens - train_solution_tokens)

    report = {
        "rows_total": total,
        "split_sizes": {k: len(v) for k, v in rows_by_split.items()},
        "split_overlap": {
            "train_val": overlap_train_val,
            "train_test": overlap_train_test,
            "val_test": overlap_val_test,
        },
        "parser_exact_match": parser_ok / max(1, total),
        "target_exact_match": target_ok / max(1, total),
        "vocab_size": len(VOCAB),
        "token_coverage": {
            "train_problem_tokens": len(train_problem_tokens),
            "test_problem_tokens": len(test_problem_tokens),
            "missing_problem_tokens_in_train_count": len(missing_problem_tokens),
            "missing_problem_tokens_in_train": [ID2TOK[t] for t in missing_problem_tokens[:32]],
            "train_solution_tokens": len(train_solution_tokens),
            "test_solution_tokens": len(test_solution_tokens),
            "missing_solution_tokens_in_train_count": len(missing_solution_tokens),
            "missing_solution_tokens_in_train": [ID2TOK[t] for t in missing_solution_tokens[:32]],
        },
    }
    report["valid"] = (
        overlap_train_val == 0
        and overlap_train_test == 0
        and overlap_val_test == 0
        and report["parser_exact_match"] == 1.0
        and report["target_exact_match"] == 1.0
    )
    return report


def one_epoch_train(model: ReasonerB, loader: DataLoader, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    model.train()
    ce = nn.CrossEntropyLoss()
    loss_sum = 0.0
    acc_mid_sum = 0.0
    acc_out_sum = 0.0
    n = 0
    for batch in loader:
        x = batch["problem_ids"].to(DEVICE)
        y = batch["solution_ids"].to(DEVICE)
        mid_logits, _ = model(x)
        mid_gold = y[:, 0]
        _, out_logits = model(x, mid_tokens_for_out=mid_gold)
        loss_mid = ce(mid_logits, y[:, 0])
        loss_out = ce(out_logits, y[:, 1])
        loss = loss_mid + loss_out
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pred_mid = mid_logits.argmax(dim=-1)
        pred_out = out_logits.argmax(dim=-1)
        acc_mid = (pred_mid == y[:, 0]).float().mean().item()
        acc_out = (pred_out == y[:, 1]).float().mean().item()
        loss_sum += float(loss.item())
        acc_mid_sum += acc_mid
        acc_out_sum += acc_out
        n += 1
    return {
        "loss": loss_sum / max(1, n),
        "acc_mid": acc_mid_sum / max(1, n),
        "acc_out": acc_out_sum / max(1, n),
    }


def run_pipeline_eval(
    model: ReasonerB,
    loader: DataLoader,
    mode: str = "intact",
    rng_seed: int = 0,
    max_samples: int = 12,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    assert mode in {"intact", "shuffle", "drop", "random_b"}
    model.eval()
    rng = random.Random(rng_seed)
    samples: List[Dict[str, object]] = []
    total = 0
    correct_out = 0
    correct_mid = 0
    group_total: Dict[str, int] = {}
    group_correct_out: Dict[str, int] = {}
    with torch.no_grad():
        for batch in loader:
            # A module: parse text -> LA(problem)
            parsed_ids = [parse_text_to_la_problem(t) for t in batch["text"]]
            clean_x = torch.tensor(parsed_ids, dtype=torch.long, device=DEVICE)
            x = torch.tensor(parsed_ids, dtype=torch.long, device=DEVICE)
            y = batch["solution_ids"].to(DEVICE)

            if mode == "shuffle":
                x_shuf = []
                for row in x.tolist():
                    idx = [0, 1, 2, 3, 4]
                    rng.shuffle(idx)
                    x_shuf.append([row[i] for i in idx])
                x = torch.tensor(x_shuf, dtype=torch.long, device=DEVICE)
            elif mode == "drop":
                x = torch.full_like(x, MASK_ID)

            mid_logits, out_logits = model(x)
            pred_mid = mid_logits.argmax(dim=-1)
            pred_out = out_logits.argmax(dim=-1)

            total += x.size(0)
            correct_mid += (pred_mid == y[:, 0]).sum().item()
            correct_out += (pred_out == y[:, 1]).sum().item()
            for i in range(x.size(0)):
                op1 = ID2TOK[int(clean_x[i, 1].item())]
                op2 = ID2TOK[int(clean_x[i, 3].item())]
                g = f"{op1}->{op2}"
                group_total[g] = group_total.get(g, 0) + 1
                group_correct_out[g] = group_correct_out.get(g, 0) + int(pred_out[i].item() == y[i, 1].item())

            if len(samples) < max_samples:
                for i in range(min(x.size(0), max_samples - len(samples))):
                    # C module: LA(solution) -> text answer
                    pred_text = la_solution_to_text([int(pred_mid[i].item()), int(pred_out[i].item())])
                    samples.append(
                        {
                            "mode": mode,
                            "key": batch["key"][i],
                            "text": batch["text"][i],
                            "target_answer": batch["answer_text"][i],
                            "pred_answer": pred_text,
                            "target_solution_tokens": [
                                ID2TOK[int(y[i, 0].item())],
                                ID2TOK[int(y[i, 1].item())],
                            ],
                            "pred_solution_tokens": [
                                ID2TOK[int(pred_mid[i].item())],
                                ID2TOK[int(pred_out[i].item())],
                            ],
                        }
                    )

    return {
        "acc_mid": correct_mid / max(1, total),
        "acc_out": correct_out / max(1, total),
        "n": total,
        "acc_out_by_op_pair": {
            k: group_correct_out[k] / group_total[k] for k in sorted(group_total.keys())
        },
    }, samples


def parse_seeds(seeds_str: str) -> List[int]:
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


def run_single_seed(
    rows_by_split: Dict[str, List[Dict[str, object]]],
    cfg: Config,
    seed: int,
    run_dir: str,
) -> Dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_ds = LADataset(rows_by_split["train"])
    val_ds = LADataset(rows_by_split["val"])
    test_ds = LADataset(rows_by_split["test"])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    model = ReasonerB(len(VOCAB), cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    metrics_path = os.path.join(run_dir, f"seed_{seed}_train_metrics.jsonl")
    with open(metrics_path, "w", encoding="utf-8") as f:
        for epoch in range(1, cfg.epochs + 1):
            train_metrics = one_epoch_train(model, train_loader, optimizer)
            val_metrics, _ = run_pipeline_eval(model, val_loader, mode="intact", rng_seed=seed + epoch)
            row = {
                "epoch": epoch,
                "seed": seed,
                "train_loss": train_metrics["loss"],
                "train_acc_mid": train_metrics["acc_mid"],
                "train_acc_out": train_metrics["acc_out"],
                "val_acc_mid": val_metrics["acc_mid"],
                "val_acc_out": val_metrics["acc_out"],
            }
            f.write(json.dumps(row) + "\n")

    test_intact, samples_intact = run_pipeline_eval(model, test_loader, mode="intact", rng_seed=seed)
    test_shuffle, _ = run_pipeline_eval(model, test_loader, mode="shuffle", rng_seed=seed + 17)
    test_drop, _ = run_pipeline_eval(model, test_loader, mode="drop", rng_seed=seed + 31)

    random_b = ReasonerB(len(VOCAB), cfg).to(DEVICE)
    test_random_b, _ = run_pipeline_eval(random_b, test_loader, mode="random_b", rng_seed=seed + 47)

    with open(os.path.join(run_dir, f"seed_{seed}_samples_intact.jsonl"), "w", encoding="utf-8") as f:
        for row in samples_intact:
            f.write(json.dumps(row) + "\n")

    summary = {
        "seed": seed,
        "test_intact": test_intact,
        "test_shuffle": test_shuffle,
        "test_drop": test_drop,
        "test_random_b": test_random_b,
        "delta_shuffle_out": test_intact["acc_out"] - test_shuffle["acc_out"],
        "delta_drop_out": test_intact["acc_out"] - test_drop["acc_out"],
        "delta_random_b_out": test_intact["acc_out"] - test_random_b["acc_out"],
    }
    with open(os.path.join(run_dir, f"seed_{seed}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def aggregate_seed_results(seed_summaries: Sequence[Dict[str, object]]) -> Dict[str, float]:
    intact = [s["test_intact"]["acc_out"] for s in seed_summaries]
    shuffle = [s["test_shuffle"]["acc_out"] for s in seed_summaries]
    drop = [s["test_drop"]["acc_out"] for s in seed_summaries]
    random_b = [s["test_random_b"]["acc_out"] for s in seed_summaries]
    return {
        "seeds": len(seed_summaries),
        "intact_mean": mean(intact),
        "intact_std": pstdev(intact) if len(intact) > 1 else 0.0,
        "shuffle_mean": mean(shuffle),
        "drop_mean": mean(drop),
        "random_b_mean": mean(random_b),
        "delta_shuffle_mean": mean([a - b for a, b in zip(intact, shuffle)]),
        "delta_drop_mean": mean([a - b for a, b in zip(intact, drop)]),
        "delta_random_b_mean": mean([a - b for a, b in zip(intact, random_b)]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TM-EXP-01 local reasoner experiment")
    parser.add_argument("--d_model", type=int, default=Config.d_model)
    parser.add_argument("--nhead", type=int, default=Config.nhead)
    parser.add_argument("--num_layers", type=int, default=Config.num_layers)
    parser.add_argument("--dim_ff", type=int, default=Config.dim_ff)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--train_ratio", type=float, default=Config.train_ratio)
    parser.add_argument("--val_ratio", type=float, default=Config.val_ratio)
    parser.add_argument("--num_min", type=int, default=Config.num_min)
    parser.add_argument("--num_max", type=int, default=Config.num_max)
    parser.add_argument("--mul_cap", type=int, default=Config.mul_cap)
    parser.add_argument("--split_seed", type=int, default=Config.split_seed)
    parser.add_argument("--template_seed", type=int, default=Config.template_seed)
    parser.add_argument("--seeds", type=str, default=Config.seeds)
    parser.add_argument("--output_root", type=str, default=Config.output_root)
    parser.add_argument(
        "--split_mode",
        type=str,
        default=Config.split_mode,
        choices=["random", "holdout_high_digits", "holdout_op_pair"],
    )
    parser.add_argument("--holdout_high_digits", type=int, default=Config.holdout_high_digits)
    parser.add_argument("--holdout_op1", type=str, default=Config.holdout_op1, choices=OPS)
    parser.add_argument("--holdout_op2", type=str, default=Config.holdout_op2, choices=OPS)
    parser.add_argument("--experiment_id", type=str, default=Config.experiment_id)
    parser.add_argument("--intact_acc_out_min", type=float, default=Config.intact_acc_out_min)
    parser.add_argument("--delta_shuffle_out_min", type=float, default=Config.delta_shuffle_out_min)
    parser.add_argument("--delta_drop_out_min", type=float, default=Config.delta_drop_out_min)
    parser.add_argument("--seed_stability_std_max", type=float, default=Config.seed_stability_std_max)
    parser.add_argument(
        "--out_condition_on_mid",
        type=int,
        default=1 if Config.out_condition_on_mid else 0,
        help="1 to condition final-state head on mid-state token, 0 for independent heads",
    )
    args = parser.parse_args()
    args.out_condition_on_mid = bool(args.out_condition_on_mid)

    cfg = Config(**vars(args))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = cfg.experiment_id.lower().replace("-", "_")
    run_dir = os.path.join(cfg.output_root, f"{run_tag}_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    prereg = {
        "experiment_id": cfg.experiment_id,
        "hypothesis": (
            "A(text->LA) + B(LA->LA) + C(LA->text) solves compositional arithmetic "
            "through the LA pivot, and perturbing LA should causally degrade performance."
        ),
        "constraints": [
            "local-only run",
            "no large artifacts/checkpoints",
            "disjoint expression split",
            "deterministic A/C, learned B",
        ],
        "acceptance_criteria": {
            "intact_acc_out_min": cfg.intact_acc_out_min,
            "delta_shuffle_out_min": cfg.delta_shuffle_out_min,
            "delta_drop_out_min": cfg.delta_drop_out_min,
            "seed_stability_std_max": cfg.seed_stability_std_max,
        },
        "dataset": {
            "type": "synthetic_generated_in_script",
            "expression_form": "((a op1 b) op2 c)",
            "num_range": [cfg.num_min, cfg.num_max],
            "ops": OPS,
            "mul_cap": cfg.mul_cap,
            "split_seed": cfg.split_seed,
            "template_seed": cfg.template_seed,
            "split_mode": cfg.split_mode,
            "holdout_high_digits": cfg.holdout_high_digits,
            "holdout_op1": cfg.holdout_op1,
            "holdout_op2": cfg.holdout_op2,
        },
        "config": asdict(cfg),
        "device": DEVICE,
        "timestamp": ts,
    }
    with open(os.path.join(run_dir, "preregistered_plan.json"), "w", encoding="utf-8") as f:
        json.dump(prereg, f, indent=2)

    exprs = build_expression_space(cfg.num_min, cfg.num_max, cfg.mul_cap)
    splits = make_splits(
        exprs,
        cfg.split_seed,
        cfg.train_ratio,
        cfg.val_ratio,
        split_mode=cfg.split_mode,
        holdout_high_digits=cfg.holdout_high_digits,
        num_max=cfg.num_max,
        holdout_op1=cfg.holdout_op1,
        holdout_op2=cfg.holdout_op2,
    )
    rows_by_split = {
        split_name: build_samples(split_exprs, cfg.template_seed + i)
        for i, (split_name, split_exprs) in enumerate(splits.items())
    }
    setup_report = evaluate_setup(rows_by_split)
    with open(os.path.join(run_dir, "setup_validation.json"), "w", encoding="utf-8") as f:
        json.dump(setup_report, f, indent=2)
    if not setup_report["valid"]:
        raise RuntimeError(f"Setup validation failed: {json.dumps(setup_report, indent=2)}")

    seed_summaries: List[Dict[str, object]] = []
    for seed in parse_seeds(cfg.seeds):
        summary = run_single_seed(rows_by_split, cfg, seed, run_dir)
        seed_summaries.append(summary)

    aggregate = aggregate_seed_results(seed_summaries)
    acceptance = prereg["acceptance_criteria"]
    verdict = {
        "intact_ok": aggregate["intact_mean"] >= acceptance["intact_acc_out_min"],
        "shuffle_ok": aggregate["delta_shuffle_mean"] >= acceptance["delta_shuffle_out_min"],
        "drop_ok": aggregate["delta_drop_mean"] >= acceptance["delta_drop_out_min"],
        "stability_ok": aggregate["intact_std"] <= acceptance["seed_stability_std_max"],
    }
    verdict["green"] = all(verdict.values())

    with open(os.path.join(run_dir, "seed_summaries.json"), "w", encoding="utf-8") as f:
        json.dump(seed_summaries, f, indent=2)
    with open(os.path.join(run_dir, "aggregate_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"aggregate": aggregate, "verdict": verdict}, f, indent=2)

    print(json.dumps({"run_dir": run_dir, "aggregate": aggregate, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
