"""
TM-EXP-04 (local): LA-pivot vs text-only baseline comparison.

Goal:
- Produce early, rigorous evidence on whether the two-model LA architecture
  can outperform a matched text-only CoT-style proxy on the same data/splits.

Protocol:
- Shared dataset/splits/seeds.
- Setup validation before interpretation.
- Multi-seed reporting.
- Causal controls for LA model (shuffle/drop/random_b).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict
from datetime import datetime
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from two_model_architecture.tm_exp01_reasoner_local import (
        Config as LAConfig,
        DEVICE,
        ID2TOK,
        MASK_ID,
        OPS,
        PAD_ID,
        ReasonerB,
        apply_op,
        build_expression_space,
        build_samples,
        evaluate_setup,
        la_solution_to_text,
        make_splits,
        parse_seeds,
        parse_text_to_la_problem,
    )
except ModuleNotFoundError:
    # Support direct execution as "python two_model_architecture/tm_exp04_compare_local.py".
    from tm_exp01_reasoner_local import (  # type: ignore
        Config as LAConfig,
        DEVICE,
        ID2TOK,
        MASK_ID,
        OPS,
        PAD_ID,
        ReasonerB,
        apply_op,
        build_expression_space,
        build_samples,
        evaluate_setup,
        la_solution_to_text,
        make_splits,
        parse_seeds,
        parse_text_to_la_problem,
    )


class JointDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, object]], tokenizer: "TextTokenizer"):
        self.rows = list(rows)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        r = self.rows[idx]
        return {
            "problem_ids": r["problem_ids"],
            "solution_ids": r["solution_ids"],
            "text": r["text"],
            "text_ids": self.tokenizer.encode(r["text"]),
            "key": r["key"],
            "answer_text": r["answer_text"],
        }


def enforce_test_solution_token_coverage(
    rows_by_split: Dict[str, List[Dict[str, object]]],
) -> Dict[str, object]:
    train_solution_tokens = {tok for r in rows_by_split["train"] for tok in r["solution_ids"]}
    test_rows = rows_by_split["test"]
    test_total = len(test_rows)

    missing_before = sorted({tok for r in test_rows for tok in r["solution_ids"] if tok not in train_solution_tokens})
    filtered_test = [r for r in test_rows if all(tok in train_solution_tokens for tok in r["solution_ids"])]
    removed = test_total - len(filtered_test)
    missing_after = sorted({tok for r in filtered_test for tok in r["solution_ids"] if tok not in train_solution_tokens})

    rows_by_split["test"] = filtered_test
    return {
        "applied": True,
        "test_total_before": test_total,
        "test_total_after": len(filtered_test),
        "removed_count": removed,
        "removed_fraction": (removed / test_total) if test_total else 0.0,
        "missing_solution_tokens_before": [ID2TOK[t] for t in missing_before],
        "missing_solution_tokens_after": [ID2TOK[t] for t in missing_after],
    }


def collate_joint(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    problem = torch.tensor([b["problem_ids"] for b in batch], dtype=torch.long)
    solution = torch.tensor([b["solution_ids"] for b in batch], dtype=torch.long)

    max_text = max(len(b["text_ids"]) for b in batch)
    text_ids = torch.full((len(batch), max_text), fill_value=0, dtype=torch.long)
    for i, b in enumerate(batch):
        ids = b["text_ids"]
        text_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    return {
        "problem_ids": problem,
        "solution_ids": solution,
        "text_ids": text_ids,
        "text": [b["text"] for b in batch],
        "key": [b["key"] for b in batch],
        "answer_text": [b["answer_text"] for b in batch],
    }


class TextTokenizer:
    def __init__(self, train_rows: Sequence[Dict[str, object]]):
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.mask = "<mask>"
        vocab = [self.pad, self.unk, self.mask]

        def tok(s: str) -> List[str]:
            return re.findall(r"-?\d+|[+\-*]|[a-z]+", s.lower())

        seen = set()
        for r in train_rows:
            for t in tok(r["text"]):
                if t not in seen:
                    seen.add(t)
                    vocab.append(t)

        self.vocab = vocab
        self.tok2id = {t: i for i, t in enumerate(vocab)}
        self.id2tok = {i: t for t, i in self.tok2id.items()}
        self._tok_fn = tok

    @property
    def pad_id(self) -> int:
        return self.tok2id[self.pad]

    @property
    def mask_id(self) -> int:
        return self.tok2id[self.mask]

    def encode(self, text: str) -> List[int]:
        return [self.tok2id.get(t, self.tok2id[self.unk]) for t in self._tok_fn(text)]


class TextReasoner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        out_vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, 32, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
            nn.ReLU(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(2 * d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
            nn.ReLU(),
        )
        self.mid_emb = nn.Embedding(out_vocab_size, d_model)
        self.head_mid = nn.Linear(d_model, out_vocab_size)
        self.head_out = nn.Linear(d_model, out_vocab_size)

    def forward(
        self,
        text_ids: torch.Tensor,
        mid_tokens_for_out: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        l = text_ids.size(1)
        if l > self.pos.size(1):
            raise ValueError(f"text length {l} exceeds max position size {self.pos.size(1)}")
        h = self.embed(text_ids) + self.pos[:, :l, :]
        h = self.encoder(h)

        # mean pooling with pad-aware mask (pad id is 0)
        mask = (text_ids != 0).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (h * mask).sum(dim=1) / denom

        z = self.proj(pooled)
        mid_logits = self.head_mid(z)
        if mid_tokens_for_out is None:
            mid_tokens_for_out = torch.argmax(mid_logits, dim=-1)
        out_in = torch.cat([z, self.mid_emb(mid_tokens_for_out)], dim=-1)
        out_logits = self.head_out(self.out_proj(out_in))
        return mid_logits, out_logits


def train_epoch_la(model: ReasonerB, loader: DataLoader, opt: torch.optim.Optimizer) -> Dict[str, float]:
    model.train()
    ce = nn.CrossEntropyLoss()
    n = 0
    loss_sum = 0.0
    acc_out = 0.0
    for batch in loader:
        x = batch["problem_ids"].to(DEVICE)
        y = batch["solution_ids"].to(DEVICE)
        mid_logits, _ = model(x)
        _, out_logits = model(x, mid_tokens_for_out=y[:, 0])
        loss = ce(mid_logits, y[:, 0]) + ce(out_logits, y[:, 1])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        acc_out += (out_logits.argmax(-1) == y[:, 1]).float().mean().item()
        loss_sum += float(loss.item())
        n += 1
    return {"loss": loss_sum / max(1, n), "acc_out": acc_out / max(1, n)}


def train_epoch_text(model: TextReasoner, loader: DataLoader, opt: torch.optim.Optimizer) -> Dict[str, float]:
    model.train()
    ce = nn.CrossEntropyLoss()
    n = 0
    loss_sum = 0.0
    acc_out = 0.0
    for batch in loader:
        x = batch["text_ids"].to(DEVICE)
        y = batch["solution_ids"].to(DEVICE)
        mid_logits, _ = model(x)
        _, out_logits = model(x, mid_tokens_for_out=y[:, 0])
        loss = ce(mid_logits, y[:, 0]) + ce(out_logits, y[:, 1])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        acc_out += (out_logits.argmax(-1) == y[:, 1]).float().mean().item()
        loss_sum += float(loss.item())
        n += 1
    return {"loss": loss_sum / max(1, n), "acc_out": acc_out / max(1, n)}


def eval_la(
    model: ReasonerB,
    loader: DataLoader,
    mode: str,
    rng_seed: int,
    max_samples: int = 8,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    assert mode in {"intact", "shuffle", "drop", "random_b"}
    rng = random.Random(rng_seed)
    model.eval()
    total = 0
    correct_out = 0
    samples: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            parsed_ids = [parse_text_to_la_problem(t) for t in batch["text"]]
            x = torch.tensor(parsed_ids, dtype=torch.long, device=DEVICE)
            y = batch["solution_ids"].to(DEVICE)
            clean_x = x.clone()

            if mode == "shuffle":
                buf = []
                for row in x.tolist():
                    idx = [0, 1, 2, 3, 4]
                    rng.shuffle(idx)
                    buf.append([row[i] for i in idx])
                x = torch.tensor(buf, dtype=torch.long, device=DEVICE)
            elif mode == "drop":
                x = torch.full_like(x, MASK_ID)

            mid_logits, out_logits = model(x)
            pred_mid = mid_logits.argmax(-1)
            pred_out = out_logits.argmax(-1)
            total += x.size(0)
            correct_out += (pred_out == y[:, 1]).sum().item()

            if len(samples) < max_samples:
                for i in range(min(x.size(0), max_samples - len(samples))):
                    samples.append(
                        {
                            "model": "la",
                            "mode": mode,
                            "text": batch["text"][i],
                            "problem_clean": [ID2TOK[int(t)] for t in clean_x[i].tolist()],
                            "target_answer": batch["answer_text"][i],
                            "pred_answer": la_solution_to_text([int(pred_mid[i].item()), int(pred_out[i].item())]),
                        }
                    )
    return {"acc_out": correct_out / max(1, total), "n": total}, samples


def eval_text(
    model: TextReasoner,
    loader: DataLoader,
    tokenizer: TextTokenizer,
    mode: str,
    rng_seed: int,
    max_samples: int = 8,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    assert mode in {"intact", "shuffle", "drop", "random_text"}
    rng = random.Random(rng_seed)
    model.eval()
    total = 0
    correct_out = 0
    samples: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["text_ids"].to(DEVICE)
            y = batch["solution_ids"].to(DEVICE)

            if mode == "shuffle":
                x_s = []
                for row in x.tolist():
                    row2 = list(row)
                    non_pad = [i for i, t in enumerate(row2) if t != tokenizer.pad_id]
                    vals = [row2[i] for i in non_pad]
                    rng.shuffle(vals)
                    for idx, v in zip(non_pad, vals):
                        row2[idx] = v
                    x_s.append(row2)
                x = torch.tensor(x_s, dtype=torch.long, device=DEVICE)
            elif mode == "drop":
                x = torch.where(x == tokenizer.pad_id, x, torch.full_like(x, tokenizer.mask_id))

            mid_logits, out_logits = model(x)
            pred_mid = mid_logits.argmax(-1)
            pred_out = out_logits.argmax(-1)
            total += x.size(0)
            correct_out += (pred_out == y[:, 1]).sum().item()

            if len(samples) < max_samples:
                for i in range(min(x.size(0), max_samples - len(samples))):
                    samples.append(
                        {
                            "model": "text",
                            "mode": mode,
                            "text": batch["text"][i],
                            "target_answer": batch["answer_text"][i],
                            "pred_answer": ID2TOK[int(pred_out[i].item())],
                        }
                    )
    return {"acc_out": correct_out / max(1, total), "n": total}, samples


def run_seed(
    rows_by_split: Dict[str, List[Dict[str, object]]],
    cfg: LAConfig,
    seed: int,
    run_dir: str,
) -> Dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = TextTokenizer(rows_by_split["train"])
    train_ds = JointDataset(rows_by_split["train"], tokenizer)
    val_ds = JointDataset(rows_by_split["val"], tokenizer)
    test_ds = JointDataset(rows_by_split["test"], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_joint)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_joint)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_joint)

    la = ReasonerB(len(ID2TOK), cfg).to(DEVICE)
    text = TextReasoner(
        vocab_size=len(tokenizer.vocab),
        out_vocab_size=len(ID2TOK),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_ff=cfg.dim_ff,
        dropout=cfg.dropout,
    ).to(DEVICE)
    opt_la = torch.optim.AdamW(la.parameters(), lr=cfg.lr)
    opt_text = torch.optim.AdamW(text.parameters(), lr=cfg.lr)

    metrics_path = os.path.join(run_dir, f"seed_{seed}_train_metrics.jsonl")
    with open(metrics_path, "w", encoding="utf-8") as f:
        for epoch in range(1, cfg.epochs + 1):
            tr_la = train_epoch_la(la, train_loader, opt_la)
            tr_text = train_epoch_text(text, train_loader, opt_text)
            va_la, _ = eval_la(la, val_loader, mode="intact", rng_seed=seed + epoch)
            va_text, _ = eval_text(text, val_loader, tokenizer, mode="intact", rng_seed=seed + epoch)
            row = {
                "epoch": epoch,
                "seed": seed,
                "train_la_loss": tr_la["loss"],
                "train_la_acc_out": tr_la["acc_out"],
                "val_la_acc_out": va_la["acc_out"],
                "train_text_loss": tr_text["loss"],
                "train_text_acc_out": tr_text["acc_out"],
                "val_text_acc_out": va_text["acc_out"],
            }
            f.write(json.dumps(row) + "\n")

    la_intact, la_samples = eval_la(la, test_loader, mode="intact", rng_seed=seed)
    la_shuffle, _ = eval_la(la, test_loader, mode="shuffle", rng_seed=seed + 101)
    la_drop, _ = eval_la(la, test_loader, mode="drop", rng_seed=seed + 203)
    la_random = ReasonerB(len(ID2TOK), cfg).to(DEVICE)
    la_random_m, _ = eval_la(la_random, test_loader, mode="random_b", rng_seed=seed + 307)

    text_intact, text_samples = eval_text(text, test_loader, tokenizer, mode="intact", rng_seed=seed)
    text_shuffle, _ = eval_text(text, test_loader, tokenizer, mode="shuffle", rng_seed=seed + 101)
    text_drop, _ = eval_text(text, test_loader, tokenizer, mode="drop", rng_seed=seed + 203)
    text_random = TextReasoner(
        vocab_size=len(tokenizer.vocab),
        out_vocab_size=len(ID2TOK),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_ff=cfg.dim_ff,
        dropout=cfg.dropout,
    ).to(DEVICE)
    text_random_m, _ = eval_text(text_random, test_loader, tokenizer, mode="random_text", rng_seed=seed + 307)

    with open(os.path.join(run_dir, f"seed_{seed}_samples_la.jsonl"), "w", encoding="utf-8") as f:
        for r in la_samples:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(run_dir, f"seed_{seed}_samples_text.jsonl"), "w", encoding="utf-8") as f:
        for r in text_samples:
            f.write(json.dumps(r) + "\n")

    summary = {
        "seed": seed,
        "la": {
            "intact": la_intact,
            "shuffle": la_shuffle,
            "drop": la_drop,
            "random": la_random_m,
            "delta_shuffle": la_intact["acc_out"] - la_shuffle["acc_out"],
            "delta_drop": la_intact["acc_out"] - la_drop["acc_out"],
        },
        "text": {
            "intact": text_intact,
            "shuffle": text_shuffle,
            "drop": text_drop,
            "random": text_random_m,
            "delta_shuffle": text_intact["acc_out"] - text_shuffle["acc_out"],
            "delta_drop": text_intact["acc_out"] - text_drop["acc_out"],
        },
        "gain_la_minus_text": la_intact["acc_out"] - text_intact["acc_out"],
    }
    with open(os.path.join(run_dir, f"seed_{seed}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def aggregate(seed_summaries: Sequence[Dict[str, object]]) -> Dict[str, float]:
    la = [s["la"]["intact"]["acc_out"] for s in seed_summaries]
    text = [s["text"]["intact"]["acc_out"] for s in seed_summaries]
    gains = [s["gain_la_minus_text"] for s in seed_summaries]
    la_dsh = [s["la"]["delta_shuffle"] for s in seed_summaries]
    la_ddr = [s["la"]["delta_drop"] for s in seed_summaries]
    return {
        "seeds": len(seed_summaries),
        "la_intact_mean": mean(la),
        "la_intact_std": pstdev(la) if len(la) > 1 else 0.0,
        "text_intact_mean": mean(text),
        "text_intact_std": pstdev(text) if len(text) > 1 else 0.0,
        "gain_mean": mean(gains),
        "gain_std": pstdev(gains) if len(gains) > 1 else 0.0,
        "la_delta_shuffle_mean": mean(la_dsh),
        "la_delta_drop_mean": mean(la_ddr),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TM-EXP-04 LA vs text-only comparison")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--num_min", type=int, default=0)
    parser.add_argument("--num_max", type=int, default=9)
    parser.add_argument("--mul_cap", type=int, default=5)
    parser.add_argument("--split_seed", type=int, default=20260227)
    parser.add_argument("--template_seed", type=int, default=1337)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--split_mode", type=str, default="holdout_op_pair", choices=["random", "holdout_op_pair"])
    parser.add_argument("--holdout_op1", type=str, default="*", choices=OPS)
    parser.add_argument("--holdout_op2", type=str, default="+", choices=OPS)
    parser.add_argument("--output_root", type=str, default="two_model_architecture/results")
    parser.add_argument(
        "--enforce_test_solution_coverage",
        type=int,
        default=1,
        help="1 removes test rows whose solution tokens are unseen in train; 0 keeps raw split",
    )

    # preregistered comparison criteria
    parser.add_argument("--gain_min", type=float, default=0.03)
    parser.add_argument("--la_delta_shuffle_min", type=float, default=0.20)
    parser.add_argument("--la_delta_drop_min", type=float, default=0.20)
    args = parser.parse_args()
    args.enforce_test_solution_coverage = bool(args.enforce_test_solution_coverage)

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
    run_dir = os.path.join(args.output_root, f"tm_exp_04_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    prereg = {
        "experiment_id": "TM-EXP-04",
        "hypothesis": "LA-pivot model outperforms matched text-only baseline on held-out composition.",
        "comparison": "same dataset, split, seeds, and training budget",
        "acceptance_criteria": {
            "gain_min": args.gain_min,
            "la_delta_shuffle_min": args.la_delta_shuffle_min,
            "la_delta_drop_min": args.la_delta_drop_min,
        },
        "split": {
            "mode": args.split_mode,
            "holdout_op1": args.holdout_op1,
            "holdout_op2": args.holdout_op2,
            "split_seed": args.split_seed,
            "enforce_test_solution_coverage": args.enforce_test_solution_coverage,
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
        split_mode=args.split_mode,
        num_max=cfg.num_max,
        holdout_op1=args.holdout_op1,
        holdout_op2=args.holdout_op2,
    )
    rows_by_split = {
        split_name: build_samples(split_exprs, cfg.template_seed + i)
        for i, (split_name, split_exprs) in enumerate(splits.items())
    }
    setup_before = evaluate_setup(rows_by_split)
    coverage_adjustment = {"applied": False}
    if args.enforce_test_solution_coverage:
        coverage_adjustment = enforce_test_solution_token_coverage(rows_by_split)
    setup = evaluate_setup(rows_by_split)
    with open(os.path.join(run_dir, "setup_validation.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "setup_before_adjustment": setup_before,
                "coverage_adjustment": coverage_adjustment,
                "setup_after_adjustment": setup,
            },
            f,
            indent=2,
        )
    if not setup["valid"]:
        raise RuntimeError(f"setup invalid: {json.dumps(setup, indent=2)}")
    if len(rows_by_split["test"]) == 0:
        raise RuntimeError("setup invalid: test split empty after coverage adjustment")

    seed_summaries: List[Dict[str, object]] = []
    for seed in parse_seeds(cfg.seeds):
        seed_summaries.append(run_seed(rows_by_split, cfg, seed, run_dir))

    agg = aggregate(seed_summaries)
    verdict = {
        "gain_ok": agg["gain_mean"] >= args.gain_min,
        "la_causal_shuffle_ok": agg["la_delta_shuffle_mean"] >= args.la_delta_shuffle_min,
        "la_causal_drop_ok": agg["la_delta_drop_mean"] >= args.la_delta_drop_min,
    }
    verdict["green"] = all(verdict.values())

    with open(os.path.join(run_dir, "seed_summaries.json"), "w", encoding="utf-8") as f:
        json.dump(seed_summaries, f, indent=2)
    with open(os.path.join(run_dir, "aggregate_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "verdict": verdict}, f, indent=2)

    print(json.dumps({"run_dir": run_dir, "aggregate": agg, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
