import json
import os
import statistics as stats
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import PredictiveDataset, collate_fn
from model import AirGapVQTransformer


def token_f1(pred_tokens, target_tokens):
    if not pred_tokens or not target_tokens:
        return 0.0
    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    target_counts = {}
    for t in target_tokens:
        target_counts[t] = target_counts.get(t, 0) + 1
    common = 0
    for t, c in pred_counts.items():
        common += min(c, target_counts.get(t, 0))
    prec = common / len(pred_tokens)
    rec = common / len(target_tokens)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def summarize_losses(values):
    if not values:
        return {"mean": None, "median": None, "count": 0}
    return {
        "mean": float(stats.mean(values)),
        "median": float(stats.median(values)),
        "count": len(values),
    }


def load_datasets(base_dir, block_size=256, n_ir_tokens=64):
    train_file = os.path.join(base_dir, "TinyStoriesV2-GPT4-train.txt")
    test_file = os.path.join(base_dir, "TinyStoriesV2-GPT4-valid.txt")

    train_dataset = PredictiveDataset(
        num_samples=200000,
        split="train",
        tiny_stories_path=train_file,
        block_size=block_size,
        n_ir_tokens=n_ir_tokens,
    )
    test_dataset = PredictiveDataset(
        num_samples=2000,
        split="test",
        tiny_stories_path=test_file,
        block_size=block_size,
        n_ir_tokens=n_ir_tokens,
        # Keep train/eval token IDs aligned for diagnostic comparability.
        vocab_tokens=train_dataset.tokens,
    )
    return train_dataset, test_dataset


def load_model(base_dir, vocab_size, block_size=256, n_ir_tokens=64):
    model = AirGapVQTransformer(
        vocab_size=vocab_size,
        num_codes=1024,
        code_dim=384,
        n_ir_tokens=n_ir_tokens,
        reasoner_layers=6,
        reasoner_heads=8,
        speaker_layers=6,
        speaker_heads=8,
        block_size=block_size,
    )
    ckpt_path = os.path.join(base_dir, "results_phase2", "model_final.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model


@torch.no_grad()
def teacher_forced_losses(model, loader, device):
    model.eval()
    loss_bins = defaultdict(lambda: {"speaker": [], "vq": []})

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        tasks = batch["tasks"]

        input_embeds = model.reasoner.transformer.wte(input_ids)
        batch_size = input_ids.shape[0]
        queries = model.thinking_queries.expand(batch_size, -1, -1)
        reasoner_input = torch.cat([input_embeds, queries], dim=1)

        positions = torch.arange(
            reasoner_input.size(1), device=reasoner_input.device
        )
        x = reasoner_input + model.reasoner.transformer.wpe(positions)
        for block in model.reasoner.transformer.h:
            x = block(x)
        x = model.reasoner.transformer.ln_f(x)
        z = x[:, -model.n_ir_tokens :, :]

        z_q, _, _ = model.vq(z, hard=True)

        # Per-sample VQ loss to check whether the bottleneck differs by task.
        vq_loss = (z_q.detach() - z).pow(2).mean(dim=(1, 2))
        vq_loss = vq_loss + model.vq.beta * (z_q - z.detach()).pow(2).mean(dim=(1, 2))

        target_embeds = model.speaker.transformer.wte(target_ids)
        speaker_input = torch.cat([z_q, target_embeds[:, :-1]], dim=1)
        speaker_logits, _ = model.speaker(inputs_embeds=speaker_input)

        relevant_logits = speaker_logits[:, model.n_ir_tokens - 1 :, :]
        per_token_loss = F.cross_entropy(
            relevant_logits.transpose(1, 2),
            target_ids,
            ignore_index=0,
            reduction="none",
        )
        token_mask = target_ids != 0
        token_counts = token_mask.sum(dim=1).clamp(min=1)
        speaker_loss = (per_token_loss * token_mask).sum(dim=1) / token_counts

        for i, task in enumerate(tasks):
            loss_bins[task]["speaker"].append(float(speaker_loss[i]))
            loss_bins[task]["vq"].append(float(vq_loss[i]))

    summary = {}
    for task, bins in loss_bins.items():
        summary[task] = {
            "speaker_loss": summarize_losses(bins["speaker"]),
            "vq_loss": summarize_losses(bins["vq"]),
        }
    return summary


@torch.no_grad()
def generation_metrics(model, loader, device, ir_mode, prefix_ks):
    model.eval()
    eos_id = loader.dataset.eos_id
    max_new = model.speaker_config.block_size - model.n_ir_tokens

    stats_bins = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "f1_sum": 0.0,
        "prefix_f1_sum": {k: 0.0 for k in prefix_ks},
    })

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        tasks = batch["tasks"]
        target_texts = batch["target_texts"]

        pred_ids, _ = model.generate(input_ids, max_new_tokens=max_new, ir_mode=ir_mode)

        for i in range(len(tasks)):
            pred_seq = pred_ids[i].tolist()
            if eos_id in pred_seq:
                pred_seq = pred_seq[: pred_seq.index(eos_id)]
            pred_text = loader.dataset.decode(pred_seq).strip().lower()
            pred_tokens = pred_text.split()

            target_text = target_texts[i].strip().lower()
            target_tokens = target_text.split()

            task = tasks[i]
            stats_bins[task]["total"] += 1

            if task == "math":
                if pred_text == target_text:
                    stats_bins[task]["correct"] += 1
                continue

            f1 = token_f1(pred_tokens, target_tokens)
            stats_bins[task]["f1_sum"] += f1
            for k in prefix_ks:
                f1_k = token_f1(pred_tokens[:k], target_tokens[:k])
                stats_bins[task]["prefix_f1_sum"][k] += f1_k

    summary = {}
    for task, bins in stats_bins.items():
        total = bins["total"]
        if total == 0:
            continue
        if task == "math":
            summary[task] = {"acc": bins["correct"] / total}
        else:
            summary[task] = {
                "f1": bins["f1_sum"] / total,
                "prefix_f1": {k: bins["prefix_f1_sum"][k] / total for k in prefix_ks},
            }
    return summary


def main():
    base_dir = os.path.dirname(__file__)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prefix_ks = (5, 10, 20)
    train_dataset, test_dataset = load_datasets(base_dir)

    loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = load_model(base_dir, vocab_size=len(train_dataset.vocab))
    model.to(device)

    results = {
        "teacher_forced_losses": teacher_forced_losses(model, loader, device),
        "generation_metrics": {},
    }

    for ir_mode in ("intact", "shuffle", "zero"):
        results["generation_metrics"][ir_mode] = generation_metrics(
            model, loader, device, ir_mode, prefix_ks
        )

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
