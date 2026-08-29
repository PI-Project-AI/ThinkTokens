"""
Tiny, local-friendly probe for diffusion-as-IR-corrector.
- Task: toy arithmetic ("a+b=" -> answer). Ground-truth IR is the pair of numbers; draft IR is corrupted with swaps/drops.
- Model: denoiser (Transformer encoder) predicts clean IR; speaker (Transformer decoder) maps clean IR to answer tokens.
- Logging: writes small JSONL metrics and a few samples per epoch to results/.

Run: python air_gap/ir_diffusion_playground/run_diffusion_corrector.py
Designed for CPU / <16GB GPU; defaults tiny.
"""
import json
import os
import random
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Data ----------
VOCAB = ["<pad>", "<bos>", "<eos>", "<sep>"] + [str(i) for i in range(20)] + ["+", "="]  # support sums up to 18
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for t, i in TOK2ID.items()}


def encode(tokens: List[str]) -> List[int]:
    return [TOK2ID[t] for t in tokens]


def decode(ids: List[int]) -> List[str]:
    return [ID2TOK[i] for i in ids]

def scramble(tokens: torch.Tensor) -> torch.Tensor:
    # Shuffle non-pad tokens; keep pad positions.
    tokens_np = tokens.cpu().numpy()
    pad_id = TOK2ID["<pad>"]
    out = []
    for seq in tokens_np:
        non_pad = [t for t in seq if t != pad_id]
        random.shuffle(non_pad)
        non_pad += [pad_id] * (len(seq) - len(non_pad))
        out.append(non_pad)
    return torch.tensor(out, device=tokens.device, dtype=tokens.dtype)


def corrupt_ir(ir_tokens: List[int], drop_p=0.25, swap_p=0.25) -> List[int]:
    out = ir_tokens[:]
    # swap
    if len(out) > 1 and random.random() < swap_p:
        i, j = random.sample(range(len(out)), 2)
        out[i], out[j] = out[j], out[i]
    # drop
    out = [t for t in out if random.random() > drop_p]
    if not out:
        out = ir_tokens[:]  # avoid empty
    return out


class ToyIRDataset(Dataset):
    def __init__(self, n=20000):
        self.samples = []
        for _ in range(n):
            a, b = random.randint(0, 9), random.randint(0, 9)
            input_tokens = [str(a), "+", str(b), "="]
            answer = str(a + b)
            ir = [TOK2ID[str(a)], TOK2ID[str(b)]]
            draft_ir = corrupt_ir(ir)
            self.samples.append(
                {
                    "input": encode(input_tokens),
                    "ir_clean": ir,
                    "ir_noisy": draft_ir,
                    "target": encode([answer]),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    pad = TOK2ID["<pad>"]
    def pad_seq(seq, maxlen):
        return seq + [pad] * (maxlen - len(seq))
    max_in = max(len(b["input"]) for b in batch)
    max_ir_noisy = max(len(b["ir_noisy"]) for b in batch)
    max_ir_clean = max(len(b["ir_clean"]) for b in batch)
    max_tgt = max(len(b["target"]) for b in batch)
    inputs = torch.tensor([pad_seq(b["input"], max_in) for b in batch])
    ir_noisy = torch.tensor([pad_seq(b["ir_noisy"], max_ir_noisy) for b in batch])
    ir_clean = torch.tensor([pad_seq(b["ir_clean"], max_ir_clean) for b in batch])
    targets = torch.tensor([pad_seq(b["target"], max_tgt) for b in batch])
    return inputs, ir_noisy, ir_clean, targets


# ---------- Model ----------
@dataclass
class Config:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 128
    dropout: float = 0.1


class IRCorrector(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.d_model, padding_idx=TOK2ID["<pad>"])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.dim_ff, dropout=cfg.dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.head = nn.Linear(cfg.d_model, vocab_size)

    def forward(self, noisy_ir):
        h = self.embed(noisy_ir)
        h = self.encoder(h)
        return self.head(h)


class Speaker(nn.Module):
    """
    Minimal speaker: pool IR and classify the answer token. This forces reliance on IR
    (no teacher-forcing tokens to leak the target).
    """
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.d_model, padding_idx=TOK2ID["<pad>"])
        self.proj = nn.Linear(cfg.d_model, cfg.dim_ff)
        self.act = nn.ReLU()
        self.head = nn.Linear(cfg.dim_ff, vocab_size)

    def forward(self, ir_clean):
        emb = self.embed(ir_clean)
        # mean pool non-pad
        pad_id = TOK2ID["<pad>"]
        mask = (ir_clean != pad_id).unsqueeze(-1).float()
        summed = (emb * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        h = self.act(self.proj(pooled))
        logits = self.head(h)  # (B, vocab)
        return logits.unsqueeze(1)  # keep time dim 1 for consistency


def accuracy(pred, target, pad):
    with torch.no_grad():
        mask = target != pad
        correct = (pred == target) * mask
        return correct.sum().item() / max(1, mask.sum().item())


def train_epoch(corrector, speaker, data, opt_c, opt_s, pad):
    ce = nn.CrossEntropyLoss(ignore_index=pad)
    corrector.train(); speaker.train()
    pbar = tqdm(data, desc="train", leave=False)
    total = 0; acc_ir = 0; acc_ans = 0
    for inputs, ir_noisy, ir_clean, targets in pbar:
        inputs, ir_noisy, ir_clean, targets = [x.to(DEVICE) for x in (inputs, ir_noisy, ir_clean, targets)]
        # Corrector loss
        logits_ir = corrector(ir_noisy)
        loss_ir = ce(logits_ir.view(-1, logits_ir.size(-1)), ir_clean.view(-1))
        # Speaker: teacher-forced
        logits_ans = speaker(ir_clean)
        loss_ans = ce(logits_ans.view(-1, logits_ans.size(-1)), targets.view(-1))
        loss = loss_ir + loss_ans
        opt_c.zero_grad(); opt_s.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(speaker.parameters(), 1.0)
        opt_c.step(); opt_s.step()
        # metrics
        pred_ir = logits_ir.argmax(-1)
        pred_ans = logits_ans.argmax(-1)
        acc_ir += accuracy(pred_ir, ir_clean, pad)
        acc_ans += accuracy(pred_ans, targets, pad)
        total += 1
        pbar.set_postfix(loss=f"{loss.item():.3f}", ir_acc=f"{acc_ir/total:.3f}", ans_acc=f"{acc_ans/total:.3f}")
    return acc_ir/total, acc_ans/total


def eval_epoch(corrector, speaker, data, pad, max_samples=4):
    ce = nn.CrossEntropyLoss(ignore_index=pad)
    corrector.eval(); speaker.eval()
    total = 0; acc_ir = 0; acc_ans = 0; acc_ans_scramble = 0; samples = []
    with torch.no_grad():
        for inputs, ir_noisy, ir_clean, targets in data:
            inputs, ir_noisy, ir_clean, targets = [x.to(DEVICE) for x in (inputs, ir_noisy, ir_clean, targets)]
            logits_ir = corrector(ir_noisy)
            logits_ans = speaker(ir_clean)
            pred_ir = logits_ir.argmax(-1)
            pred_ans = logits_ans.argmax(-1)
            # Air-gap sanity: scramble predicted IR and see if speaker fails.
            ir_scrambled = scramble(pred_ir)
            logits_ans_scramble = speaker(ir_scrambled)
            pred_ans_scramble = logits_ans_scramble.argmax(-1)
            acc_ir += accuracy(pred_ir, ir_clean, pad)
            acc_ans += accuracy(pred_ans, targets, pad)
            acc_ans_scramble += accuracy(pred_ans_scramble, targets, pad)
            total += 1
            if len(samples) < max_samples:
                samples.append({
                    "input": decode(inputs[0].tolist()),
                    "ir_noisy": decode(ir_noisy[0].tolist()),
                    "ir_pred": decode(pred_ir[0].tolist()),
                    "ir_clean": decode(ir_clean[0].tolist()),
                    "target": decode(targets[0].tolist()),
                    "pred": decode(pred_ans[0].tolist()),
                    "pred_scramble": decode(pred_ans_scramble[0].tolist()),
                })
    return acc_ir/total, acc_ans/total, acc_ans_scramble/total, samples


def main():
    random.seed(0); torch.manual_seed(0)
    os.makedirs("air_gap/ir_diffusion_playground/results", exist_ok=True)
    train_ds = ToyIRDataset(n=20000)
    val_ds = ToyIRDataset(n=2000)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate)

    cfg = Config()
    corrector = IRCorrector(len(VOCAB), cfg).to(DEVICE)
    speaker = Speaker(len(VOCAB), cfg).to(DEVICE)
    opt_c = torch.optim.AdamW(corrector.parameters(), lr=2e-3)
    opt_s = torch.optim.AdamW(speaker.parameters(), lr=2e-3)
    pad = TOK2ID["<pad>"]

    metrics_path = "air_gap/ir_diffusion_playground/results/metrics.jsonl"
    open(metrics_path, "w").close()

    for epoch in range(10):
        ir_tr, ans_tr = train_epoch(corrector, speaker, train_loader, opt_c, opt_s, pad)
        ir_va, ans_va, ans_va_scramble, samples = eval_epoch(corrector, speaker, val_loader, pad)
        entry = {"epoch": epoch+1, "ir_train_acc": ir_tr, "ans_train_acc": ans_tr,
                 "ir_val_acc": ir_va, "ans_val_acc": ans_va, "ans_val_acc_scramble": ans_va_scramble}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        if samples:
            with open(f"air_gap/ir_diffusion_playground/results/samples_epoch{epoch+1}.jsonl", "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
        print(f"Epoch {epoch+1}: IR train {ir_tr:.3f}, ans train {ans_tr:.3f}, IR val {ir_va:.3f}, ans val {ans_va:.3f}")

if __name__ == "__main__":
    main()
