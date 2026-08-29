"""
Minimal two-model LA baseline (CPU-friendly).
- Module A: encoder from text -> LA codes.
- Module B: reasoning over LA (here identity with slight noise; acts as placeholder).
- Module C: decoder from LA -> text.
Task: tiny arithmetic "a+b=" -> answer. LA is just the pair of numbers (discrete).
Logging: JSONL metrics and a few samples per epoch to results/.

Run: python two_model_architecture/run_two_model_baseline.py
"""
import json
import os
import random
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB = ["<pad>", "<bos>", "<eos>", "<sep>"] + [str(i) for i in range(20)] + ["+", "="]  # support sums up to 18
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for t, i in TOK2ID.items()}


def encode(tokens: List[str]) -> List[int]:
    return [TOK2ID[t] for t in tokens]


def decode(ids: List[int]) -> List[str]:
    return [ID2TOK[i] for i in ids]


class ToyDataset(Dataset):
    def __init__(self, n=20000):
        self.samples = []
        for _ in range(n):
            a, b = random.randint(0, 9), random.randint(0, 9)
            inp = [str(a), "+", str(b), "="]
            ans = [str(a + b)]
            la = [a, b]  # abstract code tokens before quantization
            self.samples.append({"input": encode(inp), "la": la, "target": encode(ans)})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    pad = TOK2ID["<pad>"]
    def pad_seq(seq, maxlen):
        return seq + [pad] * (maxlen - len(seq))
    max_in = max(len(b["input"]) for b in batch)
    max_tgt = max(len(b["target"]) for b in batch)
    inputs = torch.tensor([pad_seq(b["input"], max_in) for b in batch])
    targets = torch.tensor([pad_seq(b["target"], max_tgt) for b in batch])
    la = torch.tensor([b["la"] for b in batch])  # shape (B, 2)
    return inputs, la, targets


@dataclass
class Config:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 128
    dropout: float = 0.1
    la_vocab: int = 16  # quantized LA codes


class EncoderLA(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.d_model, padding_idx=TOK2ID["<pad>"])
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.dim_ff, dropout=cfg.dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.head = nn.Linear(cfg.d_model, cfg.la_vocab)

    def forward(self, x):
        h = self.embed(x)
        h = self.encoder(h)
        # Pool by taking last timestep
        pooled = h[:, -1, :]
        logits = self.head(pooled)  # (B, la_vocab)
        return logits


class ReasonerLA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.embed = nn.Embedding(cfg.la_vocab, cfg.d_model)
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_ff),
            nn.ReLU(),
            nn.Linear(cfg.dim_ff, cfg.d_model),
        )
        self.head = nn.Linear(cfg.d_model, cfg.la_vocab)

    def forward(self, la_codes):
        h = self.embed(la_codes)
        h = self.net(h)
        return self.head(h)


class DecoderLA(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.d_model, padding_idx=TOK2ID["<pad>"])
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.dim_ff, dropout=cfg.dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_layers)
        self.head = nn.Linear(cfg.d_model, vocab_size)

    def forward(self, la_emb, tgt_inp):
        tgt = self.embed(tgt_inp)
        h = self.decoder(tgt, la_emb)
        return self.head(h)


def accuracy(pred, target, pad):
    with torch.no_grad():
        mask = target != pad
        correct = (pred == target) * mask
        return correct.sum().item() / max(1, mask.sum().item())


def train_epoch(enc, reasoner, dec, data, opt_all, cfg):
    ce = nn.CrossEntropyLoss(ignore_index=TOK2ID["<pad>"])
    enc.train(); reasoner.train(); dec.train()
    pbar = tqdm(data, desc="train", leave=False)
    total = 0; acc_tgt = 0
    for inputs, la, targets in pbar:
        inputs, la, targets = [x.to(DEVICE) for x in (inputs, la, targets)]
        # Encode to LA logits and sample codes
        logits_la = enc(inputs)
        la_codes = torch.argmax(logits_la, dim=-1)
        # Reasoner over LA (identity-ish)
        reason_logits = reasoner(la_codes)
        la_reason_codes = torch.argmax(reason_logits, dim=-1).unsqueeze(1)  # (B,1)
        la_emb = reasoner.embed(la_reason_codes)  # reuse embedding as memory
        # Decode to text
        logits_tgt = dec(la_emb, targets)
        loss = ce(logits_tgt.view(-1, logits_tgt.size(-1)), targets.view(-1))
        opt_all.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(reasoner.parameters())+list(dec.parameters()), 1.0)
        opt_all.step()
        pred_tgt = logits_tgt.argmax(-1)
        acc_tgt += accuracy(pred_tgt, targets, TOK2ID["<pad>"])
        total += 1
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc_tgt/total:.3f}")
    return acc_tgt/total


def eval_epoch(enc, reasoner, dec, data, cfg, max_samples=4):
    ce = nn.CrossEntropyLoss(ignore_index=TOK2ID["<pad>"])
    enc.eval(); reasoner.eval(); dec.eval()
    total = 0; acc_tgt = 0; samples = []
    with torch.no_grad():
        for inputs, la, targets in data:
            inputs, la, targets = [x.to(DEVICE) for x in (inputs, la, targets)]
            logits_la = enc(inputs)
            la_codes = torch.argmax(logits_la, dim=-1)
            reason_logits = reasoner(la_codes)
            la_reason_codes = torch.argmax(reason_logits, dim=-1).unsqueeze(1)
            la_emb = reasoner.embed(la_reason_codes)
            logits_tgt = dec(la_emb, targets)
            pred_tgt = logits_tgt.argmax(-1)
            acc_tgt += accuracy(pred_tgt, targets, TOK2ID["<pad>"])
            total += 1
            if len(samples) < max_samples:
                samples.append({
                    "input": decode(inputs[0].tolist()),
                    "la_code": int(la_codes[0].item()),
                    "la_reason": int(la_reason_codes[0,0].item()),
                    "target": decode(targets[0].tolist()),
                    "pred": decode(pred_tgt[0].tolist()),
                })
    return acc_tgt/total, samples


def main():
    random.seed(0); torch.manual_seed(0)
    os.makedirs("two_model_architecture/results", exist_ok=True)
    train_ds = ToyDataset(n=20000)
    val_ds = ToyDataset(n=2000)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate)
    cfg = Config()
    enc = EncoderLA(len(VOCAB), cfg).to(DEVICE)
    reasoner = ReasonerLA(cfg).to(DEVICE)
    dec = DecoderLA(len(VOCAB), cfg).to(DEVICE)
    opt_all = torch.optim.AdamW(list(enc.parameters())+list(reasoner.parameters())+list(dec.parameters()), lr=2e-3)
    metrics_path = "two_model_architecture/results/metrics.jsonl"
    open(metrics_path, "w").close()
    for epoch in range(10):
        acc_tr = train_epoch(enc, reasoner, dec, train_loader, opt_all, cfg)
        acc_va, samples = eval_epoch(enc, reasoner, dec, val_loader, cfg)
        entry = {"epoch": epoch+1, "train_acc": acc_tr, "val_acc": acc_va}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        if samples:
            with open(f"two_model_architecture/results/samples_epoch{epoch+1}.jsonl", "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
        print(f"Epoch {epoch+1}: train_acc {acc_tr:.3f}, val_acc {acc_va:.3f}")

if __name__ == "__main__":
    main()
