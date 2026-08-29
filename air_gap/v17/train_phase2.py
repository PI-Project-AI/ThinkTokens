import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
from data import HybridDataset, collate_fn
from model import AirGapVQTransformer

def train_phase2():
    # Phase 2: Reconstruction Fine-tune (or Reasoning if we changed objective, but V17 implies reconstruction baseline)
    # GOLD STANDARD: block_size=256, n_ir_tokens=64
    short_run = os.getenv("SHORT_RUN", "0") == "1"
    batch_size = 32
    lr = 1e-4 
    epochs = 5 if short_run else 20
    block_size = 256
    n_ir_tokens = 64
    save_dir = os.path.join(os.path.dirname(__file__), "results_phase2")
    os.makedirs(save_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phase 2 using device: {device}")
    
    base_dir = os.path.dirname(__file__)
    train_file = os.path.join(base_dir, "TinyStoriesV2-GPT4-train.txt")
    test_file = os.path.join(base_dir, "TinyStoriesV2-GPT4-valid.txt")

    train_dataset = HybridDataset(
        num_samples=50000 if short_run else 200000,
        split='train',
        tiny_stories_path=train_file,
        block_size=block_size,
        n_ir_tokens=n_ir_tokens,
    )
    test_dataset = HybridDataset(
        num_samples=2000,
        split='test',
        tiny_stories_path=test_file,
        block_size=block_size,
        n_ir_tokens=n_ir_tokens,
        # Reuse train vocab to keep token IDs aligned for eval decoding.
        vocab_tokens=train_dataset.tokens,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    vocab_size = len(train_dataset.vocab)
    model = AirGapVQTransformer(
        vocab_size=vocab_size,
        num_codes=1024,
        code_dim=384,
        n_ir_tokens=n_ir_tokens,
        reasoner_layers=6, reasoner_heads=8,
        speaker_layers=6, speaker_heads=8,
        block_size=block_size 
    ).to(device)
    
    print("Loading Phase 1 Weights...")
    ckpt_path = os.path.join(base_dir, "phase1_ae.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print("Warning: Phase 1 checkpoint not found. Training from scratch.")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"P2 Epoch {epoch+1}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            optimizer.zero_grad()
            out = model(input_ids, target_ids=target_ids, vq_hard=True) 
            loss = out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if pbar.n % 100 == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'spk': f"{out['speaker_loss'].item():.4f}"})
                
        evaluate(model, test_loader, device, epoch, save_dir)
        
    torch.save(model.state_dict(), f"{save_dir}/model_final.pt")

def evaluate(model, loader, device, epoch, save_dir, max_logged_samples=5):
    model.eval()
    metrics = {}
    metrics_shuffle = {}
    logged_samples = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            target_texts = batch['target_texts']
            tasks = batch['tasks']
            
            max_gen = model.speaker_config.block_size - model.n_ir_tokens
            
            # 1. Normal Generation
            pred_ids, _ = model.generate(input_ids, max_new_tokens=max_gen, ir_mode='intact')
            
            # 2. Shuffle Generation (Causality Test)
            pred_ids_shuf, _ = model.generate(input_ids, max_new_tokens=max_gen, ir_mode='shuffle')
            
            eos_id = loader.dataset.eos_id
            
            for i in range(len(input_ids)):
                # Decode Normal
                pred_seq = pred_ids[i].tolist()
                if eos_id in pred_seq: pred_seq = pred_seq[:pred_seq.index(eos_id)]
                pred = loader.dataset.decode(pred_seq).strip().lower()

                # Decode Shuffle
                pred_seq_shuf = pred_ids_shuf[i].tolist()
                if eos_id in pred_seq_shuf: pred_seq_shuf = pred_seq_shuf[:pred_seq_shuf.index(eos_id)]
                pred_shuf = loader.dataset.decode(pred_seq_shuf).strip().lower()

                target = target_texts[i].strip().lower()
                task = tasks[i]
                
                # Init Metrics
                if task not in metrics:
                    metrics[task] = {'correct': 0, 'total': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                    metrics_shuffle[task] = {'correct': 0, 'total': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                
                # Compute
                if task == 'math':
                    metrics[task]['correct'] += 1 if (pred == target) else 0
                    metrics_shuffle[task]['correct'] += 1 if (pred_shuf == target) else 0
                else:
                    p, r, f1 = token_f1(pred, target)
                    metrics[task]['precision'] += p
                    metrics[task]['recall'] += r
                    metrics[task]['f1'] += f1
                    
                    p_s, r_s, f1_s = token_f1(pred_shuf, target)
                    metrics_shuffle[task]['precision'] += p_s
                    metrics_shuffle[task]['recall'] += r_s
                    metrics_shuffle[task]['f1'] += f1_s

                metrics[task]['total'] += 1
                metrics_shuffle[task]['total'] += 1
                
                if len(logged_samples) < max_logged_samples:
                    logged_samples.append({
                        "task": task,
                        "input": loader.dataset.decode(input_ids[i].tolist()),
                        "target": target,
                        "pred": pred,
                        "pred_shuffle": pred_shuf
                    })
                
    print(f"\nEpoch {epoch+1} Eval:")
    epoch_metrics = {"epoch": epoch + 1, "tasks": {}}
    
    for t in sorted(metrics.keys()):
        total = metrics[t]['total']
        if total == 0: continue
        
        if t == 'math':
            acc = metrics[t]['correct'] / total
            acc_s = metrics_shuffle[t]['correct'] / total
            print(f"  {t}: {acc:.2%} | Shuffle: {acc_s:.2%}")
            epoch_metrics["tasks"][t] = {"acc": acc, "acc_shuffle": acc_s}
        else:
            prec = metrics[t]['precision'] / total
            rec = metrics[t]['recall'] / total
            f1 = metrics[t]['f1'] / total
            
            f1_s = metrics_shuffle[t]['f1'] / total
            
            print(f"  {t}: F1={f1:.3f} | Shuffle F1={f1_s:.3f}")
            epoch_metrics["tasks"][t] = {"f1": f1, "f1_shuffle": f1_s}

    # Save metrics
    with open(os.path.join(save_dir, "eval_metrics.jsonl"), "a") as f:
        f.write(json.dumps(epoch_metrics) + "\n")
        
    # Save samples
    with open(os.path.join(save_dir, f"samples_epoch{epoch+1}.jsonl"), "w") as f:
        for s in logged_samples:
            f.write(json.dumps(s) + "\n")

def token_f1(pred, target):
    pred_tokens = pred.split()
    target_tokens = target.split()
    if not pred_tokens or not target_tokens:
        return 0.0, 0.0, 0.0
    
    common = 0
    pred_counts = {}
    for t in pred_tokens: pred_counts[t] = pred_counts.get(t, 0) + 1
    
    target_counts = {}
    for t in target_tokens: target_counts[t] = target_counts.get(t, 0) + 1
    
    for t, c in pred_counts.items():
        common += min(c, target_counts.get(t, 0))
        
    prec = common / len(pred_tokens)
    rec = common / len(target_tokens)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

if __name__ == "__main__":
    train_phase2()
