import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
import numpy as np

from data import AlgorithmicDataset, collate_fn
from model import AirGapVQTransformer

def train():
    # V14 Hyperparams (Algorithmic)
    batch_size = 128
    lr = 3e-4
    epochs = 30 
    eval_interval = 500
    log_interval = 100
    save_dir = "air_gap/v14/results"
    os.makedirs(save_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Data: Algorithmic Traces
    train_dataset = AlgorithmicDataset(num_samples=200000, split='train')
    test_dataset = AlgorithmicDataset(num_samples=2000, split='test') 
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    vocab_size = len(train_dataset.vocab) 
    print(f"Vocab Size: {vocab_size}")
    
    # V14 Model (Same Config as V13)
    model = AirGapVQTransformer(
        vocab_size=vocab_size,
        num_codes=1024,
        code_dim=384,
        n_ir_tokens=64, 
        reasoner_layers=6,
        reasoner_heads=8,
        speaker_layers=6,
        speaker_heads=8,
        block_size=256
    ).to(device)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    
    step = 0
    history = []
    
    warmup_epochs = 5
    warmup_steps = warmup_epochs * len(train_loader)
    
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            if step < warmup_steps:
                hard = False
                frac = step / warmup_steps
                temp = 2.0 * (1 - frac) + 0.5 * frac
            else:
                hard = True
                temp = 0.5
            
            optimizer.zero_grad()
            
            out = model(input_ids, target_ids, vq_hard=hard, vq_temp=temp)
            
            speaker_loss = out['speaker_loss']
            vq_loss = out['vq_loss']
            
            entropy = torch.tensor(0.0, device=device)
            if 'encodings' in out['vq_info']:
                encodings = out['vq_info']['encodings'] 
                avg_probs = encodings.mean(dim=0) 
                entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
                diversity_loss = -entropy
                loss = speaker_loss + vq_loss + 0.1 * diversity_loss
            else:
                loss = speaker_loss + vq_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            perplexity = out['vq_info']['perplexity'].item()
            
            if step % log_interval == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'spk': f"{speaker_loss.item():.4f}", 
                    'ppl': f"{perplexity:.1f}",
                    'ent': f"{entropy.item():.2f}"
                })
                history.append({
                    'step': step,
                    'loss': loss.item(),
                    'speaker_loss': speaker_loss.item(),
                    'perplexity': perplexity
                })
                
            if step % eval_interval == 0:
                evaluate(model, test_loader, device, step, save_dir, train_dataset)
                model.train()
                
            step += 1
            
    # Final eval
    evaluate(model, test_loader, device, step, save_dir, train_dataset, final=True)
    
    torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pt"))
    with open(os.path.join(save_dir, "history.json"), 'w') as f:
        json.dump(history, f)

def evaluate(model, loader, device, step, save_dir, dataset, final=False):
    model.eval() 
    
    metrics = {}
    limit = 500 if not final else len(loader.dataset)
    count = 0
    examples = []
    
    with torch.no_grad():
        for batch in loader:
            if count >= limit: break
            
            input_ids = batch['input_ids'].to(device)
            input_texts = batch['input_texts']
            target_texts = batch['target_texts']
            batch_tasks = batch['tasks']
            
            pred_ids, ir_indices = model.generate(input_ids, max_new_tokens=5, ir_mode='intact')
            
            eos_id = dataset.eos_id
            
            for i in range(len(input_ids)):
                if count >= limit: break
                
                pred_seq = pred_ids[i].tolist()
                if eos_id in pred_seq:
                    pred_seq = pred_seq[:pred_seq.index(eos_id)]
                
                pred_str = dataset.decode(pred_seq).strip()
                target_str = target_texts[i].strip()
                task = batch_tasks[i]
                
                if task not in metrics: metrics[task] = {'correct': 0, 'total': 0}
                
                is_correct = (pred_str == target_str)
                
                metrics[task]['correct'] += 1 if is_correct else 0
                metrics[task]['total'] += 1
                
                if len(examples) < 10: 
                    examples.append({
                        'task': task,
                        'input': input_texts[i],
                        'target': target_str,
                        'pred': pred_str,
                        'ir': ir_indices[i].tolist(),
                        'correct': is_correct
                    })
                count += 1
    
    log_entry = {'step': step}
    print(f"\nStep {step} Eval:")
    
    total_correct = sum(m['correct'] for m in metrics.values())
    total_count = sum(m['total'] for m in metrics.values())
    print(f"  Overall: {total_correct/total_count:.2%} ({total_correct}/{total_count})")
    
    for t in sorted(metrics.keys()):
        tot = metrics[t]['total']
        acc = metrics[t]['correct'] / tot if tot > 0 else 0
        print(f"  Task {t}: {acc:.2%}")
        log_entry[f'acc_{t}'] = acc
        
    snap = {
        'step': step,
        'metrics': log_entry,
        'examples': examples
    }
    with open(os.path.join(save_dir, f"snapshot_{step}.json"), 'w') as f:
        json.dump(snap, f, indent=2)
    
    log_file = os.path.join(save_dir, "eval_log.json")
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            data = json.load(f)
    else:
        data = []
    data.append(log_entry)
    with open(log_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    train()
