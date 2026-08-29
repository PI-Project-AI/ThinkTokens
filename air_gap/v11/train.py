import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import time
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from data import ArithmeticDataset, collate_fn
from model import AirGapVQTransformer, GPTConfig

def train():
    # V11 Hyperparams (Scale Up)
    batch_size = 128
    lr = 3e-4
    epochs = 40
    eval_interval = 500 # check less frequently as steps are faster/more numerous?
    log_interval = 50
    save_dir = "air_gap/v11/results"
    os.makedirs(save_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Data: 3-digit mixed ops
    # Increase num_samples to 200k for sufficient training on harder task
    train_dataset = ArithmeticDataset(num_samples=200000, complexity=3, split='train')
    test_dataset = ArithmeticDataset(num_samples=2000, complexity=3, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    vocab_size = 20 
    
    # Scaled Model
    model = AirGapVQTransformer(
        vocab_size=vocab_size,
        num_codes=1024,
        code_dim=384, # Increased from 128
        n_ir_tokens=32, # Increased from 16
        reasoner_layers=6,
        reasoner_heads=8, # 384/8 = 48 dim per head
        speaker_layers=6,
        speaker_heads=8,
        block_size=256 # Increased context window just in case
    ).to(device)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    step = 0
    history = []
    
    # Gumbel Schedule: Slower anneal for harder task
    # Soft for first 8 epochs?
    warmup_epochs = 8
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Gumbel Logic
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
                diversity_loss = torch.tensor(0.0)
                loss = speaker_loss + vq_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            perplexity = out['vq_info']['perplexity'].item()
            
            if step % log_interval == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'spk': f"{speaker_loss.item():.4f}", 
                    'ppl': f"{perplexity:.1f}",
                    'ent': f"{entropy.item():.2f}",
                    'mode': 'H' if hard else 'S'
                })
                history.append({
                    'step': step,
                    'loss': loss.item(),
                    'speaker_loss': speaker_loss.item(),
                    'vq_loss': vq_loss.item(),
                    'diversity_loss': diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else 0,
                    'perplexity': perplexity,
                    'temp': temp,
                    'hard': hard
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
    results = {}
    modes = ['intact', 'random', 'shuffle'] 
    
    for mode in modes:
        correct = 0
        total = 0
        limit = 200 if not final else len(loader.dataset)
        count = 0
        examples = []
        
        with torch.no_grad():
            for batch in loader:
                if count >= limit: break
                
                input_ids = batch['input_ids'].to(device)
                # Target IDs not strictly needed for generation, but needed for checking correctness if we didn't pass text
                # But we use target_texts
                input_texts = batch['input_texts']
                target_texts = batch['target_texts']
                
                pred_ids, ir_indices = model.generate(input_ids, max_new_tokens=15, ir_mode=mode)
                
                for i in range(len(input_ids)):
                    if count >= limit: break
                    
                    pred_str = dataset.decode(pred_ids[i].tolist())
                    target_str = target_texts[i]
                    
                    is_correct = (pred_str.strip() == target_str.strip())
                    correct += 1 if is_correct else 0
                    total += 1
                    
                    if len(examples) < 5:
                        examples.append({
                            'input': input_texts[i],
                            'target': target_str,
                            'pred': pred_str,
                            'ir': ir_indices[i].tolist(),
                            'correct': is_correct
                        })
                    count += 1
        
        acc = correct / total if total > 0 else 0
        results[mode] = acc
        
        if mode == 'intact':
            snap = {
                'step': step,
                'accuracy': acc,
                'examples': examples
            }
            with open(os.path.join(save_dir, f"snapshot_{step}.json"), 'w') as f:
                json.dump(snap, f, indent=2)
                
    print(f"\nStep {step} Eval: Intact={results['intact']:.2%}, Random={results['random']:.2%}, Shuffle={results['shuffle']:.2%}")
    
    log_file = os.path.join(save_dir, "eval_log.json")
    entry = {'step': step, **results}
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(log_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    train()