import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
from data import ScaledHybridDataset, collate_fn
from model import AirGapVQTransformer
import random

def train_phase2():
    # Phase 2 keeps the same block/IR layout as Phase 1 to make eval comparable.
    n_ir_tokens = 64
    block_size = 512
    batch_size = 64 # For H100
    lr = 1e-4 
    epochs = 15 # Shorter fine-tune for H100
    save_dir = "nano_architectures_v18/results_phase2"
    os.makedirs(save_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phase 2 using device: {device}")
    
    train_file_name = "TinyStoriesV2-GPT4-train.txt"
    test_file_name = "TinyStoriesV2-GPT4-valid.txt"

    train_dataset = ScaledHybridDataset(
        num_samples=2000000,
        split='train',
        tiny_stories_filename=train_file_name,
        block_size=block_size,
        n_ir_tokens=n_ir_tokens,
    )
    test_dataset = ScaledHybridDataset(
        num_samples=20000,
        split='test',
        tiny_stories_filename=test_file_name,
        block_size=block_size,
        n_ir_tokens=n_ir_tokens,
    ) # Larger test set
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    vocab_size = len(train_dataset.vocab)
    model = AirGapVQTransformer(
        vocab_size=vocab_size,
        num_codes=4096,
        code_dim=768,
        n_ir_tokens=n_ir_tokens,
        reasoner_layers=12, reasoner_heads=12,
        speaker_layers=12, speaker_heads=12,
        block_size=block_size
    ).to(device)
    
    print("Loading Phase 1 Weights...")
    model.load_state_dict(torch.load("nano_architectures_v18/phase1_ae.pt"))
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"P2 Epoch {epoch+1}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            tasks = batch['tasks']
            
            optimizer.zero_grad()
            out = model(input_ids, target_ids=target_ids, vq_hard=True) 
            loss = out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if pbar.n % 100 == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        evaluate(model, test_loader, device, epoch)
        
    torch.save(model.state_dict(), f"{save_dir}/model_final.pt")

def evaluate(model, loader, device, epoch):
    model.eval()
    metrics = {}
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            target_texts = batch['target_texts']
            tasks = batch['tasks']
            
            # Generate up to max_target_length
            max_gen_tokens_for_speaker = model.speaker_config.block_size - model.n_ir_tokens
            requested_new_tokens = max(len(t) for t in target_texts) # Generate up to the longest target
            actual_max_new_tokens = min(requested_new_tokens, max_gen_tokens_for_speaker)

            pred_ids, _ = model.generate(input_ids, max_new_tokens=actual_max_new_tokens)
            
            eos_id = loader.dataset.eos_id
            
            for i in range(len(input_ids)):
                pred_seq = pred_ids[i].tolist()
                if eos_id in pred_seq: pred_seq = pred_seq[:pred_seq.index(eos_id)]
                
                pred = loader.dataset.decode(pred_seq).strip().lower()
            target = target_texts[i].strip().lower()
            task = tasks[i]
            
            if task not in metrics: metrics[task] = {'correct': 0, 'total': 0}
            
            if task == 'math':
                # Math remains exact match to preserve a clean signal.
                is_correct = (pred == target)
                metrics[task]['correct'] += 1 if is_correct else 0
            else:
                # Story prediction: use token-level F1 to avoid permanent zero scores from exact-match only.
                p, r, f1 = token_f1(pred, target)
                metrics[task].setdefault('precision', 0.0)
                metrics[task].setdefault('recall', 0.0)
                metrics[task].setdefault('f1', 0.0)
                metrics[task]['precision'] += p
                metrics[task]['recall'] += r
                metrics[task]['f1'] += f1
                is_correct = None

            metrics[task]['total'] += 1

            # Log a few examples for manual review (especially for story_pred)
            if random.random() < 0.01: # Log ~1% of examples
                print(f"\n--- Example (Task: {task}, Correct: {is_correct}) ---")
                print(f"Input: {loader.dataset.decode(input_ids[i].tolist())}")
                print(f"Target: {target}")
                print(f"Pred: {pred}")

                
    print(f"\nEpoch {epoch} Eval:")
    for t in sorted(metrics.keys()):
        if t == 'math':
            acc = metrics[t]['correct'] / metrics[t]['total'] if metrics[t]['total'] > 0 else 0.0
            print(f"  {t}: {acc:.2%}")
        else:
            total = metrics[t]['total']
            prec = metrics[t]['precision'] / total if total else 0.0
            rec = metrics[t]['recall'] / total if total else 0.0
            f1 = metrics[t]['f1'] / total if total else 0.0
            print(f"  {t}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

def token_f1(pred, target):
    # Simple token-level precision/recall to avoid zeroing story metrics on near-misses.
    pred_tokens = pred.split()
    target_tokens = target.split()
    if not pred_tokens or not target_tokens:
        return 0.0, 0.0, 0.0
    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    target_counts = {}
    for t in target_tokens:
        target_counts[t] = target_counts.get(t, 0) + 1
    overlap = 0
    for t, c in pred_counts.items():
        overlap += min(c, target_counts.get(t, 0))
    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(target_tokens) if target_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

if __name__ == "__main__":
    train_phase2()
