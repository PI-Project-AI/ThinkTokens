import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
from data import HybridDataset, collate_fn
from model import AirGapVQTransformer

def train_phase1():
    # Phase 1: Reconstruction/pretraining.
    # GOLD STANDARD: block_size=256, n_ir_tokens=64
    short_run = os.getenv("SHORT_RUN", "0") == "1"
    batch_size = 32
    lr = 3e-4
    epochs = 5 if short_run else 15
    block_size = 256
    n_ir_tokens = 64
    save_dir = os.path.join(os.path.dirname(__file__), "results_phase1")
    os.makedirs(save_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phase 1 using device: {device}")
    
    # We assume the user runs this from the project root, so we look for data relative to this script
    base_dir = os.path.dirname(__file__)
    train_file = os.path.join(base_dir, "TinyStoriesV2-GPT4-train.txt")
    val_file = os.path.join(base_dir, "TinyStoriesV2-GPT4-valid.txt")

    dataset = HybridDataset(
        num_samples=50000 if short_run else 200000,
        split='train',
        tiny_stories_path=train_file,
        block_size=block_size,
        n_ir_tokens=n_ir_tokens,
    )
    val_dataset = HybridDataset(
        num_samples=2000,
        split='test',
        tiny_stories_path=val_file,
        block_size=block_size,
        n_ir_tokens=n_ir_tokens,
        # Keep train/val token IDs aligned for meaningful reconstruction eval.
        vocab_tokens=dataset.tokens,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    vocab_size = len(dataset.vocab)
    print(f"Vocab: {vocab_size}")
    
    model = AirGapVQTransformer(
        vocab_size=vocab_size,
        num_codes=1024,
        code_dim=384,
        n_ir_tokens=n_ir_tokens, 
        reasoner_layers=6, reasoner_heads=8,
        speaker_layers=6, speaker_heads=8,
        block_size=block_size 
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"P1 Epoch {epoch+1}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            # For reconstruction, target is same as input
            target_ids = input_ids.clone()
            
            optimizer.zero_grad()
            out = model(input_ids, target_ids=target_ids, vq_hard=False) 
            loss = out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'ppl': f"{out['vq_info']['perplexity'].item():.1f}"})
                
        acc = eval_reconstruction(model, val_loader, device, epoch, dataset.eos_id)
        
        # Log
        with open(os.path.join(save_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps({"epoch": epoch+1, "reconstruction_acc": acc}) + "\n")

    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "phase1_ae.pt"))
    print("Phase 1 Complete. Model saved.")

def eval_reconstruction(model, loader, device, epoch, eos_id):
    model.eval()
    total_correct_tokens = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            max_gen_tokens_for_speaker = model.speaker_config.block_size - model.n_ir_tokens
            requested_new_tokens = input_ids.size(1) 
            actual_max_new_tokens = min(requested_new_tokens, max_gen_tokens_for_speaker)

            generated_ids, _ = model.generate(input_ids, max_new_tokens=actual_max_new_tokens, ir_mode='intact')
            
            for i in range(len(input_ids)):
                target_seq = input_ids[i].tolist() 
                pred_seq = generated_ids[i].tolist()
                
                if eos_id in pred_seq: pred_seq = pred_seq[:pred_seq.index(eos_id)]
                pred_seq = [t for t in pred_seq if t != 0] # Remove PAD

                pred_seq = pred_seq[:len(target_seq)]
                
                correct_tokens = sum(1 for p, t in zip(pred_seq, target_seq) if p == t)
                total_correct_tokens += correct_tokens
                total_tokens += len(target_seq)

    acc = total_correct_tokens / total_tokens if total_tokens > 0 else 0.0
    print(f"P1 Epoch {epoch+1} Reconstruction Accuracy: {acc:.2%}")
    model.train() 
    return acc

if __name__ == "__main__":
    train_phase1()
