import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
from data import PredictiveDataset, collate_fn
from model import AirGapVQTransformer

def train_phase1():
    # Phase 1: Predictive Pre-training
    # Input: Story Start -> IR -> Story End
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

    dataset = PredictiveDataset(
        num_samples=50000 if short_run else 200000,
        split='train',
        tiny_stories_path=train_file,
        block_size=block_size,
        n_ir_tokens=n_ir_tokens,
    )
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
            target_ids = batch['target_ids'].to(device)
            
            optimizer.zero_grad()
            # Soft VQ for Phase 1
            out = model(input_ids, target_ids=target_ids, vq_hard=False) 
            loss = out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'ppl': f"{out['vq_info']['perplexity'].item():.1f}"})

    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "phase1_ae.pt"))
    print("Phase 1 Complete. Model saved.")

if __name__ == "__main__":
    train_phase1()