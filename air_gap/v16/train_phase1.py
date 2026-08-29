import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
from data import TrinityDataset, collate_fn
from model import AirGapVQTransformer

def train_phase1():
    batch_size = 128
    lr = 3e-4
    epochs = 15 # Shorter AE phase? 20 is safe.
    save_dir = "air_gap/v16/results_phase1"
    os.makedirs(save_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = TrinityDataset(num_samples=200000, split='train')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    vocab_size = len(dataset.vocab)
    print(f"Vocab: {vocab_size}")
    
    model = AirGapVQTransformer(
        vocab_size=vocab_size,
        num_codes=1024,
        code_dim=384,
        n_ir_tokens=32,
        reasoner_layers=6, reasoner_heads=8,
        speaker_layers=6, speaker_heads=8
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"P1 Epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            optimizer.zero_grad()
            out = model(input_ids, target_ids=input_ids, vq_hard=False)
            loss = out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if pbar.n % 100 == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'ppl': f"{out['vq_info']['perplexity'].item():.1f}"})
                
    torch.save(model.state_dict(), "air_gap/v16/phase1_ae.pt")

if __name__ == "__main__":
    train_phase1()
