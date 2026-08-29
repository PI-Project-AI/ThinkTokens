import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
from data import MixedReasoningDataset, collate_fn
from model import AirGapVQTransformer

def train_phase1():
    # Phase 1: Auto-Encoder (Input -> IR -> Input)
    batch_size = 128
    lr = 3e-4
    epochs = 20
    log_interval = 100
    save_dir = "air_gap/v15/results_phase1"
    os.makedirs(save_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phase 1 using device: {device}")
    
    # Data (200k Mixed)
    dataset = MixedReasoningDataset(num_samples=200000, split='train')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    vocab_size = len(dataset.vocab)
    print(f"Vocab: {vocab_size}")
    
    # Model
    model = AirGapVQTransformer(
        vocab_size=vocab_size,
        num_codes=1024,
        code_dim=384,
        n_ir_tokens=32, # Tight bottleneck
        reasoner_layers=6, reasoner_heads=8,
        speaker_layers=6, speaker_heads=8
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"P1 Epoch {epoch+1}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            # Target is INPUT (Reconstruction)
            # Shifted? NanoGPT forward usually takes input_ids and targets
            # If target is input, we want Speaker to predict Input given IR.
            # Speaker Input: IR
            # Speaker Target: Input
            
            optimizer.zero_grad()
            
            # We must trick the forward pass.
            # Normal forward: Reasoner(input) -> IR -> Speaker(IR + target_embeds) -> Loss(target)
            # Here target = input_ids.
            
            out = model(input_ids, target_ids=input_ids, vq_hard=False) # Soft VQ for easier gradient in P1?
            # Actually, let's anneal or keep Soft for P1 to ensure codebook usage.
            
            loss = out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if pbar.n % log_interval == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'ppl': f"{out['vq_info']['perplexity'].item():.1f}"})
                
    # Save Weights
    torch.save(model.state_dict(), "air_gap/v15/phase1_ae.pt")
    print("Phase 1 Complete. Model saved.")

if __name__ == "__main__":
    train_phase1()
