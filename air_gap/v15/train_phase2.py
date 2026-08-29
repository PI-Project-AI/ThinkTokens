import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
from data import MixedReasoningDataset, collate_fn
from model import AirGapVQTransformer

def train_phase2():
    # Phase 2: Reasoning Fine-Tune (Input -> IR -> Answer)
    batch_size = 128
    lr = 1e-4 # Lower LR
    epochs = 30
    save_dir = "air_gap/v15/results_phase2"
    os.makedirs(save_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phase 2 using device: {device}")
    
    # Data
    train_dataset = MixedReasoningDataset(num_samples=200000, split='train')
    test_dataset = MixedReasoningDataset(num_samples=2000, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Model
    vocab_size = len(train_dataset.vocab)
    model = AirGapVQTransformer(
        vocab_size=vocab_size,
        num_codes=1024,
        code_dim=384,
        n_ir_tokens=32,
        reasoner_layers=6, reasoner_heads=8,
        speaker_layers=6, speaker_heads=8
    ).to(device)
    
    # Load Phase 1
    print("Loading Phase 1 Weights...")
    model.load_state_dict(torch.load("air_gap/v15/phase1_ae.pt"))
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"P2 Epoch {epoch+1}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device) # Actual Answer
            
            optimizer.zero_grad()
            
            # Hard VQ for reasoning phase (Testing Discrete Hypothesis)
            out = model(input_ids, target_ids=target_ids, vq_hard=True) 
            
            loss = out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if pbar.n % 100 == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'spk': f"{out['speaker_loss'].item():.4f}"})
                
        # Simple Eval
        evaluate(model, test_loader, device, epoch)
        
    torch.save(model.state_dict(), f"{save_dir}/model_final.pt")

def evaluate(model, loader, device, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            target_texts = batch['target_texts']
            
            # Generate
            pred_ids, _ = model.generate(input_ids, max_new_tokens=10)
            
            for i in range(len(input_ids)):
                pred = loader.dataset.decode(pred_ids[i].tolist()).strip()
                target = target_texts[i].strip()
                if pred == target: correct += 1
                total += 1
                if total <= 2: print(f"T: {target} | P: {pred}")
                
    print(f"Epoch {epoch} Acc: {correct/total:.2%}")

if __name__ == "__main__":
    train_phase2()
