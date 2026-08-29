import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
from data import TrinityDataset, collate_fn
from model import AirGapVQTransformer

def train_phase2():
    batch_size = 128
    lr = 1e-4
    epochs = 20
    save_dir = "air_gap/v16/results_phase2"
    os.makedirs(save_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dataset = TrinityDataset(num_samples=200000, split='train')
    test_dataset = TrinityDataset(num_samples=3000, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    vocab_size = len(train_dataset.vocab)
    model = AirGapVQTransformer(
        vocab_size=vocab_size,
        num_codes=1024,
        code_dim=384,
        n_ir_tokens=32,
        reasoner_layers=6, reasoner_heads=8,
        speaker_layers=6, speaker_heads=8
    ).to(device)
    
    print("Loading Phase 1...")
    model.load_state_dict(torch.load("air_gap/v16/phase1_ae.pt"))
    
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
            
            pred_ids, _ = model.generate(input_ids, max_new_tokens=15)
            
            eos_id = loader.dataset.eos_id
            
            for i in range(len(input_ids)):
                pred_seq = pred_ids[i].tolist()
                if eos_id in pred_seq: pred_seq = pred_seq[:pred_seq.index(eos_id)]
                
                pred = loader.dataset.decode(pred_seq).strip()
                target = target_texts[i].strip()
                task = tasks[i]
                
                if task not in metrics: metrics[task] = {'correct': 0, 'total': 0}
                
                # Loose match for Chat/Story?
                # For Math: Exact match.
                # For Chat/Story: Target is fixed in synthetic, so exact match is fair proxy for "learning the distribution".
                
                is_correct = (pred == target)
                metrics[task]['correct'] += 1 if is_correct else 0
                metrics[task]['total'] += 1
                
    print(f"\nEpoch {epoch} Eval:")
    for t in sorted(metrics.keys()):
        acc = metrics[t]['correct'] / metrics[t]['total']
        print(f"  {t}: {acc:.2%}")

if __name__ == "__main__":
    train_phase2()
