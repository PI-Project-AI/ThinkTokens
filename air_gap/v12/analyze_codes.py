import torch
from torch.utils.data import DataLoader
import json
import os
import numpy as np
from data import MixedReasoningDataset, collate_fn
from model import AirGapVQTransformer

def analyze_overlap():
    batch_size = 128
    save_dir = "air_gap/v12/results"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data
    dataset = MixedReasoningDataset(num_samples=3000, split='test') # 1000 per task approx
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    vocab_size = len(dataset.vocab)
    
    # Load Model
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
    
    model.load_state_dict(torch.load(os.path.join(save_dir, "model_final.pt"), map_location=device))
    model.eval()
    
    # Collect codes per task
    task_codes = {'math': set(), 'logic': set(), 'nav': set()}
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            batch_tasks = batch['tasks']
            
            _, ir_indices = model.generate(input_ids, max_new_tokens=1) # Don't need generation, just IR
            
            ir_indices = ir_indices.cpu().numpy() # [B, T]
            
            for i, task in enumerate(batch_tasks):
                unique_codes = set(ir_indices[i])
                task_codes[task].update(unique_codes)
                
    # Analyze Jaccard
    tasks = ['math', 'logic', 'nav']
    print("\nActive Codes per Task:")
    for t in tasks:
        print(f"  {t}: {len(task_codes[t])}")
        
    print("\nJaccard Similarity (Intersection / Union):")
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            t1, t2 = tasks[i], tasks[j]
            s1, s2 = task_codes[t1], task_codes[t2]
            
            intersection = len(s1.intersection(s2))
            union = len(s1.union(s2))
            jaccard = intersection / union if union > 0 else 0
            
            print(f"  {t1} vs {t2}: {jaccard:.4f} (Int: {intersection}, Union: {union})")

if __name__ == "__main__":
    analyze_overlap()
