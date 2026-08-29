import torch
from torch.utils.data import DataLoader
import json
import os
from data import MixedReasoningDataset, collate_fn
from model import AirGapVQTransformer

def evaluate_final():
    batch_size = 128
    save_dir = "air_gap/v15/results_phase2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data (Test split)
    test_dataset = MixedReasoningDataset(num_samples=3000, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    vocab_size = len(test_dataset.vocab)
    
    # Model
    model = AirGapVQTransformer(
        vocab_size=vocab_size,
        num_codes=1024,
        code_dim=384,
        n_ir_tokens=32,
        reasoner_layers=6, reasoner_heads=8,
        speaker_layers=6, speaker_heads=8
    ).to(device)
    
    # Load Weights
    print("Loading Final Model...")
    model.load_state_dict(torch.load(f"{save_dir}/model_final.pt", map_location=device))
    model.eval()
    
    metrics = {}
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            target_texts = batch['target_texts']
            tasks = batch['tasks']
            
            pred_ids, _ = model.generate(input_ids, max_new_tokens=15, ir_mode='intact')
            
            eos_id = test_dataset.eos_id
            
            for i in range(len(input_ids)):
                pred_seq = pred_ids[i].tolist()
                if eos_id in pred_seq:
                    pred_seq = pred_seq[:pred_seq.index(eos_id)]
                
                pred = test_dataset.decode(pred_seq).strip()
                target = target_texts[i].strip()
                task = tasks[i]
                
                if task not in metrics: metrics[task] = {'correct': 0, 'total': 0}
                
                # Relaxed check (ignore spaces)
                is_correct = (pred.replace(' ', '') == target.replace(' ', ''))
                metrics[task]['correct'] += 1 if is_correct else 0
                metrics[task]['total'] += 1

    print("\nFinal Evaluation:")
    for t in sorted(metrics.keys()):
        tot = metrics[t]['total']
        acc = metrics[t]['correct'] / tot if tot > 0 else 0
        print(f"  Task {t}: {acc:.2%}")

if __name__ == "__main__":
    evaluate_final()
