import torch
from torch.utils.data import DataLoader
import json
import os
from data import BabiDataset, collate_fn
from model import AirGapVQTransformer

def run_ablations():
    batch_size = 64
    save_dir = "air_gap/v13/results"
    data_dir = "air_gap/v13/tasks_1-20_v1-2/en-10k"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Data (Test split)
    test_dataset = BabiDataset(data_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    vocab_size = len(test_dataset.vocab)
    
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
    
    model_path = os.path.join(save_dir, "model_final.pt")
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Run Ablations
    modes = ['intact', 'random', 'shuffle']
    results = {}
    
    for mode in modes:
        print(f"\nRunning Ablation: {mode.upper()}")
        task_metrics = {i: {'correct': 0, 'total': 0} for i in range(1, 21)}
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                tasks = batch['tasks']
                
                # Generate
                pred_ids, _ = model.generate(input_ids, max_new_tokens=5, ir_mode=mode)
                
                eos_id = test_dataset.eos_id
                
                for i in range(len(input_ids)):
                    pred_seq = pred_ids[i].tolist()
                    if eos_id in pred_seq:
                        pred_seq = pred_seq[:pred_seq.index(eos_id)]
                    
                    # Convert to text
                    pred_str = test_dataset.decode(pred_seq).strip().lower()
                    target_str = test_dataset.decode(target_ids[i].tolist()).strip().lower() # Target IDs include EOS usually, decode handles it?
                    # target_ids in batch includes EOS. Decode removes specials.
                    
                    is_correct = (pred_str == target_str)
                    
                    task_id = tasks[i]
                    task_metrics[task_id]['correct'] += 1 if is_correct else 0
                    task_metrics[task_id]['total'] += 1
        
        # Aggregate
        mode_res = {}
        total_correct = 0
        total_count = 0
        
        print(f"{'Task':<5} | {'Acc':<10} | {'Count':<10}")
        print("-" * 30)
        
        for t in range(1, 21):
            correct = task_metrics[t]['correct']
            total = task_metrics[t]['total']
            acc = correct / total if total > 0 else 0.0
            
            mode_res[f"task_{t}"] = acc
            total_correct += correct
            total_count += total
            
            print(f"{t:<5} | {acc:.2%} | {total:<10}")
            
        overall = total_correct / total_count if total_count > 0 else 0.0
        mode_res["overall"] = overall
        results[mode] = mode_res
        print("-" * 30)
        print(f"OVERALL: {overall:.2%}\n")

    # Save Results
    with open(os.path.join(save_dir, "ablation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_ablations()
