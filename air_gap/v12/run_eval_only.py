import torch
from torch.utils.data import DataLoader
import json
import os
from data import MixedReasoningDataset, collate_fn
from model import AirGapVQTransformer

def eval_only():
    batch_size = 128
    save_dir = "air_gap/v12/results"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Data
    train_dataset = MixedReasoningDataset(num_samples=100, split='train') # Dummy init to get vocab
    test_dataset = MixedReasoningDataset(num_samples=3000, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    vocab_size = len(train_dataset.vocab)
    
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
    
    evaluate(model, test_loader, device, 9999, save_dir, train_dataset, final=True)

def evaluate(model, loader, device, step, save_dir, dataset, final=False):
    model.eval()
    
    # Breakdown by task
    tasks = ['math', 'logic', 'nav']
    metrics = {t: {'correct': 0, 'total': 0} for t in tasks}
    
    # Ablations
    modes = ['intact', 'random', 'shuffle']
    results = {}
    
    for mode in modes:
        print(f"Running mode: {mode}")
        metrics = {t: {'correct': 0, 'total': 0} for t in tasks}
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                input_texts = batch['input_texts']
                target_texts = batch['target_texts']
                batch_tasks = batch['tasks']
                
                pred_ids, _ = model.generate(input_ids, max_new_tokens=15, ir_mode=mode)
                
                eos_id = dataset.eos_id
                
                for i in range(len(input_ids)):
                    # Truncate at EOS
                    pred_seq = pred_ids[i].tolist()
                    if eos_id in pred_seq:
                        pred_seq = pred_seq[:pred_seq.index(eos_id)]
                    
                    pred_str = dataset.decode(pred_seq)
                    target_str = target_texts[i]
                    task = batch_tasks[i]
                    
                    is_correct = (pred_str.strip().replace(' ', '') == target_str.strip().replace(' ', ''))
                    
                    metrics[task]['correct'] += 1 if is_correct else 0
                    metrics[task]['total'] += 1
        
        # Aggregate
        res_mode = {}
        for t in tasks:
            tot = metrics[t]['total']
            acc = metrics[t]['correct'] / tot if tot > 0 else 0
            res_mode[t] = acc
            print(f"  {t}: {acc:.2%}")
        results[mode] = res_mode

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    eval_only()
