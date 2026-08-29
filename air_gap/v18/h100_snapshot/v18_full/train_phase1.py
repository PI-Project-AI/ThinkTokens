import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
from data import ScaledHybridDataset, collate_fn
from model import AirGapVQTransformer

def train_phase1():
    # Phase 1: Predictive Pre-training (Input -> IR -> Next Segment/Math)
    # Keep IR tokens modest (64) and block size 512 to leave room for real context/targets.
    n_ir_tokens = 64
    block_size = 512
    batch_size = 64 # Larger batch size for H100
    lr = 3e-4
    epochs = 10 # Fewer epochs, larger dataset
    save_dir = "nano_architectures_v18/results_phase1"
    os.makedirs(save_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phase 1 using device: {device}")
    
    # Data (Hybrid)
    train_file_name = "TinyStoriesV2-GPT4-train.txt"
    val_file_name = "TinyStoriesV2-GPT4-valid.txt"

    dataset = ScaledHybridDataset(
        num_samples=2000000,
        split='train',
        tiny_stories_filename=train_file_name,
        block_size=block_size,
        n_ir_tokens=n_ir_tokens,
    )
    val_dataset = ScaledHybridDataset(
        num_samples=20000,
        split='test',
        tiny_stories_filename=val_file_name,
        block_size=block_size,
        n_ir_tokens=n_ir_tokens,
    ) # Larger val set
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    vocab_size = len(dataset.vocab)
    print(f"Vocab: {vocab_size}")
    
    # Model (Scaled V18)
    model = AirGapVQTransformer(
        vocab_size=vocab_size,
        num_codes=4096,
        code_dim=768,
        n_ir_tokens=n_ir_tokens, 
        reasoner_layers=12, reasoner_heads=12,
        speaker_layers=12, speaker_heads=12,
        block_size=block_size # For H100
    ).to(device)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"P1 Epoch {epoch+1}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device) # Predictive target
            
            optimizer.zero_grad()
            # Soft VQ for Phase 1
            out = model(input_ids, target_ids=target_ids, vq_hard=False) 
            loss = out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'ppl': f"{out['vq_info']['perplexity'].item():.1f}"})

        # Evaluate predictive reconstruction (similar to V17 eval_reconstruction)
        eval_predictive_reconstruction(model, val_loader, device, epoch, dataset.eos_id, dataset.decode)

    torch.save(model.state_dict(), "nano_architectures_v18/phase1_ae.pt")
    print("Phase 1 Complete. Model saved.")

def eval_predictive_reconstruction(model, loader, device, epoch, eos_id, decoder_fn):
    model.eval()
    total_correct_tokens = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            # Calculate max_new_tokens for generation
            max_gen_tokens_for_speaker = model.speaker_config.block_size - model.n_ir_tokens
            # For predictive reconstruction, we want to generate up to target length.
            # max_new_tokens should be len(target_ids) to cover the full target
            requested_new_tokens = target_ids.size(1) # We want to predict up to the length of the target
            actual_max_new_tokens = min(requested_new_tokens, max_gen_tokens_for_speaker)

            generated_ids, _ = model.generate(input_ids, max_new_tokens=actual_max_new_tokens, ir_mode='intact')
            
            for i in range(len(input_ids)):
                current_target_seq = target_ids[i].tolist() # The actual target sequence
                pred_seq = generated_ids[i].tolist()
                
                # Remove EOS and PAD from generated for comparison
                if eos_id in pred_seq: pred_seq = pred_seq[:pred_seq.index(eos_id)]
                pred_seq = [t for t in pred_seq if t != 0] # Remove PAD

                # Truncate pred_seq to target length for fair comparison
                pred_seq = pred_seq[:len(current_target_seq)]
                
                correct_tokens = sum(1 for p, t in zip(pred_seq, current_target_seq) if p == t)
                total_correct_tokens += correct_tokens
                total_tokens += len(current_target_seq) # Count tokens in the target

    predictive_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0.0
    print(f"P1 Epoch {epoch+1} Predictive Reconstruction Accuracy: {predictive_accuracy:.2%}")
    model.train() # Set back to train mode

if __name__ == "__main__":
    train_phase1()
