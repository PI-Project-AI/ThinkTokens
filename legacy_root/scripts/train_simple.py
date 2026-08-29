#!/usr/bin/env python3
"""Simplified training script using VQ language model."""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
from pathlib import Path
from vq_model_v2 import VQLanguageModel

# Configuration
CONFIG = {
    'model_name': "EleutherAI/pythia-410m",
    'num_codes': 512,
    'batch_size': 4,
    'epochs': 2,
    'learning_rate': 2e-5,
    'max_length': 256,
    'max_samples': 500,  # Use small subset for quick testing
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

def main():
    device = CONFIG['device']
    print(f"Device: {device}")
    print(f"Config: {json.dumps({k: str(v) if not isinstance(v, (int, float)) else v for k, v in CONFIG.items()}, indent=2)}\n")

    # Setup directories
    Path("results").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("gsm8k", "main")['train'].select(range(min(CONFIG['max_samples'], 7473)))

    def preprocess(examples):
        texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples['question'], examples['answer'])]
        encodings = tokenizer(texts, truncation=True, max_length=CONFIG['max_length'], padding='max_length')
        encodings['labels'] = encodings['input_ids'].copy()
        return encodings

    print("Preprocessing dataset...")
    dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    print(f"Dataset size: {len(dataset)}, Batches per epoch: {len(dataloader)}\n")

    # Initialize model
    print("Initializing model...")
    model = VQLanguageModel(CONFIG['model_name'], CONFIG['num_codes']).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])

    # Training loop
    history = {'epochs': [], 'losses': []}

    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        model.train()

        total_loss = 0
        total_lm_loss = 0
        total_vq_loss = 0
        code_usage = 0

        pbar = tqdm(dataloader, desc=f"Training")
        for i, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            if outputs['lm_loss'] is not None:
                total_lm_loss += outputs['lm_loss'].item()
            total_vq_loss += outputs['vq_loss'].item()
            code_usage += outputs['code_usage_pct']

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'code_usage': f"{outputs['code_usage_pct']:.1f}%"
            })

        # Epoch metrics
        avg_loss = total_loss / len(dataloader)
        avg_lm_loss = total_lm_loss / len(dataloader) if total_lm_loss > 0 else 0
        avg_vq_loss = total_vq_loss / len(dataloader)
        avg_code_usage = code_usage / len(dataloader)

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  LM Loss: {avg_lm_loss:.4f}")
        print(f"  VQ Loss: {avg_vq_loss:.4f}")
        print(f"  Code Usage: {avg_code_usage:.1f}%")

        history['epochs'].append(epoch + 1)
        history['losses'].append({
            'total_loss': avg_loss,
            'lm_loss': avg_lm_loss,
            'vq_loss': avg_vq_loss,
            'code_usage': avg_code_usage
        })

        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': CONFIG,
            'epoch': epoch,
            'metrics': history['losses'][-1]
        }
        torch.save(checkpoint, f"checkpoints/epoch_{epoch+1}.pt")

    # Save final model
    torch.save(model.state_dict(), "checkpoints/final_model.pt")

    # Save history
    with open("results/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    with open("results/training_config.json", 'w') as f:
        config_to_save = {k: str(v) if not isinstance(v, (int, float)) else v for k, v in CONFIG.items()}
        json.dump(config_to_save, f, indent=2)

    print("\n✓ Training complete!")
    return model, history

if __name__ == '__main__':
    main()
