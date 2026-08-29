#!/usr/bin/env python3
"""Training script for VQ reasoning model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import json
from pathlib import Path
from vq_model import VQReasoningModel

# Configuration
CONFIG = {
    'model_name': "EleutherAI/pythia-410m",
    'num_codes': 512,
    'batch_size': 8,
    'epochs': 3,
    'learning_rate': 5e-5,
    'max_length': 512,
    'warmup_steps': 100,
    'accumulation_steps': 1,
    'checkpoint_interval': 500,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

def preprocess_function(examples):
    """Preprocess dataset examples."""
    texts = []
    for q, a in zip(examples['question'], examples['answer']):
        # Format: Question + Answer with separator
        text = f"Question: {q}\nAnswer: {a}"
        texts.append(text)

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=CONFIG['max_length'],
        padding='max_length',
        return_tensors='pt'
    )

    # For language modeling, labels = input_ids
    encodings['labels'] = encodings['input_ids'].clone()

    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': encodings['labels']
    }

def train_epoch(model, train_loader, optimizer, scheduler, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    lm_loss_total = 0
    vq_loss_total = 0
    code_usage_total = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(CONFIG['device'])
        attention_mask = batch['attention_mask'].to(CONFIG['device'])
        labels = batch['labels'].to(CONFIG['device'])

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs['loss']
        lm_loss = outputs['lm_loss']
        vq_loss = outputs['vq_loss']

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % CONFIG['accumulation_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Metrics
        total_loss += loss.item()
        lm_loss_total += lm_loss.item() if lm_loss is not None else 0
        vq_loss_total += vq_loss.item()
        code_usage_total += outputs['vq_metrics']['code_usage_pct'].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'lm_loss': lm_loss.item() if lm_loss is not None else 0,
            'vq_loss': vq_loss.item(),
            'codes': f"{outputs['vq_metrics']['code_usage'].item():.0f}/{CONFIG['num_codes']}"
        })

        # Checkpoint
        if (batch_idx + 1) % CONFIG['checkpoint_interval'] == 0:
            checkpoint_path = Path(f"checkpoints/epoch_{epoch+1}_step_{batch_idx+1}.pt")
            checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'step': batch_idx
            }, checkpoint_path)

    avg_loss = total_loss / num_batches
    avg_lm_loss = lm_loss_total / num_batches
    avg_vq_loss = vq_loss_total / num_batches
    avg_code_usage = code_usage_total / num_batches

    return {
        'avg_loss': avg_loss,
        'avg_lm_loss': avg_lm_loss,
        'avg_vq_loss': avg_vq_loss,
        'avg_code_usage': avg_code_usage
    }

def train():
    """Main training loop."""
    global tokenizer
    print(f"Device: {CONFIG['device']}")
    print(f"Config: {json.dumps({k: str(v) if not isinstance(v, (int, float)) else v for k, v in CONFIG.items()}, indent=2)}\n")

    # Setup
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    # Load tokenizer and dataset
    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("gsm8k", "main")

    # Use smaller dataset for faster testing (can be increased)
    print("Using reduced dataset for faster training...")
    train_dataset = dataset['train'].select(range(min(1000, len(dataset['train']))))

    # Process dataset
    print("Processing dataset...")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Processing train set"
    )
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0
    )

    print(f"Dataset size: {len(train_dataset)}")
    print(f"Num batches per epoch: {len(train_loader)}\n")

    # Model
    print("Initializing VQ model...")
    model = VQReasoningModel(
        CONFIG['model_name'],
        num_codes=CONFIG['num_codes']
    ).to(CONFIG['device'])

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=0.01
    )

    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG['warmup_steps'],
        num_training_steps=total_steps
    )

    # Training loop
    training_history = {
        'epochs': [],
        'losses': []
    }

    for epoch in range(CONFIG['epochs']):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"{'='*70}")

        epoch_metrics = train_epoch(model, train_loader, optimizer, scheduler, epoch)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Loss: {epoch_metrics['avg_loss']:.4f}")
        print(f"  Avg LM Loss: {epoch_metrics['avg_lm_loss']:.4f}")
        print(f"  Avg VQ Loss: {epoch_metrics['avg_vq_loss']:.4f}")
        print(f"  Avg Code Usage: {epoch_metrics['avg_code_usage']:.1f}%")

        training_history['epochs'].append(epoch + 1)
        training_history['losses'].append({
            'total_loss': epoch_metrics['avg_loss'],
            'lm_loss': epoch_metrics['avg_lm_loss'],
            'vq_loss': epoch_metrics['avg_vq_loss'],
            'code_usage': epoch_metrics['avg_code_usage']
        })

        # Save checkpoint
        checkpoint_path = checkpoints_dir / f"epoch_{epoch+1}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': CONFIG,
            'epoch': epoch,
            'metrics': epoch_metrics
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_model_path = checkpoints_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved: {final_model_path}")

    # Save training history
    history_path = results_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved: {history_path}")

    # Save config
    config_path = results_dir / "training_config.json"
    with open(config_path, 'w') as f:
        config_to_save = {k: str(v) if not isinstance(v, (int, float)) else v for k, v in CONFIG.items()}
        json.dump(config_to_save, f, indent=2)

    return model, training_history

if __name__ == '__main__':
    model, history = train()
    print(f"\nTraining complete!")
