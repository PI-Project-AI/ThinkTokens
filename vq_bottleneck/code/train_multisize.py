#!/usr/bin/env python3
"""Multi-scale training script for VQ models across different model sizes."""

import torch
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
from pathlib import Path
from vq_model_v2 import VQLanguageModel
import sys

# Model configurations for different sizes
MODEL_CONFIGS = {
    "160M": {
        "model_name": "EleutherAI/pythia-160m",
        "batch_size": 8,
        "learning_rate": 2e-5,
    },
    "410M": {
        "model_name": "EleutherAI/pythia-410m",
        "batch_size": 4,
        "learning_rate": 2e-5,
    },
    "1.4B": {
        "model_name": "EleutherAI/pythia-1.4b",
        "batch_size": 1,
        "learning_rate": 1e-5,  # Lower LR for larger model
    },
    "2.8B": {
        "model_name": "EleutherAI/pythia-2.8b",
        "batch_size": 1,
        "learning_rate": 5e-6,  # Even lower for very large model
    },
}

DATASET_CONFIGS = {
    "quick": {
        "max_samples": 500,
        "epochs": 2,
        "max_length": 256,
    },
    "medium": {
        "max_samples": 2000,
        "epochs": 3,
        "max_length": 384,
    },
    "full": {
        "max_samples": 7473,
        "epochs": 3,
        "max_length": 512,
    },
}

def get_config(model_size, dataset_size):
    """Get combined configuration."""
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(MODEL_CONFIGS.keys())}")
    if dataset_size not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset size: {dataset_size}. Choose from {list(DATASET_CONFIGS.keys())}")

    config = {**MODEL_CONFIGS[model_size], **DATASET_CONFIGS[dataset_size]}
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['num_codes'] = 512
    config['model_size'] = model_size
    config['dataset_size'] = dataset_size
    return config

def create_results_dir(model_size):
    """Create results directory for specific model size."""
    results_dir = Path(f"results_{model_size}")
    results_dir.mkdir(exist_ok=True)
    return results_dir

def create_checkpoints_dir(model_size):
    """Create checkpoints directory for specific model size."""
    checkpoints_dir = Path(f"checkpoints_{model_size}")
    checkpoints_dir.mkdir(exist_ok=True)
    return checkpoints_dir

def preprocess_function(examples, tokenizer, max_length):
    """Preprocess dataset examples."""
    texts = []
    for q, a in zip(examples['question'], examples['answer']):
        text = f"Question: {q}\nAnswer: {a}"
        texts.append(text)

    encodings = tokenizer(texts, truncation=True, max_length=max_length, padding='max_length')
    encodings['labels'] = encodings['input_ids'].copy()
    return encodings

def train_epoch(model, train_loader, optimizer, scheduler, epoch, config, use_amp=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    lm_loss_total = 0
    vq_loss_total = 0
    code_usage_total = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")

    for batch in pbar:
        input_ids = batch['input_ids'].to(config['device'])
        attention_mask = batch['attention_mask'].to(config['device'])
        labels = batch['labels'].to(config['device'])

        optimizer.zero_grad()

        # Mixed precision training with bfloat16 (no scaler needed for bfloat16)
        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss']

            # Backward (no scaling needed for bfloat16)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            # Standard training
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Metrics
        total_loss += loss.item()
        if outputs['lm_loss'] is not None:
            lm_loss_total += outputs['lm_loss'].item()
        vq_loss_total += outputs['vq_loss'].item()
        code_usage_total += outputs['code_usage_pct']
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'codes': f"{outputs['code_usage']:.0f}/{config['num_codes']}"
        })

    avg_loss = total_loss / num_batches
    avg_lm_loss = lm_loss_total / num_batches if lm_loss_total > 0 else 0
    avg_vq_loss = vq_loss_total / num_batches
    avg_code_usage = code_usage_total / num_batches

    return {
        'avg_loss': avg_loss,
        'avg_lm_loss': avg_lm_loss,
        'avg_vq_loss': avg_vq_loss,
        'avg_code_usage': avg_code_usage
    }

def train_model(model_size, dataset_size="quick"):
    """Train VQ model at specified scale."""

    config = get_config(model_size, dataset_size)
    results_dir = create_results_dir(model_size)
    checkpoints_dir = create_checkpoints_dir(model_size)

    print(f"\n{'='*80}")
    print(f"TRAINING VQ MODEL - {model_size} ({dataset_size} dataset)")
    print(f"{'='*80}")
    print(f"Model: {config['model_name']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Max Samples: {config['max_samples']}")
    print(f"Device: {config['device']}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("gsm8k", "main")['train'].select(range(min(config['max_samples'], 7473)))

    # Preprocess
    print("Preprocessing dataset...")
    def preprocess_fn(examples):
        return preprocess_function(examples, tokenizer, config['max_length'])

    train_dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Processing"
    )
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    print(f"Dataset: {len(train_dataset)} samples, {len(dataloader)} batches per epoch\n")

    # Initialize model
    print("Initializing model...")
    # Enable gradient checkpointing for large models (1.4B+)
    use_checkpointing = model_size in ["1.4B", "2.8B"]
    model = VQLanguageModel(
        config['model_name'],
        config['num_codes'],
        use_gradient_checkpointing=use_checkpointing
    ).to(config['device'])

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=len(dataloader) * config['epochs']
    )

    # No gradient scaler needed for bfloat16 (it has better numerical stability than float16)
    # Scaler is only needed for float16
    if use_checkpointing:
        print("Mixed precision training enabled (bfloat16)\n")

    # Training loop
    history = {'epochs': [], 'losses': []}

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        metrics = train_epoch(model, dataloader, optimizer, scheduler, epoch, config, use_checkpointing)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {metrics['avg_loss']:.4f}")
        print(f"  LM Loss: {metrics['avg_lm_loss']:.4f}")
        print(f"  VQ Loss: {metrics['avg_vq_loss']:.4f}")
        print(f"  Code Usage: {metrics['avg_code_usage']:.1f}%")

        history['epochs'].append(epoch + 1)
        history['losses'].append({
            'total_loss': metrics['avg_loss'],
            'lm_loss': metrics['avg_lm_loss'],
            'vq_loss': metrics['avg_vq_loss'],
            'code_usage': metrics['avg_code_usage']
        })

        # Save checkpoint
        checkpoint_path = checkpoints_dir / f"epoch_{epoch+1}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'epoch': epoch,
            'metrics': metrics
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_model_path = checkpoints_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved: {final_model_path}")

    # Save history
    history_path = results_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save config
    config_path = results_dir / "training_config.json"
    with open(config_path, 'w') as f:
        config_to_save = {
            'model_name': config['model_name'],
            'model_size': config['model_size'],
            'dataset_size': config['dataset_size'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'epochs': config['epochs'],
            'max_samples': config['max_samples'],
            'num_codes': config['num_codes']
        }
        json.dump(config_to_save, f, indent=2)

    print(f"\n✓ Training complete for {model_size}!")
    print(f"Results saved to: {results_dir}")
    print(f"Checkpoints saved to: {checkpoints_dir}")

    return model, history

def main():
    parser = argparse.ArgumentParser(description="Multi-scale VQ model training")
    parser.add_argument("--model", type=str, default="410M",
                       choices=list(MODEL_CONFIGS.keys()) + ["all"],
                       help="Model size to train")
    parser.add_argument("--dataset", type=str, default="quick",
                       choices=list(DATASET_CONFIGS.keys()),
                       help="Dataset size (quick/medium/full)")

    args = parser.parse_args()

    if args.model == "all":
        print("Training all model sizes...")
        for model_size in ["160M", "410M", "1.4B"]:
            try:
                train_model(model_size, args.dataset)
            except Exception as e:
                print(f"Error training {model_size}: {e}")
                continue
        print("\n✓ All models trained!")
    else:
        train_model(args.model, args.dataset)

if __name__ == '__main__':
    main()
