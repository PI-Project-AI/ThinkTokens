"""
Training script for Seed + Emergent IR-CoT model.

Implements:
- Two-pass training (Input → IR → Answer)
- Combined losses (CE, VQ, cycle, coverage)
- Causality diagnostics
- Checkpoint saving
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
import argparse

# Import model components
from tokenizer_utils import extend_tokenizer_for_ir
from models.causal_ir_model import CausalIRModel
from evaluation.causal_tests import CausalityTester
from evaluation.answer_matching import exact_match
from ir_grammar import validate_ir_integrity, IRGrammarEnforcer


class ArithmeticDataset(Dataset):
    """Dataset for arithmetic problems."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        """
        Args:
            data_path: Path to JSON file with examples
            tokenizer: Extended tokenizer
            max_length: Maximum sequence length
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # Tokenize problem
        problem_tokens = self.tokenizer(
            example['problem'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize answer
        answer_tokens = self.tokenizer(
            example['answer'],
            max_length=20,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize snippet for local cycle
        snippet_tokens = self.tokenizer(
            example['snippet'],
            max_length=10,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': problem_tokens['input_ids'].squeeze(0),
            'attention_mask': problem_tokens['attention_mask'].squeeze(0),
            'answer_ids': answer_tokens['input_ids'].squeeze(0),
            'snippet_ids': snippet_tokens['input_ids'].squeeze(0)
        }


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch: int
):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    loss_components = {
        'answer_loss': 0,
        'ir_lm_loss': 0,
        'vq_loss': 0,
        'cycle_loss': 0,
        'coverage_loss': 0,
        'diversity_loss': 0
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        answer_ids = batch['answer_ids'].to(device)
        snippet_ids = batch['snippet_ids'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            answer_ids=answer_ids,
            input_snippets=snippet_ids,
            mode='train'
        )

        loss = outputs['total_loss']
        breakdown = outputs['loss_breakdown']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Accumulate losses
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += breakdown[key]

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'ans': f"{breakdown['answer_loss']:.4f}",
            'vq': f"{breakdown['vq_loss']:.4f}"
        })

    # Average losses
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return avg_loss, loss_components


def evaluate(model, dataloader, device, tokenizer):
    """Evaluate model on validation set with exact matching and IR integrity."""
    model.eval()

    total_loss = 0
    correct = 0
    total = 0
    all_ir_buffers = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            answer_ids = batch['answer_ids'].to(device)

            # Forward pass for loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                answer_ids=answer_ids,
                mode='train'
            )

            total_loss += outputs['total_loss'].item()
            ir_buffer = outputs['ir_token_ids']
            all_ir_buffers.append(ir_buffer)

            # Generate answers for accuracy
            gen_outputs = model.generate_answer(input_ids, attention_mask, max_answer_length=10)
            pred_answer_ids = gen_outputs['answer_ids']

            # Decode and compare with exact matching
            for j in range(pred_answer_ids.shape[0]):
                pred_text = tokenizer.decode(
                    pred_answer_ids[j],
                    skip_special_tokens=True
                ).strip()

                true_text = tokenizer.decode(
                    answer_ids[j],
                    skip_special_tokens=True
                ).strip()

                matches, _, _ = exact_match(pred_text, true_text)
                if matches:
                    correct += 1
                total += 1

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0

    # Check IR integrity
    if all_ir_buffers:
        all_ir = torch.cat(all_ir_buffers, dim=0)
        integrity = validate_ir_integrity(
            all_ir,
            model.ir_token_ids,
            min_codes_per_span=3,
            max_codes_per_span=6
        )
        ir_error_rate = integrity['error_rate']
    else:
        ir_error_rate = 0.0

    return avg_loss, accuracy, ir_error_rate


def main(args):
    """Main training loop."""
    print("="*60)
    print("IR-CoT Training")
    print("="*60)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Setup tokenizer
    print("\nSetting up tokenizer...")
    tokenizer, ir_token_ids = extend_tokenizer_for_ir(
        base_model_name=args.model_name,
        num_codes=args.num_codes
    )

    # Initialize model
    print(f"\nInitializing model: {args.model_name}")
    model = CausalIRModel(
        base_model_name=args.model_name,
        ir_token_ids=ir_token_ids,
        num_codes=args.num_codes,
        code_dim=args.code_dim,
        snippet_length=args.snippet_length,
        use_local_cycle=args.use_local_cycle,
        cycle_weight=args.cycle_weight,
        vq_weight=args.vq_weight,
        coverage_weight=args.coverage_weight
    )
    model.to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup datasets
    print("\nLoading datasets...")
    train_dataset = ArithmeticDataset(args.train_data, tokenizer)
    val_dataset = ArithmeticDataset(args.val_data, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    best_val_loss = float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.num_epochs} ---")

        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        print(f"\nTrain Loss: {train_loss:.4f}")
        print("Loss components:")
        for key, val in train_components.items():
            print(f"  {key}: {val:.4f}")

        # Validate
        val_loss, val_acc, ir_error_rate = evaluate(model, val_loader, device, tokenizer)
        print(f"\nVal Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.2%}")
        print(f"IR Error Rate: {ir_error_rate:.2%}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "best_model.pt"

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'args': vars(args)
            }, checkpoint_path)

            print(f"Saved best model to {checkpoint_path}")

        # Run causality tests every N epochs
        if epoch % args.test_frequency == 0:
            print("\n--- Running Causality Tests ---")
            tester = CausalityTester(model, tokenizer, ir_token_ids)

            # Load small test set
            test_data_path = args.test_data if args.test_data else args.val_data
            with open(test_data_path, 'r') as f:
                test_data = json.load(f)[:100]  # Use first 100 for quick test

            results = tester.run_all_tests(test_data, batch_size=8)

            # Save results
            results_path = output_dir / f"causality_tests_epoch{epoch}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IR-CoT model")

    # Model args
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m',
                       help='Base model name')
    parser.add_argument('--num_codes', type=int, default=512,
                       help='Number of VQ codes')
    parser.add_argument('--code_dim', type=int, default=128,
                       help='Code embedding dimension')
    parser.add_argument('--snippet_length', type=int, default=10,
                       help='Snippet length for local cycle')

    # Training args
    parser.add_argument('--train_data', type=str,
                       default='../data/arithmetic/train.json',
                       help='Path to training data')
    parser.add_argument('--val_data', type=str,
                       default='../data/arithmetic/val.json',
                       help='Path to validation data')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test data (optional)')

    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')

    # Loss weights
    parser.add_argument('--use_local_cycle', action='store_true', default=True,
                       help='Use local cycle loss')
    parser.add_argument('--cycle_weight', type=float, default=0.05,
                       help='Weight for cycle loss')
    parser.add_argument('--vq_weight', type=float, default=0.1,
                       help='Weight for VQ loss')
    parser.add_argument('--coverage_weight', type=float, default=0.02,
                       help='Weight for coverage loss')

    # Output args
    parser.add_argument('--output_dir', type=str,
                       default='../checkpoints/ir_cot_v1',
                       help='Output directory for checkpoints')
    parser.add_argument('--test_frequency', type=int, default=5,
                       help='Run causality tests every N epochs')

    args = parser.parse_args()

    main(args)
