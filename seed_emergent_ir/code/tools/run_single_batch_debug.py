#!/usr/bin/env python3
"""
Single-Batch Deterministic Debug (P0-P6)
Isolates pipeline breakage with comprehensive probes.
"""
import torch
import torch.nn.functional as F
import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from models.causal_ir_model_v2 import CausalIRModelV2
from ir_grammar import IRGrammarEnforcer
import argparse


def setup_deterministic(seed=42):
    """Enable full determinism."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def p0_special_token_sanity(tokenizer, ir_token_ids):
    """P0: Verify all special token IDs are correct and unique."""
    print("\n" + "="*60)
    print("P0: Special Token Sanity Check")
    print("="*60)

    results = {
        "passed": True,
        "tokens": {},
        "errors": []
    }

    # Collect all token IDs
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    results["tokens"]["PAD"] = pad_id
    results["tokens"]["EOS"] = eos_id
    results["tokens"]["IR_START"] = ir_token_ids["ir_start"]
    results["tokens"]["IR_END"] = ir_token_ids["ir_end"]
    results["tokens"]["CODE_START"] = ir_token_ids["code_start"]
    results["tokens"]["CODE_END"] = ir_token_ids["code_end"]

    # Tag tokens
    for tag in ["goal", "assume", "step", "check", "branch"]:
        results["tokens"][f"{tag.upper()}_OPEN"] = ir_token_ids[tag]
        results["tokens"][f"{tag.upper()}_CLOSE"] = ir_token_ids[f"{tag}_end"]

    print(f"PAD: {pad_id}")
    print(f"EOS: {eos_id}")
    print(f"IR_START: {ir_token_ids['ir_start']}")
    print(f"IR_END: {ir_token_ids['ir_end']}")
    print(f"CODE range: [{ir_token_ids['code_start']}, {ir_token_ids['code_end']}]")

    # Assert uniqueness
    critical_ids = [pad_id, eos_id, ir_token_ids["ir_start"], ir_token_ids["ir_end"]]
    if len(critical_ids) != len(set(critical_ids)):
        error = f"FAIL: Critical tokens not unique! {critical_ids}"
        print(error)
        results["errors"].append(error)
        results["passed"] = False
    else:
        print("✓ Critical tokens are unique")

    # Assert IR_END != EOS != PAD
    if ir_token_ids["ir_end"] == eos_id:
        error = f"FAIL: IR_END ({ir_token_ids['ir_end']}) == EOS ({eos_id})"
        print(error)
        results["errors"].append(error)
        results["passed"] = False
    else:
        print(f"✓ IR_END ({ir_token_ids['ir_end']}) != EOS ({eos_id})")

    if eos_id == pad_id:
        error = f"FAIL: EOS ({eos_id}) == PAD ({pad_id})"
        print(error)
        results["errors"].append(error)
        results["passed"] = False
    else:
        print(f"✓ EOS ({eos_id}) != PAD ({pad_id})")

    status = "PASSED" if results["passed"] else "FAILED"
    print(f"\nP0 Status: {status}")

    return results


def p1_fsm_reachability(grammar, ir_token_ids, vocab_size):
    """P1: Verify IR_END can be reached via FSM."""
    print("\n" + "="*60)
    print("P1: FSM Reachability Test")
    print("="*60)

    results = {
        "passed": True,
        "trace": [],
        "errors": []
    }

    # Simulate: <IR_START> → (open → CODE{3} → close){3} → <IR_END>
    sequence = [ir_token_ids["ir_start"]]

    # Add 3 spans with 3 codes each
    for span_idx in range(3):
        # Open tag (use GOAL)
        sequence.append(ir_token_ids["goal"])

        # Add 3 codes
        code_start = ir_token_ids["code_start"]
        for code_idx in range(3):
            sequence.append(code_start + code_idx)

        # Close tag
        sequence.append(ir_token_ids["goal_end"])

    # Now check if IR_END is allowed
    allowed_mask = grammar.get_valid_next_tokens(sequence, vocab_size)

    ir_end_allowed = allowed_mask[ir_token_ids["ir_end"]].item()

    results["sequence_length"] = len(sequence)
    results["ir_end_allowed"] = bool(ir_end_allowed)
    results["allowed_count"] = allowed_mask.sum().item()

    print(f"Simulated sequence length: {len(sequence)}")
    print(f"Sequence: {sequence[:10]}... (truncated)")
    print(f"Allowed tokens at terminal: {allowed_mask.sum().item()}")
    print(f"IR_END ({ir_token_ids['ir_end']}) allowed: {ir_end_allowed}")

    if not ir_end_allowed:
        error = "FAIL: IR_END not in allowed set at terminal state!"
        print(error)
        results["errors"].append(error)
        results["passed"] = False

        # Debug: show what IS allowed
        allowed_ids = torch.where(allowed_mask)[0].tolist()
        print(f"Allowed IDs: {allowed_ids[:20]}... (truncated)")
    else:
        print("✓ IR_END is reachable")

    status = "PASSED" if results["passed"] else "FAILED"
    print(f"\nP1 Status: {status}")

    return results


def p2_mask_application_test(grammar, ir_token_ids, vocab_size):
    """P2: Verify masks are applied correctly to logits."""
    print("\n" + "="*60)
    print("P2: Mask Application Test")
    print("="*60)

    results = {
        "passed": True,
        "tests": [],
        "errors": []
    }

    # Test 3 states: code slot, tag close, terminal

    # Test 1: After open tag (should allow codes)
    seq1 = [ir_token_ids["ir_start"], ir_token_ids["goal"]]
    mask1 = grammar.get_valid_next_tokens(seq1, vocab_size)

    logits1 = torch.randn(vocab_size)
    logits1[~mask1] = float('-inf')
    probs1 = F.softmax(logits1, dim=-1)

    # Check that disallowed tokens have ~0 prob
    disallowed_prob = probs1[~mask1].max().item()

    test1 = {
        "state": "after_open_tag",
        "allowed_count": mask1.sum().item(),
        "max_disallowed_prob": disallowed_prob,
        "passed": disallowed_prob < 1e-6
    }
    results["tests"].append(test1)

    print(f"Test 1 (after open tag):")
    print(f"  Allowed: {mask1.sum().item()}/{vocab_size}")
    print(f"  Max disallowed prob: {disallowed_prob:.2e}")
    print(f"  Status: {'PASS' if test1['passed'] else 'FAIL'}")

    if not test1["passed"]:
        results["passed"] = False
        results["errors"].append("Test 1 failed: disallowed tokens have non-zero prob")

    # Test 2: After codes (should allow close tag)
    seq2 = [ir_token_ids["ir_start"], ir_token_ids["goal"]]
    for i in range(3):
        seq2.append(ir_token_ids["code_start"] + i)

    mask2 = grammar.get_valid_next_tokens(seq2, vocab_size)

    close_allowed = mask2[ir_token_ids["goal_end"]].item()

    test2 = {
        "state": "after_codes",
        "close_tag_allowed": bool(close_allowed),
        "allowed_count": mask2.sum().item(),
        "passed": close_allowed
    }
    results["tests"].append(test2)

    print(f"\nTest 2 (after codes):")
    print(f"  Close tag allowed: {close_allowed}")
    print(f"  Status: {'PASS' if test2['passed'] else 'FAIL'}")

    if not test2["passed"]:
        results["passed"] = False
        results["errors"].append("Test 2 failed: close tag not allowed after codes")

    # Test 3: After min spans (should allow IR_END)
    seq3 = [ir_token_ids["ir_start"]]
    for _ in range(3):
        seq3.append(ir_token_ids["goal"])
        for i in range(3):
            seq3.append(ir_token_ids["code_start"] + i)
        seq3.append(ir_token_ids["goal_end"])

    mask3 = grammar.get_valid_next_tokens(seq3, vocab_size)
    ir_end_allowed = mask3[ir_token_ids["ir_end"]].item()

    test3 = {
        "state": "after_min_spans",
        "ir_end_allowed": bool(ir_end_allowed),
        "allowed_count": mask3.sum().item(),
        "passed": ir_end_allowed
    }
    results["tests"].append(test3)

    print(f"\nTest 3 (after min spans):")
    print(f"  IR_END allowed: {ir_end_allowed}")
    print(f"  Status: {'PASS' if test3['passed'] else 'FAIL'}")

    if not test3["passed"]:
        results["passed"] = False
        results["errors"].append("Test 3 failed: IR_END not allowed after min spans")

    status = "PASSED" if results["passed"] else "FAILED"
    print(f"\nP2 Status: {status}")

    return results


def p3_greedy_decode_trace(model, tokenizer, input_ids, ir_token_ids, vocab_size, device):
    """P3: Trace real greedy IR generation step-by-step."""
    print("\n" + "="*60)
    print("P3: Greedy Decode Trace (1 sample)")
    print("="*60)

    results = {
        "passed": True,
        "steps": [],
        "errors": [],
        "ir_end_ever_allowed": False,
        "ir_end_chosen": False
    }

    # Use first sample only
    input_ids_single = input_ids[0:1].to(device)

    # Generate IR
    with torch.no_grad():
        ir_output = model.ir_generator._generate_ir(
            input_ids_single,
            attention_mask=None,
            temperature=1.0,
            max_length=50
        )

    ir_buffer = ir_output["ir_token_ids"][0].cpu().tolist()

    print(f"Generated IR length: {len(ir_buffer)}")
    print(f"IR sequence: {ir_buffer}")

    # Decode tokens
    ir_text = tokenizer.decode(ir_buffer, skip_special_tokens=False)
    print(f"IR text: {ir_text[:200]}...")

    # Check if IR_END present
    ir_end_present = ir_token_ids["ir_end"] in ir_buffer
    results["ir_end_chosen"] = ir_end_present

    print(f"\nIR_END ({ir_token_ids['ir_end']}) in sequence: {ir_end_present}")

    # Re-trace generation to capture step-by-step info
    print("\nRe-tracing with detailed logging...")

    ir_buffer_trace = [ir_token_ids["ir_start"]]
    grammar = model.ir_generator.grammar

    for step in range(min(50, len(ir_buffer))):
        # Get allowed mask
        allowed_mask = grammar.get_valid_next_tokens(ir_buffer_trace, vocab_size)

        ir_end_allowed = allowed_mask[ir_token_ids["ir_end"]].item()
        if ir_end_allowed:
            results["ir_end_ever_allowed"] = True

        # Record step info
        step_info = {
            "step": step,
            "current_length": len(ir_buffer_trace),
            "allowed_count": allowed_mask.sum().item(),
            "ir_end_allowed": bool(ir_end_allowed),
            "chosen_token": ir_buffer[step] if step < len(ir_buffer) else None
        }

        results["steps"].append(step_info)

        if step < 5 or step % 10 == 0:  # Log first 5 and every 10th
            print(f"Step {step}: allowed={allowed_mask.sum().item()}, IR_END_ok={ir_end_allowed}, chose={step_info['chosen_token']}")

        # Check if we chose IR_END
        if step < len(ir_buffer) and ir_buffer[step] == ir_token_ids["ir_end"]:
            print(f"Step {step}: Chose IR_END!")
            break

        # Advance trace
        if step < len(ir_buffer):
            ir_buffer_trace.append(ir_buffer[step])

    # Analysis
    if not results["ir_end_ever_allowed"]:
        error = "FAIL: IR_END was NEVER in allowed set during generation!"
        print(f"\n{error}")
        results["errors"].append(error)
        results["passed"] = False
    elif not results["ir_end_chosen"]:
        error = "FAIL: IR_END was allowed but never chosen!"
        print(f"\n{error}")
        results["errors"].append(error)
        results["passed"] = False
    else:
        print("\n✓ IR_END was allowed and chosen")

    status = "PASSED" if results["passed"] else "FAILED"
    print(f"\nP3 Status: {status}")

    return results


def p4_answer_pad_eos_guard(model, batch, device):
    """P4: Verify answer generation isn't collapsing."""
    print("\n" + "="*60)
    print("P4: Answer PAD/EOS Guard")
    print("="*60)

    results = {
        "passed": True,
        "errors": [],
        "answer_loss": None,
        "num_answer_tokens": 0
    }

    # Forward pass
    input_ids = batch["input_ids"].to(device)
    ir_ids = batch.get("ir_ids")
    answer_ids = batch.get("answer_ids")

    if ir_ids is not None:
        ir_ids = ir_ids.to(device)
    if answer_ids is not None:
        answer_ids = answer_ids.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            target_ir_ids=ir_ids,
            answer_ids=answer_ids,
            mode="train"
        )

    # P4 FIX: answer_loss is in loss_breakdown
    loss_breakdown = outputs.get("loss_breakdown", {})
    answer_loss = loss_breakdown.get("answer_loss", 0.0)
    results["answer_loss"] = answer_loss

    print(f"Answer loss: {answer_loss:.6f}")

    # Check if answer loss is zero
    if answer_loss == 0.0:
        error = "FAIL: Answer loss is exactly 0.0!"
        print(error)
        results["errors"].append(error)
        results["passed"] = False
    else:
        print("✓ Answer loss > 0")

    # Check answer targets
    if answer_ids is not None:
        # Count non-pad tokens
        pad_id = model.tokenizer.pad_token_id
        non_pad = (answer_ids != pad_id).sum().item()
        results["num_answer_tokens"] = non_pad

        print(f"Non-PAD answer tokens: {non_pad}")

        if non_pad == 0:
            error = "FAIL: No non-PAD tokens in answer targets!"
            print(error)
            results["errors"].append(error)
            results["passed"] = False
        else:
            print("✓ Answer has non-PAD tokens")

    status = "PASSED" if results["passed"] else "FAILED"
    print(f"\nP4 Status: {status}")

    return results


def p5_diversity_grad_path(model, batch, device):
    """P5: Verify gradients flow through Gumbel."""
    print("\n" + "="*60)
    print("P5: Diversity/Grad Path (Gumbel)")
    print("="*60)

    results = {
        "passed": True,
        "errors": [],
        "gumbel_active": False,
        "grad_flows": False
    }

    # Enable Gumbel mode
    model.ir_generator.vq.vq.current_step = 500  # Within Gumbel range
    model.current_step = 500

    results["gumbel_active"] = (model.ir_generator.vq.vq.current_step <
                                 model.ir_generator.vq.vq.gumbel_steps)

    print(f"Gumbel active: {results['gumbel_active']}")

    # Forward pass with gradients
    input_ids = batch["input_ids"].to(device)
    ir_ids = batch.get("ir_ids")
    answer_ids = batch.get("answer_ids")

    if ir_ids is not None:
        ir_ids = ir_ids.to(device)
    if answer_ids is not None:
        answer_ids = answer_ids.to(device)

    # P5 FIX: Enable gradients for VQ codebook explicitly
    model.train()
    model.ir_generator.vq.vq.codebook.weight.requires_grad = True
    model.zero_grad()  # Clear any existing gradients

    outputs = model(
        input_ids=input_ids,
        target_ir_ids=ir_ids,
        answer_ids=answer_ids,
        mode="train"
    )

    total_loss = outputs["total_loss"]

    # Backward
    total_loss.backward()

    # Check if VQ embeddings have gradients
    vq_embeddings = model.ir_generator.vq.vq.codebook.weight
    has_grad = vq_embeddings.grad is not None

    results["grad_flows"] = has_grad

    if has_grad:
        grad_norm = vq_embeddings.grad.norm().item()
        print(f"✓ VQ embeddings have gradients (norm={grad_norm:.6f})")
    else:
        error = "FAIL: VQ embeddings have no gradients!"
        print(error)
        results["errors"].append(error)
        results["passed"] = False

    # Check diversity/coverage loss (P5 FIX: in loss_breakdown)
    loss_breakdown = outputs.get("loss_breakdown", {})
    diversity_loss = loss_breakdown.get("diversity_loss", 0.0)
    results["diversity_loss"] = diversity_loss

    print(f"Diversity loss: {diversity_loss:.6f}")

    status = "PASSED" if results["passed"] else "FAILED"
    print(f"\nP5 Status: {status}")

    return results


def p6_tag_supervision(model, batch, device):
    """P6: Verify tag prediction accuracy."""
    print("\n" + "="*60)
    print("P6: Tag Supervision")
    print("="*60)

    results = {
        "passed": True,
        "errors": [],
        "tag_loss": None,
        "tag_accuracy": None
    }

    # Forward pass
    input_ids = batch["input_ids"].to(device)
    ir_ids = batch.get("ir_ids")

    if ir_ids is not None:
        ir_ids = ir_ids.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                target_ir_ids=ir_ids,
                mode="train"
            )

        # P6 FIX: tag_lm_loss is in loss_breakdown
        loss_breakdown = outputs.get("loss_breakdown", {})
        tag_loss = loss_breakdown.get("tag_lm_loss", 0.0)
        results["tag_loss"] = tag_loss

        print(f"Tag LM loss: {tag_loss:.6f}")

        # Check if tag loss is reasonable
        if tag_loss == 0.0:
            error = "FAIL: Tag LM loss is 0.0!"
            print(error)
            results["errors"].append(error)
            results["passed"] = False
        else:
            print("✓ Tag LM loss > 0")
    else:
        print("No IR targets provided, skipping tag supervision test")

    status = "PASSED" if results["passed"] else "FAILED"
    print(f"\nP6 Status: {status}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dump_path", type=str,
                       default="../checkpoints/ir_cot_debug/debug_single_batch.json")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--train_data", type=str, default="../data/arithmetic/train.json")
    args = parser.parse_args()

    print("="*60)
    print("SINGLE-BATCH DETERMINISTIC DEBUG (P0-P6)")
    print("="*60)

    # Setup
    setup_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    # Load tokenizer and add IR tokens
    print("\nLoading tokenizer...")
    from tokenizer_utils import extend_tokenizer_for_ir
    tokenizer, ir_token_ids = extend_tokenizer_for_ir(args.model_name, num_codes=512)
    vocab_size = len(tokenizer)

    print(f"Vocab size: {vocab_size}")

    # Initialize model
    print("\nInitializing model...")
    model = CausalIRModelV2(
        base_model_name=args.model_name,
        ir_token_ids=ir_token_ids,
        num_codes=512,
        code_dim=128,
        pad_token_id=tokenizer.pad_token_id,
        use_contrastive=True,
        contrastive_weight=0.3,
        contrastive_T=0.07,
        use_gumbel_warmstart=True,
        gumbel_tau=0.6,
        gumbel_steps=1500
    )

    # Set tokenizer for model
    model.tokenizer = tokenizer
    model = model.to(device)

    # Load single batch
    print("\nLoading single batch...")
    import json as json_lib
    with open(args.train_data) as f:
        data = json_lib.load(f)

    # Take first 2 examples
    examples = data[:2]

    # Prepare batch
    batch = {
        "input_ids": [],
        "ir_ids": [],
        "answer_ids": []
    }

    for ex in examples:
        # Tokenize input
        input_tokens = tokenizer(ex["problem"], return_tensors="pt", padding=False)
        batch["input_ids"].append(input_tokens["input_ids"][0])

        # P5 FIX: Create realistic IR with code tokens (4 spans, 3 codes each)
        dummy_ir = f"<IR_START><STEP>c000c001c002</STEP><STEP>c003c004c005</STEP><STEP>c006c007c008</STEP><STEP>c009c010c011</STEP><IR_END>"
        ir_tokens = tokenizer(dummy_ir, return_tensors="pt", padding=False)
        batch["ir_ids"].append(ir_tokens["input_ids"][0])

        # Tokenize answer
        answer_tokens = tokenizer(ex["answer"], return_tensors="pt", padding=False)
        batch["answer_ids"].append(answer_tokens["input_ids"][0])

    # Pad batch
    from torch.nn.utils.rnn import pad_sequence
    batch["input_ids"] = pad_sequence(batch["input_ids"], batch_first=True,
                                      padding_value=tokenizer.pad_token_id)
    batch["ir_ids"] = pad_sequence(batch["ir_ids"], batch_first=True,
                                   padding_value=tokenizer.pad_token_id)
    batch["answer_ids"] = pad_sequence(batch["answer_ids"], batch_first=True,
                                       padding_value=tokenizer.pad_token_id)

    print(f"Batch size: {batch['input_ids'].shape[0]}")
    print(f"Input shape: {batch['input_ids'].shape}")

    # Run all probes
    all_results = {}

    # P0: Special token sanity
    all_results["p0_special_tokens"] = p0_special_token_sanity(tokenizer, ir_token_ids)

    # P1: FSM reachability
    grammar = IRGrammarEnforcer(ir_token_ids, min_codes_per_span=3, max_codes_per_span=6,
                                min_spans=3, max_spans=12)
    all_results["p1_fsm_reachability"] = p1_fsm_reachability(grammar, ir_token_ids, vocab_size)

    # P2: Mask application
    all_results["p2_mask_application"] = p2_mask_application_test(grammar, ir_token_ids, vocab_size)

    # P3: Greedy decode trace
    all_results["p3_greedy_decode"] = p3_greedy_decode_trace(
        model, tokenizer, batch["input_ids"], ir_token_ids, vocab_size, device
    )

    # P4: Answer PAD/EOS guard
    all_results["p4_answer_guard"] = p4_answer_pad_eos_guard(model, batch, device)

    # P5: Diversity/grad path
    all_results["p5_diversity_grad"] = p5_diversity_grad_path(model, batch, device)

    # P6: Tag supervision
    all_results["p6_tag_supervision"] = p6_tag_supervision(model, batch, device)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for probe_name, probe_results in all_results.items():
        status = "✓ PASS" if probe_results["passed"] else "✗ FAIL"
        print(f"{probe_name.upper()}: {status}")
        if not probe_results["passed"]:
            all_passed = False
            for error in probe_results.get("errors", []):
                print(f"  - {error}")

    print("\n" + "="*60)
    if all_passed:
        print("ALL PROBES PASSED ✓")
    else:
        print("SOME PROBES FAILED ✗")
    print("="*60)

    # Save results
    dump_path = Path(args.dump_path)
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dump_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {dump_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
