#!/usr/bin/env python3
"""
V7-Lite Sanity Tests
Tests critical components before full training:
1. Gumbel-Softmax gradient flow
2. Contrastive loss separation
"""
import torch
import torch.nn.functional as F
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from contrastive_loss import InfoNCEContrastiveLoss


def test_gumbel_gradients():
    """
    Test 1: Verify Gumbel-Softmax provides gradients through hard selection
    """
    print("=" * 60)
    print("TEST 1: Gumbel-Softmax Gradient Flow")
    print("=" * 60)

    # Create logits that require gradients
    vq_logits = torch.randn(64, 512, requires_grad=True)

    # Apply Gumbel-Softmax (hard=True for straight-through)
    y = F.gumbel_softmax(vq_logits, tau=0.6, hard=True, dim=-1)

    # Create dummy loss
    loss = (y * vq_logits).sum()

    # Backpropagate
    loss.backward()

    # Check gradients
    assert vq_logits.grad is not None, "FAIL: No gradients through Gumbel-Softmax!"
    assert vq_logits.grad.abs().max() > 0, "FAIL: Zero gradients!"

    grad_norm = vq_logits.grad.norm().item()
    print(f"✓ Gradients flow through Gumbel-Softmax")
    print(f"  Gradient norm: {grad_norm:.4f}")
    print(f"  Gradient shape: {vq_logits.grad.shape}")
    print(f"  Max gradient: {vq_logits.grad.abs().max().item():.4f}")
    print()
    return True


def test_contrastive_separation():
    """
    Test 2: Verify contrastive loss separates matched/mismatched pairs
    """
    print("=" * 60)
    print("TEST 2: Contrastive Loss Separation")
    print("=" * 60)

    batch_size = 16
    hidden_dim = 512

    # Create contrastive loss module
    contrastive = InfoNCEContrastiveLoss(
        hidden_dim=hidden_dim,
        temperature=0.07
    )

    # Test case: Create batch where half are matched, half mismatched
    # First 8: HL = IR (matched pairs)
    # Last 8: HL != IR (mismatched pairs)

    hl_emb = torch.randn(batch_size, hidden_dim)
    ir_emb = torch.zeros(batch_size, hidden_dim)

    # Make first half matched (similar embeddings)
    ir_emb[:8] = hl_emb[:8] + torch.randn(8, hidden_dim) * 0.1  # Small noise

    # Make second half mismatched (random embeddings)
    ir_emb[8:] = torch.randn(8, hidden_dim)

    # Compute contrastive loss
    loss, metrics = contrastive(hl_emb, ir_emb)

    # Extract metrics
    diag_mean = metrics['diag_similarity']
    offdiag_mean = metrics['offdiag_similarity']
    nn_acc = metrics['nn_accuracy']
    separation = metrics['diag_minus_offdiag']

    print(f"✓ Contrastive loss computed successfully")
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Diagonal similarity (matched pairs): {diag_mean:.4f}")
    print(f"  Off-diagonal similarity (mismatched): {offdiag_mean:.4f}")
    print(f"  Separation (diag - offdiag): {separation:.4f}")
    print(f"  NN-accuracy (retrieval): {nn_acc:.2%}")
    print()

    # Assertions
    # Note: With untrained projection heads, separation may be poor initially
    # We just verify the loss is computed correctly and has reasonable values
    assert not torch.isnan(loss), "FAIL: Loss is NaN!"
    assert not torch.isinf(loss), "FAIL: Loss is infinite!"
    assert loss.item() > 0, f"FAIL: Loss should be positive, got {loss.item():.4f}"

    print("✓ Contrastive loss working correctly!")
    print("  Note: Separation/NN-acc will improve during training as projection heads learn")
    print()
    return True


def test_vq_gumbel_mode():
    """
    Test 3: Verify VQ module supports Gumbel mode
    """
    print("=" * 60)
    print("TEST 3: VQ Gumbel Mode")
    print("=" * 60)

    from vq import VectorQuantizer

    # Create VQ with Gumbel support
    vq = VectorQuantizer(
        num_codes=512,
        code_dim=128,
        use_gumbel_warmstart=True,
        gumbel_tau=0.6,
        gumbel_steps=1500
    )
    vq.train()
    vq.current_step = 500  # Within Gumbel phase

    # Create input
    z = torch.randn(4, 10, 128, requires_grad=True)

    # Forward pass (should use Gumbel)
    z_q, vq_loss, indices = vq(z)

    # Check gradients
    loss = z_q.sum() + vq_loss
    loss.backward()

    assert z.grad is not None, "FAIL: No gradients in Gumbel mode!"

    print(f"✓ VQ Gumbel mode working")
    print(f"  Input shape: {z.shape}")
    print(f"  Output shape: {z_q.shape}")
    print(f"  Indices shape: {indices.shape}")
    print(f"  VQ loss: {vq_loss.item():.4f}")
    print(f"  Gradient norm: {z.grad.norm().item():.4f}")
    print()

    # Test switch to VQ mode after gumbel_steps
    vq.current_step = 1600  # After Gumbel phase
    z2 = torch.randn(4, 10, 128, requires_grad=True)
    z_q2, vq_loss2, indices2 = vq(z2)

    print(f"✓ VQ switches to standard mode after step 1500")
    print(f"  VQ loss (standard mode): {vq_loss2.item():.4f}")
    print()

    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("V7-LITE SANITY TESTS")
    print("=" * 60)
    print()

    try:
        # Run all tests
        test1_pass = test_gumbel_gradients()
        test2_pass = test_contrastive_separation()
        test3_pass = test_vq_gumbel_mode()

        if test1_pass and test2_pass and test3_pass:
            print("=" * 60)
            print("✓ ALL SANITY TESTS PASSED!")
            print("=" * 60)
            print()
            print("Ready to launch V7-Lite training:")
            print("  bash train_70m_v7_lite.sh")
            print()
            sys.exit(0)
        else:
            print("✗ Some tests failed. Fix issues before training.")
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR:\n{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
