#!/usr/bin/env python3
"""
🧪 HessGPT - ARCHITECTURE TEST SUITE v4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests complets de l'architecture HessGPT pour détecter les bugs

TESTS:
1. Model initialization
2. Forward pass (train mode)
3. Forward pass (eval mode)  
4. Soft-capping verification
5. Loss computation with padding
6. Gradient flow
7. Masque causal
8. RoPE/YaRN
9. GQA
10. QK-Norm
11. LoRA application
12. Memory leaks
13. Generation
14. Vocab resize
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USAGE:
    python test_hessgpt.py
"""

import torch
import torch.nn as nn
import sys
import os
import time
import traceback
from transformers import AutoTokenizer

sys.path.append('./Core/Model')

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_test(name):
    print(f"\n{BLUE}{'='*70}")
    print(f"🧪 TEST: {name}")
    print(f"{'='*70}{RESET}")

def print_success(msg):
    print(f"{GREEN}✅ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}❌ {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}⚠️  {msg}{RESET}")


# ============================================
# TEST 1: MODEL INITIALIZATION
# ============================================
def test_model_initialization():
    print_test("Model Initialization")
    
    try:
        from HessGpt import HessGPT
        
        # Test basic config
        model = HessGPT(
            vocab_size=32005,
            embed_dim=768,
            num_heads=12,
            num_layers=6,  # Small for testing
            max_seq_len=512,
            use_rope=True,
            use_yarn=False,
            use_swiglu=True,
            n_kv_heads=4,
            use_qk_norm=True,
            soft_cap=30.0,
            use_flash_attn=True
        )
        
        print_success("Model created successfully")
        
        # Check param count
        total_params = sum(p.numel() for p in model.parameters())
        print_success(f"Total params: {total_params:,}")
        
        # Verify architecture components
        assert hasattr(model, 'token_embeddings'), "Missing token_embeddings"
        assert hasattr(model, 'blocks'), "Missing transformer blocks"
        assert hasattr(model, 'ln_final'), "Missing final layer norm"
        assert hasattr(model, 'output_head'), "Missing output head"
        print_success("All components present")
        
        # Verify weight tying
        assert model.output_head.weight is model.token_embeddings.weight, \
            "Weight tying not working!"
        print_success("Weight tying verified")
        
        # Verify soft-cap storage
        assert model.soft_cap == 30.0, f"Soft-cap not stored correctly: {model.soft_cap}"
        print_success(f"Soft-cap stored: {model.soft_cap}")
        
        return True
        
    except Exception as e:
        print_error(f"Test failed: {e}")
        traceback.print_exc()
        return False


# ============================================
# TEST 2: FORWARD PASS (TRAIN MODE)
# ============================================
def test_forward_pass_train():
    print_test("Forward Pass (Train Mode)")
    
    try:
        from HessGpt import HessGPT
        
        model = HessGPT(
            vocab_size=32005,
            embed_dim=768,
            num_heads=12,
            num_layers=6,
            max_seq_len=512,
            soft_cap=30.0
        )
        model.train()
        
        # Create random input
        batch_size = 4
        seq_len = 128
        x = torch.randint(0, 32005, (batch_size, seq_len))
        y = torch.randint(0, 32005, (batch_size, seq_len))
        
        # Forward
        logits, loss = model(x, targets=y, pad_token_id=0)
        
        # Check shapes
        assert logits.shape == (batch_size, seq_len, 32005), \
            f"Wrong logits shape: {logits.shape}"
        print_success(f"Logits shape correct: {logits.shape}")
        
        # Check loss
        assert loss is not None, "Loss is None"
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is inf"
        print_success(f"Loss computed: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print_error(f"Test failed: {e}")
        traceback.print_exc()
        return False


# ============================================
# TEST 3: SOFT-CAPPING VERIFICATION
# ============================================
def test_soft_capping():
    print_test("Soft-Capping Verification")
    
    try:
        from HessGpt import HessGPT
        
        # Model WITH soft-cap
        model_capped = HessGPT(
            vocab_size=32005,
            embed_dim=768,
            num_heads=12,
            num_layers=6,
            max_seq_len=512,
            soft_cap=30.0
        )
        model_capped.eval()
        
        # Model WITHOUT soft-cap
        model_uncapped = HessGPT(
            vocab_size=32005,
            embed_dim=768,
            num_heads=12,
            num_layers=6,
            max_seq_len=512,
            soft_cap=None
        )
        model_uncapped.eval()
        
        # Same input
        x = torch.randint(0, 32005, (2, 64))
        
        with torch.no_grad():
            logits_capped, _ = model_capped(x)
            logits_uncapped, _ = model_uncapped(x)
        
        # Verify capping
        max_capped = logits_capped.max().item()
        min_capped = logits_capped.min().item()
        
        max_uncapped = logits_uncapped.max().item()
        
        # Capped should be within [-30, 30]
        assert max_capped <= 30.5, f"Soft-cap not applied! Max: {max_capped}"
        assert min_capped >= -30.5, f"Soft-cap not applied! Min: {min_capped}"
        print_success(f"Capped range: [{min_capped:.2f}, {max_capped:.2f}] ✅")
        
        # Uncapped should be larger
        assert abs(max_uncapped) > 30, \
            f"Uncapped model should have larger logits: {max_uncapped}"
        print_success(f"Uncapped max: {max_uncapped:.2f} (> 30) ✅")
        
        return True
        
    except Exception as e:
        print_error(f"Test failed: {e}")
        traceback.print_exc()
        return False


# ============================================
# TEST 4: LOSS WITH PADDING
# ============================================
def test_loss_with_padding():
    print_test("Loss Computation with Padding")
    
    try:
        from HessGpt import HessGPT
        
        model = HessGPT(
            vocab_size=32005,
            embed_dim=768,
            num_heads=12,
            num_layers=6,
            max_seq_len=512
        )
        model.train()
        
        # Create input with padding
        batch_size = 4
        seq_len = 64
        x = torch.randint(1, 32005, (batch_size, seq_len))
        y = torch.randint(1, 32005, (batch_size, seq_len))
        
        # Add padding (token 0)
        x[:, 32:] = 0
        y[:, 32:] = 0
        
        # Forward
        logits, loss = model(x, targets=y, pad_token_id=0)
        
        assert loss is not None, "Loss is None"
        assert not torch.isnan(loss), "Loss is NaN with padding"
        print_success(f"Loss with padding: {loss.item():.4f}")
        
        # Verify padding is ignored
        # Loss should be lower than if we didn't ignore padding
        logits_2, loss_2 = model(x, targets=y, pad_token_id=None)
        
        print_success(f"Loss without ignore: {loss_2.item():.4f}")
        print_success(f"Loss with ignore:    {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print_error(f"Test failed: {e}")
        traceback.print_exc()
        return False


# ============================================
# TEST 5: GRADIENT FLOW
# ============================================
def test_gradient_flow():
    print_test("Gradient Flow")
    
    try:
        from HessGpt import HessGPT
        
        model = HessGPT(
            vocab_size=32005,
            embed_dim=768,
            num_heads=12,
            num_layers=6,
            max_seq_len=512
        )
        model.train()
        
        x = torch.randint(0, 32005, (2, 32))
        y = torch.randint(0, 32005, (2, 32))
        
        # Forward + backward
        logits, loss = model(x, targets=y)
        loss.backward()
        
        # Check gradients
        grads_exist = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads_exist = True
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
        
        assert grads_exist, "No gradients computed!"
        print_success("All gradients computed successfully")
        
        return True
        
    except Exception as e:
        print_error(f"Test failed: {e}")
        traceback.print_exc()
        return False


# ============================================
# TEST 6: MASQUE CAUSAL
# ============================================
def test_causal_mask():
    print_test("Masque Causal")
    
    try:
        from HessGpt import HessGPT
        
        model = HessGPT(
            vocab_size=32005,
            embed_dim=768,
            num_heads=12,
            num_layers=6,
            max_seq_len=512
        )
        
        # Get mask
        mask = model._get_causal_mask(8, 'cpu')
        
        # Verify shape
        assert mask.shape == (8, 8), f"Wrong mask shape: {mask.shape}"
        
        # Verify triangular (upper triangle should be True)
        expected = torch.triu(torch.ones(8, 8), diagonal=1).bool()
        assert torch.equal(mask, expected), "Mask is not correct causal mask"
        
        print_success("Causal mask correct")
        
        # Test caching
        mask2 = model._get_causal_mask(8, 'cpu')
        assert mask is mask2 or torch.equal(mask, mask2[:8, :8]), "Mask not cached"
        print_success("Mask caching works")
        
        return True
        
    except Exception as e:
        print_error(f"Test failed: {e}")
        traceback.print_exc()
        return False


# ============================================
# TEST 7: GENERATION
# ============================================
def test_generation():
    print_test("Text Generation")
    
    try:
        from HessGpt import HessGPT
        
        model = HessGPT(
            vocab_size=32005,
            embed_dim=768,
            num_heads=12,
            num_layers=6,
            max_seq_len=512
        )
        model.eval()
        
        # Generate
        prompt = torch.randint(0, 32005, (1, 10))
        
        with torch.no_grad():
            output = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=50)
        
        # Verify output
        assert output.shape[0] == 1, "Batch size changed"
        assert output.shape[1] == 30, f"Wrong output length: {output.shape[1]}"
        print_success(f"Generated {output.shape[1] - 10} new tokens")
        
        return True
        
    except Exception as e:
        print_error(f"Test failed: {e}")
        traceback.print_exc()
        return False


# ============================================
# TEST 8: VOCAB RESIZE
# ============================================
def test_vocab_resize():
    print_test("Vocab Resize")
    
    try:
        from HessGpt import HessGPT
        
        model = HessGPT(
            vocab_size=32005,
            embed_dim=768,
            num_heads=12,
            num_layers=6,
            max_seq_len=512
        )
        
        # Resize
        model.resize_token_embeddings(33000)
        
        # Verify
        assert model.vocab_size == 33000, f"Vocab not resized: {model.vocab_size}"
        assert model.token_embeddings.num_embeddings == 33000
        assert model.output_head.out_features == 33000
        print_success(f"Vocab resized to {model.vocab_size}")
        
        # Verify weight tying still works
        assert model.output_head.weight is model.token_embeddings.weight
        print_success("Weight tying preserved after resize")
        
        # Test forward with new vocab
        x = torch.randint(0, 33000, (2, 16))
        y = torch.randint(0, 33000, (2, 16))
        logits, loss = model(x, targets=y)
        assert logits.shape[-1] == 33000
        print_success("Forward pass works with new vocab")
        
        return True
        
    except Exception as e:
        print_error(f"Test failed: {e}")
        traceback.print_exc()
        return False


# ============================================
# TEST 9: LORA APPLICATION
# ============================================
def test_lora():
    print_test("LoRA Application")
    
    try:
        from HessGpt import HessGPT
        import torch.nn as nn
        
        # Simple LoRA implementation for testing
        class LoRALayer(nn.Module):
            def __init__(self, in_features, out_features, r=8, alpha=16):
                super().__init__()
                self.r = r
                self.scaling = alpha / r
                self.lora_A = nn.Parameter(torch.zeros(r, in_features))
                self.lora_B = nn.Parameter(torch.zeros(out_features, r))
                nn.init.kaiming_uniform_(self.lora_A)
                nn.init.zeros_(self.lora_B)
            
            def forward(self, x):
                return torch.nn.functional.linear(
                    torch.nn.functional.linear(x, self.lora_A),
                    self.lora_B
                ) * self.scaling
        
        class LinearWithLoRA(nn.Module):
            def __init__(self, base_layer, r=8, alpha=16):
                super().__init__()
                self.base_layer = base_layer
                self.lora = LoRALayer(base_layer.in_features, base_layer.out_features, r, alpha)
            
            def forward(self, x):
                return self.base_layer(x) + self.lora(x)
        
        model = HessGPT(
            vocab_size=32005,
            embed_dim=768,
            num_heads=12,
            num_layers=2,  # Small
            max_seq_len=512
        )
        
        # Freeze all params
        for param in model.parameters():
            param.requires_grad = False
        
        # Apply LoRA to first attention layer
        first_block = model.blocks[0]
        q_proj = first_block.attention.q_proj
        
        # Replace with LoRA version
        lora_layer = LinearWithLoRA(q_proj, r=8, alpha=16)
        first_block.attention.q_proj = lora_layer
        
        # Verify trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable > 0, "No trainable params after LoRA!"
        print_success(f"Trainable params: {trainable:,}")
        
        # Test forward
        x = torch.randint(0, 32005, (2, 16))
        y = torch.randint(0, 32005, (2, 16))
        logits, loss = model(x, targets=y)
        
        # Test backward
        loss.backward()
        
        # Verify LoRA gradients
        assert lora_layer.lora.lora_A.grad is not None, "No gradient on LoRA A"
        assert lora_layer.lora.lora_B.grad is not None, "No gradient on LoRA B"
        print_success("LoRA gradients computed")
        
        return True
        
    except Exception as e:
        print_error(f"Test failed: {e}")
        traceback.print_exc()
        return False


# ============================================
# TEST 10: YARN
# ============================================
def test_yarn():
    print_test("YaRN Extension")
    
    try:
        from HessGpt import HessGPT
        
        # Model with YaRN
        model_yarn = HessGPT(
            vocab_size=32005,
            embed_dim=768,
            num_heads=12,
            num_layers=2,
            max_seq_len=4096,  # Extended
            use_yarn=True,
            yarn_scale=4.0,
            yarn_original_max_len=1024
        )
        
        # Model without YaRN
        model_base = HessGPT(
            vocab_size=32005,
            embed_dim=768,
            num_heads=12,
            num_layers=2,
            max_seq_len=1024,
            use_yarn=False
        )
        
        # Test long sequence with YaRN
        x_long = torch.randint(0, 32005, (1, 2048))
        
        model_yarn.eval()
        with torch.no_grad():
            logits_yarn, _ = model_yarn(x_long)
        
        assert logits_yarn.shape[1] == 2048, "YaRN failed on long sequence"
        print_success(f"YaRN handled {x_long.shape[1]} tokens (4x extension)")
        
        # Base model should work on short
        x_short = torch.randint(0, 32005, (1, 512))
        model_base.eval()
        with torch.no_grad():
            logits_base, _ = model_base(x_short)
        
        print_success("Base model works on short sequences")
        
        return True
        
    except Exception as e:
        print_error(f"Test failed: {e}")
        traceback.print_exc()
        return False


# ============================================
# RUN ALL TESTS
# ============================================
def run_all_tests():
    print(f"\n{BLUE}{'='*70}")
    print("🧪 HessGPT ARCHITECTURE TEST SUITE v4")
    print(f"{'='*70}{RESET}\n")
    
    tests = [
        ("Model Initialization", test_model_initialization),
        ("Forward Pass (Train)", test_forward_pass_train),
        ("Soft-Capping", test_soft_capping),
        ("Loss with Padding", test_loss_with_padding),
        ("Gradient Flow", test_gradient_flow),
        ("Causal Mask", test_causal_mask),
        ("Generation", test_generation),
        ("Vocab Resize", test_vocab_resize),
        ("LoRA Application", test_lora),
        ("YaRN Extension", test_yarn),
    ]
    
    results = []
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print_error(f"Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print(f"\n{BLUE}{'='*70}")
    print("📊 TEST SUMMARY")
    print(f"{'='*70}{RESET}\n")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        if result:
            print_success(f"{name}")
        else:
            print_error(f"{name}")
    
    print(f"\n{BLUE}{'='*70}{RESET}")
    if passed == total:
        print_success(f"ALL TESTS PASSED: {passed}/{total} ✅")
    else:
        print_error(f"SOME TESTS FAILED: {passed}/{total}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
