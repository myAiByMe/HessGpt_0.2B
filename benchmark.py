#!/usr/bin/env python3
"""
📊 HessGPT BENCHMARK EVALUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Évalue le model sur benchmarks standards:

Common Sense Reasoning:
  ✓ WinoGrande
  ✓ ARC Easy
  ✓ ARC Challenge

Question Answering:
  ✓ PIQA
  ✓ HellaSwag
  ✓ MMLU (5-shot)
  ✓ OpenBookQA

Usage:
  python benchmark.py --checkpoint ./tinyModel/hessgpt_mini_pretrain.pt
  python benchmark.py --checkpoint ./checkpoints/HessGpt_pretrain.pt
  python benchmark.py --checkpoint ./tinyModel/hessgpt_mini_0.2B_merged.pt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import sys
import os
import json
import argparse
import numpy as np
from datetime import datetime

sys.path.append('./Core/Model')
from HessGpt import HessGPT

# ============================================
# ARGS
# ============================================
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True,
                    help='Model checkpoint path')
parser.add_argument('--output', type=str, default=None,
                    help='Output JSON file (default: auto-generated)')
parser.add_argument('--max-samples', type=int, default=None,
                    help='Limit samples per benchmark (for quick test)')
parser.add_argument('--batch-size', type=int, default=8,
                    help='Batch size for inference')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 80)
print("📊 HessGPT Benchmark Evaluation")
print(f"   Checkpoint: {args.checkpoint}")
print(f"   Device: {device}")
print("=" * 80)

# ============================================
# LOAD MODEL
# ============================================
print(f"\n🏗️  Loading model...")
checkpoint = torch.load(args.checkpoint, map_location='cpu')
config = checkpoint.get('config', {})

model = HessGPT(
    vocab_size=config.get('vocab_size', 32009),
    embed_dim=config.get('embed_dim', 1280),
    num_heads=config.get('num_heads', 20),
    num_layers=config.get('num_layers', 22),
    max_seq_len=config.get('max_seq_len', 1024),
    dropout=0.0,
    use_rope=True,
    use_yarn=config.get('use_yarn', False),
    yarn_scale=config.get('yarn_scale', 1.0),
    yarn_original_max_len=config.get('yarn_original_max_len', 1024),
    use_swiglu=True,
    n_kv_heads=config.get('n_kv_heads', 5),
    use_qk_norm=True,
    soft_cap=30.0,
    use_flash_attn=True
)

model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model = model.to(device)
model.eval()

params = model.count_parameters()
print(f"   ✅ Model loaded: {params['total']/1e6:.1f}M params")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
special_tokens = checkpoint.get('special_tokens', {})
if special_tokens:
    tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens.keys())})
tokenizer.pad_token = tokenizer.eos_token

# ============================================
# EVALUATION FUNCTIONS
# ============================================
def get_loglikelihood(model, tokenizer, text, device):
    """Calcule log-likelihood d'un texte"""
    tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').to(device)
    
    with torch.no_grad():
        logits, _ = model(tokens)
        
        # Shift for autoregressive loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokens[:, 1:].contiguous()
        
        # Log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs, 
            2, 
            shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Sum log probs
        total_log_prob = token_log_probs.sum().item()
    
    return total_log_prob

def evaluate_multiple_choice(model, tokenizer, question, choices, device):
    """Évalue multiple choice en comparant log-likelihood"""
    scores = []
    
    for choice in choices:
        # Format: question + choice
        text = f"{question}\n{choice}"
        score = get_loglikelihood(model, tokenizer, text, device)
        scores.append(score)
    
    # Choisir le plus haut score
    pred_idx = np.argmax(scores)
    return pred_idx, scores

# ============================================
# BENCHMARK 1: WinoGrande
# ============================================
def eval_winogrande(model, tokenizer, device, max_samples=None):
    """
    WinoGrande: Commonsense reasoning
    Format: Sentence with blank, 2 options
    """
    print(f"\n{'='*70}")
    print("📝 WinoGrande (Commonsense Reasoning)")
    print(f"{'='*70}")
    
    dataset = load_dataset("winogrande", "winogrande_xl", split="validation")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for sample in tqdm(dataset, desc="WinoGrande"):
        sentence = sample['sentence']
        option1 = sample['option1']
        option2 = sample['option2']
        answer = int(sample['answer']) - 1  # 1 or 2 → 0 or 1
        
        # Replace _ with options
        choice1 = sentence.replace('_', option1)
        choice2 = sentence.replace('_', option2)
        
        pred_idx, _ = evaluate_multiple_choice(
            model, tokenizer, "", [choice1, choice2], device
        )
        
        if pred_idx == answer:
            correct += 1
        total += 1
    
    accuracy = correct / total * 100
    print(f"   ✅ Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return {"accuracy": accuracy, "correct": correct, "total": total}

# ============================================
# BENCHMARK 2: ARC Easy/Challenge
# ============================================
def eval_arc(model, tokenizer, device, subset="ARC-Easy", max_samples=None):
    """
    ARC: AI2 Reasoning Challenge
    Format: Science questions, multiple choice
    """
    print(f"\n{'='*70}")
    print(f"📝 {subset} (Science QA)")
    print(f"{'='*70}")
    
    dataset = load_dataset("ai2_arc", subset, split="test")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for sample in tqdm(dataset, desc=subset):
        question = sample['question']
        choices = sample['choices']['text']
        answer_key = sample['answerKey']
        
        # Convert answer key to index
        if answer_key.isdigit():
            answer_idx = int(answer_key) - 1
        else:
            answer_idx = ord(answer_key) - ord('A')
        
        pred_idx, _ = evaluate_multiple_choice(
            model, tokenizer, question, choices, device
        )
        
        if pred_idx == answer_idx:
            correct += 1
        total += 1
    
    accuracy = correct / total * 100
    print(f"   ✅ Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return {"accuracy": accuracy, "correct": correct, "total": total}

# ============================================
# BENCHMARK 3: PIQA
# ============================================
def eval_piqa(model, tokenizer, device, max_samples=None):
    """
    PIQA: Physical Interaction QA
    Format: Goal + 2 solutions
    """
    print(f"\n{'='*70}")
    print("📝 PIQA (Physical Commonsense)")
    print(f"{'='*70}")
    
    dataset = load_dataset("piqa", split="validation")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for sample in tqdm(dataset, desc="PIQA"):
        goal = sample['goal']
        sol1 = sample['sol1']
        sol2 = sample['sol2']
        answer = sample['label']
        
        # Format choices
        choice1 = f"{goal} {sol1}"
        choice2 = f"{goal} {sol2}"
        
        pred_idx, _ = evaluate_multiple_choice(
            model, tokenizer, "", [choice1, choice2], device
        )
        
        if pred_idx == answer:
            correct += 1
        total += 1
    
    accuracy = correct / total * 100
    print(f"   ✅ Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return {"accuracy": accuracy, "correct": correct, "total": total}

# ============================================
# BENCHMARK 4: HellaSwag
# ============================================
def eval_hellaswag(model, tokenizer, device, max_samples=None):
    """
    HellaSwag: Commonsense reasoning about events
    Format: Context + 4 endings
    """
    print(f"\n{'='*70}")
    print("📝 HellaSwag (Event Completion)")
    print(f"{'='*70}")
    
    dataset = load_dataset("hellaswag", split="validation")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for sample in tqdm(dataset, desc="HellaSwag"):
        ctx = sample['ctx']
        endings = sample['endings']
        answer = int(sample['label'])
        
        pred_idx, _ = evaluate_multiple_choice(
            model, tokenizer, ctx, endings, device
        )
        
        if pred_idx == answer:
            correct += 1
        total += 1
    
    accuracy = correct / total * 100
    print(f"   ✅ Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return {"accuracy": accuracy, "correct": correct, "total": total}

# ============================================
# BENCHMARK 5: OpenBookQA
# ============================================
def eval_openbookqa(model, tokenizer, device, max_samples=None):
    """
    OpenBookQA: Elementary science questions
    Format: Question + 4 choices
    """
    print(f"\n{'='*70}")
    print("📝 OpenBookQA (Elementary Science)")
    print(f"{'='*70}")
    
    dataset = load_dataset("openbookqa", "main", split="test")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for sample in tqdm(dataset, desc="OpenBookQA"):
        question = sample['question_stem']
        choices = sample['choices']['text']
        answer_key = sample['answerKey']
        
        # Convert answer key to index
        answer_idx = ord(answer_key) - ord('A')
        
        pred_idx, _ = evaluate_multiple_choice(
            model, tokenizer, question, choices, device
        )
        
        if pred_idx == answer_idx:
            correct += 1
        total += 1
    
    accuracy = correct / total * 100
    print(f"   ✅ Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return {"accuracy": accuracy, "correct": correct, "total": total}

# ============================================
# BENCHMARK 6: MMLU (simplified)
# ============================================
def eval_mmlu(model, tokenizer, device, max_samples=None):
    """
    MMLU: Massive Multitask Language Understanding
    Format: Multiple subjects, 4-choice questions
    """
    print(f"\n{'='*70}")
    print("📝 MMLU (Multitask Understanding)")
    print(f"{'='*70}")
    
    # Use a few representative subjects
    subjects = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 
                'college_chemistry', 'computer_security', 'econometrics', 
                'global_facts', 'high_school_geography', 'jurisprudence']
    
    all_correct = 0
    all_total = 0
    
    for subject in subjects:
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples // len(subjects), len(dataset))))
            
            correct = 0
            total = 0
            
            for sample in dataset:
                question = sample['question']
                choices = sample['choices']
                answer = sample['answer']
                
                pred_idx, _ = evaluate_multiple_choice(
                    model, tokenizer, question, choices, device
                )
                
                if pred_idx == answer:
                    correct += 1
                total += 1
            
            all_correct += correct
            all_total += total
            
        except Exception as e:
            print(f"   ⚠️  Skipped {subject}: {e}")
    
    accuracy = all_correct / all_total * 100 if all_total > 0 else 0
    print(f"   ✅ Overall Accuracy: {accuracy:.2f}% ({all_correct}/{all_total})")
    
    return {"accuracy": accuracy, "correct": all_correct, "total": all_total}

# ============================================
# RUN ALL BENCHMARKS
# ============================================
print(f"\n{'='*80}")
print("🚀 RUNNING BENCHMARKS")
print(f"{'='*80}")

results = {
    'model': args.checkpoint,
    'params': f"{params['total']/1e6:.1f}M",
    'config': config,
    'timestamp': datetime.now().isoformat(),
    'benchmarks': {}
}

# Run all benchmarks
results['benchmarks']['winogrande'] = eval_winogrande(model, tokenizer, device, args.max_samples)
results['benchmarks']['arc_easy'] = eval_arc(model, tokenizer, device, "ARC-Easy", args.max_samples)
results['benchmarks']['arc_challenge'] = eval_arc(model, tokenizer, device, "ARC-Challenge", args.max_samples)
results['benchmarks']['piqa'] = eval_piqa(model, tokenizer, device, args.max_samples)
results['benchmarks']['hellaswag'] = eval_hellaswag(model, tokenizer, device, args.max_samples)
results['benchmarks']['openbookqa'] = eval_openbookqa(model, tokenizer, device, args.max_samples)
results['benchmarks']['mmlu'] = eval_mmlu(model, tokenizer, device, args.max_samples)

# ============================================
# SUMMARY
# ============================================
print(f"\n{'='*80}")
print("📊 BENCHMARK SUMMARY")
print(f"{'='*80}")

print(f"\nModel: {args.checkpoint}")
print(f"Params: {params['total']/1e6:.1f}M")
print(f"\nResults:")

for bench_name, bench_results in results['benchmarks'].items():
    acc = bench_results['accuracy']
    print(f"   {bench_name:20s}: {acc:6.2f}%")

# Calculate average
avg_acc = np.mean([r['accuracy'] for r in results['benchmarks'].values()])
print(f"\n   {'Average':20s}: {avg_acc:6.2f}%")

results['average_accuracy'] = avg_acc

# ============================================
# SAVE RESULTS
# ============================================
if args.output is None:
    # Auto-generate filename
    model_name = os.path.basename(args.checkpoint).replace('.pt', '')
    args.output = f'./benchmark_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

with open(args.output, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved: {args.output}")

print(f"\n{'='*80}")
print("✅ BENCHMARKING COMPLETED")
print(f"{'='*80}\n")
