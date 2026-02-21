#!/usr/bin/env python3
"""
🔥 HessGPT Mini 0.2B - PRETRAIN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ NO CLI - Lance direct avec config hardcodée
✅ Data folder: ./data (comme download script)
✅ Model folder: ./tinyModel (pas checkpoints)
✅ 5 chunks × 1B tokens = 5B tokens (ratio 25:1)
✅ Tokens spéciaux Gold Standard 2026 (7 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Just run: python pretrain_mini_0.2B.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
import math
import json
import gc
from tqdm import tqdm
from transformers import AutoTokenizer
from datetime import datetime
import numpy as np

sys.path.append('./Core/Model')
from HessGpt import HessGPT

# ============================================
# CONFIGURATION HARDCODÉE (NO CLI)
# ============================================
NUM_CHUNKS = 5               # 5 chunks de 1B tokens
DATA_DIR = './data/ultra_filtered'
MODEL_DIR = './tinyModel'    # ✅ PAS checkpoints
CHECKPOINT_FILE = f'{MODEL_DIR}/hessgpt_mini_pretrain.pt'

# ============================================
# TOKENS SPÉCIAUX — Gold Standard 2026 (7 tokens)
# ============================================
# Philosophie : le minimum nécessaire pour un agent moderne.
# Pas plus → ne pas diluer l'attention du modèle.
#
#   <think> / </think>           → Raisonnement interne (Thinking Mode)
#   <tool_call> / </tool_call>   → Déclenchement outil / recherche web
#   <tool_response> / </tool_response> → Réinjection résultat dans le modèle
#   <code>                       → Routage vers module logique 0.6B
#
SPECIAL_TOKENS = {
    '<|system|>':      32000,
    '<|user|>':        32001,
    '<|assistant|>':   32002,
    '<|end|>':         32003,
    '<think>':         32004,
    '</think>':        32005,
    '<tool_call>':     32006,
    '</tool_call>':    32007,
    '<tool_response>': 32008,
    '</tool_response>':32009,
    '<code>':          32010,
}

# ============================================
# CONFIG MINI 0.2B
# ============================================
CONFIG = {
    'vocab_size':    32011,   # 32000 Mistral + 11 special tokens
    'embed_dim':     896,
    'num_heads':     14,
    'n_kv_heads':    7,
    'num_layers':    16,
    'max_seq_len':   1024,
    'dropout':       0.0,
    'use_rope':      True,
    'use_yarn':      False,
    'yarn_scale':    1.0,
    'yarn_original_max_len': 1024,
    'use_swiglu':    True,
    'use_qk_norm':   True,
    'soft_cap':      30.0,
    'use_flash_attn': True,

    'batch_size':              32,
    'gradient_accumulation':   4,
    'max_grad_norm':           1.0,
    'learning_rate':           5e-4,
    'weight_decay':            0.1,
    'adam_beta1':              0.9,
    'adam_beta2':              0.95,
    'adam_eps':                1e-8,

    'warmup_ratio':  0.03,
    'decay_ratio':   0.15,
    'min_lr_ratio':  0.1,

    'validate_every_steps': 500,
    'val_batches':          50,

    # --- System ---
    'use_compile':    True,
    'compile_mode':   'default',
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 80)
print("🔥 HessGPT Mini 0.2B — PRETRAIN")
print(f"   Data:   {DATA_DIR}")
print(f"   Model:  {MODEL_DIR}")
print(f"   Chunks: {NUM_CHUNKS}")
print(f"   Vocab:  {CONFIG['vocab_size']} ({len(SPECIAL_TOKENS)} special tokens)")
print("=" * 80)

# Create model dir
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================
# TOKENIZER
# ============================================
print(f"\n📝 Loading Mistral tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.add_special_tokens({
    'additional_special_tokens': list(SPECIAL_TOKENS.keys())
})
tokenizer.pad_token = tokenizer.eos_token
print(f"   ✅ Vocab size: {len(tokenizer)}")
print(f"   ✅ Special tokens: {list(SPECIAL_TOKENS.keys())}")

# ============================================
# MODEL
# ============================================
print(f"\n🏗️  Creating model...")
model = HessGPT(
    vocab_size=CONFIG['vocab_size'],
    embed_dim=CONFIG['embed_dim'],
    num_heads=CONFIG['num_heads'],
    num_layers=CONFIG['num_layers'],
    max_seq_len=CONFIG['max_seq_len'],
    dropout=CONFIG['dropout'],
    use_rope=CONFIG['use_rope'],
    use_yarn=CONFIG['use_yarn'],
    yarn_scale=CONFIG['yarn_scale'],
    yarn_original_max_len=CONFIG['yarn_original_max_len'],
    use_swiglu=CONFIG['use_swiglu'],
    n_kv_heads=CONFIG['n_kv_heads'],
    use_qk_norm=CONFIG['use_qk_norm'],
    soft_cap=CONFIG['soft_cap'],
    use_flash_attn=CONFIG['use_flash_attn']
).to(device)

params = model.count_parameters()
print(f"   ✅ Model: {params['total']/1e6:.1f}M params")

if CONFIG['use_compile'] and device == 'cuda':
    print(f"\n⚡ torch.compile...")
    try:
        model = torch.compile(model, mode=CONFIG['compile_mode'])
        print(f"   ✅ Compilé")
    except Exception as e:
        print(f"   ⚠️  Échec compile: {e}")

# ============================================
# SCAN CHUNKS
# ============================================
def scan_chunks(data_dir):
    chunks = []
    for entry in sorted(os.listdir(data_dir)):
        if not entry.startswith('chunk'):
            continue
        chunk_dir = os.path.join(data_dir, entry)
        if not os.path.isdir(chunk_dir):
            continue
        stats_file = os.path.join(chunk_dir, 'stats.json')
        if not os.path.exists(stats_file):
            continue
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        chunks.append({
            'id': stats['chunk_id'],
            'dir': chunk_dir,
            'stats': stats
        })
    return sorted(chunks, key=lambda x: x['id'])

chunks = scan_chunks(DATA_DIR)[:NUM_CHUNKS]
print(f"\n📦 Found {len(chunks)} chunks")

# ============================================
# DATASET
# ============================================
class ChunkDataset(Dataset):
    def __init__(self, chunk_dir, seq_len, pad_token_id):
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id

        # Load all .npy files (skip stats.json)
        files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.npy')])

        all_tokens = []
        for f in files:
            path = os.path.join(chunk_dir, f)
            tokens = np.load(path)
            all_tokens.append(tokens)

        self.tokens = np.concatenate(all_tokens)
        self.num_samples = len(self.tokens) // (seq_len + 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = self.tokens[start : start + self.seq_len + 1]

        x = torch.from_numpy(chunk[:-1].copy()).long()
        y = torch.from_numpy(chunk[1:].copy()).long()

        return x, y

# ============================================
# SCHEDULER (WSD — Warmup Stable Decay)
# ============================================
class WSDScheduler:
    def __init__(self, optimizer, max_lr, total_steps, warmup_ratio, decay_ratio, min_lr_ratio):
        self.optimizer    = optimizer
        self.max_lr       = max_lr
        self.min_lr       = max_lr * min_lr_ratio
        self.total_steps  = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.decay_steps  = int(total_steps * decay_ratio)
        self.stable_steps = total_steps - self.warmup_steps - self.decay_steps
        self.current_step = 0

    def get_lr(self):
        step = self.current_step
        if step < self.warmup_steps:
            return self.max_lr * (step / max(self.warmup_steps, 1))
        elif step < self.warmup_steps + self.stable_steps:
            return self.max_lr
        else:
            decay_step = step - self.warmup_steps - self.stable_steps
            progress   = min(decay_step / max(self.decay_steps, 1), 1.0)
            cosine     = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine

    def step(self):
        lr = self.get_lr()
        self.current_step += 1
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def get_last_lr(self):
        return [self.get_lr()]

# ============================================
# CHECKPOINT RESUME
# ============================================
start_chunk     = 0
global_step     = 0
resume_checkpoint = None

if os.path.exists(CHECKPOINT_FILE):
    print(f"\n📂 Found existing checkpoint: {CHECKPOINT_FILE}")
    print(f"   Loading for resume...")

    resume_checkpoint = torch.load(CHECKPOINT_FILE, map_location='cpu')

    model.load_state_dict(resume_checkpoint['model_state_dict'])

    global_step = resume_checkpoint.get('global_step', 0)
    start_chunk = resume_checkpoint.get('completed_chunks', 0)

    print(f"   ✅ Resuming from:")
    print(f"      Completed chunks: {start_chunk}/{NUM_CHUNKS}")
    print(f"      Global step:      {global_step:,}")

    model = model.to(device)
else:
    print(f"\n🆕 No checkpoint found, starting fresh")

# ============================================
# TRAINING
# ============================================
print(f"\n🚀 Starting training from chunk {start_chunk + 1}...")

# Estimate total steps
total_steps = 0
for chunk in chunks:
    tokens  = chunk['stats']['total_tokens']
    samples = tokens // (CONFIG['max_seq_len'] + 1)
    batches = math.ceil(samples / CONFIG['batch_size'])
    steps   = math.ceil(batches / CONFIG['gradient_accumulation'])
    total_steps += steps

print(f"   Total steps: {total_steps:,}")

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay'],
    betas=(CONFIG['adam_beta1'], CONFIG['adam_beta2']),
    eps=CONFIG['adam_eps']
)

if resume_checkpoint is not None and 'optimizer_state_dict' in resume_checkpoint:
    optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
    print(f"   ✅ Optimizer state resumed")

# Scheduler
scheduler = WSDScheduler(
    optimizer,
    max_lr=CONFIG['learning_rate'],
    total_steps=total_steps,
    warmup_ratio=CONFIG['warmup_ratio'],
    decay_ratio=CONFIG['decay_ratio'],
    min_lr_ratio=CONFIG['min_lr_ratio']
)

if resume_checkpoint is not None and 'scheduler_state_dict' in resume_checkpoint:
    scheduler.current_step = resume_checkpoint['scheduler_state_dict'].get('current_step', 0)
    print(f"   ✅ Scheduler resumed (step: {scheduler.current_step:,})")

# ── Training loop ──────────────────────────────────────────────────────────
model.train()

for chunk_idx, chunk in enumerate(chunks):

    if chunk_idx < start_chunk:
        print(f"\n⏭️  Skipping chunk {chunk_idx + 1}/{NUM_CHUNKS} (already completed)")
        continue

    print(f"\n{'='*70}")
    print(f"CHUNK {chunk_idx + 1}/{NUM_CHUNKS}")
    print(f"{'='*70}")

    dataset = ChunkDataset(chunk['dir'], CONFIG['max_seq_len'], tokenizer.pad_token_id)
    loader  = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print(f"   Samples: {len(dataset):,}")

    pbar = tqdm(loader, desc=f"Chunk {chunk_idx+1}")

    for batch_idx, (x, y) in enumerate(pbar):
        x = x.to(device)
        y = y.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(x, targets=y, pad_token_id=tokenizer.pad_token_id)
            loss = loss / CONFIG['gradient_accumulation']

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()

        if (batch_idx + 1) % CONFIG['gradient_accumulation'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            pbar.set_postfix({
                'loss': f'{loss.item() * CONFIG["gradient_accumulation"]:.4f}',
                'lr':   f'{scheduler.get_last_lr()[0]:.2e}'
            })

    gc.collect()
    torch.cuda.empty_cache()

    # Save after each chunk
    print(f"\n   💾 Saving checkpoint...")
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': {'current_step': scheduler.current_step},
        'config':               CONFIG,
        'global_step':          global_step,
        'completed_chunks':     chunk_idx + 1,
        'tokenizer':            'mistralai/Mistral-7B-v0.1',
        'special_tokens':       SPECIAL_TOKENS,
    }, CHECKPOINT_FILE)
    print(f"   ✅ Checkpoint saved (chunk {chunk_idx + 1}/{NUM_CHUNKS} done)")

# ============================================
# SAVE FINAL
# ============================================
print(f"\n💾 Saving final checkpoint...")
torch.save({
    'model_state_dict':     model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': {'current_step': scheduler.current_step},
    'config':               CONFIG,
    'global_step':          global_step,
    'completed_chunks':     NUM_CHUNKS,
    'tokenizer':            'mistralai/Mistral-7B-v0.1',
    'special_tokens':       SPECIAL_TOKENS,
}, CHECKPOINT_FILE)
print(f"   ✅ Saved: {CHECKPOINT_FILE}")

print(f"\n{'='*70}")
print(f"✅ PRETRAIN COMPLETED")
print(f"{'='*70}")
print(f"\n   Model:      {params['total']/1e6:.1f}M params")
print(f"   Chunks:     {NUM_CHUNKS}")
print(f"   Steps:      {global_step:,}")
print(f"   Checkpoint: {CHECKPOINT_FILE}")

print(f"\n🎯 NEXT: Run SFT LoRA")
print(f"   python sft_lora_function_calling.py")