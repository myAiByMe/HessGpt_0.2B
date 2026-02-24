#!/usr/bin/env python3
"""
🔥 HessGPT Mini 0.2B - Math/Code Injection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Injecte 400M tokens de math haute qualité
✅ LR très bas (5e-5) pour ne pas perturber le pretrain
✅ 1 seule epoch sur chunkMath/
✅ Save auto toutes les 50 min
✅ Resume compatible avec le checkpoint pretrain
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Datasets dans chunkMath/ :
  🔢 FineMath-4plus           240M → finemath.npy
  📐 Nemotron-CC-Math-v1 4+   160M → nemotron_math.npy
  Total : 400M tokens

USAGE:
  1. Place hessgpt_mini_pretrain.pt dans ./tinyModel/
  2. Place chunkMath/ dans ./data/ultra_filtered/
  3. Lance : python pretrain_math_injection.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
import math
import gc
from tqdm import tqdm
from transformers import AutoTokenizer
from datetime import datetime
import numpy as np

sys.path.append('./Core/Model')
from HessGpt import HessGPT

# ============================================
# CONFIGURATION
# ============================================
MATH_CHUNK_DIR  = './data/ultra_filtered/chunkMath'
MODEL_DIR       = './tinyModel'
PRETRAIN_CKPT   = f'{MODEL_DIR}/hessgpt_mini_pretrain.pt'
OUTPUT_CKPT     = f'{MODEL_DIR}/hessgpt_mini_math_injected.pt'
SAVE_EVERY_MINUTES = 50

# ============================================
# TOKENS SPÉCIAUX — Gold Standard 2026
# ============================================
SPECIAL_TOKENS = {
    '<|system|>':       32000,
    '<|user|>':         32001,
    '<|assistant|>':    32002,
    '<|end|>':          32003,
    '<think>':          32004,
    '</think>':         32005,
    '<tool_call>':      32006,
    '</tool_call>':     32007,
    '<tool_response>':  32008,
    '</tool_response>': 32009,
    '<code>':           32010,
}

# ============================================
# CONFIG INJECTION
# LR très bas pour ne pas perturber le pretrain
# ============================================
CONFIG = {
    'vocab_size':    32011,
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

    'batch_size':              80,    # plus élevé car 1 seule epoch courte
    'gradient_accumulation':   4,
    'max_grad_norm':           1.0,

    # LR très bas — on affine, on ne réapprend pas
    'learning_rate':           5e-5,
    'weight_decay':            0.1,
    'adam_beta1':              0.9,
    'adam_beta2':              0.95,
    'adam_eps':                1e-8,

    # Cosine decay simple — pas de warmup agressif
    'warmup_ratio':  0.02,
    'min_lr_ratio':  0.1,

    'use_compile':   True,
    'compile_mode':  'default',
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 80)
print("🔥 HessGPT Mini 0.2B — Math/Code Injection")
print(f"   Input:   {PRETRAIN_CKPT}")
print(f"   Output:  {OUTPUT_CKPT}")
print(f"   Data:    {MATH_CHUNK_DIR}")
print(f"   LR:      {CONFIG['learning_rate']:.0e}  (très bas — injection douce)")
print("=" * 80)

# ============================================
# DATASET
# ============================================
class MathDataset(Dataset):
    def __init__(self, chunk_dir, seq_len):
        self.seq_len = seq_len

        files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.npy')])
        if not files:
            raise FileNotFoundError(f"❌ Aucun .npy dans {chunk_dir}")

        print(f"\n📦 Chargement dataset math...")
        all_tokens = []
        for f in files:
            path   = os.path.join(chunk_dir, f)
            tokens = np.load(path)
            print(f"   📂 {f}: {len(tokens)/1e6:.1f}M tokens")
            all_tokens.append(tokens)

        self.tokens      = np.concatenate(all_tokens)
        self.num_samples = len(self.tokens) // (seq_len + 1)
        print(f"   ✅ Total: {len(self.tokens)/1e6:.1f}M tokens → {self.num_samples:,} samples")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = self.tokens[start : start + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].copy()).long()
        y = torch.from_numpy(chunk[1:].copy()).long()
        return x, y


# ============================================
# SCHEDULER — Cosine simple
# ============================================
class CosineScheduler:
    def __init__(self, optimizer, max_lr, total_steps, warmup_ratio, min_lr_ratio):
        self.optimizer    = optimizer
        self.max_lr       = max_lr
        self.min_lr       = max_lr * min_lr_ratio
        self.total_steps  = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.current_step = 0

    def get_lr(self):
        step = self.current_step
        if step < self.warmup_steps:
            return self.max_lr * (step / max(self.warmup_steps, 1))
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
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
# SAVE
# ============================================
def save_checkpoint(model, optimizer, scheduler,
                    global_step, step_in_epoch,
                    batch_idx, loss_val, path):
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_step':       scheduler.current_step,

        'epoch':             'math_injection',
        'step_in_epoch':     step_in_epoch,
        'global_step':       global_step,
        'batch_idx':         batch_idx,

        'config':            CONFIG,
        'tokenizer':         'mistralai/Mistral-7B-v0.1',
        'special_tokens':    SPECIAL_TOKENS,
        'saved_at':          datetime.now().isoformat(),
        'last_loss':         loss_val,
        'injection_type':    'math_finemath_nemotron',
    }, path)

    status = "terminé" if batch_idx == -1 else f"batch {batch_idx}"
    print(f"\n   💾 Save  |  step={step_in_epoch}  global={global_step:,}  loss={loss_val:.4f}  ({status})")
    print(f"      → {path}")


# ============================================
# TOKENIZER
# ============================================
print(f"\n📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.add_special_tokens({'additional_special_tokens': list(SPECIAL_TOKENS.keys())})
tokenizer.pad_token = tokenizer.eos_token
print(f"   ✅ Vocab: {len(tokenizer)}")

# ============================================
# MODEL
# ============================================
print(f"\n🏗️  Loading pretrained model...")

if not os.path.exists(PRETRAIN_CKPT):
    raise FileNotFoundError(f"❌ Checkpoint pretrain introuvable: {PRETRAIN_CKPT}")

ckpt = torch.load(PRETRAIN_CKPT, map_location='cpu')
cfg  = ckpt.get('config', {})

model = HessGPT(
    vocab_size=CONFIG['vocab_size'],
    embed_dim=cfg.get('embed_dim', CONFIG['embed_dim']),
    num_heads=cfg.get('num_heads', CONFIG['num_heads']),
    num_layers=cfg.get('num_layers', CONFIG['num_layers']),
    max_seq_len=CONFIG['max_seq_len'],
    dropout=CONFIG['dropout'],
    use_rope=CONFIG['use_rope'],
    use_yarn=CONFIG['use_yarn'],
    yarn_scale=CONFIG['yarn_scale'],
    yarn_original_max_len=CONFIG['yarn_original_max_len'],
    use_swiglu=CONFIG['use_swiglu'],
    n_kv_heads=cfg.get('n_kv_heads', CONFIG['n_kv_heads']),
    use_qk_norm=CONFIG['use_qk_norm'],
    soft_cap=CONFIG['soft_cap'],
    use_flash_attn=CONFIG['use_flash_attn']
)

# Fix torch.compile prefix
state_dict = ckpt['model_state_dict']
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model = model.to(device)

pretrain_epoch = ckpt.get('epoch', '?')
pretrain_loss  = ckpt.get('last_loss', '?')
pretrain_step  = ckpt.get('global_step', 0)
print(f"   ✅ Pretrain epoch={pretrain_epoch} | loss={pretrain_loss:.4f} | global_step={pretrain_step:,}")

# ============================================
# DATASET
# ============================================
dataset = MathDataset(MATH_CHUNK_DIR, CONFIG['max_seq_len'])

total_steps = math.ceil(
    math.ceil(len(dataset) / CONFIG['batch_size']) / CONFIG['gradient_accumulation']
)
print(f"\n   Steps total: {total_steps:,}")
print(f"   Batch size:  {CONFIG['batch_size']} × grad_accum {CONFIG['gradient_accumulation']} = effective {CONFIG['batch_size']*CONFIG['gradient_accumulation']}")

# ============================================
# OPTIMIZER + SCHEDULER
# ============================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay'],
    betas=(CONFIG['adam_beta1'], CONFIG['adam_beta2']),
    eps=CONFIG['adam_eps']
)

scheduler = CosineScheduler(
    optimizer,
    max_lr=CONFIG['learning_rate'],
    total_steps=total_steps,
    warmup_ratio=CONFIG['warmup_ratio'],
    min_lr_ratio=CONFIG['min_lr_ratio']
)

# ============================================
# RESUME si injection déjà commencée
# ============================================
global_step   = 0
step_in_epoch = 0
resume_batch  = -1
current_loss  = 0.0

if os.path.exists(OUTPUT_CKPT):
    print(f"\n📂 Checkpoint injection trouvé → resume...")
    inj_ckpt = torch.load(OUTPUT_CKPT, map_location='cpu')

    state_dict = inj_ckpt['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)

    optimizer.load_state_dict(inj_ckpt['optimizer_state_dict'])
    scheduler.current_step = inj_ckpt.get('scheduler_step', 0)
    global_step   = inj_ckpt.get('global_step', 0)
    step_in_epoch = inj_ckpt.get('step_in_epoch', 0)
    resume_batch  = inj_ckpt.get('batch_idx', -1)
    current_loss  = inj_ckpt.get('last_loss', 0.0)

    print(f"   Sauvegardé le : {inj_ckpt.get('saved_at', '?')}")
    print(f"   Step: {step_in_epoch} | Global: {global_step:,} | Loss: {current_loss:.4f}")
    print(f"   ⚡ Reprise au batch {resume_batch}")
else:
    print(f"\n🆕 Démarrage injection fresh")

print(f"   LR actuel : {scheduler.get_lr():.2e}  (step {scheduler.current_step}/{total_steps})")

# ============================================
# COMPILE
# ============================================
if CONFIG['use_compile'] and device == 'cuda':
    print(f"\n⚡ torch.compile...")
    try:
        model = torch.compile(model, mode=CONFIG['compile_mode'])
        print(f"   ✅ OK")
    except Exception as e:
        print(f"   ⚠️  Échec: {e}")

# ============================================
# DATALOADER
# ============================================
loader = DataLoader(
    dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=min(4, os.cpu_count()),
    pin_memory=True
)

# ============================================
# TRAINING
# ============================================
print(f"\n{'='*70}")
print(f"🚀 MATH INJECTION  |  {len(dataset):,} samples  |  LR={CONFIG['learning_rate']:.0e}")
print(f"{'='*70}")

model.train()
last_save_time = time.time()
save_interval  = SAVE_EVERY_MINUTES * 60

pbar = tqdm(loader, desc="Math Injection")

for batch_idx, (x, y) in enumerate(pbar):

    # ── Skip batches déjà vus ─────────────────────────────────────────
    if resume_batch > 0 and batch_idx < resume_batch:
        if batch_idx % 500 == 0:
            print(f"\r   ⏭️  Skip {batch_idx}/{resume_batch}...", end='', flush=True)
        continue
    if resume_batch > 0 and batch_idx == resume_batch:
        print(f"\n   ✅ Reprise effective au batch {batch_idx}")
        resume_batch = -1

    # ── Forward ──────────────────────────────────────────────────────
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, loss = model(x, targets=y, pad_token_id=tokenizer.pad_token_id)
        loss = loss / CONFIG['gradient_accumulation']

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"\n   ⚠️  NaN/Inf batch {batch_idx}, skip")
        optimizer.zero_grad(set_to_none=True)
        continue

    loss.backward()

    # ── Optimizer step ────────────────────────────────────────────────
    if (batch_idx + 1) % CONFIG['gradient_accumulation'] == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        global_step   += 1
        step_in_epoch += 1
        current_loss   = loss.item() * CONFIG['gradient_accumulation']

        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'lr':   f'{scheduler.get_last_lr()[0]:.2e}',
            'step': f'{step_in_epoch}/{total_steps}',
        })

    # ── Auto-save toutes les N minutes ───────────────────────────────
    if time.time() - last_save_time >= save_interval:
        save_checkpoint(model, optimizer, scheduler,
                        global_step, step_in_epoch,
                        batch_idx, current_loss, OUTPUT_CKPT)
        last_save_time = time.time()

# ============================================
# FIN — save final
# ============================================
gc.collect()
torch.cuda.empty_cache()

print(f"\n\n{'='*70}")
print(f"✅ MATH INJECTION TERMINÉE !")

save_checkpoint(model, optimizer, scheduler,
                global_step, step_in_epoch,
                -1, current_loss, OUTPUT_CKPT)

print(f"\n{'='*70}")
print(f"📊 RÉSUMÉ")
print(f"{'='*70}")
print(f"   Steps:       {step_in_epoch:,}")
print(f"   Last loss:   {current_loss:.4f}")
print(f"   Checkpoint:  {OUTPUT_CKPT}")
print(f"\n{'='*70}")
print(f"📋 PROCHAINE ÉTAPE")
print(f"{'='*70}")
print(f"   ✅ Modèle prêt pour le SFT LoRA")
print(f"   → Utilise {OUTPUT_CKPT} comme PRETRAIN_CHECKPOINT dans le SFT")
print(f"   → python sft_lora_function_calling.py")
