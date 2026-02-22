#!/usr/bin/env python3
"""
🔥 HessGPT Mini 0.2B - PRETRAIN v3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ NO CLI, NO EDIT — Lance direct
✅ Chunk auto-détecté dans ./data/
✅ Epoch/Step dans les métadonnées du .pt
✅ Save auto toutes les 50 minutes
✅ Multi-compte friendly
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stratégie multi-compte :
  1. Place ton chunk dans ./data/  (1 seul chunk à la fois)
  2. Place le .pt dans ./tinyModel/ si tu reprends
  3. Lance : python pretrain_mini_0.2B_v3.py
  4. À la fin : download le .pt, change de compte, place chunk suivant + .pt, relance
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
# CONFIGURATION
# ============================================
DATA_DIR        = './data/ultra_filtered'
MODEL_DIR       = './tinyModel'
CHECKPOINT_FILE = f'{MODEL_DIR}/hessgpt_mini_pretrain.pt'
TOTAL_CHUNKS    = 5
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
# CONFIG MINI 0.2B
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

    'batch_size':              70,
    'gradient_accumulation':   4,
    'max_grad_norm':           1.0,
    'learning_rate':           8e-4,
    'weight_decay':            0.1,
    'adam_beta1':              0.9,
    'adam_beta2':              0.95,
    'adam_eps':                1e-8,

    'warmup_ratio':  0.03,
    'decay_ratio':   0.15,
    'min_lr_ratio':  0.1,

    'use_compile':   True,
    'compile_mode':  'default',
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================
# AUTO-DETECT CHUNK
# ============================================
def find_chunk(data_dir):
    """
    Cherche le premier (et unique) dossier chunk dans ./data/
    Accepte : chunk000, chunk001, chunk_0, chunk_1, etc.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Dossier data introuvable: {data_dir}")

    candidates = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if not entry.lower().startswith('chunk'):
            continue
        # Vérifie qu'il contient des .npy
        npy_files = [f for f in os.listdir(full_path) if f.endswith('.npy')]
        if len(npy_files) == 0:
            continue
        candidates.append((entry, full_path))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"❌ Aucun dossier chunk avec des .npy trouvé dans {data_dir}\n"
            f"   Place ton chunk dans {data_dir}/chunk000/ par exemple"
        )

    if len(candidates) > 1:
        print(f"   ⚠️  Plusieurs chunks trouvés, on prend le premier: {candidates[0][0]}")

    chunk_name, chunk_path = candidates[0]

    # Extrait l'ID numérique du nom
    # chunk000→0 | chunk_000→0 | chunk_3→3 | chunk2→2
    digits = ''.join(filter(str.isdigit, chunk_name))
    chunk_id = int(digits.lstrip('0') or '0') if digits else 0

    return chunk_path, chunk_id, chunk_name


# ============================================
# DATASET
# ============================================
class ChunkDataset(Dataset):
    def __init__(self, chunk_dir, seq_len):
        self.seq_len = seq_len

        files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.npy')])
        all_tokens = []
        for f in files:
            tokens = np.load(os.path.join(chunk_dir, f))
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
# SCHEDULER WSD
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
# SAVE
# ============================================
def save_checkpoint(model, optimizer, scheduler,
                    global_step, step_in_epoch, epoch,
                    chunk_id, batch_idx, loss_val):
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': {'current_step': scheduler.current_step},

        # ── Métadonnées human-readable ──
        'epoch':            epoch,           # 1-indexed (epoch 1 = chunk 0)
        'step_in_epoch':    step_in_epoch,   # steps depuis le début de ce chunk
        'global_step':      global_step,     # steps cumulés tous chunks

        # ── Pour le resume ──
        'completed_chunk_id': chunk_id,      # chunk en cours ou terminé
        'batch_idx_in_chunk': batch_idx,     # -1 si chunk terminé

        # ── Meta ──
        'config':           CONFIG,
        'tokenizer':        'mistralai/Mistral-7B-v0.1',
        'special_tokens':   SPECIAL_TOKENS,
        'saved_at':         datetime.now().isoformat(),
        'last_loss':        loss_val,
    }, CHECKPOINT_FILE)

    status = "chunk terminé" if batch_idx == -1 else f"batch {batch_idx}"
    print(f"\n   💾 Save  |  epoch={epoch}  step={step_in_epoch}  global={global_step:,}  loss={loss_val:.4f}  ({status})")
    print(f"      → {CHECKPOINT_FILE}")


# ============================================
# DETECT CHUNK
# ============================================
print("\n🔍 Détection du chunk...")
chunk_path, chunk_id, chunk_name = find_chunk(DATA_DIR)
epoch = chunk_id + 1   # epoch 1-indexed

print(f"   ✅ Chunk trouvé : {chunk_name}  →  epoch {epoch} / {TOTAL_CHUNKS}")
print(f"   Path: {chunk_path}")

print("=" * 80)
print("🔥 HessGPT Mini 0.2B — PRETRAIN v3")
print(f"   Epoch:  {epoch} / {TOTAL_CHUNKS}  (chunk {chunk_name})")
print(f"   Model:  {MODEL_DIR}")
print(f"   Save:   toutes les {SAVE_EVERY_MINUTES} min")
print("=" * 80)

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
print(f"   ✅ {params['total']/1e6:.1f}M params")

# ============================================
# DATASET + STEPS ESTIMATION
# ============================================
print(f"\n📦 Chargement dataset...")
dataset = ChunkDataset(chunk_path, CONFIG['max_seq_len'])
print(f"   ✅ {len(dataset):,} samples")

steps_per_chunk = math.ceil(
    math.ceil(len(dataset) / CONFIG['batch_size']) / CONFIG['gradient_accumulation']
)
total_steps = steps_per_chunk * TOTAL_CHUNKS
print(f"   Steps/chunk: {steps_per_chunk:,}  |  Total estimé: {total_steps:,}")

# ============================================
# OPTIMIZER
# ============================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay'],
    betas=(CONFIG['adam_beta1'], CONFIG['adam_beta2']),
    eps=CONFIG['adam_eps']
)

# ============================================
# CHECKPOINT RESUME
# ============================================
global_step    = 0
step_in_epoch  = 0
resume_batch   = -1
current_loss   = 0.0

if os.path.exists(CHECKPOINT_FILE):
    print(f"\n📂 Checkpoint trouvé → resume...")
    ckpt = torch.load(CHECKPOINT_FILE, map_location='cpu')

    # Fix torch.compile prefix _orig_mod.
    state_dict = ckpt['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    global_step   = ckpt.get('global_step', 0)
    current_loss  = ckpt.get('last_loss', 0.0)
    saved_chunk   = ckpt.get('completed_chunk_id', -1)
    saved_epoch   = ckpt.get('epoch', 0)
    saved_step    = ckpt.get('step_in_epoch', 0)
    saved_at      = ckpt.get('saved_at', 'unknown')

    print(f"   Sauvegardé le : {saved_at}")
    print(f"   Checkpoint    : epoch {saved_epoch}  step {saved_step}  global {global_step:,}")
    print(f"   Ce run        : epoch {epoch}")

    if saved_chunk == chunk_id:
        # Même chunk → reprise mid-epoch
        resume_batch  = ckpt.get('batch_idx_in_chunk', -1)
        step_in_epoch = ckpt.get('step_in_epoch', 0)
        print(f"   ⚡ Même chunk → reprise au batch {resume_batch}  (step_in_epoch={step_in_epoch})")
    else:
        # Nouveau chunk → epoch suivante, step_in_epoch repart à 0
        step_in_epoch = 0
        resume_batch  = -1
        print(f"   🆕 Nouveau chunk → step_in_epoch repart à 0, scheduler continue (global={global_step:,})")

else:
    print(f"\n🆕 Aucun checkpoint, démarrage fresh")
    model = model.to(device)

# ============================================
# SCHEDULER — positionné au bon global_step
# ============================================
scheduler = WSDScheduler(
    optimizer,
    max_lr=CONFIG['learning_rate'],
    total_steps=total_steps,
    warmup_ratio=CONFIG['warmup_ratio'],
    decay_ratio=CONFIG['decay_ratio'],
    min_lr_ratio=CONFIG['min_lr_ratio']
)
scheduler.current_step = global_step
print(f"\n   LR actuel : {scheduler.get_lr():.2e}  (step {scheduler.current_step}/{total_steps})")

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
print(f"🚀 TRAINING  |  epoch {epoch}/{TOTAL_CHUNKS}  |  {len(dataset):,} samples")
print(f"{'='*70}")

model.train()
last_save_time = time.time()
save_interval  = SAVE_EVERY_MINUTES * 60

pbar = tqdm(loader, desc=f"Epoch {epoch}")

for batch_idx, (x, y) in enumerate(pbar):

    # ── Skip batches déjà vus (resume mid-chunk) ─────────────────────────
    if resume_batch > 0 and batch_idx < resume_batch:
        if batch_idx % 500 == 0:
            print(f"\r   ⏭️  Skip {batch_idx}/{resume_batch}...", end='', flush=True)
        continue
    if resume_batch > 0 and batch_idx == resume_batch:
        print(f"\n   ✅ Reprise effective au batch {batch_idx}")
        resume_batch = -1

    # ── Forward ──────────────────────────────────────────────────────────
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

    # ── Optimizer step ────────────────────────────────────────────────────
    if (batch_idx + 1) % CONFIG['gradient_accumulation'] == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        global_step   += 1
        step_in_epoch += 1
        current_loss   = loss.item() * CONFIG['gradient_accumulation']

        pbar.set_postfix({
            'loss':   f'{current_loss:.4f}',
            'lr':     f'{scheduler.get_last_lr()[0]:.2e}',
            'ep_stp': f'{step_in_epoch}',
            'g_stp':  f'{global_step:,}'
        })

    # ── Auto-save toutes les N minutes ───────────────────────────────────
    if time.time() - last_save_time >= save_interval:
        save_checkpoint(model, optimizer, scheduler,
                        global_step, step_in_epoch, epoch,
                        chunk_id, batch_idx, current_loss)
        last_save_time = time.time()

# ============================================
# FIN DU CHUNK → save avec step_in_epoch=0
# (prêt pour la prochaine epoch)
# ============================================
gc.collect()
torch.cuda.empty_cache()

print(f"\n\n{'='*70}")
print(f"✅ Epoch {epoch} terminée !")

# On sauvegarde avec batch_idx=-1 et step_in_epoch=0
# → la prochaine session détectera chunk_id+1 et repartira proprement
save_checkpoint(model, optimizer, scheduler,
                global_step, 0, epoch,
                chunk_id, -1, current_loss)

# ============================================
# RÉSUMÉ + INSTRUCTIONS
# ============================================
print(f"\n{'='*70}")
print(f"📊 RÉSUMÉ")
print(f"{'='*70}")
print(f"   Epoch:       {epoch} / {TOTAL_CHUNKS}")
print(f"   Global step: {global_step:,}")
print(f"   Last loss:   {current_loss:.4f}")
print(f"   Checkpoint:  {CHECKPOINT_FILE}")

next_epoch = epoch + 1
next_chunk_id = chunk_id + 1
if next_chunk_id < TOTAL_CHUNKS:
    print(f"\n{'='*70}")
    print(f"📋 PROCHAINE SESSION (epoch {next_epoch})")
    print(f"{'='*70}")
    print(f"   1. Download : {CHECKPOINT_FILE}")
    print(f"   2. Change de compte / machine")
    print(f"   3. Upload   : chunk{next_chunk_id:03d}/  dans ./data/")
    print(f"   4. Upload   : hessgpt_mini_pretrain.pt  dans ./tinyModel/")
    print(f"   5. Lance    : python pretrain_mini_0.2B_v3.py")
    print(f"   → Le script détectera automatiquement chunk{next_chunk_id:03d} = epoch {next_epoch}")
else:
    print(f"\n   🎉 TOUS LES CHUNKS TERMINÉS → Lance le SFT !")
    print(f"   python sft_lora_function_calling.py")
