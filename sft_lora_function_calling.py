#!/usr/bin/env python3
"""
🔥 HessGPT Mini 0.2B - SFT LoRA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ NO CLI - Lance direct
✅ YaRN 1024 → 4096
✅ LoRA Rank 64 sur toutes les couches (q/k/v/o + SwiGLU)
✅ Tokens spéciaux Gold Standard 2026 (identiques au pretrain)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATASETS (210k samples total) :
  33% General Assistant  → HuggingFaceTB/smoltalk           (70k)
  14% Thinking Mode      → Magpie-Reasoning-V2 CoT DeepSeek (30k)
  14% Function Calling   → Salesforce/xlam-function-calling  (30k)
  10% Raisonnement étape → Nous-Hermes-2 WikiHow             (20k)
  14% Personnalité stable→ Tulu 3 Personas math-grade        (30k)
   5% Longues séquences  → LongAlign-10k YaRN calib          (10k)
  10% Task decomposition → WikiHow b-mc2                     (20k)

Just run: python sft_lora_function_calling.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
import os
import random
import math
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

sys.path.append('./Core/Model')
from HessGpt import HessGPT

# ============================================
# CONFIG HARDCODÉE
# ============================================
PRETRAIN_CHECKPOINT = './tinyModel/hessgpt_mini_pretrain.pt'
OUTPUT_DIR          = './tinyModel/lora_adapters'
EPOCHS              = 3
BATCH_SIZE          = 2
MAX_SEQ_LEN         = 4096
LORA_RANK           = 64
LEARNING_RATE       = 1e-4

# ============================================
# TOKENS SPÉCIAUX — Gold Standard 2026
# ⚠️  Doit être IDENTIQUE au pretrain
# ============================================
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

VOCAB_SIZE = 32011   # 32000 Mistral + 11 special tokens

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 80)
print("🔥 HessGPT Mini 0.2B — SFT LoRA")
print(f"   Input:  {PRETRAIN_CHECKPOINT}")
print(f"   Output: {OUTPUT_DIR}")
print("=" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# TOKENIZER
# ============================================
print(f"\n📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.add_special_tokens({
    'additional_special_tokens': list(SPECIAL_TOKENS.keys())
})
tokenizer.pad_token = tokenizer.eos_token
print(f"   ✅ Vocab: {len(tokenizer)} tokens")

# ============================================
# LOAD MODEL
# ============================================
print(f"\n🏗️  Loading pretrained model...")
checkpoint = torch.load(PRETRAIN_CHECKPOINT, map_location='cpu')
config     = checkpoint.get('config', {})

model = HessGPT(
    vocab_size=VOCAB_SIZE,
    embed_dim=config.get('embed_dim', 896),
    num_heads=config.get('num_heads', 14),
    num_layers=config.get('num_layers', 16),
    max_seq_len=MAX_SEQ_LEN,
    dropout=0.0,
    use_rope=True,
    use_yarn=True,                # YaRN activé pour le SFT
    yarn_scale=4.0,               # 1024 → 4096
    yarn_original_max_len=1024,
    use_swiglu=True,
    n_kv_heads=config.get('n_kv_heads', 7),
    use_qk_norm=True,
    soft_cap=30.0,
    use_flash_attn=True
)

model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model = model.to(device)
print(f"   ✅ Model loaded | YaRN x4 activé (1024 → 4096 tokens)")

# ============================================
# LoRA
# ============================================
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r, alpha, dropout):
        super().__init__()
        self.r       = r
        self.scaling = alpha / r

        self.lora_A  = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B  = nn.Parameter(torch.zeros(r, out_features))
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling

class LinearWithLoRA(nn.Module):
    def __init__(self, base_layer, r, alpha, dropout):
        super().__init__()
        self.base_layer = base_layer
        self.lora = LoRALayer(
            base_layer.in_features,
            base_layer.out_features,
            r, alpha, dropout
        )

    def forward(self, x):
        return self.base_layer(x) + self.lora(x)

print(f"\n🔧 Applying LoRA (r={LORA_RANK}, alpha={LORA_RANK*2})...")

# Freeze base
for param in model.parameters():
    param.requires_grad = False

# Toutes les couches attention + SwiGLU
TARGET_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'out_proj',   # Attention
    'gate_proj', 'up_proj', 'down_proj',          # SwiGLU FFN
]

for name, module in model.named_modules():
    for target in TARGET_MODULES:
        if target in name and isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name  = name.split('.')[-1]
            parent      = model.get_submodule(parent_name) if parent_name else model

            lora_layer = LinearWithLoRA(
                module,
                r=LORA_RANK,
                alpha=LORA_RANK * 2,
                dropout=0.05
            )
            setattr(parent, child_name, lora_layer)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   ✅ LoRA appliqué sur: {TARGET_MODULES}")
print(f"   📊 Trainable: {trainable_params/1e6:.2f}M / {total_params/1e6:.1f}M ({100*trainable_params/total_params:.2f}%)")

# ============================================
# DATASET
# ============================================
class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len):
        self.data        = data
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text   = self._format(sample)

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        input_ids  = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:],  dtype=torch.long)

        # Masque le prompt → entraîne uniquement sur les réponses assistant
        labels = target_ids.clone()
        assistant_token_id = SPECIAL_TOKENS['<|assistant|>']
        for i, token_id in enumerate(input_ids):
            if token_id == assistant_token_id:
                labels[:i+1] = -100
                break

        return input_ids, labels

    def _format(self, sample):
        """
        Gère tous les formats de datasets :
          - messages (liste de dicts role/content)  → smoltalk, xLAM, Magpie
          - system/user/assistant (clés plates)     → WikiHow
        """
        if 'messages' in sample:
            result = ""
            for msg in sample['messages']:
                role    = msg.get('role', '')
                content = msg.get('content', '')

                if role == 'system':
                    result += f"<|system|>{content}<|end|>"

                elif role == 'user':
                    result += f"<|user|>{content}<|end|>"

                elif role == 'assistant':
                    # Thinking Mode : si le contenu commence par <think>
                    # on le laisse tel quel (le modèle apprend à le générer)
                    tool_calls = msg.get('tool_calls')
                    if tool_calls:
                        for tc in tool_calls:
                            fname = tc.get('function', {}).get('name', '')
                            args  = tc.get('function', {}).get('arguments', '')
                            result += (
                                f"<|assistant|>"
                                f"<tool_call>{fname}\n{args}</tool_call>"
                                f"<|end|>"
                            )
                    else:
                        result += f"<|assistant|>{content}<|end|>"

                elif role == 'tool':
                    result += f"<tool_response>{content}</tool_response>"

            return result

        # Format plat (WikiHow, etc.)
        system    = sample.get('system', 'You are a helpful assistant.')
        user      = sample.get('user',   sample.get('prompt',      sample.get('instruction', '')))
        assistant = sample.get('assistant', sample.get('response', sample.get('output', '')))

        return (
            f"<|system|>{system}<|end|>"
            f"<|user|>{user}<|end|>"
            f"<|assistant|>{assistant}<|end|>"
        )

def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    labels    = [item[1] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels    = torch.nn.utils.rnn.pad_sequence(labels,    batch_first=True, padding_value=-100)

    return input_ids, labels

# ============================================
# CHARGEMENT DES DATASETS
# ============================================
print(f"\n📦 Loading datasets...")
all_samples = []

# ── 1. SmolTalk 70k (46% — socle assistant poli) ──────────────────────────
print(f"\n1️⃣  SmolTalk (70k) — socle assistant...")
try:
    smoltalk = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
    smoltalk_samples = list(smoltalk.take(70000))
    all_samples.extend(smoltalk_samples)
    print(f"   ✅ SmolTalk: {len(smoltalk_samples):,}")
except Exception as e:
    print(f"   ⚠️  SmolTalk: {e}")

# ── 2. Magpie-Reasoning 30k (20% — Thinking Mode) ─────────────────────────
print(f"\n2️⃣  Magpie-Reasoning CoT (30k) — Thinking Mode...")
try:
    magpie = load_dataset(
        "Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B",
        split="train",
        streaming=True
    )
    magpie_samples = list(magpie.take(30000))
    all_samples.extend(magpie_samples)
    print(f"   ✅ Magpie-Reasoning: {len(magpie_samples):,}")
except Exception as e:
    print(f"   ⚠️  Magpie-Reasoning: {e}")

# ── 3. xLAM Function Calling 30k (20% — tool use) ─────────────────────────
print(f"\n3️⃣  xLAM Function Calling (30k) — tool use...")
try:
    xlam = load_dataset(
        "Salesforce/xlam-function-calling-60k",
        split="train",
        streaming=True
    )
    xlam_samples = list(xlam.take(30000))
    all_samples.extend(xlam_samples)
    print(f"   ✅ xLAM: {len(xlam_samples):,}")
except Exception as e:
    print(f"   ⚠️  xLAM: {e}")

# ── 4. WikiHow 20k (14% — raisonnement étape par étape) ───────────────────
print(f"\n4️⃣  Nous-Hermes WikiHow (20k) — raisonnement étapes...")
try:
    wikihow = load_dataset(
        "MaziyarPanahi/Nous-Hermes-2-Mixtral-8x7B-SFT-Wikihow",
        split="train",
        streaming=True
    )
    wikihow_samples = list(wikihow.take(20000))
    all_samples.extend(wikihow_samples)
    print(f"   ✅ WikiHow: {len(wikihow_samples):,}")
except Exception as e:
    print(f"   ⚠️  WikiHow: {e}")

# ── 5. Tulu 3 Personas (30k — personnalité stable, indispensable) ──────────
print(f"\n5️⃣  Tulu 3 Personas (30k) — personnalité stable...")
try:
    tulu = load_dataset(
        "allenai/tulu-3-sft-personas-math-grade",
        split="train",
        streaming=True
    )
    tulu_samples = list(tulu.take(30000))
    all_samples.extend(tulu_samples)
    print(f"   ✅ Tulu 3: {len(tulu_samples):,}")
except Exception as e:
    print(f"   ⚠️  Tulu 3: {e}")

# ── 6. LongAlign-10k (10k — calibration longues séquences) ────────────────
print(f"\n6️⃣  LongAlign-10k (10k) — YaRN calibration...")
try:
    longalign = load_dataset(
        "THUDM/LongAlign-10k",
        split="train",
        streaming=True
    )
    longalign_samples = list(longalign.take(10000))
    all_samples.extend(longalign_samples)
    print(f"   ✅ LongAlign: {len(longalign_samples):,}")
except Exception as e:
    print(f"   ⚠️  LongAlign: {e}")

# ── Shuffle ────────────────────────────────────────────────────────────────
random.shuffle(all_samples)

print(f"\n📊 Dataset total: {len(all_samples):,} samples")
print(f"   SmolTalk         33%  ~70k")
print(f"   Magpie-CoT       14%  ~30k")
print(f"   xLAM             14%  ~30k")
print(f"   WikiHow          10%  ~20k")
print(f"   Tulu 3 Personas  14%  ~30k")
print(f"   LongAlign         5%  ~10k")
print(f"   WikiHow b-mc2    10%  ~20k")

train_dataset = SFTDataset(all_samples, tokenizer, MAX_SEQ_LEN)
train_loader  = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn
)

# ============================================
# TRAINING
# ============================================
print(f"\n🏋️  Training (epochs={EPOCHS}, lr={LEARNING_RATE}, seq={MAX_SEQ_LEN})...")

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    weight_decay=0.01
)

model.train()

for epoch in range(EPOCHS):
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print(f"{'='*70}")

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for input_ids, labels in pbar:
        input_ids = input_ids.to(device)
        labels    = labels.to(device)

        logits, _ = model(input_ids)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

# ============================================
# SAVE LoRA ADAPTERS
# ============================================
print(f"\n💾 Saving LoRA adapters...")

lora_state = {
    name: param.data.cpu()
    for name, param in model.named_parameters()
    if param.requires_grad
}

torch.save({
    'lora_state_dict': lora_state,
    'config': {
        'lora_rank':      LORA_RANK,
        'lora_alpha':     LORA_RANK * 2,
        'target_modules': TARGET_MODULES,
        'max_seq_len':    MAX_SEQ_LEN,
        'yarn_scale':     4.0,
    },
    'special_tokens': SPECIAL_TOKENS,
}, f'{OUTPUT_DIR}/lora_weights.pt')

print(f"   ✅ Saved: {OUTPUT_DIR}/lora_weights.pt")

print(f"\n{'='*70}")
print(f"✅ SFT COMPLETED")
print(f"{'='*70}")
print(f"\n🎯 NEXT: Test function calling + thinking mode")
print(f"   python test_function_calling.py")