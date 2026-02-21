#!/usr/bin/env python3
"""
🔥 Ultra-Filtered Dataset Downloader v4.4 - TRONCAGE AU NIVEAU DOCUMENT
CORRECTIF: Troncage intelligent qui ne coupe JAMAIS une phrase/document

CHANGEMENTS v4.4:
✅ Troncage au niveau DOCUMENT (pas de phrases coupées)
✅ Suppression logique de skip bugguée (reprise via checkpoints uniquement)
✅ Tolérance intelligente: accepte N tokens ±500K (≈1-2 documents)
✅ Garantit des documents COMPLETS dans le dataset final

ARCHITECTURE:
  Download DCLM 250M:
  ├─ 0-100M:   Collecte docs complets → checkpoint_1.npy (exactement 100M)
  ├─ 100-200M: Collecte docs complets → checkpoint_2.npy (exactement 100M)
  ├─ 200-250M: Collecte jusqu'à atteindre ≥250M
  └─ FUSION:   checkpoint_1 + checkpoint_2 + RAM → MERGE
               → TRUNCATE au dernier document COMPLET ≤250.5M
               → Résultat: 249.5M-250.5M tokens (tous docs complets)
               → dclm_baseline.npy
               → DELETE temp_checkpoints/

USAGE Lightning.AI (4h limit):
  python download_ultra_filtered_mistral_v44.py --num-chunks 1
  → Si timeout: relance, reprend au dernier checkpoint (max 99M tokens perdus)
"""

import os
import sys
import json
import torch
import psutil
import hashlib
import signal
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Tuple
import shutil

# Check and install zstandard if needed
try:
    import zstandard
except ImportError:
    print("📦 Installing zstandard for DCLM dataset...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "zstandard", "-q"])
    import zstandard
    print("✅ zstandard installed!")

import numpy as np

# ============================================
# TIMEOUT HANDLER (3h30 = 12,600 seconds)
# ============================================
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("⏰ Timeout 3h30 atteint!")

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    'output_dir': './data/ultra_filtered',
    'checkpoint_dir': './temp_checkpoints',
    'tokenizer_name': 'mistralai/Mistral-7B-v0.1',
    'batch_size_docs': 20000,
    'num_chunks': 10,
    'tokens_per_chunk': 1_000_000_000,
    
    # ✅ Checkpoint system
    'checkpoint_interval': 100_000_000,  # 100M tokens per checkpoint
    
    # ✅ v4.4: Tolérance pour troncage au niveau document
    'token_tolerance': 500_000,  # Accepte ±500K tokens (≈1-2 gros documents)
    
    # Timeout protection (3h30 pour Lightning.AI)
    'dataset_timeout': 12600,  # 3h30 en secondes
    
    # Filtres ultra-stricts
    'min_text_length': 500,
    'max_text_length': 100000,
    'min_alpha_ratio': 0.7,
    'max_special_chars_ratio': 0.15,
    'min_avg_word_length': 3.0,
    'max_avg_word_length': 12.0,
    'min_unique_words_ratio': 0.4,
    'max_line_repetition_ratio': 0.3,
    
    # Deduplication
    'enable_dedup': True,
    'dedup_method': 'hash',
}

# ============================================
# DATASETS - DISTRIBUTION 25-15-10-20-10
# ============================================
DATASETS = [
    {
        'name': 'dclm_baseline',
        'source': 'mlfoundations/dclm-baseline-1.0',
        'config': None,
        'split': 'train',
        'text_key': 'text',
        'streaming': True,
        'description': '🌍 DCLM (diversité conversationnelle)',
        'tokens_per_chunk': 250_000_000,
    },
    {
        'name': 'refinedweb',
        'source': 'tiiuae/falcon-refinedweb',
        'config': 'default',
        'split': 'train',
        'text_key': 'content',
        'streaming': True,
        'description': '🦅 RefinedWeb (web premium Falcon)',
        'tokens_per_chunk': 150_000_000,
    },
    # finepdfs_edu commenté temporairement (bloqué)
    #{
    #    'name': 'finepdfs_edu',
    #    'source': 'HuggingFaceFW/fineweb-edu',
    #    'config': 'sample-10BT',
    #    'split': 'train',
    #    'text_key': 'text',
    #    'streaming': True,
    #    'description': '📄 FineWeb-Edu PDF-like (structure académique)',
    #    'tokens_per_chunk': 200_000_000,
    #    'filter_for_academic': True,
    #},
    {
        'name': 'fineweb_edu',
        'source': 'HuggingFaceFW/fineweb-edu',
        'config': 'sample-10BT',
        'split': 'train',
        'text_key': 'text',
        'streaming': True,
        'description': '🎓 FineWeb-Edu (éducatif accessible)',
        'tokens_per_chunk': 100_000_000,
    },
    {
        'name': 'cosmopedia_v2',
        'source': 'HuggingFaceTB/cosmopedia-v2',
        'config': 'default',
        'split': 'train',
        'text_key': 'text',
        'streaming': True,
        'description': '🌌 Cosmopedia v2 (synthétique haute qualité)',
        'tokens_per_chunk': 200_000_000,
    },
    {
        'name': 'wikipedia',
        'source': 'wikimedia/wikipedia',
        'config': '20231101.en',
        'split': 'train',
        'text_key': 'text',
        'streaming': True,
        'description': '📚 Wikipedia (connaissances factuelles)',
        'tokens_per_chunk': 100_000_000,
    },
]

# ============================================
# TOKENS SPÉCIAUX - MISTRAL
# ============================================
SPECIAL_TOKENS = {
    '<|system|>': 32000,
    '<|user|>': 32001,
    '<|assistant|>': 32002,
    '<|end|>': 32003,
    '<code>': 32004,
}

# ============================================
# FILTERS (same as v4.3)
# ============================================
CODE_PATTERNS = [
    r'\bdef\s+\w+\s*\(', r'\bfunction\s+\w+\s*\(', r'\bclass\s+\w+\s*[:{]',
    r'\bimport\s+\w+', r'\bfrom\s+\w+\s+import', r'#include\s*<',
    r'\bpublic\s+static\s+void', r'\bprivate\s+\w+\s+\w+\s*\(',
    r'=>\s*{', r'\bconst\s+\w+\s*=', r'\bvar\s+\w+\s*=', r'\blet\s+\w+\s*=',
    r'SELECT\s+.+FROM', r'INSERT\s+INTO', r'CREATE\s+TABLE',
    r'```\w+', r'<\?php', r'<script>', r'</script>',
    r'System\.out\.println', r'console\.log', r'printf\s*\(',
]

MATH_PATTERNS = [
    r'\\begin\{equation\}', r'\\frac\{', r'\\sum_', r'\\int_', r'\\sqrt\{',
    r'\$\$', r'\\alpha|\\beta|\\gamma|\\delta',
    r'Theorem\s+\d+', r'Lemma\s+\d+', r'Proof\.', r'Q\.E\.D\.',
    r'\b[a-z]\s*=\s*[a-z]\s*\+\s*[a-z]\b', r'\bf\(x\)\s*=', r'\d+x\s*[\+\-]\s*\d+',
]

CODE_REGEX = [re.compile(p, re.IGNORECASE) for p in CODE_PATTERNS]
MATH_REGEX = [re.compile(p, re.IGNORECASE) for p in MATH_PATTERNS]

def contains_code_or_math(text: str) -> bool:
    for pattern in CODE_REGEX:
        if pattern.search(text):
            return True
    for pattern in MATH_REGEX:
        if pattern.search(text):
            return True
    return False

def filter_academic_structure(text: str) -> bool:
    if len(text) < 1500:
        return False
    academic_markers = [
        'introduction', 'conclusion', 'abstract', 'summary',
        'methodology', 'results', 'discussion', 'references',
        'figure', 'table', 'section', 'chapter',
        'furthermore', 'therefore', 'however', 'moreover',
        'consequently', 'nevertheless', 'accordingly'
    ]
    text_lower = text.lower()
    marker_count = sum(1 for marker in academic_markers if marker in text_lower)
    if marker_count < 3:
        return False
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 5:
        return False
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
    if avg_sentence_length < 15:
        return False
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) < 3:
        return False
    return True

def filter_document(text: str, filter_academic: bool = False) -> bool:
    if len(text) < CONFIG['min_text_length'] or len(text) > CONFIG['max_text_length']:
        return False
    if contains_code_or_math(text):
        return False
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars / len(text) < CONFIG['min_alpha_ratio']:
        return False
    special_chars = sum(not c.isalnum() and not c.isspace() for c in text)
    if special_chars / len(text) > CONFIG['max_special_chars_ratio']:
        return False
    if text.count('http') > 3 or text.count('www.') > 3:
        return False
    spam_patterns = [
        'click here', 'buy now', 'subscribe', 'follow us',
        'copyright ©', 'all rights reserved', 'terms of service',
        'cookies policy', 'privacy policy'
    ]
    text_lower = text.lower()
    spam_count = sum(text_lower.count(pattern) for pattern in spam_patterns)
    if spam_count > 2:
        return False
    words = text.split()
    if len(words) < 50:
        return False
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < CONFIG['min_avg_word_length'] or avg_word_len > CONFIG['max_avg_word_length']:
        return False
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < CONFIG['min_unique_words_ratio']:
        return False
    lines = text.split('\n')
    if len(lines) > 10:
        unique_lines = len(set(lines))
        line_repetition = 1 - (unique_lines / len(lines))
        if line_repetition > CONFIG['max_line_repetition_ratio']:
            return False
    sentences_end = text.count('.') + text.count('!') + text.count('?')
    if sentences_end < len(words) / 30:
        return False
    if filter_academic:
        if not filter_academic_structure(text):
            return False
    return True

class DocumentDeduplicator:
    def __init__(self):
        self.seen_hashes = set()
        self.num_duplicates = 0
    
    def compute_hash(self, text: str) -> str:
        normalized = ' '.join(text.lower().split())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, text: str) -> bool:
        text_hash = self.compute_hash(text)
        if text_hash in self.seen_hashes:
            self.num_duplicates += 1
            return True
        self.seen_hashes.add(text_hash)
        return False
    
    def get_stats(self) -> Dict:
        return {
            'unique_docs': len(self.seen_hashes),
            'duplicates_found': self.num_duplicates,
            'dedup_rate': self.num_duplicates / max(len(self.seen_hashes) + self.num_duplicates, 1) * 100
        }

# ============================================
# DOCUMENT TRACKER - Pour troncage intelligent
# ============================================
class DocumentTracker:
    """Track documents and their token boundaries for intelligent truncation"""
    def __init__(self):
        self.documents = []  # List of (start_idx, end_idx, num_tokens)
        self.current_position = 0
    
    def add_document(self, num_tokens: int):
        """Add a document with its token count"""
        start = self.current_position
        end = start + num_tokens
        self.documents.append((start, end, num_tokens))
        self.current_position = end
    
    def find_truncation_point(self, target_tokens: int, tolerance: int) -> Tuple[int, int]:
        """
        Find best truncation point that doesn't cut documents
        
        Args:
            target_tokens: Desired number of tokens
            tolerance: Acceptable deviation (±tolerance)
        
        Returns:
            (truncation_index, num_docs_kept)
        """
        if not self.documents:
            return 0, 0
        
        # Find last complete document that fits within target + tolerance
        best_idx = 0
        best_docs = 0
        
        for i, (start, end, num_tokens) in enumerate(self.documents):
            if end <= target_tokens + tolerance:
                best_idx = end
                best_docs = i + 1
            else:
                break
        
        return best_idx, best_docs
    
    def get_stats(self) -> Dict:
        """Get statistics about documents"""
        if not self.documents:
            return {
                'num_docs': 0,
                'total_tokens': 0,
                'avg_tokens_per_doc': 0,
                'min_tokens': 0,
                'max_tokens': 0,
            }
        
        token_counts = [doc[2] for doc in self.documents]
        return {
            'num_docs': len(self.documents),
            'total_tokens': self.current_position,
            'avg_tokens_per_doc': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
        }

# ============================================
# DOWNLOADER v4.4 (TRONCAGE AU NIVEAU DOCUMENT)
# ============================================
class UltraFilteredDownloader:
    def __init__(self):
        self.output_dir = Path(CONFIG['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = Path(CONFIG['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print("🔥 Ultra-Filtered Downloader v4.4 (Troncage Document)")
        print(f"📊 Distribution: 25-15-10-20-10 (finepdfs_edu skipped)")
        print(f"💾 Checkpoints: Tous les {CONFIG['checkpoint_interval']/1e6:.0f}M tokens")
        print(f"⏰ Timeout: {CONFIG['dataset_timeout']/3600:.1f}h par dataset")
        print(f"✅ NOUVEAU: Troncage au niveau DOCUMENT (pas de phrases coupées)")
        print(f"📏 Tolérance: ±{CONFIG['token_tolerance']/1e3:.0f}K tokens par dataset")
        
        print(f"\n📝 Loading Mistral tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['tokenizer_name'])
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': list(SPECIAL_TOKENS.keys())
        })
        
        print(f"   ✅ Mistral tokenizer ready ({len(self.tokenizer)} tokens)")
        
        if CONFIG['enable_dedup']:
            self.deduplicator = DocumentDeduplicator()
            print(f"   ✅ Deduplication enabled")
        else:
            self.deduplicator = None
        
        self.state_file = self.output_dir / 'downloader_state.json'
        self.load_state()
    
    def load_state(self):
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
            print(f"✅ État chargé: {self.state['completed_chunks']} chunks complétés")
        else:
            self.state = {'completed_chunks': 0}
            print("🆕 Nouvel état créé")
    
    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_existing_checkpoints(self, name: str) -> List[Path]:
        """Get existing checkpoint files for a dataset"""
        pattern = f"{name}_checkpoint_*.npy"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        return checkpoints
    
    def get_checkpoint_tokens_count(self, checkpoints: List[Path]) -> int:
        """Count total tokens in existing checkpoints"""
        if not checkpoints:
            return 0
        return len(checkpoints) * CONFIG['checkpoint_interval']
    
    def save_checkpoint(self, name: str, checkpoint_num: int, tokens: List[int]) -> Path:
        """Save a checkpoint file"""
        checkpoint_file = self.checkpoint_dir / f"{name}_checkpoint_{checkpoint_num}.npy"
        tokens_array = np.array(tokens, dtype=np.int32)
        np.save(checkpoint_file, tokens_array)
        return checkpoint_file
    
    def merge_checkpoints_smart(
        self, 
        name: str, 
        checkpoints: List[Path], 
        final_tokens: List[int],
        doc_tracker: DocumentTracker,
        target_tokens: int
    ) -> np.ndarray:
        """
        ✅ v4.4: Merge checkpoints + truncate at DOCUMENT level
        
        Args:
            name: Dataset name
            checkpoints: List of checkpoint files
            final_tokens: Remaining tokens in RAM
            doc_tracker: Document tracker with boundaries
            target_tokens: Target number of tokens
        
        Returns:
            Merged array truncated at document boundary
        """
        print(f"      🔀 Fusion des checkpoints...")
        
        all_arrays = []
        
        # Load all checkpoint files
        for i, checkpoint_path in enumerate(checkpoints):
            checkpoint_data = np.load(checkpoint_path)
            all_arrays.append(checkpoint_data)
            print(f"         Checkpoint {i+1}: {len(checkpoint_data)/1e6:.1f}M tokens")
        
        # Add final tokens from RAM
        if final_tokens:
            final_array = np.array(final_tokens, dtype=np.int32)
            all_arrays.append(final_array)
            print(f"         RAM finale: {len(final_tokens)/1e6:.1f}M tokens")
        
        # Concatenate all
        merged = np.concatenate(all_arrays)
        print(f"      📊 Pré-troncage: {len(merged)/1e6:.1f}M tokens")
        
        # ✅ v4.4: SMART TRUNCATION at document boundary
        trunc_idx, num_docs_kept = doc_tracker.find_truncation_point(
            target_tokens, 
            CONFIG['token_tolerance']
        )
        
        if trunc_idx > 0:
            merged = merged[:trunc_idx]
            actual_tokens = len(merged)
            deviation = actual_tokens - target_tokens
            
            print(f"      ✂️  Troncage intelligent:")
            print(f"         Documents complets: {num_docs_kept}")
            print(f"         Tokens finaux: {actual_tokens:,} ({actual_tokens/1e6:.1f}M)")
            print(f"         Cible: {target_tokens:,} ({target_tokens/1e6:.1f}M)")
            print(f"         Déviation: {deviation:+,} tokens ({deviation/1e3:+.1f}K)")
            
            if abs(deviation) > CONFIG['token_tolerance']:
                print(f"      ⚠️  WARNING: Déviation > tolérance ({CONFIG['token_tolerance']/1e3:.0f}K)")
        else:
            print(f"      ⚠️  WARNING: Pas assez de tokens pour troncage intelligent")
        
        # Get doc stats
        doc_stats = doc_tracker.get_stats()
        print(f"      📄 Documents: {doc_stats['num_docs']:,}")
        print(f"      📊 Moy tokens/doc: {doc_stats['avg_tokens_per_doc']:.0f}")
        
        return merged
    
    def cleanup_checkpoints(self, name: str):
        """Delete all checkpoint files for a dataset"""
        checkpoints = self.get_existing_checkpoints(name)
        for checkpoint_path in checkpoints:
            checkpoint_path.unlink()
        if checkpoints:
            print(f"      🗑️  {len(checkpoints)} checkpoints supprimés")
    
    def download_dataset_for_chunk(self, dataset_config: Dict, chunk_id: int) -> Dict:
        name = dataset_config['name']
        target_tokens = dataset_config['tokens_per_chunk']
        filter_academic = dataset_config.get('filter_for_academic', False)
        
        print(f"\n   🎯 {name}: Target {target_tokens/1e6:.1f}M tokens (±{CONFIG['token_tolerance']/1e3:.0f}K)")
        if filter_academic:
            print(f"      ✅ Academic filter enabled")
        
        # ✅ CHECK EXISTING CHECKPOINTS
        existing_checkpoints = self.get_existing_checkpoints(name)
        tokens_already_downloaded = self.get_checkpoint_tokens_count(existing_checkpoints)
        
        if tokens_already_downloaded > 0:
            print(f"      🔄 REPRISE: {tokens_already_downloaded/1e6:.1f}M tokens déjà en checkpoints")
            print(f"      → Reprend à {tokens_already_downloaded/1e6:.1f}M / {target_tokens/1e6:.1f}M")
        
        # ✅ SETUP TIMEOUT (3h30)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(CONFIG['dataset_timeout'])
        
        try:
            # Load dataset
            try:
                if dataset_config['config']:
                    dataset = load_dataset(
                        dataset_config['source'],
                        dataset_config['config'],
                        split=dataset_config['split'],
                        streaming=dataset_config['streaming'],
                        trust_remote_code=True
                    )
                else:
                    dataset = load_dataset(
                        dataset_config['source'],
                        split=dataset_config['split'],
                        streaming=dataset_config['streaming'],
                        trust_remote_code=True
                    )
            except Exception as e:
                print(f"   ❌ Erreur chargement {name}: {e}")
                signal.alarm(0)
                return None
            
            # Collection stats
            all_tokens = []
            doc_tracker = DocumentTracker()  # ✅ v4.4: Track documents
            
            num_docs_total = 0
            num_docs_passed = 0
            num_docs_duplicate = 0
            num_docs_code_math = 0
            num_docs_quality = 0
            num_docs_academic = 0
            
            # ✅ Checkpoint tracking
            checkpoint_counter = len(existing_checkpoints)
            total_tokens_collected = tokens_already_downloaded
            checkpoint_files = list(existing_checkpoints)
            
            text_key = dataset_config['text_key']
            
            # Progress bar (start from checkpoint position)
            pbar = tqdm(
                total=target_tokens + CONFIG['token_tolerance'],  # Allow overshoot
                initial=tokens_already_downloaded,
                desc=f"   {name}",
                unit="tokens",
                unit_scale=True,
            )
            
            # ✅ v4.4: NO SKIP LOGIC - start fresh or from checkpoints
            
            batch_docs = []
            for doc in dataset:
                batch_docs.append(doc)
                
                if len(batch_docs) >= CONFIG['batch_size_docs']:
                    for d in batch_docs:
                        num_docs_total += 1
                        
                        text = d.get(text_key, '')
                        if not text:
                            continue
                        
                        # Apply filters
                        if CONFIG['enable_dedup'] and self.deduplicator.is_duplicate(text):
                            num_docs_duplicate += 1
                            continue
                        
                        if contains_code_or_math(text):
                            num_docs_code_math += 1
                            continue
                        
                        if not filter_document(text, filter_academic=filter_academic):
                            if filter_academic:
                                num_docs_academic += 1
                            else:
                                num_docs_quality += 1
                            continue
                        
                        # Tokenize
                        tokens = self.tokenizer.encode(text, add_special_tokens=False)
                        
                        # ✅ v4.4: Track document boundary
                        doc_tracker.add_document(len(tokens))
                        
                        all_tokens.extend(tokens)
                        total_tokens_collected += len(tokens)
                        num_docs_passed += 1
                        
                        pbar.update(len(tokens))
                        
                        # ✅ CHECK IF CHECKPOINT NEEDED (every 100M tokens)
                        if total_tokens_collected >= (checkpoint_counter + 1) * CONFIG['checkpoint_interval']:
                            checkpoint_counter += 1
                            
                            # Save checkpoint
                            checkpoint_path = self.save_checkpoint(
                                name, checkpoint_counter, all_tokens
                            )
                            checkpoint_files.append(checkpoint_path)
                            
                            print(f"\n      💾 Checkpoint {checkpoint_counter} sauvegardé: {len(all_tokens)/1e6:.1f}M tokens")
                            print(f"         Fichier: {checkpoint_path.name}")
                            
                            # Clear RAM (keep only overflow tokens)
                            overflow = total_tokens_collected - (checkpoint_counter * CONFIG['checkpoint_interval'])
                            if overflow > 0:
                                all_tokens = all_tokens[-overflow:]
                            else:
                                all_tokens = []
                        
                        # ✅ Check target with tolerance
                        if total_tokens_collected >= target_tokens + CONFIG['token_tolerance']:
                            break
                    
                    batch_docs = []
                    
                    if total_tokens_collected >= target_tokens + CONFIG['token_tolerance']:
                        break
            
            pbar.close()
            signal.alarm(0)  # ✅ Cancel timeout alarm
            
            # Stats
            pass_rate = (num_docs_passed / num_docs_total * 100) if num_docs_total > 0 else 0
            
            print(f"   ✅ Download terminé: collecté {total_tokens_collected/1e6:.1f}M tokens")
            print(f"      Filtrage: {num_docs_passed:,}/{num_docs_total:,} docs ({pass_rate:.1f}% pass rate)")
            
            num_rejected = num_docs_total - num_docs_passed
            if num_rejected > 0:
                print(f"      ❌ Rejetés: {num_rejected:,}")
                if num_docs_duplicate > 0:
                    print(f"         - Duplicates: {num_docs_duplicate:,} ({num_docs_duplicate/num_rejected*100:.1f}%)")
                if num_docs_code_math > 0:
                    print(f"         - Code/Math: {num_docs_code_math:,} ({num_docs_code_math/num_rejected*100:.1f}%)")
                if num_docs_quality > 0:
                    print(f"         - Qualité: {num_docs_quality:,} ({num_docs_quality/num_rejected*100:.1f}%)")
                if num_docs_academic > 0:
                    print(f"         - Non-académique: {num_docs_academic:,} ({num_docs_academic/num_rejected*100:.1f}%)")
            
            if CONFIG['enable_dedup'] and self.deduplicator:
                dedup_stats = self.deduplicator.get_stats()
                print(f"      🔄 Dedup: {dedup_stats['unique_docs']:,} unique docs, "
                      f"{dedup_stats['duplicates_found']:,} duplicates ({dedup_stats['dedup_rate']:.1f}%)")
            
            # ✅ RETURN: checkpoints + final tokens + doc tracker
            return {
                'checkpoints': checkpoint_files,
                'final_tokens': all_tokens,
                'doc_tracker': doc_tracker,
                'target_tokens': target_tokens,
                'num_docs': num_docs_passed,
                'num_docs_total': num_docs_total,
                'num_docs_duplicate': num_docs_duplicate,
                'num_docs_code_math': num_docs_code_math,
                'num_docs_quality': num_docs_quality,
                'num_docs_academic': num_docs_academic,
                'pass_rate': pass_rate,
            }
        
        except TimeoutError:
            signal.alarm(0)
            print(f"\n   ⏰ TIMEOUT 3h30 atteint pour {name}!")
            print(f"      Checkpoints sauvés: {checkpoint_counter} × 100M = {checkpoint_counter * 100}M tokens")
            print(f"      RAM actuelle: {len(all_tokens)/1e6:.1f}M tokens (sera perdue)")
            print(f"      Perte max à la reprise: {len(all_tokens)/1e6:.1f}M tokens")
            print(f"      → Relancez le script, il reprendra au checkpoint {checkpoint_counter}")
            return None
        except Exception as e:
            signal.alarm(0)
            print(f"   ❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_chunk(self, chunk_id: int):
        """
        ✅ v4.4: Checkpoint system + document-level truncation
        """
        print(f"\n{'='*70}")
        print(f"🔥 CREATING CHUNK {chunk_id + 1}/{CONFIG['num_chunks']}")
        print(f"{'='*70}")
        
        chunk_dir = self.output_dir / f"chunk_{chunk_id:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing stats
        stats_file = chunk_dir / 'stats.json'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                chunk_stats = json.load(f)
            print(f"   📂 Chunk partiellement complété, reprise...")
        else:
            chunk_stats = {
                'chunk_id': chunk_id,
                'datasets': {},
                'total_tokens': 0,
                'total_docs': 0,
                'total_size_mb': 0,
                'timestamp': time.time(),
                'tokenizer': CONFIG['tokenizer_name'],
                'special_tokens': SPECIAL_TOKENS,
            }
        
        # Reset deduplicator
        if CONFIG['enable_dedup']:
            self.deduplicator = DocumentDeduplicator()
        
        # Process each dataset
        for dataset_config in DATASETS:
            name = dataset_config['name']
            dataset_file = chunk_dir / f"{name}.npy"
            target_tokens = dataset_config['tokens_per_chunk']
            
            # ✅ CHECK IF DATASET ALREADY EXISTS (with tolerance)
            if dataset_file.exists():
                try:
                    test_load = np.load(dataset_file, mmap_mode='r')
                    actual_tokens = len(test_load)
                    
                    # ✅ v4.4: Accept within tolerance
                    deviation = abs(actual_tokens - target_tokens)
                    if deviation <= CONFIG['token_tolerance']:
                        print(f"\n   ✅ {name}: DÉJÀ TÉLÉCHARGÉ ({actual_tokens/1e6:.1f}M tokens, {deviation/1e3:+.1f}K déviation)")
                        print(f"      Fichier: {dataset_file}")
                        
                        if name not in chunk_stats['datasets']:
                            chunk_stats['datasets'][name] = {
                                'tokens': actual_tokens,
                                'docs': 0,
                                'size_mb': dataset_file.stat().st_size / (1024 * 1024)
                            }
                            chunk_stats['total_tokens'] += actual_tokens
                            chunk_stats['total_size_mb'] += chunk_stats['datasets'][name]['size_mb']
                        
                        # ✅ CLEANUP old checkpoints for this dataset
                        self.cleanup_checkpoints(name)
                        
                        continue  # ✅ SKIP
                    else:
                        print(f"\n   ⚠️  {name}: Déviation trop grande ({actual_tokens/1e6:.1f}M, {deviation/1e3:+.1f}K)")
                        print(f"      Suppression et re-download...")
                        dataset_file.unlink()
                except Exception as e:
                    print(f"\n   ⚠️  {name}: Fichier corrompu ({e})")
                    print(f"      Suppression et re-download...")
                    if dataset_file.exists():
                        dataset_file.unlink()
            
            # ✅ DOWNLOAD DATASET (with checkpoints)
            print(f"\n   🔽 {name}: TÉLÉCHARGEMENT EN COURS...")
            result = self.download_dataset_for_chunk(dataset_config, chunk_id)
            
            if result is None:
                print(f"   ⚠️  Dataset {name} non complété (timeout ou erreur)")
                print(f"   💡 Checkpoints sauvegardés, relancez pour reprendre")
                with open(stats_file, 'w') as f:
                    json.dump(chunk_stats, f, indent=2)
                return None
            
            # ✅ v4.4: MERGE with SMART TRUNCATION at document level
            merged_tokens = self.merge_checkpoints_smart(
                name, 
                result['checkpoints'], 
                result['final_tokens'],
                result['doc_tracker'],
                result['target_tokens']
            )
            
            # Verify within tolerance
            actual_tokens = len(merged_tokens)
            deviation = abs(actual_tokens - target_tokens)
            
            if deviation > CONFIG['token_tolerance']:
                print(f"      ⚠️  WARNING: Déviation > tolérance ({deviation/1e3:.1f}K > {CONFIG['token_tolerance']/1e3:.0f}K)")
                print(f"      Acceptation quand même (documents complets prioritaires)")
            
            # Save final .npy
            np.save(dataset_file, merged_tokens)
            size_mb = dataset_file.stat().st_size / (1024 * 1024)
            print(f"      ✅ Fichier final: {dataset_file}")
            print(f"      📦 Taille: {size_mb:.1f} MB")
            print(f"      📊 Tokens: {actual_tokens:,} ({actual_tokens/1e6:.1f}M)")
            
            # ✅ CLEANUP checkpoints
            self.cleanup_checkpoints(name)
            
            # Update stats
            chunk_stats['datasets'][name] = {
                'tokens': actual_tokens,
                'tokens_target': target_tokens,
                'tokens_deviation': actual_tokens - target_tokens,
                'docs': result['num_docs'],
                'docs_total': result['num_docs_total'],
                'docs_duplicate': result.get('num_docs_duplicate', 0),
                'docs_code_math': result.get('num_docs_code_math', 0),
                'docs_quality': result.get('num_docs_quality', 0),
                'docs_academic': result.get('num_docs_academic', 0),
                'pass_rate': result['pass_rate'],
                'size_mb': size_mb
            }
            chunk_stats['total_tokens'] += actual_tokens
            chunk_stats['total_docs'] += result['num_docs']
            chunk_stats['total_size_mb'] += size_mb
            
            # Save stats
            with open(stats_file, 'w') as f:
                json.dump(chunk_stats, f, indent=2)
            print(f"      ✅ Stats sauvegardées")
        
        # Final summary
        print(f"\n{'='*70}")
        print(f"✅ CHUNK {chunk_id + 1} COMPLETED")
        print(f"{'='*70}")
        print(f"📊 Total tokens: {chunk_stats['total_tokens']/1e9:.2f}B")
        print(f"📄 Total docs: {chunk_stats['total_docs']:,}")
        print(f"💾 Total size: {chunk_stats['total_size_mb']:.1f} MB")
        
        # Update state
        self.state['completed_chunks'] = chunk_id + 1
        self.save_state()
        
        return chunk_stats
    
    def run(self):
        start_chunk = self.state['completed_chunks']
        
        print(f"\n🚀 Starting from chunk {start_chunk + 1}")
        print(f"📊 Target: {CONFIG['num_chunks']} chunks total")
        
        all_stats = []
        
        for chunk_id in range(start_chunk, CONFIG['num_chunks']):
            chunk_stats = self.create_chunk(chunk_id)
            
            if chunk_stats is None:
                print(f"\n⏰ Chunk {chunk_id + 1} interrompu")
                print(f"💾 Checkpoints sauvegardés dans: {self.checkpoint_dir}")
                print(f"💡 Relancez le script pour reprendre")
                break
            
            all_stats.append(chunk_stats)
            
            import gc
            gc.collect()
        
        if len(all_stats) > 0:
            print(f"\n{'='*70}")
            print(f"🎉 DOWNLOAD SESSION TERMINÉE")
            print(f"{'='*70}")
            
            total_tokens = sum(s['total_tokens'] for s in all_stats)
            total_docs = sum(s['total_docs'] for s in all_stats)
            total_size = sum(s['total_size_mb'] for s in all_stats)
            
            print(f"\n📊 STATISTIQUES:")
            print(f"   • Chunks: {len(all_stats)}")
            print(f"   • Tokens: {total_tokens/1e9:.2f}B")
            print(f"   • Documents: {total_docs:,}")
            print(f"   • Taille: {total_size/1024:.2f} GB")
            print(f"\n📂 Données: {self.output_dir}")
            print(f"💾 Checkpoints: {self.checkpoint_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-chunks', type=int, default=10)
    args = parser.parse_args()
    
    CONFIG['num_chunks'] = args.num_chunks
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║  🔥 ULTRA-FILTERED DATASET DOWNLOADER v4.4                   ║
║                                                               ║
║  ✅ Troncage au niveau DOCUMENT (pas de phrases coupées)     ║
║  📏 Tolérance: ±500K tokens (≈1-2 docs)                      ║
║  💾 Checkpoints: 100M tokens (perte max 99M)                 ║
║  🔀 Fusion: checkpoints → merge → smart truncate → .npy      ║
║  🗑️  Cleanup: suppression auto checkpoints après fusion      ║
║  ⏰ Timeout: 3h30 par dataset (Lightning.AI)                 ║
║                                                               ║
║  🚫 finepdfs_edu: SKIPPED (temporairement commenté)          ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    downloader = UltraFilteredDownloader()
    downloader.run()