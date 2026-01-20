"""
Improved BPE (Byte Pair Encoding) Tokenizer
Learns common subwords and words from training data
Much more efficient than character-level tokenization
"""

import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import pickle

class BPETokenizer:
    """
    Simple BPE tokenizer that learns subword units from text.
    This allows the model to understand common words as single tokens.
    """
    
    def __init__(self, vocab_size=2000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '<unk>': 3
        }
        # Reverse mapping
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        self.token_to_id = self.special_tokens.copy()
        
        # BPE merge rules
        self.merges = []
        self.cache = {}
        
    def train(self, texts: List[str], verbose=True):
        """Train BPE on a corpus of texts"""
        if verbose:
            print(f"\n[BPE] Training tokenizer on {len(texts)} texts...")
            print(f"[BPE] Target vocabulary size: {self.vocab_size}")
        
        # Step 1: Initialize with character-level vocabulary
        vocab = Counter()
        for text in texts:
            # Pre-tokenize: split on whitespace and add end-of-word marker
            words = text.lower().split()
            for word in words:
                word = ' '.join(list(word)) + ' </w>'
                vocab[word] += 1
        
        if verbose:
            print(f"[BPE] Initial vocabulary: {len(vocab)} unique words")
        
        # Step 2: Learn merges
        num_merges = self.vocab_size - len(self.special_tokens) - 256  # Reserve space for bytes
        
        for i in range(num_merges):
            # Find most frequent pair
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"[BPE] Learned {i+1}/{num_merges} merges... (latest: {best_pair})")
        
        if verbose:
            print(f"[BPE] Training complete! Learned {len(self.merges)} merge rules")
        
        # Step 3: Build final vocabulary
        self._build_vocab(vocab)
        
        if verbose:
            print(f"[BPE] Final vocabulary size: {len(self.token_to_id)}")
            print(f"[BPE] Sample tokens: {list(self.token_to_id.keys())[:20]}")
    
    def _get_stats(self, vocab: Dict[str, int]) -> Counter:
        """Count frequency of adjacent pairs"""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """Merge a pair in the vocabulary"""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
        
        return new_vocab
    
    def _build_vocab(self, vocab: Dict[str, int]):
        """Build token-to-id mapping from learned merges"""
        # Start with special tokens
        current_id = len(self.special_tokens)
        
        # Add all unique tokens from vocabulary
        all_tokens = set()
        for word in vocab.keys():
            tokens = word.split()
            all_tokens.update(tokens)
        
        # Sort for deterministic ordering
        for token in sorted(all_tokens):
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using learned merges"""
        if word in self.cache:
            return self.cache[word]
        
        # Start with characters
        word = ' '.join(list(word.lower())) + ' </w>'
        
        # Apply merge rules
        for pair in self.merges:
            bigram = ' '.join(pair)
            if bigram in word:
                word = word.replace(bigram, ''.join(pair))
        
        tokens = word.split()
        self.cache[word] = tokens
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if not isinstance(text, str):
            text = str(text)
        
        # Split into words
        words = text.split()
        
        # Tokenize each word
        token_ids = []
        for word in words:
            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                token_id = self.token_to_id.get(token, self.special_tokens['<unk>'])
                token_ids.append(token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for tid in token_ids:
            tid = int(tid)  # Handle numpy types
            if tid in self.id_to_token:
                token = self.id_to_token[tid]
                # Skip special tokens in output
                if token not in ['<pad>', '<sos>', '<eos>', '<unk>']:
                    tokens.append(token)
        
        # Join tokens and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()
    
    def save(self, path: str):
        """Save tokenizer to disk"""
        data = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'token_to_id': self.token_to_id,
            'id_to_token': {int(k): v for k, v in self.id_to_token.items()},
            'merges': self.merges
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[BPE] Saved tokenizer to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load tokenizer from disk"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.special_tokens = data['special_tokens']
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        tokenizer.merges = [tuple(pair) for pair in data['merges']]
        
        print(f"[BPE] Loaded tokenizer from {path}")
        print(f"[BPE] Vocabulary size: {len(tokenizer.token_to_id)}")
        return tokenizer


# Simple wrapper for compatibility with existing code
class SimpleTokenizer:
    """
    Compatibility wrapper - now uses BPE internally
    """
    def __init__(self, bpe_tokenizer=None):
        if bpe_tokenizer is None:
            # Create default BPE tokenizer
            self.tokenizer = BPETokenizer(vocab_size=2000)
        else:
            self.tokenizer = bpe_tokenizer
        
        self.special_tokens = self.tokenizer.special_tokens
        self.vocab_size = len(self.tokenizer.token_to_id)
    
    def encode(self, text):
        return self.tokenizer.encode(text)
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
    
    def save(self, path):
        self.tokenizer.save(path)
    
    @classmethod
    def load(cls, path):
        bpe = BPETokenizer.load(path)
        return cls(bpe_tokenizer=bpe)
