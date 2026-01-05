import json
import os

class SimpleTokenizer:
    """
    A byte-level BPE-style tokenizer that requires no pre-training for basic usage.
    It maps bytes to tokens, ensuring 100% coverage (no <unk>).
    """
    def __init__(self):
        self.encoder = {i: i for i in range(256)}
        self.decoder = {i: i for i in range(256)}
        self.vocab_size = 256
        self.special_tokens = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2
        }
        # Ideally shift regular bytes to avoid collision with specials if we used them,
        # but for this simple "Experimental" version, we treat bytes as raw IDs.
        # To be cleaner:
        self.byte_offset = 3 
        self.vocab_size = 256 + 3

    def encode(self, text):
        """
        Encodes text into a list of token IDs.
        """
        if isinstance(text, str):
            # Encode string to bytes
            b = text.encode('utf-8')
        else:
            b = text
        
        # Map bytes to IDs (shifting by offset to reserve 0,1,2 for specials)
        ids = [x + self.byte_offset for x in b]
        return ids

    def decode(self, tokens):
        """
        Decodes a list of token IDs back to a string.
        """
        bytes_list = []
        for t in tokens:
            # Handle numpy scalar types
            t_int = int(t)
            if t_int >= self.byte_offset:
                byte_val = t_int - self.byte_offset
                # Safety: clip to valid byte range
                if 0 <= byte_val < 256:
                    bytes_list.append(byte_val)
        
        return bytes(bytes_list).decode('utf-8', errors='replace')

    def save(self, path):
        pass # Nothing to save for stateless tokenizer

    @classmethod
    def load(cls, path):
        return cls()
