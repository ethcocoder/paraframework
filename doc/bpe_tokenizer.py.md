# bpe_tokenizer.py

## Overview
An implementation of the BPE (Byte Pair Encoding) subword tokenization algorithm. It provides the `BPETokenizer` class for learning and applying subword units, making text processing significantly more efficient than character-level methods.

## Purpose
Subword tokenization strikes a balance between character-level (very fine) and word-level (too coarse) tokenization. It allows the model to learn a fixed-size vocabulary that can represent any text, including rare or unseen words, by breaking them down into meaningful sub-units.

## Key Features
- **Vocab Learning**: Can be trained on any text corpus to learn the most frequent character combinations.
- **Subword Mapping**: Automatically merges frequent pairs into single tokens.
- **Special Tokens**: Supports `<pad>`, `<sos>`, `<eos>`, and `<unk>` tokens natively.
- **Persistence**: Full support for saving and loading learned vocabularies and merge rules to/from JSON.
- **Performance**: Includes a caching mechanism (`self.cache`) to speed up tokenization of repeated words.

## Components

### `BPETokenizer` (Class)

#### Core Methods
- **`__init__(vocab_size=2000)`**: Sets up the initial special tokens and reverse mappings.
- **`train(texts, verbose=True)`**: 
  - Iteratively finds the most frequent adjacent character/token pairs in the corpus.
  - Merges these pairs into new tokens until the `vocab_size` is reached.
  - Handles pre-tokenization (lowercase + word boundary markers `</w>`).
- **`encode(text)`**: Converts a string into a list of integer IDs using learned merge rules.
- **`decode(token_ids)`**: Reasons backwards from IDs to text, handling word boundaries and ignoring special tokens in the final output.
- **`save(path)` / `load(path)`**: Serializes the vocabulary and merge rules to a JSON file.

#### Internal Helpers
- **`_get_stats(vocab)`**: Calculates frequencies of all adjacent pairs.
- **`_merge_vocab(pair, vocab)`**: Applies a merge rule to the training vocabulary.
- **`_tokenize_word(word)`**: Appies merge rules to a single word.

### `SimpleTokenizer` (Class)
A compatibility wrapper that presents a standard interface used by other parts of the Paradox framework while using the BPE engine internally.

## Implementation Details
- **Pre-tokenization**: Words are treated as units, and an `</w>` suffix is added to each word to distinguish between tokens that end words and those that are internal segments.
- **Determinism**: Sorting is used during vocabulary building to ensure consistent ID assignment.
- **Efficiency**: Tokenization of a word applies merges in the order they were learned, ensuring the most frequent (and usually longest) subwords are prioritized.

## Configuration
Typical usage involves a `vocab_size` between 2000 and 10000. Larger sizes allow for more "whole word" tokens but require more training data and model capacity.
