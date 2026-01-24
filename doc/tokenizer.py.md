# tokenizer.py

## Overview
A lightweight, byte-level tokenizer designed for universal coverage. It serves as the baseline tokenizer for the Paradox Framework, ensuring that any input text can be processed without "Out of Vocabulary" (OOV) errors.

## Purpose
`tokenizer.py` provides a fail-safe way to convert text into numbers. By operating at the byte level rather than the word or character level, it can handle any UTF-8 string, including emojis, special characters, and multiple languages, while keeping the vocabulary size very small (259 tokens).

## Components

### `SimpleTokenizer` (Class)
A stateless tokenizer that maps raw bytes to integers.

#### Features
- **100% Coverage**: Since it maps bytes (0-255), it can never encounter a character it doesn't know.
- **Special Tokens**: Reserves the first 3 IDs for framework control:
  - `0`: `<pad>` (Padding)
  - `1`: `<sos>` (Start of Sentence)
  - `2`: `<eos>` (End of Sentence)
- **Byte Shifting**: All raw byte values are shifted by `3` (the `byte_offset`) so that byte `0` becomes ID `3`, and so on, up to ID `258`.

#### Methods
- **`encode(text)`**: 
  - Converts string input into UTF-8 bytes.
  - Returns a list of shifted integer IDs.
- **`decode(tokens)`**: 
  - Subtracts the `byte_offset`.
  - Converts the resulting bytes back into a UTF-8 string.
  - Uses `errors='replace'` to handle partial or corrupted byte sequences gracefully.
- **`save()` / `load()`**: Included for API compatibility with other tokenizers, but since this tokenizer is logic-based rather than learned, these methods are effectively no-ops.

## Comparison with BPE Tokenizer
| Feature | SimpleTokenizer | BPETokenizer |
| :--- | :--- | :--- |
| **Vocab Size** | Fixed (259) | Learned (e.g., 2000+) |
| **Granularity** | Byte-level | Subword-level |
| **Efficiency** | Lower (more tokens per word) | Higher (fewer tokens per word) |
| **Coverage** | 100% Guaranteed | High (has `<unk>`) |
| **Pre-training** | None required | Required on corpus |

## Usage
This tokenizer is ideal for:
- Initial base training before a BPE corpus is ready.
- Processing highly technical or garbled text.
- Minimal-memory constraints.
