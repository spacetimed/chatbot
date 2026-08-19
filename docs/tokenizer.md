# Chatbot: Tokenizer specification

Three different versions of the tokenizer will be created in Python, C++, and Rust. This document defines the global behavior that each implementation must follow to ensure benchmarking stays fair.

The Python implementation (`tokenizer.py`) will serve as the readable reference baseline implementation. The C++ and Rust implementations (`tokenizer.cpp`, `tokenizer.rs`) must deterministically create the same output as this Python baseline.

## File structure

- `tokenizer_driver.py` (see **Entry point: Running the driver**) provides the command-line interface used to run and benchmark a selected tokenizer implementation. It supplies input files and configuration, then saves the resulting artifacts.
- `tokenizer.py`, `tokenizer.cpp`, and `tokenizer.rs` implement this specification in Python, C++, and Rust. Each implementation must be able to train a vocabulary, encode text, and decode token IDs.

The implementations may use different internal algorithms and optimizations, but must produce the same merge rules and token IDs given the same corpus and configuration.

## Entry point: Running the driver

The driver is basically just the harness to run one of the three implementations. It's the outer-file which lets you select the language, the mode of operation, and options such as a benchmarking mode.

```sh
python -m chatbot.tokenizer_driver <command> --language {python,cpp,rust} [options]
```

Command:

- `train`: Train a vocabulary from a corpus and write learned merge rules to `rules.json`.
- `encode`: Load `rules.json`, encode input text, and write `tokens.bin`.
- `decode`: Load `rules.json`, decode `tokens.bin`, and write `decoded.txt`.

Options:

- `--language {python,cpp,rust}`: Select the tokenizer implementation.
- `--vocab-size N`: Set the mergeable vocabulary size during vocabulary training.
- `--benchmark`: Provide performance output for benchmarking.

## Data flow

Three operations will be supported: training, encoding, and decoding.

### 1. Training

Training learns a vocabulary from a text corpus.

Inputs:

- A corpus of training text.
- A tokenizer configuration, including the desired mergeable vocabulary size and special tokens.

Output:

- `rules.json`, containing the pre-tokenization configuration and ordered merge rules needed to reconstruct the learned vocabulary.

A sample `rules.json` output from one of the runs:

```json
{
  "format": "bpe",
  "language": "py",
  "pre_tokenizer": "gpt2",
  "mergeable_vocab_size": 50256,
  "vocab_size": 50261,
  "special_tokens": {
    "<|endoftext|>": 50256,
    "<|system|>": 50257,
    "<|user|>": 50258,
    "<|assistant|>": 50259,
    "<|endofturn|>": 50260
  },
  "regex": "'s|'t|'re|'ve|'m|'ll|'d| ?[\\p{L}]+| ?[\\p{N}]+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
  "merges": [
    [116, 104, 256]
  ]
}
```

Each merge is stored as `[left_token, right_token, resulting_token]`. The order of the merges determines their priority.

### 2. Encoding

Encoding applies a trained vocabulary to text.

Inputs:

- Input text.
- The trained `rules.json` artifact.
- The set of special tokens, if any, that may be recognized in the input.

Output:

- `tokens.bin`, containing the encoded text as an ordered sequence of token IDs.

### 3. Decoding

Decoding converts token IDs back into plaintext.

Inputs:

- A `tokens.bin` file containing an ordered sequence of token IDs.
- The same `rules.json` artifact used to encode the text.

Output:

- `decoded.txt`, created by replacing each token ID with its byte sequence and joining those sequences in their original order.


## Shared representation

- Input text is converted into UTF-8 bytes.
- Token IDs `[0, 255]` represent their corresponding byte values.
- Learned token IDs begin at `256`, increasing sequentially.
- `mergeable_vocab_size` includes the `256` base byte tokens and all learned tokens.
- Special-token IDs begin after the mergeable vocabulary.
- `vocab_size` includes byte tokens, learned tokens, and special tokens.
- Use the GPT-2 pre-tokenization regex pattern:

```regex
's|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Pre-tokenization must preserve all text. BPE is applied separately to each resulting piece, and merges cannot cross piece boundaries.

Special tokens are recognized only when explicitly enabled by the caller. Otherwise, their written forms are encoded as ordinary text. Recognized special tokens bypass pre-tokenization and BPE because they already have fixed token IDs.

`<|endoftext|>` separates training documents. The role and end-of-turn tokens format conversations during supervised fine-tuning and inference.

## Vocabulary training (procedure)

1. Split training text using the GPT-2 pre-tokenization pattern.
2. Convert each piece into base byte token IDs.
3. Count every adjacent pair within each piece. Pairs cannot cross piece boundaries.
4. Select the pair which occurs most frequently (if multiple pairs have the same frequency, select the smallest `(left_id, right_id)` pair).
5. Assign the selected pair the next available token ID.
6. Replace its non-overlapping occurrences from left to right.
7. Repeat until the desired mergeable vocabulary size is reached or no pairs remain.
8. Save learned merge rules to the `merges` list in `rules.json`, in the order they were created.

## Encoding (procedure)

1. Load the pre-tokenization pattern, special tokens, and ordered merge rules from `rules.json`.
2. Separate explicitly enabled special tokens from ordinary text while preserving their original order.
3. Split the ordinary text using the GPT-2 pre-tokenization pattern.
4. Convert each piece into base byte token IDs.
5. Find the adjacent pair with the earliest matching merge rule within each piece.
6. Replace its non-overlapping occurrences from left to right.
7. Repeat until no merge rule can be applied.
8. Combine the resulting IDs with any special-token IDs in their original order.
9. Write the token IDs to `tokens.bin`.

## Decoding (procedure)

1. Load the vocabulary and special tokens from `rules.json`.
2. Read the token IDs from `tokens.bin`.
3. Replace each token ID with the byte sequence or special-token text it represents.
4. Join the results in their original order.
5. Write the result to `decoded.txt`.

## Correctness and performance

- All implementations must produce the same pre-tokenization pieces and token IDs for the same text and `rules.json`.
- Decoding an encoded input must reproduce the original input exactly.
- `rules.json` files are compared by their parsed contents rather than JSON whitespace or key order.
- `tokens.bin` and `decoded.txt` outputs are compared byte-for-byte.
- `tokens.bin` stores token IDs as unsigned 32-bit integers in little-endian order.
- Correctness must be verified before comparing performance.
- Training, encoding, and decoding are benchmarked separately.
- Core benchmarks exclude driver startup, file-loading, and file-writing time.
