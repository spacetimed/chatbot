# Chatbot: BPE Tokenizer Specification

This document defines the behavior shared by the Python, C++, and Rust tokenizer implementations of Chatbot. Python will serve as the readable reference implementation and vocabulary trainer. All implementations must follow the specification and produce identical token IDs for the same artifact and input.

In other words, I'm creating this document for *myself* to ensure that I am comparing the same "work" across the three tokenizers I will build.

## Data flow

Three operations will be supported: vocabulary training, encoding, and decoding. Vocabulary training creates one shared tokenizer artifact. Given that artifact and the same input text, every implementation must produce identical encoded tokens and decoded text.

**1. Vocabulary training**

```
input:  1. a corpus of text
        2. desired mergeable vocabulary size

output:  tokenizer.json (Artifact 1), where:
    - common sequences ("th", "the", etc.) are assigned new token IDs
    - the learned vocabulary and ordered merge rules are stored
```

tokenizer.json example:

```json
{
  "format": "chatbot-bpe",
  "version": 1,
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
  "pattern": "'s|'t|'re|'ve|'m|'ll|'d| ?[\\p{L}]+| ?[\\p{N}]+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
  "merges": [
    [116, 104, 256]
  ]
}
```

Each merge is stored as `[left_token, right_token, resulting_token]`. Merge order determines priority.

**2. Encoding**

```
input:  1. input.txt
        2. tokenizer.json (Artifact 1)

output:  tokens.bin (Artifact 2), where:
    - input.txt is represented as an ordered sequence of token IDs
    - token IDs are produced using the vocabulary and merge rules in tokenizer.json
```

**3. Decoding**

```
input:  1. tokens.bin (Artifact 2)
        2. tokenizer.json (Artifact 1)

output:  decoded.txt (Artifact 3), where:
    - each token ID is replaced with its corresponding byte sequence
    - byte sequences are joined in order to reproduce input.txt exactly
```

For the same input and tokenizer artifact:

```
Python tokens.bin == C++ tokens.bin == Rust tokens.bin
Python decoded.txt == C++ decoded.txt == Rust decoded.txt == input.txt
```

## Shared representation

- Input text is represented as UTF-8 bytes.
- Token IDs `[0,255]` represent corresponding byte values.
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
8. Save learned merge rules to the `merges` list in `tokenizer.json`, in the order they were created.

## Encoding (procedure)

1. Load the pre-tokenization pattern, special tokens, and ordered merge rules from `tokenizer.json`.
2. Separate explicitly enabled special tokens from ordinary text while preserving their original order.
3. Split the ordinary text using the GPT-2 pre-tokenization pattern.
4. Convert each piece into base byte token IDs.
5. Find the adjacent pair with the earliest matching merge rule within each piece.
6. Replace its non-overlapping occurrences from left to right.
7. Repeat until no merge rule can be applied.
8. Combine the resulting IDs with any special-token IDs in their original order.
9. Write the token IDs to `tokens.bin`.

## Decoding (procedure)

1. Load the vocabulary and special tokens from `tokenizer.json`.
2. Read the token IDs from `tokens.bin`.
3. Replace each token ID with the byte sequence or special-token text it represents.
4. Join the results in their original order.
5. Write the result to `decoded.txt`.

## Correctness, evaluating performance

- All implementations must produce the same pre-tokenization pieces and token IDs for the same text and `tokenizer.json`.
- Decoding an encoded input must reproduce the original input exactly.
- `tokens.bin` stores token IDs as unsigned 32-bit integers in little-endian order.
- Correctness must be verified before comparing performance.
- Core encoding and decoding benchmarks will exclude file-loading and file-writing time.
