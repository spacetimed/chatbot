import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import regex

# tokenizer spec information: ./docs/tokenizer.md

# constants used by tokenizer.json artifact
TOKENIZER_FORMAT = "chatbot-bpe"
TOKENIZER_VERSION = 1
GPT2_PATTERN_TEXT = r"'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
GPT2_PATTERN = regex.compile(GPT2_PATTERN_TEXT)


# helper functions (pretokenize, count_pairs, merge_pairs, write_tokens, read_tokens)
def pretokenize(
    text: str,
) -> list[str]:
    # partition text with boundaries (using GPT2_PATTERN)

    pieces = GPT2_PATTERN.findall(text)
    return pieces


def count_pairs(
    pieces: list[list[int]],
) -> dict[tuple[int, int], int]:

    # same functionality as old tokenizer.py (karpathy-style)

    counts: dict[tuple[int, int], int] = {}

    for piece in pieces:
        for pair in pairwise(piece):
            counts[pair] = counts.get(pair, 0) + 1

    return counts


def merge_pair(
    token_ids: list[int],
    pair: tuple[int, int],
    new_token_id: int,
) -> list[int]:

    # same functionality as old tokenizer.py (karpathy-style)

    merged = []
    index = 0

    while index < len(token_ids):
        if index < len(token_ids) - 1 and (token_ids[index], token_ids[index + 1]) == pair:
            merged.append(new_token_id)
            index += 2
        else:
            merged.append(token_ids[index])
            index += 1

    return merged


def write_tokens(
    path: Path,
    token_ids: list[int],
) -> None:

    # encodes the text and then writes it to file

    tokens = np.asarray(token_ids, dtype="<u4")
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(path)


def read_tokens(
    path: Path,
) -> list[int]:

    # same as write_tokens but reversed

    if path.stat().st_size % 4 != 0:
        raise ValueError("tokens.bin size must be divisible by four bytes")

    return np.fromfile(path, dtype="<u4").tolist()


class BPETokenizer:
    def __init__(
        self,
        mergeable_vocab_size: int,
        special_tokens: dict[str, int] | None = None,
    ) -> None:

        if mergeable_vocab_size < 256:
            raise ValueError("mergeable_vocab_size must be at least 256")

        self.mergeable_vocab_size = mergeable_vocab_size
        self.special_tokens = dict(special_tokens or {})
        self.vocab_size = mergeable_vocab_size + len(self.special_tokens)

        # special tokens
        expected_special_ids = set(range(mergeable_vocab_size, self.vocab_size))

        if any(not token for token in self.special_tokens):
            raise ValueError("special tokens cannot be empty")

        if set(self.special_tokens.values()) != expected_special_ids:
            raise ValueError("special token IDs must begin after the mergeable vocabulary")

        self.merges: dict[tuple[int, int], int] = {}
        self._reset_vocab()

    def train(
        self,
        text: str,
    ) -> None:

        pieces = [list(piece.encode("utf-8")) for piece in pretokenize(text)]

        self.merges = {}
        self._reset_vocab()

        for new_token_id in range(256, self.mergeable_vocab_size):
            pair_counts = count_pairs(pieces)

            if not pair_counts:
                break

            selected_pair = min(
                pair_counts,
                key=lambda pair: (-pair_counts[pair], pair),
            )
            pieces = [merge_pair(piece, selected_pair, new_token_id) for piece in pieces]

            self.merges[selected_pair] = new_token_id
            self.vocab[new_token_id] = self.vocab[selected_pair[0]] + self.vocab[selected_pair[1]]

    def encode(
        self,
        text: str,
        allowed_special: set[str] | None = None,
    ) -> list[int]:

        allowed_special = set(allowed_special or ())
        unknown_special = allowed_special - self.special_tokens.keys()

        if unknown_special:
            raise ValueError(f"unknown special tokens: {sorted(unknown_special)}")

        if not allowed_special:
            return self._encode_ordinary(text)

        ordered_special = sorted(
            allowed_special,
            key=lambda token: (-len(token), token),
        )
        special_pattern = regex.compile("|".join(regex.escape(token) for token in ordered_special))

        token_ids = []
        previous_end = 0

        for match in special_pattern.finditer(text):
            token_ids.extend(self._encode_ordinary(text[previous_end : match.start()]))
            token_ids.append(self.special_tokens[match.group()])
            previous_end = match.end()

        token_ids.extend(self._encode_ordinary(text[previous_end:]))

        return token_ids

    def _encode_ordinary(
        self,
        text: str,
    ) -> list[int]:

        token_ids = []

        for piece in pretokenize(text):
            token_ids.extend(self._encode_piece(piece))

        return token_ids

    def _encode_piece(
        self,
        piece: str,
    ) -> list[int]:

        token_ids = list(piece.encode("utf-8"))

        while len(token_ids) >= 2:
            mergeable_pairs = {pair for pair in pairwise(token_ids) if pair in self.merges}

            if not mergeable_pairs:
                break

            selected_pair = min(mergeable_pairs, key=self.merges.__getitem__)
            token_ids = merge_pair(token_ids, selected_pair, self.merges[selected_pair])

        return token_ids

    def decode_bytes(
        self,
        token_ids: list[int],
    ) -> bytes:

        return b"".join(self.vocab[token_id] for token_id in token_ids)

    def decode(
        self,
        token_ids: list[int],
        errors: str = "replace",
    ) -> str:

        return self.decode_bytes(token_ids).decode("utf-8", errors=errors)

    def to_dict(
        self,
    ) -> dict:

        # for tokenizer.json artifact

        return {
            "format": TOKENIZER_FORMAT,
            "version": TOKENIZER_VERSION,
            "pre_tokenizer": "gpt2",
            "mergeable_vocab_size": self.mergeable_vocab_size,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "pattern": GPT2_PATTERN_TEXT,
            "merges": [
                [left_token, right_token, new_token] for (left_token, right_token), new_token in self.merges.items()
            ],
        }

    @classmethod
    def from_dict(
        cls,
        state: dict,
    ) -> "BPETokenizer":

        if state.get("format") != TOKENIZER_FORMAT or state.get("version") != TOKENIZER_VERSION:
            raise ValueError("unsupported tokenizer artifact")

        if state.get("pre_tokenizer") != "gpt2" or state.get("pattern") != GPT2_PATTERN_TEXT:
            raise ValueError("unsupported pre-tokenizer configuration")

        tokenizer = cls(
            state["mergeable_vocab_size"],
            state.get("special_tokens"),
        )

        if state.get("vocab_size") != tokenizer.vocab_size:
            raise ValueError("tokenizer vocabulary sizes do not match")

        for expected_token, (left_token, right_token, new_token) in enumerate(state["merges"], start=256):
            if new_token != expected_token or new_token >= tokenizer.mergeable_vocab_size:
                raise ValueError("merge token IDs are not sequential")

            tokenizer.merges[(left_token, right_token)] = new_token
            tokenizer.vocab[new_token] = tokenizer.vocab[left_token] + tokenizer.vocab[right_token]

        return tokenizer

    def save(
        self,
        path: Path,
    ) -> None:

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> "BPETokenizer":

        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _reset_vocab(
        self,
    ) -> None:

        self.vocab = {token_id: bytes([token_id]) for token_id in range(256)}

        for token, token_id in self.special_tokens.items():
            self.vocab[token_id] = token.encode("utf-8")
