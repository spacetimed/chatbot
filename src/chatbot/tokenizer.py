from __future__ import annotations


class BPETokenizer:
    """A small byte-pair encoding tokenizer."""

    def __init__(self, vocab_size: int):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        self.vocab_size = vocab_size
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = self._base_vocab()

    def train(self, text: str, verbose: bool = False) -> None:
        """Learn byte-pair merges from raw text."""
        ids = list(text.encode("utf-8"))
        num_merges = self.vocab_size - 256

        self.merges = {}
        self.vocab = self._base_vocab()

        for i in range(num_merges):
            stats = self.get_pair_stats(ids)
            if not stats:
                break

            pair = max(stats, key=stats.get)
            new_id = 256 + i

            ids = self.merge_pair(ids, pair, new_id)
            self.merges[pair] = new_id

            if verbose:
                count = stats[pair]
                print(f"merge {i}: {pair} -> {new_id}, count={count}, len={len(ids)}")

        for pair, new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text: str) -> list[int]:
        """Convert raw text into token ids."""
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            stats = self.get_pair_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            new_id = self.merges[pair]
            ids = self.merge_pair(ids, pair, new_id)

        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert token ids back into text."""
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        """Return plain checkpoint-safe tokenizer state."""
        return {
            "type": "bpe",
            "vocab_size": self.vocab_size,
            "merges": [
                {"pair": list(pair), "new_id": new_id}
                for pair, new_id in self.merges.items()
            ],
        }

    @classmethod
    def from_dict(cls, state: dict) -> "BPETokenizer":
        tokenizer = cls(state["vocab_size"])
        tokenizer.merges = {
            tuple(item["pair"]): item["new_id"] for item in state["merges"]
        }
        tokenizer.vocab = tokenizer._base_vocab()

        for pair, new_id in tokenizer.merges.items():
            tokenizer.vocab[new_id] = (
                tokenizer.vocab[pair[0]] + tokenizer.vocab[pair[1]]
            )

        return tokenizer

    @staticmethod
    def get_pair_stats(ids: list[int]) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        merged_ids = []
        i = 0

        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                merged_ids.append(new_id)
                i += 2
            else:
                merged_ids.append(ids[i])
                i += 1

        return merged_ids

    @staticmethod
    def _base_vocab() -> dict[int, bytes]:
        return {idx: bytes([idx]) for idx in range(256)}
