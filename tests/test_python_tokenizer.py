import struct

from chatbot.tokenizer import BPETokenizer, pretokenize
from chatbot.tokenizer_driver import TokenizerIO

# some basic tests to ensure our python tokenizer works fine


def test_gpt2_pretokenization():
    text = "Hello, world! 123"
    pieces = pretokenize(text)

    assert pieces == ["Hello", ",", " world", "!", " 123"]
    assert "".join(pieces) == text


def test_bpe_training_encoding_and_decoding():
    tokenizer = BPETokenizer(mergeable_vocab_size=258)
    tokenizer.train("abab abab")

    assert tokenizer.merges == {
        (97, 98): 256,
        (256, 256): 257,
    }
    assert tokenizer.encode("ab") == [256]
    assert tokenizer.encode("abab") == [257]
    assert tokenizer.decode([257]) == "abab"


def test_special_tokens():
    special_tokens = {
        "<|endoftext|>": 256,
        "<|user|>": 257,
        "<|endofturn|>": 258,
    }
    tokenizer = BPETokenizer(
        mergeable_vocab_size=256,
        special_tokens=special_tokens,
    )
    text = "<|user|>hello<|endofturn|>"

    ordinary_ids = tokenizer.encode(text)
    special_ids = tokenizer.encode(
        text,
        allowed_special={"<|user|>", "<|endofturn|>"},
    )

    assert ordinary_ids == list(text.encode("utf-8"))
    assert special_ids == [257, 104, 101, 108, 108, 111, 258]
    assert tokenizer.decode(special_ids) == text


def test_tokenizer_json_round_trip(tmp_path):
    tokenizer = BPETokenizer(
        mergeable_vocab_size=258,
        special_tokens={"<|endoftext|>": 258},
    )
    tokenizer.train("abab abab")
    artifact_path = tmp_path / "rules.json"

    TokenizerIO.save_rules(artifact_path, tokenizer.to_dict())
    loaded = BPETokenizer.from_dict(TokenizerIO.load_rules(artifact_path))

    assert loaded.to_dict() == tokenizer.to_dict()
    assert loaded.encode("abab") == [257]
    assert loaded.decode([258]) == "<|endoftext|>"


def test_binary_token_round_trip(tmp_path):
    token_ids = [1, 256, 50_260]
    artifact_path = tmp_path / "tokens.bin"

    TokenizerIO.save_tokens(artifact_path, token_ids)

    assert artifact_path.read_bytes() == struct.pack("<III", *token_ids)
    assert TokenizerIO.load_tokens(artifact_path) == token_ids
