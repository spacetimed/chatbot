import numpy as np

from chatbot._tokenizer_cpp import BPETokenizer as CppBPETokenizer

from chatbot.tokenizer import BPETokenizer as PythonBPETokenizer


def test_cpp_matches_python_tokenizer():
    mergeable_vocab_size = 272
    special_tokens = {
        "<|endoftext|>": 272,
        "<|user|>": 273,
        "<|assistant|>": 274,
    }
    corpus = "Hello, world! It's naïve. Numbers 12345.\nhello hello hello"

    python_tokenizer = PythonBPETokenizer(mergeable_vocab_size, special_tokens)
    cpp_tokenizer = CppBPETokenizer(mergeable_vocab_size, special_tokens)

    python_tokenizer.train(corpus)
    cpp_tokenizer.train(corpus)

    python_state = python_tokenizer.to_dict()
    cpp_state = cpp_tokenizer.to_dict()
    cpp_state["language"] = "py"

    assert cpp_state == python_state

    text = "<|user|>Hello, naïve world!<|assistant|>Hi!"
    allowed_special = {"<|user|>", "<|assistant|>"}

    python_ids = python_tokenizer.encode(text, allowed_special)
    cpp_ids = cpp_tokenizer.encode(text, allowed_special)

    assert cpp_ids == python_ids
    assert cpp_tokenizer.decode_bytes(cpp_ids) == python_tokenizer.decode_bytes(python_ids)
    assert cpp_tokenizer.decode_bytes(np.asarray(cpp_ids, dtype="<u4")) == python_tokenizer.decode_bytes(python_ids)
    assert CppBPETokenizer.from_dict(python_state).encode(text, allowed_special) == python_ids
