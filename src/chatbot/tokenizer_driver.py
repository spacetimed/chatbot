import argparse
import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from chatbot.dataset_loader import load_documents
from chatbot.tokenizer import BPETokenizer

SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|endofturn|>",
)

FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
FINEWEB_REVISION = "v1.0.0"

TOKENIZER_ARTIFACTS = Path("artifacts/tokenizer")
TOKENIZER_LANGUAGES = ("python", "cpp", "rust")
BENCHMARK_RESULTS = Path("benchmarks/benchmarks.jsonl")


class TokenizerIO:
    @staticmethod
    def save_rules(path: Path, state: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @staticmethod
    def load_rules(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def save_tokens(path: Path, token_ids: list[int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.asarray(token_ids, dtype="<u4").tofile(path)

    @staticmethod
    def load_tokens(path: Path) -> list[int]:
        if path.stat().st_size % 4 != 0:
            raise ValueError("tokens.bin size must be divisible by four bytes")
        return np.fromfile(path, dtype="<u4").tolist()

    @staticmethod
    def save_decoded(path: Path, decoded: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(decoded)


def create_artifact_directories() -> None:
    # create artifacts/{python,cpp,rust} if they do not exist
    for language in TOKENIZER_LANGUAGES:
        (TOKENIZER_ARTIFACTS / language).mkdir(parents=True, exist_ok=True)


def artifact_path(
    language: str,
    filename: str,
) -> Path:
    return TOKENIZER_ARTIFACTS / language / filename


def log_benchmark(
    result: dict,
) -> None:
    BENCHMARK_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        **result,
    }
    with BENCHMARK_RESULTS.open("a", encoding="utf-8") as benchmark_file:
        benchmark_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"logged benchmark: {BENCHMARK_RESULTS}")


def create_special_tokens(
    mergeable_vocab_size: int,
) -> dict[str, int]:

    return {token: mergeable_vocab_size + index for index, token in enumerate(SPECIAL_TOKENS)}


def get_tokenizer_implementation(
    language: str,
) -> type:

    if language == "python":
        return BPETokenizer

    if language == "cpp":
        from chatbot._tokenizer_cpp import BPETokenizer as CppBPETokenizer

        return CppBPETokenizer

    raise SystemExit(f"the {language} tokenizer is not implemented yet")


def run_train(
    args: argparse.Namespace,
) -> None:

    Tokenizer = get_tokenizer_implementation(args.language)

    output_path = args.output or artifact_path(args.language, "rules.json")
    dataset_cache = args.dataset_cache or Path(f"datasets/fineweb_edu_{args.dataset_bytes}_bytes.jsonl")
    documents = load_documents(
        dataset_name=FINEWEB_DATASET,
        dataset_config=FINEWEB_CONFIG,
        dataset_split="train",
        dataset_revision=FINEWEB_REVISION,
        cache_path=dataset_cache,
        max_bytes=args.dataset_bytes,
    )
    text = "\n\n".join(documents)
    corpus = text.encode("utf-8")
    input_bytes = len(corpus)

    repetitions = args.repeat if args.benchmark else 1
    elapsed_seconds = []

    for _ in range(repetitions):
        tokenizer = Tokenizer(
            args.vocab_size,
            create_special_tokens(args.vocab_size),
        )

        start_time = perf_counter()
        tokenizer.train(text)
        elapsed_seconds.append(perf_counter() - start_time)

    TokenizerIO.save_rules(output_path, tokenizer.to_dict())

    print(f"dataset: {FINEWEB_DATASET}/{FINEWEB_CONFIG}")
    print(f"dataset cache: {dataset_cache}")
    print(f"documents: {len(documents):,}")
    print(f"corpus bytes: {input_bytes:,}")
    print(f"saved rules: {output_path}")
    print(f"mergeable vocabulary size: {tokenizer.mergeable_vocab_size:,}")
    print(f"learned merges: {len(tokenizer.merges):,}")
    print(f"total vocabulary size: {tokenizer.vocab_size:,}")

    if args.benchmark:
        median_elapsed = median(elapsed_seconds)
        throughput = input_bytes / median_elapsed
        print(f"training median: {median_elapsed:.4f}s across {args.repeat} repetitions")
        print(f"training throughput: {throughput:,.0f} bytes/s")
        log_benchmark(
            {
                "label": args.label,
                "operation": "train",
                "language": args.language,
                "dataset": FINEWEB_DATASET,
                "dataset_config": FINEWEB_CONFIG,
                "dataset_revision": FINEWEB_REVISION,
                "dataset_cache": str(dataset_cache),
                "documents": len(documents),
                "input_bytes": input_bytes,
                "input_sha256": sha256(corpus).hexdigest(),
                "mergeable_vocab_size": tokenizer.mergeable_vocab_size,
                "vocab_size": tokenizer.vocab_size,
                "learned_merges": len(tokenizer.merges),
                "repeat": args.repeat,
                "elapsed_seconds": elapsed_seconds,
                "minimum_seconds": min(elapsed_seconds),
                "median_seconds": median_elapsed,
                "maximum_seconds": max(elapsed_seconds),
                "median_throughput_bytes_per_second": throughput,
                "output": str(output_path),
            }
        )


def run_encode(
    args: argparse.Namespace,
) -> None:

    Tokenizer = get_tokenizer_implementation(args.language)

    rules_path = args.rules or artifact_path(args.language, "rules.json")
    output_path = args.output or artifact_path(args.language, "tokens.bin")
    tokenizer = Tokenizer.from_dict(TokenizerIO.load_rules(rules_path))
    text = args.input.read_text(encoding="utf-8")
    encoded_text = text.encode("utf-8")
    allowed_special = set(args.allow_special)
    repetitions = args.repeat if args.benchmark else 1
    elapsed_seconds = []

    if args.benchmark:
        tokenizer.encode(text, allowed_special=allowed_special)

    for _ in range(repetitions):
        start_time = perf_counter()
        token_ids = tokenizer.encode(
            text,
            allowed_special=allowed_special,
        )
        elapsed_seconds.append(perf_counter() - start_time)

    TokenizerIO.save_tokens(output_path, token_ids)

    print(f"saved tokens: {output_path}")
    print(f"tokens: {len(token_ids):,}")

    if args.benchmark:
        input_bytes = len(encoded_text)
        median_elapsed = median(elapsed_seconds)
        throughput = input_bytes / median_elapsed
        print(f"encoding median: {median_elapsed:.4f}s across {args.repeat} repetitions")
        print(f"encoding throughput: {throughput:,.0f} bytes/s")
        log_benchmark(
            {
                "label": args.label,
                "operation": "encode",
                "language": args.language,
                "input": str(args.input),
                "input_bytes": input_bytes,
                "input_sha256": sha256(encoded_text).hexdigest(),
                "tokens": len(token_ids),
                "rules": str(rules_path),
                "repeat": args.repeat,
                "elapsed_seconds": elapsed_seconds,
                "minimum_seconds": min(elapsed_seconds),
                "median_seconds": median_elapsed,
                "maximum_seconds": max(elapsed_seconds),
                "median_throughput_bytes_per_second": throughput,
                "output": str(output_path),
            }
        )


def run_decode(
    args: argparse.Namespace,
) -> None:

    Tokenizer = get_tokenizer_implementation(args.language)

    rules_path = args.rules or artifact_path(args.language, "rules.json")
    output_path = args.output or artifact_path(args.language, "decoded.txt")
    tokenizer = Tokenizer.from_dict(TokenizerIO.load_rules(rules_path))
    token_ids = TokenizerIO.load_tokens(args.input)
    encoded_tokens = args.input.read_bytes()
    repetitions = args.repeat if args.benchmark else 1
    elapsed_seconds = []

    if args.benchmark:
        tokenizer.decode_bytes(token_ids)

    for _ in range(repetitions):
        start_time = perf_counter()
        decoded = tokenizer.decode_bytes(token_ids)
        elapsed_seconds.append(perf_counter() - start_time)

    TokenizerIO.save_decoded(output_path, decoded)

    print(f"saved text: {output_path}")
    print(f"decoded bytes: {len(decoded):,}")

    if args.benchmark:
        median_elapsed = median(elapsed_seconds)
        throughput = len(decoded) / median_elapsed
        print(f"decoding median: {median_elapsed:.4f}s across {args.repeat} repetitions")
        print(f"decoding throughput: {throughput:,.0f} bytes/s")
        log_benchmark(
            {
                "label": args.label,
                "operation": "decode",
                "language": args.language,
                "input": str(args.input),
                "input_sha256": sha256(encoded_tokens).hexdigest(),
                "tokens": len(token_ids),
                "decoded_bytes": len(decoded),
                "decoded_sha256": sha256(decoded).hexdigest(),
                "rules": str(rules_path),
                "repeat": args.repeat,
                "elapsed_seconds": elapsed_seconds,
                "minimum_seconds": min(elapsed_seconds),
                "median_seconds": median_elapsed,
                "maximum_seconds": max(elapsed_seconds),
                "median_throughput_bytes_per_second": throughput,
                "output": str(output_path),
            }
        )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select between three tokenizer implementations. Train, run, and benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    common_parser = argparse.ArgumentParser(add_help=False)

    common_parser.add_argument(
        "--language",
        choices=TOKENIZER_LANGUAGES,
        required=True,
    )
    common_parser.add_argument(
        "--benchmark",
        action="store_true",
    )
    common_parser.add_argument(
        "--repeat",
        type=int,
    )
    common_parser.add_argument(
        "--label",
    )
    train_parser = subparsers.add_parser(
        "train",
        parents=[common_parser],
        help="train a vocabulary and write rules.json",
    )
    train_parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="mergeable vocabulary size",
    )
    train_parser.add_argument(
        "--dataset-bytes",
        type=int,
        default=5_000_000,
        help="maximum FineWeb-Edu text bytes to stream when creating the cache",
    )
    train_parser.add_argument(
        "--dataset-cache",
        type=Path,
        help="existing or new JSONL cache path (default includes dataset byte limit)",
    )
    train_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="rules.json output path",
    )
    train_parser.set_defaults(run=run_train)

    encode_parser = subparsers.add_parser(
        "encode",
        parents=[common_parser],
        help="encode text and write tokens.bin",
    )
    encode_parser.add_argument("input", type=Path)
    encode_parser.add_argument("--rules", type=Path, default=None)
    encode_parser.add_argument("--output", type=Path, default=None)
    encode_parser.add_argument("--allow-special", action="append", default=[])
    encode_parser.set_defaults(run=run_encode)

    decode_parser = subparsers.add_parser(
        "decode",
        parents=[common_parser],
        help="decode tokens.bin and write decoded.txt",
    )
    decode_parser.add_argument("input", type=Path)
    decode_parser.add_argument("--rules", type=Path, default=None)
    decode_parser.add_argument("--output", type=Path, default=None)
    decode_parser.set_defaults(run=run_decode)

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    if args.benchmark and args.repeat is None:
        parser.error("--repeat is required with --benchmark")
    if args.benchmark and args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.benchmark and args.label is None:
        parser.error("--label is required with --benchmark")
    if not args.benchmark and (args.repeat is not None or args.label is not None):
        parser.error("--repeat and --label require --benchmark")

    create_artifact_directories()
    args.run(args)


if __name__ == "__main__":
    main()
