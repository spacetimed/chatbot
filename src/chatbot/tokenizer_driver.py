import argparse
import json
from datetime import datetime
from pathlib import Path
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
BENCHMARK_RESULTS = TOKENIZER_ARTIFACTS / "benchmarks/results.jsonl"


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
    BENCHMARK_RESULTS.parent.mkdir(parents=True, exist_ok=True)


def artifact_path(
    language: str,
    filename: str,
) -> Path:
    return TOKENIZER_ARTIFACTS / language / filename


def log_benchmark(
    result: dict,
) -> None:
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


def require_implementation(
    language: str,
) -> None:

    if language != "python":
        raise SystemExit(f"the {language} tokenizer is not implemented yet")


def run_train(
    args: argparse.Namespace,
) -> None:

    require_implementation(args.language)

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
    input_bytes = len(text.encode("utf-8"))

    tokenizer = BPETokenizer(
        args.vocab_size,
        create_special_tokens(args.vocab_size),
    )

    start_time = perf_counter()
    tokenizer.train(text)
    elapsed = perf_counter() - start_time

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
        throughput = input_bytes / elapsed
        print(f"training time: {elapsed:.4f}s")
        print(f"training throughput: {throughput:,.0f} bytes/s")
        log_benchmark(
            {
                "operation": "train",
                "language": args.language,
                "dataset": FINEWEB_DATASET,
                "dataset_config": FINEWEB_CONFIG,
                "dataset_revision": FINEWEB_REVISION,
                "dataset_cache": str(dataset_cache),
                "documents": len(documents),
                "input_bytes": input_bytes,
                "mergeable_vocab_size": tokenizer.mergeable_vocab_size,
                "vocab_size": tokenizer.vocab_size,
                "learned_merges": len(tokenizer.merges),
                "elapsed_seconds": elapsed,
                "throughput_bytes_per_second": throughput,
                "output": str(output_path),
            }
        )


def run_encode(
    args: argparse.Namespace,
) -> None:

    require_implementation(args.language)

    rules_path = args.rules or artifact_path(args.language, "rules.json")
    output_path = args.output or artifact_path(args.language, "tokens.bin")
    tokenizer = BPETokenizer.from_dict(TokenizerIO.load_rules(rules_path))
    text = args.input.read_text(encoding="utf-8")

    start_time = perf_counter()
    token_ids = tokenizer.encode(
        text,
        allowed_special=set(args.allow_special),
    )
    elapsed = perf_counter() - start_time

    TokenizerIO.save_tokens(output_path, token_ids)

    print(f"saved tokens: {output_path}")
    print(f"tokens: {len(token_ids):,}")

    if args.benchmark:
        input_bytes = len(text.encode("utf-8"))
        throughput = input_bytes / elapsed
        print(f"encoding time: {elapsed:.4f}s")
        print(f"encoding throughput: {throughput:,.0f} bytes/s")
        log_benchmark(
            {
                "operation": "encode",
                "language": args.language,
                "input": str(args.input),
                "input_bytes": input_bytes,
                "tokens": len(token_ids),
                "rules": str(rules_path),
                "elapsed_seconds": elapsed,
                "throughput_bytes_per_second": throughput,
                "output": str(output_path),
            }
        )


def run_decode(
    args: argparse.Namespace,
) -> None:

    require_implementation(args.language)

    rules_path = args.rules or artifact_path(args.language, "rules.json")
    output_path = args.output or artifact_path(args.language, "decoded.txt")
    tokenizer = BPETokenizer.from_dict(TokenizerIO.load_rules(rules_path))
    token_ids = TokenizerIO.load_tokens(args.input)

    start_time = perf_counter()
    decoded = tokenizer.decode_bytes(token_ids)
    elapsed = perf_counter() - start_time

    TokenizerIO.save_decoded(output_path, decoded)

    print(f"saved text: {output_path}")
    print(f"decoded bytes: {len(decoded):,}")

    if args.benchmark:
        throughput = len(decoded) / elapsed
        print(f"decoding time: {elapsed:.4f}s")
        print(f"decoding throughput: {throughput:,.0f} bytes/s")
        log_benchmark(
            {
                "operation": "decode",
                "language": args.language,
                "input": str(args.input),
                "tokens": len(token_ids),
                "decoded_bytes": len(decoded),
                "rules": str(rules_path),
                "elapsed_seconds": elapsed,
                "throughput_bytes_per_second": throughput,
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
    create_artifact_directories()
    args.run(args)


if __name__ == "__main__":
    main()
