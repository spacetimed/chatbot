import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from chatbot.tokenizer import BPETokenizer, read_tokens, write_tokens

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


def load_fineweb_corpus(
    cache_path: Path,
    max_bytes: int,
) -> tuple[str, int, int]:

    if max_bytes <= 0:
        raise ValueError("dataset-bytes must be positive")

    if cache_path.exists():
        print(f"loading cached FineWeb-Edu corpus: {cache_path}")
        documents = [json.loads(line)["text"] for line in cache_path.read_text(encoding="utf-8").splitlines()]
        text = "\n\n".join(documents)
        return text, len(documents), len(text.encode("utf-8"))

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("install project dependencies with: python -m pip install -e .") from error

    print(f"streaming up to {max_bytes:,} bytes from FineWeb-Edu...")

    dataset = load_dataset(
        FINEWEB_DATASET,
        name=FINEWEB_CONFIG,
        split="train",
        revision=FINEWEB_REVISION,
        streaming=True,
    )

    documents = []
    total_bytes = 0

    for example in dataset:
        document = example["text"]
        document_bytes = len(document.encode("utf-8"))

        if not document:
            continue

        if documents and total_bytes + document_bytes > max_bytes:
            break

        documents.append(document)
        total_bytes += document_bytes

    if not documents:
        raise ValueError("FineWeb-Edu stream returned no documents")

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with cache_path.open("w", encoding="utf-8") as cache_file:
        for document in documents:
            cache_file.write(json.dumps({"text": document}, ensure_ascii=False) + "\n")

    text = "\n\n".join(documents)
    return text, len(documents), len(text.encode("utf-8"))


def run_train(
    args: argparse.Namespace,
) -> None:

    require_implementation(args.language)

    output_path = args.output or artifact_path(args.language, "rules.json")
    dataset_cache = args.dataset_cache or Path(f"datasets/fineweb_edu_{args.dataset_bytes}_bytes.jsonl")
    text, document_count, input_bytes = load_fineweb_corpus(
        dataset_cache,
        args.dataset_bytes,
    )
    tokenizer = BPETokenizer(
        args.vocab_size,
        create_special_tokens(args.vocab_size),
    )

    start_time = perf_counter()
    tokenizer.train(text)
    elapsed = perf_counter() - start_time

    tokenizer.save(output_path)

    print(f"dataset: {FINEWEB_DATASET}/{FINEWEB_CONFIG}")
    print(f"dataset cache: {dataset_cache}")
    print(f"documents: {document_count:,}")
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
                "documents": document_count,
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
    tokenizer = BPETokenizer.load(rules_path)
    text = args.input.read_text(encoding="utf-8")

    start_time = perf_counter()
    token_ids = tokenizer.encode(
        text,
        allowed_special=set(args.allow_special),
    )
    elapsed = perf_counter() - start_time

    write_tokens(output_path, token_ids)

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
    tokenizer = BPETokenizer.load(rules_path)
    token_ids = read_tokens(args.input)

    start_time = perf_counter()
    decoded = tokenizer.decode_bytes(token_ids)
    elapsed = perf_counter() - start_time

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(decoded)

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
