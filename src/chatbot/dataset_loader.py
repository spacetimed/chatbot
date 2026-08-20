import json
from pathlib import Path
from datasets import load_dataset


def load_documents(
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    dataset_revision: str,
    cache_path: Path,
    max_bytes: int,
) -> list[str]:

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    if cache_path.exists():
        print(f"loading cached dataset: {cache_path}")
        return [json.loads(line)["text"] for line in cache_path.read_text(encoding="utf-8").splitlines()]

    print(f"streaming up to {max_bytes:,} bytes from {dataset_name}...")

    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split=dataset_split,
        revision=dataset_revision,
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
        raise ValueError("dataset stream returned no documents")

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with cache_path.open("w", encoding="utf-8") as cache_file:
        for document in documents:
            cache_file.write(json.dumps({"text": document}, ensure_ascii=False) + "\n")

    return documents
