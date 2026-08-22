#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <cpp|python> <label>" >&2
    exit 1
fi

language="$1"
label="$2"

if [[ "$language" != "cpp" && "$language" != "python" ]]; then
    echo "language must be cpp or python" >&2
    exit 1
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="$project_root/.venv/bin/python"

cd "$project_root"

"$python" -m chatbot.tokenizer_driver train \
    --language "$language" \
    --vocab-size 1000 \
    --dataset-bytes 100000 \
    --benchmark \
    --repeat 3 \
    --label "$label"

"$python" -m chatbot.tokenizer_driver encode \
    --language "$language" \
    --benchmark \
    --repeat 5 \
    --label "$label" \
    datasets/fineweb_edu_1000000_bytes.txt

"$python" -m chatbot.tokenizer_driver decode \
    --language "$language" \
    --benchmark \
    --repeat 20 \
    --label "$label" \
    "artifacts/tokenizer/$language/tokens.bin"

if [[ "$language" == "cpp" ]]; then
    if ! cmp -s artifacts/tokenizer/cpp/tokens.bin artifacts/tokenizer/python/tokens.bin || \
       ! cmp -s artifacts/tokenizer/cpp/decoded.txt artifacts/tokenizer/python/decoded.txt; then
        echo "benchmark invalid! incorrect output" >&2
        exit 1
    fi

    echo "benchmark valid: C++ output matches Python"
fi
