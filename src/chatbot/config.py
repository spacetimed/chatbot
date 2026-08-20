from dataclasses import dataclass
from pathlib import Path


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_embed: int
    n_head: int
    n_layer: int
    dropout: float = 0.0


@dataclass
class TrainConfig:
    # support for hugging face datasets (instead of plato)
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    dataset_split: str = "train"
    dataset_revision: str = "v1.0.0"
    dataset_bytes: int = 1_000_000
    dataset_cache: Path = Path("datasets/fineweb_edu_1000000_bytes.jsonl")

    tokenizer_path: Path = Path("artifacts/tokenizer/python/rules.json")

    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_save_latest: bool = True
    checkpoint_save_best: bool = True

    seed: int = 1337
    batch_size: int = 16

    max_steps: int = 2_000
    warmup_steps: int = 100

    log_interval: int = 100
    eval_interval: int = 250
    eval_batches: int = 20

    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    max_grad_norm: float = 1.0

    train_split: float = 0.9

    # GPT-style AdamW configuration
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
