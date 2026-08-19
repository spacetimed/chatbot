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
    dataset_path: Path = Path("datasets/plato.txt")

    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_save_latest: bool = True
    checkpoint_save_best: bool = True

    seed: int = 1337
    batch_size: int = 16
    max_steps: int = 100
    learning_rate: float = 3e-4

    train_split: float = 0.9
    eval_interval: int = 25
    eval_batches: int = 10

    tokenizer_vocab_size: int = 256
