from pathlib import Path

import torch

from chatbot.config import GPTConfig, TrainConfig
from chatbot.model import GPT
from chatbot.tokenizer import BPETokenizer


# TokenBatchLoader will take a long token stream, select chunks containing BxT+1 tokens, and reshape them into B sequences of length T
# labels are made from shifting the same tokens by 1 position
class TokenBatchLoader:
    def __init__(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        block_size: int,
    ) -> None:

        self.tokens = tokens
        self.batch_size = batch_size
        self.block_size = block_size

        self.tokens_per_batch = batch_size * block_size

        if tokens.ndim != 1:
            raise ValueError("tokens must be a one-dimensional tensor")

        minimum_tokens = self.tokens_per_batch + 1
        if len(tokens) < minimum_tokens:
            raise ValueError(f"batch loader requires at least {minimum_tokens} tokens, but received {len(tokens)}")

        self.reset()

    def reset(self) -> None:
        self.current_position = 0

    def next_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # sequentially returns two [B,T] tensors from dataset (input token IDs, target token IDs)

        batch_end = self.current_position + self.tokens_per_batch + 1

        # wrap back around dataset
        if batch_end > len(self.tokens):
            self.reset()
            batch_end = self.tokens_per_batch + 1

        buffer = self.tokens[self.current_position : batch_end]

        x = buffer[:-1].view(self.batch_size, self.block_size)
        y = buffer[1:].view(self.batch_size, self.block_size)

        self.current_position += self.tokens_per_batch

        return x, y


# cuda > mps > cpu (priority order)
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def prepare_data(
    config: TrainConfig,
) -> tuple[BPETokenizer, torch.Tensor, torch.Tensor]:
    # returns three objects:
    #   tokenizer:    trained BPETokenizer
    #   train_tokens: 1D CPU tensor containing training token IDs
    #   val_tokens:   1D CPU tensor containing the validation token IDs

    if not (0.0 < config.train_split < 1.0):
        raise ValueError("train_split must be within range (0,1)")

    text = config.dataset_path.read_text(encoding="utf-8")

    if not text:
        raise ValueError(f"dataset is empty: {config.dataset_path}")

    split_index = int(len(text) * config.train_split)  # index to split training/validation data

    train_text = text[:split_index]
    val_text = text[split_index:]

    tokenizer = BPETokenizer(config.tokenizer_vocab_size)
    tokenizer.train(train_text)

    train_token_ids = tokenizer.encode(train_text)
    val_token_ids = tokenizer.encode(val_text)

    train_tokens = torch.tensor(
        train_token_ids,
        dtype=torch.long,
    )
    val_tokens = torch.tensor(
        val_token_ids,
        dtype=torch.long,
    )

    return tokenizer, train_tokens, val_tokens


def main() -> None:
    train_config = TrainConfig()

    torch.manual_seed(train_config.seed)
    device = get_device()

    tokenizer, train_tokens, val_tokens = prepare_data(train_config)

    # model architecture
    model_config = GPTConfig(
        vocab_size=len(tokenizer.vocab),
        block_size=64,
        n_embed=64,
        n_head=4,
        n_layer=2,
        dropout=0.15,
    )

    # training data
    train_loader = TokenBatchLoader(
        train_tokens,
        train_config.batch_size,
        model_config.block_size,
    )

    # validation data
    val_loader = TokenBatchLoader(
        val_tokens,
        train_config.batch_size,
        model_config.block_size,
    )

    # for training loss measurements during eval
    train_eval_loader = TokenBatchLoader(
        train_tokens,
        train_config.batch_size,
        model_config.block_size,
    )

    model = GPT(model_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
    )

    # information/specs
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device: {device}")
    print(f"training tokens: {len(train_tokens)}")
    print(f"validation tokens: {len(val_tokens)}")
    print(f"vocab size: {len(tokenizer.vocab)}")
    print(f"parameters: {parameter_count:,}")

    # training loop
    model.train()
    for step in range(1, train_config.max_steps + 1):
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)

        _, loss = model(x, y)

        loss.backward()
        optimizer.step()

        print(f"step {step:4d} | train loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
