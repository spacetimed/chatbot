import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from chatbot.config import GPTConfig, TrainConfig
from chatbot.dataset_loader import load_documents
from chatbot.model import GPT
from chatbot.tokenizer import BPETokenizer
from chatbot.tokenizer_driver import TokenizerIO


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


# wait for asynchronous accelerator work to finish before measuring time
def synchronize_device(
    device: torch.device,
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def get_learning_rate(
    step: int,
    config: TrainConfig,
) -> float:

    # get learning rate relative to current training step
    # gradually increases then decreases

    if not (0 <= config.warmup_steps < config.max_steps):
        raise ValueError("warmup_steps must be non-negative and less than max_steps")

    if not (0.0 <= config.min_learning_rate <= config.learning_rate):
        raise ValueError("min_learning_rate must be between zero and learning_rate")

    # linearly increase from small learning rate to maximum learning rate
    if (config.warmup_steps > 0) and (step <= config.warmup_steps):
        return config.learning_rate * step / config.warmup_steps

    # decay from maximum learning rate to minimum learning rate
    decay_start = max(config.warmup_steps, 1)

    if step >= config.max_steps:
        return config.min_learning_rate

    decay_ratio = (step - decay_start) / (config.max_steps - decay_start)
    cosine_coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return config.min_learning_rate + cosine_coefficient * (config.learning_rate - config.min_learning_rate)


def prepare_data(
    config: TrainConfig,
) -> tuple[BPETokenizer, torch.Tensor, torch.Tensor]:
    # returns three objects:
    #   tokenizer:    loaded BPETokenizer
    #   train_tokens: 1D CPU tensor containing training token IDs
    #   val_tokens:   1D CPU tensor containing the validation token IDs

    if not (0.0 < config.train_split < 1.0):
        raise ValueError("train_split must be within range (0,1)")

    documents = load_documents(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        dataset_split=config.dataset_split,
        dataset_revision=config.dataset_revision,
        cache_path=config.dataset_cache,
        max_bytes=config.dataset_bytes,
    )

    split_index = int(len(documents) * config.train_split)

    if split_index == 0 or split_index == len(documents):
        raise ValueError("dataset must contain enough documents for training and validation splits")

    train_documents = documents[:split_index]
    val_documents = documents[split_index:]

    document_separator = "<|endoftext|>"
    train_text = document_separator.join(train_documents)
    val_text = document_separator.join(val_documents)

    tokenizer = BPETokenizer.from_dict(TokenizerIO.load_rules(config.tokenizer_path))
    allowed_special = {document_separator}

    train_token_ids = tokenizer.encode(train_text, allowed_special=allowed_special)
    val_token_ids = tokenizer.encode(val_text, allowed_special=allowed_special)

    train_tokens = torch.tensor(
        train_token_ids,
        dtype=torch.long,
    )
    val_tokens = torch.tensor(
        val_token_ids,
        dtype=torch.long,
    )

    return tokenizer, train_tokens, val_tokens


@torch.no_grad()
def evaluate_loss(
    model: GPT,
    loader: TokenBatchLoader,
    device: torch.device,
    eval_batches: int,
) -> float:
    # used mid-training to evaluate current model with several [B,T] batches

    if eval_batches <= 0:
        raise ValueError("eval_batches must be positive")

    was_training = model.training
    model.eval()
    loader.reset()

    total_loss = 0.0

    for _ in range(eval_batches):
        x, y = loader.next_batch()
        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)
        total_loss += loss.item()

    if was_training:
        model.train()

    return total_loss / eval_batches


def evaluate_and_report(
    model: GPT,
    train_loader: TokenBatchLoader,
    val_loader: TokenBatchLoader,
    device: torch.device,
    eval_batches: int,
    step: int,
) -> dict[str, float]:
    # prints training information at current step; returns dict (for checkpoint storage)
    losses = {
        "train": evaluate_loss(model, train_loader, device, eval_batches),
        "val": evaluate_loss(model, val_loader, device, eval_batches),
    }

    print(f"eval  | step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    return losses


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    tokenizer: BPETokenizer,
    train_config: TrainConfig,
    step: int,
    losses: dict[str, float],
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    train_config_state = asdict(train_config)
    train_config_state["dataset_cache"] = str(train_config.dataset_cache)
    train_config_state["tokenizer_path"] = str(train_config.tokenizer_path)
    train_config_state["checkpoint_dir"] = str(train_config.checkpoint_dir)

    checkpoint = {
        "checkpoint_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(model.config),
        "train_config": train_config_state,
        "tokenizer": tokenizer.to_dict(),
        "step": step,
        "losses": losses,
        "best_val_loss": best_val_loss,
    }

    torch.save(checkpoint, path)


def save_evaluation_checkpoints(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    tokenizer: BPETokenizer,
    train_config: TrainConfig,
    step: int,
    losses: dict[str, float],
    best_val_loss: float,
) -> float:
    is_best = losses["val"] < best_val_loss

    if is_best:
        best_val_loss = losses["val"]

    if train_config.checkpoint_save_latest:
        save_checkpoint(
            train_config.checkpoint_dir / "latest.pt",
            model,
            optimizer,
            tokenizer,
            train_config,
            step,
            losses,
            best_val_loss,
        )

    if train_config.checkpoint_save_best and is_best:
        save_checkpoint(
            train_config.checkpoint_dir / "best.pt",
            model,
            optimizer,
            tokenizer,
            train_config,
            step,
            losses,
            best_val_loss,
        )

    return best_val_loss


def create_optimizer(
    model: GPT,
    config: TrainConfig,
) -> torch.optim.AdamW:

    # create optimizer with custom AdamW configuration

    # filter parameters with gradients
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    # matrix weights used by Linear and Embedding layers receive weight decay
    decay_parameters = [parameter for parameter in parameters if parameter.dim() >= 2]

    # biases and LayerNorm parameters do not receive weight decay
    no_decay_parameters = [parameter for parameter in parameters if parameter.dim() < 2]

    parameter_groups = [
        {
            "params": decay_parameters,
            "weight_decay": config.weight_decay,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]

    decay_parameter_count = sum(parameter.numel() for parameter in decay_parameters)
    no_decay_parameter_count = sum(parameter.numel() for parameter in no_decay_parameters)

    print(f"decayed parameters: {decay_parameter_count:,}")
    print(f"non-decayed parameters: {no_decay_parameter_count:,}")

    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=1e-8,
    )

    return optimizer


def main() -> None:
    train_config = TrainConfig()

    torch.manual_seed(train_config.seed)
    device = get_device()

    tokenizer, train_tokens, val_tokens = prepare_data(train_config)

    # model architecture
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
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

    optimizer = create_optimizer(
        model,
        train_config,
    )

    # information/specs
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device: {device}")
    print(f"training tokens: {len(train_tokens)}")
    print(f"validation tokens: {len(val_tokens)}")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"parameters: {parameter_count:,}")

    best_val_loss = float("inf")

    # measure step zero
    losses = evaluate_and_report(
        model,
        train_eval_loader,
        val_loader,
        device,
        train_config.eval_batches,
        step=0,
    )
    best_val_loss = save_evaluation_checkpoints(
        model,
        optimizer,
        tokenizer,
        train_config,
        step=0,
        losses=losses,
        best_val_loss=best_val_loss,
    )

    if train_config.log_interval <= 0:
        raise ValueError("log_interval must be positive")

    interval_loss = 0.0
    interval_seconds = 0.0
    interval_tokens = 0
    interval_steps = 0
    interval_grad_norm = 0.0

    # training loop
    model.train()
    for step in range(1, train_config.max_steps + 1):
        # synchronize before starting so previous accelerator work is not included
        synchronize_device(device)
        step_start = perf_counter()

        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)

        _, loss = model(x, y)

        loss.backward()
        # gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            train_config.max_grad_norm,
        )

        step_learning_rate = get_learning_rate(step, train_config)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = step_learning_rate

        optimizer.step()

        # synchronize again so the full training step finishes before stopping the timer
        synchronize_device(device)
        step_seconds = perf_counter() - step_start

        interval_loss += loss.item()
        interval_seconds += step_seconds
        interval_tokens += train_loader.tokens_per_batch
        interval_steps += 1
        interval_grad_norm += grad_norm.item()

        if step % train_config.log_interval == 0 or step == train_config.max_steps:
            average_loss = interval_loss / interval_steps
            average_grad_norm = interval_grad_norm / interval_steps
            average_step_ms = (interval_seconds / interval_steps) * 1_000
            tokens_per_second = interval_tokens / interval_seconds

            print(
                f"train | step {step:4d} | "
                f"loss {average_loss:.4f} | "
                f"lr {step_learning_rate:.2e} | "
                f"grad norm {average_grad_norm:.4f} | "
                f"{average_step_ms:.2f} ms/step | "
                f"{tokens_per_second:,.0f} tok/s"
            )

            interval_loss = 0.0
            interval_grad_norm = 0.0
            interval_seconds = 0.0
            interval_tokens = 0
            interval_steps = 0

        if step % train_config.eval_interval == 0 or step == train_config.max_steps:
            losses = evaluate_and_report(
                model,
                train_eval_loader,
                val_loader,
                device,
                train_config.eval_batches,
                step,
            )
            best_val_loss = save_evaluation_checkpoints(
                model,
                optimizer,
                tokenizer,
                train_config,
                step,
                losses,
                best_val_loss,
            )


if __name__ == "__main__":
    main()
