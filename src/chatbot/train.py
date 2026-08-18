from pathlib import Path
from datetime import datetime
from time import perf_counter

import torch

from chatbot.config import GPTConfig
from chatbot.model import GPT
from chatbot.tokenizer import BPETokenizer


# config
data_path = Path("datasets/plato.txt")
checkpoint_path = Path("checkpoints/gplato.pt")
log_path = Path("logs/gplato_train.log")
seed = 1337

# hyperparameters -----
batch_size = 16
block_size = 64
tokenizer_vocab_size = 256
n_embed = 64
num_heads = 4
n_layers = 2

lr = 3e-4

dropout = 0.15

max_iters = 100
eval_interval = 25
eval_iters = 10

gen_tokens = 100

temperature = 0.8  # soften sampling
# ---------------------


def format_duration(seconds: float):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_text(path: Path):
    return path.read_text(encoding="utf-8")


def build_tokenizer(text: str):
    tokenizer = BPETokenizer(tokenizer_vocab_size)
    tokenizer.train(text, verbose=True)
    return tokenizer


def split_data(data: torch.Tensor):
    n = int(0.9 * len(data))
    return data[:n], data[n:]


def get_batch(split, train_data, val_data, block_size, batch_size, device):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,), device=device)
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters):
    out = {}
    was_training = model.training
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split, train_data, val_data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    if was_training:
        model.train()

    return out


def save_checkpoint(path, model, optimizer, tokenizer, step, losses, best_val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "vocab_size": model.vocab_size,
            "block_size": model.block_size,
            "n_embed": model.n_embed,
            "num_heads": model.num_heads,
            "n_layers": model.n_layer,
            "dropout": model.dropout,
        },
        "tokenizer": tokenizer.to_dict(),
        "step": step,
        "losses": losses,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, path)


def log_run(path, elapsed_seconds, device, n_params, vocab_size, losses):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(f"run: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"device: {device}\n")
        f.write(f"parameters: {n_params}\n")
        f.write(f"vocab_size: {vocab_size}\n")
        f.write(
            "hyperparameters: "
            f"batch_size={batch_size}, "
            f"block_size={block_size}, "
            f"tokenizer_vocab_size={tokenizer_vocab_size}, "
            f"n_embed={n_embed}, "
            f"num_heads={num_heads}, "
            f"n_layers={n_layers}, "
            f"dropout={dropout}, "
            f"lr={lr}, "
            f"max_iters={max_iters}, "
            f"eval_interval={eval_interval}, "
            f"eval_iters={eval_iters}\n"
        )
        f.write(
            "losses: "
            f"train={losses['train']:.4f}, "
            f"val={losses['val']:.4f}\n"
        )
        f.write(f"elapsed: {format_duration(elapsed_seconds)} ({elapsed_seconds:.2f}s)\n")
        f.write(f"checkpoint: {checkpoint_path}\n\n")


def main():
    torch.manual_seed(seed)
    device = get_device()

    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        print("enabled cuda tf32 matmul")

    text = load_text(data_path)
    tokenizer = build_tokenizer(text)
    token_ids = tokenizer.encode(text)
    vocab_size = len(tokenizer.vocab)

    data = torch.tensor(token_ids, dtype=torch.long, device=device)
    train_data, val_data = split_data(data)

    model_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embed=n_embed,
        n_head=num_heads,
        n_layer=n_layers,
        dropout=dropout,
    )
    model = GPT(model_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=(device == "cuda"))

    n_params = sum(p.numel() for p in model.parameters())

    print(f"device: {device}")
    print(f"characters in dataset: {len(text)}")
    print(f"tokens in dataset: {len(token_ids)}")
    print(f"vocab size: {vocab_size}")
    print(f"parameters: {n_params:,}")

    last_losses = None
    best_step = None
    best_losses = None
    best_val_loss = float("inf")
    train_start = perf_counter()
    last_eval_time = train_start
    for step in range(max_iters + 1):
        if step % eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                block_size,
                batch_size,
                device,
                eval_iters,
            )
            last_losses = losses
            saved_best = False
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_step = step
                best_losses = losses
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    tokenizer,
                    step,
                    losses,
                    best_val_loss,
                )
                saved_best = True

            now = perf_counter()
            total_elapsed = now - train_start
            eval_delta = now - last_eval_time
            last_eval_time = now
            print(
                f"step {step:4d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"elapsed {format_duration(total_elapsed)} | "
                f"+{format_duration(eval_delta)}"
                f"{' | saved best' if saved_best else ''}"
            )

        xb, yb = get_batch("train", train_data, val_data, block_size, batch_size, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    elapsed_seconds = perf_counter() - train_start

    if last_losses is None:
        last_losses = estimate_loss(
            model,
            train_data,
            val_data,
            block_size,
            batch_size,
            device,
            eval_iters,
        )

    if best_losses is None:
        best_losses = last_losses
        best_step = max_iters
        best_val_loss = last_losses["val"]
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            tokenizer,
            best_step,
            best_losses,
            best_val_loss,
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"\nsaved best checkpoint: {checkpoint_path}")
    print(f"best step: {best_step} | best val loss: {best_val_loss:.4f}")
    print(f"time taken to train: {format_duration(elapsed_seconds)} ({elapsed_seconds:.2f}s)")

    log_run(log_path, elapsed_seconds, device, n_params, vocab_size, best_losses)
    print(f"logged run: {log_path}")

    context_ids = tokenizer.encode("\n")
    context = torch.tensor([context_ids], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=gen_tokens, temperature=temperature)
    print("\n--- sample ---")
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
