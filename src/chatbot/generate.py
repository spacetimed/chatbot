from pathlib import Path

import torch

from chatbot.config import GPTConfig
from chatbot.model import GPT
from chatbot.tokenizer import BPETokenizer


checkpoint_path = Path("checkpoints/gplato.pt")
max_new_tokens = 1_000
temperature = 0.8
prompt = ""
stream = True
refresh_every = 4


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_text_codec(checkpoint):
    if "tokenizer" in checkpoint:
        tokenizer = BPETokenizer.from_dict(checkpoint["tokenizer"])
        return tokenizer.encode, tokenizer.decode

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    def encode(text: str):
        return [stoi[ch] for ch in text]

    def decode(ids):
        return "".join(itos[int(i)] for i in ids)

    return encode, decode


def clear_screen():
    print("\033[H\033[J", end="")


@torch.no_grad()
def generate_stream(model, context, max_new_tokens, temperature, decode, header):
    model.eval()
    idx = context

    for step in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        if step % refresh_every == 0 or step == max_new_tokens - 1:
            clear_screen()
            print(header)
            print(decode(idx[0].tolist()))
            print(f"\n[{step + 1}/{max_new_tokens} tokens]", flush=True)

    return idx


def main():
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model_config = GPTConfig(**config)
    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    encode, decode = load_text_codec(checkpoint)

    if prompt:
        context_ids = encode(prompt)
    else:
        context_ids = encode("\n")

    context = torch.tensor([context_ids], dtype=torch.long, device=device)

    header = (
        f"device: {device}\n"
        f"loaded checkpoint: {checkpoint_path}\n"
        f"checkpoint step: {checkpoint.get('step')}\n"
        "\n--- sample ---"
    )

    if stream:
        generate_stream(
            model,
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            decode=decode,
            header=header,
        )
    else:
        generated = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        print(header)
        print(decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
