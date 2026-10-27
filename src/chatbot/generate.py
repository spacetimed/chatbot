import argparse
from pathlib import Path

import torch

from chatbot.config import GPTConfig
from chatbot.model import Transformer
from chatbot.tokenizer import BPETokenizer

# generate.py rewrite

# target usage:
#   python -m chatbot.generate \
#       --checkpoint checkpoints/best.pt \
#       --prompt "Machine learning is" \
#       --max-new-tokens 200 \
#       --temperature 0.8 \
#       --top-k 50 \
#       --seed 1337

# today's scope:
#  1. load input checkpoint
#  2. reconstruct GPTConfig/weighs/tokenizer
#  3. prompt functionality
#  4. temperature/top-k/token count/seed/device options
#  5. generate/decode a completion
#  6. confirm seed produces deterministic output


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="load model weights from checkpoint and generate text from a prompt")

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=False, default=1.0)

    return parser.parse_args()
    # soon: temperature, top-k


def load_checkpoint(checkpoint_path: Path, device: str) -> tuple[Transformer, BPETokenizer]:

    # checkpoint -> model, tokenizer

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = GPTConfig(**checkpoint["config"])
    model = Transformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = BPETokenizer.from_dict(checkpoint["tokenizer"])

    return model, tokenizer

@torch.no_grad()
def generate(model: Transformer, idx, max_new_tokens, temperature=1.0) -> torch.Tensor:

    if temperature <= 0: raise ValueError("temperature must be above 0")
    if idx.size(1) + max_new_tokens > model.block_size: raise ValueError("request exceeds context limit")

    was_training = model.training
    model.eval()

    # todo special token for stopping then break loop here

    # data flow (to jog my memory)
    # idx: begins as [ [L] ] (so 1xL) (flattened token stream)

    # enable kv caches
    model.enable_caches()

    for i in range(max_new_tokens):
        # let T = min(L, block_size:=256)

        idx_cond = idx[:, -model.block_size:] if i==0 else idx[:,-1:] # [1,T] if first pass, [1,1] else with kv cache

        logits, _ = model(idx_cond)  # [1,T,V]
        # extract final token T-1's prediction. note -1 "selects" last T's V basically
        logits = logits[:, -1, :]  # [1,V]
        logits = logits / temperature  # temperature scaling (logit/(1/t) = logit*t)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

    if was_training: model.train()

    return idx

def main():

    # parse arguments
    args = parse_args()

    # get device, seed
    device = get_device()
    torch.manual_seed(args.seed)

    # load checkpoint, ensure exists
    model: Transformer
    tokenizer: BPETokenizer
    model, tokenizer = load_checkpoint(args.checkpoint, device)

    prompt_tokens = tokenizer.encode(args.prompt)
    if not prompt_tokens: raise ValueError("prompt must produce at least one token")

    context = torch.tensor(
        [prompt_tokens],
        dtype=torch.long,
        device=device,
    )

    generated = generate(model, context, max_new_tokens=args.tokens, temperature=args.temperature)
    output_tokens = generated[0].tolist()
    output_text = tokenizer.decode(output_tokens)

    print(f"device: {device}")
    print(output_text)

if __name__ == "__main__":
    main()
