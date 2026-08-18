import torch

from chatbot.config import GPTConfig
from chatbot.model import GPT

# basic low-compute tests to ensure the model is functional

def test_model():

    torch.manual_seed(1337)
    config = GPTConfig(
        vocab_size=64,
        block_size=16,
        n_embed=32,
        n_head=4,
        n_layer=2,
        dropout=0.0,
    )
    model = GPT(config)
    tokens = torch.randint(0, 64, (2, 9)) # 2x9 containing rand int [0,64)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    logits, loss = model(inputs, targets)

    assert logits.shape == (2, 8, 64)
    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()
    assert model.token_embedding_table.weight.grad is not None
    

def test_force_overfit():
    # Force the model to overfit one batch; ensure loss drops substantially overall.

    torch.manual_seed(1337)
    config = GPTConfig(
        vocab_size=64,
        block_size=16,
        n_embed=32,
        n_head=4,
        n_layer=2,
        dropout=0.0,
    )
    model = GPT(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    tokens = torch.randint(0, 64, (2, 9)) # 2x9 containing rand int [0,64)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    _, initial_loss = model(inputs, targets)
    initial_loss_value = initial_loss.item()

    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()

    _, final_loss = model(inputs, targets)
    final_loss_value = final_loss.item()

    assert final_loss_value < initial_loss_value * 0.1, (
        f"expected at least 90% loss reduction, "
        f"but loss changed from {initial_loss_value:.4f} "
        f"to {final_loss_value:.4f}"
    )
