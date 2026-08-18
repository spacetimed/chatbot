import torch
from chatbot.model import GPT

# basic low-compute tests to ensure the model is functional

def test_model():
    torch.manual_seed(1337)

    model = GPT(
        vocab_size=64,
        block_size=16,
        n_embed=32,
        num_heads=4,
        n_layer=2,
        dropout=0.0,
    )

    tokens = torch.randint(0, 64, (2, 9)) # 2x9 containing rand int [0,64)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    logits, loss = model(inputs, targets)

    assert logits.shape == (2, 8, 64)
    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()
    assert model.token_embedding_table.weight.grad is not None