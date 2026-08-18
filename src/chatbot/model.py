import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    tril: torch.Tensor

    def __init__(self, block_size, n_embed, head_size, dropout=0.0):
        super().__init__()

        self.block_size = block_size
        self.head_size = head_size

        # Q,K,V linear layers
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        # register triangular mask as buffer
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        _, T, _ = x.shape

        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")

        q = self.query(x)  # [B,T,H]
        k = self.key(x)  # [B,T,H]
        v = self.value(x)  # [B,T,H]

        wei = q @ k.transpose(-2, -1)  # attention scores: q dot k pairs
        wei = wei * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # [B,T,T]
        wei = self.dropout(wei)

        out = wei @ v  # [B,T,H] context-rich vectors
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, block_size, n_embed, num_heads, dropout=0.0):
        super().__init__()

        assert n_embed % num_heads == 0  # avoid inadvertent truncating

        head_size = n_embed // num_heads

        self.heads = nn.ModuleList(
            [SingleHeadAttention(block_size, n_embed, head_size, dropout) for _ in range(num_heads)]
        )

        # project concatenated head outputs back into model channel space
        self.proj = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # run x example on all single-heads, concatenate output, project to n_embed
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # applies the same MLP independently to every (b,t) vector
        return self.net(x)


class Block(nn.Module):
    def __init__(self, block_size, n_embed, num_heads, dropout=0.0):
        super().__init__()

        self.sa = MultiHeadAttention(block_size, n_embed, num_heads, dropout)
        self.ffwd = FeedForward(n_embed, dropout)

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            block_size, 
            n_embed, 
            num_heads, 
            n_layer, 
            dropout=0.0
        ):
        super().__init__()

        # constants
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.num_heads = num_heads
        self.n_layer = n_layer
        self.dropout = dropout

        # embedding tables
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        # transformer blocks; transformer -> (MHA -> [SHA, ...]) + FF)
        self.blocks = nn.Sequential(
            *[Block(block_size, n_embed, num_heads, dropout) for _ in range(n_layer)]
        )

        # final normalization + vocab projection
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        _, T = idx.shape

        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")

        # idx: [B,T]
        tok_emb = self.token_embedding_table(idx)  # [B,T,C]
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # [T,C]

        x = tok_emb + pos_emb  # [B,T,C]
        x = self.blocks(x)  # [B,T,C]
        x = self.ln_f(x)  # [B,T,C]

        logits = self.lm_head(x)  # [B,T,vocab_size]

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits_flat = logits.reshape(B * T, V)
            targets_flat = targets.reshape(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        if was_training:
            self.train()

        return idx
