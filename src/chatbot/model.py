import torch
import torch.nn as nn
import torch.nn.functional as F

from chatbot.config import GPTConfig

# start vectorized attention write to prep for GPT-2
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.config = config
        self.block_size = config.block_size
        self.n_embed = config.n_embed
        self.n_head = config.n_head

        assert config.n_embed % config.n_head == 0
        self.head_size = config.n_embed // config.n_head

        # one projection for all Q,K,V values
        # linear layer which takes C vector from each BxT example -> transform to 3C values (representing QKV concatenated)
        self.c_attn = nn.Linear(
            config.n_embed,
            3 * config.n_embed,
            bias=True,
        )

        # 1 output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        # separate dropout modules for clarity
        self.attn_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(
            torch.ones(config.block_size, config.block_size, dtype=torch.bool)
        )

        # 4 dimensional mask: [1,1,block_size,block_size]
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # vectorized QKV summary:
        #   start with [B,T,C] (each (b,t) position contains an n_embed vector)
        #   pass into linear layer to produce [B,T,3C]
        #   (each 3C vector contains concatenated q,k,v vectors)
        #   unpack that [B,T,3C] tensor into three [B,T,C] tensors
        #   (just de-concatenating q, k, and v)
        #   separate each C component according to C=HxD
        #   (H = n_head; D = dimensionality of each head)
        #   so each token (b,t) has an HxD grid composed from partitioning the C-vector

        B, T, C = x.shape  # [B,T,C] = [batch, token, n_embed]

        if T > self.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.block_size}"
            )

        # concatenated QKV vector for each (b,t) position after the linear layer.
        qkv = self.c_attn(x)  # [B,T,3C]

        # extract Q, K, and V from the three C-sized chunks of 3C.
        q, k, v = qkv.split(self.n_embed, dim=-1)  # each is [B,T,C]

        q = q.view(B, T, self.n_head, self.head_size)  # [B,T,H,D]
        q = q.transpose(1, 2)  # [B,H,T,D]

        k = k.view(B, T, self.n_head, self.head_size)  # [B,T,H,D]
        k = k.transpose(1, 2)  # [B,H,T,D]

        v = v.view(B, T, self.n_head, self.head_size)  # [B,T,H,D]
        v = v.transpose(1, 2)  # [B,H,T,D]

        # at this point, each (b,t) token position has three tiny HxD grids: query, key, value grids

        # key = [B,H,T,D] -> [B,H,D,T]
        #           -2 -1 <- these dim offsets get swapped
        # for each (b,h), compute [T,D] @ [D,T] to compare every query token
        # against every key token; this produces one [T,T] score matrix per head
        scores = q @ k.transpose(-2, -1)  # [B,H,T,T]
        scores = scores * (self.head_size ** -0.5)

        # triangular mask prevents each token from attending to future tokens
        scores = scores.masked_fill(
            ~self.causal_mask[:, :, :T, :T],
            float("-inf"),
        )

        # softmax
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # use attention weights to take weighted combination of value vectors
        out = weights @ v  # [B,H,T,D]

        # restore organization where each (b,t) position contains HxD grid
        out = out.transpose(1, 2)  # [B,T,H,D]

        # before transpose: [B, H, T, D]
        # after transpose:  [B, T, H, D]
        #                          | /
        #                          |/
        # use .view to:     [B, T, C]
        # need .contiguous() for view because transpose() changed tensor's order
        out = out.contiguous().view(B, T, C)  # [B,T,C]

        # mix information across heads, then apply residual-stream dropout
        out = self.c_proj(out)  # [B,T,C]
        out = self.residual_dropout(out)

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
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.sa = CausalSelfAttention(config)
        self.ffwd = FeedForward(config.n_embed, config.dropout)

        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        # forward pass remains unchanged for vectorized attn
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        # constants
        self.config = config
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.n_embed = config.n_embed
        self.n_heads = config.n_head
        self.n_layer = config.n_layer
        self.dropout = config.dropout

        # embedding tables
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embed)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embed)

        # transformer blocks; transformer -> (MHA -> [SHA, ...]) + FF)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )

        # final normalization + vocab projection
        self.ln_f = nn.LayerNorm(self.n_embed)
        self.lm_head = nn.Linear(self.n_embed, self.vocab_size)

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
