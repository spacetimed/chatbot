import torch
import torch.nn as nn
import torch.nn.functional as F

from chatbot.config import GPTConfig


class KVCache(nn.Module):
    """
    stores kv activations for one TransformerBlock
    stores a copy of K,V ; each TransformerBlock forward pass will fetch/update
    no computation. just a datastore. first pass populates empty cache
    """

    # first attempt
    def __init__(self):
          super().__init__()
          self.register_buffer("k", None, persistent=False)
          self.register_buffer("v", None, persistent=False)

    def extend(self, k, v):
        self.k = k if self.k is None else torch.cat((self.k, k), dim=2)
        self.v = v if self.v is None else torch.cat((self.v, v), dim=2)
        return self.k, self.v

    def reset(self):
        self.k = self.v = None


# start vectorized attention write to prep for GPT-2
class Attention(nn.Module):
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
        self.c_proj.SCALE_RESIDUAL = True

        # separate dropout modules for clarity
        self.attn_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        self.kv_cache = None

        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))

        # 4 dimensional mask: [1,1,block_size,block_size]
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.block_size, config.block_size),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape # [B,T,C] = [batch, token, n_embed]

        if T > self.block_size: raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")

        # concatenated QKV vector for each (b,t) position after the linear layer.
        qkv = self.c_attn(x) # [B,T,3C] concatenated q,k,v
        q,k,v = qkv.split(self.n_embed, dim=-1) # 3x[B,T,C] (partition qkv from concatenated vectors)

        q = q.view(B, T, self.n_head, self.head_size) # [B,T,H,D] -> each (b,t) now contains [HxD] s.t. each row of H contains D vec
        k = k.view(B, T, self.n_head, self.head_size) # [B,T,H,D]
        v = v.view(B, T, self.n_head, self.head_size) # [B,T,H,D]

        q,k,v = map(lambda x: x.transpose(1,2), (q,k,v)) # from gpt-fast; quite succinct [B,T,H,D] -> [B,H,T,D]

        # kv cache: extend K,V based off window of history
        #  cache disabled: leave k,v unchanged
        #  cache enabled + empty: store current k,v
        #  cache enabled + populated: replace local k,v with extended history
        if self.kv_cache is not None:
            k,v = self.kv_cache.extend(k,v)

        # replace manual attention calculation with pytorch's optimized implementation
        # i still kept my old comments (from the manual attention) below.
        # pytorch's implementation can dispatch flash attention if supported
        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=(self.config.dropout if self.training else 0.0),
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        out = self.residual_dropout(out)

        # === old manual attention calculation ===
        # # key = [B,H,T,D] -> [B,H,D,T]
        # #           -2 -1 <- these dim offsets get swapped
        # # for each (b,h), compute [T,D] @ [D,T] to compare every query token
        # # against every key token; this produces one [T,T] score matrix per head
        # scores = q @ k.transpose(-2, -1)  # [B,H,T,T]
        # scores = scores * (self.head_size**-0.5)

        # # triangular mask prevents each token from attending to future tokens
        # scores = scores.masked_fill(
        #     ~self.causal_mask[:, :, :T, :T],
        #     float("-inf"),
        # )

        # # softmax (each row is basically a probability distribution for each token idx=i relative to all preceding)
        # weights = F.softmax(scores, dim=-1)
        # weights = self.attn_dropout(weights)

        # # use attention weights to take weighted combination of value vectors
        # out = weights @ v  # [B,H,T,D]

        # # restore organization where each (b,t) position contains HxD grid
        # out = out.transpose(1, 2)  # [B,T,H,D]

        # # before transpose: [B, H, T, D]
        # # after transpose:  [B, T, H, D]
        # #                          | /
        # #                          |/
        # # use .view to:     [B, T, C]
        # # need .contiguous() for view because transpose() changed tensor's order
        # out = out.contiguous().view(B, T, C)  # [B,T,C]

        # # mix information across heads, then apply residual-stream dropout
        # out = self.c_proj(out)  # [B,T,C]
        # out = self.residual_dropout(out)
        # === end manual attention calculation ===

        return out


# replace old FeedForward class
class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        # fully-connected expansion layer: expands each token's hidden representation from C -> 4C features
        # [B,T,C] -> [B,T,4C]
        self.c_fc = nn.Linear(
            in_features=config.n_embed,
            out_features=4 * config.n_embed,
        )

        # non-linear transformation to each of the 4C features
        # approximate tanh to match GPT-2's original GELU calc
        self.gelu = nn.GELU(approximate="tanh")

        # output projection layer (compress 4C dim back to C)
        # [B,T,4C] -> [B,T,C]
        self.c_proj = nn.Linear(
            in_features=4 * config.n_embed,
            out_features=config.n_embed,
        )
        self.c_proj.SCALE_RESIDUAL = True

        # random zeroing for training
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # apply the layers/transformations above
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.sa = Attention(config)
        self.mlp = MLP(config)

        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
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
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embed)  # weight [V,C]
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embed)

        # transformer blocks; transformer -> (MHA -> [SHA, ...]) + FF)
        # self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))

        # final normalization + vocab projection
        self.ln_f = nn.LayerNorm(self.n_embed)
        self.lm_head = nn.Linear(self.n_embed, self.vocab_size, bias=False)  # weight [V,C]

        # share the [V,C] weight matrix between token embedding and output projection
        # gpt-2 does this bc the input and output sides refer to same vocabulary
        #  and sharing the weights lowers params by a lot!
        self.lm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:

        if isinstance(module, nn.Linear):
            std = 0.02  # gpt-2

            # 1/ √(2*number of layers)
            # scale residual-output projections to keep accumulated variance tame
            if getattr(module, "SCALE_RESIDUAL", False):
                std *= (2 * self.config.n_layer) ** -0.5

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=std,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(self, idx, targets=None) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = idx.shape

        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")

        # idx: [B,T]
        tok_emb = self.token_embedding_table(idx)  # [B,T,C]
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # [T,C]

        x = tok_emb + pos_emb  # [B,T,C]
        for block in self.blocks: x = block(x) # [B,T,C] ; sequentially propagate through TransformerBlock
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
    def generate(self, idx, max_new_tokens, temperature=1.0) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be above 0")

        was_training = self.training
        self.eval()

        # todo special token for stopping then break loop here

        # data flow (to jog my memory)
        # idx: begins as [ [L] ] (so 1xL) (flattened token stream)

        for _ in range(max_new_tokens):
            # let T = min(L, block_size:=256)

            idx_cond = idx[:, -self.block_size :]  # [1,T] (essentially truncated to max size of 256)

            logits, _ = self(idx_cond)  # [1,T,V]
            # extract final token T-1's prediction. note -1 "selects" last T's V basically
            logits = logits[:, -1, :]  # [1,V]
            logits = logits / temperature  # temperature scaling (logit/(1/t) = logit*t)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        if was_training:
            self.train()

        return idx
