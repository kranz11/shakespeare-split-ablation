import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionHead(nn.Module):
    def __init__(self,input_dim:int,head_size:int,block_size:int,dropout:float):
        super().__init__()
        self.query = nn.Linear(input_dim, head_size, bias=False)
        self.key = nn.Linear(input_dim, head_size, bias=False)
        self.value = nn.Linear(input_dim, head_size, bias=False)
        self.register_buffer(
            "causal_mask", torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_scores = q@k.transpose(-2,-1) * (k.shape[-1] ** -0.5)
        attn_scores = attn_scores.masked_fill(self.causal_mask[:T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T, T)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ v  # (B, T, head_size)
        return out

class Multiheadattention(nn.Module):
    def __init__(self,input_dims:int,num_heads:int,block_size:int,dropout:float):
        super().__init__()
        assert input_dims % num_heads == 0, "input_dim must be divisible by num_heads"
        head_size = input_dims // num_heads
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(input_dims, head_size, block_size, dropout)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(input_dims, input_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Feedforward(nn.Module):
    def __init__(self,input_dims:int,dropout:float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dims,4*input_dims),
            nn.GELU(),
            nn.Linear(4*input_dims,input_dims),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)

class Transformer_decoder(nn.Module):
    def __init__(self,input_dim: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(input_dim)
        self.attn = Multiheadattention(input_dim,num_heads,block_size,dropout)
        self.ln2 = nn.LayerNorm(input_dim)
        self.feedfwd = Feedforward(input_dim,dropout)

    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.feedfwd(self.ln2(x))
        return x

class SmallLM(nn.Module):
    def __init__(self,input_dim: int, vocab_size: int, num_heads: int, num_layers: int, block_size: int, dropout: float):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, input_dim)
        self.position_embedding = nn.Embedding(block_size, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                Transformer_decoder(input_dim, num_heads, block_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(input_dim)
        self.lm_head = nn.Linear(input_dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, (
            f"Sequence length {T} exceeds block_size {self.block_size}"
        )

        # Look up token embeddings: (B, T, input_dim)
        tok_emb = self.token_embedding(idx)

        # Look up positional embeddings for positions 0..T-1: (T, input_dim)
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(positions)

        # Combine token identity + position, then pass through the model.
        x = self.dropout(tok_emb + pos_emb)  # (B, T, input_dim), broadcasting over batch

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy expects (N, num_classes) and (N,), so we
            # flatten the batch and time dimensions together.
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Autoregressive text generation: repeatedly predict the next
        token, sample from that distribution, append it, and repeat.

        idx: (batch, time) tensor of token ids to use as the starting
                "prompt".
        """
        self.eval()
        for _ in range(max_new_tokens):
            # The model only has learned positional embeddings up to
            # block_size, so if our context is longer than that we crop
            # it down to just the most recent block_size tokens.
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            # We only care about the prediction for the very last
            # position -- that's the "next token" prediction.
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # (B, vocab_size)

            if top_k is not None:
                # Optionally restrict sampling to only the top_k most
                # likely tokens, which tends to produce more coherent
                # text than sampling from the full distribution.
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)  # append and continue

        self.train()
        return idx
