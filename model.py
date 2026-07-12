"""
STEP 2: The model itself.

This is a small GPT (Generative Pre-trained Transformer) -- the same family
of architecture behind ChatGPT, Claude, Llama, etc, just much smaller.

Core idea in plain terms:
  - The model reads a chunk of characters (its "context")
  - "Attention" lets every character look at every earlier character and
    decide which ones matter for predicting what comes next
  - It outputs a probability for "what character comes next"
  - During training we compare its guess to the real next character and
    nudge its internal numbers (weights) to be less wrong next time

We are NOT using a pretrained model. Every number in this network starts
random and only learns from the Shakespeare text we feed it.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class SelfAttentionHead(nn.Module):
    """One 'attention head': lets each character look back at earlier ones."""
    def __init__(self, embed_size, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        # a mask so characters can only look at PAST characters, not future ones
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # how much each token "attends" to others
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # can't see the future
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Runs several attention heads in parallel, each learning different patterns."""
    def __init__(self, num_heads, embed_size, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_size, head_size, block_size, dropout) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(head_size * num_heads, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """A small 2-layer network applied to each position -- lets the model 'think'
    about what attention found."""
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block = attention + feedforward, with residual connections."""
    def __init__(self, embed_size, num_heads, block_size, dropout):
        super().__init__()
        head_size = embed_size // num_heads
        self.sa = MultiHeadAttention(num_heads, embed_size, head_size, block_size, dropout)
        self.ff = FeedForward(embed_size, dropout)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(*[
            Block(embed_size, num_heads, block_size, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=20):
        """Given some starting text (as numbers), keep predicting and appending
        the next character, one at a time.

        temperature: lower = more confident/conservative (less random),
                     higher = more random/creative. 1.0 is "as trained".
        top_k: only sample from the k most likely next characters, instead
               of the full distribution -- cuts out weird, unlikely picks.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # last position's prediction

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
