"""Llama-style decoder-only transformer, 29.4M params: RoPE, pre-norm RMSNorm,
SwiGLU, no biases, tied embeddings. Running this file checks the pieces and
then overfits one batch."""

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 4096
N_LAYER = 8
N_HEAD = 8
DIM = 512
HEAD_DIM = DIM // N_HEAD
CTX = 512
THETA = 10000.0
INIT_STD = 0.02

def hidden(dim):
    """4*dim cut by 2/3 for SwiGLU's third matrix, rounded up to 256."""
    return 256 * math.ceil(8 * dim / 3 / 256)

HIDDEN = hidden(DIM)  # 1536

def rope_tables(head_dim, ctx, device=None):
    """cos/sin of the RoPE angles at every position, both (ctx, head_dim)."""
    # float64 because the angle at the last position is a few hundred radians,
    # where float32 spacing already swamps the relative positions we want.
    freqs = THETA ** (-torch.arange(0, head_dim, 2, device=device).double() / head_dim)
    angles = torch.outer(torch.arange(ctx, device=device).double(), freqs)
    # Duplicating rather than interleaving pairs channel i with i + head_dim/2,
    # so the rotation is one concat and two products.
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos().float(), angles.sin().float()

def rope(x, cos, sin):
    """Rotate x, (B, heads, T, head_dim), by its position angles."""
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # In fp32: a sum of 512 bf16 squares loses enough precision to shift
        # the scale of every activation.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)

class Attention(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin, cache=None, use_cache=False):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, -1).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, -1).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, -1).transpose(1, 2)
        q, k = rope(q, cos, sin), rope(k, cos, sin)  # position enters via q, k only
        past = 0 if cache is None else cache[0].shape[2]
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        if past and T > 1:
            # Cached chunks need a causal mask offset by the prefix length.
            mask = torch.arange(k.shape[2], device=x.device)[None, :] <= (
                past + torch.arange(T, device=x.device)[:, None])
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            # A single new query can see the entire cache. is_causal=True
            # would align it with the FIRST key rather than the last one.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=past == 0)
        return self.wo(y.transpose(1, 2).reshape(B, T, C)), (k, v) if use_cache else None

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden(dim), bias=False)
        self.w2 = nn.Linear(hidden(dim), dim, bias=False)
        self.w3 = nn.Linear(dim, hidden(dim), bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_head)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x, cos, sin, cache=None, use_cache=False):
        attention, cache = self.attn(self.attn_norm(x), cos, sin, cache, use_cache)
        x = x + attention
        return x + self.mlp(self.mlp_norm(x)), cache

class Sprout(nn.Module):
    def __init__(self, n_layer=N_LAYER, dim=DIM, n_head=N_HEAD, ctx=CTX, vocab=VOCAB_SIZE,
                 pos="rope"):
        super().__init__()
        assert pos in ("rope", "learned"), pos
        self.ctx = ctx
        self.tok_emb = nn.Embedding(vocab, dim)
        # GPT-2's learned absolute positions, the alternative to RoPE.
        self.pos_emb = nn.Embedding(ctx, dim) if pos == "learned" else None
        self.blocks = nn.ModuleList(Block(dim, n_head) for _ in range(n_layer))
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok_emb.weight  # tie before init: one tensor only

        self.apply(init_weights)
        # The projections writing into the residual stream start smaller, so
        # its variance grows like a sum of 2*n_layer contributions.
        for block in self.blocks:
            for w in (block.attn.wo.weight, block.mlp.w2.weight):
                nn.init.normal_(w, std=INIT_STD / math.sqrt(2 * n_layer))

        cos, sin = rope_tables(dim // n_head, ctx)
        if self.pos_emb is not None:
            cos, sin = torch.ones_like(cos), torch.zeros_like(sin)  # rotation by 0
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _hidden(self, idx, cache=None, use_cache=False):
        T = idx.shape[1]
        if cache is not None:
            assert use_cache and len(cache) == len(self.blocks)
        offset = 0 if cache is None else cache[0][0].shape[2]
        assert 0 < T and offset + T <= self.ctx, "input exceeds the model context"
        if cache is not None:
            for k, v in cache:
                assert k.shape == v.shape == (idx.shape[0], self.blocks[0].attn.n_head,
                                               offset, self.cos.shape[1])
                assert k.device == v.device == idx.device
        cos, sin = self.cos[offset:offset + T], self.sin[offset:offset + T]
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(offset, offset + T, device=idx.device))
        updated = []
        for i, block in enumerate(self.blocks):
            x, entry = block(x, cos, sin, None if cache is None else cache[i], use_cache)
            if use_cache:
                updated.append(entry)
        return self.norm(x), tuple(updated)

    def forward(self, idx, targets=None):
        logits = self.head(self._hidden(idx)[0])
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    @torch.no_grad()
    def forward_cached(self, idx, cache=None):
        """Return chunk logits and per-layer (K, V), each (B, heads, T, head_dim)."""
        assert not self.training, "KV caching requires model.eval()"
        x, cache = self._hidden(idx, cache, use_cache=True)
        return self.head(x), cache

    @torch.no_grad()
    def generate_tokens(self, idx, n, temperature=1.0, stop=None, top_k=None, use_cache=True):
        """Yield (B, 1) token tensors; cache lives only in this iterator."""
        assert not self.training, "generation requires model.eval()"
        assert idx.ndim == 2 and idx.shape[0] > 0 and idx.shape[1] > 0
        assert n >= 0 and temperature >= 0, (n, temperature)
        assert top_k is None or 0 < top_k <= self.head.out_features, top_k
        assert stop is None or 0 <= stop < self.head.out_features, stop
        cache = None
        finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        for _ in range(n):
            if use_cache:
                if cache is None or cache[0][0].shape[2] == self.ctx:
                    # Sliding the window changes every retained hidden state.
                    # Rebuild it at position zero to match uncached generation.
                    logits, cache = self.forward_cached(idx[:, -self.ctx:])
                else:
                    logits, cache = self.forward_cached(idx[:, -1:], cache)
                logits = logits[:, -1]
            else:
                logits = self(idx[:, -self.ctx:])[0][:, -1]
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    cutoff = logits.topk(top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                nxt = torch.multinomial(logits.softmax(-1), 1)
            if stop is not None:
                nxt = nxt.masked_fill(finished[:, None], stop)
                finished |= nxt[:, 0] == stop
            idx = torch.cat([idx, nxt], dim=1)
            yield nxt
            if stop is not None and finished.all():
                break

    @torch.no_grad()
    def generate(self, idx, n, temperature=1.0, stop=None, top_k=None, use_cache=True):
        """Append up to n tokens; temperature=0 selects greedy decoding."""
        tokens = list(self.generate_tokens(idx, n, temperature, stop, top_k, use_cache))
        return torch.cat([idx, *tokens], dim=1) if tokens else idx

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, std=INIT_STD)


@torch.no_grad()
def check_cache(device):
    """Check cached chunks, sliding windows, streaming, and batched stopping."""
    torch.manual_seed(1337)
    for pos in ("rope", "learned"):
        model = Sprout(n_layer=2, dim=32, n_head=4, ctx=64, vocab=256, pos=pos).to(device).eval()
        idx = torch.randint(256, (2, 13), device=device)
        ref = model(idx)[0]
        first, cache = model.forward_cached(idx[:, :5])
        single, cache = model.forward_cached(idx[:, 5:6], cache)
        last, cache = model.forward_cached(idx[:, 6:], cache)
        logits = torch.cat([first, single, last], dim=1)
        error = (logits - ref).abs().max().item()
        assert torch.allclose(logits, ref, atol=1e-6, rtol=1e-5), error
        assert all(k.shape == v.shape == (2, 4, 13, 8) for k, v in cache)
        assert all(not k.requires_grad and not v.requires_grad for k, v in cache)

        cached = model.generate(idx, 224, temperature=0)
        plain = model.generate(idx, 224, temperature=0, use_cache=False)
        assert torch.equal(cached, plain), (cached != plain).nonzero()
        # A prefix longer than the context must use exactly the final window.
        assert torch.equal(model.generate(cached, 3, temperature=0),
                           model.generate(cached, 3, temperature=0, use_cache=False))
        assert torch.equal(model.generate(idx, 0), idx)
        tokens = list(model.generate_tokens(idx, 224, temperature=0))
        assert torch.equal(torch.cat([idx, *tokens], dim=1), cached)

        stop = int(cached[0, idx.shape[1]])
        stopped = model.generate(idx, 32, temperature=0, stop=stop)
        assert (stopped[0, idx.shape[1]:] == stop).all()
        assert torch.equal(stopped, model.generate(idx, 32, temperature=0, stop=stop,
                                                  use_cache=False))
        if stopped.shape[1] < idx.shape[1] + 32:
            assert (stopped[:, -1] == stop).all()
        print(f"{pos} on {device}: 224 greedy tokens match exactly across context rollover; "
              f"chunk error {error:.2e}; streaming and batched stopping pass")

def main(device):
    check_cache(device)
    torch.manual_seed(1337)
    model = Sprout()

    # Params, against a hand count of the same architecture.
    expected = (
        VOCAB_SIZE * DIM
        + N_LAYER * (4 * DIM * DIM + 3 * DIM * HIDDEN + 2 * DIM)
        + DIM
    )
    actual = sum(p.numel() for p in model.parameters())
    print(f"params: {actual:,} (hand count {expected:,})")
    assert actual == expected, (actual, expected)
    assert model.head.weight is model.tok_emb.weight
    with torch.random.fork_rng(devices=[]):  # leave the seeded checks below as they were
        learned = sum(p.numel() for p in Sprout(pos="learned").parameters())
    print(f"params with learned positions: {learned:,} (+ctx*dim = {CTX * DIM:,})")
    assert learned == expected + CTX * DIM

    # RMSNorm against the definition written longhand.
    x = torch.randn(4, 7, DIM)
    norm = RMSNorm(DIM)
    with torch.no_grad():
        norm.weight.normal_()
    ref = norm.weight * x / torch.sqrt((x * x).sum(-1, keepdim=True) / DIM + norm.eps)
    print(f"rmsnorm vs reference: {(norm(x) - ref).abs().max():.2e}")
    assert torch.allclose(norm(x), ref, atol=1e-6)

    # RoPE at position 0 is a no-op, and q.k depends only on the gap m - n.
    cos, sin = (t.double() for t in rope_tables(HEAD_DIM, CTX))
    q = torch.randn(2, 1, 1, HEAD_DIM, dtype=torch.float64)
    k = torch.randn(2, 1, 1, HEAD_DIM, dtype=torch.float64)
    print(f"rope at position 0: {(rope(q, cos[0], sin[0]) - q).abs().max():.2e}")
    assert torch.allclose(rope(q, cos[0], sin[0]), q, atol=1e-6)

    def dot(m, n):
        return (rope(q, cos[m], sin[m]) * rope(k, cos[n], sin[n])).sum(-1)

    worst = max(
        (dot(m, n) - dot(m + d, n + d)).abs().max().item()
        for m, n, d in [(0, 0, 7), (3, 11, 100), (17, 5, 300), (100, 200, 211)]
    )
    print(f"rope relative shift, worst gap: {worst:.2e}")
    assert worst < 1e-6

    # Flash attention against the masked softmax it replaces.
    q, k, v = (torch.randn(2, N_HEAD, 64, HEAD_DIM) for _ in range(3))
    scores = q @ k.transpose(-2, -1) / math.sqrt(HEAD_DIM)
    causal = torch.ones(64, 64, dtype=torch.bool).tril()
    manual = scores.masked_fill(~causal, float("-inf")).softmax(-1) @ v
    flash = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"manual attention vs sdpa: {(manual - flash).abs().max():.2e}")
    assert torch.allclose(manual, flash, atol=1e-5)

    # A model that has learned nothing pays about ln(V) per token.
    val = np.memmap("data/val.bin", dtype=np.uint16, mode="r")
    real = torch.from_numpy(val[: 8 * (CTX + 1)].astype(np.int64)).view(8, CTX + 1)
    model.eval()
    with torch.no_grad():
        for name, b in [("random", torch.randint(VOCAB_SIZE, (8, CTX + 1))), ("real", real)]:
            logits, loss = model(b[:, :-1], b[:, 1:])
            s = logits.std().item()
            print(f"init loss ({name}): {loss:.4f}, logit std {s:.3f}, "
                  f"ln(V) + std^2/2 = {math.log(VOCAB_SIZE) + s ** 2 / 2:.4f}")

    # Causality: change one token and nothing before it moves, bitwise.
    idx = torch.randint(VOCAB_SIZE, (1, 64))
    t = 40
    with torch.no_grad():
        before = model(idx)[0]
        idx[0, t] = (idx[0, t] + 1) % VOCAB_SIZE
        after = model(idx)[0]
    print(f"causality: same up to t={t}: {torch.equal(before[:, :t], after[:, :t])}, "
          f"changed at t: {(before[:, t] - after[:, t]).abs().max():.3f}")
    assert torch.equal(before[:, :t], after[:, :t])
    assert not torch.equal(before[:, t], after[:, t])

    # Overfit one batch, the only check that runs backward. 32x128 rather than
    # the full context, which needs ~9GB of activations.
    model = Sprout().to(device)
    model.train()
    seq = 128
    batch = torch.from_numpy(val[: 32 * (seq + 1)].astype(np.int64)).view(32, seq + 1)
    x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    print(f"overfitting 32x{seq} tokens on {device}")
    start = time.time()
    for step in range(301):
        loss = model(x, y)[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Once the batch is memorized the gradients are tiny and Adam's step
        # blows up; without clipping the loss spikes back out.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"  step {step:3d}  loss {loss.item():.4f}  {time.time() - start:.0f}s")
    assert loss.item() < 0.1, loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-cache", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    args = parser.parse_args()
    if args.check_cache:
        check_cache(args.device)
    else:
        main(args.device)
