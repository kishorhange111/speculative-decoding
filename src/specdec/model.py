"""GPT-2 written out by hand, with a KV cache that can be extended by several tokens at once
and rolled back. Those are the two operations speculative decoding needs: the target scores
γ + 1 tokens in one forward, and rejected draft tokens are removed from both caches.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

NAMES = (
    "ln_1.weight",
    "ln_1.bias",
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "attn.c_proj.bias",
    "ln_2.weight",
    "ln_2.bias",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    "mlp.c_proj.bias",
)


class GPT2:
    """Weights in Hugging Face's GPT-2 layout (Conv1D, so ``x @ W``)."""

    def __init__(self, sd: dict, n_layer: int, n_head: int, eps: float = 1e-5, max_len=None):
        self.n_layer, self.n_head = n_layer, n_head
        self.wte, self.wpe = sd["transformer.wte.weight"], sd["transformer.wpe.weight"]
        self.d = self.wte.shape[1]
        self.hd, self.eps = self.d // n_head, eps
        self.lnf = (sd["transformer.ln_f.weight"], sd["transformer.ln_f.bias"])
        self.layers = [{n: sd[f"transformer.h.{i}.{n}"] for n in NAMES} for i in range(n_layer)]
        self.max_len = max_len or self.wpe.shape[0]
        self.reset()

    @classmethod
    def from_pretrained(cls, name: str, device="cuda", dtype=torch.float16):
        from transformers import GPT2LMHeadModel

        hf = GPT2LMHeadModel.from_pretrained(name, torch_dtype=dtype).to(device)
        sd = {k: v.detach() for k, v in hf.state_dict().items()}
        return cls(sd, hf.config.n_layer, hf.config.n_head, hf.config.layer_norm_epsilon)

    @property
    def vocab(self) -> int:
        return self.wte.shape[0]

    def reset(self, batch: int = 1):
        shape = (self.n_layer, 2, batch, self.n_head, self.max_len, self.hd)
        self.kv = torch.empty(shape, device=self.wte.device, dtype=self.wte.dtype)
        self.n = 0  # tokens currently cached

    def truncate(self, n: int):
        if not 0 <= n <= self.n:
            raise ValueError(f"cannot truncate a cache of {self.n} tokens to {n}")
        self.n = n

    def _ln(self, x, w, b):
        return F.layer_norm(x, (self.d,), w, b, self.eps)

    @torch.no_grad()
    def extend(self, idx: torch.Tensor) -> torch.Tensor:
        """Append tokens ``idx (b, t)`` after the cached ones; return logits (b, t, vocab)."""
        b, t = idx.shape
        n0, n1 = self.n, self.n + t
        if n1 > self.max_len:
            raise ValueError("context too long")
        x = self.wte[idx] + self.wpe[n0:n1]
        # new token i (absolute n0 + i) sees cached tokens and new tokens ≤ i
        mask = torch.ones(t, n1, dtype=torch.bool, device=idx.device).tril(n0)
        for L, kv in zip(self.layers, self.kv, strict=True):
            h = self._ln(x, L["ln_1.weight"], L["ln_1.bias"])
            qkv = (h @ L["attn.c_attn.weight"] + L["attn.c_attn.bias"]).view(
                b, t, 3, self.n_head, self.hd
            )
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            kv[0, :, :, n0:n1], kv[1, :, :, n0:n1] = k, v
            y = F.scaled_dot_product_attention(q, kv[0, :, :, :n1], kv[1, :, :, :n1], mask)
            x = x + y.transpose(1, 2).reshape(b, t, self.d) @ L["attn.c_proj.weight"]
            x = x + L["attn.c_proj.bias"]
            h = self._ln(x, L["ln_2.weight"], L["ln_2.bias"])
            h = F.gelu(h @ L["mlp.c_fc.weight"] + L["mlp.c_fc.bias"], approximate="tanh")
            x = x + h @ L["mlp.c_proj.weight"] + L["mlp.c_proj.bias"]
        self.n = n1
        return (self._ln(x, *self.lnf) @ self.wte.t()).float()


def random_gpt2(vocab=50, d=32, n_layer=2, n_head=4, ctx=128, seed=0, emb_std=1.0) -> GPT2:
    """Random weights in GPT-2 layout, for tests."""
    g = torch.Generator().manual_seed(seed)

    def r(*s, std=0.2):
        return torch.randn(*s, generator=g) * std

    sd = {
        "transformer.wte.weight": r(vocab, d, std=emb_std),
        "transformer.wpe.weight": r(ctx, d),
        "transformer.ln_f.weight": 1 + r(d),
        "transformer.ln_f.bias": r(d),
    }
    for i in range(n_layer):
        p = f"transformer.h.{i}."
        sd |= {
            p + "ln_1.weight": 1 + r(d),
            p + "ln_1.bias": r(d),
            p + "attn.c_attn.weight": r(d, 3 * d),
            p + "attn.c_attn.bias": r(3 * d),
            p + "attn.c_proj.weight": r(d, d),
            p + "attn.c_proj.bias": r(d),
            p + "ln_2.weight": 1 + r(d),
            p + "ln_2.bias": r(d),
            p + "mlp.c_fc.weight": r(d, 4 * d),
            p + "mlp.c_fc.bias": r(4 * d),
            p + "mlp.c_proj.weight": r(4 * d, d),
            p + "mlp.c_proj.bias": r(d),
        }
    return GPT2({k: v.double() for k, v in sd.items()}, n_layer, n_head)
