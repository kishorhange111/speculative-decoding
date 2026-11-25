"""Autoregressive, speculative and prompt-lookup decoding.

**The rule** (Leviathan et al. 2023; Chen et al. 2023). The draft proposes x₁..x_γ with
probabilities p_i. The target scores all of them in one forward, giving q_1..q_{γ+1}. Each
x_i is accepted with probability min(1, q_i(x_i) / p_i(x_i)). At the first rejection, a
replacement is drawn from the residual norm(max(0, q_i − p_i)) and the rest is discarded. If
all γ are accepted, a bonus token is drawn from q_{γ+1}. The output is distributed
**exactly** as sampling from the target alone, because for every token x

    P(out = x) = min(p(x), q(x)) + (1 − Σ_y min(p(y), q(y))) · norm(q − p)⁺(x) = q(x).

Greedy decoding is the same rule with one-hot p and q (temperature 0), so it needs no
separate code path: a draft token is accepted iff it is the target's argmax.

**Prompt lookup** (Saxena 2023) drafts with no model at all. It copies the γ tokens that
followed the most recent earlier occurrence of the last n-gram. Its proposal is
deterministic, a one-hot p, so the same rule applies unchanged.

**Cache invariant.** Both models' caches hold every committed token except the newest one,
``cur``. Each round feeds ``cur`` first, so rolling back after a rejection is a single
``truncate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .model import GPT2


def probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
    return (logits / temperature).softmax(-1)


def sample(p: torch.Tensor, gen=None) -> int:
    return int(torch.multinomial(p, 1, generator=gen))


def verify(draft: list[int], p: torch.Tensor, q: torch.Tensor, gen=None) -> tuple[int, int]:
    """Accept/reject ``draft`` (γ tokens) with draft probs ``p (γ, V)`` and target probs
    ``q (γ+1, V)``. Returns (number accepted, next token)."""
    for i, x in enumerate(draft):
        u = torch.rand((), generator=gen, device=q.device)
        if u * p[i, x] < q[i, x]:  # u < q/p without dividing; p(x) > 0 since x was drawn
            continue
        r = (q[i] - p[i]).clamp(min=0)
        return i, sample(r / r.sum(), gen)
    return len(draft), sample(q[len(draft)], gen)


@dataclass
class Stats:
    target_calls: int = 0
    draft_calls: int = 0
    proposed: int = 0
    accepted: int = 0
    per_round: list[int] = field(default_factory=list)

    @property
    def acceptance(self) -> float:
        return self.accepted / max(1, self.proposed)

    @property
    def tokens_per_target_call(self) -> float:
        return (sum(self.per_round) + len(self.per_round)) / max(1, len(self.per_round))


def _t(tokens, device):
    return torch.tensor([tokens], device=device)


def autoregressive(target: GPT2, prompt: list[int], n: int, temperature=0.0, gen=None):
    dev = target.wte.device
    target.reset()
    out = list(prompt)
    q = probs(target.extend(_t(prompt, dev))[0, -1], temperature)
    for _ in range(n):
        out.append(sample(q, gen))
        if len(out) - len(prompt) == n:
            break
        q = probs(target.extend(_t(out[-1:], dev))[0, -1], temperature)
    return out[len(prompt) :], Stats(target_calls=n)


def speculative(
    target: GPT2,
    draft: GPT2 | None,
    prompt: list[int],
    n: int,
    gamma: int = 4,
    temperature=0.0,
    gen=None,
    ngram: int = 3,
):
    """Generate ``n`` tokens. ``draft=None`` uses prompt lookup with ``ngram``-gram matching."""
    dev = target.wte.device
    st = Stats()
    target.reset()
    target.extend(_t(prompt[:-1], dev))
    if draft is not None:
        draft.reset()
        draft.extend(_t(prompt[:-1], dev))
    out = list(prompt)
    pending = [out[-1]]  # tokens not yet in the draft's cache (always ends with cur)
    while len(out) - len(prompt) < n:
        g = min(gamma, n - (len(out) - len(prompt)))
        base = len(out) - 1  # cache length both models share
        if draft is not None:
            ps, xs = [], []
            logits = draft.extend(_t(pending, dev))[0, -1]
            st.draft_calls += 1
            for i in range(g):
                ps.append(probs(logits, temperature))
                xs.append(sample(ps[-1], gen))
                if i < g - 1:
                    logits = draft.extend(_t(xs[-1:], dev))[0, -1]
                    st.draft_calls += 1
            p = torch.stack(ps)
        else:
            xs = lookup(out, g, ngram)
            p = torch.nn.functional.one_hot(torch.tensor(xs, dtype=torch.long), target.vocab)
            p = p.to(device=dev, dtype=torch.float32)
        q = probs(target.extend(_t([out[-1], *xs], dev))[0], temperature)
        st.target_calls += 1
        k, nxt = verify(xs, p, q, gen) if xs else (0, sample(q[0], gen))
        st.proposed, st.accepted = st.proposed + len(xs), st.accepted + k
        st.per_round.append(k)
        out += xs[:k] + [nxt]
        target.truncate(base + 1 + k)
        if draft is not None:
            if k == len(xs):  # draft never fed its own last proposal
                pending = [xs[-1], nxt] if xs else [nxt]
            else:
                draft.truncate(base + 1 + k)
                pending = [nxt]
    return out[len(prompt) : len(prompt) + n], st


def lookup(seq: list[int], g: int, ngram: int) -> list[int]:
    """The ``g`` tokens after the latest earlier occurrence of the final n-gram (n ≤ ngram,
    longest first). Empty when there is no match."""
    for m in range(min(ngram, len(seq) - 1), 0, -1):
        tail = seq[-m:]
        for s in range(len(seq) - m - 1, -1, -1):
            if seq[s : s + m] == tail:
                return seq[s + m : s + m + g]
    return []
