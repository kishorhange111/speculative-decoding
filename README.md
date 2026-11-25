# speculative-decoding

[![CI](https://github.com/kishorhange111/speculative-decoding/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/speculative-decoding/actions/workflows/ci.yml)

Speculative decoding from scratch, with a hand-written GPT-2 and KV cache: a small draft
model proposes γ tokens and the large target model checks them all in **one** forward
pass. Here gpt2 or gpt2-medium drafts for gpt2-xl (1.5B). The repo also has prompt-lookup
decoding, which drafts with no model at all.

## The algorithm ([`decode.py`](src/specdec/decode.py))

1. The draft samples x₁..x_γ, keeping its probabilities p₁..p_γ.
2. The target extends its cache by `[cur, x₁..x_γ]` in one forward, giving q₁..q_{γ+1}.
3. Each xᵢ is accepted with probability min(1, qᵢ(xᵢ)/pᵢ(xᵢ)). At the first rejection, the
   replacement comes from norm(max(0, qᵢ − pᵢ)); if every token is accepted, a bonus token
   comes from q_{γ+1}.
4. Both caches are rolled back to the accepted prefix with a single `truncate`.

The output distribution is **exactly** the target's:
`P(x) = min(p, q)(x) + (1 − Σ min(p, q)) · norm(q − p)⁺(x) = q(x)`. Greedy decoding is the
same rule with one-hot p and q, so there is a single code path. Prompt lookup proposes the
tokens that followed the last earlier match of the final n-gram. That proposal is
deterministic (one-hot p), so the same verifier applies.

## Model ([`model.py`](src/specdec/model.py))

GPT-2 is written over Hugging Face weights with a preallocated KV cache. `extend(tokens)`
appends any number of tokens under the right causal mask; `truncate(n)` rolls back.

## Tests

There are 10, on CPU with random small GPT-2s in float64. They check:
- chunked extension plus rollback equals one full forward;
- **one verify round reproduces the target distribution** (Monte Carlo, 40k draws, with a
  draft far from the target);
- **sampled speculative decoding matches the target's joint distribution** over two tokens
  (TV < 0.06). A control test shows that sampling the draft alone is easily told apart
  (TV > 0.12), so this test has power;
- greedy speculative and prompt-lookup output is identical to greedy target decoding for
  γ ∈ {1, 3, 6}, even with an unrelated draft;
- a draft identical to the target accepts 100% and needs ⌈n/(γ+1)⌉ target calls;
- n-gram lookup edge cases and the benchmark bookkeeping.

## Benchmark ([`bench.py`](src/specdec/bench.py))

Target gpt2-xl, fp16, on a T4. There are two domains, WikiText-103 prose and Python
standard-library code, with 10 prompts of 128 tokens each and 128 new tokens per prompt.
The configurations are:
- greedy decoding with draft gpt2 or gpt2-medium and γ ∈ {2, 4, 6, 8};
- prompt lookup with γ ∈ {4, 8};
- T = 1 sampling.

Reported for each: tokens/s, speed-up over plain greedy decoding, acceptance, tokens per
target call, and whether greedy output stays identical.

```
pytest -q                                          # 10 tests, CPU
python -m specdec.bench --out results              # GPU, resumable (runs.jsonl)
```

## Results (Tesla T4, Colab)

The target is gpt2-xl in fp16, decoding 10 prompts × 128 new tokens per domain. The full
tables (28 configurations) are in [`results/summary.md`](results/summary.md) and the raw
rows in [`results/runs.jsonl`](results/runs.jsonl). Plain greedy decoding of the target runs
at 37.5 tokens/s on code and 36.8 on prose.

| domain | method | γ | speed-up | acceptance | tokens / target call |
|---|---|---|---|---|---|
| code | draft gpt2 | 4 | 1.73× | 68% | 3.71 |
| code | draft gpt2-medium | 2 | 1.29× | 86% | 2.71 |
| code | prompt lookup | 8 | **1.95×** | 36% | 2.21 |
| prose | draft gpt2 | 4 | 1.52× | 59% | 3.33 |
| prose | draft gpt2-medium | 8 | 0.92× | 49% | 4.87 |
| prose | prompt lookup | 8 | **2.12×** | 26% | 2.29 |
| code, T = 1 | draft gpt2 | 4 | 1.11× | 37% | 2.45 |
| prose, T = 1 | prompt lookup | 4 | 1.01× | 3% | 1.04 |

What the numbers show:
- **Prompt lookup was the fastest method in both domains**, at 1.95× and 2.12×. It commits
  only 2.2–2.3 tokens per target call, against 3.3–3.7 for the gpt2 draft. But its drafts
  cost nothing, while the gpt2 draft runs γ forward passes before every verification.
- **More acceptance is not more speed.** gpt2-medium is accepted more often than gpt2
  (86% vs 78% on code at γ=2), but it is always slower, because a 355M draft is a large
  fraction of the 1.5B target's cost. At γ=8 on prose it is slower than not speculating
  (0.92×).
- **For a model draft, γ=4 is the sweet spot.** Acceptance falls steadily with γ (gpt2 on
  code: 78% → 68% → 58% → 50%), so past γ=4 the extra draft passes cost more than the extra
  accepted tokens save.
- **Sampling at T = 1 erases most of the gain** (1.01–1.17×). The target's distribution is
  flatter, so the `min(1, q/p)` test rejects more often. Lookup's one-hot proposals are
  almost never accepted on prose (3%).
- **Greedy outputs: 100% identical on code, 80% on prose with a model draft.** Every
  draft-model config on prose has 8 of 10 sequences identical to plain greedy decoding,
  while prompt lookup has 10 of 10. The fp64 CPU tests show that the algorithm reproduces greedy
  decoding exactly. The likely cause is fp16: scoring γ + 1 tokens in one forward rounds
  differently from scoring one, so a near-tie can flip and the sequence then diverges.
  This run did not record which tokens flipped.
