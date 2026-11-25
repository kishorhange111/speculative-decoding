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

**Results are pending** from the Colab run.
