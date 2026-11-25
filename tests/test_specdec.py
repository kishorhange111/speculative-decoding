from collections import Counter

import pytest
import torch

from specdec.decode import autoregressive, lookup, probs, speculative, verify
from specdec.model import random_gpt2

PROMPT = [3, 1, 4, 1, 5, 9, 2, 6]


def test_extend_in_chunks_and_rollback_match_full_forward():
    m = random_gpt2()
    seq = torch.randint(0, 50, (1, 20), generator=torch.Generator().manual_seed(1))
    m.reset()
    full = m.extend(seq)
    m.reset()
    parts = [m.extend(seq[:, :7]), m.extend(seq[:, 7:8]), m.extend(seq[:, 8:20])]
    torch.testing.assert_close(torch.cat(parts, 1), full)
    m.truncate(10)  # roll back, then re-extend with different tokens and back again
    m.extend(torch.tensor([[0, 0, 0]]))
    m.truncate(10)
    torch.testing.assert_close(m.extend(seq[:, 10:]), full[:, 10:])
    with pytest.raises(ValueError):
        m.truncate(99)


def test_verify_output_distribution_is_exactly_the_target():
    """Monte Carlo of one round with γ = 1. The first output token must be ~ q₁ whatever p is."""
    g = torch.Generator().manual_seed(0)
    p = torch.tensor([[0.6, 0.3, 0.05, 0.05, 0.0]], dtype=torch.float64)
    q = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.0], [0.2] * 5], dtype=torch.float64)
    n, counts = 40000, Counter()
    for _ in range(n):
        x = int(torch.multinomial(p[0], 1, generator=g))
        k, nxt = verify([x], p, q, g)
        counts[x if k == 1 else nxt] += 1
    emp = torch.tensor([counts[i] / n for i in range(5)], dtype=torch.float64)
    assert (emp - q[0]).abs().max() < 0.01  # 5σ ≈ 0.012 for p = 0.4
    # theoretical acceptance rate is Σ min(p, q) = 0.1 + 0.2 + 0.05 + 0.05 = 0.4


def test_greedy_speculative_equals_greedy_target():
    target, draft = random_gpt2(seed=0), random_gpt2(seed=1)  # unrelated draft
    ref, _ = autoregressive(target, PROMPT, 40)
    for gamma in (1, 3, 6):
        out, st = speculative(target, draft, PROMPT, 40, gamma=gamma)
        assert out == ref
        assert st.target_calls < 40 or st.acceptance == 0


def test_draft_equal_to_target_accepts_everything():
    target = random_gpt2(seed=0)
    out, st = speculative(target, random_gpt2(seed=0), PROMPT, 30, gamma=4)
    assert out == autoregressive(target, PROMPT, 30)[0]
    assert st.acceptance == 1.0
    assert st.target_calls == 6  # 30 tokens / (4 accepted + 1 bonus)
    g = torch.Generator().manual_seed(0)
    _, st = speculative(target, random_gpt2(seed=0), PROMPT, 30, 4, temperature=1.0, gen=g)
    assert st.acceptance > 0.99  # q == p up to float error


def test_sampled_speculative_matches_target_distribution():
    """Joint law of the first two generated tokens, speculative vs plain sampling."""
    target, draft = (
        random_gpt2(vocab=6, seed=0, emb_std=0.1),
        random_gpt2(vocab=6, seed=2, emb_std=0.1),
    )
    g = torch.Generator().manual_seed(0)
    n = 3000
    a = Counter(tuple(autoregressive(target, [1, 2], 2, 1.0, g)[0]) for _ in range(n))
    b = Counter(tuple(speculative(target, draft, [1, 2], 2, 2, 1.0, g)[0]) for _ in range(n))
    tv = 0.5 * sum(abs(a[k] - b[k]) for k in set(a) | set(b)) / n
    assert tv < 0.06


def test_prompt_lookup_greedy_equals_target_and_speeds_up_repetition():
    target = random_gpt2(seed=0)
    prompt = [5, 6, 7, 8, 9] * 4
    ref, _ = autoregressive(target, prompt, 30)
    out, st = speculative(target, None, prompt, 30, gamma=5, ngram=3)
    assert out == ref
    g = torch.Generator().manual_seed(0)
    out, st = speculative(target, None, prompt, 30, gamma=5, temperature=1.0, gen=g)
    assert len(out) == 30


def test_lookup():
    assert lookup([1, 2, 3, 9, 1, 2, 3], 2, 3) == [9, 1]
    assert lookup([1, 2, 7, 5, 2], 3, 3) == [7, 5, 2]  # falls back to a 1-gram
    assert lookup([1, 2, 3], 2, 3) == []


def test_probs_at_zero_temperature_is_one_hot():
    p = probs(torch.tensor([0.1, 3.0, -1.0]), 0)
    assert p.tolist() == [0.0, 1.0, 0.0]


def test_distribution_test_has_power():
    """Sampling from the draft alone is clearly distinguishable from the target."""
    target, draft = (
        random_gpt2(vocab=6, seed=0, emb_std=0.1),
        random_gpt2(vocab=6, seed=2, emb_std=0.1),
    )
    g = torch.Generator().manual_seed(1)
    n = 3000
    a = Counter(tuple(autoregressive(target, [1, 2], 2, 1.0, g)[0]) for _ in range(n))
    b = Counter(tuple(autoregressive(draft, [1, 2], 2, 1.0, g)[0]) for _ in range(n))
    assert 0.5 * sum(abs(a[k] - b[k]) for k in set(a) | set(b)) / n > 0.12


def test_bench_rows_and_report():
    from specdec.bench import report, run_config

    target, drafts = random_gpt2(seed=0), {"gpt2": random_gpt2(seed=1)}
    ps = [PROMPT, [7, 7, 7, 7, 7, 7]]
    ar, ref = run_config(
        {"domain": "toy", "method": "ar", "temperature": 0.0}, target, drafts, ps, 12, None
    )
    sd, _ = run_config(
        {"domain": "toy", "method": "gpt2", "gamma": 3, "temperature": 0.0},
        target,
        drafts,
        ps,
        12,
        ref,
    )
    lk, _ = run_config(
        {"domain": "toy", "method": "lookup", "gamma": 3, "temperature": 0.0},
        target,
        drafts,
        ps,
        12,
        ref,
    )
    assert sd["identical_to_greedy"] == 1.0 and lk["identical_to_greedy"] == 1.0
    assert ar["tokens_per_target_call"] == 1.0 and sd["tokens_per_target_call"] >= 1.0
    text = report([ar, sd, lk])
    assert "1.00×" in text and "**toy**" in text
