"""Wall-clock benchmark: gpt2-xl (1.5B) as the target, gpt2 / gpt2-medium as drafts.

    python -m specdec.bench --out results --device cuda

There are two prompt domains, each with prompts of 128 tokens continued for 128 tokens:
- ``prose``: WikiText-103 test articles;
- ``code``: Python standard-library source, where repeated identifiers favour prompt lookup.

Every configuration decodes the same prompts. Recorded per configuration: tokens/s (with CUDA
synchronisation), speed-up over plain greedy decoding of the target, draft acceptance
rate, tokens committed per target forward, and, for greedy runs, the fraction of
sequences identical to plain greedy decoding. fp16 numerics differ slightly between
scoring 1 and γ + 1 tokens per forward, so near-ties can flip. Rows are appended to
``<out>/runs.jsonl`` as they finish, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sysconfig
import time

import torch

from .decode import autoregressive, speculative
from .model import GPT2


def prompts(tok, domain: str, n: int, length: int) -> list[list[int]]:
    if domain == "prose":
        from datasets import load_dataset

        text = "".join(
            load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"]
        )
        arts = [a for a in text.split("\n = ") if len(a) > 4000]
    else:
        root = sysconfig.get_paths()["stdlib"]
        files = sorted(glob.glob(os.path.join(root, "*.py")))
        arts = [open(f, encoding="utf-8", errors="ignore").read() for f in files]
        arts = [a for a in arts if len(a) > 8000]
    out = []
    step = max(1, len(arts) // n)
    for a in arts[::step][:n]:
        ids = tok(a[1000:6000])["input_ids"]
        out.append(ids[:length])
    return out


def run_config(cfg, target, drafts, ps, n_new, ref):
    gen = torch.Generator(device=target.wte.device).manual_seed(0)
    sync = torch.cuda.synchronize if target.wte.is_cuda else (lambda: None)
    sync()
    t0 = time.time()
    outs, stats = [], []
    for p in ps:
        if cfg["method"] == "ar":
            o, s = autoregressive(target, p, n_new, cfg["temperature"], gen)
        else:
            d = drafts.get(cfg["method"])  # None = prompt lookup
            o, s = speculative(target, d, p, n_new, cfg["gamma"], cfg["temperature"], gen)
        outs.append(o)
        stats.append(s)
    sync()
    sec = time.time() - t0
    prop = sum(s.proposed for s in stats)
    rounds = sum(len(s.per_round) for s in stats)
    row = {
        **cfg,
        "seconds": sec,
        "tok_per_s": n_new * len(ps) / sec,
        "acceptance": sum(s.accepted for s in stats) / prop if prop else None,
        "tokens_per_target_call": (
            (sum(sum(s.per_round) for s in stats) + rounds) / rounds if rounds else 1.0
        ),
    }
    if ref is not None and cfg["temperature"] == 0:
        row["identical_to_greedy"] = sum(a == b for a, b in zip(outs, ref, strict=True)) / len(ps)
    return row, outs


def report(rows) -> str:
    lines = []
    for dom in dict.fromkeys(r["domain"] for r in rows):
        rs = [r for r in rows if r["domain"] == dom]
        base = next(
            (r["tok_per_s"] for r in rs if r["method"] == "ar" and r["temperature"] == 0), None
        )
        lines += [
            f"**{dom}**",
            "",
            "| method | γ | T | tokens/s | speed-up | acceptance | tokens / target call "
            "| identical to greedy |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rs:
            sp = f"{r['tok_per_s'] / base:.2f}×" if base else "—"
            acc = "—" if r.get("acceptance") is None else f"{100 * r['acceptance']:.0f}%"
            ident = r.get("identical_to_greedy")
            ident = "—" if ident is None else f"{100 * ident:.0f}%"
            lines.append(
                f"| {r['method']} | {r.get('gamma', '—')} | {r['temperature']} | "
                f"{r['tok_per_s']:.1f} | {sp} | {acc} | "
                f"{r['tokens_per_target_call']:.2f} | {ident} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target", default="gpt2-xl")
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--new", type=int, default=128)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    target = GPT2.from_pretrained(a.target, a.device)
    drafts = {n: GPT2.from_pretrained(n, a.device) for n in ("gpt2", "gpt2-medium")}
    configs = [{"method": "ar", "temperature": 0.0}]
    for name in ("gpt2", "gpt2-medium"):
        configs += [{"method": name, "gamma": g, "temperature": 0.0} for g in (2, 4, 6, 8)]
    configs += [{"method": "lookup", "gamma": g, "temperature": 0.0} for g in (4, 8)]
    configs += [
        {"method": "ar", "temperature": 1.0},
        {"method": "gpt2", "gamma": 4, "temperature": 1.0},
        {"method": "lookup", "gamma": 4, "temperature": 1.0},
    ]
    path = os.path.join(a.out, "runs.jsonl")
    rows = [json.loads(s) for s in open(path)] if os.path.exists(path) else []
    keys = ("domain", "method", "gamma", "temperature")
    done = {json.dumps({k: r[k] for k in keys if k in r}, sort_keys=True) for r in rows}
    dev = torch.cuda.get_device_name()
    for dom in ("code", "prose"):
        todo = [
            c
            for c in ({"domain": dom, **c} for c in configs)
            if json.dumps(c, sort_keys=True) not in done
        ]
        if not todo:
            continue
        ps = prompts(tok, dom, a.n_prompts, a.prompt_len)
        autoregressive(target, ps[0], 8)  # warm-up
        speculative(target, drafts["gpt2"], ps[0], 8, 4)
        ref = None
        if any(c["temperature"] == 0 for c in todo):  # greedy reference, rerun on resume
            ref_cfg = {"domain": dom, **configs[0]}
            row, ref = run_config(ref_cfg, target, drafts, ps, a.new, None)
            todo = [c for c in todo if c != ref_cfg]
            if json.dumps(ref_cfg, sort_keys=True) not in done:
                todo.insert(0, ("ref", row))
        for cfg in todo:
            if isinstance(cfg, tuple):
                row = cfg[1]
            else:
                row, _ = run_config(cfg, target, drafts, ps, a.new, ref)
            key = json.dumps({k: row[k] for k in keys if k in row}, sort_keys=True)
            row["device"] = dev
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            rows.append(row)
            done.add(key)
            log(json.dumps(row))
    text = report(rows)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
