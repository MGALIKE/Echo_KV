"""Reproducible benchmarks for echokv.

Every function here RUNS a measurement and returns a JSON-serialisable dict; the
``release/benchmarks`` driver scripts call these to produce the paper's figures and
logged results.  ``python -m echokv.benchmarks`` runs a small, locally-feasible
subset end to end (default model: a GPT-2 that fits any GPU/CPU) and writes JSON +
PNG/PDF figures, so the headline measurements are one command to reproduce.

Nothing here invents numbers: each result dict records the model, hardware, and the
raw measurement it came from.
"""
from __future__ import annotations

import json
import os
import platform
import time
from typing import List, Optional

import numpy as np
import torch

from . import (
    EchoSchedule,
    calibrate,
    echo_generate,
    evaluate_perplexity,
    make_schedule,
    measure_memory,
    profile_layers,
)
from .core import _set_attn_impl, get_attn_modules


# --------------------------------------------------------------------------- #
#  small helpers
# --------------------------------------------------------------------------- #
def hardware_tag() -> str:
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return f"{p.name} ({p.total_memory / 1e9:.1f} GB), torch {torch.__version__}"
    return f"CPU {platform.processor() or platform.machine()}, torch {torch.__version__}"


def load_model(name: str, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, attn_implementation="eager")
    if torch.cuda.is_available():
        model = model.cuda()
    return model.eval(), tok


def _filler_text(n_words: int = 600) -> str:
    base = ("The committee reviewed the quarterly report in detail, noting the "
            "steady growth in revenue, the modest rise in operating costs, and the "
            "need to invest in research over the coming year. ")
    out = []
    while len(" ".join(out).split()) < n_words:
        out.append(base)
    return " ".join(out)


# --------------------------------------------------------------------------- #
#  1. per-layer triviality
# --------------------------------------------------------------------------- #
def bench_triviality(model, tok, target_saving: float = 0.4,
                     window: int = 64, sink: int = 4, calib_texts=None,
                     calib_max_seqs: int = 24) -> dict:
    triv = profile_layers(model, tok, texts=calib_texts, window=window, sink=sink,
                          max_seqs=calib_max_seqs)
    sch = make_schedule(triv, len(triv), target_saving=target_saving,
                        window=window, sink=sink)
    return {"triviality": [float(x) for x in triv],
            "num_layers": len(triv),
            "local_layers": sorted(sch.local_layers),
            "min_triviality": 0.55, "calibration": "wikitext" if calib_texts else "default",
            "window": window, "sink": sink}


# --------------------------------------------------------------------------- #
#  2. KV-cache saving vs context length (MEASURED via the real bounded cache)
# --------------------------------------------------------------------------- #
def bench_saving_vs_context(model, tok, schedule: EchoSchedule,
                            lengths=(256, 512, 1024, 2048),
                            new_tokens: int = 8) -> dict:
    """Measure the realised KV-cache saving (actual cached key count, full vs echo)
    at a range of prompt lengths.  Saving grows toward n_local/num_layers."""
    text = _filler_text(2000)
    ids_all = tok(text, add_special_tokens=False)["input_ids"]
    rows = []
    asymptote = schedule.n_local / schedule.num_layers
    for T in lengths:
        if len(ids_all) < T:
            continue
        ids = torch.tensor([ids_all[:T]], device=next(model.parameters()).device)
        _t, stats = echo_generate(model, tok, ids, schedule,
                                  max_new_tokens=new_tokens, greedy=True,
                                  stop_on_eos=False)
        rows.append({"context": int(T), "final_len": stats["final_len"],
                     "kv_saving": float(stats["kv_saving"]),
                     "projected": float(schedule.saving(stats["final_len"]))})
    return {"asymptote": float(asymptote), "n_local": schedule.n_local,
            "num_layers": schedule.num_layers, "points": rows}


# --------------------------------------------------------------------------- #
#  3. quality (ppl gap) vs saving -- the Pareto frontier
# --------------------------------------------------------------------------- #
def bench_ppl_pareto(model, tok, ks: Optional[List[int]] = None,
                     window: int = 64, sink: int = 4, seq_len: int = 256,
                     ref_len: int = 1024, calib_texts=None,
                     calib_max_seqs: int = 24) -> dict:
    """Sweep the number of local layers; for each, report measured perplexity gap
    (echo vs full) and projected saving at ref_len.  Also runs an ANTI-ORACLE
    (localize the least-trivial layers) and a RANDOM control at each K.  Triviality
    and perplexity are both measured on `calib_texts` (a held-out corpus) when given;
    short-text perplexity is a weak proxy that does not stress long-range retrieval --
    the layer-choice signal shows up in the needle test, not here."""
    triv = profile_layers(model, tok, texts=calib_texts, window=window, sink=sink,
                          seq_len=seq_len, max_seqs=calib_max_seqs)
    L = len(triv)
    order = list(np.argsort(-triv))           # most trivial first
    anti = list(np.argsort(triv))             # least trivial first
    rng = np.random.default_rng(0)
    ks = ks or list(range(0, L + 1, max(1, L // 8)))
    full = None
    rows = []
    for k in ks:
        echo_sch = EchoSchedule([int(x) for x in order[:k]], L, window, sink)
        r = evaluate_perplexity(model, tok, echo_sch, texts=calib_texts, seq_len=seq_len)
        if full is None:
            full = r["full_ppl"]
        saving = echo_sch.saving(ref_len)
        row = {"k": int(k), "echo_ppl": r["echo_ppl"], "ppl_gap": r["ppl_gap"],
               "saving_at_ref": float(saving)}
        if 0 < k < L:
            anti_sch = EchoSchedule([int(x) for x in anti[:k]], L, window, sink)
            rand_sch = EchoSchedule(
                [int(x) for x in rng.choice(L, size=k, replace=False)], L, window, sink)
            row["anti_ppl"] = evaluate_perplexity(model, tok, anti_sch,
                                                  texts=calib_texts, seq_len=seq_len)["echo_ppl"]
            row["random_ppl"] = evaluate_perplexity(model, tok, rand_sch,
                                                    texts=calib_texts, seq_len=seq_len)["echo_ppl"]
        rows.append(row)
    return {"full_ppl": float(full), "num_layers": L, "ref_len": ref_len,
            "calibration": "wikitext" if calib_texts else "default",
            "window": window, "sink": sink, "points": rows}


# --------------------------------------------------------------------------- #
#  4. retrieval-needle survival under localization (capability, not just ppl)
# --------------------------------------------------------------------------- #
#  This is the X10 capability test in package form.  Perplexity is a proxy; the
#  capability long context exists for is RETRIEVAL.  We build fictional fact lines
#  with one needle placed deep, ask for it through the chat template, keep the
#  needles the FULL model retrieves, then localize K layers (echo / random / anti)
#  and measure how many of the model's own answers survive.  A localized layer is
#  simulated by a single masked forward (sink + sliding window), the same mechanism
#  as the schedule -- this genuinely changes the prefill that computes the answer,
#  which a real-cache generation would not (the answer is baked in at prefill).
_VOW, _CON = "aeiou", "bdfgklmnprstvz"
_ATTRS = ["capital", "currency", "river", "mountain", "language", "flower",
          "festival", "mineral", "bird", "harbor", "anthem", "motto", "gem",
          "tree", "dance", "dish", "season", "wind", "valley", "port"]


def _rname(rng, syl=None):
    syl = syl or int(rng.integers(2, 4))
    s = "".join(rng.choice(list(_CON)) + rng.choice(list(_VOW)) for _ in range(syl))
    return s[0].upper() + s[1:]


def _make_needle(rng, n_distract):
    n = n_distract + 1
    attrs = [str(a) for a in rng.choice(_ATTRS, size=n, replace=True)]
    ents = [_rname(rng) for _ in range(n)]
    if len(set(ents)) != n:
        return None
    vals = [_rname(rng, syl=3) for _ in range(n)]
    facts = [f"The {a} of {e} is {v}." for a, e, v in zip(attrs, ents, vals)]
    j = int(rng.integers(0, max(1, int(0.6 * n))))     # needle deep in the context
    ctx = " ".join(facts)
    if any(ctx.count(v) != 1 for v in vals):
        return None
    return dict(context=ctx, ent=ents[j], attr=attrs[j], gold=vals[j])


def _needle_ids(tok, item, dev):
    user = (f"Use only the context to answer.\n\nContext: {item['context']}\n\n"
            f"Question: What is the {item['attr']} of {item['ent']}? "
            f"Answer with one word.")
    text = tok.apply_chat_template([{"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)
    return torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device=dev)


def _band_mask(T, window, sink, dtype, dev):
    band = np.zeros((T, T), bool)
    for r in range(T):
        band[r, max(0, r - window + 1):r + 1] = True
    band[:, :sink] = True
    band &= np.tril(np.ones((T, T), bool))
    neg = torch.finfo(dtype).min
    return torch.where(torch.from_numpy(band), torch.tensor(0.0),
                       torch.tensor(float(neg))).to(dtype).view(1, 1, T, T).to(dev)


@torch.no_grad()
def bench_needle_survival(model, tok, ks: Optional[List[int]] = None,
                          window: int = 64, sink: int = 4, n_distract: int = 13,
                          n_trials: int = 120, n_random: int = 6,
                          seed: int = 0, calib_texts=None,
                          calib_max_seqs: int = 24) -> dict:
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    attn = get_attn_modules(model)
    triv = profile_layers(model, tok, texts=calib_texts, window=window, sink=sink,
                          max_seqs=calib_max_seqs)
    L = len(triv)
    order = list(np.argsort(-triv)); anti = list(np.argsort(triv))
    rng = np.random.default_rng(seed)
    ks = [k for k in (ks or [1, 2, 3, 4, 6, 8]) if k < L]

    def masked_argmax(ids, local_layers):
        handles = []
        if local_layers:
            add = _band_mask(ids.shape[1], window, sink, dtype, dev)

            def pre(mod, args, kwargs):
                kwargs["attention_mask"] = add
                return (args, kwargs)
            handles = [attn[l].register_forward_pre_hook(pre, with_kwargs=True)
                       for l in local_layers]
        was = _set_attn_impl(model, "eager")
        try:
            return int(model(ids, use_cache=False).logits[0, -1].argmax())
        finally:
            for h in handles:
                h.remove()
            _set_attn_impl(model, was)

    # keep needles the FULL model retrieves (greedy gen contains gold); record a0
    cands = []
    tried = 0
    while len(cands) < n_trials and tried < n_trials * 4:
        tried += 1
        it = _make_needle(rng, n_distract)
        if it is None:
            continue
        ids = _needle_ids(tok, it, dev)
        was = _set_attn_impl(model, "eager")
        g = model.generate(ids, max_new_tokens=8, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        _set_attn_impl(model, was)
        text = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        if it["gold"].lower() in text.lower():
            a0 = masked_argmax(ids, [])
            cands.append((ids, a0))

    def survival(local_layers):
        if not cands:
            return float("nan")
        return sum(masked_argmax(ids, local_layers) == a0
                   for ids, a0 in cands) / len(cands)

    rows = []
    for k in ks:
        echo_s = survival([int(x) for x in order[:k]])
        anti_s = survival([int(x) for x in anti[:k]])
        rand = [survival([int(x) for x in rng.permutation(L)[:k]])
                for _ in range(n_random)]
        rows.append({"k": int(k), "echo": echo_s, "anti": anti_s,
                     "random_mean": float(np.mean(rand)),
                     "random_min": float(np.min(rand)),
                     "random_max": float(np.max(rand))})
    return {"retrievable": len(cands), "num_layers": L, "window": window,
            "sink": sink, "n_distract": n_distract,
            "calibration": "wikitext" if calib_texts else "default", "points": rows}


# --------------------------------------------------------------------------- #
#  5. chunked-prefill peak memory (CUDA only)
# --------------------------------------------------------------------------- #
def bench_chunked_peak(model, tok, schedule: EchoSchedule, context: int = 4096,
                       prefill_chunk: int = 256, new_tokens: int = 16) -> dict:
    """Compare peak CUDA memory of naive full inference vs echo with chunked
    prefill on a long prompt.  Needs CUDA; on CPU returns the KV-byte saving only."""
    text = _filler_text(int(context / 0.6))
    ids_all = tok(text, add_special_tokens=False)["input_ids"][:context]
    ids = torch.tensor([ids_all], device=next(model.parameters()).device)
    out = measure_memory(model, tok, ids, schedule, max_new_tokens=new_tokens,
                         prefill_chunk=prefill_chunk)
    out["context"] = int(context)
    out["prefill_chunk"] = int(prefill_chunk)
    return out


# --------------------------------------------------------------------------- #
#  driver
# --------------------------------------------------------------------------- #
def _plot_triviality(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    triv = res["triviality"]; loc = set(res["local_layers"])
    colors = ["#d1495b" if i in loc else "#2e86ab" for i in range(len(triv))]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(triv)), triv, color=colors)
    ax.axhline(res["min_triviality"], ls="--", c="gray", lw=1,
               label=f"min triviality {res['min_triviality']}")
    ax.set_xlabel("layer"); ax.set_ylabel("echo triviality\n(sink+window mass)")
    ax.set_title(f"Per-layer echo triviality "
                 f"({len(loc)}/{len(triv)} localized, red)")
    ax.legend(fontsize=8); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=140)
    plt.close(fig)


def _plot_pareto(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = res["points"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sv = [p["saving_at_ref"] for p in pts]
    ax.plot(sv, [p["ppl_gap"] for p in pts], "o-", c="#2e86ab", label="echo")
    anti = [(p["saving_at_ref"], p["anti_ppl"] - res["full_ppl"])
            for p in pts if "anti_ppl" in p]
    rand = [(p["saving_at_ref"], p["random_ppl"] - res["full_ppl"])
            for p in pts if "random_ppl" in p]
    if anti:
        ax.plot([a for a, _ in anti], [b for _, b in anti], "s--", c="#d1495b",
                label="anti-oracle")
    if rand:
        ax.plot([a for a, _ in rand], [b for _, b in rand], "^:", c="gray",
                label="random")
    ax.set_xlabel(f"projected KV saving at T={res['ref_len']}")
    ax.set_ylabel("perplexity gap vs full")
    ax.set_title("Quality vs saving (lower-right is better)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=140)
    plt.close(fig)


def main(out: str = "echokv_bench", quick: bool = False,
         only: Optional[str] = None, model_name: Optional[str] = None) -> int:
    os.makedirs(out, exist_ok=True)
    name = model_name or os.environ.get("ECHOKV_BENCH_MODEL", "gpt2")
    which = set((only or "triviality,pareto,saving").split(","))
    print(f"[echokv.benchmarks] model={name} hardware={hardware_tag()}")
    model, tok = load_model(name)
    meta = {"model": name, "hardware": hardware_tag(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "echokv_version": __import__("echokv").__version__}
    results = {"_meta": meta}

    if "triviality" in which:
        r = bench_triviality(model, tok); results["triviality"] = r
        _plot_triviality(r, os.path.join(out, "fig_triviality"))
        print("  triviality:", r["local_layers"])
    if "pareto" in which:
        ks = None if not quick else [0, 2, 4, 6]
        r = bench_ppl_pareto(model, tok, ks=ks); results["ppl_pareto"] = r
        _plot_pareto(r, os.path.join(out, "fig_pareto"))
        print("  pareto full_ppl:", round(r["full_ppl"], 2))
    if "saving" in which and "triviality" in results:
        sch = calibrate(model, tok, target_saving=0.4)
        lengths = (256, 512) if quick else (256, 512, 1024, 2048)
        r = bench_saving_vs_context(model, tok, sch, lengths=lengths)
        results["saving_vs_context"] = r
        print("  saving@points:", [(p["context"], round(p["kv_saving"], 3))
                                    for p in r["points"]])

    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[echokv.benchmarks] wrote {os.path.join(out, 'results.json')} "
          f"and figures to {out}/")
    return 0


if __name__ == "__main__":          # pragma: no cover
    import sys
    raise SystemExit(main(*(sys.argv[1:2] or [])))
