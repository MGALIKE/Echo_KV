"""echokv core: calibration, schedule, echo attention, and a memory-bounded
echo KV cache for training-free inference compression.

Mechanism.  A decoder layer whose heads spend (almost) all their attention on the
sink (kernel) plus a recent window has a *trivial* echo space -- it retrieves
nothing mid-range.  We rank layers by that trivial mass on one calibration batch,
make the most-trivial K layers LOCAL (their KV cache is pruned to a small fixed
budget and kept that small as generation proceeds), and leave the rest GLOBAL
(full cache).  The local layers' cache stops growing with context, which is the
memory saving.

What a local layer keeps is a *front block* plus a *recent window*:

    front (frozen)        = kernel (sink) + protected anchors
    window (sliding)      = the most recent `window` tokens

For a text-only model the anchor set is empty, so the front is just the sink and
this reduces exactly to keeping ``[:sink] + [-window:]`` -- the original behaviour.
For a *multimodal* model the anchors are a pooled subset of the image tokens, so a
local layer keeps the image reachable on a fixed budget instead of letting recency
evict it.  This is the fix that makes the compressor correct on vision-language
models, where the image tokens sit at the *start* of the sequence and dominate the
cache: a pure-recency window would drop them and blind the model to the picture.

The only framework subtlety is that stock transformers builds a single attention
mask shared across layers and assumes every layer has the same KV length.  Once
local layers hold a shorter cache that breaks.  We register a small custom
attention ("echo_eager") that builds its own causal mask, which is exactly correct
for single-token decode (every cached key is a valid causal target) and lets pruned
local layers hold a shorter -- and non-contiguous -- cache than the global ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

__version__ = "0.2.0"

try:
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import repeat_kv
except Exception:                                   # pragma: no cover
    AttentionInterface = None
    def repeat_kv(x, n):
        if n == 1:
            return x
        b, h, s, d = x.shape
        return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


# --------------------------------------------------------------------------- #
#  model-structure helpers (handle nested multimodal models, e.g. Gemma / VLMs)
# --------------------------------------------------------------------------- #
def _attn_of(block):
    for name in ("self_attn", "attn", "attention"):
        a = getattr(block, name, None)
        if a is not None:
            return a
    return None


def _find_decoder_layers(model):
    """Return the ModuleList of decoder blocks, descending through the common
    (possibly nested, multimodal) attribute paths used by Llama/Qwen/Mistral, by
    Gemma (text decoder under a multimodal wrapper) and by VLMs such as Idefics3 /
    SmolVLM (text decoder under ``model.text_model``)."""
    import torch.nn as nn
    paths = [
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "text_model", "layers"),
        ("text_model", "layers"),
    ]
    for path in paths:
        obj, ok = model, True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if (ok and isinstance(obj, nn.ModuleList) and len(obj) > 0
                and _attn_of(obj[0]) is not None):
            return obj
    # generic fallback: first ModuleList whose blocks expose an attention submodule
    hit = []

    def walk(m):
        for child in m.children():
            if (isinstance(child, nn.ModuleList) and len(child) > 0
                    and _attn_of(child[0]) is not None):
                hit.append(child)
                return
            walk(child)
    walk(model)
    return hit[0] if hit else None


def _text_config(model):
    """The text sub-config for multimodal models (Gemma, VLMs), else the top config."""
    cfg = model.config
    for attr in ("text_config", "language_config"):
        sub = getattr(cfg, attr, None)
        if sub is not None and hasattr(sub, "num_attention_heads"):
            return sub
    return cfg


def _set_attn_impl(model, name):
    """Set the attention implementation on the top config AND any text sub-config
    (multimodal models read the sub-config), returning the previous top value."""
    prev = getattr(model.config, "_attn_implementation", "eager")
    model.config._attn_implementation = name
    sub = _text_config(model)
    if sub is not model.config:
        try:
            sub._attn_implementation = name
        except Exception:
            pass
    return prev


def get_attn_modules(model):
    """Attention submodules, one per decoder layer, across common architectures
    including nested multimodal models (Gemma, Idefics3/SmolVLM, ...)."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [b.attn for b in model.transformer.h]          # GPT-2
    if hasattr(model, "gpt_neox"):
        return [b.attention for b in model.gpt_neox.layers]    # Pythia / NeoX
    layers = _find_decoder_layers(model)                       # Llama/Qwen/Gemma/VLM...
    if layers is not None:
        return [_attn_of(b) for b in layers]
    raise RuntimeError("echokv: unknown architecture (no decoder layers found)")


def image_token_id(model, processor=None):
    """Best-effort lookup of the image placeholder token id for a VLM."""
    for src in (getattr(model, "config", None), processor,
                getattr(processor, "tokenizer", None)):
        if src is None:
            continue
        for attr in ("image_token_id", "image_token_index"):
            v = getattr(src, attr, None)
            if isinstance(v, int):
                return v
    return None


def image_token_spans(input_ids, img_id) -> List[Tuple[int, int]]:
    """Contiguous [start, end) spans of image-placeholder tokens in a 1-D id row."""
    if img_id is None:
        return []
    ids = input_ids
    if hasattr(ids, "tolist"):
        if ids.dim() > 1:
            ids = ids[0]
        ids = ids.tolist()
    spans, start = [], None
    for i, t in enumerate(ids):
        if t == img_id and start is None:
            start = i
        elif t != img_id and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(ids)))
    return spans


def _pooled_anchor_positions(spans: Sequence[Tuple[int, int]], budget: int,
                             scores=None) -> List[int]:
    """Absolute image-token positions to keep as anchors on local layers.

    With `scores` (a {position: float} importance map) keep the top-`budget`
    most-attended image tokens -- *echo-scored* selection, which spends the budget on
    the tokens the model actually reads.  Without it, keep evenly-spaced tokens
    (*uniform*).  A falsy or oversized budget keeps all image tokens."""
    allpos = [p for s, e in spans for p in range(s, e)]
    if not allpos:
        return []
    if not budget or budget >= len(allpos):
        return allpos
    if scores is not None:
        ranked = sorted(allpos, key=lambda p: -scores.get(p, 0.0))
        return sorted(ranked[:budget])
    pos: List[int] = []                      # uniform stride, proportional per span
    total = len(allpos)
    for s, e in spans:
        n = e - s
        take = min(n, max(1, int(round(budget * n / total))))
        pos.extend(int(round(x)) for x in np.linspace(s, e - 1, take))
    return sorted(set(pos))


def _device(model):
    return next(model.parameters()).device


@torch.no_grad()
def image_anchor_scores(model, inputs, img_id, query_from: Optional[int] = None):
    """Training-free importance of each image token: the attention it *receives*,
    summed over heads, query positions and layers, on one eager forward of the
    multimodal prompt.  The most-attended image tokens are the model's own visual
    anchors -- keep those (echo-scored selection) instead of a blind uniform stride.

    Returns {position: score} over image-token positions.  `query_from` restricts the
    querying rows to positions >= it (default: all rows)."""
    dev = _device(model)
    kw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    ids = kw["input_ids"]
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
        kw["input_ids"] = ids
    spans = image_token_spans(ids[0], img_id)
    img_pos = [p for s, e in spans for p in range(s, e)]
    if not img_pos:
        return {}
    T = ids.shape[1]
    q0 = 0 if query_from is None else max(0, query_from)
    was = _set_attn_impl(model, "eager")
    try:
        out = model(**kw, output_attentions=True, use_cache=False)
        score = torch.zeros(T, dtype=torch.float32, device=dev)
        for a in out.attentions:                 # (B, H, Tq, Tk)
            score += a[0].float()[:, q0:, :].sum(dim=(0, 1))    # over heads + queries
        del out
    finally:
        _set_attn_impl(model, was)
    score = score.cpu().numpy()
    return {p: float(score[p]) for p in img_pos}


# --------------------------------------------------------------------------- #
#  calibration: per-layer echo triviality
# --------------------------------------------------------------------------- #
DEFAULT_CALIB = [
    "The history of science is the study of how the natural world has been "
    "understood and explained across cultures and centuries, from early "
    "astronomy and medicine to modern physics and biology.",
    "In economics, supply and demand describe how the price of a good in a "
    "competitive market is determined by the interaction of buyers and sellers, "
    "reaching equilibrium where the quantity supplied equals the quantity demanded.",
    "A river begins at its source in high ground, gathers water from tributaries "
    "as it flows downhill, carves valleys over long periods of time, and finally "
    "reaches its mouth where it empties into a sea or a lake.",
    "Software engineering concerns the design, construction, testing and "
    "maintenance of programs; good practice favours small, well-named functions, "
    "clear interfaces, automated tests, and code that reads the way it runs.",
]


@torch.no_grad()
def profile_layers(model, tokenizer, texts: Optional[List[str]] = None,
                   window: int = 64, sink: int = 4, seq_len: int = 256,
                   eval_from: Optional[int] = None, max_seqs: int = 8):
    """Return per-layer echo triviality (mean sink+window attention mass over
    query positions), shape (num_layers,), higher = safer to make local."""
    texts = texts or DEFAULT_CALIB
    dev = _device(model)
    # pack calibration text into fixed-length sequences
    ids_all = []
    for t in texts:
        ids_all.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
    bos = (tokenizer.bos_token_id if tokenizer.bos_token_id is not None
           else tokenizer.eos_token_id)
    seqs = []
    i = 0
    while i + seq_len - 1 <= len(ids_all) and len(seqs) < max_seqs:
        seqs.append([bos] + ids_all[i:i + seq_len - 1])
        i += seq_len - 1
    if not seqs:                                   # short calibration text: pad one
        s = ([bos] + ids_all)[:seq_len]
        seqs = [s + [bos] * (seq_len - len(s))]
    tokens = torch.tensor(seqs, dtype=torch.long, device=dev)
    T = tokens.shape[1]
    ef = eval_from if eval_from is not None else T // 2
    qsel = list(range(ef, T))

    was = _set_attn_impl(model, "eager")
    acc, n = None, 0
    for b in range(0, tokens.shape[0], 2):
        out = model(tokens[b:b + 2], output_attentions=True, use_cache=False)
        for l, a in enumerate(out.attentions):     # a: (B,H,T,T)
            A = a.float()
            s = A[:, :, qsel, :sink].sum(-1)        # sink mass (first `sink` keys)
            wm = torch.zeros_like(s)
            for qi, q in enumerate(qsel):
                lo = max(0, q - window + 1)
                wm[:, :, qi] = A[:, :, q, lo:q + 1].sum(-1)
            triv = (s + wm).clamp(0, 1).mean().item()
            if acc is None:
                acc = [0.0] * len(out.attentions)
            acc[l] += triv * (tokens[b:b + 2].shape[0])
        n += tokens[b:b + 2].shape[0]
        del out
    _set_attn_impl(model, was)
    return np.array(acc) / max(n, 1)


# --------------------------------------------------------------------------- #
#  schedule
# --------------------------------------------------------------------------- #
@dataclass
class EchoSchedule:
    local_layers: List[int]
    num_layers: int
    window: int = 64
    sink: int = 4
    triviality: Optional[List[float]] = None
    calib_seq_len: int = 256

    @property
    def n_local(self) -> int:
        return len(self.local_layers)

    def saving(self, seq_len: int, anchors: int = 0) -> float:
        """Fraction of the per-token KV cache removed at this context length.  Pass
        `anchors` = number of protected image tokens kept on each local layer
        (multimodal); 0 for text."""
        kept_local = self.sink + self.window + anchors
        if seq_len <= kept_local:
            return 0.0
        return self.n_local * (seq_len - kept_local) / (self.num_layers * seq_len)

    def describe(self) -> str:
        return (f"EchoSchedule: {self.n_local}/{self.num_layers} layers local "
                f"(sink={self.sink}, window={self.window}); "
                f"local layers={sorted(self.local_layers)}")


def make_schedule(triviality, num_layers: int, target_saving: Optional[float] = 0.4,
                  n_local: Optional[int] = None, window: int = 64, sink: int = 4,
                  ref_len: int = 1024, min_triviality: float = 0.55) -> EchoSchedule:
    """Pick the local layers.  Either fix `n_local`, or choose the smallest number
    of the most-trivial layers whose projected saving at `ref_len` reaches
    `target_saving` (never localizing layers below `min_triviality`)."""
    triv = np.asarray(triviality, float)
    order = list(np.argsort(-triv))                # most trivial first
    eligible = [int(l) for l in order if triv[l] >= min_triviality]
    if n_local is not None:
        chosen = order[:n_local]
    else:
        per_layer = (ref_len - (sink + window)) / (num_layers * ref_len)
        need = int(math.ceil((target_saving or 0.0) / max(per_layer, 1e-9)))
        need = min(need, len(eligible))
        chosen = eligible[:need]
    return EchoSchedule(local_layers=[int(x) for x in chosen], num_layers=num_layers,
                        window=window, sink=sink, triviality=triv.tolist())


def calibrate(model, tokenizer, texts: Optional[List[str]] = None,
              target_saving: float = 0.4, n_local: Optional[int] = None,
              window: int = 64, sink: int = 4, seq_len: int = 256) -> EchoSchedule:
    """One-call: profile the model and return an EchoSchedule."""
    L = len(get_attn_modules(model))
    triv = profile_layers(model, tokenizer, texts, window=window, sink=sink,
                          seq_len=seq_len)
    if not np.isfinite(triv).all():
        raise RuntimeError(
            "echokv: non-finite attention during calibration -- the model is "
            "producing NaN/Inf, which means it is overflowing in float16. Load it "
            "in bfloat16 instead: AutoModelForCausalLM.from_pretrained(name, "
            "dtype=torch.bfloat16, attn_implementation='eager'). Most modern models "
            "(Qwen, Llama, Gemma, VLMs) are trained in bf16 and overflow in fp16. "
            "For a multimodal checkpoint, load the text path with "
            "AutoModelForCausalLM / AutoModelForImageTextToText and pass "
            "attn_implementation='eager'.")
    sch = make_schedule(triv, L, target_saving=target_saving, n_local=n_local,
                        window=window, sink=sink)
    sch.calib_seq_len = seq_len
    return sch


# --------------------------------------------------------------------------- #
#  echo attention (mask-width guard so local layers can hold a shorter cache)
# --------------------------------------------------------------------------- #
_INSTALLED = False


def _echo_eager(module, query, key, value, attention_mask=None, scaling=None,
                dropout: float = 0.0, **kwargs):
    """Eager attention that builds its own causal mask, so correctness does not
    depend on the shared per-forward mask the framework would otherwise impose at a
    single KV length across all layers (which breaks once a local layer is pruned).

    Prefill (q>1): standard causal masking, computed from query/key lengths.
    Decode  (q==1): every cached key is a valid causal target -> attend to all,
    which is exactly what lets pruned local layers hold a shorter (and possibly
    non-contiguous) cache.  Assumes a single (unpadded) sequence, which is how
    `echo_generate` runs.

    `scaling` is the softmax temperature.  Llama/Qwen/Gemma pass it explicitly;
    GPT-2 (and any model whose attention interface omits it) does not, so we derive
    the GPT-2 convention from the module: 1/sqrt(head_dim), optionally disabled
    (`scale_attn_weights=False`) or divided by the layer index
    (`scale_attn_by_inverse_layer_idx`)."""
    k = repeat_kv(key, getattr(module, "num_key_value_groups", 1))
    v = repeat_kv(value, getattr(module, "num_key_value_groups", 1))
    q_len, k_len = query.shape[2], k.shape[2]
    if scaling is None:
        scaling = (float(value.shape[-1]) ** -0.5
                   if getattr(module, "scale_attn_weights", True) else 1.0)
        if getattr(module, "scale_attn_by_inverse_layer_idx", False):
            scaling = scaling / float(getattr(module, "layer_idx", 0) + 1)
    aw = torch.matmul(query, k.transpose(2, 3)) * scaling
    softcap = getattr(module, "attn_logit_softcapping", None)
    if softcap is None:
        softcap = getattr(getattr(module, "config", None),
                          "attn_logit_softcapping", None)
    if softcap:                                            # Gemma-style soft cap
        aw = torch.tanh(aw / softcap) * softcap
    if q_len > 1:
        dev = aw.device
        # query row i has absolute position (k_len - q_len + i); attends to col<=pos
        rows = torch.arange(q_len, device=dev).view(-1, 1) + (k_len - q_len)
        cols = torch.arange(k_len, device=dev).view(1, -1)
        causal = cols <= rows                              # (q_len, k_len)
        aw = aw.masked_fill(~causal[None, None], float("-inf"))
    aw = torch.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout:
        aw = torch.nn.functional.dropout(aw, p=dropout, training=False)
    out = torch.matmul(aw, v).transpose(1, 2).contiguous()
    return out, aw


def install_echo_attention(model):
    """Register the echo attention and switch the model to it. Returns the previous
    implementation name so it can be restored."""
    global _INSTALLED
    if AttentionInterface is None:
        raise RuntimeError("echokv needs transformers with AttentionInterface")
    if not _INSTALLED:
        AttentionInterface.register("echo_eager", _echo_eager)
        _INSTALLED = True
    prev = _set_attn_impl(model, "echo_eager")
    return prev


def restore_attention(model, prev: str = "eager"):
    _set_attn_impl(model, prev)


# --------------------------------------------------------------------------- #
#  memory-bounded cache pruning
# --------------------------------------------------------------------------- #
def _cache_handles(past):
    """Return (layers, key_cache, value_cache, n) handling both the new unified
    Cache API (`.layers[i].keys/.values`) and the older DynamicCache
    (`.key_cache[i]`/`.value_cache[i]`)."""
    layers = getattr(past, "layers", None)
    if layers is not None:
        return layers, None, None, len(layers)
    kc = getattr(past, "key_cache", None)
    vc = getattr(past, "value_cache", None)
    n = len(kc) if kc is not None else 0
    return None, kc, vc, n


class _Pruner:
    """Bounds the KV cache of local layers to a frozen *front* block (kernel sink +
    protected anchors) plus a sliding *recent window*.

    With no anchors the front is just the sink, so this keeps ``[:sink] + [-window:]``
    -- the original text behaviour.  With anchors (pooled image tokens) the front
    also protects those positions, so a local layer stays able to read the image
    even after recency would have evicted it.  Anchors are absolute positions in the
    original prompt; the cached keys carry their original RoPE phase, so gathering a
    non-contiguous subset is positionally correct."""

    def __init__(self, local_layers, sink, window, anchors=None):
        self.local = list(local_layers)
        self.sink = sink
        self.window = window
        self.anchors = sorted(set(int(a) for a in (anchors or [])))
        self.front = {}                 # layer -> number of frozen front rows

    def step(self, past):
        layers, kc, vc, n = _cache_handles(past)
        for li in self.local:
            if li >= n:                 # cache shorter than expected -> skip safely
                continue
            if layers is not None:
                lyr = layers[li]; k, v = lyr.keys, lyr.values
            else:
                lyr = None; k, v = kc[li], vc[li]
            if k is None:
                continue
            L = k.shape[2]
            if li not in self.front:    # first prune: choose the frozen front block
                # front = kernel sink + ALL protected anchors (wherever they sit);
                # reorder them to the start so they are permanently kept.  Attention
                # is permutation-invariant over keys and RoPE phase is baked into the
                # cached keys, so reordering rows is exact.
                front = sorted(set(range(min(self.sink, L)))
                               | set(a for a in self.anchors if 0 <= a < L))
                fset = set(front)
                tail = [p for p in range(max(0, L - self.window), L) if p not in fset]
                self.front[li] = len(front)
                idx = front + tail
                if len(idx) >= L and idx == list(range(L)):
                    continue            # already ordered, nothing to drop
            else:
                f = self.front[li]
                if L <= f + self.window:
                    continue
                idx = list(range(f)) + list(range(L - self.window, L))
            sel = torch.tensor(idx, device=k.device)
            nk = k.index_select(2, sel).contiguous()
            nv = v.index_select(2, sel).contiguous()
            if lyr is not None:
                lyr.keys, lyr.values = nk, nv
            else:
                kc[li], vc[li] = nk, nv


def _cache_key_count(past, num_layers):
    """Total cached key positions summed across layers (actual, post-pruning)."""
    layers, kc, vc, n = _cache_handles(past)
    total = 0
    for li in range(n):
        k = layers[li].keys if layers is not None else kc[li]
        if k is not None:
            total += k.shape[2]
    return total


def _forward_last(model, **kw):
    """Forward that only materializes the last position's logits when supported
    (avoids an O(T*vocab) logits tensor at the prefill peak)."""
    try:
        return model(**kw, logits_to_keep=1)
    except TypeError:
        return model(**kw)


# --------------------------------------------------------------------------- #
#  memory-bounded generation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def echo_generate(model, tokenizer, prompt, schedule: EchoSchedule,
                  max_new_tokens: int = 128, greedy: bool = True,
                  temperature: float = 0.7, top_k: int = 50,
                  stop_on_eos: bool = True, prefill_chunk: Optional[int] = None,
                  inputs: Optional[dict] = None, image_token_id: Optional[int] = None,
                  image_budget: int = 64, protect_spans=None, keep: str = "anchor",
                  force_decode: Optional[Sequence[int]] = None, anchor_scores=None):
    """Greedy/sampled generation with the echo memory-bounded cache.  Local layers
    are pruned to a fixed budget after prefill and kept that small every step, so
    their KV cache does not grow with context.  Returns (text, stats).

    Multimodal:  pass `inputs` (a processor dict with input_ids + pixel_values, etc.)
    to prefill a vision-language model.  Give `image_token_id` (or `protect_spans`)
    so the image tokens are protected as anchors on local layers -- otherwise a
    pure-recency window evicts the image and the model goes blind to it.  Set
    `keep="recency"` to reproduce that (broken) baseline for ablation.

    `prefill_chunk`:  text-only peak control -- encode the prompt in chunks, pruning
    after each (attention scratch O(chunk*T) not O(T^2)).  Disabled automatically
    when image anchors are protected (anchors need the whole prompt at once)."""
    dev = _device(model)
    # ---- assemble prefill inputs ----
    if inputs is not None:
        prefill_kw = {k: (v.to(dev) if torch.is_tensor(v) else v)
                      for k, v in inputs.items()}
        ids = prefill_kw["input_ids"]
    else:
        if isinstance(prompt, str):
            ids = tokenizer(prompt, return_tensors="pt").to(dev).input_ids
        else:
            ids = prompt.to(dev)
        prefill_kw = {"input_ids": ids}
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)

    # ---- choose protected anchors (image tokens) ----
    anchors: List[int] = []
    n_img = 0
    if keep == "anchor":
        spans = protect_spans
        if spans is None and image_token_id is not None:
            spans = image_token_spans(ids[0], image_token_id)
        if spans:
            n_img = sum(e - s for s, e in spans)
            anchors = _pooled_anchor_positions(spans, image_budget, scores=anchor_scores)
    pruner = _Pruner(schedule.local_layers, schedule.sink, schedule.window, anchors)

    prev_impl = install_echo_attention(model)
    try:
        T0 = ids.shape[1]
        use_chunk = bool(prefill_chunk) and T0 > prefill_chunk and not anchors \
            and inputs is None
        if use_chunk:
            past = None
            for s in range(0, T0, prefill_chunk):
                chunk = ids[:, s:s + prefill_chunk]
                cpos = torch.arange(s, s + chunk.shape[1], device=dev)
                out = _forward_last(model, input_ids=chunk, past_key_values=past,
                                    use_cache=True, cache_position=cpos)
                past = out.past_key_values
                pruner.step(past)
            logits = out.logits[:, -1]
        else:
            out = _forward_last(model, **prefill_kw, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1]
            pruner.step(past)
        produced, true_len = [], T0
        eos = tokenizer.eos_token_id
        # teacher-forced decode prefix: feed these tokens through the *pruned* cache
        # (advancing the recent window) before sampling.  Used by the multimodal
        # needle to push the answer far enough into decode that recency has evicted
        # the image, so the model must rely on global layers + anchors.
        if force_decode:
            for tid in force_decode:
                nxt = torch.tensor([[int(tid)]], device=dev)
                out = model(input_ids=nxt, past_key_values=past, use_cache=True,
                            cache_position=torch.tensor([true_len], device=dev))
                logits = out.logits[:, -1]
                true_len += 1
                pruner.step(past)
        for _ in range(max_new_tokens):
            if greedy:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                lg = logits / max(temperature, 1e-5)
                if top_k:
                    v, _ = torch.topk(lg, top_k)
                    lg[lg < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(torch.softmax(lg, -1), 1)
            produced.append(int(nxt))
            if stop_on_eos and eos is not None and int(nxt) == eos:
                break
            out = model(input_ids=nxt, past_key_values=past, use_cache=True,
                        cache_position=torch.tensor([true_len], device=dev))
            logits = out.logits[:, -1]
            true_len += 1
            pruner.step(past)
        text = tokenizer.decode(produced, skip_special_tokens=True)
        # cache accounting (actual cache sizes, so anchors are counted)
        echo_keys = _cache_key_count(past, schedule.num_layers)
        full_keys = schedule.num_layers * true_len
        stats = {"final_len": true_len, "new_tokens": len(produced),
                 "n_anchor": len(anchors), "n_image": n_img, "keep": keep,
                 "anchor_select": "score" if anchor_scores is not None else "uniform",
                 "full_cache_keys": full_keys, "echo_cache_keys": echo_keys,
                 "kv_saving": 1.0 - echo_keys / max(full_keys, 1)}
        return text, stats
    finally:
        restore_attention(model, prev_impl)


def _kv_bytes_per_pos(model):
    cfg = _text_config(model)
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    elt = next(model.parameters()).element_size()
    return n_kv * head_dim * 2 * elt           # key + value


@torch.no_grad()
def measure_memory(model, tokenizer, prompt, schedule: EchoSchedule,
                   max_new_tokens: int = 256, prefill_chunk: Optional[int] = None,
                   inputs: Optional[dict] = None, image_token_id: Optional[int] = None,
                   image_budget: int = 64, keep: str = "anchor"):
    """Generate (naive full vs echo) and report KV-cache size and peak CUDA memory.

    The `full` baseline is naive inference (full cache, single-shot prefill).  The
    `echo` run uses the memory-bounded cache and, if `prefill_chunk` is set (text
    only), chunked prefill."""
    empty = EchoSchedule([], schedule.num_layers, schedule.window, schedule.sink)
    bpp = _kv_bytes_per_pos(model)              # bytes per cached position (per layer)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    _t, s_full = echo_generate(model, tokenizer, prompt, empty, max_new_tokens,
                               inputs=inputs)
    full_peak = torch.cuda.max_memory_allocated() if cuda else 0
    if cuda:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    _t, s_echo = echo_generate(model, tokenizer, prompt, schedule, max_new_tokens,
                               prefill_chunk=prefill_chunk, inputs=inputs,
                               image_token_id=image_token_id,
                               image_budget=image_budget, keep=keep)
    echo_peak = torch.cuda.max_memory_allocated() if cuda else 0
    full_kv_mb = s_full["echo_cache_keys"] * bpp / 1e6
    echo_kv_mb = s_echo["echo_cache_keys"] * bpp / 1e6
    out = {"final_len": s_echo["final_len"],
           "full_kv_mb": full_kv_mb, "echo_kv_mb": echo_kv_mb,
           "kv_saving": 1 - echo_kv_mb / max(full_kv_mb, 1e-9)}
    if cuda:
        out.update(full_peak_mb=full_peak / 1e6, echo_peak_mb=echo_peak / 1e6,
                   peak_reduction=1 - echo_peak / max(full_peak, 1))
    return out


def kv_saving_report(schedule: EchoSchedule, seq_len: int, anchors: int = 0) -> str:
    s = schedule.saving(seq_len, anchors=anchors)
    return (f"{schedule.describe()}\n"
            f"  projected KV-cache saving at T={seq_len}: {100*s:.1f}%  "
            f"(asymptote {100*schedule.n_local/schedule.num_layers:.1f}% as T->inf)")


# --------------------------------------------------------------------------- #
#  optional: quick quality check by masked simulation (no custom cache)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_perplexity(model, tokenizer, schedule: EchoSchedule,
                        texts: Optional[List[str]] = None, seq_len: int = 256):
    """Perplexity of the full model vs the echo schedule (simulated by masking the
    local layers to sink+window), on held-out text.  A fast quality sanity check
    that needs no generation."""
    texts = texts or DEFAULT_CALIB
    dev = _device(model)
    ids_all = []
    for t in texts:
        ids_all.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
    bos = (tokenizer.bos_token_id if tokenizer.bos_token_id is not None
           else tokenizer.eos_token_id)
    if len(ids_all) < 32:
        return {"note": "not enough text for a perplexity estimate (need >=32 tokens)"}
    seq_len = max(32, min(seq_len, len(ids_all) + 1))   # adapt to short text
    seqs = []
    i = 0
    while i + seq_len - 1 <= len(ids_all) and len(seqs) < 8:
        seqs.append([bos] + ids_all[i:i + seq_len - 1]); i += seq_len - 1
    if not seqs:
        seqs = [[bos] + ids_all[:seq_len - 1]]
    tokens = torch.tensor(seqs, dtype=torch.long, device=dev)
    T = tokens.shape[1]; ef = max(1, T // 2)
    attn_mods = get_attn_modules(model)
    dtype = next(model.parameters()).dtype
    neg = torch.finfo(dtype).min

    band = np.zeros((T, T), bool)
    for r in range(T):
        band[r, max(0, r - schedule.window + 1):r + 1] = True
    band[:, :schedule.sink] = True
    band &= np.tril(np.ones((T, T), bool))
    add = torch.where(torch.from_numpy(band), torch.tensor(0.0),
                      torch.tensor(float(neg))).to(dtype).view(1, 1, T, T).to(dev)

    prev = _set_attn_impl(model, "eager")

    def ppl(local_layers):
        def mk():
            def pre(mod, args, kwargs):
                kwargs["attention_mask"] = add
                return (args, kwargs)
            return pre
        handles = [attn_mods[l].register_forward_pre_hook(mk(), with_kwargs=True)
                   for l in local_layers]
        try:
            tot, cnt = 0.0, 0
            for b in range(0, tokens.shape[0], 2):
                tk = tokens[b:b + 2]
                lg = model(tk, use_cache=False).logits[:, ef - 1:-1]
                tg = tk[:, ef:]
                nll = torch.nn.functional.cross_entropy(
                    lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                    reduction="sum")
                tot += float(nll); cnt += tg.numel()
            return math.exp(tot / cnt)
        finally:
            for h in handles:
                h.remove()
    full = ppl([])
    echo = ppl(schedule.local_layers)
    _set_attn_impl(model, prev)
    return {"full_ppl": full, "echo_ppl": echo, "ppl_gap": echo - full,
            "kv_saving_at_len": schedule.saving(T)}
