# Changelog

All notable changes to `echokv` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.3.0] — 2026-06-21

Adds two training-free knobs on top of the layer schedule: **quantization** (the bit
axis) and **value-aware front selection**.

### Added
- **`kv_bits`** on `echo_generate()`, `measure_memory()`, and `evaluate_perplexity()`
  (and `echokv ... --kv-bits` on the CLI): quantize the kept KV cache KIVI-style
  (per-channel keys / per-token values) to 8 or 4 bits. Orthogonal to the layer
  schedule, so the savings multiply — e.g. ~50% token saving × 4-bit ≈ **87% of the
  fp16 KV bytes**. Validated on Qwen2.5-0.5B/3B/7B (n=100 needle survival): 4-bit costs
  a near-constant ~0.05 of retention at any localization level, and echo's layer choice
  still beats a random choice by ~0.19 at matched memory. `stats` now reports `kv_bits`
  and `kv_saving_with_quant`.
- **`front_policy` / `front_budget`** on `echo_generate()`: choose *which* long-range
  keys a local layer keeps in its frozen front block (beyond the kernel sink) —
  `"positional"` (default, StreamingLLM-style), `"value_norm"` (largest ‖v‖, VATP-style),
  or `"value_subspace"` (greedy pivoted Gram–Schmidt spanning the value subspace,
  CurDKV-style). Reads cached values only (FlashAttention/SDPA-compatible — no attention
  matrix materialised).

### Notes / honesty
- Quantization is **fake-quant** (quantize→dequantize): quality is measured exactly (the
  model decodes through real quantization noise), and the reported byte saving is what a
  **packed int cache would realise**. Realising that byte reduction *live* (lower CUDA
  peak from quant) needs a packed-storage cache kernel — future work.
- Value-aware fronts are a *classifier-ties-frontier* feature: `value_subspace` matches
  CurDKV-style selection (it does not claim to beat it). Use it to spend a small extra
  front budget on the keys the model actually reads, not as a frontier-beating ranker.

## [0.2.0] — 2026-06-19

First public release.

### Added
- `calibrate()` / `profile_layers()` / `make_schedule()`: build an `EchoSchedule`
  (which layers are local) from one calibration batch of per-layer echo triviality.
- `echo_generate()`: generation with a real memory-bounded KV cache — local layers
  are pruned to `kernel + anchors + recent window` and kept that small every step.
- `echo_eager` custom attention that builds its own causal mask, so local layers
  can hold a shorter (non-contiguous) cache than global layers. Robust to the
  GPT-2 attention interface (derives the softmax scaling when the framework does
  not pass it) as well as Llama/Qwen/Gemma.
- Multimodal support: image-token **anchors** keep a vision-language model grounded
  on its image even when recency would evict it. `image_token_id`,
  `image_token_spans`, `image_anchor_scores`, and `echo_generate(inputs=...,
  image_token_id=..., image_budget=..., keep=...)`.
- `measure_memory()` (KV bytes + peak CUDA memory, full vs echo) and chunked
  prefill (`prefill_chunk`) for peak-memory reduction.
- `evaluate_perplexity()` quick full-vs-echo quality check.
- `echokv` CLI (`calibrate` | `generate` | `benchmark`) and an importable
  `echokv.benchmarks` reproduction suite.
- fp16 guard: `calibrate()` raises a clear error if the model overflows float16.

### Known limits
- Single (unpadded) sequence generation only.
- On hybrid local/global models the headline saving is vs an all-global baseline.
- `echo_eager` prefill does not replicate native per-layer sliding windows.
- Quality is measured by perplexity / teacher-forced retrieval proxies.
