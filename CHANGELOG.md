# Changelog

All notable changes to `echokv` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

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
