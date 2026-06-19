# echokv benchmarks

Every number here comes from a script in this directory that writes its raw
measurements to `results/<name>.json` and its figure to `figures/<name>.{png,pdf}`.
Each row carries a **hardware label**:

- **local 4GB** — measured for this release on an RTX 3050 Ti (4.3 GB), PyTorch 2.6,
  transformers 5.3; bfloat16 for modern models, float32 for GPT-2. Reproduce with the
  command in the last column.
- **Colab/L4** — from notebook runs on a Colab L4 (24 GB), bfloat16, for models the
  4 GB card cannot hold. **Not** re-measured here. The underlying Colab notebooks live
  in the authors' research repository and are **not bundled** in this release; reproduce
  these rows by running the same `echokv` API (see the README quickstart) on a ≥16 GB GPU.

Calibration for the layer-locality experiments uses 24 sequences of 256 tokens from
WikiText-2 (window `w=48`, sink `s=4`), the X9/X10/X11 protocol. The deployable tool
default is `window=64` (more conservative).

## Reproduce

```bash
pip install -e ..
python run_text_benchmarks.py        # triviality, pareto, saving-vs-context, needle, peak
python run_multimodal_benchmark.py   # SmolVLM recency-vs-anchor + ceiling gate + uniform-vs-scored
python regenerate_x11_figure.py      # long-context scaling (from the logged X11 run)
python make_overlay_figure.py        # echo-vs-native layer schedule (Gemma, Colab/L4 values)
python check_claims.py               # asserts every headline number matches its logged JSON
```

`check_claims.py` cross-checks 33 headline numbers used in the README and this
table against the exact fields in `results/*.json`, and exits non-zero on any mismatch
(all 33 pass). The Colab/L4 rows below are the exception: their raw notebooks are not
bundled here (see the note above), so they are cited, not re-verified by this script.

## Headline numbers

| Quantity | Model | Value | Hardware | Script / source |
|---|---|---|---|---|
| KV saving, projected@4k | Qwen2.5-0.5B (11/24 local) | 45% (asymptote 46%) | local 4GB | `run_text_benchmarks.py` → `*_saving_vs_context.json` |
| KV saving, measured | Qwen2.5-0.5B | 37%@256 → 45%@2048 (grows with context) | local 4GB | `*_saving_vs_context.json` |
| Peak memory drop (chunked prefill) | Qwen2.5-0.5B | −33%@2k, **−65%@4k** (3466→1199 MB) | local 4GB | `*_chunked_peak.json` |
| Retrieval-needle survival | Qwen2.5-0.5B | echo **0.92–0.97** through K=4 vs random mean 0.41 (min 0.07), anti 0.66 | local 4GB | `*_needle.json` |
| Perplexity gap (held-out WikiText) | GPT-2 | echo +0.4@24% < random +2.5 < anti +9.6 (echo<random<anti at every K) | local 4GB | `gpt2_pareto.json` |
| Perplexity gap (held-out WikiText) | Qwen2.5-0.5B | echo +1.8@24%; beats anti every K; ties random on ppl (needle separates) | local 4GB | `Qwen2_5-0_5B-Instruct_pareto.json` |
| Long-context scaling (frozen schedule) | GPT-2 (K=5) | echo +1.3→+5.1, random +6.5→**+67.4** over T=256→1024 | local 4GB | `x11_scaling.json` (logged X11 run) |
| Long-context scaling | Qwen2.5-0.5B (K=10) | echo +0.9→+2.4, random +1.6→+3.3 over T=256→1024 | local 4GB | `x11_scaling.json` |
| KV saving | Qwen3-1.7B (15/28 local) | **53%**, ppl gap −0.46 (better), needle Vex→Vex | Colab/L4 | research notebook (not bundled) |
| KV saving | Gemma E4B (42-layer hybrid; see note) (23/42 local) | **54%** vs all-global baseline, ppl gap −0.65 | Colab/L4 | research notebook (not bundled) |
| Schedule rediscovery | Gemma E4B (42-layer hybrid) | **100%** of echo-local layers are native sliding-window in the run shown; a second calibration agreed on **83%** (localizing 4 native-global layers); 37% of native-global kept global | Colab/L4 | `make_overlay_figure.py` → `gemma_layer_overlay.json` |
| Multimodal grounding (all-local stress) | SmolVLM-256M (64 img tokens/card) | colour: full **100%**; recency **25%** (=chance, image evicted, saves 65%); **uniform anchors → 100% at budget 8** (saves 61%); echo-scored needs budget 32 (uniform coverage beats salience) | local 4GB | `run_multimodal_benchmark.py` → `smolvlm_multimodal.json` |
| Multimodal grounding (deployable schedule) | SmolVLM-256M (17/30 local) | colour recency **75%** (saves 37%) — global layers still carry the image, so the deployable schedule stays largely grounded without anchors | local 4GB | `smolvlm_multimodal.json` |
| Multimodal fine-detail ceiling (honest negative) | SmolVLM-256M | digit needle: full-cache ceiling collapses under the leak-free protocol → test INVALID below ~1B (gate reported, anchor sweep skipped) | local 4GB | `smolvlm_multimodal.json` |
| Multimodal grounding (all-local stress) | Gemma E4B (42-layer hybrid) | recency 25% (= chance, image evicted); anchors recover as budget grows | Colab/L4 | research notebook (not bundled) |

## Figures

| File | What | Hardware |
|---|---|---|
| `Qwen2_5-0_5B-Instruct_triviality.{png,pdf}` | per-layer triviality bars (red = localized) | local 4GB |
| `Qwen2_5-0_5B-Instruct_saving_vs_context.{png,pdf}` | measured KV saving vs context (→ asymptote) | local 4GB |
| `gpt2_pareto.{png,pdf}` | perplexity gap vs saving: echo < random < anti | local 4GB |
| `Qwen2_5-0_5B-Instruct_needle.{png,pdf}` | retrieval-needle survival under localization | local 4GB |
| `x11_scaling.{png,pdf}` | frozen schedule holds as context grows; random diverges | local 4GB (logged) |
| `Qwen2_5-0_5B-Instruct_chunked_peak.{png,pdf}` | chunked-prefill peak memory: full vs echo | local 4GB |
| `smolvlm_recency_vs_anchor.{png,pdf}` | multimodal recency-vs-anchor + ceiling gate + uniform-vs-scored | local 4GB |
| `gemma_layer_overlay.{png,pdf}` | echo-chosen local layers vs Gemma's native schedule | Colab/L4 values |

## Honest notes on the numbers

- **Saving is KV-cache memory**, growing with context toward `n_local/L`; it is not a
  reduction in token count or API billing.
- **Short-text perplexity is a weak proxy.** On Qwen the echo and random schedules tie on
  perplexity; the layer-choice signal shows up in the retrieval needle and in the
  long-context scaling, not in short-text perplexity.
- **Hybrid models:** the 54% on Gemma is vs an all-global baseline; the marginal gain
  over Gemma's own native sliding-window cache is smaller.
- **Multimodal negatives kept in:** echo-scored anchor selection does NOT beat uniform on
  redundant (colour) needles; the fine-detail (digit) needle ceiling collapses on a 256M
  VLM under the leak-free protocol, so detail-grounding under compression is not
  measurable below ~1B (the benchmark gates on the full-cache ceiling and reports it).
