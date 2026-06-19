# Contributing to echokv

Thanks for your interest. `echokv` is a small, research-grade tool with a strong
bias toward **honest, reproducible claims**. Contributions are welcome under that
bias.

## Ground rules

1. **Every number in the docs must come from a run that exists in the repo.** If you
   add a result to the README or `BENCHMARKS.md`, add the script and the logged JSON
   that produced it, and label the hardware. No estimated or "simulated-as-real"
   numbers.
2. **Keep the negatives in.** The boundary of the method (classifier, not ranker;
   hybrid-model caveats; the multimodal scope limits) is part of the science. Do not
   delete a documented limitation to make a claim look stronger.
3. **Be precise about what is saved.** The win is KV-cache *memory* that grows with
   context, not API token billing.

## Development setup

```bash
python -m venv .venv && . .venv/bin/activate   # or your env of choice
pip install -e ".[dev]"
make test          # pytest (CPU-only tests run without a GPU)
make lint          # ruff
```

## Tests

- Most tests are CPU-only and use a tiny GPT-2 (`sshleifer/tiny-gpt2` or `gpt2`).
- Tests that need a GPU or download a larger model are marked `@pytest.mark.gpu`
  / `@pytest.mark.slow`; CI runs `-m "not gpu and not slow"`.
- New attention/cache behaviour must come with a test that pins the invariant
  (e.g. `echo_eager` prefill logits equal eager; an empty schedule changes nothing).

## Pull requests

- Run `make lint test` before opening a PR.
- Describe what you measured and on what hardware.
- Keep changes focused; one behavioural change per PR where possible.

## Reporting issues

Please include: model id, transformers/torch versions, dtype (must be bf16 for
modern models), and a minimal snippet. Attach the `stats` dict from
`echo_generate` if relevant.
