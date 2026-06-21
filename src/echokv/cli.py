"""Command-line interface for echokv.

    echokv calibrate <model>  [--target-saving 0.4] [--seq-len 4096]
    echokv generate  <model>  --prompt "..." [--target-saving 0.4] [--max-new-tokens 128]
    echokv benchmark          [--out DIR] [--quick]

``calibrate`` profiles a model and prints the echo global/local schedule and the
projected KV-cache saving.  ``generate`` runs the memory-bounded cache on a prompt
and reports the realised saving.  ``benchmark`` runs the bundled reproduction suite
(see :mod:`echokv.benchmarks`).

Models must be loaded in bfloat16 -- modern checkpoints overflow float16.
"""
from __future__ import annotations

import argparse
import sys


def _load(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="eager")
    if torch.cuda.is_available():
        model = model.cuda()
    return model.eval(), tok


def _cmd_calibrate(args) -> int:
    from . import calibrate, evaluate_perplexity, kv_saving_report
    model, tok = _load(args.model)
    sch = calibrate(model, tok, target_saving=args.target_saving,
                    window=args.window, sink=args.sink, n_local=args.n_local)
    print(kv_saving_report(sch, seq_len=args.seq_len))
    if args.check_quality:
        print(evaluate_perplexity(model, tok, sch, kv_bits=args.kv_bits))
    return 0


def _cmd_generate(args) -> int:
    from . import calibrate, echo_generate
    model, tok = _load(args.model)
    sch = calibrate(model, tok, target_saving=args.target_saving,
                    window=args.window, sink=args.sink, n_local=args.n_local)
    prompt = args.prompt
    if args.chat:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True)
    text, stats = echo_generate(model, tok, prompt, sch,
                                max_new_tokens=args.max_new_tokens,
                                prefill_chunk=args.prefill_chunk,
                                kv_bits=args.kv_bits)
    print(text)
    msg = (f"\n[echokv] {sch.n_local}/{sch.num_layers} layers local; "
           f"KV tokens saved {100 * stats['kv_saving']:.0f}% at length "
           f"{stats['final_len']}")
    if args.kv_bits < 16:
        msg += (f"; with {args.kv_bits}-bit quant -> "
                f"{100 * stats['kv_saving_with_quant']:.0f}% of fp16 KV bytes saved")
    print(msg)
    return 0


def _cmd_benchmark(args) -> int:
    from . import benchmarks
    return benchmarks.main(out=args.out, quick=args.quick, only=args.only)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="echokv", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("model", help="HuggingFace model id or local path")
        sp.add_argument("--target-saving", type=float, default=0.4,
                        help="projected KV saving the schedule should reach (default 0.4)")
        sp.add_argument("--n-local", type=int, default=None,
                        help="fix the number of local layers instead of target-saving")
        sp.add_argument("--window", type=int, default=64)
        sp.add_argument("--sink", type=int, default=4)
        sp.add_argument("--kv-bits", type=int, default=16,
                        help="quantize the kept KV to this many bits, composing with "
                             "the layer schedule (8 near-lossless, 4 aggressive; "
                             "default 16 = off)")

    c = sub.add_parser("calibrate", help="profile a model and print its echo schedule")
    add_common(c)
    c.add_argument("--seq-len", type=int, default=4096,
                   help="context length to project the saving at (default 4096)")
    c.add_argument("--check-quality", action="store_true",
                   help="also run a quick full-vs-echo perplexity check")
    c.set_defaults(func=_cmd_calibrate)

    g = sub.add_parser("generate", help="generate with the memory-bounded echo cache")
    add_common(g)
    g.add_argument("--prompt", required=True)
    g.add_argument("--chat", action="store_true",
                   help="wrap the prompt with the model's chat template")
    g.add_argument("--max-new-tokens", type=int, default=128)
    g.add_argument("--prefill-chunk", type=int, default=None,
                   help="chunk the prefill to bound peak memory (text only)")
    g.set_defaults(func=_cmd_generate)

    b = sub.add_parser("benchmark", help="run the bundled reproduction benchmarks")
    b.add_argument("--out", default="echokv_bench", help="output directory")
    b.add_argument("--quick", action="store_true", help="smaller/faster configuration")
    b.add_argument("--only", default=None,
                   help="comma-separated subset of benchmarks to run")
    b.set_defaults(func=_cmd_benchmark)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":      # pragma: no cover
    sys.exit(main())
