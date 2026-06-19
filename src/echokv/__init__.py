"""echokv -- training-free KV-cache compression from the attention echo space.

The kernel ``cl(emptyset)`` of a transformer's attention echo space is its attention
sink.  Read per head or per layer, the echo space classifies which parts of the
model do genuine long-range retrieval and which only look locally.  The local ones
can run on a tiny sink+window cache; the retrieval ones keep theirs.  This package
turns that classification into a real, memory-bounded KV cache, with no training and
one calibration batch.

Typical use::

    from echokv import calibrate, echo_generate, kv_saving_report
    schedule = calibrate(model, tokenizer, target_saving=0.4)
    text, stats = echo_generate(model, tokenizer, prompt, schedule, max_new_tokens=128)
    print(kv_saving_report(schedule, seq_len=stats["final_len"]))
"""
from .core import (
    EchoSchedule,
    __version__,
    calibrate,
    echo_generate,
    evaluate_perplexity,
    get_attn_modules,
    image_anchor_scores,
    image_token_id,
    image_token_spans,
    install_echo_attention,
    kv_saving_report,
    make_schedule,
    measure_memory,
    profile_layers,
    restore_attention,
)

__all__ = [
    "EchoSchedule", "calibrate", "profile_layers", "make_schedule",
    "get_attn_modules", "install_echo_attention", "restore_attention",
    "echo_generate", "measure_memory", "kv_saving_report", "evaluate_perplexity",
    "image_token_id", "image_token_spans", "image_anchor_scores",
    "__version__",
]
