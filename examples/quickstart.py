"""Minimal echokv quickstart: calibrate, check quality, generate with the bounded cache.

    python examples/quickstart.py [model_name]

Defaults to Qwen2.5-0.5B-Instruct (a small grouped-query model). Use bfloat16.
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import echokv


def main(name="Qwen/Qwen2.5-0.5B-Instruct"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation="eager")
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    schedule = echokv.calibrate(model, tok, target_saving=0.4)
    print(echokv.kv_saving_report(schedule, seq_len=4096))
    print("quality:", echokv.evaluate_perplexity(model, tok, schedule))

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Summarise the theory of evolution in one sentence."}],
        tokenize=False, add_generation_prompt=True)
    text, stats = echokv.echo_generate(model, tok, prompt, schedule, max_new_tokens=96)
    print("\n---\n", text)
    print(f"\nKV cache saved {100 * stats['kv_saving']:.0f}% at length {stats['final_len']}")


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
