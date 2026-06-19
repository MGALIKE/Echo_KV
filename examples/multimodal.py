"""Multimodal echokv example: keep a VLM grounded on its image with anchors.

    python examples/multimodal.py

Loads SmolVLM-256M, builds an image+text prompt, and compares the broken
pure-recency baseline (keep="recency", image evicted) against anchor keeping
(keep="anchor", image protected) under an all-local schedule. The decisive test is
a *decode-time* image question — short VQA hides the failure because the answer is
baked in during prefill. See experiments/x12_multimodal_needle.py for the full
needle protocol.
"""
import torch
from PIL import Image

import echokv


def solid(color, size=224):
    return Image.new("RGB", (size, size), color)


def main(name="HuggingFaceTB/SmolVLM-256M-Instruct"):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation="eager")
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    img = solid("green")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "What is the single most dominant colour in this picture?"}]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=prompt, images=[img], return_tensors="pt")

    # all layers local = the stress regime where recency goes blind
    schedule = echokv.calibrate(model, proc.tokenizer, n_local=None, target_saving=0.9)
    itid = echokv.image_token_id(model, proc)

    for keep in ("recency", "anchor"):
        text, stats = echokv.echo_generate(
            model, proc.tokenizer, None, schedule, inputs=dict(inputs),
            image_token_id=itid, image_budget=32, keep=keep,
            max_new_tokens=20, force_decode=None)
        print(f"[{keep:7s}] saved {100*stats['kv_saving']:.0f}%  ->  {text.strip()[:80]}")


if __name__ == "__main__":
    main()
