"""End-to-end tests on real (small) models. Marked `slow` (download/run a model);
CI skips them with `-m "not gpu and not slow"`. Run locally with `make test-all`.

These pin the behavioural contract:
  * echo_eager prefill logits == eager (the cache machinery is exact at prefill)
  * an empty schedule changes nothing (backward compatibility)
  * a calibrated text schedule preserves a retrieval needle (Vex -> Vex)
  * multimodal: anchor keeping protects image tokens that recency would evict
"""
import pytest
import torch

import echokv
from echokv.core import install_echo_attention, restore_attention


def _load_lm(name, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, attn_implementation="eager")
    if torch.cuda.is_available():
        model = model.cuda()
    return model.eval(), tok


@pytest.mark.slow
def test_echo_eager_equals_eager_prefill_gpt2():
    model, tok = _load_lm("gpt2")
    ids = tok("The quick brown fox jumps over the lazy dog by the river.",
              return_tensors="pt").input_ids.to(next(model.parameters()).device)
    with torch.no_grad():
        base = model(ids).logits
        prev = install_echo_attention(model)
        echo = model(ids).logits
        restore_attention(model, prev)
    assert (base - echo).abs().max().item() < 1e-4


@pytest.mark.slow
def test_empty_schedule_matches_plain_generate_gpt2():
    model, tok = _load_lm("gpt2")
    empty = echokv.EchoSchedule([], num_layers=len(echokv.get_attn_modules(model)))
    text, _ = echokv.echo_generate(model, tok, "The capital of France is", empty,
                                   max_new_tokens=12, greedy=True, stop_on_eos=False)
    dev = next(model.parameters()).device
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        ref = model.generate(ids, max_new_tokens=12, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    ref_text = tok.decode(ref[0, ids.shape[1]:], skip_special_tokens=True)
    assert text.strip() == ref_text.strip()


@pytest.mark.slow
@pytest.mark.gpu
def test_text_backward_compat_retrieval_qwen():
    if not torch.cuda.is_available():
        pytest.skip("needs GPU/bf16")
    model, tok = _load_lm("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16)
    sch = echokv.calibrate(model, tok, target_saving=0.4)
    ctx = ("Notes. The Helio project lead is Dana. The codeword for the archive is "
           "Vex. The Orchard project lead is Sam.\n\nQuestion: What is the codeword "
           "for the archive?\nAnswer:")
    text, _ = echokv.echo_generate(model, tok, ctx, sch, max_new_tokens=6, greedy=True)
    assert "Vex" in text


@pytest.mark.slow
@pytest.mark.gpu
def test_multimodal_anchor_mechanism_smolvlm():
    """Anchor keeping protects image tokens (n_anchor>0, larger cache); recency does
    not (n_anchor==0). This is the mechanism behind the grounding fix; the grounding
    *effect* is measured in the benchmark/X12 needle, not here."""
    if not torch.cuda.is_available():
        pytest.skip("needs GPU/bf16")
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    name = "HuggingFaceTB/SmolVLM-256M-Instruct"
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    img = Image.new("RGB", (224, 224), "green")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Describe the picture."}]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = dict(proc(text=prompt, images=[img], return_tensors="pt"))
    sch = echokv.calibrate(model, proc.tokenizer, target_saving=0.9)  # aggressive: many local
    itid = echokv.image_token_id(model, proc)

    _t, s_anchor = echokv.echo_generate(model, proc.tokenizer, None, sch, inputs=inputs,
                                        image_token_id=itid, image_budget=16,
                                        keep="anchor", max_new_tokens=8)
    _t, s_recency = echokv.echo_generate(model, proc.tokenizer, None, sch, inputs=inputs,
                                         image_token_id=itid, keep="recency",
                                         max_new_tokens=8)
    assert s_anchor["n_anchor"] > 0
    assert s_recency["n_anchor"] == 0
    assert s_anchor["n_image"] > 0
