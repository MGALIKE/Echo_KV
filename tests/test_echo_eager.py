"""echo_eager must equal a reference eager attention on prefill, and attend to the
whole (possibly pruned) cache on decode. Tested at the kernel level on random
tensors -- no model download, runs in CI."""
import math
import types

import torch

from echokv.core import _echo_eager


def _ref_causal(q, k, v, scaling):
    """Reference eager causal attention (single sequence)."""
    qd = q.float()
    aw = torch.matmul(qd, k.float().transpose(2, 3)) * scaling
    ql, kl = q.shape[2], k.shape[2]
    rows = torch.arange(ql).view(-1, 1) + (kl - ql)
    cols = torch.arange(kl).view(1, -1)
    aw = aw.masked_fill(~(cols <= rows)[None, None], float("-inf"))
    aw = torch.softmax(aw, dim=-1)
    return torch.matmul(aw, v.float()).transpose(1, 2)


def test_prefill_equals_reference_mha():
    torch.manual_seed(0)
    B, H, T, D = 1, 4, 12, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    scaling = 1.0 / math.sqrt(D)
    mod = types.SimpleNamespace(num_key_value_groups=1)
    out, _ = _echo_eager(mod, q, k, v, attention_mask=None, scaling=scaling)
    ref = _ref_causal(q, k, v, scaling)
    assert out.shape == ref.shape
    assert torch.allclose(out.float(), ref, atol=1e-5)


def test_prefill_equals_reference_gqa():
    """Grouped-query: k/v have fewer heads; echo_eager must repeat them."""
    torch.manual_seed(1)
    B, Hq, T, D, groups = 1, 8, 10, 8, 2
    Hkv = Hq // groups
    q = torch.randn(B, Hq, T, D)
    k = torch.randn(B, Hkv, T, D)
    v = torch.randn(B, Hkv, T, D)
    scaling = 1.0 / math.sqrt(D)
    mod = types.SimpleNamespace(num_key_value_groups=groups)
    out, _ = _echo_eager(mod, q, k, v, attention_mask=None, scaling=scaling)
    k_rep = k.repeat_interleave(groups, dim=1)
    v_rep = v.repeat_interleave(groups, dim=1)
    ref = _ref_causal(q, k_rep, v_rep, scaling)
    assert torch.allclose(out.float(), ref, atol=1e-5)


def test_decode_attends_to_all_keys():
    """q_len==1: every cached key is a valid causal target, so no masking is applied
    -- this is what lets a pruned local layer hold a shorter/non-contiguous cache."""
    torch.manual_seed(2)
    B, H, Tk, D = 1, 2, 7, 8
    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, H, Tk, D)
    v = torch.randn(B, H, Tk, D)
    scaling = 1.0 / math.sqrt(D)
    mod = types.SimpleNamespace(num_key_value_groups=1)
    out, aw = _echo_eager(mod, q, k, v, attention_mask=None, scaling=scaling)
    # full softmax over all keys, no -inf entries
    assert torch.isfinite(aw).all()
    assert torch.allclose(aw.sum(-1), torch.ones(B, H, 1), atol=1e-5)


def test_scaling_derived_when_absent_gpt2_style():
    """When the framework does not pass scaling (GPT-2 interface), echo_eager derives
    1/sqrt(head_dim) from the module."""
    torch.manual_seed(3)
    B, H, T, D = 1, 2, 6, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    mod = types.SimpleNamespace(num_key_value_groups=1, scale_attn_weights=True,
                                scale_attn_by_inverse_layer_idx=False)
    out, _ = _echo_eager(mod, q, k, v, attention_mask=None, scaling=None)
    ref = _ref_causal(q, k, v, 1.0 / math.sqrt(D))
    assert torch.allclose(out.float(), ref, atol=1e-5)


def test_softcap_applied():
    """Gemma-style logit soft-cap squashes attention scores through tanh, changing the
    softmax distribution. Use a moderate logit scale (so softmax is NOT one-hot -- tanh
    is monotonic, so a saturated softmax would be unchanged) and a small cap that bites."""
    torch.manual_seed(4)
    B, H, T, D = 1, 1, 5, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    scaling = 1.0                # raw logits ~ N(0, D); not saturated
    capped = types.SimpleNamespace(num_key_value_groups=1, attn_logit_softcapping=0.5)
    uncapped = types.SimpleNamespace(num_key_value_groups=1)
    o1, _ = _echo_eager(capped, q, k, v, attention_mask=None, scaling=scaling)
    o2, _ = _echo_eager(uncapped, q, k, v, attention_mask=None, scaling=scaling)
    assert not torch.allclose(o1.float(), o2.float(), atol=1e-4)   # the cap changes the result
