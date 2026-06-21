"""KV-cache quantization (the bit axis): the quantizer, the in-place cache pass, and
the combined-saving accounting.  All model-free -- random tensors + a synthetic cache
object -- so they run in CI."""
import torch

from echokv.core import (
    _combined_saving,
    _quant_dequant,
    _quantize_cache_inplace,
)


def test_quant_is_noop_at_16_bits():
    x = torch.randn(1, 2, 7, 8)
    assert torch.equal(_quant_dequant(x, 16, (0, 2)), x)
    assert torch.equal(_quant_dequant(x, None, (0, 2)), x)


def test_quant_preserves_shape_and_dtype():
    x = torch.randn(1, 2, 7, 8, dtype=torch.float16)
    q = _quant_dequant(x, 4, (0, 2))
    assert q.shape == x.shape and q.dtype == torch.float16


def test_quant_error_decreases_with_bits():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 32, 16)
    errs = [(_quant_dequant(x, b, (0, 2)) - x).pow(2).mean().item()
            for b in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2]              # more bits -> less error
    assert errs[2] < 1e-3                           # 8-bit is near-lossless on N(0,1)


def test_quant_lossless_when_constant_along_reduced_axis():
    """Per-channel keys reduce over the token axis (dim 2): if every token shares the
    same per-(head,channel) value, min==max per group and quantization is exact."""
    c = torch.randn(1, 3, 1, 8)                     # one value per (head, channel)
    x = c.expand(1, 3, 20, 8).contiguous()          # constant across tokens
    q = _quant_dequant(x, 2, (0, 2))
    assert torch.allclose(q, x, atol=1e-5)


class _FakeCache:
    """Mimics the legacy DynamicCache (.key_cache / .value_cache lists), which
    `_cache_handles` supports."""
    def __init__(self, ks, vs):
        self.key_cache = ks
        self.value_cache = vs


def test_quantize_cache_inplace_changes_values_and_is_idempotent():
    torch.manual_seed(1)
    ks = [torch.randn(1, 2, 16, 8) for _ in range(3)]
    vs = [torch.randn(1, 2, 16, 8) for _ in range(3)]
    cache = _FakeCache([k.clone() for k in ks], [v.clone() for v in vs])
    _quantize_cache_inplace(cache, 4)
    # shapes preserved, values actually quantized (changed from the originals)
    for li in range(3):
        assert cache.key_cache[li].shape == ks[li].shape
        assert not torch.allclose(cache.key_cache[li], ks[li], atol=1e-4)
    # a second pass is a no-op (already on the grid)
    k_once = [k.clone() for k in cache.key_cache]
    _quantize_cache_inplace(cache, 4)
    for li in range(3):
        assert torch.allclose(cache.key_cache[li], k_once[li], atol=1e-4)


def test_quantize_cache_noop_at_16():
    ks = [torch.randn(1, 2, 5, 8)]
    vs = [torch.randn(1, 2, 5, 8)]
    cache = _FakeCache([k.clone() for k in ks], [v.clone() for v in vs])
    _quantize_cache_inplace(cache, 16)
    assert torch.equal(cache.key_cache[0], ks[0])


def test_combined_saving_multiplies_axes():
    # 50% tokens dropped + 4-bit (1/4 of the bytes) -> keep 0.5*0.25 = 0.125 -> save 87.5%
    assert abs(_combined_saving(0.5, 4) - 0.875) < 1e-9
    assert abs(_combined_saving(0.0, 8) - 0.5) < 1e-9     # quant only
    assert abs(_combined_saving(0.4, 16) - 0.4) < 1e-9    # 16-bit = token axis only
    assert _combined_saving(0.5, 4) > _combined_saving(0.5, 16)   # quant adds saving
