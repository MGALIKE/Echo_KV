"""calibrate() must raise a clear, actionable error when the model overflows float16
(non-finite attention). Tested by injecting a non-finite triviality -- no model
download, runs in CI."""
import numpy as np
import pytest

import echokv.core as core


def test_fp16_guard_raises(monkeypatch):
    monkeypatch.setattr(core, "get_attn_modules", lambda m: [object()] * 4)
    monkeypatch.setattr(core, "profile_layers",
                        lambda *a, **k: np.array([0.9, float("nan"), 0.8, 0.7]))
    with pytest.raises(RuntimeError) as exc:
        core.calibrate(model=None, tokenizer=None)
    msg = str(exc.value).lower()
    assert "bfloat16" in msg            # tells the user the fix
    assert "float16" in msg or "fp16" in msg


def test_finite_triviality_is_accepted(monkeypatch):
    monkeypatch.setattr(core, "get_attn_modules", lambda m: [object()] * 4)
    monkeypatch.setattr(core, "profile_layers",
                        lambda *a, **k: np.array([0.9, 0.6, 0.8, 0.7]))
    sch = core.calibrate(model=None, tokenizer=None, target_saving=0.3)
    assert sch.num_layers == 4
    assert all(0 <= l < 4 for l in sch.local_layers)
