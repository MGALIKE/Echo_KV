"""_Pruner correctness on a synthetic cache -- no model download, runs in CI.

We encode each cached position's index into its key/value tensor so we can assert
exactly which positions survive a prune. Two invariants matter most:
  * text (no anchors)  ==  keep [:sink] + [-window:]  (backward compatibility)
  * anchors are ALWAYS frozen to the front, even if they sit inside the window
    (the multimodal image-grounding fix).
"""
import types

import torch

from echokv.core import _Pruner


def _fake_cache(num_layers, L, D=4):
    """Old DynamicCache-style: .key_cache / .value_cache lists, no .layers.
    Position p is encoded as the constant vector [p, p, p, p] so we can read it back."""
    kc, vc = [], []
    for _ in range(num_layers):
        k = torch.arange(L, dtype=torch.float32).view(1, 1, L, 1).expand(1, 1, L, D).contiguous()
        kc.append(k.clone())
        vc.append(k.clone())
    cache = types.SimpleNamespace(key_cache=kc, value_cache=vc)
    # ensure the new-API probe returns None so the kc/vc path is used
    assert getattr(cache, "layers", None) is None
    return cache


def _positions(cache, li):
    return cache.key_cache[li][0, 0, :, 0].tolist()


def test_text_keeps_sink_plus_window():
    L, sink, window = 100, 4, 10
    cache = _fake_cache(2, L)
    pr = _Pruner(local_layers=[0], sink=sink, window=window, anchors=None)
    pr.step(cache)
    kept = _positions(cache, 0)
    expected = list(range(sink)) + list(range(L - window, L))
    assert kept == expected
    # global layer 1 untouched
    assert len(_positions(cache, 1)) == L


def test_anchors_frozen_even_inside_window():
    L, sink, window = 100, 4, 10
    anchors = [50, 95]          # 50 is mid-sequence, 95 is INSIDE the last-10 window
    cache = _fake_cache(1, L)
    pr = _Pruner(local_layers=[0], sink=sink, window=window, anchors=anchors)
    pr.step(cache)
    kept = set(_positions(cache, 0))
    # both anchors must survive
    assert 50 in kept and 95 in kept
    # sink survives
    assert {0, 1, 2, 3} <= kept
    # the front block is sink + anchors (sorted, deduped)
    assert pr.front[0] == len({0, 1, 2, 3, 50, 95})


def test_window_slides_on_second_prune():
    L, sink, window = 60, 4, 8
    cache = _fake_cache(1, L)
    pr = _Pruner(local_layers=[0], sink=sink, window=window)
    pr.step(cache)
    front = pr.front[0]
    # simulate decode growth: append 5 new positions L..L+4
    k = cache.key_cache[0]
    extra = torch.arange(L, L + 5, dtype=torch.float32).view(1, 1, 5, 1).expand(1, 1, 5, k.shape[-1])
    cache.key_cache[0] = torch.cat([k, extra.contiguous()], dim=2)
    cache.value_cache[0] = cache.key_cache[0].clone()
    pr.step(cache)
    kept = _positions(cache, 0)
    # front rows preserved + most recent `window` positions
    assert len(kept) == front + window
    assert kept[-1] == L + 4            # newest position retained
    assert kept[:front] == list(range(sink))   # sink front preserved


def test_local_cache_stops_growing():
    """The whole point: a local layer's cache size is bounded by front+window."""
    L, sink, window = 200, 4, 16
    cache = _fake_cache(1, L)
    pr = _Pruner(local_layers=[0], sink=sink, window=window)
    pr.step(cache)
    assert cache.key_cache[0].shape[2] == sink + window


def test_value_aware_front_extra_frozen():
    """Value-aware front positions are frozen like anchors (the new capability)."""
    L, sink, window = 100, 4, 10
    cache = _fake_cache(1, L)
    pr = _Pruner(local_layers=[0], sink=sink, window=window, front_extra={0: [40, 41]})
    pr.step(cache)
    kept = set(_positions(cache, 0))
    assert {40, 41} <= kept                       # value-selected keys survive
    assert {0, 1, 2, 3} <= kept                   # sink (topology) still kept
    assert pr.front[0] == len({0, 1, 2, 3, 40, 41})


def test_value_subspace_avoids_redundant_direction():
    """value_subspace must span: prefer an orthogonal key over a near-duplicate of an
    already-kept high-norm key; value_norm just takes the two largest norms."""
    from echokv.core import _value_front
    vmat = torch.zeros(40, 4)
    vmat[10] = torch.tensor([10.0, 0, 0, 0])
    vmat[20] = torch.tensor([9.9, 0, 0, 0])       # near-duplicate of pos 10's direction
    vmat[30] = torch.tensor([0.0, 5, 0, 0])       # orthogonal, smaller norm
    cand = [10, 20, 30]
    assert _value_front(vmat, cand, 2, "value_norm") == [10, 20]
    assert _value_front(vmat, cand, 2, "value_subspace") == [10, 30]
