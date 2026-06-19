"""Schedule math, anchor pooling, and image-span detection -- pure logic, runs in CI."""
import numpy as np

from echokv.core import EchoSchedule, _pooled_anchor_positions, image_token_spans, make_schedule


def test_saving_formula_and_monotonicity():
    sch = EchoSchedule(local_layers=list(range(10)), num_layers=24, window=64, sink=4)
    # below the kept budget there is no saving
    assert sch.saving(50) == 0.0
    s512 = sch.saving(512)
    s4096 = sch.saving(4096)
    # exact formula
    kept = sch.sink + sch.window
    assert abs(s4096 - 10 * (4096 - kept) / (24 * 4096)) < 1e-9
    # grows with context toward n_local/num_layers
    assert s512 < s4096 < sch.n_local / sch.num_layers
    assert sch.saving(10**9) < sch.n_local / sch.num_layers


def test_saving_with_anchors_is_smaller():
    sch = EchoSchedule(local_layers=list(range(10)), num_layers=24, window=64, sink=4)
    assert sch.saving(4096, anchors=0) > sch.saving(4096, anchors=32)


def test_make_schedule_picks_most_trivial():
    triv = np.array([0.9, 0.1, 0.8, 0.2, 0.95, 0.3])
    sch = make_schedule(triv, num_layers=6, n_local=3, window=64, sink=4)
    assert sorted(sch.local_layers) == [0, 2, 4]   # the three highest-triviality layers


def test_make_schedule_respects_min_triviality():
    triv = np.array([0.9, 0.1, 0.8, 0.2, 0.95, 0.3])
    # target large enough to want many layers, but only 3 are >= 0.55
    sch = make_schedule(triv, num_layers=6, target_saving=0.9, window=64, sink=4,
                        ref_len=1024, min_triviality=0.55)
    assert set(sch.local_layers) <= {0, 2, 4}
    assert 1 not in sch.local_layers and 3 not in sch.local_layers


def test_pooled_anchors_uniform_stride():
    spans = [(10, 26)]                # 16 image tokens
    pos = _pooled_anchor_positions(spans, budget=4)
    assert len(pos) == 4
    assert pos == sorted(pos)
    assert all(10 <= p < 26 for p in pos)


def test_pooled_anchors_all_when_budget_large():
    spans = [(0, 8)]
    pos = _pooled_anchor_positions(spans, budget=999)
    assert pos == list(range(8))


def test_pooled_anchors_scored_topk():
    spans = [(0, 6)]
    scores = {0: 0.1, 1: 0.9, 2: 0.2, 3: 0.8, 4: 0.05, 5: 0.7}
    pos = _pooled_anchor_positions(spans, budget=3, scores=scores)
    assert sorted(pos) == [1, 3, 5]   # top-3 most-attended


def test_image_token_spans():
    ids = [5, 5, 99, 99, 99, 7, 8, 99, 99]
    spans = image_token_spans(ids, img_id=99)
    assert spans == [(2, 5), (7, 9)]


def test_image_token_spans_none():
    assert image_token_spans([1, 2, 3], img_id=None) == []
