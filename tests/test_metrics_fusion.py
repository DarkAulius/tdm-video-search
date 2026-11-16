import numpy as np

from tdm.fusion import fuse, tune_weight, zscore_rows
from tdm.metrics import ranks, summarize


def test_ranks_and_recall():
    scores = np.array([
        [0.9, 0.1, 0.2],   # clip corretta al 1° posto
        [0.8, 0.5, 0.1],   # 2° posto
        [0.7, 0.6, 0.1],   # 3° posto
    ])
    r = ranks(scores)
    assert r.tolist() == [1, 2, 3]
    s = summarize(r)
    assert s["R@1"] == 33.3 and s["R@5"] == 100.0 and s["MdR"] == 2.0


def test_ties_count_as_worst_position():
    assert ranks(np.array([[0.5, 0.5], [0.0, 1.0]])).tolist() == [2, 1]


def test_zscore_with_mask():
    z = zscore_rows(np.array([[1.0, 3.0, 100.0]]), mask=np.array([True, True, False]))
    assert np.allclose(z, [[-1.0, 1.0, 0.0]], atol=1e-6)


def test_fusion_weight_zero_is_visual_only():
    rng = np.random.default_rng(0)
    vis, sp = rng.normal(size=(5, 5)), rng.normal(size=(5, 5))
    assert np.allclose(fuse(vis, sp, np.ones(5, bool), 0.0), zscore_rows(vis))


def test_tuning_prefers_informative_speech():
    rng = np.random.default_rng(1)
    n = 40
    vis = rng.normal(size=(n, n)) + 0.8 * np.eye(n)   # visivo debole
    sp = rng.normal(size=(n, n)) + 3.0 * np.eye(n)    # parlato molto informativo
    w, curve = tune_weight(vis, sp, np.ones(n, bool), [i / 10 for i in range(11)])
    assert w > 0.3
    assert len(curve) == 11
