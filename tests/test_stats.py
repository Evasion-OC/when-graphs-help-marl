"""Unit tests for stats utilities."""

from __future__ import annotations

import numpy as np

from gnnmarl.utils.stats import (
    episodes_to_threshold,
    holm_correct,
    mean_with_ci,
    pairwise_compare,
    threshold_from_best,
)


def test_episodes_to_threshold_crosses() -> None:
    # Increasing returns: rolling mean of width 5 crosses 5.0 at episode 9
    # (window covers episodes 5..9 → mean = 7.0).
    r = np.arange(20, dtype=float)
    ep = episodes_to_threshold(r, threshold=5.0, window=5)
    assert ep is not None
    assert ep < len(r)


def test_episodes_to_threshold_never() -> None:
    r = np.zeros(50)
    assert episodes_to_threshold(r, threshold=10.0, window=5) is None


def test_episodes_to_threshold_short_run() -> None:
    r = np.array([1.0, 2.0])
    assert episodes_to_threshold(r, threshold=0.5, window=5) is None


def test_holm_correct_monotone() -> None:
    raw = [0.001, 0.04, 0.03, 0.5]
    adj = holm_correct(raw)
    # Sorted by p: [0.001, 0.03, 0.04, 0.5]; adjusted [0.004, 0.09, 0.08→0.09, 0.5]
    # After monotone enforcement: [0.004, 0.09, 0.09, 0.5]
    # Re-mapped to input order:
    expected_order = [0.004, 0.09, 0.09, 0.5]
    assert np.allclose(adj, expected_order, atol=1e-6)


def test_holm_clips_at_one() -> None:
    adj = holm_correct([0.9, 0.95])
    assert all(p <= 1.0 for p in adj)


def test_pairwise_compare_welch() -> None:
    rng = np.random.default_rng(0)
    scores = {
        "a": rng.normal(loc=0.0, scale=1.0, size=20).tolist(),
        "b": rng.normal(loc=2.0, scale=1.0, size=20).tolist(),
        "c": rng.normal(loc=0.1, scale=1.0, size=20).tolist(),
    }
    rows = pairwise_compare(scores, test="welch")
    assert len(rows) == 3
    pair_ab = next(r for r in rows if {r.a, r.b} == {"a", "b"})
    assert pair_ab.p_holm < 0.01  # large mean gap should survive correction


def test_pairwise_compare_mannwhitney_runs() -> None:
    scores = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
    rows = pairwise_compare(scores, test="mannwhitney")
    assert len(rows) == 1
    assert rows[0].test == "mannwhitney"
    assert 0.0 <= rows[0].p_raw <= 1.0


def test_mean_with_ci_single_value() -> None:
    m, lo, hi = mean_with_ci([3.0])
    assert (m, lo, hi) == (3.0, 3.0, 3.0)


def test_mean_with_ci_constant() -> None:
    m, lo, hi = mean_with_ci([2.0, 2.0, 2.0])
    assert m == 2.0
    assert lo == hi == 2.0


def test_mean_with_ci_basic() -> None:
    m, lo, hi = mean_with_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m == 3.0
    assert lo < m < hi


def test_threshold_from_best() -> None:
    runs = {
        "a": np.concatenate([np.zeros(90), np.ones(10) * 10.0]),
        "b": np.concatenate([np.zeros(90), np.ones(10) * 5.0]),
    }
    # Best final-window mean = 10.0; 80% threshold = 8.0.
    t = threshold_from_best(runs, fraction=0.8)
    assert abs(t - 8.0) < 1e-9
