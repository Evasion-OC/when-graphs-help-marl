"""Statistical analysis utilities for the empirical study.

Two thin layers:

* :func:`episodes_to_threshold` collapses a learning curve into a sample-
  efficiency scalar (the metric we report in the forest plots).
* :func:`pairwise_compare` runs the algo-vs-algo comparisons that drive the
  significance claims in the paper, with Holm step-down correction across the
  resulting pair-wise family.

The choice of Welch's t (default) over Mann–Whitney mirrors the
recommendations of Henderson et al. (2018) and Agarwal et al. (2021) for
deep-RL evaluation: small seed budgets, possibly non-normal returns, but
seed-wise scores are continuous and unbounded so Welch's t is reasonable as
the primary; Mann–Whitney is offered as a non-parametric robustness check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Sample efficiency
# ---------------------------------------------------------------------------


def episodes_to_threshold(
    returns: np.ndarray,
    threshold: float,
    *,
    window: int = 25,
) -> int | None:
    """First episode index where the windowed mean reaches ``threshold``.

    A rolling mean of width ``window`` smooths over evaluation noise. Returns
    ``None`` if the threshold is never reached within the run.
    """
    if returns.ndim != 1:
        raise ValueError(f"expected 1-D returns array; got shape {returns.shape}")
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")
    if len(returns) == 0:
        return None
    # Cumulative-sum rolling mean — O(n) and avoids the lib dep.
    csum = np.concatenate([[0.0], np.cumsum(returns)])
    rolling = (csum[window:] - csum[:-window]) / window  # len = n - window + 1
    hits = np.where(rolling >= threshold)[0]
    if len(hits) == 0:
        return None
    # Translate from rolling-array index back to episode index of the right
    # edge of the window — the episode at which the smoothed score crosses.
    return int(hits[0] + window - 1)


def threshold_from_best(
    returns_by_algo: dict[str, np.ndarray],
    fraction: float = 0.8,
) -> float:
    """**Deprecated** — kept for backwards compatibility with old analyses.

    Common threshold = ``fraction`` * max-best-final-window across algos.
    This metric is panel-dependent (an added algorithm changes every other
    algorithm's reported sample efficiency), which is a measurement bias
    we no longer use in the paper. Prefer :func:`self_threshold` for the
    per-algorithm self-threshold metric, or a fixed absolute threshold
    (e.g. ``fraction * theoretical_max_return``) defined upstream by the
    caller.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
    final_means = []
    for r in returns_by_algo.values():
        tail = max(1, len(r) // 10)
        final_means.append(float(np.mean(r[-tail:])))
    return fraction * max(final_means)


def self_threshold(returns: np.ndarray, fraction: float = 0.8) -> float:
    """Self-thresholding: ``fraction`` * own final-window mean.

    Equivalent to the within-algorithm "episodes to reach a fixed fraction
    of where this algorithm eventually plateaus." This metric is invariant
    to the panel of algorithms being compared — adding or removing an
    algorithm does not change any other algorithm's number.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
    if len(returns) == 0:
        raise ValueError("self_threshold called on empty returns array")
    tail = max(1, len(returns) // 10)
    return fraction * float(np.mean(returns[-tail:]))


# ---------------------------------------------------------------------------
# Pairwise significance with Holm correction
# ---------------------------------------------------------------------------


@dataclass
class PairwiseResult:
    """One row of the pairwise comparison table."""

    a: str
    b: str
    metric_a: float
    metric_b: float
    stat: float
    p_raw: float
    p_holm: float
    test: str


def holm_correct(p_values: Sequence[float]) -> list[float]:
    """Holm–Bonferroni step-down correction.

    Order p-values ascending, multiply the i-th by ``m - i`` (where m is the
    family size), then enforce monotonicity and clip to 1.0. The output is
    aligned to the *input* ordering.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    sorted_p = np.asarray(p_values, dtype=float)[order]
    # Holm step-down: multiply rank-i p by (m - i), then enforce monotone
    # non-decreasing via forward max-accumulate (an adjusted p cannot be
    # smaller than any earlier adjusted p in the sorted order).
    adjusted = np.maximum.accumulate(sorted_p * (m - np.arange(m)))
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adjusted
    return out.tolist()


def pairwise_compare(
    scores: dict[str, Sequence[float]],
    *,
    test: str = "welch",
    alternative: str = "two-sided",
) -> list[PairwiseResult]:
    """All unordered pairs (a, b), a != b, with Holm-corrected p-values.

    Args:
        scores: ``{algo_name: per_seed_scalar_scores}``.
        test: ``"welch"`` (default; Welch's two-sample t) or ``"mannwhitney"``
            (two-sided Mann–Whitney U).
        alternative: forwarded to the underlying test.
    """
    names = list(scores.keys())
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    raw: list[tuple[str, str, float, float, float, float]] = []
    for a, b in pairs:
        xa = np.asarray(scores[a], dtype=float)
        xb = np.asarray(scores[b], dtype=float)
        if test == "welch":
            res = stats.ttest_ind(xa, xb, equal_var=False, alternative=alternative)
            stat = float(res.statistic)
            p = float(res.pvalue)
        elif test == "mannwhitney":
            res = stats.mannwhitneyu(xa, xb, alternative=alternative)
            stat = float(res.statistic)
            p = float(res.pvalue)
        else:
            raise ValueError(f"unknown test {test!r}; expected welch | mannwhitney")
        raw.append((a, b, float(xa.mean()), float(xb.mean()), stat, p))

    p_holm = holm_correct([r[5] for r in raw])
    return [
        PairwiseResult(
            a=a,
            b=b,
            metric_a=ma,
            metric_b=mb,
            stat=stat,
            p_raw=p_raw,
            p_holm=p_h,
            test=test,
        )
        for (a, b, ma, mb, stat, p_raw), p_h in zip(raw, p_holm)
    ]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def mean_with_ci(
    values: Iterable[float], *, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Mean and Student-t CI half-width for small-N seed aggregates.

    Returns ``(mean, lo, hi)``. For ``n == 1`` returns ``(x, x, x)``; for
    constant samples returns a zero-width interval.
    """
    arr = np.asarray(list(values), dtype=float)
    n = len(arr)
    if n == 0:
        raise ValueError("mean_with_ci called on empty iterable")
    mean = float(arr.mean())
    if n == 1:
        return mean, mean, mean
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        return mean, mean, mean
    se = sd / np.sqrt(n)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    half = t_crit * se
    return mean, mean - half, mean + half
