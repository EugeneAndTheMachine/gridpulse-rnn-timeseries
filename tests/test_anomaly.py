"""Tests cho anomaly detection."""
import numpy as np
from gridpulse.anomaly.thresholding import (
    BaseThreshold,
    PercentileThreshold,
    RollingThreshold,
    ZScoreThreshold,
)
from gridpulse.anomaly.event_grouping import filter_short_runs, group_anomalies


def test_percentile_threshold_flags_top_5_pct():
    scores = np.arange(100).astype(float)
    thr = PercentileThreshold(percentile=95).fit(scores)
    predictions = thr.predict(scores)
    # ~5% top scores should be flagged
    assert 4 <= predictions.sum() <= 6


def test_zscore_threshold_flags_outliers():
    np.random.seed(42)
    scores = np.random.randn(1000)
    scores[500] = 10.0  # extreme outlier
    thr = ZScoreThreshold(k=3.0).fit(scores)
    preds = thr.predict(scores)
    assert preds[500]


def test_group_anomalies_merges_close_points():
    is_anomaly = np.zeros(20, dtype=bool)
    is_anomaly[[5, 6, 7]] = True   # event 1
    is_anomaly[[15, 16]] = True    # event 2
    scores = np.zeros(20)
    scores[is_anomaly] = 1.0

    events = group_anomalies(is_anomaly, scores, max_gap=1, min_duration=1)
    assert len(events) == 2
    assert events[0].duration == 3
    assert events[1].duration == 2


def test_group_anomalies_respects_min_duration():
    is_anomaly = np.zeros(20, dtype=bool)
    is_anomaly[5] = True  # single point event
    is_anomaly[[10, 11, 12]] = True

    scores = is_anomaly.astype(float)
    events = group_anomalies(is_anomaly, scores, max_gap=0, min_duration=2)
    # Chỉ event 2 (3 points) qualifies
    assert len(events) == 1
    assert events[0].start_idx == 10


# ── Regression tests for max_gap bridging ───────────────────────────────────

def test_group_anomalies_bridges_across_gap():
    """max_gap=2 must merge [T,F,F,T] into a single event of duration 4."""
    is_anomaly = np.array([True, False, False, True])
    scores = is_anomaly.astype(float)
    events = group_anomalies(is_anomaly, scores, max_gap=2, min_duration=1)
    assert len(events) == 1
    assert events[0].start_idx == 0
    assert events[0].end_idx == 3
    assert events[0].duration == 4


def test_group_anomalies_bridges_single_gap():
    """max_gap=1 must merge [T,F,T] into one event."""
    is_anomaly = np.array([True, False, True])
    scores = is_anomaly.astype(float)
    events = group_anomalies(is_anomaly, scores, max_gap=1, min_duration=1)
    assert len(events) == 1
    assert events[0].start_idx == 0
    assert events[0].end_idx == 2


def test_group_anomalies_does_not_bridge_over_max_gap():
    """Gap of 3 > max_gap=1 must produce two separate events."""
    is_anomaly = np.array([True, False, False, False, True])
    scores = is_anomaly.astype(float)
    events = group_anomalies(is_anomaly, scores, max_gap=1, min_duration=1)
    assert len(events) == 2
    assert events[0].start_idx == 0 and events[0].end_idx == 0
    assert events[1].start_idx == 4 and events[1].end_idx == 4


def test_filter_short_runs_bridges_gap_before_length_check():
    """[T,F,T,T] with max_gap=1 becomes a run of 4 → survives min_duration=2."""
    is_anom = np.array([True, False, True, True])
    filt = filter_short_runs(is_anom, min_duration=2, max_gap=1)
    assert filt.tolist() == [True, True, True, True]


def test_filter_short_runs_zero_gap_default():
    """max_gap=0: [T,F,T,T] must drop the singleton T and keep the pair."""
    is_anom = np.array([True, False, True, True])
    filt = filter_short_runs(is_anom, min_duration=2, max_gap=0)
    assert filt.tolist() == [False, False, True, True]


# ── RollingThreshold data-leakage regression ────────────────────────────────

def test_rolling_threshold_flags_spike_without_leaking_into_stats():
    """A large spike must be flagged — its own value must not inflate the
    rolling std used to threshold itself."""
    scores = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
    thr = RollingThreshold(window=3, k=3.0).fit(scores)
    preds = thr.predict(scores)
    assert preds[-1], "spike at the end must be flagged"


def test_rolling_threshold_does_not_flag_stationary_series():
    """Constant-then-noisy series: no clear spikes → no flags."""
    rng = np.random.default_rng(0)
    scores = rng.normal(loc=0.0, scale=1.0, size=500)
    thr = RollingThreshold(window=50, k=5.0).fit(scores)
    preds = thr.predict(scores)
    # k=5 is very conservative — false-positive rate should be tiny
    assert preds.sum() < 5


def test_base_threshold_name_no_typo():
    """Ensure the base class is spelled BaseThreshold (not BaseTheshold)."""
    assert issubclass(PercentileThreshold, BaseThreshold)
    assert issubclass(ZScoreThreshold, BaseThreshold)
    assert issubclass(RollingThreshold, BaseThreshold)