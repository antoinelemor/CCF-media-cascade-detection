"""
Tests for cascade_detector.analysis.unified_impact

Uses synthetic data only — no DB, no embeddings.
"""

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import Dict, List, Set

from cascade_detector.analysis.unified_impact import (
    UnifiedImpactAnalyzer,
    UnifiedImpactResults,
    PRE_WINDOW,
    POST_WINDOW,
    PROXIMITY_SIGMA,
    IMPACT_STRONG,
    IMPACT_MODERATE,
    IMPACT_WEAK,
    DID_NOISE_THRESHOLD,
    GRANGER_SIG_THRESHOLD,
    LATE_SUPPORT_FRAME_RATIO,
    LATE_SUPPORT_EMBEDDING_SIM,
    LATE_SUPPORT_MIN_AFFINITY,
    P1_OVERLAP_FLOOR,
    P1_W_DID,
    P1_W_XCORR,
    P1_W_PROX,
    P1_W_DID_G,
    P1_W_XCORR_G,
    P1_W_GRANGER_P1,
    P1_W_PROX_G,
    P1_GRANGER_MIN_OBS,
    CONTENT_RELEVANCE_FLOOR,
    CONTENT_AFFINITY_SATURATION,
    CONTENT_DRIVER_THRESHOLD,
    STAT_SIGNIFICANCE_ALPHA,
    N_PERMUTATIONS,
    P_VALUE_FDR_ALPHA,
)
from cascade_detector.core.constants import FRAMES, FRAME_COLUMNS


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_occurrence(occurrence_id, event_type, peak_date, belonging, seed_doc_ids=None):
    """Minimal EventOccurrence-like object."""
    dates = sorted(belonging.keys())
    return SimpleNamespace(
        occurrence_id=occurrence_id,
        event_type=event_type,
        peak_date=pd.Timestamp(peak_date),
        core_start=pd.Timestamp('2018-01-01'),
        core_end=pd.Timestamp('2018-03-31'),
        first_date=pd.Timestamp('2018-01-01'),
        last_date=pd.Timestamp('2018-03-31'),
        effective_mass=sum(belonging.values()),
        belonging=belonging,
        doc_ids=list(belonging.keys()),
        seed_doc_ids=seed_doc_ids or list(belonging.keys()),
        n_articles=len(belonging),
    )


def _make_cluster(cluster_id, occurrences, peak_date, strength=0.5):
    """Minimal EventCluster-like object."""
    return SimpleNamespace(
        cluster_id=cluster_id,
        occurrences=occurrences,
        peak_date=pd.Timestamp(peak_date),
        core_start=pd.Timestamp('2018-01-15'),
        core_end=pd.Timestamp('2018-03-15'),
        strength=strength,
        total_mass=sum(o.effective_mass for o in occurrences),
        event_types=set(o.event_type for o in occurrences),
    )


def _make_cascade(cascade_id, frame, peak_date, total_score=0.6,
                   daily_composite=None):
    """Minimal CascadeResult-like object."""
    if daily_composite is None:
        idx = pd.date_range('2018-01-01', '2018-03-31', freq='D')
        daily_composite = pd.Series(np.random.randn(len(idx)) * 0.1, index=idx)
    return SimpleNamespace(
        cascade_id=cascade_id,
        frame=frame,
        peak_date=pd.Timestamp(peak_date),
        onset_date=pd.Timestamp('2018-01-01'),
        end_date=pd.Timestamp('2018-03-31'),
        total_score=total_score,
        daily_composite=daily_composite,
    )


def _make_articles(n=100, start='2018-01-01', end='2018-03-31'):
    """DataFrame with doc_id and date columns."""
    dates = pd.date_range(start, end, freq='D')
    doc_ids = list(range(1, n + 1))
    return pd.DataFrame({
        'doc_id': doc_ids,
        'date': [dates[i % len(dates)] for i in range(n)],
    })


def _make_paradigm_timeline(n_days=90, start='2018-01-01'):
    """Synthetic paradigm timeline with columns paradigm_{frame}."""
    dates = pd.date_range(start, periods=n_days, freq='D')
    data = {'date': dates}
    rng = np.random.default_rng(42)
    for frame in FRAMES:
        data[f'paradigm_{frame}'] = rng.uniform(0, 1, n_days)
    data['dominant_frames'] = 'Eco'
    data['paradigm_type'] = 'Mono-paradigm'
    data['concentration'] = 0.5
    data['coherence'] = 0.3
    return pd.DataFrame(data)


def _make_daily_composite(onset='2018-01-15', peak='2018-02-15',
                           end='2018-03-15'):
    """Synthetic Z-score series with known peak."""
    idx = pd.date_range(onset, end, freq='D')
    peak_dt = pd.Timestamp(peak)
    values = []
    for dt in idx:
        dist = abs((dt - peak_dt).days)
        values.append(2.0 * np.exp(-dist / 10.0))
    return pd.Series(values, index=idx)


def _make_phase1_row(**overrides):
    """Build a single Phase 1 row dict with sensible defaults + overrides."""
    defaults = {
        'cluster_id': 1, 'cascade_id': 'c1', 'cascade_frame': 'Eco',
        'cluster_strength': 0.5, 'proximity': 0.8,
        'diff_in_diff': 0.3, 'dose_response_corr': 0.6,
        'dose_response_lag': 5, 'article_overlap': 0.3,
        'impact_score': 0.10,
        'is_post_peak': False, 'frame_affinity': 0.15,
        'did_p_value': 0.01, 'xcorr_p_value': 0.02,
        'granger_f': 3.0, 'granger_p': 0.05,
        'content_relevance': 0.7,
    }
    defaults.update(overrides)
    return defaults


@pytest.fixture
def analyzer():
    return UnifiedImpactAnalyzer()


@pytest.fixture
def articles():
    return _make_articles(n=200)


@pytest.fixture
def paradigm_timeline():
    return _make_paradigm_timeline()


# ---------------------------------------------------------------------------
# TestDailyMass
# ---------------------------------------------------------------------------
class TestDailyMass:

    def test_basic(self, analyzer):
        """Daily mass aggregates belonging correctly."""
        # Use articles with dates that fall inside the cluster core period
        arts = pd.DataFrame({
            'doc_id': [1, 2, 3],
            'date': pd.to_datetime(['2018-02-01', '2018-02-05', '2018-02-10']),
        })
        occ = _make_occurrence(1, 'evt_weather', '2018-02-01',
                               {1: 0.8, 2: 0.6, 3: 0.4})
        cluster = _make_cluster(1, [occ], '2018-02-01')
        mass = analyzer._build_daily_mass(cluster, arts)
        assert isinstance(mass, pd.Series)
        assert mass.sum() > 0
        assert mass.sum() == pytest.approx(0.8 + 0.6 + 0.4)

    def test_max_aggregation(self, analyzer):
        """When two occurrences share a doc_id, max belonging is used."""
        arts = pd.DataFrame({
            'doc_id': [1, 2, 3],
            'date': pd.to_datetime(['2018-02-01', '2018-02-05', '2018-02-10']),
        })
        occ1 = _make_occurrence(1, 'evt_weather', '2018-02-01', {1: 0.8, 2: 0.3})
        occ2 = _make_occurrence(2, 'evt_meeting', '2018-02-01', {1: 0.5, 3: 0.9})
        cluster = _make_cluster(1, [occ1, occ2], '2018-02-01')
        mass = analyzer._build_daily_mass(cluster, arts)
        # doc_id 1 should use max(0.8, 0.5) = 0.8
        total = mass.sum()
        assert total > 0
        assert total == pytest.approx(0.8 + 0.3 + 0.9)

    def test_empty_belonging(self, analyzer, articles):
        """Empty belonging produces empty series."""
        occ = _make_occurrence(1, 'evt_weather', '2018-02-01', {})
        cluster = _make_cluster(1, [occ], '2018-02-01')
        mass = analyzer._build_daily_mass(cluster, articles)
        assert mass.empty or mass.sum() == 0

    def test_zero_fill(self, analyzer, articles):
        """Daily mass is zero-filled for core period."""
        occ = _make_occurrence(1, 'evt_weather', '2018-02-01', {1: 1.0})
        cluster = _make_cluster(1, [occ], '2018-02-01')
        mass = analyzer._build_daily_mass(cluster, articles)
        if not mass.empty:
            # Should cover the core period
            assert len(mass) >= 1


# ---------------------------------------------------------------------------
# TestTemporalProximity
# ---------------------------------------------------------------------------
class TestTemporalProximity:

    def test_same_date(self, analyzer):
        """Same date → proximity = 1.0."""
        d = pd.Timestamp('2018-06-15')
        assert analyzer._temporal_proximity(d, d) == pytest.approx(1.0)

    def test_far_apart(self, analyzer):
        """Dates far apart → proximity near 0."""
        d1 = pd.Timestamp('2018-01-01')
        d2 = pd.Timestamp('2018-12-31')
        prox = analyzer._temporal_proximity(d1, d2)
        assert prox < 0.01

    def test_symmetric(self, analyzer):
        """Proximity is symmetric."""
        d1 = pd.Timestamp('2018-03-01')
        d2 = pd.Timestamp('2018-03-10')
        assert analyzer._temporal_proximity(d1, d2) == pytest.approx(
            analyzer._temporal_proximity(d2, d1)
        )


# ---------------------------------------------------------------------------
# TestDiffInDiff
# ---------------------------------------------------------------------------
class TestDiffInDiff:

    def test_positive_shift(self, analyzer):
        """Post mean higher than pre → positive DID."""
        idx = pd.date_range('2018-01-01', '2018-03-31', freq='D')
        peak = pd.Timestamp('2018-02-15')
        values = [0.1 if dt < peak else 0.9 for dt in idx]
        target = pd.Series(values, index=idx)
        did = analyzer._diff_in_diff(target, peak)
        assert did > 0

    def test_negative_shift(self, analyzer):
        """Post mean lower than pre → negative DID."""
        idx = pd.date_range('2018-01-01', '2018-03-31', freq='D')
        peak = pd.Timestamp('2018-02-15')
        values = [0.9 if dt < peak else 0.1 for dt in idx]
        target = pd.Series(values, index=idx)
        did = analyzer._diff_in_diff(target, peak)
        assert did < 0

    def test_no_data(self, analyzer):
        """Empty series → DID = 0."""
        target = pd.Series(dtype=float)
        did = analyzer._diff_in_diff(target, pd.Timestamp('2018-02-15'))
        assert did == 0.0

    def test_flat_signal(self, analyzer):
        """Flat signal → DID ≈ 0."""
        idx = pd.date_range('2018-01-01', '2018-03-31', freq='D')
        target = pd.Series(0.5, index=idx)
        did = analyzer._diff_in_diff(target, pd.Timestamp('2018-02-15'))
        assert abs(did) < 0.01


# ---------------------------------------------------------------------------
# TestDiffInDiffWithPvalue
# ---------------------------------------------------------------------------
class TestDiffInDiffWithPvalue:

    def test_significant_shift(self, analyzer):
        """Large shift → small p-value."""
        idx = pd.date_range('2018-01-01', '2018-03-31', freq='D')
        peak = pd.Timestamp('2018-02-15')
        values = [0.1 if dt < peak else 0.9 for dt in idx]
        target = pd.Series(values, index=idx)
        did, p = analyzer._diff_in_diff_with_pvalue(target, peak)
        assert did > 0
        assert p < 0.05

    def test_flat_signal_high_p(self, analyzer):
        """Flat signal → high p-value."""
        idx = pd.date_range('2018-01-01', '2018-03-31', freq='D')
        target = pd.Series(0.5, index=idx)
        did, p = analyzer._diff_in_diff_with_pvalue(target, pd.Timestamp('2018-02-15'))
        assert abs(did) < 0.01
        # Welch test on identical means should give p ≈ 1 (or NaN → 1.0)
        assert p > 0.5

    def test_empty_series(self, analyzer):
        """Empty series → (0.0, 1.0)."""
        target = pd.Series(dtype=float)
        did, p = analyzer._diff_in_diff_with_pvalue(target, pd.Timestamp('2018-02-15'))
        assert did == 0.0
        assert p == 1.0

    def test_short_windows(self, analyzer):
        """Fewer than 3 obs per window → fallback."""
        idx = pd.date_range('2018-02-14', '2018-02-16', freq='D')
        target = pd.Series([0.5, 0.6, 0.7], index=idx)
        did, p = analyzer._diff_in_diff_with_pvalue(target, pd.Timestamp('2018-02-15'))
        assert did == 0.0
        assert p == 1.0

    def test_consistent_with_base_did(self, analyzer):
        """DID value should match the base _diff_in_diff method."""
        idx = pd.date_range('2018-01-01', '2018-03-31', freq='D')
        peak = pd.Timestamp('2018-02-15')
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, len(idx))
        target = pd.Series(values, index=idx)
        did_base = analyzer._diff_in_diff(target, peak)
        did_pval, _ = analyzer._diff_in_diff_with_pvalue(target, peak)
        assert did_pval == pytest.approx(did_base, abs=1e-10)


# ---------------------------------------------------------------------------
# TestDoseResponseWithPvalue
# ---------------------------------------------------------------------------
class TestDoseResponseWithPvalue:

    def test_correlated_signal(self, analyzer):
        """Strongly correlated signals → small p-value."""
        idx = pd.date_range('2018-01-01', periods=100, freq='D')
        source = pd.Series(np.sin(np.arange(100) * 0.1), index=idx)
        target = pd.Series(np.sin((np.arange(100) - 5) * 0.1), index=idx)
        corr, lag, p = analyzer._dose_response_with_pvalue(source, target)
        assert abs(corr) > 0.5
        assert p < 0.05

    def test_uncorrelated_signal(self, analyzer):
        """Independent signals → high p-value."""
        rng = np.random.default_rng(42)
        idx = pd.date_range('2018-01-01', periods=100, freq='D')
        source = pd.Series(rng.normal(0, 1, 100), index=idx)
        target = pd.Series(rng.normal(0, 1, 100), index=idx)
        corr, lag, p = analyzer._dose_response_with_pvalue(source, target)
        assert p > 0.01  # should not be very significant

    def test_short_signal(self, analyzer):
        """Very short signals → fallback."""
        idx = pd.date_range('2018-01-01', periods=2, freq='D')
        source = pd.Series([1.0, 2.0], index=idx)
        target = pd.Series([0.5, 1.5], index=idx)
        corr, lag, p = analyzer._dose_response_with_pvalue(source, target)
        assert corr == 0.0
        assert lag == 0
        assert p == 1.0

    def test_consistent_with_base(self, analyzer):
        """corr and lag should match the base _dose_response method."""
        idx = pd.date_range('2018-01-01', periods=100, freq='D')
        source = pd.Series(np.sin(np.arange(100) * 0.1), index=idx)
        target = pd.Series(np.sin((np.arange(100) - 3) * 0.1), index=idx)
        corr_base, lag_base = analyzer._dose_response(source, target)
        corr_pval, lag_pval, _ = analyzer._dose_response_with_pvalue(source, target)
        assert corr_pval == pytest.approx(corr_base)
        assert lag_pval == lag_base


# ---------------------------------------------------------------------------
# TestDoseResponse (original)
# ---------------------------------------------------------------------------
class TestDoseResponse:

    def test_lagged_signal(self, analyzer):
        """Source leads target by 5 days → best lag should be positive."""
        idx = pd.date_range('2018-01-01', periods=100, freq='D')
        source = pd.Series(np.sin(np.arange(100) * 0.1), index=idx)
        target = pd.Series(np.sin((np.arange(100) - 5) * 0.1), index=idx)
        corr, lag = analyzer._dose_response(source, target)
        assert abs(corr) > 0.5
        assert lag > 0  # source leads

    def test_anti_correlated(self, analyzer):
        """Anti-correlated signals → negative correlation."""
        idx = pd.date_range('2018-01-01', periods=100, freq='D')
        source = pd.Series(np.sin(np.arange(100) * 0.1), index=idx)
        target = pd.Series(-np.sin(np.arange(100) * 0.1), index=idx)
        corr, lag = analyzer._dose_response(source, target)
        assert corr < -0.5

    def test_short_signal(self, analyzer):
        """Very short signals → fallback (0.0, 0)."""
        idx = pd.date_range('2018-01-01', periods=2, freq='D')
        source = pd.Series([1.0, 2.0], index=idx)
        target = pd.Series([0.5, 1.5], index=idx)
        corr, lag = analyzer._dose_response(source, target)
        assert corr == 0.0
        assert lag == 0

    def test_lag_sign(self, analyzer):
        """When target leads source, best lag should be negative."""
        idx = pd.date_range('2018-01-01', periods=100, freq='D')
        source = pd.Series(np.sin((np.arange(100) - 5) * 0.1), index=idx)
        target = pd.Series(np.sin(np.arange(100) * 0.1), index=idx)
        corr, lag = analyzer._dose_response(source, target)
        assert lag < 0  # target leads


# ---------------------------------------------------------------------------
# TestGrangerCausality
# ---------------------------------------------------------------------------
class TestGrangerCausality:

    def test_causal_ar(self, analyzer):
        """AR(1) with causal link → significant p-value."""
        rng = np.random.default_rng(42)
        n = 200
        x = np.zeros(n)
        y = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.5 * x[i-1] + rng.normal(0, 0.1)
            y[i] = 0.5 * y[i-1] + 0.8 * x[i-1] + rng.normal(0, 0.1)

        idx = pd.date_range('2018-01-01', periods=n, freq='D')
        source = pd.Series(x, index=idx)
        target = pd.Series(y, index=idx)
        f_stat, p_val = analyzer._granger_causality(source, target)
        assert f_stat > 0
        assert p_val < 0.05

    def test_independent(self, analyzer):
        """Independent signals → high p-value."""
        rng = np.random.default_rng(123)
        n = 200
        idx = pd.date_range('2018-01-01', periods=n, freq='D')
        source = pd.Series(rng.normal(0, 1, n), index=idx)
        target = pd.Series(rng.normal(0, 1, n), index=idx)
        f_stat, p_val = analyzer._granger_causality(source, target)
        # Should not be strongly significant
        assert p_val > 0.01

    def test_too_short(self, analyzer):
        """Very short series → fallback (0.0, 1.0)."""
        idx = pd.date_range('2018-01-01', periods=5, freq='D')
        source = pd.Series([1, 2, 3, 4, 5], index=idx, dtype=float)
        target = pd.Series([5, 4, 3, 2, 1], index=idx, dtype=float)
        f_stat, p_val = analyzer._granger_causality(source, target)
        assert f_stat == 0.0
        assert p_val == 1.0


# ---------------------------------------------------------------------------
# TestContentRelevance
# ---------------------------------------------------------------------------
class TestContentRelevance:

    def test_high_affinity_high_alignment(self):
        """High values → score near 1.0."""
        cr = UnifiedImpactAnalyzer._compute_content_relevance(0.30, 0.90)
        assert cr > 0.9
        assert cr <= 1.0

    def test_zero_affinity_zero_alignment(self):
        """Zero values → score = FLOOR."""
        cr = UnifiedImpactAnalyzer._compute_content_relevance(0.0, 0.0)
        assert cr == pytest.approx(CONTENT_RELEVANCE_FLOOR)

    def test_nan_alignment_default(self):
        """NaN alignment → uses default 0.5."""
        cr = UnifiedImpactAnalyzer._compute_content_relevance(0.10, np.nan)
        # 0.60 * min(1, 0.10/0.20) + 0.40 * 0.5 = 0.30 + 0.20 = 0.50
        # result = 0.30 + 0.70 * min(1.0, 0.50) = 0.30 + 0.35 = 0.65
        assert cr == pytest.approx(0.65, abs=0.01)

    def test_saturation(self):
        """Affinity above saturation doesn't increase further."""
        cr1 = UnifiedImpactAnalyzer._compute_content_relevance(0.20, 0.5)
        cr2 = UnifiedImpactAnalyzer._compute_content_relevance(0.40, 0.5)
        # Both should have affinity_norm = 1.0 since 0.20 >= CONTENT_AFFINITY_SATURATION
        assert cr1 == pytest.approx(cr2)

    def test_floor_guarantee(self):
        """Score never drops below FLOOR."""
        cr = UnifiedImpactAnalyzer._compute_content_relevance(0.0, 0.0)
        assert cr >= CONTENT_RELEVANCE_FLOOR

    def test_monotonic_in_affinity(self):
        """Higher affinity → higher relevance (alignment fixed)."""
        cr_low = UnifiedImpactAnalyzer._compute_content_relevance(0.01, 0.5)
        cr_high = UnifiedImpactAnalyzer._compute_content_relevance(0.15, 0.5)
        assert cr_high > cr_low

    def test_monotonic_in_alignment(self):
        """Higher alignment → higher relevance (affinity fixed)."""
        cr_low = UnifiedImpactAnalyzer._compute_content_relevance(0.10, 0.1)
        cr_high = UnifiedImpactAnalyzer._compute_content_relevance(0.10, 0.9)
        assert cr_high > cr_low


# ---------------------------------------------------------------------------
# TestAttributionConfidence
# ---------------------------------------------------------------------------
class TestAttributionConfidence:

    def test_high_confidence(self):
        """All significant signals → high confidence."""
        row = {
            'did_p_value': 0.001, 'xcorr_p_value': 0.005, 'granger_p': 0.01,
            'perm_p_adjusted': 0.01,
            'content_relevance': 0.90,
            'proximity': 0.95,
        }
        conf = UnifiedImpactAnalyzer._compute_attribution_confidence(row)
        assert conf > 0.7

    def test_low_confidence(self):
        """No significant signals → low confidence."""
        row = {
            'did_p_value': 0.90, 'xcorr_p_value': 0.80, 'granger_p': 1.0,
            'perm_p_adjusted': 0.80,
            'content_relevance': CONTENT_RELEVANCE_FLOOR,
            'proximity': 0.1,
        }
        conf = UnifiedImpactAnalyzer._compute_attribution_confidence(row)
        assert conf < 0.3

    def test_bounded(self):
        """Confidence is always in [0, 1]."""
        row = {
            'did_p_value': 0.0, 'xcorr_p_value': 0.0, 'granger_p': 0.0,
            'perm_p_adjusted': 0.0,
            'content_relevance': 1.0,
            'proximity': 1.0,
        }
        conf = UnifiedImpactAnalyzer._compute_attribution_confidence(row)
        assert 0.0 <= conf <= 1.0

    def test_missing_fields_graceful(self):
        """Missing optional fields → uses defaults (no crash)."""
        row = {'proximity': 0.5}
        conf = UnifiedImpactAnalyzer._compute_attribution_confidence(row)
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# TestBenjaminiHochberg
# ---------------------------------------------------------------------------
class TestBenjaminiHochberg:

    def test_empty(self):
        """Empty array → empty result."""
        result = UnifiedImpactAnalyzer._benjamini_hochberg(np.array([]))
        assert len(result) == 0

    def test_single_value(self):
        """Single p-value → unchanged."""
        result = UnifiedImpactAnalyzer._benjamini_hochberg(np.array([0.05]))
        assert result[0] == pytest.approx(0.05)

    def test_monotonic_adjusted(self):
        """Adjusted p-values are monotonic in the sorted order."""
        p = np.array([0.01, 0.04, 0.03, 0.08, 0.50])
        adjusted = UnifiedImpactAnalyzer._benjamini_hochberg(p)
        sorted_idx = np.argsort(p)
        sorted_adj = adjusted[sorted_idx]
        for i in range(len(sorted_adj) - 1):
            assert sorted_adj[i] <= sorted_adj[i + 1] + 1e-10

    def test_known_values(self):
        """Known BH correction example."""
        p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        adjusted = UnifiedImpactAnalyzer._benjamini_hochberg(p)
        # All p-values are small enough relative to their rank that adjusted ≈ p * n / rank
        # p[0] adjusted = 0.01 * 5/1 = 0.05
        assert adjusted[0] == pytest.approx(0.05, abs=0.01)
        # All should be <= 1.0
        assert all(a <= 1.0 for a in adjusted)

    def test_bounded(self):
        """Adjusted p-values are in [0, 1]."""
        rng = np.random.default_rng(42)
        p = rng.uniform(0, 1, 50)
        adjusted = UnifiedImpactAnalyzer._benjamini_hochberg(p)
        assert all(0.0 <= a <= 1.0 for a in adjusted)


# ---------------------------------------------------------------------------
# TestCircularShift
# ---------------------------------------------------------------------------
class TestCircularShift:

    def test_identity_shift(self):
        """Shift by 0 → unchanged values."""
        idx = pd.date_range('2018-01-01', periods=5, freq='D')
        s = pd.Series([1, 2, 3, 4, 5], index=idx)
        shifted = UnifiedImpactAnalyzer._circular_shift_series(s, 0)
        assert list(shifted.values) == [1, 2, 3, 4, 5]

    def test_shift_by_one(self):
        """Shift by 1 → last value moves to front."""
        idx = pd.date_range('2018-01-01', periods=5, freq='D')
        s = pd.Series([1, 2, 3, 4, 5], index=idx)
        shifted = UnifiedImpactAnalyzer._circular_shift_series(s, 1)
        assert list(shifted.values) == [5, 1, 2, 3, 4]

    def test_preserves_index(self):
        """Shift preserves original index."""
        idx = pd.date_range('2018-01-01', periods=5, freq='D')
        s = pd.Series([1, 2, 3, 4, 5], index=idx)
        shifted = UnifiedImpactAnalyzer._circular_shift_series(s, 2)
        assert list(shifted.index) == list(idx)


# ---------------------------------------------------------------------------
# TestPhase1
# ---------------------------------------------------------------------------
class TestPhase1:

    def test_known_impact(self, analyzer, articles):
        """Cluster near cascade peak → positive impact score."""
        occ = _make_occurrence(1, 'evt_weather', '2018-02-15',
                               {i: 0.8 for i in range(1, 20)})
        cluster = _make_cluster(1, [occ], '2018-02-15', strength=0.7)
        daily_comp = _make_daily_composite('2018-01-15', '2018-02-15', '2018-03-15')
        cascade = _make_cascade('c1', 'Eco', '2018-02-15',
                                 daily_composite=daily_comp)

        df = analyzer.phase1_cluster_cascade([cluster], [cascade], articles)
        assert len(df) == 1
        assert df.iloc[0]['impact_score'] > 0

    def test_empty(self, analyzer, articles):
        """No clusters → empty DataFrame with correct columns."""
        df = analyzer.phase1_cluster_cascade([], [], articles)
        assert df.empty
        assert 'impact_score' in df.columns

    def test_strength_weighting(self, analyzer, articles):
        """Higher cluster strength → higher impact score."""
        occ = _make_occurrence(1, 'evt_weather', '2018-02-15',
                               {i: 0.8 for i in range(1, 20)})
        daily_comp = _make_daily_composite('2018-01-15', '2018-02-15', '2018-03-15')
        cascade = _make_cascade('c1', 'Eco', '2018-02-15',
                                 daily_composite=daily_comp)

        cluster_weak = _make_cluster(1, [occ], '2018-02-15', strength=0.2)
        cluster_strong = _make_cluster(2, [occ], '2018-02-15', strength=0.9)

        df_weak = analyzer.phase1_cluster_cascade([cluster_weak], [cascade], articles)
        df_strong = analyzer.phase1_cluster_cascade([cluster_strong], [cascade], articles)

        assert df_strong.iloc[0]['impact_score'] > df_weak.iloc[0]['impact_score']

    def test_proximity_weighting(self, analyzer, articles):
        """Closer cluster → higher impact than distant one."""
        occ1 = _make_occurrence(1, 'evt_weather', '2018-02-15',
                                {i: 0.8 for i in range(1, 20)})
        occ2 = _make_occurrence(2, 'evt_meeting', '2018-01-01',
                                {i: 0.8 for i in range(1, 20)})
        cluster_close = _make_cluster(1, [occ1], '2018-02-15', strength=0.5)
        cluster_far = _make_cluster(2, [occ2], '2018-01-01', strength=0.5)

        daily_comp = _make_daily_composite('2018-01-15', '2018-02-15', '2018-03-15')
        cascade = _make_cascade('c1', 'Eco', '2018-02-15',
                                 daily_composite=daily_comp)

        df_close = analyzer.phase1_cluster_cascade([cluster_close], [cascade], articles)
        df_far = analyzer.phase1_cluster_cascade([cluster_far], [cascade], articles)

        # Close cluster has higher proximity component
        assert df_close.iloc[0]['proximity'] > df_far.iloc[0]['proximity']

    def test_new_columns_present(self, analyzer, articles):
        """Phase 1 output includes new statistical columns."""
        occ = _make_occurrence(1, 'evt_weather', '2018-02-15',
                               {i: 0.8 for i in range(1, 20)})
        cluster = _make_cluster(1, [occ], '2018-02-15', strength=0.5)
        daily_comp = _make_daily_composite('2018-01-15', '2018-02-15', '2018-03-15')
        cascade = _make_cascade('c1', 'Eco', '2018-02-15',
                                 daily_composite=daily_comp)

        df = analyzer.phase1_cluster_cascade([cluster], [cascade], articles)
        for col in ['did_p_value', 'xcorr_p_value', 'granger_f', 'granger_p',
                     'content_relevance']:
            assert col in df.columns, f"Missing column: {col}"

    def test_p_values_valid_range(self, analyzer, articles):
        """P-values are in [0, 1]."""
        occ = _make_occurrence(1, 'evt_weather', '2018-02-15',
                               {i: 0.8 for i in range(1, 20)})
        cluster = _make_cluster(1, [occ], '2018-02-15', strength=0.5)
        daily_comp = _make_daily_composite('2018-01-15', '2018-02-15', '2018-03-15')
        cascade = _make_cascade('c1', 'Eco', '2018-02-15',
                                 daily_composite=daily_comp)

        df = analyzer.phase1_cluster_cascade([cluster], [cascade], articles)
        row = df.iloc[0]
        assert 0.0 <= row['did_p_value'] <= 1.0
        assert 0.0 <= row['xcorr_p_value'] <= 1.0
        assert 0.0 <= row['granger_p'] <= 1.0
        assert CONTENT_RELEVANCE_FLOOR <= row['content_relevance'] <= 1.0


# ---------------------------------------------------------------------------
# TestPhase2
# ---------------------------------------------------------------------------
class TestPhase2:

    def test_all_frames(self, analyzer, articles, paradigm_timeline):
        """Phase 2 produces rows for all 8 frames per cluster."""
        occ = _make_occurrence(1, 'evt_weather', '2018-02-15',
                               {i: 0.8 for i in range(1, 20)})
        cluster = _make_cluster(1, [occ], '2018-02-15', strength=0.5)

        df = analyzer.phase2_cluster_dominance([cluster], paradigm_timeline, articles)
        assert len(df) == len(FRAMES)
        assert set(df['frame']) == set(FRAMES)

    def test_known_causal(self, analyzer, articles):
        """Cluster with causal link to dominance → non-zero impact."""
        rng = np.random.default_rng(42)
        n = 90
        dates = pd.date_range('2018-01-01', periods=n, freq='D')

        # Build timeline where Eco dominance follows cluster mass
        timeline_data = {'date': dates}
        for frame in FRAMES:
            if frame == 'Eco':
                # Signal that ramps up at day 45
                vals = [0.2] * 45 + [0.8] * 45
                timeline_data[f'paradigm_{frame}'] = vals
            else:
                timeline_data[f'paradigm_{frame}'] = rng.uniform(0.1, 0.3, n)
        timeline_data['dominant_frames'] = 'Eco'
        timeline_data['paradigm_type'] = 'Mono-paradigm'
        timeline_data['concentration'] = 0.5
        timeline_data['coherence'] = 0.3
        timeline = pd.DataFrame(timeline_data)

        occ = _make_occurrence(1, 'evt_weather', '2018-02-15',
                               {i: 0.8 for i in range(1, 30)})
        cluster = _make_cluster(1, [occ], '2018-02-15', strength=0.6)

        df = analyzer.phase2_cluster_dominance([cluster], timeline, articles)
        eco_row = df[df['frame'] == 'Eco']
        assert len(eco_row) == 1
        assert eco_row.iloc[0]['diff_in_diff'] != 0.0

    def test_no_timeline(self, analyzer, articles):
        """No paradigm timeline → empty DataFrame."""
        occ = _make_occurrence(1, 'evt_weather', '2018-02-15', {1: 0.8})
        cluster = _make_cluster(1, [occ], '2018-02-15')
        df = analyzer.phase2_cluster_dominance([cluster], None, articles)
        assert df.empty
        assert 'granger_p' in df.columns


# ---------------------------------------------------------------------------
# TestPhase3
# ---------------------------------------------------------------------------
class TestPhase3:

    def test_cascade_influence(self, analyzer, paradigm_timeline):
        """Cascade with daily_composite → produces rows for all frames."""
        daily_comp = _make_daily_composite('2018-01-15', '2018-02-15', '2018-03-15')
        cascade = _make_cascade('c1', 'Eco', '2018-02-15',
                                 daily_composite=daily_comp, total_score=0.7)

        df = analyzer.phase3_cascade_dominance([cascade], paradigm_timeline)
        assert len(df) == len(FRAMES)
        assert all(df['cascade_frame'] == 'Eco')
        assert all(df['cascade_score'] == 0.7)

    def test_missing_daily_composite(self, analyzer, paradigm_timeline):
        """Cascade without daily_composite → skipped."""
        cascade = SimpleNamespace(
            cascade_id='c1', frame='Eco', peak_date=pd.Timestamp('2018-02-15'),
            total_score=0.5, daily_composite=None,
        )
        df = analyzer.phase3_cascade_dominance([cascade], paradigm_timeline)
        assert df.empty

    def test_output_columns(self, analyzer, paradigm_timeline):
        """Output DataFrame has expected columns."""
        daily_comp = _make_daily_composite()
        cascade = _make_cascade('c1', 'Eco', '2018-02-15',
                                 daily_composite=daily_comp)
        df = analyzer.phase3_cascade_dominance([cascade], paradigm_timeline)
        expected = {
            'cascade_id', 'frame', 'cascade_frame', 'cascade_score',
            'diff_in_diff', 'dose_response_corr', 'dose_response_lag',
            'granger_f', 'granger_p', 'impact_score',
        }
        # Phase 3 raw output doesn't include role columns (added by run_from_components)
        assert expected.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# TestRun
# ---------------------------------------------------------------------------
class TestRun:

    def test_full_run(self, analyzer, articles, paradigm_timeline):
        """Full run returns UnifiedImpactResults with all three DataFrames."""
        occ = _make_occurrence(1, 'evt_weather', '2018-02-15',
                               {i: 0.8 for i in range(1, 20)})
        cluster = _make_cluster(1, [occ], '2018-02-15', strength=0.5)
        daily_comp = _make_daily_composite()
        cascade = _make_cascade('c1', 'Eco', '2018-02-15',
                                 daily_composite=daily_comp)

        result = analyzer.run_from_components(
            clusters=[cluster],
            cascades=[cascade],
            articles=articles,
            paradigm_timeline=paradigm_timeline,
        )

        assert isinstance(result, UnifiedImpactResults)
        assert not result.cluster_cascade.empty
        assert not result.cluster_dominance.empty
        assert not result.cascade_dominance.empty
        assert result.summary['n_clusters'] == 1
        assert result.summary['n_cascades'] == 1

    def test_empty_inputs(self, analyzer):
        """Empty inputs → empty DataFrames, no errors."""
        result = analyzer.run_from_components(
            clusters=[], cascades=[],
            articles=pd.DataFrame(),
            paradigm_timeline=None,
        )
        assert isinstance(result, UnifiedImpactResults)
        assert result.cluster_cascade.empty
        assert result.cluster_dominance.empty
        assert result.cascade_dominance.empty
        assert result.summary['n_clusters'] == 0

    def test_output_schemas(self, analyzer, articles, paradigm_timeline):
        """Verify all output DataFrames have expected column sets."""
        occ = _make_occurrence(1, 'evt_weather', '2018-02-15',
                               {i: 0.8 for i in range(1, 20)})
        cluster = _make_cluster(1, [occ], '2018-02-15', strength=0.5)
        daily_comp = _make_daily_composite()
        cascade = _make_cascade('c1', 'Eco', '2018-02-15',
                                 daily_composite=daily_comp)

        result = analyzer.run_from_components(
            clusters=[cluster], cascades=[cascade],
            articles=articles, paradigm_timeline=paradigm_timeline,
        )

        p1_cols = {
            'cluster_id', 'cascade_id', 'cascade_frame', 'cluster_strength',
            'proximity', 'diff_in_diff', 'dose_response_corr',
            'dose_response_lag', 'article_overlap', 'impact_score',
            'is_post_peak', 'frame_affinity', 'embedding_alignment',
            'role', 'impact_label',
            'did_p_value', 'xcorr_p_value', 'granger_f', 'granger_p',
            'content_relevance', 'perm_p_value', 'perm_p_adjusted', 'confidence',
        }
        p2_cols = {
            'cluster_id', 'frame', 'cluster_strength',
            'diff_in_diff', 'dose_response_corr', 'dose_response_lag',
            'granger_f', 'granger_p', 'impact_score', 'role', 'impact_label',
        }
        p3_cols = {
            'cascade_id', 'frame', 'cascade_frame', 'cascade_score',
            'diff_in_diff', 'dose_response_corr', 'dose_response_lag',
            'granger_f', 'granger_p', 'impact_score',
            'role', 'impact_label', 'is_own_frame',
        }

        assert p1_cols == set(result.cluster_cascade.columns), \
            f"Missing: {p1_cols - set(result.cluster_cascade.columns)}, " \
            f"Extra: {set(result.cluster_cascade.columns) - p1_cols}"
        assert set(result.cluster_dominance.columns) == p2_cols
        assert set(result.cascade_dominance.columns) == p3_cols

    def test_permutation_columns(self, analyzer, articles, paradigm_timeline):
        """Full run produces permutation p-values and confidence."""
        occ = _make_occurrence(1, 'evt_weather', '2018-02-15',
                               {i: 0.8 for i in range(1, 20)})
        cluster = _make_cluster(1, [occ], '2018-02-15', strength=0.5)
        daily_comp = _make_daily_composite()
        cascade = _make_cascade('c1', 'Eco', '2018-02-15',
                                 daily_composite=daily_comp)

        result = analyzer.run_from_components(
            clusters=[cluster], cascades=[cascade],
            articles=articles, paradigm_timeline=paradigm_timeline,
        )

        df1 = result.cluster_cascade
        assert 'perm_p_value' in df1.columns
        assert 'perm_p_adjusted' in df1.columns
        assert 'confidence' in df1.columns
        row = df1.iloc[0]
        assert 0.0 <= row['perm_p_value'] <= 1.0
        assert 0.0 <= row['perm_p_adjusted'] <= 1.0
        assert 0.0 <= row['confidence'] <= 1.0


# ---------------------------------------------------------------------------
# TestClassifyImpact
# ---------------------------------------------------------------------------
class TestClassifyImpact:

    def test_strong(self):
        assert UnifiedImpactAnalyzer._classify_impact(0.20) == 'strong'
        assert UnifiedImpactAnalyzer._classify_impact(0.15) == 'strong'

    def test_moderate(self):
        assert UnifiedImpactAnalyzer._classify_impact(0.10) == 'moderate'
        assert UnifiedImpactAnalyzer._classify_impact(0.05) == 'moderate'

    def test_weak(self):
        assert UnifiedImpactAnalyzer._classify_impact(0.03) == 'weak'
        assert UnifiedImpactAnalyzer._classify_impact(0.01) == 'weak'

    def test_negligible(self):
        assert UnifiedImpactAnalyzer._classify_impact(0.009) == 'negligible'
        assert UnifiedImpactAnalyzer._classify_impact(0.0) == 'negligible'

    def test_boundary_exact(self):
        """Boundary values land in the higher bucket (>=)."""
        assert UnifiedImpactAnalyzer._classify_impact(IMPACT_STRONG) == 'strong'
        assert UnifiedImpactAnalyzer._classify_impact(IMPACT_MODERATE) == 'moderate'
        assert UnifiedImpactAnalyzer._classify_impact(IMPACT_WEAK) == 'weak'


# ---------------------------------------------------------------------------
# TestPhase1Roles
# ---------------------------------------------------------------------------
class TestPhase1Roles:

    def test_driver_role(self, analyzer):
        """Positive DID above noise + significant p-value + affinity → driver."""
        df = pd.DataFrame([_make_phase1_row(
            diff_in_diff=0.3, impact_score=0.10,
            did_p_value=0.01, xcorr_p_value=0.02,
            is_post_peak=False, frame_affinity=0.15,
        )])
        result = analyzer._assign_roles_phase1(df)
        assert result.iloc[0]['role'] == 'driver'
        assert result.iloc[0]['impact_label'] == 'moderate'

    def test_driver_blocked_by_low_affinity(self, analyzer):
        """Positive DID but low affinity → neutral (not driver)."""
        df = pd.DataFrame([_make_phase1_row(
            diff_in_diff=0.3, impact_score=0.10,
            did_p_value=0.01, xcorr_p_value=0.02,
            is_post_peak=False, frame_affinity=0.01,  # below threshold
        )])
        result = analyzer._assign_roles_phase1(df)
        assert result.iloc[0]['role'] == 'neutral'

    def test_suppressor_role(self, analyzer):
        """Negative DID below -noise + significant → suppressor."""
        df = pd.DataFrame([_make_phase1_row(
            diff_in_diff=-0.1, impact_score=0.05,
            did_p_value=0.02, xcorr_p_value=0.05,
        )])
        result = analyzer._assign_roles_phase1(df)
        assert result.iloc[0]['role'] == 'suppressor'

    def test_suppressor_pre_peak(self, analyzer):
        """Pre-peak suppressor remains suppressor (attentional competition)."""
        df = pd.DataFrame([_make_phase1_row(
            diff_in_diff=-0.15, impact_score=0.06,
            did_p_value=0.01, xcorr_p_value=0.03,
            is_post_peak=False, frame_affinity=0.20,
        )])
        result = analyzer._assign_roles_phase1(df)
        assert result.iloc[0]['role'] == 'suppressor'

    def test_late_support_post_peak_positive_did(self, analyzer):
        """Post-peak + positive DID + high affinity → late_support."""
        df = pd.DataFrame([_make_phase1_row(
            diff_in_diff=0.15, impact_score=0.08,
            did_p_value=0.01, xcorr_p_value=0.03,
            is_post_peak=True, frame_affinity=0.10,  # >= CONTENT_DRIVER_THRESHOLD
        )])
        result = analyzer._assign_roles_phase1(df)
        assert result.iloc[0]['role'] == 'late_support'

    def test_neutral_no_significant_pvalue(self, analyzer):
        """No significant p-value → neutral regardless of DID."""
        df = pd.DataFrame([_make_phase1_row(
            diff_in_diff=0.5, impact_score=0.10,
            did_p_value=0.50, xcorr_p_value=0.60, granger_p=1.0,
        )])
        result = analyzer._assign_roles_phase1(df)
        assert result.iloc[0]['role'] == 'neutral'

    def test_neutral_did_in_noise_band(self, analyzer):
        """DID within noise band + significant → neutral."""
        df = pd.DataFrame([_make_phase1_row(
            diff_in_diff=0.005, impact_score=0.08,
            did_p_value=0.05, xcorr_p_value=0.05,
        )])
        result = analyzer._assign_roles_phase1(df)
        assert result.iloc[0]['role'] == 'neutral'

    def test_unrelated_role(self, analyzer):
        """Impact < 0.01 → unrelated regardless of DID."""
        df = pd.DataFrame([_make_phase1_row(
            diff_in_diff=0.5, impact_score=0.005,
            did_p_value=0.01, xcorr_p_value=0.01,
        )])
        result = analyzer._assign_roles_phase1(df)
        assert result.iloc[0]['role'] == 'unrelated'

    def test_valid_roles_only(self, analyzer):
        """All assigned roles are from the valid set."""
        valid = {'driver', 'suppressor', 'late_support', 'neutral', 'unrelated'}
        rows = [
            _make_phase1_row(cluster_id=0, cascade_id='c0',
                             diff_in_diff=0.1, impact_score=0.10,
                             did_p_value=0.01, is_post_peak=False, frame_affinity=0.15),
            _make_phase1_row(cluster_id=1, cascade_id='c1',
                             diff_in_diff=-0.2, impact_score=0.03,
                             did_p_value=0.01),
            _make_phase1_row(cluster_id=2, cascade_id='c2',
                             diff_in_diff=0.005, impact_score=0.02,
                             did_p_value=0.05),
            _make_phase1_row(cluster_id=3, cascade_id='c3',
                             diff_in_diff=0.5, impact_score=0.001),
        ]
        df = pd.DataFrame(rows)
        result = analyzer._assign_roles_phase1(df)
        assert set(result['role']).issubset(valid)

    def test_empty_dataframe(self, analyzer):
        """Empty input → empty output with role and impact_label columns."""
        df = pd.DataFrame(columns=[
            'cluster_id', 'cascade_id', 'cascade_frame', 'cluster_strength',
            'proximity', 'diff_in_diff', 'dose_response_corr',
            'dose_response_lag', 'impact_score',
        ])
        result = analyzer._assign_roles_phase1(df)
        assert 'role' in result.columns
        assert 'impact_label' in result.columns
        assert result.empty


# ---------------------------------------------------------------------------
# TestPhase2Roles
# ---------------------------------------------------------------------------
class TestPhase2Roles:

    def test_catalyst(self, analyzer):
        """Significant Granger + positive DID → catalyst."""
        df = pd.DataFrame([{
            'cluster_id': 1, 'frame': 'Eco', 'cluster_strength': 0.5,
            'diff_in_diff': 0.2, 'dose_response_corr': 0.3,
            'dose_response_lag': 2, 'granger_f': 5.0, 'granger_p': 0.01,
            'impact_score': 0.08,
        }])
        result = analyzer._assign_roles_phase2(df)
        assert result.iloc[0]['role'] == 'catalyst'

    def test_disruptor(self, analyzer):
        """Significant Granger + negative DID → disruptor."""
        df = pd.DataFrame([{
            'cluster_id': 1, 'frame': 'Eco', 'cluster_strength': 0.5,
            'diff_in_diff': -0.1, 'dose_response_corr': 0.3,
            'dose_response_lag': 2, 'granger_f': 5.0, 'granger_p': 0.02,
            'impact_score': 0.06,
        }])
        result = analyzer._assign_roles_phase2(df)
        assert result.iloc[0]['role'] == 'disruptor'

    def test_inert(self, analyzer):
        """Non-significant Granger → inert."""
        df = pd.DataFrame([{
            'cluster_id': 1, 'frame': 'Eco', 'cluster_strength': 0.5,
            'diff_in_diff': 0.5, 'dose_response_corr': 0.9,
            'dose_response_lag': 2, 'granger_f': 0.5, 'granger_p': 0.30,
            'impact_score': 0.02,
        }])
        result = analyzer._assign_roles_phase2(df)
        assert result.iloc[0]['role'] == 'inert'

    def test_disruptor_zero_did(self, analyzer):
        """Significant Granger + DID exactly 0 → disruptor (<=0)."""
        df = pd.DataFrame([{
            'cluster_id': 1, 'frame': 'Eco', 'cluster_strength': 0.5,
            'diff_in_diff': 0.0, 'dose_response_corr': 0.3,
            'dose_response_lag': 2, 'granger_f': 5.0, 'granger_p': 0.03,
            'impact_score': 0.05,
        }])
        result = analyzer._assign_roles_phase2(df)
        assert result.iloc[0]['role'] == 'disruptor'


# ---------------------------------------------------------------------------
# TestPhase3Roles
# ---------------------------------------------------------------------------
class TestPhase3Roles:

    def test_amplification(self, analyzer):
        """Own frame + significant + positive DID → amplification."""
        df = pd.DataFrame([{
            'cascade_id': 'c1', 'frame': 'Eco', 'cascade_frame': 'Eco',
            'cascade_score': 0.6, 'diff_in_diff': 0.2,
            'dose_response_corr': 0.4, 'dose_response_lag': 3,
            'granger_f': 6.0, 'granger_p': 0.01, 'impact_score': 0.10,
        }])
        result = analyzer._assign_roles_phase3(df)
        assert result.iloc[0]['role'] == 'amplification'
        assert result.iloc[0]['is_own_frame'] == True

    def test_destabilisation_own_frame_negative(self, analyzer):
        """Own frame + significant + negative DID → destabilisation."""
        df = pd.DataFrame([{
            'cascade_id': 'c1', 'frame': 'Eco', 'cascade_frame': 'Eco',
            'cascade_score': 0.6, 'diff_in_diff': -0.1,
            'dose_response_corr': 0.4, 'dose_response_lag': 3,
            'granger_f': 6.0, 'granger_p': 0.02, 'impact_score': 0.08,
        }])
        result = analyzer._assign_roles_phase3(df)
        assert result.iloc[0]['role'] == 'destabilisation'

    def test_destabilisation_other_frame(self, analyzer):
        """Other frame + significant → destabilisation."""
        df = pd.DataFrame([{
            'cascade_id': 'c1', 'frame': 'Envt', 'cascade_frame': 'Eco',
            'cascade_score': 0.6, 'diff_in_diff': 0.3,
            'dose_response_corr': 0.5, 'dose_response_lag': 2,
            'granger_f': 8.0, 'granger_p': 0.01, 'impact_score': 0.12,
        }])
        result = analyzer._assign_roles_phase3(df)
        assert result.iloc[0]['role'] == 'destabilisation'
        assert result.iloc[0]['is_own_frame'] == False

    def test_dormant(self, analyzer):
        """Non-significant Granger → dormant."""
        df = pd.DataFrame([{
            'cascade_id': 'c1', 'frame': 'Eco', 'cascade_frame': 'Eco',
            'cascade_score': 0.6, 'diff_in_diff': 0.5,
            'dose_response_corr': 0.8, 'dose_response_lag': 1,
            'granger_f': 0.3, 'granger_p': 0.60, 'impact_score': 0.02,
        }])
        result = analyzer._assign_roles_phase3(df)
        assert result.iloc[0]['role'] == 'dormant'

    def test_is_own_frame_column(self, analyzer):
        """is_own_frame correctly identifies own vs other frames."""
        df = pd.DataFrame([
            {'cascade_id': 'c1', 'frame': 'Eco', 'cascade_frame': 'Eco',
             'cascade_score': 0.5, 'diff_in_diff': 0.1,
             'dose_response_corr': 0.3, 'dose_response_lag': 1,
             'granger_f': 1.0, 'granger_p': 0.20, 'impact_score': 0.03},
            {'cascade_id': 'c1', 'frame': 'Envt', 'cascade_frame': 'Eco',
             'cascade_score': 0.5, 'diff_in_diff': 0.1,
             'dose_response_corr': 0.3, 'dose_response_lag': 1,
             'granger_f': 1.0, 'granger_p': 0.20, 'impact_score': 0.03},
        ])
        result = analyzer._assign_roles_phase3(df)
        assert result.iloc[0]['is_own_frame'] == True
        assert result.iloc[1]['is_own_frame'] == False


# ---------------------------------------------------------------------------
# TestCascadeRoleAggregation
# ---------------------------------------------------------------------------
class TestCascadeRoleAggregation:

    def test_amplification_priority(self, analyzer):
        """Own-frame amplification → cascade is 'amplification'."""
        df = pd.DataFrame([
            {'cascade_id': 'c1', 'frame': 'Eco', 'cascade_frame': 'Eco',
             'role': 'amplification', 'is_own_frame': True, 'impact_score': 0.1},
            {'cascade_id': 'c1', 'frame': 'Envt', 'cascade_frame': 'Eco',
             'role': 'destabilisation', 'is_own_frame': False, 'impact_score': 0.05},
        ])
        roles = analyzer._aggregate_cascade_roles(df)
        assert roles['c1'] == 'amplification'

    def test_destabilisation_without_amplification(self, analyzer):
        """Any destabilisation but no own-frame amplification → destabilisation."""
        df = pd.DataFrame([
            {'cascade_id': 'c1', 'frame': 'Eco', 'cascade_frame': 'Eco',
             'role': 'dormant', 'is_own_frame': True, 'impact_score': 0.01},
            {'cascade_id': 'c1', 'frame': 'Envt', 'cascade_frame': 'Eco',
             'role': 'destabilisation', 'is_own_frame': False, 'impact_score': 0.08},
        ])
        roles = analyzer._aggregate_cascade_roles(df)
        assert roles['c1'] == 'destabilisation'

    def test_dormant_fallback(self, analyzer):
        """No amplification and no destabilisation → dormant."""
        df = pd.DataFrame([
            {'cascade_id': 'c1', 'frame': 'Eco', 'cascade_frame': 'Eco',
             'role': 'dormant', 'is_own_frame': True, 'impact_score': 0.01},
            {'cascade_id': 'c1', 'frame': 'Envt', 'cascade_frame': 'Eco',
             'role': 'dormant', 'is_own_frame': False, 'impact_score': 0.005},
        ])
        roles = analyzer._aggregate_cascade_roles(df)
        assert roles['c1'] == 'dormant'

    def test_multiple_cascades(self, analyzer):
        """Each cascade gets its own role."""
        df = pd.DataFrame([
            {'cascade_id': 'c1', 'frame': 'Eco', 'cascade_frame': 'Eco',
             'role': 'amplification', 'is_own_frame': True, 'impact_score': 0.1},
            {'cascade_id': 'c2', 'frame': 'Pol', 'cascade_frame': 'Pol',
             'role': 'dormant', 'is_own_frame': True, 'impact_score': 0.005},
        ])
        roles = analyzer._aggregate_cascade_roles(df)
        assert roles['c1'] == 'amplification'
        assert roles['c2'] == 'dormant'

    def test_empty(self, analyzer):
        """Empty DataFrame → empty dict."""
        df = pd.DataFrame(columns=['cascade_id', 'role', 'is_own_frame'])
        roles = analyzer._aggregate_cascade_roles(df)
        assert roles == {}


# ---------------------------------------------------------------------------
# TestSummaryEnrichment
# ---------------------------------------------------------------------------
class TestSummaryEnrichment:

    def test_summary_has_role_keys(self, analyzer, articles, paradigm_timeline):
        """Summary dict includes role/label distributions."""
        occ = _make_occurrence(1, 'evt_weather', '2018-02-15',
                               {i: 0.8 for i in range(1, 20)})
        cluster = _make_cluster(1, [occ], '2018-02-15', strength=0.5)
        daily_comp = _make_daily_composite()
        cascade = _make_cascade('c1', 'Eco', '2018-02-15',
                                 daily_composite=daily_comp)

        result = analyzer.run_from_components(
            clusters=[cluster], cascades=[cascade],
            articles=articles, paradigm_timeline=paradigm_timeline,
        )

        for key in ('phase1_roles', 'phase2_roles', 'phase3_roles',
                     'phase3_cascade_roles',
                     'phase1_impact_labels', 'phase2_impact_labels',
                     'phase3_impact_labels'):
            assert key in result.summary, f"Missing key: {key}"

    def test_summary_role_values(self, analyzer, articles, paradigm_timeline):
        """Role distributions contain valid role names."""
        occ = _make_occurrence(1, 'evt_weather', '2018-02-15',
                               {i: 0.8 for i in range(1, 20)})
        cluster = _make_cluster(1, [occ], '2018-02-15', strength=0.5)
        daily_comp = _make_daily_composite()
        cascade = _make_cascade('c1', 'Eco', '2018-02-15',
                                 daily_composite=daily_comp)

        result = analyzer.run_from_components(
            clusters=[cluster], cascades=[cascade],
            articles=articles, paradigm_timeline=paradigm_timeline,
        )

        p1_valid = {'driver', 'suppressor', 'late_support', 'neutral', 'unrelated'}
        p2_valid = {'catalyst', 'disruptor', 'inert'}
        p3_valid = {'amplification', 'destabilisation', 'dormant'}

        assert set(result.summary['phase1_roles'].keys()).issubset(p1_valid)
        assert set(result.summary['phase2_roles'].keys()).issubset(p2_valid)
        assert set(result.summary['phase3_roles'].keys()).issubset(p3_valid)

    def test_summary_empty_inputs(self, analyzer):
        """Empty inputs produce empty role/label dicts."""
        result = analyzer.run_from_components(
            clusters=[], cascades=[],
            articles=pd.DataFrame(), paradigm_timeline=None,
        )
        assert result.summary['phase1_roles'] == {}
        assert result.summary['phase2_roles'] == {}
        assert result.summary['phase3_roles'] == {}
        assert result.summary['phase3_cascade_roles'] == {}

    def test_cascade_roles_in_summary(self, analyzer, articles, paradigm_timeline):
        """phase3_cascade_roles maps each cascade_id to a valid role."""
        occ = _make_occurrence(1, 'evt_weather', '2018-02-15',
                               {i: 0.8 for i in range(1, 20)})
        cluster = _make_cluster(1, [occ], '2018-02-15', strength=0.5)
        daily_comp = _make_daily_composite()
        cascade = _make_cascade('c1', 'Eco', '2018-02-15',
                                 daily_composite=daily_comp)

        result = analyzer.run_from_components(
            clusters=[cluster], cascades=[cascade],
            articles=articles, paradigm_timeline=paradigm_timeline,
        )

        cascade_roles = result.summary['phase3_cascade_roles']
        assert 'c1' in cascade_roles
        assert cascade_roles['c1'] in {'amplification', 'destabilisation', 'dormant'}


# ---------------------------------------------------------------------------
# TestLateSupport
# ---------------------------------------------------------------------------
class TestLateSupport:
    """Tests for late support reclassification of post-peak suppressors."""

    @staticmethod
    def _make_phase1_df(rows):
        """Build a Phase 1 DataFrame from a list of row dicts."""
        return pd.DataFrame(rows)

    @staticmethod
    def _base_driver():
        """Base driver row dict with all needed columns."""
        return {
            'cluster_id': 1, 'cascade_id': 'c1', 'cascade_frame': 'Pbh',
            'cluster_strength': 0.5, 'proximity': 0.8,
            'diff_in_diff': 0.097, 'dose_response_corr': 0.5,
            'dose_response_lag': 2, 'article_overlap': 0.3,
            'impact_score': 0.10, 'is_post_peak': False,
            'frame_affinity': 0.200,
            'did_p_value': 0.01, 'xcorr_p_value': 0.02,
            'granger_f': 3.0, 'granger_p': 0.05,
            'content_relevance': 0.7,
        }

    def test_post_peak_high_affinity_reclassified(self, analyzer):
        """C316-like: post-peak suppressor with high frame_affinity → late_support."""
        driver = self._base_driver()
        suppressor = {
            'cluster_id': 2, 'cascade_id': 'c1', 'cascade_frame': 'Pbh',
            'cluster_strength': 0.5, 'proximity': 0.6,
            'diff_in_diff': -0.109, 'dose_response_corr': 0.3,
            'dose_response_lag': 5, 'article_overlap': 0.2,
            'impact_score': 0.08, 'is_post_peak': True,
            'frame_affinity': 0.200,  # above LATE_SUPPORT_MIN_AFFINITY=0.10
            'did_p_value': 0.02, 'xcorr_p_value': 0.05,
            'granger_f': 2.0, 'granger_p': 0.08,
            'content_relevance': 0.6,
        }
        df = self._make_phase1_df([driver, suppressor])
        df = analyzer._assign_roles_phase1(df)
        assert df.iloc[0]['role'] == 'driver'
        assert df.iloc[1]['role'] == 'suppressor'

        # No embedding alignment column → frame affinity alone decides
        df = analyzer._reclassify_late_support(df)
        assert df.iloc[0]['role'] == 'driver'
        assert df.iloc[1]['role'] == 'late_support'

    def test_post_peak_low_affinity_stays_suppressor(self, analyzer):
        """C322-like: post-peak suppressor with low frame_affinity stays suppressor."""
        driver = self._base_driver()
        suppressor = {
            'cluster_id': 3, 'cascade_id': 'c1', 'cascade_frame': 'Pbh',
            'cluster_strength': 0.5, 'proximity': 0.5,
            'diff_in_diff': -0.275, 'dose_response_corr': 0.2,
            'dose_response_lag': 3, 'article_overlap': 0.15,
            'impact_score': 0.06, 'is_post_peak': True,
            'frame_affinity': 0.040,  # below LATE_SUPPORT_MIN_AFFINITY=0.10
            'did_p_value': 0.03, 'xcorr_p_value': 0.10,
            'granger_f': 1.0, 'granger_p': 0.20,
            'content_relevance': 0.4,
        }
        df = self._make_phase1_df([driver, suppressor])
        df = analyzer._assign_roles_phase1(df)
        df = analyzer._reclassify_late_support(df)
        assert df.iloc[1]['role'] == 'suppressor'

    def test_pre_peak_suppressor_unchanged(self, analyzer):
        """Pre-peak suppressor with high affinity stays suppressor."""
        driver = self._base_driver()
        suppressor = {
            'cluster_id': 2, 'cascade_id': 'c1', 'cascade_frame': 'Pbh',
            'cluster_strength': 0.5, 'proximity': 0.7,
            'diff_in_diff': -0.15, 'dose_response_corr': 0.4,
            'dose_response_lag': 3, 'article_overlap': 0.25,
            'impact_score': 0.09, 'is_post_peak': False,
            'frame_affinity': 0.200,
            'did_p_value': 0.01, 'xcorr_p_value': 0.03,
            'granger_f': 3.0, 'granger_p': 0.05,
            'content_relevance': 0.7,
        }
        df = self._make_phase1_df([driver, suppressor])
        df = analyzer._assign_roles_phase1(df)
        df = analyzer._reclassify_late_support(df)
        assert df.iloc[1]['role'] == 'suppressor'

    def test_no_drivers_no_reclassification(self, analyzer):
        """No driver reference → all suppressors stay."""
        row = {
            'cluster_id': 1, 'cascade_id': 'c1', 'cascade_frame': 'Eco',
            'cluster_strength': 0.5, 'proximity': 0.5,
            'diff_in_diff': -0.10, 'dose_response_corr': 0.3,
            'dose_response_lag': 2, 'article_overlap': 0.2,
            'impact_score': 0.08, 'is_post_peak': True,
            'frame_affinity': 0.300,
            'did_p_value': 0.01, 'xcorr_p_value': 0.03,
            'granger_f': 2.0, 'granger_p': 0.08,
            'content_relevance': 0.8,
        }
        df = self._make_phase1_df([row])
        df = analyzer._assign_roles_phase1(df)
        assert df.iloc[0]['role'] == 'suppressor'
        df = analyzer._reclassify_late_support(df)
        assert df.iloc[0]['role'] == 'suppressor'

    def test_low_embedding_alignment_stays_suppressor(self, analyzer):
        """High affinity but low cosine sim → stays suppressor."""
        driver = self._base_driver()
        suppressor = {
            'cluster_id': 2, 'cascade_id': 'c1', 'cascade_frame': 'Pbh',
            'cluster_strength': 0.5, 'proximity': 0.6,
            'diff_in_diff': -0.109, 'dose_response_corr': 0.3,
            'dose_response_lag': 5, 'article_overlap': 0.2,
            'impact_score': 0.08, 'is_post_peak': True,
            'frame_affinity': 0.200,
            'did_p_value': 0.02, 'xcorr_p_value': 0.05,
            'granger_f': 2.0, 'granger_p': 0.08,
            'content_relevance': 0.6,
        }
        df = self._make_phase1_df([driver, suppressor])
        df = analyzer._assign_roles_phase1(df)
        # Manually add embedding_alignment column with low value
        df['embedding_alignment'] = [0.5, 0.10]  # 0.10 < 0.30 threshold
        df = analyzer._reclassify_late_support(df)
        assert df.iloc[1]['role'] == 'suppressor'

    def test_missing_alignment_column_frame_only(self, analyzer):
        """No embedding_alignment column → frame affinity alone decides."""
        driver = {
            'cluster_id': 1, 'cascade_id': 'c1', 'cascade_frame': 'Eco',
            'cluster_strength': 0.5, 'proximity': 0.8,
            'diff_in_diff': 0.05, 'dose_response_corr': 0.5,
            'dose_response_lag': 2, 'article_overlap': 0.3,
            'impact_score': 0.10, 'is_post_peak': False,
            'frame_affinity': 0.15,
            'did_p_value': 0.01, 'xcorr_p_value': 0.02,
            'granger_f': 3.0, 'granger_p': 0.05,
            'content_relevance': 0.7,
        }
        suppressor = {
            'cluster_id': 2, 'cascade_id': 'c1', 'cascade_frame': 'Eco',
            'cluster_strength': 0.5, 'proximity': 0.6,
            'diff_in_diff': -0.08, 'dose_response_corr': 0.3,
            'dose_response_lag': 5, 'article_overlap': 0.2,
            'impact_score': 0.06, 'is_post_peak': True,
            'frame_affinity': 0.15,
            'did_p_value': 0.02, 'xcorr_p_value': 0.05,
            'granger_f': 2.0, 'granger_p': 0.08,
            'content_relevance': 0.6,
        }
        df = self._make_phase1_df([driver, suppressor])
        df = analyzer._assign_roles_phase1(df)
        # No embedding_alignment column at all
        df = analyzer._reclassify_late_support(df)
        assert df.iloc[1]['role'] == 'late_support'

    def test_empty_dataframe(self, analyzer):
        """Empty input → no error."""
        df = pd.DataFrame(columns=[
            'cluster_id', 'cascade_id', 'cascade_frame', 'cluster_strength',
            'proximity', 'diff_in_diff', 'dose_response_corr',
            'dose_response_lag', 'impact_score', 'is_post_peak',
            'frame_affinity', 'role', 'impact_label',
        ])
        result = analyzer._reclassify_late_support(df)
        assert result.empty

    def test_nan_alignment_treated_as_pass(self, analyzer):
        """NaN embedding_alignment → frame affinity decides."""
        driver = self._base_driver()
        suppressor = {
            'cluster_id': 2, 'cascade_id': 'c1', 'cascade_frame': 'Pbh',
            'cluster_strength': 0.5, 'proximity': 0.6,
            'diff_in_diff': -0.109, 'dose_response_corr': 0.3,
            'dose_response_lag': 5, 'article_overlap': 0.2,
            'impact_score': 0.08, 'is_post_peak': True,
            'frame_affinity': 0.200,
            'did_p_value': 0.02, 'xcorr_p_value': 0.05,
            'granger_f': 2.0, 'granger_p': 0.08,
            'content_relevance': 0.6,
        }
        df = self._make_phase1_df([driver, suppressor])
        df = analyzer._assign_roles_phase1(df)
        df['embedding_alignment'] = [0.5, np.nan]  # NaN → passes alignment check
        df = analyzer._reclassify_late_support(df)
        assert df.iloc[1]['role'] == 'late_support'


# ---------------------------------------------------------------------------
# Overlap soft gate
# ---------------------------------------------------------------------------
class TestOverlapSoftGate:
    """Phase 1 overlap soft gate: floor at P1_OVERLAP_FLOOR."""

    @pytest.fixture()
    def analyzer(self):
        return UnifiedImpactAnalyzer()

    def test_zero_overlap_nonzero_impact(self, analyzer):
        """Cluster articles completely disjoint from cascade → impact > 0."""
        occ = _make_occurrence(1, 'evt_weather', '2018-02-15',
                               {d: 0.8 for d in range(1001, 1021)})
        cluster = _make_cluster(1, [occ], '2018-02-15', strength=0.5)
        cascade = _make_cascade('c1', 'Eco', '2018-02-15')
        # Articles 1-200 → no overlap with doc_ids 1001-1020
        articles = _make_articles(n=200)
        for col in FRAME_COLUMNS.values():
            articles[col] = 0.1

        df = analyzer.phase1_cluster_cascade([cluster], [cascade], articles)
        assert len(df) == 1
        row = df.iloc[0]
        assert row['article_overlap'] == 0.0
        assert row['impact_score'] > 0.0

    def test_full_overlap_unchanged(self, analyzer):
        """overlap=1.0 → effective_overlap=1.0, no attenuation."""
        # Use doc_ids within article range so overlap is high
        occ = _make_occurrence(1, 'evt_weather', '2018-02-15',
                               {d: 0.8 for d in range(1, 11)})
        cluster = _make_cluster(1, [occ], '2018-02-15', strength=0.5)
        cascade = _make_cascade('c1', 'Eco', '2018-02-15')
        articles = _make_articles(n=200)
        for col in FRAME_COLUMNS.values():
            articles[col] = 0.5

        df = analyzer.phase1_cluster_cascade([cluster], [cascade], articles)
        row = df.iloc[0]
        overlap = row['article_overlap']
        assert overlap > 0.9  # near-full overlap
        effective = P1_OVERLAP_FLOOR + (1 - P1_OVERLAP_FLOOR) * overlap
        assert abs(effective - 1.0) < 0.1  # close to 1.0

    def test_partial_overlap_monotonic(self, analyzer):
        """Higher overlap → higher impact score (all else equal)."""
        cascade = _make_cascade('c1', 'Eco', '2018-02-15')
        articles = _make_articles(n=200)
        for col in FRAME_COLUMNS.values():
            articles[col] = 0.5

        # Low overlap: most doc_ids outside article range
        occ_low = _make_occurrence(1, 'evt_weather', '2018-02-15',
                                   {**{d: 0.8 for d in range(1001, 1019)},
                                    **{d: 0.8 for d in range(1, 3)}})
        cluster_low = _make_cluster(1, [occ_low], '2018-02-15', strength=0.5)

        # High overlap: all doc_ids inside article range
        occ_high = _make_occurrence(2, 'evt_weather', '2018-02-15',
                                    {d: 0.8 for d in range(1, 21)})
        cluster_high = _make_cluster(2, [occ_high], '2018-02-15', strength=0.5)

        df_low = analyzer.phase1_cluster_cascade([cluster_low], [cascade], articles)
        df_high = analyzer.phase1_cluster_cascade([cluster_high], [cascade], articles)

        assert df_low.iloc[0]['article_overlap'] < df_high.iloc[0]['article_overlap']
        assert df_low.iloc[0]['impact_score'] < df_high.iloc[0]['impact_score']


# ---------------------------------------------------------------------------
# TestPermutation (lightweight — small N)
# ---------------------------------------------------------------------------
class TestPermutation:

    def test_basic_permutation(self, analyzer, articles):
        """Permutation test runs and returns valid p-values."""
        occ = _make_occurrence(1, 'evt_weather', '2018-02-15',
                               {i: 0.8 for i in range(1, 20)})
        cluster = _make_cluster(1, [occ], '2018-02-15', strength=0.5)
        daily_comp = _make_daily_composite('2018-01-15', '2018-02-15', '2018-03-15')
        cascade = _make_cascade('c1', 'Eco', '2018-02-15',
                                 daily_composite=daily_comp)

        df = analyzer.phase1_cluster_cascade([cluster], [cascade], articles)
        raw_p, adj_p = analyzer._compute_permutation_pvalues(
            df, [cluster], [cascade], articles, n_permutations=20,
        )
        assert len(raw_p) == len(df)
        assert len(adj_p) == len(df)
        assert all(0 <= p <= 1 for p in raw_p)
        assert all(0 <= p <= 1 for p in adj_p)

    def test_empty_df(self, analyzer, articles):
        """Empty DataFrame → empty arrays."""
        df = pd.DataFrame()
        raw_p, adj_p = analyzer._compute_permutation_pvalues(
            df, [], [], articles, n_permutations=10,
        )
        assert len(raw_p) == 0
        assert len(adj_p) == 0
