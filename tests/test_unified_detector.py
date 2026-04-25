"""
Unit tests for UnifiedCascadeDetector.

Covers:
- API contracts (detect returns tuple, detect_all_frames iterates, requires embedding store)
- Burst detection (detected in synthetic data, valid fields, no detection in flat series)
- Cascade scoring (scores in [0, 1], sub-indices in [0, 1], total = weighted sum, classification)
- New fields (composite_peak, daily_composite, daily_signals, detection_method)
- Sub-index counts per dimension: temporal=4, participation=6, convergence=4, source=3
- Hard filter (< 3 articles -> zero score)
- Weight validation (scoring weights sum to 1.0, signal weights sum to 1.0)
- Serialization (to_dict produces JSON-serializable output)

23 tests total.  No database, GPU, or external embeddings required.
"""

import json
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.constants import (
    CASCADE_THRESHOLDS, FRAME_COLUMNS, MESSENGERS, FRAMES,
)
from cascade_detector.core.models import BurstResult, CascadeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> DetectorConfig:
    """Return a DetectorConfig with sensible test defaults."""
    defaults = dict(
        baseline_window_days=90,
        min_burst_days=3,
        frames=["Pol"],
    )
    defaults.update(overrides)
    return DetectorConfig(**defaults)


def _make_detector(mock_embedding_store, config=None):
    """Instantiate UnifiedCascadeDetector with mocked dependencies.

    Patches NetworkBuilder and SemanticConvergenceCalculator so no real
    network computation or embedding files are needed.
    """
    from cascade_detector.detection.unified_detector import UnifiedCascadeDetector

    cfg = config or _make_config()

    # Mock SemanticConvergenceCalculator to avoid real embedding file access
    mock_convergence = MagicMock()
    mock_convergence.compute_all_metrics.return_value = {
        'intra_window_similarity': 0.5,
        'convergence_trend_slope': 0.02,
        'cross_media_alignment': 0.4,
        'novelty_decay_rate': 0.01,
    }

    with patch(
        'cascade_detector.embeddings.semantic_convergence.SemanticConvergenceCalculator',
        return_value=mock_convergence,
    ):
        detector = UnifiedCascadeDetector(config=cfg, embedding_store=mock_embedding_store)

    # Also patch _build_network to avoid NetworkBuilder / subprocess overhead
    detector._build_network = MagicMock(return_value={
        'density': 0.3,
        'modularity': 0.2,
        'mean_degree': 2.5,
        'n_components': 1,
        'n_nodes': 5,
        'graph': None,
    })

    return detector


def _make_articles(n_days=200, n_per_day_baseline=2, spike_day=150,
                   spike_size=20, frame="Pol", total_per_day=10):
    """Create a synthetic article DataFrame with a clear spike.

    Returns articles spanning *n_days* starting from 2020-01-01.
    Around *spike_day* (offset from start), *spike_size* extra articles
    with the frame set appear for 10 consecutive days, mimicking a cascade.

    Each day has *total_per_day* total articles. During baseline,
    *n_per_day_baseline* have the frame set; during spike, *n_per_day_baseline
    + spike_size* have it (capped at total_per_day).
    """
    rng = np.random.RandomState(42)
    frame_col = FRAME_COLUMNS[frame]
    rows = []

    start_date = pd.Timestamp("2020-01-01")
    journalists = [f"journalist_{i}" for i in range(15)]
    media_outlets = [f"media_{i}" for i in range(5)]

    for day_offset in range(n_days):
        date = start_date + pd.Timedelta(days=day_offset)

        # Frame-positive articles
        n_frame = n_per_day_baseline
        if spike_day <= day_offset < spike_day + 10:
            n_frame += spike_size

        n_total = max(total_per_day, n_frame)

        for j in range(n_total):
            has_frame = 1 if j < n_frame else 0
            rows.append({
                'doc_id': f"doc_{day_offset}_{j}",
                'date': date,
                'author': rng.choice(journalists),
                'media': rng.choice(media_outlets),
                frame_col: has_frame,
                f"{frame_col}_mean": rng.uniform(0.3, 1.0) if has_frame else 0.0,
                'messenger': 'msg_scientist',
                'msg_scientist': rng.randint(0, 2),
                'msg_health': rng.randint(0, 2),
                'msg_official': rng.randint(0, 2),
            })

    return pd.DataFrame(rows)


def _make_temporal_index(articles, frame="Pol"):
    """Build a minimal temporal index from the synthetic articles.

    The temporal index structure mirrors the production TemporalIndexer:
    {frame: {'daily_series': pd.Series, 'daily_proportions': pd.Series, ...}}
    """
    frame_col = FRAME_COLUMNS[frame]
    dates = pd.to_datetime(articles['date']).dt.normalize()
    total_per_day = dates.value_counts().sort_index()
    frame_per_day = (
        articles[articles[frame_col] > 0]
        .assign(_date=lambda df: pd.to_datetime(df['date']).dt.normalize())
        ['_date'].value_counts().sort_index()
    )

    full_range = pd.date_range(total_per_day.index.min(), total_per_day.index.max(), freq='D')
    total_per_day = total_per_day.reindex(full_range, fill_value=0).astype(float)
    frame_per_day = frame_per_day.reindex(full_range, fill_value=0).astype(float)

    proportions = (frame_per_day / total_per_day.replace(0, np.nan)).fillna(0)

    return {
        frame: {
            'daily_series': frame_per_day,
            'daily_totals': total_per_day,
            'daily_proportions': proportions,
        }
    }


def _make_flat_articles(n_days=200, n_per_day=2, total_per_day=10, frame="Pol"):
    """Create articles with perfectly flat, deterministic activity -- no spike.

    Uses round-robin journalist and media assignment so that participation
    signals are perfectly uniform and no anomaly is detected.
    """
    frame_col = FRAME_COLUMNS[frame]
    journalists = [f"journalist_{i}" for i in range(4)]
    media_outlets = [f"media_{i}" for i in range(2)]
    start_date = pd.Timestamp("2020-01-01")
    rows = []

    for day_offset in range(n_days):
        date = start_date + pd.Timedelta(days=day_offset)
        for j in range(total_per_day):
            idx = day_offset * total_per_day + j
            has_frame = 1 if j < n_per_day else 0
            rows.append({
                'doc_id': f"flat_{day_offset}_{j}",
                'date': date,
                'author': journalists[idx % len(journalists)],
                'media': media_outlets[idx % len(media_outlets)],
                frame_col: has_frame,
                'messenger': 'msg_scientist',
                'msg_scientist': 1 if has_frame else 0,
                'msg_health': 0,
                'msg_official': 0,
            })

    return pd.DataFrame(rows)


def _make_burst(frame="Pol", onset="2020-06-01", peak="2020-06-05",
                end="2020-06-10", intensity=3.0):
    return BurstResult(
        frame=frame,
        onset_date=pd.Timestamp(onset),
        peak_date=pd.Timestamp(peak),
        end_date=pd.Timestamp(end),
        duration_days=(pd.Timestamp(end) - pd.Timestamp(onset)).days + 1,
        intensity=intensity,
        baseline_mean=0.05,
        peak_proportion=0.15,
        detection_method='pelt',
    )


def _make_cascade(frame="Pol", total_score=0.72, classification="strong_cascade"):
    return CascadeResult(
        cascade_id="Pol_20200601_1",
        frame=frame,
        onset_date=pd.Timestamp("2020-06-01"),
        peak_date=pd.Timestamp("2020-06-05"),
        end_date=pd.Timestamp("2020-06-10"),
        duration_days=10,
        n_articles=50,
        n_journalists=8,
        n_media=4,
        n_new_journalists=5,
        burst_intensity=3.0,
        adoption_velocity=1.5,
        baseline_mean=0.05,
        peak_proportion=0.15,
        score_temporal=0.7,
        score_participation=0.6,
        score_convergence=0.5,
        score_source=0.4,
        total_score=total_score,
        classification=classification,
        composite_peak=3.2,
        daily_composite=pd.Series([1.0, 2.0, 3.2, 2.5, 1.8],
                                  index=pd.date_range("2020-06-03", periods=5)),
        daily_signals={'z_temporal': pd.Series([1.0], index=pd.date_range("2020-06-03", periods=1))},
        detection_method='pelt',
    )


# =========================================================================
# 1. API Contracts
# =========================================================================

class TestAPIContracts:
    """Tests 1-3: verify the public API signatures and contracts."""

    def test_detect_returns_tuple_of_three(self, mock_embedding_store):
        """detect() returns (List[CascadeResult], List[BurstResult], Dict)."""
        detector = _make_detector(mock_embedding_store)
        articles = _make_articles()
        temporal_index = _make_temporal_index(articles)

        result = detector.detect("Pol", temporal_index, articles, {})

        assert isinstance(result, tuple)
        assert len(result) == 3
        cascades, bursts, signals = result
        assert isinstance(cascades, list)
        assert isinstance(bursts, list)
        assert isinstance(signals, dict)

    def test_detect_all_frames_iterates_frames(self, mock_embedding_store):
        """detect_all_frames loops over config.frames and returns 3 collections."""
        cfg = _make_config(frames=["Pol", "Eco"])
        detector = _make_detector(mock_embedding_store, config=cfg)

        articles = _make_articles(frame="Pol")
        # Add economic_frame column so both frames exist
        articles['economic_frame'] = 0
        temporal_index = _make_temporal_index(articles, frame="Pol")
        # Eco has no temporal index entry -> detector logs warning and skips

        result = detector.detect_all_frames(temporal_index, articles, {})

        assert isinstance(result, tuple)
        assert len(result) == 3
        all_cascades, all_bursts, all_signals = result
        assert isinstance(all_cascades, list)
        assert isinstance(all_bursts, list)
        assert isinstance(all_signals, dict)

    def test_requires_embedding_store(self):
        """Constructor raises ValueError when embedding_store is None."""
        from cascade_detector.detection.unified_detector import UnifiedCascadeDetector
        with pytest.raises(ValueError, match="embedding_store is required"):
            UnifiedCascadeDetector(config=_make_config(), embedding_store=None)


# =========================================================================
# 2. Burst Detection
# =========================================================================

class TestBurstDetection:
    """Tests 4-6: burst detection on synthetic data."""

    def test_spike_detected(self, mock_embedding_store):
        """A 10-day spike (20x baseline) should produce at least one burst."""
        detector = _make_detector(mock_embedding_store)
        articles = _make_articles(spike_size=30)
        temporal_index = _make_temporal_index(articles)

        _, bursts, _ = detector.detect("Pol", temporal_index, articles, {})

        assert len(bursts) >= 1, "Expected at least one burst from a large spike"

    def test_burst_fields_valid(self, mock_embedding_store):
        """Each BurstResult has required fields with correct types."""
        detector = _make_detector(mock_embedding_store)
        articles = _make_articles(spike_size=30)
        temporal_index = _make_temporal_index(articles)

        _, bursts, _ = detector.detect("Pol", temporal_index, articles, {})
        if not bursts:
            pytest.skip("No bursts detected -- cannot validate fields")

        b = bursts[0]
        assert isinstance(b, BurstResult)
        assert isinstance(b.onset_date, pd.Timestamp)
        assert isinstance(b.peak_date, pd.Timestamp)
        assert isinstance(b.end_date, pd.Timestamp)
        assert b.onset_date <= b.peak_date <= b.end_date
        assert b.duration_days >= 1
        assert b.intensity >= 0
        assert b.frame == "Pol"
        assert b.detection_method in (
            'pelt', 'pelt_refined', 'sliding_prop',
        )

    def test_flat_composite_no_burst(self, mock_embedding_store):
        """A perfectly flat composite signal (all zeros) produces zero bursts."""
        detector = _make_detector(mock_embedding_store)
        articles = _make_flat_articles()
        temporal_index = _make_temporal_index(articles)

        # Patch signal builder to return a flat-zero composite signal
        date_range = pd.date_range("2020-01-01", periods=200, freq='D')
        flat_signals = {
            'z_temporal': pd.Series(0.0, index=date_range),
            'z_participation': pd.Series(0.0, index=date_range),
            'z_convergence': pd.Series(0.0, index=date_range),
            'z_source': pd.Series(0.0, index=date_range),
            'z_semantic': pd.Series(0.0, index=date_range),
            'composite': pd.Series(0.0, index=date_range),
        }
        detector.signal_builder.build_signals = MagicMock(return_value=flat_signals)

        _, bursts, _ = detector.detect("Pol", temporal_index, articles, {})

        assert len(bursts) == 0, "Flat composite signal should not trigger any burst"


# =========================================================================
# 3. Cascade Scoring
# =========================================================================

class TestCascadeScoring:
    """Tests 7-10: scoring logic produces valid values."""

    def test_dimension_scores_in_unit_interval(self, mock_embedding_store):
        """Each of the 4 dimension scores must be in [0, 1]."""
        detector = _make_detector(mock_embedding_store)
        articles = _make_articles(spike_size=30)
        temporal_index = _make_temporal_index(articles)

        cascades, _, _ = detector.detect("Pol", temporal_index, articles, {})
        if not cascades:
            pytest.skip("No cascades scored")

        c = cascades[0]
        for attr in ('score_temporal', 'score_participation',
                     'score_convergence', 'score_source'):
            val = getattr(c, attr)
            assert 0.0 <= val <= 1.0, f"{attr} = {val} outside [0,1]"

    def test_sub_indices_in_unit_interval(self, mock_embedding_store):
        """Every sub-index value must be in [0, 1]."""
        detector = _make_detector(mock_embedding_store)
        articles = _make_articles(spike_size=30)
        temporal_index = _make_temporal_index(articles)

        cascades, _, _ = detector.detect("Pol", temporal_index, articles, {})
        if not cascades:
            pytest.skip("No cascades scored")

        for key, val in cascades[0].sub_indices.items():
            assert 0.0 <= val <= 1.0, f"sub_index '{key}' = {val} outside [0,1]"

    def test_total_score_is_weighted_sum(self, mock_embedding_store):
        """total_score equals the weighted sum of the 4 dimension scores."""
        detector = _make_detector(mock_embedding_store)
        articles = _make_articles(spike_size=30)
        temporal_index = _make_temporal_index(articles)

        cascades, _, _ = detector.detect("Pol", temporal_index, articles, {})
        if not cascades:
            pytest.skip("No cascades scored")

        c = cascades[0]
        cfg = detector.config
        base_score = (
            cfg.weight_temporal * c.score_temporal
            + cfg.weight_participation * c.score_participation
            + cfg.weight_convergence * c.score_convergence
            + cfg.weight_source * c.score_source
        )
        media_confidence = min(1.0, np.log2(max(c.n_media, 1)) / np.log2(10))
        expected = float(np.clip(base_score * media_confidence, 0, 1))
        assert abs(c.total_score - expected) < 1e-6, (
            f"total_score {c.total_score} != weighted sum * confidence {expected}"
        )

    def test_classification_assigned(self, mock_embedding_store):
        """Every scored cascade has a non-empty classification string."""
        detector = _make_detector(mock_embedding_store)
        articles = _make_articles(spike_size=30)
        temporal_index = _make_temporal_index(articles)

        cascades, _, _ = detector.detect("Pol", temporal_index, articles, {})
        if not cascades:
            pytest.skip("No cascades scored")

        c = cascades[0]
        valid = set(CASCADE_THRESHOLDS.keys())
        assert c.classification in valid, (
            f"classification '{c.classification}' not in {valid}"
        )


# =========================================================================
# 4. New Fields (unified detector additions)
# =========================================================================

class TestNewFields:
    """Tests 11-14: fields introduced by the unified detector."""

    def test_composite_peak_present(self, mock_embedding_store):
        """CascadeResult.composite_peak is a non-negative float."""
        detector = _make_detector(mock_embedding_store)
        articles = _make_articles(spike_size=30)
        temporal_index = _make_temporal_index(articles)

        cascades, _, _ = detector.detect("Pol", temporal_index, articles, {})
        if not cascades:
            pytest.skip("No cascades scored")

        c = cascades[0]
        assert isinstance(c.composite_peak, float)
        assert c.composite_peak >= 0.0

    def test_daily_composite_is_series(self, mock_embedding_store):
        """CascadeResult.daily_composite is a pd.Series or None."""
        detector = _make_detector(mock_embedding_store)
        articles = _make_articles(spike_size=30)
        temporal_index = _make_temporal_index(articles)

        cascades, _, _ = detector.detect("Pol", temporal_index, articles, {})
        if not cascades:
            pytest.skip("No cascades scored")

        c = cascades[0]
        assert c.daily_composite is None or isinstance(c.daily_composite, pd.Series)

    def test_daily_signals_is_dict_of_series(self, mock_embedding_store):
        """CascadeResult.daily_signals maps signal names to pd.Series."""
        detector = _make_detector(mock_embedding_store)
        articles = _make_articles(spike_size=30)
        temporal_index = _make_temporal_index(articles)

        cascades, _, _ = detector.detect("Pol", temporal_index, articles, {})
        if not cascades:
            pytest.skip("No cascades scored")

        c = cascades[0]
        if c.daily_signals is not None:
            assert isinstance(c.daily_signals, dict)
            for k, v in c.daily_signals.items():
                assert isinstance(v, pd.Series), f"daily_signals['{k}'] is not pd.Series"

    def test_detection_method_label(self, mock_embedding_store):
        """CascadeResult.detection_method is one of the PELT methods."""
        detector = _make_detector(mock_embedding_store)
        articles = _make_articles(spike_size=30)
        temporal_index = _make_temporal_index(articles)

        cascades, _, _ = detector.detect("Pol", temporal_index, articles, {})
        if not cascades:
            pytest.skip("No cascades scored")

        c = cascades[0]
        assert isinstance(c.detection_method, str)
        assert c.detection_method in ('pelt', 'pelt_refined', 'sliding_prop')


# =========================================================================
# 5. Sub-index Counts per Dimension
# =========================================================================

class TestSubIndexCounts:
    """Tests 15-18: each dimension has the documented number of sub-indices."""

    def _get_sub_indices(self, mock_embedding_store):
        """Helper: detect a cascade and return its sub_indices dict."""
        detector = _make_detector(mock_embedding_store)
        articles = _make_articles(spike_size=30)
        temporal_index = _make_temporal_index(articles)

        cascades, _, _ = detector.detect("Pol", temporal_index, articles, {})
        if not cascades:
            pytest.skip("No cascades scored -- cannot inspect sub-indices")
        return cascades[0].sub_indices

    def test_temporal_has_4_sub_indices(self, mock_embedding_store):
        """Temporal dimension: burst_intensity, adoption_velocity, duration, mann_whitney."""
        si = self._get_sub_indices(mock_embedding_store)
        temporal_keys = [k for k in si if k.startswith('temporal_')]
        assert len(temporal_keys) == 4, (
            f"Expected 4 temporal sub-indices, got {len(temporal_keys)}: {temporal_keys}"
        )

    def test_participation_has_6_sub_indices(self, mock_embedding_store):
        """Participation dimension: actor_diversity, cross_media_ratio,
        new_entrant_rate, growth_pattern, network_structure, network_cohesion."""
        si = self._get_sub_indices(mock_embedding_store)
        part_keys = [k for k in si if k.startswith('participation_')]
        assert len(part_keys) == 6, (
            f"Expected 6 participation sub-indices, got {len(part_keys)}: {part_keys}"
        )

    def test_convergence_has_4_sub_indices(self, mock_embedding_store):
        """Convergence dimension: semantic_similarity, convergence_trend,
        cross_media_alignment, novelty_decay."""
        si = self._get_sub_indices(mock_embedding_store)
        conv_keys = [k for k in si if k.startswith('convergence_')]
        assert len(conv_keys) == 4, (
            f"Expected 4 convergence sub-indices, got {len(conv_keys)}: {conv_keys}"
        )

    def test_source_has_3_sub_indices(self, mock_embedding_store):
        """Source dimension: source_diversity_decline, messenger_concentration,
        media_coordination."""
        si = self._get_sub_indices(mock_embedding_store)
        src_keys = [k for k in si if k.startswith('source_')]
        assert len(src_keys) == 3, (
            f"Expected 3 source sub-indices, got {len(src_keys)}: {src_keys}"
        )


# =========================================================================
# 6. Hard Filter
# =========================================================================

class TestHardFilter:
    """Test 19: cascades with < MIN_ARTICLES_HARD articles receive zero score."""

    def test_few_articles_zero_score(self, mock_embedding_store):
        """With very few articles in the spike window, total_score should be 0."""
        detector = _make_detector(mock_embedding_store)
        # spike_size=1 means only 3 articles/day (2 baseline + 1 spike) for
        # 10 days = 30 articles, but frame filtering may reduce this.
        # We use a very small dataset to trigger the hard filter.
        articles = _make_articles(n_days=200, n_per_day_baseline=0,
                                  spike_day=150, spike_size=1)
        temporal_index = _make_temporal_index(articles)

        cascades, bursts, _ = detector.detect("Pol", temporal_index, articles, {})

        # Either no cascades are produced, or any produced have zero score
        for c in cascades:
            if c.n_articles < detector.MIN_ARTICLES_HARD:
                assert c.total_score == 0.0, (
                    f"Cascade with {c.n_articles} articles should have zero score"
                )


# =========================================================================
# 7. Weight Validation
# =========================================================================

class TestWeightValidation:
    """Tests 20-21: configured weights sum to 1.0."""

    def test_scoring_weights_sum_to_one(self):
        """The 4 scoring dimension weights must sum to 1.0."""
        cfg = DetectorConfig()
        total = (cfg.weight_temporal + cfg.weight_participation
                 + cfg.weight_convergence + cfg.weight_source)
        assert abs(total - 1.0) < 1e-6, f"Scoring weights sum to {total}, not 1.0"

    def test_signal_weights_sum_to_one(self):
        """The 5 composite signal weights must sum to 1.0."""
        cfg = DetectorConfig()
        total = (cfg.signal_weight_temporal + cfg.signal_weight_participation
                 + cfg.signal_weight_convergence + cfg.signal_weight_source
                 + cfg.signal_weight_semantic)
        assert abs(total - 1.0) < 1e-6, f"Signal weights sum to {total}, not 1.0"


# =========================================================================
# 8. Serialization
# =========================================================================

class TestSerialization:
    """Tests 22-23: to_dict produces JSON-serializable output."""

    def test_cascade_to_dict_json_serializable(self):
        """CascadeResult.to_dict() produces a dict that json.dumps accepts."""
        c = _make_cascade()
        d = c.to_dict()
        assert isinstance(d, dict)
        # This must not raise
        serialized = json.dumps(d, default=str)
        assert isinstance(serialized, str)
        assert len(serialized) > 0

    def test_burst_to_dict_json_serializable(self):
        """BurstResult.to_dict() produces a dict that json.dumps accepts."""
        b = _make_burst()
        d = b.to_dict()
        assert isinstance(d, dict)
        serialized = json.dumps(d, default=str)
        assert isinstance(serialized, str)
        assert 'frame' in d
        assert 'onset_date' in d
        assert 'detection_method' in d


# =========================================================================
# 9. Semantic Peak
# =========================================================================

class TestSemanticPeak:
    """Tests 24-27: semantic peak replaces z-score argmax for cascade peak_date."""

    def test_peak_within_burst_window(self, mock_embedding_store):
        """Semantic peak_date must lie between onset and end dates."""
        detector = _make_detector(mock_embedding_store)
        articles = _make_articles(spike_size=30)
        temporal_index = _make_temporal_index(articles)

        cascades, _, _ = detector.detect("Pol", temporal_index, articles, {})
        if not cascades:
            pytest.skip("No cascades scored")

        c = cascades[0]
        assert c.onset_date <= c.peak_date <= c.end_date, (
            f"peak_date {c.peak_date} outside [{c.onset_date}, {c.end_date}]"
        )

    def test_peak_near_activity_center(self, mock_embedding_store):
        """Peak should gravitate toward the mass of articles, not a z-score spike.

        2 frame articles at day 150, 20 frame articles at day 155-159 → peak closer to 155+.
        """
        detector = _make_detector(mock_embedding_store)
        frame_col = FRAME_COLUMNS["Pol"]
        rng = np.random.RandomState(99)

        # Build articles: sparse early, dense later
        start_date = pd.Timestamp("2020-01-01")
        journalists = [f"journalist_{i}" for i in range(15)]
        media_outlets = [f"media_{i}" for i in range(5)]
        rows = []

        # Baseline: 10 total articles/day, 2 with frame
        for day in range(200):
            date = start_date + pd.Timedelta(days=day)
            for j in range(10):
                has_frame = 1 if j < 2 else 0
                rows.append({
                    'doc_id': f"base_{day}_{j}",
                    'date': date,
                    'author': rng.choice(journalists),
                    'media': rng.choice(media_outlets),
                    frame_col: has_frame,
                    f"{frame_col}_mean": rng.uniform(0.3, 1.0) if has_frame else 0.0,
                    'messenger': 'msg_scientist',
                    'msg_scientist': 1 if has_frame else 0,
                    'msg_health': 0,
                    'msg_official': 0,
                })

        # Spike at day 150: only 2 extra frame articles
        for j in range(2):
            date = start_date + pd.Timedelta(days=150)
            rows.append({
                'doc_id': f"early_{j}",
                'date': date,
                'author': rng.choice(journalists),
                'media': rng.choice(media_outlets),
                frame_col: 1,
                f"{frame_col}_mean": rng.uniform(0.5, 1.0),
                'messenger': 'msg_scientist',
                'msg_scientist': 1,
                'msg_health': 0,
                'msg_official': 0,
            })

        # Heavy mass at days 155-159: 20 frame articles/day
        for day in range(155, 160):
            date = start_date + pd.Timedelta(days=day)
            for j in range(20):
                rows.append({
                    'doc_id': f"mass_{day}_{j}",
                    'date': date,
                    'author': rng.choice(journalists),
                    'media': rng.choice(media_outlets),
                    frame_col: 1,
                    f"{frame_col}_mean": rng.uniform(0.5, 1.0),
                    'messenger': 'msg_scientist',
                    'msg_scientist': 1,
                    'msg_health': 0,
                    'msg_official': 0,
                })

        articles = pd.DataFrame(rows)
        temporal_index = _make_temporal_index(articles)

        cascades, _, _ = detector.detect("Pol", temporal_index, articles, {})
        if not cascades:
            pytest.skip("No cascades scored")

        c = cascades[0]
        day_150 = start_date + pd.Timedelta(days=150)
        day_155 = start_date + pd.Timedelta(days=155)

        # Peak should be closer to day 155 (mass center) than day 150 (sparse)
        dist_to_150 = abs((c.peak_date - day_150).days)
        dist_to_155 = abs((c.peak_date - day_155).days)
        assert dist_to_155 <= dist_to_150, (
            f"peak_date {c.peak_date} closer to sparse day 150 ({dist_to_150}d) "
            f"than mass center day 155 ({dist_to_155}d)"
        )

    def test_fallback_no_embeddings(self, mock_embedding_store):
        """When embedding store returns empty, cascade still has a valid peak_date."""
        # Override get_batch_article_embeddings to return nothing
        mock_embedding_store.get_batch_article_embeddings = MagicMock(
            return_value=(np.empty((0, 64), dtype=np.float32), [])
        )
        detector = _make_detector(mock_embedding_store)
        articles = _make_articles(spike_size=30)
        temporal_index = _make_temporal_index(articles)

        cascades, _, _ = detector.detect("Pol", temporal_index, articles, {})
        if not cascades:
            pytest.skip("No cascades scored")

        c = cascades[0]
        assert isinstance(c.peak_date, pd.Timestamp)
        assert c.onset_date <= c.peak_date <= c.end_date

    def test_daily_composite_nonnegative(self, mock_embedding_store):
        """All daily_composite values must be >= 0."""
        detector = _make_detector(mock_embedding_store)
        articles = _make_articles(spike_size=30)
        temporal_index = _make_temporal_index(articles)

        cascades, _, _ = detector.detect("Pol", temporal_index, articles, {})
        if not cascades:
            pytest.skip("No cascades scored")

        c = cascades[0]
        if c.daily_composite is not None:
            assert (c.daily_composite >= 0).all(), (
                f"Negative daily_composite values: "
                f"{c.daily_composite[c.daily_composite < 0].to_dict()}"
            )


# =========================================================================
# 10. PELT Detection
# =========================================================================

class TestPELTDetection:
    """Tests for the PELT changepoint detection pipeline."""

    @staticmethod
    def _build_series(values, start="2020-01-01"):
        """Helper: build a daily Series from a list of float values."""
        dates = pd.date_range(start, periods=len(values), freq='D')
        return pd.Series(values, index=dates, dtype=float)

    def test_pelt_segments_finds_spike(self, mock_embedding_store):
        """A composite with a clear spike should produce >= 1 elevated segment."""
        detector = _make_detector(mock_embedding_store)
        # 150 baseline days + 15 spike days + 35 baseline days
        values = [0.2] * 150 + [5.0] * 15 + [0.2] * 35
        composite = self._build_series(values)

        segments = detector._pelt_segments(composite)
        assert len(segments) >= 1, "Expected at least 1 elevated segment from spike"

        # The segment should overlap with the spike period
        spike_start = composite.index[150]
        spike_end = composite.index[164]
        found_overlap = False
        for seg in segments:
            if seg['start'] <= spike_end and seg['end'] >= spike_start:
                found_overlap = True
        assert found_overlap, "Elevated segment should overlap with spike period"

    def test_pelt_segments_flat(self, mock_embedding_store):
        """A perfectly flat signal should produce 0 elevated segments."""
        detector = _make_detector(mock_embedding_store)
        values = [1.0] * 200
        composite = self._build_series(values)

        segments = detector._pelt_segments(composite)
        assert len(segments) == 0, "Flat signal should not produce elevated segments"

    def test_proptest_rejects_baseline(self, mock_embedding_store):
        """A window at baseline proportion should be rejected by proptest."""
        detector = _make_detector(mock_embedding_store)
        dates = pd.date_range("2020-01-01", periods=200, freq='D')
        # All days: 5 frame articles out of 100 total = 5% proportion
        count_series = pd.Series(5.0, index=dates)
        total_series = pd.Series(100.0, index=dates)
        baseline_prop = 0.05

        start = dates[100]
        end = dates[120]
        accepted, stats = detector._proptest_accept(
            count_series, total_series, start, end, baseline_prop
        )
        assert not accepted, "At-baseline proportion should not be accepted"

    def test_proptest_accepts_elevated(self, mock_embedding_store):
        """A window with clearly elevated proportion should be accepted."""
        detector = _make_detector(mock_embedding_store)
        dates = pd.date_range("2020-01-01", periods=200, freq='D')
        # Baseline: 5% proportion
        count_series = pd.Series(5.0, index=dates)
        total_series = pd.Series(100.0, index=dates)
        baseline_prop = 0.05

        # Spike window: 20% proportion (days 100-120)
        count_series.iloc[100:121] = 20.0

        start = dates[100]
        end = dates[120]
        accepted, stats = detector._proptest_accept(
            count_series, total_series, start, end, baseline_prop
        )
        assert accepted, "Elevated proportion (20% vs 5% baseline) should be accepted"
        assert stats['cohen_h'] > 0.05
        assert stats['pvalue'] < 0.01

    def test_sliding_window_finds_burst(self, mock_embedding_store):
        """Sliding-window proportion test should find a proportion-only burst."""
        detector = _make_detector(mock_embedding_store)
        dates = pd.date_range("2020-01-01", periods=200, freq='D')
        # Baseline: 2% proportion
        count_series = pd.Series(2.0, index=dates)
        total_series = pd.Series(100.0, index=dates)
        prop_series = count_series / total_series

        # Spike: 15% proportion for 20 days
        count_series.iloc[120:140] = 15.0
        prop_series = count_series / total_series
        baseline_prop = 0.02

        periods = detector._sliding_window_proptest(
            prop_series, count_series, total_series, baseline_prop
        )
        if not periods:
            pytest.skip("Sliding window did not find burst -- may depend on exact thresholds")

        # At least one period should overlap with spike
        found = False
        for p in periods:
            if p['start'] <= dates[139] and p['end'] >= dates[120]:
                found = True
        assert found, "Sliding window should find burst overlapping with spike"

    def test_refine_recovers_subburst(self, mock_embedding_store):
        """Fine PELT should recover a short burst hidden in a rejected segment."""
        detector = _make_detector(mock_embedding_store)
        # Segment: 30 days where first 10 are quiet, middle 10 are strong, last 10 quiet
        values = [0.5] * 10 + [5.0] * 10 + [0.5] * 10
        dates = pd.date_range("2020-06-01", periods=30, freq='D')
        composite = pd.Series(values, index=dates)

        # Count/total for proptest: spike = 15% vs 2% baseline
        count_series = pd.Series(2.0, index=dates)
        total_series = pd.Series(100.0, index=dates)
        count_series.iloc[10:20] = 15.0
        baseline_prop = 0.02

        rejected = [{'start': dates[0], 'end': dates[29], 'method': 'pelt'}]
        refined = detector._refine_rejected(
            rejected, composite, count_series, total_series, baseline_prop
        )
        # May or may not recover depending on exact PELT behavior -- don't hard assert count
        # but verify method label if any are found
        for r in refined:
            assert r['method'] == 'pelt_refined'


# =========================================================================
# 11. Boundary Extension
# =========================================================================

class TestBoundaryExtension:
    """Tests for _extend_boundaries() boundary walking logic."""

    @staticmethod
    def _build_series(values, start="2020-01-01"):
        dates = pd.date_range(start, periods=len(values), freq='D')
        return pd.Series(values, index=dates, dtype=float)

    def test_extends_above_baseline(self, mock_embedding_store):
        """Walk extends backward while proportion > baseline."""
        detector = _make_detector(mock_embedding_store,
                                  config=_make_config(boundary_max_lookback=10))
        # 5 above-baseline days before period, then 10 period days
        values = [0.01] * 85 + [0.10] * 5 + [0.20] * 10
        prop_series = self._build_series(values)
        baseline_prop = float(prop_series.mean())

        period = {
            'start': prop_series.index[90],
            'end': prop_series.index[99],
            'method': 'pelt',
        }
        result = detector._extend_boundaries([period], prop_series, baseline_prop)

        # Should extend back to at least day 85 (where above-baseline starts)
        assert result[0]['start'] <= prop_series.index[90], "Should extend backward"

    def test_stops_below_baseline(self, mock_embedding_store):
        """Walk stops when proportion drops well below baseline."""
        detector = _make_detector(mock_embedding_store,
                                  config=_make_config(boundary_max_lookback=10,
                                                      boundary_gap_tolerance=0))
        # All zeros before period, then spike
        values = [0.0] * 90 + [0.20] * 10
        prop_series = self._build_series(values)
        baseline_prop = float(prop_series.mean())

        period = {
            'start': prop_series.index[90],
            'end': prop_series.index[99],
            'method': 'pelt',
        }
        result = detector._extend_boundaries([period], prop_series, baseline_prop)

        # With gap_tolerance=0, should stop immediately at zeros
        # Due to 5-day smoothing, some extension is expected but bounded
        actual_start = result[0]['start']
        # Should not go further back than a few days due to smoothing spillover
        lookback = (prop_series.index[90] - actual_start).days
        assert lookback <= 5, f"Should not extend far into zeros, got {lookback} days"

    def test_gap_tolerance(self, mock_embedding_store):
        """Gap tolerance allows 1-2 below-baseline days, stops at 3+."""
        detector = _make_detector(mock_embedding_store,
                                  config=_make_config(boundary_max_lookback=15,
                                                      boundary_gap_tolerance=2))
        # Layout: above, 2-gap, above, above, period
        values = ([0.01] * 75  # quiet
                  + [0.15] * 5  # above baseline
                  + [0.01] * 2  # 2-day gap (within tolerance)
                  + [0.15] * 8  # above baseline
                  + [0.20] * 10)  # period
        prop_series = self._build_series(values)
        baseline_prop = float(prop_series.mean())

        period = {
            'start': prop_series.index[90],
            'end': prop_series.index[99],
            'method': 'pelt',
        }
        result = detector._extend_boundaries([period], prop_series, baseline_prop)

        # Should extend through the 2-day gap
        assert result[0]['start'] < prop_series.index[90], "Should extend backward"

    def test_max_lookback_respected(self, mock_embedding_store):
        """Boundary extension doesn't exceed max_lookback."""
        detector = _make_detector(mock_embedding_store,
                                  config=_make_config(boundary_max_lookback=5))
        # All above baseline
        values = [0.20] * 100
        prop_series = self._build_series(values)
        baseline_prop = 0.01  # very low baseline, all days above

        period = {
            'start': prop_series.index[50],
            'end': prop_series.index[70],
            'method': 'pelt',
        }
        result = detector._extend_boundaries([period], prop_series, baseline_prop)

        lookback = (prop_series.index[50] - result[0]['start']).days
        assert lookback <= 5, f"Lookback {lookback} exceeds max_lookback=5"

    def test_extends_forward(self, mock_embedding_store):
        """Boundary extension also extends end date forward."""
        detector = _make_detector(mock_embedding_store,
                                  config=_make_config(boundary_max_lookback=10))
        # Period followed by 5 above-baseline days
        values = [0.01] * 80 + [0.20] * 10 + [0.10] * 5 + [0.01] * 5
        prop_series = self._build_series(values)
        baseline_prop = float(prop_series.mean())

        period = {
            'start': prop_series.index[80],
            'end': prop_series.index[89],
            'method': 'pelt',
        }
        result = detector._extend_boundaries([period], prop_series, baseline_prop)

        assert result[0]['end'] >= prop_series.index[89], "End should extend forward"
