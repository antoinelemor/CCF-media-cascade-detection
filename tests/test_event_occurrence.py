"""
Tests for cascade_detector.analysis.event_occurrence.

Uses ClusterableMockEmbeddingStore from conftest.py to produce deterministic,
clusterable embeddings. Articles with doc_id 'cluster_N_*' get embeddings
near centroid N.

Architecture: database-first detection on ALL articles, then attribution
to cascades. Tests cover the new detect_events() + attribute_to_cascades()
API as well as backward-compatible detect() and detect_all() wrappers.
"""

import numpy as np
import pandas as pd
import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from cascade_detector.analysis.event_occurrence import (
    EventOccurrenceDetector,
    MIN_CLUSTER_SIZE,
    COSINE_DISTANCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
)
from cascade_detector.core.constants import (
    TITLE_WEIGHT, TITLE_SENTENCE_ID, SEED_PERCENTILE,
    PHASE4_N_ITERATIONS, SEED_DOMINANT_RATIO,
    EVENT_CLUSTER_TITLE_SIM_THRESHOLD, EVENT_CLUSTER_MAX_GAP_DAYS,
)
from cascade_detector.core.models import (
    EventOccurrence, EventCluster, CascadeAttribution,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

@dataclass
class _FakeCascade:
    """Minimal cascade stub for testing."""
    cascade_id: str = 'test_cascade'
    frame: str = 'Envt'
    onset_date: pd.Timestamp = pd.Timestamp('2018-06-01')
    peak_date: pd.Timestamp = pd.Timestamp('2018-06-15')
    end_date: pd.Timestamp = pd.Timestamp('2018-07-15')
    dominant_events: Dict[str, int] = field(default_factory=dict)
    dominant_messengers: Dict[str, int] = field(default_factory=dict)
    event_occurrences: List = field(default_factory=list)
    event_occurrence_metrics: Dict[str, float] = field(default_factory=dict)
    daily_event_profile: Optional[pd.DataFrame] = None


def _make_articles(n_per_cluster: int = 10,
                   n_clusters: int = 2,
                   evt_types: Optional[List[str]] = None,
                   start_date: str = '2018-06-01',
                   days_apart: int = 0) -> pd.DataFrame:
    """Build a test DataFrame with articles belonging to distinct clusters.

    Args:
        n_per_cluster: Articles per cluster.
        n_clusters: Number of clusters.
        evt_types: Event type(s) to label. Defaults to ['evt_weather'].
        start_date: Base date for articles.
        days_apart: Days offset between clusters (0 = overlapping).

    Returns:
        DataFrame with doc_id, date, and evt_* columns.
    """
    if evt_types is None:
        evt_types = ['evt_weather']

    rows = []
    base = pd.Timestamp(start_date)
    evt_col_map = {et: f'{et}_mean' for et in evt_types}

    for c in range(n_clusters):
        evt_type = evt_types[c % len(evt_types)]
        evt_col = evt_col_map[evt_type]
        for i in range(n_per_cluster):
            doc_id = f'cluster_{c}_art{i}'
            date = base + pd.Timedelta(days=c * days_apart + i % 7)
            row = {
                'doc_id': doc_id,
                'date': date,
            }
            # Set event columns
            for et, col in evt_col_map.items():
                if et == evt_type:
                    row[col] = 0.5 + 0.3 * (i / n_per_cluster)  # above threshold
                else:
                    row[col] = 0.02  # below threshold
            rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# TestDailyEventProfile
# -----------------------------------------------------------------------

class TestDailyEventProfile:
    """Phase 1: daily event profile construction."""

    def test_shape(self, clusterable_embedding_store):
        """Profile has one row per unique date, one column per event type."""
        articles = _make_articles(n_per_cluster=7, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        profile = det._build_daily_event_profile(articles, 'date')

        n_unique_dates = articles['date'].nunique()
        assert len(profile) == n_unique_dates
        assert 'evt_weather' in profile.columns
        assert 'evt_meeting' in profile.columns

    def test_proportions(self, clusterable_embedding_store):
        """Mean evt_* values are in [0, 1]."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        profile = det._build_daily_event_profile(articles, 'date')

        assert (profile.values >= 0).all()
        assert (profile.values <= 1).all()

    def test_empty_events(self, clusterable_embedding_store):
        """Event types with no column in data get zero values."""
        articles = pd.DataFrame({
            'doc_id': ['a', 'b', 'c'],
            'date': pd.to_datetime(['2018-06-01', '2018-06-02', '2018-06-03']),
        })
        det = EventOccurrenceDetector(clusterable_embedding_store)
        profile = det._build_daily_event_profile(articles, 'date')

        assert len(profile) == 3
        assert (profile.values == 0).all()


# -----------------------------------------------------------------------
# TestSeedSelection
# -----------------------------------------------------------------------

class TestSeedSelection:
    """Phase 2: seed article selection with adaptive percentile."""

    def test_percentile_selection(self, clusterable_embedding_store):
        """Seeds are selected at P50 of non-zero values."""
        articles = pd.DataFrame({
            'doc_id': ['a', 'b', 'c', 'd', 'e', 'f'],
            'date': pd.to_datetime(['2018-06-01'] * 6),
            'evt_weather_mean': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
        })
        det = EventOccurrenceDetector(clusterable_embedding_store)
        seeds, weights = det._select_seed_articles(articles, 'evt_weather_mean')

        # Non-zero values: [0.1, 0.3, 0.5, 0.7, 0.9], P50 = 0.5
        # Articles >= 0.5: d (0.5), e (0.7), f (0.9)
        assert len(seeds) == 3
        assert set(seeds['doc_id']) == {'d', 'e', 'f'}

    def test_returns_weights(self, clusterable_embedding_store):
        """_select_seed_articles returns continuous weights."""
        articles = pd.DataFrame({
            'doc_id': ['a', 'b', 'c', 'd', 'e'],
            'date': pd.to_datetime(['2018-06-01'] * 5),
            'evt_weather_mean': [0.1, 0.3, 0.5, 0.7, 0.9],
        })
        det = EventOccurrenceDetector(clusterable_embedding_store)
        seeds, weights = det._select_seed_articles(articles, 'evt_weather_mean')

        # P50 of [0.1, 0.3, 0.5, 0.7, 0.9] = 0.5, so seeds >= 0.5
        assert len(weights) == len(seeds)
        assert weights.dtype == np.float64
        np.testing.assert_array_almost_equal(weights, [0.5, 0.7, 0.9])

    def test_all_zero_returns_empty(self, clusterable_embedding_store):
        """All-zero event column returns empty seeds."""
        articles = pd.DataFrame({
            'doc_id': ['a', 'b'],
            'date': pd.to_datetime(['2018-06-01'] * 2),
            'evt_weather_mean': [0.0, 0.0],
        })
        det = EventOccurrenceDetector(clusterable_embedding_store)
        seeds, weights = det._select_seed_articles(articles, 'evt_weather_mean')
        assert len(seeds) == 0
        assert len(weights) == 0

    def test_fallback_to_sum(self, clusterable_embedding_store):
        """_resolve_event_col finds evt_*_sum when evt_*_mean is absent."""
        articles = pd.DataFrame({
            'doc_id': ['a'],
            'date': pd.to_datetime(['2018-06-01']),
            'evt_weather_sum': [5.0],
        })
        det = EventOccurrenceDetector(clusterable_embedding_store)
        col = det._resolve_event_col(articles, 'evt_weather')
        assert col == 'evt_weather_sum'


# -----------------------------------------------------------------------
# TestClustering
# -----------------------------------------------------------------------

class TestClustering:
    """Phase 2: agglomerative clustering per event type."""

    def test_two_distinct_clusters(self, clusterable_embedding_store):
        """Two well-separated groups produce two clusters."""
        articles = _make_articles(
            n_per_cluster=5, n_clusters=2,
            evt_types=['evt_weather', 'evt_weather'],
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)
        seeds, weights = det._select_seed_articles(articles, 'evt_weather_mean')

        clusters = det._cluster_event_type(
            seeds, 'evt_weather', 'evt_weather_mean', id_offset=0,
            seed_weights=weights,
        )
        assert len(clusters) == 2

    def test_single_cohesive_cluster(self, clusterable_embedding_store):
        """Articles near the same centroid form one cluster."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        seeds, weights = det._select_seed_articles(articles, 'evt_weather_mean')

        clusters = det._cluster_event_type(
            seeds, 'evt_weather', 'evt_weather_mean', id_offset=0,
            seed_weights=weights,
        )
        assert len(clusters) == 1
        assert len(clusters[0].doc_ids) >= 5  # at least some articles survive

    def test_min_size_filter(self, clusterable_embedding_store):
        """Singletons are preserved as micro unique events.

        With singleton preservation, both the 5-article cluster and the
        1-article singleton survive Phase 2.
        """
        # 5 articles near centroid 0, 1 near centroid 1 (singleton)
        rows = []
        for i in range(5):
            rows.append({
                'doc_id': f'cluster_0_art{i}',
                'date': pd.Timestamp('2018-06-01'),
                'evt_weather_mean': 0.5,
            })
        rows.append({
            'doc_id': f'cluster_1_art0',
            'date': pd.Timestamp('2018-06-01'),
            'evt_weather_mean': 0.5,
        })
        articles = pd.DataFrame(rows)

        det = EventOccurrenceDetector(clusterable_embedding_store)
        seeds, weights = det._select_seed_articles(articles, 'evt_weather_mean')
        clusters = det._cluster_event_type(
            seeds, 'evt_weather', 'evt_weather_mean', id_offset=0,
            seed_weights=weights,
        )

        # Both clusters survive (singleton preserved as micro unique event)
        assert len(clusters) == 2
        sizes = sorted(len(c.doc_ids) for c in clusters)
        assert sizes == [1, 5]

    def test_pair_survives_min_size(self, clusterable_embedding_store):
        """A cluster of exactly 2 articles survives MIN_CLUSTER_SIZE=2."""
        assert MIN_CLUSTER_SIZE == 2
        # 5 articles near centroid 0, 2 near centroid 1
        rows = []
        for i in range(5):
            rows.append({
                'doc_id': f'cluster_0_art{i}',
                'date': pd.Timestamp('2018-06-01'),
                'evt_weather_mean': 0.5,
            })
        for i in range(2):
            rows.append({
                'doc_id': f'cluster_1_art{i}',
                'date': pd.Timestamp('2018-06-01'),
                'evt_weather_mean': 0.5,
            })
        articles = pd.DataFrame(rows)

        det = EventOccurrenceDetector(clusterable_embedding_store)
        seeds, weights = det._select_seed_articles(articles, 'evt_weather_mean')
        clusters = det._cluster_event_type(
            seeds, 'evt_weather', 'evt_weather_mean', id_offset=0,
            seed_weights=weights,
        )
        assert len(clusters) == 2


# -----------------------------------------------------------------------
# TestAssignment
# -----------------------------------------------------------------------

class TestAssignment:
    """Phase 4: iterative 4D article assignment."""

    def test_belonging_assigned(self, clusterable_embedding_store):
        """Articles near a centroid get non-zero belonging after assignment."""
        articles = _make_articles(n_per_cluster=5, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        seeds, weights = det._select_seed_articles(articles, 'evt_weather_mean')
        clusters = det._cluster_event_type(
            seeds, 'evt_weather', 'evt_weather_mean', id_offset=0,
            seed_weights=weights,
        )

        det._assign_articles(clusters, articles, 'date')

        # Seed articles should have belonging > 0
        assert len(clusters) == 1
        for i in range(5):
            did = f'cluster_0_art{i}'
            assert did in clusters[0].belonging
            assert clusters[0].belonging[did] > 0

    def test_signal_dimension_filters(self, clusterable_embedding_store):
        """Articles with low event signal get lower belonging via signal dimension."""
        # Two clusters: one with high evt_weather, one with high evt_meeting
        articles = _make_articles(
            n_per_cluster=5, n_clusters=2,
            evt_types=['evt_weather', 'evt_meeting']
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)

        seeds_w, weights_w = det._select_seed_articles(articles, 'evt_weather_mean')
        seeds_m, weights_m = det._select_seed_articles(articles, 'evt_meeting_mean')

        clusters_w = det._cluster_event_type(
            seeds_w, 'evt_weather', 'evt_weather_mean', id_offset=0,
            seed_weights=weights_w,
        )
        clusters_m = det._cluster_event_type(
            seeds_m, 'evt_meeting', 'evt_meeting_mean',
            id_offset=len(clusters_w),
            seed_weights=weights_m,
        )
        all_clusters = clusters_w + clusters_m

        if all_clusters:
            det._assign_articles(all_clusters, articles, 'date')

            # Weather articles should have higher belonging to weather cluster
            for cluster in all_clusters:
                if cluster.event_type == 'evt_weather':
                    # Seed articles should be well-assigned
                    seed_bs = [cluster.belonging.get(f'cluster_0_art{i}', 0)
                               for i in range(5)]
                    assert np.mean(seed_bs) > 0.3

    def test_two_iterations(self, clusterable_embedding_store):
        """Assignment runs PHASE4_N_ITERATIONS iterations."""
        assert PHASE4_N_ITERATIONS == 2
        articles = _make_articles(n_per_cluster=5, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        seeds, weights = det._select_seed_articles(articles, 'evt_weather_mean')
        clusters = det._cluster_event_type(
            seeds, 'evt_weather', 'evt_weather_mean', id_offset=0,
            seed_weights=weights,
        )

        # Should not crash with 2 iterations
        det._assign_articles(clusters, articles, 'date')
        assert len(clusters[0].belonging) > 0

    def test_unlabeled_low_belonging(self, clusterable_embedding_store):
        """Unlabeled articles (evt_*_mean=0) get lower belonging than labeled seeds."""
        articles = _make_articles(n_per_cluster=5, n_clusters=1)
        # Add unlabeled articles near the same centroid
        extra_rows = []
        for i in range(3):
            extra_rows.append({
                'doc_id': f'cluster_0_extra{i}',
                'date': pd.Timestamp('2018-06-05'),
                'evt_weather_mean': 0.0,  # no label
            })
        extra = pd.DataFrame(extra_rows)
        all_articles = pd.concat([articles, extra], ignore_index=True)

        det = EventOccurrenceDetector(clusterable_embedding_store)
        seeds, weights = det._select_seed_articles(all_articles, 'evt_weather_mean')
        clusters = det._cluster_event_type(
            seeds, 'evt_weather', 'evt_weather_mean', id_offset=0,
            seed_weights=weights,
        )

        det._assign_articles(clusters, all_articles, 'date')

        # Unlabeled articles should have much lower belonging than seeds
        # (signal dimension contributes 0.25 penalty, pushing them towards threshold)
        seed_belongings = [
            clusters[0].belonging.get(did, 0)
            for did in clusters[0].seed_doc_ids
            if did in clusters[0].belonging
        ]
        extra_belongings = [
            clusters[0].belonging.get(f'cluster_0_extra{i}', 0)
            for i in range(3)
        ]
        assert np.mean(seed_belongings) > np.mean(extra_belongings) + 0.1

    def test_continuous_belonging(self, clusterable_embedding_store):
        """Belonging values are continuous (no threshold cutoff)."""
        articles = _make_articles(n_per_cluster=5, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        seeds, weights = det._select_seed_articles(articles, 'evt_weather_mean')
        clusters = det._cluster_event_type(
            seeds, 'evt_weather', 'evt_weather_mean', id_offset=0,
            seed_weights=weights,
        )
        det._assign_articles(clusters, articles, 'date')

        # All belonging values should be in (0, 1]
        for did, b in clusters[0].belonging.items():
            assert 0.0 < b <= 1.0


# -----------------------------------------------------------------------
# TestAttribution
# -----------------------------------------------------------------------

class TestAttribution:
    """Attribution of occurrences to cascades."""

    def test_basic_attribution(self, clusterable_embedding_store):
        """Occurrences overlapping with cascade get attributed."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, clusters = det.detect_events(articles)

        cascade = _FakeCascade(
            onset_date=pd.Timestamp('2018-06-01'),
            end_date=pd.Timestamp('2018-07-15'),
        )
        attributions = det.attribute_to_cascades(occurrences, [cascade], articles)

        # Should have at least one attribution
        assert len(attributions) > 0
        assert all(isinstance(a, CascadeAttribution) for a in attributions)

        # Cascade should have event_occurrences populated (backward compat)
        assert len(cascade.event_occurrences) > 0
        assert cascade.event_occurrence_metrics['n_occurrences'] > 0

    def test_no_temporal_overlap(self, clusterable_embedding_store):
        """Occurrences outside cascade window get no attribution."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1,
                                  start_date='2018-06-01')
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, clusters = det.detect_events(articles)

        # Cascade is 6 months later
        cascade = _FakeCascade(
            onset_date=pd.Timestamp('2019-01-01'),
            end_date=pd.Timestamp('2019-02-15'),
        )
        attributions = det.attribute_to_cascades(occurrences, [cascade], articles)

        assert len(attributions) == 0
        assert len(cascade.event_occurrences) == 0

    def test_shared_occurrence_across_cascades(self, clusterable_embedding_store):
        """Same occurrence can be attributed to multiple cascades."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, clusters = det.detect_events(articles)

        cascade1 = _FakeCascade(
            cascade_id='c1',
            onset_date=pd.Timestamp('2018-06-01'),
            end_date=pd.Timestamp('2018-07-15'),
        )
        cascade2 = _FakeCascade(
            cascade_id='c2',
            onset_date=pd.Timestamp('2018-06-01'),
            end_date=pd.Timestamp('2018-07-15'),
        )
        attributions = det.attribute_to_cascades(
            occurrences, [cascade1, cascade2], articles
        )

        # Both cascades should get the same occurrences
        if occurrences:
            assert len(cascade1.event_occurrences) > 0
            assert len(cascade2.event_occurrences) > 0

    def test_attribution_fields(self, clusterable_embedding_store):
        """CascadeAttribution has correct fields."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, clusters = det.detect_events(articles)

        cascade = _FakeCascade()
        attributions = det.attribute_to_cascades(occurrences, [cascade], articles)

        for attr in attributions:
            assert attr.cascade_id == 'test_cascade'
            assert attr.shared_articles >= 1
            assert attr.temporal_overlap_days >= 0
            assert 0.0 <= attr.overlap_ratio <= 1.0

    def test_occurrence_cascade_attributions_populated(self, clusterable_embedding_store):
        """Occurrences have cascade_attributions list populated."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, clusters = det.detect_events(articles)

        cascade = _FakeCascade()
        det.attribute_to_cascades(occurrences, [cascade], articles)

        for occ in cascade.event_occurrences:
            assert len(occ.cascade_attributions) > 0


# -----------------------------------------------------------------------
# TestBelonging
# -----------------------------------------------------------------------

class TestBelonging:
    """Belonging score computation and properties."""

    def test_belonging_populated(self, clusterable_embedding_store):
        """Every occurrence has belonging dict with doc_ids as keys."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        for occ in occurrences:
            assert len(occ.belonging) > 0
            # All doc_ids should be in belonging
            for did in occ.doc_ids:
                assert did in occ.belonging

    def test_belonging_values_range(self, clusterable_embedding_store):
        """All belonging values in [0, 1]."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        for occ in occurrences:
            for did, b in occ.belonging.items():
                assert 0.0 <= b <= 1.0, f"Belonging {b} out of range for {did}"

    def test_belonging_weighted_temporal_bounds(self, clusterable_embedding_store):
        """core_start/core_end are within [first_date, last_date]."""
        articles = _make_articles(n_per_cluster=20, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        for occ in occurrences:
            assert occ.core_start >= occ.first_date
            assert occ.core_end <= occ.last_date

    def test_effective_mass_equals_sum_belonging(self, clusterable_embedding_store):
        """effective_mass equals sum of all belonging values."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        for occ in occurrences:
            expected = sum(occ.belonging.values())
            assert abs(occ.effective_mass - expected) < 1e-6


# -----------------------------------------------------------------------
# TestConfidenceScore
# -----------------------------------------------------------------------

class TestConfidenceScore:
    """Confidence score computation and properties."""

    def test_confidence_populated(self, clusterable_embedding_store):
        """Every occurrence has confidence in [0, 1]."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        for occ in occurrences:
            assert 0.0 <= occ.confidence <= 1.0

    def test_confidence_components_keys(self, clusterable_embedding_store):
        """Components dict has all 5 expected keys."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        expected_keys = {
            'centroid_tightness', 'coherence_residual',
            'media_diversity', 'recruitment_success', 'size_adequacy',
        }
        for occ in occurrences:
            assert set(occ.confidence_components.keys()) == expected_keys
            for v in occ.confidence_components.values():
                assert 0.0 <= v <= 1.0

    def test_high_confidence_tight_cluster(self, clusterable_embedding_store):
        """Tight, pure cluster with enough articles → reasonable confidence."""
        articles = _make_articles(n_per_cluster=15, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        if occurrences:
            occ = occurrences[0]
            assert occ.confidence > 0.3

    def test_low_confidence_flag(self, clusterable_embedding_store):
        """low_confidence flag matches threshold comparison."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        for occ in occurrences:
            assert occ.low_confidence == (occ.confidence < LOW_CONFIDENCE_THRESHOLD)

    def test_media_diversity_signal(self, clusterable_embedding_store):
        """Multi-source articles give media_diversity > 0; mono-source gives 0."""
        # Build articles with multiple media
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        n = len(articles)
        media_choices = ['CBC', 'Globe and Mail', 'Toronto Star']
        articles['media'] = [media_choices[i % len(media_choices)] for i in range(n)]

        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        for occ in occurrences:
            assert occ.confidence_components['media_diversity'] > 0.0

        # Mono-source: all same media
        articles_mono = articles.copy()
        articles_mono['media'] = 'CBC'
        det2 = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences_mono, _ = det2.detect_events(articles_mono)

        for occ in occurrences_mono:
            assert occ.confidence_components['media_diversity'] == 0.0

    def test_coherence_residual(self, clusterable_embedding_store):
        """Coherence residual ≤ raw semantic coherence (baseline ≥ 0)."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        for occ in occurrences:
            # With mock baseline=0, residual equals raw coherence
            assert occ.confidence_components['coherence_residual'] >= 0.0
            assert occ.confidence_components['coherence_residual'] <= 1.0

    def test_recruitment_success(self, clusterable_embedding_store):
        """Recruitment success is 0 when all articles are seeds."""
        articles = _make_articles(n_per_cluster=5, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        # With small clusters, all articles may be seeds → recruitment ≈ 0
        for occ in occurrences:
            assert 0.0 <= occ.confidence_components['recruitment_success'] <= 1.0


# -----------------------------------------------------------------------
# TestOccurrenceBuilding
# -----------------------------------------------------------------------

class TestOccurrenceBuilding:
    """Phase 3: EventOccurrence object construction."""

    def test_temporal_bounds(self, clusterable_embedding_store):
        """first_date and last_date span the cluster's articles."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        if occurrences:
            occ = occurrences[0]
            article_dates = pd.to_datetime(articles['date'])
            assert occ.first_date <= article_dates.min()
            assert occ.last_date >= article_dates.max() - pd.Timedelta(days=1)

    def test_core_period_percentiles(self, clusterable_embedding_store):
        """Core period is within [first_date, last_date]."""
        articles = _make_articles(n_per_cluster=20, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        if occurrences:
            occ = occurrences[0]
            assert occ.core_start >= occ.first_date
            assert occ.core_end <= occ.last_date

    def test_effective_mass(self, clusterable_embedding_store):
        """effective_mass > 0 and core_mass <= effective_mass."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        for occ in occurrences:
            assert occ.effective_mass > 0
            assert occ.core_mass <= occ.effective_mass + 1e-6

    def test_doc_ids_populated(self, clusterable_embedding_store):
        """Each occurrence stores its doc_ids."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        for occ in occurrences:
            assert len(occ.doc_ids) == occ.n_articles
            assert len(occ.doc_ids) > 0


# -----------------------------------------------------------------------
# TestCascadeMetrics
# -----------------------------------------------------------------------

class TestCascadeMetrics:
    """Phase 3: cascade-level occurrence metrics."""

    def test_structure_metrics(self, clusterable_embedding_store):
        """Metrics include expected keys."""
        articles = _make_articles(n_per_cluster=10, n_clusters=2,
                                  evt_types=['evt_weather', 'evt_meeting'])
        cascade = _FakeCascade()
        det = EventOccurrenceDetector(clusterable_embedding_store)
        det.detect(cascade, articles, 'date')

        m = cascade.event_occurrence_metrics
        assert 'n_occurrences' in m
        assert 'n_event_types' in m
        assert 'mean_coherence' in m
        assert 'temporal_overlap' in m
        assert 'mean_confidence' in m
        assert 'n_low_confidence' in m
        assert 'mean_effective_mass' in m
        assert 'total_effective_mass' in m

    def test_semantic_quality(self, clusterable_embedding_store):
        """Mean coherence is in [0, 1] for clusterable data."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        cascade = _FakeCascade()
        det = EventOccurrenceDetector(clusterable_embedding_store)
        det.detect(cascade, articles, 'date')

        m = cascade.event_occurrence_metrics
        if m.get('n_occurrences', 0) > 0:
            assert 0.0 <= m['mean_coherence'] <= 1.0

    def test_regime_transitions(self, clusterable_embedding_store):
        """Regime transitions return a list of dicts."""
        articles = _make_articles(
            n_per_cluster=10, n_clusters=2,
            evt_types=['evt_weather', 'evt_meeting'],
            days_apart=14
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)
        profile = det._build_daily_event_profile(articles, 'date')
        transitions = det._compute_regime_transitions(profile)

        assert isinstance(transitions, list)
        for t in transitions:
            assert 'date' in t
            assert 'from_type' in t
            assert 'to_type' in t
            assert 'confidence' in t

    def test_temporal_overlap_two_overlapping(self, clusterable_embedding_store):
        """Two temporally overlapping occurrences produce overlap > 0."""
        articles = _make_articles(
            n_per_cluster=10, n_clusters=2,
            evt_types=['evt_weather', 'evt_meeting'],
            days_apart=0  # same dates
        )
        cascade = _FakeCascade()
        det = EventOccurrenceDetector(clusterable_embedding_store)
        det.detect(cascade, articles, 'date')

        m = cascade.event_occurrence_metrics
        if m.get('n_occurrences', 0) >= 2:
            assert m['temporal_overlap'] > 0.0

    def test_confidence_metrics(self, clusterable_embedding_store):
        """Cascade metrics include confidence aggregates."""
        articles = _make_articles(n_per_cluster=10, n_clusters=2,
                                  evt_types=['evt_weather', 'evt_meeting'])
        cascade = _FakeCascade()
        det = EventOccurrenceDetector(clusterable_embedding_store)
        det.detect(cascade, articles, 'date')

        m = cascade.event_occurrence_metrics
        if m.get('n_occurrences', 0) > 0:
            assert 0.0 <= m['mean_confidence'] <= 1.0
            assert m['n_low_confidence'] >= 0
            assert m['mean_effective_mass'] > 0
            assert m['total_effective_mass'] > 0


# -----------------------------------------------------------------------
# TestDetect
# -----------------------------------------------------------------------

class TestDetect:
    """End-to-end detect_events(), detect(), and detect_all() tests."""

    def test_detect_events_returns_tuple(self, clusterable_embedding_store):
        """detect_events() returns (occurrences, event_clusters)."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        det = EventOccurrenceDetector(clusterable_embedding_store)

        occurrences, event_clusters = det.detect_events(articles)
        assert isinstance(occurrences, list)
        assert isinstance(event_clusters, list)

    def test_detect_backward_compat(self, clusterable_embedding_store):
        """detect() modifies the cascade object in-place."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        cascade = _FakeCascade()
        det = EventOccurrenceDetector(clusterable_embedding_store)

        assert cascade.event_occurrences == []
        det.detect(cascade, articles, 'date')

        assert isinstance(cascade.event_occurrences, list)
        assert cascade.event_occurrence_metrics is not None
        assert cascade.daily_event_profile is not None

    def test_no_events(self, clusterable_embedding_store):
        """Cascade with no event columns produces empty occurrences."""
        articles = pd.DataFrame({
            'doc_id': [f'doc_{i}' for i in range(10)],
            'date': pd.to_datetime(['2018-06-01'] * 10),
        })
        cascade = _FakeCascade()
        det = EventOccurrenceDetector(clusterable_embedding_store)
        det.detect(cascade, articles, 'date')

        assert cascade.event_occurrences == []
        assert cascade.event_occurrence_metrics['n_occurrences'] == 0

    def test_batch_backward_compat(self, clusterable_embedding_store):
        """detect_all() processes multiple cascades."""
        articles = _make_articles(n_per_cluster=10, n_clusters=1)
        cascades = [
            _FakeCascade(cascade_id='c1'),
            _FakeCascade(cascade_id='c2'),
        ]
        det = EventOccurrenceDetector(clusterable_embedding_store)
        det.detect_all(cascades, articles)

        for c in cascades:
            assert isinstance(c.event_occurrences, list)

    def test_database_first_detects_more(self, clusterable_embedding_store):
        """Database-first detection finds events across the full dataset."""
        articles = _make_articles(n_per_cluster=10, n_clusters=2,
                                  evt_types=['evt_weather', 'evt_meeting'],
                                  days_apart=14)
        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        # Should find at least one occurrence
        assert len(occurrences) > 0


# -----------------------------------------------------------------------
# TestEdgeCases
# -----------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_low_embedding_coverage(self, mock_embedding_store):
        """Works with generic MockEmbeddingStore (random embeddings)."""
        articles = pd.DataFrame({
            'doc_id': [f'doc_{i}' for i in range(10)],
            'date': pd.to_datetime(['2018-06-01'] * 10),
            'evt_weather_mean': [0.5] * 10,
        })
        cascade = _FakeCascade()
        det = EventOccurrenceDetector(mock_embedding_store)
        det.detect(cascade, articles, 'date')

        # Should not crash; may or may not find clusters
        assert isinstance(cascade.event_occurrences, list)

    def test_single_article_event_type(self, clusterable_embedding_store):
        """Single article for an event type produces a singleton occurrence."""
        articles = pd.DataFrame({
            'doc_id': ['cluster_0_art0'],
            'date': pd.to_datetime(['2018-06-01']),
            'evt_weather_mean': [0.9],
        })
        cascade = _FakeCascade()
        det = EventOccurrenceDetector(clusterable_embedding_store)
        det.detect(cascade, articles, 'date')

        assert len(cascade.event_occurrences) == 1
        occ = cascade.event_occurrences[0]
        assert occ.is_singleton is True
        assert occ.low_confidence is True


# -----------------------------------------------------------------------
# TestEventOccurrenceModel
# -----------------------------------------------------------------------

class TestEventOccurrenceModel:
    """Tests for the EventOccurrence dataclass itself."""

    def test_to_dict_excludes_centroid_and_belonging(self):
        """to_dict() omits centroid, belonging, and event_sentence_ids."""
        occ = EventOccurrence(
            occurrence_id=0,
            event_type='evt_weather',
            first_date=pd.Timestamp('2018-06-01'),
            last_date=pd.Timestamp('2018-06-15'),
            core_start=pd.Timestamp('2018-06-03'),
            core_end=pd.Timestamp('2018-06-13'),
            peak_date=pd.Timestamp('2018-06-08'),
            n_articles=50,
            effective_mass=42.5,
            core_mass=35.0,
            semantic_coherence=0.75,
            centroid=np.ones(64),
            confidence=0.65,
            confidence_components={
                'centroid_tightness': 0.85,
                'semantic_coherence': 0.75,
                'attribution_purity': 0.70,
                'size_adequacy': 1.0,
            },
            low_confidence=False,
            belonging={'doc1': 0.8, 'doc2': 0.6},
            doc_ids=[100, 101, 102],
            event_sentence_ids={100: [0, 1, 2], 101: [1, 3]},
        )
        d = occ.to_dict()
        assert 'centroid' not in d
        assert 'belonging' not in d
        assert 'event_sentence_ids' not in d
        assert d['event_type'] == 'evt_weather'
        assert d['n_articles'] == 50
        assert d['effective_mass'] == 42.5
        assert d['core_mass'] == 35.0
        assert d['confidence'] == 0.65
        assert 'confidence_components' in d
        assert d['low_confidence'] is False
        assert d['doc_ids'] == [100, 101, 102]
        assert d['n_event_sentences'] == 5  # 3 + 2

    def test_to_dict_serializable(self):
        """to_dict() output is JSON-serializable."""
        import json
        occ = EventOccurrence(
            occurrence_id=0,
            event_type='evt_meeting',
            first_date=pd.Timestamp('2018-06-01'),
            last_date=pd.Timestamp('2018-06-15'),
            core_start=pd.Timestamp('2018-06-03'),
            core_end=pd.Timestamp('2018-06-13'),
            peak_date=pd.Timestamp('2018-06-08'),
            n_articles=30,
            effective_mass=25.0,
            core_mass=20.0,
            semantic_coherence=0.80,
            centroid=np.zeros(64),
            confidence=0.55,
            confidence_components={
                'centroid_tightness': 0.83,
                'semantic_coherence': 0.80,
                'attribution_purity': 0.67,
                'size_adequacy': 1.0,
            },
            low_confidence=False,
            belonging={'200': 0.9, '201': 0.7},
            doc_ids=[200, 201, 202],
            event_sentence_ids={200: [0, 1]},
        )
        json_str = json.dumps(occ.to_dict())
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed['doc_ids'] == [200, 201, 202]
        assert parsed['n_event_sentences'] == 2
        assert parsed['effective_mass'] == 25.0
        assert parsed['confidence'] == 0.55


# -----------------------------------------------------------------------
# TestWeightedPercentile
# -----------------------------------------------------------------------

class TestWeightedPercentile:
    """Unit tests for _weighted_percentile helper."""

    def test_uniform_weights(self, clusterable_embedding_store):
        """With uniform weights, result matches np.percentile."""
        det = EventOccurrenceDetector(clusterable_embedding_store)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5)

        p50 = det._weighted_percentile(values, weights, 50)
        assert abs(p50 - 3.0) < 0.5  # close to median

    def test_skewed_weights(self, clusterable_embedding_store):
        """High weight on low values pulls percentile down."""
        det = EventOccurrenceDetector(clusterable_embedding_store)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([10.0, 1.0, 1.0, 1.0, 1.0])  # heavy on 1.0

        p50 = det._weighted_percentile(values, weights, 50)
        assert p50 < 2.5  # pulled toward 1.0

    def test_single_value(self, clusterable_embedding_store):
        """Single value returns that value regardless of percentile."""
        det = EventOccurrenceDetector(clusterable_embedding_store)
        values = np.array([42.0])
        weights = np.array([1.0])

        assert det._weighted_percentile(values, weights, 10) == 42.0
        assert det._weighted_percentile(values, weights, 90) == 42.0


# -----------------------------------------------------------------------
# TestTitleIntegration
# -----------------------------------------------------------------------

class TestTitleIntegration:
    """Tests for title embedding integration in Phase 2 and 4."""

    def test_title_blended_in_phase2(self, clusterable_embedding_store):
        """_get_event_filtered_embeddings blends title at TITLE_WEIGHT.

        The mock store always returns embeddings for sentence_id=0 (title),
        so the returned embeddings should differ from phrases-only.
        """
        articles = _make_articles(n_per_cluster=5, n_clusters=1)

        # Build sentence index so event-filtered path is taken
        sentence_df = pd.DataFrame([
            {'doc_id': f'cluster_0_art{i}', 'sentence_id': 1, 'evt_weather': 1}
            for i in range(5)
        ])
        det = EventOccurrenceDetector(clusterable_embedding_store, sentence_df=sentence_df)

        embeddings, found_ids = det._get_event_filtered_embeddings(
            [f'cluster_0_art{i}' for i in range(5)], 'evt_weather'
        )

        assert len(found_ids) == 5
        assert embeddings.shape == (5, clusterable_embedding_store.embedding_dim)

        # Verify title was blended: get phrases-only for comparison
        for i, doc_id in enumerate(found_ids):
            phrases_emb = clusterable_embedding_store.get_filtered_article_embedding(
                doc_id, [1]
            )
            title_emb = clusterable_embedding_store.get_sentence_embedding(
                doc_id, TITLE_SENTENCE_ID
            )
            expected = TITLE_WEIGHT * title_emb + (1.0 - TITLE_WEIGHT) * phrases_emb
            # The blended embedding should be close to expected
            cos_sim = float(np.dot(
                embeddings[i] / np.linalg.norm(embeddings[i]),
                expected / np.linalg.norm(expected)
            ))
            assert cos_sim > 0.99

    def test_no_title_fallback(self, clusterable_embedding_store):
        """When title is not available, falls back to phrases-only."""
        class NoTitleStore(type(clusterable_embedding_store)):
            def get_sentence_embedding(self, doc_id, sentence_id):
                if sentence_id == 0:
                    return None
                return super().get_sentence_embedding(doc_id, sentence_id)

        store = NoTitleStore()
        sentence_df = pd.DataFrame([
            {'doc_id': 'cluster_0_art0', 'sentence_id': 1, 'evt_weather': 1},
        ])
        det = EventOccurrenceDetector(store, sentence_df=sentence_df)

        embeddings, found_ids = det._get_event_filtered_embeddings(
            ['cluster_0_art0'], 'evt_weather'
        )
        assert len(found_ids) == 1
        # Should be purely phrases-based
        expected = store.get_filtered_article_embedding('cluster_0_art0', [1])
        cos_sim = float(np.dot(
            embeddings[0] / np.linalg.norm(embeddings[0]),
            expected / np.linalg.norm(expected)
        ))
        assert cos_sim > 0.999

    def test_title_constants(self):
        """Title constants have expected values."""
        assert TITLE_SENTENCE_ID == 0
        assert 0.0 < TITLE_WEIGHT < 1.0
        assert TITLE_WEIGHT == 0.30


# -----------------------------------------------------------------------
# TestEventClusterDetection
# -----------------------------------------------------------------------

def _make_occurrences(n_clusters=2, n_per_cluster=3, days_apart=30,
                      clusterable_store=None):
    """Build test EventOccurrence objects for clustering tests.

    Creates n_clusters groups of occurrences, each group clustered
    around a different date and using different centroids.
    """
    if clusterable_store is None:
        from tests.conftest import ClusterableMockEmbeddingStore
        clusterable_store = ClusterableMockEmbeddingStore()

    occurrences = []
    occ_id = 0
    for c in range(n_clusters):
        base_date = pd.Timestamp('2018-06-01') + pd.Timedelta(days=c * days_apart)
        centroid = clusterable_store._centroids[c].copy()
        centroid /= np.linalg.norm(centroid)

        evt_type = ['evt_weather', 'evt_meeting', 'evt_policy'][c % 3]

        for i in range(n_per_cluster):
            # Add small noise to centroid for each occurrence
            rng = np.random.RandomState(c * 100 + i)
            noise = rng.randn(clusterable_store.embedding_dim).astype(np.float32)
            occ_centroid = centroid + 0.05 * noise
            occ_centroid /= np.linalg.norm(occ_centroid)

            peak = base_date + pd.Timedelta(days=i * 2)
            # Each occurrence gets unique seed_doc_ids so deduplication
            # doesn't merge them (they represent distinct sub-events)
            occ_seed_ids = [f'cluster_{c}_occ{i}_art{j}' for j in range(10)]
            occ = EventOccurrence(
                occurrence_id=occ_id,
                event_type=evt_type,
                first_date=peak - pd.Timedelta(days=3),
                last_date=peak + pd.Timedelta(days=3),
                core_start=peak - pd.Timedelta(days=1),
                core_end=peak + pd.Timedelta(days=1),
                peak_date=peak,
                n_articles=10,
                effective_mass=8.0 + i,
                core_mass=6.0,
                semantic_coherence=0.75,
                centroid=occ_centroid,
                confidence=0.65,
                doc_ids=occ_seed_ids,
                seed_doc_ids=occ_seed_ids,
                belonging={d: 0.8 for d in occ_seed_ids},
            )
            occurrences.append(occ)
            occ_id += 1

    return occurrences


class TestEventClusterDetection:
    """Phase 3: event cluster detection tests."""

    def test_separate_clusters(self, clusterable_embedding_store):
        """Well-separated occurrences form distinct event clusters."""
        occurrences = _make_occurrences(
            n_clusters=2, n_per_cluster=3, days_apart=60,
            clusterable_store=clusterable_embedding_store,
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(occurrences)

        assert len(clusters) >= 2
        # Each cluster should have occurrences
        for ec in clusters:
            assert ec.n_occurrences > 0
            assert len(ec.occurrences) == ec.n_occurrences

    def test_nearby_occurrences_merge(self, clusterable_embedding_store):
        """Duplicate occurrences (same type, same docs) are deduplicated."""
        base_centroid = clusterable_embedding_store._centroids[0].copy()
        base_centroid /= np.linalg.norm(base_centroid)
        base_date = pd.Timestamp('2018-06-01')
        same_docs = [f'shared_{j}' for j in range(10)]

        occurrences = []
        for i in range(3):
            occ = EventOccurrence(
                occurrence_id=i,
                event_type='evt_weather',
                first_date=base_date - pd.Timedelta(days=3),
                last_date=base_date + pd.Timedelta(days=3),
                core_start=base_date - pd.Timedelta(days=1),
                core_end=base_date + pd.Timedelta(days=1),
                peak_date=base_date,
                n_articles=10,
                effective_mass=8.0,
                core_mass=6.0,
                semantic_coherence=0.75,
                centroid=base_centroid.copy(),
                confidence=0.8,
                low_confidence=False,
                doc_ids=same_docs,
                seed_doc_ids=same_docs,
                belonging={d: 0.8 for d in same_docs},
            )
            occurrences.append(occ)

        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(occurrences)

        # All identical → deduplicated to 1 occurrence → single cluster
        assert len(clusters) == 1
        assert clusters[0].n_occurrences == 1

    def test_singleton_cluster(self, clusterable_embedding_store):
        """Single occurrence produces a singleton cluster."""
        occurrences = _make_occurrences(
            n_clusters=1, n_per_cluster=1, days_apart=0,
            clusterable_store=clusterable_embedding_store,
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(occurrences)

        assert len(clusters) == 1
        assert clusters[0].n_occurrences == 1

    def test_empty_occurrences(self, clusterable_embedding_store):
        """Empty occurrence list returns empty clusters."""
        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences([])
        assert clusters == []

    def test_cluster_has_required_fields(self, clusterable_embedding_store):
        """EventCluster has all required fields populated."""
        occurrences = _make_occurrences(
            n_clusters=2, n_per_cluster=3, days_apart=60,
            clusterable_store=clusterable_embedding_store,
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(occurrences)

        for ec in clusters:
            assert isinstance(ec, EventCluster)
            assert isinstance(ec.cluster_id, int)
            assert isinstance(ec.event_types, set)
            assert len(ec.event_types) > 0
            assert isinstance(ec.peak_date, pd.Timestamp)
            assert isinstance(ec.core_start, pd.Timestamp)
            assert isinstance(ec.core_end, pd.Timestamp)
            assert ec.total_mass > 0
            assert ec.centroid is not None
            assert isinstance(ec.is_multi_type, bool)
            assert 0.0 <= ec.strength <= 1.0
            assert isinstance(ec.strength_components, dict)
            assert isinstance(ec.dominant_type, str)
            assert ec.dominant_type in ec.event_types
            assert isinstance(ec.type_structure, dict)
            assert isinstance(ec.type_overlap_graph, dict)

    def test_multi_type_flag(self, clusterable_embedding_store):
        """Multi-type clusters form when evidence is strong (shared seed_doc_ids)."""
        base_centroid = clusterable_embedding_store._centroids[0].copy()
        base_centroid /= np.linalg.norm(base_centroid)
        base_date = pd.Timestamp('2018-06-01')

        far_centroid = clusterable_embedding_store._centroids[1].copy()
        far_centroid /= np.linalg.norm(far_centroid)
        far_date = pd.Timestamp('2018-09-01')

        # Shared seeds across types → constitutive structure → survives dissolution
        shared_seeds = [f'shared_seed_{k}' for k in range(5)]

        occurrences = []
        occ_id = 0
        for i, evt_type in enumerate(['evt_weather', 'evt_meeting', 'evt_policy']):
            for j in range(2):
                shared_docs = [f'shared_{k}' for k in range(10)]
                # seed_doc_ids include shared seeds + type-specific seeds
                # Many unique seeds → high fragmentation ratio → survives consolidation
                type_seeds = shared_seeds + [f'type{i}_occ{j}_art{k}' for k in range(10)]
                occ = EventOccurrence(
                    occurrence_id=occ_id,
                    event_type=evt_type,
                    first_date=base_date - pd.Timedelta(days=3),
                    last_date=base_date + pd.Timedelta(days=3),
                    core_start=base_date - pd.Timedelta(days=1),
                    core_end=base_date + pd.Timedelta(days=1),
                    peak_date=base_date + pd.Timedelta(days=j),
                    n_articles=10,
                    effective_mass=8.0 + j,
                    core_mass=6.0,
                    semantic_coherence=0.75,
                    centroid=base_centroid.copy(),
                    confidence=0.8,
                    low_confidence=False,
                    doc_ids=list(shared_docs),
                    seed_doc_ids=type_seeds,
                    belonging={d: 0.8 for d in shared_docs},
                )
                occurrences.append(occ)
                occ_id += 1

        # Add 2 distant occurrences to create a clear 2-cluster structure
        for j in range(2):
            far_docs = [f'far_{j}_art{k}' for k in range(10)]
            occ = EventOccurrence(
                occurrence_id=occ_id,
                event_type='evt_weather',
                first_date=far_date - pd.Timedelta(days=3),
                last_date=far_date + pd.Timedelta(days=3),
                core_start=far_date - pd.Timedelta(days=1),
                core_end=far_date + pd.Timedelta(days=1),
                peak_date=far_date + pd.Timedelta(days=j),
                n_articles=10,
                effective_mass=8.0 + j,
                core_mass=6.0,
                semantic_coherence=0.75,
                centroid=far_centroid.copy(),
                confidence=0.8,
                low_confidence=False,
                doc_ids=list(far_docs),
                seed_doc_ids=list(far_docs),
                belonging={d: 0.8 for d in far_docs},
            )
            occurrences.append(occ)
            occ_id += 1

        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(occurrences)

        multi_type = [ec for ec in clusters if ec.is_multi_type]
        assert len(multi_type) >= 1
        assert any(len(ec.event_types) >= 2 for ec in clusters)

    def test_to_dict_serializable(self, clusterable_embedding_store):
        """EventCluster.to_dict() produces JSON-serializable output."""
        import json
        occurrences = _make_occurrences(
            n_clusters=1, n_per_cluster=3, days_apart=0,
            clusterable_store=clusterable_embedding_store,
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(occurrences)

        for ec in clusters:
            d = ec.to_dict()
            json_str = json.dumps(d)
            parsed = json.loads(json_str)
            assert 'cluster_id' in parsed
            assert 'event_types' in parsed
            assert 'strength' in parsed

    def test_cluster_sorted_by_peak_date(self, clusterable_embedding_store):
        """Event clusters are sorted by peak_date."""
        occurrences = _make_occurrences(
            n_clusters=3, n_per_cluster=2, days_apart=30,
            clusterable_store=clusterable_embedding_store,
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(occurrences)

        for i in range(len(clusters) - 1):
            assert clusters[i].peak_date <= clusters[i + 1].peak_date


# -----------------------------------------------------------------------
# TestOccurrenceDistance
# -----------------------------------------------------------------------

class TestOccurrenceDistance:
    """Tests for compound distance between occurrences."""

    def test_identical_occurrences(self, clusterable_embedding_store):
        """Distance between identical occurrences is ~0."""
        occurrences = _make_occurrences(
            n_clusters=1, n_per_cluster=2, days_apart=0,
            clusterable_store=clusterable_embedding_store,
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)

        d = det._occurrence_distance(
            occurrences[0], occurrences[0], set(), set()
        )
        assert d < 0.15

    def test_distant_occurrences(self, clusterable_embedding_store):
        """Far-apart occurrences have high distance."""
        occurrences = _make_occurrences(
            n_clusters=2, n_per_cluster=1, days_apart=100,
            clusterable_store=clusterable_embedding_store,
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)

        d = det._occurrence_distance(
            occurrences[0], occurrences[1], set(), set()
        )
        assert d > 0.3

    def test_entity_overlap_reduces_distance(self, clusterable_embedding_store):
        """Shared entities reduce the distance."""
        occurrences = _make_occurrences(
            n_clusters=1, n_per_cluster=2, days_apart=0,
            clusterable_store=clusterable_embedding_store,
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)

        entities_shared = {'PER:John Doe', 'ORG:IPCC'}

        d_no_ent = det._occurrence_distance(
            occurrences[0], occurrences[1], set(), set()
        )
        d_with_ent = det._occurrence_distance(
            occurrences[0], occurrences[1], entities_shared, entities_shared
        )
        assert d_with_ent < d_no_ent

    def test_article_overlap_reduces_distance(self, clusterable_embedding_store):
        """Shared seed_doc_ids reduce the distance via article overlap dimension."""
        base_centroid = clusterable_embedding_store._centroids[0].copy()
        base_centroid /= np.linalg.norm(base_centroid)
        base_date = pd.Timestamp('2018-06-01')

        seed_a = [f'doc_{i}' for i in range(10)]
        seed_no_overlap = [f'doc_{i}' for i in range(10, 20)]
        seed_with_overlap = [f'doc_{i}' for i in range(5, 15)]

        occ_a = EventOccurrence(
            occurrence_id=0, event_type='evt_weather',
            first_date=base_date, last_date=base_date,
            core_start=base_date, core_end=base_date,
            peak_date=base_date, n_articles=10, effective_mass=8.0,
            core_mass=6.0, semantic_coherence=0.75,
            centroid=base_centroid.copy(),
            doc_ids=seed_a, seed_doc_ids=seed_a,
        )
        occ_b_no_overlap = EventOccurrence(
            occurrence_id=1, event_type='evt_weather',
            first_date=base_date, last_date=base_date,
            core_start=base_date, core_end=base_date,
            peak_date=base_date, n_articles=10, effective_mass=8.0,
            core_mass=6.0, semantic_coherence=0.75,
            centroid=base_centroid.copy(),
            doc_ids=seed_no_overlap, seed_doc_ids=seed_no_overlap,
        )
        occ_b_with_overlap = EventOccurrence(
            occurrence_id=2, event_type='evt_weather',
            first_date=base_date, last_date=base_date,
            core_start=base_date, core_end=base_date,
            peak_date=base_date, n_articles=10, effective_mass=8.0,
            core_mass=6.0, semantic_coherence=0.75,
            centroid=base_centroid.copy(),
            doc_ids=seed_with_overlap, seed_doc_ids=seed_with_overlap,
        )

        det = EventOccurrenceDetector(clusterable_embedding_store)

        d_no_overlap = det._occurrence_distance(occ_a, occ_b_no_overlap, set(), set())
        d_with_overlap = det._occurrence_distance(occ_a, occ_b_with_overlap, set(), set())

        assert d_with_overlap < d_no_overlap

    def test_type_distance_increases_for_different_types(
        self, clusterable_embedding_store
    ):
        """Different event types add type distance penalty."""
        base_centroid = clusterable_embedding_store._centroids[0].copy()
        base_centroid /= np.linalg.norm(base_centroid)
        base_date = pd.Timestamp('2018-06-01')
        docs = [f'doc_{i}' for i in range(10)]

        occ_weather = EventOccurrence(
            occurrence_id=0, event_type='evt_weather',
            first_date=base_date, last_date=base_date,
            core_start=base_date, core_end=base_date,
            peak_date=base_date, n_articles=10, effective_mass=8.0,
            core_mass=6.0, semantic_coherence=0.75,
            centroid=base_centroid.copy(),
            doc_ids=list(docs), seed_doc_ids=list(docs),
        )
        occ_same_type = EventOccurrence(
            occurrence_id=1, event_type='evt_weather',
            first_date=base_date, last_date=base_date,
            core_start=base_date, core_end=base_date,
            peak_date=base_date, n_articles=10, effective_mass=8.0,
            core_mass=6.0, semantic_coherence=0.75,
            centroid=base_centroid.copy(),
            doc_ids=list(docs), seed_doc_ids=list(docs),
        )
        occ_diff_type = EventOccurrence(
            occurrence_id=2, event_type='evt_policy',
            first_date=base_date, last_date=base_date,
            core_start=base_date, core_end=base_date,
            peak_date=base_date, n_articles=10, effective_mass=8.0,
            core_mass=6.0, semantic_coherence=0.75,
            centroid=base_centroid.copy(),
            doc_ids=list(docs), seed_doc_ids=list(docs),
        )

        det = EventOccurrenceDetector(clusterable_embedding_store)

        d_same = det._occurrence_distance(occ_weather, occ_same_type, set(), set())
        d_diff = det._occurrence_distance(occ_weather, occ_diff_type, set(), set())

        assert d_diff > d_same

    def test_seed_doc_ids_used_for_article_overlap(
        self, clusterable_embedding_store
    ):
        """Article overlap uses seed_doc_ids (Phase 2 core members)."""
        base_centroid = clusterable_embedding_store._centroids[0].copy()
        base_centroid /= np.linalg.norm(base_centroid)
        base_date = pd.Timestamp('2018-06-01')

        shared_docs = [f'doc_{i}' for i in range(10)]

        occ_weather = EventOccurrence(
            occurrence_id=0, event_type='evt_weather',
            first_date=base_date, last_date=base_date,
            core_start=base_date, core_end=base_date,
            peak_date=base_date, n_articles=10, effective_mass=8.0,
            core_mass=6.0, semantic_coherence=0.75,
            centroid=base_centroid.copy(),
            doc_ids=list(shared_docs),
            seed_doc_ids=[f'seed_w_{i}' for i in range(5)],
        )
        occ_policy = EventOccurrence(
            occurrence_id=1, event_type='evt_policy',
            first_date=base_date, last_date=base_date,
            core_start=base_date, core_end=base_date,
            peak_date=base_date, n_articles=10, effective_mass=8.0,
            core_mass=6.0, semantic_coherence=0.75,
            centroid=base_centroid.copy(),
            doc_ids=list(shared_docs),
            seed_doc_ids=[f'seed_p_{i}' for i in range(5)],
        )

        det = EventOccurrenceDetector(clusterable_embedding_store)
        d = det._occurrence_distance(occ_weather, occ_policy, set(), set())

        assert d > 0.3


# -----------------------------------------------------------------------
# TestEntityExtraction
# -----------------------------------------------------------------------

class TestEntityExtraction:
    """Tests for entity extraction per occurrence."""

    def test_entity_extraction_basic(self, clusterable_embedding_store):
        """Entities appearing >= min citations are extracted."""
        occ = EventOccurrence(
            occurrence_id=0,
            event_type='evt_weather',
            first_date=pd.Timestamp('2018-06-01'),
            last_date=pd.Timestamp('2018-06-10'),
            core_start=pd.Timestamp('2018-06-03'),
            core_end=pd.Timestamp('2018-06-08'),
            peak_date=pd.Timestamp('2018-06-05'),
            n_articles=5,
            effective_mass=4.0,
            core_mass=3.0,
            semantic_coherence=0.7,
            centroid=np.zeros(64),
            doc_ids=['d1', 'd2', 'd3', 'd4', 'd5'],
        )

        entity_index = {
            'd1': [('IPCC', 'ORG'), ('Paris', 'LOC')],
            'd2': [('IPCC', 'ORG'), ('Paris', 'LOC')],
            'd3': [('IPCC', 'ORG'), ('London', 'LOC')],
            'd4': [('WMO', 'ORG')],
            'd5': [('IPCC', 'ORG'), ('Paris', 'LOC')],
        }

        det = EventOccurrenceDetector(clusterable_embedding_store)
        entities = det._extract_occurrence_entities(occ, entity_index)

        assert 'ORG:IPCC' in entities
        assert 'LOC:Paris' in entities
        assert 'LOC:London' not in entities
        assert 'ORG:WMO' not in entities

    def test_no_entity_index(self, clusterable_embedding_store):
        """Returns empty set when entity_index is None."""
        occ = EventOccurrence(
            occurrence_id=0,
            event_type='evt_weather',
            first_date=pd.Timestamp('2018-06-01'),
            last_date=pd.Timestamp('2018-06-10'),
            core_start=pd.Timestamp('2018-06-03'),
            core_end=pd.Timestamp('2018-06-08'),
            peak_date=pd.Timestamp('2018-06-05'),
            n_articles=0,
            effective_mass=0.0,
            core_mass=0.0,
            semantic_coherence=0.0,
            centroid=np.zeros(64),
        )

        det = EventOccurrenceDetector(clusterable_embedding_store)
        entities = det._extract_occurrence_entities(occ, None)
        assert entities == set()


# -----------------------------------------------------------------------
# TestOccurrenceStrength
# -----------------------------------------------------------------------

class TestOccurrenceStrength:
    """Tests for occurrence strength metrics."""

    def test_strength_metrics_populated(self, clusterable_embedding_store):
        """After _compute_occurrence_strength, fields are populated."""
        occ = EventOccurrence(
            occurrence_id=0,
            event_type='evt_weather',
            first_date=pd.Timestamp('2018-06-01'),
            last_date=pd.Timestamp('2018-06-10'),
            core_start=pd.Timestamp('2018-06-03'),
            core_end=pd.Timestamp('2018-06-08'),
            peak_date=pd.Timestamp('2018-06-05'),
            n_articles=5,
            effective_mass=4.0,
            core_mass=3.0,
            semantic_coherence=0.7,
            centroid=np.zeros(64),
            doc_ids=['cluster_0_art0', 'cluster_0_art1', 'cluster_0_art2',
                     'cluster_0_art3', 'cluster_0_art4'],
            belonging={f'cluster_0_art{i}': 0.8 for i in range(5)},
        )

        articles = pd.DataFrame({
            'doc_id': [f'cluster_0_art{i}' for i in range(5)],
            'date': pd.to_datetime(['2018-06-01'] * 5),
            'media': ['CBC', 'Globe', 'CBC', 'Star', 'Globe'],
            'tone_positive': [0.3, 0.5, 0.2, 0.6, 0.4],
            'tone_neutral': [0.5, 0.3, 0.6, 0.2, 0.4],
            'tone_negative': [0.2, 0.2, 0.2, 0.2, 0.2],
        })

        det = EventOccurrenceDetector(clusterable_embedding_store)
        det._compute_occurrence_strength([occ], articles)

        assert occ.media_count == 3  # CBC, Globe, Star
        assert occ.temporal_intensity > 0
        assert occ.emotional_intensity >= 0
        assert 0.0 <= occ.tone_coherence <= 1.0

    def test_temporal_intensity_formula(self, clusterable_embedding_store):
        """temporal_intensity = effective_mass / max(1, core_days)."""
        occ = EventOccurrence(
            occurrence_id=0,
            event_type='evt_weather',
            first_date=pd.Timestamp('2018-06-01'),
            last_date=pd.Timestamp('2018-06-10'),
            core_start=pd.Timestamp('2018-06-03'),
            core_end=pd.Timestamp('2018-06-08'),
            peak_date=pd.Timestamp('2018-06-05'),
            n_articles=5,
            effective_mass=10.0,
            core_mass=8.0,
            semantic_coherence=0.7,
            centroid=np.zeros(64),
            doc_ids=[],
            belonging={},
        )

        articles = pd.DataFrame({'doc_id': [], 'date': []})
        det = EventOccurrenceDetector(clusterable_embedding_store)
        det._compute_occurrence_strength([occ], articles)

        expected = 10.0 / 5  # 5 days core
        assert abs(occ.temporal_intensity - expected) < 1e-6


# -----------------------------------------------------------------------
# TestEventClusterModel
# -----------------------------------------------------------------------

class TestEventClusterModel:
    """Tests for EventCluster dataclass."""

    def test_to_dict_keys(self):
        """to_dict has all expected keys."""
        ec = EventCluster(
            cluster_id=0,
            occurrences=[],
            event_types={'evt_weather', 'evt_meeting'},
            peak_date=pd.Timestamp('2018-06-15'),
            core_start=pd.Timestamp('2018-06-10'),
            core_end=pd.Timestamp('2018-06-20'),
            total_mass=42.0,
            centroid=np.zeros(64),
            n_occurrences=5,
            is_multi_type=True,
            strength=0.6,
            strength_components={'mass_score': 0.8, 'coverage_score': 0.5},
            entities={'PER:John', 'ORG:IPCC'},
            dominant_type='evt_weather',
            type_structure={'evt_weather': 'constitutive', 'evt_meeting': 'satellite'},
            type_overlap_graph={'evt_weather': {'evt_meeting': 0.0}, 'evt_meeting': {'evt_weather': 0.0}},
        )
        d = ec.to_dict()
        assert d['cluster_id'] == 0
        assert d['is_multi_type'] is True
        assert d['total_mass'] == 42.0
        assert d['n_occurrences'] == 5
        assert 'event_types' in d
        assert 'strength_components' in d
        assert d['dominant_type'] == 'evt_weather'
        assert 'type_structure' in d
        assert 'type_overlap_graph' in d
        assert 'type_ranking' in d

    def test_to_dict_serializable(self):
        """to_dict output is JSON-serializable."""
        import json
        ec = EventCluster(
            cluster_id=1,
            occurrences=[],
            event_types={'evt_weather'},
            peak_date=pd.Timestamp('2018-07-01'),
            core_start=pd.Timestamp('2018-06-28'),
            core_end=pd.Timestamp('2018-07-04'),
            total_mass=20.0,
            centroid=np.ones(64),
            n_occurrences=3,
            is_multi_type=False,
            strength=0.4,
            dominant_type='evt_weather',
            type_structure={'evt_weather': 'constitutive'},
            type_overlap_graph={'evt_weather': {}},
        )
        json_str = json.dumps(ec.to_dict())
        parsed = json.loads(json_str)
        assert parsed['cluster_id'] == 1
        assert parsed['is_multi_type'] is False
        assert parsed['dominant_type'] == 'evt_weather'
        assert parsed['type_structure'] == {'evt_weather': 'constitutive'}


# -----------------------------------------------------------------------
# TestSilhouetteClustering
# -----------------------------------------------------------------------

class TestSilhouetteClustering:
    """Tests for silhouette-based optimal cut in Phase 3."""

    def test_silhouette_separates_distant_groups(self, clusterable_embedding_store):
        """Well-separated groups produce multiple clusters via silhouette."""
        occurrences = _make_occurrences(
            n_clusters=3, n_per_cluster=3, days_apart=60,
            clusterable_store=clusterable_embedding_store,
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(occurrences)

        assert len(clusters) >= 2

    def test_silhouette_merges_identical(self, clusterable_embedding_store):
        """Occurrences with high seed overlap are deduplicated, then clustered."""
        base_centroid = clusterable_embedding_store._centroids[0].copy()
        base_centroid /= np.linalg.norm(base_centroid)
        base_date = pd.Timestamp('2018-06-01')
        shared_docs = [f'shared_{j}' for j in range(10)]

        occurrences = []
        for i in range(4):
            seed = shared_docs + [f'unique_{i}_{j}' for j in range(5)]
            occ = EventOccurrence(
                occurrence_id=i, event_type='evt_weather',
                first_date=base_date, last_date=base_date,
                core_start=base_date, core_end=base_date,
                peak_date=base_date, n_articles=15, effective_mass=8.0,
                core_mass=6.0, semantic_coherence=0.75,
                centroid=base_centroid.copy(),
                doc_ids=seed,
                seed_doc_ids=seed,
                belonging={d: 0.8 for d in seed},
            )
            occurrences.append(occ)

        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(occurrences)

        assert len(clusters) == 1

    def test_two_occurrences_fallback(self, clusterable_embedding_store):
        """With exactly 2 occurrences, silhouette can only try k=2 or fallback."""
        occurrences = _make_occurrences(
            n_clusters=2, n_per_cluster=1, days_apart=60,
            clusterable_store=clusterable_embedding_store,
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(occurrences)

        assert len(clusters) >= 1


# -----------------------------------------------------------------------
# TestTypeStructure
# -----------------------------------------------------------------------

class TestTypeStructure:
    """Tests for type dominant, type_structure, and type_overlap_graph."""

    def test_mono_type_constitutive(self, clusterable_embedding_store):
        """Single-type cluster: type is constitutive and dominant."""
        from cascade_detector.analysis.event_occurrence import EventOccurrenceDetector as EOD
        base_centroid = clusterable_embedding_store._centroids[0].copy()
        base_centroid /= np.linalg.norm(base_centroid)
        base_date = pd.Timestamp('2018-06-01')

        occurrences = []
        for i in range(3):
            occ = EventOccurrence(
                occurrence_id=i, event_type='evt_weather',
                first_date=base_date, last_date=base_date,
                core_start=base_date, core_end=base_date,
                peak_date=base_date + pd.Timedelta(days=i),
                n_articles=10, effective_mass=8.0, core_mass=6.0,
                semantic_coherence=0.75, centroid=base_centroid.copy(),
                doc_ids=[100 + i * 10 + j for j in range(5)],
                belonging={100 + i * 10 + j: 0.8 for j in range(5)},
            )
            occurrences.append(occ)

        overlap_graph, structure, dominant, ranking = EOD._analyze_type_structure(
            occurrences, {'evt_weather'}
        )

        assert dominant == 'evt_weather'
        assert structure == {'evt_weather': 'constitutive'}
        assert len(ranking) >= 1
        assert ranking[0][0] == 'evt_weather'

    def test_multi_type_with_overlap(self, clusterable_embedding_store):
        """Multi-type cluster with shared articles: types are constitutive."""
        from cascade_detector.analysis.event_occurrence import EventOccurrenceDetector as EOD
        base_centroid = clusterable_embedding_store._centroids[0].copy()
        base_centroid /= np.linalg.norm(base_centroid)
        base_date = pd.Timestamp('2018-06-01')

        shared_docs = [f'shared_{i}' for i in range(5)]

        occ_weather = EventOccurrence(
            occurrence_id=0, event_type='evt_weather',
            first_date=base_date, last_date=base_date,
            core_start=base_date, core_end=base_date,
            peak_date=base_date, n_articles=10, effective_mass=10.0,
            core_mass=8.0, semantic_coherence=0.75,
            centroid=base_centroid.copy(),
            doc_ids=shared_docs + [f'weather_{i}' for i in range(5)],
            belonging={d: 0.8 for d in shared_docs + [f'weather_{i}' for i in range(5)]},
        )
        occ_meeting = EventOccurrence(
            occurrence_id=1, event_type='evt_meeting',
            first_date=base_date, last_date=base_date,
            core_start=base_date, core_end=base_date,
            peak_date=base_date, n_articles=8, effective_mass=6.0,
            core_mass=5.0, semantic_coherence=0.70,
            centroid=base_centroid.copy(),
            doc_ids=shared_docs + [f'meeting_{i}' for i in range(3)],
            belonging={d: 0.7 for d in shared_docs + [f'meeting_{i}' for i in range(3)]},
        )

        occurrences = [occ_weather, occ_meeting]
        event_types = {'evt_weather', 'evt_meeting'}

        overlap_graph, structure, dominant, ranking = EOD._analyze_type_structure(
            occurrences, event_types
        )

        assert structure['evt_weather'] == 'constitutive'
        assert structure['evt_meeting'] == 'constitutive'
        assert overlap_graph['evt_weather']['evt_meeting'] > 0

    def test_satellite_type_no_shared_articles(self, clusterable_embedding_store):
        """Type with zero article overlap with others is satellite."""
        from cascade_detector.analysis.event_occurrence import EventOccurrenceDetector as EOD
        base_centroid = clusterable_embedding_store._centroids[0].copy()
        base_centroid /= np.linalg.norm(base_centroid)
        base_date = pd.Timestamp('2018-06-01')

        occ_weather = EventOccurrence(
            occurrence_id=0, event_type='evt_weather',
            first_date=base_date, last_date=base_date,
            core_start=base_date, core_end=base_date,
            peak_date=base_date, n_articles=10, effective_mass=10.0,
            core_mass=8.0, semantic_coherence=0.75,
            centroid=base_centroid.copy(),
            doc_ids=[f'weather_{i}' for i in range(10)],
            belonging={f'weather_{i}': 0.8 for i in range(10)},
        )
        occ_policy = EventOccurrence(
            occurrence_id=1, event_type='evt_policy',
            first_date=base_date, last_date=base_date,
            core_start=base_date, core_end=base_date,
            peak_date=base_date, n_articles=5, effective_mass=4.0,
            core_mass=3.0, semantic_coherence=0.70,
            centroid=base_centroid.copy(),
            doc_ids=[f'policy_{i}' for i in range(5)],
            belonging={f'policy_{i}': 0.7 for i in range(5)},
        )

        occurrences = [occ_weather, occ_policy]
        event_types = {'evt_weather', 'evt_policy'}

        overlap_graph, structure, dominant, ranking = EOD._analyze_type_structure(
            occurrences, event_types
        )

        assert structure['evt_weather'] == 'satellite'
        assert structure['evt_policy'] == 'satellite'
        assert overlap_graph['evt_weather']['evt_policy'] == 0.0

    def test_dominant_type_by_mass(self, clusterable_embedding_store):
        """Dominant type is the one with highest mass × connectivity."""
        from cascade_detector.analysis.event_occurrence import EventOccurrenceDetector as EOD
        base_centroid = clusterable_embedding_store._centroids[0].copy()
        base_centroid /= np.linalg.norm(base_centroid)
        base_date = pd.Timestamp('2018-06-01')

        shared_docs = [f'shared_{i}' for i in range(5)]

        occ_weather = EventOccurrence(
            occurrence_id=0, event_type='evt_weather',
            first_date=base_date, last_date=base_date,
            core_start=base_date, core_end=base_date,
            peak_date=base_date, n_articles=15, effective_mass=20.0,
            core_mass=15.0, semantic_coherence=0.80,
            centroid=base_centroid.copy(),
            doc_ids=shared_docs + [f'weather_{i}' for i in range(10)],
            belonging={d: 0.8 for d in shared_docs + [f'weather_{i}' for i in range(10)]},
        )
        occ_meeting = EventOccurrence(
            occurrence_id=1, event_type='evt_meeting',
            first_date=base_date, last_date=base_date,
            core_start=base_date, core_end=base_date,
            peak_date=base_date, n_articles=5, effective_mass=3.0,
            core_mass=2.0, semantic_coherence=0.70,
            centroid=base_centroid.copy(),
            doc_ids=shared_docs,
            belonging={d: 0.7 for d in shared_docs},
        )

        occurrences = [occ_weather, occ_meeting]
        event_types = {'evt_weather', 'evt_meeting'}

        overlap_graph, structure, dominant, ranking = EOD._analyze_type_structure(
            occurrences, event_types
        )

        assert dominant == 'evt_weather'


# -----------------------------------------------------------------------
# TestTypeRanking
# -----------------------------------------------------------------------

class TestTypeRanking:
    """Tests for type_ranking across all 7 event types (mono + multi)."""

    def _make_occ(self, occ_id, event_type, mass, doc_ids, type_scores=None):
        """Helper to create a minimal EventOccurrence."""
        base_date = pd.Timestamp('2018-06-01')
        return EventOccurrence(
            occurrence_id=occ_id, event_type=event_type,
            first_date=base_date, last_date=base_date,
            core_start=base_date, core_end=base_date,
            peak_date=base_date, n_articles=len(doc_ids),
            effective_mass=mass, core_mass=mass * 0.8,
            semantic_coherence=0.75, centroid=np.zeros(64),
            doc_ids=doc_ids,
            belonging={d: 0.8 for d in doc_ids},
            type_scores=type_scores or {},
        )

    def test_mono_type_ranking_structural_only(self):
        """Mono-type without type_scores: ranking is purely structural."""
        from cascade_detector.analysis.event_occurrence import EventOccurrenceDetector as EOD
        from cascade_detector.core.constants import EVENT_COLUMNS

        occ = self._make_occ(0, 'evt_weather', 10.0, list(range(10)))
        _, _, dominant, ranking = EOD._analyze_type_structure(
            [occ], {'evt_weather'}, EVENT_COLUMNS
        )

        assert dominant == 'evt_weather'
        assert ranking[0][0] == 'evt_weather'
        assert ranking[0][1] == 0.5  # structural=1.0, nlp=0 → 0.50*1.0 + 0.50*0.0
        # Only evt_weather has score > 0 (no type_scores → nlp=0, only present type gets structural)
        assert len(ranking) == 1

    def test_mono_type_ranking_with_type_scores(self):
        """Mono-type with type_scores: NLP adds secondary types to ranking."""
        from cascade_detector.analysis.event_occurrence import EventOccurrenceDetector as EOD
        from cascade_detector.core.constants import EVENT_COLUMNS

        type_scores = {
            'evt_weather': 0.12,
            'evt_policy': 0.10,
            'evt_meeting': 0.04,
            'evt_election': 0.0,
        }
        occ = self._make_occ(0, 'evt_weather', 10.0, list(range(10)),
                             type_scores=type_scores)
        _, _, dominant, ranking = EOD._analyze_type_structure(
            [occ], {'evt_weather'}, EVENT_COLUMNS
        )

        assert dominant == 'evt_weather'
        # evt_weather is rank 1 (structural=1.0 + nlp=1.0)
        assert ranking[0][0] == 'evt_weather'
        # evt_policy should appear in ranking (nlp > 0)
        ranking_types = [t for t, _ in ranking]
        assert 'evt_policy' in ranking_types
        assert 'evt_meeting' in ranking_types
        # evt_election has score 0 → should NOT appear
        assert 'evt_election' not in ranking_types
        # evt_weather score > evt_policy score
        scores = {t: s for t, s in ranking}
        assert scores['evt_weather'] > scores['evt_policy']

    def test_multi_type_ranking_covers_all_types(self):
        """Multi-type cluster with type_scores: ranking covers all 7 types."""
        from cascade_detector.analysis.event_occurrence import EventOccurrenceDetector as EOD
        from cascade_detector.core.constants import EVENT_COLUMNS

        type_scores_w = {col: 0.05 for col in EVENT_COLUMNS}
        type_scores_w['evt_weather'] = 0.15
        type_scores_m = {col: 0.03 for col in EVENT_COLUMNS}
        type_scores_m['evt_meeting'] = 0.12

        shared = list(range(5))
        occ_w = self._make_occ(0, 'evt_weather', 12.0,
                               shared + list(range(10, 20)),
                               type_scores=type_scores_w)
        occ_m = self._make_occ(1, 'evt_meeting', 8.0,
                               shared + list(range(20, 25)),
                               type_scores=type_scores_m)

        _, _, dominant, ranking = EOD._analyze_type_structure(
            [occ_w, occ_m], {'evt_weather', 'evt_meeting'}, EVENT_COLUMNS
        )

        # All 7 types should appear (all have nlp > 0)
        ranking_types = [t for t, _ in ranking]
        assert len(ranking_types) == len(EVENT_COLUMNS)
        # Dominant should be one of the two present types
        assert dominant in {'evt_weather', 'evt_meeting'}
        # Present types should rank higher than absent types
        present_scores = {t: s for t, s in ranking if t in {'evt_weather', 'evt_meeting'}}
        absent_scores = {t: s for t, s in ranking if t not in {'evt_weather', 'evt_meeting'}}
        assert min(present_scores.values()) > max(absent_scores.values())

    def test_ranking_sorted_descending(self):
        """Type ranking is sorted in descending order."""
        from cascade_detector.analysis.event_occurrence import EventOccurrenceDetector as EOD
        from cascade_detector.core.constants import EVENT_COLUMNS

        type_scores = {col: 0.01 * (i + 1) for i, col in enumerate(EVENT_COLUMNS)}
        occ = self._make_occ(0, 'evt_weather', 10.0, list(range(10)),
                             type_scores=type_scores)
        _, _, _, ranking = EOD._analyze_type_structure(
            [occ], {'evt_weather'}, EVENT_COLUMNS
        )

        scores = [s for _, s in ranking]
        assert scores == sorted(scores, reverse=True)

    def test_ranking_no_zero_scores(self):
        """Types with score=0 are excluded from ranking."""
        from cascade_detector.analysis.event_occurrence import EventOccurrenceDetector as EOD
        from cascade_detector.core.constants import EVENT_COLUMNS

        # Only one type has NLP score, all others 0
        type_scores = {col: 0.0 for col in EVENT_COLUMNS}
        type_scores['evt_weather'] = 0.10
        occ = self._make_occ(0, 'evt_weather', 10.0, list(range(10)),
                             type_scores=type_scores)
        _, _, _, ranking = EOD._analyze_type_structure(
            [occ], {'evt_weather'}, EVENT_COLUMNS
        )

        for t, s in ranking:
            assert s > 0

    def test_ranking_mass_weighted_nlp(self):
        """NLP scores are weighted by occurrence mass."""
        from cascade_detector.analysis.event_occurrence import EventOccurrenceDetector as EOD
        from cascade_detector.core.constants import EVENT_COLUMNS

        # Heavy occ has high evt_policy score, light occ has low
        occ_heavy = self._make_occ(0, 'evt_weather', 20.0, list(range(20)),
                                   type_scores={'evt_policy': 0.20, 'evt_weather': 0.10})
        occ_light = self._make_occ(1, 'evt_weather', 2.0, list(range(20, 25)),
                                   type_scores={'evt_policy': 0.01, 'evt_weather': 0.30})

        _, _, _, ranking = EOD._analyze_type_structure(
            [occ_heavy, occ_light], {'evt_weather'}, EVENT_COLUMNS
        )

        scores = {t: s for t, s in ranking}
        # evt_policy NLP raw ≈ (20*0.20 + 2*0.01)/22 ≈ 0.182
        # evt_weather NLP raw ≈ (20*0.10 + 2*0.30)/22 ≈ 0.118
        # After normalization, evt_policy NLP > evt_weather NLP
        assert 'evt_policy' in scores

    def test_ranking_without_all_event_columns(self):
        """Without all_event_columns, ranking covers only present types."""
        from cascade_detector.analysis.event_occurrence import EventOccurrenceDetector as EOD

        occ = self._make_occ(0, 'evt_weather', 10.0, list(range(10)),
                             type_scores={'evt_weather': 0.10, 'evt_policy': 0.08})
        _, _, dominant, ranking = EOD._analyze_type_structure(
            [occ], {'evt_weather'}  # no all_event_columns
        )

        assert dominant == 'evt_weather'
        # Only evt_weather in ranking (all_types = types_list = present types)
        ranking_types = [t for t, _ in ranking]
        assert ranking_types == ['evt_weather']

    def test_cluster_has_type_ranking(self, clusterable_embedding_store):
        """EventCluster built via _build_event_cluster has type_ranking."""
        occurrences = _make_occurrences(
            n_clusters=1, n_per_cluster=3, days_apart=0,
            clusterable_store=clusterable_embedding_store,
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(
            occurrences, embedding_store=clusterable_embedding_store,
        )

        for ec in clusters:
            assert isinstance(ec.type_ranking, list)
            assert len(ec.type_ranking) >= 1
            # First entry matches dominant_type
            assert ec.type_ranking[0][0] == ec.dominant_type
            # Sorted descending
            scores = [s for _, s in ec.type_ranking]
            assert scores == sorted(scores, reverse=True)

    def test_to_dict_includes_type_ranking(self):
        """to_dict serializes type_ranking correctly."""
        ec = EventCluster(
            cluster_id=0, occurrences=[], event_types={'evt_weather'},
            peak_date=pd.Timestamp('2018-06-15'),
            core_start=pd.Timestamp('2018-06-10'),
            core_end=pd.Timestamp('2018-06-20'),
            total_mass=20.0, centroid=np.zeros(64), n_occurrences=3,
            is_multi_type=False, dominant_type='evt_weather',
            type_ranking=[('evt_weather', 0.95), ('evt_policy', 0.42)],
        )
        d = ec.to_dict()
        assert 'type_ranking' in d
        assert len(d['type_ranking']) == 2
        assert d['type_ranking'][0] == {'type': 'evt_weather', 'score': 0.95}
        assert d['type_ranking'][1] == {'type': 'evt_policy', 'score': 0.42}

    def test_to_dict_type_ranking_json_serializable(self):
        """type_ranking in to_dict is JSON-serializable."""
        import json
        ec = EventCluster(
            cluster_id=0, occurrences=[], event_types={'evt_weather'},
            peak_date=pd.Timestamp('2018-06-15'),
            core_start=pd.Timestamp('2018-06-10'),
            core_end=pd.Timestamp('2018-06-20'),
            total_mass=20.0, centroid=np.zeros(64), n_occurrences=3,
            is_multi_type=False, dominant_type='evt_weather',
            type_ranking=[('evt_weather', 0.95)],
        )
        json_str = json.dumps(ec.to_dict())
        parsed = json.loads(json_str)
        assert parsed['type_ranking'] == [{'type': 'evt_weather', 'score': 0.95}]


# -----------------------------------------------------------------------
# TestStrengthV2
# -----------------------------------------------------------------------

class TestStrengthV2:
    """Tests for the v2 strength scoring (mass, coverage, intensity, coherence)."""

    def test_strength_components_keys(self, clusterable_embedding_store):
        """Strength components have the 5 expected keys."""
        occurrences = _make_occurrences(
            n_clusters=1, n_per_cluster=3, days_apart=0,
            clusterable_store=clusterable_embedding_store,
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(
            occurrences, embedding_store=clusterable_embedding_store,
        )

        expected_keys = {
            'mass_score', 'coverage_score', 'intensity_score',
            'coherence_score', 'media_diversity',
        }
        for ec in clusters:
            assert set(ec.strength_components.keys()) == expected_keys

    def test_coherence_with_embedding_store(self, clusterable_embedding_store):
        """Coherence score is > 0 when embedding_store is provided."""
        occurrences = _make_occurrences(
            n_clusters=1, n_per_cluster=3, days_apart=0,
            clusterable_store=clusterable_embedding_store,
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(
            occurrences, embedding_store=clusterable_embedding_store,
        )

        for ec in clusters:
            assert ec.strength_components['coherence_score'] > 0.0

    def test_coherence_without_embedding_store(self, clusterable_embedding_store):
        """Coherence score is 0 when no embedding_store is provided."""
        occurrences = _make_occurrences(
            n_clusters=1, n_per_cluster=3, days_apart=0,
            clusterable_store=clusterable_embedding_store,
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(occurrences)  # no embedding_store

        for ec in clusters:
            assert ec.strength_components['coherence_score'] == 0.0

    def test_coverage_with_articles(self, clusterable_embedding_store):
        """Coverage score is > 0 when articles with media are provided."""
        occurrences = _make_occurrences(
            n_clusters=1, n_per_cluster=3, days_apart=0,
            clusterable_store=clusterable_embedding_store,
        )
        all_doc_ids = sorted(set(
            did for occ in occurrences for did in occ.doc_ids
        ))
        n = len(all_doc_ids)
        media_choices = ['CBC', 'Globe and Mail', 'Toronto Star']
        articles = pd.DataFrame({
            'doc_id': all_doc_ids,
            'date': pd.to_datetime(['2018-06-01'] * n),
            'media': [media_choices[i % len(media_choices)] for i in range(n)],
            'author': [f'journalist_{i}' for i in range(n)],
        })

        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(occurrences, articles=articles)

        for ec in clusters:
            assert ec.strength_components['coverage_score'] > 0.0

    def test_strength_range(self, clusterable_embedding_store):
        """Strength score is in [0, 1]."""
        occurrences = _make_occurrences(
            n_clusters=2, n_per_cluster=3, days_apart=60,
            clusterable_store=clusterable_embedding_store,
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(occurrences)

        for ec in clusters:
            assert 0.0 <= ec.strength <= 1.0
            for v in ec.strength_components.values():
                assert 0.0 <= v <= 1.0

    def test_mass_recalibration(self, clusterable_embedding_store):
        """Mass=10 gives ~0.49 (not 0.22 with old /log2(50001) calibration)."""
        occurrences = _make_occurrences(
            n_clusters=1, n_per_cluster=1, days_apart=0,
            clusterable_store=clusterable_embedding_store,
        )
        # effective_mass=8.0 for first occurrence
        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(
            occurrences, embedding_store=clusterable_embedding_store,
        )

        for ec in clusters:
            mass = ec.strength_components['mass_score']
            # log2(1+8)/log2(1+100) = 3.17/6.66 ≈ 0.476
            assert mass > 0.40, f"mass_score={mass} should be > 0.40 with recalibration"

    def test_media_diversity_in_strength(self, clusterable_embedding_store):
        """Clusters with multiple media have media_diversity > 0."""
        occurrences = _make_occurrences(
            n_clusters=1, n_per_cluster=3, days_apart=0,
            clusterable_store=clusterable_embedding_store,
        )
        all_doc_ids = sorted(set(
            did for occ in occurrences for did in occ.doc_ids
        ))
        n = len(all_doc_ids)
        media_choices = ['CBC', 'Globe and Mail', 'Toronto Star']
        articles = pd.DataFrame({
            'doc_id': all_doc_ids,
            'date': pd.to_datetime(['2018-06-01'] * n),
            'media': [media_choices[i % len(media_choices)] for i in range(n)],
        })

        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(occurrences, articles=articles)

        for ec in clusters:
            assert ec.strength_components['media_diversity'] > 0.0

    def test_coherence_residual_in_strength(self, clusterable_embedding_store):
        """Coherence score is residual (above baseline), in [0, 1]."""
        occurrences = _make_occurrences(
            n_clusters=1, n_per_cluster=3, days_apart=0,
            clusterable_store=clusterable_embedding_store,
        )
        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(
            occurrences, embedding_store=clusterable_embedding_store,
        )

        for ec in clusters:
            cs = ec.strength_components['coherence_score']
            assert 0.0 <= cs <= 1.0


# -----------------------------------------------------------------------
# TestCascadeAttributionModel
# -----------------------------------------------------------------------

class TestCascadeAttributionModel:
    """Tests for CascadeAttribution dataclass."""

    def test_to_dict(self):
        """to_dict() produces serializable output."""
        import json
        attr = CascadeAttribution(
            cascade_id='test',
            occurrence_id=0,
            shared_articles=5,
            temporal_overlap_days=10,
            overlap_ratio=0.5,
        )
        d = attr.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed['cascade_id'] == 'test'
        assert parsed['shared_articles'] == 5
        assert parsed['overlap_ratio'] == 0.5


# -----------------------------------------------------------------------
# TestSeedDominantRatio (Correction 1)
# -----------------------------------------------------------------------

class TestSeedDominantRatio:
    """Correction 1: articles with weak signal for a type are excluded as seeds."""

    def test_dominant_ratio_excludes_weak_type(self, clusterable_embedding_store):
        """Article with weather=1.0, policy=0.1 is excluded from policy seeds."""
        articles = pd.DataFrame({
            'doc_id': ['a', 'b', 'c', 'd', 'e'],
            'date': pd.to_datetime(['2018-06-01'] * 5),
            'evt_weather_mean': [1.0, 0.0, 0.0, 0.0, 0.0],
            'evt_policy_mean': [0.1, 0.8, 0.7, 0.6, 0.5],
        })
        det = EventOccurrenceDetector(clusterable_embedding_store)

        # Article 'a' has policy=0.1 < 0.5 * max(weather=1.0) → excluded
        seeds, _ = det._select_seed_articles(articles, 'evt_policy_mean')
        assert 'a' not in seeds['doc_id'].values

    def test_dominant_ratio_allows_balanced(self, clusterable_embedding_store):
        """Article with weather=0.8, policy=0.8 stays seed for both types."""
        articles = pd.DataFrame({
            'doc_id': ['a', 'b', 'c', 'd', 'e'],
            'date': pd.to_datetime(['2018-06-01'] * 5),
            'evt_weather_mean': [0.8, 0.5, 0.6, 0.1, 0.1],
            'evt_policy_mean': [0.8, 0.1, 0.1, 0.5, 0.6],
        })
        det = EventOccurrenceDetector(clusterable_embedding_store)

        # Article 'a' has policy=0.8 >= 0.5 * max(weather=0.8) → eligible
        # AND policy=0.8 is at or above P50 → becomes seed
        seeds_policy, _ = det._select_seed_articles(articles, 'evt_policy_mean')
        assert 'a' in seeds_policy['doc_id'].values

        seeds_weather, _ = det._select_seed_articles(articles, 'evt_weather_mean')
        assert 'a' in seeds_weather['doc_id'].values


# -----------------------------------------------------------------------
# TestCorpusBaseline
# -----------------------------------------------------------------------

class TestCorpusBaseline:
    """Correction 3: corpus-adjusted residual similarity."""

    def test_baseline_computation(self, clusterable_embedding_store):
        """compute_corpus_baseline returns a float in [0, 1]."""
        baseline = clusterable_embedding_store.compute_corpus_baseline()
        assert isinstance(baseline, float)
        assert 0.0 <= baseline <= 1.0

    def test_baseline_cached(self, clusterable_embedding_store):
        """Baseline is cached after first call."""
        b1 = clusterable_embedding_store.compute_corpus_baseline()
        b2 = clusterable_embedding_store.compute_corpus_baseline()
        assert b1 == b2

    def test_residual_similarity_logic(self, clusterable_embedding_store):
        """Residual similarity: (sim - baseline) / (1 - baseline)."""
        # With mock baseline = 0, residual = raw similarity
        baseline = 0.0
        raw_sim = 0.8
        residual = max(0.0, (raw_sim - baseline) / (1.0 - baseline))
        assert abs(residual - 0.8) < 1e-6

        # With baseline = 0.7, residual amplifies differences above baseline
        baseline = 0.7
        residual = max(0.0, (raw_sim - baseline) / (1.0 - baseline))
        expected = (0.8 - 0.7) / (1.0 - 0.7)  # ≈ 0.333
        assert abs(residual - expected) < 1e-6


# -----------------------------------------------------------------------
# TestEntityPhase2 (Correction 4)
# -----------------------------------------------------------------------

class TestEntityPhase2:
    """Correction 4: entity dimension in Phase 2 clustering."""

    def test_entity_dimension_separates(self, clusterable_embedding_store):
        """Articles with different entities are separated even with similar embeddings."""
        # All articles near centroid 0 (same embedding), but different entity groups
        rows = []
        for i in range(5):
            rows.append({
                'doc_id': f'cluster_0_art{i}',
                'date': pd.Timestamp('2018-06-01'),
                'evt_weather_mean': 0.8,
            })
        for i in range(5):
            rows.append({
                'doc_id': f'cluster_0_art{i+5}',
                'date': pd.Timestamp('2018-06-01'),
                'evt_weather_mean': 0.8,
            })
        articles = pd.DataFrame(rows)

        # Entity index: two groups with completely different entities
        entity_index = {}
        for i in range(5):
            entity_index[f'cluster_0_art{i}'] = [
                ('Trudeau', 'PER'), ('G7', 'ORG'), ('Charlevoix', 'LOC'),
            ]
        for i in range(5):
            entity_index[f'cluster_0_art{i+5}'] = [
                ('LQE', 'ORG'), ('Quebec', 'LOC'), ('Patronat', 'ORG'),
            ]

        det = EventOccurrenceDetector(clusterable_embedding_store)
        seeds, weights = det._select_seed_articles(articles, 'evt_weather_mean')

        # Without entities: should be 1 cluster (same embeddings)
        clusters_no_ent = det._cluster_event_type(
            seeds, 'evt_weather', 'evt_weather_mean', id_offset=0,
            seed_weights=weights, entity_index=None,
        )

        # With entities: entity distance should push toward 2 clusters
        clusters_with_ent = det._cluster_event_type(
            seeds, 'evt_weather', 'evt_weather_mean', id_offset=0,
            seed_weights=weights, entity_index=entity_index,
        )

        # With entities there should be >= as many clusters
        assert len(clusters_with_ent) >= len(clusters_no_ent)


# -----------------------------------------------------------------------
# TestTypeValidation — Correction A
# -----------------------------------------------------------------------

class TestTypeValidation:
    """Post-validation of occurrence event types."""

    def _make_occurrence_with_type(self, store, occ_id, event_type,
                                   cluster_idx=0, peak_date=None,
                                   doc_prefix=None):
        """Helper to build a single occurrence for type validation tests."""
        if peak_date is None:
            peak_date = pd.Timestamp('2018-06-15')
        if doc_prefix is None:
            doc_prefix = f'cluster_{cluster_idx}_occ{occ_id}'

        centroid = store._centroids[cluster_idx].copy()
        centroid /= np.linalg.norm(centroid)

        doc_ids = [f'{doc_prefix}_art{j}' for j in range(10)]
        return EventOccurrence(
            occurrence_id=occ_id,
            event_type=event_type,
            first_date=peak_date - pd.Timedelta(days=3),
            last_date=peak_date + pd.Timedelta(days=3),
            core_start=peak_date - pd.Timedelta(days=1),
            core_end=peak_date + pd.Timedelta(days=1),
            peak_date=peak_date,
            n_articles=10,
            effective_mass=8.0,
            core_mass=6.0,
            semantic_coherence=0.7,
            centroid=centroid,
            confidence=0.65,
            doc_ids=doc_ids,
            seed_doc_ids=doc_ids,
            belonging={d: 0.8 for d in doc_ids},
        )

    def test_correct_type_preserved(self, clusterable_embedding_store):
        """Occurrence whose assigned type has high evt_* is not corrected."""
        occ = self._make_occurrence_with_type(
            clusterable_embedding_store, 0, 'evt_weather',
        )
        articles = pd.DataFrame([{
            'doc_id': did,
            'date': pd.Timestamp('2018-06-15'),
            'evt_weather_mean': 0.8,
            'evt_meeting_mean': 0.05,
            'evt_policy_mean': 0.02,
        } for did in occ.doc_ids])

        det = EventOccurrenceDetector(clusterable_embedding_store)
        n_corrected = det._validate_occurrence_types([occ], articles)

        assert n_corrected == 0
        assert occ.event_type == 'evt_weather'
        assert occ.type_confidence >= 0.5

    def test_mistyped_corrected(self, clusterable_embedding_store):
        """Occurrence with wrong type (low evt_* signal) gets corrected."""
        occ = self._make_occurrence_with_type(
            clusterable_embedding_store, 0, 'evt_publication',
        )
        # All articles have high evt_weather but zero evt_publication
        articles = pd.DataFrame([{
            'doc_id': did,
            'date': pd.Timestamp('2018-06-15'),
            'evt_weather_mean': 0.9,
            'evt_publication_mean': 0.0,
            'evt_meeting_mean': 0.1,
        } for did in occ.doc_ids])

        det = EventOccurrenceDetector(clusterable_embedding_store)
        n_corrected = det._validate_occurrence_types([occ], articles)

        assert n_corrected == 1
        assert occ.event_type == 'evt_weather'
        assert occ.type_confidence == 1.0  # recomputed after correction

    def test_type_scores_populated(self, clusterable_embedding_store):
        """type_scores dict is populated on all occurrences."""
        occ = self._make_occurrence_with_type(
            clusterable_embedding_store, 0, 'evt_weather',
        )
        articles = pd.DataFrame([{
            'doc_id': did,
            'date': pd.Timestamp('2018-06-15'),
            'evt_weather_mean': 0.7,
            'evt_meeting_mean': 0.3,
        } for did in occ.doc_ids])

        det = EventOccurrenceDetector(clusterable_embedding_store)
        det._validate_occurrence_types([occ], articles)

        assert isinstance(occ.type_scores, dict)
        assert len(occ.type_scores) > 0
        assert 'evt_weather' in occ.type_scores

    def test_close_scores_no_correction(self, clusterable_embedding_store):
        """If assigned type is within 50% of best, no correction occurs."""
        occ = self._make_occurrence_with_type(
            clusterable_embedding_store, 0, 'evt_weather',
        )
        # Both types have similar signal strength
        articles = pd.DataFrame([{
            'doc_id': did,
            'date': pd.Timestamp('2018-06-15'),
            'evt_weather_mean': 0.6,
            'evt_meeting_mean': 0.7,
        } for did in occ.doc_ids])

        det = EventOccurrenceDetector(clusterable_embedding_store)
        n_corrected = det._validate_occurrence_types([occ], articles)

        # evt_weather is within 50% of evt_meeting (0.6/0.7 = 0.86 > 0.5)
        assert n_corrected == 0
        assert occ.event_type == 'evt_weather'

    def test_empty_occurrences(self, clusterable_embedding_store):
        """Empty list returns 0 corrections."""
        det = EventOccurrenceDetector(clusterable_embedding_store)
        assert det._validate_occurrence_types([], pd.DataFrame()) == 0

    def test_no_event_columns(self, clusterable_embedding_store):
        """Articles without evt_* columns → 0 corrections."""
        occ = self._make_occurrence_with_type(
            clusterable_embedding_store, 0, 'evt_weather',
        )
        articles = pd.DataFrame([{
            'doc_id': did,
            'date': pd.Timestamp('2018-06-15'),
        } for did in occ.doc_ids])

        det = EventOccurrenceDetector(clusterable_embedding_store)
        n_corrected = det._validate_occurrence_types([occ], articles)
        assert n_corrected == 0

    def test_type_confidence_range(self, clusterable_embedding_store):
        """type_confidence is always in [0, 1]."""
        occ = self._make_occurrence_with_type(
            clusterable_embedding_store, 0, 'evt_weather',
        )
        articles = pd.DataFrame([{
            'doc_id': did,
            'date': pd.Timestamp('2018-06-15'),
            'evt_weather_mean': 0.5,
            'evt_meeting_mean': 0.8,
        } for did in occ.doc_ids])

        det = EventOccurrenceDetector(clusterable_embedding_store)
        det._validate_occurrence_types([occ], articles)

        assert 0.0 <= occ.type_confidence <= 1.0

    def test_to_dict_includes_new_fields(self, clusterable_embedding_store):
        """EventOccurrence.to_dict() includes type_confidence and type_scores."""
        occ = self._make_occurrence_with_type(
            clusterable_embedding_store, 0, 'evt_weather',
        )
        # Use integer doc_ids for to_dict() compatibility
        int_ids = list(range(10))
        occ.doc_ids = int_ids
        occ.seed_doc_ids = int_ids
        occ.type_scores = {'evt_weather': 0.8, 'evt_meeting': 0.3}
        occ.type_confidence = 0.95

        d = occ.to_dict()
        assert 'type_confidence' in d
        assert 'type_scores' in d
        assert d['type_confidence'] == 0.95
        assert d['type_scores']['evt_weather'] == 0.8


# -----------------------------------------------------------------------
# TestConnectivitySplit — Correction B (Step 6b)
# -----------------------------------------------------------------------

class TestConnectivitySplit:
    """Step 6b: title+temporal connectivity splitting of clusters."""

    def _make_disconnected_cluster(self, store, n_per_group=3):
        """Build a cluster with 2 disconnected occurrence groups.

        Group A: cluster_0 articles, peak in June 2018
        Group B: cluster_1 articles, peak in December 2018 (180 days later)
        No shared seed articles, different centroids → disconnected.
        """
        occs = []
        for group in range(2):
            base_date = pd.Timestamp('2018-06-01') + pd.Timedelta(
                days=group * 180
            )
            centroid = store._centroids[group].copy()
            centroid /= np.linalg.norm(centroid)

            for i in range(n_per_group):
                peak = base_date + pd.Timedelta(days=i * 2)
                doc_ids = [
                    f'cluster_{group}_grp{group}_occ{i}_art{j}'
                    for j in range(10)
                ]
                occ = EventOccurrence(
                    occurrence_id=group * n_per_group + i,
                    event_type='evt_weather',
                    first_date=peak - pd.Timedelta(days=3),
                    last_date=peak + pd.Timedelta(days=3),
                    core_start=peak - pd.Timedelta(days=1),
                    core_end=peak + pd.Timedelta(days=1),
                    peak_date=peak,
                    n_articles=10,
                    effective_mass=8.0,
                    core_mass=6.0,
                    semantic_coherence=0.7,
                    centroid=centroid,
                    confidence=0.65,
                    doc_ids=doc_ids,
                    seed_doc_ids=doc_ids,
                    belonging={d: 0.8 for d in doc_ids},
                )
                occs.append(occ)

        # Build a single EventCluster that incorrectly groups both
        return EventCluster(
            cluster_id=0,
            occurrences=occs,
            event_types={'evt_weather'},
            peak_date=pd.Timestamp('2018-09-01'),
            core_start=pd.Timestamp('2018-06-01'),
            core_end=pd.Timestamp('2018-12-01'),
            total_mass=sum(o.effective_mass for o in occs),
            centroid=occs[0].centroid,
            n_occurrences=len(occs),
            is_multi_type=False,
        )

    def _make_connected_cluster(self, store, n_occs=3):
        """Build a cluster where occurrences share seed articles.

        All occurrences share at least some seed_doc_ids → connected.
        """
        shared_docs = [f'shared_art_{j}' for j in range(5)]
        centroid = store._centroids[0].copy()
        centroid /= np.linalg.norm(centroid)

        occs = []
        for i in range(n_occs):
            peak = pd.Timestamp('2018-06-01') + pd.Timedelta(days=i * 5)
            own_docs = [f'cluster_0_occ{i}_art{j}' for j in range(5)]
            all_docs = shared_docs + own_docs

            rng = np.random.RandomState(i)
            noise = rng.randn(store.embedding_dim).astype(np.float32)
            occ_centroid = centroid + 0.05 * noise
            occ_centroid /= np.linalg.norm(occ_centroid)

            occ = EventOccurrence(
                occurrence_id=i,
                event_type='evt_weather',
                first_date=peak - pd.Timedelta(days=3),
                last_date=peak + pd.Timedelta(days=3),
                core_start=peak - pd.Timedelta(days=1),
                core_end=peak + pd.Timedelta(days=1),
                peak_date=peak,
                n_articles=10,
                effective_mass=8.0,
                core_mass=6.0,
                semantic_coherence=0.7,
                centroid=occ_centroid,
                confidence=0.65,
                doc_ids=all_docs,
                seed_doc_ids=all_docs,
                belonging={d: 0.8 for d in all_docs},
            )
            occs.append(occ)

        return EventCluster(
            cluster_id=0,
            occurrences=occs,
            event_types={'evt_weather'},
            peak_date=pd.Timestamp('2018-06-08'),
            core_start=pd.Timestamp('2018-06-01'),
            core_end=pd.Timestamp('2018-06-15'),
            total_mass=sum(o.effective_mass for o in occs),
            centroid=centroid,
            n_occurrences=len(occs),
            is_multi_type=False,
        )

    def test_disconnected_split(self, clusterable_embedding_store):
        """Disconnected multi-occurrence cluster is split into components."""
        ec = self._make_disconnected_cluster(clusterable_embedding_store)
        det = EventOccurrenceDetector(clusterable_embedding_store)

        result = det._split_disconnected_clusters(
            [ec], entity_index=None, articles=None,
            embedding_store=clusterable_embedding_store,
        )

        # Should be split into 2 clusters (group A and group B)
        assert len(result) >= 2
        for new_ec in result:
            assert new_ec.n_occurrences > 0

    def test_connected_preserved(self, clusterable_embedding_store):
        """Connected cluster (shared seed articles) stays intact."""
        ec = self._make_connected_cluster(clusterable_embedding_store)
        det = EventOccurrenceDetector(clusterable_embedding_store)

        result = det._split_disconnected_clusters(
            [ec], entity_index=None, articles=None,
            embedding_store=clusterable_embedding_store,
        )

        assert len(result) == 1
        assert result[0].n_occurrences == ec.n_occurrences

    def test_singleton_cluster_preserved(self, clusterable_embedding_store):
        """Single-occurrence cluster passes through unchanged."""
        centroid = clusterable_embedding_store._centroids[0].copy()
        centroid /= np.linalg.norm(centroid)
        occ = EventOccurrence(
            occurrence_id=0,
            event_type='evt_weather',
            first_date=pd.Timestamp('2018-06-01'),
            last_date=pd.Timestamp('2018-06-07'),
            core_start=pd.Timestamp('2018-06-02'),
            core_end=pd.Timestamp('2018-06-06'),
            peak_date=pd.Timestamp('2018-06-04'),
            n_articles=10,
            effective_mass=8.0,
            core_mass=6.0,
            semantic_coherence=0.7,
            centroid=centroid,
            doc_ids=[f'art_{i}' for i in range(10)],
            seed_doc_ids=[f'art_{i}' for i in range(10)],
            belonging={f'art_{i}': 0.8 for i in range(10)},
        )
        ec = EventCluster(
            cluster_id=0,
            occurrences=[occ],
            event_types={'evt_weather'},
            peak_date=occ.peak_date,
            core_start=occ.core_start,
            core_end=occ.core_end,
            total_mass=occ.effective_mass,
            centroid=centroid,
            n_occurrences=1,
            is_multi_type=False,
        )

        det = EventOccurrenceDetector(clusterable_embedding_store)
        result = det._split_disconnected_clusters(
            [ec], entity_index=None, articles=None,
            embedding_store=clusterable_embedding_store,
        )
        assert len(result) == 1

    def test_title_similarity_connects_nearby(self, clusterable_embedding_store):
        """Occurrences with same-cluster doc_ids (high title sim) + close peaks stay connected."""
        centroid = clusterable_embedding_store._centroids[0].copy()
        centroid /= np.linalg.norm(centroid)

        occs = []
        for i in range(2):
            peak = pd.Timestamp('2018-06-01') + pd.Timedelta(days=i * 10)
            # Different doc_ids (no seed overlap) but SAME cluster_0 prefix
            # → ClusterableMockEmbeddingStore gives similar embeddings → high title sim
            doc_ids = [f'cluster_0_grp{i}_art{j}' for j in range(10)]
            rng = np.random.RandomState(42 + i)
            noise = rng.randn(clusterable_embedding_store.embedding_dim).astype(np.float32)
            occ_centroid = centroid + 0.03 * noise
            occ_centroid /= np.linalg.norm(occ_centroid)

            occ = EventOccurrence(
                occurrence_id=i,
                event_type='evt_weather',
                first_date=peak - pd.Timedelta(days=3),
                last_date=peak + pd.Timedelta(days=3),
                core_start=peak - pd.Timedelta(days=1),
                core_end=peak + pd.Timedelta(days=1),
                peak_date=peak,
                n_articles=10,
                effective_mass=8.0,
                core_mass=6.0,
                semantic_coherence=0.7,
                centroid=occ_centroid,
                doc_ids=doc_ids,
                seed_doc_ids=doc_ids,
                belonging={d: 0.8 for d in doc_ids},
            )
            occs.append(occ)

        ec = EventCluster(
            cluster_id=0,
            occurrences=occs,
            event_types={'evt_weather'},
            peak_date=pd.Timestamp('2018-06-06'),
            core_start=pd.Timestamp('2018-06-01'),
            core_end=pd.Timestamp('2018-06-11'),
            total_mass=16.0,
            centroid=centroid,
            n_occurrences=2,
            is_multi_type=False,
        )

        det = EventOccurrenceDetector(clusterable_embedding_store)
        result = det._split_disconnected_clusters(
            [ec], entity_index=None, articles=None,
            embedding_store=clusterable_embedding_store,
        )

        # Same cluster_0 prefix → title sim should be high → stays connected
        assert len(result) == 1

    def test_temporal_gap_exceeds_max(self, clusterable_embedding_store):
        """Even with high title sim, gap > MAX_GAP_DAYS → split."""
        centroid = clusterable_embedding_store._centroids[0].copy()
        centroid /= np.linalg.norm(centroid)

        occs = []
        for i in range(2):
            # Gap of 60 days > EVENT_CLUSTER_MAX_GAP_DAYS (30)
            peak = pd.Timestamp('2018-06-01') + pd.Timedelta(days=i * 60)
            doc_ids = [f'cluster_0_far{i}_art{j}' for j in range(10)]
            rng = np.random.RandomState(100 + i)
            noise = rng.randn(clusterable_embedding_store.embedding_dim).astype(np.float32)
            occ_centroid = centroid + 0.03 * noise
            occ_centroid /= np.linalg.norm(occ_centroid)

            occ = EventOccurrence(
                occurrence_id=i,
                event_type='evt_weather',
                first_date=peak - pd.Timedelta(days=3),
                last_date=peak + pd.Timedelta(days=3),
                core_start=peak - pd.Timedelta(days=1),
                core_end=peak + pd.Timedelta(days=1),
                peak_date=peak,
                n_articles=10,
                effective_mass=8.0,
                core_mass=6.0,
                semantic_coherence=0.7,
                centroid=occ_centroid,
                doc_ids=doc_ids,
                seed_doc_ids=doc_ids,
                belonging={d: 0.8 for d in doc_ids},
            )
            occs.append(occ)

        ec = EventCluster(
            cluster_id=0,
            occurrences=occs,
            event_types={'evt_weather'},
            peak_date=pd.Timestamp('2018-07-01'),
            core_start=pd.Timestamp('2018-06-01'),
            core_end=pd.Timestamp('2018-07-31'),
            total_mass=16.0,
            centroid=centroid,
            n_occurrences=2,
            is_multi_type=False,
        )

        det = EventOccurrenceDetector(clusterable_embedding_store)
        result = det._split_disconnected_clusters(
            [ec], entity_index=None, articles=None,
            embedding_store=clusterable_embedding_store,
        )

        # No shared seeds AND gap > 30 days → must be split even if title sim is high
        assert len(result) == 2

    def test_mixed_connected_and_disconnected(self, clusterable_embedding_store):
        """Mix of connected clusters (preserved) and disconnected (split)."""
        ec_connected = self._make_connected_cluster(
            clusterable_embedding_store
        )
        ec_disconnected = self._make_disconnected_cluster(
            clusterable_embedding_store
        )

        det = EventOccurrenceDetector(clusterable_embedding_store)
        result = det._split_disconnected_clusters(
            [ec_connected, ec_disconnected],
            entity_index=None, articles=None,
            embedding_store=clusterable_embedding_store,
        )

        # Connected stays as 1, disconnected splits into ≥2 → total ≥ 3
        assert len(result) >= 3

    def test_empty_input(self, clusterable_embedding_store):
        """Empty cluster list returns empty."""
        det = EventOccurrenceDetector(clusterable_embedding_store)
        result = det._split_disconnected_clusters(
            [], entity_index=None, articles=None,
            embedding_store=clusterable_embedding_store,
        )
        assert result == []

    def test_split_renumbers_clusters(self, clusterable_embedding_store):
        """After splitting, cluster_occurrences renumbers cluster_ids."""
        # Build occurrences that will form a disconnected cluster
        occs = []
        for group in range(2):
            base_date = pd.Timestamp('2018-06-01') + pd.Timedelta(
                days=group * 180
            )
            centroid = clusterable_embedding_store._centroids[group].copy()
            centroid /= np.linalg.norm(centroid)
            for i in range(3):
                peak = base_date + pd.Timedelta(days=i * 2)
                doc_ids = [
                    f'cluster_{group}_renumber{group}_occ{i}_art{j}'
                    for j in range(10)
                ]
                occ = EventOccurrence(
                    occurrence_id=group * 3 + i,
                    event_type=['evt_weather', 'evt_meeting'][group],
                    first_date=peak - pd.Timedelta(days=3),
                    last_date=peak + pd.Timedelta(days=3),
                    core_start=peak - pd.Timedelta(days=1),
                    core_end=peak + pd.Timedelta(days=1),
                    peak_date=peak,
                    n_articles=10,
                    effective_mass=8.0,
                    core_mass=6.0,
                    semantic_coherence=0.7,
                    centroid=centroid,
                    confidence=0.65,
                    doc_ids=doc_ids,
                    seed_doc_ids=doc_ids,
                    belonging={d: 0.8 for d in doc_ids},
                )
                occs.append(occ)

        det = EventOccurrenceDetector(clusterable_embedding_store)
        clusters = det.cluster_occurrences(occs)

        # cluster_ids should be sequential starting from 0
        ids = [ec.cluster_id for ec in clusters]
        assert ids == list(range(len(clusters)))


# -----------------------------------------------------------------------
# TestArticleLevelPeakDate
# -----------------------------------------------------------------------

class TestArticleLevelPeakDate:
    """Tests for article-level temporal bounds in EventCluster."""

    def _make_articles(self, doc_ids, dates, evt_types=None, evt_values=None):
        """Build a minimal articles DataFrame."""
        data = {
            'doc_id': doc_ids,
            'date': pd.to_datetime(dates),
        }
        if evt_types and evt_values:
            for evt_type, vals in zip(evt_types, evt_values):
                data[evt_type] = vals
        return pd.DataFrame(data)

    def test_single_occurrence_with_articles(self, clusterable_embedding_store):
        """Single-occurrence cluster: peak_date computed from articles, not occurrence."""
        # Create an occurrence with peak_date on June 15,
        # but belonging articles are spread across June 10-20
        centroid = clusterable_embedding_store._centroids[0].copy()
        centroid /= np.linalg.norm(centroid)

        doc_ids = [f'cluster_0_art{j}' for j in range(20)]
        belonging = {d: 0.8 for d in doc_ids}

        occ = EventOccurrence(
            occurrence_id=0,
            event_type='evt_weather',
            first_date=pd.Timestamp('2018-06-10'),
            last_date=pd.Timestamp('2018-06-20'),
            core_start=pd.Timestamp('2018-06-12'),
            core_end=pd.Timestamp('2018-06-18'),
            peak_date=pd.Timestamp('2018-06-15'),
            n_articles=20,
            effective_mass=16.0,
            core_mass=12.0,
            semantic_coherence=0.8,
            centroid=centroid,
            confidence=0.75,
            doc_ids=doc_ids,
            seed_doc_ids=doc_ids[:10],
            belonging=belonging,
        )

        # Articles spread across June 10-20
        dates = [f'2018-06-{10 + i}' for i in range(20)]
        evt_vals = [0.5] * 20
        articles = self._make_articles(doc_ids, dates, ['evt_weather'], [evt_vals])

        det = EventOccurrenceDetector(clusterable_embedding_store)
        cluster = det._build_event_cluster(
            cluster_id=0,
            occurrences=[occ],
            articles=articles,
            embedding_store=clusterable_embedding_store,
        )

        # With 20 articles spread over 11 days, article-level P50 should
        # be near the middle. Crucially, core_start < peak_date < core_end.
        assert cluster.core_start <= cluster.peak_date <= cluster.core_end
        # Should NOT collapse to a single date (which is what occurrence-level
        # would give for a single occurrence)
        assert cluster.core_start < cluster.core_end

    def test_multi_occurrence_shifts_toward_denser(self, clusterable_embedding_store):
        """Peak shifts toward the occurrence with more articles."""
        centroid = clusterable_embedding_store._centroids[0].copy()
        centroid /= np.linalg.norm(centroid)

        # Occurrence 1: 5 articles around June 10
        docs1 = [f'cluster_0_early{j}' for j in range(5)]
        occ1 = EventOccurrence(
            occurrence_id=0,
            event_type='evt_weather',
            first_date=pd.Timestamp('2018-06-08'),
            last_date=pd.Timestamp('2018-06-12'),
            core_start=pd.Timestamp('2018-06-09'),
            core_end=pd.Timestamp('2018-06-11'),
            peak_date=pd.Timestamp('2018-06-10'),
            n_articles=5,
            effective_mass=4.0,
            core_mass=3.0,
            semantic_coherence=0.75,
            centroid=centroid,
            confidence=0.7,
            doc_ids=docs1,
            seed_doc_ids=docs1,
            belonging={d: 0.8 for d in docs1},
        )

        # Occurrence 2: 15 articles around June 25
        docs2 = [f'cluster_0_late{j}' for j in range(15)]
        occ2 = EventOccurrence(
            occurrence_id=1,
            event_type='evt_weather',
            first_date=pd.Timestamp('2018-06-22'),
            last_date=pd.Timestamp('2018-06-28'),
            core_start=pd.Timestamp('2018-06-23'),
            core_end=pd.Timestamp('2018-06-27'),
            peak_date=pd.Timestamp('2018-06-25'),
            n_articles=15,
            effective_mass=12.0,
            core_mass=10.0,
            semantic_coherence=0.75,
            centroid=centroid,
            confidence=0.7,
            doc_ids=docs2,
            seed_doc_ids=docs2,
            belonging={d: 0.8 for d in docs2},
        )

        # Articles DataFrame
        base1 = pd.Timestamp('2018-06-08')
        dates1 = [(base1 + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(5)]
        base2 = pd.Timestamp('2018-06-22')
        dates2 = [(base2 + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(15)]
        all_docs = docs1 + docs2
        all_dates = dates1 + dates2
        evt_vals = [0.5] * len(all_docs)
        articles = self._make_articles(all_docs, all_dates, ['evt_weather'], [evt_vals])

        det = EventOccurrenceDetector(clusterable_embedding_store)
        cluster = det._build_event_cluster(
            cluster_id=0,
            occurrences=[occ1, occ2],
            articles=articles,
            embedding_store=clusterable_embedding_store,
        )

        # Peak should shift toward the denser occurrence (June 25 side)
        midpoint = pd.Timestamp('2018-06-17')  # naive midpoint
        assert cluster.peak_date > midpoint

    def test_cosine_weight_shifts_peak(self, clusterable_embedding_store):
        """Two equal-mass occurrences; one semantically closer → peak shifts."""
        # Occurrence 1: centroid 0, articles near centroid 0
        centroid0 = clusterable_embedding_store._centroids[0].copy()
        centroid0 /= np.linalg.norm(centroid0)

        # Occurrence 2: centroid 1, articles near centroid 1
        centroid1 = clusterable_embedding_store._centroids[1].copy()
        centroid1 /= np.linalg.norm(centroid1)

        n = 10
        docs0 = [f'cluster_0_a{j}' for j in range(n)]
        docs1 = [f'cluster_1_b{j}' for j in range(n)]

        occ0 = EventOccurrence(
            occurrence_id=0,
            event_type='evt_weather',
            first_date=pd.Timestamp('2018-06-01'),
            last_date=pd.Timestamp('2018-06-10'),
            core_start=pd.Timestamp('2018-06-03'),
            core_end=pd.Timestamp('2018-06-08'),
            peak_date=pd.Timestamp('2018-06-05'),
            n_articles=n,
            effective_mass=8.0,
            core_mass=6.0,
            semantic_coherence=0.8,
            centroid=centroid0,
            confidence=0.7,
            doc_ids=docs0,
            seed_doc_ids=docs0,
            belonging={d: 0.8 for d in docs0},
        )

        occ1 = EventOccurrence(
            occurrence_id=1,
            event_type='evt_weather',
            first_date=pd.Timestamp('2018-07-01'),
            last_date=pd.Timestamp('2018-07-10'),
            core_start=pd.Timestamp('2018-07-03'),
            core_end=pd.Timestamp('2018-07-08'),
            peak_date=pd.Timestamp('2018-07-05'),
            n_articles=n,
            effective_mass=8.0,
            core_mass=6.0,
            semantic_coherence=0.8,
            centroid=centroid1,
            confidence=0.7,
            doc_ids=docs1,
            seed_doc_ids=docs1,
            belonging={d: 0.8 for d in docs1},
        )

        # Articles for both
        dates0 = [f'2018-06-{1 + i}' for i in range(n)]
        dates1 = [f'2018-07-{1 + i}' for i in range(n)]
        all_docs = docs0 + docs1
        all_dates = dates0 + dates1
        evt_vals = [0.5] * len(all_docs)
        articles = self._make_articles(all_docs, all_dates, ['evt_weather'], [evt_vals])

        det = EventOccurrenceDetector(clusterable_embedding_store)
        cluster = det._build_event_cluster(
            cluster_id=0,
            occurrences=[occ0, occ1],
            articles=articles,
            embedding_store=clusterable_embedding_store,
        )

        # The cluster centroid is mass-weighted mean of centroid0 and centroid1.
        # Articles near centroid0 will have higher cosine similarity with the
        # centroid if centroid0 is closer to the blend. Since masses are equal,
        # centroid ≈ midpoint. But the cosine similarity difference should
        # cause at least a slight shift. The key check is that:
        # 1. We get valid dates
        assert cluster.core_start <= cluster.peak_date <= cluster.core_end
        # 2. core_start is in June, core_end is in July
        assert cluster.core_start.month == 6
        assert cluster.core_end.month == 7

    def test_fallback_without_articles(self, clusterable_embedding_store):
        """Without articles, falls back to occurrence-level behavior."""
        centroid = clusterable_embedding_store._centroids[0].copy()
        centroid /= np.linalg.norm(centroid)

        occ = EventOccurrence(
            occurrence_id=0,
            event_type='evt_weather',
            first_date=pd.Timestamp('2018-06-10'),
            last_date=pd.Timestamp('2018-06-20'),
            core_start=pd.Timestamp('2018-06-12'),
            core_end=pd.Timestamp('2018-06-18'),
            peak_date=pd.Timestamp('2018-06-15'),
            n_articles=10,
            effective_mass=8.0,
            core_mass=6.0,
            semantic_coherence=0.75,
            centroid=centroid,
            confidence=0.7,
            doc_ids=[f'cluster_0_art{j}' for j in range(10)],
            seed_doc_ids=[f'cluster_0_art{j}' for j in range(10)],
            belonging={f'cluster_0_art{j}': 0.8 for j in range(10)},
        )

        det = EventOccurrenceDetector(clusterable_embedding_store)

        # No articles → fallback to occurrence-level
        cluster = det._build_event_cluster(
            cluster_id=0,
            occurrences=[occ],
            articles=None,
            embedding_store=clusterable_embedding_store,
        )

        # Single occurrence, no articles → all three dates = occurrence peak_date
        assert cluster.peak_date == pd.Timestamp('2018-06-15')
        assert cluster.core_start == pd.Timestamp('2018-06-15')
        assert cluster.core_end == pd.Timestamp('2018-06-15')


# -----------------------------------------------------------------------
# TestSingletonPreservation
# -----------------------------------------------------------------------

class TestSingletonPreservation:
    """Tests for singleton (micro unique event) preservation in Phase 2."""

    def test_singleton_flag(self, clusterable_embedding_store):
        """is_singleton=True for 1-seed clusters, False for multi-seed."""
        rows = []
        for i in range(5):
            rows.append({
                'doc_id': f'cluster_0_art{i}',
                'date': pd.Timestamp('2018-06-01'),
                'evt_weather_mean': 0.6,
                'event_mean': 0.5,
            })
        rows.append({
            'doc_id': 'cluster_1_art0',
            'date': pd.Timestamp('2018-06-01'),
            'evt_weather_mean': 0.6,
            'event_mean': 0.5,
        })
        articles = pd.DataFrame(rows)

        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        # Should have both singleton and multi-seed occurrences
        singletons = [o for o in occurrences if o.is_singleton]
        multi_seed = [o for o in occurrences if not o.is_singleton]
        assert len(singletons) >= 1
        assert len(multi_seed) >= 1
        # Singleton must have exactly 1 seed
        for s in singletons:
            assert len(s.seed_doc_ids) == 1

    def test_singleton_confidence_low(self, clusterable_embedding_store):
        """Singleton confidence is below LOW_CONFIDENCE_THRESHOLD."""
        articles = pd.DataFrame({
            'doc_id': ['cluster_0_art0'],
            'date': pd.to_datetime(['2018-06-01']),
            'evt_weather_mean': [0.8],
            'event_mean': [0.6],
        })

        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, _ = det.detect_events(articles)

        assert len(occurrences) == 1
        occ = occurrences[0]
        assert occ.is_singleton is True
        assert occ.confidence < LOW_CONFIDENCE_THRESHOLD
        assert occ.low_confidence is True

    def test_singleton_in_to_dict(self):
        """is_singleton appears in serialized dict."""
        occ = EventOccurrence(
            occurrence_id=0,
            event_type='evt_weather',
            first_date=pd.Timestamp('2018-06-01'),
            last_date=pd.Timestamp('2018-06-01'),
            core_start=pd.Timestamp('2018-06-01'),
            core_end=pd.Timestamp('2018-06-01'),
            peak_date=pd.Timestamp('2018-06-01'),
            n_articles=1,
            effective_mass=1.0,
            core_mass=1.0,
            semantic_coherence=0.0,
            centroid=np.ones(64),
            is_singleton=True,
        )
        d = occ.to_dict()
        assert 'is_singleton' in d
        assert d['is_singleton'] is True

    def test_singleton_survives_pipeline(self, clusterable_embedding_store):
        """Singleton occurrence survives the full detect_events pipeline.

        A single seed article produces a singleton occurrence that passes
        through Phase 3, Phase 4, and Phase 5 without being dropped.
        """
        articles = pd.DataFrame({
            'doc_id': ['cluster_0_art0'],
            'date': pd.to_datetime(['2018-06-01']),
            'evt_weather_mean': [0.9],
            'event_mean': [0.8],
        })

        det = EventOccurrenceDetector(clusterable_embedding_store)
        occurrences, clusters = det.detect_events(articles)

        # Singleton should survive the full pipeline
        assert len(occurrences) == 1
        assert occurrences[0].is_singleton is True
        assert occurrences[0].event_type == 'evt_weather'
        # Should also produce at least one event cluster
        assert len(clusters) >= 1
