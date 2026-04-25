"""
Tests for cascade_detector.analysis.stabsel_paradigm — StabSel v2 paradigm analysis.

Covers:
  - AR order selection and AR column construction
  - OLS post-selection v2 (HAC + bootstrap)
  - Temporal cross-validation and train/test split
  - Impact metrics (magnitude, shift contribution)
  - Frame centroid construction
  - Weighted lagged mass matrix (Model A) and cascade lagged matrix (Model B)
  - Role classification (per-pair and global alignment)
  - DataFrame export helpers
  - StabSelParadigmAnalyzer integration
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any

from cascade_detector.core.constants import FRAMES, FRAME_COLUMNS
from cascade_detector.analysis.stabsel_paradigm import (
    select_ar_order,
    build_ar_columns,
    ols_post_selection_v2,
    temporal_cross_validation,
    train_test_evaluation,
    compute_impact_magnitude,
    compute_shift_contribution,
    build_frame_centroid,
    build_weighted_lagged_mass_paradigm,
    build_cascade_lagged_matrix,
    classify_model_a_roles,
    classify_model_b_roles,
    run_model,
    results_to_dataframe_a,
    results_to_dataframe_b,
    StabSelParadigmAnalyzer,
    StabSelParadigmResults,
    MAX_LAG_PARADIGM,
    MAX_AR_ORDER,
    ALPHA_SIG,
    MIN_D_SUM,
)

# Import the mock from conftest
from tests.conftest import MockEmbeddingStore


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def paradigm_timeline():
    """Mock paradigm timeline: 200 days with 8 paradigm columns."""
    dates = pd.date_range('2018-04-01', periods=200, freq='D')
    data = {'date': dates}
    rng = np.random.default_rng(123)
    for frame in FRAMES:
        data[f'paradigm_{frame}'] = rng.random(200) * 0.3
    return pd.DataFrame(data)


@pytest.fixture
def mock_articles():
    """Mock articles DataFrame with frame columns and doc_ids."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range('2018-04-01', periods=200, freq='D')
    df = pd.DataFrame({
        'doc_id': [f'doc_{i}' for i in range(n)],
        'date': rng.choice(dates, size=n),
    })
    # Add frame columns (using _mean suffix as in production)
    for frame, col in FRAME_COLUMNS.items():
        df[f'{col}_mean'] = rng.random(n) * 0.5
    return df


@pytest.fixture
def embedding_store():
    return MockEmbeddingStore()


@pytest.fixture
def mock_cascade():
    """A single mock cascade."""
    cascade = Mock()
    cascade.cascade_id = 'Eco_20180601_1'
    cascade.frame = 'Eco'
    cascade.peak_date = pd.Timestamp('2018-07-01')
    cascade.total_score = 0.65
    dates = pd.date_range('2018-06-15', '2018-07-15', freq='D')
    cascade.daily_composite = pd.Series(
        np.random.default_rng(42).random(len(dates)) * 0.3,
        index=dates
    )
    return cascade


@pytest.fixture
def mock_occurrence():
    """A single mock EventOccurrence."""
    occ = Mock()
    occ.occurrence_id = 1
    occ.event_type = 'evt_weather'
    occ.peak_date = pd.Timestamp('2018-06-20')
    occ.core_start = pd.Timestamp('2018-06-15')
    occ.core_end = pd.Timestamp('2018-06-25')
    occ.effective_mass = 10.0
    occ.centroid = None
    occ.confidence = 0.6
    occ.entities = ['entity1']
    occ.seed_doc_ids = [f'doc_{i}' for i in range(10)]
    occ.belonging = {f'doc_{i}': 0.8 for i in range(20)}
    return occ


@pytest.fixture
def mock_cluster(mock_occurrence):
    """A single mock EventCluster."""
    cluster = Mock()
    cluster.cluster_id = 1
    cluster.occurrences = [mock_occurrence]
    cluster.event_types = {'evt_weather'}
    cluster.peak_date = pd.Timestamp('2018-06-20')
    cluster.core_start = pd.Timestamp('2018-06-15')
    cluster.core_end = pd.Timestamp('2018-06-25')
    cluster.total_mass = 10.0
    cluster.centroid = None
    cluster.n_occurrences = 1
    cluster.is_multi_type = False
    cluster.strength = 0.5
    cluster.entities = ['entity1']
    cluster.dominant_type = 'evt_weather'
    return cluster


# ── Test AR order selection ──────────────────────────────────────────────────

class TestSelectArOrder:

    def test_returns_valid_range(self, rng):
        y = rng.standard_normal(200)
        p = select_ar_order(y)
        assert 1 <= p <= MAX_AR_ORDER

    def test_prefers_low_for_white_noise(self, rng):
        y = rng.standard_normal(200)
        p = select_ar_order(y)
        assert p <= 3  # white noise should not need many AR terms

    def test_detects_ar1_process(self, rng):
        """AR(1) process with phi=0.8 should select p>=1."""
        y = np.zeros(300)
        y[0] = rng.standard_normal()
        for t in range(1, 300):
            y[t] = 0.8 * y[t - 1] + rng.standard_normal() * 0.1
        p = select_ar_order(y)
        assert p >= 1

    def test_short_series(self):
        """Very short series should still return valid p."""
        y = np.ones(10)
        p = select_ar_order(y)
        assert 1 <= p <= MAX_AR_ORDER

    def test_constant_series(self):
        """Constant series has zero variance — should still return valid p."""
        y = np.ones(100)
        p = select_ar_order(y)
        assert 1 <= p <= MAX_AR_ORDER


# ── Test AR column construction ──────────────────────────────────────────────

class TestBuildArColumns:

    def test_shape(self):
        y = np.arange(100, dtype=float)
        ar_cols = build_ar_columns(y, 3)
        assert ar_cols.shape == (100, 3)

    def test_values_lag1(self):
        y = np.arange(10, dtype=float)
        ar_cols = build_ar_columns(y, 1)
        # Column 0 should be lag-1: y[t-1]
        assert ar_cols[0, 0] == 0.0  # no history
        assert ar_cols[1, 0] == 0.0  # y[0]
        assert ar_cols[2, 0] == 1.0  # y[1]
        assert ar_cols[5, 0] == 4.0  # y[4]

    def test_values_lag2(self):
        y = np.arange(10, dtype=float)
        ar_cols = build_ar_columns(y, 2)
        # Column 1 should be lag-2: y[t-2]
        assert ar_cols[0, 1] == 0.0
        assert ar_cols[1, 1] == 0.0
        assert ar_cols[2, 1] == 0.0  # y[0]
        assert ar_cols[3, 1] == 1.0  # y[1]

    def test_single_lag(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ar_cols = build_ar_columns(y, 1)
        assert ar_cols.shape == (5, 1)


# ── Test OLS post-selection v2 ───────────────────────────────────────────────

class TestOlsPostSelectionV2:

    def test_empty_stable_returns_zero_r2(self, rng):
        n = 100
        y = rng.standard_normal(n)
        X = np.column_stack([np.linspace(0, 1, n), rng.standard_normal((n, 5))])
        stable_mask = np.zeros(5, dtype=bool)  # no treatment stable
        lag_labels = [(i, 0) for i in range(5)]
        result = ols_post_selection_v2(y, X, stable_mask, lag_labels, n_controls=1)
        assert result['r2'] == 0.0
        assert result['n_stable'] == 0
        assert result['cluster_results'] == {}

    def test_returns_hac_and_boot_pvalues(self, rng):
        n = 150
        X_treat = rng.standard_normal((n, 4))
        y = 0.5 * X_treat[:, 0] + rng.standard_normal(n) * 0.1
        trend = np.linspace(0, 1, n)
        X = np.column_stack([trend, X_treat])
        stable_mask = np.array([True, False, False, False])
        lag_labels = [(0, 0), (1, 0), (2, 0), (3, 0)]
        result = ols_post_selection_v2(y, X, stable_mask, lag_labels,
                                       n_controls=1, n_bootstrap=50)
        assert result['n_stable'] == 1
        assert result['r2'] > 0
        cr = result['cluster_results']
        assert 0 in cr
        assert 'p_value_hac' in cr[0]
        assert 'p_value_boot' in cr[0]
        assert 'p_value' in cr[0]
        assert cr[0]['p_value'] == cr[0]['p_value_hac']  # primary = HAC

    def test_r2_reasonable(self, rng):
        n = 200
        X_treat = rng.standard_normal((n, 2))
        y = 2.0 * X_treat[:, 0] - 1.5 * X_treat[:, 1] + rng.standard_normal(n) * 0.3
        trend = np.linspace(0, 1, n)
        X = np.column_stack([trend, X_treat])
        stable_mask = np.array([True, True])
        lag_labels = [(0, 0), (1, 0)]
        result = ols_post_selection_v2(y, X, stable_mask, lag_labels, n_controls=1)
        assert result['r2'] > 0.5

    def test_aggregates_multiple_lags(self, rng):
        """Multiple lags of same cluster should be aggregated into net_beta."""
        n = 150
        X_treat = rng.standard_normal((n, 3))
        y = X_treat[:, 0] + X_treat[:, 1] + rng.standard_normal(n) * 0.1
        trend = np.linspace(0, 1, n)
        X = np.column_stack([trend, X_treat])
        # Cluster 0 has lags 0 and 1, cluster 1 has lag 0
        stable_mask = np.array([True, True, True])
        lag_labels = [(0, 0), (0, 1), (1, 0)]
        result = ols_post_selection_v2(y, X, stable_mask, lag_labels, n_controls=1)
        cr = result['cluster_results']
        assert 0 in cr
        assert 1 in cr
        # Cluster 0 net_beta should be sum of lag 0 + lag 1 betas
        assert 'betas_by_lag' in cr[0]
        assert len(cr[0]['betas_by_lag']) == 2


# ── Test temporal cross-validation ───────────────────────────────────────────

class TestTemporalCrossValidation:

    @patch('cascade_detector.analysis.stabsel_paradigm.stability_selection')
    def test_returns_cv_metrics(self, mock_stabsel, rng):
        n = 200
        y = rng.standard_normal(n)
        X = np.column_stack([np.linspace(0, 1, n), rng.standard_normal((n, 5))])
        lag_labels = [(i, 0) for i in range(5)]
        # Mock: select first 2 treatment columns as stable
        mock_stabsel.return_value = (np.array([False, True, True, False, False, False]), None)
        result = temporal_cross_validation(y, X, lag_labels, n_controls=1, n_folds=3)
        assert 'r2_per_fold' in result
        assert 'r2_cv' in result
        assert 'stable_per_fold' in result

    @patch('cascade_detector.analysis.stabsel_paradigm.stability_selection')
    def test_small_segments_skip_cv(self, mock_stabsel, rng):
        n = 50  # segments will be < 30
        y = rng.standard_normal(n)
        X = np.column_stack([np.linspace(0, 1, n), rng.standard_normal((n, 3))])
        lag_labels = [(i, 0) for i in range(3)]
        result = temporal_cross_validation(y, X, lag_labels, n_controls=1, n_folds=3)
        assert result['r2_per_fold'] == []
        assert np.isnan(result['r2_cv'])


# ── Test train/test evaluation ───────────────────────────────────────────────

class TestTrainTestEvaluation:

    @patch('cascade_detector.analysis.stabsel_paradigm.stability_selection')
    def test_returns_r2_metrics(self, mock_stabsel, rng):
        n = 200
        y = rng.standard_normal(n)
        X = np.column_stack([np.linspace(0, 1, n), rng.standard_normal((n, 5))])
        lag_labels = [(i, 0) for i in range(5)]
        mock_stabsel.return_value = (np.array([False, True, True, False, False, False]), None)
        result = train_test_evaluation(y, X, lag_labels, n_controls=1)
        assert 'r2_train' in result
        assert 'r2_test' in result
        assert 'n_stable_train' in result

    @patch('cascade_detector.analysis.stabsel_paradigm.stability_selection')
    def test_short_test_returns_nan(self, mock_stabsel, rng):
        n = 20  # test set < 10
        y = rng.standard_normal(n)
        X = np.column_stack([np.linspace(0, 1, n), rng.standard_normal((n, 3))])
        lag_labels = [(i, 0) for i in range(3)]
        result = train_test_evaluation(y, X, lag_labels, n_controls=1, train_frac=0.95)
        assert np.isnan(result['r2_train'])
        assert np.isnan(result['r2_test'])


# ── Test impact metrics ──────────────────────────────────────────────────────

class TestImpactMetrics:

    def test_impact_magnitude(self):
        beta = np.array([3.0, 4.0])  # norm = 5
        assert compute_impact_magnitude(beta, 2.0) == pytest.approx(10.0)

    def test_impact_magnitude_zero_weight(self):
        beta = np.array([1.0, 2.0])
        assert compute_impact_magnitude(beta, 0.0) == 0.0

    def test_shift_contribution(self):
        paradigm = pd.DataFrame({
            'paradigm_Cult': [0.1, 0.2, 0.3, 0.4, 0.5],
            'paradigm_Eco': [0.5, 0.4, 0.3, 0.2, 0.1],
        }, index=pd.date_range('2018-06-01', periods=5))
        # Add remaining frames as zeros
        for f in FRAMES:
            col = f'paradigm_{f}'
            if col not in paradigm.columns:
                paradigm[col] = 0.0

        beta = np.zeros(len(FRAMES))
        beta[FRAMES.index('Cult')] = 1.0
        beta[FRAMES.index('Eco')] = 2.0

        result = compute_shift_contribution(
            beta, paradigm, pd.Timestamp('2018-06-03'), FRAMES, window=2)
        # Δ Cult = |0.5 - 0.1| = 0.4, Δ Eco = |0.1 - 0.5| = 0.4
        # contribution = 1.0 * 0.4 + 2.0 * 0.4 = 1.2
        assert result == pytest.approx(1.2)

    def test_shift_contribution_zero_beta(self):
        paradigm = pd.DataFrame({
            'paradigm_Cult': [0.1, 0.5],
        }, index=pd.date_range('2018-06-01', periods=2))
        for f in FRAMES:
            col = f'paradigm_{f}'
            if col not in paradigm.columns:
                paradigm[col] = 0.0
        beta = np.zeros(len(FRAMES))
        result = compute_shift_contribution(
            beta, paradigm, pd.Timestamp('2018-06-01'), FRAMES, window=1)
        assert result == 0.0


# ── Test role classification ─────────────────────────────────────────────────

class TestRoleClassification:

    def test_model_a_catalyst(self):
        ols = {'cluster_results': {1: {'net_beta': 0.5, 'p_value': 0.05}}}
        roles = classify_model_a_roles(ols)
        assert roles[1] == 'catalyst'

    def test_model_a_disruptor(self):
        ols = {'cluster_results': {1: {'net_beta': -0.5, 'p_value': 0.05}}}
        roles = classify_model_a_roles(ols)
        assert roles[1] == 'disruptor'

    def test_model_a_inert(self):
        ols = {'cluster_results': {1: {'net_beta': 0.5, 'p_value': 0.15}}}
        roles = classify_model_a_roles(ols)
        assert roles[1] == 'inert'

    def test_model_b_catalyst_own_frame(self):
        cascade = Mock()
        cascade.frame = 'Eco'
        ols = {'cluster_results': {'c1': {'net_beta': 0.5, 'p_value': 0.05}}}
        roles, flags = classify_model_b_roles(ols, 'Eco', {'c1': cascade})
        assert roles['c1'] == 'catalyst'
        assert flags['c1'] is True

    def test_model_b_disruptor_own_frame(self):
        cascade = Mock()
        cascade.frame = 'Eco'
        ols = {'cluster_results': {'c1': {'net_beta': -0.5, 'p_value': 0.05}}}
        roles, flags = classify_model_b_roles(ols, 'Eco', {'c1': cascade})
        assert roles['c1'] == 'disruptor'
        assert flags['c1'] is True

    def test_model_b_catalyst_cross_frame(self):
        cascade = Mock()
        cascade.frame = 'Cult'  # different from target 'Eco'
        ols = {'cluster_results': {'c1': {'net_beta': 0.5, 'p_value': 0.05}}}
        roles, flags = classify_model_b_roles(ols, 'Eco', {'c1': cascade})
        assert roles['c1'] == 'catalyst'
        assert flags['c1'] is False

    def test_model_b_inert(self):
        cascade = Mock()
        cascade.frame = 'Eco'
        ols = {'cluster_results': {'c1': {'net_beta': 0.5, 'p_value': 0.20}}}
        roles, flags = classify_model_b_roles(ols, 'Eco', {'c1': cascade})
        assert roles['c1'] == 'inert'
        assert flags['c1'] is True

    def test_model_b_missing_cascade(self):
        ols = {'cluster_results': {'c1': {'net_beta': 0.5, 'p_value': 0.05}}}
        roles, flags = classify_model_b_roles(ols, 'Eco', {})
        assert roles['c1'] == 'catalyst'
        assert flags['c1'] is False


# ── Test DataFrame export ────────────────────────────────────────────────────

class TestResultsToDataFrame:

    def test_dataframe_a_columns(self):
        results = [{
            'frame': 'Eco',
            'ar_order': 1,
            'r2_full': 0.5,
            'r2_test': 0.3,
            'r2_cv': 0.25,
            'cluster_results': {
                42: {
                    'net_beta': 0.1,
                    'p_value_hac': 0.05,
                    'p_value_boot': 0.04,
                    'role': 'catalyst',
                    'dominant_type': 'evt_weather',
                    'event_types': ['evt_weather'],
                    'strength': 0.5,
                    'D_sum': 1.0,
                    'peak_date': '2018-06-15',
                }
            }
        }]
        df = results_to_dataframe_a(results)
        assert not df.empty
        expected = {'frame', 'cluster_id', 'net_beta', 'p_value_hac',
                    'p_value_boot', 'role', 'dominant_type', 'event_types',
                    'strength', 'D_sum', 'peak_date', 'ar_order',
                    'r2_full', 'r2_test', 'r2_cv'}
        assert expected.issubset(set(df.columns))

    def test_dataframe_b_columns(self):
        results = [{
            'frame': 'Eco',
            'ar_order': 1,
            'r2_full': 0.5,
            'r2_test': 0.3,
            'r2_cv': 0.25,
            'cluster_results': {
                'Eco_20180601_1': {
                    'net_beta': 0.2,
                    'p_value_hac': 0.08,
                    'p_value_boot': 0.06,
                    'role': 'catalyst',
                    'is_own_frame': True,
                    'cascade_frame': 'Eco',
                }
            }
        }]
        df = results_to_dataframe_b(results)
        assert not df.empty
        expected = {'target_frame', 'cascade_id', 'cascade_frame', 'net_beta',
                    'p_value_hac', 'p_value_boot', 'role', 'is_own_frame',
                    'ar_order', 'r2_full', 'r2_test', 'r2_cv'}
        assert expected.issubset(set(df.columns))

    def test_dataframe_empty(self):
        df_a = results_to_dataframe_a([None])
        assert df_a.empty
        df_b = results_to_dataframe_b([None])
        assert df_b.empty

    def test_dataframe_a_multiple_results(self):
        results = [
            {'frame': 'Eco', 'ar_order': 1, 'r2_full': 0.5, 'r2_test': 0.3, 'r2_cv': 0.2,
             'cluster_results': {1: {'net_beta': 0.1, 'p_value_hac': 0.05, 'role': 'catalyst'}}},
            {'frame': 'Cult', 'ar_order': 2, 'r2_full': 0.6, 'r2_test': 0.4, 'r2_cv': 0.3,
             'cluster_results': {2: {'net_beta': -0.2, 'p_value_hac': 0.03, 'role': 'disruptor'}}},
        ]
        df = results_to_dataframe_a(results)
        assert len(df) == 2
        assert set(df['frame']) == {'Eco', 'Cult'}


# ── Test cascade lagged matrix ───────────────────────────────────────────────

class TestBuildCascadeLaggedMatrix:

    def test_shape(self, mock_cascade):
        dates = pd.date_range('2018-04-01', periods=200, freq='D')
        X, cids, lag_labels = build_cascade_lagged_matrix([mock_cascade], dates)
        expected_cols = 1 * (MAX_LAG_PARADIGM + 1)
        assert X.shape == (200, expected_cols)
        assert len(cids) == 1
        assert len(lag_labels) == expected_cols

    def test_empty_cascade(self):
        cascade = Mock()
        cascade.cascade_id = 'test'
        cascade.daily_composite = pd.Series(dtype=float)
        dates = pd.date_range('2018-04-01', periods=100, freq='D')
        X, cids, lag_labels = build_cascade_lagged_matrix([cascade], dates)
        assert X.shape[1] == 0
        assert cids == []

    def test_below_min_d_sum(self):
        cascade = Mock()
        cascade.cascade_id = 'test'
        dates = pd.date_range('2018-04-01', periods=100, freq='D')
        cascade.daily_composite = pd.Series(
            np.full(100, MIN_D_SUM / 200),  # total < MIN_D_SUM
            index=dates
        )
        X, cids, lag_labels = build_cascade_lagged_matrix([cascade], dates)
        assert X.shape[1] == 0

    def test_multiple_cascades(self, mock_cascade):
        c2 = Mock()
        c2.cascade_id = 'Cult_20180701_2'
        c2.frame = 'Cult'
        dates = pd.date_range('2018-04-01', periods=200, freq='D')
        c2.daily_composite = pd.Series(
            np.random.default_rng(99).random(200) * 0.3,
            index=dates
        )
        X, cids, lag_labels = build_cascade_lagged_matrix([mock_cascade, c2], dates)
        expected_cols = 2 * (MAX_LAG_PARADIGM + 1)
        assert X.shape == (200, expected_cols)


# ── Test alignment ───────────────────────────────────────────────────────────

class TestAlignment:

    def _make_paradigm_indexed(self):
        dates = pd.date_range('2018-04-01', periods=200, freq='D')
        data = {}
        for f in FRAMES:
            data[f'paradigm_{f}'] = np.random.default_rng(42).random(200)
        return pd.DataFrame(data, index=dates)

    def test_alignment_a_reinforcer(self):
        """Cluster with beta aligned with paradigm → reinforcer."""
        df_a = pd.DataFrame([{
            'frame': 'Eco',
            'cluster_id': 1,
            'net_beta': 5.0,  # large positive on Eco
            'p_value_hac': 0.01,
            'peak_date': '2018-06-01',
            'strength': 0.5,
            'dominant_type': 'evt_weather',
        }])
        paradigm = self._make_paradigm_indexed()
        # Make Eco dominant at that date
        paradigm.loc[:, 'paradigm_Eco'] = 0.8
        for f in FRAMES:
            if f != 'Eco':
                paradigm.loc[:, f'paradigm_{f}'] = 0.05

        result = StabSelParadigmAnalyzer._build_alignment_a(df_a, paradigm, [])
        assert not result.empty
        assert result.iloc[0]['global_role'] == 'reinforcer'

    def test_alignment_a_challenger(self):
        """Cluster with beta opposing paradigm → challenger."""
        df_a = pd.DataFrame([{
            'frame': 'Eco',
            'cluster_id': 1,
            'net_beta': -5.0,  # large negative on Eco
            'p_value_hac': 0.01,
            'peak_date': '2018-06-01',
            'strength': 0.5,
            'dominant_type': 'evt_weather',
        }])
        paradigm = self._make_paradigm_indexed()
        paradigm.loc[:, 'paradigm_Eco'] = 0.8
        for f in FRAMES:
            if f != 'Eco':
                paradigm.loc[:, f'paradigm_{f}'] = 0.05

        result = StabSelParadigmAnalyzer._build_alignment_a(df_a, paradigm, [])
        assert not result.empty
        assert result.iloc[0]['global_role'] == 'challenger'

    def test_alignment_empty_when_no_significant(self):
        df_a = pd.DataFrame([{
            'frame': 'Eco',
            'cluster_id': 1,
            'net_beta': 0.1,
            'p_value_hac': 0.50,  # not significant
            'peak_date': '2018-06-01',
            'strength': 0.5,
        }])
        paradigm = self._make_paradigm_indexed()
        result = StabSelParadigmAnalyzer._build_alignment_a(df_a, paradigm, [])
        assert result.empty

    def test_alignment_b_with_cascade(self):
        cascade = Mock()
        cascade.frame = 'Eco'
        cascade.peak_date = pd.Timestamp('2018-06-15')
        cascade.total_score = 0.7

        df_b = pd.DataFrame([{
            'target_frame': 'Eco',
            'cascade_id': 'Eco_20180601_1',
            'net_beta': 3.0,
            'p_value_hac': 0.02,
        }])
        paradigm = self._make_paradigm_indexed()
        paradigm.loc[:, 'paradigm_Eco'] = 0.8
        for f in FRAMES:
            if f != 'Eco':
                paradigm.loc[:, f'paradigm_{f}'] = 0.05

        result = StabSelParadigmAnalyzer._build_alignment_b(
            df_b, paradigm, {'Eco_20180601_1': cascade})
        assert not result.empty
        assert 'global_role' in result.columns
        assert 'impact_magnitude' in result.columns
        assert 'shift_contribution' in result.columns


# ── Test StabSelParadigmAnalyzer ─────────────────────────────────────────────

class TestStabSelParadigmAnalyzer:

    def _make_mock_results(self, paradigm_timeline, mock_articles,
                           cascades=None, clusters=None):
        """Create a minimal mock DetectionResults."""
        results = Mock()
        results.paradigm_shifts = Mock()
        results.paradigm_shifts.paradigm_timeline = paradigm_timeline
        results._articles = mock_articles
        results.cascades = cascades or []
        results.event_clusters = clusters or []
        return results

    def test_missing_paradigm_shifts_raises(self, embedding_store, mock_articles):
        results = Mock()
        results.paradigm_shifts = None
        results._articles = mock_articles

        analyzer = StabSelParadigmAnalyzer(embedding_store)
        with pytest.raises(ValueError, match="paradigm_shifts is None"):
            analyzer.run(results)

    def test_empty_timeline_raises(self, embedding_store, mock_articles):
        results = Mock()
        results.paradigm_shifts = Mock()
        results.paradigm_shifts.paradigm_timeline = pd.DataFrame()
        results._articles = mock_articles

        analyzer = StabSelParadigmAnalyzer(embedding_store)
        with pytest.raises(ValueError, match="paradigm_timeline is empty"):
            analyzer.run(results)

    def test_no_clusters_no_cascades(self, paradigm_timeline, mock_articles,
                                      embedding_store):
        """No clusters + no cascades = empty DataFrames, no crash."""
        results = self._make_mock_results(paradigm_timeline, mock_articles)
        analyzer = StabSelParadigmAnalyzer(embedding_store)
        out = analyzer.run(results)
        assert isinstance(out, StabSelParadigmResults)
        # Both models produce empty DFs if no data
        assert isinstance(out.cluster_dominance, pd.DataFrame)
        assert isinstance(out.cascade_dominance, pd.DataFrame)

    @patch('cascade_detector.analysis.stabsel_paradigm.run_model_a')
    @patch('cascade_detector.analysis.stabsel_paradigm.run_model_b')
    def test_full_run_with_mocks(self, mock_run_b, mock_run_a,
                                  paradigm_timeline, mock_articles,
                                  mock_cascade, mock_cluster, embedding_store):
        """Integration test: analyzer should produce valid StabSelParadigmResults."""
        # Mock run_model_a to return a result for first frame only
        def side_a(frame, *args, **kwargs):
            if frame == 'Eco':
                return {
                    'frame': 'Eco', 'model': 'A', 'ar_order': 1,
                    'n_entities': 5, 'n_columns': 40, 'n_stable': 3,
                    'r2_full': 0.65, 'r2_train': 0.70, 'r2_test': 0.45,
                    'r2_cv': 0.40, 'r2_cv_folds': [0.35, 0.40, 0.45],
                    'stable_cv_folds': [3, 4, 3],
                    'roles': {1: 'catalyst'},
                    'cluster_results': {
                        1: {'net_beta': 0.5, 'p_value_hac': 0.03,
                            'p_value_boot': 0.04, 'p_value': 0.03,
                            'role': 'catalyst', 'dominant_type': 'evt_weather',
                            'event_types': ['evt_weather'], 'strength': 0.5,
                            'D_sum': 1.0, 'peak_date': '2018-06-15',
                            'lag_profile': np.zeros(8), 'betas_by_lag': {0: 0.5}},
                    },
                    'cluster_meta': {},
                }
            return None

        def side_b(frame, *args, **kwargs):
            if frame == 'Eco':
                return {
                    'frame': 'Eco', 'model': 'B', 'ar_order': 1,
                    'n_entities': 3, 'n_columns': 24, 'n_stable': 1,
                    'r2_full': 0.60, 'r2_train': 0.65, 'r2_test': 0.40,
                    'r2_cv': 0.35, 'r2_cv_folds': [0.30, 0.35, 0.40],
                    'stable_cv_folds': [1, 2, 1],
                    'roles': {'Eco_20180601_1': 'catalyst'},
                    'cluster_results': {
                        'Eco_20180601_1': {
                            'net_beta': 0.3, 'p_value_hac': 0.05,
                            'p_value_boot': 0.04, 'p_value': 0.05,
                            'role': 'catalyst', 'is_own_frame': True, 'cascade_frame': 'Eco',
                            'cascade_id': 'Eco_20180601_1',
                            'lag_profile': np.zeros(8), 'betas_by_lag': {0: 0.3}},
                    },
                }
            return None

        mock_run_a.side_effect = side_a
        mock_run_b.side_effect = side_b

        results = self._make_mock_results(
            paradigm_timeline, mock_articles,
            cascades=[mock_cascade], clusters=[mock_cluster])

        analyzer = StabSelParadigmAnalyzer(embedding_store)
        out = analyzer.run(results)

        assert isinstance(out, StabSelParadigmResults)
        assert len(out.cluster_dominance) == 1
        assert len(out.cascade_dominance) == 1
        assert not out.validation.empty
        assert 'model_a' in out.summary
        assert 'model_b' in out.summary
        assert 'global' in out.summary
        assert 'model_a' in out.raw_results
        assert 'model_b' in out.raw_results


# ── Test run_model ───────────────────────────────────────────────────────────

class TestRunModel:

    @patch('cascade_detector.analysis.stabsel_paradigm.stability_selection')
    def test_run_model_empty_treatment(self, mock_stabsel, rng):
        y = rng.standard_normal(200)
        X = np.empty((200, 0))
        result = run_model('A', 'Eco', y, X, [])
        assert result is None

    @patch('cascade_detector.analysis.stabsel_paradigm.stability_selection')
    @patch('cascade_detector.analysis.stabsel_paradigm.train_test_evaluation')
    @patch('cascade_detector.analysis.stabsel_paradigm.temporal_cross_validation')
    def test_run_model_returns_result(self, mock_cv, mock_tt, mock_stabsel, rng):
        n = 200
        y = rng.standard_normal(n)
        X = rng.standard_normal((n, 8))
        lag_labels = [(0, i) for i in range(8)]

        # StabSel: protect first 2 controls + select some treatment
        full_mask = np.array([False, False, True, False, True, False, False, False, False, False])
        mock_stabsel.return_value = (full_mask, np.random.random(10))
        mock_tt.return_value = {'r2_train': 0.5, 'r2_test': 0.3, 'n_stable_train': 2}
        mock_cv.return_value = {'r2_per_fold': [0.2, 0.3], 'r2_cv': 0.25, 'stable_per_fold': [2, 3]}

        result = run_model('A', 'Eco', y, X, lag_labels, entity_ids=[0])
        assert result is not None
        assert result['frame'] == 'Eco'
        assert result['model'] == 'A'
        assert 'ar_order' in result
        assert 'r2_full' in result
        assert 'r2_cv' in result


# ── Test summary building ────────────────────────────────────────────────────

class TestBuildSummary:

    def test_summary_structure(self):
        results_a = [{'frame': 'Eco', 'ar_order': 1, 'n_entities': 5, 'n_stable': 3,
                       'r2_full': 0.5, 'r2_test': 0.3, 'r2_cv': 0.25}]
        results_b = [{'frame': 'Eco', 'ar_order': 1, 'n_entities': 3, 'n_stable': 1,
                       'r2_full': 0.6, 'r2_test': 0.4, 'r2_cv': 0.35}]
        df_a = pd.DataFrame([{
            'frame': 'Eco', 'cluster_id': 1, 'net_beta': 0.5,
            'p_value_hac': 0.05, 'role': 'catalyst',
        }])
        df_b = pd.DataFrame([{
            'target_frame': 'Eco', 'cascade_id': 'c1', 'net_beta': 0.3,
            'p_value_hac': 0.08, 'role': 'catalyst',
        }])
        df_align_a = pd.DataFrame([{'global_role': 'reinforcer'}])
        df_align_b = pd.DataFrame([{'global_role': 'reinforcer'}])

        summary = StabSelParadigmAnalyzer._build_summary(
            results_a, results_b, df_a, df_b, df_align_a, df_align_b)

        assert 'model_a' in summary
        assert 'model_b' in summary
        assert 'global' in summary
        assert summary['model_a']['Eco']['n_catalyst'] == 1
        assert summary['model_b']['Eco']['n_catalyst'] == 1
        assert summary['global']['n_reinforcers_a'] == 1
