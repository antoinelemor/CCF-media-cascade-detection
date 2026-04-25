"""
Unit tests for EventImpactAnalyzer.

Uses synthetic data only — no database or embedding store required.
"""

import numpy as np
import pandas as pd
import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional

from cascade_detector.analysis.impact_analysis import EventImpactAnalyzer


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeCascade:
    """Minimal cascade result for testing."""
    cascade_id: str = 'test_001'
    frame: str = 'Eco'
    onset_date: pd.Timestamp = pd.Timestamp('2020-03-01')
    peak_date: pd.Timestamp = pd.Timestamp('2020-03-10')
    end_date: pd.Timestamp = pd.Timestamp('2020-03-20')
    total_score: float = 0.65


@dataclass
class FakeResults:
    """Minimal detection results for testing."""
    cascades: list = field(default_factory=list)


def _make_articles(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic article-level DataFrame.

    Creates articles spanning 2020 with known event/messenger rates.
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range('2020-01-01', '2020-12-31', periods=n)

    df = pd.DataFrame({
        'doc_id': [f'doc_{i}' for i in range(n)],
        'date_converted_first': dates,
        'media': rng.choice(['media_a', 'media_b', 'media_c'], n),
        'author': [f'author_{i % 20}' for i in range(n)],
        'year': 2020,
        # Frame columns (aggregated style with _sum suffix)
        'economic_frame_sum': rng.binomial(5, 0.3, n),
        'health_frame_sum': rng.binomial(5, 0.1, n),
        # Event columns
        'evt_weather_sum': rng.binomial(3, 0.2, n),
        'evt_meeting_sum': rng.binomial(3, 0.15, n),
        'evt_publication_sum': rng.binomial(3, 0.05, n),
        # Messenger columns
        'msg_scientist_sum': rng.binomial(3, 0.1, n),
        'msg_official_sum': rng.binomial(3, 0.08, n),
        # Solution columns
        'sol_mitigation_sum': rng.binomial(3, 0.1, n),
        'sol_adaptation_sum': rng.binomial(3, 0.05, n),
    })
    return df


def _make_articles_with_cascade_signal(seed: int = 42) -> pd.DataFrame:
    """Articles where evt_weather is strongly elevated during cascade window."""
    rng = np.random.RandomState(seed)
    n = 300

    dates = pd.date_range('2020-01-01', '2020-12-31', periods=n)
    df = pd.DataFrame({
        'doc_id': [f'doc_{i}' for i in range(n)],
        'date_converted_first': dates,
        'media': rng.choice(['media_a', 'media_b'], n),
        'author': [f'author_{i % 10}' for i in range(n)],
        'year': 2020,
        'economic_frame_sum': 1,  # all articles have the frame
    })

    # evt_weather: baseline rate ~10%, cascade rate ~80%
    cascade_start = pd.Timestamp('2020-03-01')
    cascade_end = pd.Timestamp('2020-03-20')

    in_cascade = (df['date_converted_first'] >= cascade_start) & \
                 (df['date_converted_first'] <= cascade_end)

    evt_values = np.zeros(n, dtype=int)
    for i in range(n):
        if in_cascade.iloc[i]:
            evt_values[i] = rng.binomial(1, 0.8)
        else:
            evt_values[i] = rng.binomial(1, 0.1)

    df['evt_weather_sum'] = evt_values
    df['evt_meeting_sum'] = rng.binomial(1, 0.1, n)  # no signal
    df['msg_scientist_sum'] = rng.binomial(1, 0.1, n)
    df['sol_mitigation_sum'] = rng.binomial(1, 0.05, n)

    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPrevalenceRatio:
    """Test prevalence ratio computation."""

    def test_known_counts(self):
        """Prevalence ratio with known elevated rates during cascade."""
        articles = _make_articles_with_cascade_signal()
        cascade = FakeCascade()
        results = FakeResults(cascades=[cascade])

        analyzer = EventImpactAnalyzer(results, articles)
        df = analyzer.compute_prevalence_ratios()

        # evt_weather should have prevalence_ratio > 1 (strongly elevated)
        weather_row = df[df['annotation'] == 'evt_weather']
        assert len(weather_row) == 1
        pr = weather_row['prevalence_ratio'].iloc[0]
        assert pr > 2.0, f"Expected PR > 2.0 for evt_weather, got {pr}"

    def test_fisher_p_value(self):
        """Fisher's exact test should yield significant p-value for elevated event."""
        articles = _make_articles_with_cascade_signal()
        cascade = FakeCascade()
        results = FakeResults(cascades=[cascade])

        analyzer = EventImpactAnalyzer(results, articles)
        df = analyzer.compute_prevalence_ratios()

        weather_row = df[df['annotation'] == 'evt_weather']
        p_val = weather_row['p_value'].iloc[0]
        assert p_val < 0.05, f"Expected p < 0.05 for evt_weather, got {p_val}"


class TestPreOnsetSurge:
    """Test pre-onset surge computation."""

    def test_surge_with_known_pattern(self):
        """Pre-onset surge detects elevated pre-cascade activity."""
        # Create articles with elevated evt_weather in the 7 days before cascade
        rng = np.random.RandomState(123)
        n = 400
        dates = pd.date_range('2020-01-01', '2020-12-31', periods=n)

        df = pd.DataFrame({
            'doc_id': [f'doc_{i}' for i in range(n)],
            'date_converted_first': dates,
            'media': 'media_a',
            'author': 'author_0',
            'year': 2020,
            'economic_frame_sum': 1,
        })

        # Create 3 cascades with elevated pre-onset evt_weather
        cascades = [
            FakeCascade(cascade_id='c1', onset_date=pd.Timestamp('2020-03-15'),
                        peak_date=pd.Timestamp('2020-03-20'),
                        end_date=pd.Timestamp('2020-03-30'), total_score=0.7),
            FakeCascade(cascade_id='c2', onset_date=pd.Timestamp('2020-06-15'),
                        peak_date=pd.Timestamp('2020-06-20'),
                        end_date=pd.Timestamp('2020-06-30'), total_score=0.5),
            FakeCascade(cascade_id='c3', onset_date=pd.Timestamp('2020-09-15'),
                        peak_date=pd.Timestamp('2020-09-20'),
                        end_date=pd.Timestamp('2020-09-30'), total_score=0.6),
        ]

        # evt_weather: baseline ~5%, pre-onset window ~50%
        evt_values = np.zeros(n, dtype=int)
        for i in range(n):
            d = df['date_converted_first'].iloc[i]
            is_pre_onset = False
            for c in cascades:
                pre_start = c.onset_date - pd.Timedelta(days=7)
                pre_end = c.onset_date - pd.Timedelta(days=1)
                if pre_start <= d <= pre_end:
                    is_pre_onset = True
                    break
            if is_pre_onset:
                evt_values[i] = rng.binomial(1, 0.6)
            else:
                evt_values[i] = rng.binomial(1, 0.05)

        df['evt_weather_sum'] = evt_values
        df['msg_scientist_sum'] = rng.binomial(1, 0.05, n)
        df['sol_mitigation_sum'] = rng.binomial(1, 0.03, n)

        results = FakeResults(cascades=cascades)
        analyzer = EventImpactAnalyzer(results, df)
        surge_df = analyzer.compute_pre_onset_surge()

        weather_rows = surge_df[surge_df['annotation'] == 'evt_weather']
        assert len(weather_rows) == 1
        median_surge = weather_rows['median_surge'].iloc[0]
        assert median_surge > 1.0, f"Expected surge > 1.0, got {median_surge}"


class TestStrengthCorrelation:
    """Test strength correlation computation."""

    def test_correlation_direction(self):
        """Higher event prevalence should correlate with higher cascade scores."""
        rng = np.random.RandomState(99)

        # Create 5 cascades with increasing evt_weather prevalence and scores
        cascades = []
        all_articles = []

        for i, (score, evt_rate) in enumerate([
            (0.3, 0.1), (0.4, 0.2), (0.5, 0.4), (0.6, 0.6), (0.7, 0.8)
        ]):
            onset = pd.Timestamp(f'2020-{i+2:02d}-01')
            end = onset + pd.Timedelta(days=14)
            cascades.append(FakeCascade(
                cascade_id=f'c{i}', onset_date=onset, peak_date=onset + pd.Timedelta(days=7),
                end_date=end, total_score=score,
            ))
            # Create 50 articles per cascade window
            for j in range(50):
                all_articles.append({
                    'doc_id': f'doc_{i}_{j}',
                    'date_converted_first': onset + pd.Timedelta(days=rng.randint(0, 14)),
                    'media': 'media_a',
                    'author': f'author_{j % 5}',
                    'year': 2020,
                    'economic_frame_sum': 1,
                    'evt_weather_sum': rng.binomial(1, evt_rate),
                    'msg_scientist_sum': rng.binomial(1, 0.1),
                    'sol_mitigation_sum': rng.binomial(1, 0.05),
                })

        df = pd.DataFrame(all_articles)
        results = FakeResults(cascades=cascades)
        analyzer = EventImpactAnalyzer(results, df)
        corr_df = analyzer.compute_strength_correlations()

        weather_rows = corr_df[corr_df['annotation'] == 'evt_weather']
        assert len(weather_rows) == 1
        rho = weather_rows['spearman_rho'].iloc[0]
        assert rho > 0, f"Expected positive rho for evt_weather, got {rho}"


class TestBenjaminiHochberg:
    """Test BH FDR correction."""

    def test_adjusted_ge_raw(self):
        """Adjusted p-values should always be >= raw p-values."""
        raw = np.array([0.001, 0.01, 0.03, 0.05, 0.10, 0.50])
        adjusted = EventImpactAnalyzer._benjamini_hochberg(raw)

        assert len(adjusted) == len(raw)
        for r, a in zip(raw, adjusted):
            assert a >= r, f"Adjusted {a} < raw {r}"
            assert a <= 1.0, f"Adjusted {a} > 1.0"

    def test_monotonicity(self):
        """Adjusted p-values should be monotonically non-decreasing when sorted by raw."""
        raw = np.array([0.005, 0.01, 0.02, 0.04, 0.08])
        adjusted = EventImpactAnalyzer._benjamini_hochberg(raw)

        sorted_idx = np.argsort(raw)
        adj_sorted = adjusted[sorted_idx]
        for i in range(len(adj_sorted) - 1):
            assert adj_sorted[i] <= adj_sorted[i+1] + 1e-10


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_cascades(self):
        """Empty cascades should return empty DataFrames gracefully."""
        articles = _make_articles()
        results = FakeResults(cascades=[])

        analyzer = EventImpactAnalyzer(results, articles)
        output = analyzer.run()

        assert 'prevalence_ratios' in output
        assert 'pre_onset_surge' in output
        assert 'strength_correlations' in output
        assert 'summary' in output

        assert isinstance(output['prevalence_ratios'], pd.DataFrame)
        assert len(output['prevalence_ratios']) == 0

    def test_zero_occurrence_event(self):
        """Event that never occurs should be handled without error."""
        articles = _make_articles()
        # Zero out a specific event
        articles['evt_publication_sum'] = 0

        cascade = FakeCascade()
        results = FakeResults(cascades=[cascade])

        analyzer = EventImpactAnalyzer(results, articles)
        df = analyzer.compute_prevalence_ratios()

        # Should still produce a row for evt_publication (with rate=0)
        pub_row = df[df['annotation'] == 'evt_publication']
        if len(pub_row) > 0:
            assert pub_row['cascade_rate'].iloc[0] == 0.0
            assert pub_row['baseline_rate'].iloc[0] == 0.0


class TestRun:
    """Test the full run() method."""

    def test_returns_all_dataframes(self):
        """run() should return dict with all expected keys and correct column types."""
        articles = _make_articles_with_cascade_signal()
        cascade = FakeCascade()
        results = FakeResults(cascades=[cascade])

        analyzer = EventImpactAnalyzer(results, articles)
        output = analyzer.run()

        assert set(output.keys()) == {
            'prevalence_ratios', 'pre_onset_surge',
            'strength_correlations', 'summary'
        }

        # Check prevalence_ratios columns
        pr_df = output['prevalence_ratios']
        expected_pr_cols = {
            'frame', 'annotation', 'type', 'cascade_rate', 'baseline_rate',
            'prevalence_ratio', 'odds_ratio', 'ci_low', 'ci_high',
            'p_value', 'p_value_adjusted', 'n_cascade', 'n_baseline'
        }
        assert expected_pr_cols.issubset(set(pr_df.columns)), \
            f"Missing columns: {expected_pr_cols - set(pr_df.columns)}"

        # Check pre_onset_surge columns
        surge_df = output['pre_onset_surge']
        expected_surge_cols = {
            'frame', 'annotation', 'type', 'n_cascades',
            'median_surge', 'mean_surge', 'p_value', 'p_value_adjusted'
        }
        assert expected_surge_cols.issubset(set(surge_df.columns)), \
            f"Missing columns: {expected_surge_cols - set(surge_df.columns)}"

        # Check summary has n_significant_metrics
        summary = output['summary']
        assert 'n_significant_metrics' in summary.columns
