"""
Unit tests for DailySignalBuilder.

14 tests covering signal API, alignment, composite weighting,
flat/burst behaviour, semantic signal, and rolling Z-score properties.

Uses synthetic data only -- no database, GPU, or embedding files required.
MockEmbeddingStore is injected via the conftest.py fixture.
"""

import numpy as np
import pandas as pd
import pytest

from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.constants import FRAME_COLUMNS, FRAMES
from cascade_detector.detection.signal_builder import DailySignalBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "z_temporal",
    "z_participation",
    "z_convergence",
    "z_source",
    "z_semantic",
    "composite",
}

# Signal weights from DetectorConfig defaults
W_TEMPORAL = 0.25
W_PARTICIPATION = 0.20
W_CONVERGENCE = 0.20
W_SOURCE = 0.15
W_SEMANTIC = 0.20


def _make_config() -> DetectorConfig:
    """Return a default DetectorConfig."""
    return DetectorConfig()


def _date_range(n_days: int = 200, start: str = "2020-01-01") -> pd.DatetimeIndex:
    """Return a DatetimeIndex of *n_days* consecutive days."""
    return pd.date_range(start, periods=n_days, freq="D")


def _make_articles(
    dates: pd.DatetimeIndex,
    frame: str = "Pol",
    n_per_day: int = 5,
    include_doc_id: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic article DataFrame covering *dates*.

    Every article is marked as belonging to *frame* via the appropriate
    FRAME_COLUMNS column.  messenger columns are included for the
    source-signal test.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for i, date in enumerate(dates):
        for j in range(n_per_day):
            row = {
                "date": date,
                "author": f"journalist_{rng.randint(0, 15)}",
                "media": rng.choice(["media_a", "media_b", "media_c"]),
            }
            if include_doc_id:
                row["doc_id"] = f"doc_{i:04d}_{j}"
            # Set frame column to 1 so articles pass the frame filter
            frame_col = FRAME_COLUMNS.get(frame)
            if frame_col:
                row[frame_col] = 1
            # Messenger columns (needed for source signal)
            row["msg_health"] = rng.randint(0, 3)
            row["msg_scientist"] = rng.randint(0, 2)
            row["msg_official"] = rng.randint(0, 2)
            rows.append(row)
    return pd.DataFrame(rows)


def _make_temporal_index(
    dates: pd.DatetimeIndex,
    frame: str = "Pol",
    base_proportion: float = 0.05,
    seed: int = 42,
) -> dict:
    """Build a minimal temporal_index dict for one frame.

    The returned structure mirrors the output of TemporalIndexer.build_index()
    for the keys consumed by DailySignalBuilder.
    """
    rng = np.random.RandomState(seed)
    proportions = pd.Series(
        base_proportion + rng.normal(0, 0.005, len(dates)).clip(-base_proportion, None),
        index=dates,
    )
    return {frame: {"daily_proportions": proportions}}


def _build_signals(
    frame: str = "Pol",
    n_days: int = 200,
    include_doc_id: bool = True,
    embedding_store=None,
    temporal_index_override: dict = None,
    articles_override: pd.DataFrame = None,
    seed: int = 42,
) -> dict:
    """End-to-end helper: build signals and return the dict."""
    config = _make_config()
    dates = _date_range(n_days)
    articles = articles_override if articles_override is not None else _make_articles(
        dates, frame=frame, include_doc_id=include_doc_id, seed=seed
    )
    temporal_index = temporal_index_override if temporal_index_override is not None else _make_temporal_index(
        dates, frame=frame, seed=seed
    )
    builder = DailySignalBuilder(config=config, embedding_store=embedding_store)
    return builder.build_signals(
        frame=frame,
        temporal_index=temporal_index,
        articles=articles,
        indices={},
    )


# ===========================================================================
# 1-3. Signal builder API
# ===========================================================================


class TestSignalBuilderAPI:
    """Tests for the public build_signals interface."""

    def test_returns_all_keys(self, mock_embedding_store):
        """build_signals must return all 6 expected keys."""
        signals = _build_signals(embedding_store=mock_embedding_store)
        assert set(signals.keys()) == EXPECTED_KEYS

    def test_handles_missing_frame(self, mock_embedding_store):
        """Frame absent from temporal_index returns empty dict."""
        dates = _date_range(200)
        temporal_index = _make_temporal_index(dates, frame="Pol")
        articles = _make_articles(dates, frame="Eco")
        config = _make_config()
        builder = DailySignalBuilder(config=config, embedding_store=mock_embedding_store)
        # Ask for 'Eco' but temporal_index only has 'Pol'
        result = builder.build_signals(
            frame="Eco",
            temporal_index=temporal_index,
            articles=articles,
            indices={},
        )
        assert result == {}

    def test_insufficient_data(self, mock_embedding_store):
        """Too few days (< baseline_window + 10) returns empty dict."""
        config = _make_config()
        # baseline_window_days defaults to 90; need < 100 days
        n_days = 50
        dates = _date_range(n_days)
        articles = _make_articles(dates, frame="Pol")
        temporal_index = _make_temporal_index(dates, frame="Pol")
        builder = DailySignalBuilder(config=config, embedding_store=mock_embedding_store)
        result = builder.build_signals(
            frame="Pol",
            temporal_index=temporal_index,
            articles=articles,
            indices={},
        )
        assert result == {}


# ===========================================================================
# 4-5. Signal alignment
# ===========================================================================


class TestSignalAlignment:
    """Tests that all returned signals share the same index and are non-negative."""

    def test_same_datetime_index(self, mock_embedding_store):
        """All signals must share an identical DatetimeIndex."""
        signals = _build_signals(embedding_store=mock_embedding_store)
        ref_index = signals["z_temporal"].index
        for key in EXPECTED_KEYS:
            pd.testing.assert_index_equal(
                signals[key].index, ref_index, exact=True,
                obj=f"signals['{key}'].index",
            )

    def test_all_non_negative(self, mock_embedding_store):
        """All Z-score signals and the composite must be >= 0 everywhere."""
        signals = _build_signals(embedding_store=mock_embedding_store)
        for key in EXPECTED_KEYS:
            series = signals[key]
            assert (series >= -1e-12).all(), (
                f"signals['{key}'] has negative values: min={series.min()}"
            )


# ===========================================================================
# 6. Composite weighting
# ===========================================================================


class TestCompositeWeighting:
    """Test that composite == weighted sum of individual Z-scores."""

    def test_weighted_sum_matches_manual(self, mock_embedding_store):
        """Composite must equal the manually-computed weighted sum."""
        signals = _build_signals(embedding_store=mock_embedding_store)
        manual = (
            W_TEMPORAL * signals["z_temporal"]
            + W_PARTICIPATION * signals["z_participation"]
            + W_CONVERGENCE * signals["z_convergence"]
            + W_SOURCE * signals["z_source"]
            + W_SEMANTIC * signals["z_semantic"]
        )
        pd.testing.assert_series_equal(
            signals["composite"], manual, check_names=False, atol=1e-10,
        )


# ===========================================================================
# 7-9. Flat / burst behaviour
# ===========================================================================


class TestFlatAndBurstBehaviour:
    """Test that flat series produce low composite and bursts raise it."""

    def test_flat_series_low_composite(self, mock_embedding_store):
        """Perfectly flat input proportions yield near-zero composite."""
        n_days = 200
        dates = _date_range(n_days)
        # Constant proportion -- no anomaly
        flat_proportions = pd.Series(0.05, index=dates)
        temporal_index = {"Pol": {"daily_proportions": flat_proportions}}
        articles = _make_articles(dates, frame="Pol", seed=99)
        config = _make_config()
        builder = DailySignalBuilder(config=config, embedding_store=mock_embedding_store)
        signals = builder.build_signals(
            frame="Pol",
            temporal_index=temporal_index,
            articles=articles,
            indices={},
        )
        # Temporal Z-score on a constant series should be ~0 after the warm-up
        assert signals["z_temporal"].mean() < 0.5, (
            f"Flat temporal signal mean too high: {signals['z_temporal'].mean()}"
        )

    def test_burst_in_one_dimension_raises_composite(self, mock_embedding_store):
        """A spike injected into the temporal proportion should raise composite on those days."""
        n_days = 200
        dates = _date_range(n_days)
        rng = np.random.RandomState(7)

        proportions = pd.Series(
            0.05 + rng.normal(0, 0.003, n_days).clip(-0.04, None),
            index=dates,
        )
        # Inject a strong burst at days 150-160
        proportions.iloc[150:160] = 0.30

        temporal_index = {"Pol": {"daily_proportions": proportions}}
        articles = _make_articles(dates, frame="Pol", seed=7)
        config = _make_config()
        builder = DailySignalBuilder(config=config, embedding_store=mock_embedding_store)
        signals = builder.build_signals(
            frame="Pol",
            temporal_index=temporal_index,
            articles=articles,
            indices={},
        )
        burst_window = signals["composite"].iloc[150:160]
        calm_window = signals["composite"].iloc[100:110]
        assert burst_window.mean() > calm_window.mean(), (
            f"Burst composite ({burst_window.mean():.4f}) not > calm ({calm_window.mean():.4f})"
        )

    def test_multi_signal_burst_higher_than_single(self, mock_embedding_store):
        """A burst in temporal + participation should produce higher composite
        than a burst in temporal alone."""
        n_days = 200
        dates = _date_range(n_days)
        rng = np.random.RandomState(11)

        # --- Single-signal burst (temporal only) ---
        prop_single = pd.Series(
            0.05 + rng.normal(0, 0.003, n_days).clip(-0.04, None),
            index=dates,
        )
        prop_single.iloc[150:160] = 0.30

        articles_single = _make_articles(dates, frame="Pol", seed=11)
        ti_single = {"Pol": {"daily_proportions": prop_single}}

        config = _make_config()
        builder = DailySignalBuilder(config=config, embedding_store=mock_embedding_store)
        sig_single = builder.build_signals(
            frame="Pol", temporal_index=ti_single,
            articles=articles_single, indices={},
        )

        # --- Multi-signal burst (temporal + many new journalists) ---
        prop_multi = prop_single.copy()
        # Add many unique journalists during burst window
        articles_multi = _make_articles(dates, frame="Pol", n_per_day=5, seed=11)
        burst_rows = articles_multi["date"].isin(dates[150:160])
        # Give every article in the burst window a unique journalist
        articles_multi.loc[burst_rows, "author"] = [
            f"unique_journalist_{k}" for k in range(burst_rows.sum())
        ]
        ti_multi = {"Pol": {"daily_proportions": prop_multi}}
        sig_multi = builder.build_signals(
            frame="Pol", temporal_index=ti_multi,
            articles=articles_multi, indices={},
        )

        single_burst_mean = sig_single["composite"].iloc[150:160].mean()
        multi_burst_mean = sig_multi["composite"].iloc[150:160].mean()
        assert multi_burst_mean > single_burst_mean, (
            f"Multi-signal burst ({multi_burst_mean:.4f}) "
            f"not > single-signal ({single_burst_mean:.4f})"
        )


# ===========================================================================
# 10-11. Semantic signal
# ===========================================================================


class TestSemanticSignal:
    """Tests for the semantic (embedding-based) Z-score signal."""

    def test_semantic_present_and_non_negative(self, mock_embedding_store):
        """z_semantic must exist and be >= 0 when embedding_store is provided."""
        signals = _build_signals(embedding_store=mock_embedding_store)
        assert "z_semantic" in signals
        assert (signals["z_semantic"] >= -1e-12).all()

    def test_semantic_zero_when_doc_ids_absent(self, mock_embedding_store):
        """z_semantic must be identically 0 when articles lack doc_id."""
        signals = _build_signals(
            embedding_store=mock_embedding_store,
            include_doc_id=False,
        )
        assert "z_semantic" in signals
        assert (signals["z_semantic"] == 0.0).all(), (
            "z_semantic should be zero when doc_id column is absent"
        )


# ===========================================================================
# 12-14. Rolling Z-score properties
# ===========================================================================


class TestRollingZScore:
    """Tests for _rolling_zscore clipping and spike detection."""

    def test_clipped_at_zero(self, mock_embedding_store):
        """_rolling_zscore output must never be negative."""
        config = _make_config()
        builder = DailySignalBuilder(config=config, embedding_store=mock_embedding_store)

        dates = _date_range(200)
        rng = np.random.RandomState(55)
        series = pd.Series(rng.normal(0.05, 0.02, 200).clip(0, None), index=dates)

        z = builder._rolling_zscore(series)
        assert (z >= -1e-12).all(), f"Z-score has negative values: min={z.min()}"

    def test_spike_produces_high_z(self, mock_embedding_store):
        """A single large spike should produce a high rolling Z-score."""
        config = _make_config()
        builder = DailySignalBuilder(config=config, embedding_store=mock_embedding_store)

        dates = _date_range(200)
        series = pd.Series(0.05, index=dates)  # flat baseline
        series.iloc[150] = 1.0  # massive spike

        z = builder._rolling_zscore(series)
        assert z.iloc[150] > 2.0, (
            f"Spike Z-score too low: {z.iloc[150]:.4f}"
        )

    def test_z_score_monotonicity_with_spike_size(self, mock_embedding_store):
        """Larger spikes should produce larger Z-scores (all else equal)."""
        config = _make_config()
        builder = DailySignalBuilder(config=config, embedding_store=mock_embedding_store)

        dates = _date_range(200)
        z_values = []
        for spike_size in [0.10, 0.30, 0.60, 1.00]:
            series = pd.Series(0.05, index=dates)
            series.iloc[150] = spike_size
            z = builder._rolling_zscore(series)
            z_values.append(z.iloc[150])

        for i in range(len(z_values) - 1):
            assert z_values[i] <= z_values[i + 1] + 1e-10, (
                f"Z-score not monotonic: spike sizes produced Z={z_values}"
            )


# ===========================================================================
# 15-18. Orthogonalization
# ===========================================================================


class TestOrthogonalization:
    """Tests for z_convergence orthogonalization w.r.t. z_temporal."""

    def test_orthogonalized_convergence_non_negative(self, mock_embedding_store):
        """z_convergence after orthogonalization must remain >= 0."""
        signals = _build_signals(embedding_store=mock_embedding_store)
        assert (signals["z_convergence"] >= -1e-12).all(), (
            f"Orthogonalized z_convergence has negative values: "
            f"min={signals['z_convergence'].min()}"
        )

    def test_orthogonalized_reduces_correlation(self, mock_embedding_store):
        """After orthogonalization, correlation between z_convergence and
        z_temporal should be lower than the raw (pre-orthogonalization)
        correlation.

        Uses a multi-frame temporal index so that convergence (dominance
        ratio = target / sum_all_frames) is meaningful, and injects
        repeated bursts to create correlated z-score signals.
        """
        n_days = 300
        dates = _date_range(n_days)
        rng = np.random.RandomState(7)

        # Build multi-frame temporal index: Pol bursts, others stay flat
        pol_props = pd.Series(
            0.08 + rng.normal(0, 0.005, n_days).clip(-0.07, None),
            index=dates,
        )
        # Multiple burst windows to get enough nonzero overlap
        for start in [120, 150, 180, 210, 240]:
            pol_props.iloc[start:start + 8] = 0.30 + rng.uniform(0, 0.05, 8)

        # Other frames: stable background
        temporal_index = {"Pol": {"daily_proportions": pol_props}}
        for f in FRAMES:
            if f == "Pol":
                continue
            other_props = pd.Series(
                0.06 + rng.normal(0, 0.003, n_days).clip(-0.05, None),
                index=dates,
            )
            temporal_index[f] = {"daily_proportions": other_props}

        articles = _make_articles(dates, frame="Pol", seed=7)
        config = _make_config()
        builder = DailySignalBuilder(config=config, embedding_store=mock_embedding_store)

        signals = builder.build_signals(
            frame="Pol", temporal_index=temporal_index,
            articles=articles, indices={},
        )

        # Compute raw (non-orthogonalized) convergence for comparison
        z_conv_raw = builder._compute_convergence_z("Pol", temporal_index, dates)
        z_temporal = signals["z_temporal"]
        z_conv_orth = signals["z_convergence"]

        # Use only days where both signals are nonzero for meaningful correlation
        mask = (z_temporal > 0) & (z_conv_raw > 0)
        assert mask.sum() >= 10, (
            f"Not enough nonzero overlap ({mask.sum()}) for correlation test"
        )

        raw_corr = np.corrcoef(z_temporal[mask], z_conv_raw[mask])[0, 1]
        orth_corr = np.corrcoef(z_temporal[mask], z_conv_orth[mask])[0, 1]

        assert abs(orth_corr) <= abs(raw_corr) + 1e-6, (
            f"Orthogonalized correlation ({orth_corr:.3f}) should be <= "
            f"raw correlation ({raw_corr:.3f})"
        )

    def test_orthogonalize_zero_reference(self, mock_embedding_store):
        """When z_temporal is all zeros, z_convergence should be unchanged."""
        config = _make_config()
        builder = DailySignalBuilder(config=config, embedding_store=mock_embedding_store)

        dates = _date_range(100)
        reference = pd.Series(0.0, index=dates)
        target = pd.Series(np.random.RandomState(1).rand(100), index=dates)

        result = builder._orthogonalize(target, reference)
        pd.testing.assert_series_equal(result, target, check_names=False)

    def test_orthogonalize_identical_signals(self, mock_embedding_store):
        """When target == reference (perfect correlation), residual should be ~0."""
        config = _make_config()
        builder = DailySignalBuilder(config=config, embedding_store=mock_embedding_store)

        dates = _date_range(100)
        signal = pd.Series(np.random.RandomState(2).rand(100).clip(0, None), index=dates)

        result = builder._orthogonalize(signal, signal)
        # After subtracting projection and clipping, should be near zero
        assert result.max() < 1e-6, (
            f"Residual of identical signals should be ~0, got max={result.max()}"
        )
