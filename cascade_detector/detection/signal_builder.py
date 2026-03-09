"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
signal_builder.py

MAIN OBJECTIVE:
---------------
Compute 5 daily anomaly signals for a given frame, then combine them into
a single composite signal. The composite replaces the temporal-only Z-score
used in the legacy BurstDetector with a multi-signal approach that captures
temporal, participation, convergence, source, and semantic anomalies
simultaneously.

Signals:
1. Temporal anomaly   — daily frame proportion vs rolling baseline
2. Participation      — unique journalists publishing with this frame
3. Convergence        — frame dominance (target / all frames)
4. Source             — messenger concentration (1 - norm. entropy)
5. Semantic anomaly   — mean pairwise embedding similarity (content homogenization)

All signals are converted to one-sided Z-scores (clipped at 0) so that
only positive anomalies are considered.

Composite = weighted sum of the 5 Z-scores.

Author:
-------
Antoine Lemor
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.constants import FRAME_COLUMNS, FRAMES, MESSENGERS

logger = logging.getLogger(__name__)


class DailySignalBuilder:
    """Builds daily multi-signal time series for cascade detection.

    For each frame and each day in the analysis period, computes 5 anomaly
    Z-scores and combines them into a single composite signal.
    """

    def __init__(self, config: Optional[DetectorConfig] = None, embedding_store=None):
        self.config = config or DetectorConfig()
        self.embedding_store = embedding_store

    def build_signals(
        self,
        frame: str,
        temporal_index: Dict,
        articles: pd.DataFrame,
        indices: Dict,
    ) -> Dict[str, pd.Series]:
        """Build all daily signals for one frame.

        Args:
            frame: Frame short name (e.g. 'Pol').
            temporal_index: Output of TemporalIndexer.build_index().
            articles: Article-level DataFrame.
            indices: All built indices.

        Returns:
            Dict with keys: 'z_temporal', 'z_participation', 'z_convergence',
            'z_source', 'z_semantic', 'composite'. All pd.Series with same
            DatetimeIndex.
        """
        if frame not in temporal_index:
            logger.warning(f"Frame '{frame}' not in temporal index")
            return {}

        frame_data = temporal_index[frame]
        series = frame_data.get('daily_proportions')
        if series is None or len(series) < self.config.baseline_window_days + 10:
            logger.debug(f"Insufficient data for frame '{frame}'")
            return {}

        series = series.sort_index().astype(float)
        date_range = series.index

        # Resolve column names once
        date_col = self._resolve_col(articles, ['date', 'date_converted_first', 'date_converted'])
        author_col = self._resolve_col(articles, ['author', 'author_first', 'author_clean_first', 'author_clean'])

        # Compute 5 Z-score signals
        z_temporal = self._compute_temporal_z(series)
        z_participation = self._compute_participation_z(
            frame, articles, date_col, author_col, date_range
        )
        z_convergence_raw = self._compute_convergence_z(frame, temporal_index, date_range)
        z_source = self._compute_source_z(frame, articles, date_col, date_range)
        z_semantic = self._compute_semantic_z(frame, articles, date_col, date_range)

        # Orthogonalize z_convergence w.r.t. z_temporal.
        # z_temporal and z_convergence are ~0.92 correlated because both are
        # functions of daily frame proportion.  Subtracting the projection
        # removes the redundant component so convergence only captures the
        # *additional* information of frame dominance (target / all frames)
        # that is not already explained by the temporal signal.
        z_convergence = self._orthogonalize(z_convergence_raw, z_temporal)

        # Build composite
        composite = self._build_composite({
            'z_temporal': z_temporal,
            'z_participation': z_participation,
            'z_convergence': z_convergence,
            'z_source': z_source,
            'z_semantic': z_semantic,
        })

        return {
            'z_temporal': z_temporal,
            'z_participation': z_participation,
            'z_convergence': z_convergence,
            'z_source': z_source,
            'z_semantic': z_semantic,
            'composite': composite,
        }

    # =========================================================================
    # Signal 1: Temporal anomaly
    # =========================================================================

    def _compute_temporal_z(self, series: pd.Series) -> pd.Series:
        """Z-score of daily frame proportion vs rolling baseline.

        Same logic as the legacy BurstDetector, but returns continuous
        Z-score instead of boolean flags.
        """
        return self._rolling_zscore(series)

    # =========================================================================
    # Signal 2: Participation anomaly
    # =========================================================================

    def _compute_participation_z(
        self,
        frame: str,
        articles: pd.DataFrame,
        date_col: Optional[str],
        author_col: Optional[str],
        date_range: pd.DatetimeIndex,
    ) -> pd.Series:
        """Z-score of unique journalist count per day for this frame.

        Days with 0 articles for this frame → n_journalists = 0 → Z clipped to 0.
        """
        raw = pd.Series(0.0, index=date_range)

        if date_col is None or author_col is None or articles.empty:
            return raw

        # Filter articles to this frame
        frame_col = FRAME_COLUMNS.get(frame)
        frame_articles = self._filter_frame_articles(articles, frame_col)
        if frame_articles.empty:
            return raw

        dates = pd.to_datetime(frame_articles[date_col], errors='coerce').dt.normalize()
        frame_articles = frame_articles.copy()
        frame_articles['_date'] = dates

        # Count unique journalists per day
        daily_counts = (
            frame_articles.dropna(subset=['_date', author_col])
            .groupby('_date')[author_col]
            .nunique()
        )
        raw = raw.add(daily_counts, fill_value=0).fillna(0).reindex(date_range, fill_value=0)

        return self._rolling_zscore(raw)

    # =========================================================================
    # Signal 3: Convergence anomaly
    # =========================================================================

    def _compute_convergence_z(
        self,
        frame: str,
        temporal_index: Dict,
        date_range: pd.DatetimeIndex,
    ) -> pd.Series:
        """Z-score of frame dominance (target frame / all frames).

        Measures how much this frame monopolizes coverage relative to
        all other frames on each day.
        """
        # Sum all frames' daily proportions
        all_sums = pd.Series(0.0, index=date_range)
        target_series = pd.Series(0.0, index=date_range)

        for f in FRAMES:
            if f not in temporal_index:
                continue
            f_data = temporal_index[f]
            f_series = f_data.get('daily_proportions')
            if f_series is None:
                continue
            f_series = f_series.reindex(date_range, fill_value=0).astype(float)
            all_sums = all_sums + f_series
            if f == frame:
                target_series = f_series

        # Dominance ratio
        dominance = pd.Series(0.0, index=date_range)
        nonzero = all_sums > 0
        dominance[nonzero] = target_series[nonzero] / all_sums[nonzero]

        return self._rolling_zscore(dominance)

    # =========================================================================
    # Signal 4: Source anomaly
    # =========================================================================

    def _compute_source_z(
        self,
        frame: str,
        articles: pd.DataFrame,
        date_col: Optional[str],
        date_range: pd.DatetimeIndex,
    ) -> pd.Series:
        """Z-score of messenger concentration per day.

        Concentration = 1 - normalized Shannon entropy of messenger distribution.
        High concentration = few messenger types dominate = sources converging.
        """
        raw = pd.Series(0.0, index=date_range)

        if date_col is None or articles.empty:
            return raw

        frame_col = FRAME_COLUMNS.get(frame)
        frame_articles = self._filter_frame_articles(articles, frame_col)
        if frame_articles.empty:
            return raw

        dates = pd.to_datetime(frame_articles[date_col], errors='coerce').dt.normalize()
        frame_articles = frame_articles.copy()
        frame_articles['_date'] = dates

        # Find available messenger columns
        available_msg_cols = []
        for msg_col in MESSENGERS:
            for col in [msg_col, f"{msg_col}_sum", f"{msg_col}_mean"]:
                if col in frame_articles.columns:
                    available_msg_cols.append(col)
                    break

        if not available_msg_cols:
            return raw

        # Compute concentration per day
        for date in date_range:
            day_articles = frame_articles[frame_articles['_date'] == date]
            if day_articles.empty:
                continue

            msg_counts = np.array(
                [day_articles[col].sum() for col in available_msg_cols],
                dtype=float,
            )
            total = msg_counts.sum()
            if total == 0:
                continue

            probs = msg_counts / total
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            max_entropy = np.log(len(available_msg_cols))
            if max_entropy > 0:
                raw[date] = 1.0 - (entropy / max_entropy)

        return self._rolling_zscore(raw)

    # =========================================================================
    # Signal 5: Semantic anomaly
    # =========================================================================

    def _compute_semantic_z(
        self,
        frame: str,
        articles: pd.DataFrame,
        date_col: Optional[str],
        date_range: pd.DatetimeIndex,
    ) -> pd.Series:
        """Z-score of daily mean pairwise embedding similarity for this frame.

        For each day: get doc_ids of articles for this frame, compute mean
        pairwise cosine similarity via embedding_store. High similarity =
        content homogenization = cascade signal.

        Days with < 2 articles → similarity = 0 → Z clipped to 0.
        """
        raw = pd.Series(0.0, index=date_range)

        if self.embedding_store is None or date_col is None or articles.empty:
            return raw

        frame_col = FRAME_COLUMNS.get(frame)
        frame_articles = self._filter_frame_articles(articles, frame_col)
        if frame_articles.empty or 'doc_id' not in frame_articles.columns:
            return raw

        dates = pd.to_datetime(frame_articles[date_col], errors='coerce').dt.normalize()
        frame_articles = frame_articles.copy()
        frame_articles['_date'] = dates

        n_days_computed = 0
        n_days_with_articles = 0
        for date in date_range:
            day_articles = frame_articles[frame_articles['_date'] == date]
            if len(day_articles) < 2:
                continue
            n_days_with_articles += 1
            doc_ids = day_articles['doc_id'].dropna().unique().tolist()
            if len(doc_ids) < 2:
                continue
            sim = self.embedding_store.mean_pairwise_similarity(doc_ids)
            if sim > 0:
                n_days_computed += 1
            raw[date] = sim

        if n_days_with_articles > 0 and n_days_computed == 0:
            logger.warning(
                f"Semantic signal [{frame}]: 0/{n_days_with_articles} days "
                f"produced similarity > 0 — embeddings may be missing"
            )
        elif n_days_with_articles > 0:
            logger.debug(
                f"Semantic signal [{frame}]: {n_days_computed}/{n_days_with_articles} "
                f"days with similarity > 0"
            )

        return self._rolling_zscore(raw)

    # =========================================================================
    # Composite
    # =========================================================================

    def _build_composite(self, z_signals: Dict[str, pd.Series]) -> pd.Series:
        """Weighted sum of the 5 Z-score signals.

        Uses dedicated signal weights (separate from scoring dimension weights).
        """
        composite = (
            self.config.signal_weight_temporal * z_signals['z_temporal']
            + self.config.signal_weight_participation * z_signals['z_participation']
            + self.config.signal_weight_convergence * z_signals['z_convergence']
            + self.config.signal_weight_source * z_signals['z_source']
            + self.config.signal_weight_semantic * z_signals['z_semantic']
        )
        return composite

    # =========================================================================
    # Shared utilities
    # =========================================================================

    def _rolling_zscore(self, series: pd.Series) -> pd.Series:
        """One-sided rolling Z-score with trailing window, clipped at 0.

        Uses a 90-day trailing window with shift(1) to prevent lookahead.
        Z-scores are clipped to max(0, z) so only positive anomalies count.
        """
        window = self.config.baseline_window_days
        rolling_mean = series.rolling(window=window, min_periods=window // 2).mean().shift(1)
        rolling_std = series.rolling(window=window, min_periods=window // 2).std().shift(1)

        # Fill initial NaNs with global stats
        global_mean = series.mean()
        global_std = series.std()
        rolling_mean = rolling_mean.fillna(global_mean)
        rolling_std = rolling_std.fillna(global_std)

        # Floor std to prevent division by zero
        rolling_std = rolling_std.clip(lower=max(global_std * 0.1, 1e-10))

        z = (series - rolling_mean) / rolling_std
        return z.clip(lower=0.0)

    @staticmethod
    def _orthogonalize(target: pd.Series, reference: pd.Series) -> pd.Series:
        """Remove the projection of *target* onto *reference*, then re-clip to 0.

        Given two one-sided (>= 0) Z-score series, this computes:
            residual = target - (dot(target, reference) / dot(reference, reference)) * reference
            result   = max(0, residual)

        When target and reference are highly correlated (e.g. 0.92 for
        z_convergence / z_temporal), the projection captures nearly all
        of target's variance, leaving a small residual that only fires
        when convergence deviates from what temporal alone predicts.

        If reference is all-zero, returns target unchanged.
        """
        ref_vals = reference.values.astype(float)
        tgt_vals = target.values.astype(float)

        ref_dot = np.dot(ref_vals, ref_vals)
        if ref_dot < 1e-10:
            return target

        beta = np.dot(tgt_vals, ref_vals) / ref_dot
        residual = tgt_vals - beta * ref_vals
        result = pd.Series(np.clip(residual, 0.0, None), index=target.index)
        return result

    @staticmethod
    def _resolve_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
        """Return the first column name from candidates that exists in df."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _filter_frame_articles(articles: pd.DataFrame, frame_col: Optional[str]) -> pd.DataFrame:
        """Filter articles to those using a specific frame."""
        if frame_col is None:
            return articles

        for col in [frame_col, f"{frame_col}_sum", f"{frame_col}_mean"]:
            if col in articles.columns:
                return articles[articles[col] > 0]

        return articles
