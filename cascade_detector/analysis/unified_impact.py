"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
unified_impact.py

MAIN OBJECTIVE:
---------------
Unified causal impact analysis between three entity types:
  - EventClusters (detected real-world events)
  - Cascades (media cascade bursts)
  - Paradigm dominance (frame dominance over time)

Three phases measure continuous impact via diff-in-diff, dose-response
(cross-correlation), and Granger causality:
  Phase 1: Cluster → Cascade
  Phase 2: Cluster → Dominance
  Phase 3: Cascade → Dominance

Replaces the legacy EventImpactAnalyzer (annotation-level prevalence ratios).

Author:
-------
Antoine Lemor
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from cascade_detector.core.constants import FRAMES, FRAME_COLUMNS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRE_WINDOW = 14     # days before peak for diff-in-diff
POST_WINDOW = 28    # days after peak for diff-in-diff
MAX_LAG = 14        # max lag for cross-correlation
GRANGER_MAX_LAG = 7 # max lag for Granger test
PROXIMITY_SIGMA = 21  # days, Gaussian decay σ for temporal proximity

# Phase 1 weights WITH Granger (sum=1.0)
P1_W_DID_G = 0.30
P1_W_XCORR_G = 0.30
P1_W_GRANGER_P1 = 0.20
P1_W_PROX_G = 0.20

# Phase 1 weights WITHOUT Granger — fallback (sum=1.0)
P1_W_DID = 0.35
P1_W_XCORR = 0.35
P1_W_PROX = 0.30
P1_OVERLAP_FLOOR = 0.20   # soft gate: overlap=0 retains 20% of raw impact

# Phase 1 Granger minimum observations
P1_GRANGER_MIN_OBS = 21

# Phase 2/3 weights
P23_W_DID = 0.30
P23_W_XCORR = 0.30
P23_W_GRANGER = 0.40

GRANGER_P_THRESHOLD = 0.10  # p-value mapped linearly to [0,1]

# Role classification thresholds
IMPACT_STRONG = 0.15
IMPACT_MODERATE = 0.05
IMPACT_WEAK = 0.01
DID_NOISE_THRESHOLD = 0.01    # |DID| below this → neutral (no directional effect)
GRANGER_SIG_THRESHOLD = 0.05  # binary significance for role assignment

# Late support reclassification (Phase 1 post-peak suppressors)
LATE_SUPPORT_FRAME_RATIO = 0.50    # cluster affinity >= 50% of driver mean
LATE_SUPPORT_EMBEDDING_SIM = 0.30  # cosine sim floor for alignment check
LATE_SUPPORT_MIN_AFFINITY = 0.10   # absolute floor (raised from 0.05 — evaluation showed too low)

# Content relevance (multiplicative gate)
CONTENT_RELEVANCE_FLOOR = 0.30      # floor (preserves suppressors without frame)
CONTENT_AFFINITY_SATURATION = 0.20  # frame_affinity saturates at this value
CONTENT_DRIVER_THRESHOLD = 0.03     # min frame_affinity for driver role

# Statistical significance
STAT_SIGNIFICANCE_ALPHA = 0.10      # p-value threshold for stat gate

# Permutation test
N_PERMUTATIONS = 200
P_VALUE_FDR_ALPHA = 0.10


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class UnifiedImpactResults:
    """Container for unified impact analysis outputs."""
    cluster_cascade: pd.DataFrame = field(default_factory=pd.DataFrame)
    cluster_dominance: pd.DataFrame = field(default_factory=pd.DataFrame)
    cascade_dominance: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cluster_cascade': self.cluster_cascade,
            'cluster_dominance': self.cluster_dominance,
            'cascade_dominance': self.cascade_dominance,
            'summary': self.summary,
        }


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------
class UnifiedImpactAnalyzer:
    """Measures continuous causal impact between clusters, cascades, and dominance."""

    # ------------------------------------------------------------------
    # Private statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _diff_in_diff(target: pd.Series, peak_date: pd.Timestamp,
                      pre_days: int = PRE_WINDOW,
                      post_days: int = POST_WINDOW) -> float:
        """Compute diff-in-diff: mean(post) - mean(pre) around peak_date.

        Returns signed float. Returns 0.0 if insufficient data.
        """
        if target.empty:
            return 0.0

        idx = target.index
        pre_mask = (idx >= peak_date - pd.Timedelta(days=pre_days)) & (idx < peak_date)
        post_mask = (idx >= peak_date) & (idx <= peak_date + pd.Timedelta(days=post_days))

        pre_vals = target[pre_mask]
        post_vals = target[post_mask]

        if pre_vals.empty or post_vals.empty:
            return 0.0

        return float(post_vals.mean() - pre_vals.mean())

    @staticmethod
    def _dose_response(source: pd.Series, target: pd.Series,
                       max_lag: int = MAX_LAG) -> Tuple[float, int]:
        """Cross-correlation at lags [-max_lag, +max_lag].

        Positive lag = source leads target.
        Returns (max_corr, best_lag). Falls back to (0.0, 0) if series too short.
        """
        if len(source) < 3 or len(target) < 3:
            return 0.0, 0

        # Align on shared date range
        common = source.index.intersection(target.index)
        if len(common) < 3:
            return 0.0, 0

        s = source.reindex(common).fillna(0.0).values
        t = target.reindex(common).fillna(0.0).values

        # Normalize
        s_std = np.std(s)
        t_std = np.std(t)
        if s_std < 1e-12 or t_std < 1e-12:
            return 0.0, 0

        s_norm = (s - np.mean(s)) / s_std
        t_norm = (t - np.mean(t)) / t_std
        n = len(s_norm)

        best_corr = 0.0
        best_lag = 0

        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                s_slice = s_norm[:n - lag] if lag > 0 else s_norm
                t_slice = t_norm[lag:] if lag > 0 else t_norm
            else:
                s_slice = s_norm[-lag:]
                t_slice = t_norm[:n + lag]

            if len(s_slice) < 3 or len(s_slice) != len(t_slice):
                continue

            corr = float(np.mean(s_slice * t_slice))
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        return best_corr, best_lag

    @staticmethod
    def _granger_causality(source: pd.Series, target: pd.Series,
                           max_lag: int = GRANGER_MAX_LAG) -> Tuple[float, float]:
        """Granger causality test: source → target.

        Returns (f_stat, p_value) at best lag.
        Falls back to (0.0, 1.0) if series too short or test fails.
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except ImportError:
            logger.warning("statsmodels not available, skipping Granger test")
            return 0.0, 1.0

        # Align
        common = source.index.intersection(target.index)
        if len(common) < max_lag + 3:
            return 0.0, 1.0

        s = source.reindex(common).fillna(0.0).values
        t = target.reindex(common).fillna(0.0).values

        # Variance check
        if np.std(s) < 1e-12 or np.std(t) < 1e-12:
            return 0.0, 1.0

        data = np.column_stack([t, s])  # statsmodels convention: [target, source]

        try:
            results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        except Exception:
            return 0.0, 1.0

        best_f = 0.0
        best_p = 1.0
        for lag in range(1, max_lag + 1):
            if lag not in results:
                continue
            tests = results[lag][0]
            # Use ssr_ftest
            f_stat = tests['ssr_ftest'][0]
            p_val = tests['ssr_ftest'][1]
            if f_stat > best_f:
                best_f = f_stat
                best_p = p_val

        return float(best_f), float(best_p)

    @staticmethod
    def _temporal_proximity(date_a: pd.Timestamp, date_b: pd.Timestamp,
                            sigma: float = PROXIMITY_SIGMA) -> float:
        """Gaussian decay proximity between two dates."""
        dt = abs((date_a - date_b).total_seconds()) / 86400.0
        return float(np.exp(-dt**2 / (2 * sigma**2)))

    @staticmethod
    def _build_daily_mass(cluster, articles: pd.DataFrame) -> pd.Series:
        """Build daily effective mass series for an EventCluster.

        Aggregates belonging across occurrences (max per doc_id to avoid
        double-counting), then groups by article date.
        """
        # Merge belonging from all occurrences, taking max per doc_id
        combined = {}
        for occ in cluster.occurrences:
            for doc_id, score in occ.belonging.items():
                combined[doc_id] = max(combined.get(doc_id, 0.0), score)

        if not combined:
            return pd.Series(dtype=float)

        doc_ids = list(combined.keys())
        scores = [combined[d] for d in doc_ids]

        # Look up dates from articles DataFrame
        date_col = None
        for candidate in ('date', 'date_first', 'date_converted_first'):
            if candidate in articles.columns:
                date_col = candidate
                break
        if date_col is None:
            return pd.Series(dtype=float)

        id_col = 'doc_id' if 'doc_id' in articles.columns else articles.index.name
        if id_col == 'doc_id':
            date_lookup = articles.set_index('doc_id')[date_col]
        else:
            date_lookup = articles[date_col]

        rows = []
        for doc_id, score in zip(doc_ids, scores):
            if doc_id in date_lookup.index:
                rows.append({'date': pd.Timestamp(date_lookup[doc_id]), 'mass': score})

        if not rows:
            return pd.Series(dtype=float)

        mass_df = pd.DataFrame(rows)
        daily = mass_df.groupby('date')['mass'].sum()

        # Reindex to full core period with zero-fill
        full_range = pd.date_range(cluster.core_start, cluster.core_end, freq='D')
        daily = daily.reindex(full_range, fill_value=0.0)
        daily.index.name = None

        return daily

    @staticmethod
    def _get_dominance_series(paradigm_timeline: pd.DataFrame,
                              frame: str) -> pd.Series:
        """Extract paradigm_{frame} column as a DatetimeIndex Series."""
        col = f'paradigm_{frame}'
        if paradigm_timeline.empty or col not in paradigm_timeline.columns:
            return pd.Series(dtype=float)

        series = paradigm_timeline.set_index('date')[col].copy()
        series.index = pd.DatetimeIndex(series.index)
        return series

    @staticmethod
    def _normalize_did(did: float, scale: float = 0.5) -> float:
        """Normalize diff-in-diff to [0, 1] via tanh scaling."""
        return float(np.tanh(abs(did) / scale)) if scale > 0 else 0.0

    @staticmethod
    def _diff_in_diff_with_pvalue(
        target: pd.Series, peak_date: pd.Timestamp,
        pre_days: int = PRE_WINDOW, post_days: int = POST_WINDOW,
    ) -> Tuple[float, float]:
        """Compute DID with Welch's t-test p-value.

        Returns (did, p_value). Falls back to (0.0, 1.0) if < 3 obs per window.
        """
        if target.empty:
            return 0.0, 1.0

        idx = target.index
        pre_mask = (idx >= peak_date - pd.Timedelta(days=pre_days)) & (idx < peak_date)
        post_mask = (idx >= peak_date) & (idx <= peak_date + pd.Timedelta(days=post_days))

        pre_vals = target[pre_mask].values
        post_vals = target[post_mask].values

        if len(pre_vals) < 3 or len(post_vals) < 3:
            return 0.0, 1.0

        did = float(np.mean(post_vals) - np.mean(pre_vals))
        _, p_value = scipy_stats.ttest_ind(post_vals, pre_vals, equal_var=False)

        if np.isnan(p_value):
            p_value = 1.0

        return did, float(p_value)

    @staticmethod
    def _dose_response_with_pvalue(
        source: pd.Series, target: pd.Series,
        max_lag: int = MAX_LAG,
    ) -> Tuple[float, int, float]:
        """Cross-correlation with Fisher z-transform p-value.

        Returns (max_corr, best_lag, p_value).
        Falls back to (0.0, 0, 1.0) if series too short.
        """
        corr, lag = UnifiedImpactAnalyzer._dose_response(source, target, max_lag)

        if abs(corr) < 1e-12:
            return corr, lag, 1.0

        # Compute effective sample size at the best lag
        common = source.index.intersection(target.index)
        n = len(common) - abs(lag)

        if n < 4:
            return corr, lag, 1.0

        # Fisher z-transform
        r = np.clip(corr, -0.9999, 0.9999)
        z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n - 3)
        p_value = float(2.0 * (1.0 - scipy_stats.norm.cdf(abs(z))))

        return corr, lag, p_value

    @staticmethod
    def _compute_content_relevance(
        frame_affinity: float, embedding_alignment: float,
    ) -> float:
        """Compute content relevance multiplicative gate in [FLOOR, 1.0].

        Combines frame_affinity (saturating at CONTENT_AFFINITY_SATURATION)
        and embedding_alignment into a single content relevance score.
        """
        if np.isnan(embedding_alignment):
            alignment = 0.5  # default when unavailable
        else:
            alignment = np.clip(embedding_alignment, 0.0, 1.0)

        affinity_norm = min(1.0, frame_affinity / CONTENT_AFFINITY_SATURATION) \
            if CONTENT_AFFINITY_SATURATION > 0 else 0.0

        raw = 0.60 * affinity_norm + 0.40 * alignment
        result = CONTENT_RELEVANCE_FLOOR + (1.0 - CONTENT_RELEVANCE_FLOOR) * min(1.0, raw)
        return float(result)

    @staticmethod
    def _compute_attribution_confidence(row: dict) -> float:
        """Compute composite confidence score for a Phase 1 row.

        Combines 4 dimensions:
          stat_conf (0.30): min p-value significance
          perm_conf (0.30): permutation significance
          content_conf (0.25): content relevance
          temporal_conf (0.15): temporal proximity
        Returns [0, 1].
        """
        # Statistical confidence: best of DID/xcorr/Granger p-values
        p_vals = [row.get('did_p_value', 1.0), row.get('xcorr_p_value', 1.0)]
        granger_p = row.get('granger_p', 1.0)
        if granger_p < 1.0:
            p_vals.append(granger_p)
        min_p = min(p_vals)
        stat_conf = float(np.clip(1.0 - min_p / STAT_SIGNIFICANCE_ALPHA, 0.0, 1.0))

        # Permutation confidence
        perm_p = row.get('perm_p_adjusted', 1.0)
        perm_conf = float(np.clip(1.0 - perm_p / P_VALUE_FDR_ALPHA, 0.0, 1.0))

        # Content confidence
        content_conf = float(np.clip(row.get('content_relevance', CONTENT_RELEVANCE_FLOOR), 0.0, 1.0))

        # Temporal confidence (proximity)
        temporal_conf = float(np.clip(row.get('proximity', 0.0), 0.0, 1.0))

        return float(
            0.30 * stat_conf
            + 0.30 * perm_conf
            + 0.25 * content_conf
            + 0.15 * temporal_conf
        )

    @staticmethod
    def _circular_shift_series(series: pd.Series, shift: int) -> pd.Series:
        """Circular shift of series values, preserving index."""
        return pd.Series(np.roll(series.values, shift), index=series.index)

    @staticmethod
    def _benjamini_hochberg(p_values: np.ndarray, alpha: float = P_VALUE_FDR_ALPHA) -> np.ndarray:
        """Benjamini-Hochberg FDR correction.

        Returns adjusted p-values (same order as input).
        """
        n = len(p_values)
        if n == 0:
            return np.array([], dtype=float)

        # Sort and compute adjusted p-values
        order = np.argsort(p_values)
        sorted_p = p_values[order]

        adjusted = np.empty(n, dtype=float)
        adjusted[-1] = sorted_p[-1]  # largest p-value unchanged
        for i in range(n - 2, -1, -1):
            rank = i + 1
            adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n / rank)

        # Clip to [0, 1]
        adjusted = np.clip(adjusted, 0.0, 1.0)

        # Restore original order
        result = np.empty(n, dtype=float)
        result[order] = adjusted
        return result

    @staticmethod
    def _vectorized_did_batch(
        shifted_targets: np.ndarray,
        pre_mask: np.ndarray,
        post_mask: np.ndarray,
    ) -> np.ndarray:
        """Batch DID on multiple shifted targets (2D: n_perms × n_days).

        Returns array of shape (n_perms,) with DID values.
        Falls back to 0.0 where pre or post window has no data.
        """
        n_pre = pre_mask.sum()
        n_post = post_mask.sum()
        if n_pre == 0 or n_post == 0:
            return np.zeros(shifted_targets.shape[0])

        pre_mean = shifted_targets[:, pre_mask].mean(axis=1)
        post_mean = shifted_targets[:, post_mask].mean(axis=1)
        return post_mean - pre_mean

    @staticmethod
    def _vectorized_xcorr_batch(
        source_norm: np.ndarray,
        shifted_targets: np.ndarray,
        max_lag: int = MAX_LAG,
    ) -> np.ndarray:
        """Batch cross-correlation for multiple shifted targets.

        source_norm: pre-normalized source array (n_common,)
        shifted_targets: 2D array (n_perms, n_common)
        Returns array of shape (n_perms,) with max |corr| across lags.
        """
        n_perms, n = shifted_targets.shape
        if n < 3:
            return np.zeros(n_perms)

        # Normalize targets: (n_perms, n)
        t_means = shifted_targets.mean(axis=1, keepdims=True)
        t_stds = shifted_targets.std(axis=1, keepdims=True)
        # Avoid division by zero
        valid = (t_stds.ravel() > 1e-12)
        t_norm = np.zeros_like(shifted_targets)
        t_norm[valid] = (shifted_targets[valid] - t_means[valid]) / t_stds[valid]

        best_corr = np.zeros(n_perms)

        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                if lag > 0:
                    s_slice = source_norm[:n - lag]
                    t_slice = t_norm[:, lag:]
                else:
                    s_slice = source_norm
                    t_slice = t_norm
            else:
                s_slice = source_norm[-lag:]
                t_slice = t_norm[:, :n + lag]

            if len(s_slice) < 3 or t_slice.shape[1] != len(s_slice):
                continue

            # (n_perms,) dot product
            corr = (t_slice * s_slice[np.newaxis, :]).mean(axis=1)
            # Update best where |corr| is larger
            better = np.abs(corr) > np.abs(best_corr)
            best_corr[better] = corr[better]

        return best_corr

    @staticmethod
    def _compute_permutation_pvalues(
        df: pd.DataFrame,
        clusters: list,
        cascades: list,
        articles: pd.DataFrame,
        n_permutations: int = N_PERMUTATIONS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute permutation p-values via circular shift of cascade daily_composite.

        Vectorized implementation: for each cascade, all permutation shifts
        are applied as a 2D matrix (n_perms × n_days) and DID/xcorr are
        computed in batch via numpy operations.

        Returns (raw_p_values, adjusted_p_values) aligned with df index.
        """
        if df.empty:
            return np.array([], dtype=float), np.array([], dtype=float)

        analyzer = UnifiedImpactAnalyzer()
        rng = np.random.default_rng(42)

        # Build lookups
        cluster_lookup = {c.cluster_id: c for c in clusters}

        # Pre-compute daily mass per cluster
        daily_mass_cache = {}
        for c in clusters:
            daily_mass_cache[c.cluster_id] = analyzer._build_daily_mass(c, articles)

        # Observed composite per row (stat component only)
        did_norm_arr = np.tanh(np.abs(df['diff_in_diff'].values) / 0.5)
        xcorr_arr = np.abs(df['dose_response_corr'].values)
        prox_arr = df['proximity'].values
        granger_p_arr = df['granger_p'].values if 'granger_p' in df.columns else np.ones(len(df))
        granger_available = granger_p_arr < 1.0
        granger_sig_arr = np.maximum(0.0, 1.0 - granger_p_arr / GRANGER_P_THRESHOLD)

        observed = np.where(
            granger_available,
            P1_W_DID_G * did_norm_arr + P1_W_XCORR_G * xcorr_arr
            + P1_W_GRANGER_P1 * granger_sig_arr + P1_W_PROX_G * prox_arr,
            P1_W_DID * did_norm_arr + P1_W_XCORR * xcorr_arr + P1_W_PROX * prox_arr,
        )

        n_exceed = np.zeros(len(df))

        # Group rows by cascade
        cascade_ids = df['cascade_id'].values
        cluster_ids = df['cluster_id'].values
        unique_cascade_ids = np.unique(cascade_ids)

        # Build cascade target lookup
        cascade_targets = {}
        for cascade in cascades:
            target = getattr(cascade, 'daily_composite', None)
            if target is None or (hasattr(target, 'empty') and target.empty):
                continue
            if not isinstance(target.index, pd.DatetimeIndex):
                target = target.copy()
                target.index = pd.DatetimeIndex(target.index)
            cascade_targets[cascade.cascade_id] = target

        for cascade_id in unique_cascade_ids:
            target = cascade_targets.get(cascade_id)
            if target is None:
                continue

            row_mask = cascade_ids == cascade_id
            row_indices = np.where(row_mask)[0]
            if len(row_indices) == 0:
                continue

            target_vals = target.values
            target_index = target.index
            n_days = len(target_vals)
            if n_days < 2:
                continue

            # Generate all shifts at once: (n_perms, n_days)
            shifts = rng.integers(1, n_days, size=n_permutations)
            shifted_matrix = np.stack([
                np.roll(target_vals, s) for s in shifts
            ])  # (n_perms, n_days)

            # For each cluster in this cascade group, compute batch DID + xcorr
            unique_clusters_in_group = np.unique(cluster_ids[row_indices])

            for cid in unique_clusters_in_group:
                # Find rows for this (cascade, cluster) pair
                pair_mask = (cascade_ids == cascade_id) & (cluster_ids == cid)
                pair_indices = np.where(pair_mask)[0]
                if len(pair_indices) == 0:
                    continue

                cluster = cluster_lookup.get(cid)
                if cluster is None:
                    continue

                peak_date = cluster.peak_date

                # --- Batch DID ---
                # Build pre/post masks on target_index
                pre_mask = (
                    (target_index >= peak_date - pd.Timedelta(days=PRE_WINDOW))
                    & (target_index < peak_date)
                )
                post_mask = (
                    (target_index >= peak_date)
                    & (target_index <= peak_date + pd.Timedelta(days=POST_WINDOW))
                )
                pre_arr = pre_mask.values if hasattr(pre_mask, 'values') else np.asarray(pre_mask)
                post_arr = post_mask.values if hasattr(post_mask, 'values') else np.asarray(post_mask)
                did_batch = analyzer._vectorized_did_batch(
                    shifted_matrix, pre_arr, post_arr
                )  # (n_perms,)
                did_norm_batch = np.tanh(np.abs(did_batch) / 0.5)

                # --- Batch xcorr ---
                daily_mass = daily_mass_cache.get(cid, pd.Series(dtype=float))
                if len(daily_mass) >= 3:
                    common = daily_mass.index.intersection(target_index)
                    if len(common) >= 3:
                        s_vals = daily_mass.reindex(common).fillna(0.0).values
                        s_std = np.std(s_vals)
                        if s_std > 1e-12:
                            s_norm = (s_vals - np.mean(s_vals)) / s_std

                            # Extract common columns from shifted matrix
                            # Map common dates to target_index positions
                            common_pos = np.array([
                                target_index.get_loc(d) for d in common
                            ])
                            shifted_common = shifted_matrix[:, common_pos]  # (n_perms, n_common)

                            xcorr_batch = analyzer._vectorized_xcorr_batch(
                                s_norm, shifted_common
                            )  # (n_perms,)
                        else:
                            xcorr_batch = np.zeros(n_permutations)
                    else:
                        xcorr_batch = np.zeros(n_permutations)
                else:
                    xcorr_batch = np.zeros(n_permutations)

                # --- Compute surrogate composite for all perms ---
                # Use the same row-level granger/prox for all permutations
                for ri in pair_indices:
                    prox = prox_arr[ri]
                    g_avail = granger_available[ri]
                    g_sig = granger_sig_arr[ri]

                    if g_avail:
                        surrogate_batch = (
                            P1_W_DID_G * did_norm_batch
                            + P1_W_XCORR_G * np.abs(xcorr_batch)
                            + P1_W_GRANGER_P1 * g_sig
                            + P1_W_PROX_G * prox
                        )
                    else:
                        surrogate_batch = (
                            P1_W_DID * did_norm_batch
                            + P1_W_XCORR * np.abs(xcorr_batch)
                            + P1_W_PROX * prox
                        )

                    n_exceed[ri] = np.sum(surrogate_batch >= observed[ri])

        raw_p = (n_exceed + 1) / (n_permutations + 1)
        adjusted_p = UnifiedImpactAnalyzer._benjamini_hochberg(raw_p)

        return raw_p, adjusted_p

    # ------------------------------------------------------------------
    # Role classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_impact(impact_score: float) -> str:
        """Classify continuous impact score into qualitative label."""
        if impact_score >= IMPACT_STRONG:
            return 'strong'
        elif impact_score >= IMPACT_MODERATE:
            return 'moderate'
        elif impact_score >= IMPACT_WEAK:
            return 'weak'
        return 'negligible'

    @staticmethod
    def _assign_roles_phase1(df: pd.DataFrame) -> pd.DataFrame:
        """Assign role and impact_label to Phase 1 (Cluster → Cascade) rows.

        V2 logic with statistical gates:
            1. impact < IMPACT_WEAK → 'unrelated'
            2. No p-value < STAT_SIGNIFICANCE_ALPHA → 'neutral'
            3. DID > noise, post-peak → 'late_support' if affinity >= threshold, else 'neutral'
            4. DID > noise, pre-peak → 'driver' if affinity >= threshold, else 'neutral'
            5. DID < -noise → 'suppressor' (pre-peak and post-peak)
            6. |DID| <= noise → 'neutral'
        """
        if df.empty:
            df['role'] = pd.Series(dtype=str)
            df['impact_label'] = pd.Series(dtype=str)
            return df

        roles = []
        for _, row in df.iterrows():
            score = row['impact_score']
            did = row['diff_in_diff']

            # 1. Low impact → unrelated
            if score < IMPACT_WEAK:
                roles.append('unrelated')
                continue

            # 2. Statistical gate: at least one p-value significant
            p_vals = []
            for col in ('did_p_value', 'xcorr_p_value', 'granger_p'):
                v = row.get(col, 1.0)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    p_vals.append(v)
            has_significant = any(p < STAT_SIGNIFICANCE_ALPHA for p in p_vals) if p_vals else False

            if not has_significant:
                roles.append('neutral')
                continue

            # 3/4. Positive DID above noise
            if did > DID_NOISE_THRESHOLD:
                is_post_peak = row.get('is_post_peak', False)
                affinity = row.get('frame_affinity', 0.0)
                if is_post_peak:
                    if affinity >= CONTENT_DRIVER_THRESHOLD:
                        roles.append('late_support')
                    else:
                        roles.append('neutral')
                else:
                    if affinity >= CONTENT_DRIVER_THRESHOLD:
                        roles.append('driver')
                    else:
                        roles.append('neutral')
                continue

            # 5. Negative DID below -noise → suppressor
            if did < -DID_NOISE_THRESHOLD:
                roles.append('suppressor')
                continue

            # 6. DID within noise band
            roles.append('neutral')

        df['role'] = roles
        df['impact_label'] = df['impact_score'].apply(
            UnifiedImpactAnalyzer._classify_impact
        )
        return df

    @staticmethod
    def _compute_embedding_alignment(
        df: pd.DataFrame, clusters: list, cascades: list,
    ) -> pd.DataFrame:
        """Compute cosine similarity between each cluster centroid and
        its cascade's driver centroid (strength-weighted mean of driver centroids).

        Adds 'embedding_alignment' column (NaN when unavailable).
        """
        if df.empty:
            df['embedding_alignment'] = pd.Series(dtype=float)
            return df

        # Build cluster centroid lookup
        cluster_centroids = {}
        for c in clusters:
            centroid = getattr(c, 'centroid', None)
            if centroid is not None and np.isfinite(centroid).all():
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    cluster_centroids[c.cluster_id] = centroid / norm

        # Build driver centroid per cascade
        driver_centroids = {}
        drivers = df[df['role'] == 'driver']
        for cascade_id in df['cascade_id'].unique():
            cascade_drivers = drivers[drivers['cascade_id'] == cascade_id]
            if cascade_drivers.empty:
                continue

            weighted_vecs = []
            weight_sum = 0.0
            for _, row in cascade_drivers.iterrows():
                cid = row['cluster_id']
                if cid in cluster_centroids:
                    w = row.get('cluster_strength', 1.0)
                    weighted_vecs.append(cluster_centroids[cid] * w)
                    weight_sum += w

            if weighted_vecs and weight_sum > 0:
                mean_vec = np.sum(weighted_vecs, axis=0) / weight_sum
                norm = np.linalg.norm(mean_vec)
                if norm > 0:
                    driver_centroids[cascade_id] = mean_vec / norm

        # Compute alignment per row
        alignments = []
        for _, row in df.iterrows():
            cid = row['cluster_id']
            cascade_id = row['cascade_id']
            if cid in cluster_centroids and cascade_id in driver_centroids:
                sim = float(np.dot(
                    cluster_centroids[cid], driver_centroids[cascade_id]
                ))
                alignments.append(sim)
            else:
                alignments.append(np.nan)

        df['embedding_alignment'] = alignments
        return df

    @staticmethod
    def _reclassify_late_support(df: pd.DataFrame) -> pd.DataFrame:
        """Reclassify post-peak suppressors with high frame affinity as 'late_support'.

        Only modifies rows where role='suppressor' AND is_post_peak=True.
        Requires drivers as reference; no drivers → no reclassification.
        """
        if df.empty:
            return df

        if 'is_post_peak' not in df.columns:
            return df

        new_roles = df['role'].copy()
        has_alignment = 'embedding_alignment' in df.columns

        for cascade_id in df['cascade_id'].unique():
            cmask = df['cascade_id'] == cascade_id
            cascade_df = df[cmask]

            # Driver reference
            driver_rows = cascade_df[cascade_df['role'] == 'driver']
            if driver_rows.empty:
                continue

            driver_mean_affinity = driver_rows['frame_affinity'].mean()
            threshold = max(
                LATE_SUPPORT_MIN_AFFINITY,
                LATE_SUPPORT_FRAME_RATIO * driver_mean_affinity,
            )

            # Candidates: post-peak suppressors
            candidates = cascade_df[
                (cascade_df['role'] == 'suppressor')
                & (cascade_df['is_post_peak'] == True)
            ]

            for idx, row in candidates.iterrows():
                if row['frame_affinity'] < threshold:
                    continue  # low frame content → true suppressor

                if has_alignment:
                    alignment = row['embedding_alignment']
                    if not np.isnan(alignment) and alignment < LATE_SUPPORT_EMBEDDING_SIM:
                        continue  # semantically distant → true suppressor

                new_roles.at[idx] = 'late_support'

        df['role'] = new_roles
        return df

    @staticmethod
    def _assign_roles_phase2(df: pd.DataFrame) -> pd.DataFrame:
        """Assign role and impact_label to Phase 2 (Cluster → Dominance) rows."""
        if df.empty:
            df['role'] = pd.Series(dtype=str)
            df['impact_label'] = pd.Series(dtype=str)
            return df

        roles = []
        for _, row in df.iterrows():
            p_val = row['granger_p']
            did = row['diff_in_diff']

            if p_val >= GRANGER_SIG_THRESHOLD:
                roles.append('inert')
            elif did > 0:
                roles.append('catalyst')
            else:
                roles.append('disruptor')

        df['role'] = roles
        df['impact_label'] = df['impact_score'].apply(
            UnifiedImpactAnalyzer._classify_impact
        )
        return df

    @staticmethod
    def _assign_roles_phase3(df: pd.DataFrame) -> pd.DataFrame:
        """Assign role, impact_label, and is_own_frame to Phase 3 (Cascade → Dominance) rows."""
        if df.empty:
            df['role'] = pd.Series(dtype=str)
            df['impact_label'] = pd.Series(dtype=str)
            df['is_own_frame'] = pd.Series(dtype=bool)
            return df

        roles = []
        is_own = []
        for _, row in df.iterrows():
            own_frame = (row['frame'] == row['cascade_frame'])
            is_own.append(own_frame)
            p_val = row['granger_p']
            did = row['diff_in_diff']

            if p_val >= GRANGER_SIG_THRESHOLD:
                roles.append('dormant')
            elif own_frame and did > 0:
                roles.append('amplification')
            else:
                # own frame + did <= 0, or other frame + significant
                roles.append('destabilisation')

        df['role'] = roles
        df['is_own_frame'] = is_own
        df['impact_label'] = df['impact_score'].apply(
            UnifiedImpactAnalyzer._classify_impact
        )
        return df

    @staticmethod
    def _aggregate_cascade_roles(df: pd.DataFrame) -> Dict[str, str]:
        """Derive cascade-level dominant role from Phase 3 own-frame rows.

        Returns dict mapping cascade_id → role ('amplification', 'destabilisation', 'dormant').
        """
        if df.empty:
            return {}

        result = {}
        for cascade_id in df['cascade_id'].unique():
            cascade_rows = df[df['cascade_id'] == cascade_id]
            own_rows = cascade_rows[cascade_rows['is_own_frame']]

            if not own_rows.empty and 'amplification' in own_rows['role'].values:
                result[cascade_id] = 'amplification'
            elif 'destabilisation' in cascade_rows['role'].values:
                result[cascade_id] = 'destabilisation'
            else:
                result[cascade_id] = 'dormant'

        return result

    # ------------------------------------------------------------------
    # Phase 1: Cluster → Cascade
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_article_overlap(cluster, cascade_doc_ids: set) -> float:
        """Compute fraction of cluster articles that are cascade members.

        A cascade's article set = articles with frame > 0 during the cascade
        period.  The overlap is the Jaccard-like ratio |intersection| / |cluster|,
        establishing a material link: the event's articles actually participate
        in the cascade, not just co-occur temporally.
        """
        cluster_doc_ids = set()
        for occ in cluster.occurrences:
            cluster_doc_ids.update(occ.belonging.keys())

        if not cluster_doc_ids:
            return 0.0

        shared = cluster_doc_ids & cascade_doc_ids
        return len(shared) / len(cluster_doc_ids)

    def phase1_cluster_cascade(
        self,
        clusters: list,
        cascades: list,
        articles: pd.DataFrame,
    ) -> pd.DataFrame:
        """Measure impact of EventClusters on Cascades.

        Returns DataFrame with one row per (cluster_id, cascade_id).
        Impact score is modulated by article overlap via a soft gate
        (floor at P1_OVERLAP_FLOOR) to attenuate pairs without shared
        articles without zeroing them out.
        """
        # Pre-compute cascade article sets (doc_ids with frame > 0 in window)
        cascade_doc_sets = {}
        if not articles.empty:
            date_col = None
            for col in ('date', 'date_first', 'date_converted_first'):
                if col in articles.columns:
                    date_col = col
                    break
            id_col = 'doc_id' if 'doc_id' in articles.columns else None
            if date_col and id_col:
                dates = pd.to_datetime(articles[date_col], errors='coerce')
                for cascade in cascades:
                    onset = getattr(cascade, 'onset_date', None)
                    end = getattr(cascade, 'end_date', None)
                    if onset is None or end is None:
                        cascade_doc_sets[cascade.cascade_id] = set()
                        continue
                    mask = (dates >= onset) & (dates <= end)
                    frame_col = FRAME_COLUMNS.get(cascade.frame)
                    if frame_col:
                        for fc in [frame_col, f"{frame_col}_sum", f"{frame_col}_mean"]:
                            if fc in articles.columns:
                                mask = mask & (articles[fc] > 0)
                                break
                    cascade_doc_sets[cascade.cascade_id] = set(
                        articles.loc[mask, id_col].values
                    )

        # Pre-compute resolved frame column for each cascade
        cascade_frame_cols = {}
        for cascade in cascades:
            fc = FRAME_COLUMNS.get(cascade.frame)
            if fc:
                for variant in [f"{fc}_mean", f"{fc}_sum", fc]:
                    if variant in articles.columns:
                        cascade_frame_cols[cascade.cascade_id] = variant
                        break

        rows = []

        for cluster in clusters:
            daily_mass = self._build_daily_mass(cluster, articles)

            for cascade in cascades:
                # Temporal proximity
                proximity = self._temporal_proximity(
                    cluster.peak_date, cascade.peak_date
                )

                # Skip negligible pairs early
                if proximity < 0.01:
                    continue

                # Target: cascade daily_composite
                target = getattr(cascade, 'daily_composite', None)
                if target is None or (hasattr(target, 'empty') and target.empty):
                    continue

                # Ensure DatetimeIndex on target
                if not isinstance(target.index, pd.DatetimeIndex):
                    target = target.copy()
                    target.index = pd.DatetimeIndex(target.index)

                # Diff-in-diff with p-value
                did, did_p = self._diff_in_diff_with_pvalue(target, cluster.peak_date)
                did_norm = self._normalize_did(did)

                # Dose-response with p-value
                xcorr, lag, xcorr_p = self._dose_response_with_pvalue(daily_mass, target)

                # Granger causality (Phase 1 — conditional on min obs)
                common_len = len(daily_mass.index.intersection(target.index))
                granger_available = common_len >= P1_GRANGER_MIN_OBS
                if granger_available:
                    g_f, g_p = self._granger_causality(daily_mass, target)
                    # Check if Granger actually ran (not fallback)
                    granger_available = g_p < 1.0
                else:
                    g_f, g_p = 0.0, 1.0

                # Article overlap (actual shared articles, not just temporal)
                c_docs = cascade_doc_sets.get(cascade.cascade_id, set())
                overlap = self._compute_article_overlap(cluster, c_docs)

                # Impact score — conditional on Granger availability
                if granger_available:
                    granger_sig = max(0.0, 1.0 - g_p / GRANGER_P_THRESHOLD)
                    stat_score = (
                        P1_W_DID_G * did_norm
                        + P1_W_XCORR_G * abs(xcorr)
                        + P1_W_GRANGER_P1 * granger_sig
                        + P1_W_PROX_G * proximity
                    )
                else:
                    stat_score = (
                        P1_W_DID * did_norm
                        + P1_W_XCORR * abs(xcorr)
                        + P1_W_PROX * proximity
                    )

                effective_overlap = P1_OVERLAP_FLOOR + (1 - P1_OVERLAP_FLOOR) * overlap

                # Late support: is_post_peak and frame_affinity
                is_post_peak = cluster.peak_date > cascade.peak_date

                frame_affinity = 0.0
                frame_col_resolved = cascade_frame_cols.get(cascade.cascade_id)
                if frame_col_resolved and not articles.empty:
                    # Collect max belonging per doc_id across occurrences
                    doc_belongings = {}
                    for occ in cluster.occurrences:
                        for doc_id, bel in occ.belonging.items():
                            doc_belongings[doc_id] = max(
                                doc_belongings.get(doc_id, 0.0), bel
                            )
                    if doc_belongings:
                        bel_df = pd.DataFrame({
                            'doc_id': list(doc_belongings.keys()),
                            'belonging': list(doc_belongings.values()),
                        })
                        merged = bel_df.merge(
                            articles[['doc_id', frame_col_resolved]].drop_duplicates('doc_id'),
                            on='doc_id', how='inner',
                        )
                        if not merged.empty:
                            weights = merged['belonging'].values
                            vals = merged[frame_col_resolved].values.astype(float)
                            w_sum = weights.sum()
                            if w_sum > 0:
                                frame_affinity = float(
                                    (weights * vals).sum() / w_sum
                                )

                # Content relevance gate (embedding_alignment computed later)
                content_relevance = self._compute_content_relevance(
                    frame_affinity, np.nan  # alignment not yet available
                )

                impact = stat_score * cluster.strength * effective_overlap * content_relevance

                rows.append({
                    'cluster_id': cluster.cluster_id,
                    'cascade_id': cascade.cascade_id,
                    'cascade_frame': cascade.frame,
                    'cluster_strength': cluster.strength,
                    'proximity': round(proximity, 4),
                    'diff_in_diff': round(did, 6),
                    'dose_response_corr': round(xcorr, 4),
                    'dose_response_lag': lag,
                    'article_overlap': round(overlap, 4),
                    'impact_score': round(impact, 6),
                    'is_post_peak': is_post_peak,
                    'frame_affinity': round(frame_affinity, 6),
                    'did_p_value': round(did_p, 6),
                    'xcorr_p_value': round(xcorr_p, 6),
                    'granger_f': round(g_f, 4),
                    'granger_p': round(g_p, 4),
                    'content_relevance': round(content_relevance, 6),
                })

        if not rows:
            return pd.DataFrame(columns=[
                'cluster_id', 'cascade_id', 'cascade_frame', 'cluster_strength',
                'proximity', 'diff_in_diff', 'dose_response_corr',
                'dose_response_lag', 'article_overlap', 'impact_score',
                'is_post_peak', 'frame_affinity',
                'did_p_value', 'xcorr_p_value', 'granger_f', 'granger_p',
                'content_relevance',
                'role', 'impact_label', 'embedding_alignment',
                'perm_p_value', 'perm_p_adjusted', 'confidence',
            ])

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Phase 2: Cluster → Dominance
    # ------------------------------------------------------------------

    def phase2_cluster_dominance(
        self,
        clusters: list,
        paradigm_timeline: pd.DataFrame,
        articles: pd.DataFrame,
    ) -> pd.DataFrame:
        """Measure impact of EventClusters on paradigm dominance.

        Returns DataFrame with one row per (cluster_id, frame).
        Parallelized: each (cluster, frame) pair is computed independently.
        """
        if paradigm_timeline is None or paradigm_timeline.empty:
            return pd.DataFrame(columns=[
                'cluster_id', 'frame', 'cluster_strength',
                'diff_in_diff', 'dose_response_corr', 'dose_response_lag',
                'granger_f', 'granger_p', 'impact_score', 'role', 'impact_label',
            ])

        # Pre-compute dominance series per frame
        dom_cache = {}
        for frame in FRAMES:
            dom = self._get_dominance_series(paradigm_timeline, frame)
            if not dom.empty:
                dom_cache[frame] = dom

        # Pre-compute daily mass per cluster
        mass_cache = {}
        for cluster in clusters:
            mass_cache[cluster.cluster_id] = self._build_daily_mass(cluster, articles)

        def _compute_one(cluster, frame, daily_mass, dominance):
            did = self._diff_in_diff(dominance, cluster.peak_date)
            did_norm = self._normalize_did(did)
            xcorr, lag = self._dose_response(daily_mass, dominance)
            f_stat, p_val = self._granger_causality(daily_mass, dominance)
            granger_sig = max(0.0, 1.0 - p_val / GRANGER_P_THRESHOLD)
            impact = (
                P23_W_DID * did_norm
                + P23_W_XCORR * abs(xcorr)
                + P23_W_GRANGER * granger_sig
            ) * cluster.strength
            return {
                'cluster_id': cluster.cluster_id,
                'frame': frame,
                'cluster_strength': cluster.strength,
                'diff_in_diff': round(did, 6),
                'dose_response_corr': round(xcorr, 4),
                'dose_response_lag': lag,
                'granger_f': round(f_stat, 4),
                'granger_p': round(p_val, 4),
                'impact_score': round(impact, 6),
            }

        rows = []
        n_workers = min(8, len(clusters))
        if n_workers > 1 and len(clusters) * len(dom_cache) > 50:
            futures = {}
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                for cluster in clusters:
                    daily_mass = mass_cache[cluster.cluster_id]
                    for frame, dominance in dom_cache.items():
                        fut = executor.submit(
                            _compute_one, cluster, frame, daily_mass, dominance
                        )
                        futures[fut] = None
                for fut in as_completed(futures):
                    rows.append(fut.result())
        else:
            for cluster in clusters:
                daily_mass = mass_cache[cluster.cluster_id]
                for frame, dominance in dom_cache.items():
                    rows.append(_compute_one(cluster, frame, daily_mass, dominance))

        if not rows:
            return pd.DataFrame(columns=[
                'cluster_id', 'frame', 'cluster_strength',
                'diff_in_diff', 'dose_response_corr', 'dose_response_lag',
                'granger_f', 'granger_p', 'impact_score', 'role', 'impact_label',
            ])

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Phase 3: Cascade → Dominance
    # ------------------------------------------------------------------

    def phase3_cascade_dominance(
        self,
        cascades: list,
        paradigm_timeline: pd.DataFrame,
    ) -> pd.DataFrame:
        """Measure impact of Cascades on paradigm dominance.

        Returns DataFrame with one row per (cascade_id, frame).
        Parallelized: each (cascade, frame) pair is computed independently.
        """
        if paradigm_timeline is None or paradigm_timeline.empty:
            return pd.DataFrame(columns=[
                'cascade_id', 'frame', 'cascade_frame', 'cascade_score',
                'diff_in_diff', 'dose_response_corr', 'dose_response_lag',
                'granger_f', 'granger_p', 'impact_score',
                'role', 'impact_label', 'is_own_frame',
            ])

        # Pre-compute dominance series per frame
        dom_cache = {}
        for frame in FRAMES:
            dom = self._get_dominance_series(paradigm_timeline, frame)
            if not dom.empty:
                dom_cache[frame] = dom

        # Pre-process cascade sources
        cascade_sources = {}
        for cascade in cascades:
            source = getattr(cascade, 'daily_composite', None)
            if source is None or (hasattr(source, 'empty') and source.empty):
                continue
            if not isinstance(source.index, pd.DatetimeIndex):
                source = source.copy()
                source.index = pd.DatetimeIndex(source.index)
            cascade_sources[cascade.cascade_id] = source

        def _compute_one(cascade, frame, source, dominance):
            cascade_score = getattr(cascade, 'total_score', 0.0)
            did = self._diff_in_diff(dominance, cascade.peak_date)
            did_norm = self._normalize_did(did)
            xcorr, lag = self._dose_response(source, dominance)
            f_stat, p_val = self._granger_causality(source, dominance)
            granger_sig = max(0.0, 1.0 - p_val / GRANGER_P_THRESHOLD)
            impact = (
                P23_W_DID * did_norm
                + P23_W_XCORR * abs(xcorr)
                + P23_W_GRANGER * granger_sig
            ) * cascade_score
            return {
                'cascade_id': cascade.cascade_id,
                'frame': frame,
                'cascade_frame': cascade.frame,
                'cascade_score': cascade_score,
                'diff_in_diff': round(did, 6),
                'dose_response_corr': round(xcorr, 4),
                'dose_response_lag': lag,
                'granger_f': round(f_stat, 4),
                'granger_p': round(p_val, 4),
                'impact_score': round(impact, 6),
            }

        rows = []
        n_workers = min(8, len(cascades))
        if n_workers > 1 and len(cascade_sources) * len(dom_cache) > 50:
            futures = {}
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                for cascade in cascades:
                    source = cascade_sources.get(cascade.cascade_id)
                    if source is None:
                        continue
                    for frame, dominance in dom_cache.items():
                        fut = executor.submit(
                            _compute_one, cascade, frame, source, dominance
                        )
                        futures[fut] = None
                for fut in as_completed(futures):
                    rows.append(fut.result())
        else:
            for cascade in cascades:
                source = cascade_sources.get(cascade.cascade_id)
                if source is None:
                    continue
                for frame, dominance in dom_cache.items():
                    rows.append(_compute_one(cascade, frame, source, dominance))

        if not rows:
            return pd.DataFrame(columns=[
                'cascade_id', 'frame', 'cascade_frame', 'cascade_score',
                'diff_in_diff', 'dose_response_corr', 'dose_response_lag',
                'granger_f', 'granger_p', 'impact_score',
                'role', 'impact_label', 'is_own_frame',
            ])

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def run(self, results) -> UnifiedImpactResults:
        """Run all three phases from a DetectionResults object.

        Expects:
            results.event_clusters: List[EventCluster]
            results.cascades: List[CascadeResult]
            results.paradigm_shifts.paradigm_timeline: pd.DataFrame (optional)
            results._articles: pd.DataFrame
        """
        clusters = getattr(results, 'event_clusters', []) or []
        cascades = getattr(results, 'cascades', []) or []
        articles = getattr(results, '_articles', pd.DataFrame())

        # Get paradigm timeline if available
        paradigm_shifts = getattr(results, 'paradigm_shifts', None)
        paradigm_timeline = None
        if paradigm_shifts is not None:
            paradigm_timeline = getattr(paradigm_shifts, 'paradigm_timeline', None)

        return self.run_from_components(
            clusters=clusters,
            cascades=cascades,
            articles=articles,
            paradigm_timeline=paradigm_timeline,
        )

    def run_from_components(
        self,
        clusters: list,
        cascades: list,
        articles: pd.DataFrame,
        paradigm_timeline: Optional[pd.DataFrame] = None,
    ) -> UnifiedImpactResults:
        """Run all three phases from explicit components."""
        logger.info(
            f"Unified impact analysis: {len(clusters)} clusters, "
            f"{len(cascades)} cascades, "
            f"paradigm={'yes' if paradigm_timeline is not None and not paradigm_timeline.empty else 'no'}"
        )

        # Phase 1
        logger.info("  Phase 1: Cluster → Cascade...")
        df1 = self.phase1_cluster_cascade(clusters, cascades, articles)

        # Initial role assignment (needed by _compute_embedding_alignment to find drivers)
        df1 = self._assign_roles_phase1(df1)

        # Embedding alignment
        df1 = self._compute_embedding_alignment(df1, clusters, cascades)

        # Re-compute content_relevance now that embedding_alignment is available
        if not df1.empty and 'embedding_alignment' in df1.columns:
            updated_cr = []
            updated_impact = []
            for _, row in df1.iterrows():
                cr = self._compute_content_relevance(
                    row['frame_affinity'], row['embedding_alignment']
                )
                updated_cr.append(round(cr, 6))
                # Recompute impact with correct content_relevance
                did_norm = self._normalize_did(row['diff_in_diff'])
                granger_available = row.get('granger_p', 1.0) < 1.0
                if granger_available:
                    granger_sig = max(0.0, 1.0 - row['granger_p'] / GRANGER_P_THRESHOLD)
                    stat_score = (
                        P1_W_DID_G * did_norm
                        + P1_W_XCORR_G * abs(row['dose_response_corr'])
                        + P1_W_GRANGER_P1 * granger_sig
                        + P1_W_PROX_G * row['proximity']
                    )
                else:
                    stat_score = (
                        P1_W_DID * did_norm
                        + P1_W_XCORR * abs(row['dose_response_corr'])
                        + P1_W_PROX * row['proximity']
                    )
                eff_overlap = P1_OVERLAP_FLOOR + (1 - P1_OVERLAP_FLOOR) * row['article_overlap']
                updated_impact.append(round(stat_score * row['cluster_strength'] * eff_overlap * cr, 6))
            df1['content_relevance'] = updated_cr
            df1['impact_score'] = updated_impact

        # Permutation test
        if not df1.empty and len(df1) > 0:
            import time as _time
            n_cascades_with_target = sum(
                1 for c in cascades
                if getattr(c, 'daily_composite', None) is not None
            )
            logger.info(
                f"    Running permutation test ({N_PERMUTATIONS} perms, "
                f"{len(df1)} pairs, {n_cascades_with_target} cascades)..."
            )
            t0 = _time.time()
            raw_p, adj_p = self._compute_permutation_pvalues(
                df1, clusters, cascades, articles
            )
            logger.info(f"    Permutation test done in {_time.time() - t0:.1f}s")
            df1['perm_p_value'] = np.round(raw_p, 6)
            df1['perm_p_adjusted'] = np.round(adj_p, 6)
        else:
            df1['perm_p_value'] = pd.Series(dtype=float)
            df1['perm_p_adjusted'] = pd.Series(dtype=float)

        # Assign roles v2 (with statistical gates)
        df1 = self._assign_roles_phase1(df1)
        df1 = self._reclassify_late_support(df1)

        # Confidence scores
        if not df1.empty:
            df1['confidence'] = [
                round(self._compute_attribution_confidence(row), 6)
                for _, row in df1.iterrows()
            ]
        else:
            df1['confidence'] = pd.Series(dtype=float)

        n_sig1 = (df1['impact_score'] > 0.01).sum() if not df1.empty else 0
        n_late = (df1['role'] == 'late_support').sum() if not df1.empty else 0
        logger.info(f"    {len(df1)} pairs, {n_sig1} significant, {n_late} late_support")

        # Phase 2
        logger.info("  Phase 2: Cluster → Dominance...")
        df2 = self.phase2_cluster_dominance(clusters, paradigm_timeline, articles)
        df2 = self._assign_roles_phase2(df2)
        n_sig2 = (df2['impact_score'] > 0.01).sum() if not df2.empty else 0
        logger.info(f"    {len(df2)} pairs, {n_sig2} significant")

        # Phase 3
        logger.info("  Phase 3: Cascade → Dominance...")
        df3 = self.phase3_cascade_dominance(cascades, paradigm_timeline)
        df3 = self._assign_roles_phase3(df3)
        n_sig3 = (df3['impact_score'] > 0.01).sum() if not df3.empty else 0
        logger.info(f"    {len(df3)} pairs, {n_sig3} significant")

        # Cascade-level role aggregation
        cascade_roles = self._aggregate_cascade_roles(df3)

        # Role and label distributions
        def _role_counts(df, col='role'):
            if df.empty:
                return {}
            return dict(df[col].value_counts())

        def _label_counts(df):
            if df.empty:
                return {}
            return dict(df['impact_label'].value_counts())

        summary = {
            'n_clusters': len(clusters),
            'n_cascades': len(cascades),
            'has_paradigm_timeline': paradigm_timeline is not None and not paradigm_timeline.empty,
            'phase1_pairs': len(df1),
            'phase1_significant': int(n_sig1),
            'phase2_pairs': len(df2),
            'phase2_significant': int(n_sig2),
            'phase3_pairs': len(df3),
            'phase3_significant': int(n_sig3),
            'phase1_roles': _role_counts(df1),
            'phase2_roles': _role_counts(df2),
            'phase3_roles': _role_counts(df3),
            'phase3_cascade_roles': cascade_roles,
            'phase1_impact_labels': _label_counts(df1),
            'phase2_impact_labels': _label_counts(df2),
            'phase3_impact_labels': _label_counts(df3),
        }

        return UnifiedImpactResults(
            cluster_cascade=df1,
            cluster_dominance=df2,
            cascade_dominance=df3,
            summary=summary,
        )
