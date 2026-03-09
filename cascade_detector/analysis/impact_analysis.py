"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
impact_analysis.py

MAIN OBJECTIVE:
---------------
Post-processing analyzer: which events and messengers are most associated
with triggering and amplifying media cascades?

Takes DetectionResults + article DataFrame. Does not modify detection/scoring.
Computes 3 metrics per (annotation, frame) pair:
  1. Prevalence Ratio — over-representation during cascade vs baseline
  2. Pre-Onset Surge — elevated rates in 7 days before cascade onset
  3. Strength Correlation — Spearman correlation with cascade total_score

Multiple comparisons corrected with Benjamini-Hochberg FDR (alpha=0.05).

Author:
-------
Antoine Lemor
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from cascade_detector.core.constants import (
    EVENT_COLUMNS, SOLUTION_COLUMNS, MESSENGERS, FRAME_COLUMNS
)

logger = logging.getLogger(__name__)


class EventImpactAnalyzer:
    """Post-processing analyzer for event/messenger impact on cascades.

    Takes DetectionResults + article DataFrame. Does not modify detection/scoring.
    Computes 3 metrics: prevalence ratios, pre-onset surge, strength correlations.
    """

    def __init__(self, results, articles: pd.DataFrame):
        """Initialize with detection results and article-level data.

        Args:
            results: DetectionResults from pipeline.
            articles: Article-level DataFrame (from processor.aggregate_by_article).
        """
        self.cascades = results.cascades
        self.articles = articles.copy()

        # Resolve date column
        if 'date_converted_first' in self.articles.columns:
            self.date_col = 'date_converted_first'
        elif 'date_converted' in self.articles.columns:
            self.date_col = 'date_converted'
        else:
            self.date_col = 'date'

        # Ensure datetime
        self.articles[self.date_col] = pd.to_datetime(self.articles[self.date_col])

        # Resolve annotation columns (handle _sum/_mean suffixes from aggregation)
        self.annotation_cols = {}  # col_name -> (display_name, type)
        for col in EVENT_COLUMNS:
            resolved = self._resolve_col(col)
            if resolved:
                self.annotation_cols[resolved] = (col, 'event')
        for col in SOLUTION_COLUMNS:
            resolved = self._resolve_col(col)
            if resolved:
                self.annotation_cols[resolved] = (col, 'solution')
        for col in MESSENGERS:
            resolved = self._resolve_col(col)
            if resolved:
                self.annotation_cols[resolved] = (col, 'messenger')

        logger.info(
            f"EventImpactAnalyzer initialized: {len(self.cascades)} cascades, "
            f"{len(self.articles)} articles, {len(self.annotation_cols)} annotation columns"
        )

    def _resolve_col(self, col: str) -> Optional[str]:
        """Find the actual column name in articles (handles _sum suffix)."""
        if col in self.articles.columns:
            return col
        if f'{col}_sum' in self.articles.columns:
            return f'{col}_sum'
        return None

    def _get_frame_col(self, frame: str) -> Optional[str]:
        """Get the frame column name for a given frame abbreviation."""
        frame_col = FRAME_COLUMNS.get(frame)
        if frame_col is None:
            return None
        # Handle aggregated column names
        if frame_col in self.articles.columns:
            return frame_col
        if f'{frame_col}_sum' in self.articles.columns:
            return f'{frame_col}_sum'
        return None

    def _tag_cascade_periods(self, frame: str) -> pd.Series:
        """Return boolean mask on articles: True if within any cascade window of this frame."""
        mask = pd.Series(False, index=self.articles.index)
        frame_cascades = [c for c in self.cascades if c.frame == frame]

        for c in frame_cascades:
            onset = pd.Timestamp(c.onset_date)
            end = pd.Timestamp(c.end_date)
            in_window = (
                (self.articles[self.date_col] >= onset) &
                (self.articles[self.date_col] <= end)
            )
            mask = mask | in_window

        return mask

    @staticmethod
    def _benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Benjamini-Hochberg FDR correction.

        Args:
            p_values: Array of raw p-values.
            alpha: FDR level (not used in adjustment, kept for API clarity).

        Returns:
            Array of adjusted p-values.
        """
        p_values = np.asarray(p_values, dtype=float)
        n = len(p_values)
        if n == 0:
            return np.array([])

        # Handle NaN: keep them as NaN
        valid = ~np.isnan(p_values)
        if not valid.any():
            return p_values.copy()

        adjusted = np.full(n, np.nan)
        valid_p = p_values[valid]
        m = len(valid_p)

        # Sort
        sorted_idx = np.argsort(valid_p)
        sorted_p = valid_p[sorted_idx]

        # Adjust: p_adj[i] = min(p[i] * m / rank, 1.0), enforcing monotonicity
        ranks = np.arange(1, m + 1)
        adjusted_sorted = np.minimum(sorted_p * m / ranks, 1.0)

        # Enforce monotonicity (from largest to smallest)
        for i in range(m - 2, -1, -1):
            adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

        # Unsort
        result = np.empty(m)
        result[sorted_idx] = adjusted_sorted

        adjusted[valid] = result
        return adjusted

    def compute_prevalence_ratios(self) -> pd.DataFrame:
        """Compute prevalence ratios for each (annotation, frame) pair.

        For each event/messenger and frame: how much more frequently does it
        appear during cascade periods vs non-cascade periods?

        Returns:
            DataFrame with columns: frame, annotation, type, cascade_rate,
            baseline_rate, prevalence_ratio, odds_ratio, ci_low, ci_high,
            p_value, p_value_adjusted, n_cascade, n_baseline
        """
        rows = []

        frames_with_cascades = set(c.frame for c in self.cascades)

        for frame in frames_with_cascades:
            cascade_mask = self._tag_cascade_periods(frame)

            # Filter to articles relevant to this frame
            frame_col = self._get_frame_col(frame)
            if frame_col is not None:
                frame_articles = self.articles[self.articles[frame_col] > 0]
                cascade_in_frame = cascade_mask[frame_articles.index] if len(frame_articles) > 0 else pd.Series(dtype=bool)
            else:
                frame_articles = self.articles
                cascade_in_frame = cascade_mask

            cascade_articles = frame_articles[cascade_in_frame]
            baseline_articles = frame_articles[~cascade_in_frame]

            n_cascade = len(cascade_articles)
            n_baseline = len(baseline_articles)

            if n_cascade == 0 or n_baseline == 0:
                continue

            for col, (display_name, ann_type) in self.annotation_cols.items():
                # Count articles with annotation > 0
                a = int((cascade_articles[col] > 0).sum())    # cascade + annotation
                b = int(n_cascade - a)                         # cascade + no annotation
                c_count = int((baseline_articles[col] > 0).sum())  # baseline + annotation
                d = int(n_baseline - c_count)                  # baseline + no annotation

                cascade_rate = a / n_cascade if n_cascade > 0 else 0
                baseline_rate = c_count / n_baseline if n_baseline > 0 else 0

                # Prevalence ratio
                if baseline_rate > 0:
                    prevalence_ratio = cascade_rate / baseline_rate
                else:
                    prevalence_ratio = float('inf') if cascade_rate > 0 else 1.0

                # Fisher's exact test (2x2 contingency table)
                table = np.array([[a, b], [c_count, d]])
                try:
                    odds_ratio_result = stats.fisher_exact(table)
                    odds_ratio = odds_ratio_result[0]
                    p_value = odds_ratio_result[1]
                except Exception:
                    odds_ratio = np.nan
                    p_value = np.nan

                # 95% CI for odds ratio (Woolf's method)
                ci_low, ci_high = self._odds_ratio_ci(a, b, c_count, d)

                rows.append({
                    'frame': frame,
                    'annotation': display_name,
                    'type': ann_type,
                    'cascade_rate': cascade_rate,
                    'baseline_rate': baseline_rate,
                    'prevalence_ratio': prevalence_ratio,
                    'odds_ratio': odds_ratio,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'p_value': p_value,
                    'n_cascade': n_cascade,
                    'n_baseline': n_baseline,
                })

        if not rows:
            return pd.DataFrame(columns=[
                'frame', 'annotation', 'type', 'cascade_rate', 'baseline_rate',
                'prevalence_ratio', 'odds_ratio', 'ci_low', 'ci_high',
                'p_value', 'p_value_adjusted', 'n_cascade', 'n_baseline'
            ])

        df = pd.DataFrame(rows)

        # BH correction
        df['p_value_adjusted'] = self._benjamini_hochberg(df['p_value'].values)

        return df

    @staticmethod
    def _odds_ratio_ci(a: int, b: int, c: int, d: int,
                       alpha: float = 0.05) -> tuple:
        """Compute 95% CI for odds ratio using Woolf's log method."""
        # Add 0.5 continuity correction if any cell is 0
        if any(x == 0 for x in [a, b, c, d]):
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

        try:
            log_or = np.log(a * d / (b * c))
            se = np.sqrt(1/a + 1/b + 1/c + 1/d)
            z = stats.norm.ppf(1 - alpha / 2)
            ci_low = np.exp(log_or - z * se)
            ci_high = np.exp(log_or + z * se)
            return ci_low, ci_high
        except (ValueError, ZeroDivisionError):
            return np.nan, np.nan

    def compute_pre_onset_surge(self, pre_days: int = 7,
                                baseline_days: int = 30) -> pd.DataFrame:
        """Compute pre-onset surge for each (annotation, frame) pair.

        Does a given event appear at elevated rates in the pre_days before
        cascade onset vs its baseline rate in the preceding baseline_days?

        Args:
            pre_days: Number of days before onset for the test window (default 7).
            baseline_days: Number of days before the test window for baseline (default 30).

        Returns:
            DataFrame with columns: frame, annotation, type, n_cascades,
            median_surge, mean_surge, p_value, p_value_adjusted
        """
        rows = []

        frames_with_cascades = set(c.frame for c in self.cascades)

        for frame in frames_with_cascades:
            frame_cascades = [c for c in self.cascades if c.frame == frame]

            for col, (display_name, ann_type) in self.annotation_cols.items():
                surges = []

                for cascade in frame_cascades:
                    onset = pd.Timestamp(cascade.onset_date)

                    # Pre-onset window: [onset - pre_days, onset - 1]
                    pre_start = onset - pd.Timedelta(days=pre_days)
                    pre_end = onset - pd.Timedelta(days=1)

                    # Baseline window: [onset - pre_days - baseline_days, onset - pre_days - 1]
                    bl_start = pre_start - pd.Timedelta(days=baseline_days)
                    bl_end = pre_start - pd.Timedelta(days=1)

                    pre_mask = (
                        (self.articles[self.date_col] >= pre_start) &
                        (self.articles[self.date_col] <= pre_end)
                    )
                    bl_mask = (
                        (self.articles[self.date_col] >= bl_start) &
                        (self.articles[self.date_col] <= bl_end)
                    )

                    pre_articles = self.articles[pre_mask]
                    bl_articles = self.articles[bl_mask]

                    if len(pre_articles) == 0 or len(bl_articles) == 0:
                        continue

                    # Rate = proportion of articles with annotation > 0
                    pre_rate = (pre_articles[col] > 0).mean()
                    bl_rate = (bl_articles[col] > 0).mean()

                    if bl_rate > 0:
                        surge = pre_rate / bl_rate
                    else:
                        # If baseline rate is 0 and pre-onset has any, surge is infinite
                        # Skip to avoid skewing statistics
                        if pre_rate > 0:
                            surge = float('inf')
                        else:
                            surge = 1.0

                    if np.isfinite(surge):
                        surges.append(surge)

                if len(surges) < 2:
                    # Need at least 2 observations for Wilcoxon
                    rows.append({
                        'frame': frame,
                        'annotation': display_name,
                        'type': ann_type,
                        'n_cascades': len(surges),
                        'median_surge': np.median(surges) if surges else np.nan,
                        'mean_surge': np.mean(surges) if surges else np.nan,
                        'p_value': np.nan,
                    })
                    continue

                surges_arr = np.array(surges)
                median_surge = float(np.median(surges_arr))
                mean_surge = float(np.mean(surges_arr))

                # Wilcoxon signed-rank test: is surge systematically > 1.0?
                try:
                    stat, p_value = stats.wilcoxon(
                        surges_arr - 1.0,
                        alternative='greater'
                    )
                except ValueError:
                    # All values identical
                    p_value = np.nan

                rows.append({
                    'frame': frame,
                    'annotation': display_name,
                    'type': ann_type,
                    'n_cascades': len(surges),
                    'median_surge': median_surge,
                    'mean_surge': mean_surge,
                    'p_value': p_value,
                })

        if not rows:
            return pd.DataFrame(columns=[
                'frame', 'annotation', 'type', 'n_cascades',
                'median_surge', 'mean_surge', 'p_value', 'p_value_adjusted'
            ])

        df = pd.DataFrame(rows)
        df['p_value_adjusted'] = self._benjamini_hochberg(df['p_value'].values)

        return df

    def compute_strength_correlations(self) -> pd.DataFrame:
        """Compute Spearman correlations between annotation prevalence and cascade strength.

        Across cascades of a given frame, is event/messenger prevalence
        correlated with cascade total_score?

        Returns:
            DataFrame with columns: frame, annotation, type, n_cascades,
            spearman_rho, p_value, p_value_adjusted
        """
        rows = []

        frames_with_cascades = set(c.frame for c in self.cascades)

        for frame in frames_with_cascades:
            frame_cascades = [c for c in self.cascades if c.frame == frame]

            if len(frame_cascades) < 3:
                # Need at least 3 cascades for meaningful correlation
                continue

            for col, (display_name, ann_type) in self.annotation_cols.items():
                prevalences = []
                scores = []

                for cascade in frame_cascades:
                    onset = pd.Timestamp(cascade.onset_date)
                    end = pd.Timestamp(cascade.end_date)

                    mask = (
                        (self.articles[self.date_col] >= onset) &
                        (self.articles[self.date_col] <= end)
                    )
                    cascade_articles = self.articles[mask]

                    if len(cascade_articles) == 0:
                        continue

                    prevalence = (cascade_articles[col] > 0).mean()
                    prevalences.append(prevalence)
                    scores.append(cascade.total_score)

                if len(prevalences) < 3:
                    continue

                # Spearman rank correlation
                try:
                    rho, p_value = stats.spearmanr(prevalences, scores)
                except Exception:
                    rho, p_value = np.nan, np.nan

                rows.append({
                    'frame': frame,
                    'annotation': display_name,
                    'type': ann_type,
                    'n_cascades': len(prevalences),
                    'spearman_rho': rho,
                    'p_value': p_value,
                })

        if not rows:
            return pd.DataFrame(columns=[
                'frame', 'annotation', 'type', 'n_cascades',
                'spearman_rho', 'p_value', 'p_value_adjusted'
            ])

        df = pd.DataFrame(rows)
        df['p_value_adjusted'] = self._benjamini_hochberg(df['p_value'].values)

        return df

    def _build_summary(self, prevalence_df: pd.DataFrame,
                       surge_df: pd.DataFrame,
                       correlation_df: pd.DataFrame) -> pd.DataFrame:
        """Build a summary table combining all three metrics.

        Merges on (frame, annotation, type) and ranks by combined evidence.
        """
        if prevalence_df.empty and surge_df.empty and correlation_df.empty:
            return pd.DataFrame(columns=[
                'frame', 'annotation', 'type',
                'prevalence_ratio', 'prevalence_p_adj',
                'median_surge', 'surge_p_adj',
                'spearman_rho', 'correlation_p_adj',
                'n_significant_metrics',
            ])

        merge_keys = ['frame', 'annotation', 'type']

        # Prepare subsets for merge
        prev_cols = prevalence_df[merge_keys + [
            'prevalence_ratio', 'p_value_adjusted'
        ]].rename(columns={'p_value_adjusted': 'prevalence_p_adj'}) if not prevalence_df.empty else pd.DataFrame(columns=merge_keys + ['prevalence_ratio', 'prevalence_p_adj'])

        surge_cols = surge_df[merge_keys + [
            'median_surge', 'p_value_adjusted'
        ]].rename(columns={'p_value_adjusted': 'surge_p_adj'}) if not surge_df.empty else pd.DataFrame(columns=merge_keys + ['median_surge', 'surge_p_adj'])

        corr_cols = correlation_df[merge_keys + [
            'spearman_rho', 'p_value_adjusted'
        ]].rename(columns={'p_value_adjusted': 'correlation_p_adj'}) if not correlation_df.empty else pd.DataFrame(columns=merge_keys + ['spearman_rho', 'correlation_p_adj'])

        # Merge
        summary = prev_cols.merge(surge_cols, on=merge_keys, how='outer')
        summary = summary.merge(corr_cols, on=merge_keys, how='outer')

        # Count significant metrics (p_adj < 0.05)
        sig_count = (
            (summary['prevalence_p_adj'].astype(float).fillna(1.0) < 0.05).astype(int) +
            (summary['surge_p_adj'].astype(float).fillna(1.0) < 0.05).astype(int) +
            (summary['correlation_p_adj'].astype(float).fillna(1.0) < 0.05).astype(int)
        )
        summary['n_significant_metrics'] = sig_count

        # Sort by number of significant metrics (desc), then prevalence ratio
        summary = summary.sort_values(
            ['n_significant_metrics', 'prevalence_ratio'],
            ascending=[False, False]
        ).reset_index(drop=True)

        return summary

    def run(self) -> Dict[str, Any]:
        """Run all three impact analyses.

        Returns:
            Dictionary with keys:
                'prevalence_ratios': DataFrame
                'pre_onset_surge': DataFrame
                'strength_correlations': DataFrame
                'summary': DataFrame
        """
        if not self.cascades:
            logger.info("No cascades to analyze — skipping impact analysis")
            empty_prev = pd.DataFrame(columns=[
                'frame', 'annotation', 'type', 'cascade_rate', 'baseline_rate',
                'prevalence_ratio', 'odds_ratio', 'ci_low', 'ci_high',
                'p_value', 'p_value_adjusted', 'n_cascade', 'n_baseline'
            ])
            empty_surge = pd.DataFrame(columns=[
                'frame', 'annotation', 'type', 'n_cascades',
                'median_surge', 'mean_surge', 'p_value', 'p_value_adjusted'
            ])
            empty_corr = pd.DataFrame(columns=[
                'frame', 'annotation', 'type', 'n_cascades',
                'spearman_rho', 'p_value', 'p_value_adjusted'
            ])
            return {
                'prevalence_ratios': empty_prev,
                'pre_onset_surge': empty_surge,
                'strength_correlations': empty_corr,
                'summary': self._build_summary(empty_prev, empty_surge, empty_corr),
            }

        logger.info("Computing prevalence ratios...")
        prevalence_df = self.compute_prevalence_ratios()
        logger.info(f"  {len(prevalence_df)} (annotation, frame) pairs")

        logger.info("Computing pre-onset surge...")
        surge_df = self.compute_pre_onset_surge()
        logger.info(f"  {len(surge_df)} (annotation, frame) pairs")

        logger.info("Computing strength correlations...")
        corr_df = self.compute_strength_correlations()
        logger.info(f"  {len(corr_df)} (annotation, frame) pairs")

        summary = self._build_summary(prevalence_df, surge_df, corr_df)

        n_sig = (summary['n_significant_metrics'] > 0).sum() if not summary.empty else 0
        logger.info(f"Impact analysis complete: {n_sig} annotations significant on >= 1 metric")

        return {
            'prevalence_ratios': prevalence_df,
            'pre_onset_surge': surge_df,
            'strength_correlations': corr_df,
            'summary': summary,
        }
