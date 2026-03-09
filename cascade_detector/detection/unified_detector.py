"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
unified_detector.py

MAIN OBJECTIVE:
---------------
Unified multi-signal cascade detector. Replaces the sequential
BurstDetector → CascadeScorer pipeline with a single step where ALL
metrics (temporal + participation + convergence + source) are used
simultaneously to:

1. Detect WHETHER a cascade exists (multiple signals converge)
2. Determine WHEN it starts/ends (boundaries from composite evidence)
3. Score its strength (same 4 dimensions, coherent framework)

Detection is performed via PELT changepoint detection on a composite
signal (weighted sum of 5 daily Z-scores), validated by binomial
proportion tests, then each detected period is scored with full
sub-indices including network metrics.

Author:
-------
Antoine Lemor
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats as scipy_stats

from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.models import BurstResult, CascadeResult
from cascade_detector.core.constants import (
    FRAME_COLUMNS, EMOTION_COLUMNS, CASCADE_THRESHOLDS,
    EVENT_COLUMNS, MESSENGERS, FRAMES,
)
from cascade_detector.detection.signal_builder import DailySignalBuilder

logger = logging.getLogger(__name__)


class UnifiedCascadeDetector:
    """Multi-signal cascade detector: detect + score in one step.

    Replaces both BurstDetector and CascadeScorer. Detection uses a composite
    signal (4 daily Z-scores), scoring uses 4 dimensions with sub-indices.

    Normalization constants:
    - MAX_BURST_INTENSITY (5.0): 5x baseline is exceptional for frame proportions
    - MAX_ADOPTION_VELOCITY (3.0): 3 new journalists/day is very fast adoption
    - MAX_DURATION_DAYS (30.0): cascades rarely exceed 30 days
    - MAX_MEDIA (20.0): Canadian media landscape has ~20 major outlets
    """

    MAX_BURST_INTENSITY = 5.0
    MAX_ADOPTION_VELOCITY = 3.0
    MAX_DURATION_DAYS = 30.0
    MAX_MEDIA = 20.0
    MIN_ARTICLES_HARD = 3

    DATE_COLS = ['date', 'date_converted_first', 'date_converted']
    AUTHOR_COLS = ['author', 'author_first', 'author_clean_first', 'author_clean']
    MEDIA_COLS = ['media', 'media_first']

    def __init__(self, config: Optional[DetectorConfig] = None,
                 embedding_store=None):
        self.config = config or DetectorConfig()
        if embedding_store is None:
            raise ValueError(
                "embedding_store is required. Embeddings are mandatory for "
                "cascade detection. Provide an EmbeddingStore instance."
            )
        self.embedding_store = embedding_store
        self._cascade_counter = 0
        self._cascade_lock = threading.Lock()
        self.signal_builder = DailySignalBuilder(self.config, embedding_store=embedding_store)

        from cascade_detector.embeddings.semantic_convergence import SemanticConvergenceCalculator
        self._convergence_calculator = SemanticConvergenceCalculator(self.embedding_store)
        logger.info("UnifiedCascadeDetector: embedding-based convergence enabled")

    # =========================================================================
    # Public API
    # =========================================================================

    def detect_all_frames(
        self,
        temporal_index: Dict[str, Any],
        articles: pd.DataFrame,
        indices: Dict[str, Any],
    ) -> Tuple[List[CascadeResult], List[BurstResult], Dict[str, Dict[str, pd.Series]]]:
        """Detect and score cascades across all configured frames in parallel.

        Uses ThreadPoolExecutor(max_workers=4) to process frames concurrently.
        Thread safety is ensured by _cascade_lock on the shared counter.

        Args:
            temporal_index: Output of TemporalIndexer.build_index().
            articles: Article-level DataFrame.
            indices: All built indices.

        Returns:
            Tuple of:
            - List[CascadeResult]: scored cascades for all frames
            - List[BurstResult]: raw burst detections for all frames
            - Dict[str, Dict[str, pd.Series]]: per-frame signal data
        """
        all_cascades = []
        all_bursts = []
        all_signals = {}

        def _detect_frame(frame):
            if frame not in temporal_index:
                logger.warning(f"Frame '{frame}' not in temporal index, skipping.")
                return [], [], {}
            return self.detect(frame, temporal_index, articles, indices)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_detect_frame, f): f for f in self.config.frames}
            for future in as_completed(futures):
                frame = futures[future]
                cascades, bursts, signals = future.result()
                all_cascades.extend(cascades)
                all_bursts.extend(bursts)
                if signals:
                    all_signals[frame] = signals
                if cascades:
                    n_scored = len([c for c in cascades if c.total_score > 0])
                    logger.info(f"  {frame}: {len(bursts)} burst(s), {n_scored} scored cascade(s)")

        return all_cascades, all_bursts, all_signals

    def detect(
        self,
        frame: str,
        temporal_index: Dict[str, Any],
        articles: pd.DataFrame,
        indices: Dict[str, Any],
    ) -> Tuple[List[CascadeResult], List[BurstResult], Dict[str, pd.Series]]:
        """Detect and score cascades for a single frame.

        Args:
            frame: Frame short name (e.g. 'Pol').
            temporal_index: Output of TemporalIndexer.build_index().
            articles: Article-level DataFrame.
            indices: All built indices.

        Returns:
            Tuple of:
            - List[CascadeResult]: scored cascades
            - List[BurstResult]: raw burst detections
            - Dict[str, pd.Series]: signal data (z_temporal, z_participation, etc.)
        """
        # Build multi-signal time series
        signals = self.signal_builder.build_signals(frame, temporal_index, articles, indices)
        if not signals:
            return [], [], {}

        composite = signals['composite']

        # Detect burst periods from composite signal
        bursts = self._detect_from_composite(composite, frame, temporal_index)
        if not bursts:
            return [], bursts, signals

        # Score each burst
        cascades = []
        for burst in bursts:
            cascade = self._score_cascade(burst, signals, articles, indices, temporal_index)
            cascades.append(cascade)

        return cascades, bursts, signals

    # =========================================================================
    # Detection from composite signal
    # =========================================================================

    def _detect_from_composite(
        self,
        composite: pd.Series,
        frame: str,
        temporal_index: Dict[str, Any],
    ) -> List[BurstResult]:
        """PELT changepoint detection pipeline with proportion-test validation.

        6-step pipeline:
        1. PELT on smoothed composite → elevated segments
        2. Proportion test filter → kept / rejected
        3. Refine rejected segments (fine PELT + proptest)
        4. Sliding-window proportion complement (multi-scale)
        5. Union + merge + re-proptest
        6. Boundary extension + re-merge
        """
        if len(composite) < self.config.baseline_window_days + 10:
            return []

        # Extract proportion, count and total series from temporal index
        frame_data = temporal_index.get(frame, {})
        prop_series = frame_data.get('daily_proportions')
        count_series = frame_data.get('daily_series', frame_data.get('daily_counts'))
        total_series = frame_data.get('daily_totals')
        if prop_series is None or count_series is None or total_series is None:
            return []

        # Baseline = year-level proportion (volume-weighted, matching sandbox DB query)
        total_count = float(count_series.sum())
        total_total = float(total_series.sum())
        baseline_prop = total_count / total_total if total_total > 0 else 0.0
        if baseline_prop <= 0:
            return []

        # Step 1: PELT on smoothed composite → elevated segments
        pelt_segments = self._pelt_segments(composite)

        # Step 2: Proportion test filter
        kept = []
        rejected = []
        for seg in pelt_segments:
            accepted, stats = self._proptest_accept(
                count_series, total_series, seg['start'], seg['end'], baseline_prop
            )
            seg_copy = {**seg, **stats}
            if accepted:
                kept.append(seg_copy)
            else:
                rejected.append(seg_copy)

        # Step 3: Refine rejected segments (fine PELT + proptest)
        refined = self._refine_rejected(
            rejected, composite, count_series, total_series, baseline_prop
        )

        # Step 4: Sliding-window proportion complement
        sw_periods = self._sliding_window_proptest(
            prop_series, count_series, total_series, baseline_prop
        )

        # Step 5: Union + merge + re-proptest
        all_periods = kept + refined + sw_periods
        all_periods = self._merge_periods(all_periods)

        # Re-validate merged periods
        validated = []
        for period in all_periods:
            accepted, stats = self._proptest_accept(
                count_series, total_series, period['start'], period['end'], baseline_prop
            )
            if accepted:
                validated.append({**period, **stats})
        all_periods = validated

        # Step 6: Boundary extension + re-merge
        all_periods = self._extend_boundaries(all_periods, prop_series, baseline_prop)
        all_periods = self._merge_periods(all_periods)

        # Re-annotate after final merge
        final = []
        for period in all_periods:
            _, stats = self._proptest_accept(
                count_series, total_series, period['start'], period['end'], baseline_prop
            )
            final.append({**period, **stats})

        # Build BurstResult objects
        bursts = []
        for period in final:
            start = pd.Timestamp(period['start'])
            end = pd.Timestamp(period['end'])
            duration = (end - start).days + 1
            if duration < self.config.min_burst_days:
                continue

            # Get composite values in window
            mask = (composite.index >= start) & (composite.index <= end)
            window_composite = composite[mask]
            if len(window_composite) == 0:
                continue

            peak_date = window_composite.idxmax()
            if pd.isna(peak_date):
                peak_date = start + (end - start) / 2  # midpoint fallback

            # Proportion stats
            prop_mask = (prop_series.index >= start) & (prop_series.index <= end)
            window_props = prop_series[prop_mask]
            peak_proportion = float(window_props.max()) if len(window_props) > 0 else 0.0
            window_prop = period.get('window_prop', float(window_props.mean()) if len(window_props) > 0 else 0.0)
            intensity = peak_proportion / baseline_prop if baseline_prop > 0 else 0.0

            bursts.append(BurstResult(
                frame=frame,
                onset_date=start,
                peak_date=peak_date,
                end_date=end,
                duration_days=duration,
                intensity=intensity,
                baseline_mean=baseline_prop,
                peak_proportion=peak_proportion,
                detection_method=period.get('method', 'pelt'),
                cohen_h=period.get('cohen_h', 0.0),
                proptest_pvalue=period.get('pvalue', 1.0),
                window_prop=window_prop,
            ))

        return bursts

    def _pelt_segments(self, composite: pd.Series) -> List[Dict[str, Any]]:
        """Run PELT on smoothed composite, return elevated segments.

        Smooths with rolling mean, runs PELT changepoint detection,
        then keeps segments whose mean exceeds mean + threshold × std.
        Adjacent elevated segments (gap ≤ 1 day) are merged before return.
        """
        cfg = self.config
        # Smooth
        smoothed = composite.rolling(
            window=cfg.pelt_smoothing_window, center=True, min_periods=1
        ).mean()
        signal = smoothed.values.reshape(-1, 1)

        if len(signal) < 2 * cfg.pelt_min_size:
            return []

        # PELT detection
        algo = rpt.Pelt(model="rbf", min_size=cfg.pelt_min_size).fit(signal)
        try:
            breakpoints = algo.predict(pen=cfg.pelt_penalty)
        except Exception:
            return []

        # Convert breakpoints to segments
        dates = composite.index
        mean_val = float(smoothed.mean())
        std_val = float(smoothed.std())
        if std_val == 0:
            return []
        threshold = mean_val + cfg.pelt_elevation_threshold * std_val

        segments = []
        prev = 0
        for bp in breakpoints:
            seg_vals = smoothed.iloc[prev:bp]
            if float(seg_vals.mean()) > threshold:
                segments.append({
                    'start': dates[prev],
                    'end': dates[min(bp - 1, len(dates) - 1)],
                    'method': 'pelt',
                })
            prev = bp

        # Merge adjacent elevated segments (gap ≤ 1 day)
        if not segments:
            return []
        merged = [segments[0].copy()]
        for seg in segments[1:]:
            gap = (seg['start'] - merged[-1]['end']).days
            if gap <= 1:
                merged[-1]['end'] = seg['end']
            else:
                merged.append(seg.copy())

        return merged

    def _proportion_test(
        self,
        count_series: pd.Series,
        total_series: pd.Series,
        start: pd.Timestamp,
        end: pd.Timestamp,
        baseline_prop: float,
    ) -> Dict[str, float]:
        """Binomial proportion test on in-memory count/total Series.

        Returns dict with pvalue, cohen_h, ratio, window_prop.
        """
        mask = (count_series.index >= start) & (count_series.index <= end)
        k = float(count_series[mask].sum())
        n = float(total_series[mask].sum())

        if n < 1:
            return {'pvalue': 1.0, 'cohen_h': 0.0, 'ratio': 0.0, 'window_prop': 0.0}

        observed_prop = k / n
        ratio = observed_prop / baseline_prop if baseline_prop > 0 else 0.0

        # Cohen's h: 2 * (arcsin(sqrt(p_obs)) - arcsin(sqrt(p_base)))
        cohen_h = 2.0 * (np.arcsin(np.sqrt(np.clip(observed_prop, 0, 1)))
                         - np.arcsin(np.sqrt(np.clip(baseline_prop, 0, 1))))

        # Binomial test: observed count vs expected under baseline
        try:
            result = scipy_stats.binomtest(int(round(k)), int(round(n)), baseline_prop, alternative='greater')
            pvalue = float(result.pvalue)
        except (ValueError, OverflowError):
            pvalue = 1.0

        return {
            'pvalue': pvalue,
            'cohen_h': float(cohen_h),
            'ratio': float(ratio),
            'window_prop': float(observed_prop),
        }

    def _proptest_accept(
        self,
        count_series: pd.Series,
        total_series: pd.Series,
        start: pd.Timestamp,
        end: pd.Timestamp,
        baseline_prop: float,
    ) -> Tuple[bool, Dict[str, float]]:
        """Accept criterion: p < alpha AND (h > h_min OR ratio > r_min)."""
        stats = self._proportion_test(count_series, total_series, start, end, baseline_prop)
        cfg = self.config
        accepted = (
            stats['pvalue'] < cfg.proptest_alpha
            and (stats['cohen_h'] > cfg.proptest_min_cohen_h
                 or stats['ratio'] > cfg.proptest_min_ratio)
        )
        return accepted, stats

    def _refine_rejected(
        self,
        rejected: List[Dict],
        composite: pd.Series,
        count_series: pd.Series,
        total_series: pd.Series,
        baseline_prop: float,
    ) -> List[Dict[str, Any]]:
        """Fine PELT (pen=1, smoothing=3) on rejected segments to recover sub-bursts."""
        refined = []
        for seg in rejected:
            start = pd.Timestamp(seg['start'])
            end = pd.Timestamp(seg['end'])
            dur = (end - start).days + 1
            if dur < 5:
                continue

            mask = (composite.index >= start) & (composite.index <= end)
            sub = composite[mask]
            if len(sub) < 5:
                continue

            # Fine PELT with adaptive min_size
            smoothed = sub.rolling(window=3, center=True, min_periods=1).mean()
            signal = smoothed.values.reshape(-1, 1)
            min_sz = min(3, len(signal) // 2)
            if min_sz < 2:
                continue
            algo = rpt.Pelt(model="rbf", min_size=min_sz).fit(signal)
            try:
                breakpoints = algo.predict(pen=1.0)
            except Exception:
                continue

            mean_val = float(smoothed.mean())
            std_val = float(smoothed.std())
            if std_val == 0:
                continue
            threshold = mean_val + 0.3 * std_val

            dates = sub.index
            sub_segments = []
            prev = 0
            for bp in breakpoints:
                seg_vals = smoothed.iloc[prev:bp]
                if float(seg_vals.mean()) > threshold:
                    sub_segments.append({
                        'start': dates[prev],
                        'end': dates[min(bp - 1, len(dates) - 1)],
                        'method': 'pelt_refined',
                    })
                prev = bp

            # Merge adjacent sub-segments (gap ≤ 1 day) before proptest
            if not sub_segments:
                continue
            merged = [sub_segments[0].copy()]
            for ss in sub_segments[1:]:
                gap = (pd.Timestamp(ss['start']) - pd.Timestamp(merged[-1]['end'])).days
                if gap <= 1:
                    merged[-1]['end'] = ss['end']
                else:
                    merged.append(ss.copy())

            # Proptest each merged sub-segment
            for ms in merged:
                accepted, stats = self._proptest_accept(
                    count_series, total_series, ms['start'], ms['end'], baseline_prop
                )
                if accepted:
                    refined.append({**ms, **stats})

        return refined

    def _sliding_window_proptest(
        self,
        prop_series: pd.Series,
        count_series: pd.Series,
        total_series: pd.Series,
        baseline_prop: float,
    ) -> List[Dict[str, Any]]:
        """Multi-scale (5-28 day) sliding-window proportion test.

        For each window size, slide across dates and collect (start, end)
        intervals where the proptest is significant. Merge overlapping
        intervals, then re-test each merged interval. If a merged interval
        is too diluted, fall back to the best sub-window.
        """
        cfg = self.config
        window_sizes = [5, 7, 10, 14, 21, 28]
        significant = []

        for ws in window_sizes:
            rolling_count = count_series.rolling(window=ws, min_periods=ws).sum()
            rolling_total = total_series.rolling(window=ws, min_periods=ws).sum()

            for i in range(ws - 1, len(count_series)):
                rc = rolling_count.iloc[i]
                rt = rolling_total.iloc[i]
                if rt < 1 or np.isnan(rc) or np.isnan(rt):
                    continue
                obs_prop = rc / rt
                if obs_prop <= baseline_prop:
                    continue

                cohen_h = 2.0 * (np.arcsin(np.sqrt(np.clip(obs_prop, 0, 1)))
                                 - np.arcsin(np.sqrt(np.clip(baseline_prop, 0, 1))))

                try:
                    result = scipy_stats.binomtest(
                        int(round(rc)), int(round(rt)), baseline_prop, alternative='greater'
                    )
                    pval = float(result.pvalue)
                except (ValueError, OverflowError):
                    continue

                ratio = obs_prop / baseline_prop if baseline_prop > 0 else 0.0
                if pval < cfg.proptest_alpha and (cohen_h > cfg.proptest_min_cohen_h
                                                   or ratio > cfg.proptest_min_ratio):
                    e = count_series.index[i]
                    s = count_series.index[i - ws + 1]
                    significant.append((s, e))

        if not significant:
            return []

        # Merge overlapping intervals
        significant.sort()
        merged_intervals = [significant[0]]
        for s, e in significant[1:]:
            prev_s, prev_e = merged_intervals[-1]
            if s <= prev_e:  # overlapping
                merged_intervals[-1] = (prev_s, max(prev_e, e))
            else:
                merged_intervals.append((s, e))

        # For each merged interval: accept if proptest passes,
        # otherwise fall back to best sub-window
        kept = []
        for ms, me in merged_intervals:
            accepted, stats = self._proptest_accept(
                count_series, total_series, ms, me, baseline_prop
            )
            if accepted:
                kept.append({'start': ms, 'end': me, 'method': 'sliding_prop', **stats})
            else:
                # Merged interval too diluted — find best sub-window
                best_h = -1.0
                best_window = None
                for s, e in significant:
                    if s >= ms and e <= me:
                        _, sub_stats = self._proptest_accept(
                            count_series, total_series, s, e, baseline_prop
                        )
                        if sub_stats['cohen_h'] > best_h:
                            best_h = sub_stats['cohen_h']
                            best_window = (s, e, sub_stats)
                if best_window:
                    bs, be, bstats = best_window
                    kept.append({'start': bs, 'end': be, 'method': 'sliding_prop', **bstats})

        return kept

    def _extend_boundaries(
        self,
        periods: List[Dict],
        prop_series: pd.Series,
        baseline_prop: float,
    ) -> List[Dict]:
        """Walk boundaries while 5-day smooth proportion > baseline."""
        cfg = self.config
        smoothed = prop_series.rolling(window=5, center=True, min_periods=1).mean()
        result = []

        for period in periods:
            start = pd.Timestamp(period['start'])
            end = pd.Timestamp(period['end'])

            # Extend start backward
            new_start = start
            gap_count = 0
            for i in range(1, cfg.boundary_max_lookback + 1):
                candidate = start - pd.Timedelta(days=i)
                if candidate not in smoothed.index:
                    break
                if smoothed[candidate] > baseline_prop:
                    new_start = candidate
                    gap_count = 0
                else:
                    gap_count += 1
                    if gap_count > cfg.boundary_gap_tolerance:
                        break

            # Extend end forward
            new_end = end
            gap_count = 0
            for i in range(1, cfg.boundary_max_lookback + 1):
                candidate = end + pd.Timedelta(days=i)
                if candidate not in smoothed.index:
                    break
                if smoothed[candidate] > baseline_prop:
                    new_end = candidate
                    gap_count = 0
                else:
                    gap_count += 1
                    if gap_count > cfg.boundary_gap_tolerance:
                        break

            p = period.copy()
            p['start'] = new_start
            p['end'] = new_end
            result.append(p)

        return result

    def _merge_periods(self, periods: List[Dict]) -> List[Dict]:
        """Merge overlapping or adjacent periods (gap <= burst_merge_gap_days)."""
        if not periods:
            return []

        gap = self.config.burst_merge_gap_days
        sorted_periods = sorted(periods, key=lambda p: p['start'])
        merged = [sorted_periods[0].copy()]

        for period in sorted_periods[1:]:
            last = merged[-1]
            gap_days = (period['start'] - last['end']).days
            if gap_days <= gap:
                last['end'] = max(last['end'], period['end'])
                # Keep the stronger method label
                method_priority = {'pelt': 2, 'pelt_refined': 1, 'sliding_prop': 0}
                if method_priority.get(period.get('method', ''), 0) > method_priority.get(last.get('method', ''), 0):
                    last['method'] = period['method']
            else:
                merged.append(period.copy())

        return merged

    # =========================================================================
    # Scoring
    # =========================================================================

    def _score_cascade(
        self,
        burst: BurstResult,
        signals: Dict[str, pd.Series],
        articles: pd.DataFrame,
        indices: Dict[str, Any],
        temporal_index: Dict[str, Any],
    ) -> CascadeResult:
        """Score a burst into a CascadeResult with continuous 4-dimension metrics."""
        with self._cascade_lock:
            self._cascade_counter += 1
            counter_val = self._cascade_counter
        cascade_id = f"{burst.frame}_{burst.onset_date.strftime('%Y%m%d')}_{counter_val}"

        # Resolve column names
        date_col = self._resolve_col(articles, self.DATE_COLS)
        author_col = self._resolve_col(articles, self.AUTHOR_COLS)
        media_col = self._resolve_col(articles, self.MEDIA_COLS)

        if date_col is None:
            logger.warning(f"No date column found in articles.")
            return self._empty_result(cascade_id, burst, signals)

        # Extract articles in burst window using this frame
        burst_articles = self._get_burst_articles(burst, articles, date_col)
        if burst_articles.empty:
            return self._empty_result(cascade_id, burst, signals)

        # Semantic peak: embedding-weighted P50 replaces z-score argmax
        semantic_peak, daily_semantic_mass = self._compute_semantic_peak(
            burst, burst_articles, date_col
        )

        # Participation metrics
        n_articles = len(burst_articles)
        journalists = burst_articles[author_col].dropna().unique() if author_col else []
        media_outlets = burst_articles[media_col].dropna().unique() if media_col else []
        n_journalists = len(journalists)
        n_media = len(media_outlets)

        # Hard filter
        if n_articles < self.MIN_ARTICLES_HARD or n_journalists == 0:
            return self._empty_result(cascade_id, burst, signals)

        # New journalists
        n_new_journalists, _ = self._count_new_journalists(
            burst, articles, journalists, date_col, author_col
        )

        # Daily time series
        daily_articles, daily_journalists, cumulative_journalists = \
            self._build_daily_series(burst, burst_articles, date_col, author_col)

        # Adoption velocity (uses semantic peak for growth window)
        adoption_velocity = self._compute_adoption_velocity(
            burst, cumulative_journalists, peak_date=semantic_peak
        )

        # Context
        top_journalists = self._top_counts(burst_articles, author_col, n=10) if author_col else []
        top_media = self._top_counts(burst_articles, media_col, n=10) if media_col else []
        dominant_events = self._extract_dominant_events(burst_articles)
        dominant_messengers = self._extract_dominant_messengers(burst_articles)

        # Mann-Whitney U test (on daily proportions, not raw counts)
        mann_whitney_p = self._mann_whitney_test(burst, temporal_index)

        # Composite peak and daily composite for this cascade
        composite = signals.get('composite')
        composite_peak = 0.0
        daily_composite = None
        daily_signals = None
        if composite is not None:
            cmask = (composite.index >= burst.onset_date) & (composite.index <= burst.end_date)
            daily_composite = composite[cmask]
            composite_peak = float(daily_composite.max()) if len(daily_composite) > 0 else 0.0
            # Per-dimension signals for this cascade window
            daily_signals = {}
            for key in ['z_temporal', 'z_participation', 'z_convergence', 'z_source', 'z_semantic']:
                if key in signals:
                    daily_signals[key] = signals[key][cmask]

        cascade = CascadeResult(
            cascade_id=cascade_id,
            frame=burst.frame,
            onset_date=burst.onset_date,
            peak_date=semantic_peak,
            end_date=burst.end_date,
            duration_days=burst.duration_days,
            n_articles=n_articles,
            n_journalists=n_journalists,
            n_media=n_media,
            n_new_journalists=n_new_journalists,
            burst_intensity=burst.intensity,
            adoption_velocity=adoption_velocity,
            baseline_mean=burst.baseline_mean,
            peak_proportion=burst.peak_proportion,
            daily_articles=daily_articles,
            daily_journalists=daily_journalists,
            cumulative_journalists=cumulative_journalists,
            top_journalists=top_journalists,
            top_media=top_media,
            dominant_events=dominant_events,
            dominant_messengers=dominant_messengers,
            mann_whitney_p=mann_whitney_p,
            composite_peak=composite_peak,
            daily_composite=daily_semantic_mass if daily_semantic_mass is not None else daily_composite,
            daily_signals=daily_signals,
            detection_method=burst.detection_method,
        )

        # Build network
        network = self._build_network(cascade, articles)

        # Score 4 dimensions
        source_index = indices.get('sources')
        sub_temporal = self._score_temporal(cascade)
        sub_participation = self._score_participation(cascade, network, daily_articles, cumulative_journalists)
        sub_convergence = self._score_convergence(cascade, articles, source_index)
        sub_source = self._score_source(cascade, burst_articles, source_index, author_col, media_col)

        score_temporal = float(np.clip(np.mean(list(sub_temporal.values())), 0, 1))
        score_participation = float(np.clip(np.mean(list(sub_participation.values())), 0, 1))
        score_convergence = float(np.clip(np.mean(list(sub_convergence.values())), 0, 1))
        score_source = float(np.clip(np.mean(list(sub_source.values())), 0, 1))

        # Base weighted score
        base_score = (
            self.config.weight_temporal * score_temporal
            + self.config.weight_participation * score_participation
            + self.config.weight_convergence * score_convergence
            + self.config.weight_source * score_source
        )
        # Media confidence factor: cascades on few outlets get discounted
        # Reaches 1.0 at 10 media (log2(10)/log2(10) = 1)
        media_confidence = min(1.0, np.log2(max(n_media, 1)) / np.log2(10))
        total_score = float(np.clip(base_score * media_confidence, 0, 1))

        classification = self._classify(total_score)

        cascade.score_temporal = score_temporal
        cascade.score_participation = score_participation
        cascade.score_convergence = score_convergence
        cascade.score_source = score_source
        cascade.total_score = total_score
        cascade.classification = classification

        cascade.sub_indices = {
            **{f"temporal_{k}": v for k, v in sub_temporal.items()},
            **{f"participation_{k}": v for k, v in sub_participation.items()},
            **{f"convergence_{k}": v for k, v in sub_convergence.items()},
            **{f"source_{k}": v for k, v in sub_source.items()},
            'media_confidence': float(media_confidence),
        }

        # Network metrics
        cascade.network_density = network.get('density', 0.0)
        cascade.network_modularity = network.get('modularity', 0.0)
        cascade.network_mean_degree = network.get('mean_degree', 0.0)
        cascade.network_n_components = network.get('n_components', 0)
        cascade.network_avg_clustering = network.get('avg_clustering', 0.0)

        # Store full network edge list for export
        # Nodes are (journalist, media) tuples — keep them as tuples
        graph = network.get('graph')
        if graph is not None:
            cascade.network_edges = [
                (u, v, float(d.get('weight', 1.0)))
                for u, v, d in graph.edges(data=True)
            ]

        logger.info(
            f"  Scored {cascade_id}: "
            f"T={score_temporal:.3f} P={score_participation:.3f} "
            f"C={score_convergence:.3f} S={score_source:.3f} "
            f"=> {total_score:.3f} ({classification})"
        )

        return cascade

    # =========================================================================
    # Article extraction
    # =========================================================================

    @staticmethod
    def _resolve_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for col in candidates:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _weighted_percentile(values: np.ndarray, weights: np.ndarray,
                             p: float) -> float:
        """Compute weighted percentile with midpoint-centered interpolation.

        Args:
            values: Array of values.
            weights: Array of weights (same length).
            p: Percentile in [0, 100].

        Returns:
            Weighted percentile value.
        """
        if len(values) == 0:
            return 0.0
        if len(values) == 1:
            return float(values[0])

        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[sorted_idx]

        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights[-1]
        if total_weight < 1e-10:
            return float(np.percentile(values, p))

        cum_frac = (cum_weights - 0.5 * sorted_weights) / total_weight
        target = p / 100.0
        return float(np.interp(target, cum_frac, sorted_values))

    def _compute_semantic_peak(
        self,
        burst: BurstResult,
        burst_articles: pd.DataFrame,
        date_col: str,
    ) -> Tuple[pd.Timestamp, Optional[pd.Series]]:
        """Compute embedding-weighted peak date and daily mass for a burst.

        For each burst article, weight = frame_signal × cosine_sim(article, centroid).
        The centroid is the frame_signal-weighted mean of article embeddings.
        Peak date = weighted P50 of article dates.
        Daily mass = sum of weights per day.

        Returns:
            (semantic_peak_date, daily_semantic_mass) or
            (burst.peak_date, None) on fallback.
        """
        frame_col = FRAME_COLUMNS.get(burst.frame)
        if not frame_col:
            return burst.peak_date, None

        # Resolve frame signal column: _mean > _sum > binary
        signal_col = None
        for candidate in [f"{frame_col}_mean", f"{frame_col}_sum", frame_col]:
            if candidate in burst_articles.columns:
                signal_col = candidate
                break
        if signal_col is None:
            return burst.peak_date, None

        # Get doc_ids and frame signals
        if 'doc_id' not in burst_articles.columns:
            return burst.peak_date, None

        doc_ids = burst_articles['doc_id'].tolist()
        frame_signals = burst_articles[signal_col].values.astype(float)

        # Get embeddings
        embeddings, found_ids = self.embedding_store.get_batch_article_embeddings(doc_ids)
        if len(found_ids) < 2:
            return burst.peak_date, None

        # Build mapping from doc_id to index in burst_articles
        doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}
        found_mask = np.array([doc_id_to_idx[d] for d in found_ids])
        found_signals = np.maximum(frame_signals[found_mask], 0.0)

        # Compute centroid: frame_signal-weighted mean, L2-normalized
        signal_sum = found_signals.sum()
        if signal_sum < 1e-10:
            return burst.peak_date, None

        centroid = (embeddings * found_signals[:, np.newaxis]).sum(axis=0) / signal_sum
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm < 1e-10:
            return burst.peak_date, None
        centroid /= centroid_norm

        # Cosine similarity of each article embedding with centroid
        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_norms = np.maximum(emb_norms, 1e-10)
        cos_sims = np.maximum(0.0, (embeddings / emb_norms) @ centroid)

        # Per-article weight = frame_signal × cosine_similarity
        weights = found_signals * cos_sims

        if weights.sum() < 1e-10:
            return burst.peak_date, None

        # Get dates for found articles
        found_dates = pd.to_datetime(
            burst_articles.iloc[found_mask][date_col], errors='coerce'
        ).dt.normalize()

        # Daily semantic mass = sum of weights per day
        date_range = pd.date_range(burst.onset_date, burst.end_date, freq='D')
        daily_mass = pd.Series(0.0, index=date_range)
        for date_val, w in zip(found_dates, weights):
            if date_val in daily_mass.index:
                daily_mass[date_val] += w

        # Weighted P50 on daily masses (not per-article) so that days with
        # many moderate-weight articles outweigh days with few heavy articles.
        nonzero = daily_mass[daily_mass > 0]
        if nonzero.empty:
            return burst.peak_date, daily_mass

        day_ordinals = nonzero.index.values.astype('datetime64[D]').astype(float)
        day_weights = nonzero.values.astype(float)
        peak_ordinal = self._weighted_percentile(day_ordinals, day_weights, 50)
        semantic_peak = pd.Timestamp(np.datetime64(int(peak_ordinal), 'D'))
        # Clamp to burst window
        semantic_peak = max(burst.onset_date, min(semantic_peak, burst.end_date))

        return semantic_peak, daily_mass

    def _get_burst_articles(self, burst: BurstResult,
                            articles: pd.DataFrame,
                            date_col: str) -> pd.DataFrame:
        """Get articles in burst window using the burst frame."""
        dates = pd.to_datetime(articles[date_col], errors='coerce')
        mask = (dates >= burst.onset_date) & (dates <= burst.end_date)

        frame_col = FRAME_COLUMNS.get(burst.frame)
        if frame_col:
            for col in [frame_col, f"{frame_col}_sum", f"{frame_col}_mean"]:
                if col in articles.columns:
                    mask = mask & (articles[col] > 0)
                    break

        return articles[mask].copy()

    def _count_new_journalists(self, burst, articles, burst_journalists,
                                date_col, author_col):
        if not author_col:
            return 0, set()

        baseline_start = burst.onset_date - pd.Timedelta(days=self.config.baseline_window_days)
        baseline_end = burst.onset_date - pd.Timedelta(days=1)

        dates = pd.to_datetime(articles[date_col], errors='coerce')
        mask = (dates >= baseline_start) & (dates <= baseline_end)

        frame_col = FRAME_COLUMNS.get(burst.frame)
        if frame_col:
            for col in [frame_col, f"{frame_col}_sum", f"{frame_col}_mean"]:
                if col in articles.columns:
                    mask = mask & (articles[col] > 0)
                    break

        baseline_articles = articles[mask]
        baseline_journalists = set(baseline_articles[author_col].dropna().unique())
        burst_journalist_set = set(burst_journalists)

        new_journalists = burst_journalist_set - baseline_journalists
        return len(new_journalists), baseline_journalists

    # =========================================================================
    # Daily time series
    # =========================================================================

    def _build_daily_series(self, burst, burst_articles, date_col, author_col):
        date_range = pd.date_range(burst.onset_date, burst.end_date, freq='D')
        daily_articles = pd.Series(0, index=date_range, dtype=int)
        daily_journalists = pd.Series(0, index=date_range, dtype=int)
        cumulative_journalists = pd.Series(0, index=date_range, dtype=int)

        if burst_articles.empty:
            return daily_articles, daily_journalists, cumulative_journalists

        dates = pd.to_datetime(burst_articles[date_col], errors='coerce')
        burst_articles = burst_articles.copy()
        burst_articles['_date'] = dates.dt.normalize()

        seen_journalists = set()

        for date in date_range:
            day_articles = burst_articles[burst_articles['_date'] == date]
            daily_articles[date] = len(day_articles)

            if author_col and author_col in day_articles.columns:
                day_journalists = set(day_articles[author_col].dropna().unique())
                daily_journalists[date] = len(day_journalists)
                seen_journalists |= day_journalists

            cumulative_journalists[date] = len(seen_journalists)

        return daily_articles, daily_journalists, cumulative_journalists

    def _compute_adoption_velocity(self, burst, cumulative_journalists, peak_date=None):
        if cumulative_journalists.empty:
            return 0.0

        _peak = peak_date if peak_date is not None else burst.peak_date
        growth_mask = cumulative_journalists.index <= _peak
        growth = cumulative_journalists[growth_mask]

        if len(growth) < 2:
            return 0.0

        growth_days = (growth.index[-1] - growth.index[0]).days
        if growth_days <= 0:
            return 0.0

        return (growth.iloc[-1] - growth.iloc[0]) / growth_days

    # =========================================================================
    # Network building
    # =========================================================================

    def _build_network(self, cascade, articles) -> Dict[str, Any]:
        from cascade_detector.detection.network_builder import NetworkBuilder
        builder = NetworkBuilder()
        return builder.build(cascade, articles)

    # =========================================================================
    # Dimension 1: Temporal Dynamics (0.30)
    # =========================================================================

    def _score_temporal(self, cascade: CascadeResult) -> Dict[str, float]:
        burst_intensity = min(1.0, cascade.burst_intensity / self.MAX_BURST_INTENSITY)
        adoption_velocity = min(1.0, cascade.adoption_velocity / self.MAX_ADOPTION_VELOCITY)
        duration = min(1.0, cascade.duration_days / self.MAX_DURATION_DAYS)
        # Significance-gated scoring: credit only statistically significant results
        if cascade.mann_whitney_p < 0.01:
            mann_whitney = 1.0
        elif cascade.mann_whitney_p < 0.05:
            mann_whitney = 0.5
        elif cascade.mann_whitney_p < 0.10:
            mann_whitney = 0.25
        else:
            mann_whitney = 0.0

        return {
            'burst_intensity': burst_intensity,
            'adoption_velocity': adoption_velocity,
            'duration': duration,
            'mann_whitney': mann_whitney,
        }

    # =========================================================================
    # Dimension 2: Participation Breadth (0.25) — now includes network
    # =========================================================================

    def _score_participation(self, cascade, network, daily_articles, cumulative_journalists):
        actor_diversity = self._actor_diversity(cascade)
        cross_media_ratio = min(1.0, cascade.n_media / self.MAX_MEDIA)
        new_entrant_rate = (min(1.0, cascade.n_new_journalists / cascade.n_journalists)
                           if cascade.n_journalists > 0 else 0.0)
        growth_pattern = self._growth_pattern(cumulative_journalists)

        # Network sub-indices
        # Normalized clustering: ratio of observed clustering to random-graph
        # expectation (= density). Scale-free: comparable across network sizes.
        avg_clustering = float(network.get('avg_clustering', 0.0))
        density = float(network.get('density', 0.0))
        n_nodes = max(float(network.get('n_nodes', cascade.n_journalists)), 2.0)
        if density > 0 and n_nodes >= 5:
            clustering_ratio = avg_clustering / density
            # Normalize: ratio of 20x random = 1.0 (max)
            network_structure = min(1.0, clustering_ratio / 20.0)
        elif n_nodes < 5:
            # Too few nodes for meaningful structure — neutral score
            network_structure = 0.5
        else:
            network_structure = 0.0

        modularity = float(network.get('modularity', 0.0))
        network_cohesion = max(0.0, min(1.0, 1.0 - modularity))

        return {
            'actor_diversity': actor_diversity,
            'cross_media_ratio': cross_media_ratio,
            'new_entrant_rate': new_entrant_rate,
            'growth_pattern': growth_pattern,
            'network_structure': network_structure,
            'network_cohesion': network_cohesion,
        }

    def _actor_diversity(self, cascade):
        if not cascade.top_journalists:
            return 0.0

        counts = np.array([count for _, count in cascade.top_journalists], dtype=float)
        if len(counts) <= 1:
            return 0.0

        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(counts))
        if max_entropy == 0:
            return 0.0

        return min(1.0, entropy / max_entropy)

    def _growth_pattern(self, cumulative_journalists):
        """Fraction of days with new journalist adoption (monotonic growth ratio).

        Measures cascade propagation breadth over time. A value of 1.0 means
        every day brought at least one new journalist; 0.0 means no growth.
        """
        if cumulative_journalists is None or len(cumulative_journalists) < 2:
            return 0.0
        values = cumulative_journalists.values
        n = len(values)
        increasing_days = sum(1 for i in range(1, n) if values[i] > values[i - 1])
        return increasing_days / (n - 1)

    # =========================================================================
    # Dimension 3: Content Convergence (0.25)
    # =========================================================================

    def _score_convergence(self, cascade, articles=None, source_index=None):
        """Score convergence dimension using embedding-based metrics.

        Always uses SemanticConvergenceCalculator. Returns 0-valued sub-indices
        when insufficient data is available (< 2 articles or no doc_ids).
        """
        zero_result = {
            'semantic_similarity': 0.0,
            'convergence_trend': 0.0,
            'cross_media_alignment': 0.0,
            'novelty_decay': 0.0,
        }

        if articles is None:
            return zero_result

        cascade_articles = self._get_cascade_articles(cascade, articles)
        if cascade_articles.empty or 'doc_id' not in cascade_articles.columns:
            return zero_result

        article_ids = cascade_articles['doc_id'].unique().tolist()
        if len(article_ids) < 2:
            return zero_result

        # Check embedding coverage before computing convergence
        _, found_ids = self.embedding_store.get_batch_article_embeddings(article_ids)
        n_found = len(found_ids)
        coverage = n_found / len(article_ids)
        if coverage < 1.0:
            logger.info(
                f"Convergence embedding coverage for {cascade.frame} "
                f"({cascade.onset_date.strftime('%Y-%m-%d')}): "
                f"{n_found}/{len(article_ids)} ({coverage:.0%})"
            )
        if n_found < 2:
            logger.warning(
                f"Convergence: <2 embeddings found for {cascade.frame} "
                f"({cascade.onset_date.strftime('%Y-%m-%d')}), returning zeros"
            )
            return zero_result

        article_dates = {}
        article_media = {}

        if source_index:
            profiles = source_index.get('article_profiles', {})
            for doc_id in article_ids:
                profile = profiles.get(doc_id, {})
                if profile.get('date'):
                    article_dates[doc_id] = profile['date']
                if profile.get('media'):
                    article_media[doc_id] = profile['media']

        if not article_dates:
            date_col = self._resolve_col(cascade_articles, self.DATE_COLS)
            media_col = self._resolve_col(cascade_articles, self.MEDIA_COLS)
            if date_col:
                for _, row in cascade_articles.drop_duplicates('doc_id').iterrows():
                    doc_id = row['doc_id']
                    date = pd.to_datetime(row[date_col], errors='coerce')
                    if pd.notna(date):
                        article_dates[doc_id] = date.to_pydatetime()
                    if media_col and pd.notna(row.get(media_col)):
                        article_media[doc_id] = str(row[media_col])

        metrics = self._convergence_calculator.compute_all_metrics(
            article_ids=article_ids,
            article_dates=article_dates if article_dates else None,
            article_media=article_media if article_media else None,
        )

        raw_semantic = float(np.clip(metrics.get('intra_window_similarity', 0.0), 0, 1))
        syndication_ratio = float(metrics.get('syndication_ratio', 0.0))
        # Penalize semantic similarity by syndication: shared wire copy inflates
        # similarity without reflecting independent editorial convergence
        semantic_similarity = raw_semantic * (1.0 - 0.5 * syndication_ratio)
        raw_trend = metrics.get('convergence_trend_slope', 0.0)
        convergence_trend = float(np.clip((raw_trend + 0.05) / 0.10, 0, 1))
        cross_media_alignment = float(np.clip(metrics.get('cross_media_alignment', 0.0), 0, 1))
        raw_decay = metrics.get('novelty_decay_rate', 0.0)
        # Size-invariant normalization: multiply per-article slope by n_articles
        # to get total novelty drop over full cascade (not biased by cascade size)
        n_articles = max(len(article_ids), 1)
        total_decay = raw_decay * n_articles
        novelty_decay = float(np.clip(total_decay / 0.5, 0, 1))

        cascade.semantic_similarity = semantic_similarity
        cascade.convergence_trend = raw_trend
        cascade.cross_media_alignment = cross_media_alignment
        cascade.novelty_decay_rate = raw_decay
        cascade.convergence_metrics_full = {k: float(v) for k, v in metrics.items()}

        return {
            'semantic_similarity': semantic_similarity,
            'convergence_trend': convergence_trend,
            'cross_media_alignment': cross_media_alignment,
            'novelty_decay': novelty_decay,
        }

    # =========================================================================
    # Dimension 4: Source Convergence (0.20)
    # =========================================================================

    def _score_source(self, cascade, burst_articles, source_index, author_col, media_col):
        diversity_decline = self._source_diversity_decline(
            cascade, burst_articles, author_col, media_col, source_index
        )
        concentration = self._messenger_concentration(burst_articles, source_index)
        coordination = self._media_coordination(
            cascade, burst_articles, author_col, source_index
        )

        cascade.source_diversity_decline = diversity_decline
        cascade.messenger_concentration = concentration
        cascade.media_coordination = coordination

        return {
            'source_diversity_decline': diversity_decline,
            'messenger_concentration': concentration,
            'media_coordination': coordination,
        }

    def _source_diversity_decline(self, cascade, burst_articles, author_col, media_col, source_index):
        if burst_articles.empty:
            return 0.0

        date_col = self._resolve_col(burst_articles, self.DATE_COLS)
        if not date_col:
            return 0.0

        dates = pd.to_datetime(burst_articles[date_col], errors='coerce')
        midpoint = cascade.onset_date + (cascade.end_date - cascade.onset_date) / 2

        first_half = burst_articles[dates <= midpoint]
        second_half = burst_articles[dates > midpoint]

        if first_half.empty or second_half.empty:
            return 0.0

        entropy_first = self._messenger_entropy(first_half)
        entropy_second = self._messenger_entropy(second_half)

        if entropy_first == 0:
            return 0.0

        decline = (entropy_first - entropy_second) / entropy_first

        # Log-scale damping: small cascades have unreliable entropy estimates.
        # log2(3)/log2(100) ≈ 0.24, log2(30)/log2(100) ≈ 0.74, log2(100)/log2(100) = 1.0
        n = max(len(burst_articles), 1)
        confidence = np.log2(n) / np.log2(100)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return float(np.clip(decline * confidence, 0.0, 1.0))

    def _messenger_concentration(self, burst_articles, source_index):
        if burst_articles.empty:
            return 0.0
        return float(np.clip(1.0 - self._messenger_entropy(burst_articles), 0.0, 1.0))

    def _messenger_entropy(self, df):
        msg_counts = []
        for msg_col in MESSENGERS:
            for col in [msg_col, f"{msg_col}_sum", f"{msg_col}_mean"]:
                if col in df.columns:
                    msg_counts.append(float(df[col].sum()))
                    break

        if not msg_counts or sum(msg_counts) == 0:
            return 0.5

        counts = np.array(msg_counts, dtype=float)
        total = counts.sum()
        if total == 0:
            return 0.5

        probs = counts / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(msg_counts))
        if max_entropy == 0:
            return 0.5

        return float(entropy / max_entropy)

    def _media_coordination(self, cascade, burst_articles, author_col, source_index):
        """Embedding-based media coordination: pairwise cosine similarity
        between journalist centroids (mean article embeddings per journalist).

        Measures whether journalists are writing semantically similar content,
        which is a stronger signal than messenger-profile similarity.
        """
        if burst_articles.empty or not author_col:
            return 0.0

        if 'doc_id' not in burst_articles.columns:
            return 0.0

        journalists = burst_articles[author_col].dropna().unique()
        if len(journalists) < 2:
            return 0.0

        # Compute centroid for each journalist
        centroids = []
        n_journalists_skipped = 0
        for journalist in journalists:
            j_articles = burst_articles[burst_articles[author_col] == journalist]
            doc_ids = j_articles['doc_id'].dropna().unique().tolist()
            if not doc_ids:
                n_journalists_skipped += 1
                continue
            embeddings, found = self.embedding_store.get_batch_article_embeddings(doc_ids)
            if len(embeddings) == 0:
                n_journalists_skipped += 1
                continue
            centroids.append(embeddings.mean(axis=0))

        if n_journalists_skipped > 0:
            logger.info(
                f"Media coordination: {n_journalists_skipped}/{len(journalists)} "
                f"journalists skipped (no embeddings found)"
            )

        if len(centroids) < 2:
            return 0.0

        # Pairwise cosine similarity between journalist centroids
        centroids_arr = np.array(centroids, dtype=np.float32)
        norms = np.linalg.norm(centroids_arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = centroids_arr / norms
        sim_matrix = normalized @ normalized.T

        n = len(centroids)
        upper = np.triu_indices(n, k=1)
        mean_sim = float(np.mean(sim_matrix[upper]))

        return float(np.clip(mean_sim, 0.0, 1.0))

    # =========================================================================
    # Statistical test
    # =========================================================================

    def _mann_whitney_test(self, burst, temporal_index):
        """Mann-Whitney U test on daily frame proportions (burst vs baseline).

        Tests whether daily proportions during the burst window are
        significantly higher than during the baseline window. Using
        proportions (not raw article counts) controls for variation in
        total daily media volume.
        """
        frame_data = temporal_index.get(burst.frame, {})
        daily_props = frame_data.get('daily_proportions')
        if daily_props is None or daily_props.empty:
            return 1.0

        # Burst window proportions
        burst_mask = (daily_props.index >= burst.onset_date) & (daily_props.index <= burst.end_date)
        burst_values = daily_props[burst_mask].values

        # Baseline window proportions
        baseline_start = burst.onset_date - pd.Timedelta(days=self.config.baseline_window_days)
        baseline_end = burst.onset_date - pd.Timedelta(days=1)
        baseline_mask = (daily_props.index >= baseline_start) & (daily_props.index <= baseline_end)
        baseline_values = daily_props[baseline_mask].values

        if len(burst_values) < 3 or len(baseline_values) < 5:
            return 1.0

        try:
            _, p_value = scipy_stats.mannwhitneyu(
                burst_values, baseline_values,
                alternative='greater'
            )
            return float(p_value)
        except ValueError:
            return 1.0

    # =========================================================================
    # Context extraction
    # =========================================================================

    def _extract_dominant_events(self, df):
        result = {}
        for col in EVENT_COLUMNS:
            if col in df.columns:
                count = int((df[col] > 0).sum())
                if count > 0:
                    result[col] = count
            for suffix in ['_sum', '_mean']:
                full_col = f"{col}{suffix}"
                if full_col in df.columns:
                    count = int((df[full_col] > 0).sum())
                    if count > 0:
                        result[col] = count
                    break
        return result

    def _extract_dominant_messengers(self, df):
        result = {}
        for col in MESSENGERS:
            if col in df.columns:
                count = int((df[col] > 0).sum())
                if count > 0:
                    result[col] = count
            for suffix in ['_sum', '_mean']:
                full_col = f"{col}{suffix}"
                if full_col in df.columns:
                    count = int((df[full_col] > 0).sum())
                    if count > 0:
                        result[col] = count
                    break
        return result

    @staticmethod
    def _top_counts(df, column, n=10):
        if column not in df.columns:
            return []
        counts = df[column].dropna().value_counts().head(n)
        return list(zip(counts.index.tolist(), counts.values.tolist()))

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_cascade_articles(self, cascade, articles):
        date_col = self._resolve_col(articles, self.DATE_COLS)
        if not date_col:
            return pd.DataFrame()
        dates = pd.to_datetime(articles[date_col], errors='coerce')
        mask = (dates >= cascade.onset_date) & (dates <= cascade.end_date)

        # Filter by frame (matching _get_burst_articles logic)
        frame_col = FRAME_COLUMNS.get(cascade.frame)
        if frame_col:
            for col in [frame_col, f"{frame_col}_sum", f"{frame_col}_mean"]:
                if col in articles.columns:
                    mask = mask & (articles[col] > 0)
                    break

        return articles[mask]

    def _classify(self, total_score):
        for classification, threshold in sorted(
            CASCADE_THRESHOLDS.items(), key=lambda x: x[1], reverse=True
        ):
            if total_score >= threshold:
                return classification
        return 'not_cascade'

    def _empty_result(self, cascade_id, burst, signals=None):
        """Create a zero-scored CascadeResult for a burst with insufficient data."""
        composite_peak = 0.0
        daily_composite = None
        if signals and 'composite' in signals:
            composite = signals['composite']
            cmask = (composite.index >= burst.onset_date) & (composite.index <= burst.end_date)
            daily_composite = composite[cmask]
            composite_peak = float(daily_composite.max()) if len(daily_composite) > 0 else 0.0

        return CascadeResult(
            cascade_id=cascade_id,
            frame=burst.frame,
            onset_date=burst.onset_date,
            peak_date=burst.peak_date,
            end_date=burst.end_date,
            duration_days=burst.duration_days,
            n_articles=0,
            n_journalists=0,
            n_media=0,
            n_new_journalists=0,
            burst_intensity=burst.intensity,
            adoption_velocity=0.0,
            baseline_mean=burst.baseline_mean,
            peak_proportion=burst.peak_proportion,
            total_score=0.0,
            classification='not_cascade',
            composite_peak=composite_peak,
            daily_composite=daily_composite,
            detection_method=burst.detection_method,
        )
