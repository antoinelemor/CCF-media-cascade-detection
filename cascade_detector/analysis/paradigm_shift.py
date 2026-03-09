"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
paradigm_shift.py

MAIN OBJECTIVE:
---------------
Detect paradigm shifts (changes in dominant frame composition) and attribute
them to specific media cascades and triggering events.

Uses ParadigmDominanceAnalyzer from CCF-paradigm (same 4-method consensus:
information theory, network analysis, causality, proportional) to compute
weekly paradigm states, then detects transitions in the dominant frame set.

Pipeline:
  weekly frame proportions → ParadigmStateComputer (12-week window, daily step,
  parallelized across CPU cores) → ShiftDetector (compare consecutive states,
  direction-aware merge) → CascadeShiftAttributor (link to cascade results)
  → ParadigmShiftResults

Author:
-------
Antoine Lemor
"""

import logging
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from cascade_detector.core.constants import FRAMES
from cascade_detector.core.models import _jsonify

logger = logging.getLogger(__name__)


# =============================================================================
# Module-level function for multiprocessing (must be picklable)
# =============================================================================

def _analyze_window_worker(args: Tuple) -> Optional[dict]:
    """Analyze a single window in a worker process.

    Accepts and returns plain dicts/tuples (picklable) rather than
    dataclass instances. The caller converts back to ParadigmState.
    """
    window_values, window_index, date_ts, window_start_ts, window_end_ts, \
        frame_names, *extra = args
    # extra[0] = parent sys.path (optional, for spawn workers)
    if extra:
        import sys
        for p in extra[0]:
            if p not in sys.path:
                sys.path.insert(0, p)

    from ccf_paradigm import ParadigmDominanceAnalyzer

    # Reconstruct DataFrame in worker
    window = pd.DataFrame(
        window_values, index=pd.DatetimeIndex(window_index),
        columns=frame_names,
    )
    date = pd.Timestamp(date_ts)
    window_start = pd.Timestamp(window_start_ts)
    window_end = pd.Timestamp(window_end_ts)

    try:
        analyzer = ParadigmDominanceAnalyzer(
            frame_names=frame_names, n_workers=1
        )
        dominance_scores = analyzer.calculate_dominance_scores(
            window, show_progress=False
        )
        dominant_frames, details = analyzer.determine_dominant_frames(
            dominance_scores
        )

        paradigm_vector = np.array([
            dominance_scores.loc[f, 'dominance_score']
            if f in dominance_scores.index else 0.0
            for f in frame_names
        ])

        frame_scores = {
            f: float(dominance_scores.loc[f, 'dominance_score'])
            for f in frame_names
            if f in dominance_scores.index
        }

        concentration = float(
            window[dominant_frames].mean().sum()
        ) if dominant_frames else 0.0

        if len(dominant_frames) > 1:
            corr = window[dominant_frames].corr()
            upper = corr.values[np.triu_indices_from(corr.values, k=1)]
            coherence = float(np.nanmean(upper)) if len(upper) > 0 else 1.0
        else:
            coherence = 1.0

        paradigm_type = analyzer._classify_paradigm(len(dominant_frames))

        return {
            'date': date_ts,
            'window_start': window_start_ts,
            'window_end': window_end_ts,
            'dominant_frames': dominant_frames,
            'paradigm_type': paradigm_type,
            'paradigm_vector': paradigm_vector.tolist(),
            'frame_scores': frame_scores,
            'concentration': concentration,
            'coherence': coherence,
        }
    except Exception as e:
        return None


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ParadigmState:
    """Paradigm composition at a single time window."""
    date: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    dominant_frames: List[str]
    paradigm_type: str              # 'Mono-paradigm', 'Dual-paradigm', etc.
    paradigm_vector: np.ndarray     # 8-dim dominance scores
    frame_scores: Dict[str, float]
    concentration: float
    coherence: float

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify({
            'date': self.date.isoformat(),
            'window_start': self.window_start.isoformat(),
            'window_end': self.window_end.isoformat(),
            'dominant_frames': self.dominant_frames,
            'paradigm_type': self.paradigm_type,
            'paradigm_vector': self.paradigm_vector.tolist(),
            'frame_scores': self.frame_scores,
            'concentration': self.concentration,
            'coherence': self.coherence,
        })


@dataclass
class ParadigmShift:
    """A detected paradigm shift."""
    shift_id: str
    shift_date: pd.Timestamp
    shift_type: str                 # 'frame_entry', 'frame_exit', 'recomposition', 'full_replacement'
    entering_frames: List[str]
    exiting_frames: List[str]
    state_before: ParadigmState
    state_after: ParadigmState
    # Magnitude
    shift_magnitude: float          # [0,1] composite
    vector_distance: float          # cosine distance between paradigm vectors
    set_jaccard_distance: float     # 1 - |intersection|/|union| of dominant sets
    concentration_change: float
    # Shift-level dynamics (filled by ShiftDetector.qualify_shifts)
    regime_duration_days: int = 0   # days until next shift (or end of period)
    structural_change: int = 0     # change in number of dominant frames (+1, -1, etc.)
    reversible: bool = False        # does the *next* shift restore state_before?
    # Attribution (filled by attributor)
    attributed_cascades: List[Dict] = field(default_factory=list)
    attributed_events: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify({
            'shift_id': self.shift_id,
            'shift_date': self.shift_date.isoformat(),
            'shift_type': self.shift_type,
            'entering_frames': self.entering_frames,
            'exiting_frames': self.exiting_frames,
            'state_before': self.state_before.to_dict(),
            'state_after': self.state_after.to_dict(),
            'shift_magnitude': self.shift_magnitude,
            'vector_distance': self.vector_distance,
            'set_jaccard_distance': self.set_jaccard_distance,
            'concentration_change': self.concentration_change,
            'regime_duration_days': self.regime_duration_days,
            'structural_change': self.structural_change,
            'reversible': self.reversible,
            'attributed_cascades': self.attributed_cascades,
            'attributed_events': self.attributed_events,
        })


@dataclass
class ShiftEpisode:
    """A group of temporally close shifts forming a turbulent period."""
    episode_id: str
    shifts: List[ParadigmShift]
    # Timing
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    duration_days: int
    n_shifts: int
    # Episode-level dynamics
    dominant_frames_before: List[str]   # set dominant at episode start
    dominant_frames_after: List[str]    # set dominant at episode end
    reversible: bool                    # before == after
    net_structural_change: int          # len(after) - len(before)
    max_complexity: int                 # peak number of dominant frames
    regime_after_duration_days: int     # stability after episode ends

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify({
            'episode_id': self.episode_id,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'duration_days': self.duration_days,
            'n_shifts': self.n_shifts,
            'shift_ids': [s.shift_id for s in self.shifts],
            'dominant_frames_before': self.dominant_frames_before,
            'dominant_frames_after': self.dominant_frames_after,
            'reversible': self.reversible,
            'net_structural_change': self.net_structural_change,
            'max_complexity': self.max_complexity,
            'regime_after_duration_days': self.regime_after_duration_days,
        })


@dataclass
class ParadigmShiftResults:
    """Container for all paradigm shift analysis results."""
    shifts: List[ParadigmShift]
    episodes: List[ShiftEpisode]
    paradigm_timeline: pd.DataFrame  # Weekly: date, dominant_frames, paradigm_type, ...
    analysis_period: Tuple[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify({
            'n_shifts': len(self.shifts),
            'n_episodes': len(self.episodes),
            'shifts': [s.to_dict() for s in self.shifts],
            'episodes': [e.to_dict() for e in self.episodes],
            'analysis_period': self.analysis_period,
        })

    def summary(self) -> str:
        lines = [
            "Paradigm Shift Analysis",
            "=" * 50,
            f"Period: {self.analysis_period[0]} to {self.analysis_period[1]}",
            f"Paradigm states computed: {len(self.paradigm_timeline)}",
            f"Shifts detected: {len(self.shifts)}",
            f"Episodes identified: {len(self.episodes)}",
        ]
        if self.shifts:
            lines.append("")
            lines.append("Shifts:")
            for s in sorted(self.shifts, key=lambda x: x.shift_date):
                entering = ','.join(s.entering_frames) if s.entering_frames else '-'
                exiting = ','.join(s.exiting_frames) if s.exiting_frames else '-'
                rev = ' [REV]' if s.reversible else ''
                lines.append(
                    f"  {s.shift_date.strftime('%Y-%m-%d')} [{s.shift_type}] "
                    f"mag={s.shift_magnitude:.3f} "
                    f"+{entering} -{exiting} "
                    f"dur={s.regime_duration_days}d "
                    f"struct={s.structural_change:+d}{rev} "
                    f"({len(s.attributed_cascades)} cascades)"
                )
        if self.episodes:
            lines.append("")
            lines.append("Episodes:")
            for ep in self.episodes:
                before = ','.join(ep.dominant_frames_before)
                after = ','.join(ep.dominant_frames_after)
                rev = 'reversible' if ep.reversible else 'irreversible'
                lines.append(
                    f"  {ep.episode_id}: "
                    f"{ep.start_date.strftime('%Y-%m-%d')} to "
                    f"{ep.end_date.strftime('%Y-%m-%d')} "
                    f"({ep.n_shifts} shifts, {ep.duration_days}d) "
                    f"[{before}] -> [{after}] "
                    f"{rev} net={ep.net_structural_change:+d} "
                    f"max_complexity={ep.max_complexity} "
                    f"stability_after={ep.regime_after_duration_days}d"
                )
        return '\n'.join(lines)


# =============================================================================
# ParadigmStateComputer
# =============================================================================

class ParadigmStateComputer:
    """Compute paradigm states using sliding window over weekly proportions.

    Wraps ParadigmDominanceAnalyzer from CCF-paradigm to apply the same
    4-method consensus (information theory, network, causality, proportional)
    on each window of weekly frame proportions.

    The window always spans `window_size` weeks (default 12) for statistical
    robustness, but advances by `step_days` (default 1) for daily temporal
    resolution. Windows are analyzed in parallel across CPU cores.
    """

    def __init__(self, window_size: int = 12, step_days: int = 1,
                 frame_names: List[str] = None,
                 n_workers: Optional[int] = None):
        """
        Args:
            window_size: Number of weeks per analysis window.
            step_days: Days between consecutive window positions.
                       Default 1 for daily resolution.
            frame_names: Frame names (default: FRAMES).
            n_workers: Number of parallel workers. Default None = cpu_count.
                       Set to 1 for sequential execution (tests, debugging).
        """
        self.window_size = window_size
        self.step_days = max(1, step_days)
        self.frame_names = frame_names or list(FRAMES)
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()

    def compute_states(self, weekly_props: pd.DataFrame) -> List[ParadigmState]:
        """Compute paradigm states from weekly proportions.

        The weekly data is first interpolated to daily resolution. Then a
        window of `window_size` weeks is slid forward by `step_days` days.
        At each position, the daily data within the window is resampled
        back to weekly for the ParadigmDominanceAnalyzer.

        Windows are analyzed in parallel using ProcessPoolExecutor.

        Args:
            weekly_props: DataFrame with DatetimeIndex (weekly) and columns
                          matching self.frame_names. Values are proportions [0,1].

        Returns:
            List of ParadigmState, one per valid window position, ordered by date.
        """
        n_weeks = len(weekly_props)
        # Capture parent sys.path so spawn workers can find ccf_paradigm
        import sys
        parent_path = list(sys.path)

        if n_weeks < self.window_size:
            logger.warning(
                f"Only {n_weeks} weeks available, need {self.window_size}. "
                f"Using all data as single window."
            )
            result = _analyze_window_worker((
                weekly_props.values.tolist(),
                [t.isoformat() for t in weekly_props.index],
                weekly_props.index[-1].isoformat(),
                weekly_props.index[0].isoformat(),
                weekly_props.index[-1].isoformat(),
                self.frame_names,
                parent_path,
            ))
            if result:
                return [self._dict_to_state(result)]
            return []

        # Interpolate to daily for sub-daily stepping
        daily_idx = pd.date_range(
            weekly_props.index[0], weekly_props.index[-1], freq='D'
        )
        daily = (weekly_props
                 .reindex(daily_idx.union(weekly_props.index))
                 .interpolate('time')
                 .reindex(daily_idx))

        window_td = pd.Timedelta(weeks=self.window_size)
        step_td = pd.Timedelta(days=self.step_days)

        earliest_end = daily.index[0] + window_td
        latest_end = daily.index[-1]

        # Prepare all window tasks
        tasks = []
        window_end = earliest_end
        while window_end <= latest_end:
            window_start = window_end - window_td
            window_daily = daily.loc[window_start:window_end]
            window_weekly = window_daily.resample('W').mean().dropna(how='all')

            if len(window_weekly) >= self.window_size - 1:
                tasks.append((
                    window_weekly.values.tolist(),
                    [t.isoformat() for t in window_weekly.index],
                    window_end.isoformat(),
                    window_start.isoformat(),
                    window_end.isoformat(),
                    self.frame_names,
                    parent_path,
                ))
            window_end += step_td

        n_tasks = len(tasks)
        effective_workers = min(self.n_workers, n_tasks)
        logger.info(
            f"Computing {n_tasks} paradigm states "
            f"(step={self.step_days}d, window={self.window_size}w, "
            f"workers={effective_workers})"
        )

        # Execute in parallel (or sequential if n_workers=1)
        results = []
        if effective_workers <= 1:
            for task in tasks:
                r = _analyze_window_worker(task)
                if r is not None:
                    results.append(r)
        else:
            with ProcessPoolExecutor(max_workers=effective_workers) as pool:
                futures = {
                    pool.submit(_analyze_window_worker, task): i
                    for i, task in enumerate(tasks)
                }
                done = 0
                for future in as_completed(futures):
                    done += 1
                    if done % 50 == 0 or done == n_tasks:
                        logger.info(
                            f"  Progress: {done}/{n_tasks} windows analyzed"
                        )
                    r = future.result()
                    if r is not None:
                        results.append(r)

        # Sort by date and convert to ParadigmState
        results.sort(key=lambda r: r['date'])
        states = [self._dict_to_state(r) for r in results]

        logger.info(
            f"Computed {len(states)} paradigm states from {n_weeks} weeks "
            f"(step={self.step_days}d, window={self.window_size}w)"
        )
        return states

    @staticmethod
    def _dict_to_state(d: dict) -> ParadigmState:
        """Convert a worker result dict back to a ParadigmState."""
        return ParadigmState(
            date=pd.Timestamp(d['date']),
            window_start=pd.Timestamp(d['window_start']),
            window_end=pd.Timestamp(d['window_end']),
            dominant_frames=d['dominant_frames'],
            paradigm_type=d['paradigm_type'],
            paradigm_vector=np.array(d['paradigm_vector']),
            frame_scores=d['frame_scores'],
            concentration=d['concentration'],
            coherence=d['coherence'],
        )


# =============================================================================
# ShiftDetector
# =============================================================================

class ShiftDetector:
    """Detect shifts by comparing consecutive ParadigmStates.

    A shift is detected when the dominant frame set changes or the paradigm
    vector distance exceeds a threshold. Nearby shifts are merged.
    """

    def __init__(self, vector_distance_threshold: float = 0.3,
                 merge_window_weeks: int = 2):
        """
        Args:
            vector_distance_threshold: Cosine distance threshold for detecting
                vector-based shifts even when the dominant set is unchanged.
            merge_window_weeks: Merge shifts within this many weeks, but only
                when they go in the same direction (same frames involved).
        """
        self.vector_distance_threshold = vector_distance_threshold
        self.merge_window_weeks = merge_window_weeks

    def detect_shifts(self, states: List[ParadigmState]) -> List[ParadigmShift]:
        """Detect paradigm shifts from consecutive states.

        Args:
            states: Ordered list of ParadigmState (by date).

        Returns:
            List of ParadigmShift.
        """
        if len(states) < 2:
            return []

        raw_shifts = []
        for i in range(1, len(states)):
            shift = self._compare_states(states[i - 1], states[i])
            if shift is not None:
                raw_shifts.append(shift)

        if not raw_shifts:
            return []

        # Merge nearby shifts
        merged = self._merge_shifts(raw_shifts)

        logger.info(
            f"Detected {len(raw_shifts)} raw shifts, "
            f"merged to {len(merged)}"
        )
        return merged

    def _compare_states(self, before: ParadigmState,
                        after: ParadigmState) -> Optional[ParadigmShift]:
        """Compare two consecutive states; return a ParadigmShift if different."""
        set_before = set(before.dominant_frames)
        set_after = set(after.dominant_frames)

        entering = sorted(set_after - set_before)
        exiting = sorted(set_before - set_after)

        # Jaccard distance
        union = set_before | set_after
        intersection = set_before & set_after
        jaccard_dist = 1.0 - (len(intersection) / len(union)) if union else 0.0

        # Cosine distance between paradigm vectors
        vec_a = before.paradigm_vector
        vec_b = after.paradigm_vector
        if np.linalg.norm(vec_a) > 0 and np.linalg.norm(vec_b) > 0:
            vector_dist = float(cosine(vec_a, vec_b))
        else:
            vector_dist = 0.0

        concentration_change = after.concentration - before.concentration

        # Decide if this is a shift
        set_changed = bool(entering or exiting)
        vector_changed = vector_dist > self.vector_distance_threshold

        if not set_changed and not vector_changed:
            return None

        # Classify shift type
        if set_changed:
            if not intersection:
                shift_type = 'full_replacement'
            elif entering and exiting:
                shift_type = 'recomposition'
            elif entering:
                shift_type = 'frame_entry'
            else:
                shift_type = 'frame_exit'
        else:
            shift_type = 'recomposition'  # vector shift without set change

        # Composite magnitude
        shift_magnitude = (
            0.40 * jaccard_dist
            + 0.40 * min(vector_dist, 1.0)
            + 0.20 * min(abs(concentration_change), 1.0)
        )

        return ParadigmShift(
            shift_id=f"shift_{uuid.uuid4().hex[:8]}",
            shift_date=after.date,
            shift_type=shift_type,
            entering_frames=entering,
            exiting_frames=exiting,
            state_before=before,
            state_after=after,
            shift_magnitude=shift_magnitude,
            vector_distance=vector_dist,
            set_jaccard_distance=jaccard_dist,
            concentration_change=concentration_change,
        )

    def _merge_shifts(self, shifts: List[ParadigmShift]) -> List[ParadigmShift]:
        """Merge shifts that are temporally close AND go in the same direction.

        Two shifts are mergeable only if:
          1. They are within merge_window_weeks of each other, AND
          2. They involve the same frames (entering or exiting sets overlap).

        This preserves distinct transitions like Dual->Triple->Dual as
        separate events, while still merging redundant micro-shifts that
        represent the same ongoing transition (e.g. from sub-weekly stepping).
        """
        if len(shifts) <= 1:
            return shifts

        merged = [shifts[0]]
        merge_delta = pd.Timedelta(weeks=self.merge_window_weeks)

        for shift in shifts[1:]:
            last = merged[-1]
            close_enough = (shift.shift_date - last.shift_date) <= merge_delta
            same_direction = self._same_direction(last, shift)

            if close_enough and same_direction:
                # Merge: keep the higher-magnitude shift but expand frame info
                if shift.shift_magnitude >= last.shift_magnitude:
                    combined_entering = sorted(
                        set(last.entering_frames) | set(shift.entering_frames)
                    )
                    combined_exiting = sorted(
                        set(last.exiting_frames) | set(shift.exiting_frames)
                    )
                    shift.entering_frames = combined_entering
                    shift.exiting_frames = combined_exiting
                    shift.state_before = last.state_before
                    shift.shift_type = self._classify_from_frames(
                        combined_entering, combined_exiting,
                        set(last.state_before.dominant_frames),
                        set(shift.state_after.dominant_frames),
                    )
                    merged[-1] = shift
                else:
                    last.entering_frames = sorted(
                        set(last.entering_frames) | set(shift.entering_frames)
                    )
                    last.exiting_frames = sorted(
                        set(last.exiting_frames) | set(shift.exiting_frames)
                    )
                    last.state_after = shift.state_after
                    last.shift_type = self._classify_from_frames(
                        last.entering_frames, last.exiting_frames,
                        set(last.state_before.dominant_frames),
                        set(shift.state_after.dominant_frames),
                    )
            else:
                merged.append(shift)

        return merged

    @staticmethod
    def _same_direction(a: ParadigmShift, b: ParadigmShift) -> bool:
        """Check if two shifts are part of the same ongoing transition.

        Returns True (mergeable) when shifts reinforce each other:
          - Same frames entering in both, OR same frames exiting in both
          - Complementary: one enters X, the other exits Y (replacement)

        Returns False (keep separate) when shifts REVERSE each other:
          - A frame that entered in shift A exits in shift B (or vice versa).
          - This indicates a transient state (e.g. Dual->Triple->Dual)
            that should be preserved as two distinct events.
        """
        a_entering = set(a.entering_frames)
        a_exiting = set(a.exiting_frames)
        b_entering = set(b.entering_frames)
        b_exiting = set(b.exiting_frames)

        # Reversal: a frame that entered now exits, or vice versa
        reversal = bool(a_entering & b_exiting) or bool(a_exiting & b_entering)
        if reversal:
            return False

        # Reinforcing: same frames entering or exiting, or complementary
        # (one side enters, other side exits different frames = replacement)
        return True

    @staticmethod
    def _classify_from_frames(entering: List[str], exiting: List[str],
                              set_before: set, set_after: set) -> str:
        """Classify shift type from entering/exiting frames."""
        if not set_before & set_after:
            return 'full_replacement'
        if entering and exiting:
            return 'recomposition'
        if entering:
            return 'frame_entry'
        if exiting:
            return 'frame_exit'
        return 'recomposition'

    @staticmethod
    def qualify_shifts(shifts: List[ParadigmShift],
                       period_end: pd.Timestamp) -> List[ParadigmShift]:
        """Compute shift-level dynamics: duration, structural change, reversibility.

        For each shift, computes:
          - regime_duration_days: days until the next shift (or period_end)
          - structural_change: change in number of dominant frames
            (positive = complexification, negative = simplification)
          - reversible: whether the *next* shift restores state_before's
            dominant set (local reversibility only)

        Args:
            shifts: Ordered list of ParadigmShift (by shift_date).
            period_end: End date of the analysis period, used to compute
                        the last shift's regime duration.

        Returns:
            Same shifts with dynamics fields filled.
        """
        if not shifts:
            return shifts

        for i, shift in enumerate(shifts):
            # Structural change: how many dominant frames gained/lost
            n_before = len(shift.state_before.dominant_frames)
            n_after = len(shift.state_after.dominant_frames)
            shift.structural_change = n_after - n_before

            # Regime duration: days until next shift or period end
            if i < len(shifts) - 1:
                next_date = shifts[i + 1].shift_date
                shift.regime_duration_days = (next_date - shift.shift_date).days
            else:
                shift.regime_duration_days = (period_end - shift.shift_date).days

            # Local reversibility: does the next shift restore state_before?
            if i < len(shifts) - 1:
                next_shift = shifts[i + 1]
                restored_set = set(next_shift.state_after.dominant_frames)
                original_set = set(shift.state_before.dominant_frames)
                shift.reversible = (restored_set == original_set)
            else:
                shift.reversible = False

        return shifts


# =============================================================================
# CascadeShiftAttributor
# =============================================================================

class CascadeShiftAttributor:
    """Link cascades to paradigm shifts with three-role attribution.

    Three discursive roles for cascades in paradigm dynamics:
      - Amplification: cascade promotes its own frame toward dominance.
      - Déstabilisation: cascade disrupts paradigm structure without its
        own frame benefiting.
      - Dormante: cascade is active but without measurable structural
        consequence.

    Only amplification and déstabilisation cascades are considered drivers
    and contribute events to shift attribution.
    """

    def __init__(self, lookback_weeks: int = 12,
                 min_cascade_score: float = 0.40,
                 decay_horizon_days: int = 42,
                 lift_threshold: float = 0.05,
                 structural_threshold: float = 0.010):
        """
        Args:
            lookback_weeks: How far back to look for causal cascades.
            min_cascade_score: Minimum total_score to consider a cascade.
            decay_horizon_days: Linear decay horizon T (days). Weight = 0
                after T days from cascade onset.
            lift_threshold: Minimum own_lift to qualify as amplification.
            structural_threshold: Minimum structural_impact (cosine distance)
                to qualify as déstabilisation.
        """
        self.lookback_weeks = lookback_weeks
        self.min_cascade_score = min_cascade_score
        self.decay_horizon_days = decay_horizon_days
        self.lift_threshold = lift_threshold
        self.structural_threshold = structural_threshold

    def attribute(self, shifts: List[ParadigmShift],
                  cascades: list,
                  timeline: pd.DataFrame = None) -> List[ParadigmShift]:
        """Attribute cascades to shifts with three-role logic.

        Args:
            shifts: List of ParadigmShift.
            cascades: List of CascadeResult from detection pipeline.
            timeline: DataFrame with 'date', 'paradigm_{frame}', and
                'concentration' columns. Required for lift/structural
                impact computation. If None, all cascades get dormante role.

        Returns:
            Same shifts with attributed_cascades and attributed_events filled.
        """
        if not cascades:
            return shifts

        for shift in shifts:
            self._attribute_single_shift(shift, cascades, timeline)

        return shifts

    # -----------------------------------------------------------------
    # Impact measurement
    # -----------------------------------------------------------------

    def _compute_weighted_lift(self, cascade, frame: str,
                               timeline: pd.DataFrame) -> float:
        """Weighted mean dominance lift with linear temporal decay.

        For each day t from cascade onset to onset + decay_horizon:
          weight = max(0, 1 - t / decay_horizon)
          delta = dominance(t) - baseline
        baseline = mean dominance in 14 days before onset.
        """
        col = f'paradigm_{frame}'
        if col not in timeline.columns:
            return 0.0

        tl = timeline.copy()
        tl['date'] = pd.to_datetime(tl['date'])
        tl = tl.sort_values('date')

        onset = pd.Timestamp(cascade.onset_date)

        # Baseline: mean dominance in 14 days before onset
        baseline_start = onset - pd.Timedelta(days=14)
        baseline_mask = (tl['date'] >= baseline_start) & (tl['date'] < onset)
        baseline_rows = tl.loc[baseline_mask, col]
        if baseline_rows.empty:
            baseline = tl[col].mean()
        else:
            baseline = baseline_rows.mean()

        # Weighted lift over decay horizon
        horizon_end = onset + pd.Timedelta(days=self.decay_horizon_days)
        post_mask = (tl['date'] >= onset) & (tl['date'] <= horizon_end)
        post_rows = tl.loc[post_mask]

        if post_rows.empty:
            return 0.0

        total_weight = 0.0
        weighted_delta = 0.0
        for _, row in post_rows.iterrows():
            t = (row['date'] - onset).days
            w = max(0.0, 1.0 - t / self.decay_horizon_days)
            delta = row[col] - baseline
            weighted_delta += w * delta
            total_weight += w

        if total_weight == 0:
            return 0.0
        return weighted_delta / total_weight

    def _compute_structural_impact(self, cascade,
                                   timeline: pd.DataFrame) -> float:
        """Cosine distance between pre-cascade and during+post paradigm vectors.

        Same linear decay weighting. Measures how much the overall paradigm
        structure changed, regardless of which frame moved.
        """
        frame_cols = [f'paradigm_{f}' for f in FRAMES]
        available = [c for c in frame_cols if c in timeline.columns]
        if len(available) < 2:
            return 0.0

        tl = timeline.copy()
        tl['date'] = pd.to_datetime(tl['date'])
        tl = tl.sort_values('date')

        onset = pd.Timestamp(cascade.onset_date)

        # Pre-cascade vector: mean of 14 days before onset
        baseline_start = onset - pd.Timedelta(days=14)
        pre_mask = (tl['date'] >= baseline_start) & (tl['date'] < onset)
        pre_rows = tl.loc[pre_mask, available]
        if pre_rows.empty:
            pre_vector = tl[available].mean().values
        else:
            pre_vector = pre_rows.mean().values

        # Post-cascade vector: weighted mean over decay horizon
        horizon_end = onset + pd.Timedelta(days=self.decay_horizon_days)
        post_mask = (tl['date'] >= onset) & (tl['date'] <= horizon_end)
        post_rows = tl.loc[post_mask]

        if post_rows.empty:
            return 0.0

        weights = []
        vectors = []
        for _, row in post_rows.iterrows():
            t = (row['date'] - onset).days
            w = max(0.0, 1.0 - t / self.decay_horizon_days)
            weights.append(w)
            vectors.append(row[available].values.astype(float))

        total_w = sum(weights)
        if total_w == 0:
            return 0.0

        post_vector = np.zeros(len(available))
        for w, v in zip(weights, vectors):
            post_vector += w * v
        post_vector /= total_w

        # Cosine distance (0 = identical, 1 = orthogonal)
        norm_pre = np.linalg.norm(pre_vector)
        norm_post = np.linalg.norm(post_vector)
        if norm_pre == 0 or norm_post == 0:
            return 0.0

        cos_sim = np.dot(pre_vector, post_vector) / (norm_pre * norm_post)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        return float(1.0 - cos_sim)

    # -----------------------------------------------------------------
    # Single-shift attribution
    # -----------------------------------------------------------------

    def _attribute_single_shift(self, shift: ParadigmShift,
                                cascades: list,
                                timeline: pd.DataFrame = None) -> None:
        """Attribute cascades to a single shift with three-role logic."""
        lookback = pd.Timedelta(weeks=self.lookback_weeks)

        scored_cascades = []
        for cascade in cascades:
            # Filter: strength
            if cascade.total_score < self.min_cascade_score:
                continue

            # Filter: temporal — cascade must overlap or precede shift
            cascade_end = pd.Timestamp(cascade.end_date)
            cascade_onset = pd.Timestamp(cascade.onset_date)
            shift_date = shift.shift_date

            if cascade_end < (shift_date - lookback):
                continue
            if cascade_onset > shift_date:
                continue

            # Temporal overlap score (reused from before)
            temporal_score = self._temporal_overlap_score(
                cascade_onset, cascade_end, shift_date, lookback
            )

            # Compute impact metrics if timeline available
            if timeline is not None and not timeline.empty:
                own_lift = self._compute_weighted_lift(
                    cascade, cascade.frame, timeline
                )
                structural_impact = self._compute_structural_impact(
                    cascade, timeline
                )
            else:
                own_lift = 0.0
                structural_impact = 0.0

            # Assign role
            if timeline is not None and not timeline.empty:
                role, role_score, extras = self._assign_role(
                    cascade, shift, temporal_score,
                    own_lift, structural_impact, timeline
                )
            else:
                # Fallback: all dormante when no timeline
                role = 'dormante'
                role_score = 0.50 * temporal_score + 0.50 * cascade.total_score
                extras = {}

            entry = {
                'cascade_id': cascade.cascade_id,
                'frame': cascade.frame,
                'onset_date': cascade.onset_date.isoformat(),
                'end_date': cascade.end_date.isoformat(),
                'total_score': float(cascade.total_score),
                'classification': cascade.classification,
                'role': role,
                'attribution_score': float(role_score),
                'temporal_score': float(temporal_score),
                'own_lift': float(own_lift),
                'structural_impact': float(structural_impact),
            }
            entry.update(extras)
            scored_cascades.append(entry)

        # Sort: amplification first, then déstabilisation, then dormante
        # Within each role, sort by attribution_score desc
        role_order = {'amplification': 0, 'destabilisation': 1, 'dormante': 2}
        scored_cascades.sort(
            key=lambda x: (role_order.get(x['role'], 3),
                           -x['attribution_score'])
        )
        shift.attributed_cascades = scored_cascades

        # Aggregate events from driver cascades only (amplification + déstabilisation)
        event_counts: Dict[str, int] = {}
        for sc in scored_cascades:
            if sc['role'] == 'dormante':
                continue  # Dormante cascades don't contribute events
            cid = sc['cascade_id']
            for cascade in cascades:
                if cascade.cascade_id == cid:
                    for event, count in cascade.dominant_events.items():
                        event_counts[event] = event_counts.get(event, 0) + count
                    break

        shift.attributed_events = [
            {'event': evt, 'count': cnt}
            for evt, cnt in sorted(event_counts.items(), key=lambda x: -x[1])
        ]

    def _assign_role(self, cascade, shift: ParadigmShift,
                     temporal_score: float, own_lift: float,
                     structural_impact: float,
                     timeline: pd.DataFrame) -> tuple:
        """Assign role and compute role-specific score.

        Returns:
            (role, score, extras_dict)
        """
        if own_lift > self.lift_threshold:
            # Amplification: cascade promotes its own frame
            # direction_alignment distinguishes three mechanisms:
            #   1.0 = Promotion: frame NOT dominant before, ENTERS dominance during cascade
            #   0.7 = Consolidation: frame ALREADY dominant before cascade, stays dominant
            #   0.3 = Insufficient: frame never reaches dominance despite positive lift
            direction_alignment = 0.3
            if own_lift > 0 and timeline is not None and 'dominant_frames' in timeline.columns:
                onset = pd.Timestamp(cascade.onset_date)
                end = pd.Timestamp(cascade.end_date)

                # Check pre-cascade dominance (day before onset)
                pre_mask = timeline['date'] < onset
                dominant_before = False
                if pre_mask.any():
                    last_pre = timeline.loc[pre_mask, 'dominant_frames'].iloc[-1]
                    if isinstance(last_pre, str):
                        dominant_before = cascade.frame in last_pre.split(',')

                # Check dominance during cascade + 7 days (causal window)
                causal_end = end + pd.Timedelta(days=7)
                during_mask = (
                    (timeline['date'] >= onset)
                    & (timeline['date'] <= causal_end)
                )
                dominant_during = False
                for dom_str in timeline.loc[during_mask, 'dominant_frames']:
                    if isinstance(dom_str, str) and cascade.frame in dom_str.split(','):
                        dominant_during = True
                        break

                if not dominant_before and dominant_during:
                    direction_alignment = 1.0   # Promotion
                elif dominant_before and dominant_during:
                    direction_alignment = 0.7   # Consolidation

            # Normalize own_lift to [0,1] range (cap at 0.5 for practical range)
            norm_lift = min(abs(own_lift) / 0.5, 1.0)

            score = (0.30 * temporal_score
                     + 0.35 * norm_lift
                     + 0.20 * cascade.total_score
                     + 0.15 * direction_alignment)

            return 'amplification', score, {
                'direction_alignment': float(direction_alignment),
            }

        elif structural_impact > self.structural_threshold:
            # Déstabilisation: disrupts structure without self-benefit
            # concentration_disruption: abs change in concentration
            conc_disruption = abs(shift.concentration_change)
            norm_disruption = min(conc_disruption / 0.3, 1.0)

            norm_structural = min(structural_impact / 0.1, 1.0)

            score = (0.30 * temporal_score
                     + 0.35 * norm_structural
                     + 0.20 * cascade.total_score
                     + 0.15 * norm_disruption)

            return 'destabilisation', score, {
                'concentration_disruption': float(conc_disruption),
            }

        else:
            # Dormante: active but no measurable effect
            score = 0.50 * temporal_score + 0.50 * cascade.total_score
            return 'dormante', score, {}

    @staticmethod
    def _temporal_overlap_score(onset: pd.Timestamp, end: pd.Timestamp,
                                shift_date: pd.Timestamp,
                                lookback: pd.Timedelta) -> float:
        """Score temporal proximity of cascade to shift.

        Returns 1.0 for cascades peaking right at the shift, decreasing
        with distance.
        """
        window_start = shift_date - lookback

        # Overlap: what fraction of the cascade falls in the window?
        overlap_start = max(onset, window_start)
        overlap_end = min(end, shift_date)

        if overlap_end < overlap_start:
            return 0.0

        cascade_duration = max((end - onset).days, 1)
        overlap_days = (overlap_end - overlap_start).days + 1

        overlap_fraction = overlap_days / cascade_duration

        # Recency: cascades closer to shift score higher
        midpoint = onset + (end - onset) / 2
        days_before_shift = (shift_date - midpoint).days
        lookback_days = lookback.days
        recency = max(0.0, 1.0 - days_before_shift / lookback_days)

        return 0.5 * overlap_fraction + 0.5 * recency


# =============================================================================
# EpisodeAnalyzer
# =============================================================================

class EpisodeAnalyzer:
    """Group temporally close shifts into episodes and compute episode-level metrics.

    An episode is a turbulent period where shifts cluster together (gap < gap_weeks).
    Episode-level metrics capture the net effect of all shifts in the cluster,
    answering whether the turbulence produced lasting change or was ephemeral.
    """

    def __init__(self, gap_weeks: int = 3):
        """
        Args:
            gap_weeks: Maximum gap between consecutive shifts to be grouped
                       into the same episode. Default 3 weeks.
        """
        self.gap_weeks = gap_weeks

    def build_episodes(self, shifts: List[ParadigmShift],
                       period_end: pd.Timestamp) -> List[ShiftEpisode]:
        """Group shifts into episodes and compute episode-level dynamics.

        Args:
            shifts: Ordered list of qualified ParadigmShift (shift-level
                    dynamics already computed via ShiftDetector.qualify_shifts).
            period_end: End date of analysis period.

        Returns:
            List of ShiftEpisode.
        """
        if not shifts:
            return []

        groups = self._group_shifts(shifts)
        episodes = []

        for i, group in enumerate(groups):
            episode = self._build_single_episode(
                group, i, groups, period_end
            )
            episodes.append(episode)

        logger.info(f"Built {len(episodes)} episodes from {len(shifts)} shifts")
        return episodes

    def _group_shifts(self, shifts: List[ParadigmShift]) -> List[List[ParadigmShift]]:
        """Group shifts with gap < gap_weeks into clusters."""
        gap_delta = pd.Timedelta(weeks=self.gap_weeks)
        groups: List[List[ParadigmShift]] = [[shifts[0]]]

        for shift in shifts[1:]:
            if (shift.shift_date - groups[-1][-1].shift_date) <= gap_delta:
                groups[-1].append(shift)
            else:
                groups.append([shift])

        return groups

    def _build_single_episode(
        self,
        group: List[ParadigmShift],
        group_idx: int,
        all_groups: List[List[ParadigmShift]],
        period_end: pd.Timestamp,
    ) -> ShiftEpisode:
        """Build a ShiftEpisode from a group of shifts."""
        start_date = group[0].shift_date
        end_date = group[-1].shift_date
        duration_days = max((end_date - start_date).days, 1)

        # Dominant frames before = state_before of first shift
        dominant_before = list(group[0].state_before.dominant_frames)
        # Dominant frames after = state_after of last shift
        dominant_after = list(group[-1].state_after.dominant_frames)

        # Episode-level reversibility: did the paradigm return to its original state?
        reversible = set(dominant_before) == set(dominant_after)

        # Net structural change across the episode
        net_structural_change = len(dominant_after) - len(dominant_before)

        # Max complexity: peak number of dominant frames during the episode
        max_complexity = len(dominant_before)
        for shift in group:
            max_complexity = max(
                max_complexity,
                len(shift.state_after.dominant_frames),
            )

        # Regime after duration: stability after episode ends
        if group_idx < len(all_groups) - 1:
            # Next episode starts at the first shift of the next group
            next_episode_start = all_groups[group_idx + 1][0].shift_date
            regime_after_duration_days = (next_episode_start - end_date).days
        else:
            regime_after_duration_days = (period_end - end_date).days

        return ShiftEpisode(
            episode_id=f"episode_{uuid.uuid4().hex[:8]}",
            shifts=group,
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            n_shifts=len(group),
            dominant_frames_before=dominant_before,
            dominant_frames_after=dominant_after,
            reversible=reversible,
            net_structural_change=net_structural_change,
            max_complexity=max_complexity,
            regime_after_duration_days=regime_after_duration_days,
        )


# =============================================================================
# ParadigmShiftAnalyzer — top-level orchestrator
# =============================================================================

class ParadigmShiftAnalyzer:
    """Top-level orchestrator for paradigm shift detection and attribution.

    Entry point A: analyze(results) — from pipeline DetectionResults
    Entry point B: analyze_from_dataframes(weekly_props_df, cascades) — direct
    """

    def __init__(self, window_size: int = 12,
                 step_days: int = 1,
                 n_workers: Optional[int] = None,
                 vector_distance_threshold: float = 0.3,
                 merge_window_weeks: int = 2,
                 lookback_weeks: int = 12,
                 min_cascade_score: float = 0.40,
                 episode_gap_weeks: int = 3):
        self.state_computer = ParadigmStateComputer(
            window_size=window_size, step_days=step_days,
            n_workers=n_workers,
        )
        self.shift_detector = ShiftDetector(
            vector_distance_threshold=vector_distance_threshold,
            merge_window_weeks=merge_window_weeks,
        )
        self.attributor = CascadeShiftAttributor(
            lookback_weeks=lookback_weeks,
            min_cascade_score=min_cascade_score,
        )
        self.episode_analyzer = EpisodeAnalyzer(
            gap_weeks=episode_gap_weeks,
        )

    def analyze(self, results) -> ParadigmShiftResults:
        """Analyze paradigm shifts from pipeline DetectionResults.

        Extracts weekly proportions from results._indices['temporal'],
        detects shifts, and attributes them to results.cascades.

        Args:
            results: DetectionResults with _indices attached.

        Returns:
            ParadigmShiftResults.
        """
        indices = getattr(results, '_indices', {})
        temporal_idx = indices.get('temporal', {})

        # Extract weekly proportions per frame
        weekly_props = self._extract_weekly_props_from_indices(temporal_idx)
        if weekly_props is None or weekly_props.empty:
            logger.warning("No weekly proportions available for paradigm analysis")
            return ParadigmShiftResults(
                shifts=[], episodes=[],
                paradigm_timeline=pd.DataFrame(),
                analysis_period=results.analysis_period,
            )

        return self._run_pipeline(
            weekly_props, results.cascades, results.analysis_period
        )

    def analyze_from_files(self, proportions_path: str,
                           cascades_path: str) -> ParadigmShiftResults:
        """Analyze paradigm shifts from production output files.

        Args:
            proportions_path: Path to temporal_daily_proportions.parquet
                              (columns: frame, date, proportion).
            cascades_path: Path to cascades.parquet.

        Returns:
            ParadigmShiftResults.
        """
        props_df = pd.read_parquet(proportions_path)
        cascades_df = pd.read_parquet(cascades_path)

        # Pivot daily proportions to wide format, resample to weekly
        props_df['date'] = pd.to_datetime(props_df['date'])
        wide = props_df.pivot_table(
            index='date', columns='frame', values='proportion'
        )
        # Ensure all frames present
        for f in FRAMES:
            if f not in wide.columns:
                wide[f] = 0.0
        wide = wide[list(FRAMES)]
        weekly_props = wide.resample('W').mean().dropna(how='all')

        # Build mock cascade objects for attribution
        mock_cascades = self._cascades_from_dataframe(cascades_df)

        period = (
            str(weekly_props.index.min().date()),
            str(weekly_props.index.max().date()),
        )

        return self._run_pipeline(weekly_props, mock_cascades, period)

    def analyze_from_dataframes(self, weekly_props: pd.DataFrame,
                                cascades: list,
                                analysis_period: Tuple[str, str] = ('', ''),
                                ) -> ParadigmShiftResults:
        """Analyze from pre-built DataFrames (for testing or direct use).

        Args:
            weekly_props: DataFrame with DatetimeIndex and frame columns.
            cascades: List of CascadeResult.
            analysis_period: (start, end) strings.

        Returns:
            ParadigmShiftResults.
        """
        return self._run_pipeline(weekly_props, cascades, analysis_period)

    # -----------------------------------------------------------------
    # Internal pipeline
    # -----------------------------------------------------------------

    def _run_pipeline(self, weekly_props: pd.DataFrame,
                      cascades: list,
                      analysis_period: Tuple[str, str],
                      ) -> ParadigmShiftResults:
        """Core pipeline: states → shifts → qualify → attribute → episodes."""
        logger.info(
            f"Running paradigm shift analysis on {len(weekly_props)} weeks, "
            f"{len(cascades)} cascades"
        )

        # Step 1: Compute paradigm states
        states = self.state_computer.compute_states(weekly_props)

        # Step 2: Build paradigm timeline
        timeline = self._build_timeline(states)

        # Step 3: Detect shifts
        shifts = self.shift_detector.detect_shifts(states)

        # Step 4: Qualify shifts (duration, structural change, reversibility)
        period_end = pd.Timestamp(analysis_period[1]) if analysis_period[1] else (
            weekly_props.index[-1] if not weekly_props.empty else pd.Timestamp.now()
        )
        shifts = self.shift_detector.qualify_shifts(shifts, period_end)

        # Step 5: Attribute cascades (pass timeline for impact measurement)
        shifts = self.attributor.attribute(shifts, cascades, timeline=timeline)

        # Step 6: Build episodes
        episodes = self.episode_analyzer.build_episodes(shifts, period_end)

        results = ParadigmShiftResults(
            shifts=shifts,
            episodes=episodes,
            paradigm_timeline=timeline,
            analysis_period=analysis_period,
        )

        logger.info(
            f"Paradigm shift analysis complete: "
            f"{len(shifts)} shifts, {len(episodes)} episodes"
        )
        return results

    def _extract_weekly_props_from_indices(
        self, temporal_idx: Dict
    ) -> Optional[pd.DataFrame]:
        """Extract weekly proportions DataFrame from temporal index."""
        series_dict = {}
        for frame in FRAMES:
            frame_data = temporal_idx.get(frame, {})
            if not isinstance(frame_data, dict):
                continue
            wp = frame_data.get('weekly_proportions')
            if wp is not None and not wp.empty:
                series_dict[frame] = wp

        if not series_dict:
            return None

        df = pd.DataFrame(series_dict)
        # Fill missing frames with 0
        for f in FRAMES:
            if f not in df.columns:
                df[f] = 0.0
        df = df[list(FRAMES)]
        df = df.fillna(0.0)
        return df

    @staticmethod
    def _build_timeline(states: List[ParadigmState]) -> pd.DataFrame:
        """Build a DataFrame timeline from paradigm states."""
        if not states:
            return pd.DataFrame()

        rows = []
        for s in states:
            row = {
                'date': s.date,
                'dominant_frames': ','.join(s.dominant_frames),
                'paradigm_type': s.paradigm_type,
                'concentration': s.concentration,
                'coherence': s.coherence,
            }
            for i, frame in enumerate(FRAMES):
                row[f'paradigm_{frame}'] = (
                    float(s.paradigm_vector[i])
                    if i < len(s.paradigm_vector) else 0.0
                )
            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def _cascades_from_dataframe(df: pd.DataFrame) -> list:
        """Build lightweight cascade-like objects from a cascades DataFrame."""
        from types import SimpleNamespace

        cascades = []
        for _, row in df.iterrows():
            c = SimpleNamespace(
                cascade_id=row.get('cascade_id', ''),
                frame=row.get('frame', ''),
                onset_date=pd.Timestamp(row.get('onset_date', pd.NaT)),
                end_date=pd.Timestamp(row.get('end_date', pd.NaT)),
                total_score=float(row.get('total_score', 0.0)),
                classification=row.get('classification', ''),
                dominant_events=row.get('dominant_events', {}),
            )
            # dominant_events might be serialized as string
            if isinstance(c.dominant_events, str):
                import json
                try:
                    c.dominant_events = json.loads(c.dominant_events)
                except (json.JSONDecodeError, TypeError):
                    c.dominant_events = {}
            cascades.append(c)
        return cascades
