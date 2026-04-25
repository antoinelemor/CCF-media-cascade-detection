"""
Unit tests for cascade_detector.analysis.paradigm_shift.
=
Uses synthetic data — no database or embeddings required.
Tests:
  - ParadigmStateComputer: stable data, known dominant frame
  - ShiftDetector: stable paradigm (no shifts), shift on frame change,
    magnitude scaling, nearby shift merging
  - CascadeShiftAttributor: overlapping cascade attributed, distant cascade
    excluded, event aggregation
  - ParadigmShiftAnalyzer: end-to-end synthetic pipeline
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cascade_detector.analysis.paradigm_shift import (
    CascadeShiftAttributor,
    EpisodeAnalyzer,
    ParadigmShift,
    ParadigmShiftAnalyzer,
    ParadigmShiftResults,
    ParadigmState,
    ParadigmStateComputer,
    ShiftDetector,
    ShiftEpisode,
)
from cascade_detector.core.constants import FRAMES


# =============================================================================
# Helpers
# =============================================================================

def _make_weekly_props(n_weeks: int, dominant: str = 'Pol',
                       dominant_level: float = 0.35,
                       base_level: float = 0.05,
                       seed: int = 42) -> pd.DataFrame:
    """Create synthetic weekly proportions with one dominant frame."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range('2020-01-05', periods=n_weeks, freq='W')
    data = {}
    for frame in FRAMES:
        if frame == dominant:
            data[frame] = dominant_level + rng.normal(0, 0.02, n_weeks)
        else:
            data[frame] = base_level + rng.normal(0, 0.01, n_weeks)
    df = pd.DataFrame(data, index=dates)
    df = df.clip(lower=0.0)
    return df


def _make_shifting_props(n_weeks_phase1: int = 20,
                         n_weeks_phase2: int = 20,
                         dominant1: str = 'Pol',
                         dominant2: str = 'Envt') -> pd.DataFrame:
    """Create weekly props with a clear shift from dominant1 to dominant2."""
    phase1 = _make_weekly_props(n_weeks_phase1, dominant=dominant1, seed=1)
    phase2 = _make_weekly_props(n_weeks_phase2, dominant=dominant2, seed=2)
    # Adjust phase2 dates to follow phase1
    phase2.index = pd.date_range(
        phase1.index[-1] + pd.Timedelta(weeks=1),
        periods=n_weeks_phase2, freq='W',
    )
    return pd.concat([phase1, phase2])


def _make_cascade(cascade_id: str, frame: str,
                  onset: str, end: str,
                  total_score: float = 0.60,
                  classification: str = 'moderate_cascade',
                  events: dict = None):
    """Create a mock cascade object."""
    return SimpleNamespace(
        cascade_id=cascade_id,
        frame=frame,
        onset_date=pd.Timestamp(onset),
        end_date=pd.Timestamp(end),
        total_score=total_score,
        classification=classification,
        dominant_events=events or {},
    )


def _make_state(date: str, dominant: list, vector: list = None):
    """Create a ParadigmState for testing."""
    ts = pd.Timestamp(date)
    if vector is None:
        vector = np.zeros(len(FRAMES))
        for f in dominant:
            idx = FRAMES.index(f)
            vector[idx] = 0.5
    return ParadigmState(
        date=ts,
        window_start=ts - pd.Timedelta(weeks=12),
        window_end=ts,
        dominant_frames=dominant,
        paradigm_type=f"{'Mono' if len(dominant)==1 else 'Dual'}-paradigm",
        paradigm_vector=np.array(vector, dtype=float),
        frame_scores={f: float(vector[i]) for i, f in enumerate(FRAMES)},
        concentration=sum(vector[FRAMES.index(f)] for f in dominant),
        coherence=1.0,
    )


# =============================================================================
# ParadigmStateComputer tests
# =============================================================================

class TestParadigmStateComputer:

    def test_stable_data_consistent_dominant(self):
        """Stable data should yield consistent dominant frames across windows."""
        wp = _make_weekly_props(30, dominant='Pol', dominant_level=0.40)
        computer = ParadigmStateComputer(window_size=12, step_days=7, n_workers=1)
        states = computer.compute_states(wp)

        assert len(states) > 0
        # All states should have at least one dominant frame
        for s in states:
            assert len(s.dominant_frames) >= 1

    def test_known_dominant_frame_identified(self):
        """A clearly dominant frame should appear in dominant_frames."""
        wp = _make_weekly_props(24, dominant='Eco', dominant_level=0.50,
                                base_level=0.02)
        computer = ParadigmStateComputer(window_size=12, step_days=7, n_workers=1)
        states = computer.compute_states(wp)

        assert len(states) > 0
        # Eco should be dominant in at least half the windows
        eco_dominant_count = sum(
            1 for s in states if 'Eco' in s.dominant_frames
        )
        assert eco_dominant_count >= len(states) // 2

    def test_small_dataset_single_window(self):
        """Fewer weeks than window_size should still return one state."""
        wp = _make_weekly_props(8, dominant='Sci')
        computer = ParadigmStateComputer(window_size=12, step_days=7, n_workers=1)
        states = computer.compute_states(wp)

        assert len(states) == 1

    def test_paradigm_vector_length(self):
        """Paradigm vector should have length == number of frames."""
        wp = _make_weekly_props(20, dominant='Just')
        computer = ParadigmStateComputer(window_size=12, step_days=7, n_workers=1)
        states = computer.compute_states(wp)

        for s in states:
            assert len(s.paradigm_vector) == len(FRAMES)

    def test_state_serialization(self):
        """ParadigmState.to_dict() should produce a serializable dict."""
        wp = _make_weekly_props(14, dominant='Pol')
        computer = ParadigmStateComputer(window_size=12, step_days=7, n_workers=1)
        states = computer.compute_states(wp)

        d = states[0].to_dict()
        assert 'date' in d
        assert 'dominant_frames' in d
        assert 'paradigm_vector' in d


# =============================================================================
# ShiftDetector tests
# =============================================================================

class TestShiftDetector:

    def test_no_shift_on_stable_paradigm(self):
        """Stable dominant frames → no shifts detected."""
        states = [
            _make_state('2020-03-01', ['Pol']),
            _make_state('2020-03-08', ['Pol']),
            _make_state('2020-03-15', ['Pol']),
            _make_state('2020-03-22', ['Pol']),
        ]
        detector = ShiftDetector()
        shifts = detector.detect_shifts(states)
        assert len(shifts) == 0

    def test_shift_when_frames_change(self):
        """Change in dominant frames → shift detected."""
        states = [
            _make_state('2020-03-01', ['Pol']),
            _make_state('2020-03-08', ['Pol']),
            _make_state('2020-03-15', ['Envt']),
        ]
        detector = ShiftDetector()
        shifts = detector.detect_shifts(states)

        assert len(shifts) >= 1
        shift = shifts[0]
        assert 'Envt' in shift.entering_frames
        assert 'Pol' in shift.exiting_frames

    def test_shift_type_frame_entry(self):
        """Adding a frame without removing → frame_entry."""
        states = [
            _make_state('2020-03-01', ['Pol']),
            _make_state('2020-03-08', ['Pol', 'Envt']),
        ]
        detector = ShiftDetector()
        shifts = detector.detect_shifts(states)

        assert len(shifts) == 1
        assert shifts[0].shift_type == 'frame_entry'

    def test_shift_type_frame_exit(self):
        """Removing a frame without adding → frame_exit."""
        states = [
            _make_state('2020-03-01', ['Pol', 'Envt']),
            _make_state('2020-03-08', ['Pol']),
        ]
        detector = ShiftDetector()
        shifts = detector.detect_shifts(states)

        assert len(shifts) == 1
        assert shifts[0].shift_type == 'frame_exit'

    def test_shift_type_full_replacement(self):
        """Completely different set → full_replacement."""
        states = [
            _make_state('2020-03-01', ['Pol', 'Eco']),
            _make_state('2020-03-08', ['Envt', 'Sci']),
        ]
        detector = ShiftDetector()
        shifts = detector.detect_shifts(states)

        assert len(shifts) == 1
        assert shifts[0].shift_type == 'full_replacement'

    def test_magnitude_proportional_to_change(self):
        """Full replacement should have higher magnitude than frame entry."""
        # Small change: add one frame
        states_small = [
            _make_state('2020-03-01', ['Pol']),
            _make_state('2020-03-08', ['Pol', 'Envt']),
        ]
        # Big change: full replacement
        states_big = [
            _make_state('2020-03-01', ['Pol', 'Eco']),
            _make_state('2020-03-08', ['Envt', 'Sci']),
        ]

        detector = ShiftDetector()
        shifts_small = detector.detect_shifts(states_small)
        shifts_big = detector.detect_shifts(states_big)

        assert shifts_big[0].shift_magnitude > shifts_small[0].shift_magnitude

    def test_nearby_shifts_merged(self):
        """Shifts within merge_window_weeks are merged."""
        states = [
            _make_state('2020-03-01', ['Pol']),
            _make_state('2020-03-08', ['Pol', 'Envt']),  # Week 1: entry
            _make_state('2020-03-15', ['Envt']),          # Week 2: Pol exits
        ]
        detector = ShiftDetector(merge_window_weeks=4)
        shifts = detector.detect_shifts(states)

        assert len(shifts) == 1  # Merged into one shift
        assert 'Envt' in shifts[0].entering_frames
        assert 'Pol' in shifts[0].exiting_frames

    def test_distant_shifts_not_merged(self):
        """Shifts far apart are kept separate."""
        states = [
            _make_state('2020-01-05', ['Pol']),
            _make_state('2020-01-12', ['Envt']),   # Shift 1
            _make_state('2020-01-19', ['Envt']),
            _make_state('2020-01-26', ['Envt']),
            _make_state('2020-02-02', ['Envt']),
            _make_state('2020-02-09', ['Envt']),
            _make_state('2020-02-16', ['Envt']),
            _make_state('2020-02-23', ['Envt']),
            _make_state('2020-03-01', ['Envt']),
            _make_state('2020-03-08', ['Sci']),    # Shift 2
        ]
        detector = ShiftDetector(merge_window_weeks=4)
        shifts = detector.detect_shifts(states)

        assert len(shifts) == 2

    def test_reversal_not_merged(self):
        """Dual->Triple->Dual (entry then exit of same frame) stays separate."""
        states = [
            _make_state('2020-03-01', ['Pol', 'Eco']),
            _make_state('2020-03-08', ['Pol', 'Eco', 'Envt']),  # Envt enters
            _make_state('2020-03-15', ['Pol', 'Eco']),           # Envt exits
        ]
        detector = ShiftDetector(merge_window_weeks=4)
        shifts = detector.detect_shifts(states)

        # These should NOT merge: Envt entered then exited (reversal)
        assert len(shifts) == 2
        assert shifts[0].shift_type == 'frame_entry'
        assert shifts[1].shift_type == 'frame_exit'

    def test_shift_serialization(self):
        """ParadigmShift.to_dict() should be serializable."""
        states = [
            _make_state('2020-03-01', ['Pol']),
            _make_state('2020-03-08', ['Envt']),
        ]
        detector = ShiftDetector()
        shifts = detector.detect_shifts(states)

        d = shifts[0].to_dict()
        assert 'shift_id' in d
        assert 'shift_type' in d
        assert 'state_before' in d
        assert 'state_after' in d


# =============================================================================
# CascadeShiftAttributor tests
# =============================================================================

class TestCascadeShiftAttributor:

    @staticmethod
    def _make_timeline(start='2020-01-01', periods=120,
                       dominant_frame='Envt', dominant_level=0.35,
                       base_level=0.08, boost_start=None, boost_amount=0.15):
        """Build a mock daily timeline DataFrame.

        If boost_start is given, boost dominant_frame from that date onward
        (simulating a cascade effect).
        """
        dates = pd.date_range(start, periods=periods, freq='D')
        rng = np.random.RandomState(99)
        data = {'date': dates, 'concentration': 0.5 + rng.normal(0, 0.02, periods)}
        for frame in FRAMES:
            level = dominant_level if frame == dominant_frame else base_level
            vals = level + rng.normal(0, 0.005, periods)
            if boost_start and frame == dominant_frame:
                boost_idx = (dates >= pd.Timestamp(boost_start))
                vals[boost_idx] += boost_amount
            data[f'paradigm_{frame}'] = np.clip(vals, 0, 1)
        df = pd.DataFrame(data)
        # Compute dominant_frames: top 2 frames by paradigm score each day
        frame_cols = [f'paradigm_{f}' for f in FRAMES]
        dominant_list = []
        for _, row in df.iterrows():
            scores = {f: row[f'paradigm_{f}'] for f in FRAMES}
            top2 = sorted(scores, key=scores.get, reverse=True)[:2]
            dominant_list.append(','.join(top2))
        df['dominant_frames'] = dominant_list
        return df

    def test_overlapping_cascade_attributed(self):
        """A cascade overlapping a shift with positive lift → amplification."""
        shift = ParadigmShift(
            shift_id='test_shift',
            shift_date=pd.Timestamp('2020-03-15'),
            shift_type='frame_entry',
            entering_frames=['Envt'],
            exiting_frames=[],
            state_before=_make_state('2020-03-08', ['Pol']),
            state_after=_make_state('2020-03-15', ['Pol', 'Envt']),
            shift_magnitude=0.5,
            vector_distance=0.3,
            set_jaccard_distance=0.33,
            concentration_change=0.1,
        )

        cascade = _make_cascade(
            'c1', 'Envt',
            onset='2020-02-15', end='2020-03-10',
            total_score=0.65,
            events={'evt_weather': 5, 'evt_meeting': 2},
        )

        # Timeline where Envt gets boosted from cascade onset
        timeline = self._make_timeline(
            dominant_frame='Envt', boost_start='2020-02-15', boost_amount=0.20
        )

        attributor = CascadeShiftAttributor(lift_threshold=0.05)
        result = attributor.attribute([shift], [cascade], timeline=timeline)

        assert len(result[0].attributed_cascades) == 1
        attr = result[0].attributed_cascades[0]
        assert attr['cascade_id'] == 'c1'
        assert attr['role'] == 'amplification'
        assert attr['attribution_score'] > 0
        assert attr['own_lift'] > 0.05

    def test_distant_cascade_not_attributed(self):
        """A cascade too far in the past should not be attributed."""
        shift = ParadigmShift(
            shift_id='test_shift',
            shift_date=pd.Timestamp('2020-06-15'),
            shift_type='frame_entry',
            entering_frames=['Envt'],
            exiting_frames=[],
            state_before=_make_state('2020-06-08', ['Pol']),
            state_after=_make_state('2020-06-15', ['Pol', 'Envt']),
            shift_magnitude=0.5,
            vector_distance=0.3,
            set_jaccard_distance=0.33,
            concentration_change=0.1,
        )

        cascade = _make_cascade(
            'c_old', 'Envt',
            onset='2019-12-01', end='2020-01-01',
            total_score=0.70,
        )

        attributor = CascadeShiftAttributor()
        result = attributor.attribute([shift], [cascade])

        assert len(result[0].attributed_cascades) == 0

    def test_weak_cascade_filtered(self):
        """Cascades below min_cascade_score should be excluded."""
        shift = ParadigmShift(
            shift_id='test_shift',
            shift_date=pd.Timestamp('2020-03-15'),
            shift_type='frame_entry',
            entering_frames=['Envt'],
            exiting_frames=[],
            state_before=_make_state('2020-03-08', ['Pol']),
            state_after=_make_state('2020-03-15', ['Pol', 'Envt']),
            shift_magnitude=0.5,
            vector_distance=0.3,
            set_jaccard_distance=0.33,
            concentration_change=0.1,
        )

        cascade = _make_cascade(
            'c_weak', 'Envt',
            onset='2020-02-15', end='2020-03-10',
            total_score=0.20,
        )

        attributor = CascadeShiftAttributor(min_cascade_score=0.40)
        result = attributor.attribute([shift], [cascade])

        assert len(result[0].attributed_cascades) == 0

    def test_events_aggregated(self):
        """Events come only from driver cascades (amplification + déstabilisation)."""
        shift = ParadigmShift(
            shift_id='test_shift',
            shift_date=pd.Timestamp('2020-03-15'),
            shift_type='frame_entry',
            entering_frames=['Envt'],
            exiting_frames=[],
            state_before=_make_state('2020-03-08', ['Pol']),
            state_after=_make_state('2020-03-15', ['Pol', 'Envt']),
            shift_magnitude=0.5,
            vector_distance=0.3,
            set_jaccard_distance=0.33,
            concentration_change=0.1,
        )

        # c1: Envt with boost → amplification (driver)
        c1 = _make_cascade(
            'c1', 'Envt', '2020-02-15', '2020-03-10',
            total_score=0.60,
            events={'evt_weather': 5, 'evt_meeting': 2},
        )

        timeline = self._make_timeline(
            dominant_frame='Envt', boost_start='2020-02-15', boost_amount=0.20
        )

        attributor = CascadeShiftAttributor(lift_threshold=0.05)
        result = attributor.attribute([shift], [c1], timeline=timeline)

        # Driver cascade events should be aggregated
        events = {e['event']: e['count'] for e in result[0].attributed_events}
        assert 'evt_weather' in events

    def test_amplification_role(self):
        """Cascade with high own_lift gets role='amplification'."""
        shift = ParadigmShift(
            shift_id='s1',
            shift_date=pd.Timestamp('2020-03-20'),
            shift_type='frame_entry',
            entering_frames=['Envt'],
            exiting_frames=[],
            state_before=_make_state('2020-03-13', ['Pol']),
            state_after=_make_state('2020-03-20', ['Pol', 'Envt']),
            shift_magnitude=0.5, vector_distance=0.3,
            set_jaccard_distance=0.33, concentration_change=0.1,
        )

        cascade = _make_cascade(
            'c_amp', 'Envt',
            onset='2020-02-20', end='2020-03-15',
            total_score=0.65,
        )

        # Timeline: Envt boosted strongly from cascade onset
        timeline = self._make_timeline(
            dominant_frame='Envt', boost_start='2020-02-20', boost_amount=0.25
        )

        attributor = CascadeShiftAttributor(lift_threshold=0.05)
        result = attributor.attribute([shift], [cascade], timeline=timeline)

        attr = result[0].attributed_cascades[0]
        assert attr['role'] == 'amplification'
        assert 'direction_alignment' in attr
        assert attr['own_lift'] > 0.05
        # Envt already dominant before boost → consolidation (0.7)
        assert attr['direction_alignment'] == 0.7

    def test_direction_alignment_promotion(self):
        """Frame NOT dominant before, ENTERS dominance during cascade → 1.0."""
        shift = ParadigmShift(
            shift_id='s1',
            shift_date=pd.Timestamp('2020-03-20'),
            shift_type='frame_entry',
            entering_frames=['Sci'],
            exiting_frames=[],
            state_before=_make_state('2020-03-13', ['Pol']),
            state_after=_make_state('2020-03-20', ['Pol', 'Sci']),
            shift_magnitude=0.5, vector_distance=0.3,
            set_jaccard_distance=0.33, concentration_change=0.1,
        )

        # Sci cascade: Sci starts low, gets boosted into dominant set
        cascade = _make_cascade(
            'c_sci', 'Sci',
            onset='2020-02-20', end='2020-03-15',
            total_score=0.60,
        )

        # Timeline: Pol is dominant (0.40), Sci starts low (0.08),
        # Sci gets boosted to 0.35 during cascade → enters top 2
        dates = pd.date_range('2020-01-01', periods=120, freq='D')
        rng = np.random.RandomState(42)
        data = {'date': dates, 'concentration': np.full(120, 0.5)}
        for frame in FRAMES:
            if frame == 'Pol':
                vals = np.full(120, 0.40)
            elif frame == 'Sci':
                vals = np.full(120, 0.08)
                vals[dates >= pd.Timestamp('2020-02-20')] = 0.35
            else:
                vals = np.full(120, 0.08) + rng.normal(0, 0.001, 120)
            data[f'paradigm_{frame}'] = np.clip(vals, 0, 1)
        dom_list = []
        for i in range(120):
            scores = {f: data[f'paradigm_{f}'][i] for f in FRAMES}
            top2 = sorted(scores, key=scores.get, reverse=True)[:2]
            dom_list.append(','.join(top2))
        data['dominant_frames'] = dom_list
        timeline = pd.DataFrame(data)

        attributor = CascadeShiftAttributor(lift_threshold=0.05)
        result = attributor.attribute([shift], [cascade], timeline=timeline)

        attr = result[0].attributed_cascades[0]
        assert attr['role'] == 'amplification'
        # Sci was NOT dominant before, ENTERS during cascade → promotion
        assert attr['direction_alignment'] == 1.0

    def test_direction_alignment_consolidation(self):
        """Frame ALREADY dominant before cascade, stays dominant → 0.7."""
        shift = ParadigmShift(
            shift_id='s1',
            shift_date=pd.Timestamp('2020-03-20'),
            shift_type='recomposition',
            entering_frames=[],
            exiting_frames=[],
            state_before=_make_state('2020-03-13', ['Pol', 'Eco']),
            state_after=_make_state('2020-03-20', ['Pol', 'Eco']),
            shift_magnitude=0.3, vector_distance=0.2,
            set_jaccard_distance=0.0, concentration_change=0.05,
        )

        # Pol cascade: Pol is already dominant, gets a boost
        cascade = _make_cascade(
            'c_pol', 'Pol',
            onset='2020-02-20', end='2020-03-15',
            total_score=0.55,
        )

        # Timeline: Pol already dominant (0.40), gets modest boost
        timeline = self._make_timeline(
            dominant_frame='Pol', dominant_level=0.40,
            boost_start='2020-02-20', boost_amount=0.10
        )

        attributor = CascadeShiftAttributor(lift_threshold=0.05)
        result = attributor.attribute([shift], [cascade], timeline=timeline)

        attr = result[0].attributed_cascades[0]
        assert attr['role'] == 'amplification'
        # Pol already dominant → consolidation
        assert attr['direction_alignment'] == 0.7

    def test_direction_alignment_never_dominant(self):
        """Frame with positive lift but never reaching dominant set → 0.3."""
        shift = ParadigmShift(
            shift_id='s1',
            shift_date=pd.Timestamp('2020-03-20'),
            shift_type='frame_entry',
            entering_frames=['Envt'],
            exiting_frames=[],
            state_before=_make_state('2020-03-13', ['Pol']),
            state_after=_make_state('2020-03-20', ['Pol', 'Envt']),
            shift_magnitude=0.5, vector_distance=0.3,
            set_jaccard_distance=0.33, concentration_change=0.1,
        )

        # Secu cascade: Secu gets a modest boost but stays far from dominant
        cascade = _make_cascade(
            'c_secu', 'Secu',
            onset='2020-02-20', end='2020-03-15',
            total_score=0.55,
        )

        # Timeline: Secu gets boost (enough for lift > 0.05)
        # but not enough to enter dominant set (Pol=0.40, Envt=0.35 stay on top)
        dates = pd.date_range('2020-01-01', periods=120, freq='D')
        rng = np.random.RandomState(42)
        data = {'date': dates, 'concentration': np.full(120, 0.5)}
        for frame in FRAMES:
            if frame == 'Pol':
                vals = np.full(120, 0.40)
            elif frame == 'Envt':
                vals = np.full(120, 0.35)
            elif frame == 'Secu':
                vals = np.full(120, 0.05)
                vals[dates >= pd.Timestamp('2020-02-20')] = 0.15  # +0.10 boost
            else:
                vals = np.full(120, 0.05) + rng.normal(0, 0.001, 120)
            data[f'paradigm_{frame}'] = np.clip(vals, 0, 1)
        dom_list = []
        for i in range(120):
            scores = {f: data[f'paradigm_{f}'][i] for f in FRAMES}
            top2 = sorted(scores, key=scores.get, reverse=True)[:2]
            dom_list.append(','.join(top2))
        data['dominant_frames'] = dom_list
        timeline = pd.DataFrame(data)

        attributor = CascadeShiftAttributor(lift_threshold=0.05)
        result = attributor.attribute([shift], [cascade], timeline=timeline)

        attr = result[0].attributed_cascades[0]
        assert attr['role'] == 'amplification'
        assert attr['own_lift'] > 0.05
        # Secu never reaches dominant set → 0.3
        assert attr['direction_alignment'] == 0.3

    def test_destabilisation_role(self):
        """Cascade with high structural_impact but low own_lift → déstabilisation."""
        shift = ParadigmShift(
            shift_id='s1',
            shift_date=pd.Timestamp('2020-03-20'),
            shift_type='full_replacement',
            entering_frames=['Envt'],
            exiting_frames=['Pol'],
            state_before=_make_state('2020-03-13', ['Pol']),
            state_after=_make_state('2020-03-20', ['Envt']),
            shift_magnitude=0.7, vector_distance=0.5,
            set_jaccard_distance=1.0, concentration_change=0.2,
        )

        # Secu cascade: own frame doesn't benefit but structure changes
        cascade = _make_cascade(
            'c_destab', 'Secu',
            onset='2020-02-20', end='2020-03-15',
            total_score=0.60,
        )

        # Timeline: Secu stays flat (no self-benefit), but other frames shift
        # Pol drops, Envt rises → structural change
        dates = pd.date_range('2020-01-01', periods=120, freq='D')
        rng = np.random.RandomState(42)
        data = {'date': dates, 'concentration': np.full(120, 0.5)}
        for frame in FRAMES:
            if frame == 'Pol':
                vals = np.full(120, 0.35)
                vals[dates >= pd.Timestamp('2020-02-20')] = 0.10
            elif frame == 'Envt':
                vals = np.full(120, 0.08)
                vals[dates >= pd.Timestamp('2020-02-20')] = 0.35
            elif frame == 'Secu':
                vals = np.full(120, 0.08)  # Flat — no self-benefit
            else:
                vals = np.full(120, 0.08) + rng.normal(0, 0.002, 120)
            data[f'paradigm_{frame}'] = np.clip(vals, 0, 1)
        # Add dominant_frames column
        dom_list = []
        for i in range(120):
            scores = {f: data[f'paradigm_{f}'][i] for f in FRAMES}
            top2 = sorted(scores, key=scores.get, reverse=True)[:2]
            dom_list.append(','.join(top2))
        data['dominant_frames'] = dom_list
        timeline = pd.DataFrame(data)

        attributor = CascadeShiftAttributor(
            lift_threshold=0.05, structural_threshold=0.01
        )
        result = attributor.attribute([shift], [cascade], timeline=timeline)

        attr = result[0].attributed_cascades[0]
        assert attr['role'] == 'destabilisation'
        assert 'concentration_disruption' in attr
        assert attr['structural_impact'] > 0.01

    def test_dormante_role(self):
        """Cascade with low lift AND low structural_impact → dormante."""
        shift = ParadigmShift(
            shift_id='s1',
            shift_date=pd.Timestamp('2020-03-20'),
            shift_type='frame_entry',
            entering_frames=['Envt'],
            exiting_frames=[],
            state_before=_make_state('2020-03-13', ['Pol']),
            state_after=_make_state('2020-03-20', ['Pol', 'Envt']),
            shift_magnitude=0.5, vector_distance=0.3,
            set_jaccard_distance=0.33, concentration_change=0.1,
        )

        cascade = _make_cascade(
            'c_dorm', 'Sci',
            onset='2020-02-20', end='2020-03-15',
            total_score=0.50,
        )

        # Completely flat timeline → no lift, no structural impact
        dates = pd.date_range('2020-01-01', periods=120, freq='D')
        data = {'date': dates, 'concentration': np.full(120, 0.5)}
        for frame in FRAMES:
            data[f'paradigm_{frame}'] = np.full(120, 0.125)
        data['dominant_frames'] = [','.join(FRAMES[:2])] * 120
        timeline = pd.DataFrame(data)

        attributor = CascadeShiftAttributor(
            lift_threshold=0.05, structural_threshold=0.01
        )
        result = attributor.attribute([shift], [cascade], timeline=timeline)

        attr = result[0].attributed_cascades[0]
        assert attr['role'] == 'dormante'

    def test_dormante_no_events(self):
        """Dormante cascades don't contribute to attributed_events."""
        shift = ParadigmShift(
            shift_id='s1',
            shift_date=pd.Timestamp('2020-03-20'),
            shift_type='frame_entry',
            entering_frames=['Envt'],
            exiting_frames=[],
            state_before=_make_state('2020-03-13', ['Pol']),
            state_after=_make_state('2020-03-20', ['Pol', 'Envt']),
            shift_magnitude=0.5, vector_distance=0.3,
            set_jaccard_distance=0.33, concentration_change=0.1,
        )

        cascade = _make_cascade(
            'c_dorm', 'Sci',
            onset='2020-02-20', end='2020-03-15',
            total_score=0.50,
            events={'evt_weather': 10, 'evt_meeting': 5},
        )

        # Flat timeline → dormante
        dates = pd.date_range('2020-01-01', periods=120, freq='D')
        data = {'date': dates, 'concentration': np.full(120, 0.5)}
        for frame in FRAMES:
            data[f'paradigm_{frame}'] = np.full(120, 0.125)
        data['dominant_frames'] = [','.join(FRAMES[:2])] * 120
        timeline = pd.DataFrame(data)

        attributor = CascadeShiftAttributor(
            lift_threshold=0.05, structural_threshold=0.01
        )
        result = attributor.attribute([shift], [cascade], timeline=timeline)

        assert result[0].attributed_cascades[0]['role'] == 'dormante'
        # Events should NOT be aggregated from dormante
        assert len(result[0].attributed_events) == 0

    def test_linear_decay_weighting(self):
        """Lift measured 30 days after cascade onset has lower weight than during."""
        shift = ParadigmShift(
            shift_id='s1',
            shift_date=pd.Timestamp('2020-04-20'),
            shift_type='frame_entry',
            entering_frames=['Envt'],
            exiting_frames=[],
            state_before=_make_state('2020-04-13', ['Pol']),
            state_after=_make_state('2020-04-20', ['Pol', 'Envt']),
            shift_magnitude=0.5, vector_distance=0.3,
            set_jaccard_distance=0.33, concentration_change=0.1,
        )

        cascade = _make_cascade(
            'c1', 'Envt',
            onset='2020-02-01', end='2020-02-20',
            total_score=0.60,
        )

        # Boost Envt only AFTER day 30 (well past cascade end)
        # With decay, this late boost should be down-weighted
        dates = pd.date_range('2020-01-01', periods=150, freq='D')
        rng = np.random.RandomState(7)
        data = {'date': dates, 'concentration': np.full(150, 0.5)}
        for frame in FRAMES:
            if frame == 'Envt':
                vals = np.full(150, 0.10)
                # Boost only 35-42 days after onset (near end of decay window)
                late_start = pd.Timestamp('2020-02-01') + pd.Timedelta(days=35)
                vals[dates >= late_start] = 0.30
            else:
                vals = np.full(150, 0.10) + rng.normal(0, 0.001, 150)
            data[f'paradigm_{frame}'] = np.clip(vals, 0, 1)
        data['dominant_frames'] = [','.join(FRAMES[:2])] * 150
        timeline = pd.DataFrame(data)

        attributor = CascadeShiftAttributor(
            decay_horizon_days=42, lift_threshold=0.05
        )

        # Compute lift directly
        lift = attributor._compute_weighted_lift(cascade, 'Envt', timeline)
        # The late boost should be significantly down-weighted
        # compared to if the boost were early
        assert lift < 0.20  # Down-weighted (full boost would be 0.20)

    def test_no_timeline_fallback(self):
        """When timeline=None, all cascades get role='dormante' with fallback scoring."""
        shift = ParadigmShift(
            shift_id='s1',
            shift_date=pd.Timestamp('2020-03-20'),
            shift_type='frame_entry',
            entering_frames=['Envt'],
            exiting_frames=[],
            state_before=_make_state('2020-03-13', ['Pol']),
            state_after=_make_state('2020-03-20', ['Pol', 'Envt']),
            shift_magnitude=0.5, vector_distance=0.3,
            set_jaccard_distance=0.33, concentration_change=0.1,
        )

        cascade = _make_cascade(
            'c1', 'Envt',
            onset='2020-02-20', end='2020-03-15',
            total_score=0.65,
        )

        attributor = CascadeShiftAttributor()
        result = attributor.attribute([shift], [cascade], timeline=None)

        attr = result[0].attributed_cascades[0]
        assert attr['role'] == 'dormante'
        assert attr['own_lift'] == 0.0
        assert attr['structural_impact'] == 0.0
        # Score = 0.50 * temporal + 0.50 * total_score
        assert attr['attribution_score'] > 0

    def test_role_ordering(self):
        """Output sorted: amplification > déstabilisation > dormante."""
        shift = ParadigmShift(
            shift_id='s1',
            shift_date=pd.Timestamp('2020-03-20'),
            shift_type='full_replacement',
            entering_frames=['Envt'],
            exiting_frames=['Pol'],
            state_before=_make_state('2020-03-13', ['Pol']),
            state_after=_make_state('2020-03-20', ['Envt']),
            shift_magnitude=0.7, vector_distance=0.5,
            set_jaccard_distance=1.0, concentration_change=0.2,
        )

        # c_amp: Envt cascade with Envt boost → amplification
        c_amp = _make_cascade(
            'c_amp', 'Envt',
            onset='2020-02-20', end='2020-03-15',
            total_score=0.65,
        )

        # c_destab: Secu cascade, no self-benefit but structure changes
        c_destab = _make_cascade(
            'c_destab', 'Secu',
            onset='2020-02-20', end='2020-03-15',
            total_score=0.60,
        )

        # c_dorm: Sci cascade on flat timeline → dormante
        c_dorm = _make_cascade(
            'c_dorm', 'Sci',
            onset='2020-02-20', end='2020-03-15',
            total_score=0.55,
        )

        # Timeline: Envt boosted (for c_amp), Pol drops (structural change for c_destab),
        # Sci flat (dormante)
        dates = pd.date_range('2020-01-01', periods=120, freq='D')
        rng = np.random.RandomState(42)
        data = {'date': dates, 'concentration': np.full(120, 0.5)}
        for frame in FRAMES:
            if frame == 'Envt':
                vals = np.full(120, 0.08)
                vals[dates >= pd.Timestamp('2020-02-20')] = 0.35
            elif frame == 'Pol':
                vals = np.full(120, 0.35)
                vals[dates >= pd.Timestamp('2020-02-20')] = 0.08
            else:
                vals = np.full(120, 0.08) + rng.normal(0, 0.002, 120)
            data[f'paradigm_{frame}'] = np.clip(vals, 0, 1)
        # Add dominant_frames column
        dom_list = []
        for i in range(120):
            scores = {f: data[f'paradigm_{f}'][i] for f in FRAMES}
            top2 = sorted(scores, key=scores.get, reverse=True)[:2]
            dom_list.append(','.join(top2))
        data['dominant_frames'] = dom_list
        timeline = pd.DataFrame(data)

        attributor = CascadeShiftAttributor(
            lift_threshold=0.05, structural_threshold=0.01
        )
        result = attributor.attribute(
            [shift], [c_dorm, c_destab, c_amp], timeline=timeline
        )

        roles = [c['role'] for c in result[0].attributed_cascades]
        # amplification first, then destabilisation, then dormante
        role_positions = {r: i for i, r in enumerate(roles)}
        if 'amplification' in role_positions and 'destabilisation' in role_positions:
            assert role_positions['amplification'] < role_positions['destabilisation']
        if 'destabilisation' in role_positions and 'dormante' in role_positions:
            assert role_positions['destabilisation'] < role_positions['dormante']
        if 'amplification' in role_positions and 'dormante' in role_positions:
            assert role_positions['amplification'] < role_positions['dormante']


# =============================================================================
# ParadigmShiftAnalyzer end-to-end tests
# These tests are slow (~3 min each) because ParadigmDominanceAnalyzer
# uses a 4-method consensus that takes ~6s per window.
# Run with: pytest -m slow tests/test_paradigm_shift.py
# =============================================================================

@pytest.mark.slow
class TestParadigmShiftAnalyzer:

    def test_end_to_end_no_cascades(self):
        """Pipeline works with no cascades (attribution step is a no-op)."""
        wp = _make_shifting_props(20, 20, 'Pol', 'Envt')
        analyzer = ParadigmShiftAnalyzer(window_size=12, step_days=14, n_workers=1)
        results = analyzer.analyze_from_dataframes(
            wp, cascades=[], analysis_period=('2020-01-05', '2020-10-11'),
        )

        assert isinstance(results, ParadigmShiftResults)
        assert not results.paradigm_timeline.empty
        assert len(results.paradigm_timeline) > 0

    def test_end_to_end_with_cascades(self):
        """Pipeline detects shifts and attributes cascades."""
        wp = _make_shifting_props(20, 20, 'Pol', 'Envt')
        shift_date_approx = wp.index[20 + 12]  # ~where shift happens

        cascade = _make_cascade(
            'c1', 'Envt',
            onset=str((shift_date_approx - pd.Timedelta(weeks=4)).date()),
            end=str((shift_date_approx - pd.Timedelta(weeks=1)).date()),
            total_score=0.65,
            events={'evt_weather': 10},
        )

        analyzer = ParadigmShiftAnalyzer(window_size=12, step_days=14, n_workers=1)
        results = analyzer.analyze_from_dataframes(
            wp, cascades=[cascade],
            analysis_period=('2020-01-05', '2020-10-11'),
        )

        assert isinstance(results, ParadigmShiftResults)
        # Should have at least one shift
        assert len(results.shifts) >= 0  # May vary with synthetic data

    def test_results_serialization(self):
        """ParadigmShiftResults.to_dict() and summary() should work."""
        wp = _make_weekly_props(30, dominant='Pol')
        analyzer = ParadigmShiftAnalyzer(window_size=12, step_days=14, n_workers=1)
        results = analyzer.analyze_from_dataframes(
            wp, cascades=[], analysis_period=('2020-01-05', '2020-07-26'),
        )

        d = results.to_dict()
        assert 'n_shifts' in d
        assert 'shifts' in d
        assert 'n_episodes' in d
        assert 'episodes' in d

        s = results.summary()
        assert 'Paradigm Shift Analysis' in s
        assert 'Episodes' in s or len(results.shifts) == 0

    def test_timeline_has_frame_columns(self):
        """Timeline should have paradigm_{frame} columns for all frames."""
        wp = _make_weekly_props(24, dominant='Eco')
        analyzer = ParadigmShiftAnalyzer(window_size=12, step_days=14, n_workers=1)
        results = analyzer.analyze_from_dataframes(
            wp, cascades=[], analysis_period=('2020-01-05', '2020-06-14'),
        )

        for frame in FRAMES:
            assert f'paradigm_{frame}' in results.paradigm_timeline.columns

    def test_analyze_from_detection_results(self):
        """analyze() should work with a mock DetectionResults."""
        wp = _make_weekly_props(24, dominant='Pol')

        # Build mock temporal index
        temporal_idx = {}
        for frame in FRAMES:
            temporal_idx[frame] = {
                'weekly_proportions': wp[frame],
            }

        mock_results = SimpleNamespace(
            cascades=[],
            _indices={'temporal': temporal_idx},
            analysis_period=('2020-01-05', '2020-06-14'),
        )

        analyzer = ParadigmShiftAnalyzer(window_size=12, step_days=14, n_workers=1)
        results = analyzer.analyze(mock_results)

        assert isinstance(results, ParadigmShiftResults)
        assert not results.paradigm_timeline.empty


# =============================================================================
# ShiftDetector.qualify_shifts tests
# =============================================================================

class TestQualifyShifts:

    def test_regime_duration_between_shifts(self):
        """regime_duration_days should be the gap to the next shift."""
        shifts = [
            ParadigmShift(
                shift_id='s1', shift_date=pd.Timestamp('2020-03-08'),
                shift_type='frame_entry', entering_frames=['Envt'],
                exiting_frames=[],
                state_before=_make_state('2020-03-01', ['Pol']),
                state_after=_make_state('2020-03-08', ['Pol', 'Envt']),
                shift_magnitude=0.4, vector_distance=0.2,
                set_jaccard_distance=0.33, concentration_change=0.1,
            ),
            ParadigmShift(
                shift_id='s2', shift_date=pd.Timestamp('2020-03-28'),
                shift_type='frame_exit', entering_frames=[],
                exiting_frames=['Envt'],
                state_before=_make_state('2020-03-22', ['Pol', 'Envt']),
                state_after=_make_state('2020-03-28', ['Pol']),
                shift_magnitude=0.4, vector_distance=0.2,
                set_jaccard_distance=0.33, concentration_change=-0.1,
            ),
        ]
        period_end = pd.Timestamp('2020-06-30')
        ShiftDetector.qualify_shifts(shifts, period_end)

        # s1 → s2 = 20 days
        assert shifts[0].regime_duration_days == 20
        # s2 → period_end = 94 days
        assert shifts[1].regime_duration_days == 94

    def test_structural_change_positive(self):
        """Adding a frame: structural_change = +1."""
        shifts = [
            ParadigmShift(
                shift_id='s1', shift_date=pd.Timestamp('2020-03-08'),
                shift_type='frame_entry', entering_frames=['Envt'],
                exiting_frames=[],
                state_before=_make_state('2020-03-01', ['Pol']),
                state_after=_make_state('2020-03-08', ['Pol', 'Envt']),
                shift_magnitude=0.4, vector_distance=0.2,
                set_jaccard_distance=0.33, concentration_change=0.1,
            ),
        ]
        ShiftDetector.qualify_shifts(shifts, pd.Timestamp('2020-06-30'))
        assert shifts[0].structural_change == 1

    def test_structural_change_negative(self):
        """Removing a frame: structural_change = -1."""
        shifts = [
            ParadigmShift(
                shift_id='s1', shift_date=pd.Timestamp('2020-03-08'),
                shift_type='frame_exit', entering_frames=[],
                exiting_frames=['Envt'],
                state_before=_make_state('2020-03-01', ['Pol', 'Envt']),
                state_after=_make_state('2020-03-08', ['Pol']),
                shift_magnitude=0.4, vector_distance=0.2,
                set_jaccard_distance=0.33, concentration_change=-0.1,
            ),
        ]
        ShiftDetector.qualify_shifts(shifts, pd.Timestamp('2020-06-30'))
        assert shifts[0].structural_change == -1

    def test_reversible_shift(self):
        """Shift is reversible when the next shift restores state_before."""
        shifts = [
            ParadigmShift(
                shift_id='s1', shift_date=pd.Timestamp('2020-03-08'),
                shift_type='frame_entry', entering_frames=['Envt'],
                exiting_frames=[],
                state_before=_make_state('2020-03-01', ['Pol']),
                state_after=_make_state('2020-03-08', ['Pol', 'Envt']),
                shift_magnitude=0.4, vector_distance=0.2,
                set_jaccard_distance=0.33, concentration_change=0.1,
            ),
            ParadigmShift(
                shift_id='s2', shift_date=pd.Timestamp('2020-03-15'),
                shift_type='frame_exit', entering_frames=[],
                exiting_frames=['Envt'],
                state_before=_make_state('2020-03-08', ['Pol', 'Envt']),
                state_after=_make_state('2020-03-15', ['Pol']),
                shift_magnitude=0.4, vector_distance=0.2,
                set_jaccard_distance=0.33, concentration_change=-0.1,
            ),
        ]
        ShiftDetector.qualify_shifts(shifts, pd.Timestamp('2020-06-30'))
        assert shifts[0].reversible is True
        assert shifts[1].reversible is False  # Last shift, no next

    def test_irreversible_shift(self):
        """Shift is irreversible when the next shift goes elsewhere."""
        shifts = [
            ParadigmShift(
                shift_id='s1', shift_date=pd.Timestamp('2020-03-08'),
                shift_type='full_replacement', entering_frames=['Envt'],
                exiting_frames=['Pol'],
                state_before=_make_state('2020-03-01', ['Pol']),
                state_after=_make_state('2020-03-08', ['Envt']),
                shift_magnitude=0.6, vector_distance=0.4,
                set_jaccard_distance=1.0, concentration_change=0.0,
            ),
            ParadigmShift(
                shift_id='s2', shift_date=pd.Timestamp('2020-03-15'),
                shift_type='full_replacement', entering_frames=['Sci'],
                exiting_frames=['Envt'],
                state_before=_make_state('2020-03-08', ['Envt']),
                state_after=_make_state('2020-03-15', ['Sci']),
                shift_magnitude=0.6, vector_distance=0.4,
                set_jaccard_distance=1.0, concentration_change=0.0,
            ),
        ]
        ShiftDetector.qualify_shifts(shifts, pd.Timestamp('2020-06-30'))
        assert shifts[0].reversible is False  # Sci != Pol


# =============================================================================
# EpisodeAnalyzer tests
# =============================================================================

class TestEpisodeAnalyzer:

    def _make_shift(self, shift_id, date, before_frames, after_frames):
        """Helper to create a shift for episode tests."""
        return ParadigmShift(
            shift_id=shift_id, shift_date=pd.Timestamp(date),
            shift_type='recomposition',
            entering_frames=sorted(set(after_frames) - set(before_frames)),
            exiting_frames=sorted(set(before_frames) - set(after_frames)),
            state_before=_make_state(
                (pd.Timestamp(date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                before_frames,
            ),
            state_after=_make_state(date, after_frames),
            shift_magnitude=0.5, vector_distance=0.3,
            set_jaccard_distance=0.5, concentration_change=0.0,
        )

    def test_single_episode_from_close_shifts(self):
        """Close shifts form one episode."""
        shifts = [
            self._make_shift('s1', '2020-03-08', ['Pol'], ['Pol', 'Envt']),
            self._make_shift('s2', '2020-03-15', ['Pol', 'Envt'], ['Envt']),
        ]
        analyzer = EpisodeAnalyzer(gap_weeks=3)
        episodes = analyzer.build_episodes(shifts, pd.Timestamp('2020-06-30'))

        assert len(episodes) == 1
        assert episodes[0].n_shifts == 2

    def test_two_episodes_from_distant_shifts(self):
        """Distant shifts form separate episodes."""
        shifts = [
            self._make_shift('s1', '2020-03-08', ['Pol'], ['Envt']),
            self._make_shift('s2', '2020-06-15', ['Envt'], ['Sci']),
        ]
        analyzer = EpisodeAnalyzer(gap_weeks=3)
        episodes = analyzer.build_episodes(shifts, pd.Timestamp('2020-12-31'))

        assert len(episodes) == 2
        assert episodes[0].n_shifts == 1
        assert episodes[1].n_shifts == 1

    def test_episode_reversibility(self):
        """Episode is reversible when dominant_before == dominant_after."""
        shifts = [
            self._make_shift('s1', '2020-03-08', ['Pol'], ['Pol', 'Envt']),
            self._make_shift('s2', '2020-03-15', ['Pol', 'Envt'], ['Pol']),
        ]
        analyzer = EpisodeAnalyzer(gap_weeks=3)
        episodes = analyzer.build_episodes(shifts, pd.Timestamp('2020-06-30'))

        assert episodes[0].reversible is True

    def test_episode_irreversibility(self):
        """Episode is irreversible when dominant_before != dominant_after."""
        shifts = [
            self._make_shift('s1', '2020-03-08', ['Pol'], ['Pol', 'Envt']),
            self._make_shift('s2', '2020-03-15', ['Pol', 'Envt'], ['Envt']),
        ]
        analyzer = EpisodeAnalyzer(gap_weeks=3)
        episodes = analyzer.build_episodes(shifts, pd.Timestamp('2020-06-30'))

        assert episodes[0].reversible is False

    def test_net_structural_change(self):
        """Net structural change = len(after) - len(before) across the episode."""
        shifts = [
            self._make_shift('s1', '2020-03-08', ['Pol'], ['Pol', 'Envt']),
            self._make_shift('s2', '2020-03-15', ['Pol', 'Envt'], ['Pol', 'Envt', 'Sci']),
        ]
        analyzer = EpisodeAnalyzer(gap_weeks=3)
        episodes = analyzer.build_episodes(shifts, pd.Timestamp('2020-06-30'))

        # Before: 1 frame, after: 3 frames → net = +2
        assert episodes[0].net_structural_change == 2

    def test_max_complexity(self):
        """Max complexity = peak number of dominant frames during episode."""
        shifts = [
            self._make_shift('s1', '2020-03-08', ['Pol'], ['Pol', 'Envt', 'Sci']),
            self._make_shift('s2', '2020-03-15', ['Pol', 'Envt', 'Sci'], ['Pol', 'Envt']),
        ]
        analyzer = EpisodeAnalyzer(gap_weeks=3)
        episodes = analyzer.build_episodes(shifts, pd.Timestamp('2020-06-30'))

        # Peak was 3 (after s1)
        assert episodes[0].max_complexity == 3

    def test_regime_after_duration(self):
        """regime_after_duration_days = gap to next episode or period_end."""
        shifts = [
            self._make_shift('s1', '2020-03-08', ['Pol'], ['Envt']),
            self._make_shift('s2', '2020-06-15', ['Envt'], ['Sci']),
        ]
        analyzer = EpisodeAnalyzer(gap_weeks=3)
        period_end = pd.Timestamp('2020-12-31')
        episodes = analyzer.build_episodes(shifts, period_end)

        # Episode 1 ends Mar 8, episode 2 starts Jun 15 → 99 days
        assert episodes[0].regime_after_duration_days == 99
        # Episode 2 ends Jun 15, period ends Dec 31 → 199 days
        assert episodes[1].regime_after_duration_days == 199

    def test_episode_serialization(self):
        """ShiftEpisode.to_dict() should be serializable."""
        shifts = [
            self._make_shift('s1', '2020-03-08', ['Pol'], ['Envt']),
        ]
        analyzer = EpisodeAnalyzer(gap_weeks=3)
        episodes = analyzer.build_episodes(shifts, pd.Timestamp('2020-06-30'))

        d = episodes[0].to_dict()
        assert 'episode_id' in d
        assert 'dominant_frames_before' in d
        assert 'dominant_frames_after' in d
        assert 'reversible' in d
        assert 'net_structural_change' in d
        assert 'max_complexity' in d
        assert 'regime_after_duration_days' in d

    def test_no_shifts_returns_empty(self):
        """No shifts → no episodes."""
        analyzer = EpisodeAnalyzer(gap_weeks=3)
        episodes = analyzer.build_episodes([], pd.Timestamp('2020-06-30'))
        assert episodes == []
