"""
Post-detection analysis layer for CCF cascade detection.

Provides StabSelImpactAnalyzer for stability selection impact analysis
(cluster → cascade causal relationships via double-weighted treatment variables).

Provides ParadigmShiftAnalyzer for detecting paradigm shifts and attributing
them to specific media cascades and triggering events.

Legacy analyzers (UnifiedImpactAnalyzer, EventImpactAnalyzer) remain importable
for backward compatibility but are no longer part of the pipeline.
"""

from cascade_detector.analysis.impact_analysis import EventImpactAnalyzer  # backward compat
from cascade_detector.analysis.event_occurrence import EventOccurrenceDetector
from cascade_detector.analysis.paradigm_shift import (
    EpisodeAnalyzer,
    ParadigmShiftAnalyzer,
    ShiftEpisode,
)
from cascade_detector.analysis.stabsel_impact import (
    StabSelImpactAnalyzer,
    StabSelImpactResults,
)
from cascade_detector.analysis.unified_impact import (  # backward compat
    UnifiedImpactAnalyzer,
    UnifiedImpactResults,
)

from cascade_detector.core.models import CascadeAttribution

__all__ = ['EventOccurrenceDetector',
           'ParadigmShiftAnalyzer', 'EpisodeAnalyzer', 'ShiftEpisode',
           'StabSelImpactAnalyzer', 'StabSelImpactResults',
           'UnifiedImpactAnalyzer', 'UnifiedImpactResults',  # backward compat
           'CascadeAttribution']
