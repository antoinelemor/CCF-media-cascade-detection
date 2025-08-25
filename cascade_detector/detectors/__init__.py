"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
__init__.py (detectors module)

MAIN OBJECTIVE:
---------------
This script initializes the detectors module containing all Phase 3 pattern detection algorithms
that operate on Phase 1 indices and Phase 2 metrics to identify media cascades.

Dependencies:
-------------
- cascade_detector.detectors.base_detector
- cascade_detector.detectors.signal_aggregator

MAIN FEATURES:
--------------
1) Exports BaseDetector abstract class and DetectionContext
2) Exports SignalAggregator and AggregatedSignal
3) Provides access to all cascade detection algorithms
4) Coordinates Phase 3 pattern detection components

Author:
-------
Antoine Lemor
"""

from cascade_detector.detectors.base_detector import BaseDetector, DetectionContext
from cascade_detector.detectors.signal_aggregator import SignalAggregator, AggregatedSignal

__all__ = [
    'BaseDetector',
    'DetectionContext',
    'SignalAggregator',
    'AggregatedSignal',
]