"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
__init__.py (metrics module)

MAIN OBJECTIVE:
---------------
This script initializes the metrics module for cascade detection, providing access to all Phase 2
quantitative metric calculators that operate on Phase 1 indices.

Dependencies:
-------------
- cascade_detector.metrics.temporal_metrics
- cascade_detector.metrics.diversity_metrics
- cascade_detector.metrics.convergence_metrics

MAIN FEATURES:
--------------
1) Exports TemporalMetrics for time-based analysis
2) Exports DiversityMetrics for diversity measurements
3) Exports ConvergenceMetrics for convergence analysis
4) Coordinates Phase 2 metric computation components

Author:
-------
Antoine Lemor
"""

from cascade_detector.metrics.temporal_metrics import TemporalMetrics
# from cascade_detector.metrics.network_metrics import NetworkMetrics, optimize_network_computation
from cascade_detector.metrics.diversity_metrics import DiversityMetrics
from cascade_detector.metrics.convergence_metrics import ConvergenceMetrics

__all__ = [
    'TemporalMetrics',
    # 'NetworkMetrics', 
    'DiversityMetrics',
    'ConvergenceMetrics',
    # 'optimize_network_computation'
]