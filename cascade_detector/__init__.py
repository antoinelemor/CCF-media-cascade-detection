"""
CCF Media Cascade Detection Package

Public API:
    - CascadeDetectionPipeline: end-to-end cascade detection
    - DetectorConfig: configuration
    - DetectionResults, CascadeResult, BurstResult: data models
    - StabSelImpactAnalyzer: stability selection impact analysis (cluster → cascade)
"""

__version__ = "0.3.0"

from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.models import BurstResult, CascadeResult, DetectionResults
from cascade_detector.pipeline import CascadeDetectionPipeline
from cascade_detector.analysis.stabsel_impact import StabSelImpactAnalyzer

__all__ = [
    'DetectorConfig',
    'BurstResult',
    'CascadeResult',
    'DetectionResults',
    'CascadeDetectionPipeline',
    'StabSelImpactAnalyzer',
]
