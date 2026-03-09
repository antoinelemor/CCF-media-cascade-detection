"""
Core module: config, models, constants.
"""

from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.models import BurstResult, CascadeResult, DetectionResults
from cascade_detector.core.constants import FRAMES, FRAME_COLUMNS, MESSENGERS, MESSENGER_TYPES

__all__ = [
    'DetectorConfig',
    'BurstResult',
    'CascadeResult',
    'DetectionResults',
    'FRAMES',
    'FRAME_COLUMNS',
    'MESSENGERS',
    'MESSENGER_TYPES',
]
