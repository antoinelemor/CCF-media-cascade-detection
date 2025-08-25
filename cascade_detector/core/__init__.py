"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
__init__.py (core module)

MAIN OBJECTIVE:
---------------
This script initializes the core module of the cascade detector, exposing the main configuration,
models, and constants for use throughout the framework.

Dependencies:
-------------
- cascade_detector.core.config
- cascade_detector.core.models
- cascade_detector.core.constants

MAIN FEATURES:
--------------
1) Exports DetectorConfig for configuration management
2) Exports all data models (TimeWindow, CascadeCandidate, etc.)
3) Exports global constants (FRAMES, MESSENGERS, etc.)
4) Provides clean API for core components

Author:
-------
Antoine Lemor
"""

from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.models import (
    TimeWindow,
    IndexEntry,
    CascadeCandidate,
    SubIndex,
    Dimension,
    MediaCascade,
    CascadeMetrics
)
from cascade_detector.core.constants import (
    FRAMES,
    MESSENGERS,
    MESSENGER_TYPES
)

__all__ = [
    'DetectorConfig',
    'TimeWindow',
    'IndexEntry',
    'CascadeCandidate',
    'SubIndex',
    'Dimension',
    'MediaCascade',
    'CascadeMetrics',
    'FRAMES',
    'MESSENGERS',
    'MESSENGER_TYPES'
]