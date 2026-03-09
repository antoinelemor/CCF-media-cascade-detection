"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
__init__.py (indexing module)

MAIN OBJECTIVE:
---------------
This script initializes the indexing module of the cascade detector, providing access to all
indexing strategies for efficient data organization and retrieval.

Dependencies:
-------------
- cascade_detector.indexing.base_indexer
- cascade_detector.indexing.temporal_indexer
- cascade_detector.indexing.entity_indexer
- cascade_detector.indexing.source_indexer
- cascade_detector.indexing.frame_indexer
- cascade_detector.indexing.index_manager

MAIN FEATURES:
--------------
1) Exports AbstractIndexer base class
2) Exports specialized indexers (Temporal, Entity, Source, Frame)
3) Exports IndexManager for orchestrated index construction
4) Provides clean API for indexing components

Author:
-------
Antoine Lemor
"""

from cascade_detector.indexing.base_indexer import AbstractIndexer
from cascade_detector.indexing.temporal_indexer import TemporalIndexer
from cascade_detector.indexing.entity_indexer import EntityIndexer
from cascade_detector.indexing.source_indexer import SourceIndexer
from cascade_detector.indexing.frame_indexer import FrameIndexer
from cascade_detector.indexing.index_manager import IndexManager

__all__ = [
    'AbstractIndexer',
    'TemporalIndexer',
    'EntityIndexer',
    'SourceIndexer',
    'FrameIndexer',
    'IndexManager'
]