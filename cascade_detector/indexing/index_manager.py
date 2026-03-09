"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
index_manager.py

MAIN OBJECTIVE:
---------------
Sequential index construction. No multiprocessing, no shared memory,
no resource monitoring. The indexing phase runs once and is I/O-bound anyway.

Author:
-------
Antoine Lemor
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional
from datetime import datetime

import pandas as pd

from cascade_detector.indexing.temporal_indexer import TemporalIndexer
from cascade_detector.indexing.entity_indexer import EntityIndexer
from cascade_detector.indexing.source_indexer import SourceIndexer
from cascade_detector.indexing.frame_indexer import FrameIndexer
from cascade_detector.indexing.emotion_indexer import EmotionIndexer

try:
    from cascade_detector.indexing.geographic_indexer import GeographicIndexer
    HAS_GEOGRAPHIC_INDEXER = True
except ImportError:
    HAS_GEOGRAPHIC_INDEXER = False

logger = logging.getLogger(__name__)


class IndexManager:
    """Builds all indices sequentially from a DataFrame."""

    def __init__(self):
        self.temporal_indexer = TemporalIndexer()
        self.entity_indexer = EntityIndexer(
            resolve_entities=True,
            resolve_locations=True,
        )
        self.source_indexer = SourceIndexer(resolve_authors=True)
        self.frame_indexer = FrameIndexer()
        self.emotion_indexer = EmotionIndexer()

        if HAS_GEOGRAPHIC_INDEXER:
            self.geographic_indexer = GeographicIndexer(use_location_resolver=True)
        else:
            self.geographic_indexer = None

        self.indices: Dict[str, Any] = {}

    def build_all_indices(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Build all indices in parallel from data.

        5 independent indices are built concurrently via ThreadPoolExecutor.
        Geographic index depends on entities and is built after.

        Args:
            data: Processed DataFrame (sentence-level).

        Returns:
            Dictionary mapping index name to index data.
        """
        logger.info(f"Building all indices from {len(data):,} rows...")
        start = datetime.now()

        indices = {}

        indexers = [
            ('temporal', self.temporal_indexer),
            ('entities', self.entity_indexer),
            ('sources', self.source_indexer),
            ('frames', self.frame_indexer),
            ('emotions', self.emotion_indexer),
        ]

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(indexer.build_index, data): name
                for name, indexer in indexers
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    indices[name] = future.result()
                    logger.info(f"  {name} index built.")
                except Exception as e:
                    logger.error(f"  Failed to build {name} index: {e}")
                    indices[name] = {}

        # Geographic index depends on entity index
        if self.geographic_indexer and indices.get('entities'):
            try:
                logger.info("  Building geographic index...")
                indices['geographic'] = self.geographic_indexer.build_index(
                    data, entity_index=indices['entities']
                )
                logger.info("  geographic index built.")
            except Exception as e:
                logger.error(f"  Failed to build geographic index: {e}")
                indices['geographic'] = {}

        elapsed = (datetime.now() - start).total_seconds()
        logger.info(f"All indices built in {elapsed:.1f}s")

        self.indices = indices
        return indices
