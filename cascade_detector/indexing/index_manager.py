"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
index_manager.py

MAIN OBJECTIVE:
---------------
This script orchestrates the parallel construction of all indices for cascade detection, managing
resource allocation, worker coordination, and optimized data transfer between processes.

Dependencies:
-------------
- pandas
- logging
- typing
- concurrent.futures
- pathlib
- pickle
- json
- datetime
- tqdm
- multiprocessing
- psutil

MAIN FEATURES:
--------------
1) Parallel index construction with intelligent worker allocation
2) Resource monitoring and dynamic adjustment
3) Shared memory optimization for large index transfers
4) Progress tracking and performance metrics
5) Cache management and index serialization

Author:
-------
Antoine Lemor
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import pickle
import json
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
import psutil
import time

# Import shared memory transfer utilities for large indices
try:
    from cascade_detector.utils.shared_memory_transfer import (
        optimize_large_index_transfer, 
        retrieve_cached_index,
        LargeIndexCache
    )
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logging.warning("Shared memory transfer not available, using standard serialization")

# Import indexers - delay imports to avoid circular dependencies
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
from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.exceptions import IndexingError

# Force spawn method for macOS compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

logger = logging.getLogger(__name__)


class IndexManager:
    """
    Manages and orchestrates all indexing operations.
    Handles parallel processing for M4 Ultra optimization.
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize index manager.
        
        Args:
            config: Detector configuration
        """
        self.config = config or DetectorConfig()
        
        # Calculate optimal worker configuration for M4 Max
        cpu_count = mp.cpu_count()
        # For M4 Max (16 cores), optimal configuration:
        # - Entity indexer: 16 workers (actual core count for best performance)
        # - Geographic indexer: 16 workers (actual core count)
        # - Main process pool: 5-7 workers (one per indexer + overhead)
        
        entity_workers = cpu_count  # Use actual 16 cores only (no multiplication)
        geo_workers = cpu_count  # 16 for M4 Max
        
        logger.info(f"Worker configuration for {cpu_count} CPU cores:")
        logger.info(f"  • Entity indexer: {entity_workers} workers")
        logger.info(f"  • Geographic indexer: {geo_workers} workers")
        
        # Initialize indexers with optimizations
        self.temporal_indexer = TemporalIndexer()
        # Enhanced entity indexer with resolution and M4 Max optimization
        self.entity_indexer = EntityIndexer(
            resolve_entities=True,
            resolve_locations=True,  # Enable location resolution
            n_workers=entity_workers  # Use optimized worker count
        )
        # Enhanced source indexer with author resolution
        self.source_indexer = SourceIndexer(
            resolve_authors=True
        )
        self.frame_indexer = FrameIndexer()
        self.emotion_indexer = EmotionIndexer()
        
        # Initialize geographic indexer if available
        if HAS_GEOGRAPHIC_INDEXER:
            self.geographic_indexer = GeographicIndexer(
                use_location_resolver=True,
                n_workers=geo_workers  # Use optimized worker count
            )
            logger.info(f"GeographicIndexer initialized with {geo_workers} workers")
        else:
            self.geographic_indexer = None
            logger.info("GeographicIndexer not available")
        
        # Storage for all indices
        self.indices = {}
        
        # Metadata
        self.metadata = {
            'created': None,
            'updated': None,
            'data_range': {},
            'statistics': {}
        }
    
    def build_all_indices(self, data: pd.DataFrame, 
                         parallel: bool = True,
                         save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Build all indices from data.
        
        Args:
            data: Input DataFrame
            parallel: Whether to use parallel processing
            save_path: Optional path to save indices
            
        Returns:
            Dictionary with all indices
        """
        logger.info(f"Building all indices from {len(data):,} rows...")
        
        # Monitor initial resources
        self._log_resource_usage("Initial state")
        
        start_time = datetime.now()
        
        if parallel and self.config.use_parallel:
            indices = self._build_parallel(data)
        else:
            indices = self._build_sequential(data)
        
        # Store indices
        self.indices = indices
        
        # Update metadata
        self._update_metadata(data)
        
        # Save if requested
        if save_path:
            self.save_indices(save_path)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"All indices built in {elapsed:.2f} seconds")
        
        # Log final resource usage
        self._log_resource_usage("Final state")
        
        # Log statistics
        self._log_statistics()
        
        # Calculate and log performance metrics
        self._log_performance_metrics(elapsed, len(data))
        
        return indices
    
    def _build_parallel(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Build indices in parallel with optimized two-phase approach."""
        # Get system info
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        logger.info(f"Building indices in MASSIVE PARALLEL mode")
        logger.info(f"System resources:")
        logger.info(f"  • CPU cores: {cpu_count} (M4 Max: 14 cores)")
        logger.info(f"  • Total memory: {memory_gb:.1f} GB")
        logger.info(f"  • Available memory: {available_memory_gb:.1f} GB")
        logger.info(f"  • Workers configured: {self.config.n_workers}")
        logger.info(f"Data size: {len(data):,} rows")
        logger.info(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        indices = {}
        
        # Phase 1: Build independent indices in parallel (can all run simultaneously)
        independent_indexers = [
            ('temporal', self.temporal_indexer),
            ('sources', self.source_indexer),
            ('frames', self.frame_indexer),
            ('emotions', self.emotion_indexer)
        ]
        
        # Phase 1: Process ALL indices in parallel (5 indexers total)
        # Use ProcessPoolExecutor for CPU-bound indexing tasks
        total_indexers = len(independent_indexers) + 1  # +1 for entity indexer
        
        # For M4 Max (14 cores), use optimal worker count
        # Each indexer can run on its own process
        n_workers = min(total_indexers, cpu_count, self.config.n_workers)
        
        logger.info(f"  Phase 1: Using {n_workers} parallel processes for {total_indexers} indexers")
        logger.info(f"  This will utilize {n_workers}/{cpu_count} CPU cores")
        
        # Use ProcessPoolExecutor for true parallelism (bypass GIL)
        # Optimize for large data serialization
        import sys
        mp_context = 'spawn' if sys.platform == 'darwin' else 'forkserver'
        
        # Configure for better performance with large objects
        import pickle
        import gc
        gc.collect()  # Clean up before spawning processes
        
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context(mp_context)) as executor:
            futures_phase1 = {}
            
            # Submit all indexers with proper serialization
            # Each indexer runs in its own process for true parallelism
            for name, indexer in independent_indexers:
                logger.info(f"  Submitting {name} indexer to process pool...")
                # Use specialized wrapper based on indexer type
                if name == 'sources':
                    # SourceIndexer needs special handling for author resolution
                    futures_phase1[executor.submit(
                        _build_source_index_wrapper, 
                        data,
                        getattr(indexer, 'resolve_authors', True)
                    )] = name
                else:
                    # Standard indexers
                    futures_phase1[executor.submit(_build_index_wrapper, name, indexer.__class__, data)] = name
            
            # Also submit entity indexer (it has its own internal parallelization)
            logger.info(f"  Submitting entity indexer (internal workers: {self.entity_indexer.n_workers})...")
            # Entity indexer modifies data, so it needs a copy
            futures_phase1[executor.submit(
                _build_entity_index_wrapper, 
                data.copy(), 
                self.entity_indexer.resolve_entities,
                self.entity_indexer.resolve_locations,
                self.entity_indexer.n_workers
            )] = 'entities'
            
            # Wait for phase 1 completion with progress and timing
            completed_indices = []
            failed_indices = []
            index_times = {}
            
            with tqdm(total=len(futures_phase1), desc="Phase 1: Core indices", unit="index") as pbar:
                for future in as_completed(futures_phase1, timeout=3600):
                    index_name = futures_phase1[future]
                    try:
                        # Get result with timing
                        start_retrieve = time.time()
                        result = future.result(timeout=60)  # 60s timeout for retrieval
                        
                        # Check if result is cached (for large indices)
                        if CACHE_AVAILABLE and isinstance(result, dict) and result.get('_cached'):
                            logger.info(f"  {index_name} index using cache transfer (size: {result.get('_size_mb', 0):.0f} MB)")
                            # Retrieve from cache
                            result = retrieve_cached_index(result)
                        
                        retrieve_time = time.time() - start_retrieve
                        
                        indices[index_name] = result
                        completed_indices.append(index_name)
                        index_times[index_name] = retrieve_time
                        
                        # Log success with timing
                        logger.info(f"  ✓ {index_name} index built (retrieve: {retrieve_time:.1f}s)")
                        
                        # Update progress bar with current status
                        pbar.set_postfix({
                            'completed': index_name,
                            'retrieve_time': f'{retrieve_time:.1f}s'
                        })
                    except Exception as e:
                        logger.error(f"  ✗ Failed to build {index_name} index: {e}")
                        indices[index_name] = {}
                        failed_indices.append(index_name)
                    pbar.update(1)
            
            # Log summary
            if index_times:
                total_retrieve = sum(index_times.values())
                logger.info(f"  Phase 1 summary: {len(completed_indices)} succeeded, {len(failed_indices)} failed")
                logger.info(f"  Total data transfer time: {total_retrieve:.1f}s")
                if total_retrieve > 10:
                    logger.warning(f"  ⚠️ Large data transfer overhead detected ({total_retrieve:.1f}s)")
                    logger.info(f"  Consider using shared memory or file-based transfer for large indices")
        
        # Phase 2: Build dependent indices (geographic needs entity index)
        if self.geographic_indexer and 'entities' in indices and indices['entities']:
            logger.info("Phase 2: Building geographic index (depends on entity index)...")
            logger.info(f"  • Geographic indexer will use its own {self.geographic_indexer.n_workers} workers")
            
            # Monitor resource usage
            process = psutil.Process()
            logger.info(f"  • Current memory usage: {process.memory_info().rss / 1024**2:.1f} MB")
            logger.info(f"  • Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")
            
            # Run geographic indexer with its own parallelization
            with tqdm(total=1, desc="Phase 2: Geographic index", unit="index") as pbar:
                try:
                    start_time = time.time()
                    indices['geographic'] = self.geographic_indexer.build_index(
                        data, 
                        entity_index=indices['entities']
                    )
                    elapsed = time.time() - start_time
                    logger.info(f"  ✓ geographic index built in {elapsed:.2f}s")
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"  ✗ Failed to build geographic index: {e}")
                    indices['geographic'] = {}
                    pbar.update(1)
        
        return indices
    
    def _build_sequential(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Build indices sequentially."""
        logger.info("Building indices sequentially...")
        
        indices = {}
        
        # Build each index
        try:
            logger.info("  Building temporal index...")
            indices['temporal'] = self.temporal_indexer.build_index(data)
        except Exception as e:
            logger.error(f"  Failed to build temporal index: {e}")
            indices['temporal'] = {}
        
        try:
            logger.info("  Building entity index...")
            indices['entities'] = self.entity_indexer.build_index(data)
        except Exception as e:
            logger.error(f"  Failed to build entity index: {e}")
            indices['entities'] = {}
        
        try:
            logger.info("  Building source index...")
            indices['sources'] = self.source_indexer.build_index(data)
        except Exception as e:
            logger.error(f"  Failed to build source index: {e}")
            indices['sources'] = {}
        
        try:
            logger.info("  Building frame index...")
            indices['frames'] = self.frame_indexer.build_index(data)
        except Exception as e:
            logger.error(f"  Failed to build frame index: {e}")
            indices['frames'] = {}
        
        try:
            logger.info("  Building emotion index...")
            indices['emotions'] = self.emotion_indexer.build_index(data)
        except Exception as e:
            logger.error(f"  Failed to build emotion index: {e}")
            indices['emotions'] = {}
        
        # Build geographic index if available (needs entity index)
        if self.geographic_indexer and 'entities' in indices:
            try:
                logger.info("  Building geographic index...")
                indices['geographic'] = self.geographic_indexer.build_index(
                    data,
                    entity_index=indices['entities']
                )
            except Exception as e:
                logger.error(f"  Failed to build geographic index: {e}")
                indices['geographic'] = {}
        
        return indices
    
    def update_indices(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Update existing indices with new data.
        
        Args:
            new_data: New data to add
            
        Returns:
            Updated indices
        """
        logger.info(f"Updating indices with {len(new_data):,} new rows...")
        
        if not self.indices:
            logger.warning("No existing indices found, building from scratch")
            return self.build_all_indices(new_data)
        
        # Update each index
        updates = {}
        
        n_indexers = 6 if self.geographic_indexer else 5
        with ThreadPoolExecutor(max_workers=min(n_indexers, self.config.n_workers)) as executor:
            futures = {
                executor.submit(self.temporal_indexer.update_index, new_data): 'temporal',
                executor.submit(self.entity_indexer.update_index, new_data): 'entities',
                executor.submit(self.source_indexer.update_index, new_data): 'sources',
                executor.submit(self.frame_indexer.update_index, new_data): 'frames',
                executor.submit(self.emotion_indexer.update_index, new_data): 'emotions'
            }
            
            if self.geographic_indexer:
                futures[executor.submit(self.geographic_indexer.update_index, new_data)] = 'geographic'
            
            for future in as_completed(futures):
                index_name = futures[future]
                try:
                    updates[index_name] = future.result()
                    logger.info(f"  ✓ {index_name} index updated")
                except Exception as e:
                    logger.error(f"  ✗ Failed to update {index_name} index: {e}")
        
        # Update stored indices
        self.indices.update(updates)
        
        # Update metadata
        self.metadata['updated'] = datetime.now().isoformat()
        
        return self.indices
    
    def query(self, index_name: str, criteria: Dict[str, Any]) -> List[Any]:
        """
        Query a specific index.
        
        Args:
            index_name: Name of index to query
            criteria: Query criteria
            
        Returns:
            Query results
        """
        if index_name not in self.indices:
            raise IndexingError(f"Index '{index_name}' not found")
        
        if index_name == 'temporal':
            return self.temporal_indexer.query_index(criteria)
        elif index_name == 'entities':
            return self.entity_indexer.query_index(criteria)
        elif index_name == 'sources':
            return self.source_indexer.query_index(criteria)
        elif index_name == 'frames':
            return self.frame_indexer.query_index(criteria)
        elif index_name == 'emotions':
            return self.emotion_indexer.query_index(criteria)
        elif index_name == 'geographic' and self.geographic_indexer:
            return self.geographic_indexer.query_index(criteria)
        else:
            raise IndexingError(f"Unknown index type: {index_name}")
    
    def save_indices(self, path: str, format: str = 'pickle') -> None:
        """
        Save all indices to disk.
        
        Args:
            path: Directory path to save indices
            format: Save format ('pickle' or 'json')
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving indices to {path}...")
        
        # Save each index
        indexers_to_save = [
            ('temporal', self.temporal_indexer),
            ('entities', self.entity_indexer),
            ('sources', self.source_indexer),
            ('frames', self.frame_indexer),
            ('emotions', self.emotion_indexer)
        ]
        
        if self.geographic_indexer:
            indexers_to_save.append(('geographic', self.geographic_indexer))
        
        for name, indexer in indexers_to_save:
            index_path = path / f"{name}_index.{format}"
            try:
                indexer.save_index(str(index_path), format)
                logger.info(f"  ✓ {name} index saved")
            except Exception as e:
                logger.error(f"  ✗ Failed to save {name} index: {e}")
        
        # Save metadata
        meta_path = path / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        logger.info(f"All indices saved to {path}")
    
    def load_indices(self, path: str, format: str = 'pickle') -> Dict[str, Any]:
        """
        Load indices from disk.
        
        Args:
            path: Directory path containing indices
            format: Format of saved indices
            
        Returns:
            Loaded indices
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Index directory not found: {path}")
        
        logger.info(f"Loading indices from {path}...")
        
        indices = {}
        
        # Load each index
        indexers_to_load = [
            ('temporal', self.temporal_indexer),
            ('entities', self.entity_indexer),
            ('sources', self.source_indexer),
            ('frames', self.frame_indexer),
            ('emotions', self.emotion_indexer)
        ]
        
        if self.geographic_indexer:
            indexers_to_load.append(('geographic', self.geographic_indexer))
        
        for name, indexer in indexers_to_load:
            index_path = path / f"{name}_index.{format}"
            if index_path.exists():
                try:
                    indices[name] = indexer.load_index(str(index_path), format)
                    logger.info(f"  ✓ {name} index loaded")
                except Exception as e:
                    logger.error(f"  ✗ Failed to load {name} index: {e}")
                    indices[name] = {}
            else:
                logger.warning(f"  ⚠ {name} index file not found")
                indices[name] = {}
        
        # Load metadata
        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        
        self.indices = indices
        logger.info(f"Indices loaded from {path}")
        
        return indices
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all indices."""
        stats = {
            'metadata': self.metadata,
            'indices': {}
        }
        
        # Get stats for each index
        indexers_for_stats = [
            ('temporal', self.temporal_indexer),
            ('entities', self.entity_indexer),
            ('sources', self.source_indexer),
            ('frames', self.frame_indexer),
            ('emotions', self.emotion_indexer)
        ]
        
        if self.geographic_indexer:
            indexers_for_stats.append(('geographic', self.geographic_indexer))
        
        for name, indexer in indexers_for_stats:
            stats['indices'][name] = indexer.get_statistics()
        
        return stats
    
    def _update_metadata(self, data: pd.DataFrame) -> None:
        """Update metadata after building indices."""
        # Convert dates for metadata
        if 'date_converted' not in data.columns:
            data['date_converted'] = pd.to_datetime(
                data['date'].str[6:10] + '-' +
                data['date'].str[0:2] + '-' +
                data['date'].str[3:5],
                format='%Y-%m-%d',
                errors='coerce'
            )
        
        self.metadata.update({
            'created': datetime.now().isoformat(),
            'data_range': {
                'start': data['date_converted'].min().isoformat() if not data['date_converted'].isna().all() else None,
                'end': data['date_converted'].max().isoformat() if not data['date_converted'].isna().all() else None
            },
            'statistics': {
                'n_rows': len(data),
                'n_articles': data['doc_id'].nunique() if 'doc_id' in data.columns else 0,
                'n_journalists': data['author'].nunique() if 'author' in data.columns else 0,
                'n_media': data['media'].nunique() if 'media' in data.columns else 0
            }
        })
    
    def _log_statistics(self) -> None:
        """Log index statistics."""
        logger.info("\nIndex Statistics:")
        logger.info("-" * 40)
        
        if 'temporal' in self.indices:
            n_frames = len(self.indices['temporal'])
            logger.info(f"Temporal: {n_frames} frames indexed")
        
        if 'entities' in self.indices:
            # Entity index has entities as top-level keys directly
            entity_index = self.indices['entities']
            n_entities = len(entity_index) if isinstance(entity_index, dict) else 0
            logger.info(f"Entities: {n_entities} unique entities")
        
        if 'sources' in self.indices:
            n_articles = len(self.indices['sources'].get('article_profiles', {}))
            logger.info(f"Sources: {n_articles} articles profiled")
        
        if 'frames' in self.indices:
            n_articles = len(self.indices['frames'].get('article_frames', {}))
            logger.info(f"Frames: {n_articles} articles analyzed")
        
        if 'emotions' in self.indices:
            n_articles = len(self.indices['emotions'].get('article_emotions', {}))
            stats = self.indices['emotions'].get('emotion_statistics', {})
            mean_sentiment = stats.get('mean_sentiment', 0)
            logger.info(f"Emotions: {n_articles} articles, mean sentiment: {mean_sentiment:.3f}")
        
        if 'geographic' in self.indices:
            # Get stats from the geographic index directly
            geo_index = self.indices['geographic']
            n_locations = len(geo_index.get('locations', {}))
            
            # Try to get from statistics first, fallback to counting
            geo_stats = geo_index.get('statistics', {})
            if not geo_stats:
                # Count directly from index
                n_clusters = len(geo_index.get('location_clusters', {}))
            else:
                n_locations = geo_stats.get('total_locations', n_locations)
                n_clusters = geo_stats.get('n_clusters', 0)
            
            # Also show cascade indicators if available
            cascade_ind = geo_index.get('cascade_indicators', {})
            cascade_score = cascade_ind.get('overall_focus_score', 0)
            
            logger.info(f"Geographic: {n_locations} locations, cascade score: {cascade_score:.3f}")
        
        logger.info("-" * 40)
    
    def _log_resource_usage(self, phase: str) -> None:
        """Log current resource usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        vm = psutil.virtual_memory()
        
        logger.info(f"\nResource Usage - {phase}:")
        logger.info(f"  • Process memory: {memory_info.rss / 1024**2:.1f} MB")
        logger.info(f"  • System memory: {vm.used / 1024**3:.1f}/{vm.total / 1024**3:.1f} GB ({vm.percent:.1f}%)")
        logger.info(f"  • Available memory: {vm.available / 1024**3:.1f} GB")
        logger.info(f"  • CPU usage: {psutil.cpu_percent(interval=0.1):.1f}%")
        
        # Log per-core CPU usage for M4 Max monitoring
        per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
        if per_cpu:
            active_cores = sum(1 for cpu in per_cpu if cpu > 50)
            logger.info(f"  • Active CPU cores: {active_cores}/{len(per_cpu)}")
            logger.info(f"  • Per-core usage: {[f'{cpu:.0f}%' for cpu in per_cpu]}")
    
    def _log_performance_metrics(self, elapsed: float, n_rows: int) -> None:
        """Log performance metrics."""
        rows_per_second = n_rows / elapsed if elapsed > 0 else 0
        indices_per_second = len(self.indices) / elapsed if elapsed > 0 else 0
        
        logger.info(f"\nPerformance Metrics:")
        logger.info(f"  • Total time: {elapsed:.2f} seconds")
        logger.info(f"  • Rows processed: {n_rows:,}")
        logger.info(f"  • Processing rate: {rows_per_second:,.0f} rows/second")
        logger.info(f"  • Indices built: {len(self.indices)}")
        logger.info(f"  • Index build rate: {indices_per_second:.2f} indices/second")
        
        # Calculate efficiency
        cpu_count = mp.cpu_count()
        theoretical_speedup = cpu_count
        actual_speedup = self._estimate_speedup(elapsed)
        efficiency = (actual_speedup / theoretical_speedup) * 100 if theoretical_speedup > 0 else 0
        
        logger.info(f"  • Theoretical speedup: {theoretical_speedup}x (with {cpu_count} cores)")
        logger.info(f"  • Estimated actual speedup: {actual_speedup:.1f}x")
        logger.info(f"  • Parallel efficiency: {efficiency:.1f}%")
    
    def _estimate_speedup(self, parallel_time: float) -> float:
        """Estimate speedup from parallelization."""
        # This is a rough estimate based on typical single-threaded performance
        # Assumes single-threaded would take approximately n_indexers * parallel_time
        n_indexers = len(self.indices)
        estimated_sequential_time = n_indexers * parallel_time * 0.8  # 0.8 factor for overhead
        return estimated_sequential_time / parallel_time if parallel_time > 0 else 1.0


# Wrapper functions for ProcessPoolExecutor serialization
def _build_source_index_wrapper(data: pd.DataFrame, resolve_authors: bool) -> Dict[str, Any]:
    """
    Specialized wrapper for SourceIndexer with author resolution.
    
    Args:
        data: DataFrame to process
        resolve_authors: Whether to resolve authors
        
    Returns:
        Built source index
    """
    import logging
    import psutil
    import os
    import time
    from cascade_detector.indexing.source_indexer import SourceIndexer
    
    # Setup logging in subprocess
    logger = logging.getLogger(__name__)
    
    # Log process info
    process = psutil.Process(os.getpid())
    logger.info(f"[PID {os.getpid()}] Building source index")
    logger.info(f"  • Resolve authors: {resolve_authors}")
    logger.info(f"  • Memory usage: {process.memory_info().rss / 1024**2:.1f} MB")
    
    try:
        # Instantiate source indexer with configuration
        source_indexer = SourceIndexer(resolve_authors=resolve_authors)
        
        # Build index
        start_time = time.time()
        index = source_indexer.build_index(data)
        elapsed = time.time() - start_time
        
        logger.info(f"[PID {os.getpid()}] Source index built in {elapsed:.2f}s")
        logger.info(f"  • Articles profiled: {len(index.get('article_profiles', {}))}")  
        logger.info(f"  • Journalists profiled: {len(index.get('journalist_profiles', {}))}") 
        logger.info(f"  • Media profiled: {len(index.get('media_profiles', {}))}") 
        
        # Check if we should cache large indices
        try:
            from cascade_detector.utils.shared_memory_transfer import optimize_large_index_transfer
            
            # Check index size
            import sys
            index_size_mb = sys.getsizeof(index) / (1024**2)
            
            # For very large indices, use cache to avoid serialization overhead
            if index_size_mb > 50 or len(index.get('article_profiles', {})) > 50000:
                logger.info(f"[PID {os.getpid()}] Large index detected ({index_size_mb:.0f} MB), using cache transfer")
                return optimize_large_index_transfer(index, 'source')
        except ImportError:
            pass  # Cache not available, return normally
        
        return index
        
    except Exception as e:
        logger.error(f"[PID {os.getpid()}] Error building source index: {e}")
        raise


def _build_index_wrapper(indexer_name: str, indexer_class, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Wrapper function to build index in a separate process.
    This allows proper serialization with ProcessPoolExecutor.
    
    Args:
        indexer_name: Name of the indexer
        indexer_class: Class of the indexer to instantiate
        data: DataFrame to process
        
    Returns:
        Built index dictionary
    """
    import logging
    import psutil
    import os
    import time
    
    # Setup logging in subprocess
    logger = logging.getLogger(__name__)
    
    # Log process info
    process = psutil.Process(os.getpid())
    logger.info(f"[PID {os.getpid()}] Building {indexer_name} index")
    logger.info(f"  • CPU affinity: {process.cpu_affinity() if hasattr(process, 'cpu_affinity') else 'N/A'}")
    logger.info(f"  • Memory usage: {process.memory_info().rss / 1024**2:.1f} MB")
    
    try:
        # Instantiate indexer in subprocess
        indexer = indexer_class()
        
        # Build index
        start_time = time.time()
        index = indexer.build_index(data)
        elapsed = time.time() - start_time
        
        logger.info(f"[PID {os.getpid()}] {indexer_name} index built in {elapsed:.2f}s")
        return index
        
    except Exception as e:
        logger.error(f"[PID {os.getpid()}] Error building {indexer_name} index: {e}")
        raise


def _build_entity_index_wrapper(data: pd.DataFrame, 
                               resolve_entities: bool,
                               resolve_locations: bool,
                               n_workers: int) -> Dict[str, Any]:
    """
    Wrapper function to build entity index in a separate process.
    Handles the special requirements of entity indexer.
    
    Args:
        data: DataFrame to process
        resolve_entities: Whether to resolve entities
        resolve_locations: Whether to resolve locations
        n_workers: Number of workers for internal parallelization
        
    Returns:
        Built entity index
    """
    import logging
    import psutil
    import os
    import time
    from cascade_detector.indexing.entity_indexer import EntityIndexer
    
    # Setup logging in subprocess
    logger = logging.getLogger(__name__)
    
    # Log process info
    process = psutil.Process(os.getpid())
    logger.info(f"[PID {os.getpid()}] Building entity index with resolution")
    logger.info(f"  • Resolve entities: {resolve_entities}")
    logger.info(f"  • Resolve locations: {resolve_locations}")
    logger.info(f"  • Internal workers: {n_workers}")
    logger.info(f"  • Memory usage: {process.memory_info().rss / 1024**2:.1f} MB")
    
    try:
        # Instantiate entity indexer with configuration
        entity_indexer = EntityIndexer(
            resolve_entities=resolve_entities,
            resolve_locations=resolve_locations,
            n_workers=n_workers
        )
        
        # Build index
        start_time = time.time()
        index = entity_indexer.build_index(data)
        elapsed = time.time() - start_time
        
        logger.info(f"[PID {os.getpid()}] Entity index built in {elapsed:.2f}s")
        # Entity index has entities as top-level keys directly
        logger.info(f"  • Entities found: {len(index) if isinstance(index, dict) else 0}")
        # Locations are a subset of entities with type 'LOC'
        n_locations = sum(1 for e in (index or {}).values() if isinstance(e, dict) and e.get('type') == 'LOC')
        logger.info(f"  • Locations found: {n_locations}")
        
        # Check if we should cache large indices
        try:
            from cascade_detector.utils.shared_memory_transfer import optimize_large_index_transfer
            
            # Check index size
            import sys
            index_size_mb = sys.getsizeof(index) / (1024**2)
            
            # For very large entity indices, use cache
            # Entity index has entities as top-level keys directly
            if index_size_mb > 50 or len(index) > 10000:
                logger.info(f"[PID {os.getpid()}] Large entity index ({index_size_mb:.0f} MB), using cache transfer")
                return optimize_large_index_transfer(index, 'entity')
        except ImportError:
            pass  # Cache not available
        
        return index
        
    except Exception as e:
        logger.error(f"[PID {os.getpid()}] Error building entity index: {e}")
        raise