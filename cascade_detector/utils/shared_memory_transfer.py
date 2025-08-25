"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
shared_memory_transfer.py

MAIN OBJECTIVE:
---------------
This script provides shared memory and disk cache systems for efficient large index transfers
between processes, preserving all data for scientific accuracy while optimizing transfer speed.

Dependencies:
-------------
- pickle
- tempfile
- hashlib
- logging
- time
- os
- pathlib
- typing
- multiprocessing
- numpy
- json

MAIN FEATURES:
--------------
1) Shared memory allocation for inter-process communication
2) Disk-based caching with memory mapping
3) Automatic cache cleanup and management
4) Hash-based cache validation
5) Optimized serialization for large scientific data structures

Author:
-------
Antoine Lemor
"""

import pickle
import tempfile
import hashlib
import logging
import time
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import json

logger = logging.getLogger(__name__)


class LargeIndexCache:
    """
    Cache system for large indices to avoid expensive serialization.
    Uses disk-based cache with memory mapping for fastest access.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache system."""
        if cache_dir is None:
            cache_dir = Path(tempfile.gettempdir()) / "cascade_detector_cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean old cache files (older than 1 hour)
        self._clean_old_cache()
        
        logger.debug(f"Cache directory: {self.cache_dir}")
    
    def _clean_old_cache(self):
        """Remove cache files older than 1 hour."""
        current_time = time.time()
        for cache_file in self.cache_dir.glob("*.pkl"):
            if current_time - cache_file.stat().st_mtime > 3600:
                try:
                    cache_file.unlink()
                    logger.debug(f"Removed old cache: {cache_file.name}")
                except:
                    pass
    
    def _get_cache_key(self, data_hash: str, index_type: str) -> str:
        """Generate cache key."""
        return f"{index_type}_{data_hash}"
    
    def save_index(self, index: Dict[str, Any], index_type: str, data_hash: Optional[str] = None) -> str:
        """
        Save index to cache and return cache path.
        
        Args:
            index: Index data to save
            index_type: Type of index (e.g., 'source', 'entity')
            data_hash: Optional hash for cache key
            
        Returns:
            Cache file path
        """
        if data_hash is None:
            # Generate hash from index metadata
            data_hash = hashlib.md5(str(index.get('metadata', {})).encode()).hexdigest()[:8]
        
        cache_key = self._get_cache_key(data_hash, index_type)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        # Check if already cached
        if cache_path.exists():
            logger.debug(f"Index already cached: {cache_path.name}")
            return str(cache_path)
        
        # Save with highest protocol for speed
        start = time.time()
        with open(cache_path, 'wb') as f:
            pickle.dump(index, f, protocol=5)
        
        save_time = time.time() - start
        size_mb = cache_path.stat().st_size / (1024**2)
        
        logger.info(f"Cached {index_type} index: {size_mb:.1f} MB in {save_time:.2f}s")
        logger.debug(f"Cache path: {cache_path}")
        
        return str(cache_path)
    
    def load_index(self, cache_path: str) -> Dict[str, Any]:
        """
        Load index from cache.
        
        Args:
            cache_path: Path to cached index
            
        Returns:
            Loaded index
        """
        cache_path = Path(cache_path)
        
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}")
        
        start = time.time()
        with open(cache_path, 'rb') as f:
            index = pickle.load(f)
        
        load_time = time.time() - start
        size_mb = cache_path.stat().st_size / (1024**2)
        
        logger.debug(f"Loaded cached index: {size_mb:.1f} MB in {load_time:.2f}s")
        
        return index


class SharedMemoryTransfer:
    """
    Use shared memory for zero-copy transfer of large numpy arrays.
    Useful for influence networks and other matrix data.
    """
    
    def __init__(self):
        """Initialize shared memory manager."""
        self.shared_blocks = {}
    
    def share_array(self, array: np.ndarray, name: str) -> Tuple[str, tuple, str]:
        """
        Put array in shared memory.
        
        Args:
            array: Numpy array to share
            name: Unique name for this array
            
        Returns:
            Tuple of (shared_memory_name, shape, dtype)
        """
        # Create shared memory block
        shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
        
        # Copy array to shared memory
        shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        shared_array[:] = array[:]
        
        # Store reference
        self.shared_blocks[name] = shm
        
        logger.debug(f"Shared array '{name}': {array.nbytes / (1024**2):.1f} MB")
        
        return shm.name, array.shape, str(array.dtype)
    
    def get_array(self, shm_name: str, shape: tuple, dtype: str) -> np.ndarray:
        """
        Retrieve array from shared memory.
        
        Args:
            shm_name: Shared memory block name
            shape: Array shape
            dtype: Array dtype
            
        Returns:
            Numpy array (view of shared memory)
        """
        # Attach to existing shared memory
        shm = shared_memory.SharedMemory(name=shm_name)
        
        # Create array view
        array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        
        return array
    
    def cleanup(self):
        """Release all shared memory blocks."""
        for name, shm in self.shared_blocks.items():
            try:
                shm.close()
                shm.unlink()
                logger.debug(f"Released shared memory: {name}")
            except:
                pass
        self.shared_blocks.clear()


def optimize_large_index_transfer(index: Dict[str, Any], index_type: str) -> Dict[str, Any]:
    """
    Optimize transfer of large index between processes.
    
    For indices > 100MB, uses disk cache.
    For numpy arrays, uses shared memory.
    
    Args:
        index: Index to optimize
        index_type: Type of index
        
    Returns:
        Optimized index or cache reference
    """
    # Check if index has large numpy arrays
    has_large_arrays = False
    array_size_mb = 0
    
    if 'influence_network' in index and isinstance(index.get('influence_network'), np.ndarray):
        array_size_mb = index['influence_network'].nbytes / (1024**2)
        has_large_arrays = array_size_mb > 10
    
    # Estimate total size
    import sys
    total_size_mb = sys.getsizeof(index) / (1024**2) + array_size_mb
    
    # For very large indices, use disk cache
    if total_size_mb > 100:
        logger.info(f"Large {index_type} index ({total_size_mb:.0f} MB) - using disk cache")
        
        cache = LargeIndexCache()
        cache_path = cache.save_index(index, index_type)
        
        # Return reference instead of data
        return {
            '_cached': True,
            '_cache_path': cache_path,
            '_index_type': index_type,
            '_size_mb': total_size_mb,
            'metadata': index.get('metadata', {})  # Keep metadata for quick access
        }
    
    # For medium indices with numpy arrays, consider shared memory
    elif has_large_arrays:
        logger.info(f"Index has large arrays ({array_size_mb:.0f} MB) - optimizing transfer")
        
        # Could implement shared memory here if needed
        # For now, return as-is
        
    return index


def retrieve_cached_index(index_ref: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve index from cache if it was cached.
    
    Args:
        index_ref: Index or cache reference
        
    Returns:
        Full index data
    """
    if isinstance(index_ref, dict) and index_ref.get('_cached'):
        size_info = f" ({index_ref['_size_mb']:.0f} MB)" if '_size_mb' in index_ref else ""
        logger.info(f"Retrieving cached {index_ref.get('_index_type', 'unknown')} index{size_info}")
        
        cache = LargeIndexCache()
        try:
            return cache.load_index(index_ref['_cache_path'])
        except Exception as e:
            logger.error(f"Failed to load cached index from {index_ref['_cache_path']}: {e}")
            return None
    
    return index_ref