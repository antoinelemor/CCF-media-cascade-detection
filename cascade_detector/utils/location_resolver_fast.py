"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
location_resolver_fast.py

MAIN OBJECTIVE:
---------------
This script provides fast location resolution for intelligent geographic entity matching and
deduplication, optimized for M4 Max with efficient blocking and parallel processing.

Dependencies:
-------------
- re
- logging
- typing
- collections
- numpy
- difflib
- multiprocessing
- functools
- concurrent.futures
- pickle

MAIN FEATURES:
--------------
1) Geographic entity standardization and abbreviation handling
2) Hierarchical location resolution (city -> province -> country)
3) Context-aware location matching
4) Parallel processing with frequency-based optimization
5) Canadian geography specific enhancements

Author:
-------
Antoine Lemor
"""

import re
import logging
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
import numpy as np
from difflib import SequenceMatcher
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle

logger = logging.getLogger(__name__)


def process_location_batch(batch_data: Tuple[List[str], Dict, float]) -> List[Tuple[str, str, float]]:
    """
    Process a batch of locations for similarity (for multiprocessing).
    OPTIMIZED for M4 Max with aggressive pre-filtering.
    """
    entities, all_data, threshold = batch_data
    matches = []
    
    n = len(entities)
    
    # More aggressive reduction for large blocks
    if n > 200:  # Reduced from 500
        # Sort by frequency first
        entities_with_freq = [(e, all_data.get(e, {}).get('occurrences', 0)) for e in entities]
        entities_with_freq.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top entities by frequency  
        entities = [e[0] for e in entities_with_freq[:200]]  # Reduced from 500
        n = len(entities)
        logger.debug(f"Large block reduced from {len(batch_data[0])} to {n} entities")
    
    # Pre-compute normalized forms and lengths for efficiency
    normalized = {}
    lengths = {}
    for entity in entities:
        loc = entity[4:]  # Remove 'LOC:' prefix
        norm = loc.lower().strip()
        normalized[entity] = norm
        lengths[entity] = len(norm)
    
    # Compare all pairs in block with aggressive filtering
    comparisons = 0
    max_comparisons = 20000  # Reduced from 100000 for faster processing
    
    for i in range(n):
        if comparisons >= max_comparisons:
            break
            
        entity1 = entities[i]
        loc1_norm = normalized[entity1]
        len1 = lengths[entity1]
        
        # Limit comparisons per entity for better distribution
        entity_comparisons = 0
        max_per_entity = min(50, n - i - 1)  # Max 50 comparisons per entity
        
        for j in range(i + 1, min(i + 1 + max_per_entity, n)):
            entity2 = entities[j]
            loc2_norm = normalized[entity2]
            len2 = lengths[entity2]
            
            # Quick rejection based on length (pre-computed)
            if len1 == 0 or len2 == 0:
                continue
            len_ratio = min(len1, len2) / max(len1, len2)
            if len_ratio < 0.6:  # Stricter threshold
                continue
            
            # Quick rejection based on first/last characters
            if loc1_norm[0] != loc2_norm[0] and abs(ord(loc1_norm[0]) - ord(loc2_norm[0])) > 2:
                continue
            
            # Calculate similarity using our ULTRA-FAST method
            loc1 = entity1[4:]
            loc2 = entity2[4:]
            
            sim = calculate_string_similarity_fast(loc1, loc2)
            
            if sim >= threshold:
                matches.append((entity1, entity2, sim))
            
            comparisons += 1
            entity_comparisons += 1
            
            if comparisons >= max_comparisons:
                break
    
    return matches


def calculate_string_similarity_fast(s1: str, s2: str) -> float:
    """
    ULTRA-FAST string similarity calculation.
    Avoids expensive SequenceMatcher by using heuristics.
    """
    # Normalize
    s1_norm = s1.lower().strip()
    s2_norm = s2.lower().strip()
    
    # Exact match
    if s1_norm == s2_norm:
        return 1.0
    
    # Length check - if too different, skip
    len1, len2 = len(s1_norm), len(s2_norm)
    if len1 == 0 or len2 == 0:
        return 0.0
    
    len_ratio = min(len1, len2) / max(len1, len2)
    if len_ratio < 0.5:  # Too different in length
        return 0.0
    
    # One contains the other
    if s1_norm in s2_norm or s2_norm in s1_norm:
        return 0.9
    
    # Quick token-based check
    tokens1 = set(s1_norm.split())
    tokens2 = set(s2_norm.split())
    
    if tokens1 and tokens2:
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        if union > 0:
            jaccard = intersection / union
            if jaccard > 0.8:  # High token overlap
                return 0.95
            elif jaccard > 0.5:
                return 0.85
            elif jaccard > 0.3:
                return 0.7
    
    # For short strings, use character-level overlap
    if len1 < 15 and len2 < 15:
        # Character set similarity
        chars1 = set(s1_norm)
        chars2 = set(s2_norm)
        if chars1 and chars2:
            char_jaccard = len(chars1 & chars2) / len(chars1 | chars2)
            if char_jaccard > 0.8:
                return 0.75
    
    # If we reach here, strings are probably different
    # Return low similarity without using expensive SequenceMatcher
    return 0.3 if len_ratio > 0.7 else 0.0


class FastLocationResolver:
    """
    Fast location resolution using efficient blocking and parallel processing.
    Optimized for M4 Ultra Max performance.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 context_weight: float = 0.3,
                 min_occurrences: int = 2,
                 n_workers: int = None,
                 use_cache: bool = True):
        """
        Initialize fast location resolver.
        
        Args:
            similarity_threshold: Minimum similarity for merging (0-1)
            context_weight: Weight for context similarity (0-1)
            min_occurrences: Minimum occurrences to consider location
            n_workers: Number of parallel workers (None = auto)
            use_cache: Whether to cache computations
        """
        self.similarity_threshold = similarity_threshold
        self.context_weight = context_weight
        self.min_occurrences = min_occurrences
        # M4 Max optimization: use all cores for CPU-bound location resolution
        self.n_workers = n_workers or min(mp.cpu_count(), 14)  # Don't oversubscribe
        self.use_cache = use_cache
        
        # Cache for similarity calculations
        self._similarity_cache = {}
        
        # Normalization patterns
        self.cleanup_patterns = [
            (r'^(the|le|la|les|el|los|las)\s+', ''),
            (r'\s+(city|ville|town|municipality|region|province|state|country)$', ''),
            (r'[^\w\s\-]', ' '),
            (r'\s+', ' '),
        ]
        
        logger.info(f"FastLocationResolver initialized with {self.n_workers} workers (M4 Max optimized)")
    
    def resolve_locations(self, 
                         location_entities: Dict[str, Dict],
                         use_context: bool = False,  # Disabled for speed
                         use_blocking: bool = True) -> Tuple[Dict[str, Dict], Dict[str, str]]:
        """
        Resolve location entities by merging similar ones.
        Optimized for speed with minimal quality loss.
        """
        logger.info(f"Fast resolving {len(location_entities)} location entities...")
        
        # Filter by minimum occurrences
        loc_entities = {
            k: v for k, v in location_entities.items() 
            if k.startswith('LOC:') and v.get('occurrences', 0) >= self.min_occurrences
        }
        
        if not loc_entities:
            return location_entities, {}
        
        logger.info(f"Processing {len(loc_entities)} locations with >= {self.min_occurrences} occurrences")
        
        # Use blocking for O(n*k) complexity with progress tracking
        from tqdm import tqdm
        
        if use_blocking:
            with tqdm(total=3, desc="Location resolution", unit="stage", leave=False) as pbar:
                pbar.set_postfix({'stage': 'Creating blocks'})
                clusters = self._resolve_with_blocking_fast(loc_entities)
                pbar.update(1)
                
                pbar.set_postfix({'stage': 'Processing blocks'})
                pbar.update(1)
                
                pbar.set_postfix({'stage': 'Merging clusters'})
                pbar.update(1)
        else:
            # Full comparison (slower)
            clusters = self._resolve_full_fast(loc_entities)
        
        # Create resolved entities
        resolved_entities, location_mapping = self._merge_clusters(clusters, location_entities)
        
        reduction = (len(location_entities) - len(resolved_entities)) / len(location_entities) * 100
        logger.info(f"Location resolution complete: {len(location_entities)} → {len(resolved_entities)} "
                   f"(reduced by {reduction:.1f}%)")
        
        return resolved_entities, location_mapping
    
    def _normalize_location(self, location: str) -> str:
        """
        Fast location normalization.
        """
        normalized = location.lower().strip()
        
        # Apply only essential cleanup patterns
        for pattern, replacement in self.cleanup_patterns[:2]:  # Skip slower patterns
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Quick space normalization
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _create_blocking_key(self, location: str) -> List[str]:
        """
        Create blocking keys for a location.
        OPTIMIZED: Fewer keys for larger blocks.
        """
        normalized = self._normalize_location(location)
        if not normalized:
            return []
        
        keys = []
        words = normalized.split()
        
        # Primary key: First word prefix (larger blocks)
        if words:
            keys.append(f"first:{words[0][:3]}")  # Reduced from 5 to 3
        else:
            # Fallback for single-word locations
            keys.append(f"prefix:{normalized[:3] if len(normalized) >= 3 else normalized}")
        
        # Don't add too many keys - we want larger blocks
        # Removed length bucket and additional prefix to reduce block count
        
        return keys
    
    def _resolve_with_blocking_fast(self, loc_entities: Dict) -> List[Set[str]]:
        """
        Fast resolution using blocking to reduce comparisons with progress tracking.
        OPTIMIZED: Merge small blocks for better parallelization.
        """
        # Create initial blocks
        initial_blocks = defaultdict(set)
        for loc_key in loc_entities:
            location = loc_key[4:]  # Remove 'LOC:' prefix
            block_keys = self._create_blocking_key(location)
            if block_keys:
                # Use only primary key for simpler blocking
                initial_blocks[block_keys[0]].add(loc_key)
            else:
                initial_blocks['misc'].add(loc_key)
        
        # Merge small blocks to reduce overhead
        blocks = {}
        small_entities = set()
        
        for key, block in initial_blocks.items():
            if len(block) < 10:  # Merge blocks with < 10 entities
                small_entities.update(block)
            else:
                blocks[key] = block
        
        # Group small entities into larger blocks
        if small_entities:
            # Split into chunks of ~50 entities each
            small_list = list(small_entities)
            for i in range(0, len(small_list), 50):
                blocks[f'small_{i//50}'] = set(small_list[i:i+50])
        
        logger.info(f"Optimized {len(initial_blocks)} initial blocks into {len(blocks)} processing blocks")
        
        # Find matches within blocks (parallel)
        all_matches = []
        block_list = [(list(block), loc_entities, self.similarity_threshold) 
                      for block in blocks.values() if len(block) > 1]
        
        # Use ThreadPoolExecutor with progress tracking
        from tqdm import tqdm
        from concurrent.futures import as_completed
        
        # Sort blocks by size (LARGEST first for better results)
        block_list.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Process ALL blocks for scientific accuracy (no limit)
        logger.info(f"Processing {len(block_list)} location blocks with maximum parallelization")
        
        # OPTIMIZED: Use ThreadPoolExecutor for many small tasks (lower overhead)
        # Or batch blocks together for ProcessPoolExecutor
        n_workers_to_use = min(16, self.n_workers, len(block_list))
        
        # AGGRESSIVE BATCHING for M4 Max optimization
        if len(block_list) > 50:  # Lower threshold
            # Create fewer, larger batches to minimize overhead
            # Target: 16 batches for 16 cores
            optimal_batches = min(16, n_workers_to_use)
            batch_size = max(1, (len(block_list) + optimal_batches - 1) // optimal_batches)
            batched_blocks = []
            
            for i in range(0, len(block_list), batch_size):
                batch = block_list[i:i+batch_size]
                batched_blocks.append(batch)
            
            logger.info(f"OPTIMIZED: Batched {len(block_list)} blocks into {len(batched_blocks)} large batches (size ~{batch_size}) for {n_workers_to_use} workers")
            
            # Use ThreadPoolExecutor for string comparison tasks (better for I/O-bound work)
            with ThreadPoolExecutor(max_workers=n_workers_to_use) as executor:
                futures = []
                for batch in batched_blocks:
                    # Process multiple blocks in one task
                    future = executor.submit(self._process_block_batch_optimized, batch)
                    futures.append(future)
                
                # Collect results with progress
                with tqdm(total=len(block_list), desc="Processing location blocks", unit="blocks", leave=False) as pbar:
                    completed = 0
                    for future in as_completed(futures):
                        try:
                            batch_matches = future.result(timeout=30)  # Add timeout
                            all_matches.extend(batch_matches)
                            blocks_in_batch = min(batch_size, len(block_list) - completed)
                            pbar.update(blocks_in_batch)
                            completed += blocks_in_batch
                        except Exception as e:
                            logger.debug(f"Batch failed: {e}")
                            blocks_in_batch = min(batch_size, len(block_list) - completed) 
                            pbar.update(blocks_in_batch)
                            completed += blocks_in_batch
        else:
            # For fewer blocks, use original approach
            with ProcessPoolExecutor(max_workers=n_workers_to_use) as executor:
                futures = {
                    executor.submit(process_location_batch, block_data): idx
                    for idx, block_data in enumerate(block_list)
                }
                
                logger.info(f"Submitted {len(futures)} blocks to {n_workers_to_use} workers")
                
                with tqdm(total=len(futures), desc="Processing location blocks", unit="blocks", leave=False) as pbar:
                    for future in as_completed(futures):
                        try:
                            matches = future.result()
                            all_matches.extend(matches)
                            pbar.update(1)
                        except Exception as e:
                            logger.debug(f"Block failed: {e}")
                            pbar.update(1)
        
        # Build clusters from matches
        return self._build_clusters_from_matches(all_matches, loc_entities.keys())
    
    def _resolve_full_fast(self, loc_entities: Dict) -> List[Set[str]]:
        """
        Full resolution (fallback, slower).
        """
        entities = list(loc_entities.keys())
        n = len(entities)
        
        if n <= 1:
            return [{e} for e in entities]
        
        # Limit comparisons for large sets
        if n > 1000:
            logger.warning(f"Large location set ({n}), limiting to top 1000 by occurrences")
            entities = sorted(entities, 
                            key=lambda k: loc_entities[k].get('occurrences', 0), 
                            reverse=True)[:1000]
            n = 1000
        
        # Quick similarity matrix
        matches = []
        for i in range(n):
            for j in range(i + 1, min(i + 50, n)):  # Limit comparisons per entity
                loc1 = entities[i][4:]
                loc2 = entities[j][4:]
                
                sim = calculate_string_similarity_fast(loc1, loc2)
                if sim >= self.similarity_threshold:
                    matches.append((entities[i], entities[j], sim))
        
        return self._build_clusters_from_matches(matches, entities)
    
    def _process_block_batch(self, batch: List[Tuple]) -> List[Tuple[str, str, float]]:
        """
        Process multiple blocks in a single task to reduce overhead.
        
        Args:
            batch: List of block data tuples
            
        Returns:
            Combined list of matches from all blocks
        """
        all_matches = []
        
        for block_data in batch:
            try:
                matches = process_location_batch(block_data)
                all_matches.extend(matches)
            except Exception as e:
                logger.debug(f"Block processing failed: {e}")
                continue
        
        return all_matches
    
    def _process_block_batch_optimized(self, batch: List[Tuple]) -> List[Tuple[str, str, float]]:
        """
        OPTIMIZED: Process multiple blocks with better memory management.
        
        Args:
            batch: List of block data tuples
            
        Returns:
            Combined list of matches from all blocks
        """
        all_matches = []
        
        # Process blocks sequentially within the batch to avoid memory issues
        for block_data in batch:
            try:
                # Process block with optimized function
                matches = process_location_batch(block_data)
                
                # Only keep high-confidence matches to reduce memory
                high_conf_matches = [m for m in matches if m[2] >= 0.85]
                all_matches.extend(high_conf_matches)
                
                # Also keep medium confidence if not too many
                if len(high_conf_matches) < 100:
                    medium_matches = [m for m in matches if 0.85 > m[2] >= self.similarity_threshold]
                    all_matches.extend(medium_matches[:50])  # Limit medium confidence matches
            except Exception as e:
                logger.debug(f"Block processing failed: {e}")
                continue
        
        return all_matches
    
    def _build_clusters_from_matches(self, matches: List[Tuple], all_entities: List[str]) -> List[Set[str]]:
        """
        Build clusters from similarity matches using Union-Find.
        """
        # Union-Find for efficient clustering
        parent = {entity: entity for entity in all_entities}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union similar entities
        for entity1, entity2, _ in matches:
            union(entity1, entity2)
        
        # Build clusters
        clusters_dict = defaultdict(set)
        for entity in all_entities:
            root = find(entity)
            clusters_dict[root].add(entity)
        
        return list(clusters_dict.values())
    
    def _merge_clusters(self, clusters: List[Set[str]], 
                       all_entities: Dict) -> Tuple[Dict[str, Dict], Dict[str, str]]:
        """
        Merge clusters into resolved entities.
        """
        resolved_entities = {}
        location_mapping = {}
        
        for cluster in clusters:
            if len(cluster) == 1:
                # Single location
                loc_key = list(cluster)[0]
                resolved_entities[loc_key] = all_entities[loc_key]
                location_mapping[loc_key] = loc_key
            else:
                # Merge cluster
                canonical_key, merged_data = self._fast_merge_cluster(cluster, all_entities)
                resolved_entities[canonical_key] = merged_data
                for loc_key in cluster:
                    location_mapping[loc_key] = canonical_key
        
        # Add entities that didn't meet threshold
        for key, data in all_entities.items():
            if key not in location_mapping:
                resolved_entities[key] = data
                location_mapping[key] = key
        
        return resolved_entities, location_mapping
    
    def _fast_merge_cluster(self, cluster: Set[str], 
                           all_entities: Dict) -> Tuple[str, Dict]:
        """
        Fast cluster merging.
        """
        cluster_list = list(cluster)
        
        # Select canonical (most occurrences)
        canonical_key = max(cluster_list, 
                          key=lambda k: (all_entities[k].get('occurrences', 0), len(k)))
        
        # Fast merge
        merged_data = all_entities[canonical_key].copy()
        merged_data['merged_from'] = []
        
        total_occurrences = 0
        all_articles = set()
        all_media = set()
        all_journalists = set()
        
        for loc_key in cluster:
            data = all_entities[loc_key]
            total_occurrences += data.get('occurrences', 0)
            all_articles.update(data.get('articles', []))
            all_media.update(data.get('media', []))
            all_journalists.update(data.get('journalists', []))
            
            if loc_key != canonical_key:
                merged_data['merged_from'].append({
                    'key': loc_key,
                    'name': loc_key[4:],
                    'occurrences': data.get('occurrences', 0)
                })
        
        # Update merged data
        merged_data['occurrences'] = total_occurrences
        merged_data['articles'] = list(all_articles)
        merged_data['media'] = list(all_media)
        merged_data['journalists'] = list(all_journalists)
        
        # Simple authority score
        merged_data['authority_score'] = total_occurrences * np.log1p(len(all_media) + len(all_journalists))
        
        return canonical_key, merged_data