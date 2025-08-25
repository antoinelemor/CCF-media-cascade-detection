"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
parallel_entity_processor.py

MAIN OBJECTIVE:
---------------
This script provides ultra-optimized parallel entity processing for the cascade detection framework,
maximizing CPU utilization on Apple M4 Max architecture with intelligent chunking and progress tracking.

Dependencies:
-------------
- os
- multiprocessing
- concurrent.futures
- typing
- numpy
- time
- logging
- tqdm
- collections
- pandas
- json
- functools

MAIN FEATURES:
--------------
1) Intelligent CPU core allocation for M4 Max (16 performance cores)
2) Optimized chunk size calculation based on data characteristics
3) Parallel entity extraction with minimal overhead
4) Real-time progress tracking with tqdm integration
5) Efficient memory management and data serialization

Author:
-------
Antoine Lemor
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import time
import logging
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import json
from functools import partial

# Force spawn method for macOS compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

logger = logging.getLogger(__name__)


class ParallelEntityProcessor:
    """
    Ultra-fast parallel entity processing with maximum resource utilization.
    Designed for M4 Max with 16 performance cores and high memory bandwidth.
    """
    
    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize the parallel processor.
        
        Args:
            n_workers: Number of workers (None for auto-detection)
        """
        # Auto-detect optimal workers for M4 Max (16 cores)
        cpu_count = mp.cpu_count()
        if n_workers is None:
            # M4 Max: 16 cores total
            # OPTIMAL: Use actual core count, not multiplied
            # Too many workers cause overhead and contention
            self.n_workers = min(cpu_count, 16)  # Use actual cores only
        else:
            self.n_workers = n_workers
            
        # Optimal chunk sizes based on testing for M4 Max
        self.min_chunk_size = 35000  # Larger chunks to reduce overhead
        self.chunks_per_worker = 2  # 2-3 chunks per worker is optimal
        self.max_chunk_size = 100000  # Much larger chunks for efficiency
        
        logger.info(f"ParallelEntityProcessor initialized:")
        logger.info(f"  • Detected {cpu_count} CPU cores")
        logger.info(f"  • Using {self.n_workers} parallel workers")
        logger.info(f"  • Chunks per worker: {self.chunks_per_worker}")
    
    def process_entities_parallel(self, 
                                 data: pd.DataFrame,
                                 process_func: callable,
                                 desc: str = "Processing entities") -> Tuple[Dict, int, int]:
        """
        Process entities in parallel with optimal chunking and progress tracking.
        
        Args:
            data: DataFrame with entity data
            process_func: Function to process each chunk
            desc: Description for progress bar
            
        Returns:
            Tuple of (merged_results, processed_count, error_count)
        """
        if data.empty:
            return {}, 0, 0
            
        total_rows = len(data)
        
        # Calculate optimal chunk size for M4 Max performance
        # Balance between parallelization and overhead
        target_chunks = self.n_workers * self.chunks_per_worker  # 16 * 2 = 32 chunks ideal
        ideal_chunk_size = total_rows // target_chunks if target_chunks > 0 else 50000
        
        # Apply bounds
        chunk_size = max(self.min_chunk_size, min(self.max_chunk_size, ideal_chunk_size))
        
        # For very large datasets, use optimal size
        if total_rows > 1000000:  # 1M+ rows
            # For 3.5M rows, ~70k per chunk gives ~50 chunks
            chunk_size = 70000  # Optimal for M4 Max based on testing
        
        # Create chunks without unnecessary copies
        chunks = []
        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)
            # Use view instead of copy when possible to save memory
            chunk = data.iloc[i:end_idx]
            # Only copy if we need to modify the chunk
            if 'date_converted' not in chunk.columns:
                chunk = chunk.copy()
            chunks.append((chunk, i // chunk_size))
        
        n_chunks = len(chunks)
        
        # Sanity check - optimal range is 20-50 chunks for 16 cores
        if n_chunks > 50:
            logger.warning(f"Too many chunks ({n_chunks})! Reconsolidating to optimal size...")
            # Reconsolidate to optimal number
            new_chunk_size = max(35000, total_rows // 40)  # Target ~40 chunks
            chunks = []
            for i in range(0, total_rows, new_chunk_size):
                end_idx = min(i + new_chunk_size, total_rows)
                chunk = data.iloc[i:end_idx].copy()
                chunks.append((chunk, i // new_chunk_size))
            n_chunks = len(chunks)
            chunk_size = new_chunk_size
        
        logger.info(f"Parallel processing configuration:")
        logger.info(f"  • Total rows: {total_rows:,}")
        logger.info(f"  • Chunk size: {chunk_size:,} (actual: {total_rows // n_chunks if n_chunks > 0 else 0:,})")
        logger.info(f"  • Number of chunks: {n_chunks}")
        logger.info(f"  • Workers: {self.n_workers}")
        logger.info(f"  • Expected speedup: {min(self.n_workers, n_chunks)}x")
        
        # Process chunks in parallel with precise progress tracking
        merged_results = defaultdict(lambda: {
            'type': None,
            'name': None,
            'citations': [],
            'count': 0,
            'occurrences': 0,
            'journalists': set(),
            'media': set(),
            'dates': [],
            'articles': set(),
            'co_mentions': {},
            'authority_score': 0.0,
            'first_seen': None,
            'last_seen': None
        })
        
        total_processed = 0
        total_errors = 0
        
        # Use ProcessPoolExecutor with optimal configuration
        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=mp.get_context('spawn')) as executor:
            # Submit all chunks at once - executor handles queueing
            futures = {
                executor.submit(process_func, chunk_data, chunk_idx): chunk_idx
                for chunk_data, chunk_idx in chunks
            }
            
            # Process results with detailed progress bar
            with tqdm(total=n_chunks, desc=desc, unit='chunks', 
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                     smoothing=0.1) as pbar:  # Add smoothing for better time estimates
                
                for future in as_completed(futures):
                    chunk_idx = futures[future]
                    try:
                        chunk_results, processed, errors = future.result(timeout=60)
                        
                        # Merge results
                        for entity_key, entity_data in chunk_results.items():
                            if entity_key not in merged_results:
                                merged_results[entity_key] = entity_data
                            else:
                                # Merge data efficiently
                                self._merge_entity_data(merged_results[entity_key], entity_data)
                        
                        total_processed += processed
                        total_errors += errors
                        
                        # Update progress bar with stats
                        pbar.set_postfix({
                            'processed': f'{total_processed:,}',
                            'entities': f'{len(merged_results):,}',
                            'errors': total_errors
                        })
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Failed to process chunk {chunk_idx}: {e}")
                        total_errors += len(chunks[chunk_idx][0])
                        pbar.update(1)
        
        # Convert sets to lists for serialization
        final_results = {}
        for entity_key, entity_data in merged_results.items():
            entity_data['journalists'] = list(entity_data['journalists'])
            entity_data['media'] = list(entity_data['media'])
            entity_data['articles'] = list(entity_data['articles'])
            
            # Calculate date range
            if entity_data.get('dates'):
                dates = entity_data['dates']
                entity_data['first_seen'] = min(dates)
                entity_data['last_seen'] = max(dates)
                # Keep empty dates list for compatibility
                entity_data['dates'] = []
            else:
                # Ensure dates field exists even if empty
                entity_data['dates'] = []
                entity_data['first_seen'] = None
                entity_data['last_seen'] = None
            
            final_results[entity_key] = entity_data
        
        return final_results, total_processed, total_errors
    
    def _merge_entity_data(self, target: Dict, source: Dict):
        """
        Efficiently merge entity data from source into target.
        """
        target['count'] += source['count']
        target['occurrences'] += source['occurrences']
        target['citations'].extend(source['citations'])
        target['journalists'].update(source['journalists'])
        target['media'].update(source['media'])
        target['dates'].extend(source['dates'])
        target['articles'].update(source['articles'])
    
    def parallel_resolve_entities(self,
                                  entities: Dict[str, Dict],
                                  resolver,
                                  entity_type: str = "entities") -> Tuple[Dict, Dict]:
        """
        Resolve entities in parallel with progress tracking.
        
        Args:
            entities: Entity dictionary to resolve
            resolver: Entity resolver instance
            entity_type: Type description for progress bar
            
        Returns:
            Tuple of (resolved_entities, mapping)
        """
        if not entities:
            return {}, {}
        
        logger.info(f"Parallel resolution of {len(entities):,} {entity_type}...")
        
        # For entity resolution, we need to use the resolver's methods
        # which may not be easily parallelizable due to shared state
        # So we'll optimize the blocking and clustering steps instead
        
        # Create blocks in parallel
        with tqdm(total=1, desc=f"Creating blocks for {entity_type}") as pbar:
            blocks = resolver._create_blocks(entities)
            pbar.update(1)
        
        if not blocks:
            return entities, {}
        
        logger.info(f"Created {len(blocks):,} blocks for parallel processing")
        
        # Process blocks in parallel with detailed progress
        all_clusters = []
        block_items = list(blocks.items())
        
        # Split blocks into chunks for better parallelization
        chunk_size = max(1, len(block_items) // self.n_workers)
        block_chunks = [
            dict(block_items[i:i + chunk_size])
            for i in range(0, len(block_items), chunk_size)
        ]
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for chunk in block_chunks:
                future = executor.submit(resolver._process_block_chunk, chunk)
                futures.append(future)
            
            # Process with progress bar
            with tqdm(total=len(futures), 
                     desc=f"Processing {entity_type} blocks",
                     unit='chunks') as pbar:
                
                for future in as_completed(futures):
                    try:
                        chunk_clusters = future.result()
                        all_clusters.extend(chunk_clusters)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing block chunk: {e}")
                        pbar.update(1)
        
        # Merge clusters across blocks
        with tqdm(total=1, desc=f"Merging {entity_type} clusters") as pbar:
            merged_clusters = resolver._merge_cross_block_clusters(all_clusters)
            pbar.update(1)
        
        # Create resolved index
        with tqdm(total=1, desc=f"Creating resolved {entity_type} index") as pbar:
            resolved_index = resolver._create_resolved_index(merged_clusters, entities)
            pbar.update(1)
        
        # Create mapping
        mapping = {}
        for cluster in merged_clusters:
            if len(cluster) > 1:
                cluster_list = list(cluster)
                canonical = cluster_list[0]
                for entity_key in cluster_list[1:]:
                    mapping[entity_key] = canonical
        
        return resolved_index, mapping


def process_entity_chunk_optimized(chunk: pd.DataFrame, chunk_idx: int) -> Tuple[Dict, int, int]:
    """
    Optimized entity chunk processing function for parallel execution.
    Maintains full precision while maximizing speed.
    
    Args:
        chunk: DataFrame chunk to process
        chunk_idx: Index of the chunk
        
    Returns:
        Tuple of (entity_dict, processed_count, error_count)
    """
    from collections import defaultdict
    import pandas as pd
    import json
    import time
    
    start_time = time.time()
    
    # Initialize chunk results with all required fields
    chunk_entities = defaultdict(lambda: {
        'type': None,
        'name': None,
        'citations': [],
        'count': 0,
        'occurrences': 0,
        'journalists': set(),
        'media': set(),
        'dates': [],
        'articles': set(),
        'co_mentions': {},
        'authority_score': 0.0,
        'first_seen': None,
        'last_seen': None
    })
    
    processed = 0
    errors = 0
    
    # Pre-check for required columns to avoid repeated checks
    has_author = 'author' in chunk.columns
    has_media = 'media' in chunk.columns
    has_doc_id = 'doc_id' in chunk.columns
    
    # Process each row in chunk with minimal overhead
    for idx, row in chunk.iterrows():
        try:
            ner_json = row.get('ner_entities')
            # Quick skip for empty
            if pd.isna(ner_json) or ner_json == '' or ner_json == 'null':
                processed += 1
                continue
            
            # Parse NER entities
            entities = parse_ner_entities_optimized(ner_json)
            
            for entity_type, entity_list in entities.items():
                for entity_name in entity_list:
                    # Clean entity name
                    entity_name = clean_entity_name(entity_name)
                    if not entity_name:
                        continue
                    
                    # Create unique key
                    entity_key = f"{entity_type}:{entity_name}"
                    
                    # Update entity data
                    chunk_entities[entity_key]['type'] = entity_type
                    chunk_entities[entity_key]['name'] = entity_name
                    chunk_entities[entity_key]['count'] += 1
                    chunk_entities[entity_key]['occurrences'] += 1
                    chunk_entities[entity_key]['articles'].add(row.get('doc_id', ''))
                    
                    # Add citation details - optimize field access
                    date_val = row.get('date_converted', row.get('date'))
                    doc_id_val = row.get('doc_id', '') if has_doc_id else ''
                    author_val = row.get('author', 'Unknown') if has_author else 'Unknown'
                    media_val = row.get('media', 'Unknown') if has_media else 'Unknown'
                    
                    citation = {
                        'date': date_val,
                        'doc_id': doc_id_val,
                        'author': author_val,
                        'media': media_val
                    }
                    chunk_entities[entity_key]['citations'].append(citation)
                    
                    # Update sets - use cached values
                    if has_author and pd.notna(author_val) and author_val != 'Unknown':
                        chunk_entities[entity_key]['journalists'].add(author_val)
                    if has_media and pd.notna(media_val) and media_val != 'Unknown':
                        chunk_entities[entity_key]['media'].add(media_val)
                    if has_doc_id and doc_id_val:
                        chunk_entities[entity_key]['articles'].add(doc_id_val)
                    if date_val:
                        chunk_entities[entity_key]['dates'].append(date_val)
            
            processed += 1
                
        except Exception as e:
            errors += 1
            if errors <= 10:
                # Only log first few errors to avoid spam
                if errors <= 3:
                    import logging
                    logging.debug(f"Error in chunk {chunk_idx} at row {idx}: {e}")
    
    # Log performance for large chunks
    elapsed = time.time() - start_time
    if len(chunk) > 10000 and elapsed > 0:
        import logging
        rate = len(chunk) / elapsed
        logging.debug(f"Chunk {chunk_idx}: {len(chunk)} rows in {elapsed:.1f}s ({rate:.0f} rows/s)")
    
    return dict(chunk_entities), processed, errors


def parse_ner_entities_optimized(ner_json: str) -> Dict[str, List[str]]:
    """
    Optimized NER entity parsing.
    """
    import pandas as pd
    import json
    
    entity_types = ['PER', 'ORG', 'LOC']
    
    if pd.isna(ner_json) or ner_json == '' or ner_json == 'null':
        return {et: [] for et in entity_types}
    
    try:
        if isinstance(ner_json, str):
            entities = json.loads(ner_json)
        else:
            entities = ner_json
        
        result = {}
        for entity_type in entity_types:
            if entity_type in entities:
                if isinstance(entities[entity_type], list):
                    result[entity_type] = entities[entity_type]
                else:
                    result[entity_type] = [entities[entity_type]]
            else:
                result[entity_type] = []
        
        return result
        
    except (json.JSONDecodeError, TypeError):
        return {et: [] for et in entity_types}


def clean_entity_name(name: str) -> str:
    """
    Clean entity names to remove malformed punctuation.
    """
    import re
    
    if not name:
        return ""
    
    # Remove trailing punctuation
    name = re.sub(r'[,.\-;:!?]+$', '', name).strip()
    
    # Fix internal punctuation issues
    if ',' in name:
        parts = name.split(',')
        if len(parts) == 2:
            part1, part2 = parts[0].strip(), parts[1].strip()
            if part1 and part2:
                if part2 and part2[0].isupper() and not (part1 and part1[0].islower()):
                    name = f"{part1} {part2}"
    
    # Remove leading punctuation
    name = re.sub(r'^[,.\-;:!?]+', '', name).strip()
    
    return name