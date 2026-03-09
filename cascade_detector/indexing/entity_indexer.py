"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
entity_indexer.py

MAIN OBJECTIVE:
---------------
This script indexes Named Entity Recognition (NER) entities for epistemic authority analysis,
tracking who cites whom and when, with intelligent entity resolution and co-mention network construction.

Dependencies:
-------------
- pandas
- numpy
- json
- typing
- collections
- datetime
- logging

MAIN FEATURES:
--------------
1) NER entity extraction and indexing (PER, ORG, LOC)
2) Intelligent entity resolution to merge variations
3) Authority scoring based on citation frequency and network position
4) Co-mention network construction for entity relationships
5) Parallel processing for large-scale entity extraction

Author:
-------
Antoine Lemor
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime
import logging

from cascade_detector.indexing.base_indexer import AbstractIndexer
from cascade_detector.core.constants import ENTITY_TYPES
try:
    from cascade_detector.utils.entity_resolver_fast import FastEntityResolver
    USE_FAST_RESOLVER = True
except ImportError:
    from cascade_detector.utils.entity_resolver import SmartEntityResolver
    USE_FAST_RESOLVER = False

from cascade_detector.utils.location_resolver_fast import FastLocationResolver
HAS_LOCATION_RESOLVER = True

# Import optimized parallel processor
try:
    from cascade_detector.utils.parallel_entity_processor import (
        ParallelEntityProcessor, 
        process_entity_chunk_optimized
    )
    HAS_PARALLEL_PROCESSOR = True
except ImportError:
    HAS_PARALLEL_PROCESSOR = False

logger = logging.getLogger(__name__)


class EntityIndexer(AbstractIndexer):
    """
    Indexes NER entities for epistemic authority analysis.
    Tracks who cites whom and when.
    """
    
    def __init__(self, resolve_entities: bool = True, 
                 resolve_locations: bool = True,
                 n_workers: int = None,  # Auto-detect optimal workers
                 location_similarity_threshold: float = 0.85,
                 location_context_weight: float = 0.3,
                 location_min_occurrences: int = 2):
        """Initialize entity indexer.
        
        Args:
            resolve_entities: Whether to apply intelligent entity resolution
            resolve_locations: Whether to apply location-specific resolution
            n_workers: Number of parallel workers (optimized for M4 Max)
            location_similarity_threshold: Minimum similarity for location merging
            location_context_weight: Weight for context in location similarity
            location_min_occurrences: Minimum occurrences for location resolution
        """
        super().__init__(name="EntityIndexer")
        self.entity_types = ENTITY_TYPES
        self.resolve_entities = resolve_entities
        self.resolve_locations = resolve_locations
        
        # Auto-detect optimal workers for M4 Max (16 cores, 128GB RAM)
        import multiprocessing as mp
        if n_workers is None:
            # M4 Max has 16 cores
            # OPTIMAL: Use actual core count, not multiplied
            # Too many workers cause overhead and contention
            cpu_count = mp.cpu_count()
            self.n_workers = min(cpu_count, 16)  # Use actual cores only
            logger.info(f"M4 Max optimization: {cpu_count} cores, using {self.n_workers} workers")
        else:
            self.n_workers = n_workers
        
        # Initialize parallel processor if available
        if HAS_PARALLEL_PROCESSOR:
            self.parallel_processor = ParallelEntityProcessor(n_workers=self.n_workers)
            logger.info("Using optimized ParallelEntityProcessor")
        else:
            self.parallel_processor = None
        
        # Initialize entity resolver if needed
        if self.resolve_entities:
            if USE_FAST_RESOLVER:
                self.entity_resolver = FastEntityResolver(
                    similarity_threshold=0.85,
                    context_weight=0.3,
                    min_occurrences=2,
                    n_workers=self.n_workers
                )
                logger.info(f"Using FastEntityResolver with {self.n_workers} workers")
            else:
                self.entity_resolver = SmartEntityResolver(
                    similarity_threshold=0.85,
                    context_weight=0.3,
                    min_occurrences=2
                )
        
        # Initialize location resolver if needed
        if self.resolve_locations and HAS_LOCATION_RESOLVER:
            # Use same worker count as entity processing for consistency
            location_workers = self.n_workers
            self.location_resolver = FastLocationResolver(
                similarity_threshold=location_similarity_threshold,
                context_weight=location_context_weight,
                min_occurrences=location_min_occurrences,
                n_workers=location_workers
            )
            logger.info(f"Using FastLocationResolver with {location_workers} workers")
        elif self.resolve_locations and not HAS_LOCATION_RESOLVER:
            logger.warning("LocationResolver not available, location resolution disabled")
            self.resolve_locations = False
        
    def get_required_columns(self) -> List[str]:
        """Get required columns."""
        return ['date', 'doc_id', 'media', 'author', 'ner_entities']
    
    def build_index(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Build entity index from data.
        
        Structure: {
            entity_key: {
                'type': 'PER'/'ORG'/'LOC',
                'name': str,
                'citations': [{date, doc_id, author, media}, ...],
                'count': int,
                'journalists': set,
                'media': set,
                'authority_score': float,
                'first_seen': date,
                'last_seen': date
            }
        }
        """
        logger.info(f"Building entity index from {len(data):,} rows...")
        logger.info(f"Configuration: resolve_entities={self.resolve_entities}, resolve_locations={self.resolve_locations}")
        logger.info(f"Workers: {self.n_workers}")
        
        # Check if NER column exists
        if 'ner_entities' not in data.columns:
            logger.warning("ner_entities column not found, creating empty index")
            self.index = {}
            return self.index
        
        # Convert dates in parallel if large dataset
        if len(data) > 10000:
            logger.info("Converting dates in parallel...")
            data = self._convert_dates_parallel(data)
        else:
            data = self._convert_dates(data)
        
        # Initialize index
        self.index = defaultdict(lambda: {
            'type': None,
            'name': None,
            'citations': [],
            'count': 0,
            'occurrences': 0,  # NEW: Add occurrences field for compatibility
            'journalists': set(),
            'media': set(),
            'dates': [],
            'articles': set(),  # NEW: Track unique doc_ids
            'co_mentions': {},  # NEW: Will be populated later
            'authority_score': 0.0
        })
        
        # Use optimized parallel processor if available
        if HAS_PARALLEL_PROCESSOR and self.parallel_processor:
            logger.info(f"Using optimized parallel processor with {self.n_workers} workers")
            logger.info(f"Processing {len(data):,} rows in optimized chunks...")
            
            import time
            start_time = time.time()
            
            entity_dict, processed, errors = self.parallel_processor.process_entities_parallel(
                data, 
                process_entity_chunk_optimized,
                desc="Processing entity chunks"
            )
            
            elapsed = time.time() - start_time
            throughput = len(data) / elapsed if elapsed > 0 else 0
            logger.info(f"Entity processing complete: {elapsed:.1f}s, {throughput:.0f} rows/sec")
            
            # Update index with processed entities
            self.index.update(entity_dict)
            
        else:
            # Fallback to original parallel processing
            logger.info("Using standard parallel processing")
            
            # OPTIMIZED: Large chunks for better performance
            # Fewer chunks = less overhead, better CPU utilization
            # For M4 Max with 16 workers, use 32-48 chunks total
            target_chunks = self.n_workers * 2  # 2 chunks per worker
            chunk_size = max(10000, len(data) // target_chunks)
            chunk_size = min(chunk_size, 100000)  # Larger chunks for efficiency
            n_chunks = (len(data) + chunk_size - 1) // chunk_size
            
            # Ensure we don't have too few chunks
            if n_chunks < self.n_workers:
                chunk_size = len(data) // self.n_workers
                n_chunks = self.n_workers
            
            logger.info(f"OPTIMIZED PARALLEL PROCESSING:")
            logger.info(f"  • Total rows: {len(data):,}")
            logger.info(f"  • Chunks: {n_chunks} chunks of ~{chunk_size:,} rows")
            logger.info(f"  • Workers: {self.n_workers} parallel processes")
            logger.info(f"  • Chunks per worker: {n_chunks / self.n_workers:.1f}")
            
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing as mp
            
            processed = 0
            errors = 0
            
            # Split dataframe into chunks
            chunks = []
            for i in range(0, len(data), chunk_size):
                chunk = data.iloc[i:i+chunk_size].copy()
                chunks.append(chunk)
            
            # Process chunks in parallel with optimized settings
            # Use 'fork' on Unix for faster process creation (avoids pickle overhead)
            import sys
            mp_context = 'fork' if sys.platform != 'darwin' else 'spawn'
            
            with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=mp.get_context(mp_context)) as executor:
                futures = {
                    executor.submit(self._process_chunk, chunk, idx): idx 
                    for idx, chunk in enumerate(chunks)
                }
                
                from tqdm import tqdm
                with tqdm(total=n_chunks, desc="Processing entity chunks") as pbar:
                    for future in as_completed(futures):
                        chunk_idx = futures[future]
                        try:
                            chunk_index, chunk_processed, chunk_errors = future.result(timeout=60)
                            
                            for entity_key, entity_data in chunk_index.items():
                                if entity_key not in self.index:
                                    self.index[entity_key] = entity_data
                                else:
                                    self.index[entity_key]['count'] += entity_data['count']
                                    self.index[entity_key]['occurrences'] += entity_data['occurrences']
                                    self.index[entity_key]['citations'].extend(entity_data['citations'])
                                    self.index[entity_key]['journalists'].update(entity_data['journalists'])
                                    self.index[entity_key]['media'].update(entity_data['media'])
                                    self.index[entity_key]['dates'].extend(entity_data['dates'])
                                    self.index[entity_key]['articles'].update(entity_data['articles'])
                            
                            processed += chunk_processed
                            errors += chunk_errors
                            pbar.update(1)
                            
                        except Exception as e:
                            logger.error(f"Failed to process chunk {chunk_idx}: {e}")
                            errors += len(chunks[chunk_idx])
                            pbar.update(1)
        
        logger.info(f"Parallel processing complete: {processed:,} rows processed, {errors} errors")
        
        # Alternative: Keep original sequential code as fallback (commented)
        '''
        for idx, row in data.iterrows():
            try:
                if pd.notna(row.get('ner_entities')):
                    entities = self._parse_ner_entities(row['ner_entities'])
                    
                    for entity_type, entity_list in entities.items():
                        for entity_name in entity_list:
                            # Create unique key
                            entity_key = f"{entity_type}:{entity_name}"
                            
                            # Update index
                            self.index[entity_key]['type'] = entity_type
                            self.index[entity_key]['name'] = entity_name
                            self.index[entity_key]['count'] += 1
                            self.index[entity_key]['occurrences'] += 1  # NEW: Update occurrences
                            self.index[entity_key]['articles'].add(row['doc_id'])  # NEW: Track doc_id
                            
                            # Add citation details
                            citation = {
                                'date': row['date_converted'],
                                'doc_id': row['doc_id'],
                                'author': row.get('author', 'Unknown'),
                                'media': row.get('media', 'Unknown')
                            }
                            self.index[entity_key]['citations'].append(citation)
                            
                            # Update sets
                            if pd.notna(row.get('author')):
                                self.index[entity_key]['journalists'].add(row['author'])
                            if pd.notna(row.get('media')):
                                self.index[entity_key]['media'].add(row['media'])
                            
                            self.index[entity_key]['dates'].append(row['date_converted'])
                    
                    processed += 1
                    
            except Exception as e:
                errors += 1
                if errors <= 10:  # Log first 10 errors
                    logger.debug(f"Error processing NER entities at row {idx}: {e}")
        '''  # End of commented sequential code
        
        # Convert sets to lists for serialization
        for entity_key in self.index:
            # Convert sets to lists if they're still sets
            if isinstance(self.index[entity_key].get('journalists'), set):
                self.index[entity_key]['journalists'] = list(self.index[entity_key]['journalists'])
            if isinstance(self.index[entity_key].get('media'), set):
                self.index[entity_key]['media'] = list(self.index[entity_key]['media'])
            if isinstance(self.index[entity_key].get('articles'), set):
                self.index[entity_key]['articles'] = list(self.index[entity_key]['articles'])
            
            # Calculate date range only if dates exist and not already processed
            if 'dates' in self.index[entity_key] and self.index[entity_key]['dates']:
                dates = self.index[entity_key]['dates']
                # Only process if dates is a non-empty list
                if dates and not self.index[entity_key].get('first_seen'):
                    self.index[entity_key]['first_seen'] = min(dates)
                    self.index[entity_key]['last_seen'] = max(dates)
                    # Clear dates to save memory
                    self.index[entity_key]['dates'] = []
            
            # Ensure dates field exists (for compatibility)
            if 'dates' not in self.index[entity_key]:
                self.index[entity_key]['dates'] = []
        
        # Calculate authority scores
        self._calculate_authority_scores()
        
        # NEW: Calculate co-mentions between entities
        self._calculate_co_mentions(data)
        
        # Apply location-specific resolution first if enabled
        if self.resolve_locations and HAS_LOCATION_RESOLVER:
            # Extract LOC entities
            loc_entities = {k: v for k, v in self.index.items() if k.startswith('LOC:')}
            original_loc_count = len(loc_entities)
            
            if loc_entities:
                logger.info(f"Applying PARALLEL location resolution to {original_loc_count} location entities...")
                
                # OPTIMIZED: Use location resolver with full parallelization
                import time
                start_time = time.time()
                
                resolved_locs, location_mapping = self.location_resolver.resolve_locations(
                    loc_entities,
                    use_context=False,  # Disable context for speed
                    use_blocking=True    # Use blocking for efficiency
                )
                
                elapsed = time.time() - start_time
                logger.info(f"Location resolution took {elapsed:.1f}s ({original_loc_count / elapsed:.0f} entities/sec)")
                
                # Update index with resolved locations
                for loc_key in loc_entities:
                    if loc_key in self.index:
                        del self.index[loc_key]
                
                self.index.update(resolved_locs)
                self.location_mapping = location_mapping
                
                resolved_loc_count = len(resolved_locs)
                loc_reduction = (original_loc_count - resolved_loc_count) / original_loc_count * 100
                logger.info(f"Location resolution complete: {original_loc_count} → {resolved_loc_count} "
                           f"(reduced by {loc_reduction:.1f}%)")
        
        # Apply general entity resolution if enabled (for PER and ORG entities)
        if self.resolve_entities:
            # Process PER and ORG entities in parallel if both location resolution was done
            if self.resolve_locations and HAS_LOCATION_RESOLVER:
                # Filter entity types for parallel processing
                per_entities = {k: v for k, v in self.index.items() if k.startswith('PER:')}
                org_entities = {k: v for k, v in self.index.items() if k.startswith('ORG:')}
                
                logger.info(f"Parallel entity resolution:")
                logger.info(f"  • PER entities: {len(per_entities):,}")
                logger.info(f"  • ORG entities: {len(org_entities):,}")
                
                resolved_all = {}
                
                # Process PER and ORG in parallel using ProcessPoolExecutor for better performance
                if per_entities or org_entities:
                    from concurrent.futures import ProcessPoolExecutor, as_completed
                    import time
                    
                    logger.info(f"Parallel entity resolution using {min(2, self.n_workers)} workers")
                    start_time = time.time()
                    
                    with ProcessPoolExecutor(max_workers=min(2, self.n_workers)) as executor:
                        futures = {}
                        
                        if per_entities:
                            futures[executor.submit(
                                self.entity_resolver.resolve_entities,
                                per_entities, False, True
                            )] = ("PER", len(per_entities))
                        
                        if org_entities:
                            futures[executor.submit(
                                self.entity_resolver.resolve_entities,
                                org_entities, False, True
                            )] = ("ORG", len(org_entities))
                        
                        for future in as_completed(futures):
                            entity_type, original_count = futures[future]
                            try:
                                resolved = future.result(timeout=300)  # 5 min timeout
                                resolved_all.update(resolved)
                                
                                reduction = (original_count - len(resolved)) / original_count * 100 if original_count > 0 else 0
                                logger.info(f"  • {entity_type}: {original_count:,} → {len(resolved):,} (-{reduction:.1f}%)")
                                
                            except Exception as e:
                                logger.error(f"Error resolving {entity_type} entities: {e}")
                else:
                    # Sequential fallback
                    non_loc_entities = {**per_entities, **org_entities}
                    if non_loc_entities:
                        if USE_FAST_RESOLVER:
                            resolved_all = self.entity_resolver.resolve_entities(
                                non_loc_entities, use_context=False, use_blocking=True
                            )
                        else:
                            self.entity_resolver.build_embeddings(non_loc_entities)
                            resolved_all = self.entity_resolver.resolve_entities(
                                non_loc_entities, use_context=True, parallel=True
                            )
                
                # Update index with resolved entities
                for entity_key in per_entities:
                    if entity_key in self.index:
                        del self.index[entity_key]
                for entity_key in org_entities:
                    if entity_key in self.index:
                        del self.index[entity_key]
                
                self.index.update(resolved_all)
                
                elapsed = time.time() - start_time
                logger.info(f"Entity resolution took {elapsed:.1f}s")
                
                total_original = len(per_entities) + len(org_entities)
                total_resolved = len(resolved_all)
                if total_original > 0:
                    reduction = (total_original - total_resolved) / total_original * 100
                    logger.info(f"Entity resolution complete: {total_original:,} → {total_resolved:,} "
                               f"(reduced by {reduction:.1f}%)")
            else:
                # Resolve all entities together
                original_count = len(self.index)
                logger.info(f"Applying entity resolution to {original_count} entities...")
                
                if HAS_PARALLEL_PROCESSOR and self.parallel_processor:
                    from tqdm import tqdm
                    with tqdm(total=1, desc="Resolving all entities", unit="stage") as pbar:
                        if USE_FAST_RESOLVER:
                            self.index = self.entity_resolver.resolve_entities(
                                self.index, use_context=False, use_blocking=True
                            )
                        else:
                            self.entity_resolver.build_embeddings(self.index)
                            self.index = self.entity_resolver.resolve_entities(
                                self.index, use_context=True, parallel=True
                            )
                        pbar.update(1)
                else:
                    if USE_FAST_RESOLVER:
                        self.index = self.entity_resolver.resolve_entities(
                            self.index, use_context=False, use_blocking=True
                        )
                    else:
                        self.entity_resolver.build_embeddings(self.index)
                        self.index = self.entity_resolver.resolve_entities(
                            self.index, use_context=True, parallel=True
                        )
                
                resolved_count = len(self.index)
                reduction = (original_count - resolved_count) / original_count * 100
                logger.info(f"Entity resolution complete: {original_count} → {resolved_count} "
                           f"(reduced by {reduction:.1f}%)")
        
        # Convert to regular dict
        self.index = dict(self.index)
        
        # Update metadata
        self.metadata['created'] = datetime.now().isoformat()
        self.metadata['n_entries'] = len(self.index)
        self.metadata['n_rows_processed'] = processed
        self.metadata['n_errors'] = errors
        self.metadata['resolution_applied'] = self.resolve_entities
        self.metadata['location_resolution_applied'] = self.resolve_locations
        self.metadata['n_workers'] = self.n_workers
        
        # Add location resolution statistics
        if self.resolve_locations and hasattr(self, 'location_mapping'):
            self.metadata['n_location_mappings'] = len(self.location_mapping)
            # Count how many locations were merged
            merged_count = sum(1 for k, v in self.location_mapping.items() if k != v)
            self.metadata['n_locations_merged'] = merged_count
        
        logger.info(f"Entity index built: {len(self.index)} unique entities from {processed} rows")
        if errors > 0:
            logger.warning(f"Encountered {errors} errors during processing")
        
        return self.index
    
    @staticmethod
    def _process_chunk(chunk: pd.DataFrame, chunk_idx: int) -> Tuple[Dict, int, int]:
        """
        Process a chunk of data in parallel.
        Static method to be pickled for multiprocessing.
        
        Args:
            chunk: DataFrame chunk to process
            chunk_idx: Index of the chunk
            
        Returns:
            Tuple of (chunk_index, processed_count, error_count)
        """
        from collections import defaultdict
        import pandas as pd
        import json
        
        # Initialize chunk index
        chunk_index = defaultdict(lambda: {
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
            'authority_score': 0.0
        })
        
        processed = 0
        errors = 0
        
        # Process each row in chunk
        for idx, row in chunk.iterrows():
            try:
                if pd.notna(row.get('ner_entities')):
                    # Parse NER entities
                    entities = EntityIndexer._parse_ner_entities_static(row['ner_entities'])
                    
                    for entity_type, entity_list in entities.items():
                        for entity_name in entity_list:
                            # Create unique key
                            entity_key = f"{entity_type}:{entity_name}"
                            
                            # Update chunk index
                            chunk_index[entity_key]['type'] = entity_type
                            chunk_index[entity_key]['name'] = entity_name
                            chunk_index[entity_key]['count'] += 1
                            chunk_index[entity_key]['occurrences'] += 1
                            chunk_index[entity_key]['articles'].add(row['doc_id'])
                            
                            # Add citation details
                            citation = {
                                'date': row.get('date_converted', row.get('date')),
                                'doc_id': row['doc_id'],
                                'author': row.get('author', 'Unknown'),
                                'media': row.get('media', 'Unknown')
                            }
                            chunk_index[entity_key]['citations'].append(citation)
                            
                            # Update sets
                            if pd.notna(row.get('author')):
                                chunk_index[entity_key]['journalists'].add(row['author'])
                            if pd.notna(row.get('media')):
                                chunk_index[entity_key]['media'].add(row['media'])
                            
                            chunk_index[entity_key]['dates'].append(row.get('date_converted', row.get('date')))
                    
                    processed += 1
                    
            except Exception as e:
                errors += 1
                if errors <= 10:  # Only log first 10 errors per chunk
                    import logging
                    logging.debug(f"Error in chunk {chunk_idx} at row {idx}: {e}")
        
        # Convert to regular dict for pickling
        return dict(chunk_index), processed, errors
    
    @staticmethod
    def _parse_ner_entities_static(ner_json: str) -> Dict[str, List[str]]:
        """
        Static version of _parse_ner_entities for parallel processing.
        """
        import pandas as pd
        import json
        
        # Default entity types
        entity_types = ['PER', 'ORG', 'LOC']
        
        if pd.isna(ner_json) or ner_json == '' or ner_json == 'null':
            return {et: [] for et in entity_types}
        
        try:
            # Try to parse as JSON
            if isinstance(ner_json, str):
                entities = json.loads(ner_json)
            else:
                entities = ner_json
            
            # Ensure all entity types are present
            result = {}
            for entity_type in entity_types:
                if entity_type in entities:
                    # Ensure it's a list
                    if isinstance(entities[entity_type], list):
                        result[entity_type] = entities[entity_type]
                    else:
                        result[entity_type] = [entities[entity_type]]
                else:
                    result[entity_type] = []
            
            return result
            
        except (json.JSONDecodeError, TypeError):
            # Return empty if parsing fails
            return {et: [] for et in entity_types}
    
    def _convert_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert dates to ISO format."""
        if 'date_converted' not in data.columns:
            data['date_converted'] = pd.to_datetime(
                data['date'].str[6:10] + '-' +
                data['date'].str[0:2] + '-' +
                data['date'].str[3:5],
                format='%Y-%m-%d',
                errors='coerce'
            )
        return data
    
    def _convert_dates_parallel(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert dates to ISO format in parallel for large datasets."""
        if 'date_converted' in data.columns:
            return data
        
        # For very large datasets, vectorized operation is actually faster
        # than parallel processing due to pandas optimization
        logger.info(f"Converting {len(data):,} dates using vectorized operation...")
        
        # Vectorized conversion - MUCH faster than parallel for this operation
        data['date_converted'] = pd.to_datetime(
            data['date'].str[6:10] + '-' +
            data['date'].str[0:2] + '-' +
            data['date'].str[3:5],
            format='%Y-%m-%d',
            errors='coerce'
        )
        
        return data
    
    def _parse_ner_entities(self, ner_json: str) -> Dict[str, List[str]]:
        """
        Parse NER entities from JSON string.
        
        Args:
            ner_json: JSON string with NER entities
            
        Returns:
            Dictionary with entity types and lists
        """
        if pd.isna(ner_json) or ner_json == '' or ner_json == 'null':
            return {et: [] for et in self.entity_types}
        
        try:
            # Try to parse as JSON
            if isinstance(ner_json, str):
                entities = json.loads(ner_json)
            else:
                entities = ner_json
            
            # Ensure all entity types are present
            result = {}
            for entity_type in self.entity_types:
                if entity_type in entities:
                    # Ensure it's a list
                    if isinstance(entities[entity_type], list):
                        result[entity_type] = entities[entity_type]
                    else:
                        result[entity_type] = [entities[entity_type]]
                else:
                    result[entity_type] = []
            
            return result
            
        except (json.JSONDecodeError, TypeError) as e:
            # Return empty if parsing fails
            return {et: [] for et in self.entity_types}
    
    def _calculate_authority_scores(self) -> None:
        """
        Calculate authority scores for all entities.
        Authority = frequency × log(diversity)
        """
        for entity_key, entity_data in self.index.items():
            frequency = entity_data['count']
            
            # Diversity = unique journalists + unique media
            diversity = len(entity_data['journalists']) + len(entity_data['media'])
            
            # Authority score
            if diversity > 0:
                authority_score = frequency * np.log1p(diversity)
            else:
                authority_score = 0.0
            
            entity_data['authority_score'] = float(authority_score)
    
    def _calculate_co_mentions(self, data: pd.DataFrame) -> None:
        """
        Calculate co-mentions between entities based on document co-occurrence.
        OPTIMIZED: Skip if too many entities to avoid O(n²) explosion.
        """
        # Skip co-mention calculation if too many entities (causes slowdown)
        if len(self.index) > 10000:
            logger.info(f"Skipping co-mention calculation for {len(self.index):,} entities (too many)")
            return
            
        logger.info(f"Calculating co-mentions for {len(self.index):,} entities...")
        import time
        start_time = time.time()
        
        # Group entities by document
        doc_entities = defaultdict(set)
        
        for entity_key, entity_data in self.index.items():
            for article_id in entity_data.get('articles', []):
                doc_entities[article_id].add(entity_key)
        
        # Count co-occurrences (limit to avoid explosion)
        mention_count = 0
        max_mentions = 1000000  # Cap at 1M co-mentions
        
        for doc_id, entities in doc_entities.items():
            if mention_count > max_mentions:
                logger.warning(f"Co-mention calculation capped at {max_mentions:,} mentions")
                break
                
            entities = list(entities)
            # Limit entities per doc to avoid O(n²) explosion
            if len(entities) > 100:
                entities = entities[:100]
                
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    entity1, entity2 = entities[i], entities[j]
                    
                    # Update co-mentions for both entities
                    if entity2 not in self.index[entity1]['co_mentions']:
                        self.index[entity1]['co_mentions'][entity2] = 0
                    self.index[entity1]['co_mentions'][entity2] += 1
                    
                    if entity1 not in self.index[entity2]['co_mentions']:
                        self.index[entity2]['co_mentions'][entity1] = 0
                    self.index[entity2]['co_mentions'][entity1] += 1
                    
                    mention_count += 2
        
        elapsed = time.time() - start_time
        logger.info(f"Co-mention calculation took {elapsed:.1f}s ({mention_count:,} mentions)")
    
    def update_index(self, new_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Update index with new data."""
        logger.info(f"Updating entity index with {len(new_data):,} new rows...")
        
        # Process new data similar to build_index
        new_data = self._convert_dates(new_data)
        
        for idx, row in new_data.iterrows():
            try:
                if pd.notna(row.get('ner_entities')):
                    entities = self._parse_ner_entities(row['ner_entities'])
                    
                    for entity_type, entity_list in entities.items():
                        for entity_name in entity_list:
                            entity_key = f"{entity_type}:{entity_name}"
                            
                            # Initialize if new entity
                            if entity_key not in self.index:
                                self.index[entity_key] = {
                                    'type': entity_type,
                                    'name': entity_name,
                                    'citations': [],
                                    'count': 0,
                                    'journalists': [],
                                    'media': [],
                                    'authority_score': 0.0
                                }
                            
                            # Update entity data
                            self.index[entity_key]['count'] += 1
                            
                            citation = {
                                'date': row['date_converted'],
                                'doc_id': row['doc_id'],
                                'author': row.get('author', 'Unknown'),
                                'media': row.get('media', 'Unknown')
                            }
                            self.index[entity_key]['citations'].append(citation)
                            
                            # Update lists (convert to sets temporarily)
                            journalists = set(self.index[entity_key]['journalists'])
                            media = set(self.index[entity_key]['media'])
                            
                            if pd.notna(row.get('author')):
                                journalists.add(row['author'])
                            if pd.notna(row.get('media')):
                                media.add(row['media'])
                            
                            self.index[entity_key]['journalists'] = list(journalists)
                            self.index[entity_key]['media'] = list(media)
                            
            except Exception as e:
                logger.debug(f"Error updating entity at row {idx}: {e}")
        
        # Recalculate authority scores
        self._calculate_authority_scores()
        
        # Update metadata
        self.metadata['updated'] = datetime.now().isoformat()
        
        return self.index
    
    def query_index(self, criteria: Dict[str, Any]) -> List[Any]:
        """
        Query entity index.
        
        Criteria:
            - entity_type: 'PER', 'ORG', or 'LOC'
            - min_authority: Minimum authority score
            - min_count: Minimum citation count
            - journalist: Specific journalist
            - media: Specific media outlet
            - date_range: (start, end) tuple
        """
        results = []
        
        entity_type = criteria.get('entity_type')
        min_authority = criteria.get('min_authority', 0)
        min_count = criteria.get('min_count', 0)
        journalist = criteria.get('journalist')
        media = criteria.get('media')
        date_range = criteria.get('date_range')
        
        for entity_key, entity_data in self.index.items():
            # Filter by entity type
            if entity_type and not entity_key.startswith(f"{entity_type}:"):
                continue
            
            # Filter by authority score
            if entity_data['authority_score'] < min_authority:
                continue
            
            # Filter by count
            if entity_data['count'] < min_count:
                continue
            
            # Filter by journalist
            if journalist and journalist not in entity_data['journalists']:
                continue
            
            # Filter by media
            if media and media not in entity_data['media']:
                continue
            
            # Filter by date range
            if date_range:
                start_date, end_date = date_range
                citations_in_range = [
                    c for c in entity_data['citations']
                    if start_date <= c['date'] <= end_date
                ]
                if not citations_in_range:
                    continue
            
            results.append((entity_key, entity_data))
        
        return results
    
    def get_top_authorities(self, n: int = 10, 
                           entity_type: Optional[str] = None) -> List[Tuple[str, Dict]]:
        """
        Get top epistemic authorities.
        
        Args:
            n: Number of top authorities to return
            entity_type: Filter by entity type
            
        Returns:
            List of (entity_key, entity_data) tuples
        """
        # Filter by type if specified
        if entity_type:
            entities = [(k, v) for k, v in self.index.items() 
                       if k.startswith(f"{entity_type}:")]
        else:
            entities = list(self.index.items())
        
        # Sort by authority score
        entities.sort(key=lambda x: x[1]['authority_score'], reverse=True)
        
        return entities[:n]
    
    def get_entity_network(self, min_cooccurrence: int = 2) -> Dict[str, List[Tuple[str, int]]]:
        """
        Get co-occurrence network of entities.
        
        Args:
            min_cooccurrence: Minimum co-occurrences to include edge
            
        Returns:
            Adjacency list of entity co-occurrences
        """
        # Build co-occurrence matrix
        cooccurrence = defaultdict(Counter)
        
        # Group citations by document
        doc_entities = defaultdict(set)
        for entity_key, entity_data in self.index.items():
            for citation in entity_data['citations']:
                doc_entities[citation['doc_id']].add(entity_key)
        
        # Count co-occurrences
        for entities in doc_entities.values():
            entities = list(entities)
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    cooccurrence[entities[i]][entities[j]] += 1
                    cooccurrence[entities[j]][entities[i]] += 1
        
        # Filter by minimum co-occurrence
        network = {}
        for entity1, connections in cooccurrence.items():
            edges = [(entity2, count) for entity2, count in connections.items()
                    if count >= min_cooccurrence]
            if edges:
                network[entity1] = edges
        
        return network
    
    def get_entity_timeline(self, entity_key: str) -> pd.Series:
        """
        Get timeline of entity citations.
        
        Args:
            entity_key: Entity key (e.g., 'PER:John Doe')
            
        Returns:
            Time series of citation counts
        """
        if entity_key not in self.index:
            return pd.Series()
        
        # Extract dates from citations
        dates = [c['date'] for c in self.index[entity_key]['citations']]
        
        # Count by date
        date_counts = Counter(dates)
        
        # Create time series
        if date_counts:
            series = pd.Series(date_counts)
            series = series.sort_index()
            
            # Fill missing dates with 0
            date_range = pd.date_range(start=series.index.min(), 
                                      end=series.index.max(), 
                                      freq='D')
            series = series.reindex(date_range, fill_value=0)
            
            return series
        
        return pd.Series()
    
    def find_entity_bursts(self, entity_key: str, 
                          window: int = 7) -> List[Dict]:
        """
        Find burst periods for an entity.
        
        Args:
            entity_key: Entity key
            window: Rolling window size in days
            
        Returns:
            List of burst periods
        """
        timeline = self.get_entity_timeline(entity_key)
        if timeline.empty:
            return []
        
        # Calculate rolling mean and std
        rolling_mean = timeline.rolling(window=window, center=True).mean()
        rolling_std = timeline.rolling(window=window, center=True).std()
        
        # Detect bursts (> mean + 2*std)
        threshold = rolling_mean + 2 * rolling_std
        burst_mask = timeline > threshold
        
        # Extract burst periods
        bursts = []
        in_burst = False
        burst_start = None
        
        for date, is_burst in burst_mask.items():
            if is_burst and not in_burst:
                burst_start = date
                in_burst = True
            elif not is_burst and in_burst:
                bursts.append({
                    'entity': entity_key,
                    'start': burst_start,
                    'end': date,
                    'peak_count': int(timeline[burst_start:date].max()),
                    'total_citations': int(timeline[burst_start:date].sum())
                })
                in_burst = False
        
        return bursts