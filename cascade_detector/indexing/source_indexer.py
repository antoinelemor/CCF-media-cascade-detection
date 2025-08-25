"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
source_indexer.py

MAIN OBJECTIVE:
---------------
This script indexes messenger and source citations for pattern analysis, tracking which sources
are cited by whom and when, building journalist and media outlet profiles with influence metrics.

Dependencies:
-------------
- pandas
- numpy
- typing
- collections
- datetime
- logging
- concurrent.futures
- multiprocessing
- tqdm

MAIN FEATURES:
--------------
1) Journalist and media outlet profile construction
2) Intelligent author resolution to merge name variations
3) Influence network construction between sources
4) Messenger type tracking and aggregation
5) Temporal evolution of source citations

Author:
-------
Antoine Lemor
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from datetime import datetime
import logging

from cascade_detector.indexing.base_indexer import AbstractIndexer
from cascade_detector.core.constants import MESSENGERS, MESSENGER_TYPES, MESSENGER_MAIN
from cascade_detector.utils.author_resolver import AuthorResolver
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
try:
    from cascade_detector.utils.media_geography import MediaGeography
    HAS_MEDIA_GEOGRAPHY = True
except ImportError:
    HAS_MEDIA_GEOGRAPHY = False

logger = logging.getLogger(__name__)


class SourceIndexer(AbstractIndexer):
    """
    Indexes messenger/source citations for pattern analysis.
    Tracks which sources are cited by whom and when.
    """
    
    def __init__(self, resolve_authors: bool = True):
        """Initialize source indexer.
        
        Args:
            resolve_authors: Whether to apply intelligent author resolution
        """
        super().__init__(name="SourceIndexer")
        self.messenger_cols = MESSENGERS
        self.messenger_types = MESSENGER_TYPES
        self.resolve_authors = resolve_authors
        
        # Initialize author resolver if needed
        if self.resolve_authors:
            self.author_resolver = AuthorResolver(
                similarity_threshold=0.88,
                context_weight=0.4,
                min_articles=2
            )
        
        # Initialize media geography for filtering valid media
        if HAS_MEDIA_GEOGRAPHY:
            self.media_geography = MediaGeography()
            self.valid_media_only = True
            logger.info(f"SourceIndexer filtering to {len(self.media_geography.valid_media)} valid media")
        else:
            self.media_geography = None
            self.valid_media_only = False
        
    def get_required_columns(self) -> List[str]:
        """Get required columns."""
        # Only require base columns, messengers are optional
        base_cols = ['date', 'doc_id', 'sentence_id', 'media', 'author']
        return base_cols
    
    def build_index(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Build source index from data.
        
        Structure: {
            'article_profiles': {
                doc_id: {
                    'author': str,
                    'media': str,
                    'date': timestamp,
                    'messenger_proportions': {messenger_type: float},
                    'dominant_messenger': str,
                    'diversity': float
                }
            },
            'journalist_profiles': {
                journalist: {
                    'articles': [doc_ids],
                    'avg_proportions': {messenger_type: float},
                    'consistency': float
                }
            },
            'media_profiles': {
                media: {
                    'articles': [doc_ids],
                    'avg_proportions': {messenger_type: float},
                    'consistency': float
                }
            },
            'temporal_evolution': {
                date: {messenger_type: proportion}
            }
        }
        """
        logger.info(f"Building source index from {len(data):,} rows...")
        self.validate_data(data)
        
        # Convert dates
        data = self._convert_dates(data)
        
        # Initialize index structure
        self.index = {
            'article_profiles': {},
            'journalist_profiles': defaultdict(lambda: {
                'articles': [],
                'proportions_history': [],
                'avg_proportions': {},
                'consistency': 0.0
            }),
            'media_profiles': defaultdict(lambda: {
                'articles': [],
                'proportions_history': [],
                'avg_proportions': {},
                'consistency': 0.0
            }),
            'temporal_evolution': defaultdict(lambda: defaultdict(float)),
            'influence_network': None  # NEW: Will be populated after processing
        }
        
        # Process articles
        article_groups = data.groupby('doc_id')
        
        for doc_id, article_data in article_groups:
            # Skip if media is not in valid list
            media = article_data['media'].iloc[0] if not article_data.empty else None
            if self.valid_media_only and media:
                if not self.media_geography.is_valid_media(media):
                    continue  # Skip articles from invalid media
            
            # Calculate messenger proportions for article
            profile = self._calculate_article_profile(doc_id, article_data)
            self.index['article_profiles'][doc_id] = profile
            
            # Update journalist profile
            if pd.notna(profile['author']):
                self.index['journalist_profiles'][profile['author']]['articles'].append(doc_id)
                self.index['journalist_profiles'][profile['author']]['proportions_history'].append(
                    profile['messenger_proportions']
                )
            
            # Update media profile
            if pd.notna(profile['media']):
                self.index['media_profiles'][profile['media']]['articles'].append(doc_id)
                self.index['media_profiles'][profile['media']]['proportions_history'].append(
                    profile['messenger_proportions']
                )
            
            # Update temporal evolution
            week = pd.Timestamp(profile['date']).to_period('W').to_timestamp()
            for msg_type, prop in profile['messenger_proportions'].items():
                self.index['temporal_evolution'][week][msg_type] += prop
        
        # Calculate aggregated profiles
        self._calculate_aggregated_profiles()
        
        # Apply author resolution if enabled
        if self.resolve_authors:
            original_count = len(self.index['journalist_profiles'])
            logger.info(f"Applying intelligent author resolution to {original_count} journalists...")
            
            # Resolve author names
            resolved_profiles, name_mapping = self.author_resolver.resolve_authors(
                self.index['journalist_profiles'],
                self.index['article_profiles'],
                use_context=True
            )
            
            # Update journalist profiles
            self.index['journalist_profiles'] = resolved_profiles
            
            # Store name mapping for reference
            self.index['author_name_mapping'] = name_mapping
            
            # Update article profiles with canonical author names IN PARALLEL
            logger.info(f"Updating {len(self.index['article_profiles'])} article profiles with canonical names...")
            self._parallel_update_article_authors(name_mapping)
            
            resolved_count = len(resolved_profiles)
            logger.info(f"Author resolution: {original_count} → {resolved_count} unique journalists")
        
        # Run post-processing steps IN PARALLEL
        logger.info("Running post-processing steps in parallel...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks
            futures = [
                executor.submit(self._normalize_temporal_evolution),
                executor.submit(self._build_influence_network),
                executor.submit(self._finalize_profiles)  # New method for profile finalization
            ]
            
            # Wait for completion with progress
            with tqdm(total=len(futures), desc="Post-processing", unit="task") as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Post-processing task failed: {e}")
                        pbar.update(1)
        
        # Convert defaultdicts to regular dicts
        self.index['journalist_profiles'] = dict(self.index['journalist_profiles'])
        self.index['media_profiles'] = dict(self.index['media_profiles'])
        self.index['temporal_evolution'] = dict(self.index['temporal_evolution'])
        
        # Update metadata
        self.metadata['created'] = datetime.now().isoformat()
        self.metadata['n_entries'] = len(self.index['article_profiles'])
        self.metadata['n_journalists'] = len(self.index['journalist_profiles'])
        self.metadata['n_media'] = len(self.index['media_profiles'])
        self.metadata['author_resolution_applied'] = self.resolve_authors
        
        logger.info(f"Source index built: {self.metadata['n_entries']} articles, "
                   f"{self.metadata['n_journalists']} journalists, "
                   f"{self.metadata['n_media']} media outlets")
        
        # For large indices, prepare for efficient transfer WITHOUT data loss
        if self.metadata['n_entries'] > 50000:
            logger.info("Large index detected - preserving all data for scientific accuracy")
            
            # Calculate index size for monitoring
            import sys
            import numpy as np
            
            # Estimate size
            est_size = sys.getsizeof(self.index['article_profiles']) / (1024**2)
            est_size += sys.getsizeof(self.index['journalist_profiles']) / (1024**2)
            
            if self.index.get('influence_network') is not None:
                if isinstance(self.index['influence_network'], np.ndarray):
                    est_size += self.index['influence_network'].nbytes / (1024**2)
            
            logger.info(f"  Estimated index size: {est_size:.1f} MB")
            
            # Mark for potential shared memory transfer
            self.metadata['large_index'] = True
            self.metadata['estimated_size_mb'] = est_size
        
        return self.index
    
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
    
    def _calculate_article_profile(self, doc_id: str, article_data: pd.DataFrame) -> Dict:
        """Calculate messenger profile for an article."""
        # Get article metadata
        first_row = article_data.iloc[0]
        author = first_row.get('author', 'Unknown')
        media = first_row.get('media', 'Unknown')
        date = first_row['date_converted']
        
        # NEW: Extract NER entities from article
        entities = set()
        for idx, row in article_data.iterrows():
            if pd.notna(row.get('ner_entities')):
                try:
                    import json
                    ner_data = json.loads(row['ner_entities']) if isinstance(row['ner_entities'], str) else row['ner_entities']
                    for entity_type, entity_list in ner_data.items():
                        for entity_name in entity_list:
                            entities.add(f"{entity_type}:{entity_name}")
                except:
                    pass
        
        # Calculate messenger proportions
        total_sentences = len(article_data)
        messenger_proportions = {}
        
        # First check if there are any messengers at all
        messenger_sentences = 0
        if MESSENGER_MAIN in article_data.columns:
            article_data[MESSENGER_MAIN] = pd.to_numeric(
                article_data[MESSENGER_MAIN], errors='coerce'
            ).fillna(0)
            messenger_sentences = (article_data[MESSENGER_MAIN] == 1).sum()
        
        # Only analyze sub-types if there are messengers
        for messenger_col, messenger_type in self.messenger_types.items():
            if messenger_col in article_data.columns:
                # Convert to numeric and count
                article_data[messenger_col] = pd.to_numeric(
                    article_data[messenger_col], errors='coerce'
                ).fillna(0)
                
                # Count only within sentences that have messengers
                if messenger_sentences > 0:
                    # Get sentences with messengers
                    messenger_mask = article_data[MESSENGER_MAIN] == 1 if MESSENGER_MAIN in article_data.columns else pd.Series([True] * len(article_data))
                    count = (article_data.loc[messenger_mask, messenger_col] == 1).sum()
                    proportion = count / messenger_sentences  # Proportion among messenger sentences
                else:
                    count = 0
                    proportion = 0
                    
                messenger_proportions[messenger_type] = proportion
        
        # Find dominant messenger
        if messenger_proportions:
            dominant_messenger = max(messenger_proportions, key=messenger_proportions.get)
            max_proportion = messenger_proportions[dominant_messenger]
        else:
            dominant_messenger = None
            max_proportion = 0
        
        # Calculate diversity (Shannon entropy)
        proportions = np.array(list(messenger_proportions.values()))
        diversity = self._calculate_shannon_entropy(proportions)
        
        return {
            'doc_id': doc_id,
            'author': author,
            'media': media,
            'date': date,
            'entities': list(entities),  # NEW: Add entities to profile
            'n_entities': len(entities),  # NEW: Count of entities
            'messenger_proportions': messenger_proportions,
            'dominant_messenger': dominant_messenger,
            'dominant_proportion': max_proportion,
            'diversity': diversity,
            'n_sentences': total_sentences,
            'n_messenger_sentences': messenger_sentences,
            'messenger_rate': messenger_sentences / total_sentences if total_sentences > 0 else 0
        }
    
    def _calculate_shannon_entropy(self, proportions: np.ndarray) -> float:
        """Calculate Shannon entropy for diversity measurement."""
        # Remove zeros
        p = proportions[proportions > 0]
        if len(p) == 0:
            return 0
        
        # Normalize
        p = p / p.sum()
        
        # Calculate entropy
        return float(-np.sum(p * np.log(p)))
    
    def _calculate_aggregated_profiles(self) -> None:
        """Calculate aggregated profiles for journalists and media IN PARALLEL."""
        n_journalists = len(self.index['journalist_profiles'])
        n_media = len(self.index['media_profiles'])
        
        logger.info(f"Calculating aggregated profiles for {n_journalists:,} journalists and {n_media:,} media...")
        
        # Process in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            
            # Submit journalist profile calculations
            journalist_chunks = self._chunk_dict(self.index['journalist_profiles'], 1000)
            for chunk in journalist_chunks:
                future = executor.submit(self._process_profile_chunk, chunk, 'journalist')
                futures.append(future)
            
            # Submit media profile calculations
            media_chunks = self._chunk_dict(self.index['media_profiles'], 1000)
            for chunk in media_chunks:
                future = executor.submit(self._process_profile_chunk, chunk, 'media')
                futures.append(future)
            
            # Wait for completion with progress
            total_chunks = len(journalist_chunks) + len(media_chunks)
            with tqdm(total=total_chunks, desc="Aggregating profiles", unit="chunks") as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Failed to process profile chunk: {e}")
                        pbar.update(1)
    
    def _chunk_dict(self, data: Dict, chunk_size: int) -> List[Dict]:
        """Split dictionary into chunks for parallel processing."""
        items = list(data.items())
        chunks = []
        for i in range(0, len(items), chunk_size):
            chunk = dict(items[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def _process_profile_chunk(self, chunk: Dict, profile_type: str) -> None:
        """Process a chunk of profiles."""
        for name, profile in chunk.items():
            if profile.get('proportions_history'):
                # Calculate average proportions
                avg_props = defaultdict(list)
                for props in profile['proportions_history']:
                    for msg_type, value in props.items():
                        avg_props[msg_type].append(value)
                
                profile['avg_proportions'] = {
                    msg_type: np.mean(values) 
                    for msg_type, values in avg_props.items()
                }
                
                # Calculate consistency
                consistencies = []
                for msg_type, values in avg_props.items():
                    if len(values) > 1:
                        std = np.std(values)
                        consistency = 1 / (1 + std)
                        consistencies.append(consistency)
                
                profile['consistency'] = np.mean(consistencies) if consistencies else 1.0
                
                # Remove history to save memory
                del profile['proportions_history']
    
    def _normalize_temporal_evolution(self) -> None:
        """Normalize temporal evolution by week - VECTORIZED."""
        import numpy as np
        
        # Vectorized normalization for all weeks at once
        temporal_evo = self.index['temporal_evolution']
        if not temporal_evo:
            return
            
        # Process all weeks in batch
        for week, proportions in temporal_evo.items():
            if proportions:
                # Convert to numpy for vectorized ops
                values = np.array(list(proportions.values()))
                total = values.sum()
                if total > 0:
                    # Vectorized division
                    normalized = values / total
                    # Update all at once
                    for i, msg_type in enumerate(proportions.keys()):
                        proportions[msg_type] = float(normalized[i])
    
    def _build_influence_network(self) -> None:
        """
        Build influence network matrix using VECTORIZED operations.
        Optimized for M4 Max with parallel computation.
        """
        import numpy as np
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            HAS_SKLEARN = True
        except ImportError:
            HAS_SKLEARN = False
            logger.warning("sklearn not available, using numpy for similarity computation")
        from concurrent.futures import ThreadPoolExecutor
        import warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        # Get all journalists and media
        journalists = list(self.index['journalist_profiles'].keys())
        media = list(self.index['media_profiles'].keys())
        all_sources = journalists + media
        n_sources = len(all_sources)
        
        if n_sources == 0:
            self.index['influence_network'] = np.zeros((0, 0))
            return
        
        # For large networks, sample or use approximation
        MAX_SOURCES = 5000  # Limit for full computation
        if n_sources > MAX_SOURCES:
            logger.info(f"Large network ({n_sources} sources), using top {MAX_SOURCES} by activity")
            # Sort by number of articles and take top sources
            source_activity = []
            for source in all_sources:
                profile = (self.index['journalist_profiles'].get(source) or 
                          self.index['media_profiles'].get(source))
                n_articles = len(profile.get('articles', [])) if profile else 0
                source_activity.append((source, n_articles))
            
            source_activity.sort(key=lambda x: x[1], reverse=True)
            all_sources = [s[0] for s in source_activity[:MAX_SOURCES]]
            n_sources = len(all_sources)
        
        logger.info(f"Building influence network for {n_sources} sources...")
        
        # Get all messenger types
        all_msg_types = set()
        for source in all_sources:
            profile = (self.index['journalist_profiles'].get(source) or 
                      self.index['media_profiles'].get(source))
            if profile and profile.get('avg_proportions'):
                all_msg_types.update(profile['avg_proportions'].keys())
        
        msg_types = sorted(all_msg_types)
        n_features = len(msg_types)
        
        if n_features == 0:
            self.index['influence_network'] = np.zeros((n_sources, n_sources))
            return
        
        # Build feature matrix (sources x messenger_types)
        feature_matrix = np.zeros((n_sources, n_features))
        article_counts = np.zeros(n_sources)
        
        for i, source in enumerate(all_sources):
            profile = (self.index['journalist_profiles'].get(source) or 
                      self.index['media_profiles'].get(source))
            if profile and profile.get('avg_proportions'):
                props = profile['avg_proportions']
                for j, msg_type in enumerate(msg_types):
                    feature_matrix[i, j] = props.get(msg_type, 0)
                article_counts[i] = len(profile.get('articles', []))
        
        # Calculate cosine similarity matrix in one operation
        # This is MUCH faster than nested loops
        if HAS_SKLEARN:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                similarity_matrix = cosine_similarity(feature_matrix)
        else:
            # Fallback to numpy implementation
            # Normalize rows to unit vectors
            row_norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
            row_norms[row_norms == 0] = 1  # Avoid division by zero
            normalized = feature_matrix / row_norms
            # Cosine similarity = normalized dot product
            similarity_matrix = np.dot(normalized, normalized.T)
        
        # Apply article count weighting
        # Create weight matrix
        weight_matrix = np.outer(article_counts, article_counts)
        weight_matrix = np.minimum(weight_matrix / 100, 1.0)  # Normalize weights
        
        # Element-wise multiplication for final influence
        influence_matrix = similarity_matrix * weight_matrix
        
        # Zero out diagonal
        np.fill_diagonal(influence_matrix, 0)
        
        # Store the matrix
        self.index['influence_network'] = influence_matrix
        
        logger.info(f"Influence network built: {n_sources}x{n_sources} matrix")
    
    def update_index(self, new_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Update index with new data."""
        logger.info(f"Updating source index with {len(new_data):,} new rows...")
        
        # Convert dates
        new_data = self._convert_dates(new_data)
        
        # Process new articles
        article_groups = new_data.groupby('doc_id')
        
        for doc_id, article_data in article_groups:
            # Skip if already indexed
            if doc_id in self.index['article_profiles']:
                continue
            
            # Calculate profile
            profile = self._calculate_article_profile(doc_id, article_data)
            self.index['article_profiles'][doc_id] = profile
            
            # Update journalist profile
            if pd.notna(profile['author']):
                if profile['author'] not in self.index['journalist_profiles']:
                    self.index['journalist_profiles'][profile['author']] = {
                        'articles': [],
                        'avg_proportions': {},
                        'consistency': 0.0
                    }
                self.index['journalist_profiles'][profile['author']]['articles'].append(doc_id)
            
            # Update media profile
            if pd.notna(profile['media']):
                if profile['media'] not in self.index['media_profiles']:
                    self.index['media_profiles'][profile['media']] = {
                        'articles': [],
                        'avg_proportions': {},
                        'consistency': 0.0
                    }
                self.index['media_profiles'][profile['media']]['articles'].append(doc_id)
        
        # Recalculate aggregated profiles
        self._recalculate_profiles_for_update()
        
        # Update metadata
        self.metadata['updated'] = datetime.now().isoformat()
        self.metadata['n_entries'] = len(self.index['article_profiles'])
        
        return self.index
    
    def _recalculate_profiles_for_update(self) -> None:
        """Recalculate profiles after update."""
        # Simplified recalculation for updates
        for journalist, profile in self.index['journalist_profiles'].items():
            articles = profile['articles']
            if articles:
                proportions = defaultdict(list)
                for doc_id in articles:
                    if doc_id in self.index['article_profiles']:
                        article_props = self.index['article_profiles'][doc_id]['messenger_proportions']
                        for msg_type, value in article_props.items():
                            proportions[msg_type].append(value)
                
                profile['avg_proportions'] = {
                    msg_type: np.mean(values)
                    for msg_type, values in proportions.items()
                }
    
    def query_index(self, criteria: Dict[str, Any]) -> List[Any]:
        """
        Query source index.
        
        Criteria:
            - journalist: Specific journalist
            - media: Specific media
            - messenger_type: Specific messenger type
            - min_proportion: Minimum proportion for dominant messenger
            - date_range: (start, end) tuple
        """
        results = []
        
        journalist = criteria.get('journalist')
        media = criteria.get('media')
        messenger_type = criteria.get('messenger_type')
        min_proportion = criteria.get('min_proportion', 0)
        date_range = criteria.get('date_range')
        
        for doc_id, profile in self.index['article_profiles'].items():
            # Filter by journalist
            if journalist and profile['author'] != journalist:
                continue
            
            # Filter by media
            if media and profile['media'] != media:
                continue
            
            # Filter by messenger type
            if messenger_type:
                if messenger_type not in profile['messenger_proportions']:
                    continue
                if profile['messenger_proportions'][messenger_type] < min_proportion:
                    continue
            
            # Filter by date range
            if date_range:
                start_date, end_date = date_range
                if not (start_date <= profile['date'] <= end_date):
                    continue
            
            results.append((doc_id, profile))
        
        return results
    
    def _parallel_update_article_authors(self, name_mapping: Dict[str, str]) -> None:
        """
        Update article profiles with canonical author names in parallel.
        Optimized for M4 Max with batched updates.
        """
        # Convert to list for chunking
        article_items = list(self.index['article_profiles'].items())
        n_articles = len(article_items)
        
        if n_articles == 0:
            return
        
        # Process in chunks for better memory efficiency
        chunk_size = 10000
        n_chunks = (n_articles + chunk_size - 1) // chunk_size
        
        logger.info(f"Updating {n_articles:,} articles in {n_chunks} chunks...")
        
        # Use ThreadPoolExecutor for this I/O-bound task
        with ThreadPoolExecutor(max_workers=min(16, n_chunks)) as executor:
            futures = []
            
            for i in range(0, n_articles, chunk_size):
                chunk = article_items[i:i + chunk_size]
                future = executor.submit(self._update_article_chunk, chunk, name_mapping)
                futures.append(future)
            
            # Process with progress bar
            with tqdm(total=n_chunks, desc="Updating articles", unit="chunks") as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Failed to update article chunk: {e}")
                        pbar.update(1)
    
    def _update_article_chunk(self, chunk: List[Tuple[str, Dict]], name_mapping: Dict[str, str]) -> None:
        """Update a chunk of articles with canonical author names."""
        for doc_id, profile in chunk:
            if profile['author'] in name_mapping:
                profile['original_author'] = profile['author']
                profile['author'] = name_mapping[profile['author']]
    
    def _finalize_profiles(self) -> None:
        """
        Finalize journalist and media profiles.
        This is separated to run in parallel with other tasks.
        """
        # This combines what was being done sequentially
        # Already handled by _calculate_aggregated_profiles
        pass
    
    def get_journalist_profile(self, journalist: str) -> Dict:
        """Get profile for a specific journalist."""
        return self.index['journalist_profiles'].get(journalist, {})
    
    def get_media_profile(self, media: str) -> Dict:
        """Get profile for a specific media outlet."""
        return self.index['media_profiles'].get(media, {})
    
    def calculate_source_convergence(self, 
                                    start_date: pd.Timestamp,
                                    end_date: pd.Timestamp) -> float:
        """
        Calculate source convergence over a period.
        
        Convergence = decrease in diversity over time.
        """
        # Get articles in period
        articles_in_period = [
            profile for doc_id, profile in self.index['article_profiles'].items()
            if start_date <= profile['date'] <= end_date
        ]
        
        if not articles_in_period:
            return 0.0
        
        # Sort by date
        articles_in_period.sort(key=lambda x: x['date'])
        
        # Split into early and late periods
        mid_point = len(articles_in_period) // 2
        early_articles = articles_in_period[:mid_point]
        late_articles = articles_in_period[mid_point:]
        
        # Calculate average diversity for each period
        early_diversity = np.mean([a['diversity'] for a in early_articles])
        late_diversity = np.mean([a['diversity'] for a in late_articles])
        
        # Convergence = decrease in diversity
        convergence = max(0, (early_diversity - late_diversity) / early_diversity)
        
        return float(convergence)
    
    def find_source_patterns(self, 
                            window_size: int = 7,
                            min_correlation: float = 0.7) -> List[Dict]:
        """
        Find correlated source citation patterns.
        
        Args:
            window_size: Window size in days
            min_correlation: Minimum correlation threshold
            
        Returns:
            List of pattern dictionaries
        """
        patterns = []
        
        # Group articles by week
        weekly_groups = defaultdict(list)
        for doc_id, profile in self.index['article_profiles'].items():
            week = pd.Timestamp(profile['date']).to_period('W').to_timestamp()
            weekly_groups[week].append(profile)
        
        # Analyze patterns in each week
        for week, articles in weekly_groups.items():
            if len(articles) < 3:  # Need minimum articles
                continue
            
            # Calculate correlation matrix for messengers
            messenger_vectors = []
            for article in articles:
                vector = [article['messenger_proportions'].get(mt, 0) 
                         for mt in self.messenger_types.values()]
                messenger_vectors.append(vector)
            
            # Calculate pairwise correlations
            if len(messenger_vectors) > 1:
                corr_matrix = np.corrcoef(messenger_vectors)
                
                # Find high correlations
                high_corr_pairs = []
                for i in range(len(articles)):
                    for j in range(i+1, len(articles)):
                        if corr_matrix[i, j] >= min_correlation:
                            high_corr_pairs.append({
                                'week': week,
                                'article1': articles[i]['doc_id'],
                                'article2': articles[j]['doc_id'],
                                'correlation': float(corr_matrix[i, j]),
                                'author1': articles[i]['author'],
                                'author2': articles[j]['author'],
                                'media1': articles[i]['media'],
                                'media2': articles[j]['media']
                            })
                
                if high_corr_pairs:
                    patterns.append({
                        'week': week,
                        'n_correlated_pairs': len(high_corr_pairs),
                        'pairs': high_corr_pairs[:5]  # Top 5 pairs
                    })
        
        return patterns