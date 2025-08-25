"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
frame_indexer.py

MAIN OBJECTIVE:
---------------
This script indexes frame co-occurrences and patterns for multi-frame cascade detection, analyzing
how different frames appear together and evolve over time in media coverage.

Dependencies:
-------------
- pandas
- numpy
- typing
- collections
- datetime
- logging

MAIN FEATURES:
--------------
1) Frame co-occurrence matrix construction
2) Article-level frame distribution analysis
3) Temporal frame sequence tracking
4) Frame dominance and convergence metrics
5) Multi-frame pattern identification

Author:
-------
Antoine Lemor
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import logging

from cascade_detector.indexing.base_indexer import AbstractIndexer
from cascade_detector.core.constants import FRAMES

logger = logging.getLogger(__name__)


class FrameIndexer(AbstractIndexer):
    """
    Indexes frame co-occurrences and patterns.
    Enables multi-frame cascade detection.
    """
    
    def __init__(self):
        """Initialize frame indexer."""
        super().__init__(name="FrameIndexer")
        self.frames = FRAMES
        
    def get_required_columns(self) -> List[str]:
        """Get required columns."""
        # Only require base columns, frames are optional
        base_cols = ['date', 'doc_id', 'sentence_id', 'media', 'author']
        return base_cols
    
    def build_index(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Build frame co-occurrence index.
        
        Structure: {
            'cooccurrence_matrix': np.array,  # Frame x Frame matrix
            'article_frames': {
                doc_id: {
                    'frames': [active_frames],
                    'dominant_frame': str,
                    'n_frames': int,
                    'frame_proportions': {frame: proportion}
                }
            },
            'temporal_cooccurrence': {
                date: {
                    (frame1, frame2): count
                }
            },
            'frame_sequences': [
                {
                    'sequence': [frame1, frame2, ...],
                    'count': int,
                    'articles': [doc_ids]
                }
            ]
        }
        """
        logger.info(f"Building frame index from {len(data):,} rows...")
        self.validate_data(data)
        
        # Convert dates
        data = self._convert_dates(data)
        
        # Initialize index
        self.index = {
            'cooccurrence_matrix': np.zeros((len(self.frames), len(self.frames))),
            'article_frames': {},
            'temporal_cooccurrence': defaultdict(lambda: defaultdict(int)),
            'frame_sequences': [],
            'frame_statistics': {}
        }
        
        # Process each article
        article_groups = data.groupby('doc_id')
        
        for doc_id, article_data in article_groups:
            profile = self._analyze_article_frames(doc_id, article_data)
            self.index['article_frames'][doc_id] = profile
            
            # Update co-occurrence matrix
            active_frames = profile['frames']
            for i, frame1 in enumerate(active_frames):
                for frame2 in active_frames[i:]:
                    idx1 = self.frames.index(frame1)
                    idx2 = self.frames.index(frame2)
                    self.index['cooccurrence_matrix'][idx1, idx2] += 1
                    if idx1 != idx2:
                        self.index['cooccurrence_matrix'][idx2, idx1] += 1
            
            # Update temporal co-occurrence
            if len(article_data) > 0:
                date = article_data['date_converted'].iloc[0]
                week = pd.Timestamp(date).to_period('W').to_timestamp()
                for i, frame1 in enumerate(active_frames):
                    for frame2 in active_frames[i+1:]:
                        key = tuple(sorted([frame1, frame2]))
                        self.index['temporal_cooccurrence'][week][key] += 1
        
        # Detect frame sequences
        self._detect_frame_sequences(data)
        
        # Calculate frame statistics
        self._calculate_frame_statistics()
        
        # Convert defaultdicts to regular dicts
        self.index['temporal_cooccurrence'] = dict(self.index['temporal_cooccurrence'])
        for date in self.index['temporal_cooccurrence']:
            self.index['temporal_cooccurrence'][date] = dict(
                self.index['temporal_cooccurrence'][date]
            )
        
        # Update metadata
        self.metadata['created'] = datetime.now().isoformat()
        self.metadata['n_entries'] = len(self.index['article_frames'])
        self.metadata['n_cooccurrences'] = int(np.sum(self.index['cooccurrence_matrix']))
        
        logger.info(f"Frame index built: {self.metadata['n_entries']} articles indexed")
        
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
    
    def _analyze_article_frames(self, doc_id: str, article_data: pd.DataFrame) -> Dict:
        """Analyze frame usage in an article."""
        # Guard against empty data
        if article_data.empty or len(article_data) == 0:
            return {
                'doc_id': doc_id,
                'frames': [],
                'dominant_frame': None,
                'primary_frame': None,  # NEW: Alias for dominant_frame
                'n_frames': 0,
                'frame_count': 0,  # NEW: Alias for n_frames
                'frame_proportions': {},
                'frame_distribution': {},  # NEW: Alias for frame_proportions
                'frame_counts': {},
                'entities': [],  # NEW: Add entities
                'total_sentences': 0,
                'date': None,
                'author': 'Unknown',
                'media': 'Unknown'
            }
        
        # Make a copy to avoid modifying shared data in parallel execution
        article_data = article_data.copy()
        
        # Count sentences per frame
        frame_counts = {}
        total_sentences = len(article_data)
        
        for frame in self.frames:
            frame_col = f"{frame}_Detection"
            if frame_col in article_data.columns:
                article_data[frame_col] = pd.to_numeric(
                    article_data[frame_col], errors='coerce'
                ).fillna(0)
                count = (article_data[frame_col] == 1).sum()
                if count > 0:
                    frame_counts[frame] = count
        
        # Calculate proportions
        frame_proportions = {
            frame: count / total_sentences 
            for frame, count in frame_counts.items()
        }
        
        # Identify dominant frame
        if frame_counts:
            dominant_frame = max(frame_counts, key=frame_counts.get)
        else:
            dominant_frame = None
        
        # NEW: Extract entities from article
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
        
        return {
            'doc_id': doc_id,
            'frames': list(frame_counts.keys()),
            'dominant_frame': dominant_frame,
            'primary_frame': dominant_frame,  # NEW: Alias for compatibility
            'n_frames': len(frame_counts),
            'frame_count': len(frame_counts),  # NEW: Alias for compatibility
            'frame_proportions': frame_proportions,
            'frame_distribution': frame_proportions,  # NEW: Alias for compatibility  
            'frame_counts': frame_counts,
            'entities': list(entities),  # NEW: Add entities
            'total_sentences': total_sentences,
            'date': article_data['date_converted'].iloc[0] if len(article_data) > 0 else None,
            'author': article_data['author'].iloc[0] if len(article_data) > 0 and 'author' in article_data.columns else 'Unknown',
            'media': article_data['media'].iloc[0] if len(article_data) > 0 and 'media' in article_data.columns else 'Unknown'
        }
    
    def _detect_frame_sequences(self, data: pd.DataFrame) -> None:
        """Detect common frame sequences across articles."""
        # Group by media and time window to find sequences
        data['week'] = pd.to_datetime(data['date_converted']).dt.to_period('W')
        
        sequence_counter = defaultdict(lambda: {'count': 0, 'articles': []})
        
        # Look for sequences within media outlets
        for media in data['media'].unique():
            if pd.isna(media):
                continue
            
            media_data = data[data['media'] == media].sort_values('date_converted')
            
            # Get frame sequence for each week
            for week in media_data['week'].unique():
                week_articles = media_data[media_data['week'] == week]['doc_id'].unique()
                
                # Build sequence from articles in chronological order
                week_sequence = []
                for doc_id in week_articles:
                    if doc_id in self.index['article_frames']:
                        dominant = self.index['article_frames'][doc_id]['dominant_frame']
                        if dominant:
                            week_sequence.append(dominant)
                
                # Record sequences of length 2-4
                for seq_len in range(2, min(5, len(week_sequence) + 1)):
                    for i in range(len(week_sequence) - seq_len + 1):
                        seq = tuple(week_sequence[i:i+seq_len])
                        sequence_counter[seq]['count'] += 1
                        sequence_counter[seq]['articles'].extend(
                            week_articles[i:i+seq_len]
                        )
        
        # Store significant sequences
        self.index['frame_sequences'] = []
        for seq, info in sequence_counter.items():
            if info['count'] >= 3:  # Minimum occurrence threshold
                self.index['frame_sequences'].append({
                    'sequence': list(seq),
                    'count': info['count'],
                    'articles': list(set(info['articles']))[:10]  # Limit stored articles
                })
        
        # Sort by frequency
        self.index['frame_sequences'].sort(key=lambda x: x['count'], reverse=True)
    
    def _calculate_frame_statistics(self) -> None:
        """Calculate overall frame statistics."""
        stats = {}
        
        for frame in self.frames:
            # Count articles with this frame
            articles_with_frame = [
                doc_id for doc_id, profile in self.index['article_frames'].items()
                if frame in profile['frames']
            ]
            
            # Calculate statistics
            proportions = [
                profile['frame_proportions'].get(frame, 0)
                for profile in self.index['article_frames'].values()
                if frame in profile['frame_proportions']
            ]
            
            stats[frame] = {
                'n_articles': len(articles_with_frame),
                'prevalence': len(articles_with_frame) / max(1, len(self.index['article_frames'])),
                'mean_proportion': np.mean(proportions) if proportions else 0,
                'std_proportion': np.std(proportions) if proportions else 0,
                'max_proportion': max(proportions) if proportions else 0
            }
        
        self.index['frame_statistics'] = stats
    
    def update_index(self, new_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Update index with new data."""
        logger.info(f"Updating frame index with {len(new_data):,} new rows...")
        
        # Convert dates
        new_data = self._convert_dates(new_data)
        
        # Process new articles
        article_groups = new_data.groupby('doc_id')
        
        for doc_id, article_data in article_groups:
            if doc_id not in self.index['article_frames']:
                profile = self._analyze_article_frames(doc_id, article_data)
                self.index['article_frames'][doc_id] = profile
                
                # Update co-occurrence matrix
                active_frames = profile['frames']
                for i, frame1 in enumerate(active_frames):
                    for frame2 in active_frames[i:]:
                        idx1 = self.frames.index(frame1)
                        idx2 = self.frames.index(frame2)
                        self.index['cooccurrence_matrix'][idx1, idx2] += 1
                        if idx1 != idx2:
                            self.index['cooccurrence_matrix'][idx2, idx1] += 1
        
        # Recalculate statistics
        self._calculate_frame_statistics()
        
        # Update metadata
        self.metadata['updated'] = datetime.now().isoformat()
        self.metadata['n_entries'] = len(self.index['article_frames'])
        
        return self.index
    
    def query_index(self, criteria: Dict[str, Any]) -> List[Any]:
        """
        Query frame index.
        
        Criteria:
            - frames: List of frames that must be present
            - dominant_frame: Specific dominant frame
            - min_frames: Minimum number of active frames
            - date_range: (start, end) tuple
        """
        results = []
        
        required_frames = criteria.get('frames', [])
        dominant_frame = criteria.get('dominant_frame')
        min_frames = criteria.get('min_frames', 0)
        date_range = criteria.get('date_range')
        
        for doc_id, profile in self.index['article_frames'].items():
            # Check required frames
            if required_frames:
                if not all(f in profile['frames'] for f in required_frames):
                    continue
            
            # Check dominant frame
            if dominant_frame and profile['dominant_frame'] != dominant_frame:
                continue
            
            # Check minimum frames
            if profile['n_frames'] < min_frames:
                continue
            
            # Check date range
            if date_range:
                start_date, end_date = date_range
                if not (start_date <= profile['date'] <= end_date):
                    continue
            
            results.append((doc_id, profile))
        
        return results
    
    def get_cooccurrence_matrix(self, normalize: bool = True) -> np.ndarray:
        """
        Get frame co-occurrence matrix.
        
        Args:
            normalize: Whether to normalize by diagonal
            
        Returns:
            Co-occurrence matrix
        """
        matrix = self.index['cooccurrence_matrix'].copy()
        
        if normalize:
            # Normalize by diagonal (self-occurrence)
            diagonal = np.diag(matrix)
            with np.errstate(divide='ignore', invalid='ignore'):
                matrix = matrix / np.sqrt(np.outer(diagonal, diagonal))
                matrix = np.nan_to_num(matrix)
        
        return matrix
    
    def find_multi_frame_patterns(self, 
                                 min_frames: int = 2,
                                 min_support: float = 0.01) -> List[Dict]:
        """
        Find frequent multi-frame patterns.
        
        Args:
            min_frames: Minimum number of frames in pattern
            min_support: Minimum support (proportion of articles)
            
        Returns:
            List of patterns with support
        """
        patterns = defaultdict(int)
        total_articles = len(self.index['article_frames'])
        
        # Count frame combinations
        for profile in self.index['article_frames'].values():
            frames = profile['frames']
            
            if len(frames) >= min_frames:
                # Generate all subsets of size >= min_frames
                from itertools import combinations
                
                for r in range(min_frames, min(len(frames) + 1, 5)):  # Cap at 4
                    for combo in combinations(sorted(frames), r):
                        patterns[combo] += 1
        
        # Filter by minimum support
        min_count = int(min_support * total_articles)
        frequent_patterns = []
        
        for pattern, count in patterns.items():
            if count >= min_count:
                support = count / total_articles
                frequent_patterns.append({
                    'frames': list(pattern),
                    'count': count,
                    'support': support
                })
        
        # Sort by support
        frequent_patterns.sort(key=lambda x: x['support'], reverse=True)
        
        return frequent_patterns
    
    def calculate_frame_transition_matrix(self, 
                                         window_days: int = 7) -> np.ndarray:
        """
        Calculate frame transition probabilities.
        
        Args:
            window_days: Time window for transitions
            
        Returns:
            Transition probability matrix
        """
        n_frames = len(self.frames)
        transitions = np.zeros((n_frames, n_frames))
        
        # Sort articles by date
        articles = sorted(
            self.index['article_frames'].values(),
            key=lambda x: x['date']
        )
        
        # Count transitions within time windows
        for i in range(len(articles) - 1):
            article1 = articles[i]
            
            # Look ahead within window
            j = i + 1
            while j < len(articles):
                article2 = articles[j]
                
                # Check time difference
                time_diff = (article2['date'] - article1['date']).days
                if time_diff > window_days:
                    break
                
                # Count transition from dominant frames
                if article1['dominant_frame'] and article2['dominant_frame']:
                    from_idx = self.frames.index(article1['dominant_frame'])
                    to_idx = self.frames.index(article2['dominant_frame'])
                    transitions[from_idx, to_idx] += 1
                
                j += 1
        
        # Normalize to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            transitions = transitions / row_sums
            transitions = np.nan_to_num(transitions)
        
        return transitions