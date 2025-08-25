"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
emotion_indexer.py

MAIN OBJECTIVE:
---------------
This script indexes emotional tone and sentiment patterns in media coverage, tracking the evolution
of sentiment over time and across different actors for cascade emotional dynamics analysis.

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
1) Sentiment distribution analysis per article and frame
2) Temporal emotion evolution tracking
3) Media outlet emotional profiles
4) Sentiment cascade pattern detection
5) Emotional intensity metrics for cascade amplification

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
from cascade_detector.core.constants import EMOTION_COLUMNS

logger = logging.getLogger(__name__)


class EmotionIndexer(AbstractIndexer):
    """
    Indexes emotional tone and sentiment patterns.
    Tracks evolution of sentiment over time and across actors.
    """
    
    def __init__(self):
        """Initialize emotion indexer."""
        super().__init__(name="EmotionIndexer")
        self.emotion_cols = EMOTION_COLUMNS
        
    def get_required_columns(self) -> List[str]:
        """Get required columns."""
        base_cols = ['date', 'doc_id', 'sentence_id', 'media', 'author']
        # Add emotion columns to required
        emotion_cols = list(self.emotion_cols.keys())
        return base_cols + emotion_cols
    
    def build_index(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Build emotion index from data.
        
        Structure: {
            'article_emotions': {
                doc_id: {
                    'positive_rate': float,
                    'negative_rate': float,
                    'neutral_rate': float,
                    'dominant_emotion': str,
                    'emotional_intensity': float,
                    'sentiment_score': float  # -1 to 1
                }
            },
            'temporal_emotion': {
                date: {
                    'avg_sentiment': float,
                    'emotion_distribution': {pos, neg, neu}
                }
            },
            'media_emotion': {
                media: {
                    'avg_sentiment': float,
                    'emotion_profile': {...}
                }
            },
            'author_emotion': {
                author: {
                    'avg_sentiment': float,
                    'emotion_profile': {...}
                }
            }
        }
        """
        logger.info(f"Building emotion index from {len(data):,} rows...")
        self.validate_data(data)
        
        # Convert dates
        data = self._convert_dates(data)
        
        # Initialize index
        self.index = {
            'article_emotions': {},
            'temporal_emotion': defaultdict(lambda: {
                'sentences': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }),
            'media_emotion': defaultdict(lambda: {
                'sentences': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }),
            'author_emotion': defaultdict(lambda: {
                'sentences': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }),
            'emotion_statistics': {}
        }
        
        # Process each article
        article_groups = data.groupby('doc_id')
        
        for doc_id, article_data in article_groups:
            profile = self._analyze_article_emotion(doc_id, article_data)
            self.index['article_emotions'][doc_id] = profile
            
            # Update temporal index
            date = article_data['date_converted'].iloc[0]
            week = pd.Timestamp(date).to_period('W').to_timestamp()
            self._update_aggregate(self.index['temporal_emotion'][week], article_data)
            
            # Update media index
            media = article_data['media'].iloc[0] if pd.notna(article_data['media'].iloc[0]) else 'Unknown'
            self._update_aggregate(self.index['media_emotion'][media], article_data)
            
            # Update author index
            author = article_data['author'].iloc[0] if pd.notna(article_data['author'].iloc[0]) else 'Unknown'
            self._update_aggregate(self.index['author_emotion'][author], article_data)
        
        # Calculate final statistics
        self._finalize_aggregates()
        self._calculate_emotion_statistics()
        
        # Convert defaultdicts to regular dicts
        self.index['temporal_emotion'] = dict(self.index['temporal_emotion'])
        self.index['media_emotion'] = dict(self.index['media_emotion'])
        self.index['author_emotion'] = dict(self.index['author_emotion'])
        
        # Update metadata
        self.metadata['created'] = datetime.now().isoformat()
        self.metadata['n_entries'] = len(self.index['article_emotions'])
        
        logger.info(f"Emotion index built: {self.metadata['n_entries']} articles indexed")
        
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
    
    def _analyze_article_emotion(self, doc_id: str, article_data: pd.DataFrame) -> Dict:
        """Analyze emotional tone of an article."""
        total_sentences = len(article_data)
        
        # Count emotions
        emotion_counts = {}
        for col, emotion_type in self.emotion_cols.items():
            if col in article_data.columns:
                article_data[col] = pd.to_numeric(article_data[col], errors='coerce').fillna(0)
                count = (article_data[col] == 1).sum()
                emotion_counts[emotion_type] = count
        
        # Calculate rates
        emotion_rates = {
            emotion: count / total_sentences if total_sentences > 0 else 0
            for emotion, count in emotion_counts.items()
        }
        
        # Determine dominant emotion
        if emotion_counts:
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        else:
            dominant_emotion = 'neutral'
        
        # Calculate sentiment score (-1 to 1)
        positive = emotion_rates.get('positive', 0)
        negative = emotion_rates.get('negative', 0)
        sentiment_score = positive - negative  # Simple difference
        
        # Calculate emotional intensity (0 to 1)
        # High intensity = strongly positive or negative
        # Low intensity = mostly neutral
        intensity = abs(sentiment_score)
        
        return {
            'doc_id': doc_id,
            'positive_rate': emotion_rates.get('positive', 0),
            'negative_rate': emotion_rates.get('negative', 0),
            'neutral_rate': emotion_rates.get('neutral', 0),
            'dominant_emotion': dominant_emotion,
            'emotional_intensity': intensity,
            'sentiment_score': sentiment_score,
            'n_sentences': total_sentences,
            'date': article_data['date_converted'].iloc[0] if len(article_data) > 0 else None,
            'author': article_data['author'].iloc[0] if len(article_data) > 0 else 'Unknown',
            'media': article_data['media'].iloc[0] if len(article_data) > 0 else 'Unknown'
        }
    
    def _update_aggregate(self, aggregate: Dict, data: pd.DataFrame) -> None:
        """Update aggregate emotion counts."""
        aggregate['sentences'] += len(data)
        
        for col, emotion_type in self.emotion_cols.items():
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
                count = (data[col] == 1).sum()
                if emotion_type in aggregate:
                    aggregate[emotion_type] += count
    
    def _finalize_aggregates(self) -> None:
        """Calculate final statistics for aggregates."""
        # Process temporal emotions
        for date, counts in self.index['temporal_emotion'].items():
            total = counts['sentences']
            if total > 0:
                counts['positive_rate'] = counts['positive'] / total
                counts['negative_rate'] = counts['negative'] / total
                counts['neutral_rate'] = counts['neutral'] / total
                counts['avg_sentiment'] = counts['positive_rate'] - counts['negative_rate']
        
        # Process media emotions
        for media, counts in self.index['media_emotion'].items():
            total = counts['sentences']
            if total > 0:
                counts['positive_rate'] = counts['positive'] / total
                counts['negative_rate'] = counts['negative'] / total
                counts['neutral_rate'] = counts['neutral'] / total
                counts['avg_sentiment'] = counts['positive_rate'] - counts['negative_rate']
        
        # Process author emotions
        for author, counts in self.index['author_emotion'].items():
            total = counts['sentences']
            if total > 0:
                counts['positive_rate'] = counts['positive'] / total
                counts['negative_rate'] = counts['negative'] / total
                counts['neutral_rate'] = counts['neutral'] / total
                counts['avg_sentiment'] = counts['positive_rate'] - counts['negative_rate']
    
    def _calculate_emotion_statistics(self) -> None:
        """Calculate overall emotion statistics."""
        all_sentiments = [a['sentiment_score'] for a in self.index['article_emotions'].values()]
        all_intensities = [a['emotional_intensity'] for a in self.index['article_emotions'].values()]
        
        if all_sentiments:
            self.index['emotion_statistics'] = {
                'mean_sentiment': np.mean(all_sentiments),
                'std_sentiment': np.std(all_sentiments),
                'min_sentiment': min(all_sentiments),
                'max_sentiment': max(all_sentiments),
                'mean_intensity': np.mean(all_intensities),
                'std_intensity': np.std(all_intensities),
                'n_positive_articles': sum(1 for s in all_sentiments if s > 0.1),
                'n_negative_articles': sum(1 for s in all_sentiments if s < -0.1),
                'n_neutral_articles': sum(1 for s in all_sentiments if -0.1 <= s <= 0.1)
            }
        else:
            self.index['emotion_statistics'] = {}
    
    def update_index(self, new_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Update index with new data."""
        logger.info(f"Updating emotion index with {len(new_data):,} new rows...")
        
        new_data = self._convert_dates(new_data)
        
        # Process new articles
        article_groups = new_data.groupby('doc_id')
        
        for doc_id, article_data in article_groups:
            if doc_id not in self.index['article_emotions']:
                profile = self._analyze_article_emotion(doc_id, article_data)
                self.index['article_emotions'][doc_id] = profile
        
        # Recalculate statistics
        self._calculate_emotion_statistics()
        
        # Update metadata
        self.metadata['updated'] = datetime.now().isoformat()
        
        return self.index
    
    def query_index(self, criteria: Dict[str, Any]) -> List[Any]:
        """
        Query emotion index.
        
        Criteria:
            - emotion: 'positive', 'negative', or 'neutral'
            - min_sentiment: Minimum sentiment score
            - max_sentiment: Maximum sentiment score
            - author: Specific author
            - media: Specific media
            - date_range: (start, end) tuple
        """
        results = []
        
        emotion = criteria.get('emotion')
        min_sentiment = criteria.get('min_sentiment', -1.0)
        max_sentiment = criteria.get('max_sentiment', 1.0)
        author = criteria.get('author')
        media = criteria.get('media')
        date_range = criteria.get('date_range')
        
        for doc_id, profile in self.index['article_emotions'].items():
            # Filter by emotion
            if emotion and profile['dominant_emotion'] != emotion:
                continue
            
            # Filter by sentiment range
            if not (min_sentiment <= profile['sentiment_score'] <= max_sentiment):
                continue
            
            # Filter by author
            if author and profile['author'] != author:
                continue
            
            # Filter by media
            if media and profile['media'] != media:
                continue
            
            # Filter by date range
            if date_range:
                start_date, end_date = date_range
                if not (start_date <= profile['date'] <= end_date):
                    continue
            
            results.append((doc_id, profile))
        
        return results
    
    def get_sentiment_timeline(self, 
                              frequency: str = 'daily',
                              media: str = None,
                              author: str = None) -> pd.Series:
        """
        Get sentiment timeline.
        
        Args:
            frequency: 'daily' or 'weekly'
            media: Filter by specific media
            author: Filter by specific author
            
        Returns:
            Time series of sentiment scores
        """
        # Filter articles
        articles = self.index['article_emotions'].values()
        
        if media:
            articles = [a for a in articles if a['media'] == media]
        if author:
            articles = [a for a in articles if a['author'] == author]
        
        if not articles:
            return pd.Series()
        
        # Create time series
        df = pd.DataFrame(articles)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Resample based on frequency
        if frequency == 'daily':
            sentiment_series = df['sentiment_score'].resample('D').mean()
        else:  # weekly
            sentiment_series = df['sentiment_score'].resample('W').mean()
        
        return sentiment_series
    
    def detect_sentiment_shifts(self, 
                               window: int = 7,
                               threshold: float = 0.3) -> List[Dict]:
        """
        Detect significant shifts in sentiment.
        
        Args:
            window: Window size in days
            threshold: Minimum change to be considered a shift
            
        Returns:
            List of sentiment shifts
        """
        series = self.get_sentiment_timeline('daily')
        
        if series.empty:
            return []
        
        # Calculate rolling mean
        rolling_mean = series.rolling(window=window, center=True).mean()
        
        # Calculate changes
        changes = rolling_mean.diff()
        
        # Detect shifts
        shifts = []
        for date, change in changes.items():
            if abs(change) >= threshold:
                shifts.append({
                    'date': date,
                    'change': float(change),
                    'direction': 'positive' if change > 0 else 'negative',
                    'magnitude': abs(change),
                    'before_sentiment': float(rolling_mean[date - pd.Timedelta(days=1)]) if date - pd.Timedelta(days=1) in rolling_mean else 0,
                    'after_sentiment': float(rolling_mean[date]) if date in rolling_mean else 0
                })
        
        return shifts
    
    def calculate_emotional_convergence(self, 
                                       time_windows: List[pd.Timestamp]) -> float:
        """
        Calculate how emotions converge over time.
        
        Returns:
            Convergence score (0 = diverse, 1 = convergent)
        """
        if len(time_windows) < 2:
            return 0.0
        
        # Calculate emotion distribution variance for each window
        variances = []
        
        for window in time_windows:
            if window in self.index['temporal_emotion']:
                emotions = self.index['temporal_emotion'][window]
                distribution = [
                    emotions.get('positive_rate', 0),
                    emotions.get('negative_rate', 0),
                    emotions.get('neutral_rate', 0)
                ]
                variance = np.var(distribution)
                variances.append(variance)
        
        if not variances:
            return 0.0
        
        # Convergence = decrease in variance over time
        initial_variance = variances[0]
        final_variance = variances[-1]
        
        if initial_variance == 0:
            return 1.0
        
        convergence = max(0, (initial_variance - final_variance) / initial_variance)
        
        return convergence