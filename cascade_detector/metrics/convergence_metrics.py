"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
convergence_metrics.py

MAIN OBJECTIVE:
---------------
This script calculates convergence metrics for cascade detection using Phase 1 indices, measuring
narrative alignment, frame convergence, and content homogenization across media outlets.

Dependencies:
-------------
- numpy
- pandas
- typing
- scipy
- collections
- logging
- datetime

MAIN FEATURES:
--------------
1) Frame convergence measurement across outlets
2) Narrative alignment detection using similarity metrics
3) Content homogenization quantification
4) Temporal convergence tracking
5) Multi-dimensional distance calculations (cosine, Jensen-Shannon, Wasserstein)

Author:
-------
Antoine Lemor
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import wasserstein_distance, ks_2samp
try:
    from scipy.spatial.distance import jensen_shannon
except ImportError:
    # Fallback for older scipy versions
    from scipy.spatial.distance import jensenshannon as jensen_shannon
except:
    # Manual implementation if neither is available
    def jensen_shannon(p, q):
        import numpy as np
        from scipy.stats import entropy
        m = 0.5 * (p + q)
        return np.sqrt(0.5 * entropy(p, m) + 0.5 * entropy(q, m))
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ConvergenceMetrics:
    """
    Calculate convergence metrics using Phase 1 indices.
    Enhanced with multiple distance measures and temporal analysis.
    """
    
    def __init__(self, 
                 source_index: Dict[str, Any],
                 entity_index: Dict[str, Any],
                 emotion_index: Dict[str, Any],
                 temporal_index: Optional[Dict[str, Any]] = None,
                 frame_index: Optional[Dict[str, Any]] = None):
        """
        Initialize with Phase 1 indices.
        
        Args:
            source_index: Output from SourceIndexer (for source convergence)
            entity_index: Output from EntityIndexer (for entity convergence)
            emotion_index: Output from EmotionIndexer
            temporal_index: Optional temporal index
            frame_index: Optional frame index
        """
        self.source_index = source_index or {}
        self.entity_index = entity_index or {}
        self.emotion_index = emotion_index or {}
        self.temporal_index = temporal_index or {}
        self.frame_index = frame_index or {}
        self._cache = {}
    
    def calculate_frame_convergence(self, 
                                   time_windows: List[Tuple[str, str]],
                                   method: str = 'cosine',
                                   articles: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Measure how frames converge over time windows.
        Enhanced with multiple similarity measures.
        
        Args:
            time_windows: List of (start_date, end_date) tuples
            method: 'cosine', 'euclidean', 'js', or 'wasserstein'
            articles: Optional list of article IDs to consider
            
        Returns:
            Dict with convergence scores and metadata
        """
        if len(time_windows) < 2:
            return {'convergence_scores': [0.0], 'overall_convergence': 0.0}
        
        # Get frame distributions for each window
        distributions = []
        window_metadata = []
        
        for i, (start, end) in enumerate(time_windows, 1):
            window_articles = self._get_articles_in_window(start, end, articles)
            
            # Debug logging
            print(f"  [DEBUG] Window {i} ({start} to {end}): {len(window_articles)} articles", flush=True)
            
            if not window_articles:
                distributions.append({})
                window_metadata.append({'window': i, 'articles': 0, 'frames': {}})
                continue
            
            # Calculate frame distribution for window
            frame_counts = defaultdict(float)
            article_frames = self.frame_index.get('article_frames', {})
            
            articles_with_frames = 0
            for doc_id in window_articles:
                if doc_id in article_frames:
                    # Try both keys for compatibility
                    frame_dist = article_frames[doc_id].get('frame_distribution', {})
                    if not frame_dist:
                        frame_dist = article_frames[doc_id].get('frame_proportions', {})
                    
                    # Debug: log if we found frame data
                    if frame_dist:
                        articles_with_frames += 1
                        for frame, weight in frame_dist.items():
                            frame_counts[frame] += weight
                    else:
                        # If no frame distribution, use dominant frame with weight 1
                        dominant = article_frames[doc_id].get('dominant_frame')
                        if dominant:
                            articles_with_frames += 1
                            frame_counts[dominant] += 1.0
            
            # Debug: log frame counts
            print(f"    Articles with frames: {articles_with_frames}/{len(window_articles)}", flush=True)
            if frame_counts:
                print(f"    Frame counts: {dict(frame_counts)}", flush=True)
            
            # Normalize to distribution
            total = sum(frame_counts.values())
            if total > 0:
                distribution = {f: c/total for f, c in frame_counts.items()}
                distributions.append(distribution)
                window_metadata.append({
                    'window': i,
                    'articles': len(window_articles),
                    'articles_with_frames': articles_with_frames,
                    'frames': distribution
                })
            else:
                distributions.append({})
                window_metadata.append({'window': i, 'articles': len(window_articles), 'frames': {}})
        
        # Calculate convergence as reduction in distance over time
        convergence_score = self._calculate_distribution_convergence(distributions, method)
        
        # Calculate pairwise convergence scores
        pairwise_scores = []
        valid_dists = [d for d in distributions if d]
        if len(valid_dists) >= 2:
            for i in range(len(valid_dists) - 1):
                distance = self._calculate_distance(valid_dists[i], valid_dists[i+1], method)
                pairwise_scores.append(1 - distance)  # Convert distance to similarity
        
        return {
            'convergence_scores': pairwise_scores if pairwise_scores else [convergence_score],
            'overall_convergence': convergence_score,
            'method': method,
            'n_windows': len(time_windows),
            'window_metadata': window_metadata
        }
    
    def _calculate_distribution_convergence(self, 
                                          distributions: List[Dict],
                                          method: str = 'cosine') -> float:
        """
        Calculate convergence from a sequence of distributions.
        
        Args:
            distributions: List of distribution dictionaries
            method: Distance calculation method
            
        Returns:
            Convergence score
        """
        if len(distributions) < 2:
            return 0.0
        
        # Remove empty distributions
        valid_dists = [d for d in distributions if d]
        if len(valid_dists) < 2:
            return 0.0
        
        # Calculate distances between consecutive distributions
        distances = []
        
        print(f"    Valid distributions: {len(valid_dists)}", flush=True)
        for i in range(len(valid_dists) - 1):
            dist1 = valid_dists[i]
            dist2 = valid_dists[i + 1]
            
            distance = self._calculate_distance(dist1, dist2, method)
            distances.append(distance)
            print(f"    Distance between window {i+1} and {i+2}: {distance:.3f}", flush=True)
        
        if not distances:
            print(f"    No distances calculated!", flush=True)
            return 0.0
        
        # Calculate convergence considering both absolute similarity and trend
        if len(distances) == 1:
            # Only two windows, use absolute similarity
            return max(0, 1 - distances[0])
        else:
            # Multiple windows analysis
            initial_distance = distances[0]
            final_distance = distances[-1]
            avg_distance = np.mean(distances)
            
            # Case 1: Already converged (very low distances throughout)
            if avg_distance < 0.05:  # Threshold for "already similar"
                # High convergence score if maintaining similarity
                base_convergence = 1 - avg_distance  # Higher score for lower average distance
                
                # Adjust slightly based on trend
                if final_distance < initial_distance:
                    # Getting even more similar
                    return min(1.0, base_convergence * 1.1)
                elif final_distance > initial_distance * 1.5:
                    # Diverging significantly
                    return max(0, base_convergence * 0.5)
                else:
                    # Stable similarity
                    return base_convergence
            
            # Case 2: Traditional convergence (distances decreasing)
            if initial_distance == 0:
                return 1.0
            
            # Calculate relative change
            relative_change = (initial_distance - final_distance) / initial_distance
            
            # Consider trend (are distances consistently decreasing?)
            decreasing_steps = sum(1 for i in range(len(distances)-1) 
                                 if distances[i+1] < distances[i])
            trend_factor = decreasing_steps / (len(distances) - 1) if len(distances) > 1 else 0
            
            # Combine factors
            if relative_change > 0:
                # Converging
                final_convergence = 0.7 * relative_change + 0.3 * trend_factor
            else:
                # Diverging or stable - use absolute similarity
                final_convergence = max(0, 1 - avg_distance) * 0.5  # Penalize divergence
            
            return min(1.0, max(0.0, final_convergence))
    
    def _calculate_distance(self, 
                          dist1: Dict[str, float],
                          dist2: Dict[str, float],
                          method: str = 'cosine') -> float:
        """
        Calculate distance between two distributions.
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            method: Distance metric
            
        Returns:
            Distance (0 = identical, 1 = maximally different)
        """
        # Align distributions
        all_keys = set(dist1.keys()) | set(dist2.keys())
        if not all_keys:
            return 1.0
        
        vec1 = np.array([dist1.get(k, 0) for k in all_keys])
        vec2 = np.array([dist2.get(k, 0) for k in all_keys])
        
        # Normalize vectors
        if vec1.sum() > 0:
            vec1 = vec1 / vec1.sum()
        if vec2.sum() > 0:
            vec2 = vec2 / vec2.sum()
        
        # Calculate distance based on method
        if method == 'cosine':
            # Cosine distance
            if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
                return 1.0
            return cosine(vec1, vec2)
        
        elif method == 'euclidean':
            # Euclidean distance (normalized)
            return euclidean(vec1, vec2) / np.sqrt(2)  # Max euclidean distance is sqrt(2)
        
        elif method == 'js':
            # Jensen-Shannon divergence
            # Ensure no zeros for JS divergence
            vec1 = np.maximum(vec1, 1e-10)
            vec2 = np.maximum(vec2, 1e-10)
            return jensen_shannon(vec1, vec2)
        
        elif method == 'wasserstein':
            # Earth Mover's Distance
            return wasserstein_distance(vec1, vec2)
        
        else:
            # Default to cosine
            return cosine(vec1, vec2) if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0 else 1.0
    
    def calculate_emotional_convergence(self,
                                       time_windows: List[Tuple[str, str]],
                                       aspect: str = 'sentiment') -> float:
        """
        Measure emotional convergence over time.
        Enhanced with multiple emotional aspects.
        
        Args:
            time_windows: List of time window tuples
            aspect: 'sentiment', 'intensity', or 'distribution'
            
        Returns:
            Emotional convergence score
        """
        if len(time_windows) < 2:
            return 0.0
        
        if aspect == 'sentiment':
            return self._calculate_sentiment_convergence(time_windows)
        elif aspect == 'intensity':
            return self._calculate_intensity_convergence(time_windows)
        else:  # distribution
            return self._calculate_emotion_distribution_convergence(time_windows)
    
    def _calculate_sentiment_convergence(self, time_windows: List[Tuple[str, str]]) -> float:
        """
        Calculate convergence of sentiment scores.
        """
        window_sentiments = []
        
        for start, end in time_windows:
            window_articles = self._get_articles_in_window(start, end)
            
            if not window_articles:
                continue
            
            # Get sentiment scores for window
            sentiments = []
            article_emotions = self.emotion_index.get('article_emotions', {})
            
            for doc_id in window_articles:
                if doc_id in article_emotions:
                    sentiment = article_emotions[doc_id].get('sentiment_score', 0)
                    sentiments.append(sentiment)
            
            if sentiments:
                # Calculate window statistics
                window_sentiments.append({
                    'mean': np.mean(sentiments),
                    'std': np.std(sentiments),
                    'values': sentiments
                })
        
        if len(window_sentiments) < 2:
            return 0.0
        
        # Calculate convergence as reduction in variance
        variances = [w['std'] for w in window_sentiments]
        
        if variances[0] == 0:
            return 1.0 if all(v == 0 for v in variances) else 0.0
        
        # Convergence = reduction in variance
        variance_reduction = (variances[0] - variances[-1]) / variances[0]
        
        # Also check if means are converging
        means = [w['mean'] for w in window_sentiments]
        mean_variance = np.var(means)
        mean_convergence = 1 - min(1, mean_variance)  # Less variance in means = more convergence
        
        # Combine both factors
        return 0.6 * max(0, variance_reduction) + 0.4 * mean_convergence
    
    def _calculate_intensity_convergence(self, time_windows: List[Tuple[str, str]]) -> float:
        """
        Calculate convergence of emotional intensity.
        """
        window_intensities = []
        
        for start, end in time_windows:
            window_articles = self._get_articles_in_window(start, end)
            
            if not window_articles:
                continue
            
            intensities = []
            article_emotions = self.emotion_index.get('article_emotions', {})
            
            for doc_id in window_articles:
                if doc_id in article_emotions:
                    intensity = article_emotions[doc_id].get('emotional_intensity', 0)
                    intensities.append(intensity)
            
            if intensities:
                window_intensities.append({
                    'mean': np.mean(intensities),
                    'std': np.std(intensities)
                })
        
        if len(window_intensities) < 2:
            return 0.0
        
        # Convergence = reduction in intensity variance
        stds = [w['std'] for w in window_intensities]
        
        if stds[0] == 0:
            return 1.0
        
        return max(0, (stds[0] - stds[-1]) / stds[0])
    
    def _calculate_emotion_distribution_convergence(self, time_windows: List[Tuple[str, str]]) -> float:
        """
        Calculate convergence of emotion distributions.
        """
        distributions = []
        
        for start, end in time_windows:
            window_articles = self._get_articles_in_window(start, end)
            
            if not window_articles:
                distributions.append({})
                continue
            
            # Get emotion distribution for window
            emotion_counts = defaultdict(int)
            article_emotions = self.emotion_index.get('article_emotions', {})
            
            for doc_id in window_articles:
                if doc_id in article_emotions:
                    dominant = article_emotions[doc_id].get('dominant_emotion', 'neutral')
                    emotion_counts[dominant] += 1
            
            # Normalize to distribution
            total = sum(emotion_counts.values())
            if total > 0:
                distribution = {e: c/total for e, c in emotion_counts.items()}
                distributions.append(distribution)
            else:
                distributions.append({})
        
        # Calculate convergence of distributions
        return self._calculate_distribution_convergence(distributions, 'js')
    
    def calculate_narrative_convergence(self,
                                       article_groups: List[List[str]],
                                       method: str = 'jaccard') -> float:
        """
        Measure narrative convergence using entity overlap.
        Enhanced with multiple similarity measures.
        
        Args:
            article_groups: Chronological groups of article IDs
            method: 'jaccard', 'cosine', or 'overlap'
            
        Returns:
            Narrative convergence score
        """
        if len(article_groups) < 2:
            return 0.0
        
        # Extract entity sets for each group
        entity_distributions = []
        
        for group in article_groups:
            entities = defaultdict(float)
            
            # Collect entities from articles
            article_frames = self.frame_index.get('article_frames', {})
            article_profiles = self.frame_index.get('article_profiles', {})
            
            for doc_id in group:
                # Try frame index first
                if doc_id in article_frames:
                    article_entities = article_frames[doc_id].get('entities', [])
                    for entity in article_entities:
                        entities[entity] += 1
                
                # Also check article profiles
                elif doc_id in article_profiles:
                    article_entities = article_profiles[doc_id].get('entities', [])
                    for entity in article_entities:
                        entities[entity] += 1
            
            entity_distributions.append(entities)
        
        # Calculate convergence based on method
        if method == 'jaccard':
            return self._calculate_jaccard_convergence(entity_distributions)
        elif method == 'cosine':
            return self._calculate_cosine_convergence(entity_distributions)
        else:  # overlap
            return self._calculate_overlap_convergence(entity_distributions)
    
    def _calculate_jaccard_convergence(self, distributions: List[Dict]) -> float:
        """
        Calculate convergence using Jaccard similarity.
        """
        if len(distributions) < 2:
            return 0.0
        
        similarities = []
        
        for i in range(len(distributions) - 1):
            set1 = set(distributions[i].keys())
            set2 = set(distributions[i + 1].keys())
            
            if not set1 and not set2:
                similarity = 1.0
            elif not set1 or not set2:
                similarity = 0.0
            else:
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                similarity = intersection / union if union > 0 else 0
            
            similarities.append(similarity)
        
        # Convergence = increasing similarity
        if len(similarities) == 1:
            return similarities[0]
        
        # Check if similarities are increasing
        increasing_steps = sum(1 for i in range(len(similarities)-1) 
                             if similarities[i+1] > similarities[i])
        trend = increasing_steps / (len(similarities) - 1) if len(similarities) > 1 else 0
        
        # Combine average similarity with trend
        avg_similarity = np.mean(similarities)
        return 0.6 * avg_similarity + 0.4 * trend
    
    def _calculate_cosine_convergence(self, distributions: List[Dict]) -> float:
        """
        Calculate convergence using cosine similarity.
        """
        if len(distributions) < 2:
            return 0.0
        
        # Convert to vectors
        all_keys = set()
        for dist in distributions:
            all_keys.update(dist.keys())
        
        if not all_keys:
            return 0.0
        
        vectors = []
        for dist in distributions:
            vec = np.array([dist.get(k, 0) for k in all_keys])
            if vec.sum() > 0:
                vec = vec / vec.sum()
            vectors.append(vec)
        
        # Calculate similarities
        similarities = []
        for i in range(len(vectors) - 1):
            if np.linalg.norm(vectors[i]) > 0 and np.linalg.norm(vectors[i+1]) > 0:
                similarity = 1 - cosine(vectors[i], vectors[i+1])
            else:
                similarity = 0
            similarities.append(similarity)
        
        # Return average similarity
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_overlap_convergence(self, distributions: List[Dict]) -> float:
        """
        Calculate convergence using overlap coefficient.
        """
        if len(distributions) < 2:
            return 0.0
        
        overlaps = []
        
        for i in range(len(distributions) - 1):
            set1 = set(distributions[i].keys())
            set2 = set(distributions[i + 1].keys())
            
            if not set1 or not set2:
                overlap = 0.0
            else:
                intersection = len(set1 & set2)
                min_size = min(len(set1), len(set2))
                overlap = intersection / min_size if min_size > 0 else 0
            
            overlaps.append(overlap)
        
        return np.mean(overlaps) if overlaps else 0.0
    
    def calculate_earth_movers_distance(self,
                                       dist1: Dict[str, float],
                                       dist2: Dict[str, float]) -> float:
        """
        Calculate Earth Mover's Distance between two distributions.
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            
        Returns:
            EMD value (lower = more similar)
        """
        # Align distributions
        all_keys = sorted(set(dist1.keys()) | set(dist2.keys()))
        
        if not all_keys:
            return 0.0
        
        vec1 = np.array([dist1.get(k, 0) for k in all_keys])
        vec2 = np.array([dist2.get(k, 0) for k in all_keys])
        
        # Normalize
        if vec1.sum() > 0:
            vec1 = vec1 / vec1.sum()
        if vec2.sum() > 0:
            vec2 = vec2 / vec2.sum()
        
        # Calculate EMD
        return wasserstein_distance(vec1, vec2)
    
    def calculate_temporal_alignment(self,
                                    frames: List[str],
                                    window: int = 7) -> float:
        """
        Calculate temporal alignment between multiple frames.
        Measures if frames peak/decline together.
        
        Args:
            frames: List of frame names
            window: Window size for correlation
            
        Returns:
            Alignment score (0 = no alignment, 1 = perfect alignment)
        """
        if len(frames) < 2:
            return 1.0
        
        # Get time series for each frame
        series_list = []
        for frame in frames:
            if frame in self.temporal_index:
                series = self.temporal_index[frame].get('daily_series', pd.Series())
                if not series.empty:
                    series_list.append(series)
        
        if len(series_list) < 2:
            return 0.0
        
        # Align series to common dates
        aligned = pd.DataFrame({f"frame_{i}": s for i, s in enumerate(series_list)})
        aligned = aligned.fillna(0)
        
        # Calculate rolling correlations
        correlations = []
        
        for i in range(len(series_list)):
            for j in range(i + 1, len(series_list)):
                # Calculate correlation over rolling window
                rolling_corr = aligned.iloc[:, i].rolling(window).corr(aligned.iloc[:, j])
                
                # Get mean correlation (ignoring NaN)
                mean_corr = rolling_corr.dropna().mean()
                if not np.isnan(mean_corr):
                    correlations.append(mean_corr)
        
        if not correlations:
            return 0.0
        
        # Return mean correlation as alignment score
        alignment = np.mean(correlations)
        
        # Convert from [-1, 1] to [0, 1]
        return (alignment + 1) / 2
    
    def detect_convergence_points(self,
                                 time_series: pd.Series,
                                 threshold: float = 0.7) -> List[Dict]:
        """
        Detect specific points where convergence occurs.
        
        Args:
            time_series: Series to analyze
            threshold: Convergence threshold
            
        Returns:
            List of convergence points with metadata
        """
        convergence_points = []
        
        # Calculate rolling variance
        rolling_var = time_series.rolling(window=7).var()
        
        # Find points where variance drops significantly
        var_change = rolling_var.diff()
        
        # Detect significant drops in variance (convergence)
        for date, change in var_change.items():
            if pd.notna(change) and change < -threshold * rolling_var.mean():
                convergence_points.append({
                    'date': date,
                    'variance_drop': float(abs(change)),
                    'value': float(time_series[date]),
                    'rolling_variance': float(rolling_var[date]) if pd.notna(rolling_var[date]) else 0
                })
        
        return convergence_points
    
    def _get_articles_in_window(self, 
                               start: str, 
                               end: str,
                               article_filter: Optional[List[str]] = None) -> List[str]:
        """
        Get article IDs within time window.
        
        Args:
            start: Start date
            end: End date
            article_filter: Optional list of articles to consider
            
        Returns:
            List of article IDs
        """
        articles = set()
        
        # Convert dates
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
        
        # Method 1: Try to get from temporal index
        for frame_data in self.temporal_index.values():
            if 'articles_by_date' in frame_data:
                for date, article_list in frame_data['articles_by_date'].items():
                    if start_date <= pd.Timestamp(date) <= end_date:
                        if article_filter:
                            articles.update([a for a in article_list if a in article_filter])
                        else:
                            articles.update(article_list)
        
        # Method 2: If no articles found, use frame_index directly
        if not articles and self.frame_index:
            article_frames = self.frame_index.get('article_frames', {})
            for doc_id, frame_data in article_frames.items():
                # Check if article has a date in the window
                article_date = frame_data.get('date')
                if article_date:
                    try:
                        article_timestamp = pd.Timestamp(article_date)
                        if start_date <= article_timestamp <= end_date:
                            if article_filter:
                                if doc_id in article_filter:
                                    articles.add(doc_id)
                            else:
                                articles.add(doc_id)
                    except:
                        # If date parsing fails, skip this article
                        continue
        
        return list(articles)
    
    def calculate_convergence_velocity(self,
                                      distributions: List[Dict],
                                      time_gaps: List[int]) -> float:
        """
        Calculate the rate of convergence over time.
        
        Args:
            distributions: Sequence of distributions
            time_gaps: Days between distributions
            
        Returns:
            Convergence velocity (rate of change)
        """
        if len(distributions) < 2 or len(time_gaps) < len(distributions) - 1:
            return 0.0
        
        # Calculate distances between consecutive distributions
        distances = []
        for i in range(len(distributions) - 1):
            dist = self._calculate_distance(distributions[i], distributions[i+1], 'cosine')
            distances.append(dist)
        
        # Calculate velocity (change per day)
        velocities = []
        for i, (dist, gap) in enumerate(zip(distances, time_gaps)):
            if gap > 0:
                velocity = dist / gap  # Distance change per day
                velocities.append(velocity)
        
        if not velocities:
            return 0.0
        
        # Return average velocity (negative = converging)
        avg_velocity = np.mean(velocities)
        
        # Convert to positive convergence velocity
        return max(0, -avg_velocity)
    
    def calculate_source_convergence(self,
                                    sources: List[str],
                                    time_window: Optional[Tuple[str, str]] = None) -> float:
        """
        Calculate convergence of sources (media outlets or journalists).
        
        Args:
            sources: List of source identifiers (media names or journalist names)
            time_window: Optional time window for analysis
            
        Returns:
            Convergence score (0 = no convergence, 1 = full convergence)
        """
        if len(sources) < 2:
            return 0.0
        
        # Get article profiles from source index
        article_profiles = self.source_index.get('article_profiles', {})
        media_profiles = self.source_index.get('media_profiles', {})
        journalist_profiles = self.source_index.get('journalist_profiles', {})
        
        # Collect frame distributions for each source
        source_distributions = []
        
        for source in sources:
            frame_dist = defaultdict(float)
            article_count = 0
            
            # Check if source is a media outlet
            if source in media_profiles:
                media_articles = media_profiles[source].get('articles', [])
                for doc_id in media_articles:
                    if doc_id in article_profiles:
                        article = article_profiles[doc_id]
                        # Apply time window filter if provided
                        if time_window:
                            article_date = article.get('date')
                            if article_date:
                                if not (pd.Timestamp(time_window[0]) <= article_date <= pd.Timestamp(time_window[1])):
                                    continue
                        
                        # Aggregate frame distribution
                        frames = article.get('frames', {})
                        for frame, weight in frames.items():
                            frame_dist[frame] += weight
                        article_count += 1
            
            # Check if source is a journalist
            elif source in journalist_profiles:
                journalist_articles = journalist_profiles[source].get('articles', [])
                for doc_id in journalist_articles:
                    if doc_id in article_profiles:
                        article = article_profiles[doc_id]
                        # Apply time window filter if provided
                        if time_window:
                            article_date = article.get('date')
                            if article_date:
                                if not (pd.Timestamp(time_window[0]) <= article_date <= pd.Timestamp(time_window[1])):
                                    continue
                        
                        # Aggregate frame distribution
                        frames = article.get('frames', {})
                        for frame, weight in frames.items():
                            frame_dist[frame] += weight
                        article_count += 1
            
            # Normalize distribution
            if frame_dist and article_count > 0:
                total = sum(frame_dist.values())
                if total > 0:
                    normalized_dist = {k: v/total for k, v in frame_dist.items()}
                    source_distributions.append(normalized_dist)
                else:
                    source_distributions.append({})
            else:
                source_distributions.append({})
        
        # Calculate convergence between source distributions
        if len(source_distributions) < 2:
            return 0.0
        
        # Remove empty distributions
        valid_dists = [d for d in source_distributions if d]
        if len(valid_dists) < 2:
            return 0.0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(valid_dists)):
            for j in range(i + 1, len(valid_dists)):
                distance = self._calculate_distance(valid_dists[i], valid_dists[j], 'cosine')
                similarity = 1 - distance
                similarities.append(similarity)
        
        # Return average similarity as convergence score
        if similarities:
            return float(np.mean(similarities))
        return 0.0
    
    def calculate_entity_convergence(self,
                                    entities: List[str],
                                    method: str = 'co_occurrence') -> float:
        """
        Calculate convergence of entity usage patterns.
        
        Args:
            entities: List of entity identifiers
            method: 'co_occurrence' or 'context_similarity'
            
        Returns:
            Convergence score (0 = no convergence, 1 = full convergence)
        """
        if len(entities) < 2:
            return 0.0
        
        if not self.entity_index:
            return 0.0
        
        if method == 'co_occurrence':
            return self._calculate_entity_co_occurrence_convergence(entities)
        else:  # context_similarity
            return self._calculate_entity_context_convergence(entities)
    
    def _calculate_entity_co_occurrence_convergence(self, entities: List[str]) -> float:
        """
        Calculate convergence based on entity co-occurrence patterns.
        
        Args:
            entities: List of entity identifiers
            
        Returns:
            Convergence score based on co-occurrence
        """
        # Build co-occurrence matrix
        co_occurrence_counts = defaultdict(lambda: defaultdict(int))
        
        for entity in entities:
            if entity in self.entity_index:
                entity_data = self.entity_index[entity]
                co_mentions = entity_data.get('co_mentions', {})
                
                for co_entity, count in co_mentions.items():
                    if co_entity in entities:  # Only consider entities in our list
                        co_occurrence_counts[entity][co_entity] += count
        
        if not co_occurrence_counts:
            return 0.0
        
        # Calculate convergence as the density of co-occurrences
        total_possible_pairs = len(entities) * (len(entities) - 1) / 2
        actual_pairs = 0
        total_strength = 0
        
        for entity1 in entities:
            for entity2 in entities:
                if entity1 < entity2:  # Avoid double counting
                    strength = co_occurrence_counts.get(entity1, {}).get(entity2, 0)
                    strength += co_occurrence_counts.get(entity2, {}).get(entity1, 0)
                    if strength > 0:
                        actual_pairs += 1
                        total_strength += strength
        
        if total_possible_pairs > 0:
            density = actual_pairs / total_possible_pairs
            # Weight by strength of co-occurrences
            if actual_pairs > 0:
                avg_strength = total_strength / actual_pairs
                # Normalize strength (assuming max reasonable co-occurrence is 100)
                normalized_strength = min(1.0, avg_strength / 100)
                # Combine density and strength
                convergence = 0.7 * density + 0.3 * normalized_strength
            else:
                convergence = 0.0
        else:
            convergence = 0.0
        
        return min(1.0, convergence)
    
    def _calculate_entity_context_convergence(self, entities: List[str]) -> float:
        """
        Calculate convergence based on entity context similarity.
        
        Args:
            entities: List of entity identifiers
            
        Returns:
            Convergence score based on context similarity
        """
        # Collect context vectors for each entity
        context_vectors = []
        
        for entity in entities:
            if entity in self.entity_index:
                entity_data = self.entity_index[entity]
                
                # Build context vector from frames, sentiments, etc.
                context = defaultdict(float)
                
                # Frame context
                frame_dist = entity_data.get('frame_distribution', {})
                for frame, weight in frame_dist.items():
                    context[f'frame_{frame}'] = weight
                
                # Media context
                media_dist = entity_data.get('media_distribution', {})
                for media, count in media_dist.items():
                    context[f'media_{media}'] = count
                
                # Temporal context (simplified)
                temporal = entity_data.get('temporal_distribution', {})
                if temporal:
                    context['temporal_spread'] = len(temporal)
                    context['temporal_peak'] = max(temporal.values()) if temporal else 0
                
                context_vectors.append(context)
            else:
                context_vectors.append({})
        
        # Calculate pairwise similarities between context vectors
        if len(context_vectors) < 2:
            return 0.0
        
        valid_vectors = [v for v in context_vectors if v]
        if len(valid_vectors) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(valid_vectors)):
            for j in range(i + 1, len(valid_vectors)):
                distance = self._calculate_distance(valid_vectors[i], valid_vectors[j], 'cosine')
                similarity = 1 - distance
                similarities.append(similarity)
        
        if similarities:
            return float(np.mean(similarities))
        return 0.0