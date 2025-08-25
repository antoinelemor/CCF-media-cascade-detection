"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
base_detector.py

MAIN OBJECTIVE:
---------------
This script provides the abstract base class for all cascade detection algorithms, ensuring
consistent interfaces, data-driven thresholds, exhaustive metric usage, and statistical validation.

Dependencies:
-------------
- abc
- typing
- dataclasses
- datetime
- numpy
- pandas
- logging
- pathlib
- json
- pickle
- collections

MAIN FEATURES:
--------------
1) Abstract interface for detection algorithms
2) Detection context management with all indices
3) Statistical validation framework
4) Performance optimization through caching
5) Data-driven threshold calculation

Author:
-------
Antoine Lemor
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import pickle
from collections import defaultdict

# Import Phase 1-2 components
from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.models import TimeWindow, MediaCascade, CascadeCandidate
from cascade_detector.metrics.temporal_metrics import TemporalMetrics
from cascade_detector.metrics.network_metrics import NetworkMetrics
from cascade_detector.metrics.convergence_metrics import ConvergenceMetrics
from cascade_detector.metrics.diversity_metrics import DiversityMetrics


@dataclass
class DetectionContext:
    """
    Complete detection context containing all Phase 1-2 resources.
    This ensures exhaustive use of all available metrics.
    """
    
    # Phase 1 Indices (all required)
    temporal_index: Dict[str, Any]
    entity_index: Dict[str, Any]
    source_index: Dict[str, Any]
    frame_index: Dict[str, Any]
    emotion_index: Dict[str, Any]
    geographic_index: Optional[Dict[str, Any]] = None
    
    # Phase 2 Metrics instances
    temporal_metrics: Optional[TemporalMetrics] = None
    network_metrics: Optional[NetworkMetrics] = None
    convergence_metrics: Optional[ConvergenceMetrics] = None
    diversity_metrics: Optional[DiversityMetrics] = None
    
    # Detection parameters
    time_window: Tuple[datetime, datetime] = field(
        default_factory=lambda: (datetime(2019, 1, 1), datetime(2019, 12, 31))
    )
    frames: List[str] = field(
        default_factory=lambda: ['Pol', 'Eco', 'Sci', 'Just', 'Cult', 'Envt', 'Pbh', 'Secu']
    )
    
    # Adaptive thresholds (calculated, not hardcoded)
    thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Metadata for tracking
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """
        Validate that all required resources are available.
        
        Returns:
            True if context is complete and valid
        """
        # Check required indices exist (even if empty)
        required_indices = [
            self.temporal_index,
            self.entity_index,
            self.source_index,
            self.frame_index,
            self.emotion_index
        ]
        
        # Only check that indices are not None (empty dicts are valid)
        if not all(idx is not None for idx in required_indices):
            logging.warning("Some required indices are None")
            return False
        
        # Check at least one metrics instance exists
        metrics = [
            self.temporal_metrics,
            self.network_metrics,
            self.convergence_metrics,
            self.diversity_metrics
        ]
        
        if not any(m is not None for m in metrics):
            logging.warning("No Phase 2 metrics instances available")
        
        return True
    
    def initialize_metrics(self) -> None:
        """Initialize Phase 2 metrics if not already present."""
        if self.temporal_metrics is None and self.temporal_index:
            self.temporal_metrics = TemporalMetrics(self.temporal_index)
        
        if self.diversity_metrics is None:
            self.diversity_metrics = DiversityMetrics(
                frame_index=self.frame_index,
                source_index=self.source_index,
                emotion_index=self.emotion_index,
                entity_index=self.entity_index,
                geographic_index=self.geographic_index
            )
        
        if self.convergence_metrics is None:
            self.convergence_metrics = ConvergenceMetrics(
                source_index=self.source_index,
                entity_index=self.entity_index,
                emotion_index=self.emotion_index,
                temporal_index=self.temporal_index,
                frame_index=self.frame_index
            )


class BaseDetector(ABC):
    """
    Abstract base class for all Phase 3 pattern detectors.
    
    Core Principles:
    1. No hardcoded thresholds - all parameters derived from data
    2. Exhaustive metric usage - leverages ALL Phase 1-2 outputs
    3. Statistical validation - every detection is validated
    4. Performance optimization - intelligent caching
    5. Scientific rigor - reproducible and transparent
    """
    
    def __init__(self, 
                 context: DetectionContext,
                 config: Optional[DetectorConfig] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize base detector with complete context.
        
        Args:
            context: Complete detection context with all resources
            config: Optional configuration override
            cache_dir: Optional directory for caching computations
        """
        # Validate context
        if not context.validate():
            raise ValueError("Detection context is incomplete or invalid")
        
        self.context = context
        self.config = config or DetectorConfig()
        
        # Initialize metrics if needed
        self.context.initialize_metrics()
        
        # Setup caching
        self._cache = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validation state
        self._validated = False
        
        # Performance tracking
        self._performance_stats = {
            'detections': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'computation_time': 0.0
        }
        
        # Logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Calculate baseline statistics on initialization
        self._baseline_stats = self.calculate_baseline_statistics()
    
    @abstractmethod
    def detect(self, **kwargs) -> List[Any]:
        """
        Main detection method to be implemented by subclasses.
        
        Returns:
            List of detected patterns (cascades, bursts, etc.)
        """
        pass
    
    @abstractmethod
    def validate_detection(self, detection: Any) -> bool:
        """
        Validate a detection using statistical tests.
        
        Args:
            detection: Object to validate
            
        Returns:
            True if detection passes validation
        """
        pass
    
    @abstractmethod
    def score_detection(self, detection: Any) -> float:
        """
        Score a detection for ranking/filtering.
        
        Args:
            detection: Object to score
            
        Returns:
            Score in [0, 1] range
        """
        pass
    
    def get_adaptive_threshold(self,
                              metric_name: str,
                              data: Union[np.ndarray, pd.Series],
                              method: str = 'percentile',
                              **kwargs) -> float:
        """
        Calculate data-driven adaptive threshold.
        
        Methods:
        - percentile: Based on empirical percentile (default: 95th)
        - mad: Median Absolute Deviation (robust to outliers)
        - iqr: Interquartile Range method
        - zscore: Standard deviation based (default: 2 sigma)
        - kde: Kernel Density Estimation for multimodal distributions
        
        Args:
            metric_name: Name for caching
            data: Data to calculate threshold from
            method: Threshold calculation method
            **kwargs: Method-specific parameters
            
        Returns:
            Calculated threshold value
        """
        # Check cache
        cache_key = f"threshold_{metric_name}_{method}_{hash(str(kwargs))}"
        if cache_key in self._cache:
            self._performance_stats['cache_hits'] += 1
            return self._cache[cache_key]
        
        self._performance_stats['cache_misses'] += 1
        
        # Convert to numpy array
        if isinstance(data, pd.Series):
            data = data.values
        data = np.array(data)
        
        # Remove NaN and infinite values
        data = data[np.isfinite(data)]
        
        if len(data) < 10:
            self.logger.warning(f"Insufficient data for threshold {metric_name}: {len(data)} points")
            # Fallback to simple statistics
            if len(data) > 0:
                threshold = np.mean(data) + np.std(data)
            else:
                threshold = 0.0
        else:
            if method == 'percentile':
                percentile = kwargs.get('percentile', 95)
                threshold = np.percentile(data, percentile)
                
            elif method == 'mad':
                # Median Absolute Deviation
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                n_mad = kwargs.get('n_mad', 3)
                threshold = median + n_mad * mad * 1.4826  # Scale factor for normal distribution
                
            elif method == 'iqr':
                # Interquartile Range
                q1, q3 = np.percentile(data, [25, 75])
                iqr = q3 - q1
                factor = kwargs.get('factor', 1.5)
                threshold = q3 + factor * iqr
                
            elif method == 'zscore':
                # Z-score based
                mean = np.mean(data)
                std = np.std(data)
                n_sigma = kwargs.get('n_sigma', 2)
                threshold = mean + n_sigma * std
                
            elif method == 'kde':
                # Kernel Density Estimation for complex distributions
                from scipy.stats import gaussian_kde
                
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 1000)
                density = kde(x_range)
                
                # Find valleys in multimodal distribution
                from scipy.signal import find_peaks
                valleys = find_peaks(-density)[0]
                
                if len(valleys) > 0:
                    # Use rightmost valley as threshold
                    threshold = x_range[valleys[-1]]
                else:
                    # Fallback to percentile
                    threshold = np.percentile(data, kwargs.get('percentile', 95))
                    
            else:
                raise ValueError(f"Unknown threshold method: {method}")
        
        # Cache result
        self._cache[cache_key] = threshold
        
        self.logger.debug(f"Adaptive threshold for {metric_name}: {threshold:.4f} (method={method})")
        
        return threshold
    
    def calculate_baseline_statistics(self, 
                                     baseline_period: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Calculate baseline statistics for adaptive thresholding.
        Uses the first 20% of the time window as baseline period by default.
        
        Args:
            baseline_period: Optional specific baseline period
            
        Returns:
            Dictionary of baseline statistics
        """
        stats = {}
        
        # Determine baseline period
        if baseline_period is None:
            start, end = self.context.time_window
            duration = (end - start).days
            baseline_days = max(30, int(duration * 0.2))  # At least 30 days
            baseline_end = start + timedelta(days=baseline_days)
            baseline_period = (start, baseline_end)
        
        stats['baseline_period'] = baseline_period
        
        # Calculate statistics for each frame
        for frame in self.context.frames:
            if frame not in self.context.temporal_index:
                continue
                
            frame_data = self.context.temporal_index[frame]
            daily_series = frame_data.get('daily_series', pd.Series())
            
            if daily_series.empty:
                continue
            
            # Filter to baseline period
            baseline_data = daily_series[baseline_period[0]:baseline_period[1]]
            
            if not baseline_data.empty:
                stats[f"{frame}_baseline"] = {
                    'mean': float(baseline_data.mean()),
                    'std': float(baseline_data.std()),
                    'median': float(baseline_data.median()),
                    'mad': float(np.median(np.abs(baseline_data - baseline_data.median()))),
                    'q25': float(baseline_data.quantile(0.25)),
                    'q75': float(baseline_data.quantile(0.75)),
                    'max': float(baseline_data.max()),
                    'min': float(baseline_data.min())
                }
        
        # Entity statistics
        if self.context.entity_index:
            entity_counts = []
            for entity_key, entity_data in self.context.entity_index.items():
                if entity_data.get('temporal_distribution'):
                    # Count occurrences in baseline period
                    baseline_count = sum(
                        count for date, count in entity_data['temporal_distribution'].items()
                        if baseline_period[0] <= pd.Timestamp(date) <= baseline_period[1]
                    )
                    entity_counts.append(baseline_count)
            
            if entity_counts:
                stats['entity_baseline'] = {
                    'mean_occurrences': float(np.mean(entity_counts)),
                    'std_occurrences': float(np.std(entity_counts)),
                    'max_occurrences': float(np.max(entity_counts)),
                    'active_entities': len([c for c in entity_counts if c > 0])
                }
        
        # Source diversity baseline
        if self.context.source_index:
            article_profiles = self.context.source_index.get('article_profiles', {})
            baseline_articles = [
                profile for doc_id, profile in article_profiles.items()
                if profile.get('date') and baseline_period[0] <= profile['date'] <= baseline_period[1]
            ]
            
            if baseline_articles:
                media_counts = defaultdict(int)
                journalist_counts = defaultdict(int)
                
                for article in baseline_articles:
                    media_counts[article.get('media', 'Unknown')] += 1
                    journalist_counts[article.get('author', 'Unknown')] += 1
                
                stats['source_baseline'] = {
                    'n_media': len(media_counts),
                    'n_journalists': len(journalist_counts),
                    'articles_per_day': len(baseline_articles) / baseline_days,
                    'media_concentration': self._calculate_hhi(list(media_counts.values())),
                    'journalist_concentration': self._calculate_hhi(list(journalist_counts.values()))
                }
        
        self.logger.info(f"Calculated baseline statistics for period {baseline_period[0]} to {baseline_period[1]}")
        
        return stats
    
    def _calculate_hhi(self, counts: List[int]) -> float:
        """
        Calculate Herfindahl-Hirschman Index for concentration.
        
        Args:
            counts: List of counts/frequencies
            
        Returns:
            HHI value (0-10000 scale)
        """
        if not counts:
            return 0.0
        
        total = sum(counts)
        if total == 0:
            return 0.0
        
        shares = [c / total for c in counts]
        hhi = sum(s ** 2 for s in shares) * 10000
        
        return hhi
    
    def get_articles_in_window(self,
                              window: Tuple[datetime, datetime],
                              frame: Optional[str] = None,
                              min_frame_weight: float = 0.1) -> List[str]:
        """
        Get articles within a time window, optionally filtered by frame.
        
        Args:
            window: (start, end) datetime tuple
            frame: Optional frame to filter by
            min_frame_weight: Minimum frame weight to include article
            
        Returns:
            List of article IDs (doc_ids)
        """
        articles = set()
        start, end = window
        
        if frame and frame in self.context.temporal_index:
            # Use temporal index for efficient lookup
            frame_data = self.context.temporal_index[frame]
            articles_by_date = frame_data.get('articles_by_date', {})
            
            # Iterate through dates in window
            current = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            
            while current <= end_ts:
                if current in articles_by_date:
                    articles.update(articles_by_date[current])
                current += pd.Timedelta(days=1)
        else:
            # Use source index for general lookup
            article_profiles = self.context.source_index.get('article_profiles', {})
            
            for doc_id, profile in article_profiles.items():
                article_date = profile.get('date')
                if article_date and start <= article_date <= end:
                    if frame:
                        # Check frame weight
                        frame_weights = profile.get('frames', {})
                        if frame_weights.get(frame, 0) >= min_frame_weight:
                            articles.add(doc_id)
                    else:
                        articles.add(doc_id)
        
        return list(articles)
    
    def calculate_composite_score(self,
                                 features: Dict[str, float],
                                 weights: Optional[Dict[str, float]] = None,
                                 normalization: str = 'minmax') -> float:
        """
        Calculate weighted composite score from multiple features.
        
        Args:
            features: Dictionary of feature scores
            weights: Optional weights (default: equal)
            normalization: 'minmax', 'zscore', or 'none'
            
        Returns:
            Composite score in [0, 1] range
        """
        if not features:
            return 0.0
        
        # Default equal weights
        if weights is None:
            weights = {k: 1.0 for k in features.keys()}
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate weighted sum
        score = 0.0
        for feature, value in features.items():
            if feature in weights:
                # Ensure value is in [0, 1]
                value = max(0.0, min(1.0, value))
                score += value * weights[feature]
        
        return score
    
    def save_cache(self) -> None:
        """Save cache to disk if cache_dir is set."""
        if self.cache_dir and self._cache:
            cache_file = self.cache_dir / f"{self.__class__.__name__}_cache.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
            self.logger.info(f"Saved cache to {cache_file}")
    
    def load_cache(self) -> None:
        """Load cache from disk if available."""
        if self.cache_dir:
            cache_file = self.cache_dir / f"{self.__class__.__name__}_cache.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                self.logger.info(f"Loaded cache from {cache_file}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self._performance_stats.copy()
        
        # Calculate cache hit rate
        total_cache_ops = stats['cache_hits'] + stats['cache_misses']
        if total_cache_ops > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_cache_ops
        else:
            stats['cache_hit_rate'] = 0.0
        
        return stats
    
    def reset_cache(self) -> None:
        """Clear all cached computations."""
        self._cache.clear()
        self.logger.info("Cache cleared")