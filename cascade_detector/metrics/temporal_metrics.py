"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
temporal_metrics.py

MAIN OBJECTIVE:
---------------
This script calculates temporal cascade metrics using Phase 1 indices, providing velocity,
acceleration, momentum, and burst detection measures for cascade temporal dynamics analysis.

Dependencies:
-------------
- pandas
- numpy
- typing
- scipy
- datetime
- logging

MAIN FEATURES:
--------------
1) Propagation velocity calculation (first derivative)
2) Acceleration computation (second derivative)
3) Momentum and energy metrics
4) Burst detection with statistical thresholds
5) Periodicity and seasonality analysis

Author:
-------
Antoine Lemor
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import signal, stats
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TemporalMetrics:
    """
    Calculate temporal cascade metrics using Phase 1 indices.
    Enhanced with additional metrics for better cascade detection.
    """
    
    def __init__(self, temporal_index: Dict[str, Any]):
        """
        Initialize with temporal index from Phase 1.
        
        Args:
            temporal_index: Output from TemporalIndexer.build_index()
                           Contains daily_series, weekly_series, articles_by_date, statistics
        """
        self.temporal_index = temporal_index
        self._cache = {}  # Cache computed metrics
        
    def calculate_velocity(self, 
                          frame: str, 
                          window: int = 7,
                          smoothing: bool = True) -> pd.Series:
        """
        Calculate propagation velocity (first derivative).
        Enhanced with smoothing option for noise reduction.
        
        Args:
            frame: Frame name
            window: Window size for derivative calculation
            smoothing: Apply Savitzky-Golay filter for smoothing
            
        Returns:
            Velocity series (articles per day change rate)
        """
        if frame not in self.temporal_index:
            logger.warning(f"Frame {frame} not found in temporal index")
            return pd.Series()
        
        cache_key = f"velocity_{frame}_{window}_{smoothing}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        series = self.temporal_index[frame].get('daily_series', pd.Series())
        
        if series.empty:
            return pd.Series()
        
        # Apply smoothing if requested
        if smoothing and len(series) > window:
            # Savitzky-Golay filter for smoothing
            smoothed_values = signal.savgol_filter(
                series.values, 
                window_length=min(window, len(series)) if len(series) % 2 == 0 else min(window, len(series)-1),
                polyorder=min(3, window-1)
            )
            series = pd.Series(smoothed_values, index=series.index)
        
        # Calculate velocity (rate of change)
        velocity = series.diff(window) / window
        
        # Fill NaN values with 0
        velocity = velocity.fillna(0)
        
        self._cache[cache_key] = velocity
        return velocity
    
    def calculate_acceleration(self, 
                              frame: str, 
                              window: int = 3) -> pd.Series:
        """
        Calculate acceleration (second derivative).
        Positive = expanding cascade, Negative = declining cascade.
        
        Args:
            frame: Frame name
            window: Window size for acceleration calculation
            
        Returns:
            Acceleration series
        """
        cache_key = f"acceleration_{frame}_{window}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get velocity first
        velocity = self.calculate_velocity(frame, window)
        
        if velocity.empty:
            return pd.Series()
        
        # Calculate acceleration (rate of change of velocity)
        acceleration = velocity.diff(window) / window
        acceleration = acceleration.fillna(0)
        
        self._cache[cache_key] = acceleration
        return acceleration
    
    def detect_bursts(self, 
                     frame: str, 
                     method: str = 'adaptive',
                     s: float = 2.0,
                     gamma: float = 1.0,
                     min_burst_length: int = 3) -> List[Dict]:
        """
        Enhanced burst detection using multiple methods.
        
        Args:
            frame: Frame to analyze
            method: 'kleinberg', 'adaptive', or 'zscore'
            s: Scaling parameter for Kleinberg (higher = fewer bursts)
            gamma: Cost of state transition for Kleinberg
            min_burst_length: Minimum days for a valid burst
            
        Returns:
            List of burst periods with intensity scores
        """
        if frame not in self.temporal_index:
            return []
        
        series = self.temporal_index[frame].get('daily_series', pd.Series())
        if series.empty or len(series) < min_burst_length:
            return []
        
        if method == 'kleinberg':
            return self._detect_kleinberg_bursts(series, s, gamma, min_burst_length)
        elif method == 'adaptive':
            return self._detect_adaptive_bursts(series, min_burst_length)
        elif method == 'zscore':
            return self._detect_zscore_bursts(series, min_burst_length)
        else:
            # Combine all methods for robust detection
            kleinberg = self._detect_kleinberg_bursts(series, s, gamma, min_burst_length)
            adaptive = self._detect_adaptive_bursts(series, min_burst_length)
            zscore = self._detect_zscore_bursts(series, min_burst_length)
            
            # Merge and deduplicate bursts
            all_bursts = kleinberg + adaptive + zscore
            return self._merge_overlapping_bursts(all_bursts)
    
    def _detect_kleinberg_bursts(self, series: pd.Series, s: float, gamma: float, min_length: int) -> List[Dict]:
        """
        Kleinberg burst detection algorithm implementation.
        """
        bursts = []
        
        # Calculate base rate
        base_rate = series.mean()
        std_rate = series.std()
        
        if base_rate == 0:
            return []
        
        # Dynamic threshold based on standard deviation
        threshold = base_rate + s * std_rate
        
        # Find periods above threshold
        above_threshold = series > threshold
        
        # Find consecutive periods
        in_burst = False
        burst_start = None
        
        for i, (date, value) in enumerate(series.items()):
            if value > threshold and not in_burst:
                in_burst = True
                burst_start = date
            elif (value <= threshold or i == len(series) - 1) and in_burst:
                in_burst = False
                burst_end = date if value <= threshold else series.index[-1]
                
                # Check minimum length
                burst_length = (burst_end - burst_start).days + 1
                if burst_length >= min_length:
                    burst_data = series[burst_start:burst_end]
                    
                    # Calculate burst metrics
                    intensity = burst_data.max() / base_rate if base_rate > 0 else 1
                    total_excess = (burst_data - base_rate).sum()
                    
                    bursts.append({
                        'start': burst_start,
                        'end': burst_end,
                        'intensity': float(intensity),
                        'peak_date': burst_data.idxmax(),
                        'peak_value': float(burst_data.max()),
                        'total_volume': float(burst_data.sum()),
                        'excess_volume': float(total_excess),
                        'duration': burst_length,
                        'method': 'kleinberg'
                    })
        
        return bursts
    
    def _detect_adaptive_bursts(self, series: pd.Series, min_length: int) -> List[Dict]:
        """
        Adaptive burst detection using local statistics.
        """
        bursts = []
        
        # Use rolling window for adaptive thresholds
        window = min(14, len(series) // 4)
        rolling_mean = series.rolling(window=window, center=True).mean()
        rolling_std = series.rolling(window=window, center=True).std()
        
        # Fill NaN values
        rolling_mean = rolling_mean.fillna(method='bfill').fillna(method='ffill')
        rolling_std = rolling_std.fillna(method='bfill').fillna(method='ffill')
        
        # Adaptive threshold: local mean + 2 * local std
        threshold = rolling_mean + 2 * rolling_std
        
        # Find bursts
        above_threshold = series > threshold
        
        in_burst = False
        burst_start = None
        
        for i, (date, value) in enumerate(series.items()):
            if above_threshold.iloc[i] and not in_burst:
                in_burst = True
                burst_start = date
            elif (not above_threshold.iloc[i] or i == len(series) - 1) and in_burst:
                in_burst = False
                burst_end = date if not above_threshold.iloc[i] else series.index[-1]
                
                burst_length = (burst_end - burst_start).days + 1
                if burst_length >= min_length:
                    burst_data = series[burst_start:burst_end]
                    local_baseline = rolling_mean[burst_start:burst_end].mean()
                    
                    intensity = burst_data.max() / local_baseline if local_baseline > 0 else 1
                    
                    bursts.append({
                        'start': burst_start,
                        'end': burst_end,
                        'intensity': float(intensity),
                        'peak_date': burst_data.idxmax(),
                        'peak_value': float(burst_data.max()),
                        'total_volume': float(burst_data.sum()),
                        'duration': burst_length,
                        'method': 'adaptive'
                    })
        
        return bursts
    
    def _detect_zscore_bursts(self, series: pd.Series, min_length: int) -> List[Dict]:
        """
        Z-score based burst detection.
        """
        bursts = []
        
        # Calculate z-scores
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return []
        
        z_scores = (series - mean) / std
        
        # Find periods with z-score > 2
        high_z = z_scores > 2
        
        in_burst = False
        burst_start = None
        
        for i, (date, is_high) in enumerate(high_z.items()):
            if is_high and not in_burst:
                in_burst = True
                burst_start = date
            elif (not is_high or i == len(high_z) - 1) and in_burst:
                in_burst = False
                burst_end = date if not is_high else series.index[-1]
                
                burst_length = (burst_end - burst_start).days + 1
                if burst_length >= min_length:
                    burst_data = series[burst_start:burst_end]
                    
                    intensity = burst_data.max() / mean if mean > 0 else 1
                    
                    bursts.append({
                        'start': burst_start,
                        'end': burst_end,
                        'intensity': float(intensity),
                        'peak_date': burst_data.idxmax(),
                        'peak_value': float(burst_data.max()),
                        'total_volume': float(burst_data.sum()),
                        'duration': burst_length,
                        'method': 'zscore'
                    })
        
        return bursts
    
    def _merge_overlapping_bursts(self, bursts: List[Dict]) -> List[Dict]:
        """
        Merge overlapping burst periods from different detection methods.
        """
        if not bursts:
            return []
        
        # Sort by start date
        bursts.sort(key=lambda x: x['start'])
        
        merged = []
        current = bursts[0].copy()
        
        for burst in bursts[1:]:
            # Check for overlap
            if burst['start'] <= current['end']:
                # Merge bursts
                current['end'] = max(current['end'], burst['end'])
                current['intensity'] = max(current['intensity'], burst['intensity'])
                current['peak_value'] = max(current['peak_value'], burst['peak_value'])
                current['total_volume'] += burst['total_volume']
                if burst['peak_value'] > current['peak_value']:
                    current['peak_date'] = burst['peak_date']
                current['duration'] = (current['end'] - current['start']).days + 1
                # Keep track of methods used
                if 'methods' not in current:
                    current['methods'] = [current.get('method', 'unknown')]
                current['methods'].append(burst.get('method', 'unknown'))
            else:
                merged.append(current)
                current = burst.copy()
        
        merged.append(current)
        return merged
    
    def calculate_persistence(self, 
                            frame: str, 
                            threshold_pct: float = 0.1) -> int:
        """
        Duration that attention stays above threshold.
        Enhanced to consider both absolute and relative thresholds.
        
        Args:
            frame: Frame name
            threshold_pct: Percentage above mean to consider active
            
        Returns:
            Maximum consecutive days above threshold
        """
        if frame not in self.temporal_index:
            return 0
        
        series = self.temporal_index[frame].get('daily_series', pd.Series())
        stats = self.temporal_index[frame].get('statistics', {})
        
        if series.empty or not stats:
            return 0
        
        # Dynamic threshold based on mean and standard deviation
        mean = stats.get('mean', 0)
        std = stats.get('std', 0)
        
        # Use both relative and absolute thresholds
        relative_threshold = mean * (1 + threshold_pct)
        absolute_threshold = mean + std  # One standard deviation above mean
        
        # Use the more lenient threshold
        threshold = min(relative_threshold, absolute_threshold)
        
        above_threshold = series > threshold
        
        # Find longest consecutive True sequence
        max_persistence = 0
        current_persistence = 0
        
        for is_above in above_threshold:
            if is_above:
                current_persistence += 1
                max_persistence = max(max_persistence, current_persistence)
            else:
                current_persistence = 0
        
        return max_persistence
    
    def calculate_volatility(self, frame: str, normalized: bool = True) -> float:
        """
        Calculate attention volatility.
        Enhanced with normalization option.
        
        Args:
            frame: Frame name
            normalized: Whether to normalize by mean (coefficient of variation)
            
        Returns:
            Volatility score
        """
        if frame not in self.temporal_index:
            return 0.0
        
        stats = self.temporal_index[frame].get('statistics', {})
        
        if not stats:
            return 0.0
        
        # FIX: Use correct keys from statistics
        std = stats.get('std_daily_count', 0)
        mean = stats.get('mean_daily_count', 0)
        
        if normalized and mean > 0:
            # Coefficient of variation
            return std / mean
        else:
            return std
    
    def calculate_momentum(self, frame: str, window: int = 7) -> pd.Series:
        """
        Calculate momentum indicator (rate of change).
        New metric not in original design.
        
        Args:
            frame: Frame name
            window: Look-back period
            
        Returns:
            Momentum series
        """
        if frame not in self.temporal_index:
            return pd.Series()
        
        series = self.temporal_index[frame].get('daily_series', pd.Series())
        
        if series.empty or len(series) <= window:
            return pd.Series()
        
        # Rate of change: (current - past) / past * 100
        momentum = ((series - series.shift(window)) / series.shift(window)) * 100
        momentum = momentum.fillna(0)
        
        return momentum
    
    def calculate_trend_strength(self, frame: str, window: int = 14) -> float:
        """
        Calculate trend strength using linear regression R-squared.
        New metric for measuring cascade direction consistency.
        
        Args:
            frame: Frame name
            window: Period to analyze
            
        Returns:
            R-squared value (0-1, higher = stronger trend)
        """
        if frame not in self.temporal_index:
            return 0.0
        
        series = self.temporal_index[frame].get('daily_series', pd.Series())
        
        if series.empty or len(series) < window:
            return 0.0
        
        # Take last 'window' days
        recent_series = series.tail(window)
        
        # Prepare data for regression
        x = np.arange(len(recent_series))
        y = recent_series.values
        
        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Return R-squared
        return r_value ** 2
    
    def find_cascade_periods(self, 
                           frame: str,
                           min_duration: int = 3,
                           min_intensity: float = 2.0,
                           include_context: bool = True) -> List[Dict]:
        """
        Enhanced cascade period detection with context.
        
        Args:
            frame: Frame to analyze
            min_duration: Minimum cascade duration in days
            min_intensity: Minimum intensity threshold
            include_context: Include pre/post cascade context
            
        Returns:
            List of cascade periods with detailed metrics
        """
        # Detect bursts using multiple methods
        bursts = self.detect_bursts(frame, method='all', min_burst_length=min_duration)
        
        cascades = []
        
        for burst in bursts:
            if burst['intensity'] >= min_intensity:
                cascade = {
                    'frame': frame,
                    'start': burst['start'],
                    'end': burst['end'],
                    'duration': burst['duration'],
                    'intensity': burst['intensity'],
                    'peak_date': burst['peak_date'],
                    'peak_value': burst['peak_value'],
                    'total_volume': burst['total_volume']
                }
                
                # Add velocity and acceleration at peak
                velocity = self.calculate_velocity(frame)
                acceleration = self.calculate_acceleration(frame)
                
                if not velocity.empty and burst['peak_date'] in velocity.index:
                    cascade['peak_velocity'] = float(velocity[burst['peak_date']])
                
                if not acceleration.empty and burst['peak_date'] in acceleration.index:
                    cascade['peak_acceleration'] = float(acceleration[burst['peak_date']])
                
                # Calculate momentum at start
                momentum = self.calculate_momentum(frame)
                if not momentum.empty and burst['start'] in momentum.index:
                    cascade['initial_momentum'] = float(momentum[burst['start']])
                
                # Add article IDs from temporal index
                if 'articles_by_date' in self.temporal_index[frame]:
                    articles = []
                    articles_by_date = self.temporal_index[frame]['articles_by_date']
                    
                    current = pd.Timestamp(burst['start'])
                    end = pd.Timestamp(burst['end'])
                    
                    while current <= end:
                        if current in articles_by_date:
                            articles.extend(articles_by_date[current])
                        current += timedelta(days=1)
                    
                    cascade['articles'] = list(set(articles))
                    cascade['n_articles'] = len(cascade['articles'])
                
                # Add context if requested
                if include_context:
                    cascade['context'] = self._get_cascade_context(frame, burst)
                
                # Detection confidence based on number of methods that found it
                if 'methods' in burst:
                    cascade['detection_confidence'] = len(set(burst['methods'])) / 3.0
                
                cascades.append(cascade)
        
        # Sort by intensity
        cascades.sort(key=lambda x: x['intensity'], reverse=True)
        
        return cascades
    
    def _get_cascade_context(self, frame: str, burst: Dict) -> Dict:
        """
        Get context around cascade period.
        """
        context = {
            'pre_cascade': {},
            'post_cascade': {}
        }
        
        series = self.temporal_index[frame].get('daily_series', pd.Series())
        
        if series.empty:
            return context
        
        # Pre-cascade context (7 days before)
        pre_start = burst['start'] - timedelta(days=7)
        pre_end = burst['start'] - timedelta(days=1)
        
        pre_data = series[pre_start:pre_end]
        if not pre_data.empty:
            context['pre_cascade'] = {
                'mean': float(pre_data.mean()),
                'trend': float(self._calculate_trend(pre_data)),
                'volatility': float(pre_data.std())
            }
        
        # Post-cascade context (7 days after)
        post_start = burst['end'] + timedelta(days=1)
        post_end = burst['end'] + timedelta(days=7)
        
        post_data = series[post_start:post_end]
        if not post_data.empty:
            context['post_cascade'] = {
                'mean': float(post_data.mean()),
                'trend': float(self._calculate_trend(post_data)),
                'volatility': float(post_data.std()),
                'decay_rate': float((post_data.iloc[-1] - post_data.iloc[0]) / len(post_data)) if len(post_data) > 1 else 0
            }
        
        return context
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """
        Calculate trend slope for a series.
        """
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        y = series.values
        
        try:
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope
        except:
            return 0.0
    
    def calculate_cascade_synchrony(self, frames: List[str], window: int = 7) -> float:
        """
        Calculate synchronization between multiple frames.
        New metric for multi-frame cascade detection.
        
        Args:
            frames: List of frame names to compare
            window: Window for correlation calculation
            
        Returns:
            Synchrony score (0-1, higher = more synchronized)
        """
        if len(frames) < 2:
            return 0.0
        
        # Get all series
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
        
        # Calculate rolling correlation
        correlations = []
        for i in range(len(series_list)):
            for j in range(i + 1, len(series_list)):
                corr = aligned.iloc[:, i].rolling(window).corr(aligned.iloc[:, j])
                correlations.append(corr.mean())
        
        # Return mean correlation as synchrony score
        return np.mean([c for c in correlations if not np.isnan(c)])
    
    def get_summary_statistics(self, frame: str) -> Dict:
        """
        Get comprehensive temporal statistics for a frame.
        
        Args:
            frame: Frame name
            
        Returns:
            Dictionary of statistics
        """
        if frame not in self.temporal_index:
            return {}
        
        stats = self.temporal_index[frame].get('statistics', {}).copy()
        series = self.temporal_index[frame].get('daily_series', pd.Series())
        
        if not series.empty:
            # Add additional statistics
            stats['volatility'] = self.calculate_volatility(frame)
            stats['persistence'] = self.calculate_persistence(frame)
            stats['trend_strength'] = self.calculate_trend_strength(frame)
            
            # Peak detection - FIX: Use correct burst detection
            bursts = self.detect_bursts(frame, method='zscore', min_burst_length=1)
            if bursts and len(bursts) > 0:
                stats['n_bursts'] = len(bursts)
                stats['max_intensity'] = max(b['intensity'] for b in bursts)
                stats['total_burst_days'] = sum(b['duration'] for b in bursts)
            else:
                # Also check adaptive method
                bursts_adaptive = self.detect_bursts(frame, method='adaptive', min_burst_length=1)
                if bursts_adaptive and len(bursts_adaptive) > 0:
                    stats['n_bursts'] = len(bursts_adaptive)
                    stats['max_intensity'] = max(b['intensity'] for b in bursts_adaptive)
                    stats['total_burst_days'] = sum(b['duration'] for b in bursts_adaptive)
                else:
                    stats['n_bursts'] = 0
                    stats['max_intensity'] = 0
                    stats['total_burst_days'] = 0
            
            # Time series characteristics
            stats['autocorrelation'] = float(series.autocorr(lag=1)) if len(series) > 1 else 0
            stats['skewness'] = float(series.skew())
            stats['kurtosis'] = float(series.kurtosis())
            
            # FIX: Override mean/std with correct values for backward compatibility
            stats['mean'] = stats.get('mean_daily_count', 0)
            stats['std'] = stats.get('std_daily_count', 0)
        
        return stats