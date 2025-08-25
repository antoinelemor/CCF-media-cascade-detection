"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
burst_detector.py

MAIN OBJECTIVE:
---------------
This script implements advanced burst detection algorithms for identifying temporal anomalies 
in media coverage that indicate cascade formation, using multiple complementary statistical methods
for robust detection.

Dependencies:
-------------
- typing
- dataclasses
- datetime
- numpy
- pandas
- logging
- scipy
- pywt (optional)

MAIN FEATURES:
--------------
1) Multi-method burst detection (Kleinberg, z-score, wavelet, change point)
2) Adaptive threshold calculation based on local baselines
3) Burst shape classification (spike, plateau, gradual, oscillating)
4) Cross-media validation and synchronization detection
5) Statistical significance testing and confidence scoring

Author:
-------
Antoine Lemor
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging
from collections import defaultdict
import warnings
from scipy import signal, stats

logger = logging.getLogger(__name__)

# Try to import wavelet library (optional)
try:
    import pywt  # For wavelet analysis
    HAS_WAVELET = True
except ImportError:
    HAS_WAVELET = False
    logger.warning("pywt not installed. Wavelet-based burst detection will be disabled.")

from cascade_detector.detectors.base_detector import BaseDetector, DetectionContext
from cascade_detector.detectors.signal_aggregator import SignalAggregator, AggregatedSignal
from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.models import CascadeCandidate, TimeWindow


@dataclass
class BurstEvent:
    """
    Represents a detected burst event with comprehensive metadata.
    """
    # Core attributes
    frame: str
    start_date: datetime
    end_date: datetime
    peak_date: datetime
    
    # Burst characteristics
    intensity: float  # Peak intensity relative to baseline
    volume: float  # Total volume during burst
    acceleration: float  # Rate of growth
    shape: str  # 'spike', 'plateau', 'gradual', 'oscillating'
    
    # Detection metadata
    detection_methods: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    # Multi-scale features
    scales: Dict[int, float] = field(default_factory=dict)  # Window size -> intensity
    wavelet_coefficients: Optional[np.ndarray] = None
    
    # Trigger analysis
    trigger_date: Optional[datetime] = None
    trigger_confidence: float = 0.0
    trigger_type: Optional[str] = None  # 'event', 'media', 'viral', 'coordinated'
    
    # Context
    preceding_calm_days: int = 0
    following_decay_days: int = 0
    is_cascade_trigger: bool = False
    
    # Validation
    statistical_significance: float = 0.0
    false_positive_risk: float = 0.0
    
    @property
    def duration_days(self) -> int:
        """Get burst duration in days."""
        return (self.end_date - self.start_date).days + 1
    
    @property
    def is_significant(self) -> bool:
        """Check if burst is statistically significant."""
        return (self.confidence > 0.7 and 
                self.statistical_significance > 0.95 and
                self.false_positive_risk < 0.05)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        return {
            'frame': self.frame,
            'period': {
                'start': self.start_date.isoformat(),
                'end': self.end_date.isoformat(),
                'peak': self.peak_date.isoformat(),
                'duration_days': self.duration_days
            },
            'characteristics': {
                'intensity': self.intensity,
                'volume': self.volume,
                'acceleration': self.acceleration,
                'shape': self.shape
            },
            'detection': {
                'methods': self.detection_methods,
                'confidence': self.confidence,
                'significance': self.statistical_significance,
                'false_positive_risk': self.false_positive_risk
            },
            'trigger': {
                'date': self.trigger_date.isoformat() if self.trigger_date else None,
                'confidence': self.trigger_confidence,
                'type': self.trigger_type
            },
            'context': {
                'preceding_calm': self.preceding_calm_days,
                'following_decay': self.following_decay_days,
                'is_cascade_trigger': self.is_cascade_trigger
            }
        }


class BurstDetector(BaseDetector):
    """
    Advanced burst detection for media cascades.
    
    Implements multiple complementary burst detection methods:
    1. Kleinberg burst detection (already in TemporalMetrics)
    2. Wavelet-based burst detection (new)
    3. Multi-resolution temporal analysis
    4. Burst shape characterization
    5. Cascade trigger identification
    """
    
    def __init__(self,
                 context: DetectionContext,
                 config: Optional[DetectorConfig] = None,
                 signal_aggregator: Optional[SignalAggregator] = None):
        """
        Initialize burst detector.
        
        Args:
            context: Detection context with all indices and metrics
            config: Optional configuration
            signal_aggregator: Optional pre-initialized SignalAggregator
        """
        super().__init__(context, config)
        
        # Initialize or use provided SignalAggregator
        # Only create a new one if explicitly provided or if needed
        self.signal_aggregator = signal_aggregator
        if signal_aggregator is False:  # Explicitly disabled
            self.signal_aggregator = None
        elif signal_aggregator is None:  # Not provided, create default
            from cascade_detector.detectors.signal_aggregator import SignalAggregator
            self.signal_aggregator = SignalAggregator(context, config)
        
        # Configuration for SENSITIVE burst detection
        self.min_burst_duration = 1  # Minimum days for valid burst
        self.wavelet_scales = [2, 4, 8, 16, 32]  # Multi-resolution scales
        self.wavelet_type = 'morl'  # Morlet wavelet (continuous wavelet for cwt)
        
        # Adaptive threshold parameters (MORE SENSITIVE)
        self.threshold_methods = {
            'percentile': {'percentile': 90},  # Lowered from 95
            'zscore': {'n_sigma': 1.5},  # Lowered from 2.0
            'mad': {'n_mad': 2.5},  # Lowered from 3.0
            'iqr': {'factor': 1.0}  # Lowered from 1.5
        }
        
        # Burst detection sensitivity
        if config and hasattr(config, 'config') and isinstance(config.config, dict):
            self.sensitivity = config.config.get('burst_sensitivity', 'high')
        else:
            self.sensitivity = 'high'
        self._adjust_sensitivity()
        
        # Cache for detection results
        self._burst_cache = {}
        
        logger.info(f"BurstDetector initialized with {self.sensitivity} sensitivity")
    
    def _adjust_sensitivity(self):
        """Adjust thresholds based on sensitivity level."""
        if self.sensitivity == 'high':
            # Most sensitive settings
            self.threshold_methods = {
                'percentile': {'percentile': 85},
                'zscore': {'n_sigma': 1.25},
                'mad': {'n_mad': 2.0},
                'iqr': {'factor': 0.75}
            }
            self.min_burst_duration = 1
        elif self.sensitivity == 'medium':
            # Balanced settings
            self.threshold_methods = {
                'percentile': {'percentile': 90},
                'zscore': {'n_sigma': 1.5},
                'mad': {'n_mad': 2.5},
                'iqr': {'factor': 1.0}
            }
            self.min_burst_duration = 2
        else:  # low sensitivity
            # Conservative settings
            self.threshold_methods = {
                'percentile': {'percentile': 95},
                'zscore': {'n_sigma': 2.0},
                'mad': {'n_mad': 3.0},
                'iqr': {'factor': 1.5}
            }
            self.min_burst_duration = 3
    
    def detect(self, **kwargs) -> Dict[str, Any]:
        """
        Detect burst events using multiple methods.
        
        Args:
            frame: Optional specific frame to analyze
            window: Optional time window to analyze
            method: 'all', 'kleinberg', 'wavelet', 'multiscale', or 'ensemble'
            signals: Optional pre-computed signals from SignalAggregator
            
        Returns:
            Dictionary containing burst events and metadata
        """
        frame = kwargs.get('frame')
        window = kwargs.get('window', self.context.time_window)
        method = kwargs.get('method', 'ensemble')
        signals = kwargs.get('signals', None)
        
        # If signals are provided, use the optimized detection path
        if signals:
            bursts = self.detect_from_signals(signals, self.context)
            return {
                'bursts': bursts,
                'n_bursts': len(bursts),
                'method': 'from_signals',
                'window': window
            }
        
        # Otherwise, use the original detection methods
        frames = [frame] if frame else self.context.frames
        bursts = []
        
        for f in frames:
            if method == 'kleinberg':
                bursts.extend(self._detect_kleinberg_bursts(f, window))
            elif method == 'wavelet':
                bursts.extend(self._detect_wavelet_bursts(f, window))
            elif method == 'multiscale':
                bursts.extend(self._detect_multiscale_bursts(f, window))
            elif method == 'ensemble':
                bursts.extend(self._detect_ensemble_bursts(f, window))
            else:  # 'all'
                bursts.extend(self._detect_all_bursts(f, window))
        
        # Post-process bursts
        bursts = self._merge_overlapping_bursts(bursts)
        bursts = self._characterize_burst_shapes(bursts)
        bursts = self._identify_triggers(bursts)
        
        # Validate bursts (skip individual validation here as it will be done on the full result)
        validated_bursts = bursts
        
        # Return as dictionary for compatibility with validation
        result = {
            'bursts': validated_bursts,
            'n_bursts': len(validated_bursts),
            'frames_analyzed': frames,
            'time_range': window,
            'detection_method': method
        }
        
        # Add burst statistics
        if validated_bursts:
            result['summary'] = {
                'avg_intensity': np.mean([b.intensity for b in validated_bursts]),
                'max_intensity': max([b.intensity for b in validated_bursts]),
                'avg_duration': np.mean([b.duration_days for b in validated_bursts]),
                'total_burst_days': sum([b.duration_days for b in validated_bursts])
            }
        else:
            result['summary'] = {
                'avg_intensity': 0,
                'max_intensity': 0,
                'avg_duration': 0,
                'total_burst_days': 0
            }
        
        return result
    
    def _detect_adaptive_threshold_bursts(self, frame: str, window: Tuple[datetime, datetime]) -> List[BurstEvent]:
        """
        Detect bursts using adaptive statistical thresholds.
        More sensitive than Kleinberg for gradual increases.
        """
        if frame not in self.context.temporal_index:
            return []
        
        series = self.context.temporal_index[frame].get('daily_series', pd.Series())
        if series.empty:
            return []
        
        # Filter to window
        series_window = series[window[0]:window[1]]
        if len(series_window) < self.min_burst_duration:
            return []
        
        burst_events = []
        values = series_window.values
        dates = series_window.index
        
        # Try multiple threshold methods for robustness
        for method, params in self.threshold_methods.items():
            threshold = self.get_adaptive_threshold(
                f"{frame}_{method}",
                values,
                method=method,
                **params
            )
            
            # Find periods above threshold
            above_threshold = values > threshold
            
            # Group consecutive days above threshold
            in_burst = False
            burst_start = None
            burst_values = []
            
            for i, (date, value, is_burst) in enumerate(zip(dates, values, above_threshold)):
                if is_burst and not in_burst:
                    # Start new burst
                    in_burst = True
                    burst_start = date
                    burst_values = [value]
                elif is_burst and in_burst:
                    # Continue burst
                    burst_values.append(value)
                elif not is_burst and in_burst:
                    # End burst
                    if len(burst_values) >= self.min_burst_duration:
                        peak_idx = np.argmax(burst_values)
                        event = BurstEvent(
                            frame=frame,
                            start_date=pd.Timestamp(burst_start),
                            end_date=pd.Timestamp(dates[i-1]),
                            peak_date=pd.Timestamp(dates[i-len(burst_values)+peak_idx]),
                            intensity=max(burst_values) / np.mean(values),  # Relative intensity
                            volume=sum(burst_values),
                            acceleration=(max(burst_values) - np.mean(values)) / (peak_idx + 1) if peak_idx > 0 else 0,
                            shape='adaptive',
                            detection_methods=[f'adaptive_{method}'],
                            confidence=0.6 + (0.1 * len(burst_values) / 10)  # Higher confidence for longer bursts
                        )
                        burst_events.append(event)
                    in_burst = False
                    burst_values = []
            
            # Handle burst at end of window
            if in_burst and len(burst_values) >= self.min_burst_duration:
                peak_idx = np.argmax(burst_values)
                event = BurstEvent(
                    frame=frame,
                    start_date=pd.Timestamp(burst_start),
                    end_date=pd.Timestamp(dates[-1]),
                    peak_date=pd.Timestamp(dates[-len(burst_values)+peak_idx]),
                    intensity=max(burst_values) / np.mean(values),
                    volume=sum(burst_values),
                    acceleration=(max(burst_values) - np.mean(values)) / (peak_idx + 1) if peak_idx > 0 else 0,
                    shape='adaptive',
                    detection_methods=[f'adaptive_{method}'],
                    confidence=0.6 + (0.1 * len(burst_values) / 10)
                )
                burst_events.append(event)
        
        return burst_events
    
    def _detect_kleinberg_bursts(self, frame: str, window: Tuple[datetime, datetime]) -> List[BurstEvent]:
        """
        Use existing Kleinberg burst detection from TemporalMetrics.
        """
        if not self.context.temporal_metrics:
            return []
        
        # Get bursts from TemporalMetrics
        temporal_bursts = self.context.temporal_metrics.detect_bursts(
            frame, 
            method='kleinberg',
            min_burst_length=self.min_burst_duration
        )
        
        # Convert to BurstEvent objects
        burst_events = []
        for tb in temporal_bursts:
            # Check if burst is within window
            burst_start = pd.Timestamp(tb['start'])
            burst_end = pd.Timestamp(tb['end'])
            
            if not (burst_end < window[0] or burst_start > window[1]):
                event = BurstEvent(
                    frame=frame,
                    start_date=burst_start,
                    end_date=burst_end,
                    peak_date=pd.Timestamp(tb['peak_date']),
                    intensity=tb['intensity'],
                    volume=tb['total_volume'],
                    acceleration=0.0,  # Will be calculated later
                    shape='unknown',  # Will be characterized later
                    detection_methods=['kleinberg'],
                    confidence=0.7  # Base confidence for Kleinberg
                )
                burst_events.append(event)
        
        return burst_events
    
    def _detect_wavelet_bursts(self, frame: str, window: Tuple[datetime, datetime]) -> List[BurstEvent]:
        """
        Wavelet-based burst detection for multi-scale pattern recognition.
        """
        # Check if wavelet library is available
        if not HAS_WAVELET:
            logger.debug("Wavelet detection skipped - pywt not installed")
            return []
        
        # Get time series data
        if frame not in self.context.temporal_index:
            return []
        
        series = self.context.temporal_index[frame].get('daily_series', pd.Series())
        if series.empty:
            return []
        
        # Filter to window
        series_window = series[window[0]:window[1]]
        if len(series_window) < self.min_burst_duration * 2:
            return []
        
        # Pad series to power of 2 for wavelet transform
        n = len(series_window)
        next_pow2 = 2 ** int(np.ceil(np.log2(n)))
        padded = np.pad(series_window.values, (0, next_pow2 - n), mode='constant')
        
        burst_events = []
        
        # Continuous wavelet transform for each scale
        for scale in self.wavelet_scales:
            try:
                # Perform wavelet transform
                # Use scales parameter correctly for cwt
                coeffs, _ = pywt.cwt(padded, [scale], self.wavelet_type)
                coeffs = coeffs[0][:n]
                
                # Normalize coefficients
                coeffs_norm = np.abs(coeffs)
                if coeffs_norm.std() > 0:
                    coeffs_z = (coeffs_norm - coeffs_norm.mean()) / coeffs_norm.std()
                else:
                    continue
                
                # Detect peaks in wavelet coefficients
                # Peaks indicate rapid changes at this scale
                threshold = 1.5 if self.sensitivity == 'high' else 2.0  # Adaptive threshold
                peaks = coeffs_z > threshold
                
                # Find burst periods
                in_burst = False
                burst_start_idx = None
                
                for i, is_peak in enumerate(peaks):
                    if is_peak and not in_burst:
                        in_burst = True
                        burst_start_idx = i
                    elif (not is_peak or i == len(peaks) - 1) and in_burst:
                        in_burst = False
                        burst_end_idx = i
                        
                        # Check duration
                        burst_duration = burst_end_idx - burst_start_idx
                        if burst_duration >= self.min_burst_duration:
                            # Extract burst data
                            burst_data = series_window.iloc[burst_start_idx:burst_end_idx]
                            
                            # Calculate burst intensity at this scale
                            baseline = series_window.mean()
                            intensity = burst_data.max() / baseline if baseline > 0 else 1.0
                            
                            # Create burst event
                            event = BurstEvent(
                                frame=frame,
                                start_date=series_window.index[burst_start_idx],
                                end_date=series_window.index[burst_end_idx - 1],
                                peak_date=burst_data.idxmax(),
                                intensity=intensity,
                                volume=float(burst_data.sum()),
                                acceleration=0.0,  # Will be calculated
                                shape='wavelet',  # Initial shape
                                detection_methods=[f'wavelet_scale_{scale}'],
                                confidence=0.6 + min(0.3, burst_duration / 30),  # Scale-dependent confidence
                                wavelet_coefficients=coeffs[burst_start_idx:burst_end_idx]
                            )
                            
                            # Store scale-specific intensity
                            event.scales[scale] = intensity
                            
                            burst_events.append(event)
            
            except Exception as e:
                logger.warning(f"Wavelet transform failed for scale {scale}: {e}")
                continue
        
        return burst_events
    
    def _detect_multiscale_bursts(self, frame: str, window: Tuple[datetime, datetime]) -> List[BurstEvent]:
        """
        Multi-scale temporal analysis using SignalAggregator.
        """
        # Check if we already have cached signals for this frame/window
        cache_key = f"{frame}_{window[0]}_{window[1]}"
        if hasattr(self, '_signal_cache') and cache_key in self._signal_cache:
            signals_result = self._signal_cache[cache_key]
        else:
            # Get aggregated signals at multiple scales
            signals_result = self.signal_aggregator.detect(frame=frame, window=window)
            # Cache the result
            if not hasattr(self, '_signal_cache'):
                self._signal_cache = {}
            self._signal_cache[cache_key] = signals_result
        
        # Extract signals list from dict result
        if isinstance(signals_result, dict):
            signals = signals_result.get('signals', [])
        else:
            signals = signals_result if isinstance(signals_result, list) else []
        
        burst_events = []
        
        # Group signals by time window
        window_groups = defaultdict(list)
        for signal in signals:
            if hasattr(signal, 'window') and signal.window:
                window_key = (signal.window[0], signal.window[1])
                window_groups[window_key].append(signal)
        
        # Analyze each window for burst patterns
        for (start, end), window_signals in window_groups.items():
            if not window_signals:
                continue
            
            # Aggregate scores across scales
            intensity_scores = []
            burst_scores = []
            
            for signal in window_signals:
                # Check temporal features for burst indicators
                if signal.temporal_features:
                    # High activity relative to mean
                    if 'max_daily_count' in signal.temporal_features and 'mean_daily_count' in signal.temporal_features:
                        mean_count = signal.temporal_features['mean_daily_count']
                        max_count = signal.temporal_features['max_daily_count']
                        if mean_count > 0:
                            intensity = max_count / mean_count
                            intensity_scores.append(intensity)
                
                # Check velocity features for burst
                if signal.velocity_features:
                    if signal.velocity_features.get('has_burst', 0) > 0:
                        burst_scores.append(signal.velocity_features.get('burst_intensity', 0))
            
            # Determine if this window contains a burst
            if intensity_scores and burst_scores:
                avg_intensity = np.mean(intensity_scores)
                avg_burst_score = np.mean(burst_scores)
                
                # Create burst event if thresholds met
                if avg_intensity > 2.0 or avg_burst_score > 3.0:
                    # Get peak date from signals
                    peak_values = []
                    for signal in window_signals:
                        if signal.temporal_features and 'max_daily_count' in signal.temporal_features:
                            peak_values.append(signal.temporal_features['max_daily_count'])
                    
                    event = BurstEvent(
                        frame=frame,
                        start_date=start,
                        end_date=end,
                        peak_date=start + (end - start) / 2,  # Approximate
                        intensity=max(intensity_scores) if intensity_scores else 1.0,
                        volume=sum(s.n_articles for s in window_signals),
                        acceleration=0.0,
                        shape='multiscale',
                        detection_methods=['multiscale_analysis'],
                        confidence=min(0.9, avg_burst_score / 5.0)
                    )
                    
                    # Store multi-scale intensities
                    for signal in window_signals:
                        window_size = (signal.window[1] - signal.window[0]).days
                        if signal.temporal_features and 'max_daily_count' in signal.temporal_features:
                            event.scales[window_size] = signal.temporal_features['max_daily_count']
                    
                    burst_events.append(event)
        
        return burst_events
    
    def _detect_ensemble_bursts(self, frame: str, window: Tuple[datetime, datetime]) -> List[BurstEvent]:
        """
        Ensemble method combining all detection approaches.
        """
        # Get bursts from all methods
        adaptive_bursts = self._detect_adaptive_threshold_bursts(frame, window)  # NEW: Most sensitive
        kleinberg_bursts = self._detect_kleinberg_bursts(frame, window)
        wavelet_bursts = self._detect_wavelet_bursts(frame, window)
        multiscale_bursts = self._detect_multiscale_bursts(frame, window)
        
        # Combine all bursts (adaptive first for sensitivity)
        all_bursts = adaptive_bursts + kleinberg_bursts + wavelet_bursts + multiscale_bursts
        
        # Merge and enhance overlapping bursts
        merged_bursts = self._merge_overlapping_bursts(all_bursts)
        
        # Boost confidence for bursts detected by multiple methods
        for burst in merged_bursts:
            n_methods = len(set(burst.detection_methods))
            if n_methods > 1:
                # Increase confidence based on consensus
                burst.confidence = min(1.0, burst.confidence + 0.1 * (n_methods - 1))
        
        return merged_bursts
    
    def _detect_all_bursts(self, frame: str, window: Tuple[datetime, datetime]) -> List[BurstEvent]:
        """
        Detect bursts using all available methods without merging.
        """
        all_bursts = []
        all_bursts.extend(self._detect_kleinberg_bursts(frame, window))
        all_bursts.extend(self._detect_wavelet_bursts(frame, window))
        all_bursts.extend(self._detect_multiscale_bursts(frame, window))
        return all_bursts
    
    def _merge_overlapping_bursts(self, bursts: List[BurstEvent]) -> List[BurstEvent]:
        """
        Merge overlapping burst events from different detection methods.
        """
        if not bursts:
            return []
        
        # Sort by start date
        bursts.sort(key=lambda x: x.start_date)
        
        merged = []
        current = bursts[0]
        
        for burst in bursts[1:]:
            # Check for overlap (same frame)
            if burst.frame == current.frame and burst.start_date <= current.end_date:
                # Merge bursts
                current.end_date = max(current.end_date, burst.end_date)
                current.intensity = max(current.intensity, burst.intensity)
                current.volume += burst.volume
                
                # Update peak if higher
                if burst.intensity > current.intensity:
                    current.peak_date = burst.peak_date
                
                # Combine detection methods
                current.detection_methods.extend(burst.detection_methods)
                current.detection_methods = list(set(current.detection_methods))
                
                # Update confidence (weighted average)
                total_conf = current.confidence + burst.confidence
                current.confidence = total_conf / 2
                
                # Merge scales
                current.scales.update(burst.scales)
                
                # Merge wavelet coefficients if both have them
                if current.wavelet_coefficients is not None and burst.wavelet_coefficients is not None:
                    # Keep the longer coefficient array
                    if len(burst.wavelet_coefficients) > len(current.wavelet_coefficients):
                        current.wavelet_coefficients = burst.wavelet_coefficients
            else:
                # No overlap, add current and start new
                merged.append(current)
                current = burst
        
        # Add last burst
        merged.append(current)
        
        return merged
    
    def _characterize_burst_shapes(self, bursts: List[BurstEvent]) -> List[BurstEvent]:
        """
        Characterize the shape of each burst (spike, plateau, gradual, oscillating).
        """
        for burst in bursts:
            if burst.frame not in self.context.temporal_index:
                burst.shape = 'unknown'
                continue
            
            series = self.context.temporal_index[burst.frame].get('daily_series', pd.Series())
            if series.empty:
                burst.shape = 'unknown'
                continue
            
            # Get burst period data
            burst_data = series[burst.start_date:burst.end_date]
            if len(burst_data) < 3:
                burst.shape = 'spike'
                continue
            
            # Analyze shape characteristics
            values = burst_data.values
            peak_idx = np.argmax(values)
            peak_value = values[peak_idx]
            
            # Calculate shape metrics
            rise_time = peak_idx
            fall_time = len(values) - peak_idx - 1
            
            # Plateau detection (sustained high values)
            high_threshold = peak_value * 0.8
            high_count = np.sum(values > high_threshold)
            plateau_ratio = high_count / len(values)
            
            # Oscillation detection (multiple peaks)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(values, height=peak_value * 0.5)
            n_peaks = len(peaks)
            
            # Classify shape
            if plateau_ratio > 0.5:
                burst.shape = 'plateau'
            elif n_peaks >= 3:
                burst.shape = 'oscillating'
            elif rise_time <= 2 and fall_time <= 2:
                burst.shape = 'spike'
            else:
                burst.shape = 'gradual'
            
            # Calculate acceleration (rate of rise)
            if rise_time > 0:
                burst.acceleration = (peak_value - values[0]) / rise_time
        
        return bursts
    
    def _identify_triggers(self, bursts: List[BurstEvent]) -> List[BurstEvent]:
        """
        Identify potential trigger events for each burst.
        """
        for burst in bursts:
            # Analyze pre-burst period
            pre_burst_window = (
                burst.start_date - timedelta(days=7),
                burst.start_date
            )
            
            # Get signals before burst
            pre_result = self.signal_aggregator.detect(
                frame=burst.frame,
                window=pre_burst_window
            )
            
            # Extract signals from result dict
            pre_signals = pre_result.get('signals', []) if isinstance(pre_result, dict) else pre_result
            
            if not pre_signals:
                burst.trigger_type = 'spontaneous'
                burst.trigger_confidence = 0.3
                continue
            
            # Analyze patterns to identify trigger type
            trigger_indicators = {
                'event': 0.0,
                'media': 0.0,
                'viral': 0.0,
                'coordinated': 0.0
            }
            
            for signal in pre_signals:
                # Skip if signal is not an AggregatedSignal object
                if not hasattr(signal, 'entity_features'):
                    continue
                    
                # Check for event trigger (sudden entity appearance)
                if hasattr(signal, 'entity_features') and signal.entity_features:
                    new_entity_ratio = signal.entity_features.get('new_entity_ratio', 0)
                    if new_entity_ratio > 0.3:
                        trigger_indicators['event'] += new_entity_ratio
                
                # Check for media trigger (specific media leading)
                if hasattr(signal, 'source_features') and signal.source_features:
                    media_concentration = signal.source_features.get('media_concentration_hhi', 0)
                    if media_concentration > 5000:  # High concentration
                        trigger_indicators['media'] += media_concentration / 10000
                
                # Check for viral trigger (high velocity)
                if hasattr(signal, 'velocity_features') and signal.velocity_features:
                    max_velocity = signal.velocity_features.get('max_velocity', 0)
                    if max_velocity > 10:
                        trigger_indicators['viral'] += min(1.0, max_velocity / 20)
                
                # Check for coordinated trigger (simultaneous multi-source)
                if hasattr(signal, 'diversity_features') and signal.diversity_features:
                    source_diversity = signal.diversity_features.get('media_diversity', 0)
                    if hasattr(signal, 'n_media') and source_diversity > 0.8 and signal.n_media > 5:
                        trigger_indicators['coordinated'] += source_diversity
            
            # Determine primary trigger type
            if trigger_indicators:
                max_indicator = max(trigger_indicators.values())
                if max_indicator > 0.3:
                    burst.trigger_type = max(trigger_indicators, key=trigger_indicators.get)
                    burst.trigger_confidence = min(1.0, max_indicator)
                    burst.trigger_date = burst.start_date - timedelta(days=1)  # Approximate
                    
                    # Check if this could trigger a cascade
                    if burst.trigger_confidence > 0.7 and burst.intensity > 3.0:
                        burst.is_cascade_trigger = True
            
            # Calculate preceding calm period
            if burst.frame in self.context.temporal_index:
                series = self.context.temporal_index[burst.frame].get('daily_series', pd.Series())
                pre_burst_series = series[pre_burst_window[0]:pre_burst_window[1]]
                if not pre_burst_series.empty:
                    mean_activity = pre_burst_series.mean()
                    baseline = series.mean()
                    if mean_activity < baseline * 0.5:
                        burst.preceding_calm_days = len(pre_burst_series)
        
        return bursts
    
    def validate_detection(self, burst: Any) -> bool:
        """
        Validate burst detection result.
        
        Args:
            burst: BurstEvent or detection result dict
            
        Returns:
            True if burst/result is valid and significant
        """
        # Handle dict format (full detection result)
        if isinstance(burst, dict):
            bursts = burst.get('bursts', [])
            # Empty results are valid
            if not bursts:
                return True
            # Validate each burst in the list
            for b in bursts:
                if not isinstance(b, BurstEvent):
                    return False
            return True
        
        # Handle single BurstEvent (legacy)
        if not isinstance(burst, BurstEvent):
            return False
            
        # Basic validation (MORE LENIENT)
        if burst.duration_days < self.min_burst_duration:
            return False
        
        # Adjusted intensity threshold based on sensitivity
        min_intensity = 1.3 if self.sensitivity == 'high' else (1.5 if self.sensitivity == 'medium' else 2.0)
        if burst.intensity < min_intensity:
            return False
        
        # Statistical validation
        if burst.frame in self.context.temporal_index:
            series = self.context.temporal_index[burst.frame].get('daily_series', pd.Series())
            if not series.empty:
                # Get burst and non-burst periods
                burst_data = series[burst.start_date:burst.end_date]
                non_burst_data = series.drop(burst_data.index)
                
                if len(burst_data) > 0 and len(non_burst_data) > 0:
                    # Perform statistical test
                    try:
                        statistic, p_value = stats.mannwhitneyu(
                            burst_data.values,
                            non_burst_data.values,
                            alternative='greater'
                        )
                        
                        burst.statistical_significance = 1 - p_value
                        burst.false_positive_risk = p_value
                        
                        # Adjusted p-value threshold based on sensitivity
                        max_p_value = 0.1 if self.sensitivity == 'high' else (0.05 if self.sensitivity == 'medium' else 0.01)
                        if p_value > max_p_value:
                            return False
                    except:
                        # If test fails, use heuristic
                        burst_mean = burst_data.mean()
                        overall_mean = series.mean()
                        overall_std = series.std()
                        
                        if overall_std > 0:
                            z_score = (burst_mean - overall_mean) / overall_std
                            burst.statistical_significance = min(1.0, z_score / 3)
                            burst.false_positive_risk = max(0.0, 1 - burst.statistical_significance)
                            
                            # Adjusted z-score threshold based on sensitivity
                            min_z_score = 1.0 if self.sensitivity == 'high' else (1.5 if self.sensitivity == 'medium' else 2.0)
                            if z_score < min_z_score:
                                return False
        
        return True
    
    def detect_from_signals(self, signals: List['AggregatedSignal'], 
                          context: Optional['DetectionContext'] = None) -> List[BurstEvent]:
        """
        Detect bursts from aggregated signals.
        
        Args:
            signals: List of aggregated signals
            context: Optional detection context
            
        Returns:
            List of detected burst events
        """
        bursts = []
        
        # Group signals by frame
        frame_signals = defaultdict(list)
        for signal in signals:
            if hasattr(signal, 'frame') and signal.frame:
                frame_signals[signal.frame].append(signal)
        
        # Detect bursts for each frame
        for frame, frame_signal_list in frame_signals.items():
            # Get time windows from signals
            for signal in frame_signal_list:
                if hasattr(signal, 'window') and signal.window:
                    # Check if this signal indicates a burst
                    if self._is_burst_signal(signal):
                        burst = self._create_burst_from_signal(signal, frame)
                        if burst:
                            bursts.append(burst)
        
        # Also run traditional detection
        traditional_results = self.detect(method='ensemble')
        
        # Extract bursts from the results dictionary
        traditional_bursts = []
        if isinstance(traditional_results, dict):
            # Handle different possible keys in the results
            if 'bursts' in traditional_results:
                traditional_bursts = traditional_results['bursts']
            elif 'events' in traditional_results:
                traditional_bursts = traditional_results['events']
            elif 'all_bursts' in traditional_results:
                traditional_bursts = traditional_results['all_bursts']
        elif isinstance(traditional_results, list):
            traditional_bursts = traditional_results
        
        # Ensure traditional_bursts is a list
        if not isinstance(traditional_bursts, list):
            traditional_bursts = []
        
        # Merge and deduplicate
        all_bursts = self._merge_burst_lists(bursts, traditional_bursts)
        
        return all_bursts
    
    def _is_burst_signal(self, signal: 'AggregatedSignal') -> bool:
        """Check if a signal indicates a burst."""
        # Check various indicators
        if hasattr(signal, 'temporal_features'):
            tf = signal.temporal_features
            if tf.get('peak_intensity', 0) > 2.0:  # Z-score > 2
                return True
            if tf.get('acceleration', 0) > 1.5:
                return True
        
        if hasattr(signal, 'n_articles') and signal.n_articles > 50:
            # High volume
            return True
        
        return False
    
    def _create_burst_from_signal(self, signal: 'AggregatedSignal', frame: str) -> Optional[BurstEvent]:
        """Create a burst event from an aggregated signal."""
        if not hasattr(signal, 'window') or not signal.window:
            return None
        
        start_date, end_date = signal.window
        
        # Create burst event
        burst = BurstEvent(
            frame=frame,
            start_date=pd.Timestamp(start_date),
            end_date=pd.Timestamp(end_date),
            peak_date=pd.Timestamp(start_date) + (pd.Timestamp(end_date) - pd.Timestamp(start_date)) / 2,
            intensity=signal.temporal_features.get('peak_intensity', 1.0) if hasattr(signal, 'temporal_features') else 1.0,
            volume=signal.n_articles if hasattr(signal, 'n_articles') else 0,
            acceleration=signal.temporal_features.get('acceleration', 0) if hasattr(signal, 'temporal_features') else 0,
            shape='signal_detected',
            detection_methods=['signal_based'],
            confidence=0.8
        )
        
        # Add signal features
        burst.aggregated_signal = signal
        
        return burst
    
    def _merge_burst_lists(self, bursts1: List[BurstEvent], bursts2: List[BurstEvent]) -> List[BurstEvent]:
        """Merge two lists of bursts, removing duplicates."""
        all_bursts = bursts1 + bursts2
        
        # Remove duplicates based on frame and time overlap
        unique_bursts = []
        seen = set()
        
        for burst in all_bursts:
            key = (burst.frame, burst.start_date.date(), burst.end_date.date())
            if key not in seen:
                seen.add(key)
                unique_bursts.append(burst)
        
        return unique_bursts
    
    def score_detection(self, burst: Any) -> float:
        """
        Score burst detection result.
        
        Args:
            burst: BurstEvent or detection result dict
            
        Returns:
            Score in [0, 1] range
        """
        # Handle dict format (full detection result)
        if isinstance(burst, dict):
            bursts = burst.get('bursts', [])
            if not bursts:
                return 0.0
            
            # Use summary statistics if available
            summary = burst.get('summary', {})
            if summary:
                intensity_score = min(1.0, summary.get('avg_intensity', 0) / 5)
                duration_score = min(1.0, summary.get('avg_duration', 0) / 30)
                count_score = min(1.0, len(bursts) / 10)
                return (intensity_score + duration_score + count_score) / 3
            else:
                # Score based on burst count
                return min(1.0, len(bursts) / 10)
        
        # Handle single BurstEvent (legacy)
        if not isinstance(burst, BurstEvent):
            return 0.0
            
        scores = []
        
        # Intensity score (normalized)
        intensity_score = min(1.0, burst.intensity / 5.0)
        scores.append(intensity_score * 0.3)
        
        # Duration score (longer bursts more important)
        duration_score = min(1.0, burst.duration_days / 30.0)
        scores.append(duration_score * 0.2)
        
        # Confidence score
        scores.append(burst.confidence * 0.2)
        
        # Statistical significance
        scores.append(burst.statistical_significance * 0.15)
        
        # Trigger confidence (strong triggers indicate important events)
        scores.append(burst.trigger_confidence * 0.15)
        
        return sum(scores)
    
    def get_burst_summary(self, burst: BurstEvent) -> Dict[str, Any]:
        """
        Get human-readable summary of burst event.
        
        Args:
            burst: BurstEvent to summarize
            
        Returns:
            Summary dictionary
        """
        return {
            'frame': burst.frame,
            'period': f"{burst.start_date.date()} to {burst.end_date.date()}",
            'duration_days': burst.duration_days,
            'peak_date': burst.peak_date.date(),
            'intensity': f"{burst.intensity:.2f}x baseline",
            'shape': burst.shape,
            'trigger': burst.trigger_type or 'unknown',
            'trigger_confidence': f"{burst.trigger_confidence:.0%}",
            'detection_methods': ', '.join(burst.detection_methods),
            'confidence': f"{burst.confidence:.0%}",
            'statistical_significance': f"{burst.statistical_significance:.0%}",
            'is_cascade_trigger': burst.is_cascade_trigger,
            'score': f"{self.score_detection(burst):.2f}"
        }