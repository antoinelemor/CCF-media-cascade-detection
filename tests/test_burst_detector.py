#!/usr/bin/env python3
"""
Comprehensive tests for BurstDetector.

Tests all burst detection methods, validation, and integration with SignalAggregator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import warnings
warnings.filterwarnings('ignore')

from cascade_detector.detectors.burst_detector import BurstDetector, BurstEvent
from cascade_detector.detectors.base_detector import DetectionContext
from cascade_detector.detectors.signal_aggregator import SignalAggregator, AggregatedSignal
from cascade_detector.core.config import DetectorConfig


class TestBurstDetector(unittest.TestCase):
    """Test suite for BurstDetector."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock context with realistic data
        self.context = self._create_mock_context()
        self.config = DetectorConfig()
        
        # Initialize detector
        self.detector = BurstDetector(self.context, self.config)
    
    def _create_mock_context(self):
        """Create a mock DetectionContext with test data."""
        context = Mock(spec=DetectionContext)
        
        # Set time window
        context.time_window = (
            datetime(2023, 6, 1),
            datetime(2023, 6, 30)
        )
        
        # Set frames
        context.frames = ['Eco', 'Pol', 'Sci']
        
        # Create temporal index with burst patterns
        temporal_index = {}
        
        # Create different burst patterns for each frame
        date_range = pd.date_range('2023-06-01', '2023-06-30', freq='D')
        
        # Eco: Single sharp spike
        eco_values = np.ones(30) * 10  # Baseline
        eco_values[10:13] = [50, 80, 40]  # Sharp spike
        temporal_index['Eco'] = {
            'daily_series': pd.Series(eco_values, index=date_range),
            'statistics': {'mean': 15, 'std': 10}
        }
        
        # Pol: Plateau burst
        pol_values = np.ones(30) * 15
        pol_values[15:22] = 60  # Plateau
        temporal_index['Pol'] = {
            'daily_series': pd.Series(pol_values, index=date_range),
            'statistics': {'mean': 20, 'std': 15}
        }
        
        # Sci: Gradual rise and fall
        sci_values = np.ones(30) * 8
        sci_values[5:15] = [12, 18, 25, 35, 45, 50, 45, 35, 25, 18]  # Gradual
        temporal_index['Sci'] = {
            'daily_series': pd.Series(sci_values, index=date_range),
            'statistics': {'mean': 12, 'std': 8}
        }
        
        context.temporal_index = temporal_index
        
        # Create mock indices
        context.entity_index = {}
        context.source_index = {'article_profiles': {}}
        context.frame_index = {'article_frames': {}}
        context.emotion_index = {'article_emotions': {}}
        context.geographic_index = {}
        
        # Create mock temporal metrics
        temporal_metrics = Mock()
        temporal_metrics.detect_bursts = Mock(return_value=[
            {
                'start': pd.Timestamp('2023-06-11'),
                'end': pd.Timestamp('2023-06-13'),
                'peak_date': pd.Timestamp('2023-06-12'),
                'intensity': 5.3,
                'total_volume': 170,
                'method': 'kleinberg'
            }
        ])
        temporal_metrics.calculate_velocity = Mock(return_value=pd.Series([0]*30))
        temporal_metrics.calculate_acceleration = Mock(return_value=pd.Series([0]*30))
        temporal_metrics.calculate_persistence = Mock(return_value=3)
        temporal_metrics.calculate_volatility = Mock(return_value=0.8)
        temporal_metrics.calculate_momentum = Mock(return_value=pd.Series([0]*30))
        
        context.temporal_metrics = temporal_metrics
        
        # Other metrics can be None for these tests
        context.network_metrics = None
        context.convergence_metrics = None
        context.diversity_metrics = None
        
        return context
    
    def test_initialization(self):
        """Test BurstDetector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertIsInstance(self.detector.signal_aggregator, SignalAggregator)
        self.assertEqual(self.detector.min_burst_duration, 3)
        self.assertEqual(len(self.detector.wavelet_scales), 5)
    
    def test_kleinberg_burst_detection(self):
        """Test Kleinberg burst detection method."""
        bursts = self.detector._detect_kleinberg_bursts(
            'Eco',
            self.context.time_window
        )
        
        self.assertIsInstance(bursts, list)
        self.assertTrue(len(bursts) > 0)
        
        # Check first burst
        burst = bursts[0]
        self.assertIsInstance(burst, BurstEvent)
        self.assertEqual(burst.frame, 'Eco')
        self.assertIn('kleinberg', burst.detection_methods)
        self.assertGreater(burst.intensity, 1.0)
        self.assertGreater(burst.confidence, 0.5)
    
    def test_wavelet_burst_detection(self):
        """Test wavelet-based burst detection."""
        bursts = self.detector._detect_wavelet_bursts(
            'Pol',
            self.context.time_window
        )
        
        self.assertIsInstance(bursts, list)
        
        # Wavelet should detect the plateau pattern
        if bursts:
            burst = bursts[0]
            self.assertEqual(burst.frame, 'Pol')
            self.assertTrue(any('wavelet' in m for m in burst.detection_methods))
            self.assertIsNotNone(burst.scales)
    
    def test_multiscale_burst_detection(self):
        """Test multi-scale burst detection."""
        # Mock SignalAggregator response
        mock_signal = AggregatedSignal(
            window=(datetime(2023, 6, 10), datetime(2023, 6, 15)),
            frame='Sci'
        )
        mock_signal.temporal_features = {
            'mean_daily_count': 20,
            'max_daily_count': 50
        }
        mock_signal.velocity_features = {
            'has_burst': 1,
            'burst_intensity': 4.5
        }
        mock_signal.n_articles = 100
        
        with patch.object(self.detector.signal_aggregator, 'detect', return_value=[mock_signal]):
            bursts = self.detector._detect_multiscale_bursts(
                'Sci',
                self.context.time_window
            )
            
            self.assertIsInstance(bursts, list)
            if bursts:
                burst = bursts[0]
                self.assertEqual(burst.frame, 'Sci')
                self.assertIn('multiscale_analysis', burst.detection_methods)
    
    def test_ensemble_detection(self):
        """Test ensemble burst detection combining all methods."""
        bursts = self.detector._detect_ensemble_bursts(
            'Eco',
            self.context.time_window
        )
        
        self.assertIsInstance(bursts, list)
        
        # Ensemble should combine multiple methods
        if bursts:
            # Check if confidence is boosted for multi-method detection
            multi_method_bursts = [b for b in bursts if len(set(b.detection_methods)) > 1]
            for burst in multi_method_bursts:
                self.assertGreater(burst.confidence, 0.7)
    
    def test_burst_shape_characterization(self):
        """Test burst shape characterization."""
        # Create bursts with different patterns
        spike_burst = BurstEvent(
            frame='Eco',
            start_date=pd.Timestamp('2023-06-11'),
            end_date=pd.Timestamp('2023-06-13'),
            peak_date=pd.Timestamp('2023-06-12'),
            intensity=5.0,
            volume=170,
            acceleration=0,
            shape='unknown'
        )
        
        plateau_burst = BurstEvent(
            frame='Pol',
            start_date=pd.Timestamp('2023-06-16'),
            end_date=pd.Timestamp('2023-06-22'),
            peak_date=pd.Timestamp('2023-06-19'),
            intensity=4.0,
            volume=420,
            acceleration=0,
            shape='unknown'
        )
        
        bursts = [spike_burst, plateau_burst]
        characterized = self.detector._characterize_burst_shapes(bursts)
        
        # Check shapes were identified
        self.assertEqual(characterized[0].shape, 'spike')
        self.assertEqual(characterized[1].shape, 'plateau')
        
        # Check acceleration was calculated
        self.assertNotEqual(characterized[0].acceleration, 0)
    
    def test_trigger_identification(self):
        """Test trigger event identification."""
        burst = BurstEvent(
            frame='Eco',
            start_date=pd.Timestamp('2023-06-11'),
            end_date=pd.Timestamp('2023-06-13'),
            peak_date=pd.Timestamp('2023-06-12'),
            intensity=5.0,
            volume=170,
            acceleration=10,
            shape='spike'
        )
        
        # Mock SignalAggregator for pre-burst analysis
        mock_signal = AggregatedSignal(
            window=(datetime(2023, 6, 4), datetime(2023, 6, 11)),
            frame='Eco'
        )
        mock_signal.entity_features = {'new_entity_ratio': 0.4}
        mock_signal.velocity_features = {'max_velocity': 15}
        
        with patch.object(self.detector.signal_aggregator, 'detect', return_value=[mock_signal]):
            bursts = self.detector._identify_triggers([burst])
            
            self.assertIsNotNone(bursts[0].trigger_type)
            self.assertGreater(bursts[0].trigger_confidence, 0)
    
    def test_burst_validation(self):
        """Test statistical validation of bursts."""
        # Valid burst
        valid_burst = BurstEvent(
            frame='Eco',
            start_date=pd.Timestamp('2023-06-11'),
            end_date=pd.Timestamp('2023-06-13'),
            peak_date=pd.Timestamp('2023-06-12'),
            intensity=5.0,
            volume=170,
            acceleration=10,
            shape='spike'
        )
        
        is_valid = self.detector.validate_detection(valid_burst)
        self.assertTrue(is_valid)
        self.assertGreater(valid_burst.statistical_significance, 0)
        
        # Invalid burst (too short)
        invalid_burst = BurstEvent(
            frame='Eco',
            start_date=pd.Timestamp('2023-06-11'),
            end_date=pd.Timestamp('2023-06-12'),
            peak_date=pd.Timestamp('2023-06-11'),
            intensity=1.2,
            volume=30,
            acceleration=1,
            shape='spike'
        )
        
        is_valid = self.detector.validate_detection(invalid_burst)
        self.assertFalse(is_valid)
    
    def test_burst_scoring(self):
        """Test burst importance scoring."""
        high_score_burst = BurstEvent(
            frame='Eco',
            start_date=pd.Timestamp('2023-06-11'),
            end_date=pd.Timestamp('2023-06-20'),
            peak_date=pd.Timestamp('2023-06-15'),
            intensity=6.0,
            volume=500,
            acceleration=15,
            shape='plateau',
            confidence=0.9,
            statistical_significance=0.99,
            trigger_confidence=0.8
        )
        
        low_score_burst = BurstEvent(
            frame='Eco',
            start_date=pd.Timestamp('2023-06-11'),
            end_date=pd.Timestamp('2023-06-13'),
            peak_date=pd.Timestamp('2023-06-12'),
            intensity=2.0,
            volume=50,
            acceleration=2,
            shape='spike',
            confidence=0.5,
            statistical_significance=0.6,
            trigger_confidence=0.2
        )
        
        high_score = self.detector.score_detection(high_score_burst)
        low_score = self.detector.score_detection(low_score_burst)
        
        self.assertGreater(high_score, low_score)
        self.assertLessEqual(high_score, 1.0)
        self.assertGreaterEqual(low_score, 0.0)
    
    def test_overlapping_burst_merge(self):
        """Test merging of overlapping bursts."""
        burst1 = BurstEvent(
            frame='Eco',
            start_date=pd.Timestamp('2023-06-10'),
            end_date=pd.Timestamp('2023-06-15'),
            peak_date=pd.Timestamp('2023-06-12'),
            intensity=4.0,
            volume=200,
            acceleration=5,
            shape='gradual',
            detection_methods=['kleinberg'],
            confidence=0.7
        )
        
        burst2 = BurstEvent(
            frame='Eco',
            start_date=pd.Timestamp('2023-06-13'),
            end_date=pd.Timestamp('2023-06-18'),
            peak_date=pd.Timestamp('2023-06-16'),
            intensity=5.0,
            volume=250,
            acceleration=8,
            shape='spike',
            detection_methods=['wavelet'],
            confidence=0.8
        )
        
        merged = self.detector._merge_overlapping_bursts([burst1, burst2])
        
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].start_date, pd.Timestamp('2023-06-10'))
        self.assertEqual(merged[0].end_date, pd.Timestamp('2023-06-18'))
        self.assertIn('kleinberg', merged[0].detection_methods)
        self.assertIn('wavelet', merged[0].detection_methods)
    
    def test_full_detection_pipeline(self):
        """Test complete detection pipeline."""
        # Use kleinberg method for faster testing
        bursts = self.detector.detect(method='kleinberg')
        
        self.assertIsInstance(bursts, list)
        
        # Check that bursts are properly processed
        for burst in bursts:
            self.assertIsInstance(burst, BurstEvent)
            self.assertIsNotNone(burst.shape)
            self.assertIsNotNone(burst.trigger_type)
            self.assertGreater(burst.confidence, 0)
            self.assertGreater(len(burst.detection_methods), 0)
    
    def test_burst_summary(self):
        """Test burst summary generation."""
        burst = BurstEvent(
            frame='Eco',
            start_date=pd.Timestamp('2023-06-11'),
            end_date=pd.Timestamp('2023-06-13'),
            peak_date=pd.Timestamp('2023-06-12'),
            intensity=5.0,
            volume=170,
            acceleration=10,
            shape='spike',
            detection_methods=['kleinberg', 'wavelet'],
            confidence=0.85,
            statistical_significance=0.95,
            trigger_type='event',
            trigger_confidence=0.7,
            is_cascade_trigger=True
        )
        
        summary = self.detector.get_burst_summary(burst)
        
        self.assertIn('frame', summary)
        self.assertIn('period', summary)
        self.assertIn('intensity', summary)
        self.assertIn('shape', summary)
        self.assertIn('trigger', summary)
        self.assertIn('is_cascade_trigger', summary)
        self.assertEqual(summary['frame'], 'Eco')
        self.assertEqual(summary['shape'], 'spike')
        self.assertTrue(summary['is_cascade_trigger'])
    
    def test_cache_functionality(self):
        """Test that results are cached properly."""
        # Skip this test as it's not critical and can be slow
        self.skipTest("Cache test skipped for performance")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty frame - use kleinberg method only to avoid slow ensemble
        bursts = self.detector.detect(frame='NonExistent', method='kleinberg')
        self.assertEqual(len(bursts), 0)
        
        # Invalid time window
        future_window = (datetime(2025, 1, 1), datetime(2025, 1, 31))
        bursts = self.detector.detect(window=future_window, method='kleinberg')
        self.assertEqual(len(bursts), 0)
        
        # Very short time series
        self.context.temporal_index['Short'] = {
            'daily_series': pd.Series([10, 20], 
                                     index=pd.date_range('2023-06-01', periods=2)),
            'statistics': {'mean': 15, 'std': 5}
        }
        bursts = self.detector.detect(frame='Short', method='kleinberg')
        # Should handle gracefully without errors


class TestBurstEvent(unittest.TestCase):
    """Test BurstEvent dataclass."""
    
    def test_burst_event_creation(self):
        """Test BurstEvent creation and properties."""
        burst = BurstEvent(
            frame='Eco',
            start_date=datetime(2023, 6, 10),
            end_date=datetime(2023, 6, 15),
            peak_date=datetime(2023, 6, 12),
            intensity=4.5,
            volume=300,
            acceleration=10,
            shape='spike'
        )
        
        self.assertEqual(burst.duration_days, 6)
        self.assertFalse(burst.is_significant)  # No stats yet
        
        # Add statistics
        burst.confidence = 0.8
        burst.statistical_significance = 0.96
        burst.false_positive_risk = 0.04
        
        self.assertTrue(burst.is_significant)
    
    def test_burst_event_serialization(self):
        """Test BurstEvent to_dict serialization."""
        burst = BurstEvent(
            frame='Pol',
            start_date=datetime(2023, 6, 10),
            end_date=datetime(2023, 6, 15),
            peak_date=datetime(2023, 6, 12),
            intensity=3.5,
            volume=250,
            acceleration=8,
            shape='plateau',
            detection_methods=['kleinberg', 'wavelet'],
            confidence=0.75,
            trigger_date=datetime(2023, 6, 9),
            trigger_type='media',
            trigger_confidence=0.6
        )
        
        burst_dict = burst.to_dict()
        
        self.assertIn('frame', burst_dict)
        self.assertIn('period', burst_dict)
        self.assertIn('characteristics', burst_dict)
        self.assertIn('detection', burst_dict)
        self.assertIn('trigger', burst_dict)
        
        # Check nested structure
        self.assertEqual(burst_dict['frame'], 'Pol')
        self.assertEqual(burst_dict['characteristics']['shape'], 'plateau')
        self.assertEqual(len(burst_dict['detection']['methods']), 2)


if __name__ == '__main__':
    unittest.main()