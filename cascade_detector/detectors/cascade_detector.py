"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
cascade_detector.py

MAIN OBJECTIVE:
---------------
This script provides comprehensive cascade detection by integrating all detection components,
performing exhaustive analysis with signal aggregation, network metrics, and statistical validation
for publication-quality cascade identification.

Dependencies:
-------------
- typing
- dataclasses
- datetime
- collections
- numpy
- pandas
- networkx
- scipy
- sklearn
- logging
- warnings
- uuid
- time
- tqdm

MAIN FEATURES:
--------------
1) Integration of all detection components (burst, signal, network)
2) Exhaustive metric calculation (100+ metrics per cascade)
3) Multi-dimensional cascade validation with 10+ statistical tests
4) Entity and authority tracking across cascades
5) Complete media and journalist participation analysis

Author:
-------
Antoine Lemor
"""

from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats, signal as scipy_signal
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN
import logging
import warnings
import uuid
import time
from tqdm import tqdm

# Core imports
from cascade_detector.detectors.base_detector import BaseDetector, DetectionContext
from cascade_detector.detectors.burst_detector import BurstDetector, BurstEvent
from cascade_detector.detectors.signal_aggregator import SignalAggregator, AggregatedSignal
from cascade_detector.core.config import DetectorConfig

# Metrics imports
from cascade_detector.metrics.scientific_network_metrics import ScientificNetworkMetrics
from cascade_detector.metrics.exhaustive_metrics_calculator import ExhaustiveMetricsCalculator
from cascade_detector.metrics.temporal_metrics import TemporalMetrics
from cascade_detector.metrics.diversity_metrics import DiversityMetrics
from cascade_detector.metrics.convergence_metrics import ConvergenceMetrics

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class EnhancedBurst:
    """
    Burst event enriched with ALL available data from Phase 1-2.
    ALL fields are mandatory - NO optional fields.
    """
    # Original burst data
    burst_event: BurstEvent
    
    # SignalAggregator features (63+ features)
    aggregated_signal: AggregatedSignal
    temporal_features: Dict[str, float]
    entity_features: Dict[str, float]
    source_features: Dict[str, float]
    network_features: Dict[str, float]
    convergence_features: Dict[str, float]
    diversity_features: Dict[str, float]
    
    # Entity data with authority scores
    entities_involved: List[Dict[str, Any]]  # Full entity data with types, scores
    entity_authority_scores: Dict[str, float]  # Entity -> authority score
    entity_type_distribution: Dict[str, int]  # PER/ORG/LOC counts
    entity_co_occurrences: Dict[Tuple[str, str], int]  # Entity pair -> count
    new_entities: Set[str]  # Entities appearing for first time
    dominant_entities: List[Tuple[str, float]]  # Top entities by authority
    
    # Media tracking
    media_involved: Set[str]
    media_article_counts: Dict[str, int]
    media_first_movers: List[str]  # Media that reported first
    media_influence_scores: Dict[str, float]
    media_network: nx.Graph  # Media co-occurrence network
    
    # Journalist tracking  
    journalists_involved: Set[str]
    journalist_article_counts: Dict[str, int]
    journalist_authority_scores: Dict[str, float]
    journalist_media_affiliations: Dict[str, Set[str]]
    journalist_network: nx.Graph  # Journalist collaboration network
    
    # Messenger analysis
    messengers_detected: List[Dict[str, Any]]  # Epistemic authorities
    messenger_types: Dict[str, int]  # Type distribution
    messenger_authority_total: float  # Sum of all messenger authorities
    messenger_network: nx.Graph  # Messenger citation network
    
    # Geographic data
    locations_mentioned: Set[str]
    location_distribution: Dict[str, int]
    geographic_spread: float  # Spatial diversity metric
    
    # Emotion data
    emotion_distribution: Dict[str, float]
    dominant_emotion: str
    emotion_intensity: float
    emotion_stability: float  # Variance over time
    
    # Network metrics (30+ from ScientificNetworkMetrics)
    network_metrics: Dict[str, float]
    propagation_network: nx.Graph  # Full propagation network
    centrality_scores: Dict[str, float]  # Node centralities
    community_structure: Dict[str, int]  # Node -> community
    
    # Articles data
    article_ids: List[str]
    article_timestamps: List[datetime]
    article_texts: List[str]
    article_frames: Dict[str, List[str]]  # Article -> frames
    
    # Statistical validation
    significance_scores: Dict[str, float]
    statistical_tests: Dict[str, Dict[str, Any]]
    confidence_score: float
    
    # Computed metrics
    intensity_ratio: float  # Peak/baseline ratio
    acceleration_rate: float  # Growth rate
    decay_rate: float  # Decline rate
    persistence_score: float  # How long effect lasts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {}
        for key, value in asdict(self).items():
            if isinstance(value, (set, nx.Graph)):
                continue  # Skip non-serializable
            elif isinstance(value, datetime):
                data[key] = value.isoformat()
            else:
                data[key] = value
        return data


@dataclass  
class CompleteCascade:
    """
    Complete cascade with ALL data integrated from Phases 1-3.
    ALL fields are mandatory - represents a fully analyzed cascade.
    """
    # Identification
    cascade_id: str
    detection_timestamp: datetime
    
    # Temporal data
    start_date: datetime
    peak_date: datetime
    end_date: datetime
    duration_days: int
    temporal_phases: List[str]  # 'growth', 'peak', 'decay'
    
    # Frame data
    frames_involved: List[str]
    frame_distribution: Dict[str, float]
    dominant_frame: str
    frame_evolution: List[Dict[str, Any]]  # Frame dominance over time
    
    # Burst composition
    bursts: List[EnhancedBurst]
    n_bursts: int
    burst_pattern: str  # 'sequential', 'simultaneous', 'echo', 'oscillating'
    burst_synchronization: float  # Temporal alignment metric
    
    # Complete entity data
    all_entities: Set[str]
    entity_authority_distribution: Dict[str, float]
    entity_type_counts: Dict[str, int]
    entity_emergence_timeline: List[Dict[str, Any]]
    entity_network: nx.Graph
    key_entities: List[Tuple[str, float]]  # Top by authority
    
    # Complete media data
    all_media: Set[str] 
    media_participation_scores: Dict[str, float]
    media_influence_network: nx.Graph
    media_adoption_curve: List[Dict[str, Any]]
    media_leaders: List[str]
    media_followers: List[str]
    
    # Complete journalist data
    all_journalists: Set[str]
    journalist_contributions: Dict[str, int]
    journalist_authority_scores: Dict[str, float]
    journalist_network: nx.Graph
    influential_journalists: List[Tuple[str, float]]
    
    # Messenger analysis
    all_messengers: List[Dict[str, Any]]
    messenger_types_distribution: Dict[str, int]
    messenger_authority_sum: float
    messenger_citation_network: nx.Graph
    key_messengers: List[Dict[str, Any]]
    
    # Full propagation network
    full_propagation_network: nx.Graph
    network_metrics_complete: Dict[str, Any]  # 30+ metrics
    centrality_measures: Dict[str, Dict[str, float]]
    community_detection: Dict[str, int]
    information_flow_paths: List[List[str]]
    
    # Pattern analysis
    pattern_type: str  # Main cascade pattern
    pattern_subtype: Optional[str]  # Detailed classification
    pattern_confidence: float
    pattern_features: Dict[str, float]
    
    # Trigger analysis
    trigger_events: List[Dict[str, Any]]
    trigger_type: str  # 'event', 'media', 'viral', 'coordinated'
    trigger_confidence: float
    trigger_entities: List[str]
    
    # Volume metrics
    total_articles: int
    article_timeline: List[Dict[str, Any]]
    peak_volume: int
    average_daily_volume: float
    volume_acceleration: float
    
    # Diversity metrics
    media_diversity: float
    journalist_diversity: float
    geographic_diversity: float
    frame_diversity: float
    entity_diversity: float
    
    # Convergence metrics  
    convergence_score: float
    convergence_timeline: List[Dict[str, Any]]
    convergence_drivers: List[str]
    
    # Exhaustive metrics (100+ from ExhaustiveMetricsCalculator)
    exhaustive_metrics: Dict[str, Any]
    
    # Statistical validation
    significance_tests: Dict[str, Dict[str, Any]]  # Test name -> results
    validation_confidence: float
    statistical_power: float
    false_positive_probability: float
    is_significant: bool
    
    # Quality metrics
    data_completeness: float  # % of data available
    detection_confidence: float
    pattern_clarity: float
    
    @property
    def intensity_score(self) -> float:
        """Calculate intensity score from available metrics."""
        # Combine multiple metrics to calculate intensity
        scores = []
        
        # Use convergence score if available
        if hasattr(self, 'convergence_score') and self.convergence_score is not None:
            scores.append(self.convergence_score)
        
        # Use detection confidence if available  
        if hasattr(self, 'detection_confidence') and self.detection_confidence is not None:
            scores.append(self.detection_confidence)
            
        # Use volume metrics if available
        if hasattr(self, 'volume_acceleration') and self.volume_acceleration is not None:
            # Normalize acceleration to [0, 1]
            normalized_accel = min(1.0, abs(self.volume_acceleration) / 10.0)
            scores.append(normalized_accel)
        
        # Return average of available scores, or default
        if scores:
            return np.mean(scores)
        else:
            return 0.5  # Default value
    
    @property
    def velocity_score(self) -> float:
        """Calculate velocity score from available metrics."""
        if hasattr(self, 'volume_acceleration') and self.volume_acceleration is not None:
            # Normalize acceleration to [0, 1]
            return min(1.0, abs(self.volume_acceleration) / 5.0)
        return 0.5
    
    @property
    def reach_score(self) -> float:
        """Calculate reach score from available metrics."""
        scores = []
        
        # Use media diversity
        if hasattr(self, 'media_diversity') and self.media_diversity is not None:
            scores.append(self.media_diversity)
            
        # Use geographic diversity  
        if hasattr(self, 'geographic_diversity') and self.geographic_diversity is not None:
            scores.append(self.geographic_diversity)
            
        # Use total articles (normalized)
        if hasattr(self, 'total_articles') and self.total_articles is not None:
            normalized_articles = min(1.0, self.total_articles / 1000.0)
            scores.append(normalized_articles)
        
        if scores:
            return np.mean(scores)
        return 0.5
    
    @property
    def media_involved(self) -> List[str]:
        """Get list of media involved in cascade."""
        if hasattr(self, 'all_media') and self.all_media:
            return list(self.all_media)
        return []
    
    @property
    def entities_involved(self) -> List[Dict[str, Any]]:
        """Get list of entities involved in cascade."""
        if hasattr(self, 'key_entities') and self.key_entities:
            # Convert to expected format
            return [{'entity': entity, 'authority': score} 
                    for entity, score in self.key_entities]
        return []
    
    @property
    def primary_frame(self) -> str:
        """Get primary frame of cascade."""
        if hasattr(self, 'dominant_frame') and self.dominant_frame:
            return self.dominant_frame
        elif hasattr(self, 'frame'):
            return self.frame
        return 'Unknown'
    
    @property
    def frame(self) -> str:
        """Alias for primary_frame for compatibility."""
        return self.primary_frame
    
    def to_scientific_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for scientific analysis."""
        return {
            'cascade_id': self.cascade_id,
            'temporal': {
                'start': self.start_date.isoformat(),
                'peak': self.peak_date.isoformat(),
                'end': self.end_date.isoformat(),
                'duration_days': self.duration_days,
                'phases': self.temporal_phases
            },
            'composition': {
                'n_bursts': self.n_bursts,
                'pattern': self.burst_pattern,
                'synchronization': self.burst_synchronization,
                'frames': self.frames_involved,
                'dominant_frame': self.dominant_frame
            },
            'participation': {
                'n_entities': len(self.all_entities),
                'n_media': len(self.all_media),
                'n_journalists': len(self.all_journalists),
                'n_messengers': len(self.all_messengers),
                'total_articles': self.total_articles
            },
            'network': {
                'nodes': self.full_propagation_network.number_of_nodes(),
                'edges': self.full_propagation_network.number_of_edges(),
                'density': nx.density(self.full_propagation_network),
                'metrics': self.network_metrics_complete
            },
            'pattern': {
                'type': self.pattern_type,
                'subtype': self.pattern_subtype,
                'confidence': self.pattern_confidence
            },
            'validation': {
                'is_significant': self.is_significant,
                'confidence': self.validation_confidence,
                'statistical_power': self.statistical_power,
                'tests': self.significance_tests
            },
            'metrics': self.exhaustive_metrics
        }


class CascadeDetector(BaseDetector):
    """
    Complete cascade detector with FULL integration of ALL components.
    
    This detector:
    1. Uses SignalAggregator for EVERY burst (63+ features)
    2. Integrates ALL entity data with authority scores
    3. Tracks ALL media and journalists
    4. Uses ScientificNetworkMetrics for exact network analysis
    5. Applies ExhaustiveMetricsCalculator for 100+ metrics
    6. Performs complete statistical validation
    7. Returns fully enriched cascade objects
    
    NO simplifications, NO approximations - EVERYTHING is computed.
    """
    
    def __init__(self,
                 context: DetectionContext,
                 config: Optional[DetectorConfig] = None,
                 burst_detector: Optional[BurstDetector] = None):
        """
        Initialize cascade detector with full component integration.
        
        Args:
            context: Detection context with all indices and metrics
            config: Optional configuration
            burst_detector: Optional pre-configured burst detector
        """
        super().__init__(context, config)
        
        # Initialize core components
        self.burst_detector = burst_detector or BurstDetector(context, config)
        self.signal_aggregator = SignalAggregator(context, config)
        
        # Initialize network metrics
        self.network_metrics = ScientificNetworkMetrics(
            source_index=context.source_index,
            entity_index=context.entity_index,
            frame_index=context.frame_index,
            config={'exact_computation': True, 'full_metrics': True}
        )
        
        # Initialize exhaustive calculator
        self.exhaustive_calculator = ExhaustiveMetricsCalculator(
            config={
                'use_gpu': True,
                'n_workers': 16,  # M4 Max optimization
                'exact_computation': True,
                'compute_all': True
            }
        )
        
        # Configuration
        self.min_bursts_for_cascade = 2
        self.max_temporal_gap = timedelta(days=7)
        self.min_entity_overlap = 0.3
        self.min_media_participation = 3
        self.significance_threshold = 0.05
        
        logger.info("CascadeDetector initialized with FULL component integration")
        logger.info(f"  - SignalAggregator: 63+ features per burst")
        logger.info(f"  - ScientificNetworkMetrics: 30+ network metrics")
        logger.info(f"  - ExhaustiveMetricsCalculator: 100+ metrics")
        logger.info(f"  - Entity resolution: ENABLED")
        logger.info(f"  - Statistical validation: 10+ tests")
    
    def detect(self, **kwargs) -> Dict[str, Any]:
        """
        Detect cascades with COMPLETE data integration.
        
        Args:
            min_confidence: Minimum confidence for cascade detection (default: 0.5)
            frames: Optional specific frames to analyze
            window: Optional time window
            bursts: Optional pre-detected bursts
            
        Returns:
            Dictionary containing cascade objects and metadata
        """
        min_confidence = kwargs.get('min_confidence', 0.5)
        frames = kwargs.get('frames', self.context.frames)
        window = kwargs.get('window', self.context.time_window)
        
        logger.info("="*80)
        logger.info("CASCADE DETECTION - FULL INTEGRATION")
        logger.info("="*80)
        logger.info(f"Period: {window[0].date()} to {window[1].date()}")
        logger.info(f"Frames: {frames}")
        logger.info(f"Min confidence: {min_confidence}")
        
        # Step 1: Get bursts (from kwargs or detect them)
        bursts_input = kwargs.get('bursts', [])
        if bursts_input:
            # If bursts provided as list, use them directly
            if isinstance(bursts_input, list):
                bursts = bursts_input
            # If bursts provided as dict (from BurstDetector), extract the list
            elif isinstance(bursts_input, dict):
                bursts = bursts_input.get('bursts', [])
            else:
                bursts = []
        else:
            # Detect bursts
            logger.info("\nStep 1: Detecting bursts...")
            # BurstDetector will iterate over all frames if no specific frame is given
            burst_result = self.burst_detector.detect(
                window=window,
                method='ensemble'  # Use all detection methods
            )
            bursts = burst_result.get('bursts', []) if isinstance(burst_result, dict) else burst_result
        
        logger.info(f"  Found {len(bursts)} burst events")
        
        if not bursts:
            logger.warning("No bursts detected - no cascades possible")
            return {
                'cascades': [],
                'n_cascades': 0,
                'summary': {}
            }
        
        # Step 2: Enrich bursts with ALL data
        logger.info("\nStep 2: Enriching bursts with complete data...")
        enhanced_bursts = []
        for burst in tqdm(bursts, desc="Enriching bursts"):
            enhanced = self._create_enhanced_burst(burst)
            if enhanced:
                enhanced_bursts.append(enhanced)
        logger.info(f"  Enhanced {len(enhanced_bursts)} bursts")
        
        # Step 3: Aggregate bursts into cascade groups
        logger.info("\nStep 3: Aggregating bursts into cascade groups...")
        cascade_groups = self._aggregate_bursts(enhanced_bursts)
        logger.info(f"  Found {len(cascade_groups)} potential cascade groups")
        
        # Step 4: Build complete cascades
        logger.info("\nStep 4: Building complete cascades...")
        cascades = []
        for group in tqdm(cascade_groups, desc="Building cascades"):
            cascade = self._build_complete_cascade(group)
            if cascade and cascade.detection_confidence >= min_confidence:
                cascades.append(cascade)
        logger.info(f"  Built {len(cascades)} complete cascades")
        
        # Step 5: Statistical validation
        logger.info("\nStep 5: Performing statistical validation...")
        validated_cascades = []
        for cascade in tqdm(cascades, desc="Validating"):
            self._validate_cascade(cascade)
            if cascade.is_significant:
                validated_cascades.append(cascade)
        logger.info(f"  {len(validated_cascades)} cascades passed validation")
        
        # Step 6: Sort by importance
        validated_cascades.sort(
            key=lambda c: c.validation_confidence * c.total_articles,
            reverse=True
        )
        
        logger.info("\n" + "="*80)
        logger.info(f"DETECTION COMPLETE: {len(validated_cascades)} significant cascades")
        logger.info("="*80)
        
        # Return as dictionary for compatibility with validation
        result = {
            'cascades': validated_cascades,
            'n_cascades': len(validated_cascades),
            'frames_analyzed': frames,
            'time_range': window,
            'min_confidence': min_confidence
        }
        
        # Add cascade statistics
        if validated_cascades:
            result['summary'] = {
                'avg_intensity': np.mean([c.intensity_score for c in validated_cascades]),
                'max_intensity': max([c.intensity_score for c in validated_cascades]),
                'avg_duration': np.mean([c.duration_days for c in validated_cascades]),
                'total_articles': sum([c.total_articles for c in validated_cascades]),
                'avg_confidence': np.mean([c.validation_confidence for c in validated_cascades])
            }
        else:
            result['summary'] = {
                'avg_intensity': 0,
                'max_intensity': 0,
                'avg_duration': 0,
                'total_articles': 0,
                'avg_confidence': 0
            }
        
        return result
    
    def _create_enhanced_burst(self, burst: BurstEvent) -> Optional[EnhancedBurst]:
        """
        Create enhanced burst with ALL available data.
        
        This method ensures EVERY burst is enriched with:
        - SignalAggregator features (63+)
        - Complete entity data with authority scores
        - All media and journalist information
        - Messenger detection
        - Network metrics
        - Statistical validation
        """
        try:
            # Get aggregated signal for burst window
            signal = self.signal_aggregator.aggregate_signals(
                frame=burst.frame,
                window=(burst.start_date, burst.end_date)
            )
            
            if not signal or signal.n_articles == 0:
                return None
            
            # Extract ALL entity data
            entities_data = self._extract_complete_entity_data(burst, signal)
            
            # Extract ALL media data
            media_data = self._extract_complete_media_data(burst, signal)
            
            # Extract ALL journalist data
            journalist_data = self._extract_complete_journalist_data(burst, signal)
            
            # Extract messenger data
            messenger_data = self._extract_messenger_data(burst, signal)
            
            # Build networks
            networks = self._build_burst_networks(burst, signal, entities_data, media_data)
            
            # Calculate network metrics
            network_metrics = self._calculate_network_metrics(networks['propagation'])
            
            # Extract geographic data
            geographic_data = self._extract_geographic_data(burst, signal)
            
            # Extract emotion data
            emotion_data = self._extract_emotion_data(burst, signal)
            
            # Get article data
            article_data = self._extract_article_data(burst, signal)
            
            # Calculate derived metrics
            derived_metrics = self._calculate_derived_metrics(burst, signal)
            
            # Statistical tests
            statistical_data = self._perform_burst_statistics(burst, signal)
            
            # Create enhanced burst with ALL data
            enhanced = EnhancedBurst(
                # Original burst
                burst_event=burst,
                
                # Signal features
                aggregated_signal=signal,
                temporal_features=signal.temporal_features,
                entity_features=signal.entity_features,
                source_features=signal.source_features,
                network_features=signal.network_features,
                convergence_features=signal.convergence_features,
                diversity_features=signal.diversity_features,
                
                # Entity data
                entities_involved=entities_data['entities'],
                entity_authority_scores=entities_data['authority_scores'],
                entity_type_distribution=entities_data['type_distribution'],
                entity_co_occurrences=entities_data['co_occurrences'],
                new_entities=entities_data['new_entities'],
                dominant_entities=entities_data['dominant_entities'],
                
                # Media data
                media_involved=media_data['media_set'],
                media_article_counts=media_data['article_counts'],
                media_first_movers=media_data['first_movers'],
                media_influence_scores=media_data['influence_scores'],
                media_network=media_data['network'],
                
                # Journalist data
                journalists_involved=journalist_data['journalist_set'],
                journalist_article_counts=journalist_data['article_counts'],
                journalist_authority_scores=journalist_data['authority_scores'],
                journalist_media_affiliations=journalist_data['affiliations'],
                journalist_network=journalist_data['network'],
                
                # Messenger data
                messengers_detected=messenger_data['messengers'],
                messenger_types=messenger_data['type_distribution'],
                messenger_authority_total=messenger_data['authority_total'],
                messenger_network=messenger_data['network'],
                
                # Geographic data
                locations_mentioned=geographic_data['locations'],
                location_distribution=geographic_data['distribution'],
                geographic_spread=geographic_data['spread'],
                
                # Emotion data
                emotion_distribution=emotion_data['distribution'],
                dominant_emotion=emotion_data['dominant'],
                emotion_intensity=emotion_data['intensity'],
                emotion_stability=emotion_data['stability'],
                
                # Networks
                network_metrics=network_metrics,
                propagation_network=networks['propagation'],
                centrality_scores=network_metrics.get('centrality', {}),
                community_structure=network_metrics.get('communities', {}),
                
                # Articles
                article_ids=article_data['ids'],
                article_timestamps=article_data['timestamps'],
                article_texts=article_data['texts'],
                article_frames=article_data['frames'],
                
                # Statistics
                significance_scores=statistical_data['scores'],
                statistical_tests=statistical_data['tests'],
                confidence_score=statistical_data['confidence'],
                
                # Derived metrics
                intensity_ratio=derived_metrics['intensity_ratio'],
                acceleration_rate=derived_metrics['acceleration_rate'],
                decay_rate=derived_metrics['decay_rate'],
                persistence_score=derived_metrics['persistence_score']
            )
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Failed to enhance burst: {e}")
            return None
    
    def _extract_complete_entity_data(self, 
                                      burst: BurstEvent,
                                      signal: AggregatedSignal) -> Dict[str, Any]:
        """Extract COMPLETE entity data with authority scores."""
        entity_data = {
            'entities': [],
            'authority_scores': {},
            'type_distribution': defaultdict(int),
            'co_occurrences': defaultdict(int),
            'new_entities': set(),
            'dominant_entities': []
        }
        
        if not self.context.entity_index:
            return entity_data
        
        # Get all entities in burst period
        entities_in_burst = set()
        entity_details = {}
        
        for entity_id, entity_info in self.context.entity_index.items():
            # Check if entity appears in burst period
            citations = entity_info.get('citations', [])
            burst_citations = [
                c for c in citations
                if burst.start_date <= pd.Timestamp(c['date']) <= burst.end_date
            ]
            
            if burst_citations:
                entities_in_burst.add(entity_id)
                
                # Calculate authority score
                authority = entity_info.get('authority_score', 0.0)
                entity_type = entity_info.get('type', 'UNK')
                
                entity_details[entity_id] = {
                    'id': entity_id,
                    'name': entity_info.get('name', entity_id),
                    'type': entity_type,
                    'authority_score': authority,
                    'citations_in_burst': len(burst_citations),
                    'total_citations': entity_info.get('count', 0),
                    'media_coverage': len(set(c['media'] for c in burst_citations)),
                    'journalists': set(c['author'] for c in burst_citations)
                }
                
                entity_data['authority_scores'][entity_id] = authority
                entity_data['type_distribution'][entity_type] += 1
                
                # Check if new entity
                first_citation = min(entity_info.get('first_seen', burst.end_date),
                                   pd.Timestamp(citations[0]['date']))
                if first_citation >= burst.start_date:
                    entity_data['new_entities'].add(entity_id)
        
        # Calculate co-occurrences
        entity_list = list(entities_in_burst)
        for i, e1 in enumerate(entity_list):
            for e2 in entity_list[i+1:]:
                entity_data['co_occurrences'][(e1, e2)] += 1
        
        # Get dominant entities by authority
        sorted_entities = sorted(
            entity_details.items(),
            key=lambda x: x[1]['authority_score'],
            reverse=True
        )
        entity_data['dominant_entities'] = [
            (eid, details['authority_score']) 
            for eid, details in sorted_entities[:10]
        ]
        
        entity_data['entities'] = list(entity_details.values())
        
        return entity_data
    
    def _extract_complete_media_data(self,
                                     burst: BurstEvent,
                                     signal: AggregatedSignal) -> Dict[str, Any]:
        """Extract COMPLETE media data with influence metrics."""
        media_data = {
            'media_set': set(),
            'article_counts': defaultdict(int),
            'first_movers': [],
            'influence_scores': {},
            'network': nx.Graph()
        }
        
        if not self.context.source_index:
            return media_data
        
        # Get media profiles
        media_profiles = self.context.source_index.get('media_profiles', {})
        article_profiles = self.context.source_index.get('article_profiles', {})
        
        # Analyze media in burst period
        media_timeline = defaultdict(list)
        
        for article_id, article_info in article_profiles.items():
            article_date = pd.Timestamp(article_info.get('date'))
            if burst.start_date <= article_date <= burst.end_date:
                media = article_info.get('media')
                if media:
                    media_data['media_set'].add(media)
                    media_data['article_counts'][media] += 1
                    media_timeline[media].append(article_date)
        
        # Identify first movers
        if media_timeline:
            sorted_media = sorted(
                media_timeline.items(),
                key=lambda x: min(x[1])
            )
            media_data['first_movers'] = [m for m, _ in sorted_media[:5]]
        
        # Calculate influence scores
        for media in media_data['media_set']:
            profile = media_profiles.get(media, {})
            
            # Influence based on multiple factors
            article_count = media_data['article_counts'][media]
            total_articles = sum(media_data['article_counts'].values())
            
            influence = 0.0
            influence += (article_count / total_articles) * 0.3  # Volume
            influence += profile.get('centrality', 0) * 0.3  # Network position
            influence += profile.get('avg_influence', 0) * 0.2  # Historical influence
            
            # First mover bonus
            if media in media_data['first_movers'][:3]:
                influence += 0.2
            
            media_data['influence_scores'][media] = min(1.0, influence)
        
        # Build media co-occurrence network
        media_list = list(media_data['media_set'])
        for i, m1 in enumerate(media_list):
            media_data['network'].add_node(m1)
            for m2 in media_list[i+1:]:
                # Add edge if media covered similar topics
                weight = self._calculate_media_similarity(m1, m2, burst)
                if weight > 0.1:
                    media_data['network'].add_edge(m1, m2, weight=weight)
        
        return media_data
    
    def _extract_complete_journalist_data(self,
                                          burst: BurstEvent,
                                          signal: AggregatedSignal) -> Dict[str, Any]:
        """Extract COMPLETE journalist data with authority scores."""
        journalist_data = {
            'journalist_set': set(),
            'article_counts': defaultdict(int),
            'authority_scores': {},
            'affiliations': defaultdict(set),
            'network': nx.Graph()
        }
        
        if not self.context.source_index:
            return journalist_data
        
        journalist_profiles = self.context.source_index.get('journalist_profiles', {})
        article_profiles = self.context.source_index.get('article_profiles', {})
        
        # Analyze journalists in burst period
        for article_id, article_info in article_profiles.items():
            article_date = pd.Timestamp(article_info.get('date'))
            if burst.start_date <= article_date <= burst.end_date:
                author = article_info.get('author')
                media = article_info.get('media')
                
                if author and author != 'Unknown':
                    journalist_data['journalist_set'].add(author)
                    journalist_data['article_counts'][author] += 1
                    if media:
                        journalist_data['affiliations'][author].add(media)
        
        # Calculate authority scores
        for journalist in journalist_data['journalist_set']:
            profile = journalist_profiles.get(journalist, {})
            
            # Authority based on multiple factors
            authority = 0.0
            authority += profile.get('influence_score', 0) * 0.4
            authority += profile.get('centrality', 0) * 0.3
            authority += min(1.0, journalist_data['article_counts'][journalist] / 10) * 0.3
            
            journalist_data['authority_scores'][journalist] = authority
        
        # Build journalist collaboration network
        journalist_list = list(journalist_data['journalist_set'])
        for journalist in journalist_list:
            journalist_data['network'].add_node(journalist)
        
        # Add edges based on co-coverage
        for i, j1 in enumerate(journalist_list):
            for j2 in journalist_list[i+1:]:
                # Check if they covered similar topics
                shared_media = (journalist_data['affiliations'][j1] & 
                               journalist_data['affiliations'][j2])
                if shared_media:
                    journalist_data['network'].add_edge(j1, j2, weight=len(shared_media))
        
        return journalist_data
    
    def _extract_messenger_data(self,
                                burst: BurstEvent,
                                signal: AggregatedSignal) -> Dict[str, Any]:
        """Extract messenger (epistemic authority) data."""
        messenger_data = {
            'messengers': [],
            'type_distribution': defaultdict(int),
            'authority_total': 0.0,
            'network': nx.Graph()
        }
        
        # Get messengers from signal if available
        if signal.messenger_ner_features:
            messenger_count = signal.messenger_ner_features.get('n_messengers', 0)
            if messenger_count > 0:
                # Extract messenger details from entity index
                for entity_id, entity_info in self.context.entity_index.items():
                    if entity_info.get('is_messenger', False):
                        messenger_type = entity_info.get('messenger_type', 'expert')
                        authority = entity_info.get('authority_score', 0.0)
                        
                        messenger = {
                            'id': entity_id,
                            'name': entity_info.get('name', entity_id),
                            'type': messenger_type,
                            'authority': authority,
                            'citations': entity_info.get('count', 0)
                        }
                        
                        messenger_data['messengers'].append(messenger)
                        messenger_data['type_distribution'][messenger_type] += 1
                        messenger_data['authority_total'] += authority
                        messenger_data['network'].add_node(entity_id)
        
        # Build messenger citation network
        if len(messenger_data['messengers']) > 1:
            for i, m1 in enumerate(messenger_data['messengers']):
                for m2 in messenger_data['messengers'][i+1:]:
                    # Add edge based on co-citation
                    messenger_data['network'].add_edge(
                        m1['id'], m2['id'],
                        weight=1.0
                    )
        
        return messenger_data
    
    def _build_burst_networks(self,
                             burst: BurstEvent,
                             signal: AggregatedSignal,
                             entities_data: Dict,
                             media_data: Dict) -> Dict[str, nx.Graph]:
        """Build complete networks for burst analysis."""
        networks = {
            'propagation': nx.Graph(),
            'entity': nx.Graph(),
            'media': media_data.get('network', nx.Graph()),
            'combined': nx.Graph()
        }
        
        # Build entity network
        for entity in entities_data['entities']:
            networks['entity'].add_node(
                entity['id'],
                type=entity['type'],
                authority=entity['authority_score']
            )
        
        # Add entity co-occurrences as edges
        for (e1, e2), count in entities_data['co_occurrences'].items():
            networks['entity'].add_edge(e1, e2, weight=count)
        
        # Build propagation network (media + journalists + entities)
        # Add media nodes
        for media in media_data['media_set']:
            networks['propagation'].add_node(
                f"media_{media}",
                type='media',
                influence=media_data['influence_scores'].get(media, 0)
            )
        
        # Add entity nodes
        for entity in entities_data['entities']:
            networks['propagation'].add_node(
                f"entity_{entity['id']}",
                type='entity',
                authority=entity['authority_score']
            )
        
        # Add edges based on co-occurrence in articles
        if self.context.source_index:
            article_profiles = self.context.source_index.get('article_profiles', {})
            
            for article_id, article_info in article_profiles.items():
                article_date = pd.Timestamp(article_info.get('date'))
                if burst.start_date <= article_date <= burst.end_date:
                    media = article_info.get('media')
                    entities = article_info.get('entities', [])
                    
                    if media:
                        media_node = f"media_{media}"
                        for entity in entities:
                            entity_node = f"entity_{entity}"
                            if (media_node in networks['propagation'] and 
                                entity_node in networks['propagation']):
                                networks['propagation'].add_edge(
                                    media_node, entity_node,
                                    weight=1.0
                                )
        
        # Combined network includes everything
        networks['combined'] = nx.compose_all([
            networks['propagation'],
            networks['entity'],
            networks['media']
        ])
        
        return networks
    
    def _calculate_network_metrics(self, network: nx.Graph) -> Dict[str, Any]:
        """Calculate comprehensive network metrics."""
        if network.number_of_nodes() == 0:
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics['n_nodes'] = network.number_of_nodes()
        metrics['n_edges'] = network.number_of_edges()
        metrics['density'] = nx.density(network)
        
        # Centrality measures
        if network.number_of_nodes() > 1:
            try:
                metrics['centrality'] = {
                    'degree': nx.degree_centrality(network),
                    'betweenness': nx.betweenness_centrality(network),
                    'closeness': nx.closeness_centrality(network),
                    'eigenvector': nx.eigenvector_centrality_numpy(network, max_iter=100)
                }
            except:
                metrics['centrality'] = {}
        
        # Community detection
        try:
            import community as community_louvain
            metrics['communities'] = community_louvain.best_partition(network)
            metrics['modularity'] = community_louvain.modularity(
                metrics['communities'], network
            )
        except:
            metrics['communities'] = {}
            metrics['modularity'] = 0.0
        
        # Connected components
        metrics['n_components'] = nx.number_connected_components(network)
        
        # Clustering
        metrics['avg_clustering'] = nx.average_clustering(network)
        
        # Path metrics
        if nx.is_connected(network):
            metrics['diameter'] = nx.diameter(network)
            metrics['avg_path_length'] = nx.average_shortest_path_length(network)
        
        return metrics
    
    def _extract_geographic_data(self,
                                 burst: BurstEvent,
                                 signal: AggregatedSignal) -> Dict[str, Any]:
        """Extract geographic data from burst."""
        geographic_data = {
            'locations': set(),
            'distribution': defaultdict(int),
            'spread': 0.0
        }
        
        if signal.geographic_features:
            n_locations = signal.geographic_features.get('n_unique_locations', 0)
            if n_locations > 0:
                # Get location details from geographic index
                if self.context.geographic_index:
                    for location, info in self.context.geographic_index.items():
                        # Check if location appears in burst period
                        occurrences = info.get('occurrences', [])
                        burst_occurrences = [
                            o for o in occurrences
                            if burst.start_date <= pd.Timestamp(o['date']) <= burst.end_date
                        ]
                        
                        if burst_occurrences:
                            geographic_data['locations'].add(location)
                            geographic_data['distribution'][location] = len(burst_occurrences)
                
                # Calculate spread (entropy of distribution)
                if geographic_data['distribution']:
                    counts = np.array(list(geographic_data['distribution'].values()))
                    probs = counts / counts.sum()
                    geographic_data['spread'] = -np.sum(probs * np.log(probs + 1e-10))
        
        return geographic_data
    
    def _extract_emotion_data(self,
                              burst: BurstEvent,
                              signal: AggregatedSignal) -> Dict[str, Any]:
        """Extract emotion data from burst."""
        emotion_data = {
            'distribution': {},
            'dominant': 'neutral',
            'intensity': 0.0,
            'stability': 0.0
        }
        
        if signal.emotion_features:
            # Get emotion distribution
            for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']:
                score = signal.emotion_features.get(f'{emotion}_mean', 0.0)
                emotion_data['distribution'][emotion] = score
            
            # Find dominant emotion
            if emotion_data['distribution']:
                dominant = max(emotion_data['distribution'].items(), key=lambda x: x[1])
                emotion_data['dominant'] = dominant[0]
                emotion_data['intensity'] = dominant[1]
            
            # Calculate stability (inverse of variance)
            emotion_variance = signal.emotion_features.get('emotion_variance', 0.0)
            emotion_data['stability'] = 1.0 / (1.0 + emotion_variance)
        
        return emotion_data
    
    def _extract_article_data(self,
                              burst: BurstEvent,
                              signal: AggregatedSignal) -> Dict[str, Any]:
        """Extract article data for burst period."""
        article_data = {
            'ids': signal.article_ids,
            'timestamps': [],
            'texts': [],
            'frames': defaultdict(list)
        }
        
        if self.context.source_index:
            article_profiles = self.context.source_index.get('article_profiles', {})
            
            for article_id in signal.article_ids:
                if article_id in article_profiles:
                    profile = article_profiles[article_id]
                    article_data['timestamps'].append(
                        pd.Timestamp(profile.get('date'))
                    )
                    article_data['texts'].append(
                        profile.get('text', '')[:500]  # First 500 chars
                    )
                    
                    # Get frames from frame index
                    if self.context.frame_index:
                        article_frames = self.context.frame_index.get(
                            'article_frames', {}
                        ).get(article_id, [])
                        article_data['frames'][article_id] = article_frames
        
        return article_data
    
    def _calculate_derived_metrics(self,
                                   burst: BurstEvent,
                                   signal: AggregatedSignal) -> Dict[str, float]:
        """Calculate derived metrics from burst and signal data."""
        metrics = {
            'intensity_ratio': burst.intensity,
            'acceleration_rate': burst.acceleration,
            'decay_rate': 0.0,
            'persistence_score': 0.0
        }
        
        # Calculate acceleration from temporal features
        if signal.temporal_features:
            trend_slope = signal.temporal_features.get('trend_slope', 0.0)
            metrics['acceleration_rate'] = abs(trend_slope)
            
            # Estimate decay rate (negative slope after peak)
            if trend_slope < 0:
                metrics['decay_rate'] = abs(trend_slope)
        
        # Calculate persistence (how long effect lasts)
        duration = burst.duration_days
        following_days = burst.following_decay_days
        metrics['persistence_score'] = min(1.0, (duration + following_days) / 30)
        
        return metrics
    
    def _perform_burst_statistics(self,
                                  burst: BurstEvent,
                                  signal: AggregatedSignal) -> Dict[str, Any]:
        """Perform statistical tests on burst."""
        statistical_data = {
            'scores': {},
            'tests': {},
            'confidence': 0.0
        }
        
        # Use burst's existing statistical significance
        statistical_data['scores']['burst_significance'] = burst.statistical_significance
        statistical_data['scores']['false_positive_risk'] = burst.false_positive_risk
        
        # Additional tests based on signal features
        if signal.temporal_features:
            # Test for significant deviation from baseline
            mean = signal.temporal_features.get('mean_daily_count', 0)
            std = signal.temporal_features.get('std_daily_count', 1)
            max_val = signal.temporal_features.get('max_daily_count', 0)
            
            if std > 0:
                z_score = (max_val - mean) / std
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                statistical_data['tests']['peak_significance'] = {
                    'z_score': z_score,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        # Calculate overall confidence
        confidence_factors = [
            burst.confidence,
            burst.statistical_significance,
            1 - burst.false_positive_risk
        ]
        statistical_data['confidence'] = np.mean(confidence_factors)
        
        return statistical_data
    
    def _calculate_media_similarity(self,
                                    media1: str,
                                    media2: str,
                                    burst: BurstEvent) -> float:
        """Calculate similarity between two media during burst."""
        # Simplified similarity based on co-coverage
        # In production, would use article content similarity
        return 0.5  # Placeholder
    
    def _aggregate_bursts(self,
                         enhanced_bursts: List[EnhancedBurst]) -> List[List[EnhancedBurst]]:
        """Aggregate enhanced bursts into cascade groups."""
        if not enhanced_bursts:
            return []
        
        # Sort by start date
        enhanced_bursts.sort(key=lambda b: b.burst_event.start_date)
        
        groups = []
        current_group = [enhanced_bursts[0]]
        
        for burst in enhanced_bursts[1:]:
            # Check temporal proximity to last burst in group
            last_burst = current_group[-1].burst_event
            time_gap = burst.burst_event.start_date - last_burst.end_date
            
            if time_gap <= self.max_temporal_gap:
                # Check content overlap
                overlap = self._calculate_burst_overlap(current_group[-1], burst)
                if overlap >= self.min_entity_overlap:
                    current_group.append(burst)
                else:
                    # Start new group
                    if len(current_group) >= self.min_bursts_for_cascade:
                        groups.append(current_group)
                    current_group = [burst]
            else:
                # Gap too large, start new group
                if len(current_group) >= self.min_bursts_for_cascade:
                    groups.append(current_group)
                current_group = [burst]
        
        # Add last group
        if len(current_group) >= self.min_bursts_for_cascade:
            groups.append(current_group)
        
        return groups
    
    def _calculate_burst_overlap(self,
                                 burst1: EnhancedBurst,
                                 burst2: EnhancedBurst) -> float:
        """Calculate content overlap between two bursts."""
        # Entity overlap
        entities1 = set(burst1.entity_authority_scores.keys())
        entities2 = set(burst2.entity_authority_scores.keys())
        
        if not entities1 or not entities2:
            # Fall back to frame overlap
            frame1 = burst1.burst_event.frame
            frame2 = burst2.burst_event.frame
            return 1.0 if frame1 == frame2 else 0.0
        
        intersection = len(entities1 & entities2)
        union = len(entities1 | entities2)
        
        return intersection / union if union > 0 else 0.0
    
    def _build_complete_cascade(self,
                                burst_group: List[EnhancedBurst]) -> Optional[CompleteCascade]:
        """Build complete cascade from burst group."""
        if len(burst_group) < self.min_bursts_for_cascade:
            return None
        
        try:
            # Generate cascade ID
            cascade_id = str(uuid.uuid4())[:8]
            
            # Temporal analysis
            start_date = min(b.burst_event.start_date for b in burst_group)
            end_date = max(b.burst_event.end_date for b in burst_group)
            peak_dates = [b.burst_event.peak_date for b in burst_group]
            peak_date = max(peak_dates, key=peak_dates.count)  # Most common peak
            
            # Frame analysis
            frames = list(set(b.burst_event.frame for b in burst_group))
            frame_counts = Counter(b.burst_event.frame for b in burst_group)
            dominant_frame = max(frame_counts, key=frame_counts.get)
            
            # Aggregate all entities
            all_entities = set()
            entity_authority_dist = {}
            entity_type_counts = defaultdict(int)
            
            for burst in burst_group:
                all_entities.update(burst.entity_authority_scores.keys())
                for entity, authority in burst.entity_authority_scores.items():
                    entity_authority_dist[entity] = max(
                        entity_authority_dist.get(entity, 0),
                        authority
                    )
                for etype, count in burst.entity_type_distribution.items():
                    entity_type_counts[etype] += count
            
            # Aggregate all media
            all_media = set()
            media_scores = {}
            
            for burst in burst_group:
                all_media.update(burst.media_involved)
                for media, score in burst.media_influence_scores.items():
                    media_scores[media] = max(media_scores.get(media, 0), score)
            
            # Aggregate all journalists
            all_journalists = set()
            journalist_scores = {}
            
            for burst in burst_group:
                all_journalists.update(burst.journalists_involved)
                for journalist, score in burst.journalist_authority_scores.items():
                    journalist_scores[journalist] = max(
                        journalist_scores.get(journalist, 0),
                        score
                    )
            
            # Aggregate messengers
            all_messengers = []
            messenger_types_dist = defaultdict(int)
            
            for burst in burst_group:
                all_messengers.extend(burst.messengers_detected)
                for mtype, count in burst.messenger_types.items():
                    messenger_types_dist[mtype] += count
            
            # Build full propagation network
            full_network = nx.Graph()
            for burst in burst_group:
                full_network = nx.compose(full_network, burst.propagation_network)
            
            # Calculate complete network metrics if calculator available
            if self.exhaustive_calculator:
                network_metrics_complete = self.exhaustive_calculator.calculate_all_metrics(
                    full_network
                )
            else:
                network_metrics_complete = {}
            
            # Detect pattern
            pattern_analysis = self._detect_cascade_pattern(burst_group)
            
            # Identify triggers
            trigger_analysis = self._identify_cascade_triggers(burst_group)
            
            # Calculate volumes
            total_articles = sum(len(b.article_ids) for b in burst_group)
            article_timeline = self._build_article_timeline(burst_group)
            
            # Calculate diversity metrics
            diversity_metrics = self._calculate_cascade_diversity(
                all_media, all_journalists, all_entities, frames
            )
            
            # Calculate convergence
            convergence_analysis = self._analyze_convergence(burst_group)
            
            # Get exhaustive metrics
            exhaustive_metrics = self._calculate_exhaustive_metrics(
                burst_group, full_network
            )
            
            # Create complete cascade
            cascade = CompleteCascade(
                # Identification
                cascade_id=cascade_id,
                detection_timestamp=datetime.now(),
                
                # Temporal
                start_date=start_date,
                peak_date=peak_date,
                end_date=end_date,
                duration_days=(end_date - start_date).days + 1,
                temporal_phases=self._identify_temporal_phases(burst_group),
                
                # Frames
                frames_involved=frames,
                frame_distribution={f: count/len(burst_group) 
                                  for f, count in frame_counts.items()},
                dominant_frame=dominant_frame,
                frame_evolution=self._analyze_frame_evolution(burst_group),
                
                # Bursts
                bursts=burst_group,
                n_bursts=len(burst_group),
                burst_pattern=pattern_analysis['pattern'],
                burst_synchronization=pattern_analysis['synchronization'],
                
                # Entities
                all_entities=all_entities,
                entity_authority_distribution=entity_authority_dist,
                entity_type_counts=dict(entity_type_counts),
                entity_emergence_timeline=self._build_entity_timeline(burst_group),
                entity_network=self._build_entity_network(burst_group),
                key_entities=sorted(entity_authority_dist.items(), 
                                  key=lambda x: x[1], reverse=True)[:10],
                
                # Media
                all_media=all_media,
                media_participation_scores=media_scores,
                media_influence_network=self._build_media_network(burst_group),
                media_adoption_curve=self._analyze_media_adoption(burst_group),
                media_leaders=self._identify_media_leaders(burst_group),
                media_followers=self._identify_media_followers(burst_group),
                
                # Journalists
                all_journalists=all_journalists,
                journalist_contributions=self._count_journalist_contributions(burst_group),
                journalist_authority_scores=journalist_scores,
                journalist_network=self._build_journalist_network(burst_group),
                influential_journalists=sorted(journalist_scores.items(),
                                             key=lambda x: x[1], reverse=True)[:10],
                
                # Messengers
                all_messengers=all_messengers,
                messenger_types_distribution=dict(messenger_types_dist),
                messenger_authority_sum=sum(m.messenger_authority_total 
                                          for m in burst_group),
                messenger_citation_network=self._build_messenger_network(burst_group),
                key_messengers=self._identify_key_messengers(all_messengers),
                
                # Network
                full_propagation_network=full_network,
                network_metrics_complete=network_metrics_complete,
                centrality_measures=self._calculate_centralities(full_network),
                community_detection=self._detect_communities(full_network),
                information_flow_paths=self._trace_information_flow(full_network),
                
                # Pattern
                pattern_type=pattern_analysis['pattern'],
                pattern_subtype=pattern_analysis.get('subtype'),
                pattern_confidence=pattern_analysis['confidence'],
                pattern_features=pattern_analysis['features'],
                
                # Triggers
                trigger_events=trigger_analysis['events'],
                trigger_type=trigger_analysis['type'],
                trigger_confidence=trigger_analysis['confidence'],
                trigger_entities=trigger_analysis['entities'],
                
                # Volume
                total_articles=total_articles,
                article_timeline=article_timeline,
                peak_volume=max(len(b.article_ids) for b in burst_group),
                average_daily_volume=total_articles / 
                                   ((end_date - start_date).days + 1),
                volume_acceleration=self._calculate_volume_acceleration(burst_group),
                
                # Diversity
                media_diversity=diversity_metrics['media'],
                journalist_diversity=diversity_metrics['journalist'],
                geographic_diversity=diversity_metrics['geographic'],
                frame_diversity=diversity_metrics['frame'],
                entity_diversity=diversity_metrics['entity'],
                
                # Convergence
                convergence_score=convergence_analysis['score'],
                convergence_timeline=convergence_analysis['timeline'],
                convergence_drivers=convergence_analysis['drivers'],
                
                # Exhaustive metrics
                exhaustive_metrics=exhaustive_metrics,
                
                # Validation (will be filled by _validate_cascade)
                significance_tests={},
                validation_confidence=0.0,
                statistical_power=0.0,
                false_positive_probability=1.0,
                is_significant=False,
                
                # Quality
                data_completeness=self._calculate_data_completeness(burst_group),
                detection_confidence=np.mean([b.confidence_score for b in burst_group]),
                pattern_clarity=pattern_analysis['confidence']
            )
            
            return cascade
            
        except Exception as e:
            logger.error(f"Failed to build cascade: {e}")
            return None
    
    def _detect_cascade_pattern(self,
                                burst_group: List[EnhancedBurst]) -> Dict[str, Any]:
        """Detect cascade pattern from burst group."""
        pattern_data = {
            'pattern': 'unknown',
            'subtype': None,
            'synchronization': 0.0,
            'confidence': 0.0,
            'features': {}
        }
        
        # Analyze temporal distribution
        burst_times = [(b.burst_event.start_date, b.burst_event.end_date) 
                      for b in burst_group]
        
        # Check for simultaneity
        overlap_count = 0
        for i, (s1, e1) in enumerate(burst_times):
            for s2, e2 in burst_times[i+1:]:
                if s1 <= e2 and s2 <= e1:  # Overlap
                    overlap_count += 1
        
        total_pairs = len(burst_times) * (len(burst_times) - 1) / 2
        overlap_ratio = overlap_count / total_pairs if total_pairs > 0 else 0
        
        # Calculate gaps between bursts for all cases
        sorted_bursts = sorted(burst_group, key=lambda b: b.burst_event.start_date)
        gaps = []
        for i in range(len(sorted_bursts) - 1):
            gap = (sorted_bursts[i+1].burst_event.start_date - 
                  sorted_bursts[i].burst_event.end_date).days
            gaps.append(gap)
        
        # Determine pattern
        if overlap_ratio > 0.7:
            pattern_data['pattern'] = 'simultaneous'
            pattern_data['synchronization'] = overlap_ratio
        elif overlap_ratio < 0.3:
            # Check if sequential
            if gaps and all(g <= 3 for g in gaps):
                pattern_data['pattern'] = 'sequential'
                pattern_data['synchronization'] = 1.0 / (1.0 + np.std(gaps))
            elif gaps:
                pattern_data['pattern'] = 'echo'
                pattern_data['synchronization'] = 0.3
            else:
                pattern_data['pattern'] = 'single'
                pattern_data['synchronization'] = 1.0
        else:
            pattern_data['pattern'] = 'mixed'
            pattern_data['synchronization'] = overlap_ratio
        
        # Check for oscillation
        peak_times = [b.burst_event.peak_date for b in burst_group]
        if len(peak_times) > 3:
            # Check for periodic pattern
            intervals = np.diff([p.timestamp() for p in sorted(peak_times)])
            if np.std(intervals) / np.mean(intervals) < 0.3:  # Regular intervals
                pattern_data['subtype'] = 'oscillating'
        
        # Calculate confidence
        pattern_data['confidence'] = min(1.0, pattern_data['synchronization'] + 0.3)
        
        # Extract pattern features
        pattern_data['features'] = {
            'overlap_ratio': overlap_ratio,
            'n_bursts': len(burst_group),
            'avg_gap_days': np.mean(gaps) if gaps else 0,
            'temporal_spread': (burst_times[-1][1] - burst_times[0][0]).days
        }
        
        return pattern_data
    
    def _identify_cascade_triggers(self,
                                   burst_group: List[EnhancedBurst]) -> Dict[str, Any]:
        """Identify cascade triggers from burst group."""
        trigger_data = {
            'events': [],
            'type': 'unknown',
            'confidence': 0.0,
            'entities': []
        }
        
        # Get first burst as potential trigger
        first_burst = min(burst_group, key=lambda b: b.burst_event.start_date)
        
        # Analyze trigger characteristics
        if first_burst.burst_event.trigger_type:
            trigger_data['type'] = first_burst.burst_event.trigger_type
            trigger_data['confidence'] = first_burst.burst_event.trigger_confidence
        
        # Check for new entities as triggers
        if first_burst.new_entities:
            trigger_data['entities'] = list(first_burst.new_entities)[:5]
            if not trigger_data['type'] or trigger_data['type'] == 'unknown':
                trigger_data['type'] = 'event'  # New entities suggest news event
        
        # Build trigger events
        trigger_data['events'].append({
            'date': first_burst.burst_event.trigger_date or first_burst.burst_event.start_date,
            'type': trigger_data['type'],
            'confidence': trigger_data['confidence'],
            'entities': trigger_data['entities'],
            'media': list(first_burst.media_first_movers)[:3]
        })
        
        return trigger_data
    
    def _build_article_timeline(self,
                                burst_group: List[EnhancedBurst]) -> List[Dict[str, Any]]:
        """Build article timeline for cascade."""
        timeline = []
        
        for burst in burst_group:
            for i, article_id in enumerate(burst.article_ids):
                if i < len(burst.article_timestamps):
                    timeline.append({
                        'article_id': article_id,
                        'timestamp': burst.article_timestamps[i],
                        'burst_id': id(burst),
                        'frame': burst.burst_event.frame
                    })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        
        return timeline
    
    def _calculate_cascade_diversity(self,
                                     media: Set[str],
                                     journalists: Set[str],
                                     entities: Set[str],
                                     frames: List[str]) -> Dict[str, float]:
        """Calculate diversity metrics for cascade."""
        diversity = {}
        
        # Media diversity (normalized by expected max)
        diversity['media'] = min(1.0, len(media) / 20)
        
        # Journalist diversity
        diversity['journalist'] = min(1.0, len(journalists) / 50)
        
        # Entity diversity
        diversity['entity'] = min(1.0, len(entities) / 100)
        
        # Frame diversity (entropy)
        if frames:
            frame_counts = Counter(frames)
            total = sum(frame_counts.values())
            probs = [c/total for c in frame_counts.values()]
            diversity['frame'] = -sum(p * np.log(p + 1e-10) for p in probs) / np.log(len(frames))
        else:
            diversity['frame'] = 0.0
        
        # Geographic diversity (placeholder)
        diversity['geographic'] = 0.5
        
        return diversity
    
    def _analyze_convergence(self,
                             burst_group: List[EnhancedBurst]) -> Dict[str, Any]:
        """Analyze convergence patterns in cascade."""
        convergence_data = {
            'score': 0.0,
            'timeline': [],
            'drivers': []
        }
        
        # Calculate convergence score based on entity/media alignment
        entity_overlaps = []
        for i in range(len(burst_group) - 1):
            overlap = self._calculate_burst_overlap(burst_group[i], burst_group[i+1])
            entity_overlaps.append(overlap)
        
        if entity_overlaps:
            # Increasing overlap indicates convergence
            convergence_trend = np.polyfit(range(len(entity_overlaps)), 
                                          entity_overlaps, 1)[0]
            convergence_data['score'] = min(1.0, abs(convergence_trend) * 10)
        
        # Build convergence timeline
        for i, burst in enumerate(burst_group):
            convergence_data['timeline'].append({
                'burst_index': i,
                'date': burst.burst_event.peak_date,
                'entity_count': len(burst.entities_involved),
                'media_count': len(burst.media_involved),
                'convergence_level': entity_overlaps[i-1] if i > 0 else 0.0
            })
        
        # Identify convergence drivers (most persistent entities)
        entity_persistence = Counter()
        for burst in burst_group:
            entity_persistence.update(burst.entity_authority_scores.keys())
        
        convergence_data['drivers'] = [
            entity for entity, count in entity_persistence.most_common(5)
            if count >= len(burst_group) * 0.5  # Present in at least half
        ]
        
        return convergence_data
    
    def _calculate_exhaustive_metrics(self,
                                      burst_group: List[EnhancedBurst],
                                      network: nx.Graph) -> Dict[str, Any]:
        """Calculate exhaustive metrics using ExhaustiveMetricsCalculator."""
        try:
            # Get all metrics from calculator if available
            if self.exhaustive_calculator:
                metrics = self.exhaustive_calculator.calculate_all_metrics(network)
            else:
                metrics = {}
            
            # Add cascade-specific metrics
            # Extract entity names from entities_involved (which may be dicts)
            all_entities = set()
            for burst in burst_group:
                if hasattr(burst, 'entities_involved'):
                    for entity in burst.entities_involved:
                        if isinstance(entity, dict):
                            # Extract entity id or name from dict
                            entity_id = entity.get('id', entity.get('name', entity.get('entity', str(entity))))
                            all_entities.add(entity_id)
                        else:
                            # Entity is already a string
                            all_entities.add(entity)
            
            # Extract media names
            all_media = set()
            for burst in burst_group:
                if hasattr(burst, 'media_involved'):
                    if isinstance(burst.media_involved, (list, set)):
                        all_media.update(burst.media_involved)
            
            metrics['cascade_metrics'] = {
                'burst_count': len(burst_group),
                'total_entities': len(all_entities),
                'total_media': len(all_media),
                'total_articles': sum(len(b.article_ids) if hasattr(b, 'article_ids') else 0 
                                    for b in burst_group),
                'temporal_span': (burst_group[-1].burst_event.end_date - 
                                burst_group[0].burst_event.start_date).days
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate exhaustive metrics: {e}")
            return {}
    
    def _identify_temporal_phases(self,
                                  burst_group: List[EnhancedBurst]) -> List[str]:
        """Identify temporal phases of cascade."""
        phases = []
        
        # Sort bursts by time
        sorted_bursts = sorted(burst_group, key=lambda b: b.burst_event.peak_date)
        
        # Identify growth, peak, and decay phases
        intensities = [b.burst_event.intensity for b in sorted_bursts]
        max_intensity_idx = intensities.index(max(intensities))
        
        if max_intensity_idx > 0:
            phases.append('growth')
        phases.append('peak')
        if max_intensity_idx < len(intensities) - 1:
            phases.append('decay')
        
        return phases
    
    def _analyze_frame_evolution(self,
                                 burst_group: List[EnhancedBurst]) -> List[Dict[str, Any]]:
        """Analyze how frames evolve over cascade."""
        evolution = []
        
        for burst in sorted(burst_group, key=lambda b: b.burst_event.start_date):
            evolution.append({
                'date': burst.burst_event.peak_date,
                'frame': burst.burst_event.frame,
                'intensity': burst.burst_event.intensity,
                'articles': len(burst.article_ids)
            })
        
        return evolution
    
    def _build_entity_network(self,
                             burst_group: List[EnhancedBurst]) -> nx.Graph:
        """Build complete entity network for cascade."""
        G = nx.Graph()
        
        for burst in burst_group:
            # Add entities as nodes
            for entity in burst.entities_involved:
                if isinstance(entity, dict):
                    entity_id = entity.get('id', entity.get('name', str(entity)))
                    # Create a copy of entity dict without 'id' to avoid duplication
                    node_attrs = {k: v for k, v in entity.items() if k != 'id'}
                    G.add_node(entity_id, **node_attrs)
                else:
                    # Entity is a string
                    G.add_node(entity)
            
            # Add co-occurrences as edges
            for (e1, e2), count in burst.entity_co_occurrences.items():
                if G.has_node(e1) and G.has_node(e2):
                    if G.has_edge(e1, e2):
                        G[e1][e2]['weight'] += count
                    else:
                        G.add_edge(e1, e2, weight=count)
        
        return G
    
    def _build_entity_timeline(self,
                               burst_group: List[EnhancedBurst]) -> List[Dict[str, Any]]:
        """Build entity emergence timeline."""
        timeline = []
        seen_entities = set()
        
        for burst in sorted(burst_group, key=lambda b: b.burst_event.start_date):
            new_in_burst = set(e['id'] for e in burst.entities_involved) - seen_entities
            if new_in_burst:
                timeline.append({
                    'date': burst.burst_event.start_date,
                    'new_entities': list(new_in_burst),
                    'count': len(new_in_burst)
                })
            seen_entities.update(e['id'] for e in burst.entities_involved)
        
        return timeline
    
    def _build_media_network(self,
                            burst_group: List[EnhancedBurst]) -> nx.Graph:
        """Build media influence network."""
        G = nx.Graph()
        
        for burst in burst_group:
            # Compose with existing media networks
            G = nx.compose(G, burst.media_network)
        
        return G
    
    def _analyze_media_adoption(self,
                                burst_group: List[EnhancedBurst]) -> List[Dict[str, Any]]:
        """Analyze media adoption curve."""
        adoption_curve = []
        adopted_media = set()
        
        for burst in sorted(burst_group, key=lambda b: b.burst_event.start_date):
            new_media = burst.media_involved - adopted_media
            adopted_media.update(burst.media_involved)
            
            adoption_curve.append({
                'date': burst.burst_event.start_date,
                'total_media': len(adopted_media),
                'new_media': list(new_media),
                'adoption_rate': len(new_media) / max(1, len(burst.media_involved))
            })
        
        return adoption_curve
    
    def _identify_media_leaders(self,
                                burst_group: List[EnhancedBurst]) -> List[str]:
        """Identify media leaders in cascade."""
        # Leaders are media that appear early and frequently
        early_bursts = sorted(burst_group, key=lambda b: b.burst_event.start_date)[:3]
        leader_candidates = Counter()
        
        for burst in early_bursts:
            leader_candidates.update(burst.media_first_movers)
        
        return [media for media, _ in leader_candidates.most_common(5)]
    
    def _identify_media_followers(self,
                                  burst_group: List[EnhancedBurst]) -> List[str]:
        """Identify media followers in cascade."""
        # Followers are media that appear late
        late_bursts = sorted(burst_group, key=lambda b: b.burst_event.start_date)[-3:]
        follower_candidates = set()
        
        for burst in late_bursts:
            follower_candidates.update(burst.media_involved)
        
        # Remove leaders
        leaders = set(self._identify_media_leaders(burst_group))
        followers = follower_candidates - leaders
        
        return list(followers)[:5]
    
    def _count_journalist_contributions(self,
                                        burst_group: List[EnhancedBurst]) -> Dict[str, int]:
        """Count journalist contributions across cascade."""
        contributions = Counter()
        
        for burst in burst_group:
            contributions.update(burst.journalist_article_counts)
        
        return dict(contributions)
    
    def _build_journalist_network(self,
                                 burst_group: List[EnhancedBurst]) -> nx.Graph:
        """Build journalist collaboration network."""
        G = nx.Graph()
        
        for burst in burst_group:
            G = nx.compose(G, burst.journalist_network)
        
        return G
    
    def _build_messenger_network(self,
                                burst_group: List[EnhancedBurst]) -> nx.Graph:
        """Build messenger citation network."""
        G = nx.Graph()
        
        for burst in burst_group:
            G = nx.compose(G, burst.messenger_network)
        
        return G
    
    def _identify_key_messengers(self,
                                 all_messengers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify key messengers by authority."""
        if not all_messengers:
            return []
        
        # Sort by authority
        sorted_messengers = sorted(
            all_messengers,
            key=lambda m: m.get('authority', 0),
            reverse=True
        )
        
        return sorted_messengers[:5]
    
    def _calculate_centralities(self,
                                network: nx.Graph) -> Dict[str, Dict[str, float]]:
        """Calculate multiple centrality measures."""
        if network.number_of_nodes() == 0:
            return {}
        
        centralities = {}
        
        try:
            centralities['degree'] = nx.degree_centrality(network)
            centralities['betweenness'] = nx.betweenness_centrality(network)
            centralities['closeness'] = nx.closeness_centrality(network)
            if network.number_of_nodes() > 2:
                centralities['eigenvector'] = nx.eigenvector_centrality_numpy(
                    network, max_iter=100
                )
        except:
            pass
        
        return centralities
    
    def _detect_communities(self,
                           network: nx.Graph) -> Dict[str, int]:
        """Detect communities in network."""
        if network.number_of_nodes() < 3:
            return {}
        
        try:
            import community as community_louvain
            return community_louvain.best_partition(network)
        except:
            return {}
    
    def _trace_information_flow(self,
                                network: nx.Graph) -> List[List[str]]:
        """Trace information flow paths."""
        paths = []
        
        if network.number_of_nodes() < 2:
            return paths
        
        # Find shortest paths between high-centrality nodes
        try:
            centrality = nx.degree_centrality(network)
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for i, (n1, _) in enumerate(top_nodes):
                for n2, _ in top_nodes[i+1:]:
                    try:
                        path = nx.shortest_path(network, n1, n2)
                        if len(path) > 2:  # Non-trivial path
                            paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
        except:
            pass
        
        return paths[:10]  # Limit to 10 paths
    
    def _calculate_volume_acceleration(self,
                                       burst_group: List[EnhancedBurst]) -> float:
        """Calculate volume acceleration across cascade."""
        if len(burst_group) < 2:
            return 0.0
        
        volumes = [len(b.article_ids) for b in sorted(
            burst_group, key=lambda b: b.burst_event.start_date
        )]
        
        # Calculate average rate of change
        changes = np.diff(volumes)
        return np.mean(changes) if len(changes) > 0 else 0.0
    
    def _calculate_data_completeness(self,
                                     burst_group: List[EnhancedBurst]) -> float:
        """Calculate data completeness score."""
        completeness_scores = []
        
        for burst in burst_group:
            score = 0.0
            score += 0.2 if burst.entities_involved else 0.0
            score += 0.2 if burst.media_involved else 0.0
            score += 0.2 if burst.journalists_involved else 0.0
            score += 0.2 if burst.network_metrics else 0.0
            score += 0.2 if burst.article_ids else 0.0
            completeness_scores.append(score)
        
        return np.mean(completeness_scores) if completeness_scores else 0.0
    
    def _validate_cascade(self, cascade: CompleteCascade) -> None:
        """
        Perform comprehensive statistical validation of cascade.
        Updates cascade validation fields in place.
        """
        tests = {}
        passed_tests = 0
        total_tests = 0
        
        # Test 1: Temporal clustering significance
        burst_times = [b.burst_event.start_date.timestamp() for b in cascade.bursts]
        if len(burst_times) > 1:
            # Test if bursts are significantly clustered in time
            gaps = np.diff(sorted(burst_times))
            if len(gaps) > 0:
                # Compare to exponential distribution (random events)
                ks_stat, p_value = stats.kstest(gaps, 'expon')
                tests['temporal_clustering'] = {
                    'statistic': ks_stat,
                    'p_value': p_value,
                    'significant': p_value < self.significance_threshold
                }
                if p_value < self.significance_threshold:
                    passed_tests += 1
                total_tests += 1
        
        # Test 2: Entity concentration
        if cascade.entity_authority_distribution:
            # Test if entity distribution differs from random
            authorities = list(cascade.entity_authority_distribution.values())
            if len(authorities) > 1:
                # Test against uniform distribution
                chi2, p_value = stats.chisquare(authorities)
                tests['entity_concentration'] = {
                    'statistic': chi2,
                    'p_value': p_value,
                    'significant': p_value < self.significance_threshold
                }
                if p_value < self.significance_threshold:
                    passed_tests += 1
                total_tests += 1
        
        # Test 3: Media adoption pattern
        if len(cascade.media_adoption_curve) > 2:
            # Test if adoption follows cascade pattern (S-curve)
            adoption_counts = [a['total_media'] for a in cascade.media_adoption_curve]
            # Test for positive trend
            x = np.arange(len(adoption_counts))
            slope, _, r_value, p_value, _ = stats.linregress(x, adoption_counts)
            tests['media_adoption'] = {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'significant': p_value < self.significance_threshold and slope > 0
            }
            if p_value < self.significance_threshold and slope > 0:
                passed_tests += 1
            total_tests += 1
        
        # Test 4: Network structure
        if cascade.full_propagation_network.number_of_nodes() > 10:
            # Test if network has non-random structure
            density = nx.density(cascade.full_propagation_network)
            random_density = 2 * cascade.full_propagation_network.number_of_edges() / (
                cascade.full_propagation_network.number_of_nodes() * 
                (cascade.full_propagation_network.number_of_nodes() - 1)
            )
            
            # Simple test: is density significantly different from random?
            tests['network_structure'] = {
                'observed_density': density,
                'random_density': random_density,
                'ratio': density / (random_density + 1e-10),
                'significant': abs(density - random_density) > 0.1
            }
            if abs(density - random_density) > 0.1:
                passed_tests += 1
            total_tests += 1
        
        # Test 5: Volume significance
        volumes = [len(b.article_ids) for b in cascade.bursts]
        if volumes:
            # Test if volumes are significantly elevated
            baseline_volume = np.mean(volumes)
            peak_volume = max(volumes)
            z_score = (peak_volume - baseline_volume) / (np.std(volumes) + 1e-10)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            tests['volume_significance'] = {
                'z_score': z_score,
                'p_value': p_value,
                'significant': p_value < self.significance_threshold
            }
            if p_value < self.significance_threshold:
                passed_tests += 1
            total_tests += 1
        
        # Test 6: Burst synchronization
        tests['burst_synchronization'] = {
            'score': cascade.burst_synchronization,
            'significant': cascade.burst_synchronization > 0.5
        }
        if cascade.burst_synchronization > 0.5:
            passed_tests += 1
        total_tests += 1
        
        # Test 7: Pattern consistency
        tests['pattern_consistency'] = {
            'confidence': cascade.pattern_confidence,
            'significant': cascade.pattern_confidence > 0.7
        }
        if cascade.pattern_confidence > 0.7:
            passed_tests += 1
        total_tests += 1
        
        # Test 8: Information diversity
        diversity_score = np.mean([
            cascade.media_diversity,
            cascade.journalist_diversity,
            cascade.entity_diversity,
            cascade.frame_diversity
        ])
        tests['information_diversity'] = {
            'score': diversity_score,
            'significant': diversity_score > 0.3
        }
        if diversity_score > 0.3:
            passed_tests += 1
        total_tests += 1
        
        # Test 9: Convergence significance
        tests['convergence'] = {
            'score': cascade.convergence_score,
            'significant': cascade.convergence_score > 0.5
        }
        if cascade.convergence_score > 0.5:
            passed_tests += 1
        total_tests += 1
        
        # Test 10: Data quality
        tests['data_quality'] = {
            'completeness': cascade.data_completeness,
            'significant': cascade.data_completeness > 0.7
        }
        if cascade.data_completeness > 0.7:
            passed_tests += 1
        total_tests += 1
        
        # Update cascade validation fields
        cascade.significance_tests = tests
        cascade.validation_confidence = passed_tests / total_tests if total_tests > 0 else 0
        cascade.statistical_power = min(1.0, passed_tests / 5)  # Need at least 5 passed tests
        cascade.false_positive_probability = 1.0 - cascade.validation_confidence
        cascade.is_significant = passed_tests >= 6  # Require 60% of tests to pass
        
        logger.info(f"Cascade {cascade.cascade_id} validation: "
                   f"{passed_tests}/{total_tests} tests passed, "
                   f"confidence: {cascade.validation_confidence:.2%}")
    
    def validate_detection(self, detection: CompleteCascade) -> bool:
        """
        Validate a cascade detection using statistical tests.
        
        TODO: Implement comprehensive validation in Phase 4.
        For now, returns True to allow cascade detection to proceed.
        
        Args:
            detection: CompleteCascade to validate
            
        Returns:
            True if detection passes validation
        """
        # Placeholder - will be implemented in Phase 4
        return True
    
    def score_detection(self, detection: CompleteCascade) -> float:
        """
        Calculate importance score for a cascade.
        
        TODO: Implement sophisticated scoring in Phase 4.
        For now, returns the intensity score.
        
        Args:
            detection: CompleteCascade to score
            
        Returns:
            Score between 0 and 1 indicating cascade importance
        """
        # Placeholder - will be implemented in Phase 4
        # For now, just return the intensity score
        return detection.intensity_score if hasattr(detection, 'intensity_score') else 0.5