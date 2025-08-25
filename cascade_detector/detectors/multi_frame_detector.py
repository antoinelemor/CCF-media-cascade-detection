"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
multi_frame_detector.py

MAIN OBJECTIVE:
---------------
This script analyzes cascade patterns across multiple frames simultaneously, detecting frame
convergence, cross-frame influence propagation, and paradigm shift indicators in media coverage.

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
- tqdm

MAIN FEATURES:
--------------
1) Frame convergence and divergence pattern detection
2) Cross-frame influence propagation analysis
3) Frame competition and dominance measurement
4) Paradigm shift indicator identification
5) Multi-frame cascade synchronization tracking

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
from scipy import stats, spatial
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
import logging
import warnings
from tqdm import tqdm

# Import base components
from cascade_detector.detectors.base_detector import BaseDetector, DetectionContext
from cascade_detector.detectors.cascade_detector import (
    CascadeDetector, CompleteCascade, EnhancedBurst
)
from cascade_detector.core.config import DetectorConfig

# Import metrics
from cascade_detector.metrics.convergence_metrics import ConvergenceMetrics
from cascade_detector.metrics.diversity_metrics import DiversityMetrics

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class FrameInteraction:
    """
    Represents interaction between two frames during a time period.
    """
    frame1: str
    frame2: str
    period: Tuple[datetime, datetime]
    
    # Interaction metrics
    entity_overlap: float  # Jaccard similarity of entities
    media_overlap: float  # Jaccard similarity of media
    temporal_correlation: float  # Correlation of activity patterns
    semantic_similarity: float  # Content similarity
    
    # Influence metrics
    lead_lag_score: float  # Which frame leads (-1 to 1)
    causality_score: float  # Granger causality measure
    information_transfer: float  # Mutual information
    
    # Competition metrics
    competition_intensity: float  # How much frames compete
    dominance_ratio: float  # Relative dominance
    substitution_rate: float  # Rate of frame substitution
    
    # Network metrics
    shared_network: nx.Graph  # Shared propagation network
    bridge_nodes: List[str]  # Nodes connecting frames
    
    def get_interaction_strength(self) -> float:
        """Calculate overall interaction strength."""
        return np.mean([
            self.entity_overlap,
            self.media_overlap,
            abs(self.temporal_correlation),
            self.semantic_similarity,
            self.information_transfer
        ])
    
    def get_relationship_type(self) -> str:
        """Determine relationship type between frames."""
        if self.competition_intensity > 0.7:
            return 'competitive'
        elif self.get_interaction_strength() > 0.6:
            if abs(self.lead_lag_score) > 0.5:
                return 'leader-follower'
            else:
                return 'synergistic'
        elif self.get_interaction_strength() < 0.2:
            return 'independent'
        else:
            return 'weakly-coupled'


@dataclass
class MultiFramePattern:
    """
    Detected pattern across multiple frames.
    """
    pattern_id: str
    pattern_type: str  # 'convergence', 'divergence', 'rotation', 'competition', 'emergence'
    
    # Frames involved
    frames: List[str]
    frame_roles: Dict[str, str]  # Frame -> role (leader, follower, competitor, etc.)
    
    # Temporal data
    start_date: datetime
    end_date: datetime
    peak_date: datetime
    duration_days: int
    
    # Pattern characteristics
    strength: float  # Pattern strength (0-1)
    stability: float  # Pattern stability over time
    directionality: float  # Unidirectional vs bidirectional
    
    # Cascades involved
    cascades: List[CompleteCascade]
    cascade_alignment: float  # How well cascades align
    
    # Frame dynamics
    frame_trajectories: Dict[str, List[float]]  # Frame -> intensity over time
    convergence_points: List[datetime]  # When frames converge
    divergence_points: List[datetime]  # When frames diverge
    
    # Influence flow
    influence_network: nx.DiGraph  # Directed influence between frames
    influence_paths: List[List[str]]  # Main influence pathways
    
    # Statistical validation
    significance: float
    confidence: float
    statistical_tests: Dict[str, Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'frames': self.frames,
            'frame_roles': self.frame_roles,
            'period': {
                'start': self.start_date.isoformat(),
                'end': self.end_date.isoformat(),
                'duration_days': self.duration_days
            },
            'metrics': {
                'strength': self.strength,
                'stability': self.stability,
                'directionality': self.directionality,
                'cascade_alignment': self.cascade_alignment,
                'significance': self.significance,
                'confidence': self.confidence
            },
            'cascades': len(self.cascades),
            'influence_paths': self.influence_paths
        }


@dataclass
class ParadigmShiftIndicator:
    """
    Indicators of potential paradigm shift across frames.
    """
    shift_type: str  # 'gradual', 'sudden', 'oscillating', 'cascade'
    
    # Frames involved
    declining_frames: List[str]  # Frames losing dominance
    emerging_frames: List[str]  # Frames gaining dominance
    stable_frames: List[str]  # Frames maintaining position
    
    # Temporal markers
    shift_start: Optional[datetime]
    shift_peak: Optional[datetime]
    shift_end: Optional[datetime]
    
    # Shift metrics
    shift_magnitude: float  # Size of the shift
    shift_velocity: float  # Speed of change
    shift_acceleration: float  # Acceleration of change
    
    # Drivers
    key_entities: List[str]  # Entities driving shift
    key_media: List[str]  # Media driving shift
    key_events: List[Dict[str, Any]]  # Events triggering shift
    
    # Predictive indicators
    early_signals: List[Dict[str, Any]]  # Early warning signals
    tipping_points: List[datetime]  # Critical transition points
    
    # Validation
    confidence: float
    supporting_evidence: List[str]


class MultiFrameDetector(BaseDetector):
    """
    Detects and analyzes cascade patterns across multiple frames.
    
    This detector:
    1. Identifies frame interactions and relationships
    2. Detects multi-frame patterns (convergence, divergence, competition)
    3. Tracks frame dominance and transitions
    4. Identifies paradigm shift indicators
    5. Analyzes cross-frame influence propagation
    
    Fully integrated with CascadeDetector outputs and provides input for:
    - CrossMediaTracker (frame-specific media behavior)
    - SequenceDetector (temporal frame sequences)
    - EchoChamberDetector (frame-based echo chambers)
    - PolarizationDetector (frame-based polarization)
    """
    
    def __init__(self,
                 context: DetectionContext,
                 config: Optional[DetectorConfig] = None,
                 cascade_detector: Optional[CascadeDetector] = None):
        """
        Initialize multi-frame detector.
        
        Args:
            context: Detection context with all indices and metrics
            config: Optional configuration
            cascade_detector: Optional pre-configured cascade detector
        """
        super().__init__(context, config)
        
        # Initialize cascade detector if not provided
        self.cascade_detector = cascade_detector or CascadeDetector(context, config)
        
        # Configuration
        self.min_frame_overlap = 0.1  # Minimum overlap for interaction
        self.min_pattern_strength = 0.5  # Minimum strength for pattern
        self.convergence_threshold = 0.7  # Threshold for convergence
        self.divergence_threshold = 0.3  # Threshold for divergence
        self.paradigm_shift_threshold = 0.6  # Threshold for paradigm shift
        
        # Analysis windows
        self.window_sizes = [7, 14, 30, 60]  # Days
        self.min_cascades_for_pattern = 3
        
        # Cache
        self._frame_interactions_cache = {}
        self._pattern_cache = {}
        
        logger.info("MultiFrameDetector initialized")
        logger.info(f"  - Analyzing {len(context.frames)} frames")
        logger.info(f"  - Window sizes: {self.window_sizes} days")
        logger.info(f"  - Min pattern strength: {self.min_pattern_strength}")
    
    def detect(self, **kwargs) -> Dict[str, Any]:
        """
        Detect multi-frame patterns and interactions.
        
        Args:
            cascades: Optional pre-detected cascades
            window: Optional time window to analyze
            frames: Optional specific frames to analyze
            
        Returns:
            Dictionary containing:
            - frame_interactions: List of FrameInteraction objects
            - multi_frame_patterns: List of MultiFramePattern objects
            - paradigm_indicators: List of ParadigmShiftIndicator objects
            - frame_dominance: Frame dominance timeline
            - cross_frame_network: Cross-frame influence network
        """
        # Get cascades
        cascades = kwargs.get('cascades')
        if cascades is None:
            logger.info("Detecting cascades first...")
            cascades = self.cascade_detector.detect(**kwargs)
        
        if not cascades:
            logger.warning("No cascades found - cannot analyze multi-frame patterns")
            return {
                'frame_interactions': [],
                'multi_frame_patterns': [],
                'paradigm_indicators': [],
                'frame_dominance': {},
                'cross_frame_network': nx.DiGraph()
            }
        
        window = kwargs.get('window', self.context.time_window)
        frames = kwargs.get('frames', self.context.frames)
        
        logger.info("="*80)
        logger.info("MULTI-FRAME DETECTION")
        logger.info("="*80)
        logger.info(f"Analyzing {len(cascades)} cascades across {len(frames)} frames")
        logger.info(f"Period: {window[0].date()} to {window[1].date()}")
        
        # Step 1: Analyze frame interactions
        logger.info("\nStep 1: Analyzing frame interactions...")
        frame_interactions = self._analyze_frame_interactions(cascades, frames, window)
        logger.info(f"  Found {len(frame_interactions)} frame interactions")
        
        # Step 2: Detect multi-frame patterns
        logger.info("\nStep 2: Detecting multi-frame patterns...")
        patterns = self._detect_multi_frame_patterns(cascades, frame_interactions, window)
        logger.info(f"  Found {len(patterns)} multi-frame patterns")
        
        # Step 3: Calculate frame dominance
        logger.info("\nStep 3: Calculating frame dominance timeline...")
        dominance_timeline = self._calculate_frame_dominance(cascades, frames, window)
        logger.info(f"  Tracked dominance for {len(dominance_timeline)} time points")
        
        # Step 4: Detect paradigm shift indicators
        logger.info("\nStep 4: Detecting paradigm shift indicators...")
        paradigm_indicators = self._detect_paradigm_shifts(
            patterns, dominance_timeline, cascades
        )
        logger.info(f"  Found {len(paradigm_indicators)} paradigm shift indicators")
        
        # Step 5: Build cross-frame network
        logger.info("\nStep 5: Building cross-frame influence network...")
        cross_frame_network = self._build_cross_frame_network(
            frame_interactions, patterns
        )
        logger.info(f"  Network has {cross_frame_network.number_of_nodes()} nodes, "
                   f"{cross_frame_network.number_of_edges()} edges")
        
        # Step 6: Validate patterns
        logger.info("\nStep 6: Validating patterns...")
        validated_patterns = []
        for pattern in patterns:
            if self._validate_pattern(pattern):
                validated_patterns.append(pattern)
        logger.info(f"  {len(validated_patterns)} patterns passed validation")
        
        # Compile results
        results = {
            'frame_interactions': frame_interactions,
            'multi_frame_patterns': validated_patterns,
            'paradigm_indicators': paradigm_indicators,
            'frame_dominance': dominance_timeline,
            'cross_frame_network': cross_frame_network,
            'summary': self._generate_summary(
                frame_interactions, validated_patterns, paradigm_indicators
            )
        }
        
        logger.info("\n" + "="*80)
        logger.info("MULTI-FRAME DETECTION COMPLETE")
        logger.info("="*80)
        
        return results
    
    def _analyze_frame_interactions(self,
                                    cascades: List[CompleteCascade],
                                    frames: List[str],
                                    window: Tuple[datetime, datetime]) -> List[FrameInteraction]:
        """Analyze interactions between frames."""
        interactions = []
        
        # Group cascades by frame
        frame_cascades = defaultdict(list)
        for cascade in cascades:
            for frame in cascade.frames_involved:
                if frame in frames:
                    frame_cascades[frame].append(cascade)
        
        # Analyze pairwise frame interactions
        for i, frame1 in enumerate(frames):
            for frame2 in frames[i+1:]:
                if frame1 in frame_cascades and frame2 in frame_cascades:
                    interaction = self._calculate_frame_interaction(
                        frame1, frame2,
                        frame_cascades[frame1],
                        frame_cascades[frame2],
                        window
                    )
                    
                    if interaction and interaction.get_interaction_strength() > self.min_frame_overlap:
                        interactions.append(interaction)
        
        return interactions
    
    def _calculate_frame_interaction(self,
                                     frame1: str,
                                     frame2: str,
                                     cascades1: List[CompleteCascade],
                                     cascades2: List[CompleteCascade],
                                     window: Tuple[datetime, datetime]) -> Optional[FrameInteraction]:
        """Calculate interaction metrics between two frames."""
        try:
            # Collect entities and media from each frame
            entities1 = set()
            entities2 = set()
            media1 = set()
            media2 = set()
            
            for cascade in cascades1:
                entities1.update(cascade.all_entities)
                media1.update(cascade.all_media)
            
            for cascade in cascades2:
                entities2.update(cascade.all_entities)
                media2.update(cascade.all_media)
            
            # Calculate overlaps
            entity_overlap = self._jaccard_similarity(entities1, entities2)
            media_overlap = self._jaccard_similarity(media1, media2)
            
            # Calculate temporal correlation
            temporal_correlation = self._calculate_temporal_correlation(
                cascades1, cascades2, window
            )
            
            # Calculate semantic similarity (simplified)
            semantic_similarity = self._calculate_semantic_similarity(
                cascades1, cascades2
            )
            
            # Calculate lead-lag relationship
            lead_lag_score = self._calculate_lead_lag(cascades1, cascades2)
            
            # Calculate causality (simplified Granger causality)
            causality_score = self._calculate_causality(cascades1, cascades2, window)
            
            # Calculate information transfer
            information_transfer = self._calculate_information_transfer(
                cascades1, cascades2
            )
            
            # Calculate competition metrics
            competition_intensity = self._calculate_competition(
                cascades1, cascades2, window
            )
            
            # Calculate dominance
            dominance_ratio = len(cascades1) / (len(cascades1) + len(cascades2))
            
            # Calculate substitution rate
            substitution_rate = self._calculate_substitution_rate(
                cascades1, cascades2, window
            )
            
            # Build shared network
            shared_network = self._build_shared_network(cascades1, cascades2)
            
            # Find bridge nodes
            bridge_nodes = self._find_bridge_nodes(
                shared_network, entities1, entities2, media1, media2
            )
            
            return FrameInteraction(
                frame1=frame1,
                frame2=frame2,
                period=window,
                entity_overlap=entity_overlap,
                media_overlap=media_overlap,
                temporal_correlation=temporal_correlation,
                semantic_similarity=semantic_similarity,
                lead_lag_score=lead_lag_score,
                causality_score=causality_score,
                information_transfer=information_transfer,
                competition_intensity=competition_intensity,
                dominance_ratio=dominance_ratio,
                substitution_rate=substitution_rate,
                shared_network=shared_network,
                bridge_nodes=bridge_nodes
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate frame interaction: {e}")
            return None
    
    def _jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _calculate_temporal_correlation(self,
                                        cascades1: List[CompleteCascade],
                                        cascades2: List[CompleteCascade],
                                        window: Tuple[datetime, datetime]) -> float:
        """Calculate temporal correlation between frame activities."""
        # Create daily activity series
        days = pd.date_range(window[0], window[1], freq='D')
        activity1 = np.zeros(len(days))
        activity2 = np.zeros(len(days))
        
        for cascade in cascades1:
            for i, day in enumerate(days):
                if cascade.start_date <= day <= cascade.end_date:
                    activity1[i] += cascade.total_articles
        
        for cascade in cascades2:
            for i, day in enumerate(days):
                if cascade.start_date <= day <= cascade.end_date:
                    activity2[i] += cascade.total_articles
        
        # Calculate correlation
        if np.std(activity1) > 0 and np.std(activity2) > 0:
            correlation = np.corrcoef(activity1, activity2)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        return 0.0
    
    def _calculate_semantic_similarity(self,
                                       cascades1: List[CompleteCascade],
                                       cascades2: List[CompleteCascade]) -> float:
        """Calculate semantic similarity between frames (simplified)."""
        # Use entity overlap as proxy for semantic similarity
        all_entities1 = set()
        all_entities2 = set()
        
        for cascade in cascades1:
            all_entities1.update(cascade.all_entities)
        
        for cascade in cascades2:
            all_entities2.update(cascade.all_entities)
        
        return self._jaccard_similarity(all_entities1, all_entities2)
    
    def _calculate_lead_lag(self,
                           cascades1: List[CompleteCascade],
                           cascades2: List[CompleteCascade]) -> float:
        """Calculate lead-lag relationship (-1: frame2 leads, 1: frame1 leads)."""
        if not cascades1 or not cascades2:
            return 0.0
        
        # Compare average start times
        avg_start1 = np.mean([c.start_date.timestamp() for c in cascades1])
        avg_start2 = np.mean([c.start_date.timestamp() for c in cascades2])
        
        # Normalize to [-1, 1]
        diff = avg_start1 - avg_start2
        max_diff = max(abs(avg_start1), abs(avg_start2))
        
        if max_diff > 0:
            return np.clip(-diff / max_diff, -1, 1)
        return 0.0
    
    def _calculate_causality(self,
                            cascades1: List[CompleteCascade],
                            cascades2: List[CompleteCascade],
                            window: Tuple[datetime, datetime]) -> float:
        """Calculate simplified causality measure."""
        # Check if cascades in frame1 tend to precede cascades in frame2
        precedence_count = 0
        total_pairs = 0
        
        for c1 in cascades1:
            for c2 in cascades2:
                if c1.end_date < c2.start_date:
                    # Check if they share entities (potential influence)
                    shared = c1.all_entities & c2.all_entities
                    if shared:
                        precedence_count += 1
                total_pairs += 1
        
        return precedence_count / total_pairs if total_pairs > 0 else 0.0
    
    def _calculate_information_transfer(self,
                                        cascades1: List[CompleteCascade],
                                        cascades2: List[CompleteCascade]) -> float:
        """Calculate information transfer between frames."""
        # Use mutual information of entity distributions
        all_entities = set()
        for cascade in cascades1 + cascades2:
            all_entities.update(cascade.all_entities)
        
        if not all_entities:
            return 0.0
        
        # Create entity vectors
        entity_list = list(all_entities)
        vec1 = np.zeros(len(entity_list))
        vec2 = np.zeros(len(entity_list))
        
        for cascade in cascades1:
            for i, entity in enumerate(entity_list):
                if entity in cascade.all_entities:
                    vec1[i] += 1
        
        for cascade in cascades2:
            for i, entity in enumerate(entity_list):
                if entity in cascade.all_entities:
                    vec2[i] += 1
        
        # Normalize
        if vec1.sum() > 0:
            vec1 = vec1 / vec1.sum()
        if vec2.sum() > 0:
            vec2 = vec2 / vec2.sum()
        
        # Calculate mutual information (simplified)
        mi = 0.0
        for i in range(len(entity_list)):
            if vec1[i] > 0 and vec2[i] > 0:
                joint = (vec1[i] + vec2[i]) / 2
                mi += joint * np.log(joint / (vec1[i] * vec2[i]))
        
        return min(1.0, mi)
    
    def _calculate_competition(self,
                               cascades1: List[CompleteCascade],
                               cascades2: List[CompleteCascade],
                               window: Tuple[datetime, datetime]) -> float:
        """Calculate competition intensity between frames."""
        # Frames compete if they occur at similar times but with different entities/media
        temporal_overlap = 0
        content_divergence = 0
        
        for c1 in cascades1:
            for c2 in cascades2:
                # Check temporal overlap
                overlap_start = max(c1.start_date, c2.start_date)
                overlap_end = min(c1.end_date, c2.end_date)
                
                if overlap_start < overlap_end:
                    temporal_overlap += (overlap_end - overlap_start).days
                    
                    # Check content divergence
                    entity_similarity = self._jaccard_similarity(
                        c1.all_entities, c2.all_entities
                    )
                    content_divergence += (1 - entity_similarity)
        
        # Normalize
        max_overlap = (window[1] - window[0]).days * len(cascades1) * len(cascades2)
        if max_overlap > 0:
            competition = (temporal_overlap / max_overlap) * (content_divergence / (len(cascades1) * len(cascades2)))
            return min(1.0, competition)
        return 0.0
    
    def _calculate_substitution_rate(self,
                                     cascades1: List[CompleteCascade],
                                     cascades2: List[CompleteCascade],
                                     window: Tuple[datetime, datetime]) -> float:
        """Calculate rate at which one frame substitutes another."""
        # Check if frame2 cascades tend to end when frame1 cascades begin
        substitutions = 0
        
        for c1 in cascades1:
            for c2 in cascades2:
                # Check if c2 ends around when c1 starts
                time_diff = abs((c2.end_date - c1.start_date).days)
                if time_diff <= 3:  # Within 3 days
                    # Check if they share media (substitution in same outlets)
                    if c1.all_media & c2.all_media:
                        substitutions += 1
        
        max_substitutions = min(len(cascades1), len(cascades2))
        return substitutions / max_substitutions if max_substitutions > 0 else 0.0
    
    def _build_shared_network(self,
                              cascades1: List[CompleteCascade],
                              cascades2: List[CompleteCascade]) -> nx.Graph:
        """Build network of shared elements between frames."""
        G = nx.Graph()
        
        # Add nodes for shared entities
        shared_entities = set()
        for c1 in cascades1:
            for c2 in cascades2:
                shared = c1.all_entities & c2.all_entities
                shared_entities.update(shared)
        
        for entity in shared_entities:
            G.add_node(f"entity_{entity}", type='entity')
        
        # Add nodes for shared media
        shared_media = set()
        for c1 in cascades1:
            for c2 in cascades2:
                shared = c1.all_media & c2.all_media
                shared_media.update(shared)
        
        for media in shared_media:
            G.add_node(f"media_{media}", type='media')
        
        # Add edges based on co-occurrence
        for c1 in cascades1:
            for c2 in cascades2:
                for entity in c1.all_entities & c2.all_entities:
                    for media in c1.all_media & c2.all_media:
                        G.add_edge(f"entity_{entity}", f"media_{media}")
        
        return G
    
    def _find_bridge_nodes(self,
                          network: nx.Graph,
                          entities1: Set,
                          entities2: Set,
                          media1: Set,
                          media2: Set) -> List[str]:
        """Find nodes that bridge between frames."""
        bridges = []
        
        # Entities that appear in both frames
        bridge_entities = entities1 & entities2
        bridges.extend([f"entity_{e}" for e in list(bridge_entities)[:10]])
        
        # Media that cover both frames
        bridge_media = media1 & media2
        bridges.extend([f"media_{m}" for m in list(bridge_media)[:10]])
        
        return bridges
    
    def _detect_multi_frame_patterns(self,
                                     cascades: List[CompleteCascade],
                                     interactions: List[FrameInteraction],
                                     window: Tuple[datetime, datetime]) -> List[MultiFramePattern]:
        """Detect patterns across multiple frames."""
        patterns = []
        
        # Group cascades by overlapping time periods
        cascade_groups = self._group_cascades_temporally(cascades)
        
        for group in cascade_groups:
            if len(group) >= self.min_cascades_for_pattern:
                # Check if multiple frames involved
                frames_in_group = set()
                for cascade in group:
                    frames_in_group.update(cascade.frames_involved)
                
                if len(frames_in_group) >= 2:
                    pattern = self._analyze_pattern(group, frames_in_group, interactions)
                    if pattern and pattern.strength >= self.min_pattern_strength:
                        patterns.append(pattern)
        
        return patterns
    
    def _group_cascades_temporally(self,
                                   cascades: List[CompleteCascade]) -> List[List[CompleteCascade]]:
        """Group cascades that overlap temporally."""
        if not cascades:
            return []
        
        # Sort cascades by start date
        sorted_cascades = sorted(cascades, key=lambda c: c.start_date)
        
        groups = []
        current_group = [sorted_cascades[0]]
        
        for cascade in sorted_cascades[1:]:
            # Check if cascade overlaps with any in current group
            overlaps = False
            for gc in current_group:
                if cascade.start_date <= gc.end_date and cascade.end_date >= gc.start_date:
                    overlaps = True
                    break
            
            if overlaps:
                current_group.append(cascade)
            else:
                if len(current_group) >= self.min_cascades_for_pattern:
                    groups.append(current_group)
                current_group = [cascade]
        
        # Add last group
        if len(current_group) >= self.min_cascades_for_pattern:
            groups.append(current_group)
        
        return groups
    
    def _analyze_pattern(self,
                        cascade_group: List[CompleteCascade],
                        frames: Set[str],
                        interactions: List[FrameInteraction]) -> Optional[MultiFramePattern]:
        """Analyze a group of cascades for multi-frame patterns."""
        try:
            import uuid
            pattern_id = str(uuid.uuid4())[:8]
            
            # Determine temporal boundaries
            start_date = min(c.start_date for c in cascade_group)
            end_date = max(c.end_date for c in cascade_group)
            peak_dates = [c.peak_date for c in cascade_group]
            peak_date = max(peak_dates, key=peak_dates.count)
            
            # Analyze frame trajectories
            frame_trajectories = self._calculate_frame_trajectories(
                cascade_group, frames, start_date, end_date
            )
            
            # Determine pattern type
            pattern_type, pattern_metrics = self._determine_pattern_type(
                frame_trajectories, cascade_group, interactions
            )
            
            # Determine frame roles
            frame_roles = self._determine_frame_roles(
                cascade_group, frames, pattern_type
            )
            
            # Calculate pattern strength
            strength = self._calculate_pattern_strength(
                cascade_group, frame_trajectories, pattern_type
            )
            
            # Calculate stability
            stability = self._calculate_pattern_stability(frame_trajectories)
            
            # Calculate directionality
            directionality = self._calculate_directionality(
                cascade_group, interactions, frames
            )
            
            # Calculate cascade alignment
            cascade_alignment = self._calculate_cascade_alignment(cascade_group)
            
            # Find convergence/divergence points
            convergence_points = self._find_convergence_points(frame_trajectories)
            divergence_points = self._find_divergence_points(frame_trajectories)
            
            # Build influence network
            influence_network = self._build_influence_network(
                cascade_group, frames, interactions
            )
            
            # Find influence paths
            influence_paths = self._find_influence_paths(influence_network)
            
            # Statistical validation
            statistical_tests = self._validate_multi_frame_pattern(
                cascade_group, frame_trajectories, pattern_type
            )
            
            significance = np.mean([
                test['p_value'] < 0.05 
                for test in statistical_tests.values()
                if 'p_value' in test
            ])
            
            confidence = strength * stability * significance
            
            return MultiFramePattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                frames=list(frames),
                frame_roles=frame_roles,
                start_date=start_date,
                end_date=end_date,
                peak_date=peak_date,
                duration_days=(end_date - start_date).days,
                strength=strength,
                stability=stability,
                directionality=directionality,
                cascades=cascade_group,
                cascade_alignment=cascade_alignment,
                frame_trajectories=frame_trajectories,
                convergence_points=convergence_points,
                divergence_points=divergence_points,
                influence_network=influence_network,
                influence_paths=influence_paths,
                significance=significance,
                confidence=confidence,
                statistical_tests=statistical_tests
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze pattern: {e}")
            return None
    
    def _calculate_frame_trajectories(self,
                                      cascade_group: List[CompleteCascade],
                                      frames: Set[str],
                                      start_date: datetime,
                                      end_date: datetime) -> Dict[str, List[float]]:
        """Calculate intensity trajectories for each frame."""
        trajectories = {frame: [] for frame in frames}
        
        # Create daily timeline
        days = pd.date_range(start_date, end_date, freq='D')
        
        for day in days:
            frame_intensities = {frame: 0.0 for frame in frames}
            
            # Sum intensities from active cascades
            for cascade in cascade_group:
                if cascade.start_date <= day <= cascade.end_date:
                    # Distribute intensity among cascade frames
                    cascade_intensity = cascade.total_articles / cascade.duration_days
                    for frame in cascade.frames_involved:
                        if frame in frames:
                            frame_intensities[frame] += cascade_intensity
            
            # Add to trajectories
            for frame in frames:
                trajectories[frame].append(frame_intensities[frame])
        
        return trajectories
    
    def _determine_pattern_type(self,
                                trajectories: Dict[str, List[float]],
                                cascades: List[CompleteCascade],
                                interactions: List[FrameInteraction]) -> Tuple[str, Dict]:
        """Determine the type of multi-frame pattern."""
        # Calculate convergence/divergence metrics
        convergence_score = self._calculate_convergence_score(trajectories)
        divergence_score = 1 - convergence_score
        
        # Check for rotation (frames taking turns)
        rotation_score = self._calculate_rotation_score(trajectories)
        
        # Check for competition
        competition_score = np.mean([
            i.competition_intensity 
            for i in interactions
        ]) if interactions else 0
        
        # Check for emergence (new frame rising)
        emergence_score = self._calculate_emergence_score(trajectories, cascades)
        
        # Determine dominant pattern
        scores = {
            'convergence': convergence_score,
            'divergence': divergence_score,
            'rotation': rotation_score,
            'competition': competition_score,
            'emergence': emergence_score
        }
        
        pattern_type = max(scores, key=scores.get)
        
        return pattern_type, scores
    
    def _calculate_convergence_score(self,
                                     trajectories: Dict[str, List[float]]) -> float:
        """Calculate how much frames converge over time."""
        if len(trajectories) < 2:
            return 0.0
        
        # Calculate pairwise correlations over time
        frame_arrays = list(trajectories.values())
        correlations = []
        
        for i in range(len(frame_arrays)):
            for j in range(i+1, len(frame_arrays)):
                if np.std(frame_arrays[i]) > 0 and np.std(frame_arrays[j]) > 0:
                    corr = np.corrcoef(frame_arrays[i], frame_arrays[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_rotation_score(self,
                                  trajectories: Dict[str, List[float]]) -> float:
        """Calculate if frames show rotation pattern (taking turns)."""
        if len(trajectories) < 2:
            return 0.0
        
        # Check if peaks occur at different times
        peak_times = []
        for frame, trajectory in trajectories.items():
            if max(trajectory) > 0:
                peak_time = trajectory.index(max(trajectory))
                peak_times.append(peak_time)
        
        if len(peak_times) < 2:
            return 0.0
        
        # Calculate dispersion of peak times
        dispersion = np.std(peak_times) / (np.mean(peak_times) + 1)
        return min(1.0, dispersion)
    
    def _calculate_emergence_score(self,
                                   trajectories: Dict[str, List[float]],
                                   cascades: List[CompleteCascade]) -> float:
        """Calculate if new frames are emerging."""
        emergence_scores = []
        
        for frame, trajectory in trajectories.items():
            if len(trajectory) > 1:
                # Check if trajectory shows growth
                first_half = np.mean(trajectory[:len(trajectory)//2])
                second_half = np.mean(trajectory[len(trajectory)//2:])
                
                if first_half > 0:
                    growth_rate = (second_half - first_half) / first_half
                    emergence_scores.append(max(0, growth_rate))
        
        return np.mean(emergence_scores) if emergence_scores else 0.0
    
    def _determine_frame_roles(self,
                               cascades: List[CompleteCascade],
                               frames: Set[str],
                               pattern_type: str) -> Dict[str, str]:
        """Determine role of each frame in the pattern."""
        roles = {}
        
        # Calculate frame metrics
        frame_metrics = {}
        for frame in frames:
            frame_cascades = [c for c in cascades if frame in c.frames_involved]
            
            if frame_cascades:
                frame_metrics[frame] = {
                    'count': len(frame_cascades),
                    'avg_start': np.mean([c.start_date.timestamp() for c in frame_cascades]),
                    'total_articles': sum(c.total_articles for c in frame_cascades),
                    'avg_influence': np.mean([c.validation_confidence for c in frame_cascades])
                }
        
        if pattern_type == 'convergence':
            # Identify leaders and followers
            sorted_frames = sorted(
                frame_metrics.items(),
                key=lambda x: x[1]['avg_start']
            )
            for i, (frame, _) in enumerate(sorted_frames):
                if i == 0:
                    roles[frame] = 'initiator'
                elif i < len(sorted_frames) / 2:
                    roles[frame] = 'early_adopter'
                else:
                    roles[frame] = 'follower'
        
        elif pattern_type == 'competition':
            # Identify competitors
            sorted_frames = sorted(
                frame_metrics.items(),
                key=lambda x: x[1]['total_articles'],
                reverse=True
            )
            for i, (frame, _) in enumerate(sorted_frames):
                if i == 0:
                    roles[frame] = 'dominant_competitor'
                else:
                    roles[frame] = 'challenger'
        
        elif pattern_type == 'rotation':
            # All frames are alternating
            for frame in frames:
                roles[frame] = 'alternating'
        
        elif pattern_type == 'emergence':
            # Identify emerging and declining frames
            for frame in frames:
                frame_cascades = [c for c in cascades if frame in c.frames_involved]
                if frame_cascades:
                    early_count = len([c for c in frame_cascades 
                                     if c.start_date < cascades[len(cascades)//2].start_date])
                    late_count = len(frame_cascades) - early_count
                    
                    if late_count > early_count * 1.5:
                        roles[frame] = 'emerging'
                    elif early_count > late_count * 1.5:
                        roles[frame] = 'declining'
                    else:
                        roles[frame] = 'stable'
        
        else:  # divergence or other
            for frame in frames:
                roles[frame] = 'independent'
        
        return roles
    
    def _calculate_pattern_strength(self,
                                    cascades: List[CompleteCascade],
                                    trajectories: Dict[str, List[float]],
                                    pattern_type: str) -> float:
        """Calculate strength of the detected pattern."""
        if pattern_type == 'convergence':
            # Strength based on correlation increase
            return self._calculate_convergence_score(trajectories)
        
        elif pattern_type == 'competition':
            # Strength based on mutual exclusivity
            overlaps = []
            for i, c1 in enumerate(cascades):
                for c2 in cascades[i+1:]:
                    if set(c1.frames_involved) != set(c2.frames_involved):
                        overlap = self._calculate_temporal_overlap(c1, c2)
                        overlaps.append(1 - overlap)  # Non-overlap
            return np.mean(overlaps) if overlaps else 0.0
        
        elif pattern_type == 'rotation':
            # Strength based on regularity of alternation
            return self._calculate_rotation_score(trajectories)
        
        else:
            # Default: based on cascade alignment
            return self._calculate_cascade_alignment(cascades)
    
    def _calculate_temporal_overlap(self,
                                    cascade1: CompleteCascade,
                                    cascade2: CompleteCascade) -> float:
        """Calculate temporal overlap between two cascades."""
        overlap_start = max(cascade1.start_date, cascade2.start_date)
        overlap_end = min(cascade1.end_date, cascade2.end_date)
        
        if overlap_start < overlap_end:
            overlap_days = (overlap_end - overlap_start).days
            total_days = max(
                (cascade1.end_date - cascade1.start_date).days,
                (cascade2.end_date - cascade2.start_date).days
            )
            return overlap_days / total_days if total_days > 0 else 0.0
        return 0.0
    
    def _calculate_pattern_stability(self,
                                     trajectories: Dict[str, List[float]]) -> float:
        """Calculate stability of pattern over time."""
        # Calculate variance in trajectories
        variances = []
        for trajectory in trajectories.values():
            if len(trajectory) > 1 and np.mean(trajectory) > 0:
                cv = np.std(trajectory) / np.mean(trajectory)  # Coefficient of variation
                variances.append(cv)
        
        if variances:
            # Lower variance means higher stability
            avg_cv = np.mean(variances)
            return 1.0 / (1.0 + avg_cv)
        return 0.5
    
    def _calculate_directionality(self,
                                  cascades: List[CompleteCascade],
                                  interactions: List[FrameInteraction],
                                  frames: Set[str]) -> float:
        """Calculate directionality of influence flow."""
        if not interactions:
            return 0.0
        
        # Calculate average absolute lead-lag scores
        lead_lag_scores = [abs(i.lead_lag_score) for i in interactions]
        return np.mean(lead_lag_scores)
    
    def _calculate_cascade_alignment(self,
                                     cascades: List[CompleteCascade]) -> float:
        """Calculate how well cascades align temporally."""
        if len(cascades) < 2:
            return 0.0
        
        # Calculate pairwise temporal overlaps
        overlaps = []
        for i, c1 in enumerate(cascades):
            for c2 in cascades[i+1:]:
                overlap = self._calculate_temporal_overlap(c1, c2)
                overlaps.append(overlap)
        
        return np.mean(overlaps) if overlaps else 0.0
    
    def _find_convergence_points(self,
                                 trajectories: Dict[str, List[float]]) -> List[datetime]:
        """Find time points where frames converge."""
        convergence_points = []
        
        if len(trajectories) < 2:
            return convergence_points
        
        # Calculate pairwise distances at each time point
        frame_arrays = list(trajectories.values())
        n_timepoints = len(frame_arrays[0])
        
        for t in range(n_timepoints):
            values = [arr[t] for arr in frame_arrays]
            if np.std(values) < np.mean(values) * 0.2:  # Low variance relative to mean
                # This is a convergence point
                convergence_points.append(t)
        
        return convergence_points
    
    def _find_divergence_points(self,
                                trajectories: Dict[str, List[float]]) -> List[datetime]:
        """Find time points where frames diverge."""
        divergence_points = []
        
        if len(trajectories) < 2:
            return divergence_points
        
        # Calculate pairwise distances at each time point
        frame_arrays = list(trajectories.values())
        n_timepoints = len(frame_arrays[0])
        
        for t in range(1, n_timepoints):
            prev_values = [arr[t-1] for arr in frame_arrays]
            curr_values = [arr[t] for arr in frame_arrays]
            
            prev_std = np.std(prev_values)
            curr_std = np.std(curr_values)
            
            if curr_std > prev_std * 1.5:  # Significant increase in variance
                divergence_points.append(t)
        
        return divergence_points
    
    def _build_influence_network(self,
                                 cascades: List[CompleteCascade],
                                 frames: Set[str],
                                 interactions: List[FrameInteraction]) -> nx.DiGraph:
        """Build directed network of influence between frames."""
        G = nx.DiGraph()
        
        # Add frame nodes
        for frame in frames:
            frame_cascades = [c for c in cascades if frame in c.frames_involved]
            G.add_node(
                frame,
                weight=len(frame_cascades),
                articles=sum(c.total_articles for c in frame_cascades)
            )
        
        # Add edges based on interactions
        for interaction in interactions:
            if interaction.frame1 in frames and interaction.frame2 in frames:
                # Direction based on lead-lag
                if interaction.lead_lag_score > 0:
                    # Frame1 leads
                    G.add_edge(
                        interaction.frame1,
                        interaction.frame2,
                        weight=interaction.causality_score,
                        lead_lag=interaction.lead_lag_score
                    )
                elif interaction.lead_lag_score < 0:
                    # Frame2 leads
                    G.add_edge(
                        interaction.frame2,
                        interaction.frame1,
                        weight=interaction.causality_score,
                        lead_lag=-interaction.lead_lag_score
                    )
        
        return G
    
    def _find_influence_paths(self,
                             influence_network: nx.DiGraph) -> List[List[str]]:
        """Find main influence pathways in the network."""
        paths = []
        
        if influence_network.number_of_nodes() < 2:
            return paths
        
        # Find nodes with high out-degree (influencers)
        out_degrees = dict(influence_network.out_degree())
        influencers = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Find nodes with high in-degree (influenced)
        in_degrees = dict(influence_network.in_degree())
        influenced = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Find paths from influencers to influenced
        for source, _ in influencers:
            for target, _ in influenced:
                if source != target:
                    try:
                        path = nx.shortest_path(influence_network, source, target)
                        if len(path) > 1:
                            paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
        
        return paths[:10]  # Limit to 10 paths
    
    def _validate_multi_frame_pattern(self,
                                      cascades: List[CompleteCascade],
                                      trajectories: Dict[str, List[float]],
                                      pattern_type: str) -> Dict[str, Dict[str, Any]]:
        """Validate multi-frame pattern statistically."""
        tests = {}
        
        # Test 1: Temporal alignment
        if len(cascades) > 1:
            start_times = [c.start_date.timestamp() for c in cascades]
            # Test if start times are more clustered than random
            ks_stat, p_value = stats.kstest(
                (start_times - np.mean(start_times)) / np.std(start_times),
                'norm'
            )
            tests['temporal_alignment'] = {
                'statistic': ks_stat,
                'p_value': p_value
            }
        
        # Test 2: Frame correlation
        if len(trajectories) > 1:
            # Test if correlations are significant
            correlations = []
            p_values = []
            
            frame_arrays = list(trajectories.values())
            for i in range(len(frame_arrays)):
                for j in range(i+1, len(frame_arrays)):
                    if len(frame_arrays[i]) > 3:
                        corr, p_val = stats.pearsonr(frame_arrays[i], frame_arrays[j])
                        correlations.append(corr)
                        p_values.append(p_val)
            
            if correlations:
                tests['frame_correlation'] = {
                    'mean_correlation': np.mean(correlations),
                    'p_value': np.mean(p_values)
                }
        
        # Test 3: Pattern-specific test
        if pattern_type == 'convergence':
            # Test if variance decreases over time
            if len(trajectories) > 1:
                frame_arrays = list(trajectories.values())
                early_var = np.var([arr[:len(arr)//2] for arr in frame_arrays])
                late_var = np.var([arr[len(arr)//2:] for arr in frame_arrays])
                
                tests['convergence_test'] = {
                    'early_variance': early_var,
                    'late_variance': late_var,
                    'convergence': late_var < early_var
                }
        
        return tests
    
    def _calculate_frame_dominance(self,
                                   cascades: List[CompleteCascade],
                                   frames: List[str],
                                   window: Tuple[datetime, datetime]) -> Dict[datetime, Dict[str, float]]:
        """Calculate frame dominance over time."""
        dominance_timeline = {}
        
        # Create weekly bins
        weeks = pd.date_range(window[0], window[1], freq='W')
        
        for week in weeks:
            week_end = week + timedelta(days=7)
            frame_scores = {frame: 0.0 for frame in frames}
            
            # Calculate dominance scores for this week
            for cascade in cascades:
                if cascade.start_date <= week_end and cascade.end_date >= week:
                    # Cascade is active during this week
                    for frame in cascade.frames_involved:
                        if frame in frames:
                            # Score based on articles and validation confidence
                            score = cascade.total_articles * cascade.validation_confidence
                            frame_scores[frame] += score
            
            # Normalize scores
            total_score = sum(frame_scores.values())
            if total_score > 0:
                for frame in frames:
                    frame_scores[frame] /= total_score
            
            dominance_timeline[week] = frame_scores
        
        return dominance_timeline
    
    def _detect_paradigm_shifts(self,
                                patterns: List[MultiFramePattern],
                                dominance_timeline: Dict[datetime, Dict[str, float]],
                                cascades: List[CompleteCascade]) -> List[ParadigmShiftIndicator]:
        """Detect indicators of paradigm shifts."""
        indicators = []
        
        if not dominance_timeline:
            return indicators
        
        # Analyze dominance changes
        timeline_dates = sorted(dominance_timeline.keys())
        frames = list(list(dominance_timeline.values())[0].keys())
        
        for i in range(1, len(timeline_dates)):
            prev_dominance = dominance_timeline[timeline_dates[i-1]]
            curr_dominance = dominance_timeline[timeline_dates[i]]
            
            # Calculate change in dominance
            changes = {}
            for frame in frames:
                changes[frame] = curr_dominance.get(frame, 0) - prev_dominance.get(frame, 0)
            
            # Check for significant shifts
            max_change = max(abs(c) for c in changes.values())
            
            if max_change > self.paradigm_shift_threshold:
                indicator = self._create_paradigm_indicator(
                    changes, timeline_dates[i], patterns, cascades
                )
                if indicator:
                    indicators.append(indicator)
        
        return indicators
    
    def _create_paradigm_indicator(self,
                                   changes: Dict[str, float],
                                   shift_date: datetime,
                                   patterns: List[MultiFramePattern],
                                   cascades: List[CompleteCascade]) -> Optional[ParadigmShiftIndicator]:
        """Create paradigm shift indicator from dominance changes."""
        try:
            # Classify frames
            declining_frames = [f for f, c in changes.items() if c < -0.1]
            emerging_frames = [f for f, c in changes.items() if c > 0.1]
            stable_frames = [f for f, c in changes.items() if abs(c) <= 0.1]
            
            if not declining_frames and not emerging_frames:
                return None
            
            # Calculate shift magnitude
            shift_magnitude = max(abs(c) for c in changes.values())
            
            # Calculate shift velocity (change rate)
            shift_velocity = shift_magnitude  # Per week
            
            # Identify key entities and media
            relevant_cascades = [
                c for c in cascades
                if c.start_date <= shift_date <= c.end_date
            ]
            
            key_entities = []
            key_media = []
            
            for cascade in relevant_cascades:
                if any(f in emerging_frames for f in cascade.frames_involved):
                    key_entities.extend([e for e, _ in cascade.key_entities[:3]])
                    key_media.extend(cascade.media_leaders[:3])
            
            # Identify early signals
            early_signals = []
            for pattern in patterns:
                if pattern.end_date < shift_date and pattern.pattern_type in ['emergence', 'competition']:
                    early_signals.append({
                        'date': pattern.start_date,
                        'type': pattern.pattern_type,
                        'frames': pattern.frames
                    })
            
            return ParadigmShiftIndicator(
                shift_type='sudden' if shift_velocity > 0.5 else 'gradual',
                declining_frames=declining_frames,
                emerging_frames=emerging_frames,
                stable_frames=stable_frames,
                shift_start=shift_date - timedelta(days=7),
                shift_peak=shift_date,
                shift_end=shift_date + timedelta(days=7),
                shift_magnitude=shift_magnitude,
                shift_velocity=shift_velocity,
                shift_acceleration=0.0,  # Would need more time points
                key_entities=list(set(key_entities))[:10],
                key_media=list(set(key_media))[:10],
                key_events=[],  # Would need event detection
                early_signals=early_signals[:5],
                tipping_points=[shift_date],
                confidence=min(1.0, shift_magnitude),
                supporting_evidence=[
                    f"Dominance shift of {shift_magnitude:.2f}",
                    f"{len(emerging_frames)} frames emerging",
                    f"{len(declining_frames)} frames declining"
                ]
            )
            
        except Exception as e:
            logger.error(f"Failed to create paradigm indicator: {e}")
            return None
    
    def _build_cross_frame_network(self,
                                   interactions: List[FrameInteraction],
                                   patterns: List[MultiFramePattern]) -> nx.DiGraph:
        """Build comprehensive cross-frame influence network."""
        G = nx.DiGraph()
        
        # Add nodes from interactions
        for interaction in interactions:
            if not G.has_node(interaction.frame1):
                G.add_node(interaction.frame1, type='frame')
            if not G.has_node(interaction.frame2):
                G.add_node(interaction.frame2, type='frame')
            
            # Add edges based on influence direction
            if interaction.lead_lag_score > 0.2:
                G.add_edge(
                    interaction.frame1,
                    interaction.frame2,
                    weight=interaction.get_interaction_strength(),
                    relationship=interaction.get_relationship_type()
                )
            elif interaction.lead_lag_score < -0.2:
                G.add_edge(
                    interaction.frame2,
                    interaction.frame1,
                    weight=interaction.get_interaction_strength(),
                    relationship=interaction.get_relationship_type()
                )
        
        # Add pattern information
        for pattern in patterns:
            for frame in pattern.frames:
                if G.has_node(frame):
                    G.nodes[frame]['patterns'] = G.nodes[frame].get('patterns', [])
                    G.nodes[frame]['patterns'].append(pattern.pattern_type)
        
        return G
    
    def _validate_pattern(self, pattern: MultiFramePattern) -> bool:
        """Validate a multi-frame pattern."""
        # Check minimum requirements
        if pattern.strength < self.min_pattern_strength:
            return False
        
        if pattern.confidence < 0.3:
            return False
        
        if len(pattern.cascades) < self.min_cascades_for_pattern:
            return False
        
        # Check statistical significance
        significant_tests = sum(
            1 for test in pattern.statistical_tests.values()
            if test.get('p_value', 1.0) < 0.05
        )
        
        if significant_tests < len(pattern.statistical_tests) * 0.5:
            return False
        
        return True
    
    def _generate_summary(self,
                         interactions: List[FrameInteraction],
                         patterns: List[MultiFramePattern],
                         indicators: List[ParadigmShiftIndicator]) -> Dict[str, Any]:
        """Generate summary of multi-frame analysis."""
        summary = {
            'n_interactions': len(interactions),
            'n_patterns': len(patterns),
            'n_paradigm_indicators': len(indicators),
            'pattern_types': Counter([p.pattern_type for p in patterns]),
            'avg_pattern_strength': np.mean([p.strength for p in patterns]) if patterns else 0,
            'avg_pattern_confidence': np.mean([p.confidence for p in patterns]) if patterns else 0,
            'paradigm_shift_detected': len(indicators) > 0,
            'dominant_relationship_types': Counter([
                i.get_relationship_type() for i in interactions
            ]),
            'frames_analyzed': len(set(
                i.frame1 for i in interactions
            ).union(set(i.frame2 for i in interactions)))
        }
        
        if indicators:
            summary['paradigm_shifts'] = [
                {
                    'type': ind.shift_type,
                    'magnitude': ind.shift_magnitude,
                    'emerging_frames': ind.emerging_frames,
                    'declining_frames': ind.declining_frames
                }
                for ind in indicators
            ]
        
        return summary
    
    def validate_detection(self, detection: Any) -> bool:
        """
        Validate a multi-frame detection using statistical tests.
        
        Args:
            detection: MultiFramePattern or ParadigmShiftIndicator to validate
            
        Returns:
            True if detection passes validation
        """
        if isinstance(detection, MultiFramePattern):
            # Validate pattern has minimum requirements
            if detection.strength < self.min_pattern_strength:
                return False
            
            if detection.confidence < 0.3:
                return False
            
            if len(detection.cascades) < self.min_cascades_for_pattern:
                return False
            
            # Check statistical significance
            if hasattr(detection, 'statistical_tests'):
                significant_tests = sum(
                    1 for test in detection.statistical_tests.values()
                    if test.get('p_value', 1.0) < 0.05
                )
                if significant_tests < len(detection.statistical_tests) * 0.5:
                    return False
            
            return True
            
        elif isinstance(detection, ParadigmShiftIndicator):
            # Validate paradigm shift has sufficient evidence
            if detection.shift_magnitude < 0.3:
                return False
            
            if detection.confidence < 0.5:
                return False
            
            if len(detection.evidence.get('patterns', [])) < 2:
                return False
            
            return True
            
        return False
    
    def score_detection(self, detection: Any) -> float:
        """
        Score a multi-frame detection for importance.
        
        Args:
            detection: MultiFramePattern or ParadigmShiftIndicator to score
            
        Returns:
            Score in [0, 1] range
        """
        if isinstance(detection, MultiFramePattern):
            scores = []
            
            # Pattern strength (40% weight)
            scores.append(min(1.0, detection.strength) * 0.4)
            
            # Confidence (20% weight)
            scores.append(detection.confidence * 0.2)
            
            # Number of frames involved (20% weight)
            n_frames = len(detection.frames_involved)
            frame_score = min(1.0, n_frames / 5.0)  # Max at 5+ frames
            scores.append(frame_score * 0.2)
            
            # Number of cascades (20% weight)
            n_cascades = len(detection.cascades)
            cascade_score = min(1.0, n_cascades / 10.0)  # Max at 10+ cascades
            scores.append(cascade_score * 0.2)
            
            return sum(scores)
            
        elif isinstance(detection, ParadigmShiftIndicator):
            scores = []
            
            # Shift magnitude (40% weight)
            scores.append(min(1.0, detection.shift_magnitude) * 0.4)
            
            # Confidence (30% weight)
            scores.append(detection.confidence * 0.3)
            
            # Evidence strength (30% weight)
            n_patterns = len(detection.evidence.get('patterns', []))
            evidence_score = min(1.0, n_patterns / 5.0)
            scores.append(evidence_score * 0.3)
            
            return sum(scores)
            
        return 0.0