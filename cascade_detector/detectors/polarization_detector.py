"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
polarization_detector.py

MAIN OBJECTIVE:
---------------
This script detects and analyzes opinion polarization patterns in media cascades, identifying
ideological divergence, partisan clustering, and narrative wars in climate change coverage.

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
- uuid
- hashlib

MAIN FEATURES:
--------------
1) Opinion polarization and ideological divergence detection
2) Media polarization and partisan clustering analysis
3) Frame polarization and narrative war identification
4) Polarization cascade and amplification tracking
5) Depolarization opportunity and bridge building detection

Author:
-------
Antoine Lemor
"""

from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter, deque
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats, spatial, optimize
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.stats import ks_2samp, mannwhitneyu, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import logging
import warnings
from tqdm import tqdm
import uuid
import hashlib

# Import base components
from cascade_detector.detectors.base_detector import BaseDetector, DetectionContext
from cascade_detector.detectors.cascade_detector import (
    CascadeDetector, CompleteCascade, EnhancedBurst
)
from cascade_detector.detectors.multi_frame_detector import (
    MultiFrameDetector, MultiFramePattern, FrameInteraction
)
from cascade_detector.detectors.cross_media_tracker import (
    CrossMediaTracker, MediaProfile, MediaCoordination, MediaCluster
)
from cascade_detector.detectors.sequence_detector import (
    SequenceDetector, CascadeSequence, TemporalMotif, InformationPathway
)
from cascade_detector.detectors.echo_chamber_detector import (
    EchoChamberDetector, EchoChamber, FilterBubble, HomophilyPattern, BridgeActor
)
from cascade_detector.core.config import DetectorConfig

# Import metrics
from cascade_detector.metrics.scientific_network_metrics import ScientificNetworkMetrics
from cascade_detector.metrics.exhaustive_metrics_calculator import ExhaustiveMetricsCalculator
from cascade_detector.metrics.convergence_metrics import ConvergenceMetrics
from cascade_detector.metrics.diversity_metrics import DiversityMetrics

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class PolarizationPattern:
    """
    Represents a detected polarization pattern with full characteristics.
    """
    pattern_id: str
    pattern_type: str  # 'opinion', 'affective', 'issue', 'frame', 'media', 'audience'
    
    # Poles identification
    poles: List[Set[str]]  # Groups at each pole
    pole_characteristics: List[Dict[str, Any]]  # Characteristics of each pole
    n_poles: int  # Number of poles (2 for bipolar, >2 for multipolar)
    
    # Temporal evolution
    formation_date: datetime
    peak_polarization_date: datetime
    current_date: datetime
    duration_days: int
    
    # Polarization metrics
    polarization_score: float  # Overall polarization (0-1)
    distance_between_poles: float  # Ideological/content distance
    within_pole_homogeneity: List[float]  # Homogeneity within each pole
    between_pole_heterogeneity: float  # Difference between poles
    
    # Dynamics
    polarization_velocity: float  # Rate of polarization increase
    polarization_acceleration: float  # Acceleration of polarization
    stability: float  # Stability of polarization pattern
    reversibility: float  # Likelihood of depolarization
    
    # Content analysis
    contested_frames: List[str]  # Frames in dispute
    contested_entities: List[str]  # Entities viewed differently
    divisive_narratives: List[str]  # Narratives causing division
    consensus_topics: List[str]  # Topics with agreement
    
    # Network structure
    polarization_network: nx.Graph  # Network showing polarization
    cross_pole_edges: List[Tuple[str, str]]  # Connections across poles
    within_pole_density: List[float]  # Density within each pole
    bridge_nodes: List[str]  # Nodes connecting poles
    
    # Media involvement
    media_distribution: Dict[int, Set[str]]  # Pole -> media outlets
    partisan_media: Dict[str, int]  # Media -> pole affiliation
    neutral_media: Set[str]  # Non-aligned media
    
    # Cascade involvement
    polarizing_cascades: List[CompleteCascade]  # Cascades driving polarization
    cascade_alignment: Dict[str, int]  # Cascade -> pole alignment
    
    # Echo chamber interaction
    associated_chambers: List[str]  # Related echo chamber IDs
    chamber_pole_mapping: Dict[str, int]  # Chamber -> pole
    
    # Amplification mechanisms
    amplification_factors: Dict[str, float]  # Mechanism -> strength
    feedback_loops: List[List[str]]  # Reinforcement loops
    
    # Depolarization potential
    common_ground: Dict[str, float]  # Shared beliefs/values
    bridge_topics: List[str]  # Topics that could unite
    mediation_potential: float  # Potential for mediation
    
    # Statistical validation
    significance: float
    confidence: float
    robustness: float
    
    def get_polarization_severity(self) -> str:
        """Categorize polarization severity."""
        if self.polarization_score > 0.8:
            return 'extreme'
        elif self.polarization_score > 0.6:
            return 'high'
        elif self.polarization_score > 0.4:
            return 'moderate'
        elif self.polarization_score > 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def get_polarization_type(self) -> str:
        """Determine specific polarization type."""
        if self.n_poles == 2:
            return 'bipolar'
        elif self.n_poles > 2:
            return 'multipolar'
        else:
            return 'unipolar'


@dataclass
class OpinionCluster:
    """
    Cluster of similar opinions in polarization space.
    """
    cluster_id: str
    
    # Members
    members: Set[str]  # Media/journalists in cluster
    core_members: Set[str]  # Most representative members
    
    # Opinion characteristics
    opinion_vector: np.ndarray  # Position in opinion space
    opinion_strength: float  # How strongly held
    opinion_consistency: float  # Internal consistency
    
    # Content
    characteristic_frames: List[str]
    characteristic_entities: List[str]
    characteristic_narratives: List[str]
    
    # Dynamics
    drift_velocity: np.ndarray  # Movement in opinion space
    hardening_rate: float  # Rate of opinion strengthening
    
    # Relationships
    allied_clusters: List[str]  # Similar clusters
    opposing_clusters: List[str]  # Opposing clusters
    
    def distance_to(self, other: 'OpinionCluster') -> float:
        """Calculate distance to another cluster."""
        return np.linalg.norm(self.opinion_vector - other.opinion_vector)


@dataclass
class PolarizationAxis:
    """
    Axis along which polarization occurs.
    """
    axis_id: str
    axis_name: str  # e.g., 'economic', 'social', 'environmental'
    
    # Dimension
    dimension_vector: np.ndarray  # Direction in feature space
    variance_explained: float  # How much variance this axis explains
    
    # Poles
    positive_pole: Set[str]  # Actors at positive end
    negative_pole: Set[str]  # Actors at negative end
    neutral_zone: Set[str]  # Actors in middle
    
    # Issues defining axis
    defining_issues: List[str]
    defining_frames: List[str]
    
    # Dynamics
    axis_stability: float  # How stable the axis is
    axis_salience: float  # How important the axis is
    
    def project(self, actor_vector: np.ndarray) -> float:
        """Project actor onto axis."""
        return np.dot(actor_vector, self.dimension_vector)


@dataclass
class AffectivePolarization:
    """
    Emotional/affective polarization between groups.
    """
    polarization_id: str
    
    # Groups
    in_group: Set[str]
    out_groups: List[Set[str]]
    
    # Affective measures
    in_group_warmth: float  # Positive affect toward in-group
    out_group_coldness: List[float]  # Negative affect toward out-groups
    affective_gap: float  # Difference in affect
    
    # Behavioral manifestations
    hostile_language_rate: float  # Rate of hostile language
    dehumanization_score: float  # Dehumanization of out-group
    moral_superiority: float  # Belief in moral superiority
    
    # Triggers
    emotional_triggers: List[str]  # What triggers negative affect
    identity_threats: List[str]  # Perceived threats to identity
    
    # Consequences
    dialogue_breakdown: float  # Breakdown in communication
    violence_risk: float  # Risk of violence/conflict
    
    def get_affective_intensity(self) -> float:
        """Calculate overall affective intensity."""
        return self.in_group_warmth + np.mean(self.out_group_coldness)


@dataclass
class IssuePolarization:
    """
    Polarization on specific issues.
    """
    issue_id: str
    issue_name: str
    
    # Positions
    position_spectrum: np.ndarray  # Range of positions
    position_distribution: Dict[float, int]  # Position -> count
    
    # Poles
    extreme_positions: List[float]  # Positions at extremes
    moderate_positions: List[float]  # Positions in middle
    
    # Actors
    actor_positions: Dict[str, float]  # Actor -> position
    position_clusters: List[Set[str]]  # Clusters of similar positions
    
    # Dynamics
    position_hardening: float  # Rate of position hardening
    position_convergence: float  # Rate of convergence (negative if diverging)
    
    # Framing
    competing_frames: Dict[str, Set[str]]  # Frame -> supporters
    frame_wars: List[Tuple[str, str]]  # Competing frame pairs
    
    def get_polarization_degree(self) -> float:
        """Calculate degree of polarization on issue."""
        if len(self.position_distribution) < 2:
            return 0.0
        
        positions = list(self.position_distribution.keys())
        counts = list(self.position_distribution.values())
        
        # Calculate bimodality
        hist, bins = np.histogram(positions, weights=counts, bins=10)
        
        # Check for two peaks
        peaks = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append(i)
        
        if len(peaks) >= 2:
            # Calculate distance between peaks
            peak_distance = abs(peaks[-1] - peaks[0]) / len(hist)
            
            # Calculate concentration at peaks
            peak_concentration = sum(hist[p] for p in peaks) / sum(hist)
            
            return peak_distance * peak_concentration
        
        return 0.0


@dataclass
class DepolarizationOpportunity:
    """
    Opportunity for reducing polarization.
    """
    opportunity_id: str
    opportunity_type: str  # 'common_ground', 'bridge_issue', 'neutral_broker', 'cross_cutting'
    
    # Mechanism
    mechanism: str  # How depolarization could work
    
    # Actors
    bridge_actors: List[str]  # Actors who could bridge
    target_groups: List[Set[str]]  # Groups to be bridged
    
    # Content
    unifying_topics: List[str]  # Topics that could unite
    shared_values: Dict[str, float]  # Values shared across poles
    
    # Conditions
    required_conditions: List[str]  # What needs to happen
    obstacles: List[str]  # What prevents depolarization
    
    # Potential
    success_probability: float  # Likelihood of success
    impact_magnitude: float  # Potential impact if successful
    
    # Strategy
    intervention_points: List[str]  # Where to intervene
    timing_windows: List[Tuple[datetime, datetime]]  # When to act
    
    def get_feasibility(self) -> float:
        """Calculate feasibility of opportunity."""
        return self.success_probability * (1 - len(self.obstacles) / 10)


class PolarizationDetector(BaseDetector):
    """
    Detects and analyzes polarization patterns.
    """
    
    def __init__(self, 
                 context: DetectionContext,
                 config: Optional[DetectorConfig] = None):
        """Initialize polarization detector."""
        super().__init__(context, config)
        self.name = "PolarizationDetector"
        
        # Initialize component detectors
        self.cascade_detector = CascadeDetector(context, config, None)  # burst_detector=None
        self.multi_frame_detector = MultiFrameDetector(context, config, self.cascade_detector)
        self.cross_media_tracker = CrossMediaTracker(context, config, self.cascade_detector, self.multi_frame_detector)
        self.sequence_detector = SequenceDetector(context, config)
        self.echo_chamber_detector = EchoChamberDetector(context, config)
        
        # Initialize metrics with required parameters
        self.network_metrics = ScientificNetworkMetrics(
            source_index=context.source_index,
            entity_index=context.entity_index
        )
        self.exhaustive_calculator = ExhaustiveMetricsCalculator()
        self.convergence_metrics = ConvergenceMetrics(
            source_index=context.source_index,
            entity_index=context.entity_index,
            emotion_index=context.emotion_index,
            temporal_index=context.temporal_index,
            frame_index=context.frame_index
        )
        self.diversity_metrics = DiversityMetrics(
            frame_index=context.frame_index,
            source_index=context.source_index,
            emotion_index=context.emotion_index,
            entity_index=context.entity_index,
            geographic_index=context.geographic_index
        )
        
        # Detection parameters
        self.min_pole_size = 3  # Minimum actors per pole
        self.polarization_threshold = 0.5  # Minimum polarization score
        self.min_duration_days = 14  # Minimum duration for stable pattern
        self.distance_threshold = 0.6  # Minimum distance between poles
        
        # Storage
        self.detected_patterns = []
        self.opinion_clusters = []
        self.polarization_axes = []
        self.affective_polarizations = []
        self.issue_polarizations = []
        self.depolarization_opportunities = []
        
        logger.info(f"PolarizationDetector initialized with config: {config}")
    
    def detect(self, **kwargs) -> Dict[str, Any]:
        """
        Main detection method for polarization patterns.
        """
        logger.info("Starting polarization detection...")
        
        # Get cascades from kwargs or detect them
        cascades = kwargs.get('cascades', [])
        if not cascades:
            cascade_result = self.cascade_detector.detect()
            cascades = cascade_result.get('cascades', []) if isinstance(cascade_result, dict) else cascade_result
        
        # Get sequences from kwargs or detect them
        sequences = kwargs.get('sequences', [])
        if not sequences:
            sequence_result = self.sequence_detector.detect(cascades=cascades)
            # Handle both list and dict formats
            if isinstance(sequence_result, dict):
                sequences = sequence_result.get('sequences', [])
            elif isinstance(sequence_result, list):
                sequences = sequence_result
            else:
                sequences = []
        
        # Get echo chamber results from kwargs or detect them
        echo_results = kwargs.get('echo_results', {})
        if not echo_results:
            echo_results = self.echo_chamber_detector.detect(cascades=cascades, sequences=sequences)
        echo_chambers = echo_results.get('echo_chambers', []) if isinstance(echo_results, dict) else echo_results
        logger.info(f"Working with {len(cascades)} cascades, {len(sequences)} sequences, "
                   f"{len(echo_chambers)} echo chambers")
        
        # Get cross-media tracking results
        cross_media_results = kwargs.get('cross_media_results', {})
        if not cross_media_results:
            cross_media_results = self.cross_media_tracker.detect(cascades=cascades)
        
        media_profiles = cross_media_results.get('media_profiles', {})
        media_clusters = cross_media_results.get('media_clusters', [])
        
        # Build opinion space
        opinion_space = self._build_opinion_space(cascades, media_profiles)
        logger.info(f"Built opinion space with {opinion_space.shape[0]} actors")
        
        # Detect polarization using multiple methods
        patterns = []
        
        # Method 1: Opinion clustering
        opinion_patterns = self._detect_opinion_polarization(
            opinion_space, media_profiles, cascades
        )
        patterns.extend(opinion_patterns)
        logger.info(f"Found {len(opinion_patterns)} opinion polarization patterns")
        
        # Method 2: Network polarization
        network_patterns = self._detect_network_polarization(
            media_profiles, cascades, echo_chambers
        )
        patterns.extend(network_patterns)
        logger.info(f"Found {len(network_patterns)} network polarization patterns")
        
        # Method 3: Frame polarization
        frame_patterns = self._detect_frame_polarization(
            cascades, media_profiles, context
        )
        patterns.extend(frame_patterns)
        logger.info(f"Found {len(frame_patterns)} frame polarization patterns")
        
        # Method 4: Temporal polarization
        temporal_patterns = self._detect_temporal_polarization(
            sequences, cascades, media_profiles
        )
        patterns.extend(temporal_patterns)
        logger.info(f"Found {len(temporal_patterns)} temporal polarization patterns")
        
        # Method 5: Echo chamber polarization
        chamber_patterns = self._detect_chamber_polarization(
            echo_chambers, cascades, media_profiles
        )
        patterns.extend(chamber_patterns)
        logger.info(f"Found {len(chamber_patterns)} chamber-based polarization patterns")
        
        # Merge and deduplicate patterns
        unique_patterns = self._merge_patterns(patterns)
        logger.info(f"After merging: {len(unique_patterns)} unique patterns")
        
        # Analyze each pattern in detail
        analyzed_patterns = []
        for pattern in tqdm(unique_patterns, desc="Analyzing polarization"):
            # Detailed analysis
            pattern = self._analyze_polarization_dynamics(pattern, cascades)
            pattern = self._identify_amplification_mechanisms(pattern, sequences)
            pattern = self._analyze_content_divergence(pattern, cascades)
            pattern = self._identify_bridge_actors(pattern, media_profiles)
            pattern = self._calculate_depolarization_potential(pattern, echo_chambers)
            
            # Validation
            if self._validate_pattern(pattern):
                analyzed_patterns.append(pattern)
                
                # Extract opinion clusters
                clusters = self._extract_opinion_clusters(pattern, opinion_space)
                self.opinion_clusters.extend(clusters)
                
                # Identify polarization axes
                axes = self._identify_polarization_axes(pattern, opinion_space)
                self.polarization_axes.extend(axes)
                
                # Detect affective polarization
                affective = self._detect_affective_polarization(pattern, cascades)
                if affective:
                    self.affective_polarizations.append(affective)
                
                # Analyze issue polarization
                issues = self._analyze_issue_polarization(pattern, cascades)
                self.issue_polarizations.extend(issues)
                
                # Find depolarization opportunities
                opportunities = self._find_depolarization_opportunities(
                    pattern, echo_chambers, media_profiles
                )
                self.depolarization_opportunities.extend(opportunities)
        
        logger.info(f"Detected {len(analyzed_patterns)} valid polarization patterns")
        logger.info(f"Found {len(self.opinion_clusters)} opinion clusters")
        logger.info(f"Identified {len(self.polarization_axes)} polarization axes")
        logger.info(f"Detected {len(self.affective_polarizations)} affective polarizations")
        logger.info(f"Analyzed {len(self.issue_polarizations)} issue polarizations")
        logger.info(f"Found {len(self.depolarization_opportunities)} depolarization opportunities")
        
        # Store results
        self.detected_patterns = analyzed_patterns
        
        return analyzed_patterns
    
    def _build_opinion_space(self, cascades: List[CompleteCascade],
                            media_profiles: Dict[str, MediaProfile]) -> np.ndarray:
        """Build multidimensional opinion space from cascade data."""
        # Collect all media actors
        all_media = set()
        for cascade in cascades:
            all_media.update(cascade.media_involved)
        
        if not all_media:
            return np.array([])
        
        # Build feature matrix
        features = []
        media_list = sorted(all_media)
        
        for media in media_list:
            media_features = []
            
            # Frame preferences
            frame_prefs = np.zeros(8)  # Assuming 8 frames
            if media in media_profiles:
                profile = media_profiles[media]
                for i, frame in enumerate(['Pol', 'Eco', 'Sci', 'Just', 'Cult', 'Envt', 'Pbh', 'Secu']):
                    frame_prefs[i] = profile.frame_preferences.get(frame, 0)
            media_features.extend(frame_prefs)
            
            # Entity preferences (top 50)
            entity_vector = np.zeros(50)
            entity_counts = Counter()
            for cascade in cascades:
                if media in cascade.media_involved:
                    for entity in cascade.entities_involved[:10]:
                        entity_counts[entity['entity']] += 1
            
            for i, (entity, count) in enumerate(entity_counts.most_common(50)):
                if i < 50:
                    entity_vector[i] = count
            media_features.extend(entity_vector)
            
            # Behavioral features
            if media in media_profiles:
                profile = media_profiles[media]
                media_features.extend([
                    profile.leader_score,
                    profile.follower_score,
                    profile.amplifier_score,
                    profile.innovator_score,
                    profile.influence_score,
                    profile.consistency_score
                ])
            else:
                media_features.extend([0, 0, 0, 0, 0, 0])
            
            features.append(media_features)
        
        # Convert to numpy array and normalize
        opinion_space = np.array(features)
        
        if opinion_space.shape[0] > 0:
            scaler = StandardScaler()
            opinion_space = scaler.fit_transform(opinion_space)
        
        return opinion_space
    
    def _detect_opinion_polarization(self, opinion_space: np.ndarray,
                                    media_profiles: Dict[str, MediaProfile],
                                    cascades: List[CompleteCascade]) -> List[PolarizationPattern]:
        """Detect polarization through opinion clustering."""
        patterns = []
        
        if opinion_space.shape[0] < self.min_pole_size * 2:
            return patterns
        
        media_list = sorted(set().union(*[c.media_involved for c in cascades]))
        
        # Try different numbers of clusters
        for n_clusters in range(2, min(6, opinion_space.shape[0] // self.min_pole_size)):
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(opinion_space)
            
            # Calculate silhouette score
            if n_clusters > 1:
                silhouette = silhouette_score(opinion_space, labels)
                
                if silhouette > 0.3:  # Reasonable clustering
                    # Check for polarization
                    cluster_centers = kmeans.cluster_centers_
                    
                    # Calculate distances between clusters
                    distances = pdist(cluster_centers)
                    
                    if np.min(distances) > self.distance_threshold:
                        # Found polarization pattern
                        pattern = self._create_opinion_pattern(
                            labels, cluster_centers, media_list, 
                            opinion_space, cascades
                        )
                        
                        if pattern:
                            patterns.append(pattern)
        
        return patterns
    
    def _detect_network_polarization(self, media_profiles: Dict[str, MediaProfile],
                                    cascades: List[CompleteCascade],
                                    echo_chambers: List[EchoChamber]) -> List[PolarizationPattern]:
        """Detect polarization through network structure."""
        patterns = []
        
        # Build media network
        G = nx.Graph()
        media_pairs = defaultdict(int)
        
        for cascade in cascades:
            media_list = list(cascade.media_involved)
            for i in range(len(media_list)):
                for j in range(i+1, len(media_list)):
                    pair = tuple(sorted([media_list[i], media_list[j]]))
                    media_pairs[pair] += cascade.intensity_score
        
        # Add edges
        for (m1, m2), weight in media_pairs.items():
            G.add_edge(m1, m2, weight=weight)
        
        if G.number_of_nodes() < self.min_pole_size * 2:
            return patterns
        
        # Find network communities
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        
        # Group by community
        communities = defaultdict(set)
        for node, comm_id in partition.items():
            communities[comm_id].add(node)
        
        # Check for polarization between communities
        for c1_id, c1_nodes in communities.items():
            for c2_id, c2_nodes in communities.items():
                if c1_id >= c2_id:
                    continue
                
                if len(c1_nodes) >= self.min_pole_size and len(c2_nodes) >= self.min_pole_size:
                    # Calculate inter-community edges
                    inter_edges = sum(1 for u, v in G.edges() 
                                    if (u in c1_nodes and v in c2_nodes) or 
                                       (u in c2_nodes and v in c1_nodes))
                    
                    # Calculate expected edges
                    expected = (len(c1_nodes) * len(c2_nodes)) / G.number_of_nodes()
                    
                    if inter_edges < expected * 0.5:  # Fewer connections than expected
                        # Found polarization
                        pattern = self._create_network_pattern(
                            [c1_nodes, c2_nodes], G, cascades, echo_chambers
                        )
                        
                        if pattern:
                            patterns.append(pattern)
        
        return patterns
    
    def _detect_frame_polarization(self, cascades: List[CompleteCascade],
                                  media_profiles: Dict[str, MediaProfile],
                                  context: DetectionContext) -> List[PolarizationPattern]:
        """Detect polarization on specific frames."""
        patterns = []
        
        # Analyze frame usage by media
        media_frames = defaultdict(Counter)
        
        for cascade in cascades:
            for media in cascade.media_involved:
                media_frames[media][cascade.primary_frame] += cascade.intensity_score
        
        # Find opposing frame coalitions
        frames = ['Pol', 'Eco', 'Sci', 'Just', 'Cult', 'Envt', 'Pbh', 'Secu']
        
        for i, frame1 in enumerate(frames):
            for frame2 in frames[i+1:]:
                # Find media strongly associated with each frame
                frame1_media = set()
                frame2_media = set()
                
                for media, frame_counts in media_frames.items():
                    total = sum(frame_counts.values())
                    if total > 0:
                        frame1_pref = frame_counts[frame1] / total
                        frame2_pref = frame_counts[frame2] / total
                        
                        if frame1_pref > 0.5:
                            frame1_media.add(media)
                        elif frame2_pref > 0.5:
                            frame2_media.add(media)
                
                if (len(frame1_media) >= self.min_pole_size and 
                    len(frame2_media) >= self.min_pole_size):
                    # Check for polarization
                    overlap = len(frame1_media & frame2_media)
                    
                    if overlap == 0:  # No overlap - polarized
                        pattern = self._create_frame_pattern(
                            frame1, frame2, frame1_media, frame2_media,
                            cascades, media_profiles
                        )
                        
                        if pattern:
                            patterns.append(pattern)
        
        return patterns
    
    def _detect_temporal_polarization(self, sequences: List[CascadeSequence],
                                     cascades: List[CompleteCascade],
                                     media_profiles: Dict[str, MediaProfile]) -> List[PolarizationPattern]:
        """Detect polarization emerging over time."""
        patterns = []
        
        for sequence in sequences:
            # Check for divergence in sequence
            if sequence.narrative_shifts and len(sequence.narrative_shifts) > 2:
                # Multiple narrative shifts suggest polarization
                
                # Track media divergence
                media_groups = self._track_media_divergence(sequence)
                
                if len(media_groups) >= 2:
                    # Found temporal polarization
                    pattern = self._create_temporal_pattern(
                        sequence, media_groups, cascades
                    )
                    
                    if pattern:
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_chamber_polarization(self, echo_chambers: List[EchoChamber],
                                    cascades: List[CompleteCascade],
                                    media_profiles: Dict[str, MediaProfile]) -> List[PolarizationPattern]:
        """Detect polarization between echo chambers."""
        patterns = []
        
        if len(echo_chambers) < 2:
            return patterns
        
        # Check for opposing chambers
        for i, chamber1 in enumerate(echo_chambers):
            for chamber2 in echo_chambers[i+1:]:
                # Calculate chamber opposition
                members1 = chamber1.core_members | chamber1.peripheral_members
                members2 = chamber2.core_members | chamber2.peripheral_members
                
                # Check for minimal overlap
                overlap = len(members1 & members2)
                
                if overlap < min(len(members1), len(members2)) * 0.1:
                    # Check for content opposition
                    frames1 = set(chamber1.dominant_frames)
                    frames2 = set(chamber2.dominant_frames)
                    
                    if not frames1 & frames2:  # No frame overlap
                        # Found polarized chambers
                        pattern = self._create_chamber_pattern(
                            chamber1, chamber2, cascades
                        )
                        
                        if pattern:
                            patterns.append(pattern)
        
        return patterns
    
    def _merge_patterns(self, patterns: List[PolarizationPattern]) -> List[PolarizationPattern]:
        """Merge overlapping polarization patterns."""
        if len(patterns) <= 1:
            return patterns
        
        merged = []
        used = set()
        
        for i, p1 in enumerate(patterns):
            if i in used:
                continue
            
            # Get all members of poles
            p1_members = set()
            for pole in p1.poles:
                p1_members.update(pole)
            
            # Look for overlapping patterns
            for j, p2 in enumerate(patterns[i+1:], i+1):
                if j in used:
                    continue
                
                p2_members = set()
                for pole in p2.poles:
                    p2_members.update(pole)
                
                # Calculate overlap
                overlap = len(p1_members & p2_members) / len(p1_members | p2_members) \
                         if p1_members | p2_members else 0
                
                if overlap > 0.5:
                    # Merge patterns
                    p1.poles = self._merge_poles(p1.poles, p2.poles)
                    p1.polarizing_cascades.extend(p2.polarizing_cascades)
                    p1.polarizing_cascades = list(set(p1.polarizing_cascades))
                    used.add(j)
            
            merged.append(p1)
            used.add(i)
        
        return merged
    
    def _merge_poles(self, poles1: List[Set[str]], 
                    poles2: List[Set[str]]) -> List[Set[str]]:
        """Merge pole sets from two patterns."""
        merged_poles = []
        
        # Start with poles1
        for pole in poles1:
            merged_poles.append(pole.copy())
        
        # Add non-overlapping poles from poles2
        for pole2 in poles2:
            best_match = None
            best_overlap = 0
            
            for i, merged_pole in enumerate(merged_poles):
                overlap = len(pole2 & merged_pole) / len(pole2 | merged_pole) \
                         if pole2 | merged_pole else 0
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = i
            
            if best_match is not None and best_overlap > 0.3:
                merged_poles[best_match].update(pole2)
            else:
                merged_poles.append(pole2)
        
        return merged_poles
    
    def _create_opinion_pattern(self, labels: np.ndarray,
                               centers: np.ndarray,
                               media_list: List[str],
                               opinion_space: np.ndarray,
                               cascades: List[CompleteCascade]) -> Optional[PolarizationPattern]:
        """Create polarization pattern from opinion clustering."""
        # Group media by cluster
        poles = defaultdict(set)
        for i, label in enumerate(labels):
            if i < len(media_list):
                poles[label].add(media_list[i])
        
        poles_list = list(poles.values())
        
        if len(poles_list) < 2:
            return None
        
        # Calculate distances between poles
        distances = pdist(centers)
        
        # Find relevant cascades
        relevant_cascades = []
        for cascade in cascades:
            involved_poles = sum(1 for pole in poles_list 
                               if cascade.media_involved & pole)
            if involved_poles >= 2:
                relevant_cascades.append(cascade)
        
        if not relevant_cascades:
            return None
        
        # Calculate temporal bounds
        formation_date = min(c.start_date for c in relevant_cascades)
        peak_date = max(c.peak_date for c in relevant_cascades 
                       if c.intensity_score > 0.5)
        
        # Calculate polarization metrics
        polarization_score = np.mean(distances) / np.sqrt(centers.shape[1])
        
        # Calculate homogeneity within poles
        within_homogeneity = []
        for label in range(len(poles_list)):
            mask = labels == label
            if np.sum(mask) > 1:
                pole_points = opinion_space[mask]
                pole_distances = pdist(pole_points)
                homogeneity = 1 / (1 + np.mean(pole_distances)) if len(pole_distances) > 0 else 1
                within_homogeneity.append(homogeneity)
        
        return PolarizationPattern(
            pattern_id=str(uuid.uuid4()),
            pattern_type='opinion',
            poles=poles_list,
            pole_characteristics=[{'center': centers[i].tolist()} 
                                 for i in range(len(poles_list))],
            n_poles=len(poles_list),
            formation_date=formation_date,
            peak_polarization_date=peak_date,
            current_date=datetime.now(),
            duration_days=(datetime.now() - formation_date).days,
            polarization_score=min(polarization_score, 1.0),
            distance_between_poles=float(np.mean(distances)),
            within_pole_homogeneity=within_homogeneity,
            between_pole_heterogeneity=float(np.std(distances)),
            polarization_velocity=0.0,  # Will be calculated
            polarization_acceleration=0.0,  # Will be calculated
            stability=0.0,  # Will be calculated
            reversibility=0.0,  # Will be calculated
            contested_frames=[],  # Will be identified
            contested_entities=[],  # Will be identified
            divisive_narratives=[],  # Will be identified
            consensus_topics=[],  # Will be identified
            polarization_network=nx.Graph(),  # Will be built
            cross_pole_edges=[],  # Will be identified
            within_pole_density=[],  # Will be calculated
            bridge_nodes=[],  # Will be identified
            media_distribution={i: poles[i] for i in range(len(poles_list))},
            partisan_media={},  # Will be calculated
            neutral_media=set(),  # Will be identified
            polarizing_cascades=relevant_cascades,
            cascade_alignment={},  # Will be calculated
            associated_chambers=[],  # Will be linked
            chamber_pole_mapping={},  # Will be mapped
            amplification_factors={},  # Will be calculated
            feedback_loops=[],  # Will be identified
            common_ground={},  # Will be identified
            bridge_topics=[],  # Will be identified
            mediation_potential=0.0,  # Will be calculated
            significance=0.0,  # Will be calculated
            confidence=0.0,  # Will be calculated
            robustness=0.0  # Will be calculated
        )
    
    def _create_network_pattern(self, communities: List[Set[str]],
                               network: nx.Graph,
                               cascades: List[CompleteCascade],
                               echo_chambers: List[EchoChamber]) -> Optional[PolarizationPattern]:
        """Create polarization pattern from network communities."""
        if len(communities) < 2:
            return None
        
        # Find relevant cascades
        relevant_cascades = []
        for cascade in cascades:
            involved_communities = sum(1 for comm in communities 
                                     if cascade.media_involved & comm)
            if involved_communities >= 2:
                relevant_cascades.append(cascade)
        
        if not relevant_cascades:
            return None
        
        formation_date = min(c.start_date for c in relevant_cascades)
        
        # Calculate network metrics
        within_density = []
        for comm in communities:
            if len(comm) > 1:
                subgraph = network.subgraph(comm)
                density = nx.density(subgraph)
                within_density.append(density)
        
        # Find cross-pole edges
        cross_edges = []
        for u, v in network.edges():
            comm_u = next((i for i, comm in enumerate(communities) if u in comm), None)
            comm_v = next((i for i, comm in enumerate(communities) if v in comm), None)
            
            if comm_u is not None and comm_v is not None and comm_u != comm_v:
                cross_edges.append((u, v))
        
        # Link to echo chambers
        associated_chambers = []
        chamber_mapping = {}
        
        for chamber in echo_chambers:
            chamber_members = chamber.core_members | chamber.peripheral_members
            
            for i, comm in enumerate(communities):
                overlap = len(chamber_members & comm) / len(chamber_members) \
                         if chamber_members else 0
                
                if overlap > 0.5:
                    associated_chambers.append(chamber.chamber_id)
                    chamber_mapping[chamber.chamber_id] = i
        
        return PolarizationPattern(
            pattern_id=str(uuid.uuid4()),
            pattern_type='media',
            poles=communities,
            pole_characteristics=[{'size': len(comm)} for comm in communities],
            n_poles=len(communities),
            formation_date=formation_date,
            peak_polarization_date=formation_date,  # Will be updated
            current_date=datetime.now(),
            duration_days=(datetime.now() - formation_date).days,
            polarization_score=1 - len(cross_edges) / network.number_of_edges() 
                             if network.number_of_edges() > 0 else 0,
            distance_between_poles=1 - len(cross_edges) / (len(communities[0]) * len(communities[1]))
                                  if len(communities) >= 2 else 0,
            within_pole_homogeneity=within_density,
            between_pole_heterogeneity=0.0,  # Will be calculated
            polarization_velocity=0.0,
            polarization_acceleration=0.0,
            stability=0.0,
            reversibility=0.0,
            contested_frames=[],
            contested_entities=[],
            divisive_narratives=[],
            consensus_topics=[],
            polarization_network=network,
            cross_pole_edges=cross_edges,
            within_pole_density=within_density,
            bridge_nodes=[],
            media_distribution={i: comm for i, comm in enumerate(communities)},
            partisan_media={},
            neutral_media=set(),
            polarizing_cascades=relevant_cascades,
            cascade_alignment={},
            associated_chambers=associated_chambers,
            chamber_pole_mapping=chamber_mapping,
            amplification_factors={},
            feedback_loops=[],
            common_ground={},
            bridge_topics=[],
            mediation_potential=0.0,
            significance=0.0,
            confidence=0.0,
            robustness=0.0
        )
    
    def _create_frame_pattern(self, frame1: str, frame2: str,
                             media1: Set[str], media2: Set[str],
                             cascades: List[CompleteCascade],
                             media_profiles: Dict[str, MediaProfile]) -> Optional[PolarizationPattern]:
        """Create frame-based polarization pattern."""
        # Find cascades involving both groups
        relevant_cascades = []
        for cascade in cascades:
            if (cascade.media_involved & media1) and (cascade.media_involved & media2):
                relevant_cascades.append(cascade)
        
        if not relevant_cascades:
            return None
        
        formation_date = min(c.start_date for c in relevant_cascades)
        
        return PolarizationPattern(
            pattern_id=str(uuid.uuid4()),
            pattern_type='frame',
            poles=[media1, media2],
            pole_characteristics=[
                {'dominant_frame': frame1},
                {'dominant_frame': frame2}
            ],
            n_poles=2,
            formation_date=formation_date,
            peak_polarization_date=formation_date,
            current_date=datetime.now(),
            duration_days=(datetime.now() - formation_date).days,
            polarization_score=1.0,  # Complete frame separation
            distance_between_poles=1.0,  # Maximum frame distance
            within_pole_homogeneity=[1.0, 1.0],  # Perfect frame homogeneity
            between_pole_heterogeneity=1.0,
            polarization_velocity=0.0,
            polarization_acceleration=0.0,
            stability=0.0,
            reversibility=0.0,
            contested_frames=[frame1, frame2],
            contested_entities=[],
            divisive_narratives=[],
            consensus_topics=[],
            polarization_network=nx.Graph(),
            cross_pole_edges=[],
            within_pole_density=[],
            bridge_nodes=[],
            media_distribution={0: media1, 1: media2},
            partisan_media={},
            neutral_media=set(),
            polarizing_cascades=relevant_cascades,
            cascade_alignment={},
            associated_chambers=[],
            chamber_pole_mapping={},
            amplification_factors={},
            feedback_loops=[],
            common_ground={},
            bridge_topics=[],
            mediation_potential=0.0,
            significance=0.0,
            confidence=0.0,
            robustness=0.0
        )
    
    def _create_temporal_pattern(self, sequence: CascadeSequence,
                                media_groups: List[Set[str]],
                                cascades: List[CompleteCascade]) -> Optional[PolarizationPattern]:
        """Create temporal polarization pattern."""
        if len(media_groups) < 2:
            return None
        
        return PolarizationPattern(
            pattern_id=str(uuid.uuid4()),
            pattern_type='temporal',
            poles=media_groups,
            pole_characteristics=[{} for _ in media_groups],
            n_poles=len(media_groups),
            formation_date=sequence.start_date,
            peak_polarization_date=sequence.cascades[-1].peak_date if sequence.cascades else datetime.now(),
            current_date=datetime.now(),
            duration_days=sequence.duration_days,
            polarization_score=1 - sequence.coherence_score,  # Divergence
            distance_between_poles=0.0,  # Will be calculated
            within_pole_homogeneity=[],  # Will be calculated
            between_pole_heterogeneity=0.0,
            polarization_velocity=len(sequence.narrative_shifts) / sequence.duration_days 
                                if sequence.duration_days > 0 else 0,
            polarization_acceleration=0.0,
            stability=sequence.coherence_score,
            reversibility=1 - sequence.predictability,
            contested_frames=[],
            contested_entities=[],
            divisive_narratives=[],
            consensus_topics=[],
            polarization_network=nx.Graph(),
            cross_pole_edges=[],
            within_pole_density=[],
            bridge_nodes=[],
            media_distribution={i: group for i, group in enumerate(media_groups)},
            partisan_media={},
            neutral_media=set(),
            polarizing_cascades=sequence.cascades,
            cascade_alignment={},
            associated_chambers=[],
            chamber_pole_mapping={},
            amplification_factors={},
            feedback_loops=[],
            common_ground={},
            bridge_topics=[],
            mediation_potential=0.0,
            significance=sequence.significance,
            confidence=sequence.confidence,
            robustness=0.0
        )
    
    def _create_chamber_pattern(self, chamber1: EchoChamber,
                               chamber2: EchoChamber,
                               cascades: List[CompleteCascade]) -> Optional[PolarizationPattern]:
        """Create polarization pattern from opposing echo chambers."""
        members1 = chamber1.core_members | chamber1.peripheral_members
        members2 = chamber2.core_members | chamber2.peripheral_members
        
        # Find cascades involving both chambers
        relevant_cascades = []
        for cascade in cascades:
            if (cascade.media_involved & members1) or (cascade.media_involved & members2):
                relevant_cascades.append(cascade)
        
        if not relevant_cascades:
            return None
        
        formation_date = min(chamber1.formation_date, chamber2.formation_date)
        
        return PolarizationPattern(
            pattern_id=str(uuid.uuid4()),
            pattern_type='chamber',
            poles=[members1, members2],
            pole_characteristics=[
                {
                    'chamber_id': chamber1.chamber_id,
                    'isolation': chamber1.isolation_score,
                    'homogeneity': chamber1.content_homogeneity
                },
                {
                    'chamber_id': chamber2.chamber_id,
                    'isolation': chamber2.isolation_score,
                    'homogeneity': chamber2.content_homogeneity
                }
            ],
            n_poles=2,
            formation_date=formation_date,
            peak_polarization_date=max(chamber1.peak_date, chamber2.peak_date),
            current_date=datetime.now(),
            duration_days=(datetime.now() - formation_date).days,
            polarization_score=(chamber1.isolation_score + chamber2.isolation_score) / 2,
            distance_between_poles=1 - len(members1 & members2) / len(members1 | members2)
                                  if members1 | members2 else 0,
            within_pole_homogeneity=[chamber1.content_homogeneity, chamber2.content_homogeneity],
            between_pole_heterogeneity=abs(chamber1.content_homogeneity - chamber2.content_homogeneity),
            polarization_velocity=0.0,
            polarization_acceleration=0.0,
            stability=(chamber1.stability + chamber2.stability) / 2,
            reversibility=1 - (chamber1.isolation_score + chamber2.isolation_score) / 2,
            contested_frames=list(set(chamber1.dominant_frames) | set(chamber2.dominant_frames)),
            contested_entities=[],
            divisive_narratives=[],
            consensus_topics=[],
            polarization_network=nx.Graph(),
            cross_pole_edges=[],
            within_pole_density=[chamber1.density, chamber2.density],
            bridge_nodes=list(set(b for b, _ in chamber1.bridge_nodes[:3]) | 
                            set(b for b, _ in chamber2.bridge_nodes[:3])),
            media_distribution={0: members1, 1: members2},
            partisan_media={},
            neutral_media=set(),
            polarizing_cascades=relevant_cascades,
            cascade_alignment={},
            associated_chambers=[chamber1.chamber_id, chamber2.chamber_id],
            chamber_pole_mapping={chamber1.chamber_id: 0, chamber2.chamber_id: 1},
            amplification_factors={
                'echo_chamber': (chamber1.echo_strength + chamber2.echo_strength) / 2
            },
            feedback_loops=[],
            common_ground={},
            bridge_topics=[],
            mediation_potential=0.0,
            significance=(chamber1.significance + chamber2.significance) / 2,
            confidence=0.0,
            robustness=(chamber1.robustness + chamber2.robustness) / 2
        )
    
    def _track_media_divergence(self, sequence: CascadeSequence) -> List[Set[str]]:
        """Track how media diverge through a sequence."""
        if not sequence.cascades:
            return []
        
        # Start with all media in first cascade
        initial_media = sequence.cascades[0].media_involved
        
        # Track which media stay consistent vs diverge
        consistent_media = initial_media.copy()
        
        for cascade in sequence.cascades[1:]:
            consistent_media &= cascade.media_involved
        
        # Find divergent groups
        groups = []
        remaining_media = initial_media - consistent_media
        
        # Group by which cascades they participate in
        media_cascades = defaultdict(set)
        for media in remaining_media:
            for i, cascade in enumerate(sequence.cascades):
                if media in cascade.media_involved:
                    media_cascades[media].add(i)
        
        # Cluster media by participation pattern
        if media_cascades:
            # Simple clustering based on cascade participation
            processed = set()
            for media1, cascades1 in media_cascades.items():
                if media1 in processed:
                    continue
                
                group = {media1}
                processed.add(media1)
                
                for media2, cascades2 in media_cascades.items():
                    if media2 not in processed:
                        overlap = len(cascades1 & cascades2) / len(cascades1 | cascades2) \
                                if cascades1 | cascades2 else 0
                        
                        if overlap > 0.5:
                            group.add(media2)
                            processed.add(media2)
                
                if len(group) >= self.min_pole_size:
                    groups.append(group)
        
        # Add consistent media as a group if large enough
        if len(consistent_media) >= self.min_pole_size:
            groups.append(consistent_media)
        
        return groups
    
    def _analyze_polarization_dynamics(self, pattern: PolarizationPattern,
                                      cascades: List[CompleteCascade]) -> PolarizationPattern:
        """Analyze dynamics of polarization pattern."""
        # Calculate velocity and acceleration
        if pattern.polarizing_cascades:
            # Sort cascades by time
            sorted_cascades = sorted(pattern.polarizing_cascades, 
                                   key=lambda c: c.peak_date)
            
            # Calculate polarization over time
            polarization_timeline = []
            
            for i, cascade in enumerate(sorted_cascades):
                # Measure polarization at this point
                pole_participation = []
                for pole in pattern.poles:
                    participation = len(cascade.media_involved & pole) / len(pole) \
                                  if pole else 0
                    pole_participation.append(participation)
                
                # Polarization as variance in participation
                polarization = np.var(pole_participation)
                polarization_timeline.append(polarization)
            
            if len(polarization_timeline) > 1:
                # Calculate velocity (rate of change)
                velocities = np.diff(polarization_timeline)
                pattern.polarization_velocity = np.mean(velocities)
                
                if len(velocities) > 1:
                    # Calculate acceleration
                    accelerations = np.diff(velocities)
                    pattern.polarization_acceleration = np.mean(accelerations)
            
            # Calculate stability (inverse of variance in polarization)
            if polarization_timeline:
                pattern.stability = 1 / (1 + np.std(polarization_timeline))
        
        # Analyze cascade alignment
        for cascade in pattern.polarizing_cascades:
            # Determine which pole cascade aligns with
            pole_scores = []
            for i, pole in enumerate(pattern.poles):
                score = len(cascade.media_involved & pole) / len(pole) if pole else 0
                pole_scores.append(score)
            
            if pole_scores:
                dominant_pole = np.argmax(pole_scores)
                pattern.cascade_alignment[cascade.cascade_id] = dominant_pole
        
        # Identify partisan media
        for i, pole in enumerate(pattern.poles):
            for media in pole:
                # Count cascade participation
                pole_cascades = sum(1 for cid, pole_idx in pattern.cascade_alignment.items()
                                  if pole_idx == i)
                
                if pole_cascades > len(pattern.cascade_alignment) * 0.7:
                    pattern.partisan_media[media] = i
        
        # Identify neutral media
        all_media = set()
        for cascade in cascades:
            all_media.update(cascade.media_involved)
        
        partisan = set(pattern.partisan_media.keys())
        pole_media = set()
        for pole in pattern.poles:
            pole_media.update(pole)
        
        pattern.neutral_media = all_media - partisan - pole_media
        
        return pattern
    
    def _identify_amplification_mechanisms(self, pattern: PolarizationPattern,
                                          sequences: List[CascadeSequence]) -> PolarizationPattern:
        """Identify mechanisms amplifying polarization."""
        # Check for reinforcement through sequences
        reinforcement_score = 0
        
        for sequence in sequences:
            # Check if sequence involves polarized groups
            sequence_media = set()
            for cascade in sequence.cascades:
                sequence_media.update(cascade.media_involved)
            
            for pole in pattern.poles:
                overlap = len(sequence_media & pole) / len(pole) if pole else 0
                
                if overlap > 0.5:
                    # Sequence reinforces this pole
                    reinforcement_score += sequence.predictability * sequence.coherence_score
        
        pattern.amplification_factors['sequence_reinforcement'] = reinforcement_score / len(sequences) \
                                                                 if sequences else 0
        
        # Check for echo chamber amplification
        echo_amplification = 0
        for chamber_id in pattern.associated_chambers:
            # Would need chamber data
            echo_amplification += 0.5  # Placeholder
        
        pattern.amplification_factors['echo_chamber'] = echo_amplification / len(pattern.associated_chambers) \
                                                       if pattern.associated_chambers else 0
        
        # Check for feedback loops
        if pattern.polarization_network and pattern.polarization_network.number_of_nodes() > 3:
            # Find cycles in network
            try:
                cycles = list(nx.simple_cycles(pattern.polarization_network.to_directed()))
                pattern.feedback_loops = [c for c in cycles if len(c) >= 3][:10]
                
                pattern.amplification_factors['feedback_loops'] = len(pattern.feedback_loops) / 10
            except:
                pattern.feedback_loops = []
        
        # Calculate overall amplification
        if pattern.amplification_factors:
            pattern.amplification_factors['overall'] = np.mean(list(pattern.amplification_factors.values()))
        
        return pattern
    
    def _analyze_content_divergence(self, pattern: PolarizationPattern,
                                   cascades: List[CompleteCascade]) -> PolarizationPattern:
        """Analyze content divergence between poles."""
        # Analyze frames
        pole_frames = []
        for pole in pattern.poles:
            frames = Counter()
            for cascade in pattern.polarizing_cascades:
                if cascade.media_involved & pole:
                    frames[cascade.primary_frame] += 1
            pole_frames.append(frames)
        
        # Find contested frames
        all_frames = set()
        for pf in pole_frames:
            all_frames.update(pf.keys())
        
        for frame in all_frames:
            frame_variance = np.var([pf.get(frame, 0) for pf in pole_frames])
            if frame_variance > 0:
                pattern.contested_frames.append(frame)
        
        # Analyze entities
        pole_entities = []
        for pole in pattern.poles:
            entities = Counter()
            for cascade in pattern.polarizing_cascades:
                if cascade.media_involved & pole:
                    for entity in cascade.entities_involved[:20]:
                        entities[entity['entity']] += 1
            pole_entities.append(entities)
        
        # Find contested entities
        all_entities = set()
        for pe in pole_entities:
            all_entities.update(pe.keys())
        
        contested = []
        consensus = []
        
        for entity in list(all_entities)[:100]:  # Limit to top 100
            entity_counts = [pe.get(entity, 0) for pe in pole_entities]
            
            if max(entity_counts) > 0:
                # Calculate how unevenly distributed
                cv = np.std(entity_counts) / (np.mean(entity_counts) + 1e-6)
                
                if cv > 1.0:  # High variance - contested
                    contested.append(entity)
                elif cv < 0.3:  # Low variance - consensus
                    consensus.append(entity)
        
        pattern.contested_entities = contested[:20]
        pattern.consensus_topics = consensus[:10]
        
        # Extract divisive narratives (simplified)
        pattern.divisive_narratives = [
            f"{frame} perspective" for frame in pattern.contested_frames[:3]
        ]
        
        return pattern
    
    def _identify_bridge_actors(self, pattern: PolarizationPattern,
                               media_profiles: Dict[str, MediaProfile]) -> PolarizationPattern:
        """Identify actors that could bridge polarization."""
        bridge_nodes = []
        
        # Find media connected to multiple poles
        for media in pattern.neutral_media:
            connections_to_poles = []
            
            for pole in pattern.poles:
                # Count connections through cascades
                connections = 0
                for cascade in pattern.polarizing_cascades:
                    if media in cascade.media_involved and cascade.media_involved & pole:
                        connections += 1
                
                connections_to_poles.append(connections)
            
            # Check if connects multiple poles
            connected_poles = sum(1 for c in connections_to_poles if c > 0)
            
            if connected_poles >= 2:
                bridge_nodes.append(media)
        
        pattern.bridge_nodes = bridge_nodes[:10]
        
        # Find bridge topics
        if pattern.consensus_topics:
            pattern.bridge_topics = pattern.consensus_topics[:5]
        
        return pattern
    
    def _calculate_depolarization_potential(self, pattern: PolarizationPattern,
                                           echo_chambers: List[EchoChamber]) -> PolarizationPattern:
        """Calculate potential for depolarization."""
        # Identify common ground
        if pattern.consensus_topics:
            for topic in pattern.consensus_topics:
                pattern.common_ground[topic] = 1.0  # Full agreement
        
        # Calculate reversibility based on:
        # 1. Existence of bridge actors
        bridge_factor = min(len(pattern.bridge_nodes) / 5, 1.0)
        
        # 2. Existence of common ground
        common_factor = min(len(pattern.common_ground) / 5, 1.0)
        
        # 3. Cross-pole connections
        cross_connections = len(pattern.cross_pole_edges) if pattern.cross_pole_edges else 0
        total_possible = len(pattern.poles[0]) * len(pattern.poles[1]) if len(pattern.poles) >= 2 else 1
        connection_factor = cross_connections / total_possible if total_possible > 0 else 0
        
        # 4. Stability (less stable = more reversible)
        stability_factor = 1 - pattern.stability
        
        pattern.reversibility = np.mean([bridge_factor, common_factor, 
                                        connection_factor, stability_factor])
        
        # Calculate mediation potential
        pattern.mediation_potential = pattern.reversibility * (1 - pattern.polarization_score)
        
        return pattern
    
    def _validate_pattern(self, pattern: PolarizationPattern) -> bool:
        """Validate polarization pattern."""
        # Check pole sizes
        for pole in pattern.poles:
            if len(pole) < self.min_pole_size:
                return False
        
        # Check duration
        if pattern.duration_days < self.min_duration_days:
            return False
        
        # Check polarization score
        if pattern.polarization_score < self.polarization_threshold:
            return False
        
        # Statistical validation
        significance = self._calculate_pattern_significance(pattern)
        confidence = self._calculate_pattern_confidence(pattern)
        robustness = self._calculate_pattern_robustness(pattern)
        
        pattern.significance = significance
        pattern.confidence = confidence
        pattern.robustness = robustness
        
        return significance > 0.95 and confidence > 0.7
    
    def _calculate_pattern_significance(self, pattern: PolarizationPattern) -> float:
        """Calculate statistical significance of pattern."""
        # Test against null hypothesis of random grouping
        
        # Calculate expected polarization under random assignment
        n_actors = sum(len(pole) for pole in pattern.poles)
        n_poles = pattern.n_poles
        
        if n_actors > 0 and n_poles > 1:
            # Expected variance under random assignment
            expected_var = (n_poles - 1) / n_poles
            
            # Observed variance
            pole_sizes = [len(pole) for pole in pattern.poles]
            observed_var = np.var(pole_sizes) / (n_actors ** 2)
            
            # Z-score
            z = (observed_var - expected_var) / (expected_var * 0.1)
            
            # Convert to p-value
            from scipy.stats import norm
            p_value = 1 - norm.cdf(abs(z))
            
            return 1 - p_value
        
        return 0.0
    
    def _calculate_pattern_confidence(self, pattern: PolarizationPattern) -> float:
        """Calculate confidence in pattern detection."""
        factors = []
        
        # Polarization strength
        factors.append(pattern.polarization_score)
        
        # Stability
        factors.append(pattern.stability)
        
        # Number of cascades
        cascade_factor = min(len(pattern.polarizing_cascades) / 20, 1.0)
        factors.append(cascade_factor)
        
        # Pole separation
        if pattern.distance_between_poles > 0:
            factors.append(min(pattern.distance_between_poles, 1.0))
        
        # Consistency of poles
        if pattern.within_pole_homogeneity:
            factors.append(np.mean(pattern.within_pole_homogeneity))
        
        return np.mean(factors) if factors else 0.0
    
    def _calculate_pattern_robustness(self, pattern: PolarizationPattern) -> float:
        """Calculate robustness of pattern to perturbation."""
        # Robustness based on:
        # 1. Size of poles
        size_factor = min(sum(len(pole) for pole in pattern.poles) / 50, 1.0)
        
        # 2. Duration
        duration_factor = min(pattern.duration_days / 30, 1.0)
        
        # 3. Number of supporting cascades
        cascade_factor = min(len(pattern.polarizing_cascades) / 30, 1.0)
        
        # 4. Stability
        stability_factor = pattern.stability
        
        return np.mean([size_factor, duration_factor, cascade_factor, stability_factor])
    
    def _extract_opinion_clusters(self, pattern: PolarizationPattern,
                                 opinion_space: np.ndarray) -> List[OpinionCluster]:
        """Extract opinion clusters from polarization pattern."""
        clusters = []
        
        for i, pole in enumerate(pattern.poles):
            if pole and i < len(pattern.pole_characteristics):
                # Get pole center if available
                char = pattern.pole_characteristics[i]
                
                if 'center' in char:
                    opinion_vector = np.array(char['center'])
                else:
                    opinion_vector = np.zeros(opinion_space.shape[1] if opinion_space.shape[0] > 0 else 10)
                
                # Determine core members (most central)
                core_size = max(3, len(pole) // 3)
                core_members = set(list(pole)[:core_size])
                
                cluster = OpinionCluster(
                    cluster_id=str(uuid.uuid4()),
                    members=pole,
                    core_members=core_members,
                    opinion_vector=opinion_vector,
                    opinion_strength=pattern.within_pole_homogeneity[i] 
                                   if i < len(pattern.within_pole_homogeneity) else 0.5,
                    opinion_consistency=pattern.within_pole_homogeneity[i]
                                      if i < len(pattern.within_pole_homogeneity) else 0.5,
                    characteristic_frames=pattern.contested_frames[:3] if i == 0 else [],
                    characteristic_entities=pattern.contested_entities[:5] if i == 0 else [],
                    characteristic_narratives=pattern.divisive_narratives[:2] if i == 0 else [],
                    drift_velocity=np.array([pattern.polarization_velocity]),
                    hardening_rate=pattern.polarization_acceleration,
                    allied_clusters=[],
                    opposing_clusters=[]
                )
                
                clusters.append(cluster)
        
        # Identify relationships between clusters
        for i, c1 in enumerate(clusters):
            for j, c2 in enumerate(clusters[i+1:], i+1):
                distance = c1.distance_to(c2)
                
                if distance < 0.3:
                    c1.allied_clusters.append(c2.cluster_id)
                    c2.allied_clusters.append(c1.cluster_id)
                elif distance > 0.7:
                    c1.opposing_clusters.append(c2.cluster_id)
                    c2.opposing_clusters.append(c1.cluster_id)
        
        return clusters
    
    def _identify_polarization_axes(self, pattern: PolarizationPattern,
                                   opinion_space: np.ndarray) -> List[PolarizationAxis]:
        """Identify main axes of polarization."""
        axes = []
        
        if opinion_space.shape[0] < 10:
            return axes
        
        # PCA to find principal axes
        pca = PCA(n_components=min(3, opinion_space.shape[1]))
        pca.fit(opinion_space)
        
        # Analyze each principal component
        for i, (component, variance) in enumerate(zip(pca.components_, 
                                                      pca.explained_variance_ratio_)):
            if variance < 0.1:  # Skip minor axes
                continue
            
            # Project actors onto axis
            projections = opinion_space @ component
            
            # Find poles (high and low projections)
            threshold = np.std(projections)
            positive_pole = set()
            negative_pole = set()
            neutral_zone = set()
            
            media_list = sorted(set().union(*pattern.poles))
            
            for j, proj in enumerate(projections):
                if j < len(media_list):
                    media = media_list[j]
                    
                    if proj > threshold:
                        positive_pole.add(media)
                    elif proj < -threshold:
                        negative_pole.add(media)
                    else:
                        neutral_zone.add(media)
            
            # Determine what defines this axis
            defining_issues = []
            defining_frames = pattern.contested_frames[:2] if i == 0 else []
            
            axis = PolarizationAxis(
                axis_id=str(uuid.uuid4()),
                axis_name=f"axis_{i+1}",
                dimension_vector=component,
                variance_explained=variance,
                positive_pole=positive_pole,
                negative_pole=negative_pole,
                neutral_zone=neutral_zone,
                defining_issues=defining_issues,
                defining_frames=defining_frames,
                axis_stability=pattern.stability,
                axis_salience=variance
            )
            
            axes.append(axis)
        
        return axes
    
    def _detect_affective_polarization(self, pattern: PolarizationPattern,
                                      cascades: List[CompleteCascade]) -> Optional[AffectivePolarization]:
        """Detect affective/emotional polarization."""
        if pattern.n_poles < 2:
            return None
        
        # Take first pole as in-group
        in_group = pattern.poles[0]
        out_groups = pattern.poles[1:]
        
        # Analyze emotional content (simplified)
        # Would need sentiment analysis in real implementation
        
        # Calculate affective measures
        in_group_warmth = pattern.within_pole_homogeneity[0] \
                         if pattern.within_pole_homogeneity else 0.5
        
        out_group_coldness = [1 - h for h in pattern.within_pole_homogeneity[1:]] \
                           if len(pattern.within_pole_homogeneity) > 1 else [0.5]
        
        affective_gap = in_group_warmth + np.mean(out_group_coldness)
        
        # Identify triggers
        emotional_triggers = pattern.contested_entities[:5]
        identity_threats = pattern.divisive_narratives[:3]
        
        return AffectivePolarization(
            polarization_id=str(uuid.uuid4()),
            in_group=in_group,
            out_groups=out_groups,
            in_group_warmth=in_group_warmth,
            out_group_coldness=out_group_coldness,
            affective_gap=affective_gap,
            hostile_language_rate=pattern.polarization_score * 0.5,  # Estimate
            dehumanization_score=pattern.polarization_score * 0.3,  # Estimate
            moral_superiority=in_group_warmth * 0.8,  # Estimate
            emotional_triggers=emotional_triggers,
            identity_threats=identity_threats,
            dialogue_breakdown=pattern.polarization_score,
            violence_risk=pattern.polarization_score * pattern.polarization_acceleration
        )
    
    def _analyze_issue_polarization(self, pattern: PolarizationPattern,
                                   cascades: List[CompleteCascade]) -> List[IssuePolarization]:
        """Analyze polarization on specific issues."""
        issues = []
        
        # Analyze each contested entity as potential issue
        for entity in pattern.contested_entities[:5]:
            # Find positions on this issue
            pole_positions = []
            
            for i, pole in enumerate(pattern.poles):
                # Calculate pole's position on issue
                issue_coverage = 0
                
                for cascade in pattern.polarizing_cascades:
                    if cascade.media_involved & pole:
                        entity_present = any(e['entity'] == entity 
                                           for e in cascade.entities_involved)
                        if entity_present:
                            issue_coverage += cascade.intensity_score
                
                pole_positions.append(issue_coverage)
            
            # Normalize positions
            if max(pole_positions) > 0:
                pole_positions = [p / max(pole_positions) for p in pole_positions]
            
            # Create position distribution
            position_dist = {}
            for i, pos in enumerate(pole_positions):
                position_dist[pos] = len(pattern.poles[i]) if i < len(pattern.poles) else 0
            
            # Identify position clusters
            position_clusters = []
            for i, pole in enumerate(pattern.poles):
                if i < len(pole_positions):
                    position_clusters.append(pole)
            
            issue = IssuePolarization(
                issue_id=str(uuid.uuid4()),
                issue_name=entity,
                position_spectrum=np.array(pole_positions),
                position_distribution=position_dist,
                extreme_positions=[min(pole_positions), max(pole_positions)],
                moderate_positions=[p for p in pole_positions 
                                  if min(pole_positions) < p < max(pole_positions)],
                actor_positions={m: pole_positions[i] 
                               for i, pole in enumerate(pattern.poles) 
                               for m in pole if i < len(pole_positions)},
                position_clusters=position_clusters,
                position_hardening=pattern.polarization_acceleration,
                position_convergence=-pattern.polarization_velocity,  # Negative if diverging
                competing_frames={f: set() for f in pattern.contested_frames[:2]},
                frame_wars=[(pattern.contested_frames[i], pattern.contested_frames[i+1])
                          for i in range(len(pattern.contested_frames)-1)]
            )
            
            issues.append(issue)
        
        return issues
    
    def _find_depolarization_opportunities(self, pattern: PolarizationPattern,
                                          echo_chambers: List[EchoChamber],
                                          media_profiles: Dict[str, MediaProfile]) -> List[DepolarizationOpportunity]:
        """Find opportunities for depolarization."""
        opportunities = []
        
        # Opportunity 1: Common ground topics
        if pattern.common_ground:
            opp = DepolarizationOpportunity(
                opportunity_id=str(uuid.uuid4()),
                opportunity_type='common_ground',
                mechanism='Focus on shared values and consensus topics',
                bridge_actors=pattern.bridge_nodes[:5],
                target_groups=pattern.poles[:2],
                unifying_topics=list(pattern.common_ground.keys())[:5],
                shared_values=pattern.common_ground,
                required_conditions=['Media cooperation', 'Reduced inflammatory rhetoric'],
                obstacles=['Echo chamber reinforcement', 'Partisan incentives'],
                success_probability=pattern.reversibility,
                impact_magnitude=1 - pattern.polarization_score,
                intervention_points=['Media editorial decisions', 'Content moderation'],
                timing_windows=[(datetime.now(), datetime.now() + timedelta(days=30))]
            )
            opportunities.append(opp)
        
        # Opportunity 2: Bridge actors
        if pattern.bridge_nodes:
            opp = DepolarizationOpportunity(
                opportunity_id=str(uuid.uuid4()),
                opportunity_type='neutral_broker',
                mechanism='Leverage bridge actors to facilitate dialogue',
                bridge_actors=pattern.bridge_nodes[:10],
                target_groups=pattern.poles,
                unifying_topics=pattern.bridge_topics,
                shared_values={},
                required_conditions=['Bridge actor willingness', 'Platform for dialogue'],
                obstacles=['Lack of trust', 'Pressure from poles'],
                success_probability=len(pattern.bridge_nodes) / 20,
                impact_magnitude=pattern.mediation_potential,
                intervention_points=['Bridge actor engagement', 'Dialogue facilitation'],
                timing_windows=[(datetime.now(), datetime.now() + timedelta(days=14))]
            )
            opportunities.append(opp)
        
        # Opportunity 3: Cross-cutting exposure
        if pattern.neutral_media:
            opp = DepolarizationOpportunity(
                opportunity_id=str(uuid.uuid4()),
                opportunity_type='cross_cutting',
                mechanism='Increase cross-cutting exposure through neutral media',
                bridge_actors=list(pattern.neutral_media)[:10],
                target_groups=pattern.poles,
                unifying_topics=[],
                shared_values={},
                required_conditions=['Algorithm changes', 'User engagement'],
                obstacles=['Filter bubbles', 'Selective exposure'],
                success_probability=len(pattern.neutral_media) / 50,
                impact_magnitude=0.5,
                intervention_points=['Recommendation algorithms', 'Content curation'],
                timing_windows=[(datetime.now(), datetime.now() + timedelta(days=60))]
            )
            opportunities.append(opp)
        
        return opportunities
    
    def get_pattern_summary(self, pattern: PolarizationPattern) -> Dict[str, Any]:
        """Get summary of polarization pattern."""
        return {
            'pattern_id': pattern.pattern_id,
            'type': pattern.pattern_type,
            'severity': pattern.get_polarization_severity(),
            'structure': pattern.get_polarization_type(),
            'poles': {
                'count': pattern.n_poles,
                'sizes': [len(pole) for pole in pattern.poles],
                'distance': pattern.distance_between_poles
            },
            'metrics': {
                'polarization_score': pattern.polarization_score,
                'stability': pattern.stability,
                'reversibility': pattern.reversibility,
                'velocity': pattern.polarization_velocity,
                'acceleration': pattern.polarization_acceleration
            },
            'content': {
                'contested_frames': pattern.contested_frames[:3],
                'contested_entities': pattern.contested_entities[:5],
                'consensus_topics': pattern.consensus_topics[:3]
            },
            'dynamics': {
                'duration_days': pattern.duration_days,
                'n_cascades': len(pattern.polarizing_cascades),
                'amplification': pattern.amplification_factors.get('overall', 0)
            },
            'depolarization': {
                'mediation_potential': pattern.mediation_potential,
                'n_bridge_actors': len(pattern.bridge_nodes),
                'n_bridge_topics': len(pattern.bridge_topics)
            },
            'validation': {
                'significance': pattern.significance,
                'confidence': pattern.confidence,
                'robustness': pattern.robustness
            }
        }
    
    def export_results(self, output_path: str) -> None:
        """Export all polarization detection results."""
        import json
        
        export_data = {
            'metadata': {
                'detector': self.name,
                'n_patterns': len(self.detected_patterns),
                'n_opinion_clusters': len(self.opinion_clusters),
                'n_axes': len(self.polarization_axes),
                'n_affective': len(self.affective_polarizations),
                'n_issues': len(self.issue_polarizations),
                'n_opportunities': len(self.depolarization_opportunities),
                'timestamp': datetime.now().isoformat()
            },
            'polarization_patterns': [
                self.get_pattern_summary(pattern) for pattern in self.detected_patterns
            ],
            'opinion_clusters': [
                {
                    'cluster_id': c.cluster_id,
                    'n_members': len(c.members),
                    'opinion_strength': c.opinion_strength,
                    'consistency': c.opinion_consistency,
                    'drift_velocity': c.hardening_rate
                } for c in self.opinion_clusters
            ],
            'polarization_axes': [
                {
                    'axis_id': a.axis_id,
                    'variance_explained': a.variance_explained,
                    'n_positive': len(a.positive_pole),
                    'n_negative': len(a.negative_pole),
                    'n_neutral': len(a.neutral_zone),
                    'stability': a.axis_stability
                } for a in self.polarization_axes
            ],
            'affective_polarizations': [
                {
                    'polarization_id': a.polarization_id,
                    'affective_intensity': a.get_affective_intensity(),
                    'affective_gap': a.affective_gap,
                    'dialogue_breakdown': a.dialogue_breakdown,
                    'violence_risk': a.violence_risk
                } for a in self.affective_polarizations
            ],
            'issue_polarizations': [
                {
                    'issue_id': i.issue_id,
                    'issue': i.issue_name,
                    'polarization_degree': i.get_polarization_degree(),
                    'n_positions': len(i.position_distribution),
                    'position_hardening': i.position_hardening
                } for i in self.issue_polarizations
            ],
            'depolarization_opportunities': [
                {
                    'opportunity_id': o.opportunity_id,
                    'type': o.opportunity_type,
                    'mechanism': o.mechanism,
                    'feasibility': o.get_feasibility(),
                    'success_probability': o.success_probability,
                    'impact_magnitude': o.impact_magnitude
                } for o in self.depolarization_opportunities
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported polarization analysis to {output_path}")
    
    def validate_detection(self, detection_result: Dict[str, Any]) -> bool:
        """
        Validate polarization detection results.
        
        Args:
            detection_result: Results from detect() method
            
        Returns:
            True if detection is valid, False otherwise
        """
        try:
            # Check required keys
            required_keys = [
                'polarization_patterns', 'affective_polarizations', 
                'issue_polarizations', 'polarization_cascades',
                'depolarization_opportunities', 'summary'
            ]
            
            for key in required_keys:
                if key not in detection_result:
                    logger.warning(f"Missing required key in detection result: {key}")
                    return False
            
            # Validate polarization patterns
            patterns = detection_result['polarization_patterns']
            if not isinstance(patterns, list):
                logger.warning("Polarization patterns should be a list")
                return False
            
            for pattern in patterns:
                if not isinstance(pattern, PolarizationPattern):
                    logger.warning("Invalid polarization pattern type")
                    return False
                
                # Check basic validity
                if pattern.n_poles < 2:
                    logger.warning(f"Invalid pole count in pattern {pattern.pattern_id}")
                    return False
                
                if len(pattern.poles) != pattern.n_poles:
                    logger.warning(f"Pole count mismatch in pattern {pattern.pattern_id}")
                    return False
                
                if not (0 <= pattern.polarization_score <= 1):
                    logger.warning(f"Invalid polarization score in pattern {pattern.pattern_id}")
                    return False
                
                if not (0 <= pattern.reversibility <= 1):
                    logger.warning(f"Invalid reversibility in pattern {pattern.pattern_id}")
                    return False
                
                # Validate pattern type
                valid_types = ['opinion', 'affective', 'issue', 'frame', 'media', 'audience']
                if pattern.pattern_type not in valid_types:
                    logger.warning(f"Invalid pattern type: {pattern.pattern_type}")
                    return False
            
            # Validate affective polarizations
            affective = detection_result['affective_polarizations']
            if not isinstance(affective, list):
                logger.warning("Affective polarizations should be a list")
                return False
            
            for aff in affective:
                if not isinstance(aff, AffectivePolarization):
                    logger.warning("Invalid affective polarization type")
                    return False
                
                if not (0 <= aff.hostility_level <= 1):
                    logger.warning("Invalid hostility level")
                    return False
                
                if not (0 <= aff.empathy_level <= 1):
                    logger.warning("Invalid empathy level")
                    return False
            
            # Validate issue polarizations
            issues = detection_result['issue_polarizations']
            if not isinstance(issues, list):
                logger.warning("Issue polarizations should be a list")
                return False
            
            for issue in issues:
                if not isinstance(issue, IssuePolarization):
                    logger.warning("Invalid issue polarization type")
                    return False
                
                if not (0 <= issue.position_hardening <= 1):
                    logger.warning("Invalid position hardening")
                    return False
            
            # Validate polarization cascades
            cascades = detection_result['polarization_cascades']
            if not isinstance(cascades, list):
                logger.warning("Polarization cascades should be a list")
                return False
            
            for cascade in cascades:
                if not isinstance(cascade, PolarizationCascade):
                    logger.warning("Invalid polarization cascade type")
                    return False
                
                if cascade.amplification_factor < 1:
                    logger.warning("Invalid amplification factor")
                    return False
            
            # Validate depolarization opportunities
            opportunities = detection_result['depolarization_opportunities']
            if not isinstance(opportunities, list):
                logger.warning("Depolarization opportunities should be a list")
                return False
            
            for opp in opportunities:
                if not isinstance(opp, DepolarizationOpportunity):
                    logger.warning("Invalid depolarization opportunity type")
                    return False
                
                if not (0 <= opp.success_probability <= 1):
                    logger.warning("Invalid success probability")
                    return False
                
                if not (0 <= opp.impact_magnitude <= 1):
                    logger.warning("Invalid impact magnitude")
                    return False
            
            logger.info(f"Validation successful: {len(patterns)} patterns, "
                       f"{len(affective)} affective, {len(issues)} issues")
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def score_detection(self, detection_result: Dict[str, Any]) -> float:
        """
        Score the quality of polarization detection.
        
        Scoring factors:
        - Polarization pattern clarity and strength
        - Affective polarization intensity
        - Issue polarization depth
        - Cascade detection quality
        - Depolarization opportunity identification
        
        Args:
            detection_result: Results from detect() method
            
        Returns:
            Score between 0 and 1
        """
        try:
            if not self.validate_detection(detection_result):
                return 0.0
            
            patterns = detection_result['polarization_patterns']
            affective = detection_result['affective_polarizations']
            issues = detection_result['issue_polarizations']
            cascades = detection_result['polarization_cascades']
            opportunities = detection_result['depolarization_opportunities']
            
            scores = []
            weights = []
            
            # 1. Polarization pattern score (30% weight)
            if patterns:
                # Strong patterns have high polarization scores and clear poles
                pol_scores = [p.polarization_score for p in patterns]
                distances = [p.distance_between_poles for p in patterns]
                stabilities = [p.stability for p in patterns]
                
                avg_pol = np.mean(pol_scores)
                avg_distance = min(1.0, np.mean(distances) / 2)  # Normalize distance
                avg_stability = np.mean(stabilities)
                
                pattern_score = (avg_pol + avg_distance + avg_stability) / 3
                scores.append(pattern_score)
                weights.append(0.3)
                logger.debug(f"Pattern score: {pattern_score:.3f}")
            else:
                # No patterns - might indicate no polarization
                scores.append(0.2)
                weights.append(0.3)
            
            # 2. Affective polarization score (25% weight)
            if affective:
                hostilities = [a.hostility_level for a in affective]
                toxicities = [a.toxicity_score for a in affective]
                
                # High hostility and toxicity indicate strong affective polarization
                avg_hostility = np.mean(hostilities)
                avg_toxicity = np.mean(toxicities)
                
                # Invert empathy - low empathy = high polarization
                empathies = [1 - a.empathy_level for a in affective]
                avg_inv_empathy = np.mean(empathies)
                
                affective_score = (avg_hostility + avg_toxicity + avg_inv_empathy) / 3
                scores.append(affective_score)
                weights.append(0.25)
                logger.debug(f"Affective score: {affective_score:.3f}")
            else:
                scores.append(0.3)
                weights.append(0.25)
            
            # 3. Issue polarization score (20% weight)
            if issues:
                hardenings = [i.position_hardening for i in issues]
                extremities = [i.position_extremity for i in issues]
                
                avg_hardening = np.mean(hardenings)
                avg_extremity = np.mean(extremities)
                
                issue_score = (avg_hardening + avg_extremity) / 2
                scores.append(issue_score)
                weights.append(0.2)
                logger.debug(f"Issue score: {issue_score:.3f}")
            else:
                scores.append(0.3)
                weights.append(0.2)
            
            # 4. Cascade quality score (15% weight)
            if cascades:
                amplifications = [min(c.amplification_factor / 5, 1) for c in cascades]
                velocities = [min(c.polarization_velocity, 1) for c in cascades]
                
                avg_amplification = np.mean(amplifications)
                avg_velocity = np.mean(velocities)
                
                cascade_score = (avg_amplification + avg_velocity) / 2
                scores.append(cascade_score)
                weights.append(0.15)
                logger.debug(f"Cascade score: {cascade_score:.3f}")
            else:
                scores.append(0.3)
                weights.append(0.15)
            
            # 5. Depolarization opportunity score (10% weight)
            if opportunities:
                feasibilities = [o.get_feasibility() for o in opportunities]
                probabilities = [o.success_probability for o in opportunities]
                impacts = [o.impact_magnitude for o in opportunities]
                
                avg_feasibility = np.mean(feasibilities)
                avg_probability = np.mean(probabilities)
                avg_impact = np.mean(impacts)
                
                # Good detection finds feasible, high-impact opportunities
                opp_score = (avg_feasibility + avg_probability + avg_impact) / 3
                scores.append(opp_score)
                weights.append(0.1)
                logger.debug(f"Opportunity score: {opp_score:.3f}")
            else:
                scores.append(0.2)  # Low score if no opportunities found
                weights.append(0.1)
            
            # Calculate weighted average
            total_score = np.average(scores, weights=weights)
            
            # Apply bonuses and penalties
            
            # Bonus for detecting strong polarization
            if patterns and any(p.polarization_score > 0.8 for p in patterns):
                total_score = min(1.0, total_score * 1.15)
                logger.debug("Applied bonus for strong polarization")
            
            # Bonus for comprehensive analysis
            if (len(patterns) > 3 and len(affective) > 2 and 
                len(issues) > 5 and len(opportunities) > 3):
                total_score = min(1.0, total_score * 1.1)
                logger.debug("Applied bonus for comprehensive analysis")
            
            # Bonus for detecting multipolar patterns (more complex)
            if any(p.n_poles > 2 for p in patterns):
                total_score = min(1.0, total_score * 1.05)
                logger.debug("Applied bonus for multipolar patterns")
            
            # Penalty if no significant polarization found
            if not patterns and not affective and not issues:
                total_score *= 0.5
                logger.debug("Applied penalty for no significant polarization")
            
            # Penalty for unstable patterns
            if patterns and all(p.stability < 0.3 for p in patterns):
                total_score *= 0.8
                logger.debug("Applied penalty for unstable patterns")
            
            logger.info(f"Polarization detection score: {total_score:.3f}")
            return float(total_score)
            
        except Exception as e:
            logger.error(f"Scoring error: {e}")
            return 0.0