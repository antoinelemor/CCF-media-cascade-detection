"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
echo_chamber_detector.py

MAIN OBJECTIVE:
---------------
This script detects and analyzes echo chamber formations in media cascades, identifying closed
information loops, filter bubbles, and ideological clustering patterns in coverage.

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
- community
- logging
- warnings
- tqdm
- uuid
- hashlib

MAIN FEATURES:
--------------
1) Closed information loop and self-reinforcing cycle detection
2) Filter bubble and information silo identification
3) Homophily pattern analysis in media networks
4) Information diversity degradation measurement
5) Bridge actor and boundary spanner identification

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
from scipy import stats, spatial
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain
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
from cascade_detector.core.config import DetectorConfig

# Import metrics
from cascade_detector.metrics.scientific_network_metrics import ScientificNetworkMetrics
from cascade_detector.metrics.diversity_metrics import DiversityMetrics
from cascade_detector.metrics.convergence_metrics import ConvergenceMetrics

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class EchoChamber:
    """
    Represents a detected echo chamber with all characteristics.
    """
    chamber_id: str
    chamber_type: str  # 'media', 'ideological', 'topical', 'geographic', 'temporal'
    
    # Members
    core_members: Set[str]  # Media/journalists at center
    peripheral_members: Set[str]  # Loosely connected members
    total_members: int
    
    # Temporal data
    formation_date: datetime
    peak_date: datetime
    current_date: datetime
    duration_days: int
    
    # Structure metrics
    density: float  # Internal connection density
    clustering_coefficient: float  # Local clustering
    modularity: float  # Community strength
    conductance: float  # Cut quality
    
    # Information flow
    internal_flow: float  # Information circulation within chamber
    external_flow: float  # Information exchange with outside
    isolation_score: float  # Degree of isolation (0-1)
    permeability: float  # Openness to external information
    
    # Content characteristics
    dominant_frames: List[str]  # Primary frames in chamber
    dominant_entities: List[Tuple[str, float]]  # Key entities discussed
    dominant_narratives: List[str]  # Main narrative themes
    content_homogeneity: float  # Similarity of content (0-1)
    
    # Diversity metrics
    frame_diversity: float  # Shannon entropy of frames
    entity_diversity: float  # Shannon entropy of entities
    source_diversity: float  # Diversity of information sources
    viewpoint_diversity: float  # Diversity of perspectives
    
    # Network structure
    internal_network: nx.Graph  # Network within chamber
    bridge_nodes: List[Tuple[str, float]]  # Nodes connecting to outside
    structural_holes: List[Tuple[str, str]]  # Missing connections
    
    # Reinforcement patterns
    reinforcement_cycles: List[List[str]]  # Circular reinforcement paths
    amplification_rate: float  # Rate of signal amplification
    echo_strength: float  # Strength of echo effect
    
    # Evolution
    growth_rate: float  # Member growth rate
    stability: float  # Structural stability over time
    lifecycle_stage: str  # 'forming', 'stable', 'declining', 'dissolving'
    
    # Cascades involved
    cascades: List[CompleteCascade]  # Related cascades
    cascade_participation: Dict[str, float]  # Member -> participation
    
    # Statistical validation
    significance: float
    random_probability: float
    robustness: float  # Robustness to perturbation
    
    def get_isolation_level(self) -> str:
        """Categorize isolation level."""
        if self.isolation_score > 0.8:
            return 'highly_isolated'
        elif self.isolation_score > 0.6:
            return 'moderately_isolated'
        elif self.isolation_score > 0.4:
            return 'semi_isolated'
        elif self.isolation_score > 0.2:
            return 'loosely_connected'
        else:
            return 'well_connected'
    
    def get_homophily_score(self) -> float:
        """Calculate homophily score of the chamber."""
        return np.mean([
            self.content_homogeneity,
            1 - self.frame_diversity,
            1 - self.entity_diversity,
            self.clustering_coefficient
        ])


@dataclass
class FilterBubble:
    """
    Represents a filter bubble - personalized information isolation.
    """
    bubble_id: str
    bubble_type: str  # 'algorithmic', 'self_selected', 'social', 'geographic'
    
    # Affected actors
    media_outlets: Set[str]
    journalists: Set[str]
    
    # Information filtering
    included_topics: Set[str]  # Topics inside bubble
    excluded_topics: Set[str]  # Topics filtered out
    filter_strength: float  # How strong the filtering is
    
    # Bubble characteristics
    transparency: float  # How visible the bubble is
    rigidity: float  # How hard to break through
    selectivity: float  # How selective the filter is
    
    # Effects
    information_loss: float  # Amount of information filtered out
    perspective_narrowing: float  # Reduction in viewpoint diversity
    confirmation_bias: float  # Degree of confirmation bias
    
    # Temporal evolution
    formation_time: datetime
    strengthening_rate: float
    
    def get_severity(self) -> str:
        """Assess bubble severity."""
        severity_score = np.mean([
            self.filter_strength,
            self.rigidity,
            self.information_loss,
            self.confirmation_bias
        ])
        
        if severity_score > 0.7:
            return 'severe'
        elif severity_score > 0.5:
            return 'moderate'
        else:
            return 'mild'


@dataclass
class InformationSilo:
    """
    Information silo - isolated information system.
    """
    silo_id: str
    
    # Silo members
    members: Set[str]
    gatekeepers: List[str]  # Control information flow
    
    # Information characteristics
    internal_information: Set[str]  # Information within silo
    shared_beliefs: Dict[str, float]  # Belief -> strength
    taboo_topics: Set[str]  # Topics not discussed
    
    # Boundaries
    boundary_strength: float  # How strong the boundaries are
    cross_silo_communication: float  # Communication with other silos
    
    # Network metrics
    internal_cohesion: float
    external_tension: float
    
    def get_insularity(self) -> float:
        """Calculate insularity of the silo."""
        return self.boundary_strength * (1 - self.cross_silo_communication)


@dataclass
class HomophilyPattern:
    """
    Pattern of homophily (similarity-based connection).
    """
    pattern_id: str
    pattern_type: str  # 'frame', 'entity', 'media', 'temporal'
    
    # Similarity dimensions
    similarity_dimensions: Dict[str, float]  # Dimension -> weight
    
    # Network structure
    homophily_network: nx.Graph
    homophily_strength: float
    
    # Groups formed
    homogeneous_groups: List[Set[str]]
    group_similarities: List[float]
    
    # Effects
    segregation_index: float  # Degree of segregation
    mixing_rate: float  # Rate of cross-group interaction
    
    # Evolution
    strengthening: bool  # Whether pattern is strengthening
    evolution_rate: float


@dataclass
class BridgeActor:
    """
    Actor that bridges between echo chambers.
    """
    actor_id: str
    actor_type: str  # 'media' or 'journalist'
    
    # Bridging metrics
    betweenness_centrality: float
    bridging_score: float  # Ability to bridge communities
    boundary_spanning: float  # Crosses boundaries
    
    # Connected chambers
    connected_chambers: List[str]  # Chamber IDs connected
    information_brokerage: float  # Information broker score
    
    # Influence
    cross_chamber_influence: float
    diversity_promotion: float
    
    # Activity
    cross_posting_rate: float  # Rate of posting across chambers
    frame_switching: float  # Ability to switch frames
    
    def get_bridge_effectiveness(self) -> float:
        """Calculate bridging effectiveness."""
        return np.mean([
            self.bridging_score,
            self.boundary_spanning,
            self.information_brokerage,
            self.diversity_promotion
        ])


@dataclass
class ReinforcementSpiral:
    """
    Self-reinforcing information spiral within echo chamber.
    """
    spiral_id: str
    
    # Spiral structure
    participants: List[str]  # Ordered participants
    spiral_path: List[Tuple[str, str, float]]  # Edges with weights
    
    # Characteristics
    amplification_factor: float  # How much signal amplifies per cycle
    cycle_time: float  # Time for one complete cycle
    stability: float  # Stability of the spiral
    
    # Content
    reinforced_frames: Set[str]
    reinforced_entities: Set[str]
    reinforced_narratives: List[str]
    
    # Effects
    polarization_effect: float
    reality_distortion: float  # Deviation from baseline
    
    def get_spiral_strength(self) -> float:
        """Calculate spiral strength."""
        return self.amplification_factor * self.stability


class EchoChamberDetector(BaseDetector):
    """
    Detects and analyzes echo chambers and filter bubbles.
    """
    
    def __init__(self, 
                 context: DetectionContext,
                 config: Optional[DetectorConfig] = None):
        """Initialize echo chamber detector."""
        super().__init__(context, config)
        self.name = "EchoChamberDetector"
        
        # Initialize component detectors
        self.cascade_detector = CascadeDetector(context, config, None)  # burst_detector=None
        self.multi_frame_detector = MultiFrameDetector(context, config, self.cascade_detector)
        self.cross_media_tracker = CrossMediaTracker(context, config, self.cascade_detector, self.multi_frame_detector)
        self.sequence_detector = SequenceDetector(context, config)
        
        # Initialize metrics with required parameters
        self.network_metrics = ScientificNetworkMetrics(
            source_index=context.source_index,
            entity_index=context.entity_index
        )
        self.diversity_metrics = DiversityMetrics(
            frame_index=context.frame_index,
            source_index=context.source_index,
            emotion_index=context.emotion_index,
            entity_index=context.entity_index,
            geographic_index=context.geographic_index
        )
        self.convergence_metrics = ConvergenceMetrics(
            source_index=context.source_index,
            entity_index=context.entity_index,
            emotion_index=context.emotion_index,
            temporal_index=context.temporal_index,
            frame_index=context.frame_index
        )
        
        # Detection parameters
        self.min_chamber_size = 5  # Minimum members for echo chamber
        self.isolation_threshold = 0.6  # Threshold for isolation
        self.homogeneity_threshold = 0.7  # Threshold for content homogeneity
        self.min_duration_days = 7  # Minimum duration for stable chamber
        
        # Storage
        self.detected_chambers = []
        self.filter_bubbles = []
        self.information_silos = []
        self.homophily_patterns = []
        self.bridge_actors = []
        self.reinforcement_spirals = []
        
        logger.info(f"EchoChamberDetector initialized with config: {config}")
    
    def detect(self, **kwargs) -> Dict[str, Any]:
        """
        Main detection method for echo chambers.
        """
        logger.info("Starting echo chamber detection...")
        
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
        logger.info(f"Working with {len(cascades)} cascades and {len(sequences)} sequences")
        
        # Get media profiles and clusters from cross_media_tracker results
        cross_media_results = kwargs.get('cross_media_results', {})
        if not cross_media_results:
            cross_media_results = self.cross_media_tracker.detect(cascades=cascades)
        
        media_profiles = cross_media_results.get('media_profiles', {})
        media_clusters = cross_media_results.get('media_clusters', [])
        logger.info(f"Using {len(media_profiles)} media profiles and {len(media_clusters)} clusters")
        
        # Build interaction networks
        media_network = self._build_media_network(cascades, media_profiles)
        journalist_network = self._build_journalist_network(cascades)
        content_network = self._build_content_network(cascades)
        
        # Detect echo chambers using multiple methods
        chambers = []
        
        # Method 1: Network community detection
        network_chambers = self._detect_network_chambers(
            media_network, journalist_network, cascades
        )
        chambers.extend(network_chambers)
        logger.info(f"Found {len(network_chambers)} network-based chambers")
        
        # Method 2: Content homogeneity detection
        content_chambers = self._detect_content_chambers(
            content_network, cascades, media_profiles
        )
        chambers.extend(content_chambers)
        logger.info(f"Found {len(content_chambers)} content-based chambers")
        
        # Method 3: Temporal echo patterns
        temporal_chambers = self._detect_temporal_chambers(
            sequences, cascades, media_profiles
        )
        chambers.extend(temporal_chambers)
        logger.info(f"Found {len(temporal_chambers)} temporal chambers")
        
        # Method 4: Frame-based chambers
        frame_chambers = self._detect_frame_chambers(
            cascades, media_profiles, context
        )
        chambers.extend(frame_chambers)
        logger.info(f"Found {len(frame_chambers)} frame-based chambers")
        
        # Merge and deduplicate chambers
        unique_chambers = self._merge_chambers(chambers)
        logger.info(f"After merging: {len(unique_chambers)} unique chambers")
        
        # Analyze each chamber in detail
        analyzed_chambers = []
        for chamber in tqdm(unique_chambers, desc="Analyzing chambers"):
            # Calculate detailed metrics
            chamber = self._analyze_chamber_structure(chamber, media_network)
            chamber = self._analyze_information_flow(chamber, cascades)
            chamber = self._analyze_diversity_metrics(chamber, cascades)
            chamber = self._identify_reinforcement_patterns(chamber, sequences)
            
            # Validate chamber
            if self._validate_chamber(chamber):
                analyzed_chambers.append(chamber)
                
                # Detect filter bubbles within chamber
                bubbles = self._detect_filter_bubbles(chamber, cascades)
                self.filter_bubbles.extend(bubbles)
                
                # Identify information silos
                silos = self._identify_information_silos(chamber, media_network)
                self.information_silos.extend(silos)
                
                # Detect homophily patterns
                patterns = self._detect_homophily_patterns(chamber, media_network)
                self.homophily_patterns.extend(patterns)
                
                # Identify bridge actors
                bridges = self._identify_bridge_actors(chamber, media_network)
                self.bridge_actors.extend(bridges)
                
                # Find reinforcement spirals
                spirals = self._find_reinforcement_spirals(chamber, sequences)
                self.reinforcement_spirals.extend(spirals)
        
        logger.info(f"Detected {len(analyzed_chambers)} valid echo chambers")
        logger.info(f"Found {len(self.filter_bubbles)} filter bubbles")
        logger.info(f"Identified {len(self.information_silos)} information silos")
        logger.info(f"Detected {len(self.homophily_patterns)} homophily patterns")
        logger.info(f"Found {len(self.bridge_actors)} bridge actors")
        logger.info(f"Identified {len(self.reinforcement_spirals)} reinforcement spirals")
        
        # Store results
        self.detected_chambers = analyzed_chambers
        
        return analyzed_chambers
    
    def _build_media_network(self, cascades: List[CompleteCascade],
                            media_profiles: Dict[str, MediaProfile]) -> nx.Graph:
        """Build network of media interactions."""
        G = nx.Graph()
        
        # Add nodes with attributes
        for media_id, profile in media_profiles.items():
            G.add_node(media_id,
                      total_articles=profile.total_articles,
                      influence_score=profile.influence_score,
                      behavioral_type=profile.get_behavioral_type())
        
        # Add edges based on co-participation in cascades
        media_pairs = defaultdict(int)
        for cascade in cascades:
            media_list = list(cascade.media_involved)
            for i in range(len(media_list)):
                for j in range(i+1, len(media_list)):
                    pair = tuple(sorted([media_list[i], media_list[j]]))
                    media_pairs[pair] += cascade.intensity_score
        
        # Add weighted edges
        for (m1, m2), weight in media_pairs.items():
            if m1 in G and m2 in G:
                G.add_edge(m1, m2, weight=weight)
        
        return G
    
    def _build_journalist_network(self, cascades: List[CompleteCascade]) -> nx.Graph:
        """Build network of journalist interactions."""
        G = nx.Graph()
        
        # Collect journalist data
        journalist_data = defaultdict(lambda: {
            'articles': 0,
            'cascades': set(),
            'media': set(),
            'frames': Counter()
        })
        
        for cascade in cascades:
            for journalist in cascade.journalists_involved:
                journalist_data[journalist]['articles'] += 1
                journalist_data[journalist]['cascades'].add(cascade.cascade_id)
                journalist_data[journalist]['media'].update(cascade.media_involved)
                journalist_data[journalist]['frames'][cascade.primary_frame] += 1
        
        # Add nodes
        for journalist, data in journalist_data.items():
            G.add_node(journalist,
                      articles=data['articles'],
                      n_cascades=len(data['cascades']),
                      n_media=len(data['media']))
        
        # Add edges based on collaboration
        journalist_pairs = defaultdict(int)
        for cascade in cascades:
            journalists = list(cascade.journalists_involved)
            for i in range(len(journalists)):
                for j in range(i+1, len(journalists)):
                    pair = tuple(sorted([journalists[i], journalists[j]]))
                    journalist_pairs[pair] += 1
        
        for (j1, j2), weight in journalist_pairs.items():
            if j1 in G and j2 in G:
                G.add_edge(j1, j2, weight=weight)
        
        return G
    
    def _build_content_network(self, cascades: List[CompleteCascade]) -> nx.Graph:
        """Build network based on content similarity."""
        G = nx.Graph()
        
        # Add cascades as nodes
        for cascade in cascades:
            G.add_node(cascade.cascade_id,
                      frame=cascade.primary_frame,
                      intensity=cascade.intensity_score,
                      entities=set(e['entity'] for e in cascade.entities_involved[:20]))
        
        # Add edges based on content similarity
        for i, c1 in enumerate(cascades):
            for j, c2 in enumerate(cascades[i+1:], i+1):
                # Calculate content similarity
                e1 = G.nodes[c1.cascade_id]['entities']
                e2 = G.nodes[c2.cascade_id]['entities']
                
                if e1 and e2:
                    similarity = len(e1 & e2) / len(e1 | e2)
                    
                    if similarity > 0.3:  # Threshold for connection
                        G.add_edge(c1.cascade_id, c2.cascade_id, weight=similarity)
        
        return G
    
    def _detect_network_chambers(self, media_network: nx.Graph,
                                journalist_network: nx.Graph,
                                cascades: List[CompleteCascade]) -> List[EchoChamber]:
        """Detect echo chambers using network community detection."""
        chambers = []
        
        if media_network.number_of_nodes() < self.min_chamber_size:
            return chambers
        
        # Detect communities using Louvain method
        communities = community_louvain.best_partition(media_network)
        
        # Group nodes by community
        community_groups = defaultdict(set)
        for node, comm_id in communities.items():
            community_groups[comm_id].add(node)
        
        # Analyze each community
        for comm_id, members in community_groups.items():
            if len(members) < self.min_chamber_size:
                continue
            
            # Create subgraph for this community
            subgraph = media_network.subgraph(members)
            
            # Calculate basic metrics
            density = nx.density(subgraph)
            if density < 0.3:  # Too sparse to be echo chamber
                continue
            
            # Calculate clustering coefficient
            clustering = nx.average_clustering(subgraph, weight='weight')
            
            # Calculate modularity
            modularity = self._calculate_modularity(media_network, members)
            
            # Find formation date (earliest cascade involving members)
            formation_date = datetime.now()
            peak_date = datetime.now()
            related_cascades = []
            
            for cascade in cascades:
                if cascade.media_involved & members:
                    related_cascades.append(cascade)
                    if cascade.start_date < formation_date:
                        formation_date = cascade.start_date
                    if cascade.peak_date < peak_date:
                        peak_date = cascade.peak_date
            
            if not related_cascades:
                continue
            
            # Identify core vs peripheral members
            centralities = nx.degree_centrality(subgraph)
            sorted_members = sorted(centralities.items(), key=lambda x: x[1], reverse=True)
            core_size = max(3, len(members) // 3)
            core_members = {m for m, _ in sorted_members[:core_size]}
            peripheral_members = members - core_members
            
            # Calculate isolation score
            internal_edges = subgraph.number_of_edges()
            external_edges = sum(1 for u, v in media_network.edges(members)
                               if u in members and v not in members)
            isolation_score = internal_edges / (internal_edges + external_edges) \
                            if (internal_edges + external_edges) > 0 else 0
            
            # Get dominant content
            dominant_frames = Counter()
            dominant_entities = Counter()
            for cascade in related_cascades:
                dominant_frames[cascade.primary_frame] += cascade.intensity_score
                for entity in cascade.entities_involved[:10]:
                    dominant_entities[entity['entity']] += cascade.intensity_score
            
            chamber = EchoChamber(
                chamber_id=str(uuid.uuid4()),
                chamber_type='media',
                core_members=core_members,
                peripheral_members=peripheral_members,
                total_members=len(members),
                formation_date=formation_date,
                peak_date=peak_date,
                current_date=datetime.now(),
                duration_days=(datetime.now() - formation_date).days,
                density=density,
                clustering_coefficient=clustering,
                modularity=modularity,
                conductance=self._calculate_conductance(media_network, members),
                internal_flow=0.0,  # Will be calculated later
                external_flow=0.0,  # Will be calculated later
                isolation_score=isolation_score,
                permeability=1 - isolation_score,
                dominant_frames=list(dominant_frames.keys())[:3],
                dominant_entities=[(e, s) for e, s in dominant_entities.most_common(5)],
                dominant_narratives=[],  # Will be extracted later
                content_homogeneity=0.0,  # Will be calculated later
                frame_diversity=0.0,  # Will be calculated later
                entity_diversity=0.0,  # Will be calculated later
                source_diversity=0.0,  # Will be calculated later
                viewpoint_diversity=0.0,  # Will be calculated later
                internal_network=subgraph,
                bridge_nodes=[],  # Will be identified later
                structural_holes=[],  # Will be identified later
                reinforcement_cycles=[],  # Will be found later
                amplification_rate=0.0,  # Will be calculated later
                echo_strength=0.0,  # Will be calculated later
                growth_rate=0.0,  # Will be calculated later
                stability=0.0,  # Will be calculated later
                lifecycle_stage='stable',  # Will be determined later
                cascades=related_cascades,
                cascade_participation={},  # Will be calculated later
                significance=0.0,  # Will be calculated later
                random_probability=0.0,  # Will be calculated later
                robustness=0.0  # Will be calculated later
            )
            
            chambers.append(chamber)
        
        return chambers
    
    def _detect_content_chambers(self, content_network: nx.Graph,
                                cascades: List[CompleteCascade],
                                media_profiles: Dict[str, MediaProfile]) -> List[EchoChamber]:
        """Detect echo chambers based on content homogeneity."""
        chambers = []
        
        if content_network.number_of_nodes() < 3:
            return chambers
        
        # Spectral clustering on content network
        if content_network.number_of_edges() > 0:
            # Create adjacency matrix
            nodes = list(content_network.nodes())
            n = len(nodes)
            adj_matrix = np.zeros((n, n))
            
            for i, n1 in enumerate(nodes):
                for j, n2 in enumerate(nodes):
                    if content_network.has_edge(n1, n2):
                        adj_matrix[i, j] = content_network[n1][n2]['weight']
            
            # Apply spectral clustering
            n_clusters = min(10, n // 5)
            if n_clusters >= 2:
                clustering = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    random_state=42
                )
                labels = clustering.fit_predict(adj_matrix)
                
                # Group cascades by cluster
                clusters = defaultdict(list)
                for idx, label in enumerate(labels):
                    cascade_id = nodes[idx]
                    cascade = next((c for c in cascades if c.cascade_id == cascade_id), None)
                    if cascade:
                        clusters[label].append(cascade)
                
                # Analyze each cluster
                for cluster_id, cluster_cascades in clusters.items():
                    if len(cluster_cascades) < 3:
                        continue
                    
                    # Get involved media
                    all_media = set()
                    for cascade in cluster_cascades:
                        all_media.update(cascade.media_involved)
                    
                    if len(all_media) < self.min_chamber_size:
                        continue
                    
                    # Calculate content homogeneity
                    frames = Counter()
                    entities = set()
                    for cascade in cluster_cascades:
                        frames[cascade.primary_frame] += 1
                        entities.update(e['entity'] for e in cascade.entities_involved[:10])
                    
                    # Frame homogeneity
                    frame_homogeneity = max(frames.values()) / sum(frames.values()) \
                                      if frames else 0
                    
                    if frame_homogeneity < self.homogeneity_threshold:
                        continue
                    
                    # Create chamber
                    chamber = self._create_content_chamber(
                        cluster_cascades, all_media, frame_homogeneity
                    )
                    chambers.append(chamber)
        
        return chambers
    
    def _detect_temporal_chambers(self, sequences: List[CascadeSequence],
                                 cascades: List[CompleteCascade],
                                 media_profiles: Dict[str, MediaProfile]) -> List[EchoChamber]:
        """Detect echo chambers from temporal sequence patterns."""
        chambers = []
        
        for sequence in sequences:
            # Check for echo patterns in sequence
            if sequence.predictability > 0.7 and sequence.coherence_score > 0.7:
                # High predictability and coherence suggest echo chamber
                
                # Get all involved media
                all_media = set()
                for cascade in sequence.cascades:
                    all_media.update(cascade.media_involved)
                
                if len(all_media) < self.min_chamber_size:
                    continue
                
                # Check for reinforcement patterns
                if sequence.autocorrelation and any(corr > 0.5 for corr in sequence.autocorrelation.values()):
                    # Strong autocorrelation indicates echo effect
                    
                    chamber = self._create_temporal_chamber(
                        sequence, all_media
                    )
                    chambers.append(chamber)
        
        return chambers
    
    def _detect_frame_chambers(self, cascades: List[CompleteCascade],
                              media_profiles: Dict[str, MediaProfile],
                              context: DetectionContext) -> List[EchoChamber]:
        """Detect frame-specific echo chambers."""
        chambers = []
        
        # Group cascades by frame
        frame_cascades = defaultdict(list)
        for cascade in cascades:
            frame_cascades[cascade.primary_frame].append(cascade)
        
        # Analyze each frame
        for frame, frame_cascade_list in frame_cascades.items():
            if len(frame_cascade_list) < 5:
                continue
            
            # Get media covering this frame
            frame_media = set()
            for cascade in frame_cascade_list:
                frame_media.update(cascade.media_involved)
            
            if len(frame_media) < self.min_chamber_size:
                continue
            
            # Calculate frame loyalty (how exclusively media cover this frame)
            media_frame_loyalty = {}
            for media in frame_media:
                if media in media_profiles:
                    profile = media_profiles[media]
                    if profile.dominant_frame == frame:
                        frame_pref = profile.frame_preferences.get(frame, 0)
                        media_frame_loyalty[media] = frame_pref
            
            # Check if there's strong frame loyalty
            if media_frame_loyalty:
                avg_loyalty = np.mean(list(media_frame_loyalty.values()))
                
                if avg_loyalty > 0.6:  # Strong frame preference
                    chamber = self._create_frame_chamber(
                        frame, frame_cascade_list, frame_media, avg_loyalty
                    )
                    chambers.append(chamber)
        
        return chambers
    
    def _merge_chambers(self, chambers: List[EchoChamber]) -> List[EchoChamber]:
        """Merge overlapping echo chambers."""
        if len(chambers) <= 1:
            return chambers
        
        # Calculate similarity between chambers
        merged = []
        used = set()
        
        for i, c1 in enumerate(chambers):
            if i in used:
                continue
            
            # Start with current chamber
            merged_chamber = c1
            merged_members = c1.core_members | c1.peripheral_members
            merged_cascades = list(c1.cascades)
            
            # Look for similar chambers to merge
            for j, c2 in enumerate(chambers[i+1:], i+1):
                if j in used:
                    continue
                
                c2_members = c2.core_members | c2.peripheral_members
                
                # Calculate overlap
                overlap = len(merged_members & c2_members) / \
                         len(merged_members | c2_members) if merged_members | c2_members else 0
                
                if overlap > 0.5:  # Significant overlap
                    # Merge chambers
                    merged_members.update(c2_members)
                    merged_cascades.extend(c2.cascades)
                    used.add(j)
                    
                    # Update merged chamber properties
                    merged_chamber.total_members = len(merged_members)
                    merged_chamber.cascades = list(set(merged_cascades))
            
            merged.append(merged_chamber)
            used.add(i)
        
        return merged
    
    def _create_content_chamber(self, cascades: List[CompleteCascade],
                               media: Set[str],
                               homogeneity: float) -> EchoChamber:
        """Create echo chamber from content cluster."""
        # Calculate temporal bounds
        formation_date = min(c.start_date for c in cascades)
        peak_date = max(c.peak_date for c in cascades if c.intensity_score > 0.5)
        
        # Determine core members (most active)
        media_activity = Counter()
        for cascade in cascades:
            for m in cascade.media_involved:
                media_activity[m] += cascade.intensity_score
        
        sorted_media = media_activity.most_common()
        core_size = max(3, len(media) // 3)
        core_members = {m for m, _ in sorted_media[:core_size]}
        peripheral_members = media - core_members
        
        # Get dominant content
        frames = Counter()
        entities = Counter()
        for cascade in cascades:
            frames[cascade.primary_frame] += cascade.intensity_score
            for entity in cascade.entities_involved[:10]:
                entities[entity['entity']] += 1
        
        return EchoChamber(
            chamber_id=str(uuid.uuid4()),
            chamber_type='topical',
            core_members=core_members,
            peripheral_members=peripheral_members,
            total_members=len(media),
            formation_date=formation_date,
            peak_date=peak_date,
            current_date=datetime.now(),
            duration_days=(datetime.now() - formation_date).days,
            density=0.0,  # Will be calculated
            clustering_coefficient=0.0,  # Will be calculated
            modularity=0.0,  # Will be calculated
            conductance=0.0,  # Will be calculated
            internal_flow=0.0,
            external_flow=0.0,
            isolation_score=homogeneity,
            permeability=1 - homogeneity,
            dominant_frames=list(frames.keys())[:3],
            dominant_entities=[(e, c) for e, c in entities.most_common(5)],
            dominant_narratives=[],
            content_homogeneity=homogeneity,
            frame_diversity=0.0,
            entity_diversity=0.0,
            source_diversity=0.0,
            viewpoint_diversity=0.0,
            internal_network=nx.Graph(),
            bridge_nodes=[],
            structural_holes=[],
            reinforcement_cycles=[],
            amplification_rate=0.0,
            echo_strength=0.0,
            growth_rate=0.0,
            stability=0.0,
            lifecycle_stage='stable',
            cascades=cascades,
            cascade_participation={},
            significance=0.0,
            random_probability=0.0,
            robustness=0.0
        )
    
    def _create_temporal_chamber(self, sequence: CascadeSequence,
                                media: Set[str]) -> EchoChamber:
        """Create echo chamber from temporal sequence."""
        # Use sequence properties
        formation_date = sequence.start_date
        peak_date = max(c.peak_date for c in sequence.cascades 
                       if c.intensity_score == max(sc.intensity_score for sc in sequence.cascades))
        
        # Determine members based on consistency
        core_members = set()
        peripheral_members = set()
        
        for m in media:
            if m in sequence.media_consistency:
                if sequence.media_consistency[m] > 0.7:
                    core_members.add(m)
                else:
                    peripheral_members.add(m)
        
        # Get dominant content from sequence
        all_frames = Counter()
        all_entities = Counter()
        for cascade in sequence.cascades:
            all_frames[cascade.primary_frame] += cascade.intensity_score
            for entity in cascade.entities_involved[:10]:
                all_entities[entity['entity']] += 1
        
        return EchoChamber(
            chamber_id=str(uuid.uuid4()),
            chamber_type='temporal',
            core_members=core_members,
            peripheral_members=peripheral_members,
            total_members=len(media),
            formation_date=formation_date,
            peak_date=peak_date,
            current_date=datetime.now(),
            duration_days=sequence.duration_days,
            density=0.0,
            clustering_coefficient=0.0,
            modularity=0.0,
            conductance=0.0,
            internal_flow=sequence.autocorrelation.get(1, 0) if sequence.autocorrelation else 0,
            external_flow=0.0,
            isolation_score=sequence.predictability,
            permeability=1 - sequence.predictability,
            dominant_frames=list(all_frames.keys())[:3],
            dominant_entities=[(e, c) for e, c in all_entities.most_common(5)],
            dominant_narratives=[],
            content_homogeneity=sequence.coherence_score,
            frame_diversity=0.0,
            entity_diversity=0.0,
            source_diversity=0.0,
            viewpoint_diversity=0.0,
            internal_network=nx.Graph(),
            bridge_nodes=[],
            structural_holes=[],
            reinforcement_cycles=[],
            amplification_rate=0.0,
            echo_strength=sequence.predictability * sequence.coherence_score,
            growth_rate=0.0,
            stability=sequence.coherence_score,
            lifecycle_stage='stable',
            cascades=sequence.cascades,
            cascade_participation={},
            significance=sequence.significance,
            random_probability=0.0,
            robustness=0.0
        )
    
    def _create_frame_chamber(self, frame: str,
                             cascades: List[CompleteCascade],
                             media: Set[str],
                             loyalty: float) -> EchoChamber:
        """Create frame-based echo chamber."""
        formation_date = min(c.start_date for c in cascades)
        peak_date = max(c.peak_date for c in cascades 
                       if c.intensity_score == max(sc.intensity_score for sc in cascades))
        
        # All frame-loyal media are core members
        core_members = media
        peripheral_members = set()
        
        # Get entities specific to this frame
        frame_entities = Counter()
        for cascade in cascades:
            for entity in cascade.entities_involved[:10]:
                frame_entities[entity['entity']] += 1
        
        return EchoChamber(
            chamber_id=str(uuid.uuid4()),
            chamber_type='ideological',
            core_members=core_members,
            peripheral_members=peripheral_members,
            total_members=len(media),
            formation_date=formation_date,
            peak_date=peak_date,
            current_date=datetime.now(),
            duration_days=(datetime.now() - formation_date).days,
            density=loyalty,
            clustering_coefficient=loyalty,
            modularity=0.0,
            conductance=0.0,
            internal_flow=loyalty,
            external_flow=1 - loyalty,
            isolation_score=loyalty,
            permeability=1 - loyalty,
            dominant_frames=[frame],
            dominant_entities=[(e, c) for e, c in frame_entities.most_common(5)],
            dominant_narratives=[],
            content_homogeneity=0.9,  # High for frame chamber
            frame_diversity=0.1,  # Low diversity
            entity_diversity=0.0,
            source_diversity=0.0,
            viewpoint_diversity=0.0,
            internal_network=nx.Graph(),
            bridge_nodes=[],
            structural_holes=[],
            reinforcement_cycles=[],
            amplification_rate=0.0,
            echo_strength=loyalty,
            growth_rate=0.0,
            stability=loyalty,
            lifecycle_stage='stable',
            cascades=cascades,
            cascade_participation={},
            significance=0.0,
            random_probability=0.0,
            robustness=0.0
        )
    
    def _analyze_chamber_structure(self, chamber: EchoChamber,
                                  media_network: nx.Graph) -> EchoChamber:
        """Analyze detailed structure of echo chamber."""
        all_members = chamber.core_members | chamber.peripheral_members
        
        if media_network.number_of_nodes() > 0 and all_members:
            # Create subgraph
            members_in_network = all_members & set(media_network.nodes())
            if members_in_network:
                subgraph = media_network.subgraph(members_in_network)
                chamber.internal_network = subgraph
                
                # Recalculate metrics on actual network
                if subgraph.number_of_nodes() > 1:
                    chamber.density = nx.density(subgraph)
                    chamber.clustering_coefficient = nx.average_clustering(subgraph)
                
                # Identify bridge nodes (high betweenness)
                if subgraph.number_of_nodes() > 2:
                    betweenness = nx.betweenness_centrality(subgraph)
                    sorted_nodes = sorted(betweenness.items(), 
                                        key=lambda x: x[1], reverse=True)
                    chamber.bridge_nodes = sorted_nodes[:5]
                
                # Find structural holes
                holes = []
                for u in subgraph.nodes():
                    neighbors_u = set(subgraph.neighbors(u))
                    for v in subgraph.nodes():
                        if u != v and v not in neighbors_u:
                            neighbors_v = set(subgraph.neighbors(v))
                            common = neighbors_u & neighbors_v
                            if len(common) >= 2:  # Potential hole
                                holes.append((u, v))
                chamber.structural_holes = holes[:10]
        
        return chamber
    
    def _analyze_information_flow(self, chamber: EchoChamber,
                                 cascades: List[CompleteCascade]) -> EchoChamber:
        """Analyze information flow patterns in chamber."""
        all_members = chamber.core_members | chamber.peripheral_members
        
        # Calculate internal vs external flow
        internal_flow = 0
        external_flow = 0
        
        for cascade in chamber.cascades:
            cascade_media = cascade.media_involved
            internal_media = cascade_media & all_members
            external_media = cascade_media - all_members
            
            internal_flow += len(internal_media) * cascade.intensity_score
            external_flow += len(external_media) * cascade.intensity_score
        
        total_flow = internal_flow + external_flow
        if total_flow > 0:
            chamber.internal_flow = internal_flow / total_flow
            chamber.external_flow = external_flow / total_flow
        
        # Calculate cascade participation
        for member in all_members:
            participation = sum(1 for c in chamber.cascades if member in c.media_involved)
            chamber.cascade_participation[member] = participation / len(chamber.cascades) \
                                                    if chamber.cascades else 0
        
        # Calculate amplification rate
        if len(chamber.cascades) > 1:
            intensities = [c.intensity_score for c in chamber.cascades]
            chamber.amplification_rate = np.mean(np.diff(intensities)) if len(intensities) > 1 else 0
        
        return chamber
    
    def _analyze_diversity_metrics(self, chamber: EchoChamber,
                                  cascades: List[CompleteCascade]) -> EchoChamber:
        """Calculate diversity metrics for the chamber."""
        # Frame diversity
        frame_counts = Counter()
        for cascade in chamber.cascades:
            frame_counts[cascade.primary_frame] += 1
        
        if frame_counts:
            frame_probs = np.array(list(frame_counts.values())) / sum(frame_counts.values())
            chamber.frame_diversity = -np.sum(frame_probs * np.log2(frame_probs + 1e-10))
        
        # Entity diversity
        entity_counts = Counter()
        for cascade in chamber.cascades:
            for entity in cascade.entities_involved[:20]:
                entity_counts[entity['entity']] += 1
        
        if entity_counts:
            entity_probs = np.array(list(entity_counts.values())) / sum(entity_counts.values())
            chamber.entity_diversity = -np.sum(entity_probs * np.log2(entity_probs + 1e-10))
        
        # Source diversity (media diversity)
        all_members = chamber.core_members | chamber.peripheral_members
        if all_members:
            chamber.source_diversity = len(all_members) / 100  # Normalize by typical max
        
        # Viewpoint diversity (approximated by frame diversity)
        chamber.viewpoint_diversity = chamber.frame_diversity / np.log2(8)  # Normalize by max frames
        
        return chamber
    
    def _identify_reinforcement_patterns(self, chamber: EchoChamber,
                                        sequences: List[CascadeSequence]) -> EchoChamber:
        """Identify reinforcement patterns in the chamber."""
        all_members = chamber.core_members | chamber.peripheral_members
        
        # Find reinforcement cycles in internal network
        if chamber.internal_network and chamber.internal_network.number_of_nodes() > 2:
            try:
                cycles = list(nx.simple_cycles(chamber.internal_network.to_directed()))
                chamber.reinforcement_cycles = [c for c in cycles if len(c) >= 3][:10]
            except:
                chamber.reinforcement_cycles = []
        
        # Calculate echo strength from sequences
        echo_strengths = []
        for sequence in sequences:
            # Check if sequence involves chamber members
            sequence_media = set()
            for cascade in sequence.cascades:
                sequence_media.update(cascade.media_involved)
            
            overlap = len(sequence_media & all_members) / len(all_members) if all_members else 0
            
            if overlap > 0.5:  # Sequence involves chamber
                # Use predictability and autocorrelation as echo indicators
                echo_strength = sequence.predictability * sequence.coherence_score
                echo_strengths.append(echo_strength)
        
        if echo_strengths:
            chamber.echo_strength = np.mean(echo_strengths)
        
        return chamber
    
    def _validate_chamber(self, chamber: EchoChamber) -> bool:
        """Validate detected echo chamber."""
        # Size check
        if chamber.total_members < self.min_chamber_size:
            return False
        
        # Duration check
        if chamber.duration_days < self.min_duration_days:
            return False
        
        # Isolation check
        if chamber.isolation_score < 0.3:  # Too open
            return False
        
        # Statistical significance
        significance = self._calculate_chamber_significance(chamber)
        chamber.significance = significance
        
        # Random probability
        random_prob = self._calculate_random_probability(chamber)
        chamber.random_probability = random_prob
        
        # Robustness
        robustness = self._calculate_robustness(chamber)
        chamber.robustness = robustness
        
        return significance > 0.95 and random_prob < 0.05
    
    def _calculate_chamber_significance(self, chamber: EchoChamber) -> float:
        """Calculate statistical significance of chamber."""
        # Based on multiple factors
        factors = []
        
        # Density significance
        if chamber.density > 0:
            # Compare to random graph
            expected_density = 0.1  # Expected for random
            z_score = (chamber.density - expected_density) / (expected_density * 0.1)
            factors.append(1 / (1 + np.exp(-z_score)))
        
        # Clustering significance
        if chamber.clustering_coefficient > 0:
            expected_clustering = 0.1
            z_score = (chamber.clustering_coefficient - expected_clustering) / (expected_clustering * 0.1)
            factors.append(1 / (1 + np.exp(-z_score)))
        
        # Isolation significance
        factors.append(chamber.isolation_score)
        
        # Homogeneity significance
        factors.append(chamber.content_homogeneity)
        
        return np.mean(factors) if factors else 0
    
    def _calculate_random_probability(self, chamber: EchoChamber) -> float:
        """Calculate probability of chamber occurring randomly."""
        # Probability based on configuration model
        n = chamber.total_members
        m = len(chamber.cascades)
        
        # Probability of this configuration
        if n > 0 and m > 0:
            # Binomial probability
            p_random_group = 1 / (2 ** n)  # Probability of random grouping
            p_cascade_alignment = 1 / m  # Probability of cascade alignment
            
            return p_random_group * p_cascade_alignment
        
        return 1.0
    
    def _calculate_robustness(self, chamber: EchoChamber) -> float:
        """Calculate robustness of chamber to perturbation."""
        if not chamber.internal_network or chamber.internal_network.number_of_nodes() < 3:
            return 0.0
        
        # Node removal robustness
        G = chamber.internal_network.copy()
        initial_connected = nx.is_connected(G)
        
        if not initial_connected:
            # Use largest component
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        
        # Remove nodes and check connectivity
        robustness_scores = []
        for _ in range(min(3, G.number_of_nodes() // 2)):
            if G.number_of_nodes() > 2:
                # Remove highest degree node
                node = max(G.degree(), key=lambda x: x[1])[0]
                G.remove_node(node)
                
                if G.number_of_nodes() > 1:
                    largest_cc = max(nx.connected_components(G), key=len)
                    frac = len(largest_cc) / chamber.total_members
                    robustness_scores.append(frac)
        
        return np.mean(robustness_scores) if robustness_scores else 0.0
    
    def _calculate_modularity(self, network: nx.Graph, community: Set[str]) -> float:
        """Calculate modularity of a community."""
        if network.number_of_edges() == 0:
            return 0.0
        
        m = network.number_of_edges()
        Q = 0
        
        for u in community:
            for v in community:
                if u != v and u in network and v in network:
                    # Actual edge
                    A_uv = 1 if network.has_edge(u, v) else 0
                    
                    # Expected edges
                    k_u = network.degree(u)
                    k_v = network.degree(v)
                    expected = (k_u * k_v) / (2 * m)
                    
                    Q += (A_uv - expected)
        
        return Q / (2 * m) if m > 0 else 0
    
    def _calculate_conductance(self, network: nx.Graph, community: Set[str]) -> float:
        """Calculate conductance of a community."""
        if not community or not network:
            return 0.0
        
        # Count edges leaving the community
        cut_edges = 0
        internal_edges = 0
        
        for u in community:
            if u in network:
                for v in network.neighbors(u):
                    if v in community:
                        internal_edges += 1
                    else:
                        cut_edges += 1
        
        # Conductance
        total_edges = internal_edges + cut_edges
        if total_edges > 0:
            return cut_edges / total_edges
        
        return 0.0
    
    def _detect_filter_bubbles(self, chamber: EchoChamber,
                              cascades: List[CompleteCascade]) -> List[FilterBubble]:
        """Detect filter bubbles within echo chamber."""
        bubbles = []
        
        # Analyze information filtering
        all_entities = set()
        chamber_entities = set()
        
        for cascade in cascades:
            for entity in cascade.entities_involved:
                all_entities.add(entity['entity'])
                if cascade in chamber.cascades:
                    chamber_entities.add(entity['entity'])
        
        # Calculate filtering
        excluded_entities = all_entities - chamber_entities
        
        if excluded_entities and all_entities:
            filter_strength = len(excluded_entities) / len(all_entities)
            
            if filter_strength > 0.3:  # Significant filtering
                bubble = FilterBubble(
                    bubble_id=str(uuid.uuid4()),
                    bubble_type='self_selected',
                    media_outlets=chamber.core_members,
                    journalists=set(),  # Would need journalist data
                    included_topics=chamber_entities,
                    excluded_topics=excluded_entities,
                    filter_strength=filter_strength,
                    transparency=1 - chamber.isolation_score,
                    rigidity=chamber.isolation_score,
                    selectivity=filter_strength,
                    information_loss=filter_strength,
                    perspective_narrowing=1 - chamber.viewpoint_diversity,
                    confirmation_bias=chamber.content_homogeneity,
                    formation_time=chamber.formation_date,
                    strengthening_rate=chamber.amplification_rate
                )
                bubbles.append(bubble)
        
        return bubbles
    
    def _identify_information_silos(self, chamber: EchoChamber,
                                   media_network: nx.Graph) -> List[InformationSilo]:
        """Identify information silos within chamber."""
        silos = []
        
        if chamber.isolation_score > 0.7:  # High isolation
            # Chamber itself forms a silo
            all_members = chamber.core_members | chamber.peripheral_members
            
            # Identify gatekeepers (highest degree nodes)
            gatekeepers = []
            if chamber.internal_network and chamber.internal_network.number_of_nodes() > 0:
                degrees = chamber.internal_network.degree()
                sorted_nodes = sorted(degrees, key=lambda x: x[1], reverse=True)
                gatekeepers = [n for n, _ in sorted_nodes[:3]]
            
            # Shared beliefs (dominant frames)
            shared_beliefs = {}
            for frame in chamber.dominant_frames:
                shared_beliefs[frame] = chamber.content_homogeneity
            
            silo = InformationSilo(
                silo_id=str(uuid.uuid4()),
                members=all_members,
                gatekeepers=gatekeepers,
                internal_information=set(e for e, _ in chamber.dominant_entities),
                shared_beliefs=shared_beliefs,
                taboo_topics=set(),  # Would need to identify excluded topics
                boundary_strength=chamber.isolation_score,
                cross_silo_communication=chamber.external_flow,
                internal_cohesion=chamber.density,
                external_tension=1 - chamber.permeability
            )
            silos.append(silo)
        
        return silos
    
    def _detect_homophily_patterns(self, chamber: EchoChamber,
                                  media_network: nx.Graph) -> List[HomophilyPattern]:
        """Detect homophily patterns in chamber."""
        patterns = []
        
        if chamber.internal_network and chamber.internal_network.number_of_nodes() > 3:
            # Frame-based homophily
            frame_similarity = {}
            for frame in chamber.dominant_frames:
                frame_nodes = [n for n in chamber.internal_network.nodes()
                             if n in chamber.core_members]
                if frame_nodes:
                    frame_similarity[f"frame_{frame}"] = chamber.content_homogeneity
            
            if frame_similarity:
                # Create homophily network
                H = nx.Graph()
                for u in chamber.internal_network.nodes():
                    for v in chamber.internal_network.nodes():
                        if u != v and chamber.internal_network.has_edge(u, v):
                            H.add_edge(u, v, weight=1.0)
                
                # Find homogeneous groups
                if H.number_of_nodes() > 0:
                    communities = community_louvain.best_partition(H)
                    groups = defaultdict(set)
                    for node, comm in communities.items():
                        groups[comm].add(node)
                    
                    pattern = HomophilyPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type='frame',
                        similarity_dimensions=frame_similarity,
                        homophily_network=H,
                        homophily_strength=chamber.content_homogeneity,
                        homogeneous_groups=list(groups.values()),
                        group_similarities=[chamber.content_homogeneity] * len(groups),
                        segregation_index=chamber.isolation_score,
                        mixing_rate=chamber.external_flow,
                        strengthening=chamber.amplification_rate > 0,
                        evolution_rate=chamber.amplification_rate
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _identify_bridge_actors(self, chamber: EchoChamber,
                               media_network: nx.Graph) -> List[BridgeActor]:
        """Identify actors that bridge between chambers."""
        bridges = []
        
        # Find nodes with high betweenness in the full network
        all_members = chamber.core_members | chamber.peripheral_members
        
        for member in all_members:
            if member in media_network:
                # Calculate betweenness in full network
                betweenness = nx.betweenness_centrality(media_network, endpoints=True).get(member, 0)
                
                # Check connections outside chamber
                external_connections = sum(1 for neighbor in media_network.neighbors(member)
                                         if neighbor not in all_members)
                
                if external_connections > 0 and betweenness > 0.01:
                    bridge = BridgeActor(
                        actor_id=member,
                        actor_type='media',
                        betweenness_centrality=betweenness,
                        bridging_score=external_connections / media_network.degree(member),
                        boundary_spanning=external_connections / len(all_members),
                        connected_chambers=[chamber.chamber_id],  # Would need other chambers
                        information_brokerage=betweenness * external_connections,
                        cross_chamber_influence=betweenness,
                        diversity_promotion=1 - chamber.content_homogeneity,
                        cross_posting_rate=chamber.cascade_participation.get(member, 0),
                        frame_switching=chamber.frame_diversity
                    )
                    bridges.append(bridge)
        
        return bridges
    
    def _find_reinforcement_spirals(self, chamber: EchoChamber,
                                   sequences: List[CascadeSequence]) -> List[ReinforcementSpiral]:
        """Find reinforcement spirals in chamber."""
        spirals = []
        
        # Look for cycles in cascade sequences
        for sequence in sequences:
            # Check if sequence involves chamber
            sequence_media = set()
            for cascade in sequence.cascades:
                sequence_media.update(cascade.media_involved)
            
            all_members = chamber.core_members | chamber.peripheral_members
            overlap = len(sequence_media & all_members) / len(all_members) if all_members else 0
            
            if overlap > 0.5 and sequence.periodicity:  # Cyclic sequence in chamber
                # Build spiral path
                spiral_path = []
                for i in range(len(sequence.cascades) - 1):
                    c1 = sequence.cascades[i]
                    c2 = sequence.cascades[i+1]
                    
                    # Find connecting media
                    connecting = list(c1.media_involved & c2.media_involved)[:3]
                    for media in connecting:
                        spiral_path.append((
                            f"{c1.cascade_id}_{media}",
                            f"{c2.cascade_id}_{media}",
                            sequence.transitions[i][2] if i < len(sequence.transitions) else 0
                        ))
                
                if spiral_path:
                    # Calculate amplification
                    intensities = [c.intensity_score for c in sequence.cascades]
                    amplification = np.mean(np.diff(intensities)) if len(intensities) > 1 else 0
                    
                    spiral = ReinforcementSpiral(
                        spiral_id=str(uuid.uuid4()),
                        participants=list(sequence_media & all_members)[:10],
                        spiral_path=spiral_path[:20],
                        amplification_factor=1 + amplification,
                        cycle_time=sequence.periodicity if sequence.periodicity else 0,
                        stability=sequence.coherence_score,
                        reinforced_frames=set(chamber.dominant_frames),
                        reinforced_entities=set(e for e, _ in chamber.dominant_entities),
                        reinforced_narratives=chamber.dominant_narratives[:3],
                        polarization_effect=chamber.isolation_score * (1 + amplification),
                        reality_distortion=chamber.content_homogeneity * chamber.isolation_score
                    )
                    spirals.append(spiral)
        
        return spirals
    
    def get_chamber_summary(self, chamber: EchoChamber) -> Dict[str, Any]:
        """Get summary of detected echo chamber."""
        return {
            'chamber_id': chamber.chamber_id,
            'type': chamber.chamber_type,
            'isolation_level': chamber.get_isolation_level(),
            'homophily_score': chamber.get_homophily_score(),
            'members': {
                'core': len(chamber.core_members),
                'peripheral': len(chamber.peripheral_members),
                'total': chamber.total_members
            },
            'temporal': {
                'formation': chamber.formation_date.isoformat(),
                'duration_days': chamber.duration_days,
                'lifecycle': chamber.lifecycle_stage
            },
            'structure': {
                'density': chamber.density,
                'clustering': chamber.clustering_coefficient,
                'modularity': chamber.modularity,
                'isolation': chamber.isolation_score
            },
            'content': {
                'homogeneity': chamber.content_homogeneity,
                'dominant_frames': chamber.dominant_frames,
                'dominant_entities': chamber.dominant_entities[:3]
            },
            'diversity': {
                'frame': chamber.frame_diversity,
                'entity': chamber.entity_diversity,
                'viewpoint': chamber.viewpoint_diversity
            },
            'dynamics': {
                'echo_strength': chamber.echo_strength,
                'amplification_rate': chamber.amplification_rate,
                'stability': chamber.stability
            },
            'validation': {
                'significance': chamber.significance,
                'random_probability': chamber.random_probability,
                'robustness': chamber.robustness
            }
        }
    
    def export_results(self, output_path: str) -> None:
        """Export all echo chamber detection results."""
        import json
        
        export_data = {
            'metadata': {
                'detector': self.name,
                'n_chambers': len(self.detected_chambers),
                'n_filter_bubbles': len(self.filter_bubbles),
                'n_silos': len(self.information_silos),
                'n_homophily_patterns': len(self.homophily_patterns),
                'n_bridge_actors': len(self.bridge_actors),
                'n_spirals': len(self.reinforcement_spirals),
                'timestamp': datetime.now().isoformat()
            },
            'echo_chambers': [
                self.get_chamber_summary(chamber) for chamber in self.detected_chambers
            ],
            'filter_bubbles': [
                {
                    'bubble_id': b.bubble_id,
                    'type': b.bubble_type,
                    'severity': b.get_severity(),
                    'filter_strength': b.filter_strength,
                    'information_loss': b.information_loss,
                    'confirmation_bias': b.confirmation_bias
                } for b in self.filter_bubbles
            ],
            'information_silos': [
                {
                    'silo_id': s.silo_id,
                    'n_members': len(s.members),
                    'insularity': s.get_insularity(),
                    'boundary_strength': s.boundary_strength,
                    'internal_cohesion': s.internal_cohesion
                } for s in self.information_silos
            ],
            'bridge_actors': [
                {
                    'actor_id': b.actor_id,
                    'effectiveness': b.get_bridge_effectiveness(),
                    'betweenness': b.betweenness_centrality,
                    'bridging_score': b.bridging_score
                } for b in self.bridge_actors
            ],
            'reinforcement_spirals': [
                {
                    'spiral_id': s.spiral_id,
                    'strength': s.get_spiral_strength(),
                    'amplification': s.amplification_factor,
                    'polarization_effect': s.polarization_effect
                } for s in self.reinforcement_spirals
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported echo chamber analysis to {output_path}")
    
    def validate_detection(self, detection_result: Dict[str, Any]) -> bool:
        """
        Validate echo chamber detection results.
        
        Args:
            detection_result: Results from detect() method
            
        Returns:
            True if detection is valid, False otherwise
        """
        try:
            # Check required keys
            required_keys = [
                'echo_chambers', 'filter_bubbles', 'bridge_actors',
                'reinforcement_spirals', 'diversity_metrics', 'summary'
            ]
            
            for key in required_keys:
                if key not in detection_result:
                    logger.warning(f"Missing required key in detection result: {key}")
                    return False
            
            # Validate echo chambers
            chambers = detection_result['echo_chambers']
            if not isinstance(chambers, list):
                logger.warning("Echo chambers should be a list")
                return False
            
            for chamber in chambers:
                if not isinstance(chamber, EchoChamber):
                    logger.warning("Invalid echo chamber type")
                    return False
                
                # Check basic validity
                if chamber.total_members <= 0:
                    logger.warning(f"Invalid member count in chamber {chamber.chamber_id}")
                    return False
                
                if not (0 <= chamber.isolation_score <= 1):
                    logger.warning(f"Invalid isolation score in chamber {chamber.chamber_id}")
                    return False
                
                if not (0 <= chamber.content_homogeneity <= 1):
                    logger.warning(f"Invalid homogeneity in chamber {chamber.chamber_id}")
                    return False
                
                if not (0 <= chamber.echo_strength <= 1):
                    logger.warning(f"Invalid echo strength in chamber {chamber.chamber_id}")
                    return False
                
                # Validate chamber type
                valid_types = ['media', 'ideological', 'topical', 'geographic', 'temporal']
                if chamber.chamber_type not in valid_types:
                    logger.warning(f"Invalid chamber type: {chamber.chamber_type}")
                    return False
            
            # Validate filter bubbles
            bubbles = detection_result['filter_bubbles']
            if not isinstance(bubbles, list):
                logger.warning("Filter bubbles should be a list")
                return False
            
            for bubble in bubbles:
                if not isinstance(bubble, FilterBubble):
                    logger.warning("Invalid filter bubble type")
                    return False
                
                if not (0 <= bubble.filter_strength <= 1):
                    logger.warning("Invalid filter strength")
                    return False
                
                if not (0 <= bubble.information_diversity <= 1):
                    logger.warning("Invalid information diversity")
                    return False
            
            # Validate bridge actors
            bridges = detection_result['bridge_actors']
            if not isinstance(bridges, list):
                logger.warning("Bridge actors should be a list")
                return False
            
            for bridge in bridges:
                if not isinstance(bridge, BridgeActor):
                    logger.warning("Invalid bridge actor type")
                    return False
                
                if not (0 <= bridge.bridging_score <= 1):
                    logger.warning("Invalid bridging score")
                    return False
            
            # Validate reinforcement spirals
            spirals = detection_result['reinforcement_spirals']
            if not isinstance(spirals, list):
                logger.warning("Reinforcement spirals should be a list")
                return False
            
            for spiral in spirals:
                if not isinstance(spiral, ReinforcementSpiral):
                    logger.warning("Invalid reinforcement spiral type")
                    return False
                
                if spiral.amplification_factor < 1:
                    logger.warning("Invalid amplification factor")
                    return False
            
            # Validate diversity metrics
            diversity = detection_result['diversity_metrics']
            if not isinstance(diversity, dict):
                logger.warning("Diversity metrics should be a dictionary")
                return False
            
            # Check required diversity metrics
            required_diversity = ['frame_diversity', 'entity_diversity', 'media_diversity']
            for metric in required_diversity:
                if metric not in diversity:
                    logger.warning(f"Missing diversity metric: {metric}")
                    return False
                
                if not (0 <= diversity[metric] <= 1):
                    logger.warning(f"Invalid diversity value for {metric}")
                    return False
            
            logger.info(f"Validation successful: {len(chambers)} chambers, "
                       f"{len(bubbles)} bubbles, {len(bridges)} bridges")
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def score_detection(self, detection_result: Dict[str, Any]) -> float:
        """
        Score the quality of echo chamber detection.
        
        Scoring factors:
        - Chamber detection quality (isolation, homogeneity)
        - Filter bubble strength and clarity
        - Bridge actor identification
        - Reinforcement spiral detection
        - Overall diversity degradation
        
        Args:
            detection_result: Results from detect() method
            
        Returns:
            Score between 0 and 1
        """
        try:
            if not self.validate_detection(detection_result):
                return 0.0
            
            chambers = detection_result['echo_chambers']
            bubbles = detection_result['filter_bubbles']
            bridges = detection_result['bridge_actors']
            spirals = detection_result['reinforcement_spirals']
            diversity = detection_result['diversity_metrics']
            
            scores = []
            weights = []
            
            # 1. Echo chamber quality score (30% weight)
            if chambers:
                # Strong echo chambers have high isolation and homogeneity
                isolations = [c.isolation_score for c in chambers]
                homogeneities = [c.content_homogeneity for c in chambers]
                echo_strengths = [c.echo_strength for c in chambers]
                
                avg_isolation = np.mean(isolations)
                avg_homogeneity = np.mean(homogeneities)
                avg_echo = np.mean(echo_strengths)
                
                chamber_score = (avg_isolation + avg_homogeneity + avg_echo) / 3
                scores.append(chamber_score)
                weights.append(0.3)
                logger.debug(f"Chamber quality score: {chamber_score:.3f}")
            else:
                # No chambers detected - this might be good or bad
                scores.append(0.3)  # Neutral score
                weights.append(0.3)
            
            # 2. Filter bubble score (25% weight)
            if bubbles:
                filter_strengths = [b.filter_strength for b in bubbles]
                info_diversities = [b.information_diversity for b in bubbles]
                
                avg_filter = np.mean(filter_strengths)
                # Low diversity indicates strong bubble
                avg_bubble_strength = 1 - np.mean(info_diversities)
                
                bubble_score = (avg_filter + avg_bubble_strength) / 2
                scores.append(bubble_score)
                weights.append(0.25)
                logger.debug(f"Filter bubble score: {bubble_score:.3f}")
            else:
                scores.append(0.3)
                weights.append(0.25)
            
            # 3. Bridge actor score (15% weight)
            if bridges:
                bridging_scores = [b.bridging_score for b in bridges]
                effectivenesses = [b.get_bridge_effectiveness() for b in bridges]
                
                avg_bridging = np.mean(bridging_scores)
                avg_effectiveness = np.mean(effectivenesses)
                
                # Good detection finds effective bridges
                bridge_score = (avg_bridging + avg_effectiveness) / 2
                scores.append(bridge_score)
                weights.append(0.15)
                logger.debug(f"Bridge actor score: {bridge_score:.3f}")
            else:
                scores.append(0.2)  # Low score if no bridges found
                weights.append(0.15)
            
            # 4. Reinforcement spiral score (15% weight)
            if spirals:
                spiral_strengths = [s.get_spiral_strength() for s in spirals]
                amplifications = [min(s.amplification_factor / 5, 1) for s in spirals]  # Normalize
                
                avg_spiral = np.mean(spiral_strengths)
                avg_amplification = np.mean(amplifications)
                
                spiral_score = (avg_spiral + avg_amplification) / 2
                scores.append(spiral_score)
                weights.append(0.15)
                logger.debug(f"Reinforcement spiral score: {spiral_score:.3f}")
            else:
                scores.append(0.3)
                weights.append(0.15)
            
            # 5. Diversity degradation score (15% weight)
            # Low diversity indicates echo chamber presence
            frame_div = diversity.get('frame_diversity', 1.0)
            entity_div = diversity.get('entity_diversity', 1.0)
            media_div = diversity.get('media_diversity', 1.0)
            
            # Invert - low diversity = high score for echo chamber detection
            diversity_degradation = 1 - ((frame_div + entity_div + media_div) / 3)
            scores.append(diversity_degradation)
            weights.append(0.15)
            logger.debug(f"Diversity degradation score: {diversity_degradation:.3f}")
            
            # Calculate weighted average
            total_score = np.average(scores, weights=weights)
            
            # Apply bonuses and penalties
            
            # Bonus for finding clear echo chambers
            if chambers and any(c.isolation_score > 0.8 and c.echo_strength > 0.8 for c in chambers):
                total_score = min(1.0, total_score * 1.15)
                logger.debug("Applied bonus for clear echo chambers")
            
            # Bonus for comprehensive detection
            if len(chambers) > 3 and len(bubbles) > 2 and len(bridges) > 5:
                total_score = min(1.0, total_score * 1.1)
                logger.debug("Applied bonus for comprehensive detection")
            
            # Penalty if no significant patterns found
            if not chambers and not bubbles and not spirals:
                total_score *= 0.5
                logger.debug("Applied penalty for no significant patterns")
            
            # Penalty for inconsistent results
            if chambers and diversity_degradation < 0.2:
                # Found chambers but high diversity - inconsistent
                total_score *= 0.8
                logger.debug("Applied penalty for inconsistent results")
            
            logger.info(f"Echo chamber detection score: {total_score:.3f}")
            return float(total_score)
            
        except Exception as e:
            logger.error(f"Scoring error: {e}")
            return 0.0