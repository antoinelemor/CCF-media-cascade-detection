"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
sequence_detector.py

MAIN OBJECTIVE:
---------------
This script analyzes temporal sequences of cascade patterns, detecting cascade chains, recurring
patterns, narrative evolution, and information flow pathways through time series analysis.

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
- dtaidistance (optional)

MAIN FEATURES:
--------------
1) Cascade chain and dependency detection
2) Recurring pattern and cycle identification
3) Narrative evolution tracking over time
4) Trigger-response sequence analysis
5) Temporal motif and signature extraction

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
from scipy import stats, signal as scipy_signal
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist, cosine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import logging
import warnings

# Optional import for DTW
try:
    from dtaidistance import dtw, dtw_ndim
    HAS_DTW = True
except ImportError:
    HAS_DTW = False
    warnings.warn("dtaidistance not installed. Some sequence comparison features will be disabled.")
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
from cascade_detector.core.config import DetectorConfig

# Import metrics
from cascade_detector.metrics.scientific_network_metrics import ScientificNetworkMetrics
from cascade_detector.metrics.exhaustive_metrics_calculator import ExhaustiveMetricsCalculator
from cascade_detector.metrics.temporal_metrics import TemporalMetrics

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class CascadeSequence:
    """
    Represents a sequence of related cascades over time.
    """
    sequence_id: str
    sequence_type: str  # 'chain', 'cycle', 'evolution', 'mutation', 'branching'
    
    # Cascades in sequence
    cascades: List[CompleteCascade]
    cascade_order: List[str]  # Ordered cascade IDs
    
    # Temporal structure
    start_date: datetime
    end_date: datetime
    duration_days: int
    
    # Sequence characteristics
    length: int  # Number of cascades
    periodicity: Optional[float]  # For cyclic patterns
    trend: str  # 'increasing', 'decreasing', 'stable', 'oscillating'
    
    # Relationships between cascades
    transitions: List[Tuple[str, str, float]]  # (cascade1, cascade2, strength)
    dependency_graph: nx.DiGraph  # Dependencies between cascades
    influence_flow: Dict[str, Dict[str, float]]  # Influence between cascades
    
    # Content evolution
    entity_evolution: Dict[str, List[Set[str]]]  # Entity sets over time
    frame_evolution: Dict[str, List[float]]  # Frame intensities over time
    narrative_shifts: List[Dict[str, Any]]  # Major narrative changes
    
    # Media and journalist tracking
    media_consistency: Dict[str, float]  # Media -> consistency score
    journalist_consistency: Dict[str, float]  # Journalist -> consistency
    key_propagators: List[Tuple[str, float]]  # Main sequence propagators
    
    # Temporal patterns
    inter_cascade_intervals: List[float]  # Time gaps between cascades
    acceleration_pattern: List[float]  # Speed changes over sequence
    decay_pattern: List[float]  # Decay rates through sequence
    
    # Statistical properties
    autocorrelation: Dict[int, float]  # Lag -> correlation
    entropy: float  # Sequence entropy
    predictability: float  # How predictable the sequence is
    
    # Validation
    significance: float
    confidence: float
    coherence_score: float  # Internal coherence of sequence
    
    def get_sequence_signature(self) -> np.ndarray:
        """Generate numerical signature of the sequence."""
        signatures = []
        for cascade in self.cascades:
            # Extract key features from each cascade
            sig = [
                cascade.intensity_score,
                cascade.velocity_score,
                cascade.reach_score,
                cascade.persistence_score,
                len(cascade.media_involved),
                len(cascade.entities_involved)
            ]
            signatures.append(sig)
        return np.array(signatures)
    
    def get_temporal_profile(self) -> pd.Series:
        """Get temporal intensity profile of the sequence."""
        dates = []
        intensities = []
        for cascade in self.cascades:
            dates.append(cascade.peak_date)
            intensities.append(cascade.intensity_score)
        
        if dates:
            series = pd.Series(intensities, index=dates)
            return series.sort_index()
        return pd.Series()


@dataclass
class TemporalMotif:
    """
    Recurring temporal pattern in cascade sequences.
    """
    motif_id: str
    motif_type: str  # 'periodic', 'burst-decay', 'escalation', 'oscillation'
    
    # Pattern definition
    pattern_signature: np.ndarray  # Numerical representation
    pattern_length: int  # In time units
    
    # Occurrences
    occurrences: List[Dict[str, Any]]  # Where pattern appears
    frequency: float  # How often it occurs
    regularity: float  # How regular occurrences are
    
    # Characteristics
    avg_intensity: float
    avg_duration: float
    variability: float  # How much instances vary
    
    # Triggers and effects
    common_triggers: List[str]  # Common trigger types
    common_outcomes: List[str]  # Common outcomes
    
    # Statistical validation
    statistical_significance: float
    random_probability: float


@dataclass
class NarrativeEvolution:
    """
    Evolution of narrative elements through a sequence.
    """
    evolution_id: str
    
    # Narrative elements
    core_entities: Set[str]  # Persistent entities
    evolving_entities: Dict[str, List[Tuple[datetime, str]]]  # Entity changes
    
    # Frame dynamics
    dominant_frames: List[Tuple[datetime, str]]  # Dominant frame over time
    frame_transitions: List[Dict[str, Any]]  # Frame transition events
    
    # Content drift
    semantic_drift: float  # Overall semantic change
    drift_trajectory: List[float]  # Drift over time
    
    # Mutation events
    mutations: List[Dict[str, Any]]  # Major content changes
    mutation_rate: float  # Rate of change
    
    # Stability metrics
    narrative_coherence: float
    thematic_consistency: float


@dataclass
class InformationPathway:
    """
    Path of information flow through cascade sequences.
    """
    pathway_id: str
    
    # Path structure
    source_cascade: str
    sink_cascade: str
    intermediate_cascades: List[str]
    
    # Flow characteristics
    flow_strength: float
    flow_speed: float  # Information/time
    bottlenecks: List[str]  # Cascades that slow flow
    amplifiers: List[str]  # Cascades that amplify
    
    # Content transformation
    content_preservation: float  # How much content preserved
    content_mutations: List[Dict[str, Any]]  # Changes along path
    
    # Media pathway
    media_path: List[List[str]]  # Media involved at each step
    journalist_path: List[List[str]]  # Journalists at each step


class SequenceDetector(BaseDetector):
    """
    Detects and analyzes temporal sequences of cascades.
    """
    
    def __init__(self, 
                 context: DetectionContext,
                 config: Optional[DetectorConfig] = None):
        """Initialize sequence detector."""
        super().__init__(context, config)
        self.name = "SequenceDetector"
        
        # Initialize component detectors
        self.cascade_detector = CascadeDetector(context, config, None)  # burst_detector=None
        self.multi_frame_detector = MultiFrameDetector(context, config, self.cascade_detector)
        self.cross_media_tracker = CrossMediaTracker(context, config, self.cascade_detector, self.multi_frame_detector)
        
        # Initialize metrics calculators with required parameters
        self.network_metrics = ScientificNetworkMetrics(
            source_index=context.source_index,
            entity_index=context.entity_index
        )
        self.exhaustive_calculator = ExhaustiveMetricsCalculator()
        # TemporalMetrics needs temporal_index
        if context.temporal_index:
            self.temporal_metrics = TemporalMetrics(context.temporal_index)
        else:
            self.temporal_metrics = None
        
        # Sequence detection parameters
        self.min_sequence_length = 3  # Minimum cascades for sequence
        self.max_gap_days = 30  # Maximum gap between related cascades
        self.similarity_threshold = 0.6  # For cascade similarity
        self.coherence_threshold = 0.5  # For sequence coherence
        
        # Storage
        self.detected_sequences = []
        self.temporal_motifs = []
        self.narrative_evolutions = []
        self.information_pathways = []
        
        logger.info(f"SequenceDetector initialized with config: {config}")
    
    def detect(self, **kwargs) -> List[CascadeSequence]:
        """
        Main detection method for cascade sequences.
        """
        logger.info("Starting sequence detection...")
        
        # Get cascades from kwargs or detect them
        cascades = kwargs.get('cascades', [])
        if not cascades:
            cascade_result = self.cascade_detector.detect()
            # Extract cascades from the result dict
            cascades = cascade_result.get('cascades', []) if isinstance(cascade_result, dict) else cascade_result
        logger.info(f"Working with {len(cascades)} detected cascades")
        
        # Get multi-frame patterns from kwargs or detect them
        multi_frame_patterns = kwargs.get('multi_frame_patterns', [])
        if not multi_frame_patterns:
            multi_frame_result = self.multi_frame_detector.detect(cascades=cascades)
            # Extract patterns from the result dict
            if isinstance(multi_frame_result, dict):
                multi_frame_patterns = multi_frame_result.get('patterns', [])
            else:
                multi_frame_patterns = multi_frame_result if isinstance(multi_frame_result, list) else []
        logger.info(f"Using {len(multi_frame_patterns)} multi-frame patterns")
        
        # Get media coordination patterns from kwargs or detect them
        media_coordinations = kwargs.get('media_coordinations', [])
        if not media_coordinations:
            cross_media_result = self.cross_media_tracker.detect(cascades=cascades)
            # Extract media coordinations from the result dict
            media_coordinations = cross_media_result.get('media_coordinations', []) if isinstance(cross_media_result, dict) else []
        logger.info(f"Using {len(media_coordinations)} media coordination patterns")
        
        # Build cascade similarity matrix
        similarity_matrix = self._build_similarity_matrix(cascades)
        
        # Detect sequence candidates
        sequence_candidates = self._detect_sequence_candidates(
            cascades, similarity_matrix
        )
        logger.info(f"Found {len(sequence_candidates)} sequence candidates")
        
        # Analyze each candidate
        sequences = []
        for candidate in tqdm(sequence_candidates, desc="Analyzing sequences"):
            sequence = self._analyze_sequence(
                candidate, cascades, multi_frame_patterns, media_coordinations
            )
            
            if sequence and self._validate_sequence(sequence):
                sequences.append(sequence)
                
                # Extract temporal motifs
                motifs = self._extract_temporal_motifs(sequence)
                self.temporal_motifs.extend(motifs)
                
                # Track narrative evolution
                evolution = self._track_narrative_evolution(sequence)
                if evolution:
                    self.narrative_evolutions.append(evolution)
                
                # Identify information pathways
                pathways = self._identify_information_pathways(sequence)
                self.information_pathways.extend(pathways)
        
        logger.info(f"Detected {len(sequences)} valid sequences")
        logger.info(f"Extracted {len(self.temporal_motifs)} temporal motifs")
        logger.info(f"Tracked {len(self.narrative_evolutions)} narrative evolutions")
        logger.info(f"Identified {len(self.information_pathways)} information pathways")
        
        # Store results
        self.detected_sequences = sequences
        
        # Return as dictionary for consistency
        result = {
            'sequences': sequences,
            'temporal_motifs': self.temporal_motifs,
            'narrative_evolutions': self.narrative_evolutions,
            'information_pathways': self.information_pathways,
            'n_sequences': len(sequences),
            'summary': {
                'total_cascades_involved': sum(s.length for s in sequences),
                'avg_sequence_length': np.mean([s.length for s in sequences]) if sequences else 0,
                'max_sequence_length': max([s.length for s in sequences]) if sequences else 0
            }
        }
        
        return result
    
    def _build_similarity_matrix(self, cascades: List[CompleteCascade]) -> np.ndarray:
        """Build similarity matrix between cascades."""
        n = len(cascades)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                similarity = self._calculate_cascade_similarity(
                    cascades[i], cascades[j]
                )
                matrix[i, j] = similarity
                matrix[j, i] = similarity
        
        return matrix
    
    def _calculate_cascade_similarity(self, c1: CompleteCascade, 
                                     c2: CompleteCascade) -> float:
        """Calculate similarity between two cascades."""
        # Entity similarity (Jaccard)
        entities1 = set(e['entity'] for e in c1.entities_involved)
        entities2 = set(e['entity'] for e in c2.entities_involved)
        entity_sim = len(entities1 & entities2) / len(entities1 | entities2) if entities1 | entities2 else 0
        
        # Media similarity
        media_sim = len(c1.media_involved & c2.media_involved) / \
                   len(c1.media_involved | c2.media_involved) if c1.media_involved | c2.media_involved else 0
        
        # Frame similarity
        frame_sim = 1.0 if c1.primary_frame == c2.primary_frame else 0.3
        
        # Temporal proximity (decay with time)
        time_diff = abs((c1.peak_date - c2.peak_date).days)
        temporal_sim = np.exp(-time_diff / 30)  # 30-day decay constant
        
        # Network structure similarity
        network_sim = self._calculate_network_similarity(
            c1.propagation_network, c2.propagation_network
        )
        
        # Weighted combination
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # Entity, media, frame, temporal, network
        similarities = [entity_sim, media_sim, frame_sim, temporal_sim, network_sim]
        
        return np.dot(weights, similarities)
    
    def _calculate_network_similarity(self, g1: nx.Graph, g2: nx.Graph) -> float:
        """Calculate structural similarity between networks."""
        if not g1 or not g2 or g1.number_of_nodes() == 0 or g2.number_of_nodes() == 0:
            return 0.0
        
        # Compare basic metrics
        metrics1 = [
            g1.number_of_nodes(),
            g1.number_of_edges(),
            nx.density(g1) if g1.number_of_nodes() > 1 else 0
        ]
        metrics2 = [
            g2.number_of_nodes(),
            g2.number_of_edges(),
            nx.density(g2) if g2.number_of_nodes() > 1 else 0
        ]
        
        # Normalize and compare
        if max(metrics1 + metrics2) > 0:
            metrics1 = np.array(metrics1) / max(metrics1 + metrics2)
            metrics2 = np.array(metrics2) / max(metrics1.tolist() + metrics2.tolist())
            return 1 - cosine(metrics1, metrics2)
        return 0.0
    
    def _detect_sequence_candidates(self, cascades: List[CompleteCascade],
                                   similarity_matrix: np.ndarray) -> List[List[int]]:
        """Detect candidate sequences using clustering and temporal ordering."""
        candidates = []
        
        # Sort cascades by time
        sorted_indices = np.argsort([c.start_date for c in cascades])
        
        # Method 1: Temporal chains
        chains = self._detect_temporal_chains(cascades, similarity_matrix, sorted_indices)
        candidates.extend(chains)
        
        # Method 2: Clustering similar cascades
        clusters = self._cluster_similar_cascades(similarity_matrix)
        for cluster in clusters:
            if len(cluster) >= self.min_sequence_length:
                # Order cluster by time
                cluster_ordered = sorted(cluster, 
                                       key=lambda i: cascades[i].start_date)
                candidates.append(cluster_ordered)
        
        # Method 3: Detect cycles
        cycles = self._detect_cyclic_patterns(cascades, similarity_matrix)
        candidates.extend(cycles)
        
        # Remove duplicates and short sequences
        unique_candidates = []
        seen = set()
        for candidate in candidates:
            if len(candidate) >= self.min_sequence_length:
                key = tuple(sorted(candidate))
                if key not in seen:
                    seen.add(key)
                    unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _detect_temporal_chains(self, cascades: List[CompleteCascade],
                               similarity_matrix: np.ndarray,
                               sorted_indices: np.ndarray) -> List[List[int]]:
        """Detect chains of cascades following temporal order."""
        chains = []
        used = set()
        
        for i in sorted_indices:
            if i in used:
                continue
            
            chain = [i]
            current = i
            used.add(i)
            
            # Look for next cascade in chain
            for j in sorted_indices:
                if j <= current or j in used:
                    continue
                
                # Check temporal gap
                time_gap = (cascades[j].start_date - cascades[current].end_date).days
                if time_gap > self.max_gap_days:
                    break
                
                # Check similarity
                if similarity_matrix[current, j] >= self.similarity_threshold:
                    chain.append(j)
                    current = j
                    used.add(j)
            
            if len(chain) >= self.min_sequence_length:
                chains.append(chain)
        
        return chains
    
    def _cluster_similar_cascades(self, similarity_matrix: np.ndarray) -> List[List[int]]:
        """Cluster cascades based on similarity."""
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        # Hierarchical clustering
        linkage_matrix = linkage(distance_matrix[np.triu_indices_from(distance_matrix, k=1)],
                                method='ward')
        
        # Cut tree at threshold
        clusters = fcluster(linkage_matrix, 
                          1 - self.similarity_threshold,
                          criterion='distance')
        
        # Group indices by cluster
        cluster_groups = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            cluster_groups[cluster_id].append(idx)
        
        return list(cluster_groups.values())
    
    def _detect_cyclic_patterns(self, cascades: List[CompleteCascade],
                               similarity_matrix: np.ndarray) -> List[List[int]]:
        """Detect cyclic/periodic patterns in cascades."""
        cycles = []
        
        # Build time series of cascade occurrences
        dates = [c.peak_date for c in cascades]
        date_range = pd.date_range(min(dates), max(dates), freq='D')
        
        # For each frame, look for periodicity
        frames = set(c.primary_frame for c in cascades)
        for frame in frames:
            frame_cascades = [(i, c) for i, c in enumerate(cascades) 
                            if c.primary_frame == frame]
            
            if len(frame_cascades) < self.min_sequence_length * 2:
                continue
            
            # Create time series
            ts = pd.Series(0, index=date_range)
            for i, c in frame_cascades:
                ts[c.peak_date] = c.intensity_score
            
            # Detect periodicity using autocorrelation
            if len(ts) > 50:
                autocorr = [ts.autocorr(lag) for lag in range(1, min(100, len(ts)//2))]
                
                # Find peaks in autocorrelation
                peaks, _ = scipy_signal.find_peaks(autocorr, height=0.3)
                
                if len(peaks) > 0:
                    period = peaks[0] + 1
                    
                    # Group cascades by period
                    for start_idx in range(len(frame_cascades) - self.min_sequence_length + 1):
                        cycle_candidate = []
                        current_date = frame_cascades[start_idx][1].peak_date
                        
                        for i, c in frame_cascades[start_idx:]:
                            if abs((c.peak_date - current_date).days) < period * 0.2:
                                cycle_candidate.append(i)
                                current_date += timedelta(days=period)
                        
                        if len(cycle_candidate) >= self.min_sequence_length:
                            cycles.append(cycle_candidate)
        
        return cycles
    
    def _analyze_sequence(self, candidate: List[int],
                         cascades: List[CompleteCascade],
                         multi_frame_patterns: List[MultiFramePattern],
                         media_coordinations: List[MediaCoordination]) -> Optional[CascadeSequence]:
        """Analyze a candidate sequence in detail."""
        sequence_cascades = [cascades[i] for i in candidate]
        
        if len(sequence_cascades) < self.min_sequence_length:
            return None
        
        # Determine sequence type
        sequence_type = self._determine_sequence_type(sequence_cascades)
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(sequence_cascades)
        
        # Calculate influence flow
        influence_flow = self._calculate_influence_flow(sequence_cascades)
        
        # Track entity evolution
        entity_evolution = self._track_entity_evolution(sequence_cascades)
        
        # Track frame evolution
        frame_evolution = self._track_frame_evolution(sequence_cascades)
        
        # Identify narrative shifts
        narrative_shifts = self._identify_narrative_shifts(sequence_cascades)
        
        # Calculate media consistency
        media_consistency = self._calculate_media_consistency(sequence_cascades)
        
        # Calculate journalist consistency
        journalist_consistency = self._calculate_journalist_consistency(sequence_cascades)
        
        # Identify key propagators
        key_propagators = self._identify_key_propagators(sequence_cascades)
        
        # Calculate temporal patterns
        inter_cascade_intervals = self._calculate_inter_cascade_intervals(sequence_cascades)
        acceleration_pattern = self._calculate_acceleration_pattern(sequence_cascades)
        decay_pattern = self._calculate_decay_pattern(sequence_cascades)
        
        # Calculate statistical properties
        autocorrelation = self._calculate_sequence_autocorrelation(sequence_cascades)
        entropy = self._calculate_sequence_entropy(sequence_cascades)
        predictability = self._calculate_predictability(sequence_cascades)
        
        # Calculate coherence
        coherence_score = self._calculate_coherence(sequence_cascades)
        
        # Build transitions
        transitions = []
        for i in range(len(candidate) - 1):
            strength = self._calculate_transition_strength(
                sequence_cascades[i], sequence_cascades[i+1]
            )
            transitions.append((
                sequence_cascades[i].cascade_id,
                sequence_cascades[i+1].cascade_id,
                strength
            ))
        
        # Determine trend
        intensities = [c.intensity_score for c in sequence_cascades]
        trend = self._determine_trend(intensities)
        
        # Calculate periodicity for cyclic patterns
        periodicity = None
        if sequence_type == 'cycle':
            periodicity = self._calculate_periodicity(sequence_cascades)
        
        return CascadeSequence(
            sequence_id=str(uuid.uuid4()),
            sequence_type=sequence_type,
            cascades=sequence_cascades,
            cascade_order=[c.cascade_id for c in sequence_cascades],
            start_date=sequence_cascades[0].start_date,
            end_date=sequence_cascades[-1].end_date,
            duration_days=(sequence_cascades[-1].end_date - sequence_cascades[0].start_date).days,
            length=len(sequence_cascades),
            periodicity=periodicity,
            trend=trend,
            transitions=transitions,
            dependency_graph=dependency_graph,
            influence_flow=influence_flow,
            entity_evolution=entity_evolution,
            frame_evolution=frame_evolution,
            narrative_shifts=narrative_shifts,
            media_consistency=media_consistency,
            journalist_consistency=journalist_consistency,
            key_propagators=key_propagators,
            inter_cascade_intervals=inter_cascade_intervals,
            acceleration_pattern=acceleration_pattern,
            decay_pattern=decay_pattern,
            autocorrelation=autocorrelation,
            entropy=entropy,
            predictability=predictability,
            significance=0.0,  # Will be calculated in validation
            confidence=0.0,  # Will be calculated in validation
            coherence_score=coherence_score
        )
    
    def _determine_sequence_type(self, cascades: List[CompleteCascade]) -> str:
        """Determine the type of sequence."""
        # Check for cycles
        peak_dates = [c.peak_date for c in cascades]
        intervals = [(peak_dates[i+1] - peak_dates[i]).days 
                    for i in range(len(peak_dates)-1)]
        
        if len(intervals) > 2:
            cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 1
            if cv < 0.3:  # Regular intervals
                return 'cycle'
        
        # Check for evolution (gradual content change)
        entity_overlap = []
        for i in range(len(cascades)-1):
            e1 = set(e['entity'] for e in cascades[i].entities_involved)
            e2 = set(e['entity'] for e in cascades[i+1].entities_involved)
            overlap = len(e1 & e2) / len(e1 | e2) if e1 | e2 else 0
            entity_overlap.append(overlap)
        
        if entity_overlap and np.mean(entity_overlap) < 0.5:
            return 'evolution'
        
        # Check for mutation (sudden changes)
        if entity_overlap and min(entity_overlap) < 0.2:
            return 'mutation'
        
        # Check for branching (diverging paths)
        frames = [c.primary_frame for c in cascades]
        if len(set(frames)) > len(frames) * 0.6:
            return 'branching'
        
        # Default to chain
        return 'chain'
    
    def _build_dependency_graph(self, cascades: List[CompleteCascade]) -> nx.DiGraph:
        """Build dependency graph between cascades."""
        G = nx.DiGraph()
        
        for i, cascade in enumerate(cascades):
            G.add_node(cascade.cascade_id, 
                      index=i,
                      peak_date=cascade.peak_date,
                      intensity=cascade.intensity_score)
        
        # Add edges based on influence
        for i in range(len(cascades)):
            for j in range(i+1, len(cascades)):
                # Check temporal ordering
                if cascades[j].start_date > cascades[i].peak_date:
                    # Calculate dependency strength
                    dep_strength = self._calculate_dependency_strength(
                        cascades[i], cascades[j]
                    )
                    if dep_strength > 0.3:
                        G.add_edge(cascades[i].cascade_id,
                                 cascades[j].cascade_id,
                                 weight=dep_strength)
        
        return G
    
    def _calculate_dependency_strength(self, c1: CompleteCascade,
                                      c2: CompleteCascade) -> float:
        """Calculate dependency strength between cascades."""
        # Entity overlap
        e1 = set(e['entity'] for e in c1.entities_involved)
        e2 = set(e['entity'] for e in c2.entities_involved)
        entity_overlap = len(e1 & e2) / len(e1) if e1 else 0
        
        # Media overlap
        media_overlap = len(c1.media_involved & c2.media_involved) / \
                       len(c1.media_involved) if c1.media_involved else 0
        
        # Temporal proximity
        time_gap = (c2.start_date - c1.end_date).days
        temporal_factor = np.exp(-time_gap / 7) if time_gap >= 0 else 0
        
        # Intensity correlation
        intensity_factor = min(c2.intensity_score / c1.intensity_score, 1.0) \
                          if c1.intensity_score > 0 else 0
        
        return np.mean([entity_overlap, media_overlap, temporal_factor, intensity_factor])
    
    def _calculate_influence_flow(self, cascades: List[CompleteCascade]) -> Dict[str, Dict[str, float]]:
        """Calculate influence flow between cascades."""
        influence = defaultdict(dict)
        
        for i in range(len(cascades)):
            for j in range(i+1, len(cascades)):
                # Forward influence
                forward = self._calculate_directional_influence(
                    cascades[i], cascades[j]
                )
                if forward > 0:
                    influence[cascades[i].cascade_id][cascades[j].cascade_id] = forward
                
                # Backward influence (feedback)
                backward = self._calculate_directional_influence(
                    cascades[j], cascades[i]
                ) * 0.3  # Reduce backward influence weight
                if backward > 0:
                    influence[cascades[j].cascade_id][cascades[i].cascade_id] = backward
        
        return dict(influence)
    
    def _calculate_directional_influence(self, source: CompleteCascade,
                                        target: CompleteCascade) -> float:
        """Calculate directional influence from source to target."""
        # Skip if target comes before source
        if target.start_date < source.end_date:
            return 0.0
        
        # Entity propagation
        source_entities = set(e['entity'] for e in source.entities_involved)
        target_entities = set(e['entity'] for e in target.entities_involved)
        entity_propagation = len(source_entities & target_entities) / \
                           len(source_entities) if source_entities else 0
        
        # Media propagation
        media_propagation = len(source.media_involved & target.media_involved) / \
                          len(source.media_involved) if source.media_involved else 0
        
        # Messenger propagation
        source_messengers = set(m['entity'] for m in source.messengers)
        target_messengers = set(m['entity'] for m in target.messengers)
        messenger_propagation = len(source_messengers & target_messengers) / \
                              len(source_messengers) if source_messengers else 0
        
        # Time decay
        time_gap = (target.start_date - source.end_date).days
        decay = np.exp(-time_gap / 14)  # 2-week decay
        
        return (entity_propagation * 0.4 + media_propagation * 0.3 + 
                messenger_propagation * 0.3) * decay
    
    def _track_entity_evolution(self, cascades: List[CompleteCascade]) -> Dict[str, List[Set[str]]]:
        """Track how entities evolve through the sequence."""
        evolution = defaultdict(list)
        
        # Track each entity type separately
        for cascade in cascades:
            # Group entities by type
            by_type = defaultdict(set)
            for entity_data in cascade.entities_involved:
                entity_type = entity_data.get('type', 'UNKNOWN')
                entity = entity_data.get('entity', '')
                by_type[entity_type].add(entity)
            
            for entity_type, entities in by_type.items():
                evolution[entity_type].append(entities)
        
        return dict(evolution)
    
    def _track_frame_evolution(self, cascades: List[CompleteCascade]) -> Dict[str, List[float]]:
        """Track frame intensities through the sequence."""
        evolution = defaultdict(list)
        
        all_frames = set()
        for cascade in cascades:
            all_frames.add(cascade.primary_frame)
            all_frames.update(cascade.secondary_frames)
        
        for frame in all_frames:
            for cascade in cascades:
                if frame == cascade.primary_frame:
                    intensity = 1.0
                elif frame in cascade.secondary_frames:
                    intensity = 0.5
                else:
                    intensity = 0.0
                evolution[frame].append(intensity * cascade.intensity_score)
        
        return dict(evolution)
    
    def _identify_narrative_shifts(self, cascades: List[CompleteCascade]) -> List[Dict[str, Any]]:
        """Identify major narrative shifts in the sequence."""
        shifts = []
        
        for i in range(len(cascades) - 1):
            c1, c2 = cascades[i], cascades[i+1]
            
            # Check entity shift
            e1 = set(e['entity'] for e in c1.entities_involved)
            e2 = set(e['entity'] for e in c2.entities_involved)
            entity_shift = 1 - (len(e1 & e2) / len(e1 | e2) if e1 | e2 else 1)
            
            # Check frame shift
            frame_shift = 0 if c1.primary_frame == c2.primary_frame else 1
            
            # Check messenger shift
            m1 = set(m['entity'] for m in c1.messengers)
            m2 = set(m['entity'] for m in c2.messengers)
            messenger_shift = 1 - (len(m1 & m2) / len(m1 | m2) if m1 | m2 else 1)
            
            # Check media shift
            media_shift = 1 - (len(c1.media_involved & c2.media_involved) / 
                             len(c1.media_involved | c2.media_involved) 
                             if c1.media_involved | c2.media_involved else 1)
            
            # Calculate overall shift
            shift_score = np.mean([entity_shift, frame_shift, messenger_shift, media_shift])
            
            if shift_score > 0.5:
                shifts.append({
                    'position': i,
                    'from_cascade': c1.cascade_id,
                    'to_cascade': c2.cascade_id,
                    'shift_score': shift_score,
                    'entity_shift': entity_shift,
                    'frame_shift': frame_shift,
                    'messenger_shift': messenger_shift,
                    'media_shift': media_shift,
                    'date': c2.start_date
                })
        
        return shifts
    
    def _calculate_media_consistency(self, cascades: List[CompleteCascade]) -> Dict[str, float]:
        """Calculate consistency score for each media outlet."""
        media_participation = defaultdict(list)
        
        for i, cascade in enumerate(cascades):
            all_media = set()
            for c in cascades:
                all_media.update(c.media_involved)
            
            for media in all_media:
                if media in cascade.media_involved:
                    media_participation[media].append(1)
                else:
                    media_participation[media].append(0)
        
        consistency = {}
        for media, participation in media_participation.items():
            # Calculate consistency as low variance in participation
            if len(participation) > 1:
                consistency[media] = 1 - np.std(participation)
            else:
                consistency[media] = 1.0 if participation[0] == 1 else 0.0
        
        return consistency
    
    def _calculate_journalist_consistency(self, cascades: List[CompleteCascade]) -> Dict[str, float]:
        """Calculate consistency score for each journalist."""
        journalist_participation = defaultdict(list)
        
        for cascade in cascades:
            all_journalists = set()
            for c in cascades:
                all_journalists.update(c.journalists_involved)
            
            for journalist in all_journalists:
                if journalist in cascade.journalists_involved:
                    journalist_participation[journalist].append(1)
                else:
                    journalist_participation[journalist].append(0)
        
        consistency = {}
        for journalist, participation in journalist_participation.items():
            if len(participation) > 1:
                consistency[journalist] = 1 - np.std(participation)
            else:
                consistency[journalist] = 1.0 if participation[0] == 1 else 0.0
        
        return consistency
    
    def _identify_key_propagators(self, cascades: List[CompleteCascade]) -> List[Tuple[str, float]]:
        """Identify key propagators (media/journalists) in the sequence."""
        propagator_scores = defaultdict(float)
        
        # Score media
        for cascade in cascades:
            for media in cascade.media_involved:
                # Base score from participation
                propagator_scores[f"media:{media}"] += cascade.intensity_score
                
                # Bonus for being first mover
                if media in cascade.media_first_movers[:3]:
                    propagator_scores[f"media:{media}"] += cascade.intensity_score * 0.5
        
        # Score journalists  
        for cascade in cascades:
            for journalist in cascade.journalists_involved:
                propagator_scores[f"journalist:{journalist}"] += cascade.intensity_score * 0.8
        
        # Sort and return top propagators
        sorted_propagators = sorted(propagator_scores.items(),
                                  key=lambda x: x[1], reverse=True)
        return sorted_propagators[:20]
    
    def _calculate_inter_cascade_intervals(self, cascades: List[CompleteCascade]) -> List[float]:
        """Calculate time intervals between cascades."""
        intervals = []
        for i in range(len(cascades) - 1):
            interval = (cascades[i+1].start_date - cascades[i].end_date).days
            intervals.append(max(0, interval))
        return intervals
    
    def _calculate_acceleration_pattern(self, cascades: List[CompleteCascade]) -> List[float]:
        """Calculate acceleration pattern through sequence."""
        pattern = []
        for cascade in cascades:
            # Use cascade's own acceleration if available
            if hasattr(cascade, 'acceleration_rate'):
                pattern.append(cascade.acceleration_rate)
            else:
                # Calculate from velocity
                pattern.append(cascade.velocity_score)
        return pattern
    
    def _calculate_decay_pattern(self, cascades: List[CompleteCascade]) -> List[float]:
        """Calculate decay pattern through sequence."""
        pattern = []
        for cascade in cascades:
            # Use cascade's own decay rate if available
            if hasattr(cascade, 'decay_rate'):
                pattern.append(cascade.decay_rate)
            else:
                # Estimate from duration and intensity
                decay = 1.0 / (cascade.duration_days + 1) * cascade.intensity_score
                pattern.append(decay)
        return pattern
    
    def _calculate_sequence_autocorrelation(self, cascades: List[CompleteCascade]) -> Dict[int, float]:
        """Calculate autocorrelation of sequence intensities."""
        intensities = [c.intensity_score for c in cascades]
        
        if len(intensities) < 3:
            return {}
        
        autocorr = {}
        for lag in range(1, min(len(intensities)//2, 10)):
            if len(intensities) > lag:
                correlation = np.corrcoef(intensities[:-lag], intensities[lag:])[0, 1]
                autocorr[lag] = correlation if not np.isnan(correlation) else 0.0
        
        return autocorr
    
    def _calculate_sequence_entropy(self, cascades: List[CompleteCascade]) -> float:
        """Calculate entropy of the sequence."""
        # Use intensity distribution
        intensities = [c.intensity_score for c in cascades]
        if not intensities:
            return 0.0
        
        # Normalize to probabilities
        total = sum(intensities)
        if total == 0:
            return 0.0
        
        probs = [i/total for i in intensities]
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(intensities))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_predictability(self, cascades: List[CompleteCascade]) -> float:
        """Calculate how predictable the sequence is."""
        if len(cascades) < 3:
            return 0.0
        
        # Use multiple features for prediction
        features = []
        for cascade in cascades:
            features.append([
                cascade.intensity_score,
                cascade.velocity_score,
                len(cascade.media_involved),
                len(cascade.entities_involved)
            ])
        
        features = np.array(features)
        
        # Calculate prediction error using simple linear model
        errors = []
        for i in range(2, len(features)):
            # Use previous 2 to predict current
            X = features[i-2:i]
            y_true = features[i]
            y_pred = np.mean(X, axis=0)
            
            error = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-6))
            errors.append(error)
        
        if errors:
            # Convert error to predictability (inverse)
            avg_error = np.mean(errors)
            predictability = 1 / (1 + avg_error)
            return predictability
        
        return 0.0
    
    def _calculate_coherence(self, cascades: List[CompleteCascade]) -> float:
        """Calculate internal coherence of the sequence."""
        if len(cascades) < 2:
            return 1.0
        
        coherence_factors = []
        
        # Entity coherence
        all_entities = set()
        for cascade in cascades:
            all_entities.update(e['entity'] for e in cascade.entities_involved)
        
        entity_overlaps = []
        for i in range(len(cascades) - 1):
            e1 = set(e['entity'] for e in cascades[i].entities_involved)
            e2 = set(e['entity'] for e in cascades[i+1].entities_involved)
            if e1 and e2:
                overlap = len(e1 & e2) / len(e1 | e2)
                entity_overlaps.append(overlap)
        
        if entity_overlaps:
            coherence_factors.append(np.mean(entity_overlaps))
        
        # Frame coherence
        frame_consistency = []
        for i in range(len(cascades) - 1):
            if cascades[i].primary_frame == cascades[i+1].primary_frame:
                frame_consistency.append(1.0)
            elif cascades[i+1].primary_frame in cascades[i].secondary_frames:
                frame_consistency.append(0.5)
            else:
                frame_consistency.append(0.0)
        
        if frame_consistency:
            coherence_factors.append(np.mean(frame_consistency))
        
        # Media coherence
        media_overlaps = []
        for i in range(len(cascades) - 1):
            m1 = cascades[i].media_involved
            m2 = cascades[i+1].media_involved
            if m1 and m2:
                overlap = len(m1 & m2) / len(m1 | m2)
                media_overlaps.append(overlap)
        
        if media_overlaps:
            coherence_factors.append(np.mean(media_overlaps))
        
        # Temporal coherence (regular spacing)
        intervals = self._calculate_inter_cascade_intervals(cascades)
        if intervals and len(intervals) > 1:
            cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
            temporal_coherence = 1 / (1 + cv)
            coherence_factors.append(temporal_coherence)
        
        return np.mean(coherence_factors) if coherence_factors else 0.0
    
    def _calculate_transition_strength(self, c1: CompleteCascade,
                                      c2: CompleteCascade) -> float:
        """Calculate strength of transition between cascades."""
        # Entity continuity
        e1 = set(e['entity'] for e in c1.entities_involved)
        e2 = set(e['entity'] for e in c2.entities_involved)
        entity_continuity = len(e1 & e2) / len(e1) if e1 else 0
        
        # Media continuity
        media_continuity = len(c1.media_involved & c2.media_involved) / \
                         len(c1.media_involved) if c1.media_involved else 0
        
        # Temporal proximity
        gap = (c2.start_date - c1.end_date).days
        temporal_strength = np.exp(-gap / 7) if gap >= 0 else 0
        
        # Intensity transfer
        intensity_transfer = min(c2.intensity_score / (c1.intensity_score + 1e-6), 2.0) / 2.0
        
        return np.mean([entity_continuity, media_continuity, 
                       temporal_strength, intensity_transfer])
    
    def _determine_trend(self, intensities: List[float]) -> str:
        """Determine trend from intensity values."""
        if len(intensities) < 2:
            return 'stable'
        
        # Fit linear trend
        x = np.arange(len(intensities))
        slope, _ = np.polyfit(x, intensities, 1)
        
        # Calculate variance
        cv = np.std(intensities) / (np.mean(intensities) + 1e-6)
        
        if cv > 0.5:
            return 'oscillating'
        elif slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_periodicity(self, cascades: List[CompleteCascade]) -> float:
        """Calculate periodicity for cyclic patterns."""
        peak_dates = [c.peak_date for c in cascades]
        
        if len(peak_dates) < 3:
            return None
        
        # Calculate intervals
        intervals = [(peak_dates[i+1] - peak_dates[i]).days 
                    for i in range(len(peak_dates)-1)]
        
        if intervals:
            return np.median(intervals)
        
        return None
    
    def _validate_sequence(self, sequence: CascadeSequence) -> bool:
        """Validate a detected sequence."""
        # Check minimum length
        if sequence.length < self.min_sequence_length:
            return False
        
        # Check coherence
        if sequence.coherence_score < self.coherence_threshold:
            return False
        
        # Statistical validation
        significance = self._calculate_statistical_significance(sequence)
        confidence = self._calculate_confidence(sequence)
        
        sequence.significance = significance
        sequence.confidence = confidence
        
        return significance > 0.95 and confidence > 0.7
    
    def _calculate_statistical_significance(self, sequence: CascadeSequence) -> float:
        """Calculate statistical significance of the sequence."""
        # Test against null hypothesis of random occurrence
        
        # Calculate probability of observing this sequence by chance
        n_cascades = len(sequence.cascades)
        
        # Probability based on coherence and predictability
        random_prob = (1 - sequence.coherence_score) * (1 - sequence.predictability)
        
        # Binomial test
        from scipy.stats import binom_test
        p_value = binom_test(n_cascades, n_cascades, random_prob, alternative='greater')
        
        return 1 - p_value
    
    def _calculate_confidence(self, sequence: CascadeSequence) -> float:
        """Calculate confidence in the sequence detection."""
        factors = []
        
        # Length factor
        length_factor = min(sequence.length / 10, 1.0)
        factors.append(length_factor)
        
        # Coherence factor
        factors.append(sequence.coherence_score)
        
        # Predictability factor
        factors.append(sequence.predictability)
        
        # Transition strength
        if sequence.transitions:
            avg_transition = np.mean([t[2] for t in sequence.transitions])
            factors.append(avg_transition)
        
        # Statistical significance
        factors.append(sequence.significance)
        
        return np.mean(factors)
    
    def _extract_temporal_motifs(self, sequence: CascadeSequence) -> List[TemporalMotif]:
        """Extract recurring temporal motifs from the sequence."""
        motifs = []
        
        # Get sequence signature
        signature = sequence.get_sequence_signature()
        
        if len(signature) < 3:
            return motifs
        
        # Look for recurring patterns using sliding window
        for window_size in range(2, min(len(signature)//2 + 1, 5)):
            for i in range(len(signature) - window_size * 2 + 1):
                pattern = signature[i:i+window_size]
                
                # Search for repetitions
                occurrences = []
                for j in range(i+window_size, len(signature)-window_size+1):
                    candidate = signature[j:j+window_size]
                    
                    # Calculate similarity
                    similarity = 1 - cosine(pattern.flatten(), candidate.flatten())
                    
                    if similarity > 0.8:
                        occurrences.append({
                            'position': j,
                            'similarity': similarity,
                            'cascade_id': sequence.cascades[j].cascade_id
                        })
                
                if len(occurrences) >= 2:
                    # Found a motif
                    motif = TemporalMotif(
                        motif_id=str(uuid.uuid4()),
                        motif_type=self._classify_motif_type(pattern),
                        pattern_signature=pattern,
                        pattern_length=window_size,
                        occurrences=occurrences,
                        frequency=len(occurrences) / len(signature),
                        regularity=self._calculate_motif_regularity(occurrences),
                        avg_intensity=np.mean(pattern[:, 0]),  # First column is intensity
                        avg_duration=window_size,
                        variability=np.std(pattern[:, 0]),
                        common_triggers=self._identify_common_triggers(
                            sequence, [i] + [o['position'] for o in occurrences]
                        ),
                        common_outcomes=self._identify_common_outcomes(
                            sequence, [i] + [o['position'] for o in occurrences]
                        ),
                        statistical_significance=self._calculate_motif_significance(
                            len(occurrences), len(signature), window_size
                        ),
                        random_probability=1 / (len(signature) - window_size + 1)
                    )
                    motifs.append(motif)
        
        return motifs
    
    def _classify_motif_type(self, pattern: np.ndarray) -> str:
        """Classify the type of temporal motif."""
        intensities = pattern[:, 0]  # First column is intensity
        
        # Check for periodicity
        if len(intensities) > 2:
            autocorr = np.corrcoef(intensities[:-1], intensities[1:])[0, 1]
            if abs(autocorr) > 0.5:
                return 'periodic'
        
        # Check for burst-decay
        if len(intensities) >= 2:
            if intensities[0] < intensities[1] and intensities[-1] < intensities[1]:
                return 'burst-decay'
        
        # Check for escalation
        if len(intensities) >= 3:
            diffs = np.diff(intensities)
            if all(d > 0 for d in diffs):
                return 'escalation'
        
        # Check for oscillation
        if len(intensities) >= 3:
            signs = np.sign(np.diff(intensities))
            changes = np.sum(np.diff(signs) != 0)
            if changes >= len(intensities) - 2:
                return 'oscillation'
        
        return 'complex'
    
    def _calculate_motif_regularity(self, occurrences: List[Dict]) -> float:
        """Calculate how regular motif occurrences are."""
        if len(occurrences) < 2:
            return 0.0
        
        positions = [o['position'] for o in occurrences]
        intervals = np.diff(sorted(positions))
        
        if len(intervals) > 0:
            cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
            return 1 / (1 + cv)
        
        return 0.0
    
    def _identify_common_triggers(self, sequence: CascadeSequence,
                                 positions: List[int]) -> List[str]:
        """Identify common triggers for motif occurrences."""
        triggers = Counter()
        
        for pos in positions:
            if pos > 0:
                prev_cascade = sequence.cascades[pos-1]
                
                # Check for event triggers
                if prev_cascade.trigger_event:
                    triggers[prev_cascade.trigger_event['type']] += 1
                
                # Check for entity triggers
                for entity in prev_cascade.entities_involved[:3]:
                    triggers[f"entity:{entity['entity']}"] += 1
        
        return [t for t, _ in triggers.most_common(5)]
    
    def _identify_common_outcomes(self, sequence: CascadeSequence,
                                 positions: List[int]) -> List[str]:
        """Identify common outcomes of motif occurrences."""
        outcomes = Counter()
        
        for pos in positions:
            if pos < len(sequence.cascades) - 1:
                next_cascade = sequence.cascades[pos+1]
                
                # Check cascade type
                outcomes[f"cascade_type:{next_cascade.cascade_type}"] += 1
                
                # Check frame
                outcomes[f"frame:{next_cascade.primary_frame}"] += 1
                
                # Check intensity change
                curr_cascade = sequence.cascades[pos]
                intensity_change = next_cascade.intensity_score - curr_cascade.intensity_score
                if intensity_change > 0.2:
                    outcomes["intensity_increase"] += 1
                elif intensity_change < -0.2:
                    outcomes["intensity_decrease"] += 1
        
        return [o for o, _ in outcomes.most_common(5)]
    
    def _calculate_motif_significance(self, n_occurrences: int,
                                     sequence_length: int,
                                     motif_length: int) -> float:
        """Calculate statistical significance of a motif."""
        # Expected occurrences under random assumption
        expected = (sequence_length - motif_length + 1) / (2 ** motif_length)
        
        # Poisson test
        from scipy.stats import poisson
        p_value = 1 - poisson.cdf(n_occurrences - 1, expected)
        
        return 1 - p_value
    
    def _track_narrative_evolution(self, sequence: CascadeSequence) -> Optional[NarrativeEvolution]:
        """Track narrative evolution through the sequence."""
        if len(sequence.cascades) < 3:
            return None
        
        # Identify core entities (present in >50% of cascades)
        entity_counts = Counter()
        for cascade in sequence.cascades:
            for entity_data in cascade.entities_involved:
                entity_counts[entity_data['entity']] += 1
        
        threshold = len(sequence.cascades) * 0.5
        core_entities = {e for e, c in entity_counts.items() if c >= threshold}
        
        # Track evolving entities
        evolving_entities = defaultdict(list)
        for i, cascade in enumerate(sequence.cascades):
            cascade_entities = {e['entity'] for e in cascade.entities_involved}
            new_entities = cascade_entities - core_entities
            
            for entity in new_entities:
                evolving_entities[entity].append((cascade.peak_date, cascade.cascade_id))
        
        # Track dominant frames
        dominant_frames = [(c.peak_date, c.primary_frame) for c in sequence.cascades]
        
        # Identify frame transitions
        frame_transitions = []
        for i in range(len(sequence.cascades) - 1):
            if sequence.cascades[i].primary_frame != sequence.cascades[i+1].primary_frame:
                frame_transitions.append({
                    'position': i,
                    'from_frame': sequence.cascades[i].primary_frame,
                    'to_frame': sequence.cascades[i+1].primary_frame,
                    'date': sequence.cascades[i+1].start_date
                })
        
        # Calculate semantic drift
        drift_trajectory = []
        for i in range(1, len(sequence.cascades)):
            e1 = set(e['entity'] for e in sequence.cascades[i-1].entities_involved)
            e2 = set(e['entity'] for e in sequence.cascades[i].entities_involved)
            
            if e1 and e2:
                drift = 1 - (len(e1 & e2) / len(e1 | e2))
                drift_trajectory.append(drift)
        
        semantic_drift = np.mean(drift_trajectory) if drift_trajectory else 0.0
        
        # Identify mutations
        mutations = []
        for shift in sequence.narrative_shifts:
            if shift['shift_score'] > 0.7:
                mutations.append({
                    'position': shift['position'],
                    'date': shift['date'],
                    'shift_score': shift['shift_score'],
                    'type': 'major_shift'
                })
        
        # Calculate coherence and consistency
        narrative_coherence = sequence.coherence_score
        
        # Thematic consistency based on frame stability
        frame_changes = len(frame_transitions)
        thematic_consistency = 1 - (frame_changes / (len(sequence.cascades) - 1)) \
                             if len(sequence.cascades) > 1 else 1.0
        
        return NarrativeEvolution(
            evolution_id=str(uuid.uuid4()),
            core_entities=core_entities,
            evolving_entities=dict(evolving_entities),
            dominant_frames=dominant_frames,
            frame_transitions=frame_transitions,
            semantic_drift=semantic_drift,
            drift_trajectory=drift_trajectory,
            mutations=mutations,
            mutation_rate=len(mutations) / len(sequence.cascades),
            narrative_coherence=narrative_coherence,
            thematic_consistency=thematic_consistency
        )
    
    def _identify_information_pathways(self, sequence: CascadeSequence) -> List[InformationPathway]:
        """Identify information flow pathways through the sequence."""
        pathways = []
        
        # Use dependency graph to find paths
        G = sequence.dependency_graph
        
        if not G or G.number_of_nodes() < 2:
            return pathways
        
        # Find all paths from sources to sinks
        sources = [n for n in G.nodes() if G.in_degree(n) == 0]
        sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
        
        for source in sources:
            for sink in sinks:
                if source != sink:
                    try:
                        # Find all simple paths
                        paths = list(nx.all_simple_paths(G, source, sink, cutoff=5))
                        
                        for path in paths[:3]:  # Limit to top 3 paths
                            if len(path) >= 2:
                                pathway = self._analyze_information_pathway(
                                    path, sequence
                                )
                                if pathway:
                                    pathways.append(pathway)
                    except nx.NetworkXNoPath:
                        continue
        
        return pathways
    
    def _analyze_information_pathway(self, path: List[str],
                                    sequence: CascadeSequence) -> Optional[InformationPathway]:
        """Analyze a specific information pathway."""
        if len(path) < 2:
            return None
        
        # Map cascade IDs to cascades
        cascade_map = {c.cascade_id: c for c in sequence.cascades}
        
        path_cascades = [cascade_map[cid] for cid in path if cid in cascade_map]
        
        if len(path_cascades) < 2:
            return None
        
        # Calculate flow strength
        flow_strengths = []
        for i in range(len(path_cascades) - 1):
            strength = self._calculate_directional_influence(
                path_cascades[i], path_cascades[i+1]
            )
            flow_strengths.append(strength)
        
        flow_strength = np.mean(flow_strengths) if flow_strengths else 0.0
        
        # Calculate flow speed
        time_gaps = []
        for i in range(len(path_cascades) - 1):
            gap = (path_cascades[i+1].start_date - path_cascades[i].end_date).days
            time_gaps.append(max(1, gap))
        
        flow_speed = 1 / np.mean(time_gaps) if time_gaps else 0.0
        
        # Identify bottlenecks (slow transitions)
        bottlenecks = []
        if time_gaps:
            threshold = np.percentile(time_gaps, 75)
            for i, gap in enumerate(time_gaps):
                if gap > threshold:
                    bottlenecks.append(path[i+1])
        
        # Identify amplifiers (intensity increases)
        amplifiers = []
        for i in range(len(path_cascades) - 1):
            if path_cascades[i+1].intensity_score > path_cascades[i].intensity_score * 1.5:
                amplifiers.append(path[i+1])
        
        # Calculate content preservation
        content_preservation_scores = []
        for i in range(len(path_cascades) - 1):
            e1 = set(e['entity'] for e in path_cascades[i].entities_involved)
            e2 = set(e['entity'] for e in path_cascades[i+1].entities_involved)
            
            if e1:
                preservation = len(e1 & e2) / len(e1)
                content_preservation_scores.append(preservation)
        
        content_preservation = np.mean(content_preservation_scores) \
                              if content_preservation_scores else 0.0
        
        # Track content mutations
        content_mutations = []
        for i in range(len(path_cascades) - 1):
            e1 = set(e['entity'] for e in path_cascades[i].entities_involved)
            e2 = set(e['entity'] for e in path_cascades[i+1].entities_involved)
            
            new_entities = e2 - e1
            lost_entities = e1 - e2
            
            if new_entities or lost_entities:
                content_mutations.append({
                    'position': i,
                    'new_entities': list(new_entities)[:5],
                    'lost_entities': list(lost_entities)[:5],
                    'cascade_id': path[i+1]
                })
        
        # Track media and journalist paths
        media_path = [list(c.media_involved)[:10] for c in path_cascades]
        journalist_path = [list(c.journalists_involved)[:10] for c in path_cascades]
        
        return InformationPathway(
            pathway_id=str(uuid.uuid4()),
            source_cascade=path[0],
            sink_cascade=path[-1],
            intermediate_cascades=path[1:-1],
            flow_strength=flow_strength,
            flow_speed=flow_speed,
            bottlenecks=bottlenecks,
            amplifiers=amplifiers,
            content_preservation=content_preservation,
            content_mutations=content_mutations,
            media_path=media_path,
            journalist_path=journalist_path
        )
    
    def get_sequence_summary(self, sequence: CascadeSequence) -> Dict[str, Any]:
        """Get summary of a detected sequence."""
        return {
            'sequence_id': sequence.sequence_id,
            'type': sequence.sequence_type,
            'length': sequence.length,
            'duration_days': sequence.duration_days,
            'period': {
                'start': sequence.start_date.isoformat(),
                'end': sequence.end_date.isoformat()
            },
            'trend': sequence.trend,
            'periodicity': sequence.periodicity,
            'cascades': sequence.cascade_order,
            'coherence': sequence.coherence_score,
            'predictability': sequence.predictability,
            'entropy': sequence.entropy,
            'significance': sequence.significance,
            'confidence': sequence.confidence,
            'key_propagators': sequence.key_propagators[:5],
            'narrative_shifts': len(sequence.narrative_shifts),
            'statistics': {
                'avg_intensity': np.mean([c.intensity_score for c in sequence.cascades]),
                'avg_velocity': np.mean([c.velocity_score for c in sequence.cascades]),
                'total_media': len(set().union(*[c.media_involved for c in sequence.cascades])),
                'total_entities': len(set().union(*[
                    set(e['entity'] for e in c.entities_involved) for c in sequence.cascades
                ]))
            }
        }
    
    def export_sequences(self, output_path: str) -> None:
        """Export detected sequences to file."""
        import json
        
        export_data = {
            'metadata': {
                'detector': self.name,
                'n_sequences': len(self.detected_sequences),
                'n_motifs': len(self.temporal_motifs),
                'n_evolutions': len(self.narrative_evolutions),
                'n_pathways': len(self.information_pathways),
                'timestamp': datetime.now().isoformat()
            },
            'sequences': [
                self.get_sequence_summary(seq) for seq in self.detected_sequences
            ],
            'motifs': [
                {
                    'motif_id': m.motif_id,
                    'type': m.motif_type,
                    'frequency': m.frequency,
                    'regularity': m.regularity,
                    'significance': m.statistical_significance
                } for m in self.temporal_motifs
            ],
            'narrative_evolutions': [
                {
                    'evolution_id': e.evolution_id,
                    'semantic_drift': e.semantic_drift,
                    'mutation_rate': e.mutation_rate,
                    'coherence': e.narrative_coherence,
                    'consistency': e.thematic_consistency
                } for e in self.narrative_evolutions
            ],
            'pathways': [
                {
                    'pathway_id': p.pathway_id,
                    'source': p.source_cascade,
                    'sink': p.sink_cascade,
                    'length': len(p.intermediate_cascades) + 2,
                    'flow_strength': p.flow_strength,
                    'flow_speed': p.flow_speed,
                    'content_preservation': p.content_preservation
                } for p in self.information_pathways
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported sequences to {output_path}")
    
    def validate_detection(self, detection_result: Dict[str, Any]) -> bool:
        """
        Validate sequence detection results.
        
        Args:
            detection_result: Results from detect() method
            
        Returns:
            True if detection is valid, False otherwise
        """
        try:
            # Check if detection_result is a list of CascadeSequence
            if isinstance(detection_result, list):
                sequences = detection_result
            elif isinstance(detection_result, dict) and 'sequences' in detection_result:
                sequences = detection_result['sequences']
            else:
                logger.warning("Invalid detection result format")
                return False
            
            if not isinstance(sequences, list):
                logger.warning("Sequences should be a list")
                return False
            
            # Empty results are valid
            if len(sequences) == 0:
                logger.info("No sequences found - empty but valid")
                return True
            
            # Validate each sequence
            for seq in sequences:
                if not isinstance(seq, CascadeSequence):
                    logger.warning("Invalid sequence type")
                    return False
                
                # Check basic validity
                if not seq.cascades:
                    logger.warning(f"Empty cascade list in sequence {seq.sequence_id}")
                    return False
                
                if seq.length != len(seq.cascades):
                    logger.warning(f"Length mismatch in sequence {seq.sequence_id}")
                    return False
                
                if seq.length != len(seq.cascade_order):
                    logger.warning(f"Order list mismatch in sequence {seq.sequence_id}")
                    return False
                
                # Check temporal validity
                if seq.end_date < seq.start_date:
                    logger.warning(f"Invalid date range in sequence {seq.sequence_id}")
                    return False
                
                # Check dependency graph
                if not isinstance(seq.dependency_graph, nx.DiGraph):
                    logger.warning(f"Invalid dependency graph in sequence {seq.sequence_id}")
                    return False
                
                # Validate trend
                valid_trends = ['increasing', 'decreasing', 'stable', 'oscillating']
                if seq.trend not in valid_trends:
                    logger.warning(f"Invalid trend '{seq.trend}' in sequence {seq.sequence_id}")
                    return False
                
                # Validate sequence type
                valid_types = ['chain', 'cycle', 'evolution', 'mutation', 'branching']
                if seq.sequence_type not in valid_types:
                    logger.warning(f"Invalid type '{seq.sequence_type}' in sequence {seq.sequence_id}")
                    return False
                
                # Check coherence score
                if not (0 <= seq.coherence_score <= 1):
                    logger.warning(f"Invalid coherence score in sequence {seq.sequence_id}")
                    return False
            
            # Check for additional result components if present
            if isinstance(detection_result, dict):
                if 'temporal_motifs' in detection_result:
                    motifs = detection_result['temporal_motifs']
                    if not isinstance(motifs, list):
                        logger.warning("Temporal motifs should be a list")
                        return False
                    
                    for motif in motifs:
                        if not isinstance(motif, TemporalMotif):
                            logger.warning("Invalid temporal motif type")
                            return False
                
                if 'narrative_evolutions' in detection_result:
                    evolutions = detection_result['narrative_evolutions']
                    if not isinstance(evolutions, list):
                        logger.warning("Narrative evolutions should be a list")
                        return False
                    
                    for evolution in evolutions:
                        if not isinstance(evolution, NarrativeEvolution):
                            logger.warning("Invalid narrative evolution type")
                            return False
                
                if 'information_pathways' in detection_result:
                    pathways = detection_result['information_pathways']
                    if not isinstance(pathways, list):
                        logger.warning("Information pathways should be a list")
                        return False
                    
                    for pathway in pathways:
                        if not isinstance(pathway, InformationPathway):
                            logger.warning("Invalid information pathway type")
                            return False
            
            logger.info(f"Validation successful: {len(sequences)} sequences")
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def score_detection(self, detection_result: Dict[str, Any]) -> float:
        """
        Score the quality of sequence detection.
        
        Scoring factors:
        - Sequence length and complexity
        - Coherence and predictability
        - Statistical significance
        - Temporal pattern quality
        - Narrative evolution clarity
        
        Args:
            detection_result: Results from detect() method
            
        Returns:
            Score between 0 and 1
        """
        try:
            if not self.validate_detection(detection_result):
                return 0.0
            
            # Extract sequences
            if isinstance(detection_result, list):
                sequences = detection_result
                motifs = []
                evolutions = []
                pathways = []
            else:
                sequences = detection_result.get('sequences', [])
                motifs = detection_result.get('temporal_motifs', [])
                evolutions = detection_result.get('narrative_evolutions', [])
                pathways = detection_result.get('information_pathways', [])
            
            if not sequences:
                logger.info("No sequences detected - score 0.0")
                return 0.0
            
            scores = []
            weights = []
            
            # 1. Sequence quality score (30% weight)
            seq_lengths = [seq.length for seq in sequences]
            seq_coherences = [seq.coherence_score for seq in sequences]
            seq_significances = [seq.statistical_significance for seq in sequences]
            
            avg_length = np.mean(seq_lengths)
            avg_coherence = np.mean(seq_coherences) if seq_coherences else 0
            avg_significance = np.mean(seq_significances) if seq_significances else 0
            
            # Good sequences are long (5+), coherent, and significant
            length_score = min(1.0, avg_length / 5)
            seq_quality = (length_score + avg_coherence + avg_significance) / 3
            
            scores.append(seq_quality)
            weights.append(0.3)
            logger.debug(f"Sequence quality score: {seq_quality:.3f}")
            
            # 2. Temporal pattern score (20% weight)
            if motifs:
                motif_regularities = [m.regularity for m in motifs]
                motif_frequencies = [m.frequency for m in motifs]
                
                avg_regularity = np.mean(motif_regularities)
                avg_frequency = np.mean(motif_frequencies)
                
                # Normalize frequency (assume 10+ occurrences is good)
                freq_score = min(1.0, avg_frequency / 10)
                temporal_score = (avg_regularity + freq_score) / 2
            else:
                temporal_score = 0.3  # Base score if no motifs
            
            scores.append(temporal_score)
            weights.append(0.2)
            logger.debug(f"Temporal pattern score: {temporal_score:.3f}")
            
            # 3. Narrative evolution score (20% weight)
            if evolutions:
                coherences = [e.narrative_coherence for e in evolutions]
                consistencies = [e.thematic_consistency for e in evolutions]
                
                avg_nar_coherence = np.mean(coherences)
                avg_consistency = np.mean(consistencies)
                
                narrative_score = (avg_nar_coherence + avg_consistency) / 2
            else:
                narrative_score = 0.3  # Base score if no evolutions
            
            scores.append(narrative_score)
            weights.append(0.2)
            logger.debug(f"Narrative evolution score: {narrative_score:.3f}")
            
            # 4. Information pathway score (15% weight)
            if pathways:
                flow_strengths = [p.flow_strength for p in pathways]
                preservations = [p.content_preservation for p in pathways]
                
                avg_flow = np.mean(flow_strengths)
                avg_preservation = np.mean(preservations)
                
                pathway_score = (avg_flow + avg_preservation) / 2
            else:
                pathway_score = 0.3  # Base score if no pathways
            
            scores.append(pathway_score)
            weights.append(0.15)
            logger.debug(f"Information pathway score: {pathway_score:.3f}")
            
            # 5. Diversity score (15% weight)
            seq_types = Counter(seq.sequence_type for seq in sequences)
            type_diversity = len(seq_types) / 5  # 5 possible types
            
            # Check for different patterns
            has_chains = any(s.sequence_type == 'chain' for s in sequences)
            has_cycles = any(s.sequence_type == 'cycle' for s in sequences)
            has_evolutions = any(s.sequence_type == 'evolution' for s in sequences)
            
            pattern_diversity = (has_chains + has_cycles + has_evolutions) / 3
            diversity_score = (type_diversity + pattern_diversity) / 2
            
            scores.append(diversity_score)
            weights.append(0.15)
            logger.debug(f"Diversity score: {diversity_score:.3f}")
            
            # Calculate weighted average
            total_score = np.average(scores, weights=weights)
            
            # Apply bonuses and penalties
            
            # Bonus for finding long sequences
            if any(seq.length >= 10 for seq in sequences):
                total_score = min(1.0, total_score * 1.1)
                logger.debug("Applied bonus for long sequences")
            
            # Bonus for high-confidence sequences
            if any(seq.confidence_score > 0.9 for seq in sequences):
                total_score = min(1.0, total_score * 1.05)
                logger.debug("Applied bonus for high-confidence sequences")
            
            # Penalty for only short sequences
            if all(seq.length <= 2 for seq in sequences):
                total_score *= 0.7
                logger.debug("Applied penalty for only short sequences")
            
            # Penalty for low significance
            if all(seq.statistical_significance < 0.5 for seq in sequences):
                total_score *= 0.8
                logger.debug("Applied penalty for low significance")
            
            logger.info(f"Sequence detection score: {total_score:.3f}")
            return float(total_score)
            
        except Exception as e:
            logger.error(f"Scoring error: {e}")
            return 0.0