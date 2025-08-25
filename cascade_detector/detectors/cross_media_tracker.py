"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
cross_media_tracker.py

MAIN OBJECTIVE:
---------------
This script tracks and analyzes media behavior patterns across cascades, identifying coordination,
leader-follower relationships, and cross-media information flow dynamics.

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

MAIN FEATURES:
--------------
1) Media coordination and synchronization detection
2) Leader-follower relationship identification
3) Media clustering and alliance discovery
4) Frame adoption pattern analysis by media outlet
5) Cross-media information flow tracking

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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import logging
import warnings
from tqdm import tqdm
import uuid

# Import base components
from cascade_detector.detectors.base_detector import BaseDetector, DetectionContext
from cascade_detector.detectors.cascade_detector import (
    CascadeDetector, CompleteCascade, EnhancedBurst
)
from cascade_detector.detectors.multi_frame_detector import (
    MultiFrameDetector, MultiFramePattern, FrameInteraction
)
from cascade_detector.core.config import DetectorConfig

# Import metrics
from cascade_detector.metrics.scientific_network_metrics import ScientificNetworkMetrics
from cascade_detector.metrics.diversity_metrics import DiversityMetrics

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class MediaProfile:
    """
    Comprehensive profile of a media outlet's behavior.
    """
    media_id: str
    media_name: str
    
    # Activity metrics
    total_articles: int
    total_cascades: int
    active_frames: Set[str]
    dominant_frame: Optional[str]
    
    # Temporal patterns
    activity_timeline: List[Dict[str, Any]]  # Date -> activity
    burst_participation: List[str]  # Burst IDs participated in
    average_response_time: float  # Hours to join cascade
    consistency_score: float  # Regularity of participation
    
    # Network position
    centrality_scores: Dict[str, float]  # Various centrality measures
    influence_score: float  # Overall influence metric
    authority_score: float  # Authority in network
    hub_score: float  # Hub score in network
    
    # Behavioral patterns
    leader_score: float  # Tendency to lead cascades (0-1)
    follower_score: float  # Tendency to follow (0-1)
    amplifier_score: float  # Tendency to amplify (0-1)
    innovator_score: float  # Tendency to introduce new elements
    
    # Journalist network
    journalists: Set[str]  # All journalists
    key_journalists: List[Tuple[str, float]]  # Top journalists with scores
    journalist_mobility: float  # Journalist turnover rate
    
    # Content patterns
    entity_preferences: Dict[str, float]  # Preferred entities to cover
    frame_preferences: Dict[str, float]  # Frame adoption preferences
    topic_diversity: float  # Diversity of topics covered
    
    # Collaboration patterns
    frequent_collaborators: List[Tuple[str, float]]  # Media outlets frequently covered together
    competitive_outlets: List[Tuple[str, float]]  # Competing outlets
    
    def get_behavioral_type(self) -> str:
        """Classify media behavioral type."""
        scores = {
            'leader': self.leader_score,
            'follower': self.follower_score,
            'amplifier': self.amplifier_score,
            'innovator': self.innovator_score
        }
        return max(scores, key=scores.get)


@dataclass
class MediaCoordination:
    """
    Detected coordination pattern between media outlets.
    """
    coordination_id: str
    media_group: List[str]  # Media outlets involved
    
    # Temporal data
    start_date: datetime
    end_date: datetime
    duration_days: int
    
    # Coordination metrics
    synchronization_score: float  # How synchronized (0-1)
    coordination_strength: float  # Strength of coordination
    coordination_type: str  # 'simultaneous', 'sequential', 'hierarchical'
    
    # Pattern details
    shared_cascades: List[str]  # Cascade IDs
    shared_frames: Set[str]  # Frames covered together
    shared_entities: Set[str]  # Entities covered together
    
    # Temporal patterns
    lag_times: Dict[str, float]  # Media -> average lag time
    response_pattern: str  # 'immediate', 'delayed', 'staggered'
    
    # Network structure
    coordination_network: nx.Graph  # Network of coordinated media
    central_media: List[str]  # Most central in coordination
    
    # Statistical validation
    significance: float  # Statistical significance
    random_probability: float  # Probability of random occurrence
    
    def is_significant(self) -> bool:
        """Check if coordination is statistically significant."""
        return self.significance > 0.95 and self.random_probability < 0.05


@dataclass
class MediaCluster:
    """
    Cluster of media outlets with similar behavior.
    """
    cluster_id: str
    cluster_type: str  # 'behavioral', 'temporal', 'topical', 'network'
    
    # Members
    core_members: List[str]  # Core cluster members
    peripheral_members: List[str]  # Peripheral members
    
    # Cluster characteristics
    coherence_score: float  # Internal coherence (0-1)
    separation_score: float  # Separation from other clusters
    stability_score: float  # Temporal stability
    
    # Behavioral profile
    dominant_behavior: str  # 'leader', 'follower', 'amplifier', 'innovator'
    frame_preferences: Dict[str, float]  # Preferred frames
    temporal_pattern: str  # 'early_adopter', 'mainstream', 'late_adopter'
    
    # Network properties
    internal_density: float  # Density within cluster
    external_connections: int  # Connections outside cluster
    bridge_media: List[str]  # Media connecting to other clusters
    
    # Evolution
    formation_date: Optional[datetime]
    peak_date: Optional[datetime]
    dissolution_date: Optional[datetime]
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get cluster summary statistics."""
        return {
            'id': self.cluster_id,
            'type': self.cluster_type,
            'size': len(self.core_members) + len(self.peripheral_members),
            'coherence': self.coherence_score,
            'stability': self.stability_score,
            'behavior': self.dominant_behavior
        }


@dataclass
class JournalistNetwork:
    """
    Network of journalist relationships and mobility.
    """
    # Journalist profiles
    journalist_profiles: Dict[str, Dict[str, Any]]  # Journalist -> profile
    
    # Mobility patterns
    journalist_moves: List[Dict[str, Any]]  # Journalist movement records
    mobility_rate: float  # Overall mobility rate
    
    # Influence network
    influence_network: nx.DiGraph  # Directed influence network
    influential_journalists: List[Tuple[str, float]]  # Top influencers
    
    # Collaboration patterns
    collaboration_network: nx.Graph  # Collaboration network
    collaboration_clusters: List[Set[str]]  # Collaboration groups
    
    # Cross-media bridges
    bridge_journalists: List[str]  # Journalists connecting media
    
    def get_key_journalists(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top n influential journalists."""
        return self.influential_journalists[:n]


@dataclass
class InformationFlow:
    """
    Information flow pattern between media outlets.
    """
    flow_id: str
    source_media: str
    target_media: str
    
    # Flow characteristics
    flow_strength: float  # Strength of flow (0-1)
    flow_direction: str  # 'unidirectional', 'bidirectional', 'reciprocal'
    flow_type: str  # 'direct', 'indirect', 'cascading'
    
    # Content
    transmitted_entities: Set[str]  # Entities that flow
    transmitted_frames: Set[str]  # Frames that flow
    
    # Temporal
    average_lag: float  # Average time lag in hours
    consistency: float  # Consistency of flow pattern
    
    # Validation
    causality_score: float  # Granger causality or similar
    confidence: float  # Confidence in flow detection


class CrossMediaTracker(BaseDetector):
    """
    Tracks and analyzes media behavior patterns across cascades and frames.
    
    This detector:
    1. Builds comprehensive media profiles from cascade participation
    2. Detects media coordination and synchronization patterns
    3. Identifies media clusters and alliances
    4. Tracks journalist networks and mobility
    5. Maps information flow between media outlets
    6. Analyzes frame adoption strategies by media
    
    Integrates with:
    - CascadeDetector: Uses cascade data for media analysis
    - MultiFrameDetector: Uses frame patterns for media-frame analysis
    - Prepares for: EchoChamberDetector and PolarizationDetector
    """
    
    def __init__(self,
                 context: DetectionContext,
                 config: Optional[DetectorConfig] = None,
                 cascade_detector: Optional[CascadeDetector] = None,
                 multi_frame_detector: Optional[MultiFrameDetector] = None):
        """
        Initialize cross-media tracker.
        
        Args:
            context: Detection context with all indices and metrics
            config: Optional configuration
            cascade_detector: Optional pre-configured cascade detector
            multi_frame_detector: Optional pre-configured multi-frame detector
        """
        super().__init__(context, config)
        
        # Initialize detectors
        self.cascade_detector = cascade_detector or CascadeDetector(context, config)
        self.multi_frame_detector = multi_frame_detector or MultiFrameDetector(
            context, config, self.cascade_detector
        )
        
        # Network metrics calculator
        self.network_metrics = ScientificNetworkMetrics(
            source_index=context.source_index,
            entity_index=context.entity_index,
            frame_index=context.frame_index,
            config={'exact_computation': True}
        )
        
        # Configuration
        self.min_media_activity = 5  # Minimum articles for analysis
        self.coordination_threshold = 0.6  # Threshold for coordination detection
        self.cluster_min_size = 3  # Minimum cluster size
        self.flow_min_strength = 0.3  # Minimum flow strength
        self.lag_window_hours = 48  # Window for lag analysis
        
        # Cache
        self._media_profiles_cache = {}
        self._coordination_cache = {}
        self._flow_cache = {}
        
        logger.info("CrossMediaTracker initialized")
        logger.info(f"  - Tracking media behavior across cascades")
        logger.info(f"  - Coordination threshold: {self.coordination_threshold}")
        logger.info(f"  - Minimum media activity: {self.min_media_activity} articles")
    
    def detect(self, **kwargs) -> Dict[str, Any]:
        """
        Detect and analyze cross-media patterns.
        
        Args:
            cascades: Optional pre-detected cascades
            multi_frame_results: Optional multi-frame detection results
            window: Optional time window
            media_list: Optional specific media outlets to analyze
            
        Returns:
            Dictionary containing:
            - media_profiles: Dict of MediaProfile objects
            - media_coordinations: List of MediaCoordination patterns
            - media_clusters: List of MediaCluster objects
            - journalist_network: JournalistNetwork analysis
            - information_flows: List of InformationFlow patterns
            - media_network: Overall media relationship network
        """
        # Get cascades
        cascades = kwargs.get('cascades')
        if cascades is None:
            logger.info("Detecting cascades first...")
            cascades = self.cascade_detector.detect(**kwargs)
        
        if not cascades:
            logger.warning("No cascades found - cannot analyze media patterns")
            return self._empty_results()
        
        # Get multi-frame results
        multi_frame_results = kwargs.get('multi_frame_results')
        if multi_frame_results is None:
            logger.info("Running multi-frame detection...")
            multi_frame_results = self.multi_frame_detector.detect(cascades=cascades, **kwargs)
        
        window = kwargs.get('window', self.context.time_window)
        media_list = kwargs.get('media_list')
        
        logger.info("="*80)
        logger.info("CROSS-MEDIA TRACKING")
        logger.info("="*80)
        logger.info(f"Analyzing {len(cascades)} cascades")
        logger.info(f"Period: {window[0].date()} to {window[1].date()}")
        
        # Step 1: Build media profiles
        logger.info("\nStep 1: Building media profiles...")
        media_profiles = self._build_media_profiles(cascades, multi_frame_results, media_list)
        logger.info(f"  Built profiles for {len(media_profiles)} media outlets")
        
        # Step 2: Detect media coordination
        logger.info("\nStep 2: Detecting media coordination patterns...")
        coordinations = self._detect_media_coordination(cascades, media_profiles)
        logger.info(f"  Found {len(coordinations)} coordination patterns")
        
        # Step 3: Identify media clusters
        logger.info("\nStep 3: Identifying media clusters...")
        clusters = self._identify_media_clusters(media_profiles, cascades)
        logger.info(f"  Found {len(clusters)} media clusters")
        
        # Step 4: Analyze journalist networks
        logger.info("\nStep 4: Analyzing journalist networks...")
        journalist_network = self._analyze_journalist_network(cascades, media_profiles)
        logger.info(f"  Analyzed {len(journalist_network.journalist_profiles)} journalists")
        
        # Step 5: Map information flows
        logger.info("\nStep 5: Mapping information flows...")
        information_flows = self._map_information_flows(cascades, media_profiles)
        logger.info(f"  Identified {len(information_flows)} information flow patterns")
        
        # Step 6: Build comprehensive media network
        logger.info("\nStep 6: Building comprehensive media network...")
        media_network = self._build_media_network(
            media_profiles, coordinations, clusters, information_flows
        )
        logger.info(f"  Network has {media_network.number_of_nodes()} nodes, "
                   f"{media_network.number_of_edges()} edges")
        
        # Step 7: Analyze frame adoption by media
        logger.info("\nStep 7: Analyzing frame adoption patterns...")
        frame_adoption = self._analyze_frame_adoption(
            cascades, media_profiles, multi_frame_results
        )
        
        # Compile results
        results = {
            'media_profiles': media_profiles,
            'media_coordinations': coordinations,
            'media_clusters': clusters,
            'journalist_network': journalist_network,
            'information_flows': information_flows,
            'media_network': media_network,
            'frame_adoption': frame_adoption,
            'summary': self._generate_summary(
                media_profiles, coordinations, clusters, information_flows
            )
        }
        
        logger.info("\n" + "="*80)
        logger.info("CROSS-MEDIA TRACKING COMPLETE")
        logger.info("="*80)
        
        return results
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure."""
        return {
            'media_profiles': {},
            'media_coordinations': [],
            'media_clusters': [],
            'journalist_network': JournalistNetwork(
                journalist_profiles={},
                journalist_moves=[],
                mobility_rate=0.0,
                influence_network=nx.DiGraph(),
                influential_journalists=[],
                collaboration_network=nx.Graph(),
                collaboration_clusters=[],
                bridge_journalists=[]
            ),
            'information_flows': [],
            'media_network': nx.Graph(),
            'frame_adoption': {},
            'summary': {}
        }
    
    def _build_media_profiles(self,
                              cascades: List[CompleteCascade],
                              multi_frame_results: Dict[str, Any],
                              media_list: Optional[List[str]] = None) -> Dict[str, MediaProfile]:
        """Build comprehensive profiles for each media outlet."""
        profiles = {}
        
        # Collect media data from cascades
        media_data = defaultdict(lambda: {
            'articles': 0,
            'cascades': [],
            'frames': set(),
            'entities': Counter(),
            'journalists': set(),
            'bursts': [],
            'timestamps': []
        })
        
        for cascade in cascades:
            for media in cascade.all_media:
                if media_list and media not in media_list:
                    continue
                
                media_data[media]['cascades'].append(cascade.cascade_id)
                media_data[media]['articles'] += cascade.media_participation_scores.get(media, 0)
                media_data[media]['frames'].update(cascade.frames_involved)
                
                # Collect entities
                for entity in cascade.all_entities:
                    media_data[media]['entities'][entity] += 1
                
                # Collect journalists
                for journalist in cascade.all_journalists:
                    # Check if journalist works for this media
                    if self._journalist_works_for_media(journalist, media, cascade):
                        media_data[media]['journalists'].add(journalist)
                
                # Collect burst participation
                for burst in cascade.bursts:
                    if media in burst.media_involved:
                        media_data[media]['bursts'].append(id(burst))
                        media_data[media]['timestamps'].extend(burst.article_timestamps)
        
        # Build profiles
        for media, data in media_data.items():
            if data['articles'] < self.min_media_activity:
                continue
            
            profile = self._create_media_profile(media, data, cascades, multi_frame_results)
            if profile:
                profiles[media] = profile
        
        return profiles
    
    def _journalist_works_for_media(self,
                                    journalist: str,
                                    media: str,
                                    cascade: CompleteCascade) -> bool:
        """Check if journalist works for specific media outlet."""
        # Check in cascade's burst data
        for burst in cascade.bursts:
            for affiliation_media in burst.journalist_media_affiliations.get(journalist, set()):
                if affiliation_media == media:
                    return True
        return False
    
    def _create_media_profile(self,
                             media: str,
                             data: Dict,
                             cascades: List[CompleteCascade],
                             multi_frame_results: Dict[str, Any]) -> Optional[MediaProfile]:
        """Create detailed media profile."""
        try:
            # Calculate temporal patterns
            activity_timeline = self._calculate_activity_timeline(media, cascades)
            avg_response_time = self._calculate_average_response_time(media, cascades)
            consistency_score = self._calculate_consistency_score(activity_timeline)
            
            # Calculate network metrics
            centrality_scores = self._calculate_media_centrality(media, cascades)
            influence_score = centrality_scores.get('eigenvector', 0.0)
            authority_score = centrality_scores.get('authority', 0.0)
            hub_score = centrality_scores.get('hub', 0.0)
            
            # Calculate behavioral scores
            behavioral_scores = self._calculate_behavioral_scores(media, data, cascades)
            
            # Identify key journalists
            key_journalists = self._identify_key_journalists(
                data['journalists'], cascades
            )
            
            # Calculate journalist mobility
            journalist_mobility = self._calculate_journalist_mobility(
                data['journalists'], cascades
            )
            
            # Determine dominant frame
            frame_counts = Counter()
            for cascade_id in data['cascades']:
                cascade = next((c for c in cascades if c.cascade_id == cascade_id), None)
                if cascade:
                    frame_counts.update(cascade.frames_involved)
            
            dominant_frame = max(frame_counts, key=frame_counts.get) if frame_counts else None
            
            # Calculate entity preferences
            total_entities = sum(data['entities'].values())
            entity_preferences = {
                entity: count / total_entities
                for entity, count in data['entities'].most_common(20)
            } if total_entities > 0 else {}
            
            # Calculate frame preferences
            frame_preferences = {
                frame: count / len(data['cascades'])
                for frame, count in frame_counts.items()
            } if data['cascades'] else {}
            
            # Calculate topic diversity
            topic_diversity = self._calculate_diversity(list(data['entities'].keys()))
            
            # Identify collaborators and competitors
            collaborators = self._identify_collaborators(media, cascades)
            competitors = self._identify_competitors(media, cascades, multi_frame_results)
            
            return MediaProfile(
                media_id=media,
                media_name=media,
                total_articles=data['articles'],
                total_cascades=len(data['cascades']),
                active_frames=data['frames'],
                dominant_frame=dominant_frame,
                activity_timeline=activity_timeline,
                burst_participation=data['bursts'],
                average_response_time=avg_response_time,
                consistency_score=consistency_score,
                centrality_scores=centrality_scores,
                influence_score=influence_score,
                authority_score=authority_score,
                hub_score=hub_score,
                leader_score=behavioral_scores['leader'],
                follower_score=behavioral_scores['follower'],
                amplifier_score=behavioral_scores['amplifier'],
                innovator_score=behavioral_scores['innovator'],
                journalists=data['journalists'],
                key_journalists=key_journalists,
                journalist_mobility=journalist_mobility,
                entity_preferences=entity_preferences,
                frame_preferences=frame_preferences,
                topic_diversity=topic_diversity,
                frequent_collaborators=collaborators[:10],
                competitive_outlets=competitors[:10]
            )
            
        except Exception as e:
            logger.error(f"Failed to create profile for {media}: {e}")
            return None
    
    def _calculate_activity_timeline(self,
                                     media: str,
                                     cascades: List[CompleteCascade]) -> List[Dict[str, Any]]:
        """Calculate media activity timeline."""
        timeline = []
        
        for cascade in cascades:
            if media in cascade.all_media:
                # Find when media joined cascade
                join_date = None
                for burst in cascade.bursts:
                    if media in burst.media_involved:
                        join_date = burst.burst_event.start_date
                        break
                
                if join_date:
                    timeline.append({
                        'date': join_date,
                        'cascade_id': cascade.cascade_id,
                        'articles': cascade.media_participation_scores.get(media, 0),
                        'frame': cascade.dominant_frame
                    })
        
        return sorted(timeline, key=lambda x: x['date'])
    
    def _calculate_average_response_time(self,
                                         media: str,
                                         cascades: List[CompleteCascade]) -> float:
        """Calculate average time for media to join cascades."""
        response_times = []
        
        for cascade in cascades:
            if media in cascade.all_media:
                # Find when media joined relative to cascade start
                for burst in cascade.bursts:
                    if media in burst.media_involved:
                        response_time = (burst.burst_event.start_date - cascade.start_date).total_seconds() / 3600
                        response_times.append(response_time)
                        break
        
        return np.mean(response_times) if response_times else 0.0
    
    def _calculate_consistency_score(self,
                                     timeline: List[Dict[str, Any]]) -> float:
        """Calculate consistency of media participation."""
        if len(timeline) < 2:
            return 0.0
        
        # Calculate intervals between participations
        intervals = []
        for i in range(1, len(timeline)):
            interval = (timeline[i]['date'] - timeline[i-1]['date']).days
            intervals.append(interval)
        
        if intervals:
            # Lower variance means higher consistency
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            cv = std_interval / (mean_interval + 1) if mean_interval > 0 else 1
            return 1.0 / (1.0 + cv)
        
        return 0.5
    
    def _calculate_media_centrality(self,
                                    media: str,
                                    cascades: List[CompleteCascade]) -> Dict[str, float]:
        """Calculate various centrality measures for media."""
        # Build media co-occurrence network
        G = nx.Graph()
        
        for cascade in cascades:
            media_list = list(cascade.all_media)
            for i, m1 in enumerate(media_list):
                G.add_node(m1)
                for m2 in media_list[i+1:]:
                    if G.has_edge(m1, m2):
                        G[m1][m2]['weight'] += 1
                    else:
                        G.add_edge(m1, m2, weight=1)
        
        if media not in G:
            return {}
        
        centrality = {}
        
        try:
            # Degree centrality
            centrality['degree'] = nx.degree_centrality(G).get(media, 0)
            
            # Betweenness centrality
            centrality['betweenness'] = nx.betweenness_centrality(G).get(media, 0)
            
            # Closeness centrality
            centrality['closeness'] = nx.closeness_centrality(G).get(media, 0)
            
            # Eigenvector centrality
            if G.number_of_nodes() > 2:
                centrality['eigenvector'] = nx.eigenvector_centrality_numpy(G, max_iter=100).get(media, 0)
            
            # HITS algorithm
            hubs, authorities = nx.hits(G, max_iter=100)
            centrality['hub'] = hubs.get(media, 0)
            centrality['authority'] = authorities.get(media, 0)
            
        except:
            pass
        
        return centrality
    
    def _calculate_behavioral_scores(self,
                                     media: str,
                                     data: Dict,
                                     cascades: List[CompleteCascade]) -> Dict[str, float]:
        """Calculate behavioral tendency scores for media."""
        scores = {
            'leader': 0.0,
            'follower': 0.0,
            'amplifier': 0.0,
            'innovator': 0.0
        }
        
        cascade_positions = []
        cascade_sizes = []
        entity_novelty = []
        
        for cascade_id in data['cascades']:
            cascade = next((c for c in cascades if c.cascade_id == cascade_id), None)
            if not cascade:
                continue
            
            # Determine media position in cascade
            media_position = self._get_media_position_in_cascade(media, cascade)
            cascade_positions.append(media_position)
            
            # Check cascade amplification
            cascade_sizes.append(cascade.total_articles)
            
            # Check entity novelty
            new_entities = 0
            for burst in cascade.bursts:
                if media in burst.media_involved:
                    new_entities += len(burst.new_entities)
            entity_novelty.append(new_entities)
        
        if cascade_positions:
            # Leader score: often among first
            scores['leader'] = len([p for p in cascade_positions if p < 0.3]) / len(cascade_positions)
            
            # Follower score: often among last
            scores['follower'] = len([p for p in cascade_positions if p > 0.7]) / len(cascade_positions)
            
            # Amplifier score: participates in large cascades
            avg_cascade_size = np.mean(cascade_sizes)
            overall_avg = np.mean([c.total_articles for c in cascades])
            scores['amplifier'] = min(1.0, avg_cascade_size / (overall_avg + 1))
            
            # Innovator score: introduces new entities
            scores['innovator'] = min(1.0, np.mean(entity_novelty) / 10)
        
        return scores
    
    def _get_media_position_in_cascade(self,
                                       media: str,
                                       cascade: CompleteCascade) -> float:
        """Get relative position of media in cascade (0=first, 1=last)."""
        # Find when media joined
        media_join_time = None
        for burst in cascade.bursts:
            if media in burst.media_involved:
                media_join_time = burst.burst_event.start_date
                break
        
        if not media_join_time:
            return 1.0
        
        # Calculate relative position
        duration = (cascade.end_date - cascade.start_date).total_seconds()
        if duration > 0:
            position = (media_join_time - cascade.start_date).total_seconds() / duration
            return min(1.0, max(0.0, position))
        
        return 0.5
    
    def _identify_key_journalists(self,
                                  journalists: Set[str],
                                  cascades: List[CompleteCascade]) -> List[Tuple[str, float]]:
        """Identify key journalists for a media outlet."""
        journalist_scores = {}
        
        for journalist in journalists:
            score = 0.0
            article_count = 0
            
            for cascade in cascades:
                if journalist in cascade.all_journalists:
                    # Get journalist's contribution
                    contribution = cascade.journalist_contributions.get(journalist, 0)
                    article_count += contribution
                    
                    # Get journalist's authority
                    authority = cascade.journalist_authority_scores.get(journalist, 0)
                    score += authority * contribution
            
            if article_count > 0:
                journalist_scores[journalist] = score / article_count
        
        # Sort by score
        sorted_journalists = sorted(
            journalist_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_journalists[:10]
    
    def _calculate_journalist_mobility(self,
                                       journalists: Set[str],
                                       cascades: List[CompleteCascade]) -> float:
        """Calculate journalist mobility/turnover rate."""
        if len(cascades) < 2:
            return 0.0
        
        # Track journalists over time
        early_journalists = set()
        late_journalists = set()
        
        mid_point = len(cascades) // 2
        
        for i, cascade in enumerate(sorted(cascades, key=lambda c: c.start_date)):
            cascade_journalists = cascade.all_journalists & journalists
            
            if i < mid_point:
                early_journalists.update(cascade_journalists)
            else:
                late_journalists.update(cascade_journalists)
        
        if not early_journalists:
            return 0.0
        
        # Calculate turnover
        retained = len(early_journalists & late_journalists)
        turnover = 1.0 - (retained / len(early_journalists))
        
        return turnover
    
    def _calculate_diversity(self, items: List[str]) -> float:
        """Calculate diversity using entropy."""
        if not items:
            return 0.0
        
        counts = Counter(items)
        total = sum(counts.values())
        
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log(p)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _identify_collaborators(self,
                                media: str,
                                cascades: List[CompleteCascade]) -> List[Tuple[str, float]]:
        """Identify media outlets that frequently collaborate."""
        collaborations = Counter()
        
        for cascade in cascades:
            if media in cascade.all_media:
                for other_media in cascade.all_media:
                    if other_media != media:
                        collaborations[other_media] += 1
        
        # Normalize by total cascades
        total = sum(collaborations.values())
        if total > 0:
            normalized = [
                (m, count / total)
                for m, count in collaborations.most_common()
            ]
            return normalized
        
        return []
    
    def _identify_competitors(self,
                             media: str,
                             cascades: List[CompleteCascade],
                             multi_frame_results: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Identify competing media outlets."""
        competitors = {}
        
        # Check frame interactions for competition
        frame_interactions = multi_frame_results.get('frame_interactions', [])
        
        for interaction in frame_interactions:
            if interaction.competition_intensity > 0.5:
                # Check if media are on different sides
                for cascade in cascades:
                    if media in cascade.all_media:
                        media_frame = cascade.dominant_frame
                        
                        for other_media in cascade.all_media:
                            if other_media != media:
                                # Check if they cover different frames
                                other_cascades = [
                                    c for c in cascades
                                    if other_media in c.all_media
                                ]
                                
                                for oc in other_cascades:
                                    if oc.dominant_frame != media_frame:
                                        competitors[other_media] = competitors.get(other_media, 0) + 1
        
        # Normalize scores
        if competitors:
            max_score = max(competitors.values())
            normalized = [
                (m, score / max_score)
                for m, score in sorted(competitors.items(), key=lambda x: x[1], reverse=True)
            ]
            return normalized
        
        return []
    
    def _detect_media_coordination(self,
                                   cascades: List[CompleteCascade],
                                   media_profiles: Dict[str, MediaProfile]) -> List[MediaCoordination]:
        """Detect coordination patterns between media outlets."""
        coordinations = []
        
        # Build temporal co-occurrence matrix
        cooccurrence_matrix = self._build_cooccurrence_matrix(cascades, media_profiles)
        
        # Find groups with high co-occurrence
        coordination_groups = self._find_coordination_groups(cooccurrence_matrix)
        
        for group in coordination_groups:
            coordination = self._analyze_coordination(group, cascades, media_profiles)
            if coordination and coordination.coordination_strength >= self.coordination_threshold:
                coordinations.append(coordination)
        
        return coordinations
    
    def _build_cooccurrence_matrix(self,
                                   cascades: List[CompleteCascade],
                                   media_profiles: Dict[str, MediaProfile]) -> np.ndarray:
        """Build media co-occurrence matrix."""
        media_list = list(media_profiles.keys())
        n_media = len(media_list)
        matrix = np.zeros((n_media, n_media))
        
        for cascade in cascades:
            cascade_media = [m for m in cascade.all_media if m in media_list]
            
            for i, m1 in enumerate(media_list):
                if m1 in cascade_media:
                    for j, m2 in enumerate(media_list):
                        if m2 in cascade_media and i != j:
                            matrix[i, j] += 1
        
        # Normalize
        max_val = matrix.max()
        if max_val > 0:
            matrix = matrix / max_val
        
        return matrix
    
    def _find_coordination_groups(self,
                                  cooccurrence_matrix: np.ndarray) -> List[List[int]]:
        """Find groups of coordinated media using clustering."""
        if cooccurrence_matrix.shape[0] < 3:
            return []
        
        # Use hierarchical clustering
        linkage_matrix = linkage(cooccurrence_matrix, method='ward')
        
        # Cut tree to get clusters
        clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')
        
        # Group indices by cluster
        groups = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            groups[cluster_id].append(idx)
        
        # Filter groups by size
        return [
            group for group in groups.values()
            if len(group) >= self.cluster_min_size
        ]
    
    def _analyze_coordination(self,
                              group_indices: List[int],
                              cascades: List[CompleteCascade],
                              media_profiles: Dict[str, MediaProfile]) -> Optional[MediaCoordination]:
        """Analyze coordination pattern for a media group."""
        try:
            media_list = list(media_profiles.keys())
            media_group = [media_list[i] for i in group_indices]
            
            # Find shared cascades
            shared_cascades = []
            shared_frames = set()
            shared_entities = set()
            
            for cascade in cascades:
                group_in_cascade = [m for m in media_group if m in cascade.all_media]
                if len(group_in_cascade) >= 2:
                    shared_cascades.append(cascade.cascade_id)
                    shared_frames.update(cascade.frames_involved)
                    shared_entities.update(cascade.all_entities)
            
            if not shared_cascades:
                return None
            
            # Calculate temporal patterns
            temporal_data = self._analyze_temporal_coordination(media_group, cascades)
            
            # Calculate synchronization score
            synchronization = self._calculate_synchronization(media_group, cascades)
            
            # Determine coordination type
            coord_type = self._determine_coordination_type(temporal_data)
            
            # Build coordination network
            coord_network = self._build_coordination_network(media_group, cascades)
            
            # Find central media
            if coord_network.number_of_nodes() > 0:
                centrality = nx.degree_centrality(coord_network)
                central_media = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                central_media = [m for m, _ in central_media[:3]]
            else:
                central_media = []
            
            # Calculate statistical significance
            significance, random_prob = self._calculate_coordination_significance(
                media_group, cascades, len(shared_cascades)
            )
            
            return MediaCoordination(
                coordination_id=str(uuid.uuid4())[:8],
                media_group=media_group,
                start_date=min(c.start_date for c in cascades if c.cascade_id in shared_cascades),
                end_date=max(c.end_date for c in cascades if c.cascade_id in shared_cascades),
                duration_days=len(set(
                    d.date() for c in cascades if c.cascade_id in shared_cascades
                    for d in pd.date_range(c.start_date, c.end_date)
                )),
                synchronization_score=synchronization,
                coordination_strength=len(shared_cascades) / len(cascades),
                coordination_type=coord_type,
                shared_cascades=shared_cascades,
                shared_frames=shared_frames,
                shared_entities=shared_entities,
                lag_times=temporal_data['lag_times'],
                response_pattern=temporal_data['pattern'],
                coordination_network=coord_network,
                central_media=central_media,
                significance=significance,
                random_probability=random_prob
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze coordination: {e}")
            return None
    
    def _analyze_temporal_coordination(self,
                                       media_group: List[str],
                                       cascades: List[CompleteCascade]) -> Dict[str, Any]:
        """Analyze temporal coordination patterns."""
        lag_times = {}
        timestamps = defaultdict(list)
        
        for cascade in cascades:
            cascade_media = [m for m in media_group if m in cascade.all_media]
            if len(cascade_media) >= 2:
                # Get join times for each media
                join_times = {}
                for media in cascade_media:
                    for burst in cascade.bursts:
                        if media in burst.media_involved:
                            join_times[media] = burst.burst_event.start_date
                            timestamps[media].append(burst.burst_event.start_date)
                            break
                
                # Calculate lags
                if join_times:
                    min_time = min(join_times.values())
                    for media, join_time in join_times.items():
                        lag = (join_time - min_time).total_seconds() / 3600
                        if media not in lag_times:
                            lag_times[media] = []
                        lag_times[media].append(lag)
        
        # Average lag times
        avg_lag_times = {
            media: np.mean(lags) if lags else 0
            for media, lags in lag_times.items()
        }
        
        # Determine response pattern
        if avg_lag_times:
            avg_lag = np.mean(list(avg_lag_times.values()))
            std_lag = np.std(list(avg_lag_times.values()))
            
            if avg_lag < 6:
                pattern = 'immediate'
            elif std_lag < 12:
                pattern = 'synchronized'
            else:
                pattern = 'staggered'
        else:
            pattern = 'unknown'
        
        return {
            'lag_times': avg_lag_times,
            'pattern': pattern,
            'timestamps': dict(timestamps)
        }
    
    def _calculate_synchronization(self,
                                   media_group: List[str],
                                   cascades: List[CompleteCascade]) -> float:
        """Calculate synchronization score for media group."""
        synchronization_scores = []
        
        for cascade in cascades:
            cascade_media = [m for m in media_group if m in cascade.all_media]
            if len(cascade_media) >= 2:
                # Get participation times
                times = []
                for media in cascade_media:
                    for burst in cascade.bursts:
                        if media in burst.media_involved:
                            times.append(burst.burst_event.start_date.timestamp())
                            break
                
                if len(times) >= 2:
                    # Calculate temporal dispersion
                    time_range = max(times) - min(times)
                    cascade_duration = (cascade.end_date - cascade.start_date).total_seconds()
                    
                    if cascade_duration > 0:
                        # Lower dispersion = higher synchronization
                        sync = 1 - (time_range / cascade_duration)
                        synchronization_scores.append(max(0, sync))
        
        return np.mean(synchronization_scores) if synchronization_scores else 0.0
    
    def _determine_coordination_type(self,
                                     temporal_data: Dict[str, Any]) -> str:
        """Determine type of coordination from temporal patterns."""
        pattern = temporal_data['pattern']
        lag_times = temporal_data['lag_times']
        
        if pattern == 'immediate' or pattern == 'synchronized':
            return 'simultaneous'
        elif pattern == 'staggered':
            # Check if there's a clear order
            if lag_times:
                sorted_media = sorted(lag_times.items(), key=lambda x: x[1])
                lags = [lag for _, lag in sorted_media]
                
                # Check if lags form a sequence
                if all(lags[i] < lags[i+1] for i in range(len(lags)-1)):
                    return 'sequential'
                else:
                    return 'hierarchical'
        
        return 'mixed'
    
    def _build_coordination_network(self,
                                    media_group: List[str],
                                    cascades: List[CompleteCascade]) -> nx.Graph:
        """Build network of coordinated media."""
        G = nx.Graph()
        
        for media in media_group:
            G.add_node(media)
        
        # Add edges based on co-participation
        for cascade in cascades:
            cascade_media = [m for m in media_group if m in cascade.all_media]
            
            for i, m1 in enumerate(cascade_media):
                for m2 in cascade_media[i+1:]:
                    if G.has_edge(m1, m2):
                        G[m1][m2]['weight'] += 1
                    else:
                        G.add_edge(m1, m2, weight=1)
        
        return G
    
    def _calculate_coordination_significance(self,
                                             media_group: List[str],
                                             cascades: List[CompleteCascade],
                                             n_shared: int) -> Tuple[float, float]:
        """Calculate statistical significance of coordination."""
        # Null hypothesis: media participate independently
        individual_probs = []
        
        for media in media_group:
            media_cascades = sum(1 for c in cascades if media in c.all_media)
            prob = media_cascades / len(cascades)
            individual_probs.append(prob)
        
        # Expected co-occurrence under independence
        expected_shared = len(cascades) * np.prod(individual_probs)
        
        # Binomial test
        if expected_shared > 0:
            # Use Poisson approximation for large n
            from scipy.stats import poisson
            random_prob = 1 - poisson.cdf(n_shared - 1, expected_shared)
            significance = 1 - random_prob
        else:
            random_prob = 0.0
            significance = 1.0
        
        return significance, random_prob
    
    def _identify_media_clusters(self,
                                 media_profiles: Dict[str, MediaProfile],
                                 cascades: List[CompleteCascade]) -> List[MediaCluster]:
        """Identify clusters of media with similar behavior."""
        if len(media_profiles) < self.cluster_min_size:
            return []
        
        clusters = []
        
        # Create feature matrix
        feature_matrix = self._create_media_feature_matrix(media_profiles)
        
        if feature_matrix.shape[0] < self.cluster_min_size:
            return []
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)
        
        # Try different clustering approaches
        
        # 1. Behavioral clustering
        behavioral_clusters = self._cluster_by_behavior(features_scaled, media_profiles)
        clusters.extend(behavioral_clusters)
        
        # 2. Temporal clustering
        temporal_clusters = self._cluster_by_temporal_patterns(media_profiles, cascades)
        clusters.extend(temporal_clusters)
        
        # 3. Network clustering
        network_clusters = self._cluster_by_network_position(media_profiles, cascades)
        clusters.extend(network_clusters)
        
        return clusters
    
    def _create_media_feature_matrix(self,
                                     media_profiles: Dict[str, MediaProfile]) -> np.ndarray:
        """Create feature matrix for media clustering."""
        media_list = list(media_profiles.keys())
        features = []
        
        for media in media_list:
            profile = media_profiles[media]
            
            feature_vec = [
                profile.leader_score,
                profile.follower_score,
                profile.amplifier_score,
                profile.innovator_score,
                profile.influence_score,
                profile.authority_score,
                profile.hub_score,
                profile.average_response_time,
                profile.consistency_score,
                profile.topic_diversity,
                profile.journalist_mobility,
                len(profile.active_frames),
                profile.total_articles,
                profile.total_cascades
            ]
            
            features.append(feature_vec)
        
        return np.array(features)
    
    def _cluster_by_behavior(self,
                             features: np.ndarray,
                             media_profiles: Dict[str, MediaProfile]) -> List[MediaCluster]:
        """Cluster media by behavioral patterns."""
        clusters = []
        media_list = list(media_profiles.keys())
        
        # Use KMeans for behavioral clustering
        n_clusters = min(5, len(media_list) // 3)
        if n_clusters < 2:
            return clusters
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features[:, :4])  # Use behavioral scores
        
        # Create cluster objects
        for cluster_id in range(n_clusters):
            cluster_members = [media_list[i] for i, l in enumerate(labels) if l == cluster_id]
            
            if len(cluster_members) >= self.cluster_min_size:
                cluster = self._create_media_cluster(
                    cluster_members,
                    'behavioral',
                    media_profiles,
                    features[labels == cluster_id]
                )
                if cluster:
                    clusters.append(cluster)
        
        return clusters
    
    def _cluster_by_temporal_patterns(self,
                                      media_profiles: Dict[str, MediaProfile],
                                      cascades: List[CompleteCascade]) -> List[MediaCluster]:
        """Cluster media by temporal response patterns."""
        clusters = []
        
        # Create temporal feature matrix
        temporal_features = []
        media_list = []
        
        for media, profile in media_profiles.items():
            temporal_vec = [
                profile.average_response_time,
                profile.consistency_score,
                len(profile.burst_participation)
            ]
            temporal_features.append(temporal_vec)
            media_list.append(media)
        
        if len(media_list) < self.cluster_min_size:
            return clusters
        
        # Use DBSCAN for temporal clustering
        temporal_features = np.array(temporal_features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(temporal_features)
        
        dbscan = DBSCAN(eps=0.5, min_samples=self.cluster_min_size)
        labels = dbscan.fit_predict(features_scaled)
        
        # Create cluster objects
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise
                continue
            
            cluster_members = [media_list[i] for i, l in enumerate(labels) if l == cluster_id]
            
            if len(cluster_members) >= self.cluster_min_size:
                cluster = self._create_media_cluster(
                    cluster_members,
                    'temporal',
                    media_profiles,
                    features_scaled[labels == cluster_id]
                )
                if cluster:
                    clusters.append(cluster)
        
        return clusters
    
    def _cluster_by_network_position(self,
                                     media_profiles: Dict[str, MediaProfile],
                                     cascades: List[CompleteCascade]) -> List[MediaCluster]:
        """Cluster media by network position."""
        clusters = []
        
        # Build media network
        G = nx.Graph()
        for media in media_profiles:
            G.add_node(media)
        
        for cascade in cascades:
            media_list = [m for m in cascade.all_media if m in media_profiles]
            for i, m1 in enumerate(media_list):
                for m2 in media_list[i+1:]:
                    if G.has_edge(m1, m2):
                        G[m1][m2]['weight'] += 1
                    else:
                        G.add_edge(m1, m2, weight=1)
        
        # Use community detection
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(G)
            
            # Create clusters from communities
            community_groups = defaultdict(list)
            for node, comm_id in communities.items():
                community_groups[comm_id].append(node)
            
            for comm_id, members in community_groups.items():
                if len(members) >= self.cluster_min_size:
                    cluster = self._create_media_cluster(
                        members,
                        'network',
                        media_profiles,
                        None
                    )
                    if cluster:
                        clusters.append(cluster)
        except:
            pass
        
        return clusters
    
    def _create_media_cluster(self,
                             members: List[str],
                             cluster_type: str,
                             media_profiles: Dict[str, MediaProfile],
                             features: Optional[np.ndarray]) -> Optional[MediaCluster]:
        """Create media cluster object."""
        try:
            # Calculate cluster metrics
            if features is not None and len(features) > 1:
                # Internal coherence (average pairwise similarity)
                from scipy.spatial.distance import pdist
                distances = pdist(features, metric='euclidean')
                coherence = 1 / (1 + np.mean(distances))
            else:
                coherence = 0.5
            
            # Determine dominant behavior
            behavior_scores = defaultdict(float)
            frame_counts = Counter()
            
            for media in members:
                profile = media_profiles[media]
                behavior_scores['leader'] += profile.leader_score
                behavior_scores['follower'] += profile.follower_score
                behavior_scores['amplifier'] += profile.amplifier_score
                behavior_scores['innovator'] += profile.innovator_score
                
                for frame in profile.active_frames:
                    frame_counts[frame] += 1
            
            # Normalize
            for behavior in behavior_scores:
                behavior_scores[behavior] /= len(members)
            
            dominant_behavior = max(behavior_scores, key=behavior_scores.get)
            
            # Frame preferences
            total_frame_mentions = sum(frame_counts.values())
            frame_preferences = {
                frame: count / total_frame_mentions
                for frame, count in frame_counts.most_common(5)
            } if total_frame_mentions > 0 else {}
            
            # Temporal pattern
            avg_response = np.mean([
                media_profiles[m].average_response_time
                for m in members
            ])
            
            if avg_response < 12:
                temporal_pattern = 'early_adopter'
            elif avg_response < 48:
                temporal_pattern = 'mainstream'
            else:
                temporal_pattern = 'late_adopter'
            
            # Identify core vs peripheral members
            member_scores = [
                (m, media_profiles[m].influence_score)
                for m in members
            ]
            member_scores.sort(key=lambda x: x[1], reverse=True)
            
            core_size = max(self.cluster_min_size, len(members) // 2)
            core_members = [m for m, _ in member_scores[:core_size]]
            peripheral_members = [m for m, _ in member_scores[core_size:]]
            
            return MediaCluster(
                cluster_id=str(uuid.uuid4())[:8],
                cluster_type=cluster_type,
                core_members=core_members,
                peripheral_members=peripheral_members,
                coherence_score=coherence,
                separation_score=0.5,  # Would need other clusters to calculate
                stability_score=0.7,  # Placeholder
                dominant_behavior=dominant_behavior,
                frame_preferences=frame_preferences,
                temporal_pattern=temporal_pattern,
                internal_density=0.5,  # Placeholder
                external_connections=0,  # Placeholder
                bridge_media=[],  # Would need network analysis
                formation_date=None,
                peak_date=None,
                dissolution_date=None
            )
            
        except Exception as e:
            logger.error(f"Failed to create media cluster: {e}")
            return None
    
    def _analyze_journalist_network(self,
                                    cascades: List[CompleteCascade],
                                    media_profiles: Dict[str, MediaProfile]) -> JournalistNetwork:
        """Analyze journalist relationships and mobility."""
        journalist_profiles = {}
        journalist_moves = []
        
        # Build journalist profiles
        all_journalists = set()
        for cascade in cascades:
            all_journalists.update(cascade.all_journalists)
        
        for journalist in all_journalists:
            profile = self._build_journalist_profile(journalist, cascades, media_profiles)
            if profile:
                journalist_profiles[journalist] = profile
        
        # Track journalist moves
        moves = self._track_journalist_moves(journalist_profiles, cascades)
        journalist_moves.extend(moves)
        
        # Calculate mobility rate
        mobility_rate = len(journalist_moves) / len(journalist_profiles) if journalist_profiles else 0.0
        
        # Build influence network
        influence_network = self._build_journalist_influence_network(
            journalist_profiles, cascades
        )
        
        # Identify influential journalists
        influential = []
        if influence_network.number_of_nodes() > 0:
            try:
                pagerank = nx.pagerank(influence_network)
                influential = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:20]
            except:
                pass
        
        # Build collaboration network
        collaboration_network = self._build_journalist_collaboration_network(
            journalist_profiles, cascades
        )
        
        # Find collaboration clusters
        collaboration_clusters = []
        if collaboration_network.number_of_nodes() > 0:
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(collaboration_network)
                
                groups = defaultdict(set)
                for node, comm_id in communities.items():
                    groups[comm_id].add(node)
                
                collaboration_clusters = list(groups.values())
            except:
                pass
        
        # Identify bridge journalists
        bridge_journalists = self._identify_bridge_journalists(
            journalist_profiles, media_profiles, cascades
        )
        
        return JournalistNetwork(
            journalist_profiles=journalist_profiles,
            journalist_moves=journalist_moves,
            mobility_rate=mobility_rate,
            influence_network=influence_network,
            influential_journalists=influential,
            collaboration_network=collaboration_network,
            collaboration_clusters=collaboration_clusters,
            bridge_journalists=bridge_journalists
        )
    
    def _build_journalist_profile(self,
                                  journalist: str,
                                  cascades: List[CompleteCascade],
                                  media_profiles: Dict[str, MediaProfile]) -> Optional[Dict[str, Any]]:
        """Build profile for a journalist."""
        try:
            articles = 0
            media_affiliations = set()
            frames_covered = set()
            entities_covered = set()
            
            for cascade in cascades:
                if journalist in cascade.all_journalists:
                    articles += cascade.journalist_contributions.get(journalist, 0)
                    frames_covered.update(cascade.frames_involved)
                    entities_covered.update(cascade.all_entities)
                    
                    # Find media affiliations
                    for burst in cascade.bursts:
                        affiliations = burst.journalist_media_affiliations.get(journalist, set())
                        media_affiliations.update(affiliations)
            
            if articles == 0:
                return None
            
            return {
                'name': journalist,
                'articles': articles,
                'media_affiliations': list(media_affiliations),
                'frames_covered': list(frames_covered),
                'n_entities': len(entities_covered),
                'authority_score': np.mean([
                    c.journalist_authority_scores.get(journalist, 0)
                    for c in cascades
                    if journalist in c.all_journalists
                ])
            }
            
        except Exception as e:
            logger.error(f"Failed to build journalist profile: {e}")
            return None
    
    def _track_journalist_moves(self,
                                journalist_profiles: Dict[str, Dict],
                                cascades: List[CompleteCascade]) -> List[Dict[str, Any]]:
        """Track journalist movements between media outlets."""
        moves = []
        
        for journalist, profile in journalist_profiles.items():
            affiliations = profile.get('media_affiliations', [])
            
            if len(affiliations) > 1:
                # Journalist has worked for multiple outlets
                # Try to determine chronology from cascades
                affiliation_timeline = []
                
                for cascade in sorted(cascades, key=lambda c: c.start_date):
                    if journalist in cascade.all_journalists:
                        for burst in cascade.bursts:
                            media_set = burst.journalist_media_affiliations.get(journalist, set())
                            for media in media_set:
                                affiliation_timeline.append({
                                    'date': burst.burst_event.start_date,
                                    'media': media
                                })
                
                # Detect changes
                if affiliation_timeline:
                    current_media = None
                    for item in sorted(affiliation_timeline, key=lambda x: x['date']):
                        if item['media'] != current_media:
                            if current_media is not None:
                                moves.append({
                                    'journalist': journalist,
                                    'from_media': current_media,
                                    'to_media': item['media'],
                                    'date': item['date']
                                })
                            current_media = item['media']
        
        return moves
    
    def _build_journalist_influence_network(self,
                                            journalist_profiles: Dict[str, Dict],
                                            cascades: List[CompleteCascade]) -> nx.DiGraph:
        """Build directed influence network for journalists."""
        G = nx.DiGraph()
        
        # Add nodes
        for journalist in journalist_profiles:
            G.add_node(journalist)
        
        # Add edges based on cascade participation patterns
        for cascade in cascades:
            journalists_in_cascade = [
                j for j in cascade.all_journalists
                if j in journalist_profiles
            ]
            
            if len(journalists_in_cascade) > 1:
                # Order by authority score
                sorted_journalists = sorted(
                    journalists_in_cascade,
                    key=lambda j: cascade.journalist_authority_scores.get(j, 0),
                    reverse=True
                )
                
                # Add directed edges from high to low authority
                for i, j1 in enumerate(sorted_journalists[:-1]):
                    for j2 in sorted_journalists[i+1:]:
                        if G.has_edge(j1, j2):
                            G[j1][j2]['weight'] += 1
                        else:
                            G.add_edge(j1, j2, weight=1)
        
        return G
    
    def _build_journalist_collaboration_network(self,
                                                journalist_profiles: Dict[str, Dict],
                                                cascades: List[CompleteCascade]) -> nx.Graph:
        """Build collaboration network for journalists."""
        G = nx.Graph()
        
        # Add nodes
        for journalist in journalist_profiles:
            G.add_node(journalist)
        
        # Add edges based on co-participation
        for cascade in cascades:
            journalists_in_cascade = [
                j for j in cascade.all_journalists
                if j in journalist_profiles
            ]
            
            # Create edges between all pairs
            for i, j1 in enumerate(journalists_in_cascade):
                for j2 in journalists_in_cascade[i+1:]:
                    if G.has_edge(j1, j2):
                        G[j1][j2]['weight'] += 1
                    else:
                        G.add_edge(j1, j2, weight=1)
        
        return G
    
    def _identify_bridge_journalists(self,
                                     journalist_profiles: Dict[str, Dict],
                                     media_profiles: Dict[str, MediaProfile],
                                     cascades: List[CompleteCascade]) -> List[str]:
        """Identify journalists who bridge between media outlets."""
        bridge_journalists = []
        
        for journalist, profile in journalist_profiles.items():
            affiliations = profile.get('media_affiliations', [])
            
            # Bridge journalists work for multiple outlets
            if len(affiliations) > 1:
                # Check if they connect otherwise unconnected media
                media_pairs = []
                for i, m1 in enumerate(affiliations):
                    for m2 in affiliations[i+1:]:
                        if m1 in media_profiles and m2 in media_profiles:
                            # Check if these media rarely collaborate
                            m1_collaborators = set(
                                m for m, _ in media_profiles[m1].frequent_collaborators
                            )
                            if m2 not in m1_collaborators:
                                bridge_journalists.append(journalist)
                                break
        
        return list(set(bridge_journalists))[:20]  # Limit to top 20
    
    def _map_information_flows(self,
                               cascades: List[CompleteCascade],
                               media_profiles: Dict[str, MediaProfile]) -> List[InformationFlow]:
        """Map information flow patterns between media outlets."""
        flows = []
        
        # Analyze pairwise media relationships
        media_list = list(media_profiles.keys())
        
        for i, source_media in enumerate(media_list):
            for target_media in media_list[i+1:]:
                flow = self._analyze_information_flow(
                    source_media, target_media, cascades
                )
                
                if flow and flow.flow_strength >= self.flow_min_strength:
                    flows.append(flow)
        
        return flows
    
    def _analyze_information_flow(self,
                                  source: str,
                                  target: str,
                                  cascades: List[CompleteCascade]) -> Optional[InformationFlow]:
        """Analyze information flow from source to target media."""
        try:
            # Track temporal patterns
            flow_instances = []
            transmitted_entities = set()
            transmitted_frames = set()
            
            for cascade in cascades:
                if source in cascade.all_media and target in cascade.all_media:
                    # Find when each media joined
                    source_time = None
                    target_time = None
                    
                    for burst in cascade.bursts:
                        if source in burst.media_involved and source_time is None:
                            source_time = burst.burst_event.start_date
                        if target in burst.media_involved and target_time is None:
                            target_time = burst.burst_event.start_date
                    
                    if source_time and target_time:
                        lag = (target_time - source_time).total_seconds() / 3600
                        
                        if 0 < lag <= self.lag_window_hours:
                            # Potential flow from source to target
                            flow_instances.append(lag)
                            transmitted_entities.update(cascade.all_entities)
                            transmitted_frames.update(cascade.frames_involved)
            
            if not flow_instances:
                return None
            
            # Calculate flow metrics
            avg_lag = np.mean(flow_instances)
            consistency = 1.0 / (1.0 + np.std(flow_instances) / (avg_lag + 1))
            flow_strength = len(flow_instances) / len(cascades)
            
            # Determine flow direction
            if avg_lag > 0:
                flow_direction = 'unidirectional'
            else:
                flow_direction = 'bidirectional'
            
            # Simple causality approximation
            causality_score = flow_strength * consistency
            
            return InformationFlow(
                flow_id=str(uuid.uuid4())[:8],
                source_media=source,
                target_media=target,
                flow_strength=flow_strength,
                flow_direction=flow_direction,
                flow_type='direct' if avg_lag < 6 else 'cascading',
                transmitted_entities=transmitted_entities,
                transmitted_frames=transmitted_frames,
                average_lag=avg_lag,
                consistency=consistency,
                causality_score=causality_score,
                confidence=min(1.0, flow_strength + consistency) / 2
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze information flow: {e}")
            return None
    
    def _build_media_network(self,
                             media_profiles: Dict[str, MediaProfile],
                             coordinations: List[MediaCoordination],
                             clusters: List[MediaCluster],
                             flows: List[InformationFlow]) -> nx.DiGraph:
        """Build comprehensive media relationship network."""
        G = nx.DiGraph()
        
        # Add media nodes with attributes
        for media, profile in media_profiles.items():
            G.add_node(
                media,
                type='media',
                behavior=profile.get_behavioral_type(),
                influence=profile.influence_score,
                articles=profile.total_articles
            )
        
        # Add edges from information flows
        for flow in flows:
            G.add_edge(
                flow.source_media,
                flow.target_media,
                weight=flow.flow_strength,
                type='flow',
                lag=flow.average_lag
            )
        
        # Add coordination information
        for coord in coordinations:
            for i, m1 in enumerate(coord.media_group):
                for m2 in coord.media_group[i+1:]:
                    if not G.has_edge(m1, m2):
                        G.add_edge(
                            m1, m2,
                            weight=coord.coordination_strength,
                            type='coordination'
                        )
        
        # Add cluster information
        for cluster in clusters:
            all_members = cluster.core_members + cluster.peripheral_members
            for member in all_members:
                if G.has_node(member):
                    G.nodes[member]['cluster'] = cluster.cluster_id
                    G.nodes[member]['cluster_role'] = (
                        'core' if member in cluster.core_members else 'peripheral'
                    )
        
        return G
    
    def _analyze_frame_adoption(self,
                                cascades: List[CompleteCascade],
                                media_profiles: Dict[str, MediaProfile],
                                multi_frame_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how media adopt different frames."""
        frame_adoption = {
            'adoption_curves': {},
            'leader_media': {},
            'follower_media': {},
            'frame_loyalty': {}
        }
        
        # Get unique frames
        frames = set()
        for cascade in cascades:
            frames.update(cascade.frames_involved)
        
        for frame in frames:
            # Track adoption over time
            adoption_timeline = self._track_frame_adoption(frame, cascades, media_profiles)
            frame_adoption['adoption_curves'][frame] = adoption_timeline
            
            # Identify leaders and followers for this frame
            leaders, followers = self._identify_frame_leaders_followers(
                frame, cascades, media_profiles
            )
            frame_adoption['leader_media'][frame] = leaders
            frame_adoption['follower_media'][frame] = followers
        
        # Calculate frame loyalty (tendency to stick with frames)
        for media, profile in media_profiles.items():
            if profile.frame_preferences:
                # Calculate entropy of frame preferences
                probs = list(profile.frame_preferences.values())
                entropy = -sum(p * np.log(p + 1e-10) for p in probs)
                max_entropy = np.log(len(probs))
                
                # Lower entropy = higher loyalty
                loyalty = 1 - (entropy / max_entropy) if max_entropy > 0 else 1.0
                frame_adoption['frame_loyalty'][media] = loyalty
        
        return frame_adoption
    
    def _track_frame_adoption(self,
                              frame: str,
                              cascades: List[CompleteCascade],
                              media_profiles: Dict[str, MediaProfile]) -> List[Dict[str, Any]]:
        """Track adoption of a specific frame over time."""
        adoption_timeline = []
        adopted_media = set()
        
        frame_cascades = [c for c in cascades if frame in c.frames_involved]
        frame_cascades.sort(key=lambda c: c.start_date)
        
        for cascade in frame_cascades:
            new_adopters = []
            
            for media in cascade.all_media:
                if media in media_profiles and media not in adopted_media:
                    new_adopters.append(media)
                    adopted_media.add(media)
            
            if new_adopters:
                adoption_timeline.append({
                    'date': cascade.start_date,
                    'new_adopters': new_adopters,
                    'total_adopted': len(adopted_media),
                    'adoption_rate': len(new_adopters) / len(media_profiles)
                })
        
        return adoption_timeline
    
    def _identify_frame_leaders_followers(self,
                                          frame: str,
                                          cascades: List[CompleteCascade],
                                          media_profiles: Dict[str, MediaProfile]) -> Tuple[List[str], List[str]]:
        """Identify leaders and followers for a specific frame."""
        # Track first adoption times
        adoption_times = {}
        
        for cascade in cascades:
            if frame in cascade.frames_involved:
                for media in cascade.all_media:
                    if media in media_profiles and media not in adoption_times:
                        adoption_times[media] = cascade.start_date
        
        if not adoption_times:
            return [], []
        
        # Sort by adoption time
        sorted_media = sorted(adoption_times.items(), key=lambda x: x[1])
        
        # First 20% are leaders, last 20% are followers
        n_media = len(sorted_media)
        n_leaders = max(1, n_media // 5)
        n_followers = max(1, n_media // 5)
        
        leaders = [m for m, _ in sorted_media[:n_leaders]]
        followers = [m for m, _ in sorted_media[-n_followers:]]
        
        return leaders, followers
    
    def _generate_summary(self,
                         media_profiles: Dict[str, MediaProfile],
                         coordinations: List[MediaCoordination],
                         clusters: List[MediaCluster],
                         flows: List[InformationFlow]) -> Dict[str, Any]:
        """Generate summary of cross-media analysis."""
        # Behavioral distribution
        behavioral_dist = Counter()
        for profile in media_profiles.values():
            behavioral_dist[profile.get_behavioral_type()] += 1
        
        # Coordination summary
        coord_summary = {
            'n_coordinations': len(coordinations),
            'avg_group_size': np.mean([len(c.media_group) for c in coordinations]) if coordinations else 0,
            'significant_coordinations': len([c for c in coordinations if c.is_significant()])
        }
        
        # Cluster summary
        cluster_summary = {
            'n_clusters': len(clusters),
            'cluster_types': Counter([c.cluster_type for c in clusters]),
            'avg_cluster_size': np.mean([
                len(c.core_members) + len(c.peripheral_members)
                for c in clusters
            ]) if clusters else 0
        }
        
        # Flow summary
        flow_summary = {
            'n_flows': len(flows),
            'avg_flow_strength': np.mean([f.flow_strength for f in flows]) if flows else 0,
            'dominant_flow_type': Counter([f.flow_type for f in flows]).most_common(1)[0][0] if flows else None
        }
        
        return {
            'n_media_analyzed': len(media_profiles),
            'behavioral_distribution': dict(behavioral_dist),
            'coordination': coord_summary,
            'clusters': cluster_summary,
            'information_flows': flow_summary,
            'top_influencers': sorted(
                media_profiles.items(),
                key=lambda x: x[1].influence_score,
                reverse=True
            )[:10]
        }
    
    def validate_detection(self, detection_result: Dict[str, Any]) -> bool:
        """
        Validate cross-media detection results.
        
        Args:
            detection_result: Results from detect() method
            
        Returns:
            True if detection is valid, False otherwise
        """
        try:
            # Check required keys
            required_keys = [
                'media_profiles', 'media_coordinations', 'media_clusters',
                'journalist_network', 'information_flows', 'media_network'
            ]
            
            for key in required_keys:
                if key not in detection_result:
                    logger.warning(f"Missing required key in detection result: {key}")
                    return False
            
            # Validate media profiles
            profiles = detection_result['media_profiles']
            if not isinstance(profiles, dict):
                logger.warning("Media profiles should be a dictionary")
                return False
            
            # Check that we have meaningful profiles
            if len(profiles) == 0:
                logger.info("No media profiles found - empty but valid")
                return True
            
            # Validate each profile has required attributes
            for media_id, profile in profiles.items():
                if not isinstance(profile, MediaProfile):
                    logger.warning(f"Invalid profile type for {media_id}")
                    return False
                
                # Check basic validity
                if profile.total_articles < 0:
                    logger.warning(f"Invalid article count for {media_id}")
                    return False
                
                if not (0 <= profile.leader_score <= 1):
                    logger.warning(f"Invalid leader score for {media_id}")
                    return False
            
            # Validate coordinations
            coordinations = detection_result['media_coordinations']
            if not isinstance(coordinations, list):
                logger.warning("Media coordinations should be a list")
                return False
            
            for coord in coordinations:
                if not isinstance(coord, MediaCoordination):
                    logger.warning("Invalid coordination type")
                    return False
                
                if len(coord.media_group) < 2:
                    logger.warning("Coordination must involve at least 2 media")
                    return False
            
            # Validate clusters
            clusters = detection_result['media_clusters']
            if not isinstance(clusters, list):
                logger.warning("Media clusters should be a list")
                return False
            
            for cluster in clusters:
                if not isinstance(cluster, MediaCluster):
                    logger.warning("Invalid cluster type")
                    return False
                
                total_members = len(cluster.core_members) + len(cluster.peripheral_members)
                if total_members < self.cluster_min_size:
                    logger.warning(f"Cluster too small: {total_members} < {self.cluster_min_size}")
                    return False
            
            # Validate information flows
            flows = detection_result['information_flows']
            if not isinstance(flows, list):
                logger.warning("Information flows should be a list")
                return False
            
            for flow in flows:
                if not isinstance(flow, InformationFlow):
                    logger.warning("Invalid flow type")
                    return False
                
                if not (0 <= flow.flow_strength <= 1):
                    logger.warning("Invalid flow strength")
                    return False
            
            # Validate network
            network = detection_result['media_network']
            if not isinstance(network, (nx.Graph, nx.DiGraph)):
                logger.warning("Media network should be a NetworkX graph")
                return False
            
            # Validate journalist network
            journalist_net = detection_result['journalist_network']
            if not isinstance(journalist_net, JournalistNetwork):
                logger.warning("Invalid journalist network type")
                return False
            
            logger.info(f"Validation successful: {len(profiles)} profiles, "
                       f"{len(coordinations)} coordinations, {len(clusters)} clusters")
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def score_detection(self, detection_result: Dict[str, Any]) -> float:
        """
        Score the quality of cross-media detection.
        
        Scoring factors:
        - Coverage: How many media outlets were analyzed
        - Coordination detection: Quality and significance of coordinations
        - Clustering quality: Coherence and separation of clusters
        - Information flow strength: Strength and confidence of flows
        - Network richness: Connectivity and structure of media network
        
        Args:
            detection_result: Results from detect() method
            
        Returns:
            Score between 0 and 1
        """
        try:
            if not self.validate_detection(detection_result):
                return 0.0
            
            scores = []
            weights = []
            
            # 1. Coverage score (20% weight)
            profiles = detection_result['media_profiles']
            if profiles:
                # Assume good coverage if we have 10+ media
                coverage_score = min(1.0, len(profiles) / 10)
                scores.append(coverage_score)
                weights.append(0.2)
                logger.debug(f"Coverage score: {coverage_score:.3f}")
            
            # 2. Coordination quality score (25% weight)
            coordinations = detection_result['media_coordinations']
            if coordinations:
                # Average coordination strength and significance
                coord_strengths = [c.coordination_strength for c in coordinations]
                coord_sigs = [c.significance for c in coordinations if c.is_significant()]
                
                avg_strength = np.mean(coord_strengths) if coord_strengths else 0
                sig_ratio = len(coord_sigs) / len(coordinations) if coordinations else 0
                
                coord_score = (avg_strength + sig_ratio) / 2
                scores.append(coord_score)
                weights.append(0.25)
                logger.debug(f"Coordination score: {coord_score:.3f}")
            else:
                scores.append(0.0)
                weights.append(0.25)
            
            # 3. Clustering quality score (20% weight)
            clusters = detection_result['media_clusters']
            if clusters:
                # Average coherence and stability
                coherences = [c.coherence_score for c in clusters]
                stabilities = [c.stability_score for c in clusters]
                
                avg_coherence = np.mean(coherences) if coherences else 0
                avg_stability = np.mean(stabilities) if stabilities else 0
                
                cluster_score = (avg_coherence + avg_stability) / 2
                scores.append(cluster_score)
                weights.append(0.2)
                logger.debug(f"Cluster score: {cluster_score:.3f}")
            else:
                scores.append(0.0)
                weights.append(0.2)
            
            # 4. Information flow score (20% weight)
            flows = detection_result['information_flows']
            if flows:
                # Average flow strength and confidence
                flow_strengths = [f.flow_strength for f in flows]
                flow_confidences = [f.confidence for f in flows]
                
                avg_flow_strength = np.mean(flow_strengths) if flow_strengths else 0
                avg_confidence = np.mean(flow_confidences) if flow_confidences else 0
                
                flow_score = (avg_flow_strength + avg_confidence) / 2
                scores.append(flow_score)
                weights.append(0.2)
                logger.debug(f"Flow score: {flow_score:.3f}")
            else:
                scores.append(0.0)
                weights.append(0.2)
            
            # 5. Network richness score (15% weight)
            network = detection_result['media_network']
            if network and network.number_of_nodes() > 0:
                # Density and connectivity
                density = nx.density(network)
                
                # Check if network is connected (for undirected version)
                G_undirected = network.to_undirected() if isinstance(network, nx.DiGraph) else network
                is_connected = nx.is_connected(G_undirected)
                connectivity_score = 1.0 if is_connected else 0.5
                
                network_score = (density + connectivity_score) / 2
                scores.append(network_score)
                weights.append(0.15)
                logger.debug(f"Network score: {network_score:.3f}")
            else:
                scores.append(0.0)
                weights.append(0.15)
            
            # Calculate weighted average
            total_score = np.average(scores, weights=weights)
            
            # Apply penalties
            
            # Penalty for too few media
            if len(profiles) < 3:
                total_score *= 0.5
                logger.debug("Applied penalty for too few media")
            
            # Penalty for no significant coordinations
            if coordinations and not any(c.is_significant() for c in coordinations):
                total_score *= 0.8
                logger.debug("Applied penalty for no significant coordinations")
            
            # Bonus for rich results
            if (len(profiles) > 20 and len(coordinations) > 5 and 
                len(clusters) > 3 and len(flows) > 10):
                total_score = min(1.0, total_score * 1.2)
                logger.debug("Applied bonus for rich results")
            
            logger.info(f"Cross-media tracking score: {total_score:.3f}")
            return float(total_score)
            
        except Exception as e:
            logger.error(f"Scoring error: {e}")
            return 0.0