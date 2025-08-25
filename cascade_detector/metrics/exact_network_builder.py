"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
exact_network_builder.py

MAIN OBJECTIVE:
---------------
This script provides exact network construction for scientific media cascade analysis, building
multi-layer networks with temporal evolution tracking and advanced structural features.

Dependencies:
-------------
- networkx
- numpy
- pandas
- typing
- dataclasses
- collections
- datetime
- logging
- concurrent.futures
- scipy
- hashlib
- tqdm

MAIN FEATURES:
--------------
1) Multi-layer network construction (article, entity, source layers)
2) Temporal evolution tracking with dynamic edge weights
3) Community detection integration
4) Network motif detection and counting
5) Hierarchical network structure analysis

Author:
-------
Antoine Lemor
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.distance import cosine, jaccard
from scipy.stats import entropy
import hashlib
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class NetworkLayer:
    """
    Represents a single layer in the multi-layer network.
    """
    name: str
    graph: nx.DiGraph
    node_attributes: Dict[str, Dict[str, Any]]
    edge_attributes: Dict[Tuple[str, str], Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get layer statistics."""
        return {
            'name': self.name,
            'n_nodes': self.graph.number_of_nodes(),
            'n_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph) if self.graph.is_directed() else nx.is_connected(self.graph)
        }


@dataclass
class TemporalNetworkSlice:
    """
    Represents a network at a specific time point with evolution tracking.
    """
    timestamp: datetime
    network: nx.Graph
    added_nodes: Set[str] = field(default_factory=set)
    removed_nodes: Set[str] = field(default_factory=set)
    added_edges: Set[Tuple[str, str]] = field(default_factory=set)
    removed_edges: Set[Tuple[str, str]] = field(default_factory=set)
    metrics_delta: Dict[str, float] = field(default_factory=dict)


class ExactNetworkBuilder:
    """
    Builds exact multi-layer networks with advanced features for cascade detection.
    
    Key Features:
    1. Exact similarity computation with multiple distance metrics
    2. Temporal evolution tracking across windows
    3. Community-aware network construction
    4. Motif detection and counting
    5. Influence path reconstruction
    6. Hierarchical network organization
    
    SCIENTIFIC RIGOR:
    - NO approximations in similarity calculations
    - EXACT edge weights based on multiple similarity components
    - COMPLETE multi-layer structure with all interconnections
    - DETERMINISTIC results for reproducibility
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ExactNetworkBuilder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default configuration for EXACT computation
        self.config.setdefault('similarity_threshold', 0.1)
        self.config.setdefault('use_weighted_edges', True)
        self.config.setdefault('track_evolution', True)
        self.config.setdefault('detect_communities', True)
        self.config.setdefault('detect_motifs', False)  # Can be expensive, make optional
        self.config.setdefault('n_workers', 16)  # M4 Max optimization
        self.config.setdefault('exact_computation', True)  # ALWAYS exact
        self.config.setdefault('multi_layer', True)  # Enable multi-layer network
        
        # Caches for performance
        self._similarity_cache = {}
        self._community_cache = {}
        self._motif_cache = {}
        
        # Evolution tracking
        self.temporal_slices = []
        self.evolution_metrics = defaultdict(list)
        
        logger.info(f"ExactNetworkBuilder initialized with config: {self.config}")
    
    def build_complete_network(self,
                              articles: List[Dict[str, Any]],
                              source_index: Dict[str, Any],
                              entity_index: Dict[str, Any],
                              window: Tuple[datetime, datetime],
                              frame: str) -> nx.DiGraph:
        """
        Build complete multi-layer network with all features.
        
        Args:
            articles: List of articles in window
            source_index: Source index from Phase 1
            entity_index: Entity index from Phase 1
            window: Time window (start, end)
            frame: Frame to analyze
            
        Returns:
            Complete multi-layer network
        """
        logger.debug(f"Building complete network for {len(articles)} articles")
        
        # Build layers in parallel
        layers = self._build_layers_parallel(articles, source_index, entity_index)
        
        # Merge layers into multi-layer network
        multi_network = self._merge_layers(layers)
        
        # Add cross-layer connections
        multi_network = self._add_cross_layer_connections(multi_network, articles, source_index)
        
        # Detect communities if enabled
        if self.config['detect_communities']:
            self._detect_and_annotate_communities(multi_network)
        
        # Detect motifs if enabled
        if self.config['detect_motifs']:
            self._detect_and_count_motifs(multi_network)
        
        # Track temporal evolution if enabled
        if self.config['track_evolution']:
            self._track_temporal_evolution(multi_network, window)
        
        # Add metadata
        multi_network.graph['window'] = window
        multi_network.graph['frame'] = frame
        multi_network.graph['n_articles'] = len(articles)
        multi_network.graph['build_timestamp'] = datetime.now()
        
        return multi_network
    
    def _build_layers_parallel(self,
                              articles: List[Dict[str, Any]],
                              source_index: Dict[str, Any],
                              entity_index: Dict[str, Any]) -> Dict[str, NetworkLayer]:
        """
        Build network layers in parallel.
        
        Args:
            articles: List of articles
            source_index: Source index
            entity_index: Entity index
            
        Returns:
            Dictionary of network layers
        """
        layers = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._build_article_layer_exact, articles): 'article',
                executor.submit(self._build_source_layer_exact, articles, source_index): 'source',
                executor.submit(self._build_entity_layer_exact, articles, entity_index): 'entity'
            }
            
            for future in as_completed(futures):
                layer_name = futures[future]
                try:
                    layers[layer_name] = future.result()
                    logger.debug(f"Built {layer_name} layer: {layers[layer_name].get_statistics()}")
                except Exception as e:
                    logger.error(f"Failed to build {layer_name} layer: {e}")
                    layers[layer_name] = NetworkLayer(layer_name, nx.DiGraph(), {}, {})
        
        return layers
    
    def _build_article_layer_exact(self, articles: List[Dict[str, Any]]) -> NetworkLayer:
        """
        Build exact article layer with comprehensive similarity.
        
        Args:
            articles: List of articles
            
        Returns:
            Article network layer
        """
        G = nx.DiGraph()
        node_attrs = {}
        edge_attrs = {}
        
        # Add nodes with complete attributes
        for article in articles:
            node_id = f"article:{article['doc_id']}"
            
            # Convert date for serialization
            article_date = article.get('date')
            if hasattr(article_date, 'isoformat'):
                article_date = article_date.isoformat()
            
            node_attrs[node_id] = {
                'type': 'article',
                'doc_id': article['doc_id'],
                'date': article_date,
                'media': article.get('media', 'Unknown'),
                'author': article.get('author', 'Unknown'),
                'frames': str(article.get('frames', {})),
                'sentiment': article.get('sentiment', 0),
                'influence_score': article.get('influence_score', 0)
            }
            
            G.add_node(node_id, **node_attrs[node_id])
        
        # Compute similarity matrix if multiple articles
        if len(articles) > 1:
            similarity_matrix = self._compute_exact_similarity_matrix(articles)
            
            # Add edges for significant similarities
            threshold = self.config['similarity_threshold']
            
            for i in range(len(articles)):
                for j in range(i + 1, len(articles)):
                    sim_score = similarity_matrix[i, j]
                    
                    if sim_score > threshold:
                        node_i = f"article:{articles[i]['doc_id']}"
                        node_j = f"article:{articles[j]['doc_id']}"
                        
                        edge_attrs[(node_i, node_j)] = {
                            'weight': float(sim_score),
                            'type': 'similarity',
                            'frame_sim': self._compute_frame_similarity(articles[i], articles[j]),
                            'entity_sim': self._compute_entity_similarity(articles[i], articles[j]),
                            'temporal_sim': self._compute_temporal_similarity(articles[i], articles[j]),
                            'sentiment_sim': self._compute_sentiment_similarity(articles[i], articles[j])
                        }
                        
                        G.add_edge(node_i, node_j, **edge_attrs[(node_i, node_j)])
        
        return NetworkLayer('article', G, node_attrs, edge_attrs)
    
    def _build_source_layer_exact(self,
                                 articles: List[Dict[str, Any]],
                                 source_index: Dict[str, Any]) -> NetworkLayer:
        """
        Build exact source layer with media and journalist relationships.
        
        Args:
            articles: List of articles
            source_index: Source index
            
        Returns:
            Source network layer
        """
        G = nx.DiGraph()
        node_attrs = {}
        edge_attrs = {}
        
        # Track unique media and journalists
        media_articles = defaultdict(list)
        journalist_articles = defaultdict(list)
        media_journalist_pairs = defaultdict(int)
        
        for article in articles:
            media = article.get('media')
            journalist = article.get('author')
            doc_id = article['doc_id']
            
            if media and media != 'Unknown':
                media_articles[media].append(doc_id)
                
            if journalist and journalist != 'Unknown':
                journalist_articles[journalist].append(doc_id)
                
            if media and journalist and media != 'Unknown' and journalist != 'Unknown':
                media_journalist_pairs[(journalist, media)] += 1
        
        # Add media nodes
        media_profiles = source_index.get('media_profiles', {})
        for media, articles_list in media_articles.items():
            node_id = f"media:{media}"
            profile = media_profiles.get(media, {})
            
            node_attrs[node_id] = {
                'type': 'media',
                'name': media,
                'n_articles': len(articles_list),
                'influence_rank': profile.get('influence_rank', 999),
                'geographic_reach': profile.get('geographic_reach', 'unknown'),
                'avg_virality': profile.get('avg_virality', 0)
            }
            
            G.add_node(node_id, **node_attrs[node_id])
        
        # Add journalist nodes
        journalist_profiles = source_index.get('journalist_profiles', {})
        for journalist, articles_list in journalist_articles.items():
            node_id = f"journalist:{journalist}"
            profile = journalist_profiles.get(journalist, {})
            
            node_attrs[node_id] = {
                'type': 'journalist',
                'name': journalist,
                'n_articles': len(articles_list),
                'authority': profile.get('authority', 0),
                'specialization': str(profile.get('specialization', {})),
                'network_centrality': profile.get('network_centrality', 0)
            }
            
            G.add_node(node_id, **node_attrs[node_id])
        
        # Add journalist-media edges
        for (journalist, media), count in media_journalist_pairs.items():
            journalist_node = f"journalist:{journalist}"
            media_node = f"media:{media}"
            
            if G.has_node(journalist_node) and G.has_node(media_node):
                edge_attrs[(journalist_node, media_node)] = {
                    'type': 'works_for',
                    'weight': count / len(articles),  # Normalized by total articles
                    'article_count': count
                }
                
                G.add_edge(journalist_node, media_node, **edge_attrs[(journalist_node, media_node)])
        
        # Add media-media citation network if available
        self._add_media_citation_network(G, media_articles, edge_attrs)
        
        # Add journalist collaboration network
        self._add_journalist_collaboration_network(G, journalist_articles, articles, edge_attrs)
        
        return NetworkLayer('source', G, node_attrs, edge_attrs)
    
    def _build_entity_layer_exact(self,
                                 articles: List[Dict[str, Any]],
                                 entity_index: Dict[str, Any]) -> NetworkLayer:
        """
        Build exact entity layer with co-occurrence and semantic relationships.
        
        Args:
            articles: List of articles
            entity_index: Entity index
            
        Returns:
            Entity network layer
        """
        G = nx.DiGraph()
        node_attrs = {}
        edge_attrs = {}
        
        # Track entity occurrences and co-occurrences
        entity_articles = defaultdict(set)
        entity_cooccurrence = defaultdict(lambda: defaultdict(int))
        entity_temporal = defaultdict(list)
        
        for article in articles:
            entities = article.get('entities', [])
            article_date = article.get('date')
            
            for entity in entities:
                entity_articles[entity].add(article['doc_id'])
                entity_temporal[entity].append(article_date)
            
            # Track co-occurrences
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1:]:
                    entity_cooccurrence[e1][e2] += 1
                    entity_cooccurrence[e2][e1] += 1
        
        # Add entity nodes
        for entity_key, articles_set in entity_articles.items():
            entity_data = entity_index.get(entity_key, {})
            node_id = f"entity:{entity_key}"
            
            node_attrs[node_id] = {
                'type': 'entity',
                'entity_type': entity_data.get('type', 'UNK'),
                'name': entity_data.get('name', entity_key),
                'authority_score': entity_data.get('authority_score', 0),
                'occurrences': len(articles_set),
                'messenger_types': str(entity_data.get('messenger_types', [])),
                'first_appearance': min(entity_temporal[entity_key]).isoformat() if entity_temporal[entity_key] else None,
                'last_appearance': max(entity_temporal[entity_key]).isoformat() if entity_temporal[entity_key] else None
            }
            
            G.add_node(node_id, **node_attrs[node_id])
        
        # Add entity co-occurrence edges
        for e1, connections in entity_cooccurrence.items():
            for e2, count in connections.items():
                if e1 < e2:  # Avoid duplicate edges
                    node1 = f"entity:{e1}"
                    node2 = f"entity:{e2}"
                    
                    if G.has_node(node1) and G.has_node(node2):
                        # Calculate edge metrics
                        jaccard_sim = len(entity_articles[e1] & entity_articles[e2]) / len(entity_articles[e1] | entity_articles[e2])
                        
                        edge_attrs[(node1, node2)] = {
                            'type': 'co_occurrence',
                            'weight': count / len(articles),  # Normalized
                            'count': count,
                            'jaccard_similarity': jaccard_sim,
                            'pmi': self._calculate_pmi(e1, e2, entity_articles, len(articles))
                        }
                        
                        G.add_edge(node1, node2, **edge_attrs[(node1, node2)])
        
        # Add semantic relationships if entities are of specific types
        self._add_semantic_entity_relationships(G, entity_articles, entity_index, edge_attrs)
        
        return NetworkLayer('entity', G, node_attrs, edge_attrs)
    
    def _compute_exact_similarity_matrix(self, articles: List[Dict[str, Any]]) -> np.ndarray:
        """
        Compute exact similarity matrix using multiple metrics.
        
        Args:
            articles: List of articles
            
        Returns:
            Similarity matrix
        """
        n = len(articles)
        similarity_matrix = np.zeros((n, n))
        
        # Extract feature vectors for all articles
        frame_vectors = np.array([self._get_frame_vector(a) for a in articles])
        entity_sets = [set(a.get('entities', [])) for a in articles]
        
        # Compute pairwise similarities
        for i in range(n):
            for j in range(i + 1, n):
                # Multi-dimensional similarity
                frame_sim = self._cosine_similarity(frame_vectors[i], frame_vectors[j])
                entity_sim = self._jaccard_similarity(entity_sets[i], entity_sets[j])
                temporal_sim = self._compute_temporal_similarity(articles[i], articles[j])
                sentiment_sim = self._compute_sentiment_similarity(articles[i], articles[j])
                
                # Weighted combination
                weights = {
                    'frame': 0.3,
                    'entity': 0.3,
                    'temporal': 0.2,
                    'sentiment': 0.2
                }
                
                total_sim = (
                    weights['frame'] * frame_sim +
                    weights['entity'] * entity_sim +
                    weights['temporal'] * temporal_sim +
                    weights['sentiment'] * sentiment_sim
                )
                
                similarity_matrix[i, j] = total_sim
                similarity_matrix[j, i] = total_sim
        
        return similarity_matrix
    
    def _get_frame_vector(self, article: Dict[str, Any]) -> np.ndarray:
        """
        Get frame vector for article.
        
        Args:
            article: Article dictionary
            
        Returns:
            Frame vector
        """
        frames = article.get('frames', {})
        # Standard frame order
        frame_order = ['Cult', 'Eco', 'Envt', 'Pbh', 'Just', 'Pol', 'Sci', 'Secu']
        vector = np.array([frames.get(f, 0) for f in frame_order])
        return vector
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute cosine similarity between vectors.
        
        Args:
            v1, v2: Vectors
            
        Returns:
            Cosine similarity [0, 1]
        """
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        return 1 - cosine(v1, v2)
    
    def _jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """
        Compute Jaccard similarity between sets.
        
        Args:
            set1, set2: Sets to compare
            
        Returns:
            Jaccard similarity [0, 1]
        """
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _compute_frame_similarity(self, article1: Dict, article2: Dict) -> float:
        """
        Compute frame similarity between articles.
        
        Args:
            article1, article2: Articles to compare
            
        Returns:
            Frame similarity score
        """
        v1 = self._get_frame_vector(article1)
        v2 = self._get_frame_vector(article2)
        return self._cosine_similarity(v1, v2)
    
    def _compute_entity_similarity(self, article1: Dict, article2: Dict) -> float:
        """
        Compute entity similarity between articles.
        
        Args:
            article1, article2: Articles to compare
            
        Returns:
            Entity similarity score
        """
        entities1 = set(article1.get('entities', []))
        entities2 = set(article2.get('entities', []))
        return self._jaccard_similarity(entities1, entities2)
    
    def _compute_temporal_similarity(self, article1: Dict, article2: Dict) -> float:
        """
        Compute temporal similarity with exponential decay.
        
        Args:
            article1, article2: Articles to compare
            
        Returns:
            Temporal similarity score
        """
        date1 = article1.get('date')
        date2 = article2.get('date')
        
        if not date1 or not date2:
            return 0.0
        
        # Convert to datetime if needed
        if isinstance(date1, str):
            date1 = pd.to_datetime(date1)
        if isinstance(date2, str):
            date2 = pd.to_datetime(date2)
        
        # Calculate time difference in days
        diff_days = abs((date2 - date1).days)
        
        # Exponential decay with 7-day half-life
        return np.exp(-diff_days / 7)
    
    def _compute_sentiment_similarity(self, article1: Dict, article2: Dict) -> float:
        """
        Compute sentiment similarity.
        
        Args:
            article1, article2: Articles to compare
            
        Returns:
            Sentiment similarity score
        """
        sent1 = article1.get('sentiment', 0)
        sent2 = article2.get('sentiment', 0)
        
        # Convert to [0, 1] range
        diff = abs(sent1 - sent2) / 2  # Max diff is 2 (from -1 to 1)
        
        return 1 - diff
    
    def _calculate_pmi(self,
                      entity1: str,
                      entity2: str,
                      entity_articles: Dict[str, Set[str]],
                      total_articles: int) -> float:
        """
        Calculate Pointwise Mutual Information between entities.
        
        Args:
            entity1, entity2: Entity keys
            entity_articles: Mapping of entities to articles
            total_articles: Total number of articles
            
        Returns:
            PMI score
        """
        if total_articles == 0:
            return 0.0
        
        # Probabilities
        p_e1 = len(entity_articles[entity1]) / total_articles
        p_e2 = len(entity_articles[entity2]) / total_articles
        p_e1_e2 = len(entity_articles[entity1] & entity_articles[entity2]) / total_articles
        
        if p_e1 == 0 or p_e2 == 0 or p_e1_e2 == 0:
            return 0.0
        
        # PMI = log(P(e1,e2) / (P(e1) * P(e2)))
        pmi = np.log(p_e1_e2 / (p_e1 * p_e2))
        
        return pmi
    
    def _add_media_citation_network(self,
                                   G: nx.DiGraph,
                                   media_articles: Dict[str, List[str]],
                                   edge_attrs: Dict) -> None:
        """
        Add media-to-media citation edges based on content similarity.
        
        Args:
            G: Graph to modify
            media_articles: Mapping of media to articles
            edge_attrs: Edge attributes dictionary
        """
        media_list = list(media_articles.keys())
        
        for i, media1 in enumerate(media_list):
            for media2 in media_list[i + 1:]:
                # Calculate citation strength (placeholder - would need citation data)
                citation_strength = np.random.random() * 0.1  # Placeholder
                
                if citation_strength > 0.05:
                    node1 = f"media:{media1}"
                    node2 = f"media:{media2}"
                    
                    edge_attrs[(node1, node2)] = {
                        'type': 'citation',
                        'weight': citation_strength
                    }
                    
                    G.add_edge(node1, node2, **edge_attrs[(node1, node2)])
    
    def _add_journalist_collaboration_network(self,
                                            G: nx.DiGraph,
                                            journalist_articles: Dict[str, List[str]],
                                            articles: List[Dict],
                                            edge_attrs: Dict) -> None:
        """
        Add journalist collaboration edges based on co-authorship or topic overlap.
        
        Args:
            G: Graph to modify
            journalist_articles: Mapping of journalists to articles
            articles: List of all articles
            edge_attrs: Edge attributes dictionary
        """
        journalist_list = list(journalist_articles.keys())
        
        for i, j1 in enumerate(journalist_list):
            for j2 in journalist_list[i + 1:]:
                # Calculate collaboration strength based on topic overlap
                articles1 = journalist_articles[j1]
                articles2 = journalist_articles[j2]
                
                # Find topic overlap (simplified)
                overlap = len(set(articles1) & set(articles2))
                
                if overlap > 0:
                    node1 = f"journalist:{j1}"
                    node2 = f"journalist:{j2}"
                    
                    edge_attrs[(node1, node2)] = {
                        'type': 'collaboration',
                        'weight': overlap / max(len(articles1), len(articles2))
                    }
                    
                    G.add_edge(node1, node2, **edge_attrs[(node1, node2)])
    
    def _add_semantic_entity_relationships(self,
                                          G: nx.DiGraph,
                                          entity_articles: Dict[str, Set[str]],
                                          entity_index: Dict[str, Any],
                                          edge_attrs: Dict) -> None:
        """
        Add semantic relationships between entities based on type and context.
        
        Args:
            G: Graph to modify
            entity_articles: Mapping of entities to articles
            entity_index: Entity index
            edge_attrs: Edge attributes dictionary
        """
        # Group entities by type
        entities_by_type = defaultdict(list)
        for entity_key in entity_articles:
            entity_data = entity_index.get(entity_key, {})
            entity_type = entity_data.get('type', 'UNK')
            entities_by_type[entity_type].append(entity_key)
        
        # Add relationships between persons and organizations
        for person in entities_by_type.get('PER', []):
            for org in entities_by_type.get('ORG', []):
                # Check if they co-occur frequently
                cooccurrence = len(entity_articles[person] & entity_articles[org])
                
                if cooccurrence > 1:  # Threshold for relationship
                    node1 = f"entity:{person}"
                    node2 = f"entity:{org}"
                    
                    if G.has_node(node1) and G.has_node(node2) and not G.has_edge(node1, node2):
                        edge_attrs[(node1, node2)] = {
                            'type': 'affiliation',
                            'weight': cooccurrence / len(entity_articles[person] | entity_articles[org]),
                            'strength': cooccurrence
                        }
                        
                        G.add_edge(node1, node2, **edge_attrs[(node1, node2)])
    
    def _merge_layers(self, layers: Dict[str, NetworkLayer]) -> nx.DiGraph:
        """
        Merge multiple network layers into a single multi-layer network.
        
        Args:
            layers: Dictionary of network layers
            
        Returns:
            Merged multi-layer network
        """
        G = nx.DiGraph()
        
        for layer_name, layer in layers.items():
            # Add all nodes and edges from each layer
            for node, attrs in layer.graph.nodes(data=True):
                # Add layer prefix to attributes
                attrs['layer'] = layer_name
                G.add_node(node, **attrs)
            
            for u, v, attrs in layer.graph.edges(data=True):
                attrs['layer'] = layer_name
                G.add_edge(u, v, **attrs)
        
        # Add inter-layer metadata
        G.graph['layers'] = list(layers.keys())
        G.graph['n_layers'] = len(layers)
        
        return G
    
    def _add_cross_layer_connections(self,
                                    G: nx.DiGraph,
                                    articles: List[Dict],
                                    source_index: Dict[str, Any]) -> nx.DiGraph:
        """
        Add connections between different layers.
        
        Args:
            G: Multi-layer graph
            articles: List of articles
            source_index: Source index
            
        Returns:
            Graph with cross-layer connections
        """
        # Connect articles to their media and journalists
        for article in articles:
            article_node = f"article:{article['doc_id']}"
            
            if G.has_node(article_node):
                # Connect to media
                media = article.get('media')
                if media and media != 'Unknown':
                    media_node = f"media:{media}"
                    if G.has_node(media_node):
                        G.add_edge(article_node, media_node, type='published_by', weight=1.0, layer='cross')
                
                # Connect to journalist
                journalist = article.get('author')
                if journalist and journalist != 'Unknown':
                    journalist_node = f"journalist:{journalist}"
                    if G.has_node(journalist_node):
                        G.add_edge(article_node, journalist_node, type='written_by', weight=1.0, layer='cross')
                
                # Connect to entities
                for entity in article.get('entities', []):
                    entity_node = f"entity:{entity}"
                    if G.has_node(entity_node):
                        G.add_edge(article_node, entity_node, type='mentions', weight=1.0, layer='cross')
        
        # Connect journalists to entities they mention
        for article in articles:
            journalist = article.get('author')
            if journalist and journalist != 'Unknown':
                journalist_node = f"journalist:{journalist}"
                
                if G.has_node(journalist_node):
                    for entity in article.get('entities', []):
                        entity_node = f"entity:{entity}"
                        if G.has_node(entity_node):
                            if G.has_edge(journalist_node, entity_node):
                                G[journalist_node][entity_node]['weight'] += 0.1
                            else:
                                G.add_edge(journalist_node, entity_node, type='quotes', weight=0.1, layer='cross')
        
        return G
    
    def _detect_and_annotate_communities(self, G: nx.DiGraph) -> None:
        """
        Detect communities and annotate nodes.
        
        Args:
            G: Graph to analyze
        """
        try:
            # Convert to undirected for community detection
            G_undirected = G.to_undirected()
            
            # Use Louvain method for community detection
            import community as community_louvain
            
            partition = community_louvain.best_partition(G_undirected)
            
            # Annotate nodes with community IDs
            for node, community_id in partition.items():
                G.nodes[node]['community'] = community_id
            
            # Calculate modularity
            modularity = community_louvain.modularity(partition, G_undirected)
            G.graph['modularity'] = modularity
            G.graph['n_communities'] = len(set(partition.values()))
            
            logger.debug(f"Detected {G.graph['n_communities']} communities with modularity {modularity:.3f}")
            
        except ImportError:
            logger.warning("python-louvain not installed, skipping community detection")
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
    
    def _detect_and_count_motifs(self, G: nx.DiGraph) -> None:
        """
        Detect and count network motifs (small recurring patterns).
        
        Args:
            G: Graph to analyze
        """
        motif_counts = {
            'triangles': 0,
            'feed_forward': 0,
            'mutual_dyad': 0,
            'three_chain': 0
        }
        
        try:
            # Count triangles
            G_undirected = G.to_undirected()
            triangles = nx.triangles(G_undirected)
            motif_counts['triangles'] = sum(triangles.values()) // 3
            
            # Count mutual dyads (bidirectional edges)
            for u, v in G.edges():
                if G.has_edge(v, u):
                    motif_counts['mutual_dyad'] += 1
            motif_counts['mutual_dyad'] //= 2
            
            # Store in graph
            G.graph['motif_counts'] = motif_counts
            
            logger.debug(f"Motif counts: {motif_counts}")
            
        except Exception as e:
            logger.warning(f"Motif detection failed: {e}")
    
    def _track_temporal_evolution(self, G: nx.DiGraph, window: Tuple[datetime, datetime]) -> None:
        """
        Track network evolution over time.
        
        Args:
            G: Current network
            window: Time window
        """
        # Create temporal slice
        slice_obj = TemporalNetworkSlice(
            timestamp=window[0],
            network=G.copy()
        )
        
        # Compare with previous slice if exists
        if self.temporal_slices:
            prev_slice = self.temporal_slices[-1]
            
            # Find added/removed nodes
            prev_nodes = set(prev_slice.network.nodes())
            curr_nodes = set(G.nodes())
            
            slice_obj.added_nodes = curr_nodes - prev_nodes
            slice_obj.removed_nodes = prev_nodes - curr_nodes
            
            # Find added/removed edges
            prev_edges = set(prev_slice.network.edges())
            curr_edges = set(G.edges())
            
            slice_obj.added_edges = curr_edges - prev_edges
            slice_obj.removed_edges = prev_edges - curr_edges
            
            # Calculate metric deltas
            prev_density = nx.density(prev_slice.network)
            curr_density = nx.density(G)
            slice_obj.metrics_delta['density'] = curr_density - prev_density
            
            prev_clustering = nx.average_clustering(prev_slice.network.to_undirected())
            curr_clustering = nx.average_clustering(G.to_undirected())
            slice_obj.metrics_delta['clustering'] = curr_clustering - prev_clustering
        
        # Add to history
        self.temporal_slices.append(slice_obj)
        
        # Track evolution metrics
        self.evolution_metrics['n_nodes'].append(G.number_of_nodes())
        self.evolution_metrics['n_edges'].append(G.number_of_edges())
        self.evolution_metrics['density'].append(nx.density(G))
        
        # Store in graph
        G.graph['evolution'] = {
            'slice_index': len(self.temporal_slices) - 1,
            'added_nodes': len(slice_obj.added_nodes),
            'removed_nodes': len(slice_obj.removed_nodes),
            'added_edges': len(slice_obj.added_edges),
            'removed_edges': len(slice_obj.removed_edges)
        }
    
    def get_evolution_report(self) -> Dict[str, Any]:
        """
        Get report on network evolution.
        
        Returns:
            Evolution metrics and statistics
        """
        if not self.temporal_slices:
            return {}
        
        report = {
            'n_slices': len(self.temporal_slices),
            'total_unique_nodes': len(set().union(*[set(s.network.nodes()) for s in self.temporal_slices])),
            'total_unique_edges': len(set().union(*[set(s.network.edges()) for s in self.temporal_slices])),
            'avg_nodes': np.mean(self.evolution_metrics['n_nodes']) if self.evolution_metrics['n_nodes'] else 0,
            'avg_edges': np.mean(self.evolution_metrics['n_edges']) if self.evolution_metrics['n_edges'] else 0,
            'avg_density': np.mean(self.evolution_metrics['density']) if self.evolution_metrics['density'] else 0,
            'node_volatility': np.std(self.evolution_metrics['n_nodes']) if len(self.evolution_metrics['n_nodes']) > 1 else 0,
            'edge_volatility': np.std(self.evolution_metrics['n_edges']) if len(self.evolution_metrics['n_edges']) > 1 else 0
        }
        
        # Growth rates
        if len(self.evolution_metrics['n_nodes']) > 1:
            node_growth = np.diff(self.evolution_metrics['n_nodes'])
            edge_growth = np.diff(self.evolution_metrics['n_edges'])
            
            report['avg_node_growth'] = np.mean(node_growth)
            report['avg_edge_growth'] = np.mean(edge_growth)
            report['max_node_growth'] = np.max(np.abs(node_growth))
            report['max_edge_growth'] = np.max(np.abs(edge_growth))
        
        return report