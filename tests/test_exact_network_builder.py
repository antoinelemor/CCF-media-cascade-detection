"""
Comprehensive tests for ExactNetworkBuilder.

Tests verify exact network construction with all advanced features.
"""

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
import tempfile
import shutil
import json
from unittest.mock import Mock, patch, MagicMock

# Import the module to test
from cascade_detector.metrics.exact_network_builder import (
    ExactNetworkBuilder, NetworkLayer, TemporalNetworkSlice
)


class TestNetworkLayer(unittest.TestCase):
    """Test NetworkLayer dataclass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph = nx.DiGraph()
        self.graph.add_edges_from([('a', 'b'), ('b', 'c')])
        self.node_attrs = {'a': {'type': 'article'}, 'b': {'type': 'media'}}
        self.edge_attrs = {('a', 'b'): {'weight': 0.5}}
    
    def test_layer_creation(self):
        """Test creating a network layer."""
        layer = NetworkLayer(
            name='test',
            graph=self.graph,
            node_attributes=self.node_attrs,
            edge_attributes=self.edge_attrs
        )
        
        self.assertEqual(layer.name, 'test')
        self.assertEqual(layer.graph.number_of_nodes(), 3)
        self.assertEqual(layer.graph.number_of_edges(), 2)
    
    def test_layer_statistics(self):
        """Test getting layer statistics."""
        layer = NetworkLayer(
            name='test',
            graph=self.graph,
            node_attributes=self.node_attrs,
            edge_attributes=self.edge_attrs
        )
        
        stats = layer.get_statistics()
        
        self.assertEqual(stats['name'], 'test')
        self.assertEqual(stats['n_nodes'], 3)
        self.assertEqual(stats['n_edges'], 2)
        self.assertIn('density', stats)
        self.assertIn('is_connected', stats)


class TestExactNetworkBuilder(unittest.TestCase):
    """Test ExactNetworkBuilder main functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test articles
        self.articles = [
            {
                'doc_id': 'doc1',
                'date': datetime(2021, 1, 1),
                'media': 'Media1',
                'author': 'Author1',
                'frames': {'economic': 0.8, 'political': 0.2},
                'entities': ['entity1', 'entity2'],
                'sentiment': 0.5
            },
            {
                'doc_id': 'doc2',
                'date': datetime(2021, 1, 2),
                'media': 'Media2',
                'author': 'Author2',
                'frames': {'economic': 0.6, 'social': 0.4},
                'entities': ['entity2', 'entity3'],
                'sentiment': -0.3
            },
            {
                'doc_id': 'doc3',
                'date': datetime(2021, 1, 3),
                'media': 'Media1',
                'author': 'Author1',
                'frames': {'political': 0.9, 'social': 0.1},
                'entities': ['entity1', 'entity3', 'entity4'],
                'sentiment': 0.1
            }
        ]
        
        # Create test indices
        self.source_index = {
            'article_profiles': {
                'doc1': self.articles[0],
                'doc2': self.articles[1],
                'doc3': self.articles[2]
            },
            'media_profiles': {
                'Media1': {
                    'influence_rank': 1,
                    'geographic_reach': 'national',
                    'avg_virality': 0.7
                },
                'Media2': {
                    'influence_rank': 2,
                    'geographic_reach': 'regional',
                    'avg_virality': 0.5
                }
            },
            'journalist_profiles': {
                'Author1': {
                    'authority': 0.8,
                    'specialization': {'economic': 0.7},
                    'network_centrality': 0.6
                },
                'Author2': {
                    'authority': 0.6,
                    'specialization': {'social': 0.8},
                    'network_centrality': 0.4
                }
            }
        }
        
        self.entity_index = {
            'entity1': {
                'type': 'PER',
                'name': 'Person 1',
                'authority_score': 0.9,
                'messenger_types': ['Expert']
            },
            'entity2': {
                'type': 'ORG',
                'name': 'Organization 1',
                'authority_score': 0.7,
                'messenger_types': []
            },
            'entity3': {
                'type': 'LOC',
                'name': 'Location 1',
                'authority_score': 0.5,
                'messenger_types': []
            },
            'entity4': {
                'type': 'PER',
                'name': 'Person 2',
                'authority_score': 0.6,
                'messenger_types': ['Politician']
            }
        }
        
        # Default config
        self.config = {
            'similarity_threshold': 0.1,
            'use_weighted_edges': True,
            'track_evolution': False,
            'detect_communities': False,
            'detect_motifs': False
        }
    
    def test_initialization(self):
        """Test ExactNetworkBuilder initialization."""
        builder = ExactNetworkBuilder(self.config)
        
        self.assertEqual(builder.config['similarity_threshold'], 0.1)
        self.assertTrue(builder.config['use_weighted_edges'])
        self.assertFalse(builder.config['track_evolution'])
    
    def test_build_complete_network(self):
        """Test building complete multi-layer network."""
        builder = ExactNetworkBuilder(self.config)
        
        window = (datetime(2021, 1, 1), datetime(2021, 1, 3))
        frame = 'economic'
        
        network = builder.build_complete_network(
            articles=self.articles,
            source_index=self.source_index,
            entity_index=self.entity_index,
            window=window,
            frame=frame
        )
        
        # Check network properties
        self.assertIsInstance(network, nx.DiGraph)
        self.assertGreater(network.number_of_nodes(), 0)
        
        # Check metadata
        self.assertEqual(network.graph['frame'], frame)
        self.assertEqual(network.graph['n_articles'], 3)
        self.assertIn('layers', network.graph)
    
    def test_build_article_layer(self):
        """Test building article layer."""
        builder = ExactNetworkBuilder(self.config)
        
        layer = builder._build_article_layer_exact(self.articles)
        
        # Check layer structure
        self.assertIsInstance(layer, NetworkLayer)
        self.assertEqual(layer.name, 'article')
        self.assertEqual(layer.graph.number_of_nodes(), 3)
        
        # Check nodes
        self.assertTrue(layer.graph.has_node('article:doc1'))
        self.assertTrue(layer.graph.has_node('article:doc2'))
        self.assertTrue(layer.graph.has_node('article:doc3'))
        
        # Check edges (should have similarity edges)
        self.assertGreater(layer.graph.number_of_edges(), 0)
    
    def test_build_source_layer(self):
        """Test building source layer."""
        builder = ExactNetworkBuilder(self.config)
        
        layer = builder._build_source_layer_exact(self.articles, self.source_index)
        
        # Check layer structure
        self.assertIsInstance(layer, NetworkLayer)
        self.assertEqual(layer.name, 'source')
        
        # Check media nodes
        self.assertTrue(layer.graph.has_node('media:Media1'))
        self.assertTrue(layer.graph.has_node('media:Media2'))
        
        # Check journalist nodes
        self.assertTrue(layer.graph.has_node('journalist:Author1'))
        self.assertTrue(layer.graph.has_node('journalist:Author2'))
        
        # Check edges
        self.assertGreater(layer.graph.number_of_edges(), 0)
    
    def test_build_entity_layer(self):
        """Test building entity layer."""
        builder = ExactNetworkBuilder(self.config)
        
        layer = builder._build_entity_layer_exact(self.articles, self.entity_index)
        
        # Check layer structure
        self.assertIsInstance(layer, NetworkLayer)
        self.assertEqual(layer.name, 'entity')
        
        # Check entity nodes
        self.assertTrue(layer.graph.has_node('entity:entity1'))
        self.assertTrue(layer.graph.has_node('entity:entity2'))
        self.assertTrue(layer.graph.has_node('entity:entity3'))
        self.assertTrue(layer.graph.has_node('entity:entity4'))
        
        # Check co-occurrence edges
        edges = list(layer.graph.edges(data=True))
        self.assertGreater(len(edges), 0)
        
        # Check edge attributes
        if edges:
            edge_data = edges[0][2]
            self.assertIn('type', edge_data)
            self.assertIn('weight', edge_data)
    
    def test_similarity_computations(self):
        """Test various similarity computation methods."""
        builder = ExactNetworkBuilder(self.config)
        
        article1 = self.articles[0]
        article2 = self.articles[1]
        
        # Test frame similarity
        frame_sim = builder._compute_frame_similarity(article1, article2)
        self.assertGreaterEqual(frame_sim, 0)
        self.assertLessEqual(frame_sim, 1)
        
        # Test entity similarity
        entity_sim = builder._compute_entity_similarity(article1, article2)
        self.assertGreaterEqual(entity_sim, 0)
        self.assertLessEqual(entity_sim, 1)
        # Should be > 0 since both share entity2
        self.assertGreater(entity_sim, 0)
        
        # Test temporal similarity
        temporal_sim = builder._compute_temporal_similarity(article1, article2)
        self.assertGreaterEqual(temporal_sim, 0)
        self.assertLessEqual(temporal_sim, 1)
        
        # Test sentiment similarity
        sentiment_sim = builder._compute_sentiment_similarity(article1, article2)
        self.assertGreaterEqual(sentiment_sim, 0)
        self.assertLessEqual(sentiment_sim, 1)
    
    def test_exact_similarity_matrix(self):
        """Test exact similarity matrix computation."""
        builder = ExactNetworkBuilder(self.config)
        
        sim_matrix = builder._compute_exact_similarity_matrix(self.articles)
        
        # Check shape
        n = len(self.articles)
        self.assertEqual(sim_matrix.shape, (n, n))
        
        # Check symmetry
        np.testing.assert_array_almost_equal(sim_matrix, sim_matrix.T)
        
        # Check diagonal (self-similarity should be 0 as initialized)
        np.testing.assert_array_equal(np.diag(sim_matrix), np.zeros(n))
        
        # Check range
        self.assertTrue(np.all(sim_matrix >= 0))
        self.assertTrue(np.all(sim_matrix <= 1))
    
    def test_pmi_calculation(self):
        """Test Pointwise Mutual Information calculation."""
        builder = ExactNetworkBuilder(self.config)
        
        entity_articles = {
            'entity1': {'doc1', 'doc3'},
            'entity2': {'doc1', 'doc2'},
            'entity3': {'doc2', 'doc3'}
        }
        
        pmi = builder._calculate_pmi('entity1', 'entity3', entity_articles, 3)
        
        # PMI should be a real number
        self.assertIsInstance(pmi, float)
        
        # Test edge case - no co-occurrence
        entity_articles['entity5'] = {'doc4'}
        pmi_zero = builder._calculate_pmi('entity1', 'entity5', entity_articles, 4)
        self.assertEqual(pmi_zero, 0.0)
    
    def test_layer_merging(self):
        """Test merging multiple layers."""
        builder = ExactNetworkBuilder(self.config)
        
        # Create test layers
        layers = {
            'article': builder._build_article_layer_exact(self.articles),
            'source': builder._build_source_layer_exact(self.articles, self.source_index),
            'entity': builder._build_entity_layer_exact(self.articles, self.entity_index)
        }
        
        # Merge layers
        merged = builder._merge_layers(layers)
        
        # Check that all nodes are present
        article_nodes = sum(1 for n in merged.nodes() if n.startswith('article:'))
        media_nodes = sum(1 for n in merged.nodes() if n.startswith('media:'))
        journalist_nodes = sum(1 for n in merged.nodes() if n.startswith('journalist:'))
        entity_nodes = sum(1 for n in merged.nodes() if n.startswith('entity:'))
        
        self.assertEqual(article_nodes, 3)
        self.assertEqual(media_nodes, 2)
        self.assertEqual(journalist_nodes, 2)
        self.assertEqual(entity_nodes, 4)
        
        # Check metadata
        self.assertIn('layers', merged.graph)
        self.assertEqual(merged.graph['n_layers'], 3)
    
    def test_cross_layer_connections(self):
        """Test adding cross-layer connections."""
        builder = ExactNetworkBuilder(self.config)
        
        # Build complete network
        network = builder.build_complete_network(
            articles=self.articles,
            source_index=self.source_index,
            entity_index=self.entity_index,
            window=(datetime(2021, 1, 1), datetime(2021, 1, 3)),
            frame='economic'
        )
        
        # Check for cross-layer edges
        cross_layer_edges = [
            (u, v, d) for u, v, d in network.edges(data=True)
            if d.get('layer') == 'cross'
        ]
        
        self.assertGreater(len(cross_layer_edges), 0)
        
        # Check specific connections
        # Articles should connect to their media
        self.assertTrue(
            network.has_edge('article:doc1', 'media:Media1') or
            network.has_edge('media:Media1', 'article:doc1')
        )
    
    def test_temporal_evolution_tracking(self):
        """Test temporal evolution tracking."""
        config = self.config.copy()
        config['track_evolution'] = True
        
        builder = ExactNetworkBuilder(config)
        
        # Build networks for two windows
        window1 = (datetime(2021, 1, 1), datetime(2021, 1, 2))
        network1 = builder.build_complete_network(
            articles=self.articles[:2],
            source_index=self.source_index,
            entity_index=self.entity_index,
            window=window1,
            frame='economic'
        )
        
        window2 = (datetime(2021, 1, 2), datetime(2021, 1, 3))
        network2 = builder.build_complete_network(
            articles=self.articles[1:],
            source_index=self.source_index,
            entity_index=self.entity_index,
            window=window2,
            frame='economic'
        )
        
        # Check evolution tracking
        self.assertEqual(len(builder.temporal_slices), 2)
        
        # Check evolution report
        report = builder.get_evolution_report()
        self.assertIn('n_slices', report)
        self.assertEqual(report['n_slices'], 2)
        self.assertIn('avg_nodes', report)
        self.assertIn('avg_edges', report)
    
    def test_community_detection(self):
        """Test community detection when enabled."""
        config = self.config.copy()
        config['detect_communities'] = True
        
        builder = ExactNetworkBuilder(config)
        
        # Build network
        network = builder.build_complete_network(
            articles=self.articles,
            source_index=self.source_index,
            entity_index=self.entity_index,
            window=(datetime(2021, 1, 1), datetime(2021, 1, 3)),
            frame='economic'
        )
        
        # Check if community detection was attempted
        # (May fail if python-louvain not installed)
        if 'modularity' in network.graph:
            self.assertIsInstance(network.graph['modularity'], float)
            self.assertIn('n_communities', network.graph)
    
    def test_motif_detection(self):
        """Test motif detection when enabled."""
        config = self.config.copy()
        config['detect_motifs'] = True
        
        builder = ExactNetworkBuilder(config)
        
        # Build network
        network = builder.build_complete_network(
            articles=self.articles,
            source_index=self.source_index,
            entity_index=self.entity_index,
            window=(datetime(2021, 1, 1), datetime(2021, 1, 3)),
            frame='economic'
        )
        
        # Check if motif detection was performed
        if 'motif_counts' in network.graph:
            motifs = network.graph['motif_counts']
            self.assertIn('triangles', motifs)
            self.assertIn('mutual_dyad', motifs)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        builder = ExactNetworkBuilder(self.config)
        
        # Test with empty articles
        network = builder.build_complete_network(
            articles=[],
            source_index=self.source_index,
            entity_index=self.entity_index,
            window=(datetime(2021, 1, 1), datetime(2021, 1, 2)),
            frame='economic'
        )
        
        self.assertEqual(network.number_of_nodes(), 0)
        
        # Test with single article
        network = builder.build_complete_network(
            articles=[self.articles[0]],
            source_index=self.source_index,
            entity_index=self.entity_index,
            window=(datetime(2021, 1, 1), datetime(2021, 1, 2)),
            frame='economic'
        )
        
        self.assertGreater(network.number_of_nodes(), 0)
        
        # Test with missing data
        incomplete_article = {
            'doc_id': 'doc_incomplete',
            'date': datetime(2021, 1, 1)
            # Missing other fields
        }
        
        network = builder.build_complete_network(
            articles=[incomplete_article],
            source_index={},
            entity_index={},
            window=(datetime(2021, 1, 1), datetime(2021, 1, 2)),
            frame='economic'
        )
        
        # Should handle gracefully
        self.assertIsInstance(network, nx.DiGraph)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Generate larger test dataset
        self.articles = []
        for i in range(20):
            date = datetime(2021, 1, 1) + timedelta(days=i % 7)
            self.articles.append({
                'doc_id': f'doc{i}',
                'date': date,
                'media': f'Media{i % 3}',
                'author': f'Author{i % 5}',
                'frames': {
                    'economic': np.random.random(),
                    'political': np.random.random(),
                    'social': np.random.random()
                },
                'entities': [f'entity{j}' for j in range(i % 4, i % 4 + 3)],
                'sentiment': np.random.uniform(-1, 1),
                'influence_score': np.random.random()
            })
        
        self.source_index = {
            'article_profiles': {f'doc{i}': self.articles[i] for i in range(20)},
            'media_profiles': {
                f'Media{i}': {
                    'influence_rank': i + 1,
                    'geographic_reach': ['local', 'regional', 'national'][i],
                    'avg_virality': np.random.random()
                }
                for i in range(3)
            },
            'journalist_profiles': {
                f'Author{i}': {
                    'authority': np.random.random(),
                    'specialization': {'economic': np.random.random()},
                    'network_centrality': np.random.random()
                }
                for i in range(5)
            }
        }
        
        self.entity_index = {
            f'entity{i}': {
                'type': ['PER', 'ORG', 'LOC'][i % 3],
                'name': f'Entity {i}',
                'authority_score': np.random.random(),
                'messenger_types': []
            }
            for i in range(10)
        }
    
    def test_full_pipeline_performance(self):
        """Test complete pipeline with performance metrics."""
        import time
        
        config = {
            'similarity_threshold': 0.1,
            'use_weighted_edges': True,
            'track_evolution': True,
            'detect_communities': False,  # Skip if dependency missing
            'detect_motifs': True
        }
        
        builder = ExactNetworkBuilder(config)
        
        # Build multiple networks
        windows = [
            (datetime(2021, 1, 1), datetime(2021, 1, 3)),
            (datetime(2021, 1, 3), datetime(2021, 1, 5)),
            (datetime(2021, 1, 5), datetime(2021, 1, 7))
        ]
        
        networks = []
        start_time = time.time()
        
        for window in windows:
            network = builder.build_complete_network(
                articles=[a for a in self.articles if window[0] <= a['date'] <= window[1]],
                source_index=self.source_index,
                entity_index=self.entity_index,
                window=window,
                frame='economic'
            )
            networks.append(network)
        
        elapsed_time = time.time() - start_time
        
        # Verify networks were built
        self.assertEqual(len(networks), 3)
        
        for network in networks:
            self.assertGreater(network.number_of_nodes(), 0)
            self.assertIn('window', network.graph)
            self.assertIn('frame', network.graph)
        
        # Check evolution tracking
        evolution_report = builder.get_evolution_report()
        self.assertEqual(evolution_report['n_slices'], 3)
        
        # Performance check
        self.assertLess(elapsed_time, 5.0)  # Should complete in < 5 seconds
        
        print(f"Built {len(networks)} networks in {elapsed_time:.2f} seconds")
        print(f"Evolution report: {evolution_report}")


if __name__ == '__main__':
    unittest.main(verbosity=2)