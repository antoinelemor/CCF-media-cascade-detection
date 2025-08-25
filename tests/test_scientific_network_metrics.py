"""
Comprehensive tests for ScientificNetworkMetrics.

These tests verify the EXACT computation functionality with NO approximations.
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
import warnings

# Import the module to test
from cascade_detector.metrics.scientific_network_metrics import (
    ScientificNetworkMetrics, NetworkSnapshot, ComputationStats
)


class TestNetworkSnapshot(unittest.TestCase):
    """Test NetworkSnapshot dataclass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.window = (datetime(2021, 1, 1), datetime(2021, 1, 7))
        self.frame = "economic"
        self.network = nx.Graph()
        self.network.add_edges_from([(1, 2), (2, 3), (3, 4)])
    
    def test_snapshot_creation(self):
        """Test creating a network snapshot."""
        snapshot = NetworkSnapshot(
            window=self.window,
            frame=self.frame,
            network=self.network
        )
        
        self.assertEqual(snapshot.window, self.window)
        self.assertEqual(snapshot.frame, self.frame)
        self.assertEqual(snapshot.network.number_of_nodes(), 4)
        self.assertEqual(snapshot.network.number_of_edges(), 3)
    
    def test_snapshot_summary(self):
        """Test getting snapshot summary."""
        snapshot = NetworkSnapshot(
            window=self.window,
            frame=self.frame,
            network=self.network,
            computation_time=5.5
        )
        
        summary = snapshot.get_summary()
        
        self.assertEqual(summary['frame'], 'economic')
        self.assertEqual(summary['n_nodes'], 4)
        self.assertEqual(summary['n_edges'], 3)
        self.assertAlmostEqual(summary['density'], 0.5, places=2)
        self.assertEqual(summary['computation_time'], '5.50s')
    
    def test_snapshot_hash(self):
        """Test snapshot hash generation."""
        snapshot = NetworkSnapshot(
            window=self.window,
            frame=self.frame,
            network=self.network
        )
        
        hash1 = snapshot.get_hash()
        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 32)  # MD5 hash length
        
        # Same parameters should give same hash
        snapshot2 = NetworkSnapshot(
            window=self.window,
            frame=self.frame,
            network=nx.Graph()
        )
        hash2 = snapshot2.get_hash()
        self.assertEqual(hash1, hash2)


class TestComputationStats(unittest.TestCase):
    """Test ComputationStats tracking."""
    
    def test_stats_update(self):
        """Test updating computation statistics."""
        stats = ComputationStats(total_windows=10)
        
        # Update with some computations
        stats.update(2.5)
        stats.update(3.0)
        stats.update(2.0)
        
        self.assertEqual(stats.completed_windows, 3)
        self.assertAlmostEqual(stats.total_time, 7.5, places=1)
        self.assertAlmostEqual(stats.avg_time_per_window, 2.5, places=1)
    
    def test_stats_report(self):
        """Test generating statistics report."""
        stats = ComputationStats(
            total_windows=100,
            completed_windows=80,
            failed_windows=5,
            total_time=240,
            cache_hits=60,
            cache_misses=40,
            peak_memory_gb=8.5
        )
        stats.avg_time_per_window = 3.0
        
        report = stats.get_report()
        
        self.assertEqual(report['progress'], '80/100')
        self.assertEqual(report['success_rate'], '94.1%')
        self.assertEqual(report['avg_time_per_window'], '3.00s')
        self.assertEqual(report['cache_hit_rate'], '60.0%')
        self.assertEqual(report['peak_memory'], '8.50 GB')


class TestScientificNetworkMetrics(unittest.TestCase):
    """Test ScientificNetworkMetrics main functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock indices
        self.source_index = {
            'article_profiles': {
                'doc1': {
                    'doc_id': 'doc1',
                    'date': datetime(2021, 1, 1),
                    'media': 'Media1',
                    'author': 'Author1',
                    'frames': {'economic': 0.8, 'political': 0.2},
                    'entities': ['entity1', 'entity2'],
                    'sentiment': 0.5,
                    'influence_score': 0.7
                },
                'doc2': {
                    'doc_id': 'doc2',
                    'date': datetime(2021, 1, 2),
                    'media': 'Media2',
                    'author': 'Author2',
                    'frames': {'economic': 0.6, 'social': 0.4},
                    'entities': ['entity2', 'entity3'],
                    'sentiment': -0.3,
                    'influence_score': 0.5
                },
                'doc3': {
                    'doc_id': 'doc3',
                    'date': datetime(2021, 1, 3),
                    'media': 'Media1',
                    'author': 'Author1',
                    'frames': {'political': 0.9},
                    'entities': ['entity1', 'entity3', 'entity4'],
                    'sentiment': 0.1,
                    'influence_score': 0.8
                }
            },
            'media_profiles': {
                'Media1': {'influence_rank': 1, 'geographic_reach': 'national'},
                'Media2': {'influence_rank': 2, 'geographic_reach': 'regional'}
            },
            'journalist_profiles': {
                'Author1': {'authority': 0.8, 'specialization': {'economic': 0.7}},
                'Author2': {'authority': 0.6, 'specialization': {'social': 0.8}}
            }
        }
        
        self.entity_index = {
            'entity1': {'type': 'PER', 'name': 'Person 1', 'authority_score': 0.9},
            'entity2': {'type': 'ORG', 'name': 'Organization 1', 'authority_score': 0.7},
            'entity3': {'type': 'LOC', 'name': 'Location 1', 'authority_score': 0.5},
            'entity4': {'type': 'PER', 'name': 'Person 2', 'authority_score': 0.6}
        }
        
        # Create temp directory for output
        self.temp_dir = tempfile.mkdtemp()
        
        # Configuration
        self.config = {
            'output_dir': self.temp_dir,
            'parallel': False,  # Disable for testing
            'n_workers': 1,
            'checkpoint_frequency': 2,
            'gpu_enabled': False
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ScientificNetworkMetrics initialization."""
        metrics = ScientificNetworkMetrics(
            source_index=self.source_index,
            entity_index=self.entity_index,
            config=self.config
        )
        
        # Check configuration
        self.assertTrue(metrics.config['exact_computation'])
        self.assertTrue(metrics.config['no_approximation'])
        self.assertTrue(metrics.config['full_metrics'])
        
        # Check output directory creation
        output_dir = Path(self.temp_dir)
        self.assertTrue(output_dir.exists())
        self.assertTrue((output_dir / 'networks').exists())
        self.assertTrue((output_dir / 'metrics').exists())
        self.assertTrue((output_dir / 'logs').exists())
        self.assertTrue((output_dir / 'checkpoints').exists())
    
    def test_get_window_articles(self):
        """Test retrieving articles for a window."""
        metrics = ScientificNetworkMetrics(
            source_index=self.source_index,
            entity_index=self.entity_index,
            config=self.config
        )
        
        window = (datetime(2021, 1, 1), datetime(2021, 1, 2))
        articles = metrics._get_window_articles(window, 'economic')
        
        # Should get 2 articles with economic frame
        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[0]['doc_id'], 'doc1')
        self.assertEqual(articles[1]['doc_id'], 'doc2')
        
        # Test with different frame
        articles_political = metrics._get_window_articles(window, 'political')
        self.assertEqual(len(articles_political), 1)  # Only doc1 has political > 0.1
    
    def test_build_article_layer(self):
        """Test building article layer network."""
        metrics = ScientificNetworkMetrics(
            source_index=self.source_index,
            entity_index=self.entity_index,
            config=self.config
        )
        
        articles = [
            self.source_index['article_profiles']['doc1'],
            self.source_index['article_profiles']['doc2']
        ]
        window = (datetime(2021, 1, 1), datetime(2021, 1, 2))
        
        G = metrics._build_article_layer(articles, window)
        
        # Check nodes
        self.assertEqual(G.number_of_nodes(), 2)
        self.assertTrue(G.has_node('article:doc1'))
        self.assertTrue(G.has_node('article:doc2'))
        
        # Check node attributes
        node_data = G.nodes['article:doc1']
        self.assertEqual(node_data['type'], 'article')
        self.assertEqual(node_data['media'], 'Media1')
        self.assertEqual(node_data['sentiment'], 0.5)
        
        # Check edges (similarity-based)
        if G.number_of_edges() > 0:
            edges = list(G.edges(data=True))
            edge_data = edges[0][2]
            self.assertIn('weight', edge_data)
            # Check individual similarity components (not nested dict for GraphML compatibility)
            self.assertIn('frame_similarity', edge_data)
            self.assertIn('entity_similarity', edge_data)
            self.assertIn('temporal_similarity', edge_data)
            self.assertIn('sentiment_similarity', edge_data)
    
    def test_build_source_layer(self):
        """Test building source layer network."""
        metrics = ScientificNetworkMetrics(
            source_index=self.source_index,
            entity_index=self.entity_index,
            config=self.config
        )
        
        articles = [
            self.source_index['article_profiles']['doc1'],
            self.source_index['article_profiles']['doc2']
        ]
        
        G = metrics._build_source_layer(articles)
        
        # Check media nodes
        self.assertTrue(G.has_node('media:Media1'))
        self.assertTrue(G.has_node('media:Media2'))
        
        # Check journalist nodes
        self.assertTrue(G.has_node('journalist:Author1'))
        self.assertTrue(G.has_node('journalist:Author2'))
        
        # Check edges
        self.assertTrue(G.has_edge('article:doc1', 'media:Media1'))
        self.assertTrue(G.has_edge('article:doc1', 'journalist:Author1'))
        self.assertTrue(G.has_edge('journalist:Author1', 'media:Media1'))
    
    def test_build_entity_layer(self):
        """Test building entity layer network."""
        metrics = ScientificNetworkMetrics(
            source_index=self.source_index,
            entity_index=self.entity_index,
            config=self.config
        )
        
        articles = [
            self.source_index['article_profiles']['doc1'],
            self.source_index['article_profiles']['doc2']
        ]
        
        G = metrics._build_entity_layer(articles)
        
        # Check entity nodes
        self.assertTrue(G.has_node('entity:entity1'))
        self.assertTrue(G.has_node('entity:entity2'))
        self.assertTrue(G.has_node('entity:entity3'))
        
        # Check article-entity edges
        self.assertTrue(G.has_edge('article:doc1', 'entity:entity1'))
        self.assertTrue(G.has_edge('article:doc1', 'entity:entity2'))
        
        # Check entity co-occurrence edges
        # entity2 appears in both doc1 and doc2 with entity1 and entity3
        edges_data = {(u, v): d for u, v, d in G.edges(data=True)}
        co_occurrence_edges = [(u, v) for (u, v), d in edges_data.items() 
                               if d.get('type') == 'co_occurrence']
        self.assertGreater(len(co_occurrence_edges), 0)
    
    def test_compute_window_network(self):
        """Test computing complete network for a window."""
        metrics = ScientificNetworkMetrics(
            source_index=self.source_index,
            entity_index=self.entity_index,
            config=self.config
        )
        
        window = (datetime(2021, 1, 1), datetime(2021, 1, 3))
        frame = 'economic'
        
        snapshot = metrics.compute_window_network(window, frame)
        
        # Check snapshot properties
        self.assertEqual(snapshot.window, window)
        self.assertEqual(snapshot.frame, frame)
        self.assertIsInstance(snapshot.network, nx.DiGraph)
        
        # Check metadata
        self.assertIn('n_nodes', snapshot.metadata)
        self.assertIn('n_edges', snapshot.metadata)
        self.assertIn('density', snapshot.metadata)
        
        # Check that network has nodes
        self.assertGreater(snapshot.network.number_of_nodes(), 0)
        
        # Check computation time
        self.assertGreater(snapshot.computation_time, 0)
    
    def test_similarity_functions(self):
        """Test similarity computation functions."""
        metrics = ScientificNetworkMetrics(
            source_index=self.source_index,
            entity_index=self.entity_index,
            config=self.config
        )
        
        article1 = self.source_index['article_profiles']['doc1']
        article2 = self.source_index['article_profiles']['doc2']
        
        # Test frame similarity
        frame_sim = metrics._frame_similarity(article1, article2)
        self.assertGreaterEqual(frame_sim, 0)
        self.assertLessEqual(frame_sim, 1)
        
        # Test entity similarity (Jaccard)
        entity_sim = metrics._entity_similarity(article1, article2)
        # Both share entity2, so similarity > 0
        self.assertGreater(entity_sim, 0)
        self.assertLessEqual(entity_sim, 1)
        
        # Test temporal similarity
        temporal_sim = metrics._temporal_similarity(article1, article2, None)
        self.assertGreater(temporal_sim, 0)
        self.assertLessEqual(temporal_sim, 1)
        
        # Test sentiment similarity
        sentiment_sim = metrics._sentiment_similarity(article1, article2)
        self.assertGreaterEqual(sentiment_sim, 0)
        self.assertLessEqual(sentiment_sim, 1)
        
        # Test overall similarity
        overall_sim = metrics._compute_article_similarity(article1, article2)
        self.assertGreaterEqual(overall_sim, 0)
        self.assertLessEqual(overall_sim, 1)
    
    def test_metric_computation(self):
        """Test metric computation functions."""
        metrics = ScientificNetworkMetrics(
            source_index=self.source_index,
            entity_index=self.entity_index,
            config=self.config
        )
        
        # Create a simple test graph
        G = nx.karate_club_graph()
        
        # Get all metric functions
        metric_funcs = metrics._get_all_metric_functions()
        
        # Check that we have the expected categories
        self.assertIn('centrality', metric_funcs)
        self.assertIn('clustering', metric_funcs)
        self.assertIn('structure', metric_funcs)
        
        # Test safe computation
        degree_result = metrics._safe_compute(
            metric_funcs['centrality']['degree'], G, 'degree'
        )
        self.assertIsNotNone(degree_result)
        self.assertEqual(len(degree_result), G.number_of_nodes())
        
        # Test complete metric computation
        all_metrics = metrics._compute_all_metrics_exact(G)
        
        # Check that metrics were computed
        self.assertIn('centrality', all_metrics)
        self.assertIn('clustering', all_metrics)
        self.assertIn('structure', all_metrics)
        
        # Check specific metrics
        if 'centrality' in all_metrics:
            self.assertIn('degree', all_metrics['centrality'])
        if 'structure' in all_metrics:
            self.assertIn('density', all_metrics['structure'])
    
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        metrics = ScientificNetworkMetrics(
            source_index=self.source_index,
            entity_index=self.entity_index,
            config=self.config
        )
        
        # Create some mock data
        window = (datetime(2021, 1, 1), datetime(2021, 1, 2))
        snapshot = NetworkSnapshot(
            window=window,
            frame='economic',
            network=nx.Graph()
        )
        
        key = metrics._get_key(window, 'economic')
        metrics.window_networks[key] = snapshot
        metrics.stats.completed_windows = 5
        
        # Save checkpoint
        metrics._save_checkpoint()
        
        # Create new instance and load checkpoint
        metrics2 = ScientificNetworkMetrics(
            source_index=self.source_index,
            entity_index=self.entity_index,
            config=self.config
        )
        
        checkpoint = metrics2._load_checkpoint()
        
        self.assertIsNotNone(checkpoint)
        self.assertIn('completed', checkpoint)
        self.assertIn('stats', checkpoint)
        self.assertEqual(checkpoint['stats'].completed_windows, 5)
    
    def test_compute_all_windows(self):
        """Test computing networks for all windows."""
        metrics = ScientificNetworkMetrics(
            source_index=self.source_index,
            entity_index=self.entity_index,
            config=self.config
        )
        
        # Create test windows
        windows = [
            (datetime(2021, 1, 1), datetime(2021, 1, 2)),
            (datetime(2021, 1, 2), datetime(2021, 1, 3))
        ]
        frames = ['economic']
        
        # Run computation
        results = metrics.compute_all_windows(
            windows, frames, resume_from_checkpoint=False
        )
        
        # Check results
        self.assertEqual(len(results), 2)  # 2 windows * 1 frame
        
        # Check that statistics were updated
        self.assertEqual(metrics.stats.completed_windows, 2)
        self.assertGreater(metrics.stats.total_time, 0)
    
    def test_safe_metric_functions(self):
        """Test safe metric computation functions."""
        metrics = ScientificNetworkMetrics(
            source_index=self.source_index,
            entity_index=self.entity_index,
            config=self.config
        )
        
        # Test with disconnected graph
        G = nx.Graph()
        G.add_edges_from([(1, 2), (3, 4)])  # Two components
        
        # These should handle disconnected graphs gracefully
        diameter = metrics._safe_diameter(G)
        self.assertIsNone(diameter)  # Disconnected graph
        
        radius = metrics._safe_radius(G)
        self.assertIsNone(radius)  # Disconnected graph
        
        avg_path = metrics._safe_average_path_length(G)
        # Should compute for largest component
        self.assertIsNotNone(avg_path)
        
        # Test with directed graph
        G_directed = nx.DiGraph()
        G_directed.add_edges_from([(1, 2), (2, 3), (3, 1)])
        
        diameter_dir = metrics._safe_diameter(G_directed)
        # Should work with strongly connected directed graph
        if nx.is_strongly_connected(G_directed):
            self.assertIsNotNone(diameter_dir)
    
    def test_empty_network_handling(self):
        """Test handling of empty networks."""
        metrics = ScientificNetworkMetrics(
            source_index=self.source_index,
            entity_index=self.entity_index,
            config=self.config
        )
        
        # Test with window that has no articles
        window = (datetime(2022, 1, 1), datetime(2022, 1, 2))  # No articles in this range
        frame = 'economic'
        
        snapshot = metrics.compute_window_network(window, frame)
        
        # Should return empty network
        self.assertEqual(snapshot.network.number_of_nodes(), 0)
        self.assertEqual(snapshot.network.number_of_edges(), 0)
        self.assertEqual(len(snapshot.metrics), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create more comprehensive test data
        self.source_index = {
            'article_profiles': {}
        }
        
        # Generate 50 test articles
        for i in range(50):
            date = datetime(2021, 1, 1) + timedelta(days=i % 7)
            self.source_index['article_profiles'][f'doc{i}'] = {
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
            }
        
        self.entity_index = {
            f'entity{i}': {
                'type': ['PER', 'ORG', 'LOC'][i % 3],
                'name': f'Entity {i}',
                'authority_score': np.random.random()
            }
            for i in range(10)
        }
        
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'output_dir': self.temp_dir,
            'parallel': True,
            'n_workers': 2,
            'checkpoint_frequency': 1,
            'gpu_enabled': False
        }
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline(self):
        """Test the complete pipeline with multiple windows and frames."""
        metrics = ScientificNetworkMetrics(
            source_index=self.source_index,
            entity_index=self.entity_index,
            config=self.config
        )
        
        # Create test windows
        windows = [
            (datetime(2021, 1, 1), datetime(2021, 1, 3)),
            (datetime(2021, 1, 3), datetime(2021, 1, 5)),
            (datetime(2021, 1, 5), datetime(2021, 1, 7))
        ]
        frames = ['economic', 'political']
        
        # Run full computation
        results = metrics.compute_all_windows(
            windows, frames, resume_from_checkpoint=False
        )
        
        # Verify results
        expected_computations = len(windows) * len(frames)
        self.assertEqual(len(results), expected_computations)
        
        # Check that all networks were computed
        for window in windows:
            for frame in frames:
                key = metrics._get_key(window, frame)
                self.assertIn(key, results)
                
                snapshot = results[key]
                self.assertIsInstance(snapshot, NetworkSnapshot)
                self.assertGreater(snapshot.network.number_of_nodes(), 0)
        
        # Check that files were saved
        output_dir = Path(self.temp_dir)
        network_files = list((output_dir / 'networks').glob('*/network.graphml'))
        self.assertGreater(len(network_files), 0)
        
        metric_files = list((output_dir / 'metrics').glob('*.json'))
        self.assertEqual(len(metric_files), expected_computations)
        
        # Check checkpoint
        checkpoint_file = output_dir / 'checkpoints' / 'latest.pkl'
        self.assertTrue(checkpoint_file.exists())
        
        # Check final report
        log_files = list((output_dir / 'logs').glob('report_*.json'))
        self.assertGreater(len(log_files), 0)
        
        # Load and verify report
        with open(log_files[0], 'r') as f:
            report = json.load(f)
        
        self.assertIn('summary', report)
        self.assertIn('network_statistics', report)
        self.assertEqual(report['network_statistics']['total_networks'], expected_computations)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)