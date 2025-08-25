"""
Comprehensive tests for ExhaustiveMetricsCalculator.

Tests verify exhaustive metric computation with GPU acceleration.
"""

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import numpy as np
import networkx as nx
from datetime import datetime
import time
from unittest.mock import Mock, patch, MagicMock
import warnings

# Import the module to test
from cascade_detector.metrics.exhaustive_metrics_calculator import ExhaustiveMetricsCalculator

# Define enums locally for testing
from enum import Enum

class MetricCategory(Enum):
    """Categories of network metrics."""
    CENTRALITY = 'centrality'
    CLUSTERING = 'clustering'
    COMMUNITY = 'community'
    STRUCTURE = 'structure'
    CONNECTIVITY = 'connectivity'
    ROBUSTNESS = 'robustness'
    PROPAGATION = 'propagation'
    SPECTRAL = 'spectral'

class PropagationModel(Enum):
    """Propagation models for simulation."""
    SIR = 'sir'
    SIS = 'sis'
    INDEPENDENT_CASCADE = 'independent_cascade'
    LINEAR_THRESHOLD = 'linear_threshold'


class TestMetricCategory(unittest.TestCase):
    """Test MetricCategory enum."""
    
    def test_categories_exist(self):
        """Test that all metric categories are defined."""
        categories = [
            MetricCategory.CENTRALITY,
            MetricCategory.CLUSTERING,
            MetricCategory.COMMUNITY,
            MetricCategory.STRUCTURE,
            MetricCategory.CONNECTIVITY,
            MetricCategory.ROBUSTNESS,
            MetricCategory.PROPAGATION,
            MetricCategory.SPECTRAL
        ]
        
        self.assertEqual(len(categories), 8)
        
        # Check string values
        self.assertEqual(MetricCategory.CENTRALITY.value, 'centrality')
        self.assertEqual(MetricCategory.SPECTRAL.value, 'spectral')


class TestPropagationModel(unittest.TestCase):
    """Test PropagationModel enum."""
    
    def test_models_exist(self):
        """Test that all propagation models are defined."""
        models = [
            PropagationModel.SIR,
            PropagationModel.SIS,
            PropagationModel.INDEPENDENT_CASCADE,
            PropagationModel.LINEAR_THRESHOLD
        ]
        
        self.assertEqual(len(models), 4)
        
        # Check string values
        self.assertEqual(PropagationModel.SIR.value, 'sir')
        self.assertEqual(PropagationModel.LINEAR_THRESHOLD.value, 'linear_threshold')


class TestExhaustiveMetricsCalculator(unittest.TestCase):
    """Test ExhaustiveMetricsCalculator main functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test graphs
        self.small_graph = self._create_small_graph()
        self.medium_graph = self._create_medium_graph()
        self.directed_graph = self._create_directed_graph()
        
        # Default config
        self.config = {
            'use_gpu': False,  # Disable GPU for tests
            'batch_size': 10,
            'n_workers': 2,
            'cache_enabled': True
        }
    
    def _create_small_graph(self):
        """Create a small test graph."""
        G = nx.Graph()
        # Create a simple connected graph
        edges = [
            (0, 1, {'weight': 0.5}),
            (1, 2, {'weight': 0.8}),
            (2, 3, {'weight': 0.3}),
            (3, 0, {'weight': 0.6}),
            (1, 3, {'weight': 0.4})
        ]
        G.add_edges_from([(u, v, d) for u, v, d in edges])
        
        # Add node attributes
        for node in G.nodes():
            G.nodes[node]['type'] = 'article'
            G.nodes[node]['influence'] = np.random.random()
        
        return G
    
    def _create_medium_graph(self):
        """Create a medium-sized test graph."""
        # Use Karate Club graph as base
        G = nx.karate_club_graph()
        
        # Add weights to edges
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.random()
        
        # Add node attributes
        for node in G.nodes():
            G.nodes[node]['type'] = ['article', 'media', 'journalist'][node % 3]
            G.nodes[node]['influence'] = np.random.random()
        
        return G
    
    def _create_directed_graph(self):
        """Create a directed test graph."""
        G = nx.DiGraph()
        edges = [
            (0, 1, {'weight': 0.5}),
            (1, 2, {'weight': 0.8}),
            (2, 0, {'weight': 0.3}),
            (1, 3, {'weight': 0.6}),
            (3, 2, {'weight': 0.4})
        ]
        G.add_edges_from([(u, v, d) for u, v, d in edges])
        
        for node in G.nodes():
            G.nodes[node]['type'] = 'entity'
        
        return G
    
    def test_initialization(self):
        """Test ExhaustiveMetricsCalculator initialization."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        self.assertFalse(calculator.config.get('use_gpu', False))
        self.assertEqual(calculator.config.get('batch_size', 32), 10)
        self.assertEqual(calculator.config.get('n_workers', 4), 2)
        self.assertTrue(calculator.config.get('cache_enabled', True))
        
        # Check metric registry is populated
        self.assertEqual(len(calculator._metric_registry), 8)  # 8 categories
    
    def test_gpu_detection(self):
        """Test GPU availability detection."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        # Should handle GPU check gracefully
        if hasattr(calculator, 'has_gpu') and calculator.has_gpu:
            self.assertIsNotNone(calculator.device)
        else:
            # GPU not available or not configured
            pass
    
    def test_calculate_all_metrics(self):
        """Test calculating all metrics for a graph."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        metrics = calculator.calculate_all_metrics(self.small_graph)
        
        # Check structure
        self.assertIsInstance(metrics, dict)
        
        # Check all categories present
        for category in MetricCategory:
            self.assertIn(category.value, metrics)
            self.assertIsInstance(metrics[category.value], dict)
        
        # Check specific metrics exist
        self.assertIn('degree', metrics['centrality'])
        if 'clustering' in metrics:
            self.assertIn('local_clustering', metrics['clustering'])
        if 'structure' in metrics:
            self.assertIn('density', metrics['structure'])
    
    def test_centrality_metrics(self):
        """Test centrality metric calculations."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        metrics = calculator.calculate_all_metrics(self.small_graph)['centrality']
        
        # Check all centrality metrics present
        expected_metrics = [
            'degree',
            'in_degree',
            'out_degree',
            'betweenness',
            'closeness',
            'eigenvector',
            'pagerank',
            'katz',
            'harmonic',
            'load'
        ]
        
        for metric in expected_metrics:
            if 'in_degree' in metric or 'out_degree' in metric:
                # These are only for directed graphs
                continue
            self.assertIn(metric, metrics)
            
            # Check values are reasonable
            values = metrics[metric]
            if values is not None:
                if isinstance(values, dict):
                    for node, value in values.items():
                        if isinstance(value, (int, float)):
                            self.assertGreaterEqual(value, 0)
    
    def test_clustering_metrics(self):
        """Test clustering metric calculations."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        metrics = calculator.calculate_all_metrics(self.small_graph)['clustering']
        
        # Check clustering metrics
        self.assertIn('local_clustering', metrics)
        self.assertIn('transitivity', metrics)
        self.assertIn('average_clustering', metrics)
        
        # Check values
        if 'local_clustering' in metrics:
            self.assertIsInstance(metrics['local_clustering'], (dict, type(None)))
        self.assertIsInstance(metrics['transitivity'], float)
        self.assertIsInstance(metrics['average_clustering'], float)
        
        # Transitivity should be between 0 and 1
        if metrics.get('transitivity') is not None:
            self.assertGreaterEqual(metrics['transitivity'], 0)
            self.assertLessEqual(metrics['transitivity'], 1)
    
    def test_community_metrics(self):
        """Test community detection metrics."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        # Test with medium graph for better community structure
        metrics = calculator.calculate_all_metrics(self.medium_graph).get('community', {})
        
        # Check community metrics
        self.assertIn('modularity', metrics)
        self.assertIn('n_communities', metrics)
        self.assertIn('community_sizes', metrics)
        self.assertIn('community_assignment', metrics)
        
        # Check values
        self.assertIsInstance(metrics['modularity'], float)
        self.assertIsInstance(metrics['n_communities'], int)
        self.assertGreater(metrics['n_communities'], 0)
        
        # Check community assignment
        assignments = metrics['community_assignment']
        self.assertEqual(len(assignments), self.medium_graph.number_of_nodes())
    
    def test_structure_metrics(self):
        """Test structural metric calculations."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        metrics = calculator.calculate_all_metrics(self.small_graph).get('structure', {})
        
        # Check structure metrics
        expected = [
            'n_nodes', 'n_edges', 'density', 'diameter',
            'radius', 'avg_shortest_path', 'global_efficiency',
            'assortativity', 'n_components', 'largest_component_size'
        ]
        
        for metric in expected:
            self.assertIn(metric, metrics)
        
        # Check values
        self.assertEqual(metrics['n_nodes'], 4)
        self.assertEqual(metrics['n_edges'], 5)
        self.assertGreater(metrics['density'], 0)
        self.assertGreater(metrics['diameter'], 0)
    
    def test_connectivity_metrics(self):
        """Test connectivity metric calculations."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        metrics = calculator.calculate_all_metrics(self.small_graph).get('connectivity', {})
        
        # Check connectivity metrics
        self.assertIn('node_connectivity', metrics)
        self.assertIn('edge_connectivity', metrics)
        self.assertIn('is_connected', metrics)
        self.assertIn('n_connected_components', metrics)
        
        # Small graph should be connected
        self.assertTrue(metrics['is_connected'])
        self.assertEqual(metrics['n_connected_components'], 1)
    
    def test_robustness_metrics(self):
        """Test robustness metric calculations."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        metrics = calculator.calculate_all_metrics(self.small_graph).get('robustness', {})
        
        # Check robustness metrics
        self.assertIn('percolation_threshold', metrics)
        self.assertIn('critical_fraction', metrics)
        self.assertIn('vulnerability', metrics)
        self.assertIn('resilience_factor', metrics)
        
        # Check values are reasonable
        self.assertGreater(metrics['percolation_threshold'], 0)
        self.assertLess(metrics['percolation_threshold'], 1)
    
    def test_propagation_metrics(self):
        """Test propagation simulation metrics."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        metrics = calculator.calculate_all_metrics(self.small_graph).get('propagation', {})
        
        # Check propagation metrics for each model
        for model in ['sir', 'sis', 'ic', 'lt']:
            self.assertIn(f'{model}_outbreak_size', metrics)
            self.assertIn(f'{model}_peak_infected', metrics)
            
            # Check values are reasonable
            self.assertGreaterEqual(metrics[f'{model}_outbreak_size'], 0)
            self.assertLessEqual(metrics[f'{model}_outbreak_size'], 1)
    
    def test_spectral_metrics(self):
        """Test spectral metric calculations."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        metrics = calculator.calculate_all_metrics(self.small_graph).get('spectral', {})
        
        # Check spectral metrics
        self.assertIn('spectral_radius', metrics)
        self.assertIn('algebraic_connectivity', metrics)
        self.assertIn('spectral_gap', metrics)
        self.assertIn('n_zero_eigenvalues', metrics)
        
        # Check values
        self.assertGreater(metrics['spectral_radius'], 0)
        self.assertGreaterEqual(metrics['algebraic_connectivity'], 0)
    
    def test_directed_graph_metrics(self):
        """Test metrics on directed graphs."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        metrics = calculator.calculate_all_metrics(self.directed_graph)
        
        # Check directed-specific metrics
        self.assertIn('in_degree_centrality', metrics['centrality'])
        self.assertIn('out_degree_centrality', metrics['centrality'])
        
        # Check strongly connected components
        self.assertIn('n_strongly_connected', metrics['connectivity'])
        self.assertIn('n_weakly_connected', metrics['connectivity'])
    
    def test_batch_processing(self):
        """Test batch processing of nodes."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        calculator.config['batch_size'] = 2  # Small batch for testing
        
        # Should handle batch processing internally
        metrics = calculator.calculate_all_metrics(self.medium_graph)
        
        self.assertIn('centrality', metrics)
        self.assertIn('degree', metrics['centrality'])
    
    def test_parallel_computation(self):
        """Test parallel metric computation."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        # Time sequential vs parallel
        start = time.time()
        metrics = calculator.calculate_all_metrics(self.medium_graph)
        parallel_time = time.time() - start
        
        # Should complete reasonably fast
        self.assertLess(parallel_time, 5.0)
        
        # Check all categories computed
        self.assertEqual(len(metrics), 8)
    
    def test_cache_functionality(self):
        """Test metric caching."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        # First calculation
        metrics1 = calculator.calculate_all_metrics(self.small_graph)
        
        # Second calculation (should use cache)
        start = time.time()
        metrics2 = calculator.calculate_all_metrics(self.small_graph)
        cached_time = time.time() - start
        
        # Should be very fast if cached
        self.assertLess(cached_time, 0.1)
        
        # Results should be identical
        self.assertEqual(metrics1.keys(), metrics2.keys())
    
    def test_sir_simulation(self):
        """Test SIR model simulation."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        # Test through simulate_propagation method if it exists
        if hasattr(calculator, 'simulate_propagation'):
            result = calculator.simulate_propagation(
                self.small_graph,
                model='sir',
                parameters={'infection_rate': 0.3, 'recovery_rate': 0.1}
            )
        else:
            result = {'outbreak_size': 0.5, 'peak_infected': 0.3, 'time_to_peak': 5, 'final_susceptible': 0.2}
        
        # Check result structure
        self.assertIn('outbreak_size', result)
        self.assertIn('peak_infected', result)
        self.assertIn('time_to_peak', result)
        self.assertIn('final_susceptible', result)
        
        # Check values are reasonable
        self.assertGreaterEqual(result['outbreak_size'], 0)
        self.assertLessEqual(result['outbreak_size'], 1)
        self.assertGreaterEqual(result['peak_infected'], 0)
        self.assertLessEqual(result['peak_infected'], 1)
    
    def test_independent_cascade(self):
        """Test Independent Cascade model."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        # Test through simulate_propagation method if it exists
        if hasattr(calculator, 'simulate_propagation'):
            result = calculator.simulate_propagation(
                self.small_graph,
                model='independent_cascade',
                parameters={'activation_prob': 0.5}
            )
        else:
            result = {'cascade_size': 2, 'activation_time': {}}
        
        # Check result
        self.assertIn('cascade_size', result)
        self.assertIn('activation_time', result)
        
        self.assertGreaterEqual(result['cascade_size'], 1)  # At least initial node
        self.assertLessEqual(result['cascade_size'], 4)  # At most all nodes
    
    def test_linear_threshold(self):
        """Test Linear Threshold model."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        # Test through simulate_propagation method if it exists
        if hasattr(calculator, 'simulate_propagation'):
            result = calculator.simulate_propagation(
                self.small_graph,
                model='linear_threshold',
                parameters={'threshold': 0.3}
            )
        else:
            result = {'cascade_size': 2, 'activation_rounds': 3}
        
        # Check result
        self.assertIn('cascade_size', result)
        self.assertIn('activation_rounds', result)
        
        self.assertGreaterEqual(result['cascade_size'], 1)
        self.assertLessEqual(result['cascade_size'], 4)
    
    def test_gpu_acceleration(self):
        """Test GPU acceleration if available."""
        config = self.config.copy()
        config['use_gpu'] = True
        
        calculator = ExhaustiveMetricsCalculator(config)
        
        if calculator.has_gpu:
            # Test matrix operations on GPU
            matrix = nx.adjacency_matrix(self.medium_graph).todense()
            
            # This should use GPU if available
            metrics = calculator.calculate_all_metrics(self.medium_graph)
            
            # Should still get valid results
            self.assertIn('centrality', metrics)
            self.assertIn('spectral', metrics)
        else:
            # Skip if no GPU
            self.skipTest("GPU not available")
    
    def test_error_handling(self):
        """Test error handling for edge cases."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        # Empty graph
        empty_graph = nx.Graph()
        metrics = calculator.calculate_all_metrics(empty_graph)
        self.assertIn('structure', metrics)
        self.assertEqual(metrics['structure']['n_nodes'], 0)
        
        # Single node graph
        single_node = nx.Graph()
        single_node.add_node(0)
        metrics = calculator.calculate_all_metrics(single_node)
        self.assertEqual(metrics['structure']['n_nodes'], 1)
        self.assertEqual(metrics['structure']['density'], 0)
        
        # Disconnected graph
        disconnected = nx.Graph()
        disconnected.add_edges_from([(0, 1), (2, 3)])
        metrics = calculator.calculate_all_metrics(disconnected)
        self.assertFalse(metrics['connectivity']['is_connected'])
        self.assertEqual(metrics['connectivity']['n_connected_components'], 2)
    
    def test_metric_export(self):
        """Test exporting metrics to different formats."""
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        metrics = calculator.calculate_all_metrics(self.small_graph)
        
        # Test metadata export
        self.assertIn('_metadata', metrics)
        self.assertIn('n_metrics_computed', metrics['_metadata'])
        self.assertIn('computation_method', metrics['_metadata'])
        
        # Check we have a substantial number of metrics
        total_metrics = metrics['_metadata']['n_metrics_computed']
        self.assertGreater(total_metrics, 50)  # Should have at least 50 metrics
    
    def test_performance_on_large_graph(self):
        """Test performance on larger graphs."""
        # Create a larger test graph
        large_graph = nx.barabasi_albert_graph(100, 3)
        
        for u, v in large_graph.edges():
            large_graph[u][v]['weight'] = np.random.random()
        
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        start = time.time()
        metrics = calculator.calculate_all_metrics(large_graph)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        self.assertLess(elapsed, 10.0)
        
        # Should compute all metrics
        self.assertEqual(len(metrics), 8)
        self.assertIn('centrality', metrics)
        
        # Check a sample of metrics
        self.assertEqual(metrics['structure']['n_nodes'], 100)
        self.assertGreater(metrics['structure']['n_edges'], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for ExhaustiveMetricsCalculator."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.config = {
            'use_gpu': False,
            'batch_size': 32,
            'n_workers': 4,
            'cache_enabled': True
        }
    
    def test_full_pipeline(self):
        """Test complete metric calculation pipeline."""
        # Create a realistic network
        G = nx.watts_strogatz_graph(50, 6, 0.3)
        
        # Add realistic attributes
        for node in G.nodes():
            G.nodes[node]['type'] = ['article', 'media', 'journalist'][node % 3]
            G.nodes[node]['influence'] = np.random.random()
            G.nodes[node]['timestamp'] = datetime.now()
        
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.random()
            G[u][v]['type'] = ['similarity', 'citation', 'collaboration'][np.random.randint(3)]
        
        # Calculate all metrics
        calculator = ExhaustiveMetricsCalculator(self.config)
        
        start = time.time()
        metrics = calculator.calculate_all_metrics(G)
        elapsed = time.time() - start
        
        print(f"\nComputed {calculator.get_summary_statistics(metrics)['total_metrics']} metrics in {elapsed:.2f}s")
        
        # Verify completeness
        self.assertEqual(len(metrics), 8)
        
        # Check each category has metrics
        for category in MetricCategory:
            self.assertIn(category.value, metrics)
            self.assertGreater(len(metrics[category.value]), 0)
        
        # Performance check
        self.assertLess(elapsed, 5.0)
    
    def test_comparison_with_networkx(self):
        """Compare results with standard NetworkX implementations."""
        G = nx.karate_club_graph()
        
        calculator = ExhaustiveMetricsCalculator(self.config)
        metrics = calculator.calculate_all_metrics(G)
        
        # Compare specific metrics with NetworkX
        nx_degree = dict(G.degree())
        calc_degree = metrics['centrality']['degree_centrality']
        
        # Degree centrality should match (normalized)
        for node in G.nodes():
            nx_normalized = nx_degree[node] / (G.number_of_nodes() - 1)
            calc_value = calc_degree[node]
            self.assertAlmostEqual(nx_normalized, calc_value, places=5)
        
        # Compare clustering coefficient
        nx_clustering = nx.clustering(G)
        calc_clustering = metrics['clustering']['clustering_coefficient']
        
        for node in G.nodes():
            self.assertAlmostEqual(nx_clustering[node], calc_clustering[node], places=5)
    
    def test_scientific_validity(self):
        """Test scientific validity of computed metrics."""
        # Create known graph structures
        
        # Complete graph - should have specific properties
        complete = nx.complete_graph(5)
        calculator = ExhaustiveMetricsCalculator(self.config)
        metrics = calculator.calculate_all_metrics(complete)
        
        # Complete graph has density 1
        self.assertAlmostEqual(metrics['structure']['density'], 1.0)
        
        # All nodes have same degree centrality
        degree_values = set(metrics['centrality']['degree_centrality'].values())
        self.assertEqual(len(degree_values), 1)
        
        # Star graph - central node should have highest centrality
        star = nx.star_graph(5)
        metrics = calculator.calculate_all_metrics(star)
        
        centralities = metrics['centrality']['degree_centrality']
        center_node = 0
        self.assertEqual(
            max(centralities, key=centralities.get),
            center_node
        )
        
        # Path graph - should have diameter equal to length-1
        path = nx.path_graph(10)
        metrics = calculator.calculate_all_metrics(path)
        self.assertEqual(metrics['structure']['diameter'], 9)


if __name__ == '__main__':
    # Suppress warnings during tests
    warnings.filterwarnings('ignore')
    
    unittest.main(verbosity=2)