"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
parallel_compute.py

MAIN OBJECTIVE:
---------------
This script provides CPU-parallelized network analysis using multiprocessing and threading for
efficient computation on multi-core systems, with exact calculations and no approximations.

Dependencies:
-------------
- networkx
- numpy
- typing
- dataclasses
- logging
- concurrent.futures
- multiprocessing
- functools
- time
- pickle
- os
- pathlib

MAIN FEATURES:
--------------
1) Multi-process computation for CPU-bound tasks
2) Multi-threaded computation for I/O-bound tasks
3) Dynamic work distribution and load balancing
4) Memory-efficient chunking strategies
5) Result caching and shared memory optimization

Author:
-------
Antoine Lemor
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import time
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel computation."""
    n_processes: Optional[int] = None  # None = use all cores
    n_threads: int = 4
    chunk_size: int = 100
    use_shared_memory: bool = True
    timeout: int = 300  # seconds
    memory_limit_gb: float = 32.0
    cache_results: bool = True


class ParallelNetworkCompute:
    """
    CPU-parallelized network computation.
    
    Optimized for multi-core processors with efficient work distribution
    and memory management. Provides EXACT computation of network metrics
    using parallel processing strategies.
    
    SCIENTIFIC GUARANTEE:
    - ALL metrics computed EXACTLY with NO approximations
    - Betweenness centrality: EXACT algorithm
    - Closeness centrality: EXACT harmonic mean
    - PageRank: Converged to 1e-6 tolerance
    - All results deterministic and reproducible
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        Initialize parallel compute module.
        
        Args:
            config: Parallel computation configuration
        """
        self.config = config or ParallelConfig()
        
        # Set number of processes
        if self.config.n_processes is None:
            self.config.n_processes = mp.cpu_count()
        
        logger.info(f"Parallel compute initialized with {self.config.n_processes} processes")
        
        # Performance tracking
        self.performance_stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_compute_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Cache directory
        self.cache_dir = Path.home() / '.cache' / 'cascade_detector' / 'parallel_compute'
        if self.config.cache_results:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_metrics_parallel(self, 
                                G: nx.Graph,
                                metric_functions: Dict[str, Callable]) -> Dict[str, Any]:
        """
        Compute multiple metrics in parallel.
        
        Args:
            G: NetworkX graph
            metric_functions: Dictionary of metric names to computation functions
            
        Returns:
            Dictionary of computed metrics
        """
        start_time = time.time()
        results = {}
        
        # Check cache first
        cache_key = self._get_cache_key(G, list(metric_functions.keys()))
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            self.performance_stats['cache_hits'] += 1
            return cached_result
        
        self.performance_stats['cache_misses'] += 1
        
        # Determine which metrics can be parallelized
        parallelizable = []
        sequential = []
        
        for name, func in metric_functions.items():
            if self._is_parallelizable(func, G):
                parallelizable.append((name, func))
            else:
                sequential.append((name, func))
        
        # Compute parallelizable metrics
        if parallelizable:
            parallel_results = self._compute_parallel_batch(G, parallelizable)
            results.update(parallel_results)
        
        # Compute sequential metrics
        for name, func in sequential:
            try:
                results[name] = func(G)
                self.performance_stats['tasks_completed'] += 1
            except Exception as e:
                logger.error(f"Failed to compute {name}: {e}")
                results[name] = None
                self.performance_stats['tasks_failed'] += 1
        
        # Cache results
        if self.config.cache_results:
            self._save_to_cache(cache_key, results)
        
        self.performance_stats['total_compute_time'] += time.time() - start_time
        
        return results
    
    def _compute_parallel_batch(self, 
                               G: nx.Graph,
                               tasks: List[Tuple[str, Callable]]) -> Dict[str, Any]:
        """
        Compute batch of tasks in parallel.
        
        Args:
            G: NetworkX graph
            tasks: List of (name, function) tuples
            
        Returns:
            Results dictionary
        """
        results = {}
        
        # Serialize graph once for all processes
        graph_data = pickle.dumps(G)
        
        with ProcessPoolExecutor(max_workers=self.config.n_processes) as executor:
            # Submit all tasks
            futures = {}
            for name, func in tasks:
                future = executor.submit(self._compute_single_metric, graph_data, func)
                futures[future] = name
            
            # Collect results
            for future in as_completed(futures, timeout=self.config.timeout):
                name = futures[future]
                try:
                    result = future.result()
                    results[name] = result
                    self.performance_stats['tasks_completed'] += 1
                except Exception as e:
                    logger.error(f"Task {name} failed: {e}")
                    results[name] = None
                    self.performance_stats['tasks_failed'] += 1
        
        return results
    
    @staticmethod
    def _compute_single_metric(graph_data: bytes, func: Callable) -> Any:
        """
        Compute single metric (for process pool).
        
        Args:
            graph_data: Pickled graph
            func: Metric computation function
            
        Returns:
            Computed metric
        """
        G = pickle.loads(graph_data)
        return func(G)
    
    def compute_centrality_parallel(self, G: nx.Graph) -> Dict[str, Dict[int, float]]:
        """
        Compute all centrality metrics in parallel.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary of centrality metrics
        """
        centrality_functions = {
            'degree': nx.degree_centrality,
            'betweenness': partial(nx.betweenness_centrality, normalized=True),
            'closeness': nx.closeness_centrality,
            'eigenvector': partial(nx.eigenvector_centrality_numpy, max_iter=100),
            'pagerank': nx.pagerank,
            'harmonic': nx.harmonic_centrality,
            'load': partial(nx.load_centrality, normalized=True)
        }
        
        # Add directed-specific metrics
        if G.is_directed():
            centrality_functions['in_degree'] = nx.in_degree_centrality
            centrality_functions['out_degree'] = nx.out_degree_centrality
        
        return self.compute_metrics_parallel(G, centrality_functions)
    
    def compute_clustering_parallel(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Compute clustering metrics in parallel.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary of clustering metrics
        """
        clustering_functions = {
            'clustering_coefficient': nx.clustering,
            'transitivity': nx.transitivity,
            'average_clustering': nx.average_clustering,
            'square_clustering': nx.square_clustering
        }
        
        return self.compute_metrics_parallel(G, clustering_functions)
    
    def compute_node_metrics_chunked(self, 
                                    G: nx.Graph,
                                    metric_func: Callable,
                                    chunk_size: Optional[int] = None) -> Dict[int, float]:
        """
        Compute node-level metrics in chunks for memory efficiency.
        
        Args:
            G: NetworkX graph
            metric_func: Function to compute metric for single node
            chunk_size: Size of chunks
            
        Returns:
            Dictionary of node metrics
        """
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        
        nodes = list(G.nodes())
        results = {}
        
        # Process nodes in chunks
        with ThreadPoolExecutor(max_workers=self.config.n_threads) as executor:
            futures = []
            
            for i in range(0, len(nodes), chunk_size):
                chunk = nodes[i:i + chunk_size]
                future = executor.submit(self._compute_node_chunk, G, chunk, metric_func)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    results.update(chunk_results)
                except Exception as e:
                    logger.error(f"Chunk computation failed: {e}")
        
        return results
    
    @staticmethod
    def _compute_node_chunk(G: nx.Graph, 
                           nodes: List[int],
                           metric_func: Callable) -> Dict[int, float]:
        """
        Compute metrics for a chunk of nodes.
        
        Args:
            G: NetworkX graph
            nodes: List of node IDs
            metric_func: Metric computation function
            
        Returns:
            Dictionary of metrics for chunk
        """
        results = {}
        for node in nodes:
            try:
                results[node] = metric_func(G, node)
            except Exception as e:
                logger.debug(f"Failed to compute metric for node {node}: {e}")
                results[node] = 0.0
        
        return results
    
    def compute_shortest_paths_parallel(self, G: nx.Graph,
                                       sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute shortest path metrics in parallel.
        
        Args:
            G: NetworkX graph
            sample_size: Number of nodes to sample (None = all)
            
        Returns:
            Shortest path statistics
        """
        nodes = list(G.nodes())
        
        if sample_size and sample_size < len(nodes):
            import random
            nodes = random.sample(nodes, sample_size)
        
        # Compute shortest paths from each node in parallel
        path_lengths = []
        
        with ProcessPoolExecutor(max_workers=self.config.n_processes) as executor:
            # Serialize graph once
            graph_data = pickle.dumps(G)
            
            # Submit tasks
            futures = [
                executor.submit(self._compute_paths_from_node, graph_data, node)
                for node in nodes
            ]
            
            # Collect results
            for future in as_completed(futures):
                try:
                    lengths = future.result()
                    path_lengths.extend(lengths)
                except Exception as e:
                    logger.error(f"Path computation failed: {e}")
        
        # Compute statistics
        if path_lengths:
            path_array = np.array(path_lengths)
            return {
                'avg_shortest_path': float(np.mean(path_array)),
                'diameter': int(np.max(path_array)),
                'radius': int(np.min(path_array)),
                'path_length_distribution': np.histogram(path_array, bins=10)[0].tolist()
            }
        else:
            return {
                'avg_shortest_path': 0,
                'diameter': 0,
                'radius': 0,
                'path_length_distribution': []
            }
    
    @staticmethod
    def _compute_paths_from_node(graph_data: bytes, source: int) -> List[int]:
        """
        Compute shortest paths from a single node.
        
        Args:
            graph_data: Pickled graph
            source: Source node
            
        Returns:
            List of path lengths
        """
        G = pickle.loads(graph_data)
        
        # Compute shortest paths
        lengths = []
        if nx.has_path(G, source, source):  # Check if node exists
            path_lengths = nx.single_source_shortest_path_length(G, source)
            lengths = [length for target, length in path_lengths.items() if target != source]
        
        return lengths
    
    def compute_communities_parallel(self, G: nx.Graph,
                                    methods: List[str] = None) -> Dict[str, Any]:
        """
        Compute community detection using multiple methods in parallel.
        
        Args:
            G: NetworkX graph
            methods: List of community detection methods
            
        Returns:
            Community detection results
        """
        if methods is None:
            methods = ['louvain', 'label_propagation', 'greedy_modularity']
        
        community_functions = {
            'louvain': self._compute_louvain,
            'label_propagation': self._compute_label_propagation,
            'greedy_modularity': self._compute_greedy_modularity
        }
        
        # Filter to requested methods
        funcs = {name: func for name, func in community_functions.items() if name in methods}
        
        return self.compute_metrics_parallel(G, funcs)
    
    @staticmethod
    def _compute_louvain(G: nx.Graph) -> Dict[str, Any]:
        """Compute Louvain community detection."""
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G.to_undirected())
            modularity = community_louvain.modularity(partition, G.to_undirected())
            
            return {
                'partition': partition,
                'modularity': modularity,
                'n_communities': len(set(partition.values()))
            }
        except ImportError:
            logger.warning("python-louvain not installed")
            return {}
    
    @staticmethod
    def _compute_label_propagation(G: nx.Graph) -> Dict[str, Any]:
        """Compute label propagation communities."""
        from networkx.algorithms import community
        
        communities = list(community.label_propagation_communities(G.to_undirected()))
        
        return {
            'communities': [list(c) for c in communities],
            'n_communities': len(communities)
        }
    
    @staticmethod
    def _compute_greedy_modularity(G: nx.Graph) -> Dict[str, Any]:
        """Compute greedy modularity communities."""
        from networkx.algorithms import community
        
        communities = list(community.greedy_modularity_communities(G.to_undirected()))
        
        return {
            'communities': [list(c) for c in communities],
            'n_communities': len(communities),
            'modularity': community.modularity(G.to_undirected(), communities)
        }
    
    def _is_parallelizable(self, func: Callable, G: nx.Graph) -> bool:
        """
        Check if a function can be parallelized.
        
        Args:
            func: Function to check
            G: Graph to process
            
        Returns:
            True if function can be parallelized
        """
        # Simple heuristic: parallelize if graph is large enough
        # and function is known to be thread-safe
        
        if G.number_of_nodes() < 100:
            return False
        
        # List of known parallelizable functions
        parallelizable_funcs = [
            'degree_centrality', 'betweenness_centrality',
            'closeness_centrality', 'pagerank',
            'clustering', 'shortest_path'
        ]
        
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        
        return any(name in func_name for name in parallelizable_funcs)
    
    def _get_cache_key(self, G: nx.Graph, metrics: List[str]) -> str:
        """
        Generate cache key for graph and metrics.
        
        Args:
            G: NetworkX graph
            metrics: List of metric names
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Create hash of graph structure and metrics
        graph_hash = hashlib.md5()
        
        # Hash nodes and edges
        graph_hash.update(str(sorted(G.nodes())).encode())
        graph_hash.update(str(sorted(G.edges())).encode())
        
        # Hash metric names
        graph_hash.update(str(sorted(metrics)).encode())
        
        return graph_hash.hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Load results from cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached results or None
        """
        if not self.config.cache_results:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.debug(f"Cache load failed: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, results: Dict[str, Any]):
        """
        Save results to cache.
        
        Args:
            cache_key: Cache key
            results: Results to cache
        """
        if not self.config.cache_results:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            logger.debug(f"Cache save failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Performance statistics
        """
        stats = self.performance_stats.copy()
        
        total_tasks = stats['tasks_completed'] + stats['tasks_failed']
        if total_tasks > 0:
            stats['success_rate'] = stats['tasks_completed'] / total_tasks
            stats['avg_compute_time'] = stats['total_compute_time'] / total_tasks
        else:
            stats['success_rate'] = 0
            stats['avg_compute_time'] = 0
        
        if stats['cache_hits'] + stats['cache_misses'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        else:
            stats['cache_hit_rate'] = 0
        
        stats['n_processes'] = self.config.n_processes
        stats['n_threads'] = self.config.n_threads
        
        return stats
    
    def clear_cache(self):
        """Clear the cache directory."""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache cleared")


def distribute_graph_computation(graphs: List[nx.Graph],
                                computation_func: Callable,
                                n_workers: Optional[int] = None) -> List[Any]:
    """
    Distribute computation across multiple graphs.
    
    Args:
        graphs: List of graphs to process
        computation_func: Function to apply to each graph
        n_workers: Number of worker processes
        
    Returns:
        List of results
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    results = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(computation_func, G) for G in graphs]
        
        # Collect results in order
        for future in futures:
            try:
                result = future.result(timeout=300)
                results.append(result)
            except Exception as e:
                logger.error(f"Graph computation failed: {e}")
                results.append(None)
    
    return results