"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
parallel_computer.py

MAIN OBJECTIVE:
---------------
This script provides optimized parallel shortest path computation for network analysis,
using chunked processing to avoid memory bottlenecks on M4 Max architecture.

Dependencies:
-------------
- os
- multiprocessing
- concurrent.futures
- typing
- numpy
- networkx
- time
- logging
- tqdm
- warnings

MAIN FEATURES:
--------------
1) Parallel shortest path computation with chunking
2) Memory-efficient aggregated statistics
3) Optimized for M4 Max with intelligent worker allocation
4) Progress tracking with tqdm
5) Automatic fallback for large networks

Author:
-------
Antoine Lemor
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx
import time
import logging
from tqdm import tqdm
import warnings

# Force spawn method for macOS compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

logger = logging.getLogger(__name__)


def compute_paths_chunk(args: Tuple) -> Tuple[float, float, int]:
    """
    Compute shortest paths for a chunk of nodes.
    Returns aggregated statistics instead of all paths to reduce memory.
    
    Args:
        args: (graph_dict, source_nodes)
        
    Returns:
        Tuple of (sum_of_lengths, max_length, count)
    """
    graph_dict, sources = args
    
    # Reconstruct graph
    G = nx.from_dict_of_lists(graph_dict)
    
    # Aggregate statistics
    total_length = 0
    max_length = 0
    path_count = 0
    
    for source in sources:
        try:
            # Compute shortest paths from this source
            lengths = nx.single_source_shortest_path_length(G, source)
            
            for target, length in lengths.items():
                # Only count unique pairs (avoid double counting in undirected graph)
                if source < target:
                    total_length += length
                    path_count += 1
                    if length > max_length:
                        max_length = length
        except Exception:
            continue
    
    return total_length, max_length, path_count


def parallel_compute_statistics(graph, n_workers: Optional[int] = None) -> Tuple[float, int]:
    """
    Compute graph statistics in parallel using aggregated approach.
    
    Args:
        graph: NetworkX graph
        n_workers: Number of workers (None for auto)
        
    Returns:
        Tuple of (average_path_length, diameter)
    """
    # Force optimal worker configuration for M4 Max
    if n_workers is None:
        # Limit to reasonable number to avoid macOS issues
        n_workers = min(16, mp.cpu_count())
    
    nodes = list(graph.nodes())
    n_nodes = len(nodes)
    
    if n_nodes == 0:
        return -1, -1
    
    print(f"\n📊 Computing statistics for {n_nodes:,} nodes using {n_workers} workers", flush=True)
    
    # Convert graph once
    graph_dict = nx.to_dict_of_lists(graph)
    
    # Create chunks - more chunks than workers for better load balancing
    chunk_size = max(1, n_nodes // (n_workers * 4))
    chunks = []
    
    for i in range(0, n_nodes, chunk_size):
        chunk_nodes = nodes[i:min(i + chunk_size, n_nodes)]
        chunks.append((graph_dict, chunk_nodes))
    
    print(f"   Processing {len(chunks)} chunks (~{chunk_size} nodes each)", flush=True)
    
    # Process chunks in parallel
    total_length = 0
    max_length = 0
    total_count = 0
    
    start_time = time.time()
    
    # Use simpler approach for better compatibility
    print(f"   Starting parallel computation with {n_workers} workers", flush=True)
    
    completed_chunks = 0
    failed_chunks = 0
    
    # Create executor with explicit cleanup
    executor = ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('spawn'))
    
    try:
        # Submit all tasks
        futures = [executor.submit(compute_paths_chunk, chunk) for chunk in chunks]
        
        # Collect results with progress bar
        with tqdm(total=len(futures), desc="   Computing") as pbar:
            for future in as_completed(futures, timeout=300):  # 5 min total timeout
                try:
                    chunk_sum, chunk_max, chunk_count = future.result(timeout=10)  # 10s per chunk
                    total_length += chunk_sum
                    total_count += chunk_count
                    if chunk_max > max_length:
                        max_length = chunk_max
                    completed_chunks += 1
                    
                except Exception as e:
                    logger.error(f"Chunk failed: {e}")
                    failed_chunks += 1
                
                pbar.update(1)
                pbar.set_postfix({'completed': completed_chunks, 'failed': failed_chunks})
    
    finally:
        # Force cleanup
        print(f"\n   Shutting down executor...", flush=True)
        try:
            # Try new API first
            executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            # Fall back to old API
            executor.shutdown(wait=True)
        print(f"   Executor shut down", flush=True)
    
    print(f"   Completed {completed_chunks}/{len(chunks)} chunks ({failed_chunks} failed)", flush=True)
    
    # This should be outside the try block
    print(f"   ProcessPoolExecutor closed, computing final statistics", flush=True)
    elapsed = time.time() - start_time
    
    if total_count > 0:
        avg_length = total_length / total_count
        diameter = max_length
        
        print(f"   ✅ Complete in {elapsed:.1f}s", flush=True)
        print(f"   • Paths analyzed: {total_count:,}", flush=True)
        print(f"   • Average length: {avg_length:.4f}", flush=True)
        print(f"   • Diameter: {diameter}", flush=True)
        
        return avg_length, diameter
    
    print("   ⚠️ No paths computed", flush=True)
    return -1, -1


def compute_exact_metrics(graph, n_workers: Optional[int] = None) -> Dict:
    """
    Compute exact network metrics efficiently.
    
    Args:
        graph: NetworkX graph
        n_workers: Number of workers
        
    Returns:
        Dictionary with network statistics
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    
    print(f"\n🔍 Analyzing graph: {n_nodes:,} nodes, {n_edges:,} edges")
    
    # For small graphs, use NetworkX directly
    if n_nodes < 1000:
        print("   Using direct computation for small graph")
        try:
            avg_length = nx.average_shortest_path_length(graph)
            diameter = nx.diameter(graph)
        except:
            avg_length = -1
            diameter = -1
    else:
        # Use parallel computation for large graphs
        avg_length, diameter = parallel_compute_statistics(graph, n_workers)
    
    return {
        'avg_path_length': avg_length,
        'diameter': diameter,
        'n_nodes': n_nodes,
        'n_edges': n_edges
    }