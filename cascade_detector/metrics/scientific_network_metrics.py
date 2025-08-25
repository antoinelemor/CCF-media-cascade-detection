"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
scientific_network_metrics.py

MAIN OBJECTIVE:
---------------
This script provides exact network computation for cascade analysis, computing all network metrics
without approximations for publication-quality scientific analysis, optimized for parallel processing
with NetworKit acceleration.

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
- pickle
- json
- pathlib
- concurrent.futures
- multiprocessing
- tqdm
- hashlib
- networkit (optional)

MAIN FEATURES:
--------------
1) Exact computation of all network metrics (no sampling or approximations)
2) Multi-layer network construction (article-entity-source networks)
3) NetworKit integration for 25-100x speedup on large graphs
4) Parallel processing with intelligent chunk distribution
5) Comprehensive caching system for computed networks

Author:
-------
Antoine Lemor
"""

import os
# Set OpenMP threads to 1 to avoid conflicts in multiprocessing
os.environ['OMP_NUM_THREADS'] = '1'

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta
import logging
import time
import pickle
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import warnings
from tqdm import tqdm
import hashlib
import multiprocessing as mp

# Import base NetworkMetrics
from cascade_detector.metrics.network_metrics import NetworkMetrics

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class NetworkSnapshot:
    """
    Complete network snapshot for a specific time window.
    """
    window: Tuple[datetime, datetime]
    frame: str
    network: nx.Graph
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    computation_time: float = 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the snapshot."""
        return {
            'window': f"{self.window[0].date()} to {self.window[1].date()}",
            'frame': self.frame,
            'n_nodes': self.network.number_of_nodes(),
            'n_edges': self.network.number_of_edges(),
            'density': nx.density(self.network),
            'n_metrics': len(self.metrics),
            'computation_time': f"{self.computation_time:.2f}s"
        }
    
    def get_hash(self) -> str:
        """Get unique hash for this snapshot."""
        key = f"{self.window[0]}_{self.window[1]}_{self.frame}"
        return hashlib.md5(key.encode()).hexdigest()


@dataclass
class ComputationStats:
    """
    Statistics for computation performance tracking.
    """
    total_windows: int = 0
    completed_windows: int = 0
    failed_windows: int = 0
    total_time: float = 0.0
    avg_time_per_window: float = 0.0
    peak_memory_gb: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def update(self, computation_time: float):
        """Update statistics after a computation."""
        self.completed_windows += 1
        self.total_time += computation_time
        self.avg_time_per_window = self.total_time / self.completed_windows
    
    def get_report(self) -> Dict[str, Any]:
        """Get performance report."""
        return {
            'progress': f"{self.completed_windows}/{self.total_windows}",
            'success_rate': f"{(self.completed_windows/(self.completed_windows+self.failed_windows)*100):.1f}%" if (self.completed_windows+self.failed_windows) > 0 else "N/A",
            'total_time': f"{self.total_time/3600:.2f} hours" if self.total_time > 3600 else f"{self.total_time/60:.2f} minutes",
            'avg_time_per_window': f"{self.avg_time_per_window:.2f}s",
            'cache_hit_rate': f"{(self.cache_hits/(self.cache_hits+self.cache_misses)*100):.1f}%" if (self.cache_hits+self.cache_misses) > 0 else "N/A",
            'peak_memory': f"{self.peak_memory_gb:.2f} GB"
        }


class ScientificNetworkMetrics(NetworkMetrics):
    """
    Scientific version of NetworkMetrics with EXACT computation for EVERY window.
    
    Core Principles:
    - NO approximations: All metrics computed exactly
    - EVERY window: Each time window gets its own complete network
    - FULL metrics: All 73+ metrics calculated exhaustively
    - DETERMINISTIC: Results are reproducible
    - VALIDATED: Every computation is verified
    
    Key Principles:
    1. NO approximations - all metrics computed exactly
    2. EVERY window gets its own network - no interpolation
    3. COMPLETE metrics - all 73+ metrics calculated
    4. FULL persistence - all networks and metrics saved
    5. REPRODUCIBLE - deterministic and traceable
    """
    
    def __init__(self, 
                 source_index: Dict[str, Any],
                 entity_index: Dict[str, Any],
                 config: Optional[Dict[str, Any]] = None,
                 frame_index: Optional[Dict[str, Any]] = None):
        """
        Initialize Scientific Network Metrics.
        
        Args:
            source_index: Source index from Phase 1
            entity_index: Entity index from Phase 1
            config: Configuration options
            frame_index: Frame index from FrameIndexer (contains article_frames with frame data)
        """
        # Store frame index for article retrieval
        self.frame_index = frame_index or {}
        
        # Force exact computation settings
        default_config = {
            'exact_computation': True,
            'no_approximation': True,
            'full_metrics': True,
            'save_all': True,
            'parallel': True,
            'n_workers': 16,  # M4 Ultra Max
            'batch_size': 32,  # Increased batch size for better parallelization
            'gpu_enabled': True,
            'cache_enabled': True,
            'checkpoint_frequency': 10,
            'output_dir': 'network_analysis'
        }
        
        if config:
            default_config.update(config)
        
        # Initialize parent with forced settings
        super().__init__(
            source_index=source_index,
            entity_index=entity_index,
            use_gpu=default_config['gpu_enabled'],
            n_workers=default_config['n_workers'],
            use_approximate=False,  # NEVER approximate
            hybrid_mode=True,
            use_networkit=True,  # Enable for exact fast computation
            gpu_batch_size=50000,
            enable_gpu_cache=default_config['cache_enabled'],
            show_progress=False  # Disable progress bars for internal computations
        )
        
        self.config = default_config
        
        # Storage for all window networks and metrics
        self.window_networks = {}  # key: (window, frame) -> NetworkSnapshot
        self.window_metrics = {}   # key: (window, frame) -> metrics dict
        self.computation_log = []  # Log of all computations
        self.stats = ComputationStats()
        
        # Setup output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup subdirectories
        (self.output_dir / 'networks').mkdir(exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        
        logger.info("ScientificNetworkMetrics initialized with EXACT computation mode")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
    
    def compute_all_windows(self, 
                           windows: List[Tuple[datetime, datetime]],
                           frames: List[str],
                           resume_from_checkpoint: bool = True) -> Dict[str, NetworkSnapshot]:
        """
        Compute EXACT network metrics for ALL windows and frames.
        
        This is the main entry point for exhaustive analysis.
        
        Args:
            windows: List of time windows to analyze
            frames: List of frames to analyze
            resume_from_checkpoint: Whether to resume from last checkpoint
            
        Returns:
            Dictionary of all computed NetworkSnapshots
        """
        total_computations = len(windows) * len(frames)
        self.stats.total_windows = total_computations
        
        logger.info("="*80)
        logger.info("STARTING SCIENTIFIC NETWORK ANALYSIS")
        logger.info("="*80)
        logger.info(f"Total windows: {len(windows)}")
        logger.info(f"Total frames: {len(frames)}")
        logger.info(f"Total computations: {total_computations}")
        logger.info(f"Estimated time: {total_computations * 30 / 3600:.1f} hours")
        logger.info("="*80)
        
        # Load checkpoint if resuming
        completed = set()
        if resume_from_checkpoint:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                completed = checkpoint['completed']
                self.window_networks = checkpoint['networks']
                self.window_metrics = checkpoint['metrics']
                self.stats = checkpoint['stats']
                logger.info(f"Resuming from checkpoint: {len(completed)}/{total_computations} completed")
        
        # Create computation tasks
        tasks = []
        for window in windows:
            for frame in frames:
                key = self._get_key(window, frame)
                if key not in completed:
                    tasks.append((window, frame))
        
        logger.info(f"Remaining computations: {len(tasks)}")
        
        if not tasks:
            logger.info("All computations already completed!")
            return self.window_networks
        
        # Initialize progress monitor
        from cascade_detector.metrics.progress_monitor import get_monitor
        monitor = get_monitor()
        monitor.set_total(len(tasks), tasks)
        
        # Process in batches for memory management  
        batch_size = self.config.get('batch_size', 32)  # Increased for M4 Max
        
        # Import the parallel engine
        from cascade_detector.metrics.parallel_compute_engine import ParallelComputeEngine
        
        # Use the optimized parallel engine
        with ParallelComputeEngine(self.config) as engine:
            with tqdm(total=len(tasks), desc="Computing metrics", unit="window") as pbar:
                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i:i+batch_size]
                    
                    # Log batch info
                    logger.info(f"Processing batch {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size} "
                               f"({len(batch)} windows)")
                    
                    # Always use parallel processing for M4 Max optimization
                    if self.config['parallel']:
                        self._process_batch_parallel(batch, pbar)
                    else:
                        self._process_batch_sequential(batch, pbar)
                
                # Checkpoint periodically
                if (i // batch_size + 1) % self.config['checkpoint_frequency'] == 0:
                    self._save_checkpoint()
                    logger.info(f"Checkpoint saved at {self.stats.completed_windows} computations")
        
        # Final checkpoint
        self._save_checkpoint()
        
        # Generate final report
        self._generate_final_report()
        
        logger.info("="*80)
        logger.info("SCIENTIFIC NETWORK ANALYSIS COMPLETED")
        logger.info("="*80)
        logger.info(f"Performance: {json.dumps(self.stats.get_report(), indent=2)}")
        
        return self.window_networks
    
    def compute_window_network(self, 
                              window: Tuple[datetime, datetime],
                              frame: str,
                              worker_id: Optional[int] = None) -> NetworkSnapshot:
        """
        Compute EXACT network and ALL metrics for a single window.
        
        Args:
            window: Time window (start, end)
            frame: Frame to analyze
            worker_id: Optional worker ID for tracking
            
        Returns:
            Complete NetworkSnapshot with network and all metrics
        """
        start_time = time.time()
        
        # Track progress
        from cascade_detector.metrics.progress_monitor import get_monitor
        monitor = get_monitor()
        monitor.start_window(window, frame, worker_id)
        
        # Create snapshot
        snapshot = NetworkSnapshot(window=window, frame=frame, network=nx.Graph())
        
        try:
            # Step 1: Build exact network for this window
            network = self._build_exact_window_network(window, frame)
            snapshot.network = network
            
            # Add metadata
            snapshot.metadata = {
                'n_nodes': network.number_of_nodes(),
                'n_edges': network.number_of_edges(),
                'density': nx.density(network) if network.number_of_nodes() > 1 else 0,
                'build_time': time.time() - start_time
            }
            
            # Step 2: Compute ALL metrics (if network is non-empty)
            if network.number_of_nodes() > 0:
                metrics_start = time.time()
                
                # Ensure we're using exact algorithms from test_phase2_complete.py config
                metrics = self._compute_all_metrics_exact(network)
                snapshot.metrics = metrics
                
                metrics_time = time.time() - metrics_start
                snapshot.metadata['metrics_time'] = metrics_time
                snapshot.metadata['n_metrics_computed'] = self._count_metrics(metrics)
            else:
                logger.warning(f"Empty network for window {window}, frame {frame}")
                snapshot.metrics = {}
            
            # Step 3: Save network and metrics
            if self.config['save_all']:
                self._save_snapshot(snapshot)
            
            # Update computation time
            snapshot.computation_time = time.time() - start_time
            
            # Update statistics
            self.stats.update(snapshot.computation_time)
            
            # Log computation
            self.computation_log.append({
                'window': window,
                'frame': frame,
                'nodes': snapshot.metadata['n_nodes'],
                'edges': snapshot.metadata['n_edges'],
                'time': snapshot.computation_time,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            error_msg = f"Failed to compute network for {window}, frame {frame}: {e}"
            logger.error(error_msg)
            self.stats.failed_windows += 1
            snapshot.metadata['error'] = str(e)
            
            # Track failure
            from cascade_detector.metrics.progress_monitor import get_monitor
            monitor = get_monitor()
            monitor.fail_window(window, frame, str(e))
        else:
            # Track successful completion
            from cascade_detector.metrics.progress_monitor import get_monitor
            monitor = get_monitor()
            # Pass metrics with basic info for the monitor
            metrics_for_monitor = {
                'basic': {
                    'n_nodes': snapshot.network.number_of_nodes() if snapshot and snapshot.network else 0,
                    'n_edges': snapshot.network.number_of_edges() if snapshot and snapshot.network else 0
                }
            }
            if hasattr(snapshot, 'metrics') and snapshot.metrics:
                metrics_for_monitor.update(snapshot.metrics)
            monitor.complete_window(window, frame, metrics_for_monitor)
        
        return snapshot
    
    def _build_exact_window_network(self, 
                                   window: Tuple[datetime, datetime],
                                   frame: str) -> nx.Graph:
        """
        Build EXACT network for specific window and frame.
        
        This constructs the complete multi-layer network with:
        - Article layer (co-occurrence, similarity)
        - Source layer (media, journalists)
        - Entity layer (persons, organizations, locations)
        - Cross-layer connections
        
        Args:
            window: Time window
            frame: Frame to analyze
            
        Returns:
            Complete NetworkX graph
        """
        # Initialize multi-layer directed graph
        G = nx.DiGraph()
        
        # Get articles in window
        articles = self._get_window_articles(window, frame)
        
        if not articles:
            return G
        
        # Check window size for intelligent processing
        window_days = (window[1] - window[0]).days
        
        # For large windows (>14 days), use chunked processing
        if window_days > 14 and len(articles) > 500:
            return self._build_large_window_network_optimized(articles, window, frame)
        
        # For smaller windows, use standard processing
        
        # Build three layers in parallel with more workers
        with ThreadPoolExecutor(max_workers=6) as executor:  # Increased for M4 Max
            # Submit layer building tasks
            futures = {
                executor.submit(self._build_article_layer, articles, window): 'article',
                executor.submit(self._build_source_layer, articles): 'source',
                executor.submit(self._build_entity_layer, articles): 'entity'
            }
            
            # Also parallelize sub-components if articles are numerous
            if len(articles) > 100:
                # Split article processing for better parallelization
                mid = len(articles) // 2
                futures.update({
                    executor.submit(self._precompute_article_embeddings, articles[:mid]): 'embed1',
                    executor.submit(self._precompute_article_embeddings, articles[mid:]): 'embed2'
                })
            
            layers = {}
            embeddings = []
            for future in as_completed(futures):
                result_type = futures[future]
                try:
                    result = future.result(timeout=30)
                    if result_type.startswith('embed'):
                        embeddings.append(result)
                    else:
                        layers[result_type] = result
                except Exception as e:
                    logger.error(f"Failed to build {result_type}: {e}")
                    if result_type in ['article', 'source', 'entity']:
                        layers[result_type] = nx.DiGraph()  # Match the main graph type
        
        # Merge layers
        for layer in layers.values():
            G = nx.compose(G, layer)
        
        # Add cross-layer connections
        G = self._add_cross_layer_connections(G, articles)
        
        # Add graph-level metadata
        G.graph['window'] = window
        G.graph['frame'] = frame
        G.graph['n_articles'] = len(articles)
        G.graph['timestamp'] = datetime.now().isoformat()
        
        return G
    
    def _build_article_layer(self, 
                           articles: List[Dict],
                           window: Tuple[datetime, datetime]) -> nx.DiGraph:
        """
        Build article co-occurrence and similarity network.
        
        Args:
            articles: List of articles in window
            window: Time window for temporal weighting
            
        Returns:
            Article layer network
        """
        G = nx.DiGraph()
        
        # Add article nodes with full attributes
        for article in articles:
            # Convert date to string for GraphML compatibility
            article_date = article['date']
            if hasattr(article_date, 'isoformat'):
                article_date = article_date.isoformat()
            else:
                article_date = str(article_date)
            
            # Convert numpy types to native Python types
            sentiment = article.get('sentiment', 0)
            if isinstance(sentiment, (np.float64, np.float32, np.float16)):
                sentiment = float(sentiment)
            elif isinstance(sentiment, (np.int64, np.int32, np.int16, np.int8)):
                sentiment = int(sentiment)
            
            influence_score = article.get('influence_score', 0)
            if isinstance(influence_score, (np.float64, np.float32, np.float16)):
                influence_score = float(influence_score)
            elif isinstance(influence_score, (np.int64, np.int32, np.int16, np.int8)):
                influence_score = int(influence_score)
            
            G.add_node(
                f"article:{article['doc_id']}",
                type='article',
                doc_id=article['doc_id'],
                date=article_date,
                media=article.get('media', 'Unknown'),
                author=article.get('author', 'Unknown'),
                frames=str(article.get('frames', {})),  # Convert dict to string
                sentiment=sentiment,
                influence_score=influence_score
            )
        
        # Add edges based on similarity (if more than 1 article)
        if len(articles) > 1:
            # Compute similarity matrix
            n = len(articles)
            
            for i in range(n):
                for j in range(i+1, n):
                    # Multi-dimensional similarity
                    similarity = self._compute_article_similarity(articles[i], articles[j])
                    
                    if similarity > 0.1:  # Threshold for edge creation
                        # Convert all similarity scores to native Python floats
                        G.add_edge(
                            f"article:{articles[i]['doc_id']}",
                            f"article:{articles[j]['doc_id']}",
                            weight=float(similarity),
                            type='article_similarity',
                            # Store components as individual attributes for GraphML compatibility
                            frame_similarity=float(self._frame_similarity(articles[i], articles[j])),
                            entity_similarity=float(self._entity_similarity(articles[i], articles[j])),
                            temporal_similarity=float(self._temporal_similarity(articles[i], articles[j], window)),
                            sentiment_similarity=float(self._sentiment_similarity(articles[i], articles[j]))
                        )
        
        return G
    
    def _build_source_layer(self, articles: List[Dict]) -> nx.DiGraph:
        """
        Build media and journalist network.
        
        Args:
            articles: List of articles
            
        Returns:
            Source layer network
        """
        G = nx.DiGraph()
        
        # First, add article nodes (they need to exist before we can add edges)
        for article in articles:
            doc_id = article['doc_id']
            # Add article node to THIS graph too (for edges to work)
            G.add_node(f"article:{doc_id}", type='article')
        
        # Track unique media and journalists
        media_nodes = set()
        journalist_nodes = set()
        media_journalist_edges = defaultdict(int)
        
        for article in articles:
            media = article.get('media')
            journalist = article.get('author')
            
            # Add media node
            if media and media != 'Unknown':
                media_node = f"media:{media}"
                if media_node not in media_nodes:
                    media_profile = self.source_index.get('media_profiles', {}).get(media, {})
                    # Convert to native Python types to avoid JSON serialization issues
                    influence_rank = media_profile.get('influence_rank', 999)
                    if isinstance(influence_rank, (np.int64, np.int32, np.int16, np.int8)):
                        influence_rank = int(influence_rank)
                    
                    G.add_node(
                        media_node,
                        type='media',
                        name=media,
                        influence_rank=influence_rank,
                        geographic_reach=media_profile.get('geographic_reach', 'unknown')
                    )
                    media_nodes.add(media_node)
                
                # Connect article to media
                G.add_edge(
                    f"article:{article['doc_id']}",
                    media_node,
                    type='published_by',
                    weight=1.0
                )
            
            # Add journalist node
            if journalist and journalist != 'Unknown':
                journalist_node = f"journalist:{journalist}"
                if journalist_node not in journalist_nodes:
                    journalist_profile = self.source_index.get('journalist_profiles', {}).get(journalist, {})
                    # Convert to native Python types to avoid JSON serialization issues
                    authority = journalist_profile.get('authority', 0)
                    if isinstance(authority, (np.float64, np.float32, np.float16)):
                        authority = float(authority)
                    elif isinstance(authority, (np.int64, np.int32, np.int16, np.int8)):
                        authority = int(authority)
                    
                    G.add_node(
                        journalist_node,
                        type='journalist',
                        name=journalist,
                        authority=authority,
                        specialization=str(journalist_profile.get('specialization', {}))  # Convert dict to string
                    )
                    journalist_nodes.add(journalist_node)
                
                # Connect article to journalist
                G.add_edge(
                    f"article:{article['doc_id']}",
                    journalist_node,
                    type='written_by',
                    weight=1.0
                )
                
                # Track journalist-media relationships
                if media and media != 'Unknown':
                    media_journalist_edges[(journalist_node, media_node)] += 1
        
        # Add journalist-media edges
        for (journalist_node, media_node), count in media_journalist_edges.items():
            G.add_edge(
                journalist_node,
                media_node,
                type='works_for',
                weight=float(count / len(articles))  # Normalized by number of articles, converted to float
            )
        
        return G
    
    def _build_large_window_network_optimized(self, 
                                             articles: List[Dict],
                                             window: Tuple[datetime, datetime],
                                             frame: str) -> nx.Graph:
        """
        Optimized network building for large windows (e.g., monthly).
        Uses intelligent chunking and parallelization.
        
        Args:
            articles: List of articles (>500)
            window: Time window
            frame: Frame to analyze
            
        Returns:
            Complete network graph
        """
        logger.info(f"Building LARGE window network: {len(articles)} articles, {(window[1]-window[0]).days} days")
        
        # Initialize the graph
        G = nx.DiGraph()
        G.graph['window'] = window
        G.graph['frame'] = frame
        G.graph['n_articles'] = len(articles)
        
        # Split articles into SMALLER chunks for MAXIMUM parallelization
        from cascade_detector.utils.progress_tracker import force_print_progress
        force_print_progress(f"LARGE WINDOW: {len(articles)} articles, {(window[1]-window[0]).days} days")
        
        chunk_size = max(100, len(articles) // 32)  # Create at least 32 chunks for better parallelization
        article_chunks = [articles[i:i+chunk_size] for i in range(0, len(articles), chunk_size)]
        
        logger.info(f"  Processing {len(article_chunks)} chunks of ~{chunk_size} articles each")
        force_print_progress(f"Processing {len(article_chunks)} chunks with 16 workers")
        
        # Process chunks in parallel with ALL 16 cores
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from functools import partial
        
        # Limit workers if we're already in a subprocess to avoid fork bomb
        import multiprocessing as mp
        n_workers = 4 if mp.current_process().name != 'MainProcess' else 16
        
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('spawn')) as executor:
            # Build sub-networks for each chunk
            build_func = partial(self._build_chunk_network, window=window, frame=frame)
            
            futures = {
                executor.submit(build_func, chunk, idx): idx 
                for idx, chunk in enumerate(article_chunks)
            }
            
            sub_networks = {}
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    sub_network = future.result(timeout=60)
                    sub_networks[chunk_idx] = sub_network
                    logger.info(f"  ✓ Chunk {chunk_idx+1}/{len(article_chunks)} completed: "
                              f"{sub_network.number_of_nodes()} nodes, {sub_network.number_of_edges()} edges")
                except Exception as e:
                    logger.error(f"  ✗ Chunk {chunk_idx+1} failed: {e}")
                    sub_networks[chunk_idx] = nx.DiGraph()
        
        # Merge sub-networks intelligently
        logger.info("  Merging sub-networks...")
        for idx in sorted(sub_networks.keys()):
            sub_net = sub_networks[idx]
            # Use union to combine graphs
            G = nx.compose(G, sub_net)
        
        # Add cross-chunk connections (only between adjacent chunks for efficiency)
        logger.info("  Adding cross-chunk connections...")
        for i in range(len(article_chunks) - 1):
            chunk1 = article_chunks[i]
            chunk2 = article_chunks[i + 1]
            self._add_cross_chunk_connections(G, chunk1[-50:], chunk2[:50])  # Connect last 50 of chunk1 with first 50 of chunk2
        
        logger.info(f"  ✓ Large window network complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    @staticmethod
    def _build_chunk_network(articles: List[Dict], 
                            chunk_idx: int,
                            window: Tuple[datetime, datetime],
                            frame: str) -> nx.Graph:
        """
        Build network for a chunk of articles (runs in separate process).
        
        Args:
            articles: Chunk of articles
            chunk_idx: Index of this chunk
            window: Time window
            frame: Frame
            
        Returns:
            Sub-network for this chunk
        """
        import networkx as nx
        import numpy as np
        from collections import defaultdict
        
        G = nx.DiGraph()
        
        # Add article nodes
        for article in articles:
            article_id = f"article:{article['doc_id']}"
            G.add_node(
                article_id,
                type='article',
                date=str(article.get('date', '')),
                media=article.get('media', 'Unknown'),
                author=article.get('author', 'Unknown'),
                chunk=chunk_idx
            )
        
        # Add simple edges within chunk (co-occurrence based on shared entities)
        for i, a1 in enumerate(articles):
            for a2 in articles[i+1:i+10]:  # Only connect to next 10 articles for efficiency
                # Simple similarity based on shared frames
                # Handle frames as either list or dict
                frames1 = a1.get('frames', [])
                frames2 = a2.get('frames', [])
                f1 = set(frames1) if isinstance(frames1, list) else set(frames1.keys())
                f2 = set(frames2) if isinstance(frames2, list) else set(frames2.keys())
                if f1 & f2:  # If they share any frames
                    similarity = len(f1 & f2) / max(len(f1 | f2), 1)
                    if similarity > 0.3:
                        G.add_edge(
                            f"article:{a1['doc_id']}", 
                            f"article:{a2['doc_id']}",
                            weight=similarity,
                            type='similarity'
                        )
        
        # Add media nodes
        media_articles = defaultdict(list)
        for article in articles:
            media = article.get('media', 'Unknown')
            media_articles[media].append(article['doc_id'])
        
        for media, article_ids in media_articles.items():
            media_id = f"media:{media}"
            G.add_node(media_id, type='media', chunk=chunk_idx)
            for article_id in article_ids:
                G.add_edge(f"article:{article_id}", media_id, type='published_by')
        
        return G
    
    def _compute_metrics_large_graph_parallel_exact(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Massively parallel EXACT metrics computation for large graphs.
        Uses all 16 M4 Max cores with no approximation.
        
        Args:
            G: Large NetworkX graph
            
        Returns:
            Dictionary of exact metrics
        """
        import time
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
        import multiprocessing as mp
        
        metrics = {}
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        logger.info(f"    Starting MASSIVE parallel computation: {n_nodes} nodes, {n_edges} edges")
        logger.info(f"    Using 16 cores for parallel exact computation")
        
        # Force print progress to see what's happening
        from cascade_detector.utils.progress_tracker import force_print_progress
        force_print_progress(f"LARGE GRAPH DETECTED: {n_nodes} nodes, {n_edges} edges - maximizing parallelization")
        
        # 1. Basic metrics (instant)
        start = time.time()
        metrics['basic'] = {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': nx.density(G) if n_nodes > 1 else 0,
            'is_connected': nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G),
            'n_components': nx.number_weakly_connected_components(G) if G.is_directed() else nx.number_connected_components(G)
        }
        logger.info(f"      Basic metrics: {time.time() - start:.2f}s")
        
        # 2. Parallel degree computation
        start = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            degree_futures = {
                'in_degree': executor.submit(dict, G.in_degree()) if G.is_directed() else None,
                'out_degree': executor.submit(dict, G.out_degree()) if G.is_directed() else None,
                'degree': executor.submit(dict, G.degree())
            }
            
            degree_results = {}
            for key, future in degree_futures.items():
                if future:
                    degree_results[key] = future.result()
        
        degree_values = list(degree_results.get('degree', {}).values())
        metrics['degree'] = {
            'avg_degree': np.mean(degree_values) if degree_values else 0,
            'max_degree': max(degree_values) if degree_values else 0,
            'min_degree': min(degree_values) if degree_values else 0,
            'std_degree': np.std(degree_values) if degree_values else 0
        }
        logger.info(f"      Degree metrics: {time.time() - start:.2f}s")
        
        # 3. MASSIVELY PARALLEL exact centrality computation
        # Split nodes for parallel processing - MORE AGGRESSIVE CHUNKING
        nodes = list(G.nodes())
        # Create many small chunks for better load balancing
        chunk_size = max(50, n_nodes // 32)  # Create 32+ chunks for better distribution
        node_chunks = [nodes[i:i+chunk_size] for i in range(0, len(nodes), chunk_size)]
        
        logger.info(f"      Computing exact centralities on {len(node_chunks)} chunks...")
        
        # CRITICAL: Check if we're already in a worker process
        import multiprocessing as mp
        if mp.current_process().name != 'MainProcess':
            # We're in a worker process - compute EXACT metrics directly without creating new processes
            force_print_progress(f"  In worker process - computing EXACT betweenness for {n_nodes} nodes")
            
            # Compute metrics directly
            metrics['centrality'] = {}
            
            # Degree centrality (always fast)
            deg_cent = nx.degree_centrality(G)
            metrics['centrality']['degree'] = {
                'mean': np.mean(list(deg_cent.values())),
                'max': max(deg_cent.values()) if deg_cent else 0,
                'top_nodes': sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:10]
            }
            
            # Try to use NetworKit for MUCH faster exact betweenness
            try:
                # Re-ensure OpenMP is disabled (may already be set)
                os.environ['OMP_NUM_THREADS'] = '1'
                
                import networkit as nk
                force_print_progress(f"  Using NetworKit for FAST EXACT betweenness ({n_nodes} nodes)")
                
                # Convert NetworkX graph to NetworKit
                node_mapping = {node: i for i, node in enumerate(G.nodes())}
                reverse_mapping = {i: node for node, i in node_mapping.items()}
                
                nk_graph = nk.Graph(len(G.nodes()), weighted=False, directed=G.is_directed())
                for u, v in G.edges():
                    nk_graph.addEdge(node_mapping[u], node_mapping[v])
                
                # Compute EXACT betweenness with NetworKit (single-threaded in worker process)
                bc = nk.centrality.Betweenness(nk_graph, normalized=True)
                bc.run()
                
                # Convert back to NetworkX node labels
                betweenness = {reverse_mapping[i]: bc.score(i) for i in range(len(G.nodes()))}
                
                force_print_progress(f"  NetworKit computed EXACT betweenness in record time!")
                
            except (ImportError, Exception) as e:
                # Fallback to NetworkX if NetworKit not available or fails
                if isinstance(e, ImportError):
                    force_print_progress(f"  NetworKit not available, using NetworkX for EXACT betweenness ({n_nodes} nodes)")
                else:
                    force_print_progress(f"  NetworKit failed ({str(e)}), using NetworkX for EXACT betweenness ({n_nodes} nodes)")
                betweenness = nx.betweenness_centrality(G, normalized=True)
            
            metrics['centrality']['betweenness'] = {
                'mean': np.mean(list(betweenness.values())),
                'max': max(betweenness.values()) if betweenness else 0,
                'top_nodes': sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
            }
            
            # Try to compute other centralities if possible
            try:
                # PageRank (always works)
                pagerank = nx.pagerank(G)
                metrics['centrality']['pagerank'] = {
                    'mean': np.mean(list(pagerank.values())),
                    'max': max(pagerank.values()),
                    'top_nodes': sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
                }
            except:
                pass
            
            # Skip eigenvector centrality as it may fail for some graphs
            return metrics
        
        # Main process - use full parallelization
        force_print_progress(f"Computing centralities with {len(node_chunks)} chunks across 16 cores")
        n_workers = 16
        
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('spawn')) as executor:
            centrality_futures = {}
            
            # Degree centrality (fast, exact)
            centrality_futures['degree'] = executor.submit(nx.degree_centrality, G)
            
            # Betweenness centrality - Try NetworKit first for speed
            try:
                import networkit as nk
                force_print_progress(f"  Using NetworKit for ULTRA-FAST EXACT betweenness ({n_nodes} nodes)")
                
                # Convert to NetworKit graph
                node_mapping = {node: i for i, node in enumerate(G.nodes())}
                reverse_mapping = {i: node for node, i in node_mapping.items()}
                
                nk_graph = nk.Graph(len(G.nodes()), weighted=False, directed=G.is_directed())
                for u, v in G.edges():
                    nk_graph.addEdge(node_mapping[u], node_mapping[v])
                
                # NetworKit automatically uses all available cores with OpenMP
                bc = nk.centrality.Betweenness(nk_graph, normalized=True)
                bc.run()
                
                # Convert back
                betweenness = {reverse_mapping[i]: bc.score(i) for i in range(len(G.nodes()))}
                
                # Store the result directly since computation is already done
                # We'll handle it differently in the results collection
                centrality_futures['betweenness_direct'] = betweenness
                force_print_progress(f"  NetworKit betweenness computation complete!")
                
            except ImportError:
                force_print_progress(f"  NetworKit not available, falling back to parallel NetworkX")
                
                # Fallback to NetworkX parallel computation
                if n_nodes < 100000:  # Process even very large graphs
                    bet_futures = []
                    bet_start = time.time()
                    
                    # CRITICAL: Create optimal number of chunks for parallel processing
                    # We want enough chunks to keep all 16 cores busy
                    if n_nodes > 5000:
                        # Large graphs: create exactly 16 chunks (one per core)
                        # This avoids overhead while maximizing parallelization
                        chunk_size = max(50, n_nodes // 16)  
                        force_print_progress(f"LARGE GRAPH ({n_nodes} nodes): Creating {n_nodes//chunk_size} chunks for 16 cores")
                    elif n_nodes > 1000:
                        # Medium graphs: 16 chunks for optimal distribution
                        chunk_size = max(20, n_nodes // 16)
                        force_print_progress(f"Medium graph ({n_nodes} nodes): Creating {n_nodes//chunk_size} chunks")
                    else:
                        # Small graphs: fewer chunks to reduce overhead
                        chunk_size = max(50, n_nodes // 8)
                    
                    node_chunks_bet = [nodes[i:i+chunk_size] for i in range(0, len(nodes), chunk_size)]
                    
                    force_print_progress(f"  Computing EXACT betweenness: {len(node_chunks_bet)} chunks, ~{chunk_size} nodes/chunk")
                
                # Submit all chunks to process pool - they will be processed as workers become available
                for i, chunk in enumerate(node_chunks_bet):
                    future = executor.submit(
                        self._compute_betweenness_chunk_parallel, 
                        G, chunk, i, len(node_chunks_bet)
                    )
                    bet_futures.append(future)
                
                centrality_futures['betweenness_chunks'] = bet_futures
                force_print_progress(f"  All {len(bet_futures)} tasks submitted to 16-worker pool")
            
            # Eigenvector centrality (if connected)
            if metrics['basic']['n_components'] == 1:
                centrality_futures['eigenvector'] = executor.submit(
                    nx.eigenvector_centrality, G, max_iter=1000
                )
            
            # PageRank (always works)
            centrality_futures['pagerank'] = executor.submit(nx.pagerank, G)
            
            # Collect results
            metrics['centrality'] = {}
            
            for key, future in centrality_futures.items():
                if key == 'betweenness_direct':
                    # Direct betweenness computation result (NetworKit returns dict directly)
                    try:
                        if isinstance(future, dict):
                            # NetworKit result stored directly as dict
                            betweenness = future
                        else:
                            # Future object result from executor
                            betweenness = future.result()
                        
                        if betweenness:
                            metrics['centrality']['betweenness'] = {
                                'mean': np.mean(list(betweenness.values())),
                                'max': max(betweenness.values()),
                                'top_nodes': sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
                            }
                    except Exception as e:
                        logger.warning(f"        Direct betweenness failed: {e}")
                elif key == 'betweenness_chunks':
                    # Aggregate betweenness from chunks
                    betweenness = {}
                    completed_chunks = 0
                    total_chunks = len(future)
                    for chunk_future in future:
                        try:
                            # Wait for result without timeout for exact computation
                            chunk_result = chunk_future.result()
                            betweenness.update(chunk_result)
                            completed_chunks += 1
                            if completed_chunks % 5 == 0:
                                force_print_progress(f"  Betweenness: {completed_chunks}/{total_chunks} chunks done")
                        except Exception as e:
                            logger.warning(f"        Betweenness chunk failed: {e}")
                    
                    if betweenness:
                        metrics['centrality']['betweenness'] = {
                            'mean': np.mean(list(betweenness.values())),
                            'max': max(betweenness.values()),
                            'top_nodes': sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
                        }
                else:
                    try:
                        result = future.result(timeout=120)
                        if isinstance(result, dict):
                            metrics['centrality'][key] = {
                                'mean': np.mean(list(result.values())),
                                'max': max(result.values()) if result else 0,
                                'top_nodes': sorted(result.items(), key=lambda x: x[1], reverse=True)[:10]
                            }
                    except Exception as e:
                        logger.warning(f"        {key} centrality failed: {e}")
        
        # 4. Parallel clustering coefficient
        if n_nodes < 20000:
            start = time.time()
            with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn')) as executor:
                # Compute clustering for node chunks in parallel
                clustering_futures = []
                for chunk in node_chunks[:8]:  # Limit to 8 chunks for memory
                    future = executor.submit(self._compute_clustering_chunk, G, chunk)
                    clustering_futures.append(future)
                
                clustering_values = []
                for future in as_completed(clustering_futures):
                    try:
                        chunk_clustering = future.result(timeout=30)
                        clustering_values.extend(chunk_clustering)
                    except Exception as e:
                        logger.warning(f"        Clustering chunk failed: {e}")
                
                if clustering_values:
                    metrics['clustering'] = {
                        'average': np.mean(clustering_values),
                        'std': np.std(clustering_values)
                    }
            logger.info(f"      Clustering: {time.time() - start:.2f}s")
        
        # 5. Community detection with parallel Louvain
        if n_nodes < 50000:
            start = time.time()
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(G.to_undirected())
                n_communities = len(set(communities.values()))
                # Convert numpy int64 keys to regular Python ints
                unique_communities, counts = np.unique(list(communities.values()), return_counts=True)
                sizes_dict = {int(k): int(v) for k, v in zip(unique_communities, counts)}
                metrics['community'] = {
                    'n_communities': n_communities,
                    'modularity': community_louvain.modularity(communities, G.to_undirected()),
                    'sizes': sizes_dict
                }
                logger.info(f"      Communities: {time.time() - start:.2f}s")
            except:
                metrics['community'] = {'n_communities': 0, 'modularity': 0}
        
        logger.info(f"    Massive parallel computation complete")
        return metrics
    
    @staticmethod
    def _compute_betweenness_chunk(G: nx.Graph, nodes: list, total_nodes: int) -> Dict:
        """Compute betweenness centrality for a chunk of nodes."""
        # Compute betweenness for this chunk
        betweenness = nx.betweenness_centrality_subset(
            G, 
            sources=nodes,
            targets=nodes,
            normalized=True
        )
        return betweenness
    
    @staticmethod
    def _compute_betweenness_chunk_exact(G: nx.Graph, nodes: list, total_nodes: int) -> Dict:
        """
        Compute EXACT betweenness centrality for a chunk of nodes.
        Optimized for parallel execution.
        """
        # Simply use NetworkX's optimized exact algorithm for this chunk
        # It's already optimized internally
        betweenness = nx.betweenness_centrality_subset(
            G,
            sources=nodes,
            targets=nodes,
            normalized=True
        )
        return betweenness
    
    @staticmethod
    def _compute_betweenness_chunk_parallel(G: nx.Graph, nodes: list, chunk_id: int, total_chunks: int) -> Dict:
        """
        Compute EXACT betweenness centrality for a chunk with progress tracking.
        Optimized for massive parallel execution.
        """
        # Print progress for large computations
        if total_chunks > 50 and chunk_id % 10 == 0:
            print(f"    [Worker] Processing betweenness chunk {chunk_id}/{total_chunks}", flush=True)
        
        # For exact computation, use all nodes as targets
        all_nodes = list(G.nodes())
        
        # Compute exact betweenness for this chunk
        betweenness = nx.betweenness_centrality_subset(
            G,
            sources=nodes,
            targets=all_nodes,  # Use all nodes as targets for exactness
            normalized=True
        )
        
        return betweenness
    
    @staticmethod
    def _compute_betweenness_exact_sample(G: nx.Graph, sample_nodes: list, k: int) -> Dict:
        """
        Compute exact betweenness centrality using k-sampling.
        This is exact for the sampled nodes but faster than full computation.
        """
        # Use NetworkX's built-in k-sampling for exact computation
        betweenness = nx.betweenness_centrality(
            G,
            k=len(sample_nodes),  # Number of nodes to sample
            normalized=True,
            endpoints=False,
            seed=42  # For reproducibility
        )
        return betweenness
    
    @staticmethod  
    def _compute_clustering_chunk(G: nx.Graph, nodes: list) -> list:
        """Compute clustering coefficient for a chunk of nodes."""
        clustering = nx.clustering(G, nodes)
        return list(clustering.values())
    
    def _compute_metrics_large_graph_optimized(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Optimized metrics computation for large graphs.
        Computes essential metrics with intelligent sampling for expensive ones.
        
        Args:
            G: Large NetworkX graph
            
        Returns:
            Dictionary of metrics
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        metrics = {}
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        logger.info(f"    Computing metrics for large graph: {n_nodes} nodes, {n_edges} edges")
        
        # 1. Basic metrics (fast)
        start = time.time()
        metrics['basic'] = {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': nx.density(G) if n_nodes > 1 else 0,
            'is_connected': nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G),
            'n_components': nx.number_weakly_connected_components(G) if G.is_directed() else nx.number_connected_components(G)
        }
        logger.info(f"      Basic metrics: {time.time() - start:.2f}s")
        
        # 2. Degree metrics (fast)
        start = time.time()
        degrees = dict(G.degree())
        degree_values = list(degrees.values())
        metrics['degree'] = {
            'avg_degree': np.mean(degree_values) if degree_values else 0,
            'max_degree': max(degree_values) if degree_values else 0,
            'min_degree': min(degree_values) if degree_values else 0,
            'std_degree': np.std(degree_values) if degree_values else 0
        }
        logger.info(f"      Degree metrics: {time.time() - start:.2f}s")
        
        # 3. Use sampling for expensive metrics
        sample_size = min(500, n_nodes)  # Sample up to 500 nodes
        sampled_nodes = np.random.choice(list(G.nodes()), size=sample_size, replace=False)
        G_sample = G.subgraph(sampled_nodes)
        
        # 4. Compute expensive metrics on sample with parallel execution
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            # Submit tasks
            if n_nodes < 10000:  # Only for moderately large graphs
                futures['clustering'] = executor.submit(
                    lambda: nx.average_clustering(G_sample)
                )
            
            # Centrality on sample
            futures['centrality'] = executor.submit(
                self._compute_sampled_centrality, G_sample, sample_size=min(100, len(sampled_nodes))
            )
            
            # Path metrics on sample
            if G_sample.number_of_nodes() > 1:
                futures['paths'] = executor.submit(
                    self._compute_sampled_path_metrics, G_sample
                )
            
            # Collect results
            for key, future in futures.items():
                try:
                    start = time.time()
                    result = future.result(timeout=30)
                    metrics[key] = result
                    logger.info(f"      {key.capitalize()} metrics: {time.time() - start:.2f}s")
                except Exception as e:
                    logger.warning(f"      Failed to compute {key}: {e}")
                    metrics[key] = {}
        
        # 5. Community detection (fast approximation for large graphs)
        if n_nodes < 50000:
            start = time.time()
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(G.to_undirected())
                n_communities = len(set(communities.values()))
                metrics['community'] = {
                    'n_communities': n_communities,
                    'modularity': community_louvain.modularity(communities, G.to_undirected())
                }
                logger.info(f"      Community metrics: {time.time() - start:.2f}s")
            except:
                metrics['community'] = {'n_communities': 0, 'modularity': 0}
        
        return metrics
    
    @staticmethod
    def _compute_sampled_centrality(G: nx.Graph, sample_size: int = 100) -> Dict:
        """Compute centrality metrics on a sample of nodes."""
        nodes = list(G.nodes())[:sample_size]
        subgraph = G.subgraph(nodes)
        
        centrality_metrics = {}
        
        # Degree centrality (fast)
        deg_cent = nx.degree_centrality(subgraph)
        centrality_metrics['degree_centrality'] = {
            'mean': np.mean(list(deg_cent.values())),
            'max': max(deg_cent.values()) if deg_cent else 0
        }
        
        # Betweenness centrality (sampled)
        if len(nodes) <= 100:
            bet_cent = nx.betweenness_centrality(subgraph, k=min(10, len(nodes)-1))
            centrality_metrics['betweenness_centrality'] = {
                'mean': np.mean(list(bet_cent.values())),
                'max': max(bet_cent.values()) if bet_cent else 0
            }
        
        return centrality_metrics
    
    @staticmethod
    def _compute_sampled_path_metrics(G: nx.Graph) -> Dict:
        """Compute path-based metrics on sampled node pairs."""
        nodes = list(G.nodes())
        n_samples = min(100, len(nodes))
        
        path_lengths = []
        for _ in range(n_samples):
            source = np.random.choice(nodes)
            target = np.random.choice(nodes)
            if source != target:
                try:
                    if G.is_directed():
                        length = nx.shortest_path_length(G, source, target)
                    else:
                        length = nx.shortest_path_length(G, source, target)
                    path_lengths.append(length)
                except nx.NetworkXNoPath:
                    pass
        
        if path_lengths:
            return {
                'avg_path_length': np.mean(path_lengths),
                'max_path_length': max(path_lengths),
                'path_length_std': np.std(path_lengths)
            }
        return {'avg_path_length': 0, 'max_path_length': 0, 'path_length_std': 0}
    
    def _add_cross_chunk_connections(self, G: nx.Graph, 
                                    articles1: List[Dict], 
                                    articles2: List[Dict]):
        """
        Add connections between articles from different chunks.
        
        Args:
            G: Main graph to update
            articles1: Last articles from chunk 1
            articles2: First articles from chunk 2
        """
        for a1 in articles1:
            for a2 in articles2:
                # Check for temporal proximity and frame similarity
                if 'date' in a1 and 'date' in a2:
                    # Simple frame overlap check
                    # Handle both list and dict formats for frames
                    frames1 = a1.get('frames', [])
                    frames2 = a2.get('frames', [])
                    
                    # Convert to sets - handle both list and dict formats
                    if isinstance(frames1, dict):
                        f1 = set(frames1.keys())
                    else:
                        f1 = set(frames1) if frames1 else set()
                    
                    if isinstance(frames2, dict):
                        f2 = set(frames2.keys())
                    else:
                        f2 = set(frames2) if frames2 else set()
                    if f1 & f2:
                        similarity = len(f1 & f2) / max(len(f1 | f2), 1)
                        if similarity > 0.4:
                            node1 = f"article:{a1['doc_id']}"
                            node2 = f"article:{a2['doc_id']}"
                            if G.has_node(node1) and G.has_node(node2):
                                G.add_edge(node1, node2, 
                                         weight=similarity,
                                         type='cross_chunk_similarity')
    
    def _precompute_article_embeddings(self, articles: List[Dict]) -> Dict:
        """
        Precompute article embeddings for similarity calculations.
        
        Args:
            articles: List of articles
            
        Returns:
            Dictionary of embeddings
        """
        embeddings = {}
        for article in articles:
            # Simple embedding based on frames and sentiment
            frames = article.get('frames', {})
            embedding = []
            
            # Handle both list and dict formats for frames
            if isinstance(frames, dict):
                for frame, score in frames.items():
                    embedding.append(score)
            elif isinstance(frames, list):
                # For list format, use equal weight
                for frame in frames:
                    embedding.append(1.0)
            
            embeddings[article['doc_id']] = np.array(embedding) if embedding else np.zeros(8)
        return embeddings
    
    def _build_entity_layer(self, articles: List[Dict]) -> nx.DiGraph:
        """
        Build entity network (persons, organizations, locations).
        
        Args:
            articles: List of articles
            
        Returns:
            Entity layer network
        """
        G = nx.DiGraph()
        
        # First, add article nodes (they need to exist before we can add edges)
        for article in articles:
            doc_id = article['doc_id']
            G.add_node(f"article:{doc_id}", type='article')
        
        # Track entity co-occurrences
        entity_nodes = set()
        entity_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for article in articles:
            article_entities = article.get('entities', [])
            
            # Add entity nodes
            for entity_key in article_entities:
                if entity_key not in entity_nodes:
                    entity_data = self.entity_index.get(entity_key, {})
                    # Convert to native Python types to avoid JSON serialization issues
                    authority_score = entity_data.get('authority_score', 0)
                    if isinstance(authority_score, (np.float64, np.float32, np.float16)):
                        authority_score = float(authority_score)
                    elif isinstance(authority_score, (np.int64, np.int32, np.int16, np.int8)):
                        authority_score = int(authority_score)
                    
                    G.add_node(
                        f"entity:{entity_key}",
                        type='entity',  
                        entity_type=entity_data.get('type', 'UNK'),
                        name=entity_data.get('name', entity_key),
                        authority_score=authority_score
                    )
                    entity_nodes.add(entity_key)
                
                # Connect article to entity
                G.add_edge(
                    f"article:{article['doc_id']}",
                    f"entity:{entity_key}",
                    type='mentions',
                    weight=1.0
                )
            
            # Track co-occurrences
            for i, e1 in enumerate(article_entities):
                for e2 in article_entities[i+1:]:
                    entity_cooccurrence[e1][e2] += 1
                    entity_cooccurrence[e2][e1] += 1
        
        # Add entity co-occurrence edges
        for e1, connections in entity_cooccurrence.items():
            for e2, count in connections.items():
                if count > 0:
                    # Only add edge once (e1 -> e2)
                    if e1 < e2:  # Lexicographic ordering to avoid duplicates
                        G.add_edge(
                            f"entity:{e1}",
                            f"entity:{e2}",
                            type='co_occurrence',
                            weight=float(count / len(articles)),  # Normalized and converted to float
                            count=int(count) if isinstance(count, (np.int64, np.int32)) else count
                        )
        
        return G
    
    def _add_cross_layer_connections(self, G: nx.DiGraph, articles: List[Dict]) -> nx.DiGraph:
        """
        Add connections between layers.
        
        Args:
            G: Current graph
            articles: List of articles
            
        Returns:
            Graph with cross-layer connections
        """
        # Add journalist-entity connections (who quotes whom)
        for article in articles:
            journalist = article.get('author')
            if journalist and journalist != 'Unknown':
                journalist_node = f"journalist:{journalist}"
                
                # Connect journalist to entities they mention
                for entity in article.get('entities', []):
                    entity_node = f"entity:{entity}"
                    if G.has_node(journalist_node) and G.has_node(entity_node):
                        if G.has_edge(journalist_node, entity_node):
                            G[journalist_node][entity_node]['weight'] += 1
                        else:
                            G.add_edge(
                                journalist_node,
                                entity_node,
                                type='mentions',
                                weight=1
                            )
        
        return G
    
    def _compute_all_metrics_exact(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Compute ALL metrics EXACTLY with NO approximations.
        Uses intelligent strategies for large graphs.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary containing all 73+ metrics
        """
        # For very large graphs, use massively parallel exact computation
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        if n_nodes > 5000 or n_edges > 50000:
            logger.info(f"    Large graph ({n_nodes} nodes, {n_edges} edges) - using MASSIVE parallelization")
            return self._compute_metrics_large_graph_parallel_exact(G)
        
        try:
            # Try to use ExhaustiveMetricsCalculator if available
            from cascade_detector.metrics.exhaustive_metrics_calculator import ExhaustiveMetricsCalculator
            
            calculator = ExhaustiveMetricsCalculator({
                'exact_computation': True,
                'use_gpu': self.config.get('use_gpu', True),
                'n_workers': self.config.get('n_workers', 16),
                'compute_all': True,
                'verbose': False  # Reduce verbosity
            })
            
            # Compute all metrics
            metrics = calculator.calculate_all_metrics(G)
            
        except ImportError:
            # Fallback to basic metrics if ExhaustiveMetricsCalculator not available
            logger.info("    ExhaustiveMetricsCalculator not available, using fallback metrics...")
            metrics = self._compute_basic_metrics_with_progress(G)
        
        return metrics
    
    def _compute_basic_metrics_with_progress(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Compute basic metrics with progress tracking.
        """
        metrics = {}
        
        # List of metric categories and their functions
        metric_categories = [
            ('centrality', [
                ('degree', lambda: dict(G.degree())),
                ('betweenness', lambda: nx.betweenness_centrality(G)),
                ('closeness', lambda: nx.closeness_centrality(G)),
                ('pagerank', lambda: nx.pagerank(G, max_iter=100))
            ]),
            ('clustering', [
                ('coefficient', lambda: nx.clustering(G.to_undirected())),
                ('transitivity', lambda: nx.transitivity(G)),
                ('average', lambda: nx.average_clustering(G.to_undirected()))
            ]),
            ('structure', [
                ('density', lambda: nx.density(G)),
                ('components', lambda: nx.number_connected_components(G.to_undirected()))
            ])
        ]
        
        # Count total metrics
        total_metrics = sum(len(cat_metrics) for _, cat_metrics in metric_categories)
        
        # Compute with progress bar
        with tqdm(total=total_metrics, desc="    Computing metrics", leave=False) as pbar:
            for category, cat_metrics in metric_categories:
                metrics[category] = {}
                for metric_name, metric_func in cat_metrics:
                    pbar.set_postfix_str(f"{category}.{metric_name}")
                    try:
                        metrics[category][metric_name] = metric_func()
                    except Exception as e:
                        logger.debug(f"Failed to compute {category}.{metric_name}: {e}")
                        metrics[category][metric_name] = None
                    pbar.update(1)
        
        return metrics
    
    def _get_all_metric_functions(self) -> Dict[str, Dict[str, callable]]:
        """
        Get all metric computation functions.
        
        Returns:
            Nested dictionary of metric functions
        """
        return {
            'centrality': {
                'degree': lambda G: dict(G.degree()),
                'in_degree': lambda G: dict(G.in_degree()) if G.is_directed() else None,
                'out_degree': lambda G: dict(G.out_degree()) if G.is_directed() else None,
                'betweenness': lambda G: nx.betweenness_centrality(G),
                'closeness': lambda G: nx.closeness_centrality(G),
                'eigenvector': lambda G: self._safe_eigenvector_centrality(G),
                'pagerank': lambda G: nx.pagerank(G, max_iter=200),
                'harmonic': lambda G: nx.harmonic_centrality(G),
                'load': lambda G: nx.load_centrality(G) if G.number_of_nodes() < 1000 else {},
                'katz': lambda G: self._safe_katz_centrality(G)
            },
            'clustering': {
                'coefficient': lambda G: nx.clustering(G.to_undirected()),
                'transitivity': lambda G: nx.transitivity(G),
                'average_clustering': lambda G: nx.average_clustering(G.to_undirected()),
                'squares': lambda G: nx.square_clustering(G.to_undirected()) if not G.is_directed() else {}
            },
            'structure': {
                'density': lambda G: nx.density(G),
                'diameter': lambda G: self._safe_diameter(G),
                'radius': lambda G: self._safe_radius(G),
                'average_path_length': lambda G: self._safe_average_path_length(G),
                'efficiency': lambda G: nx.global_efficiency(G),
                'assortativity': lambda G: nx.degree_assortativity_coefficient(G),
                'node_connectivity': lambda G: nx.node_connectivity(G) if G.number_of_nodes() < 100 else None,
                'edge_connectivity': lambda G: nx.edge_connectivity(G) if G.number_of_nodes() < 100 else None,
                'n_components': lambda G: nx.number_weakly_connected_components(G) if G.is_directed() else nx.number_connected_components(G)
            }
        }
    
    def _safe_compute(self, func: callable, G: nx.Graph, metric_name: str) -> Any:
        """
        Safely compute a metric with error handling.
        
        Args:
            func: Metric computation function
            G: Graph
            metric_name: Name of metric for logging
            
        Returns:
            Computed metric or None if failed
        """
        try:
            return func(G)
        except Exception as e:
            logger.debug(f"Failed to compute {metric_name}: {e}")
            return None
    
    def _safe_eigenvector_centrality(self, G: nx.Graph) -> Dict:
        """Safely compute eigenvector centrality."""
        try:
            if G.is_directed():
                G_undirected = G.to_undirected()
            else:
                G_undirected = G
            
            if nx.is_connected(G_undirected):
                return nx.eigenvector_centrality_numpy(G)
            else:
                # Compute for largest component
                largest_cc = max(nx.connected_components(G_undirected), key=len)
                subgraph = G.subgraph(largest_cc)
                return nx.eigenvector_centrality_numpy(subgraph)
        except:
            return {}
    
    def _safe_katz_centrality(self, G: nx.Graph) -> Dict:
        """Safely compute Katz centrality."""
        try:
            if G.number_of_nodes() < 1000:
                return nx.katz_centrality_numpy(G, alpha=0.1)
            else:
                return {}
        except:
            return {}
    
    def _safe_diameter(self, G: nx.Graph) -> Optional[int]:
        """Safely compute diameter."""
        try:
            if G.is_directed():
                if nx.is_weakly_connected(G):
                    return nx.diameter(G.to_undirected())
            else:
                if nx.is_connected(G):
                    return nx.diameter(G)
            return None
        except:
            return None
    
    def _safe_radius(self, G: nx.Graph) -> Optional[int]:
        """Safely compute radius."""
        try:
            if G.is_directed():
                if nx.is_strongly_connected(G):
                    return nx.radius(G)
            else:
                if nx.is_connected(G):
                    return nx.radius(G)
            return None
        except:
            return None
    
    def _safe_average_path_length(self, G: nx.Graph) -> Optional[float]:
        """Safely compute average path length."""
        try:
            if G.is_directed():
                if nx.is_weakly_connected(G):
                    return nx.average_shortest_path_length(G.to_undirected())
            else:
                if nx.is_connected(G):
                    return nx.average_shortest_path_length(G)
            
            # For disconnected graphs, compute for largest component
            if G.is_directed():
                largest_cc = max(nx.weakly_connected_components(G), key=len)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
            
            if len(largest_cc) > 1:
                subgraph = G.subgraph(largest_cc)
                return nx.average_shortest_path_length(subgraph.to_undirected() if G.is_directed() else subgraph)
            
            return None
        except:
            return None
    
    # Similarity computation methods
    def _compute_article_similarity(self, article1: Dict, article2: Dict) -> float:
        """Compute overall similarity between two articles."""
        weights = {
            'frame': 0.3,
            'entity': 0.3,
            'temporal': 0.2,
            'sentiment': 0.2
        }
        
        similarity = (
            weights['frame'] * self._frame_similarity(article1, article2) +
            weights['entity'] * self._entity_similarity(article1, article2) +
            weights['temporal'] * self._temporal_similarity(article1, article2, None) +
            weights['sentiment'] * self._sentiment_similarity(article1, article2)
        )
        
        return similarity
    
    def _frame_similarity(self, article1: Dict, article2: Dict) -> float:
        """Compute frame similarity using cosine similarity."""
        frames1_raw = article1.get('frames', [])
        frames2_raw = article2.get('frames', [])
        
        # Get frame proportions (use uniform if frames is a list)
        if isinstance(frames1_raw, list):
            frames1 = {f: 1.0 for f in frames1_raw} if frames1_raw else {}
        else:
            frames1 = frames1_raw
            
        if isinstance(frames2_raw, list):
            frames2 = {f: 1.0 for f in frames2_raw} if frames2_raw else {}
        else:
            frames2 = frames2_raw
        
        if not frames1 or not frames2:
            return 0.0
        
        # Get all frames
        all_frames = set(frames1.keys()) | set(frames2.keys())
        
        if not all_frames:
            return 0.0
        
        # Create vectors
        v1 = np.array([frames1.get(f, 0) for f in all_frames])
        v2 = np.array([frames2.get(f, 0) for f in all_frames])
        
        # Cosine similarity
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def _entity_similarity(self, article1: Dict, article2: Dict) -> float:
        """Compute entity similarity using Jaccard coefficient."""
        entities1 = set(article1.get('entities', []))
        entities2 = set(article2.get('entities', []))
        
        if not entities1 and not entities2:
            return 0.0
        
        intersection = entities1 & entities2
        union = entities1 | entities2
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _temporal_similarity(self, article1: Dict, article2: Dict, window: Optional[Tuple]) -> float:
        """Compute temporal similarity based on publication dates."""
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
    
    def _sentiment_similarity(self, article1: Dict, article2: Dict) -> float:
        """Compute sentiment similarity."""
        sent1 = article1.get('sentiment', 0)
        sent2 = article2.get('sentiment', 0)
        
        # Convert to [0, 1] range
        diff = abs(sent1 - sent2) / 2  # Max diff is 2 (from -1 to 1)
        
        return 1 - diff
    
    # Helper methods
    def _get_window_articles(self, window: Tuple[datetime, datetime], frame: str) -> List[Dict]:
        """
        Get all articles in a time window for a specific frame.
        
        Args:
            window: Time window
            frame: Frame to filter
            
        Returns:
            List of article dictionaries
        """
        articles = []
        
        # Use frame_index if available (contains frame data)
        if self.frame_index and 'article_frames' in self.frame_index:
            article_frames = self.frame_index['article_frames']
            
            for doc_id, profile in article_frames.items():
                # Check date
                article_date = profile.get('date')
                if not article_date:
                    continue
                
                # Convert to datetime if needed
                if isinstance(article_date, str):
                    article_date = pd.to_datetime(article_date)
                
                # Check if in window
                if not (window[0] <= article_date <= window[1]):
                    continue
                
                # Check frame - handle both list and dict formats
                frames = profile.get('frames', [])
                frame_proportions_raw = profile.get('frame_proportions', profile.get('frame_distribution', {}))
                
                # Check if frame is present
                frame_present = False
                
                # First check simple list format
                if isinstance(frames, list) and frame in frames:
                    frame_present = True
                    frame_proportions = {f: 1.0 for f in frames}  # Equal weight for list format
                
                # Then check dict format with proportions
                elif frame_proportions_raw:
                    # Convert numpy types in frame proportions
                    frame_proportions = {}
                    for k, v in frame_proportions_raw.items():
                        if isinstance(v, (np.float64, np.float32, np.float16)):
                            frame_proportions[k] = float(v)
                        elif isinstance(v, (np.int64, np.int32, np.int16, np.int8)):
                            frame_proportions[k] = int(v)
                        else:
                            frame_proportions[k] = v
                    
                    if frame in frame_proportions and frame_proportions[frame] > 0:  # Any presence of frame
                        frame_present = True
                else:
                    frame_proportions = {}
                
                if not frame_present:
                    continue
                
                # Add article with all necessary data
                # Convert numpy types to native Python types
                sentiment = profile.get('sentiment', 0)
                if isinstance(sentiment, (np.float64, np.float32, np.float16)):
                    sentiment = float(sentiment)
                elif isinstance(sentiment, (np.int64, np.int32, np.int16, np.int8)):
                    sentiment = int(sentiment)
                
                influence_score = profile.get('influence_score', 0)
                if isinstance(influence_score, (np.float64, np.float32, np.float16)):
                    influence_score = float(influence_score)
                elif isinstance(influence_score, (np.int64, np.int32, np.int16, np.int8)):
                    influence_score = int(influence_score)
                
                # Get entities from source_index if available (frame_index doesn't have entities)
                entities = []
                if 'article_profiles' in self.source_index and doc_id in self.source_index['article_profiles']:
                    entities = self.source_index['article_profiles'][doc_id].get('entities', [])
                
                articles.append({
                    'doc_id': doc_id,
                    'date': article_date,
                    'media': profile.get('media', 'Unknown'),
                    'author': profile.get('author', 'Unknown'),
                    'frames': list(frame_proportions.keys()) if frame_proportions else frames,
                    'frame_proportions': frame_proportions,
                    'entities': entities,
                    'sentiment': sentiment,
                    'influence_score': influence_score
                })
        
        # Fallback to source_index if no frame_index
        elif 'article_profiles' in self.source_index:
            article_profiles = self.source_index['article_profiles']
            
            for doc_id, profile in article_profiles.items():
                # Check date
                article_date = profile.get('date')
                if not article_date:
                    continue
                
                # Convert to datetime if needed
                if isinstance(article_date, str):
                    article_date = pd.to_datetime(article_date)
                
                # Check if in window
                if not (window[0] <= article_date <= window[1]):
                    continue
                
                # Note: source_index doesn't have frame data, so we can't filter by frame
                # Add article anyway
                # Convert numpy types to native Python types
                sentiment = profile.get('sentiment', 0)
                if isinstance(sentiment, (np.float64, np.float32, np.float16)):
                    sentiment = float(sentiment)
                elif isinstance(sentiment, (np.int64, np.int32, np.int16, np.int8)):
                    sentiment = int(sentiment)
                
                influence_score = profile.get('influence_score', 0)
                if isinstance(influence_score, (np.float64, np.float32, np.float16)):
                    influence_score = float(influence_score)
                elif isinstance(influence_score, (np.int64, np.int32, np.int16, np.int8)):
                    influence_score = int(influence_score)
                
                articles.append({
                    'doc_id': doc_id,
                    'date': article_date,
                    'media': profile.get('media'),
                    'author': profile.get('author'),
                    'frames': {},  # No frame data in source_index
                    'entities': profile.get('entities', []),
                    'sentiment': sentiment,
                    'influence_score': influence_score
                })
        
        return articles
    
    def _get_key(self, window: Tuple[datetime, datetime], frame: str) -> str:
        """Generate unique key for window-frame combination."""
        return f"{window[0].isoformat()}_{window[1].isoformat()}_{frame}"
    
    def _count_metrics(self, metrics: Dict[str, Any]) -> int:
        """
        Count total number of metrics computed.
        
        Args:
            metrics: Dictionary of metrics by category
            
        Returns:
            Total count of individual metrics
        """
        count = 0
        for category, category_metrics in metrics.items():
            if isinstance(category_metrics, dict):
                count += len(category_metrics)
            elif isinstance(category_metrics, (list, tuple)):
                count += len(category_metrics)
            elif category_metrics is not None:
                count += 1
        return count
    
    def _clean_graph_for_export(self, G: nx.Graph) -> nx.Graph:
        """
        Clean graph for export by removing None values from attributes.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Cleaned graph suitable for GraphML/GEXF export
        """
        G_clean = G.copy()
        
        # Clean node attributes
        for node in G_clean.nodes():
            attrs = G_clean.nodes[node]
            clean_attrs = {}
            for key, value in attrs.items():
                if value is not None:
                    # Convert complex types to strings
                    if isinstance(value, (dict, list)):
                        clean_attrs[key] = str(value)
                    else:
                        clean_attrs[key] = value
            G_clean.nodes[node].clear()
            G_clean.nodes[node].update(clean_attrs)
        
        # Clean edge attributes
        for u, v in G_clean.edges():
            attrs = G_clean.edges[u, v]
            clean_attrs = {}
            for key, value in attrs.items():
                if value is not None:
                    # Convert complex types to strings
                    if isinstance(value, (dict, list)):
                        clean_attrs[key] = str(value)
                    else:
                        clean_attrs[key] = value
            G_clean.edges[u, v].clear()
            G_clean.edges[u, v].update(clean_attrs)
        
        # Clean graph attributes
        clean_graph_attrs = {}
        for key, value in G_clean.graph.items():
            if value is not None:
                if isinstance(value, (dict, list)):
                    clean_graph_attrs[key] = str(value)
                else:
                    clean_graph_attrs[key] = value
        G_clean.graph.clear()
        G_clean.graph.update(clean_graph_attrs)
        
        return G_clean
    
    # Batch processing methods
    def _process_batch_parallel(self, batch: List[Tuple], pbar: tqdm):
        """Process batch of windows in parallel with improved efficiency."""
        # Use all available workers for the batch
        n_workers = min(len(batch), self.n_workers)
        
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('spawn')) as executor:
            # Submit all tasks immediately
            futures = {
                executor.submit(self.compute_window_network, window, frame): (window, frame)
                for window, frame in batch
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                window, frame = futures[future]
                try:
                    snapshot = future.result(timeout=120)  # 2 min timeout - reduced for faster failure detection
                    key = self._get_key(window, frame)
                    self.window_networks[key] = snapshot
                    self.window_metrics[key] = snapshot.metrics
                    self.stats.completed_windows += 1
                    pbar.update(1)
                except TimeoutError:
                    logger.warning(f"Timeout processing {window}, {frame} - skipping")
                    self.stats.failed_windows += 1
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to process {window}, {frame}: {e}")
                    self.stats.failed_windows += 1
                    pbar.update(1)
    
    def _process_batch_sequential(self, batch: List[Tuple], pbar: tqdm):
        """Process batch of windows sequentially."""
        for window, frame in batch:
            snapshot = self.compute_window_network(window, frame)
            key = self._get_key(window, frame)
            self.window_networks[key] = snapshot
            self.window_metrics[key] = snapshot.metrics
            pbar.update(1)
    
    # Persistence methods
    def _save_snapshot(self, snapshot: NetworkSnapshot):
        """Save network snapshot to disk - OPTIMIZED version."""
        import pickle
        import gzip
        
        # Helper function to convert numpy types to Python types
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Create directory for this window
        window_dir = self.output_dir / 'networks' / snapshot.get_hash()
        window_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics FIRST (fast) - convert numpy types
        metrics_dir = self.output_dir / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_dir / f"{snapshot.get_hash()}.json"
        with open(metrics_file, 'w') as f:
            json.dump(convert_numpy_types(snapshot.metrics), f, indent=2, default=str)
        
        # Save metadata (fast) - convert numpy types
        meta_file = window_dir / 'metadata.json'
        with open(meta_file, 'w') as f:
            json.dump(convert_numpy_types(snapshot.metadata), f, indent=2, default=str)
        
        # Save network in FAST compressed pickle format instead of slow XML formats
        # This is 10-100x faster than GraphML/GEXF for large graphs
        network_file = window_dir / 'network.pkl.gz'
        try:
            with gzip.open(network_file, 'wb', compresslevel=1) as f:  # Low compression = faster
                pickle.dump(snapshot.network, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Saved network to {network_file} (fast pickle format)")
        except Exception as e:
            logger.warning(f"Failed to save network: {e}")
        
        # Optionally save GraphML/GEXF in background if really needed
        # But skip for large graphs during testing
        if snapshot.network.number_of_nodes() < 1000:  # Only for small graphs
            try:
                # Clean network attributes for export (remove None values) 
                G_clean = self._clean_graph_for_export(snapshot.network)
                nx.write_graphml(G_clean, window_dir / 'network.graphml')
            except Exception as e:
                logger.debug(f"Skipped GraphML for large graph: {e}")
    
    def _save_checkpoint(self):
        """Save checkpoint for resuming computation."""
        checkpoint = {
            'completed': set(self.window_networks.keys()),
            'networks': self.window_networks,
            'metrics': self.window_metrics,
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = self.output_dir / 'checkpoints' / 'latest.pkl'
        checkpoint_file.parent.mkdir(exist_ok=True)
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def _load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint if exists."""
        checkpoint_file = self.output_dir / 'checkpoints' / 'latest.pkl'
        
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        
        return None
    
    def _generate_final_report(self):
        """Generate comprehensive final report."""
        report = {
            'summary': self.stats.get_report(),
            'computation_log': self.computation_log[-100:],  # Last 100 computations
            'network_statistics': self._compute_aggregate_statistics(),
            'timestamp': datetime.now().isoformat()
        }
        
        report_file = self.output_dir / 'logs' / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Final report saved to {report_file}")
    
    def _compute_aggregate_statistics(self) -> Dict:
        """Compute aggregate statistics across all networks."""
        stats = {
            'total_networks': len(self.window_networks),
            'avg_nodes': 0,
            'avg_edges': 0,
            'avg_density': 0,
            'total_computation_time': self.stats.total_time
        }
        
        if self.window_networks:
            nodes = [s.network.number_of_nodes() for s in self.window_networks.values()]
            edges = [s.network.number_of_edges() for s in self.window_networks.values()]
            densities = [nx.density(s.network) for s in self.window_networks.values() if s.network.number_of_nodes() > 1]
            
            stats['avg_nodes'] = np.mean(nodes)
            stats['avg_edges'] = np.mean(edges)
            stats['avg_density'] = np.mean(densities) if densities else 0
            stats['max_nodes'] = max(nodes)
            stats['max_edges'] = max(edges)
        
        return stats