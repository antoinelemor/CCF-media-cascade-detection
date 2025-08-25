"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
parallel_signal_aggregator.py

MAIN OBJECTIVE:
---------------
This script extends SignalAggregator with parallel processing capabilities, computing multiple
time windows simultaneously with intelligent scheduling to optimize resource usage on M4 Max.

Dependencies:
-------------
- logging
- time
- typing
- datetime
- concurrent.futures
- multiprocessing
- queue
- numpy

MAIN FEATURES:
--------------
1) Parallel window computation with priority scheduling
2) Intelligent resource allocation between large and small windows
3) Real-time progress monitoring and ETA calculation
4) Dynamic worker allocation based on window complexity
5) Memory-efficient batch processing

Author:
-------
Antoine Lemor
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from queue import PriorityQueue
import numpy as np

from cascade_detector.detectors.signal_aggregator import SignalAggregator, AggregatedSignal
from cascade_detector.detectors.base_detector import DetectionContext
from cascade_detector.core.config import DetectorConfig
from cascade_detector.metrics.parallel_compute_engine import ParallelComputeEngine, ComputeTask
from cascade_detector.metrics.progress_monitor import get_monitor
from cascade_detector.utils.progress_tracker import (
    ForceProgressBar, force_print_progress, 
    get_tracker, reset_tracker, DetailedProgressTracker
)

logger = logging.getLogger(__name__)


class WindowTask:
    """Represents a window computation task with priority."""
    
    def __init__(self, frame: str, window: Tuple[datetime, datetime], 
                 window_size: int, priority: int = 0):
        self.frame = frame
        self.window = window
        self.window_size = window_size
        self.priority = priority  # Lower is higher priority
        
        # Estimate computational complexity
        self.estimated_complexity = window_size ** 2  # Simplified estimate
        
    def __lt__(self, other):
        """For priority queue ordering."""
        # Prioritize smaller windows (they complete faster)
        return self.window_size < other.window_size


class ParallelSignalAggregator(SignalAggregator):
    """
    Enhanced SignalAggregator with parallel window computation.
    
    Features:
    - Parallel computation of multiple windows
    - Intelligent scheduling to avoid resource conflicts
    - Priority-based execution (small windows first)
    - Dynamic load balancing
    """
    
    def __init__(self, context: DetectionContext, config: Optional[DetectorConfig] = None):
        """Initialize with parallel computation support."""
        super().__init__(context, config)
        
        # Parallel configuration
        self.n_parallel_workers = min(mp.cpu_count(), 16)  # M4 Max has 16 performance cores
        self.use_parallel = config.use_parallel if config and hasattr(config, 'use_parallel') else True
        
        # Initialize parallel compute engine
        self.compute_engine = ParallelComputeEngine({
            'max_workers': self.n_parallel_workers,
            'use_gpu': self.config.use_gpu if hasattr(self.config, 'use_gpu') else True,
            'memory_limit_gb': 100  # Conservative limit for M4 Max
        })
        
        logger.info(f"ParallelSignalAggregator initialized with {self.n_parallel_workers} workers")
    
    def detect(self, **kwargs) -> Dict[str, Any]:
        """
        Detect patterns with parallel window computation.
        
        This override computes multiple windows in parallel for better performance.
        """
        # Extract parameters
        frames = kwargs.get('frames', self.context.frames)
        window = kwargs.get('window', self.context.time_window)
        
        # Initialize progress tracker
        tracker = get_tracker()
        
        if self.config.verbose:
            logger.info(f"ParallelSignalAggregator.detect() starting")
            logger.info(f"  - Frames: {frames}")
            logger.info(f"  - Window: {window[0].date()} to {window[1].date()}")
            logger.info(f"  - Window sizes: {self.window_sizes}")
            logger.info(f"  - Parallel workers: {self.n_parallel_workers}")
        
        # Generate all window tasks
        window_tasks = self._generate_window_tasks(frames, window)
        
        # Set up progress tracking with detailed window information including sizes
        window_details = [(task.window, task.frame, task.window_size) for task in window_tasks]
        tracker.set_total_windows(len(window_tasks), window_details)
        
        if self.config.verbose:
            logger.info(f"  - Total window tasks: {len(window_tasks)}")
        
        # Compute windows in parallel or sequentially based on configuration
        if self.use_parallel and len(window_tasks) > 4:
            signals = self._compute_parallel_windows(window_tasks)
        else:
            # Fall back to sequential computation for small workloads
            signals = self._compute_sequential_windows(window_tasks)
        
        # Prepare result
        result = {
            'signals': signals,
            'n_signals': len(signals),
            'n_frames': len(frames),
            'n_windows': len(window_tasks),
            'window_sizes': self.window_sizes,
            'time_range': window,
            'frames_analyzed': frames,
            'computation_mode': 'parallel' if self.use_parallel else 'sequential'
        }
        
        # Add summary statistics
        if signals:
            result['summary'] = {
                'avg_intensity': np.mean([s.temporal_features.get('intensity_score', 0) for s in signals]),
                'max_intensity': max([s.temporal_features.get('intensity_score', 0) for s in signals]),
                'total_articles': sum([s.n_articles for s in signals]),
                'signals_with_data': len([s for s in signals if s.n_articles > 0])
            }
        
        if self.config.verbose:
            logger.info(f"ParallelSignalAggregator.detect() completed: {len(signals)} signals")
        
        return result
    
    def _generate_window_tasks(self, frames: List[str], 
                               window: Tuple[datetime, datetime]) -> List[WindowTask]:
        """Generate all window computation tasks."""
        tasks = []
        
        for frame in frames:
            for window_size in self.window_sizes:
                # Generate sliding windows
                windows = self._generate_windows(window, window_size)
                
                for i, win in enumerate(windows):
                    # Priority based on window size and position
                    # Smaller windows and earlier windows have higher priority
                    priority = window_size * 100 + i
                    
                    task = WindowTask(
                        frame=frame,
                        window=win,
                        window_size=window_size,
                        priority=priority
                    )
                    tasks.append(task)
        
        # Sort by priority (smaller windows first)
        tasks.sort(key=lambda t: (t.window_size, t.priority))
        
        return tasks
    
    def _compute_parallel_windows(self, window_tasks: List[WindowTask]) -> List[AggregatedSignal]:
        """
        Compute multiple windows in parallel with intelligent scheduling.
        
        Strategy:
        1. Group windows by size
        2. Process small windows with high parallelism
        3. Process large windows with limited parallelism
        4. Avoid processing multiple large windows simultaneously
        """
        signals = []
        
        # Group tasks by window size
        size_groups = {}
        for task in window_tasks:
            if task.window_size not in size_groups:
                size_groups[task.window_size] = []
            size_groups[task.window_size].append(task)
        
        # Process each size group
        tracker = get_tracker()
        
        # IMPORTANT: Process LARGE windows first to test optimizations
        force_print_progress("Processing windows from LARGEST to SMALLEST for better resource testing")
        for size in sorted(size_groups.keys(), reverse=True):
            tasks = size_groups[size]
            
            # Determine parallelism level based on window size
            # For 448 small windows, we should process them in smaller batches
            if size <= 3:
                # Very small windows - process in SMALLER batches to avoid blocking
                n_workers = self.n_parallel_workers  # Use all 16 cores
                # CRITICAL: Use batch size = n_workers to avoid overloading
                batch_size = n_workers  # Only 16 windows at a time
                force_print_progress(f"\nProcessing {len(tasks)} windows of size {size} days")
                force_print_progress(f"  Strategy: {n_workers} workers, batch size {batch_size} (1 window per worker)")
                
                # Process in smaller batches to keep all cores busy without blocking
                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i:i+batch_size]
                    force_print_progress(f"  Batch {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1}: {len(batch)} windows")
                    batch_signals = self._process_task_group(batch, n_workers)
                    signals.extend(batch_signals)
                    # Update progress after each batch
                    tracker.update_batch_complete(len(batch))
                continue  # Skip the regular processing below
                
            elif size <= 7:
                # Small windows - high parallelism
                n_workers = min(self.n_parallel_workers, len(tasks))
            elif size <= 30:
                # Medium windows - moderate parallelism
                n_workers = min(self.n_parallel_workers // 2, len(tasks))
            else:
                # Large windows - limited parallelism to avoid memory issues
                n_workers = min(4, len(tasks))
            
            force_print_progress(f"\nProcessing {len(tasks)} windows of size {size} days with {n_workers} workers")
            
            # Process this group in parallel
            group_signals = self._process_task_group(tasks, n_workers)
            signals.extend(group_signals)
        
        return signals
    
    def _process_task_group(self, tasks: List[WindowTask], n_workers: int) -> List[AggregatedSignal]:
        """Process a group of tasks with specified parallelism."""
        signals = []
        tracker = get_tracker()
        
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('spawn')) as executor:
            # Submit all tasks (but don't track start yet - they're just queued)
            futures = {}
            for task in tasks:
                future = executor.submit(
                    self._compute_single_window_static,
                    task.frame,
                    task.window,
                    self.context,
                    self.config
                )
                futures[future] = task
                # Note: Window start will be tracked when it actually begins processing
            
            # Process and collect results as they complete
            completed = 0
            for future in as_completed(futures):
                task = futures[future]
                try:
                    # No timeout - wait for exact computation
                    signal = future.result()
                    if signal and signal.n_articles > 0:
                        signals.append(signal)
                    
                    # Track window completion with metrics if available
                    completed += 1
                    # Extract network metrics from signal for progress display
                    if signal and hasattr(signal, 'network_features'):
                        metrics = {
                            'basic': {
                                'n_nodes': signal.network_features.get('network_nodes', 0),
                                'n_edges': signal.network_features.get('network_edges', 0)
                            }
                        }
                        # Use the monitor if available
                        from cascade_detector.metrics.progress_monitor import get_monitor
                        monitor = get_monitor()
                        if monitor:
                            monitor.complete_window(task.window, task.frame, metrics)
                        else:
                            tracker.complete_window(task.window, task.frame, success=True)
                    else:
                        tracker.complete_window(task.window, task.frame, success=True)
                    
                    # Print progress every window for small windows
                    if task.window_size <= 3 and completed % 5 == 0:
                        force_print_progress(f"    Batch progress: {completed}/{len(tasks)} windows")
                        
                    if self.config.verbose:
                        logger.info(f"  Completed {task.frame} window "
                                  f"{task.window[0].date()} to {task.window[1].date()}")
                        
                except Exception as e:
                    # Track window failure
                    tracker.complete_window(task.window, task.frame, success=False)
                    logger.error(f"Failed to compute window {task.frame} "
                                f"{task.window[0].date()} to {task.window[1].date()}: {e}")
        
        return signals
    
    @staticmethod
    def _compute_single_window_static(frame: str, window: Tuple[datetime, datetime],
                                     context: DetectionContext, 
                                     config: DetectorConfig) -> Optional[AggregatedSignal]:
        """
        Static method for computing a single window (for parallel execution).
        
        This runs in a separate process, so it needs to be self-contained.
        """
        # Import necessary modules (in separate process)
        import warnings
        warnings.filterwarnings('ignore')
        
        # Create a temporary aggregator instance
        from cascade_detector.detectors.signal_aggregator import SignalAggregator
        
        # Disable verbose logging in child process
        temp_config = DetectorConfig()
        if config:
            temp_config.__dict__.update(config.__dict__)
        temp_config.verbose = False
        
        # Create aggregator and compute signal
        aggregator = SignalAggregator(context, temp_config)
        signal = aggregator.aggregate_signals(frame, window)
        
        return signal
    
    def _compute_sequential_windows(self, window_tasks: List[WindowTask]) -> List[AggregatedSignal]:
        """Fall back to sequential computation for small workloads."""
        signals = []
        tracker = get_tracker()
        
        for task in window_tasks:
            # Track window start
            tracker.start_window(task.window, task.frame, task.window_size)
            
            try:
                signal = self.aggregate_signals(task.frame, task.window)
                if signal and signal.n_articles > 0:
                    signals.append(signal)
                
                # Track successful completion
                tracker.complete_window(task.window, task.frame, success=True)
            except Exception as e:
                # Track failure
                tracker.complete_window(task.window, task.frame, success=False)
                logger.error(f"Failed to compute window {task.frame} {task.window}: {e}")
        
        return signals
    
    def cleanup(self):
        """Clean up parallel resources."""
        if hasattr(self, 'compute_engine'):
            self.compute_engine.shutdown()


def create_optimized_aggregator(context: DetectionContext, 
                                config: Optional[DetectorConfig] = None) -> SignalAggregator:
    """
    Factory function to create the best aggregator for the system.
    
    Returns ParallelSignalAggregator if conditions are met, otherwise standard SignalAggregator.
    """
    # Check if parallel processing is beneficial
    if mp.cpu_count() >= 8:  # Sufficient cores for parallelization
        logger.info("Creating ParallelSignalAggregator for multi-core system")
        return ParallelSignalAggregator(context, config)
    else:
        logger.info("Creating standard SignalAggregator for limited-core system")
        return SignalAggregator(context, config)