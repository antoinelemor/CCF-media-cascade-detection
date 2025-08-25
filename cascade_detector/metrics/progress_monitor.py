"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
progress_monitor.py

MAIN OBJECTIVE:
---------------
This script provides detailed progress monitoring for cascade detection, tracking window computations
with real-time statistics, performance metrics, and ETA calculations.

Dependencies:
-------------
- time
- logging
- typing
- datetime
- collections
- threading

MAIN FEATURES:
--------------
1) Window-level computation tracking with timing
2) Real-time progress statistics and ETA
3) Performance metrics by window size
4) Failed window tracking and reporting
5) Thread-safe operation monitoring

Author:
-------
Antoine Lemor
"""

import time
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class ProgressMonitor:
    """
    Monitor and report progress of window computations.
    Provides detailed tracking of which windows are being processed.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.window_times = {}  # Track time per window
        self.current_windows = {}  # Track currently processing windows
        self.completed_windows = set()
        self.failed_windows = set()
        self.lock = threading.Lock()
        
        # Statistics
        self.total_windows = 0
        self.windows_by_size = defaultdict(int)
        self.windows_by_frame = defaultdict(int)
        
    def set_total(self, total: int, window_details: Optional[List[Tuple]] = None):
        """Set total number of windows to process."""
        self.total_windows = total
        
        if window_details:
            for window, frame in window_details:
                window_size = (window[1] - window[0]).days if hasattr(window[0], 'days') else 0
                self.windows_by_size[window_size] += 1
                self.windows_by_frame[frame] += 1
        
        logger.info(f"\n{'='*80}")
        logger.info(f"WINDOW COMPUTATION PLAN")
        logger.info(f"{'='*80}")
        logger.info(f"Total windows to process: {total}")
        
        if self.windows_by_size:
            logger.info(f"\nWindows by size:")
            for size, count in sorted(self.windows_by_size.items()):
                logger.info(f"  • {size:2d} days: {count:3d} windows")
        
        if self.windows_by_frame:
            logger.info(f"\nWindows by frame:")
            for frame, count in sorted(self.windows_by_frame.items()):
                logger.info(f"  • {frame:8s}: {count:3d} windows")
        
        logger.info(f"{'='*80}\n")
    
    def start_window(self, window: Tuple, frame: str, worker_id: Optional[int] = None):
        """Mark a window as started."""
        with self.lock:
            key = self._make_key(window, frame)
            self.current_windows[key] = {
                'start_time': time.time(),
                'window': window,
                'frame': frame,
                'worker_id': worker_id
            }
            
            # Log current status
            window_size = self._get_window_size(window)
            logger.info(f"[STARTED] Frame: {frame:8s} | "
                       f"Window: {self._format_window(window)} ({window_size:2d} days) | "
                       f"Worker: {worker_id if worker_id else 'main'} | "
                       f"Progress: {len(self.completed_windows)}/{self.total_windows}")
    
    def complete_window(self, window: Tuple, frame: str, metrics: Optional[Dict] = None):
        """Mark a window as completed."""
        with self.lock:
            key = self._make_key(window, frame)
            
            if key in self.current_windows:
                start_time = self.current_windows[key]['start_time']
                elapsed = time.time() - start_time
                self.window_times[key] = elapsed
                del self.current_windows[key]
            else:
                elapsed = 0
            
            self.completed_windows.add(key)
            
            # Calculate statistics
            window_size = self._get_window_size(window)
            n_nodes = metrics.get('basic', {}).get('n_nodes', 0) if metrics else 0
            n_edges = metrics.get('basic', {}).get('n_edges', 0) if metrics else 0
            
            # Log completion
            logger.info(f"[COMPLETE] Frame: {frame:8s} | "
                       f"Window: {self._format_window(window)} | "
                       f"Time: {elapsed:5.1f}s | "
                       f"Graph: {n_nodes:5d} nodes, {n_edges:6d} edges | "
                       f"Progress: {len(self.completed_windows)}/{self.total_windows}")
            
            # Show overall progress
            if len(self.completed_windows) % 10 == 0:
                self._show_progress_summary()
    
    def fail_window(self, window: Tuple, frame: str, error: str):
        """Mark a window as failed."""
        with self.lock:
            key = self._make_key(window, frame)
            
            if key in self.current_windows:
                del self.current_windows[key]
            
            self.failed_windows.add(key)
            
            logger.error(f"[FAILED] Frame: {frame:8s} | "
                        f"Window: {self._format_window(window)} | "
                        f"Error: {error} | "
                        f"Progress: {len(self.completed_windows)}/{self.total_windows}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        with self.lock:
            elapsed_total = time.time() - self.start_time
            completed = len(self.completed_windows)
            failed = len(self.failed_windows)
            in_progress = len(self.current_windows)
            remaining = self.total_windows - completed - failed
            
            # Calculate rates
            rate = completed / elapsed_total if elapsed_total > 0 else 0
            eta = remaining / rate if rate > 0 else 0
            
            return {
                'completed': completed,
                'failed': failed,
                'in_progress': in_progress,
                'remaining': remaining,
                'total': self.total_windows,
                'elapsed_time': elapsed_total,
                'rate': rate,
                'eta_seconds': eta,
                'current_windows': list(self.current_windows.values())
            }
    
    def _show_progress_summary(self):
        """Show progress summary."""
        status = self.get_status()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PROGRESS SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Completed: {status['completed']}/{status['total']} "
                   f"({100*status['completed']/status['total']:.1f}%)")
        logger.info(f"Failed: {status['failed']}")
        logger.info(f"In Progress: {status['in_progress']}")
        logger.info(f"Rate: {status['rate']:.2f} windows/sec")
        logger.info(f"ETA: {status['eta_seconds']/60:.1f} minutes")
        
        if status['current_windows']:
            logger.info(f"\nCurrently processing:")
            for win in status['current_windows']:
                elapsed = time.time() - win['start_time']
                logger.info(f"  • {win['frame']:8s} | {self._format_window(win['window'])} "
                           f"| {elapsed:.1f}s | Worker {win['worker_id']}")
        
        # Show average times by window size
        if self.window_times:
            times_by_size = defaultdict(list)
            for key, elapsed in self.window_times.items():
                # Parse key to get window size
                parts = key.split('_')
                if len(parts) >= 3:
                    try:
                        # Extract dates from key
                        window_str = '_'.join(parts[1:3])
                        # Approximate window size from string
                        for size in [3, 7, 14, 30]:
                            if str(size) in window_str or f"{size}d" in window_str:
                                times_by_size[size].append(elapsed)
                                break
                    except:
                        pass
            
            if times_by_size:
                logger.info(f"\nAverage time by window size:")
                for size in sorted(times_by_size.keys()):
                    times = times_by_size[size]
                    avg_time = sum(times) / len(times)
                    logger.info(f"  • {size:2d} days: {avg_time:5.1f}s (n={len(times)})")
        
        logger.info(f"{'='*60}\n")
    
    @staticmethod
    def _make_key(window: Tuple, frame: str) -> str:
        """Create unique key for window-frame pair."""
        window_str = f"{window[0]}_{window[1]}" if window else "unknown"
        return f"{frame}_{window_str}"
    
    @staticmethod
    def _format_window(window: Tuple) -> str:
        """Format window for display."""
        if not window or len(window) < 2:
            return "unknown"
        
        start, end = window
        if hasattr(start, 'strftime'):
            return f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
        else:
            return f"{start} to {end}"
    
    @staticmethod
    def _get_window_size(window: Tuple) -> int:
        """Get window size in days."""
        if not window or len(window) < 2:
            return 0
        
        start, end = window
        if hasattr(start, 'days'):
            return (end - start).days
        elif hasattr(end, '__sub__'):
            try:
                delta = end - start
                if hasattr(delta, 'days'):
                    return delta.days
            except:
                pass
        return 0


# Global instance for easy access
_global_monitor = None

def get_monitor() -> ProgressMonitor:
    """Get global progress monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ProgressMonitor()
    return _global_monitor

def reset_monitor():
    """Reset global monitor."""
    global _global_monitor
    _global_monitor = ProgressMonitor()
    return _global_monitor