"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
progress_tracker.py

MAIN OBJECTIVE:
---------------
This script provides progress tracking utilities for cascade detection, forcing output display
with detailed window tracking and real-time performance metrics.

Dependencies:
-------------
- sys
- time
- typing
- datetime
- tqdm

MAIN FEATURES:
--------------
1) Detailed window-level progress tracking
2) Real-time performance metrics and ETA calculation
3) Force output display to bypass buffering
4) Window size and frame distribution analysis
5) Failed window tracking and retry management

Author:
-------
Antoine Lemor
"""

import sys
import time
from typing import Optional, Callable, Dict, Any, List, Tuple
from datetime import datetime
from tqdm import tqdm


class DetailedProgressTracker:
    """
    Enhanced progress tracker with detailed window information.
    Provides real-time visibility into cascade detection progress.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.window_stats = {}
        self.current_windows = {}
        self.completed_count = 0
        self.failed_count = 0
        self.total_windows = 0
        
    def set_total_windows(self, total: int, window_details: List[Tuple] = None):
        """Set total number of windows to process."""
        self.total_windows = total
        
        if window_details:
            # Analyze window distribution
            size_distribution = {}
            frame_distribution = {}
            
            for detail in window_details:
                # Handle both 2-tuple (window, frame) and 3-tuple (window, frame, size) formats
                if len(detail) == 3:
                    window, frame, size = detail
                else:
                    window, frame = detail
                    # Calculate window size
                    if hasattr(window[0], 'days'):
                        size = (window[1] - window[0]).days
                    else:
                        size = 0
                    
                size_distribution[size] = size_distribution.get(size, 0) + 1
                frame_distribution[frame] = frame_distribution.get(frame, 0) + 1
            
            self.window_stats = {
                'size_distribution': size_distribution,
                'frame_distribution': frame_distribution
            }
            
            # Force print initial stats
            self._print_plan()
    
    def _print_plan(self):
        """Print the processing plan."""
        force_print_progress("=" * 80)
        force_print_progress(f"WINDOW PROCESSING PLAN: {self.total_windows} total windows")
        force_print_progress("=" * 80)
        
        if self.window_stats.get('size_distribution'):
            force_print_progress("Windows by size:")
            for size, count in sorted(self.window_stats['size_distribution'].items()):
                force_print_progress(f"  • {size:2d} days: {count:4d} windows")
        
        if self.window_stats.get('frame_distribution'):
            force_print_progress("Windows by frame:")
            for frame, count in sorted(self.window_stats['frame_distribution'].items()):
                force_print_progress(f"  • {frame:8s}: {count:4d} windows")
        
        force_print_progress("=" * 80)
    
    def start_window(self, window: Tuple, frame: str, size: int = None):
        """Track a window starting processing."""
        key = f"{frame}_{window[0]}_{window[1]}"
        self.current_windows[key] = {
            'start_time': time.time(),
            'frame': frame,
            'window': window,
            'size': size or self._calculate_window_size(window)
        }
        
        # Print status every 10 windows
        if len(self.current_windows) % 10 == 0:
            self._print_current_status()
    
    def complete_window(self, window: Tuple, frame: str, success: bool = True):
        """Mark a window as completed."""
        key = f"{frame}_{window[0]}_{window[1]}"
        
        if key in self.current_windows:
            elapsed = time.time() - self.current_windows[key]['start_time']
            del self.current_windows[key]
        else:
            elapsed = 0
        
        if success:
            self.completed_count += 1
        else:
            self.failed_count += 1
        
        # Print progress update (already handled inside _print_progress_update)
        self._print_progress_update(frame, window, elapsed, success)
    
    def update_batch_complete(self, n_windows: int):
        """Update progress after a batch completes."""
        # Just force print current progress
        percent = (self.completed_count / self.total_windows * 100) if self.total_windows > 0 else 0
        total_elapsed = time.time() - self.start_time
        rate = self.completed_count / total_elapsed if total_elapsed > 0 else 0
        eta = (self.total_windows - self.completed_count) / rate if rate > 0 else 0
        
        message = (f"[BATCH] Progress: {self.completed_count}/{self.total_windows} ({percent:.1f}%) | "
                  f"Rate: {rate:.1f} win/s | ETA: {eta/60:.1f} min")
        force_print_progress(message)
    
    def _print_progress_update(self, frame: str, window: Tuple, elapsed: float, success: bool):
        """Print detailed progress update."""
        percent = (self.completed_count / self.total_windows * 100) if self.total_windows > 0 else 0
        total_elapsed = time.time() - self.start_time
        rate = self.completed_count / total_elapsed if total_elapsed > 0 else 0
        eta = (self.total_windows - self.completed_count) / rate if rate > 0 else 0
        
        status = "✓" if success else "✗"
        window_str = self._format_window(window)
        window_size = self._calculate_window_size(window)
        
        # Print every completion if elapsed time > 10s (likely a large window)
        # Or every 10 completions for small windows
        if elapsed > 10.0:
            message = (f"[{status}] SLOW WINDOW: {frame:8s} | {window_str} ({window_size}d) | "
                      f"Time: {elapsed:.1f}s | Progress: {self.completed_count}/{self.total_windows}")
            force_print_progress(message)
        elif self.completed_count % 10 == 0 or not success:
            message = (f"[{status}] Progress: {self.completed_count}/{self.total_windows} ({percent:.1f}%) | "
                      f"Rate: {rate:.1f} win/s | "
                      f"ETA: {eta/60:.1f} min | "
                      f"Active: {len(self.current_windows)}")
            
            force_print_progress(message)
    
    def _print_current_status(self):
        """Print status of currently processing windows."""
        if self.current_windows:
            force_print_progress(f"\n[ACTIVE] {len(self.current_windows)} windows in progress:")
            for key, info in list(self.current_windows.items())[:3]:  # Show first 3
                elapsed = time.time() - info['start_time']
                force_print_progress(f"  • {info['frame']:8s} | {self._format_window(info['window'])} | {elapsed:.1f}s")
            if len(self.current_windows) > 3:
                force_print_progress(f"  ... and {len(self.current_windows) - 3} more")
    
    def _format_window(self, window: Tuple) -> str:
        """Format window for display."""
        if not window or len(window) < 2:
            return "unknown"
        
        start, end = window
        if hasattr(start, 'strftime'):
            return f"{start.strftime('%m/%d')}-{end.strftime('%m/%d')}"
        return f"{start}-{end}"
    
    def _calculate_window_size(self, window: Tuple) -> int:
        """Calculate window size in days."""
        if window and len(window) >= 2:
            try:
                # Try to calculate delta
                delta = window[1] - window[0]
                if hasattr(delta, 'days'):
                    return delta.days
            except:
                pass
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        total_elapsed = time.time() - self.start_time
        rate = self.completed_count / total_elapsed if total_elapsed > 0 else 0
        
        return {
            'completed': self.completed_count,
            'failed': self.failed_count,
            'in_progress': len(self.current_windows),
            'remaining': self.total_windows - self.completed_count - self.failed_count,
            'total': self.total_windows,
            'elapsed_time': total_elapsed,
            'rate': rate,
            'eta_seconds': (self.total_windows - self.completed_count) / rate if rate > 0 else 0
        }


# Global tracker instance
_global_tracker = None

def get_tracker() -> DetailedProgressTracker:
    """Get global progress tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = DetailedProgressTracker()
    return _global_tracker

def reset_tracker() -> DetailedProgressTracker:
    """Reset global tracker."""
    global _global_tracker
    _global_tracker = DetailedProgressTracker()
    return _global_tracker


class ForceProgressBar:
    """
    Progress bar that forces immediate display.
    Works around buffering issues with tqdm.
    """
    
    def __init__(self, total: int, desc: str = "", unit: str = "it"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.unit = unit
        self.start_time = time.time()
        self.last_update = 0
        
        # Also use global tracker if available
        self.tracker = get_tracker()
        
        # Print initial status
        self._print_status()
    
    def __enter__(self):
        """Support context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support context manager protocol."""
        self.close()
        return False
    
    def _print_status(self):
        """Force print current status."""
        elapsed = time.time() - self.start_time
        percent = (self.current / self.total * 100) if self.total > 0 else 0
        
        # Build progress bar
        bar_length = 40
        filled = int(bar_length * self.current / self.total) if self.total > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Calculate rate
        rate = self.current / elapsed if elapsed > 0 else 0
        
        # Build status line
        status = f"\r{self.desc}: {bar} {self.current}/{self.total} {self.unit} [{percent:.1f}%] ({rate:.1f} {self.unit}/s)"
        
        # Force immediate output
        print(status, end='', flush=True)
        sys.stdout.flush()
        
        # Also print to stderr as backup
        if time.time() - self.last_update > 0.5:  # Update stderr every 0.5s
            print(f"\n[Progress] {self.desc}: {self.current}/{self.total} ({percent:.1f}%)", file=sys.stderr, flush=True)
            self.last_update = time.time()
    
    def update(self, n: int = 1):
        """Update progress."""
        self.current = min(self.current + n, self.total)
        self._print_status()
    
    def set_description(self, desc: str):
        """Update description."""
        self.desc = desc
        self._print_status()
    
    def refresh(self):
        """Refresh the progress bar display."""
        self._print_status()
    
    def set_postfix(self, postfix_dict, refresh=True):
        """Compatibility with tqdm."""
        # Just ignore postfix for now
        if refresh:
            self.refresh()
    
    def close(self):
        """Close progress bar."""
        print()  # New line
        sys.stdout.flush()


def track_progress(iterable, desc: str = "", total: Optional[int] = None):
    """
    Track progress with forced output.
    
    Args:
        iterable: Iterable to track
        desc: Description
        total: Total items (if not provided, will try to get from iterable)
    
    Returns:
        Iterator with progress tracking
    """
    if total is None:
        try:
            total = len(iterable)
        except:
            total = None
    
    if total:
        pbar = ForceProgressBar(total, desc)
        for item in iterable:
            yield item
            pbar.update()
        pbar.close()
    else:
        # Fallback to simple counter
        for i, item in enumerate(iterable):
            if i % 100 == 0:
                print(f"[Progress] {desc}: {i} items processed", flush=True)
            yield item


def force_print_progress(message: str):
    """
    Force print a progress message immediately.
    
    Args:
        message: Message to print
    """
    # Print to both stdout and stderr
    print(f"\n>>> {message}", flush=True)
    print(f">>> {message}", file=sys.stderr, flush=True)
    
    # Force flush
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Try to force OS-level flush
    try:
        import os
        if hasattr(sys.stdout, 'fileno'):
            os.fsync(sys.stdout.fileno())
    except:
        pass