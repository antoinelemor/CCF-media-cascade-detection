"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
config.py

MAIN OBJECTIVE:
---------------
This script manages the configuration settings for the cascade detection framework, providing 
centralized configuration management with environment variable overrides.

Dependencies:
-------------
- os
- dataclasses
- typing

MAIN FEATURES:
--------------
1) Central configuration dataclass for all detector parameters
2) Environment variable integration for flexible deployment
3) Default values for all configuration parameters
4) Support for database, detection, network, and analysis settings

Author:
-------
Antoine Lemor
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from cascade_detector.core.constants import *


@dataclass
class DetectorConfig:
    """
    Central configuration for the cascade detector.
    Can be overridden via environment variables or config files.
    """
    
    # Database configuration
    db_host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", "CCF"))
    db_user: str = field(default_factory=lambda: os.getenv("DB_USER", "antoine"))
    db_password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    db_table: str = DB_TABLE
    
    # Frames to analyze
    frames: List[str] = field(default_factory=lambda: FRAMES.copy())
    
    # Messengers configuration
    messengers: List[str] = field(default_factory=lambda: MESSENGERS.copy())
    messenger_types: Dict[str, str] = field(default_factory=lambda: MESSENGER_TYPES.copy())
    
    # Performance settings (optimized for M4 Max)
    n_workers: int = field(default_factory=lambda: int(os.getenv("N_WORKERS", str(DEFAULT_N_WORKERS))))
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", str(DEFAULT_BATCH_SIZE))))
    cache_size_gb: int = field(default_factory=lambda: int(os.getenv("CACHE_SIZE_GB", str(DEFAULT_CACHE_SIZE_GB))))
    use_parallel: bool = True
    enable_cache: bool = True
    
    # GPU settings (Metal Performance Shaders)
    use_gpu: bool = True
    use_hybrid_mode: bool = True  # Re-enabled with rigorous synchronization
    gpu_batch_size: int = 50000
    gpu_sync_mode: str = "strict"  # strict, normal, or minimal synchronization
    
    # Detection parameters (data-driven, these are fallbacks)
    min_cascade_days: int = MIN_CASCADE_DAYS
    max_cascade_days: int = MAX_CASCADE_DAYS
    min_articles: int = MIN_ARTICLES
    min_journalists: int = MIN_JOURNALISTS
    min_media: int = MIN_MEDIA
    
    # Adaptive window parameters
    window_min_days: int = WINDOW_MIN_DAYS
    window_max_days: int = WINDOW_MAX_DAYS
    window_overlap: float = WINDOW_OVERLAP
    adaptive_windows: bool = True
    
    # Scoring configuration
    dimension_weight: float = DIMENSION_WEIGHT
    subindex_weight: float = SUBINDEX_WEIGHT
    require_validation: bool = True
    min_confidence: float = 0.5
    
    # Output configuration
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "results/"))
    export_format: str = "json"
    save_networks: bool = True
    verbose: bool = True
    
    # Date handling
    exclude_2025: bool = True
    date_format_db: str = DATE_FORMAT_DB
    date_format_iso: str = DATE_FORMAT_ISO
    
    def validate(self) -> bool:
        """Validate configuration consistency."""
        # Check weights sum to 1
        total_weight = 5 * self.dimension_weight
        assert abs(total_weight - 1.0) < 1e-6, f"Dimension weights must sum to 1.0, got {total_weight}"
        
        # Check each dimension has 4 sub-indices
        subindex_total = 4 * self.subindex_weight
        assert abs(subindex_total - self.dimension_weight) < 1e-6, \
            f"Sub-index weights must sum to dimension weight: {subindex_total} != {self.dimension_weight}"
        
        # Check window parameters
        assert self.window_min_days < self.window_max_days, \
            "Window min days must be less than max days"
        assert 0 < self.window_overlap < 1, \
            "Window overlap must be between 0 and 1"
        
        return True
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'database': {
                'host': self.db_host,
                'port': self.db_port,
                'name': self.db_name,
                'user': self.db_user,
                'table': self.db_table
            },
            'frames': self.frames,
            'messengers': self.messenger_types,
            'performance': {
                'n_workers': self.n_workers,
                'batch_size': self.batch_size,
                'cache_size_gb': self.cache_size_gb,
                'parallel': self.use_parallel
            },
            'detection': {
                'min_cascade_days': self.min_cascade_days,
                'max_cascade_days': self.max_cascade_days,
                'min_articles': self.min_articles,
                'min_journalists': self.min_journalists,
                'min_media': self.min_media
            },
            'windows': {
                'min_days': self.window_min_days,
                'max_days': self.window_max_days,
                'overlap': self.window_overlap,
                'adaptive': self.adaptive_windows
            },
            'scoring': {
                'dimension_weight': self.dimension_weight,
                'subindex_weight': self.subindex_weight,
                'min_confidence': self.min_confidence
            }
        }
    
    @classmethod
    def from_file(cls, path: str) -> 'DetectorConfig':
        """Load configuration from JSON file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)