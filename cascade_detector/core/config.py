"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
config.py

MAIN OBJECTIVE:
---------------
Minimal configuration for the cascade detection framework.
No GPU or parallel settings. Embeddings are mandatory.

Author:
-------
Antoine Lemor
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict

from cascade_detector.core.constants import (
    FRAMES, MESSENGERS, DB_NAME, DB_TABLE, DATE_FORMAT_DB, DATE_FORMAT_ISO,
    MIN_ARTICLES, MIN_JOURNALISTS, MIN_MEDIA,
)


@dataclass
class DetectorConfig:
    """Central configuration for the cascade detector."""

    # Database
    db_host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", DB_NAME))
    db_user: str = field(default_factory=lambda: os.getenv("DB_USER", "antoine"))
    db_password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    db_table: str = DB_TABLE

    # Frames to analyze
    frames: List[str] = field(default_factory=lambda: FRAMES.copy())
    messengers: List[str] = field(default_factory=lambda: MESSENGERS.copy())

    # PELT changepoint detection
    pelt_penalty: float = 2.0            # PELT penalty parameter (higher = fewer changepoints)
    pelt_smoothing_window: int = 7       # smoothing window before PELT (days)
    pelt_min_size: int = 7               # minimum PELT segment size (days)
    pelt_elevation_threshold: float = 0.3  # elevated segment cutoff: mean + threshold × std
    proptest_alpha: float = 0.01         # significance level for binomial proportion test
    proptest_min_cohen_h: float = 0.05   # minimum effect size (Cohen's h)
    proptest_min_ratio: float = 1.5      # minimum proportion ratio above baseline
    min_burst_days: int = 3
    baseline_window_days: int = 90
    burst_merge_gap_days: int = 1        # merge gap (days); PELT produces tight segments
    boundary_max_lookback: int = 7       # max days for boundary extension walk
    boundary_gap_tolerance: int = 2      # consecutive below-baseline days tolerated

    # Cascade validation
    min_articles: int = MIN_ARTICLES
    min_journalists: int = MIN_JOURNALISTS
    min_media: int = MIN_MEDIA
    min_adoption_breadth: float = 0.60  # fraction of growth-phase days with new journalists
    min_new_entrant_rate: float = 0.0   # no hard minimum, used as sub-index

    # Scoring weights (4 dimensions, sum = 1.0)
    weight_temporal: float = 0.25
    weight_participation: float = 0.25
    weight_convergence: float = 0.25
    weight_source: float = 0.25

    # Composite signal weights (5 signals, sum = 1.0)
    signal_weight_temporal: float = 0.25
    signal_weight_participation: float = 0.20
    signal_weight_convergence: float = 0.20
    signal_weight_source: float = 0.15
    signal_weight_semantic: float = 0.20

    # Embeddings (mandatory — required for semantic signal and convergence scoring)
    embedding_dir: str = field(default_factory=lambda: os.getenv("EMBEDDING_DIR", "data/embeddings"))
    embedding_dim: int = 1024

    # Date handling
    exclude_2025: bool = True
    date_format_db: str = DATE_FORMAT_DB
    date_format_iso: str = DATE_FORMAT_ISO

    # Output
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "results"))
    verbose: bool = False

    def validate(self) -> bool:
        """Validate configuration consistency."""
        # Scoring dimension weights (4 dimensions)
        scoring_weight = (self.weight_temporal + self.weight_participation
                          + self.weight_convergence + self.weight_source)
        assert abs(scoring_weight - 1.0) < 0.01, \
            f"Scoring dimension weights must sum to 1.0, got {scoring_weight}"
        # Composite signal weights (5 signals)
        signal_weight = (self.signal_weight_temporal + self.signal_weight_participation
                         + self.signal_weight_convergence + self.signal_weight_source
                         + self.signal_weight_semantic)
        assert abs(signal_weight - 1.0) < 0.01, \
            f"Signal weights must sum to 1.0, got {signal_weight}"
        assert self.pelt_penalty > 0, "PELT penalty must be positive"
        assert 0 < self.proptest_alpha < 1, "proptest_alpha must be in (0, 1)"
        assert self.proptest_min_cohen_h >= 0, "proptest_min_cohen_h must be >= 0"
        assert self.proptest_min_ratio >= 1, "proptest_min_ratio must be >= 1"
        assert self.min_burst_days >= 1, "Minimum burst days must be >= 1"
        assert self.baseline_window_days >= 30, "Baseline window must be >= 30 days"
        assert self.boundary_max_lookback >= 0, "boundary_max_lookback must be >= 0"
        return True

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'database': {
                'host': self.db_host,
                'port': self.db_port,
                'name': self.db_name,
                'user': self.db_user,
                'table': self.db_table,
            },
            'frames': self.frames,
            'detection': {
                'pelt_penalty': self.pelt_penalty,
                'pelt_smoothing_window': self.pelt_smoothing_window,
                'pelt_min_size': self.pelt_min_size,
                'pelt_elevation_threshold': self.pelt_elevation_threshold,
                'proptest_alpha': self.proptest_alpha,
                'proptest_min_cohen_h': self.proptest_min_cohen_h,
                'proptest_min_ratio': self.proptest_min_ratio,
                'min_burst_days': self.min_burst_days,
                'baseline_window_days': self.baseline_window_days,
                'burst_merge_gap_days': self.burst_merge_gap_days,
                'boundary_max_lookback': self.boundary_max_lookback,
                'boundary_gap_tolerance': self.boundary_gap_tolerance,
                'min_articles': self.min_articles,
                'min_journalists': self.min_journalists,
                'min_media': self.min_media,
            },
            'scoring': {
                'weight_temporal': self.weight_temporal,
                'weight_participation': self.weight_participation,
                'weight_convergence': self.weight_convergence,
                'weight_source': self.weight_source,
            },
            'signal_weights': {
                'signal_weight_temporal': self.signal_weight_temporal,
                'signal_weight_participation': self.signal_weight_participation,
                'signal_weight_convergence': self.signal_weight_convergence,
                'signal_weight_source': self.signal_weight_source,
                'signal_weight_semantic': self.signal_weight_semantic,
            },
        }
