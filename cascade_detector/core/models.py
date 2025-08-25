"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
models.py

MAIN OBJECTIVE:
---------------
This script defines the core data models and structures used throughout the cascade detection 
framework, including time windows, cascade candidates, metrics, and scoring dimensions.

Dependencies:
-------------
- dataclasses
- typing
- pandas
- networkx
- datetime

MAIN FEATURES:
--------------
1) TimeWindow class for adaptive temporal analysis windows
2) CascadeCandidate for detected cascade events
3) Multi-dimensional scoring system with Dimension and SubIndex classes
4) CascadeMetrics for comprehensive cascade quantification
5) MediaCascade as the complete cascade analysis container

Author:
-------
Antoine Lemor
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import networkx as nx
from datetime import datetime


@dataclass
class TimeWindow:
    """Adaptive temporal window for analysis."""
    start: pd.Timestamp
    end: pd.Timestamp
    size_days: int
    variance: float
    data_points: int
    is_adaptive: bool = True
    
    @property
    def duration(self) -> pd.Timedelta:
        """Get window duration."""
        return self.end - self.start
    
    def contains(self, date: pd.Timestamp) -> bool:
        """Check if date is within window."""
        return self.start <= date <= self.end
    
    def overlap_ratio(self, other: 'TimeWindow') -> float:
        """Calculate overlap ratio with another window."""
        latest_start = max(self.start, other.start)
        earliest_end = min(self.end, other.end)
        
        if latest_start <= earliest_end:
            overlap_days = (earliest_end - latest_start).days
            total_days = max((self.end - self.start).days, 
                            (other.end - other.start).days)
            return overlap_days / total_days if total_days > 0 else 0
        return 0


@dataclass
class IndexEntry:
    """Single entry in an index for fast lookup."""
    frame: str
    date: pd.Timestamp
    doc_id: str
    sentence_id: int
    media: str
    author: str
    value: float
    entities: Dict[str, List[str]] = field(default_factory=dict)
    messengers: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'frame': self.frame,
            'date': self.date.isoformat(),
            'doc_id': self.doc_id,
            'sentence_id': self.sentence_id,
            'media': self.media,
            'author': self.author,
            'value': self.value,
            'entities': self.entities,
            'messengers': self.messengers,
            'metadata': self.metadata
        }


@dataclass
class CascadeCandidate:
    """Detected cascade candidate."""
    id: str
    frame: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    peak_date: pd.Timestamp
    window: TimeWindow
    detection_method: str
    confidence: float
    data_indices: List[int]
    
    # Optional fields populated during analysis
    n_articles: int = 0
    n_journalists: int = 0
    n_media: int = 0
    intensity_peak: float = 0.0
    
    @property
    def duration_days(self) -> int:
        """Get cascade duration in days."""
        return (self.end_date - self.start_date).days
    
    def is_valid(self, min_articles: int = 10, 
                 min_journalists: int = 3,
                 min_media: int = 2) -> bool:
        """Check if candidate meets minimum criteria."""
        return (self.n_articles >= min_articles and
                self.n_journalists >= min_journalists and
                self.n_media >= min_media)


@dataclass
class SubIndex:
    """Sub-index component of a dimension."""
    name: str
    value: float  # Normalized [0, 1]
    weight: float  # Should be 0.05 (5%)
    components: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    
    @property
    def weighted_value(self) -> float:
        """Get weighted contribution."""
        return self.value * self.weight
    
    def validate(self) -> bool:
        """Validate sub-index."""
        assert 0 <= self.value <= 1, f"Value must be [0,1], got {self.value}"
        assert abs(self.weight - 0.05) < 1e-6, f"Weight must be 0.05, got {self.weight}"
        return True


@dataclass
class Dimension:
    """Scoring dimension with 4 sub-indices."""
    name: str
    score: float  # Aggregated score [0, 1]
    weight: float  # Should be 0.20 (20%)
    sub_indices: List[SubIndex]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def weighted_score(self) -> float:
        """Get weighted contribution to total score."""
        return self.score * self.weight
    
    def calculate_score(self) -> float:
        """Calculate dimension score from sub-indices."""
        if not self.sub_indices:
            return 0.0
        total = sum(si.value for si in self.sub_indices)
        self.score = total / len(self.sub_indices)
        return self.score
    
    def validate(self) -> bool:
        """Validate dimension structure."""
        assert len(self.sub_indices) == 4, \
            f"Dimension {self.name} must have exactly 4 sub-indices, got {len(self.sub_indices)}"
        assert abs(self.weight - 0.20) < 1e-6, \
            f"Dimension weight must be 0.20, got {self.weight}"
        for si in self.sub_indices:
            si.validate()
        return True


@dataclass 
class CascadeMetrics:
    """Comprehensive metrics for a cascade."""
    # Temporal metrics
    onset_date: pd.Timestamp
    peak_date: pd.Timestamp
    end_date: pd.Timestamp
    duration_days: int
    
    # Velocity metrics
    initial_velocity: float
    peak_velocity: float
    acceleration: float
    momentum: float
    
    # Network metrics
    network_density: float
    clustering_coefficient: float
    avg_path_length: float
    centrality_scores: Dict[str, float] = field(default_factory=dict)
    
    # Participation metrics
    n_journalists: int = 0
    n_media: int = 0
    n_articles: int = 0
    
    # Diversity metrics
    journalist_diversity: float = 0.0
    media_diversity: float = 0.0
    source_diversity: float = 0.0
    
    # Content metrics
    frame_convergence: float = 0.0
    content_homogenization: float = 0.0
    sentiment_intensity: float = 0.0
    
    # Authority metrics
    epistemic_authorities: List[Tuple[str, float]] = field(default_factory=list)
    dominant_messengers: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'temporal': {
                'onset': self.onset_date.isoformat(),
                'peak': self.peak_date.isoformat(),
                'end': self.end_date.isoformat(),
                'duration_days': self.duration_days
            },
            'velocity': {
                'initial': self.initial_velocity,
                'peak': self.peak_velocity,
                'acceleration': self.acceleration,
                'momentum': self.momentum
            },
            'network': {
                'density': self.network_density,
                'clustering': self.clustering_coefficient,
                'path_length': self.avg_path_length
            },
            'participation': {
                'journalists': self.n_journalists,
                'media': self.n_media,
                'articles': self.n_articles
            },
            'diversity': {
                'journalist': self.journalist_diversity,
                'media': self.media_diversity,
                'source': self.source_diversity
            },
            'content': {
                'convergence': self.frame_convergence,
                'homogenization': self.content_homogenization,
                'sentiment': self.sentiment_intensity
            }
        }


@dataclass
class MediaCascade:
    """Complete media cascade with all analysis."""
    id: str
    candidate: CascadeCandidate
    dimensions: List[Dimension]
    total_score: float  # [0, 1]
    classification: str  # 'weak', 'moderate', 'strong', 'exceptional'
    confidence: float
    
    # Detailed metrics
    metrics: CascadeMetrics
    
    # Networks
    diffusion_network: Optional[nx.Graph] = None
    journalist_network: Optional[nx.Graph] = None
    entity_network: Optional[nx.Graph] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    validation: Dict[str, float] = field(default_factory=dict)
    is_significant: bool = False
    
    def validate_dimensions(self) -> bool:
        """Validate dimension structure."""
        assert len(self.dimensions) == 5, \
            f"Must have exactly 5 dimensions, got {len(self.dimensions)}"
        
        for dim in self.dimensions:
            dim.validate()
        
        # Check total weight
        total_weight = sum(d.weight for d in self.dimensions)
        assert abs(total_weight - 1.0) < 1e-6, \
            f"Dimension weights must sum to 1.0, got {total_weight}"
        
        return True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        return {
            'id': self.id,
            'frame': self.candidate.frame,
            'period': {
                'start': self.candidate.start_date.isoformat(),
                'end': self.candidate.end_date.isoformat(),
                'peak': self.candidate.peak_date.isoformat(),
                'duration_days': self.candidate.duration_days
            },
            'scores': {
                'total': self.total_score,
                'dimensions': {d.name: d.score for d in self.dimensions},
                'classification': self.classification,
                'confidence': self.confidence
            },
            'metrics': self.metrics.to_dict(),
            'validation': self.validation,
            'is_significant': self.is_significant,
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"MediaCascade(id={self.id}, frame={self.candidate.frame}, "
                f"score={self.total_score:.3f}, class={self.classification})")