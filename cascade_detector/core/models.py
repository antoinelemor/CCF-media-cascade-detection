"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
models.py

MAIN OBJECTIVE:
---------------
Data models for the minimalist cascade detection framework.
Three core dataclasses: BurstResult, CascadeResult, DetectionResults.

Author:
-------
Antoine Lemor
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Set, Tuple, Optional
import numpy as np
import pandas as pd
import json
from datetime import datetime


def _jsonify(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@dataclass
class BurstResult:
    """A detected burst in a frame's time series.

    A burst is a statistically significant deviation from baseline activity,
    detected via PELT changepoint detection + binomial proportion tests.
    A burst is a necessary but not sufficient condition for a cascade.
    """
    frame: str
    onset_date: pd.Timestamp
    peak_date: pd.Timestamp
    end_date: pd.Timestamp
    duration_days: int
    intensity: float            # peak_proportion / baseline_mean
    baseline_mean: float
    peak_proportion: float
    detection_method: str       # 'pelt', 'pelt_refined', 'sliding_prop'
    cohen_h: float = 0.0       # Cohen's h effect size from proportion test
    proptest_pvalue: float = 1.0  # p-value from binomial proportion test
    window_prop: float = 0.0   # observed proportion during burst window

    def to_dict(self) -> Dict[str, Any]:
        return {
            'frame': self.frame,
            'onset_date': self.onset_date.isoformat(),
            'peak_date': self.peak_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'duration_days': self.duration_days,
            'intensity': self.intensity,
            'baseline_mean': self.baseline_mean,
            'peak_proportion': self.peak_proportion,
            'detection_method': self.detection_method,
            'cohen_h': self.cohen_h,
            'proptest_pvalue': self.proptest_pvalue,
            'window_prop': self.window_prop,
        }


@dataclass
class EventOccurrence:
    """A distinct real-world event occurrence within a cascade.

    Identified via embedding-based clustering of articles sharing an event type.
    Each occurrence is a dated, semantically validated event with quality metrics.

    Soft assignment: every article contributes to clusters proportionally to its
    belonging score (semantic + temporal proximity). No hard threshold — the
    belonging dict maps doc_id → float in [0, 1].
    """
    occurrence_id: int
    event_type: str              # seed evt_* column name
    first_date: pd.Timestamp     # first article with belonging > 0
    last_date: pd.Timestamp      # last article with belonging > 0
    core_start: pd.Timestamp     # belonging-weighted percentile 10
    core_end: pd.Timestamp       # belonging-weighted percentile 90
    peak_date: pd.Timestamp      # belonging-weighted median (P50)
    n_articles: int              # articles with belonging > 0
    effective_mass: float        # Σ(belonging) — total weight of cluster
    core_mass: float             # Σ(belonging) for articles in core period
    semantic_coherence: float    # mean intra-cluster cosine similarity
    centroid: np.ndarray         # for inter-cascade comparison
    confidence: float = 0.0             # composite confidence [0, 1]
    confidence_components: Dict[str, float] = field(default_factory=dict)
    low_confidence: bool = False
    belonging: Dict = field(default_factory=dict)  # doc_id → float
    doc_ids: List = field(default_factory=list)     # article IDs in this occurrence
    seed_doc_ids: List = field(default_factory=list)  # Phase 2 core members (before soft-assignment)
    event_sentence_ids: Dict = field(default_factory=dict)  # doc_id → [sentence_ids] where evt_type=1
    cascade_attributions: List[Dict] = field(default_factory=list)  # CascadeAttribution dicts
    is_singleton: bool = False  # True if Phase 2 seed count was 1 (micro unique event)

    # Type validation (post-Phase 5)
    type_confidence: float = 1.0                    # assigned type score / best type score
    type_scores: Dict[str, float] = field(default_factory=dict)  # all candidate type scores

    # Strength metrics (Phase 3)
    media_count: int = 0                 # distinct media outlets covering this occurrence
    temporal_intensity: float = 0.0      # effective_mass / max(1, core_duration_days)
    emotional_intensity: float = 0.0     # belonging-weighted mean |sentiment_score|
    tone_coherence: float = 0.0          # 1 - entropy(pos, neg, neu) averaged per article
    entities: Set[str] = field(default_factory=set)  # entities with >= min citations

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        Excludes centroid (numpy array), belonging (too large),
        and event_sentence_ids (large).
        Includes doc_ids as a list for downstream analysis.
        """
        return _jsonify({
            'occurrence_id': self.occurrence_id,
            'event_type': self.event_type,
            'first_date': self.first_date.isoformat(),
            'last_date': self.last_date.isoformat(),
            'core_start': self.core_start.isoformat(),
            'core_end': self.core_end.isoformat(),
            'peak_date': self.peak_date.isoformat(),
            'n_articles': self.n_articles,
            'effective_mass': self.effective_mass,
            'core_mass': self.core_mass,
            'semantic_coherence': self.semantic_coherence,
            'confidence': self.confidence,
            'confidence_components': self.confidence_components,
            'low_confidence': self.low_confidence,
            'is_singleton': self.is_singleton,
            'type_confidence': self.type_confidence,
            'type_scores': self.type_scores,
            'doc_ids': [int(d) for d in self.doc_ids],
            'seed_doc_ids': [int(d) for d in self.seed_doc_ids],
            'n_event_sentences': sum(len(v) for v in self.event_sentence_ids.values()),
            'media_count': self.media_count,
            'temporal_intensity': self.temporal_intensity,
            'emotional_intensity': self.emotional_intensity,
            'tone_coherence': self.tone_coherence,
            'entities': sorted(self.entities),
        })


@dataclass
class EventCluster:
    """A cluster of event occurrences forming a meta-event.

    Detected in Phase 3 by pooling all occurrences across cascades and
    clustering them based on temporal, semantic, and entity proximity.
    Multi-type clusters (e.g., COP24 generating meeting + policy + publication)
    indicate coordinated real-world events.
    """
    cluster_id: int
    occurrences: List[EventOccurrence]
    event_types: Set[str]           # types present (inter-type if len > 1)
    peak_date: pd.Timestamp         # Article-level composite-weighted P50
    core_start: pd.Timestamp        # Article-level composite-weighted P10
    core_end: pd.Timestamp          # Article-level composite-weighted P90
    total_mass: float               # Σ(effective_mass) of occurrences
    centroid: np.ndarray            # mass-weighted mean of occurrence centroids
    n_occurrences: int
    is_multi_type: bool             # True if len(event_types) > 1
    strength: float = 0.0           # composite strength score
    strength_components: Dict[str, float] = field(default_factory=dict)
    entities: Set[str] = field(default_factory=set)  # shared entities
    dominant_type: str = ''                                    # most central event type
    type_structure: Dict[str, str] = field(default_factory=dict)  # {type: 'constitutive'|'satellite'}
    type_overlap_graph: Dict[str, Dict[str, float]] = field(default_factory=dict)  # type→{type: jaccard}
    type_ranking: List[Tuple[str, float]] = field(default_factory=list)  # [(type, score), ...] desc, all types

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        # Aggregate doc_ids and seed_doc_ids from occurrences
        all_doc_ids = set()
        all_seed_doc_ids = set()
        for occ in self.occurrences:
            for d in occ.doc_ids:
                try:
                    all_doc_ids.add(int(d))
                except (ValueError, TypeError):
                    all_doc_ids.add(d)
            for d in occ.seed_doc_ids:
                try:
                    all_seed_doc_ids.add(int(d))
                except (ValueError, TypeError):
                    all_seed_doc_ids.add(d)
        return _jsonify({
            'cluster_id': self.cluster_id,
            'occurrence_ids': [o.occurrence_id for o in self.occurrences],
            'event_types': sorted(self.event_types),
            'peak_date': self.peak_date.isoformat(),
            'core_start': self.core_start.isoformat(),
            'core_end': self.core_end.isoformat(),
            'total_mass': self.total_mass,
            'n_occurrences': self.n_occurrences,
            'n_articles': len(all_doc_ids),
            'is_multi_type': self.is_multi_type,
            'strength': self.strength,
            'strength_components': self.strength_components,
            'entities': sorted(self.entities),
            'dominant_type': self.dominant_type,
            'type_structure': self.type_structure,
            'type_overlap_graph': self.type_overlap_graph,
            'type_ranking': [{'type': t, 'score': round(s, 4)} for t, s in self.type_ranking],
            'doc_ids': sorted(all_doc_ids),
            'seed_doc_ids': sorted(all_seed_doc_ids),
        })


@dataclass
class CascadeAttribution:
    """Links an event occurrence to a cascade via article/temporal overlap.

    Attribution is computed post-detection: occurrences exist independently
    of cascades, and this dataclass records which cascades they feed into.
    """
    cascade_id: str
    occurrence_id: int
    shared_articles: int
    temporal_overlap_days: int
    overlap_ratio: float           # shared_articles / occurrence.n_articles

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify({
            'cascade_id': self.cascade_id,
            'occurrence_id': self.occurrence_id,
            'shared_articles': self.shared_articles,
            'temporal_overlap_days': self.temporal_overlap_days,
            'overlap_ratio': self.overlap_ratio,
        })


@dataclass
class CascadeResult:
    """A scored media cascade with continuous metrics across 4 dimensions.

    Every detected burst is scored on temporal dynamics, participation breadth,
    content convergence, and source convergence. The total_score determines
    the classification (strong/moderate/weak/not_cascade). No binary validation
    gate — the score itself decides.
    """
    # Identification
    cascade_id: str
    frame: str

    # Temporal boundaries
    onset_date: pd.Timestamp
    peak_date: pd.Timestamp
    end_date: pd.Timestamp
    duration_days: int

    # Participation
    n_articles: int
    n_journalists: int
    n_media: int
    n_new_journalists: int      # new to this frame (vs 90-day baseline)

    # Burst metrics
    burst_intensity: float
    adoption_velocity: float    # new journalists per day in growth phase
    baseline_mean: float
    peak_proportion: float

    # Scores (4 dimensions)
    score_temporal: float = 0.0
    score_participation: float = 0.0
    score_convergence: float = 0.0
    score_source: float = 0.0
    total_score: float = 0.0
    classification: str = 'not_cascade'
    sub_indices: Dict[str, float] = field(default_factory=dict)

    # Network (5 metrics)
    network_density: float = 0.0
    network_modularity: float = 0.0
    network_mean_degree: float = 0.0
    network_n_components: int = 0
    network_avg_clustering: float = 0.0

    # Semantic convergence (embedding-based, optional)
    semantic_similarity: float = 0.0        # intra-window cosine similarity
    convergence_trend: float = 0.0          # slope of similarity over time
    cross_media_alignment: float = 0.0      # inter-outlet centroid similarity
    novelty_decay_rate: float = 0.0         # rate at which new articles add less info

    # Source convergence (dimension 4)
    source_diversity_decline: float = 0.0   # diversity drop between 1st and 2nd half
    messenger_concentration: float = 0.0    # 1 - normalized entropy of messenger types
    media_coordination: float = 0.0         # cosine similarity between journalist messenger profiles

    # Multi-signal detection (unified detector)
    composite_peak: float = 0.0                          # max composite value during cascade
    daily_composite: Optional[pd.Series] = None          # composite time series (for paper figures)
    daily_signals: Optional[Dict[str, pd.Series]] = None # per-dimension Z-scores (for paper)
    detection_method: str = 'composite'                  # detection method label

    # Time series (for paper figures)
    daily_articles: Optional[pd.Series] = None
    daily_journalists: Optional[pd.Series] = None
    cumulative_journalists: Optional[pd.Series] = None

    # Context
    top_journalists: List[Tuple[str, int]] = field(default_factory=list)
    top_media: List[Tuple[str, int]] = field(default_factory=list)
    dominant_events: Dict[str, int] = field(default_factory=dict)
    dominant_messengers: Dict[str, int] = field(default_factory=dict)

    # Event occurrences (embedding-based clustering)
    event_occurrences: List['EventOccurrence'] = field(default_factory=list)
    event_occurrence_metrics: Dict[str, float] = field(default_factory=dict)
    daily_event_profile: Optional[pd.DataFrame] = None

    # Full data capture
    network_edges: List[Tuple[Any, Any, float]] = field(default_factory=list)
    convergence_metrics_full: Dict[str, float] = field(default_factory=dict)

    # Statistical test
    mann_whitney_p: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary (excludes pd.Series)."""
        return _jsonify({
            'cascade_id': self.cascade_id,
            'frame': self.frame,
            'onset_date': self.onset_date.isoformat(),
            'peak_date': self.peak_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'duration_days': self.duration_days,
            'n_articles': self.n_articles,
            'n_journalists': self.n_journalists,
            'n_media': self.n_media,
            'n_new_journalists': self.n_new_journalists,
            'burst_intensity': self.burst_intensity,
            'adoption_velocity': self.adoption_velocity,
            'baseline_mean': self.baseline_mean,
            'peak_proportion': self.peak_proportion,
            'score_temporal': self.score_temporal,
            'score_participation': self.score_participation,
            'score_convergence': self.score_convergence,
            'score_source': self.score_source,
            'total_score': self.total_score,
            'classification': self.classification,
            'sub_indices': self.sub_indices,
            'network_density': self.network_density,
            'network_modularity': self.network_modularity,
            'network_mean_degree': self.network_mean_degree,
            'network_n_components': self.network_n_components,
            'network_avg_clustering': self.network_avg_clustering,
            'semantic_similarity': self.semantic_similarity,
            'convergence_trend': self.convergence_trend,
            'cross_media_alignment': self.cross_media_alignment,
            'novelty_decay_rate': self.novelty_decay_rate,
            'composite_peak': self.composite_peak,
            'detection_method': self.detection_method,
            'source_diversity_decline': self.source_diversity_decline,
            'messenger_concentration': self.messenger_concentration,
            'media_coordination': self.media_coordination,
            'top_journalists': self.top_journalists,
            'top_media': self.top_media,
            'dominant_events': self.dominant_events,
            'dominant_messengers': self.dominant_messengers,
            'event_occurrences': [o.to_dict() for o in self.event_occurrences],
            'event_occurrence_metrics': self.event_occurrence_metrics,
            'mann_whitney_p': self.mann_whitney_p,
            'network_edges': [
                {'source': u, 'target': v, 'weight': w}
                for u, v, w in self.network_edges
            ],
            'convergence_metrics_full': self.convergence_metrics_full,
        })

    def __repr__(self) -> str:
        return (f"CascadeResult(id={self.cascade_id}, frame={self.frame}, "
                f"score={self.total_score:.3f}, class={self.classification})")


@dataclass
class DetectionResults:
    """Container for all detection results from a pipeline run."""
    cascades: List[CascadeResult]
    all_bursts: List[BurstResult]
    n_cascades_by_frame: Dict[str, int] = field(default_factory=dict)
    n_cascades_by_classification: Dict[str, int] = field(default_factory=dict)
    analysis_period: Tuple[str, str] = ('', '')
    n_articles_analyzed: int = 0
    runtime_seconds: float = 0.0
    detection_parameters: Dict[str, Any] = field(default_factory=dict)
    frame_signals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    event_impact: Optional[Any] = None  # StabSelImpactResults
    paradigm_impact: Optional[Any] = None  # StabSelParadigmResults
    event_clusters: List['EventCluster'] = field(default_factory=list)
    all_occurrences: List['EventOccurrence'] = field(default_factory=list)
    cascade_attributions: List['CascadeAttribution'] = field(default_factory=list)
    paradigm_shifts: Optional[Any] = None  # ParadigmShiftResults

    @staticmethod
    def _json_col(x):
        """Safely serialize any value to a JSON string for Parquet."""
        if x is None:
            return '{}'
        try:
            return json.dumps(x, default=str)
        except (TypeError, ValueError):
            return str(x)

    def to_dataframe(self) -> pd.DataFrame:
        """One row per cascade, all scores and metadata."""
        if not self.cascades:
            return pd.DataFrame()
        rows = [c.to_dict() for c in self.cascades]
        df = pd.DataFrame(rows)
        # Convert complex columns to JSON strings for Parquet compatibility
        for col in ['top_journalists', 'top_media', 'dominant_events',
                     'dominant_messengers', 'event_occurrences',
                     'event_occurrence_metrics', 'network_edges',
                     'sub_indices', 'convergence_metrics_full']:
            if col in df.columns:
                df[col] = df[col].apply(self._json_col)
        return df

    def to_json(self, path: str) -> None:
        """Export results to JSON file."""
        data = {
            'cascades': [c.to_dict() for c in self.cascades],
            'all_bursts': [b.to_dict() for b in self.all_bursts],
            'event_clusters': [ec.to_dict() for ec in self.event_clusters],
            'all_occurrences': [o.to_dict() for o in self.all_occurrences],
            'cascade_attributions': [a.to_dict() for a in self.cascade_attributions],
            'n_cascades_by_frame': self.n_cascades_by_frame,
            'n_cascades_by_classification': self.n_cascades_by_classification,
            'analysis_period': self.analysis_period,
            'n_articles_analyzed': self.n_articles_analyzed,
            'runtime_seconds': self.runtime_seconds,
            'detection_parameters': self.detection_parameters,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def summary(self) -> str:
        """Human-readable summary of results."""
        lines = [
            f"Cascade Detection Results",
            f"{'=' * 50}",
            f"Analysis period: {self.analysis_period[0]} to {self.analysis_period[1]}",
            f"Articles analyzed: {self.n_articles_analyzed:,}",
            f"Runtime: {self.runtime_seconds:.1f}s",
            f"",
            f"Bursts detected: {len(self.all_bursts)}",
            f"Cascades validated: {len(self.cascades)}",
            f"",
        ]

        if self.n_cascades_by_frame:
            lines.append("By frame:")
            for frame, count in sorted(self.n_cascades_by_frame.items()):
                lines.append(f"  {frame}: {count}")
            lines.append("")

        if self.n_cascades_by_classification:
            lines.append("By classification:")
            for cls, count in sorted(self.n_cascades_by_classification.items()):
                lines.append(f"  {cls}: {count}")
            lines.append("")

        if self.cascades:
            lines.append("Top cascades:")
            for c in sorted(self.cascades, key=lambda x: x.total_score, reverse=True)[:5]:
                lines.append(
                    f"  [{c.classification}] {c.frame} "
                    f"({c.onset_date.strftime('%Y-%m-%d')} to {c.end_date.strftime('%Y-%m-%d')}) "
                    f"score={c.total_score:.3f}"
                )

        return '\n'.join(lines)
