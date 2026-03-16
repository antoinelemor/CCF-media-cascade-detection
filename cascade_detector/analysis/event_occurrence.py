"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
event_occurrence.py

MAIN OBJECTIVE:
---------------
Database-first event occurrence detection. Events are detected on ALL articles
of the analysis period (not per-cascade), then attributed to cascades by
temporal and article overlap.

Phases:
  1. Build daily event profile (date × 8 evt_types) — descriptive
  2. Cluster per event type on ALL articles (agglomerative, cosine distance)
  3. Cluster mono-type occurrences into multi-type meta-events
  Merge: multi-type clusters replace their constituent mono-type clusters
  4. Iterative 4D assignment of ALL articles to ALL clusters (2 iterations)
  5. Build EventOccurrence objects with confidence metrics
  Attribution: Link occurrences to cascades by temporal + article overlap

Phase 4 distance (article → cluster):
  - Temporal (0.25): 1 - exp(-|date - peak| / 14)
  - Semantic (0.35): 1 - max(0, cos(emb_article, centroid))
  - Entity  (0.15): 1 - Jaccard(entities_article, entities_cluster); 0.5 if empty
  - Signal  (0.25): 1 - evt_*_mean for the cluster's event type

Author:
-------
Antoine Lemor
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from sklearn.metrics import silhouette_score
from scipy.spatial.distance import squareform
from scipy import sparse as sp

from cascade_detector.core.constants import (
    EVENT_MAIN,
    EVENT_COLUMNS,
    ENTITY_TYPES,
    TITLE_SENTENCE_ID,
    TITLE_WEIGHT,
    SEED_PERCENTILE,
    SEED_WEIGHT_TYPE,
    SEED_WEIGHT_GLOBAL,
    SEED_DOMINANT_RATIO,
    MIN_CLUSTER_SIZE,
    PHASE2_SEMANTIC_WEIGHT,
    PHASE2_TEMPORAL_WEIGHT,
    PHASE2_ENTITY_WEIGHT,
    PHASE4_TEMPORAL_WEIGHT,
    PHASE4_SEMANTIC_WEIGHT,
    PHASE4_ENTITY_WEIGHT,
    PHASE4_SIGNAL_WEIGHT,
    PHASE4_N_ITERATIONS,
    PHASE4_TEMPORAL_SCALE,
    EVENT_CLUSTER_TEMPORAL_WEIGHT,
    EVENT_CLUSTER_SEMANTIC_WEIGHT,
    EVENT_CLUSTER_ENTITY_WEIGHT,
    EVENT_CLUSTER_ARTICLE_WEIGHT,
    EVENT_CLUSTER_TYPE_WEIGHT,
    EVENT_CLUSTER_TEMPORAL_SCALE,
    EVENT_CLUSTER_MIN_ENTITY_CITATIONS,
    EVENT_CLUSTER_STRENGTH_MASS_WEIGHT,
    EVENT_CLUSTER_STRENGTH_COVERAGE_WEIGHT,
    EVENT_CLUSTER_STRENGTH_INTENSITY_WEIGHT,
    EVENT_CLUSTER_STRENGTH_COHERENCE_WEIGHT,
    EVENT_CLUSTER_STRENGTH_DIVERSITY_WEIGHT,
    EVENT_CLUSTER_TITLE_SIM_THRESHOLD,
    EVENT_CLUSTER_MAX_GAP_DAYS,
)
from cascade_detector.core.models import (
    EventOccurrence, EventCluster, CascadeAttribution,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants (module-level, non-configurable)
# =============================================================================

COSINE_DISTANCE_THRESHOLD = 0.30   # fcluster distance on compound metric
TEMPORAL_SCALE_DAYS = 14.0         # days separation for exponential decay
REGIME_SMOOTHING_WINDOW = 3        # days for rolling average
CORE_PERIOD_LOW = 10               # percentile for core period start
CORE_PERIOD_HIGH = 90              # percentile for core period end

# Confidence
LOW_CONFIDENCE_THRESHOLD = 0.40    # flag clusters below this


# =============================================================================
# Internal dataclass
# =============================================================================

@dataclass
class _RawCluster:
    """Internal cluster representation (not exported)."""
    cluster_id: int
    event_type: str       # seed evt_* type
    doc_ids: List[str] = field(default_factory=list)
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Belonging dict populated during assignment: doc_id → float
    belonging: Dict[str, float] = field(default_factory=dict)
    # Phase 2 seed doc_ids (preserved before soft-assignment inflation)
    seed_doc_ids: List[str] = field(default_factory=list)


# =============================================================================
# EventOccurrenceDetector
# =============================================================================

class EventOccurrenceDetector:
    """Detect distinct event occurrences via embedding clustering.

    Database-first: detects events on ALL articles of the period,
    then attributes them to cascades. Uses EmbeddingStore for semantic
    features in both clustering and assignment phases.

    When sentence-level data is available, uses event-filtered embeddings:
    for each article, only the sentences where the event type is active
    are mean-pooled, giving a more specific representation than the full
    article embedding. Falls back to full article embeddings if sentence
    data is not provided.
    """

    def __init__(self, embedding_store, sentence_df: Optional[pd.DataFrame] = None):
        """Initialize with an embedding store.

        Args:
            embedding_store: EmbeddingStore or MockEmbeddingStore instance.
            sentence_df: Optional sentence-level DataFrame with doc_id,
                sentence_id, and evt_* columns (binary 0/1). When provided,
                enables event-filtered embeddings for more precise clustering.
        """
        self.embedding_store = embedding_store
        self._evt_sentence_index = None
        if sentence_df is not None:
            self._build_event_sentence_index(sentence_df)

    def _build_event_sentence_index(self, sentence_df: pd.DataFrame) -> None:
        """Build index mapping (doc_id, evt_type) → [sentence_ids].

        Pre-computes which sentences in each article are labeled for each
        event type, avoiding repeated DataFrame lookups during clustering.
        """
        self._evt_sentence_index = {}

        # Ensure doc_id and sentence_id columns exist
        if 'doc_id' not in sentence_df.columns:
            return
        sid_col = 'sentence_id' if 'sentence_id' in sentence_df.columns else None
        if sid_col is None:
            # Try to find sentence identifier
            for col in ['sent_id', 'sentence']:
                if col in sentence_df.columns:
                    sid_col = col
                    break
        if sid_col is None:
            return

        for evt_type in EVENT_COLUMNS:
            if evt_type not in sentence_df.columns:
                continue
            # Select rows where this event type is active (binary = 1)
            evt_mask = sentence_df[evt_type] == 1
            if not evt_mask.any():
                continue
            evt_rows = sentence_df.loc[evt_mask, ['doc_id', sid_col]]
            for doc_id, grp in evt_rows.groupby('doc_id'):
                key = (doc_id, evt_type)
                self._evt_sentence_index[key] = grp[sid_col].tolist()

        n_entries = len(self._evt_sentence_index)
        n_docs = len(set(k[0] for k in self._evt_sentence_index))
        logger.info(
            f"Event sentence index: {n_entries} (doc_id, evt_type) entries "
            f"across {n_docs} articles"
        )

    # ------------------------------------------------------------------
    # Public API — database-first
    # ------------------------------------------------------------------

    def detect_events(
        self,
        articles: pd.DataFrame,
        entity_index: Optional[dict] = None,
    ) -> Tuple[List[EventOccurrence], List[EventCluster]]:
        """Database-first event detection on ALL articles.

        Args:
            articles: Article-level DataFrame with doc_id, date, evt_* columns.
            entity_index: Optional doc_id → [(entity_text, entity_type)] mapping.

        Returns:
            Tuple of (all_occurrences, event_clusters).
        """
        date_col = self._resolve_date_col(articles)

        # Phase 1: Daily event profile (descriptive)
        daily_profile = self._build_daily_event_profile(articles, date_col)

        # Phase 2: Cluster per event type on ALL articles
        all_clusters: List[_RawCluster] = []
        cluster_id_counter = 0

        for evt_type in EVENT_COLUMNS:
            evt_col = self._resolve_event_col(articles, evt_type)
            if evt_col is None:
                continue

            seeds, weights = self._select_seed_articles(articles, evt_col)
            if len(seeds) == 0:
                continue

            clusters = self._cluster_event_type(
                seeds, evt_type, evt_col, cluster_id_counter,
                date_col=date_col, seed_weights=weights,
                entity_index=entity_index,
            )
            cluster_id_counter += len(clusters)
            all_clusters.extend(clusters)

        if not all_clusters:
            logger.info("No event clusters found in Phase 2")
            return [], []

        n_singletons = sum(1 for c in all_clusters if len(c.seed_doc_ids) == 1)
        logger.info(
            f"Phase 2: {len(all_clusters)} mono-type clusters "
            f"({n_singletons} singletons)"
        )

        # Build temporary occurrences for Phase 3
        temp_occurrences = self._build_occurrences(
            all_clusters, articles, date_col
        )

        if not temp_occurrences:
            return [], []

        # Phase 3: Cluster mono-type occurrences into meta-events
        event_clusters = self.cluster_occurrences(
            temp_occurrences,
            entity_index=entity_index,
            articles=articles,
            embedding_store=self.embedding_store,
        )

        # Merge Phase 3 results back into raw clusters
        merged_clusters = self._merge_phase3_into_raw(
            all_clusters, event_clusters
        )

        logger.info(
            f"Post-merge: {len(merged_clusters)} clusters "
            f"({sum(1 for ec in event_clusters if ec.is_multi_type)} multi-type)"
        )

        # Phase 4: Iterative 4D assignment of ALL articles to ALL clusters
        self._assign_articles(
            merged_clusters, articles, date_col, entity_index
        )

        # Phase 5: Build definitive occurrences with confidence metrics
        occurrences = self._build_occurrences(
            merged_clusters, articles, date_col
        )

        # Post-validation: correct mistyped occurrences
        n_corrected = self._validate_occurrence_types(occurrences, articles)
        if n_corrected:
            logger.info(f"Type validation: corrected {n_corrected} occurrence(s)")

        # Remap event_clusters to reference final occurrences.
        # Clusters still hold refs to temp_occurrences (pre-Phase 5) whose IDs
        # are stale after renumbering. Match by seed_doc_ids signature.
        seed_to_final = {}
        for occ in occurrences:
            key = (occ.event_type, frozenset(occ.seed_doc_ids))
            seed_to_final[key] = occ
        n_remapped = 0
        n_dropped = 0
        for ec in event_clusters:
            new_occs = []
            for old_occ in ec.occurrences:
                key = (old_occ.event_type, frozenset(old_occ.seed_doc_ids))
                final_occ = seed_to_final.get(key)
                if final_occ is not None:
                    new_occs.append(final_occ)
                    n_remapped += 1
                else:
                    # No matching final occurrence — drop stale ref
                    n_dropped += 1
            ec.occurrences = new_occs
            ec.n_occurrences = len(new_occs)
            ec.event_types = {o.event_type for o in new_occs} if new_occs else ec.event_types
            ec.is_multi_type = len(ec.event_types) > 1 if new_occs else ec.is_multi_type
        # Remove empty clusters
        n_before = len(event_clusters)
        event_clusters[:] = [ec for ec in event_clusters if ec.occurrences]
        for i, ec in enumerate(event_clusters):
            ec.cluster_id = i
        logger.info(
            f"Remapped {n_remapped} occurrence refs in event clusters "
            f"(final ID space), dropped {n_dropped} stale refs, "
            f"removed {n_before - len(event_clusters)} empty clusters"
        )

        # Update type_ranking on clusters using final occurrences' type_scores
        self._update_cluster_type_rankings(event_clusters, occurrences)

        logger.info(
            f"Final: {len(occurrences)} event occurrences, "
            f"{len(event_clusters)} event clusters"
        )

        return occurrences, event_clusters

    def attribute_to_cascades(
        self,
        occurrences: List[EventOccurrence],
        cascades: list,
        articles: pd.DataFrame,
    ) -> List[CascadeAttribution]:
        """Attribute occurrences to cascades by temporal + article overlap.

        For each (cascade, occurrence) pair:
        1. Temporal overlap: intersection of core periods
        2. Articles shared: articles in cascade window with belonging > 0
        3. Attribution if temporal_overlap > 0 AND shared_articles >= 1

        Also populates cascade.event_occurrences and cascade.event_occurrence_metrics
        for backward compatibility.

        Args:
            occurrences: All detected EventOccurrence objects.
            cascades: List of CascadeResult objects.
            articles: Article-level DataFrame.

        Returns:
            List of CascadeAttribution objects.
        """
        if not occurrences or not cascades:
            return []

        date_col = self._resolve_date_col(articles)

        # Build date lookup
        if 'doc_id' in articles.columns:
            id_to_date = dict(zip(articles['doc_id'],
                                  pd.to_datetime(articles[date_col])))
        else:
            id_to_date = dict(zip(articles.index,
                                  pd.to_datetime(articles[date_col])))

        attributions = []

        # Pre-compute: for each occurrence, build a set of doc_ids
        # within its date range for fast intersection with cascade articles
        occ_doc_sets = []
        for occ in occurrences:
            occ_doc_sets.append(set(occ.doc_ids))

        # Pre-compute: for each cascade, build set of doc_ids in its window
        cascade_doc_sets = []
        for cascade in cascades:
            onset = pd.Timestamp(cascade.onset_date)
            end = pd.Timestamp(cascade.end_date)
            c_docs = set()
            for did, d in id_to_date.items():
                if onset <= d <= end:
                    c_docs.add(did)
            cascade_doc_sets.append(c_docs)

        for ci, cascade in enumerate(cascades):
            onset = pd.Timestamp(cascade.onset_date)
            end = pd.Timestamp(cascade.end_date)
            cascade_occs = []

            for oi, occ in enumerate(occurrences):
                # Temporal overlap (core periods)
                overlap_start = max(occ.core_start, onset)
                overlap_end = min(occ.core_end, end)
                temporal_overlap_days = max(0, (overlap_end - overlap_start).days)

                if temporal_overlap_days <= 0:
                    continue

                # Articles shared: set intersection (much faster than per-doc loop)
                shared = len(occ_doc_sets[oi] & cascade_doc_sets[ci])

                if shared < 1:
                    continue

                overlap_ratio = shared / max(1, occ.n_articles)

                attr = CascadeAttribution(
                    cascade_id=cascade.cascade_id,
                    occurrence_id=occ.occurrence_id,
                    shared_articles=shared,
                    temporal_overlap_days=temporal_overlap_days,
                    overlap_ratio=overlap_ratio,
                )
                attributions.append(attr)
                cascade_occs.append(occ)

                # Track attribution on occurrence
                occ.cascade_attributions.append(attr.to_dict())

            # Backward compatibility: populate cascade fields
            cascade.event_occurrences = cascade_occs
            cascade.event_occurrence_metrics = self._compute_cascade_metrics(
                cascade_occs
            )

        n_with = sum(1 for c in cascades if c.event_occurrences)
        total_attr = len(attributions)
        logger.info(
            f"Attribution: {total_attr} attributions across "
            f"{n_with}/{len(cascades)} cascades"
        )

        return attributions

    # ------------------------------------------------------------------
    # Backward-compatible wrappers
    # ------------------------------------------------------------------

    def detect_all(self, cascades: list, articles: pd.DataFrame) -> None:
        """Detect event occurrences for all cascades (backward-compatible).

        Calls detect_events() + attribute_to_cascades() under the hood.
        Modifies cascades in-place.

        Args:
            cascades: List of CascadeResult objects.
            articles: Article-level DataFrame with doc_id, date, evt_* columns.
        """
        if not cascades:
            return

        occurrences, event_clusters = self.detect_events(articles)
        self.attribute_to_cascades(occurrences, cascades, articles)

        n_with = sum(1 for c in cascades if c.event_occurrences)
        total_occ = sum(len(c.event_occurrences) for c in cascades)
        logger.info(
            f"Event occurrences: {total_occ} occurrences across "
            f"{n_with}/{len(cascades)} cascades"
        )

    def detect(self, cascade, burst_articles: pd.DataFrame,
               date_col: Optional[str] = None) -> None:
        """Detect event occurrences for a single cascade (backward-compatible).

        Uses database-first approach on the provided articles.
        Modifies cascade in-place.

        Args:
            cascade: CascadeResult to modify.
            burst_articles: Articles (typically within cascade window).
            date_col: Date column name (auto-detected if None).
        """
        if date_col is None:
            date_col = self._resolve_date_col(burst_articles)

        # Phase 1: Daily event profile
        daily_profile = self._build_daily_event_profile(burst_articles, date_col)
        cascade.daily_event_profile = daily_profile

        # Use detect_events on the provided articles
        occurrences, event_clusters = self.detect_events(burst_articles)

        if not occurrences:
            cascade.event_occurrences = []
            cascade.event_occurrence_metrics = self._compute_cascade_metrics([])
            return

        # For single-cascade backward compat, all occurrences go to this cascade
        cascade.event_occurrences = occurrences
        cascade.event_occurrence_metrics = self._compute_cascade_metrics(
            occurrences
        )

    # ------------------------------------------------------------------
    # Phase 1: Daily event profile
    # ------------------------------------------------------------------

    def _build_daily_event_profile(self, articles: pd.DataFrame,
                                   date_col: str) -> pd.DataFrame:
        """Build a DataFrame of daily event type proportions.

        Returns:
            DataFrame with date index and one column per event type,
            values are mean evt_* scores per day.
        """
        daily_data = {}
        dates = pd.to_datetime(articles[date_col])

        for evt_type in EVENT_COLUMNS:
            evt_col = self._resolve_event_col(articles, evt_type)
            if evt_col is not None:
                daily_data[evt_type] = articles.groupby(dates)[evt_col].mean()
            else:
                daily_data[evt_type] = pd.Series(
                    0.0, index=dates.unique()
                ).sort_index()

        profile = pd.DataFrame(daily_data)
        profile.index.name = 'date'
        return profile.fillna(0.0)

    # ------------------------------------------------------------------
    # Phase 2: Per-type clustering
    # ------------------------------------------------------------------

    def _resolve_event_col(self, articles: pd.DataFrame,
                           evt_type: str) -> Optional[str]:
        """Find the actual column name for an event type.

        Checks evt_type, evt_type_mean, evt_type_sum in order.
        """
        for suffix in ['', '_mean', '_sum']:
            col = f'{evt_type}{suffix}'
            if col in articles.columns:
                return col
        return None

    def _select_seed_articles(self, articles: pd.DataFrame,
                              evt_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Select seed articles using composite score for robust event clustering.

        Composite seed score:
            seed_score = SEED_WEIGHT_TYPE × evt_*_mean + SEED_WEIGHT_GLOBAL × event_mean

        Seeds = articles with evt_*_mean > 0 AND seed_score >= P50 of that group.
        The seed_score serves as weight for Phase 2 weighted centroids.

        Falls back to evt_*_mean-only scoring if event_mean column is absent.

        Args:
            articles: Full article DataFrame.
            evt_col: Event column name (e.g. 'evt_weather_mean').

        Returns:
            Tuple of (seed_articles DataFrame, weights array).
        """
        nonzero = articles[articles[evt_col] > 0]
        if nonzero.empty:
            return articles.iloc[:0].copy(), np.array([], dtype=np.float64)

        # Compute composite seed score
        event_mean_col = f'{EVENT_MAIN}_mean'
        if event_mean_col in articles.columns:
            seed_scores = (
                SEED_WEIGHT_TYPE * articles[evt_col].values +
                SEED_WEIGHT_GLOBAL * articles[event_mean_col].values
            )
        else:
            # Fallback: evt_*_mean only (backward compat for tests)
            seed_scores = articles[evt_col].values.copy()

        # Threshold: P50 of seed_score among articles with evt_*_mean > 0
        nonzero_mask = articles[evt_col].values > 0

        # Dominant ratio filter: exclude articles where this event type is
        # much weaker than the article's strongest type
        evt_cols_in_df = [
            c for c in [f'{et}_mean' for et in EVENT_COLUMNS]
            if c in articles.columns
        ]
        if len(evt_cols_in_df) > 1 and evt_col in evt_cols_in_df:
            all_evt_values = articles[evt_cols_in_df].values
            max_evt_per_article = all_evt_values.max(axis=1)
            this_evt_values = articles[evt_col].values
            dominant_mask = this_evt_values >= SEED_DOMINANT_RATIO * max_evt_per_article
            nonzero_mask = nonzero_mask & dominant_mask

        nonzero_scores = seed_scores[nonzero_mask]
        if len(nonzero_scores) == 0:
            return articles.iloc[:0].copy(), np.array([], dtype=np.float64)
        threshold = np.percentile(nonzero_scores, SEED_PERCENTILE)
        threshold = max(threshold, 1e-6)

        # Seeds = nonzero evt_*_mean AND above threshold (AND dominant ratio)
        seed_mask = nonzero_mask & (seed_scores >= threshold)

        seeds = articles[seed_mask].copy()
        weights = seed_scores[seed_mask].astype(np.float64)
        return seeds, weights

    def _get_event_filtered_embeddings(
        self, doc_ids: list, evt_type: str
    ) -> Tuple[np.ndarray, list]:
        """Get event-filtered embeddings for articles, integrating title signal.

        For each article:
        1. Mean-pools only the sentences where evt_type = 1 → emb_phrases
        2. Retrieves title embedding (sentence_id=0) → emb_title
        3. Blends: TITLE_WEIGHT * emb_title + (1-TITLE_WEIGHT) * emb_phrases

        Falls back gracefully: no title → emb_phrases only, no sentences → article embedding.

        Args:
            doc_ids: List of document IDs.
            evt_type: Event type name (e.g. 'evt_weather').

        Returns:
            Tuple of (embeddings array [n_found, dim], list of found doc_ids).
        """
        if self._evt_sentence_index is None:
            # No sentence data — fall back to full article embeddings
            return self.embedding_store.get_batch_article_embeddings(doc_ids)

        embeddings = []
        found_ids = []

        for doc_id in doc_ids:
            key = (doc_id, evt_type)
            sentence_ids = self._evt_sentence_index.get(key)

            if sentence_ids:
                # Mean-pool only event-relevant sentences
                emb_phrases = self.embedding_store.get_filtered_article_embedding(
                    doc_id, sentence_ids
                )
            else:
                # Fallback: full article embedding
                emb_phrases = self.embedding_store.get_article_embedding(doc_id)

            if emb_phrases is None:
                continue

            # Integrate title embedding
            emb_title = self.embedding_store.get_sentence_embedding(
                doc_id, TITLE_SENTENCE_ID
            )
            if emb_title is not None:
                emb = TITLE_WEIGHT * emb_title + (1.0 - TITLE_WEIGHT) * emb_phrases
            else:
                emb = emb_phrases

            embeddings.append(emb)
            found_ids.append(doc_id)

        if not embeddings:
            return np.empty((0, self.embedding_store.embedding_dim),
                            dtype=np.float32), []

        return np.array(embeddings, dtype=np.float32), found_ids

    def _cluster_event_type(self, seeds: pd.DataFrame, evt_type: str,
                            evt_col: str,
                            id_offset: int,
                            date_col: Optional[str] = None,
                            seed_weights: Optional[np.ndarray] = None,
                            entity_index: Optional[dict] = None) -> List[_RawCluster]:
        """Cluster seed articles for a single event type.

        Uses agglomerative clustering with average linkage on a compound
        distance that combines semantic, temporal, and entity dimensions.

        Compound distance = PHASE2_SEMANTIC_WEIGHT * cosine_dist
                          + PHASE2_TEMPORAL_WEIGHT * temporal_dist
                          + PHASE2_ENTITY_WEIGHT * entity_dist

        where temporal_dist = 1 - exp(-days_apart / TEMPORAL_SCALE_DAYS).

        Args:
            seeds: Seed articles (already filtered by threshold).
            evt_type: Event type name (e.g. 'evt_weather').
            evt_col: Actual column name in DataFrame.
            id_offset: Starting cluster ID.
            date_col: Date column name (auto-detected if None).
            seed_weights: Continuous weights for seeds (evt_*_mean values).

        Returns:
            List of _RawCluster objects (may be empty if insufficient embeddings).
        """
        doc_ids = seeds['doc_id'].tolist() if 'doc_id' in seeds.columns else seeds.index.tolist()

        # Get embeddings (event-filtered if sentence data available)
        embeddings, found_ids = self._get_event_filtered_embeddings(
            doc_ids, evt_type
        )
        if len(found_ids) == 0:
            return []

        # Resolve date column
        if date_col is None:
            date_col = self._resolve_date_col(seeds)

        # Build weight lookup from seed_weights
        weight_lookup = {}
        if seed_weights is not None and 'doc_id' in seeds.columns:
            for did, w in zip(seeds['doc_id'].tolist(), seed_weights):
                weight_lookup[did] = w

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = embeddings / norms

        if len(found_ids) == 1:
            w = weight_lookup.get(found_ids[0], 1.0)
            return [_RawCluster(
                cluster_id=id_offset,
                event_type=evt_type,
                doc_ids=found_ids,
                centroid=normalized[0],
                weights=np.array([w]),
                seed_doc_ids=list(found_ids),
            )]

        # Cosine distance (corpus-adjusted residual)
        raw_cosine_dists = pdist(normalized, metric='cosine')
        raw_cosine_dists = np.clip(raw_cosine_dists, 0, 2)
        baseline = self.embedding_store.compute_corpus_baseline()
        raw_sims = 1.0 - raw_cosine_dists
        residual_sims = np.clip((raw_sims - baseline) / (1.0 - baseline), 0.0, 1.0)
        cosine_dists = 1.0 - residual_sims

        # Temporal distance
        if 'doc_id' in seeds.columns:
            id_to_date = dict(zip(seeds['doc_id'], pd.to_datetime(seeds[date_col])))
        else:
            id_to_date = dict(zip(seeds.index, pd.to_datetime(seeds[date_col])))
        date_ordinals = np.array([
            id_to_date[did].toordinal() if did in id_to_date else 0
            for did in found_ids
        ], dtype=np.float64)
        temporal_dists_days = pdist(date_ordinals.reshape(-1, 1), metric='euclidean')
        temporal_dists = 1.0 - np.exp(-temporal_dists_days / TEMPORAL_SCALE_DAYS)

        # Entity distance (Jaccard) via condensed pairwise
        if entity_index:
            n_seeds = len(found_ids)
            entity_dists = np.zeros(n_seeds * (n_seeds - 1) // 2, dtype=np.float64)
            idx_e = 0
            for ii in range(n_seeds):
                ents_i = set(
                    f"{etype}:{etext}"
                    for etext, etype in entity_index.get(found_ids[ii], [])
                    if etype in ENTITY_TYPES
                )
                for jj in range(ii + 1, n_seeds):
                    ents_j = set(
                        f"{etype}:{etext}"
                        for etext, etype in entity_index.get(found_ids[jj], [])
                        if etype in ENTITY_TYPES
                    )
                    if ents_i or ents_j:
                        inter = len(ents_i & ents_j)
                        union = len(ents_i | ents_j)
                        entity_dists[idx_e] = 1.0 - (inter / union) if union > 0 else 0.5
                    else:
                        entity_dists[idx_e] = 0.5
                    idx_e += 1
        else:
            entity_dists = np.full_like(cosine_dists, 0.5)

        # Compound distance (3D)
        compound_dists = (
            PHASE2_SEMANTIC_WEIGHT * cosine_dists
            + PHASE2_TEMPORAL_WEIGHT * temporal_dists
            + PHASE2_ENTITY_WEIGHT * entity_dists
        )

        # Agglomerative clustering on compound distance
        Z = linkage(compound_dists, method='average')
        labels = fcluster(Z, t=COSINE_DISTANCE_THRESHOLD, criterion='distance')

        # Group articles by cluster label
        cluster_groups: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            cluster_groups.setdefault(label, []).append(idx)

        # Build clusters — singletons preserved as micro unique events
        clusters = []
        for label, indices in cluster_groups.items():

            cluster_doc_ids = [found_ids[i] for i in indices]
            cluster_embeddings = normalized[indices]

            # Weighted centroid
            cluster_weights = np.array([
                weight_lookup.get(found_ids[i],
                                  self._get_evt_weight(seeds, found_ids[i], evt_col))
                for i in indices
            ])
            weight_sum = cluster_weights.sum()
            if weight_sum > 0:
                centroid = (cluster_embeddings * cluster_weights[:, None]).sum(axis=0)
                centroid /= weight_sum
            else:
                centroid = cluster_embeddings.mean(axis=0)

            # Normalize centroid
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid /= c_norm

            clusters.append(_RawCluster(
                cluster_id=id_offset + len(clusters),
                event_type=evt_type,
                doc_ids=cluster_doc_ids,
                centroid=centroid,
                weights=cluster_weights,
                seed_doc_ids=list(cluster_doc_ids),
            ))

        return clusters

    def _get_evt_weight(self, seeds: pd.DataFrame, doc_id: str,
                        evt_col: str) -> float:
        """Get evt_* weight for a doc_id from seeds DataFrame."""
        if 'doc_id' in seeds.columns:
            match = seeds.loc[seeds['doc_id'] == doc_id, evt_col]
            if not match.empty:
                return float(match.iloc[0])
        return 1.0

    # ------------------------------------------------------------------
    # Phase 4: Iterative 4D article assignment
    # ------------------------------------------------------------------

    def _assign_articles(
        self,
        clusters: List[_RawCluster],
        articles: pd.DataFrame,
        date_col: str,
        entity_index: Optional[dict] = None,
    ) -> None:
        """Assign ALL articles to clusters using iterative 4D distance.

        Distance dimensions:
        - Temporal (0.25): 1 - exp(-|date - peak| / 14)
        - Semantic (0.35): 1 - max(0, cos(emb_article, centroid))
        - Entity  (0.15): 1 - Jaccard(entities_article, entities_cluster); 0.5 if empty
        - Signal  (0.25): 1 - evt_*_mean for the cluster's event type

        belonging = max(0, 1 - distance / threshold) where threshold is
        adaptive per cluster, derived from P90 of seed-to-centroid distances
        × 1.5. This ensures articles far from any seed get belonging=0.

        Runs PHASE4_N_ITERATIONS iterations, recalculating centroids
        and cluster properties between iterations.

        Modifies clusters in-place (belonging, doc_ids, centroid).

        Args:
            clusters: List of _RawCluster objects.
            articles: All articles in the analysis period.
            date_col: Date column name.
            entity_index: Optional doc_id → [(entity_text, entity_type)] mapping.
        """
        if not clusters:
            return

        import time as _time

        # --- Pre-computation (once) ---
        t0_pre = _time.time()

        # Date lookup
        doc_ids_all = articles['doc_id'].tolist() if 'doc_id' in articles.columns else articles.index.tolist()
        if 'doc_id' in articles.columns:
            id_to_date = dict(zip(articles['doc_id'],
                                  pd.to_datetime(articles[date_col])))
        else:
            id_to_date = dict(zip(articles.index,
                                  pd.to_datetime(articles[date_col])))

        # Batch embeddings for all articles
        logger.info(f"  Phase 4: loading {len(doc_ids_all)} article embeddings...")
        t0_emb = _time.time()
        embeddings_arr, emb_found_ids = self.embedding_store.get_batch_article_embeddings(
            doc_ids_all
        )
        logger.info(f"  Phase 4: embeddings loaded in {_time.time() - t0_emb:.1f}s "
                     f"({len(emb_found_ids)}/{len(doc_ids_all)} found)")
        # Normalize
        norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embeddings_arr = embeddings_arr / norms
        emb_lookup = dict(zip(emb_found_ids, embeddings_arr))

        # Entity set per article
        art_entities: Dict[str, set] = {}
        if entity_index:
            for doc_id in doc_ids_all:
                ents = entity_index.get(doc_id, [])
                art_entities[doc_id] = {
                    f"{etype}:{etext}" for etext, etype in ents
                    if etype in ENTITY_TYPES
                }

        # evt_*_mean per (doc_id, evt_type)
        article_evt_means: Dict[str, Dict[str, float]] = {}
        for doc_id in doc_ids_all:
            article_evt_means[doc_id] = {}
        for evt_type in EVENT_COLUMNS:
            evt_col = self._resolve_event_col(articles, evt_type)
            if evt_col is None:
                continue
            if 'doc_id' in articles.columns:
                for did, val in zip(articles['doc_id'], articles[evt_col]):
                    if did in article_evt_means:
                        article_evt_means[did][evt_type] = float(val)
            else:
                for did, val in zip(articles.index, articles[evt_col]):
                    if did in article_evt_means:
                        article_evt_means[did][evt_type] = float(val)

        # --- Iterative assignment ---
        belonging_matrix = None  # will be (n_art, n_cl) after first iteration

        for iteration in range(PHASE4_N_ITERATIONS):
            t_props = _time.time()

            # A. Compute cluster properties
            cluster_peak_dates: Dict[int, float] = {}  # ordinal
            cluster_entity_sets: Dict[int, set] = {}

            if iteration == 0:
                # Use seed articles for peak dates and entity sets
                for cluster in clusters:
                    dates_ord = []
                    wts = []
                    for did in cluster.seed_doc_ids:
                        d = id_to_date.get(did)
                        if d is not None:
                            dates_ord.append(d.toordinal())
                            w_val = article_evt_means.get(did, {}).get(
                                cluster.event_type, 0.0
                            )
                            wts.append(max(w_val, 0.01))
                    if dates_ord:
                        cluster_peak_dates[cluster.cluster_id] = self._weighted_percentile(
                            np.array(dates_ord, dtype=np.float64),
                            np.array(wts, dtype=np.float64),
                            50,
                        )
                    else:
                        all_dates_list = list(id_to_date.values())
                        if all_dates_list:
                            cluster_peak_dates[cluster.cluster_id] = float(
                                np.median([d.toordinal() for d in all_dates_list])
                            )

                    source_ids = cluster.seed_doc_ids
                    ent_counts: Dict[str, int] = {}
                    for did in source_ids:
                        for ent in art_entities.get(did, set()):
                            ent_counts[ent] = ent_counts.get(ent, 0) + 1
                    cluster_entity_sets[cluster.cluster_id] = {
                        ent for ent, cnt in ent_counts.items()
                        if cnt >= EVENT_CLUSTER_MIN_ENTITY_CITATIONS
                    }
            else:
                # Use belonging_matrix from previous iteration (pure numpy)
                # Peak dates: weighted median using belonging weights
                for j, cluster in enumerate(clusters):
                    bel_col = belonging_matrix[:, j]  # (n_art,)
                    mask = bel_col > 0
                    if mask.any():
                        cluster_peak_dates[cluster.cluster_id] = self._weighted_percentile(
                            art_date_ords[mask].astype(np.float64),
                            bel_col[mask].astype(np.float64),
                            50,
                        )

                # Entity sets: belonging-weighted counts via sparse matmul
                if art_ent_sp is not None and n_ent > 0:
                    # (n_cl, n_ent) = belonging.T @ art_ent_sp (dense @ sparse → dense)
                    ent_weights = belonging_matrix.T @ art_ent_sp  # np dense @ scipy sparse
                    # Threshold: entity needs >= EVENT_CLUSTER_MIN_ENTITY_CITATIONS
                    idx_to_ent_local = {v: k for k, v in ent_to_idx.items()}
                    for j, cl in enumerate(clusters):
                        cluster_entity_sets[cl.cluster_id] = set()
                        ent_row = ent_weights[j]
                        above = np.where(ent_row >= EVENT_CLUSTER_MIN_ENTITY_CITATIONS)[0]
                        if len(above) > 0:
                            cluster_entity_sets[cl.cluster_id] = {
                                idx_to_ent_local[i] for i in above
                            }
                else:
                    for cl in clusters:
                        cluster_entity_sets[cl.cluster_id] = set()

            # B. Compute 4D distance → belonging matrix (pure numpy, no scatter)
            peak_ord_default = float(np.median([
                d.toordinal() for d in id_to_date.values()
            ])) if id_to_date else 0.0

            # One-time pre-computation on first iteration
            if iteration == 0:
                valid_ids = [did for did in doc_ids_all
                             if did in emb_lookup and did in id_to_date]
                if not valid_ids:
                    break
                n_art = len(valid_ids)
                valid_id_to_row = {did: i for i, did in enumerate(valid_ids)}

                # Article embeddings: (n_art, dim) float32
                art_emb_matrix = np.array(
                    [emb_lookup[did] for did in valid_ids], dtype=np.float32
                )
                # Article date ordinals: (n_art,) float32
                art_date_ords = np.array(
                    [float(id_to_date[did].toordinal()) for did in valid_ids],
                    dtype=np.float32,
                )
                # Signal vectors per event type: {evt_type: (n_art,)} float32
                evt_type_to_signal = {}
                for evt_type in set(cl.event_type for cl in clusters):
                    evt_type_to_signal[evt_type] = np.array([
                        article_evt_means.get(did, {}).get(evt_type, 0.0)
                        for did in valid_ids
                    ], dtype=np.float32)

                # Entity: encode as binary matrices for vectorized Jaccard
                all_entity_strs: set = set()
                art_ent_sets = []
                for did in valid_ids:
                    s = art_entities.get(did, set())
                    art_ent_sets.append(s)
                    all_entity_strs.update(s)
                for cl in clusters:
                    all_entity_strs.update(
                        cluster_entity_sets.get(cl.cluster_id, set())
                    )

                if all_entity_strs:
                    ent_to_idx = {e: i for i, e in enumerate(all_entity_strs)}
                    n_ent = len(ent_to_idx)
                    # Build sparse CSR matrix for articles
                    rows_sp, cols_sp = [], []
                    for i, s in enumerate(art_ent_sets):
                        for e in s:
                            rows_sp.append(i)
                            cols_sp.append(ent_to_idx[e])
                    art_ent_sp = sp.csr_matrix(
                        (np.ones(len(rows_sp), dtype=np.float32),
                         (rows_sp, cols_sp)),
                        shape=(n_art, n_ent),
                    )
                    art_ent_counts = np.asarray(art_ent_sp.sum(axis=1)).ravel()
                else:
                    n_ent = 0
                    art_ent_sp = None
                    art_ent_counts = np.zeros(n_art, dtype=np.float32)

            n_cl = len(clusters)
            t_iter_start = _time.time()
            logger.info(
                f"  Phase 4 iteration {iteration + 1}: "
                f"{n_art} articles × {n_cl} clusters "
                f"(n_ent={n_ent if art_ent_sp is not None else 0}, "
                f"props={_time.time() - t_props:.1f}s)"
            )

            # --- Centroids matrix: (n_cl, dim) float32, normalized ---
            centroid_matrix = np.array(
                [cl.centroid for cl in clusters], dtype=np.float32
            )
            c_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
            centroid_matrix /= np.maximum(c_norms, 1e-10)

            # Cluster peak ordinals: (n_cl,) float32
            cl_peak_ords = np.array([
                cluster_peak_dates.get(cl.cluster_id, peak_ord_default)
                for cl in clusters
            ], dtype=np.float32)

            # --- 1. Semantic distance: single matmul (n_art, n_cl), corpus-adjusted ---
            t1 = _time.time()
            raw_sim = np.clip(art_emb_matrix @ centroid_matrix.T, 0.0, None)
            baseline = self.embedding_store.compute_corpus_baseline()
            residual_sim = np.clip(
                (raw_sim - baseline) / (1.0 - baseline), 0.0, 1.0
            )
            semantic_dist = 1.0 - residual_sim
            t2 = _time.time()

            # --- 2. Temporal distance: (n_art, n_cl) ---
            temporal_dist = 1.0 - np.exp(
                -np.abs(art_date_ords[:, None] - cl_peak_ords[None, :])
                / np.float32(PHASE4_TEMPORAL_SCALE)
            )
            t3 = _time.time()

            # --- 3. Signal distance: (n_art, n_cl) ---
            cl_evt_types = [cl.event_type for cl in clusters]
            signal_dist = np.column_stack([
                1.0 - evt_type_to_signal.get(et, np.zeros(n_art, dtype=np.float32))
                for et in cl_evt_types
            ]) if n_cl > 0 else np.ones((n_art, 0), dtype=np.float32)
            t4 = _time.time()

            # --- 4. Entity distance: vectorized Jaccard via sparse matmul ---
            entity_dist = np.full((n_art, n_cl), 0.5, dtype=np.float32)
            if art_ent_sp is not None and n_ent > 0:
                # Build sparse cluster entity matrix
                cl_rows, cl_cols = [], []
                for j, cl in enumerate(clusters):
                    for e in cluster_entity_sets.get(cl.cluster_id, set()):
                        idx = ent_to_idx.get(e)
                        if idx is not None:
                            cl_rows.append(j)
                            cl_cols.append(idx)
                cl_ent_sp = sp.csr_matrix(
                    (np.ones(len(cl_rows), dtype=np.float32),
                     (cl_rows, cl_cols)),
                    shape=(n_cl, n_ent),
                )
                cl_ent_counts = np.asarray(cl_ent_sp.sum(axis=1)).ravel()
                # Sparse matmul: (n_art, n_ent) @ (n_ent, n_cl) → dense (n_art, n_cl)
                intersection = np.asarray((art_ent_sp @ cl_ent_sp.T).todense())
                union = (art_ent_counts[:, None]
                         + cl_ent_counts[None, :]
                         - intersection)
                has_ent = union > 0
                entity_dist[has_ent] = (
                    1.0 - intersection[has_ent].astype(np.float32)
                    / union[has_ent].astype(np.float32)
                )
            t5 = _time.time()

            logger.info(
                f"    distances: semantic={t2-t1:.1f}s temporal={t3-t2:.1f}s "
                f"signal={t4-t3:.1f}s entity={t5-t4:.1f}s"
            )

            # --- Compound distance (pure numpy) ---
            distance_matrix = (
                PHASE4_TEMPORAL_WEIGHT * temporal_dist +
                PHASE4_SEMANTIC_WEIGHT * semantic_dist +
                PHASE4_ENTITY_WEIGHT * entity_dist +
                PHASE4_SIGNAL_WEIGHT * signal_dist
            )

            # --- Adaptive threshold per cluster ---
            # Iteration 0: bootstrap from seed distances
            # Iteration 1+: recalibrate from cluster core (articles with
            #   belonging >= P75), letting the cluster define its own boundary.
            THRESHOLD_MULTIPLIER = 2.0
            THRESHOLD_CAP = 0.5
            CORE_PERCENTILE = 75  # P75 of belonging defines core membership
            cluster_thresholds = np.empty(n_cl, dtype=np.float32)

            if iteration == 0:
                # Bootstrap: use seed distances
                for j, cl in enumerate(clusters):
                    seed_rows = [valid_id_to_row[did] for did in cl.seed_doc_ids
                                 if did in valid_id_to_row]
                    if len(seed_rows) >= 3:
                        seed_dists = distance_matrix[seed_rows, j]
                        cluster_thresholds[j] = np.median(seed_dists) * THRESHOLD_MULTIPLIER
                    elif seed_rows:
                        seed_dists = distance_matrix[seed_rows, j]
                        cluster_thresholds[j] = np.max(seed_dists) * THRESHOLD_MULTIPLIER
                    else:
                        cluster_thresholds[j] = 0.3
            else:
                # Recalibrate: use core members from previous belonging
                prev_belonging = belonging_matrix  # from previous iteration
                for j, cl in enumerate(clusters):
                    col_bel = prev_belonging[:, j]
                    pos_mask = col_bel > 0
                    if pos_mask.sum() >= 3:
                        # Core = articles with belonging >= P75 of positive entries
                        bel_threshold = np.percentile(col_bel[pos_mask], CORE_PERCENTILE)
                        core_rows = np.where(col_bel >= bel_threshold)[0]
                        core_dists = distance_matrix[core_rows, j]
                        cluster_thresholds[j] = np.median(core_dists) * THRESHOLD_MULTIPLIER
                    else:
                        # Fallback to seeds if cluster has too few members
                        seed_rows = [valid_id_to_row[did] for did in cl.seed_doc_ids
                                     if did in valid_id_to_row]
                        if seed_rows:
                            seed_dists = distance_matrix[seed_rows, j]
                            cluster_thresholds[j] = np.median(seed_dists) * THRESHOLD_MULTIPLIER
                        else:
                            cluster_thresholds[j] = 0.3

            # Floor and cap
            cluster_thresholds = np.clip(cluster_thresholds, 0.05, THRESHOLD_CAP)

            # belonging = max(0, 1 - distance / threshold)
            belonging_matrix = np.maximum(
                np.float32(0.0),
                1.0 - distance_matrix / cluster_thresholds[None, :]
            )

            n_nonzero = int((belonging_matrix > 0).sum())
            logger.info(
                f"    {n_nonzero:,} nonzero / {n_art * n_cl:,} total "
                f"(thresholds: mean={cluster_thresholds.mean():.3f} "
                f"min={cluster_thresholds.min():.3f} max={cluster_thresholds.max():.3f}) "
                f"({_time.time() - t_iter_start:.1f}s)"
            )

            # D. Recalculate centroids: single matmul
            #    new_centroids = (belonging * signal_weight).T @ art_emb / sum_weights
            #    Build weight matrix: (n_art, n_cl) = belonging * max(signal, 0.01)
            signal_vals = np.column_stack([
                evt_type_to_signal.get(et, np.zeros(n_art, dtype=np.float32))
                for et in cl_evt_types
            ]) if n_cl > 0 else np.zeros((n_art, 0), dtype=np.float32)
            weight_matrix = belonging_matrix * np.maximum(signal_vals, np.float32(0.01))
            w_sums = weight_matrix.sum(axis=0)  # (n_cl,)
            # new_centroids = weight_matrix.T @ art_emb_matrix → (n_cl, dim)
            new_centroids = weight_matrix.T @ art_emb_matrix
            # Normalize: divide by weight sums, then L2-normalize
            active = w_sums > 0
            if active.any():
                new_centroids[active] /= w_sums[active, None]
                norms = np.linalg.norm(new_centroids[active], axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                new_centroids[active] /= norms
            # Write back to clusters
            for j, cluster in enumerate(clusters):
                if active[j]:
                    cluster.centroid = new_centroids[j].astype(np.float64)

            logger.info(
                f"    Centroids updated ({_time.time() - t_iter_start:.1f}s total)"
            )

        # --- Final scatter: belonging_matrix → cluster dicts (once) ---
        # With adaptive thresholds, belonging values naturally reach 0 for
        # distant articles. We collect only articles with belonging > 0,
        # ensuring seeds are always included.
        t_scatter = _time.time()
        n_kept_total = 0

        for j, cluster in enumerate(clusters):
            col_vals = belonging_matrix[:, j]
            # Articles with positive belonging
            mask = col_vals > 0
            # Ensure seeds are always included
            for did in cluster.seed_doc_ids:
                r = valid_id_to_row.get(did)
                if r is not None:
                    mask[r] = True

            rows_j = np.where(mask)[0]
            vals_j = col_vals[rows_j].astype(np.float64)
            dids_j = [valid_ids[r] for r in rows_j]
            cluster.doc_ids = dids_j
            cluster.belonging = dict(zip(dids_j, vals_j.tolist()))
            n_kept_total += len(rows_j)

        logger.info(
            f"  Phase 4 scatter: {n_kept_total:,} entries kept / "
            f"{n_art * n_cl:,} total "
            f"(adaptive thresholds, "
            f"mean threshold={cluster_thresholds.mean():.3f}) "
            f"in {_time.time() - t_scatter:.1f}s"
        )

    # ------------------------------------------------------------------
    # Phase 3 merge
    # ------------------------------------------------------------------

    def _merge_phase3_into_raw(
        self,
        mono_clusters: List[_RawCluster],
        event_clusters: List[EventCluster],
    ) -> List[_RawCluster]:
        """Merge Phase 3 multi-type clusters back into raw clusters.

        For each multi-type EventCluster:
        - Creates a new _RawCluster with fused centroid and dominant_type
        - Removes the constituent mono-type _RawClusters

        Mono-type EventClusters keep their original _RawCluster.

        Args:
            mono_clusters: Original Phase 2 mono-type clusters.
            event_clusters: Phase 3 event clusters.

        Returns:
            Merged list of _RawCluster objects.
        """
        # Build lookup: occurrence_id → _RawCluster
        occ_to_raw: Dict[int, _RawCluster] = {}
        for rc in mono_clusters:
            occ_to_raw[rc.cluster_id] = rc

        # Track which raw clusters to remove (absorbed into multi-type)
        absorbed_ids = set()
        new_clusters = []

        for ec in event_clusters:
            if not ec.is_multi_type:
                continue

            # Collect constituent raw cluster IDs
            constituent_ids = set()
            merged_seed_ids = []
            merged_doc_ids = []
            for occ in ec.occurrences:
                constituent_ids.add(occ.occurrence_id)
                merged_seed_ids.extend(occ.seed_doc_ids)
                merged_doc_ids.extend(occ.doc_ids)

            absorbed_ids.update(constituent_ids)

            # Create merged _RawCluster
            merged = _RawCluster(
                cluster_id=-1,  # will be renumbered
                event_type=ec.dominant_type,
                doc_ids=list(set(merged_doc_ids)),
                centroid=ec.centroid.copy(),
                weights=np.array([]),
                seed_doc_ids=list(set(merged_seed_ids)),
            )
            new_clusters.append(merged)

        # Keep non-absorbed mono-type clusters
        result = [rc for rc in mono_clusters if rc.cluster_id not in absorbed_ids]
        result.extend(new_clusters)

        # Renumber
        for i, rc in enumerate(result):
            rc.cluster_id = i

        return result

    # ------------------------------------------------------------------
    # Belonging score computation (kept for title integration tests)
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_percentile(values: np.ndarray, weights: np.ndarray,
                             p: float) -> float:
        """Compute weighted percentile.

        Args:
            values: Array of values.
            weights: Array of weights (same length).
            p: Percentile in [0, 100].

        Returns:
            Weighted percentile value.
        """
        if len(values) == 0:
            return 0.0
        if len(values) == 1:
            return float(values[0])

        # Sort by values
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[sorted_idx]

        # Cumulative weight
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights[-1]
        if total_weight < 1e-10:
            return float(np.percentile(values, p))

        # Normalized cumulative weights (centered at midpoint of each interval)
        cum_frac = (cum_weights - 0.5 * sorted_weights) / total_weight

        # Interpolate
        target = p / 100.0
        return float(np.interp(target, cum_frac, sorted_values))

    # ------------------------------------------------------------------
    # Phase 5: Build occurrences and metrics
    # ------------------------------------------------------------------

    def _build_occurrences(self, clusters: List[_RawCluster],
                           articles: pd.DataFrame,
                           date_col: str) -> List[EventOccurrence]:
        """Build EventOccurrence objects from clusters with belonging scores.

        Args:
            clusters: Final clusters with belonging dicts.
            articles: All articles.
            date_col: Date column name.

        Returns:
            List of EventOccurrence objects sorted by first_date.
        """
        # Build date lookup
        if 'doc_id' in articles.columns:
            id_to_date = dict(zip(articles['doc_id'],
                                  pd.to_datetime(articles[date_col])))
        else:
            id_to_date = dict(zip(articles.index,
                                  pd.to_datetime(articles[date_col])))

        occurrences = []

        for cluster in clusters:
            if not cluster.doc_ids:
                continue
            # If no belonging dict, use seed_doc_ids with weight 1.0
            if not cluster.belonging:
                cluster.belonging = {did: 1.0 for did in cluster.doc_ids}

            # Collect dates and belonging weights
            dates = []
            belongings = []
            valid_doc_ids = []
            for did in cluster.doc_ids:
                d = id_to_date.get(did)
                b = cluster.belonging.get(did, 0.0)
                if d is not None and b > 0:
                    dates.append(d)
                    belongings.append(b)
                    valid_doc_ids.append(did)

            if not valid_doc_ids:
                continue

            dates_ts = pd.Series(dates)
            belongings_arr = np.array(belongings, dtype=np.float64)

            # Temporal bounds (belonging-weighted)
            first_date = dates_ts.min()
            last_date = dates_ts.max()

            date_ordinals = np.array([d.toordinal() for d in dates],
                                     dtype=np.float64)

            if len(dates) >= 2:
                core_start_ord = self._weighted_percentile(
                    date_ordinals, belongings_arr, CORE_PERIOD_LOW
                )
                core_end_ord = self._weighted_percentile(
                    date_ordinals, belongings_arr, CORE_PERIOD_HIGH
                )
                core_start = pd.Timestamp.fromordinal(int(core_start_ord))
                core_end = pd.Timestamp.fromordinal(int(core_end_ord))
            else:
                core_start = first_date
                core_end = last_date

            # Peak date: belonging-weighted median (P50)
            peak_ord = self._weighted_percentile(
                date_ordinals, belongings_arr, 50
            )
            peak_date = pd.Timestamp.fromordinal(int(peak_ord))

            # Effective mass and core mass
            effective_mass = float(belongings_arr.sum())
            core_mask = np.array([
                core_start <= d <= core_end for d in dates
            ])
            core_mass = float(belongings_arr[core_mask].sum()) if core_mask.any() else 0.0

            # Semantic coherence (mean pairwise cosine similarity)
            semantic_coherence = self.embedding_store.mean_pairwise_similarity(
                valid_doc_ids
            )

            # Confidence score (5 sub-signals, equal weight)
            centroid_tightness = float(np.mean(belongings_arr))

            # Semantic coherence RESIDUAL (above corpus baseline)
            baseline = self.embedding_store.compute_corpus_baseline()
            coherence_residual = (
                max(0.0, (semantic_coherence - baseline) / (1.0 - baseline))
                if baseline < 1.0 else 0.0
            )

            # Media diversity: 1 - 1/n_media (0 for mono-source)
            n_media = 1
            if articles is not None:
                media_col = next(
                    (c for c in ['media', 'media_first'] if c in articles.columns), None
                )
                doc_col = 'doc_id' if 'doc_id' in articles.columns else None
                if media_col and doc_col:
                    cluster_arts = articles[articles[doc_col].isin(valid_doc_ids)]
                    n_media = max(1, cluster_arts[media_col].dropna().nunique())
            media_diversity = 1.0 - 1.0 / n_media

            # Recruitment success: ratio of total articles to seeds
            seed_set = set(cluster.seed_doc_ids)
            n_seeds = max(1, sum(1 for did in valid_doc_ids if did in seed_set))
            n_total = len(valid_doc_ids)
            recruitment_ratio = n_total / n_seeds
            recruitment_success = min(1.0, max(0.0, (recruitment_ratio - 1.0) / 2.0))

            size_adequacy = min(1.0, effective_mass / 10.0)

            confidence = (
                0.20 * centroid_tightness +
                0.20 * coherence_residual +
                0.20 * media_diversity +
                0.20 * recruitment_success +
                0.20 * size_adequacy
            )

            confidence_components = {
                'centroid_tightness': centroid_tightness,
                'coherence_residual': coherence_residual,
                'media_diversity': media_diversity,
                'recruitment_success': recruitment_success,
                'size_adequacy': size_adequacy,
            }

            low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

            # Event-specific sentence_ids
            evt_sids: Dict[str, list] = {}
            if self._evt_sentence_index is not None:
                for did in valid_doc_ids:
                    key = (did, cluster.event_type)
                    sids = self._evt_sentence_index.get(key)
                    if sids:
                        evt_sids[did] = sids

            is_singleton = len(cluster.seed_doc_ids) == 1

            occurrences.append(EventOccurrence(
                occurrence_id=cluster.cluster_id,
                event_type=cluster.event_type,
                first_date=first_date,
                last_date=last_date,
                core_start=core_start,
                core_end=core_end,
                peak_date=peak_date,
                n_articles=len(valid_doc_ids),
                effective_mass=effective_mass,
                core_mass=core_mass,
                semantic_coherence=semantic_coherence,
                centroid=cluster.centroid,
                confidence=confidence,
                confidence_components=confidence_components,
                low_confidence=low_confidence,
                belonging=dict(cluster.belonging),
                doc_ids=list(valid_doc_ids),
                seed_doc_ids=list(cluster.seed_doc_ids),
                event_sentence_ids=evt_sids,
                is_singleton=is_singleton,
            ))

        # Sort by first_date
        occurrences.sort(key=lambda o: o.first_date)

        # Renumber occurrence_id sequentially
        for i, occ in enumerate(occurrences):
            occ.occurrence_id = i

        return occurrences

    def _validate_occurrence_types(
        self,
        occurrences: List[EventOccurrence],
        articles: pd.DataFrame,
    ) -> int:
        """Post-validate occurrence types and correct mistyped ones.

        For each occurrence, computes a composite type_score per candidate type:
          type_score[t] = 0.50 * norm_evt_mean[t] + 0.50 * title_evt_sim[t]

        Signal 1 (norm_evt_mean): belonging-weighted mean of evt_*_mean across
        articles, normalized so the best candidate = 1.0.

        Signal 2 (title_evt_sim): cosine similarity between the occurrence's
        title centroid and the mean embedding of event-labeled sentences for
        each candidate type. Measures whether the titles are about that type.

        Correction: retype if assigned type_score < 50% of the best candidate
        AND the best candidate score > 0.

        Returns:
            Number of occurrences whose type was corrected.
        """
        if not occurrences:
            return 0

        # Resolve event columns
        evt_cols = {}
        for evt_type in EVENT_COLUMNS:
            col = self._resolve_event_col(articles, evt_type)
            if col is not None:
                evt_cols[evt_type] = col

        if not evt_cols:
            return 0

        # Build doc_id → article row lookup
        doc_col = 'doc_id' if 'doc_id' in articles.columns else None
        if doc_col:
            art_lookup = articles.set_index('doc_id')
        else:
            art_lookup = articles

        n_corrected = 0

        for occ in occurrences:
            # --- Signal 1: belonging-weighted mean of evt_* per candidate type ---
            evt_means: Dict[str, float] = {}
            total_belonging = 0.0

            for did, bel in occ.belonging.items():
                if bel <= 0 or did not in art_lookup.index:
                    continue
                row = art_lookup.loc[did]
                total_belonging += bel
                for evt_type, col in evt_cols.items():
                    val = float(row.get(col, 0.0)) if hasattr(row, 'get') else 0.0
                    evt_means[evt_type] = evt_means.get(evt_type, 0.0) + bel * val

            if total_belonging > 0:
                for t in evt_means:
                    evt_means[t] /= total_belonging

            max_evt_mean = max(evt_means.values()) if evt_means else 0.0

            # Normalize so best = 1.0
            norm_evt_means: Dict[str, float] = {}
            for t in evt_cols:
                raw = evt_means.get(t, 0.0)
                norm_evt_means[t] = raw / max_evt_mean if max_evt_mean > 0 else 0.0

            # --- Signal 2: title centroid ↔ event-sentence similarity ---
            title_evt_sims: Dict[str, float] = {}

            # Compute title centroid from seed articles
            seeds = occ.seed_doc_ids if occ.seed_doc_ids else occ.doc_ids
            title_embs = []
            for did in seeds:
                e = self.embedding_store.get_sentence_embedding(did, TITLE_SENTENCE_ID)
                if e is not None:
                    title_embs.append(e)

            if title_embs:
                title_centroid = np.mean(title_embs, axis=0).astype(np.float32)
                t_norm = np.linalg.norm(title_centroid)
                if t_norm > 0:
                    title_centroid /= t_norm
                else:
                    title_centroid = None
            else:
                title_centroid = None

            if title_centroid is not None and self._evt_sentence_index is not None:
                for evt_type in evt_cols:
                    # Gather event-labeled sentence embeddings from seed articles
                    evt_embs = []
                    for did in seeds:
                        sids = self._evt_sentence_index.get((did, evt_type))
                        if sids:
                            emb = self.embedding_store.get_filtered_article_embedding(
                                did, sids
                            )
                            if emb is not None:
                                evt_embs.append(emb)

                    if evt_embs:
                        evt_centroid = np.mean(evt_embs, axis=0).astype(np.float32)
                        e_norm = np.linalg.norm(evt_centroid)
                        if e_norm > 0:
                            evt_centroid /= e_norm
                            title_evt_sims[evt_type] = float(
                                np.dot(title_centroid, evt_centroid)
                            )
                        else:
                            title_evt_sims[evt_type] = 0.0
                    else:
                        title_evt_sims[evt_type] = 0.0
            else:
                # No title embeddings or no sentence index → signal 2 is 0
                for evt_type in evt_cols:
                    title_evt_sims[evt_type] = 0.0

            # --- Composite type_score ---
            type_scores: Dict[str, float] = {}
            for evt_type in evt_cols:
                type_scores[evt_type] = (
                    0.50 * norm_evt_means.get(evt_type, 0.0)
                    + 0.50 * max(0.0, title_evt_sims.get(evt_type, 0.0))
                )

            # Store scores on occurrence
            occ.type_scores = type_scores

            # Find best candidate
            best_type = max(type_scores, key=type_scores.get)
            best_score = type_scores[best_type]
            assigned_score = type_scores.get(occ.event_type, 0.0)

            # Type confidence = ratio of assigned score to best score
            occ.type_confidence = (
                assigned_score / best_score if best_score > 0 else 1.0
            )

            # Correct if assigned score < 50% of best AND best > 0
            if best_score > 0 and assigned_score < 0.50 * best_score:
                old_type = occ.event_type
                occ.event_type = best_type
                logger.debug(
                    f"Occurrence {occ.occurrence_id}: retyped "
                    f"{old_type} → {best_type} "
                    f"(confidence {occ.type_confidence:.2f}, "
                    f"scores: {old_type}={assigned_score:.3f}, "
                    f"{best_type}={best_score:.3f})"
                )
                # Recalculate confidence after correction
                occ.type_confidence = 1.0
                n_corrected += 1

        return n_corrected

    def _compute_cascade_metrics(self, occurrences: List[EventOccurrence]
                                 ) -> Dict[str, float]:
        """Compute cascade-level event occurrence metrics.

        All metrics are descriptive (not integrated into total_score).

        Returns:
            Dict with keys:
                n_occurrences: number of distinct event occurrences
                n_event_types: number of distinct event types represented
                mean_coherence: mean semantic coherence across occurrences
                temporal_overlap: fraction of occurrence pairs with overlapping dates
                mean_confidence: mean confidence score across occurrences
                n_low_confidence: number of low-confidence occurrences
                mean_effective_mass: mean effective_mass per occurrence
                total_effective_mass: sum of effective_mass across all occurrences
        """
        if not occurrences:
            return {
                'n_occurrences': 0,
                'n_event_types': 0,
                'mean_coherence': 0.0,
                'temporal_overlap': 0.0,
                'mean_confidence': 0.0,
                'n_low_confidence': 0,
                'mean_effective_mass': 0.0,
                'total_effective_mass': 0.0,
            }

        n = len(occurrences)
        metrics: Dict[str, float] = {}
        metrics['n_occurrences'] = float(n)
        metrics['n_event_types'] = float(
            len(set(o.event_type for o in occurrences))
        )

        # Mean coherence
        coherences = [o.semantic_coherence for o in occurrences]
        metrics['mean_coherence'] = float(np.mean(coherences))

        # Temporal overlap (computed on core periods)
        if n >= 2:
            n_overlapping = 0
            n_pairs = 0
            for i in range(n):
                for j in range(i + 1, n):
                    n_pairs += 1
                    if (occurrences[i].core_start <= occurrences[j].core_end and
                            occurrences[j].core_start <= occurrences[i].core_end):
                        n_overlapping += 1
            metrics['temporal_overlap'] = n_overlapping / n_pairs if n_pairs > 0 else 0.0
        else:
            metrics['temporal_overlap'] = 0.0

        # Confidence metrics
        confidences = [o.confidence for o in occurrences]
        metrics['mean_confidence'] = float(np.mean(confidences))
        metrics['n_low_confidence'] = float(sum(1 for o in occurrences if o.low_confidence))

        # Effective mass
        masses = [o.effective_mass for o in occurrences]
        metrics['mean_effective_mass'] = float(np.mean(masses))
        metrics['total_effective_mass'] = float(np.sum(masses))

        return metrics

    def _compute_regime_transitions(self, daily_profile: pd.DataFrame
                                    ) -> List[Dict]:
        """Compute regime transitions from daily event profile.

        Uses rolling average smoothing and identifies days where the
        dominant event type changes.

        Args:
            daily_profile: DataFrame with date index and evt_* columns.

        Returns:
            List of dicts with keys: date, from_type, to_type, confidence.
        """
        if daily_profile.empty or len(daily_profile) < REGIME_SMOOTHING_WINDOW:
            return []

        # Smooth with rolling average
        smoothed = daily_profile.rolling(
            window=REGIME_SMOOTHING_WINDOW, center=True, min_periods=1
        ).mean()

        # Find dominant type per day
        dominant = smoothed.idxmax(axis=1)

        # Detect transitions
        transitions = []
        prev_type = dominant.iloc[0]
        for i in range(1, len(dominant)):
            curr_type = dominant.iloc[i]
            if curr_type != prev_type:
                # Confidence = margin between top and second type
                day_values = smoothed.iloc[i]
                sorted_vals = day_values.sort_values(ascending=False)
                if len(sorted_vals) >= 2:
                    confidence = float(sorted_vals.iloc[0] - sorted_vals.iloc[1])
                else:
                    confidence = float(sorted_vals.iloc[0])

                transitions.append({
                    'date': smoothed.index[i],
                    'from_type': prev_type,
                    'to_type': curr_type,
                    'confidence': confidence,
                })
                prev_type = curr_type

        return transitions

    # ------------------------------------------------------------------
    # Phase 3: Event cluster detection (meta-events)
    # ------------------------------------------------------------------

    def cluster_occurrences(
        self,
        all_occurrences: List[EventOccurrence],
        entity_index: Optional[dict] = None,
        articles: Optional[pd.DataFrame] = None,
        embedding_store=None,
    ) -> List[EventCluster]:
        """Phase 3: Cluster occurrences into event clusters (meta-events).

        Called after detect_all() with pooled occurrences from all cascades.
        Independent of cascade structure. Uses silhouette score to find the
        optimal number of clusters (no fixed distance threshold).

        Args:
            all_occurrences: All EventOccurrence objects across all cascades.
            entity_index: Optional entity index from EntityIndexer.build_index().
                Maps doc_id → list of (entity_text, entity_type) tuples.
            articles: Optional article DataFrame with media, tone_* columns
                for computing strength metrics.
            embedding_store: Optional EmbeddingStore for semantic coherence scoring.

        Returns:
            List of EventCluster objects.
        """
        if len(all_occurrences) < 2:
            if all_occurrences:
                return [self._single_occurrence_cluster(
                    all_occurrences[0], 0, entity_index, articles,
                    embedding_store=embedding_store,
                )]
            return []

        # Step 0: Deduplicate occurrences across cascades.
        all_occurrences = self._deduplicate_occurrences(all_occurrences)
        logger.info(
            f"Phase 3: {len(all_occurrences)} unique occurrences after deduplication"
        )

        if len(all_occurrences) < 2:
            if all_occurrences:
                return [self._single_occurrence_cluster(
                    all_occurrences[0], 0, entity_index, articles,
                    embedding_store=embedding_store,
                )]
            return []

        # Step 1: Extract entities per occurrence
        occ_entities = {}
        for occ in all_occurrences:
            occ_entities[occ.occurrence_id] = self._extract_occurrence_entities(
                occ, entity_index
            )
            occ.entities = occ_entities[occ.occurrence_id]

        # Step 2: Compute strength metrics per occurrence
        if articles is not None:
            self._compute_occurrence_strength(all_occurrences, articles)

        # Step 3: Compute pairwise compound distance matrix
        n = len(all_occurrences)
        dist_condensed = np.zeros(n * (n - 1) // 2, dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = self._occurrence_distance(
                    all_occurrences[i], all_occurrences[j],
                    occ_entities.get(all_occurrences[i].occurrence_id, set()),
                    occ_entities.get(all_occurrences[j].occurrence_id, set()),
                )
                dist_condensed[idx] = d
                idx += 1

        # Step 4: HAC with average linkage + silhouette-based optimal cut
        Z = linkage(dist_condensed, method='average')
        dist_square = squareform(dist_condensed)

        best_score = -1.0
        best_labels = None
        max_k = n - 1  # k < n (need at least one non-singleton cluster)

        for k in range(2, max_k + 1):
            labels = fcluster(Z, t=k, criterion='maxclust')
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(dist_square, labels, metric='precomputed')
            if score > best_score:
                best_score = score
                best_labels = labels

        # If silhouette is non-positive, no meaningful cluster separation exists
        # Decision: if occurrences are close (max distance < 0.5), merge all (k=1)
        # Otherwise, keep singletons — better than creating spurious mega-clusters
        if best_labels is None or best_score <= 0.0:
            max_dist = float(np.max(dist_condensed)) if len(dist_condensed) > 0 else 0.0
            if max_dist < 0.5:
                best_labels = np.ones(n, dtype=int)
            else:
                best_labels = np.arange(1, n + 1)

        # Step 5: Group occurrences by cluster label
        cluster_groups: Dict[int, List[EventOccurrence]] = {}
        for occ, label in zip(all_occurrences, best_labels):
            cluster_groups.setdefault(int(label), []).append(occ)

        # Step 6: Build EventCluster objects
        event_clusters = []
        for label, occs in sorted(cluster_groups.items()):
            ec = self._build_event_cluster(
                cluster_id=len(event_clusters),
                occurrences=occs,
                entity_index=entity_index,
                articles=articles,
                embedding_store=embedding_store,
            )
            event_clusters.append(ec)

        # Step 6b: Enforce title+temporal connectivity on multi-occurrence clusters
        n_before = len(event_clusters)
        event_clusters = self._split_disconnected_clusters(
            event_clusters, entity_index, articles, embedding_store
        )
        n_split = len(event_clusters) - n_before
        if n_split > 0:
            logger.info(
                f"Step 6b: split into {n_split} additional clusters "
                f"(title+temporal connectivity)"
            )

        # Sort by peak date
        event_clusters.sort(key=lambda ec: ec.peak_date)

        # Renumber
        for i, ec in enumerate(event_clusters):
            ec.cluster_id = i

        logger.info(
            f"Phase 3: {len(event_clusters)} event clusters from "
            f"{len(all_occurrences)} occurrences "
            f"({sum(1 for ec in event_clusters if ec.is_multi_type)} multi-type), "
            f"silhouette={best_score:.3f}"
        )

        return event_clusters

    def _split_disconnected_clusters(
        self,
        event_clusters: List[EventCluster],
        entity_index: Optional[dict],
        articles: Optional[pd.DataFrame],
        embedding_store,
    ) -> List[EventCluster]:
        """Step 6b: Split multi-occurrence clusters with disconnected components.

        Two occurrences are connected if:
          - seed_overlap > 0 (shared seed articles), OR
          - title_sim >= EVENT_CLUSTER_TITLE_SIM_THRESHOLD AND
            peak gap <= EVENT_CLUSTER_MAX_GAP_DAYS

        Uses BFS to find connected components. Disconnected components become
        separate clusters.
        """
        result = []

        for ec in event_clusters:
            if ec.n_occurrences <= 1:
                result.append(ec)
                continue

            occs = ec.occurrences
            n = len(occs)

            # Compute title centroids per occurrence (from seed articles)
            title_centroids = []
            for occ in occs:
                seeds = occ.seed_doc_ids if occ.seed_doc_ids else occ.doc_ids
                embs = []
                for did in seeds:
                    e = self.embedding_store.get_sentence_embedding(
                        did, TITLE_SENTENCE_ID
                    )
                    if e is not None:
                        embs.append(e)
                if embs:
                    c = np.mean(embs, axis=0).astype(np.float32)
                    c_norm = np.linalg.norm(c)
                    title_centroids.append(c / c_norm if c_norm > 0 else c)
                else:
                    title_centroids.append(None)

            seed_sets = [
                set(o.seed_doc_ids) if o.seed_doc_ids else set(o.doc_ids)
                for o in occs
            ]

            # Build adjacency matrix
            adj = [[False] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    # Criterion 1: shared seed articles
                    if seed_sets[i] & seed_sets[j]:
                        adj[i][j] = adj[j][i] = True
                        continue
                    # Criterion 2: title similarity + temporal proximity
                    gap = abs((occs[i].peak_date - occs[j].peak_date).days)
                    if gap > EVENT_CLUSTER_MAX_GAP_DAYS:
                        continue
                    if (title_centroids[i] is not None
                            and title_centroids[j] is not None):
                        sim = float(np.dot(
                            title_centroids[i], title_centroids[j]
                        ))
                        if sim >= EVENT_CLUSTER_TITLE_SIM_THRESHOLD:
                            adj[i][j] = adj[j][i] = True

            # BFS connected components
            visited = [False] * n
            components: List[List[int]] = []
            for start in range(n):
                if visited[start]:
                    continue
                comp: List[int] = []
                queue = deque([start])
                while queue:
                    node = queue.popleft()
                    if visited[node]:
                        continue
                    visited[node] = True
                    comp.append(node)
                    for nb in range(n):
                        if adj[node][nb] and not visited[nb]:
                            queue.append(nb)
                components.append(comp)

            if len(components) == 1:
                result.append(ec)
            else:
                for comp in components:
                    comp_occs = [occs[i] for i in comp]
                    new_ec = self._build_event_cluster(
                        cluster_id=0,
                        occurrences=comp_occs,
                        entity_index=entity_index,
                        articles=articles,
                        embedding_store=embedding_store,
                    )
                    result.append(new_ec)

        return result

    @staticmethod
    def _deduplicate_occurrences(
        occurrences: List[EventOccurrence],
        jaccard_threshold: float = 0.5,
    ) -> List[EventOccurrence]:
        """Merge duplicate occurrences from overlapping cascades.

        The same real-world event produces similar occurrences across different
        frame cascades. This method groups occurrences by event_type and merges
        those with high seed_doc_id overlap (Jaccard > threshold).

        Uses greedy agglomeration: sort by effective_mass descending, then
        merge smaller duplicates into the representative with highest mass.

        Args:
            occurrences: All occurrences pooled across cascades.
            jaccard_threshold: Minimum Jaccard similarity on seed_doc_ids to merge.

        Returns:
            Deduplicated list of occurrences with merged doc_ids/seed_doc_ids.
        """
        if len(occurrences) <= 1:
            return list(occurrences)

        # Group by event_type
        by_type: Dict[str, List[EventOccurrence]] = {}
        for occ in occurrences:
            by_type.setdefault(occ.event_type, []).append(occ)

        deduplicated = []

        for evt_type, type_occs in by_type.items():
            # Sort by effective_mass descending — best representative first
            type_occs.sort(key=lambda o: o.effective_mass, reverse=True)

            merged_flags = [False] * len(type_occs)

            for i in range(len(type_occs)):
                if merged_flags[i]:
                    continue

                rep = type_occs[i]
                rep_seeds = set(rep.seed_doc_ids) if rep.seed_doc_ids else set(rep.doc_ids)

                for j in range(i + 1, len(type_occs)):
                    if merged_flags[j]:
                        continue

                    cand = type_occs[j]
                    cand_seeds = set(cand.seed_doc_ids) if cand.seed_doc_ids else set(cand.doc_ids)

                    # Jaccard similarity on seed_doc_ids
                    union_size = len(rep_seeds | cand_seeds)
                    if union_size == 0:
                        continue
                    jaccard = len(rep_seeds & cand_seeds) / union_size
                    if jaccard >= jaccard_threshold:
                        # Merge: expand rep's doc_ids and seed_doc_ids
                        merged_doc_ids = set(rep.doc_ids) | set(cand.doc_ids)
                        rep.doc_ids = list(merged_doc_ids)
                        rep_seeds = rep_seeds | cand_seeds
                        rep.seed_doc_ids = list(rep_seeds)
                        rep.n_articles = len(merged_doc_ids)
                        # Keep higher effective_mass and core_mass
                        rep.effective_mass = max(rep.effective_mass, cand.effective_mass)
                        merged_flags[j] = True

                deduplicated.append(rep)

        # Renumber occurrence_ids
        for i, occ in enumerate(deduplicated):
            occ.occurrence_id = i

        return deduplicated

    def _occurrence_distance(
        self,
        occ_a: EventOccurrence,
        occ_b: EventOccurrence,
        entities_a: set,
        entities_b: set,
    ) -> float:
        """Compute compound distance between two occurrences.

        Five dimensions weighted by EVENT_CLUSTER_*_WEIGHT constants:
        - Temporal: 1 - exp(-|peak_A - peak_B| / scale)
        - Semantic: cosine distance between centroids
        - Entities: Jaccard distance (1 - |A∩B|/|A∪B|)
        - Article overlap: Jaccard on seed_doc_ids (Phase 2 core members)
        - Type: 0 if same event_type, 1 if different

        Cross-type clustering relies on temporal + semantic + entity overlap.
        Article overlap uses seed_doc_ids (not inflated doc_ids) because
        Phase 4 soft-assignment creates artificial cross-type overlap.
        """
        same_type = occ_a.event_type == occ_b.event_type

        # Temporal
        days_apart = abs((occ_a.peak_date - occ_b.peak_date).days)
        temporal_dist = 1.0 - np.exp(-days_apart / EVENT_CLUSTER_TEMPORAL_SCALE)

        # Semantic (cosine distance between centroids, corpus-adjusted)
        c_a = occ_a.centroid
        c_b = occ_b.centroid
        norm_a = np.linalg.norm(c_a)
        norm_b = np.linalg.norm(c_b)
        if norm_a > 1e-10 and norm_b > 1e-10:
            cos_sim = float(np.dot(c_a / norm_a, c_b / norm_b))
            baseline = self.embedding_store.compute_corpus_baseline()
            residual_sim = max(0.0, (cos_sim - baseline) / (1.0 - baseline))
            semantic_dist = 1.0 - residual_sim
        else:
            semantic_dist = 1.0

        # Entities (Jaccard distance)
        if entities_a or entities_b:
            intersection = len(entities_a & entities_b)
            union = len(entities_a | entities_b)
            entity_dist = 1.0 - (intersection / union) if union > 0 else 0.5
        else:
            entity_dist = 0.5  # neutral when both empty

        # Article overlap (Jaccard on seed_doc_ids — Phase 2 core members)
        docs_a = set(occ_a.seed_doc_ids) if occ_a.seed_doc_ids else set(occ_a.doc_ids)
        docs_b = set(occ_b.seed_doc_ids) if occ_b.seed_doc_ids else set(occ_b.doc_ids)
        if docs_a or docs_b:
            docs_intersection = len(docs_a & docs_b)
            docs_union = len(docs_a | docs_b)
            article_dist = 1.0 - (docs_intersection / docs_union) if docs_union > 0 else 1.0
        else:
            article_dist = 1.0  # no overlap possible when both empty

        # Event type (binary: same type = 0, different = 1)
        type_dist = 0.0 if same_type else 1.0

        return (
            EVENT_CLUSTER_TEMPORAL_WEIGHT * temporal_dist +
            EVENT_CLUSTER_SEMANTIC_WEIGHT * semantic_dist +
            EVENT_CLUSTER_ENTITY_WEIGHT * entity_dist +
            EVENT_CLUSTER_ARTICLE_WEIGHT * article_dist +
            EVENT_CLUSTER_TYPE_WEIGHT * type_dist
        )

    def _extract_occurrence_entities(
        self,
        occurrence: EventOccurrence,
        entity_index: Optional[dict],
    ) -> set:
        """Extract entities mentioned in >= min citations articles of the occurrence.

        Args:
            occurrence: EventOccurrence with doc_ids.
            entity_index: Maps doc_id → list of (entity_text, entity_type).

        Returns:
            Set of entity strings that appear in >= EVENT_CLUSTER_MIN_ENTITY_CITATIONS
            articles within this occurrence.
        """
        if not entity_index or not occurrence.doc_ids:
            return set()

        entity_counts: Dict[str, int] = {}
        for doc_id in occurrence.doc_ids:
            doc_entities = entity_index.get(doc_id, [])
            # Deduplicate per document
            seen = set()
            for ent_text, ent_type in doc_entities:
                if ent_type in ENTITY_TYPES:
                    key = f"{ent_type}:{ent_text}"
                    if key not in seen:
                        seen.add(key)
                        entity_counts[key] = entity_counts.get(key, 0) + 1

        return {
            ent for ent, count in entity_counts.items()
            if count >= EVENT_CLUSTER_MIN_ENTITY_CITATIONS
        }

    def _compute_occurrence_strength(
        self,
        occurrences: List[EventOccurrence],
        articles: pd.DataFrame,
    ) -> None:
        """Compute strength metrics for each occurrence (in-place).

        Metrics:
        - media_count: distinct media outlets
        - temporal_intensity: effective_mass / max(1, core_duration_days)
        - emotional_intensity: belonging-weighted mean |sentiment|
        - tone_coherence: 1 - normalized entropy of tone distribution
        """
        # Build lookups — column may be 'media' or 'media_first' after aggregation
        media_col = next(
            (c for c in ['media', 'media_first'] if c in articles.columns), None
        )
        has_media = media_col is not None
        doc_col = 'doc_id' if 'doc_id' in articles.columns else None

        if doc_col:
            media_lookup = dict(zip(articles[doc_col], articles[media_col])) if has_media else {}

            # Tone columns
            tone_cols = {}
            for col in ['tone_positive', 'tone_neutral', 'tone_negative']:
                if col in articles.columns:
                    tone_cols[col] = dict(zip(articles[doc_col], articles[col]))

        for occ in occurrences:
            # media_count
            if has_media and doc_col:
                media_set = set()
                for did in occ.doc_ids:
                    m = media_lookup.get(did)
                    if m is not None:
                        media_set.add(m)
                occ.media_count = len(media_set)

            # temporal_intensity
            core_days = max(1, (occ.core_end - occ.core_start).days)
            occ.temporal_intensity = occ.effective_mass / core_days

            # emotional_intensity & tone_coherence
            if tone_cols and doc_col:
                weighted_emotions = []
                tone_sums = np.zeros(3)  # pos, neu, neg
                total_weight = 0.0

                for did in occ.doc_ids:
                    b = occ.belonging.get(did, 0.0)
                    if b <= 0:
                        continue

                    pos = tone_cols.get('tone_positive', {}).get(did, 0.0) or 0.0
                    neu = tone_cols.get('tone_neutral', {}).get(did, 0.0) or 0.0
                    neg = tone_cols.get('tone_negative', {}).get(did, 0.0) or 0.0

                    # Sentiment score: pos - neg (simple polarity)
                    sentiment = float(pos) - float(neg)
                    weighted_emotions.append(b * abs(sentiment))

                    tone_sums[0] += b * float(pos)
                    tone_sums[1] += b * float(neu)
                    tone_sums[2] += b * float(neg)
                    total_weight += b

                if total_weight > 0:
                    occ.emotional_intensity = sum(weighted_emotions) / total_weight

                    # Tone coherence: 1 - normalized entropy
                    probs = tone_sums / (tone_sums.sum() + 1e-10)
                    probs = probs[probs > 0]
                    if len(probs) > 1:
                        entropy = -float(np.sum(probs * np.log2(probs)))
                        max_entropy = np.log2(3.0)  # max for 3 categories
                        occ.tone_coherence = 1.0 - (entropy / max_entropy)
                    else:
                        occ.tone_coherence = 1.0

    def _compute_article_level_temporal_bounds(
        self,
        occurrences: List[EventOccurrence],
        centroid: np.ndarray,
        event_types: set,
        articles: Optional[pd.DataFrame],
        embedding_store,
    ) -> Optional[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Compute article-level temporal bounds for an EventCluster.

        Pools articles from all occurrences and computes composite-weighted
        P10/P50/P90 percentiles at article granularity. Much more precise
        than occurrence-level percentiles (often just 1-3 data points).

        Args:
            occurrences: Constituent occurrences.
            centroid: Pre-computed cluster centroid (unit-norm).
            event_types: Event types present in the cluster.
            articles: Article-level DataFrame with doc_id, date, evt_* columns.
            embedding_store: EmbeddingStore for cosine similarity computation.

        Returns:
            (peak_date, core_start, core_end) or None if articles unavailable.
        """
        if articles is None or articles.empty:
            return None

        # Pool belonging dicts — for articles in multiple occurrences, take max
        pooled_belonging: Dict[str, float] = {}
        for occ in occurrences:
            for doc_id, score in occ.belonging.items():
                if score > pooled_belonging.get(doc_id, 0.0):
                    pooled_belonging[doc_id] = score

        if not pooled_belonging:
            return None

        # Resolve date column and index articles by doc_id
        date_col = self._resolve_date_col(articles)
        id_col = 'doc_id' if 'doc_id' in articles.columns else articles.columns[0]

        # Filter articles to those in pooled_belonging
        article_ids = list(pooled_belonging.keys())
        mask = articles[id_col].isin(article_ids)
        matched = articles.loc[mask, [id_col, date_col] + [
            c for c in event_types if c in articles.columns
        ]].copy()

        if matched.empty:
            return None

        matched_dates = pd.to_datetime(matched[date_col])
        matched_doc_ids = matched[id_col].values

        # Build per-article ordinal dates
        ordinals = np.array([d.toordinal() for d in matched_dates], dtype=np.float64)

        # --- Belonging component ---
        belongings = np.array([pooled_belonging.get(str(d), 0.0) for d in matched_doc_ids],
                              dtype=np.float64)

        # --- Cosine similarity component ---
        cosine_sims = None
        if embedding_store is not None:
            doc_id_list = [str(d) for d in matched_doc_ids]
            emb_arr, found_ids = embedding_store.get_batch_article_embeddings(doc_id_list)
            if len(found_ids) > 0:
                # Map found embeddings back to matched articles
                found_set = {fid: idx for idx, fid in enumerate(found_ids)}
                cosine_sims = np.zeros(len(matched_doc_ids), dtype=np.float64)
                c_norm = np.linalg.norm(centroid)
                for i, did in enumerate(doc_id_list):
                    emb_idx = found_set.get(did)
                    if emb_idx is not None:
                        emb = emb_arr[emb_idx]
                        e_norm = np.linalg.norm(emb)
                        if e_norm > 0 and c_norm > 0:
                            cosine_sims[i] = max(0.0, float(np.dot(emb, centroid) / (e_norm * c_norm)))

        # --- Event signal component ---
        evt_cols = [c for c in event_types if c in matched.columns]
        if evt_cols:
            evt_signals = matched[evt_cols].max(axis=1).values.astype(np.float64)
        else:
            evt_signals = None

        # --- Composite weight ---
        if cosine_sims is not None and evt_signals is not None:
            # Full: w = belonging^0.40 × cosine_sim^0.30 × evt_signal^0.30
            weights = (
                np.power(np.maximum(belongings, 1e-10), 0.40)
                * np.power(np.maximum(cosine_sims, 1e-10), 0.30)
                * np.power(np.maximum(evt_signals, 1e-10), 0.30)
            )
        elif evt_signals is not None:
            # No embeddings: w = belonging^0.55 × evt_signal^0.45
            weights = (
                np.power(np.maximum(belongings, 1e-10), 0.55)
                * np.power(np.maximum(evt_signals, 1e-10), 0.45)
            )
        else:
            # No evt signals: w = belonging
            weights = belongings.copy()

        # Filter out zero-weight articles
        pos_mask = weights > 1e-10
        if not np.any(pos_mask):
            return None

        ordinals = ordinals[pos_mask]
        weights = weights[pos_mask]

        p10 = self._weighted_percentile(ordinals, weights, 10)
        p50 = self._weighted_percentile(ordinals, weights, 50)
        p90 = self._weighted_percentile(ordinals, weights, 90)

        peak_date = pd.Timestamp.fromordinal(int(p50))
        core_start = pd.Timestamp.fromordinal(int(p10))
        core_end = pd.Timestamp.fromordinal(int(p90))

        return peak_date, core_start, core_end

    def _build_event_cluster(
        self,
        cluster_id: int,
        occurrences: List[EventOccurrence],
        entity_index: Optional[dict] = None,
        articles: Optional[pd.DataFrame] = None,
        embedding_store=None,
    ) -> EventCluster:
        """Build an EventCluster from a group of occurrences."""
        event_types = set(occ.event_type for occ in occurrences)

        # Mass-weighted centroid (computed first — needed for article-level temporal bounds)
        masses = np.array([occ.effective_mass for occ in occurrences], dtype=np.float64)
        total_mass = float(masses.sum())
        if total_mass > 0:
            centroid = np.zeros_like(occurrences[0].centroid, dtype=np.float32)
            for occ, m in zip(occurrences, masses):
                centroid += float(m) * occ.centroid
            centroid /= total_mass
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid /= c_norm
        else:
            centroid = np.mean([occ.centroid for occ in occurrences], axis=0)

        # Article-level temporal bounds (preferred) or occurrence-level fallback
        article_temporal = self._compute_article_level_temporal_bounds(
            occurrences, centroid, event_types, articles, embedding_store,
        )
        if article_temporal is not None:
            peak_date, core_start, core_end = article_temporal
        else:
            # Fallback: mass-weighted percentiles on occurrence-level peak_dates
            peak_ords = np.array([occ.peak_date.toordinal() for occ in occurrences],
                                 dtype=np.float64)
            if len(occurrences) >= 2:
                p10 = self._weighted_percentile(peak_ords, masses, 10)
                p50 = self._weighted_percentile(peak_ords, masses, 50)
                p90 = self._weighted_percentile(peak_ords, masses, 90)
            else:
                p10 = p50 = p90 = float(peak_ords[0])
            core_start = pd.Timestamp.fromordinal(int(p10))
            peak_date = pd.Timestamp.fromordinal(int(p50))
            core_end = pd.Timestamp.fromordinal(int(p90))

        # Shared entities (intersection across all occurrences)
        if len(occurrences) > 1:
            shared = set.intersection(*(occ.entities for occ in occurrences))
        else:
            shared = occurrences[0].entities.copy()

        # Type overlap graph & structure analysis (with full type ranking)
        type_overlap_graph, type_structure, dominant_type, type_ranking = (
            self._analyze_type_structure(occurrences, event_types, EVENT_COLUMNS)
        )

        # Strength score
        strength, strength_components = self._compute_cluster_strength(
            occurrences, total_mass, articles=articles,
            embedding_store=embedding_store,
        )

        return EventCluster(
            cluster_id=cluster_id,
            occurrences=occurrences,
            event_types=event_types,
            peak_date=peak_date,
            core_start=core_start,
            core_end=core_end,
            total_mass=total_mass,
            centroid=centroid,
            n_occurrences=len(occurrences),
            is_multi_type=len(event_types) > 1,
            strength=strength,
            strength_components=strength_components,
            entities=shared,
            dominant_type=dominant_type,
            type_structure=type_structure,
            type_overlap_graph=type_overlap_graph,
            type_ranking=type_ranking,
        )

    @staticmethod
    def _analyze_type_structure(
        occurrences: List[EventOccurrence],
        event_types: set,
        all_event_columns: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str], str, List[Tuple[str, float]]]:
        """Analyze type overlap graph, structure, dominant type, and type ranking.

        For each pair of event types in the cluster, computes Jaccard overlap
        on their pooled doc_ids. Determines constitutive vs satellite types
        and ranks all event types by combined structural + NLP score.

        Args:
            occurrences: List of EventOccurrence in this cluster.
            event_types: Set of event types present in the cluster.
            all_event_columns: All EVENT_COLUMNS for full-type ranking.
                If None, ranking covers only present types.

        Returns:
            Tuple of (type_overlap_graph, type_structure, dominant_type, type_ranking).
        """
        types_list = sorted(event_types)
        all_types = sorted(all_event_columns) if all_event_columns else types_list

        # Pool seed_doc_ids per type (Phase 2 core members, not soft-assigned)
        type_docs: Dict[str, set] = {}
        type_mass: Dict[str, float] = {}
        for occ in occurrences:
            ids = occ.seed_doc_ids if occ.seed_doc_ids else occ.doc_ids
            type_docs.setdefault(occ.event_type, set()).update(ids)
            type_mass[occ.event_type] = type_mass.get(occ.event_type, 0.0) + occ.effective_mass

        total_mass = sum(type_mass.values()) or 1.0

        # Build overlap graph
        type_overlap_graph: Dict[str, Dict[str, float]] = {}
        for t in types_list:
            type_overlap_graph[t] = {}

        for i, t_a in enumerate(types_list):
            for t_b in types_list[i + 1:]:
                docs_a = type_docs.get(t_a, set())
                docs_b = type_docs.get(t_b, set())
                union_size = len(docs_a | docs_b)
                if union_size > 0:
                    jaccard = len(docs_a & docs_b) / union_size
                else:
                    jaccard = 0.0
                type_overlap_graph[t_a][t_b] = jaccard
                type_overlap_graph[t_b][t_a] = jaccard

        # Classify types: constitutive vs satellite
        type_structure: Dict[str, str] = {}
        n_types = len(types_list)
        for t in types_list:
            if n_types <= 1:
                type_structure[t] = 'constitutive'
            else:
                # Constitutive if Jaccard > 0 with at least 1 other type
                has_overlap = any(
                    type_overlap_graph[t].get(other, 0.0) > 0
                    for other in types_list if other != t
                )
                type_structure[t] = 'constitutive' if has_overlap else 'satellite'

        # --- Structural score per type (for present types) ---
        structural_raw: Dict[str, float] = {}
        for t in types_list:
            mass_norm = type_mass.get(t, 0.0) / total_mass
            if n_types > 1:
                connectivity = float(np.mean([
                    type_overlap_graph[t].get(other, 0.0)
                    for other in types_list if other != t
                ]))
            else:
                connectivity = 1.0
            structural_raw[t] = 0.6 * mass_norm + 0.4 * connectivity

        # --- NLP score per type (all 7 types via occ.type_scores) ---
        nlp_raw: Dict[str, float] = {}
        total_occ_mass = sum(occ.effective_mass for occ in occurrences) or 1.0
        has_any_type_scores = any(occ.type_scores for occ in occurrences)

        for t in all_types:
            if has_any_type_scores:
                weighted_sum = sum(
                    occ.effective_mass * occ.type_scores.get(t, 0.0)
                    for occ in occurrences
                )
                nlp_raw[t] = weighted_sum / total_occ_mass
            else:
                nlp_raw[t] = 0.0

        # --- Normalize both to [0, 1] (best = 1.0) ---
        max_structural = max(structural_raw.values()) if structural_raw else 1.0
        max_nlp = max(nlp_raw.values()) if nlp_raw else 1.0

        structural_norm: Dict[str, float] = {}
        for t in all_types:
            structural_norm[t] = (structural_raw.get(t, 0.0) / max_structural) if max_structural > 0 else 0.0

        nlp_norm: Dict[str, float] = {}
        for t in all_types:
            nlp_norm[t] = (nlp_raw[t] / max_nlp) if max_nlp > 0 else 0.0

        # --- Combined score & ranking ---
        type_ranking: List[Tuple[str, float]] = []
        for t in all_types:
            combined = 0.50 * structural_norm[t] + 0.50 * nlp_norm[t]
            if combined > 0:
                type_ranking.append((t, combined))

        type_ranking.sort(key=lambda x: x[1], reverse=True)

        # Dominant type from ranking (replaces old calculation)
        dominant_type = type_ranking[0][0] if type_ranking else (types_list[0] if types_list else '')

        return type_overlap_graph, type_structure, dominant_type, type_ranking

    def _update_cluster_type_rankings(
        self,
        event_clusters: List[EventCluster],
        final_occurrences: List[EventOccurrence],
    ) -> None:
        """Update type_ranking on clusters using final occurrences' type_scores.

        Phase 3 clusters are built from temp occurrences (before type validation).
        After _validate_occurrence_types populates type_scores on final occurrences,
        this method matches them to clusters and recomputes type_ranking.
        """
        if not final_occurrences or not event_clusters:
            return

        # Build seed_doc_ids → final occurrence lookup
        occ_by_seeds: Dict[frozenset, EventOccurrence] = {}
        for occ in final_occurrences:
            seeds = frozenset(occ.seed_doc_ids if occ.seed_doc_ids else occ.doc_ids)
            if seeds:
                occ_by_seeds[seeds] = occ

        for ec in event_clusters:
            # Match cluster's temp occurrences to final occurrences by seed overlap
            matched_occs: List[EventOccurrence] = []
            for temp_occ in ec.occurrences:
                temp_seeds = set(temp_occ.seed_doc_ids if temp_occ.seed_doc_ids else temp_occ.doc_ids)
                if not temp_seeds:
                    continue

                # Find best matching final occurrence (highest Jaccard on seed_doc_ids)
                best_match = None
                best_jaccard = 0.0
                for final_seeds, final_occ in occ_by_seeds.items():
                    if final_occ.event_type != temp_occ.event_type:
                        continue
                    intersection = len(temp_seeds & final_seeds)
                    union = len(temp_seeds | final_seeds)
                    if union > 0:
                        jaccard = intersection / union
                        if jaccard > best_jaccard:
                            best_jaccard = jaccard
                            best_match = final_occ

                if best_match is not None and best_match.type_scores:
                    matched_occs.append(best_match)

            if not matched_occs:
                continue

            # Recompute type_ranking with matched final occurrences
            _, _, dominant_type, type_ranking = self._analyze_type_structure(
                matched_occs, ec.event_types, EVENT_COLUMNS
            )
            ec.type_ranking = type_ranking
            ec.dominant_type = dominant_type

    @staticmethod
    def _compute_cluster_strength(
        occurrences: List[EventOccurrence],
        total_mass: float,
        articles: Optional[pd.DataFrame] = None,
        embedding_store=None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute composite strength score for an event cluster.

        Five dimensions:
        - mass_score (0.20): log2(1 + total_mass) / log2(1 + 100)
        - coverage_score (0.25): composite of media, journalist, geographic spread
        - intensity_score (0.20): log2(1 + mean_temporal_intensity) / log2(1 + 50)
        - coherence_score (0.15): residual coherence above corpus baseline
        - media_diversity (0.20): 1 - 1/n_media
        """
        # Mass recalibrated for event scale (not corpus)
        mass_score = min(1.0, np.log2(1.0 + total_mass) / np.log2(1.0 + 100.0))

        # Intensity
        intensities = [occ.temporal_intensity for occ in occurrences]
        mean_intensity = float(np.mean(intensities)) if intensities else 0.0
        intensity_score = min(1.0, np.log2(1.0 + mean_intensity) / np.log2(1.0 + 50.0))

        # Coverage (media + journalist + geographic)
        coverage_score = 0.0
        n_media_unique = 1
        if articles is not None:
            all_doc_ids = list(set(
                did for occ in occurrences
                for did in (occ.seed_doc_ids if occ.seed_doc_ids else occ.doc_ids)
            ))
            media_col = next(
                (c for c in ['media', 'media_first'] if c in articles.columns), None
            )
            doc_col = 'doc_id' if 'doc_id' in articles.columns else None

            if doc_col and media_col:
                cluster_articles = articles[articles[doc_col].isin(all_doc_ids)]
                media_list = cluster_articles[media_col].dropna().unique().tolist()
                n_media_unique = max(1, len(media_list))
                media_score = min(1.0, np.log2(1.0 + n_media_unique) / np.log2(1.0 + 20.0))

                # Journalist score
                author_col = 'author' if 'author' in articles.columns else None
                if author_col:
                    authors = cluster_articles[author_col].dropna().unique()
                    media_set = set(media_list)
                    journalists = [a for a in authors if a not in media_set]
                    n_journalists = len(journalists)
                else:
                    n_journalists = 0
                journalist_score = min(1.0, np.log2(1.0 + n_journalists) / np.log2(1.0 + 50.0))

                # Geographic score
                geo_score = 0.0
                try:
                    from cascade_detector.utils.media_geography import MediaGeography
                    geo = MediaGeography()
                    spread = geo.calculate_geographic_spread(media_list)
                    geo_score = float(spread.get('cascade_geographic_score', 0.0))
                except Exception:
                    geo_score = 0.0

                coverage_score = (media_score + journalist_score + geo_score) / 3.0

        # Coherence RESIDUAL (above corpus baseline)
        coherence_score = 0.0
        if embedding_store is not None:
            all_doc_ids = list(set(
                did for occ in occurrences
                for did in (occ.seed_doc_ids if occ.seed_doc_ids else occ.doc_ids)
            ))
            if len(all_doc_ids) >= 2:
                raw = embedding_store.mean_pairwise_similarity(all_doc_ids)
                baseline = embedding_store.compute_corpus_baseline()
                coherence_score = (
                    max(0.0, (raw - baseline) / (1.0 - baseline))
                    if baseline < 1.0 else 0.0
                )

        # Media diversity (NEW)
        media_diversity = 1.0 - 1.0 / n_media_unique

        strength = (
            EVENT_CLUSTER_STRENGTH_MASS_WEIGHT * mass_score +
            EVENT_CLUSTER_STRENGTH_COVERAGE_WEIGHT * coverage_score +
            EVENT_CLUSTER_STRENGTH_INTENSITY_WEIGHT * intensity_score +
            EVENT_CLUSTER_STRENGTH_COHERENCE_WEIGHT * coherence_score +
            EVENT_CLUSTER_STRENGTH_DIVERSITY_WEIGHT * media_diversity
        )

        components = {
            'mass_score': mass_score,
            'coverage_score': coverage_score,
            'intensity_score': intensity_score,
            'coherence_score': coherence_score,
            'media_diversity': media_diversity,
        }

        return strength, components

    def _single_occurrence_cluster(
        self,
        occ: EventOccurrence,
        cluster_id: int,
        entity_index: Optional[dict],
        articles: Optional[pd.DataFrame],
        embedding_store=None,
    ) -> EventCluster:
        """Wrap a single occurrence as a singleton event cluster."""
        occ.entities = self._extract_occurrence_entities(occ, entity_index)
        if articles is not None:
            self._compute_occurrence_strength([occ], articles)
        return self._build_event_cluster(
            cluster_id, [occ], entity_index,
            articles=articles, embedding_store=embedding_store,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_date_col(articles: pd.DataFrame) -> str:
        """Find the date column in articles DataFrame."""
        if 'date_converted_first' in articles.columns:
            return 'date_converted_first'
        if 'date_converted' in articles.columns:
            return 'date_converted'
        return 'date'

    @staticmethod
    def _get_cascade_articles(cascade, articles: pd.DataFrame,
                              date_col: str) -> pd.DataFrame:
        """Filter articles to those within the cascade window."""
        dates = pd.to_datetime(articles[date_col])
        onset = pd.Timestamp(cascade.onset_date)
        end = pd.Timestamp(cascade.end_date)
        mask = (dates >= onset) & (dates <= end)
        return articles[mask].copy()
