"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
pipeline.py

MAIN OBJECTIVE:
---------------
Single orchestrator for cascade detection. Sequential processing.

Pipeline steps:
1. Load & process data from PostgreSQL
2. Load embedding store (mandatory)
3. Build indices (sequential)
4. Detect and score cascades in one step (unified multi-signal detector)
5. Return DetectionResults

The UnifiedCascadeDetector replaces the sequential BurstDetector + CascadeScorer
pipeline. Detection uses a composite signal (5 daily Z-scores including semantic
anomaly), scoring uses 4 dimensions with sub-indices including network metrics.
Embeddings are mandatory for both semantic signal detection and convergence scoring.

Author:
-------
Antoine Lemor
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Optional, List, Union

import pandas as pd

from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.models import DetectionResults
from cascade_detector.data.connector import DatabaseConnector
from cascade_detector.data.processor import DataProcessor
from cascade_detector.indexing.index_manager import IndexManager
from cascade_detector.detection.unified_detector import UnifiedCascadeDetector

logger = logging.getLogger(__name__)


class CascadeDetectionPipeline:
    """Orchestrates the full cascade detection pipeline."""

    def __init__(self, config: Optional[DetectorConfig] = None,
                 embedding_store=None):
        self.config = config or DetectorConfig()
        self.connector = DatabaseConnector(self.config)
        self.processor = DataProcessor()
        self.index_manager = IndexManager()

        # Load embedding store (mandatory)
        if embedding_store is None:
            embedding_store = self._load_embedding_store()

        self.detector = UnifiedCascadeDetector(self.config, embedding_store=embedding_store)

    def _load_embedding_store(self):
        """Load EmbeddingStore from configured directory. Raises if not found."""
        from cascade_detector.embeddings.embedding_store import EmbeddingStore
        store = EmbeddingStore(
            self.config.embedding_dir,
            embedding_dim=self.config.embedding_dim,
        )
        logger.info(f"Loaded embedding store from {self.config.embedding_dir}")
        return store

    def _fetch_paradigm_buffer(self, start_date: str) -> Optional[dict]:
        """Fetch prior data buffer for paradigm warm-up.

        The paradigm sliding window requires 12 weeks of history before
        producing its first state.  When running year-by-year, this means
        Q1 has no paradigm coverage.  This method loads weekly proportions
        from the 14 weeks preceding ``start_date`` (12-week window + 2-week
        margin for weekly resampling alignment) so the sliding window is
        warm from day 1 of the target period.

        Returns:
            Temporal index dict for the buffer period, or None if
            unavailable (first year, empty result).
        """
        buffer_end = pd.Timestamp(start_date) - pd.Timedelta(days=1)
        # 14 weeks = 12 (window) + 2 (margin for weekly resampling alignment)
        buffer_start = buffer_end - pd.Timedelta(weeks=14)

        try:
            buffer_df = self.connector.get_frame_data(
                str(buffer_start.date()), str(buffer_end.date())
            )
        except Exception as e:
            logger.info(f"  Paradigm buffer: DB query failed ({e})")
            return None

        if buffer_df is None or buffer_df.empty:
            logger.info("  Paradigm buffer: no prior data available")
            return None

        buffer_df = self.processor.process_frame_data(buffer_df)

        from cascade_detector.indexing.temporal_indexer import TemporalIndexer
        temp_indexer = TemporalIndexer()
        buffer_temporal = temp_indexer.build_index(buffer_df)

        logger.info(
            f"  Paradigm buffer: {buffer_start.date()} to {buffer_end.date()} "
            f"({len(buffer_df):,} rows)"
        )
        return buffer_temporal

    def run(self, start_date: str, end_date: str,
            target_end_date: Optional[str] = None,
            frames: Optional[List[str]] = None,
            checkpoint_dir: Optional[Union[str, Path]] = None) -> DetectionResults:
        """Run the full cascade detection pipeline.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD). May extend past
                ``target_end_date`` to capture boundary cascades.
            target_end_date: If set, cascades are trimmed so that only
                those with ``onset_date <= target_end_date`` are kept.
                Defaults to ``end_date`` when *None*.
            frames: Frames to analyze (default: all 8).
            checkpoint_dir: Optional directory for per-step checkpointing.

        Returns:
            DetectionResults with all cascades, bursts, and metadata.
        """
        t0 = time.time()

        if target_end_date is None:
            target_end_date = end_date

        if frames:
            self.config.frames = frames

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)

        logger.info(f"Starting cascade detection: {start_date} to {end_date}"
                     f" (target end: {target_end_date})")
        logger.info(f"Frames: {self.config.frames}")

        # Step 1: Load & process data
        step1_file = checkpoint_dir / 'step1_data.pkl' if checkpoint_dir else None
        if step1_file and step1_file.exists():
            logger.info("Step 1: Loading data from checkpoint...")
            with open(step1_file, 'rb') as f:
                df, articles = pickle.load(f)
            logger.info(f"  Restored {len(df):,} rows, {len(articles):,} articles")
        else:
            logger.info("Step 1: Loading data from database...")
            df = self.connector.get_frame_data(start_date, end_date)
            logger.info(f"  Loaded {len(df):,} sentence-level rows")

            df = self.processor.process_frame_data(df)
            articles = self.processor.aggregate_by_article(df)
            logger.info(f"  Aggregated to {len(articles):,} articles")

            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                with open(step1_file, 'wb') as f:
                    pickle.dump((df, articles), f)
                logger.info("  Checkpoint saved (step 1)")

        # Step 2: Build indices
        step2_file = checkpoint_dir / 'step2_indices.pkl' if checkpoint_dir else None
        if step2_file and step2_file.exists():
            logger.info("Step 2: Loading indices from checkpoint...")
            with open(step2_file, 'rb') as f:
                indices = pickle.load(f)
            logger.info("  Restored indices")
        else:
            logger.info("Step 2: Building indices...")
            indices = self.index_manager.build_all_indices(df)

            if checkpoint_dir:
                with open(step2_file, 'wb') as f:
                    pickle.dump(indices, f)
                logger.info("  Checkpoint saved (step 2)")

        # Step 3: Detect and score cascades (unified multi-signal)
        logger.info("Step 3: Detecting and scoring cascades (multi-signal)...")
        cascades, bursts, frame_signals = self.detector.detect_all_frames(
            indices.get('temporal', {}), articles, indices
        )
        logger.info(f"  Total bursts detected: {len(bursts)}")
        logger.info(f"  Total cascades scored: {len(cascades)}")

        # Filter out empty cascades (bursts with insufficient articles)
        n_before = len(cascades)
        cascades = [c for c in cascades if c.n_articles > 0]
        n_filtered = n_before - len(cascades)
        if n_filtered > 0:
            logger.info(f"  Filtered {n_filtered} empty cascades (n_articles=0)")

        # Build results
        elapsed = time.time() - t0

        n_by_frame = {}
        n_by_class = {}
        for c in cascades:
            n_by_frame[c.frame] = n_by_frame.get(c.frame, 0) + 1
            n_by_class[c.classification] = n_by_class.get(c.classification, 0) + 1

        results = DetectionResults(
            cascades=cascades,
            all_bursts=bursts,
            n_cascades_by_frame=n_by_frame,
            n_cascades_by_classification=n_by_class,
            analysis_period=(start_date, target_end_date),
            n_articles_analyzed=len(articles),
            runtime_seconds=elapsed,
            detection_parameters=self.config.to_dict().get('detection', {}),
            frame_signals=frame_signals,
        )

        # Step 3.5: Event detection (database-first, all articles)
        logger.info("Step 3.5: Detecting events (database-first on all articles)...")
        from cascade_detector.analysis.event_occurrence import EventOccurrenceDetector
        event_occ_detector = EventOccurrenceDetector(
            self.detector.embedding_store, sentence_df=df
        )

        # Build reverse entity index: doc_id → [(entity_text, entity_type)]
        raw_entity_index = indices.get('entities', {})
        doc_entity_index = {}
        for entity_key, info in raw_entity_index.items():
            ent_type = info.get('type', '')
            ent_name = info.get('name', entity_key)
            for citation in info.get('citations', []):
                doc_id = citation.get('doc_id')
                if doc_id is not None:
                    doc_entity_index.setdefault(doc_id, []).append(
                        (ent_name, ent_type)
                    )
        logger.info(f"  Built doc-level entity index: {len(doc_entity_index)} articles with entities")

        all_occurrences, event_clusters = event_occ_detector.detect_events(
            articles, entity_index=doc_entity_index
        )
        results.all_occurrences = all_occurrences
        results.event_clusters = event_clusters
        logger.info(
            f"  {len(all_occurrences)} occurrences, "
            f"{len(event_clusters)} event clusters detected"
        )

        # Step 3.6: Attribution to cascades
        logger.info("Step 3.6: Attributing events to cascades...")
        attributions = event_occ_detector.attribute_to_cascades(
            all_occurrences, cascades, articles
        )
        results.cascade_attributions = attributions
        logger.info(f"  {len(attributions)} cascade attributions")

        # Store indices and articles on results for production export
        results._indices = indices
        results._articles = articles

        # Trim cascades to target period (for year-boundary handling)
        if target_end_date != end_date:
            trim_end = pd.Timestamp(target_end_date)
            n_before = len(results.cascades)
            results.cascades = [
                c for c in results.cascades
                if pd.Timestamp(c.onset_date) <= trim_end
            ]
            results.all_bursts = [
                b for b in results.all_bursts
                if pd.Timestamp(b.onset_date) <= trim_end
            ]
            n_trimmed = n_before - len(results.cascades)
            if n_trimmed > 0:
                logger.info(
                    f"  Trimmed {n_trimmed} out-of-period cascades "
                    f"(onset > {target_end_date})"
                )
                # Recompute counts
                n_by_frame = {}
                n_by_class = {}
                for c in results.cascades:
                    n_by_frame[c.frame] = n_by_frame.get(c.frame, 0) + 1
                    n_by_class[c.classification] = (
                        n_by_class.get(c.classification, 0) + 1
                    )
                results.n_cascades_by_frame = n_by_frame
                results.n_cascades_by_classification = n_by_class

        # Step 4: Paradigm shift analysis
        logger.info("Step 4: Analyzing paradigm shifts...")
        buffer_temporal = self._fetch_paradigm_buffer(start_date)
        from cascade_detector.analysis.paradigm_shift import ParadigmShiftAnalyzer
        shift_analyzer = ParadigmShiftAnalyzer()
        results.paradigm_shifts = shift_analyzer.analyze(
            results, buffer_temporal=buffer_temporal
        )
        logger.info(f"  {len(results.paradigm_shifts.shifts)} paradigm shifts detected")

        # Step 5: StabSel impact analysis (cluster → cascade)
        logger.info("Step 5: StabSel impact analysis (cluster → cascade)...")
        from cascade_detector.analysis.stabsel_impact import StabSelImpactAnalyzer
        impact_analyzer = StabSelImpactAnalyzer(self.detector.embedding_store)
        results.event_impact = impact_analyzer.run(results)

        # Step 5b: StabSel paradigm impact analysis (cluster/cascade → paradigm dominance)
        if results.paradigm_shifts is not None:
            logger.info("Step 5b: StabSel paradigm impact analysis...")
            from cascade_detector.analysis.stabsel_paradigm import StabSelParadigmAnalyzer
            paradigm_analyzer = StabSelParadigmAnalyzer(self.detector.embedding_store)
            try:
                results.paradigm_impact = paradigm_analyzer.run(results)
                n_a = len(results.paradigm_impact.cluster_dominance)
                n_b = len(results.paradigm_impact.cascade_dominance)
                logger.info(f"  Model A: {n_a} cluster-frame pairs, "
                            f"Model B: {n_b} cascade-frame pairs")
            except Exception as e:
                logger.warning(f"  Paradigm impact analysis failed: {e}")
                results.paradigm_impact = None
        else:
            logger.info("  Skipping paradigm impact (no paradigm_shifts)")

        logger.info(f"\n{results.summary()}")

        return results
