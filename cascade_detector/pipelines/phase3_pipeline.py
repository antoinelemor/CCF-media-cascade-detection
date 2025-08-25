"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
phase3_pipeline.py

MAIN OBJECTIVE:
---------------
This script orchestrates all Phase 3 cascade detectors in a unified pipeline, ensuring complete
data flow, result aggregation, and scientific validation across all detection algorithms.

Dependencies:
-------------
- typing
- dataclasses
- datetime
- collections
- numpy
- pandas
- networkx
- pathlib
- json
- pickle
- logging
- warnings
- tqdm
- uuid
- time
- traceback
- concurrent.futures

MAIN FEATURES:
--------------
1) Orchestration of all Phase 3 detectors (burst, cascade, multi-frame, etc.)
2) Complete data flow management between detection stages
3) Result aggregation and validation across detectors
4) Performance optimization with parallel processing
5) Comprehensive reporting and visualization support

Author:
-------
Antoine Lemor
"""

from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import json
import pickle
import logging
import warnings
from tqdm import tqdm
import uuid
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Import all detectors
from cascade_detector.detectors.base_detector import BaseDetector, DetectionContext
from cascade_detector.detectors.signal_aggregator import SignalAggregator, AggregatedSignal
from cascade_detector.detectors.burst_detector import BurstDetector, BurstEvent
from cascade_detector.detectors.cascade_detector import (
    CascadeDetector, CompleteCascade, EnhancedBurst
)
from cascade_detector.detectors.multi_frame_detector import (
    MultiFrameDetector, MultiFramePattern, FrameInteraction, ParadigmShiftIndicator
)
from cascade_detector.detectors.cross_media_tracker import (
    CrossMediaTracker, MediaProfile, MediaCoordination, MediaCluster,
    JournalistNetwork, InformationFlow
)
from cascade_detector.detectors.sequence_detector import (
    SequenceDetector, CascadeSequence, TemporalMotif, NarrativeEvolution,
    InformationPathway
)
from cascade_detector.detectors.echo_chamber_detector import (
    EchoChamberDetector, EchoChamber, FilterBubble, InformationSilo,
    HomophilyPattern, BridgeActor, ReinforcementSpiral
)
from cascade_detector.detectors.polarization_detector import (
    PolarizationDetector, PolarizationPattern, OpinionCluster, PolarizationAxis,
    AffectivePolarization, IssuePolarization, DepolarizationOpportunity
)

# Import all Phase 1-2 components
from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.models import TimeWindow, MediaCascade
from cascade_detector.data.processor import DataProcessor
from cascade_detector.data.connector import DatabaseConnector
from cascade_detector.indexing.index_manager import IndexManager

# Import metrics
from cascade_detector.metrics.scientific_network_metrics import ScientificNetworkMetrics
from cascade_detector.metrics.exhaustive_metrics_calculator import ExhaustiveMetricsCalculator
from cascade_detector.metrics.temporal_metrics import TemporalMetrics
from cascade_detector.metrics.diversity_metrics import DiversityMetrics
from cascade_detector.metrics.convergence_metrics import ConvergenceMetrics

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class Phase3Results:
    """
    Complete results from Phase 3 pipeline execution.
    """
    # Execution metadata
    pipeline_id: str
    execution_timestamp: datetime
    execution_duration: float  # seconds
    config_used: DetectorConfig
    
    # Data statistics
    data_statistics: Dict[str, Any]
    indices_statistics: Dict[str, Any]
    
    # Detection results (all mandatory)
    aggregated_signals: List[AggregatedSignal]
    burst_events: List[BurstEvent]
    cascades: List[CompleteCascade]
    multi_frame_patterns: List[MultiFramePattern]
    paradigm_shifts: List[ParadigmShiftIndicator]
    media_profiles: Dict[str, MediaProfile]
    media_coordinations: List[MediaCoordination]
    media_clusters: List[MediaCluster]
    journalist_networks: List[JournalistNetwork]
    information_flows: List[InformationFlow]
    cascade_sequences: List[CascadeSequence]
    temporal_motifs: List[TemporalMotif]
    narrative_evolutions: List[NarrativeEvolution]
    information_pathways: List[InformationPathway]
    echo_chambers: List[EchoChamber]
    filter_bubbles: List[FilterBubble]
    information_silos: List[InformationSilo]
    homophily_patterns: List[HomophilyPattern]
    bridge_actors: List[BridgeActor]
    reinforcement_spirals: List[ReinforcementSpiral]
    polarization_patterns: List[PolarizationPattern]
    opinion_clusters: List[OpinionCluster]
    polarization_axes: List[PolarizationAxis]
    affective_polarizations: List[AffectivePolarization]
    issue_polarizations: List[IssuePolarization]
    depolarization_opportunities: List[DepolarizationOpportunity]
    
    # Aggregate metrics
    aggregate_metrics: Dict[str, float]
    
    # Validation results
    validation_results: Dict[str, Any]
    
    # Errors and warnings
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline results."""
        return {
            'pipeline_id': self.pipeline_id,
            'execution': {
                'timestamp': self.execution_timestamp.isoformat(),
                'duration_seconds': self.execution_duration,
                'duration_readable': f"{self.execution_duration/60:.2f} minutes"
            },
            'results_count': {
                'signals': len(self.aggregated_signals),
                'bursts': len(self.burst_events),
                'cascades': len(self.cascades),
                'multi_frame_patterns': len(self.multi_frame_patterns),
                'paradigm_shifts': len(self.paradigm_shifts),
                'media_profiles': len(self.media_profiles),
                'media_coordinations': len(self.media_coordinations),
                'media_clusters': len(self.media_clusters),
                'sequences': len(self.cascade_sequences),
                'temporal_motifs': len(self.temporal_motifs),
                'echo_chambers': len(self.echo_chambers),
                'filter_bubbles': len(self.filter_bubbles),
                'polarization_patterns': len(self.polarization_patterns),
                'depolarization_opportunities': len(self.depolarization_opportunities)
            },
            'key_findings': self._extract_key_findings(),
            'aggregate_metrics': self.aggregate_metrics,
            'validation': self.validation_results,
            'issues': {
                'errors': len(self.errors),
                'warnings': len(self.warnings)
            }
        }
    
    def _extract_key_findings(self) -> Dict[str, Any]:
        """Extract key findings from results."""
        findings = {}
        
        # Most intense cascades
        if self.cascades:
            top_cascades = sorted(self.cascades, 
                                key=lambda c: c.intensity_score, 
                                reverse=True)[:3]
            findings['top_cascades'] = [
                {
                    'id': c.cascade_id,
                    'intensity': c.intensity_score,
                    'frame': c.primary_frame,
                    'media_count': len(c.media_involved)
                } for c in top_cascades
            ]
        
        # Strongest polarization
        if self.polarization_patterns:
            strongest_polarization = max(self.polarization_patterns,
                                        key=lambda p: p.polarization_score)
            findings['strongest_polarization'] = {
                'id': strongest_polarization.pattern_id,
                'score': strongest_polarization.polarization_score,
                'type': strongest_polarization.pattern_type,
                'n_poles': strongest_polarization.n_poles
            }
        
        # Largest echo chamber
        if self.echo_chambers:
            largest_chamber = max(self.echo_chambers,
                                key=lambda e: e.total_members)
            findings['largest_echo_chamber'] = {
                'id': largest_chamber.chamber_id,
                'members': largest_chamber.total_members,
                'isolation': largest_chamber.isolation_score
            }
        
        # Most promising depolarization opportunity
        if self.depolarization_opportunities:
            best_opportunity = max(self.depolarization_opportunities,
                                 key=lambda o: o.success_probability * o.impact_magnitude)
            findings['best_depolarization_opportunity'] = {
                'id': best_opportunity.opportunity_id,
                'type': best_opportunity.opportunity_type,
                'feasibility': best_opportunity.get_feasibility()
            }
        
        return findings


@dataclass
class PipelineConfig:
    """
    Configuration for Phase 3 pipeline execution.
    """
    # Detector configurations
    detector_config: DetectorConfig = field(default_factory=DetectorConfig)
    
    # Execution settings
    parallel_execution: bool = True
    n_workers: int = 8
    use_gpu: bool = False
    
    # Data settings
    time_window: Tuple[datetime, datetime] = field(
        default_factory=lambda: (datetime(2019, 1, 1), datetime(2019, 12, 31))
    )
    frames_to_analyze: List[str] = field(
        default_factory=lambda: ['Pol', 'Eco', 'Sci', 'Just', 'Cult', 'Envt', 'Pbh', 'Secu']
    )
    
    # Caching settings
    use_cache: bool = True
    cache_dir: Path = Path("cache/phase3")
    
    # Export settings
    export_results: bool = True
    export_dir: Path = Path("results/phase3")
    export_formats: List[str] = field(default_factory=lambda: ['json', 'pickle', 'csv'])
    
    # Validation settings
    validate_results: bool = True
    validation_threshold: float = 0.95
    
    # Memory management
    batch_size: int = 10000
    max_memory_gb: float = 32.0
    
    # Logging
    verbose: bool = True
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'parallel_execution': self.parallel_execution,
            'n_workers': self.n_workers,
            'use_gpu': self.use_gpu,
            'time_window': (self.time_window[0].isoformat(), 
                          self.time_window[1].isoformat()),
            'frames': self.frames_to_analyze,
            'use_cache': self.use_cache,
            'export_results': self.export_results,
            'validate_results': self.validate_results
        }


class Phase3Pipeline:
    """
    Complete orchestration pipeline for Phase 3 cascade detection.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize Phase 3 pipeline."""
        self.config = config or PipelineConfig()
        self.pipeline_id = str(uuid.uuid4())
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger(f"Phase3Pipeline_{self.pipeline_id[:8]}")
        
        # Initialize all detectors
        self.logger.info("Initializing Phase 3 detectors...")
        
        # Note: Detectors will be properly initialized with context during run()
        # For now, store the config for later initialization
        self.signal_aggregator = None
        self.burst_detector = None
        self.cascade_detector = None
        self.multi_frame_detector = None
        self.cross_media_tracker = None
        self.sequence_detector = None
        self.echo_chamber_detector = None
        self.polarization_detector = None
        
        # Phase 1-2 components
        self.data_processor = DataProcessor(self.config.detector_config)
        self.index_manager = IndexManager(self.config.detector_config)
        
        # Metrics calculators (will be initialized with indices during run)
        self.network_metrics = None
        self.exhaustive_calculator = None
        self.temporal_metrics = None
        self.diversity_metrics = None
        self.convergence_metrics = None
        
        # Storage
        self.results: Optional[Phase3Results] = None
        self.context: Optional[DetectionContext] = None
        
        # Cache
        if self.config.use_cache:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Export directory
        if self.config.export_results:
            self.config.export_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Phase3Pipeline initialized with ID: {self.pipeline_id}")
    
    def run(self, data: pd.DataFrame, 
            indices: Optional[Dict[str, Any]] = None,
            context: Optional[DetectionContext] = None) -> Phase3Results:
        """
        Execute complete Phase 3 pipeline.
        
        Args:
            data: Input DataFrame with cascade data
            indices: Pre-built indices (optional)
            context: Detection context (optional)
            
        Returns:
            Complete Phase 3 results
        """
        self.logger.info(f"Starting Phase 3 pipeline execution...")
        start_time = time.time()
        
        # Initialize results
        errors = []
        warnings = []
        
        try:
            # Step 1: Prepare data and context
            self.logger.info("Step 1: Preparing data and context...")
            data, indices, context = self._prepare_data_and_context(data, indices, context)
            self.context = context
            
            # Initialize metrics calculators if not already initialized
            if self.network_metrics is None:
                self.logger.info("Initializing metrics calculators...")
                self.network_metrics = ScientificNetworkMetrics(
                    source_index=indices.get('sources', indices.get('source', {})),
                    entity_index=indices.get('entities', indices.get('entity', {}))
                )
                self.exhaustive_calculator = ExhaustiveMetricsCalculator()
                self.temporal_metrics = TemporalMetrics(
                    temporal_index=indices.get('temporal', {})
                )
                self.diversity_metrics = DiversityMetrics(
                    frame_index=indices.get('frames', indices.get('frame', {})),
                    source_index=indices.get('sources', indices.get('source', {})),
                    emotion_index=indices.get('emotions', indices.get('emotion', {}))
                )
                self.convergence_metrics = ConvergenceMetrics(
                    source_index=indices.get('sources', indices.get('source', {})),
                    entity_index=indices.get('entities', indices.get('entity', {})),
                    emotion_index=indices.get('emotions', indices.get('emotion', {})),
                    temporal_index=indices.get('temporal', {}),
                    frame_index=indices.get('frames', indices.get('frame', {}))
                )
            
            # Initialize detectors with context if not already initialized
            if self.signal_aggregator is None:
                self.logger.info("Initializing detectors with context...")
                self.signal_aggregator = SignalAggregator(context, self.config.detector_config)
                self.burst_detector = BurstDetector(context, self.config.detector_config)
                self.cascade_detector = CascadeDetector(context, self.burst_detector, self.config.detector_config)
                # TODO: These detectors need their abstract methods implemented
                # self.multi_frame_detector = MultiFrameDetector(context, self.config.detector_config)
                # self.cross_media_tracker = CrossMediaTracker(context, self.config.detector_config)
                # self.sequence_detector = SequenceDetector(context, self.config.detector_config)
                # self.echo_chamber_detector = EchoChamberDetector(context, self.config.detector_config)
                # self.polarization_detector = PolarizationDetector(context, self.config.detector_config)
                self.logger.warning("Some detectors are not yet fully implemented")
            
            # Step 2: Signal aggregation
            self.logger.info("Step 2: Aggregating signals from Phase 1-2...")
            aggregated_signals = self._aggregate_signals(context)
            
            # Step 3: Burst detection
            self.logger.info("Step 3: Detecting temporal bursts...")
            burst_events = self._detect_bursts(context, aggregated_signals)
            
            # Step 4: Cascade detection with full integration
            self.logger.info("Step 4: Detecting cascades with full integration...")
            cascades = self._detect_cascades(context, burst_events, aggregated_signals)
            
            # Step 5: Multi-frame pattern detection
            self.logger.info("Step 5-9: Advanced detectors skipped - not fully implemented")
            multi_frame_results = {'patterns': [], 'paradigm_shifts': []}
            media_tracking_results = {'profiles': {}, 'coordinations': [], 'clusters': [], 
                                    'journalist_networks': [], 'information_flows': []}
            sequence_results = {'sequences': [], 'motifs': [], 'evolutions': [], 'pathways': []}
            echo_chamber_results = {'chambers': [], 'bubbles': [], 'silos': [], 
                                  'homophily': [], 'bridges': [], 'spirals': []}
            polarization_results = {'patterns': [], 'opinion_clusters': [], 'axes': [], 
                                  'affective': [], 'issues': [], 'opportunities': []}
            
            # Step 10: Aggregate and validate results
            self.logger.info("Step 10: Aggregating and validating results...")
            aggregate_metrics = self._calculate_aggregate_metrics(
                cascades, multi_frame_results, echo_chamber_results, polarization_results
            )
            
            validation_results = self._validate_results(
                cascades, sequence_results, echo_chamber_results, polarization_results
            )
            
            # Calculate execution time
            execution_duration = time.time() - start_time
            
            # Create results object
            self.results = Phase3Results(
                pipeline_id=self.pipeline_id,
                execution_timestamp=datetime.now(),
                execution_duration=execution_duration,
                config_used=self.config.detector_config,
                data_statistics=self._calculate_data_statistics(data),
                indices_statistics=self._calculate_indices_statistics(indices),
                aggregated_signals=aggregated_signals,
                burst_events=burst_events,
                cascades=cascades,
                multi_frame_patterns=multi_frame_results['patterns'],
                paradigm_shifts=multi_frame_results['paradigm_shifts'],
                media_profiles=media_tracking_results['profiles'],
                media_coordinations=media_tracking_results['coordinations'],
                media_clusters=media_tracking_results['clusters'],
                journalist_networks=media_tracking_results['journalist_networks'],
                information_flows=media_tracking_results['information_flows'],
                cascade_sequences=sequence_results['sequences'],
                temporal_motifs=sequence_results['motifs'],
                narrative_evolutions=sequence_results['evolutions'],
                information_pathways=sequence_results['pathways'],
                echo_chambers=echo_chamber_results['chambers'],
                filter_bubbles=echo_chamber_results['bubbles'],
                information_silos=echo_chamber_results['silos'],
                homophily_patterns=echo_chamber_results['homophily'],
                bridge_actors=echo_chamber_results['bridges'],
                reinforcement_spirals=echo_chamber_results['spirals'],
                polarization_patterns=polarization_results['patterns'],
                opinion_clusters=polarization_results['opinion_clusters'],
                polarization_axes=polarization_results['axes'],
                affective_polarizations=polarization_results['affective'],
                issue_polarizations=polarization_results['issues'],
                depolarization_opportunities=polarization_results['opportunities'],
                aggregate_metrics=aggregate_metrics,
                validation_results=validation_results,
                errors=errors,
                warnings=warnings
            )
            
            # Export results if configured
            if self.config.export_results:
                self._export_results(self.results)
            
            # Cache results if configured
            if self.config.use_cache:
                self._cache_results(self.results)
            
            self.logger.info(f"Pipeline execution completed in {execution_duration:.2f} seconds")
            self.logger.info(f"Detected: {len(cascades)} cascades, "
                           f"{len(echo_chamber_results['chambers'])} echo chambers, "
                           f"{len(polarization_results['patterns'])} polarization patterns")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            errors.append({
                'stage': 'pipeline_execution',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
            # Return partial results if possible
            return self._create_error_results(errors, warnings)
    
    def _prepare_data_and_context(self, data: pd.DataFrame,
                                 indices: Optional[Dict[str, Any]],
                                 context: Optional[DetectionContext]) -> Tuple[pd.DataFrame, Dict[str, Any], DetectionContext]:
        """Prepare data and detection context."""
        # Process data
        if 'date_converted' not in data.columns:
            self.logger.info("Processing frame data...")
            data = self.data_processor.process_frame_data(data, show_progress=True)
        
        # Build indices if not provided
        if indices is None:
            self.logger.info("Building indices from data...")
            indices = self.index_manager.build_all_indices(
                data, 
                parallel=self.config.parallel_execution
            )
        
        # Create context if not provided
        if context is None:
            self.logger.info("Creating detection context...")
            # Log available indices
            self.logger.info(f"Available indices: {list(indices.keys())}")
            for idx_name, idx_data in indices.items():
                if isinstance(idx_data, dict):
                    self.logger.info(f"  {idx_name}: {len(idx_data)} entries")
                else:
                    self.logger.info(f"  {idx_name}: {type(idx_data)}")
            
            context = DetectionContext(
                temporal_index=indices.get('temporal', {}),
                entity_index=indices.get('entities', indices.get('entity', {})),  # Handle both names
                source_index=indices.get('sources', indices.get('source', {})),   # Handle both names
                frame_index=indices.get('frames', indices.get('frame', {})),       # Handle both names
                emotion_index=indices.get('emotions', indices.get('emotion', {})), # Handle both names
                geographic_index=indices.get('geographic'),
                time_window=self.config.time_window,
                frames=self.config.frames_to_analyze
            )
            
            # Initialize metrics
            context.initialize_metrics()
        
        # Validate context with detailed logging
        if not context.validate():
            self.logger.error("Context validation failed")
            self.logger.error(f"temporal_index: {bool(context.temporal_index)}")
            self.logger.error(f"entity_index: {bool(context.entity_index)}")
            self.logger.error(f"source_index: {bool(context.source_index)}")
            self.logger.error(f"frame_index: {bool(context.frame_index)}")
            self.logger.error(f"emotion_index: {bool(context.emotion_index)}")
            raise ValueError("Invalid detection context")
        
        return data, indices, context
    
    def _aggregate_signals(self, context: DetectionContext) -> List[AggregatedSignal]:
        """Aggregate all signals from Phase 1-2."""
        signals = []
        
        # Define time windows for aggregation
        windows = self._generate_time_windows(context.time_window)
        
        for window in tqdm(windows, desc="Aggregating signals"):
            for frame in context.frames:
                signal = self.signal_aggregator.aggregate_signals(
                    frame=frame,
                    window=window
                )
                
                if signal and signal.n_articles > 0:
                    signals.append(signal)
        
        self.logger.info(f"Aggregated {len(signals)} signals")
        return signals
    
    def _detect_bursts(self, context: DetectionContext,
                      signals: List[AggregatedSignal]) -> List[BurstEvent]:
        """Detect temporal bursts."""
        bursts = self.burst_detector.detect_from_signals(signals, context)
        
        # Filter significant bursts
        significant_bursts = [b for b in bursts if b.is_significant]
        
        self.logger.info(f"Detected {len(significant_bursts)} significant bursts")
        return significant_bursts
    
    def _detect_cascades(self, context: DetectionContext,
                        bursts: List[BurstEvent],
                        signals: List[AggregatedSignal]) -> List[CompleteCascade]:
        """Detect cascades with full integration."""
        # CascadeDetector already integrates everything
        # The detect method expects kwargs, not positional args
        cascades = self.cascade_detector.detect()
        
        self.logger.info(f"Detected {len(cascades)} complete cascades")
        return cascades
    
    def _detect_multi_frame_patterns(self, context: DetectionContext,
                                    cascades: List[CompleteCascade]) -> Dict[str, Any]:
        """Detect multi-frame patterns."""
        # MultiFrameDetector doesn't take context (uses kwargs)
        patterns = self.multi_frame_detector.detect()
        paradigm_shifts = self.multi_frame_detector.paradigm_shifts
        
        self.logger.info(f"Detected {len(patterns)} multi-frame patterns, "
                        f"{len(paradigm_shifts)} paradigm shifts")
        
        return {
            'patterns': patterns,
            'paradigm_shifts': paradigm_shifts
        }
    
    def _track_cross_media(self, context: DetectionContext,
                          cascades: List[CompleteCascade]) -> Dict[str, Any]:
        """Track cross-media behavior."""
        # Build media profiles
        profiles = self.cross_media_tracker.build_media_profiles(context)
        
        # Detect coordination
        coordinations = self.cross_media_tracker.detect_coordination(context)
        
        # Detect clusters
        clusters = self.cross_media_tracker.detect_clusters(context)
        
        # Build journalist networks
        journalist_networks = self.cross_media_tracker.build_journalist_networks(context)
        
        # Analyze information flows
        information_flows = self.cross_media_tracker.analyze_information_flows(context)
        
        self.logger.info(f"Built {len(profiles)} media profiles, "
                        f"detected {len(coordinations)} coordinations, "
                        f"{len(clusters)} clusters")
        
        return {
            'profiles': profiles,
            'coordinations': coordinations,
            'clusters': clusters,
            'journalist_networks': journalist_networks,
            'information_flows': information_flows
        }
    
    def _detect_sequences(self, context: DetectionContext,
                         cascades: List[CompleteCascade]) -> Dict[str, Any]:
        """Detect temporal sequences."""
        # SequenceDetector handles everything
        sequences = self.sequence_detector.detect(context)
        
        # Get additional results
        motifs = self.sequence_detector.temporal_motifs
        evolutions = self.sequence_detector.narrative_evolutions
        pathways = self.sequence_detector.information_pathways
        
        self.logger.info(f"Detected {len(sequences)} sequences, "
                        f"{len(motifs)} temporal motifs, "
                        f"{len(pathways)} information pathways")
        
        return {
            'sequences': sequences,
            'motifs': motifs,
            'evolutions': evolutions,
            'pathways': pathways
        }
    
    def _detect_echo_chambers(self, context: DetectionContext,
                             cascades: List[CompleteCascade],
                             sequence_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect echo chambers and related patterns."""
        # EchoChamberDetector handles everything
        chambers = self.echo_chamber_detector.detect(context)
        
        # Get additional results
        bubbles = self.echo_chamber_detector.filter_bubbles
        silos = self.echo_chamber_detector.information_silos
        homophily = self.echo_chamber_detector.homophily_patterns
        bridges = self.echo_chamber_detector.bridge_actors
        spirals = self.echo_chamber_detector.reinforcement_spirals
        
        self.logger.info(f"Detected {len(chambers)} echo chambers, "
                        f"{len(bubbles)} filter bubbles, "
                        f"{len(bridges)} bridge actors")
        
        return {
            'chambers': chambers,
            'bubbles': bubbles,
            'silos': silos,
            'homophily': homophily,
            'bridges': bridges,
            'spirals': spirals
        }
    
    def _detect_polarization(self, context: DetectionContext,
                            cascades: List[CompleteCascade],
                            sequence_results: Dict[str, Any],
                            echo_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect polarization patterns."""
        # PolarizationDetector handles everything
        patterns = self.polarization_detector.detect(context)
        
        # Get additional results
        opinion_clusters = self.polarization_detector.opinion_clusters
        axes = self.polarization_detector.polarization_axes
        affective = self.polarization_detector.affective_polarizations
        issues = self.polarization_detector.issue_polarizations
        opportunities = self.polarization_detector.depolarization_opportunities
        
        self.logger.info(f"Detected {len(patterns)} polarization patterns, "
                        f"{len(opportunities)} depolarization opportunities")
        
        return {
            'patterns': patterns,
            'opinion_clusters': opinion_clusters,
            'axes': axes,
            'affective': affective,
            'issues': issues,
            'opportunities': opportunities
        }
    
    def _calculate_aggregate_metrics(self, cascades: List[CompleteCascade],
                                    multi_frame_results: Dict[str, Any],
                                    echo_results: Dict[str, Any],
                                    polarization_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate aggregate metrics across all detections."""
        metrics = {}
        
        # Cascade metrics
        if cascades:
            metrics['avg_cascade_intensity'] = np.mean([c.intensity_score for c in cascades])
            metrics['max_cascade_intensity'] = max(c.intensity_score for c in cascades)
            metrics['total_cascade_reach'] = sum(len(c.media_involved) for c in cascades)
            metrics['avg_cascade_duration'] = np.mean([c.duration_days for c in cascades])
        
        # Multi-frame metrics
        if multi_frame_results['patterns']:
            metrics['frame_convergence'] = np.mean([
                p.convergence_score for p in multi_frame_results['patterns']
            ])
            metrics['paradigm_shift_likelihood'] = len(multi_frame_results['paradigm_shifts']) / \
                                                  len(multi_frame_results['patterns']) \
                                                  if multi_frame_results['patterns'] else 0
        
        # Echo chamber metrics
        if echo_results['chambers']:
            metrics['avg_echo_isolation'] = np.mean([
                c.isolation_score for c in echo_results['chambers']
            ])
            metrics['echo_chamber_prevalence'] = sum(
                c.total_members for c in echo_results['chambers']
            ) / 1000  # Normalize
        
        # Polarization metrics
        if polarization_results['patterns']:
            metrics['avg_polarization'] = np.mean([
                p.polarization_score for p in polarization_results['patterns']
            ])
            metrics['max_polarization'] = max(
                p.polarization_score for p in polarization_results['patterns']
            )
            metrics['depolarization_potential'] = np.mean([
                o.success_probability * o.impact_magnitude 
                for o in polarization_results['opportunities']
            ]) if polarization_results['opportunities'] else 0
        
        # Overall system metrics
        metrics['information_diversity'] = self._calculate_diversity(cascades)
        metrics['network_fragmentation'] = self._calculate_fragmentation(echo_results)
        metrics['cascade_predictability'] = self._calculate_predictability(cascades)
        
        return metrics
    
    def _calculate_diversity(self, cascades: List[CompleteCascade]) -> float:
        """Calculate overall information diversity."""
        if not cascades:
            return 0.0
        
        # Frame diversity
        frame_counts = Counter()
        for cascade in cascades:
            frame_counts[cascade.primary_frame] += 1
        
        if frame_counts:
            probs = np.array(list(frame_counts.values())) / sum(frame_counts.values())
            frame_diversity = -np.sum(probs * np.log2(probs + 1e-10))
        else:
            frame_diversity = 0.0
        
        # Entity diversity
        all_entities = set()
        for cascade in cascades:
            all_entities.update(e['entity'] for e in cascade.entities_involved[:50])
        
        entity_diversity = np.log2(len(all_entities) + 1) / 10  # Normalize
        
        # Media diversity
        all_media = set()
        for cascade in cascades:
            all_media.update(cascade.media_involved)
        
        media_diversity = len(all_media) / 100  # Normalize
        
        return np.mean([frame_diversity, entity_diversity, media_diversity])
    
    def _calculate_fragmentation(self, echo_results: Dict[str, Any]) -> float:
        """Calculate network fragmentation."""
        if not echo_results['chambers']:
            return 0.0
        
        # Number of isolated groups
        n_chambers = len(echo_results['chambers'])
        
        # Average isolation
        avg_isolation = np.mean([c.isolation_score for c in echo_results['chambers']])
        
        # Cross-chamber connectivity (inverse)
        disconnection = 1 - np.mean([c.permeability for c in echo_results['chambers']])
        
        return np.mean([n_chambers / 20, avg_isolation, disconnection])  # Normalize
    
    def _calculate_predictability(self, cascades: List[CompleteCascade]) -> float:
        """Calculate cascade predictability."""
        if len(cascades) < 2:
            return 0.0
        
        # Sort by time
        sorted_cascades = sorted(cascades, key=lambda c: c.start_date)
        
        # Calculate inter-cascade intervals
        intervals = []
        for i in range(len(sorted_cascades) - 1):
            interval = (sorted_cascades[i+1].start_date - sorted_cascades[i].end_date).days
            intervals.append(max(0, interval))
        
        if intervals:
            # Low variance in intervals = high predictability
            cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
            predictability = 1 / (1 + cv)
            return predictability
        
        return 0.0
    
    def _validate_results(self, cascades: List[CompleteCascade],
                         sequence_results: Dict[str, Any],
                         echo_results: Dict[str, Any],
                         polarization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate detection results."""
        validation = {
            'passed': True,
            'checks': {}
        }
        
        # Check cascade validity
        cascade_validity = all(
            c.significance > self.config.validation_threshold 
            for c in cascades[:10]  # Check top 10
        )
        validation['checks']['cascade_validity'] = cascade_validity
        
        # Check sequence coherence
        sequence_coherence = all(
            s.coherence_score > 0.5 
            for s in sequence_results['sequences'][:10]
        )
        validation['checks']['sequence_coherence'] = sequence_coherence
        
        # Check echo chamber significance
        chamber_significance = all(
            c.significance > self.config.validation_threshold
            for c in echo_results['chambers'][:10]
        )
        validation['checks']['chamber_significance'] = chamber_significance
        
        # Check polarization confidence
        polarization_confidence = all(
            p.confidence > 0.7
            for p in polarization_results['patterns'][:10]
        )
        validation['checks']['polarization_confidence'] = polarization_confidence
        
        # Overall validation
        validation['passed'] = all(validation['checks'].values())
        validation['score'] = sum(validation['checks'].values()) / len(validation['checks'])
        
        return validation
    
    def _generate_time_windows(self, time_range: Tuple[datetime, datetime],
                              window_size: int = 7) -> List[Tuple[datetime, datetime]]:
        """Generate time windows for analysis."""
        windows = []
        current = time_range[0]
        
        while current < time_range[1]:
            window_end = min(current + timedelta(days=window_size), time_range[1])
            windows.append((current, window_end))
            current = window_end
        
        return windows
    
    def _calculate_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics about input data."""
        return {
            'n_rows': len(data),
            'date_range': {
                'start': data['date_converted'].min().isoformat() if 'date_converted' in data.columns else None,
                'end': data['date_converted'].max().isoformat() if 'date_converted' in data.columns else None
            },
            'n_unique_media': data['media'].nunique() if 'media' in data.columns else 0,
            'n_unique_authors': data['author'].nunique() if 'author' in data.columns else 0,
            'frames_present': list(data.columns[data.columns.str.contains('frame_')]) if any('frame_' in col for col in data.columns) else []
        }
    
    def _calculate_indices_statistics(self, indices: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics about indices."""
        stats = {}
        
        for index_name, index_data in indices.items():
            if isinstance(index_data, dict):
                stats[index_name] = {
                    'n_entries': len(index_data),
                    'memory_mb': self._estimate_memory(index_data)
                }
        
        return stats
    
    def _estimate_memory(self, obj: Any) -> float:
        """Estimate memory usage of object in MB."""
        import sys
        
        def get_size(obj, seen=None):
            size = sys.getsizeof(obj)
            if seen is None:
                seen = set()
            
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            
            seen.add(obj_id)
            
            if isinstance(obj, dict):
                size += sum([get_size(v, seen) for v in obj.values()])
                size += sum([get_size(k, seen) for k in obj.keys()])
            elif hasattr(obj, '__dict__'):
                size += get_size(obj.__dict__, seen)
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
                size += sum([get_size(i, seen) for i in obj])
            
            return size
        
        return get_size(obj) / (1024 * 1024)  # Convert to MB
    
    def _export_results(self, results: Phase3Results) -> None:
        """Export results to configured formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"phase3_results_{self.pipeline_id[:8]}_{timestamp}"
        
        # Export to JSON
        if 'json' in self.config.export_formats:
            json_path = self.config.export_dir / f"{base_name}.json"
            self._export_json(results, json_path)
            self.logger.info(f"Exported results to {json_path}")
        
        # Export to pickle
        if 'pickle' in self.config.export_formats:
            pickle_path = self.config.export_dir / f"{base_name}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(results, f)
            self.logger.info(f"Exported results to {pickle_path}")
        
        # Export summary to CSV
        if 'csv' in self.config.export_formats:
            csv_path = self.config.export_dir / f"{base_name}_summary.csv"
            self._export_summary_csv(results, csv_path)
            self.logger.info(f"Exported summary to {csv_path}")
    
    def _export_json(self, results: Phase3Results, path: Path) -> None:
        """Export results to JSON format."""
        # Convert to serializable format
        export_data = {
            'pipeline_id': results.pipeline_id,
            'execution': {
                'timestamp': results.execution_timestamp.isoformat(),
                'duration': results.execution_duration
            },
            'summary': results.get_summary(),
            'cascades': [
                {
                    'id': c.cascade_id,
                    'intensity': c.intensity_score,
                    'frame': c.primary_frame,
                    'start': c.start_date.isoformat(),
                    'peak': c.peak_date.isoformat(),
                    'media_count': len(c.media_involved)
                } for c in results.cascades[:100]  # Limit to top 100
            ],
            'echo_chambers': [
                {
                    'id': e.chamber_id,
                    'members': e.total_members,
                    'isolation': e.isolation_score,
                    'type': e.chamber_type
                } for e in results.echo_chambers
            ],
            'polarization': [
                {
                    'id': p.pattern_id,
                    'score': p.polarization_score,
                    'type': p.pattern_type,
                    'n_poles': p.n_poles
                } for p in results.polarization_patterns
            ],
            'aggregate_metrics': results.aggregate_metrics,
            'validation': results.validation_results
        }
        
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def _export_summary_csv(self, results: Phase3Results, path: Path) -> None:
        """Export summary statistics to CSV."""
        summary_data = []
        
        # Cascade summary
        for cascade in results.cascades[:100]:
            summary_data.append({
                'type': 'cascade',
                'id': cascade.cascade_id,
                'metric1': cascade.intensity_score,
                'metric2': len(cascade.media_involved),
                'metric3': cascade.duration_days,
                'label': cascade.primary_frame
            })
        
        # Echo chamber summary
        for chamber in results.echo_chambers:
            summary_data.append({
                'type': 'echo_chamber',
                'id': chamber.chamber_id,
                'metric1': chamber.isolation_score,
                'metric2': chamber.total_members,
                'metric3': chamber.echo_strength,
                'label': chamber.chamber_type
            })
        
        # Polarization summary
        for pattern in results.polarization_patterns:
            summary_data.append({
                'type': 'polarization',
                'id': pattern.pattern_id,
                'metric1': pattern.polarization_score,
                'metric2': pattern.n_poles,
                'metric3': pattern.reversibility,
                'label': pattern.pattern_type
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(path, index=False)
    
    def _cache_results(self, results: Phase3Results) -> None:
        """Cache results for future use."""
        cache_file = self.config.cache_dir / f"results_{self.pipeline_id}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Also save metadata
        metadata_file = self.config.cache_dir / f"metadata_{self.pipeline_id}.json"
        metadata = {
            'pipeline_id': self.pipeline_id,
            'timestamp': results.execution_timestamp.isoformat(),
            'duration': results.execution_duration,
            'n_cascades': len(results.cascades),
            'n_echo_chambers': len(results.echo_chambers),
            'n_polarization_patterns': len(results.polarization_patterns)
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _create_error_results(self, errors: List[Dict[str, Any]],
                            warnings: List[Dict[str, Any]]) -> Phase3Results:
        """Create results object with errors."""
        return Phase3Results(
            pipeline_id=self.pipeline_id,
            execution_timestamp=datetime.now(),
            execution_duration=0.0,
            config_used=self.config.detector_config,
            data_statistics={},
            indices_statistics={},
            aggregated_signals=[],
            burst_events=[],
            cascades=[],
            multi_frame_patterns=[],
            paradigm_shifts=[],
            media_profiles={},
            media_coordinations=[],
            media_clusters=[],
            journalist_networks=[],
            information_flows=[],
            cascade_sequences=[],
            temporal_motifs=[],
            narrative_evolutions=[],
            information_pathways=[],
            echo_chambers=[],
            filter_bubbles=[],
            information_silos=[],
            homophily_patterns=[],
            bridge_actors=[],
            reinforcement_spirals=[],
            polarization_patterns=[],
            opinion_clusters=[],
            polarization_axes=[],
            affective_polarizations=[],
            issue_polarizations=[],
            depolarization_opportunities=[],
            aggregate_metrics={},
            validation_results={'passed': False, 'error': 'Pipeline execution failed'},
            errors=errors,
            warnings=warnings
        )
    
    def load_cached_results(self, pipeline_id: str) -> Optional[Phase3Results]:
        """Load cached results from previous run."""
        cache_file = self.config.cache_dir / f"results_{pipeline_id}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                results = pickle.load(f)
            
            self.logger.info(f"Loaded cached results for pipeline {pipeline_id}")
            return results
        
        return None
    
    def get_results_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of current results."""
        if self.results:
            return self.results.get_summary()
        return None
    
    def visualize_results(self, output_dir: Optional[Path] = None) -> None:
        """Generate visualizations of results (requires matplotlib)."""
        if not self.results:
            self.logger.warning("No results to visualize")
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            output_dir = output_dir or self.config.export_dir / "visualizations"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create visualization suite
            self._visualize_cascade_timeline(output_dir)
            self._visualize_polarization_map(output_dir)
            self._visualize_echo_chamber_network(output_dir)
            self._visualize_information_flow(output_dir)
            
            self.logger.info(f"Visualizations saved to {output_dir}")
            
        except ImportError:
            self.logger.warning("Matplotlib not available, skipping visualizations")
    
    def _visualize_cascade_timeline(self, output_dir: Path) -> None:
        """Visualize cascade timeline."""
        # Implementation would create timeline visualization
        pass
    
    def _visualize_polarization_map(self, output_dir: Path) -> None:
        """Visualize polarization landscape."""
        # Implementation would create polarization visualization
        pass
    
    def _visualize_echo_chamber_network(self, output_dir: Path) -> None:
        """Visualize echo chamber network structure."""
        # Implementation would create network visualization
        pass
    
    def _visualize_information_flow(self, output_dir: Path) -> None:
        """Visualize information flow patterns."""
        # Implementation would create flow visualization
        pass


def run_phase3_pipeline(data_path: str,
                       config: Optional[PipelineConfig] = None) -> Phase3Results:
    """
    Convenience function to run Phase 3 pipeline.
    
    Args:
        data_path: Path to input data
        config: Pipeline configuration
        
    Returns:
        Phase 3 results
    """
    # Load data
    data = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_pickle(data_path)
    
    # Create pipeline
    pipeline = Phase3Pipeline(config)
    
    # Run pipeline
    results = pipeline.run(data)
    
    return results