# CCF Media Cascade Detection Framework

A scientific framework for detecting and scoring information cascades in large-scale media corpora, built for the [Canadian Climate Framing (CCF) project](https://ccf-project.ca).

## Overview

This framework analyzes climate change media coverage across 20 major Canadian newspapers (1978–2024) to detect **media cascades** — coordinated surges of coverage where content converges, new journalists enter, and narratives amplify across outlets.

The system operates on the CCF database of **9.2 million sentence-level annotations** extracted from 266,271 articles. Each sentence is annotated with 60+ features: 8 climate frames, 9 messenger types, 8 event types, 3 emotional tones, named entities, and metadata. Pre-computed sentence embeddings (BAAI/bge-m3, 1024 dimensions) enable semantic analysis across the bilingual English-French corpus.

### Core research question

> When does a news topic shift from routine reporting to a self-reinforcing cascade where content converges, new journalists enter, and coverage amplifies across outlets?

### What is a cascade?

A media cascade, as operationalized in this framework, is a period during which:

1. **Temporal anomaly** — Coverage of a specific frame rises significantly above its 90-day trailing baseline
2. **Participation broadening** — New journalists and media outlets begin covering the topic
3. **Content convergence** — Articles become semantically more similar over time (measured via embeddings)
4. **Source coordination** — Messenger profiles concentrate and journalist-level content centroids align

The framework distinguishes cascades from routine coverage spikes by requiring anomalies across multiple dimensions, not just volume.

## Architecture

### Pipeline

The `CascadeDetectionPipeline` executes seven sequential steps:

```
PostgreSQL (9.2M sentences, 77 columns)
    │
    ▼
Step 1: LOAD & PROCESS
    DatabaseConnector → raw DataFrame (sentence-level)
    DataProcessor → cleaned DataFrame + article-level aggregation
    │
    ▼
Step 2: BUILD INDICES
    IndexManager → 6 specialized indices
    (temporal, frame, emotion, entity, source, geographic)
    │
    ▼
Step 3: DETECT & SCORE (per frame × 8 frames)
    DailySignalBuilder → 5 daily Z-score signals → composite
    Signal orthogonalization (z_convergence ⊥ z_temporal)
    Burst detection → Z-score flagging + CUSUM change-point
    Semantic peak → embedding-weighted P50 for cascade peak_date
    Cascade scoring → 4 dimensions, 17 sub-indices
    Media confidence factor → log₂(n_media)/log₂(10) multiplier
    Classification → strong / moderate / weak / not_cascade
    │
    ▼
Step 3.5: EVENT OCCURRENCE DETECTION (database-first)
    EventOccurrenceDetector → detect events on ALL articles
    Phase 2: per-type HAC clustering on seeds (title+sentence embeddings)
    Phase 3: multi-type EventCluster merge (silhouette-optimized)
    Phase 4: iterative 4D soft assignment (all articles → clusters)
    Phase 5: confidence scoring (5 components)
    │
    ▼
Step 3.6: EVENT ATTRIBUTION
    attribute_to_cascades() → link occurrences to cascades
    Temporal overlap + shared articles → CascadeAttribution objects
    │
    ▼
Step 4: PARADIGM SHIFT ANALYSIS
    ParadigmShiftAnalyzer → paradigm states, shifts, episodes
    Uses CCF-paradigm 4-method consensus (12-week window, daily step)
    Shift detection → qualify (duration, structure, reversibility)
    Three-role cascade attribution → amplification / déstabilisation / dormante
    Episode grouping → two-level dynamics qualification
    │
    ▼
Step 5: STABILITY SELECTION IMPACT ANALYSIS (cluster → cascade)
    StabSelImpactAnalyzer → variable selection + causal inference
    Treatment: D_j(t,l) = Σ belonging(a,j) × frame_signal(a) × cosine_sim(a, centroid)
    Stability Selection (B=100, ElasticNet, π≥0.60) → stable cluster set
    OLS post-selection + residual bootstrap → net_beta, p-values
    Role classification → driver / suppressor / neutral (α=0.10)
    │
    ▼
RESULTS
    DetectionResults → JSON, Parquet, DataFrame exports
    EventClusters, EventOccurrences, CascadeAttributions, StabSelImpactResults
```

### Step 3 — Detection: five daily anomaly signals

For each of the 8 climate frames, five daily Z-score signals are computed against a 90-day trailing baseline (no lookahead bias):

| Signal | Measures | Raw input |
|--------|----------|-----------|
| **Temporal** | Frame proportion anomaly | Daily frame share of total coverage |
| **Participation** | Journalist engagement anomaly | Daily unique journalist count for the frame |
| **Convergence** | Frame dominance anomaly | Frame share relative to all 8 frames |
| **Source** | Messenger concentration anomaly | 1 − normalized Shannon entropy of 9 messenger types |
| **Semantic** | Content homogenization anomaly | Mean pairwise cosine similarity of article embeddings |

The convergence signal is **orthogonalized** with respect to the temporal signal (projection subtracted, then re-clipped to 0) to remove their ~0.92 correlation — both derive from daily frame proportion, but convergence captures additional frame-dominance information not explained by temporal alone.

These are combined into a **weighted composite signal** (weights: 0.25, 0.20, 0.20, 0.15, 0.20) used for burst detection via dual methods: Z-score flagging (≥2σ for ≥3 consecutive days) and CUSUM change-point detection (Page, 1954).

### Step 3 — Scoring: four dimensions, 17 sub-indices

Every detected burst is scored on a continuous [0, 1] scale across four equally-weighted dimensions. There is no binary validation gate — the score determines the classification.

| Dimension | Weight | Sub-indices | Key metrics |
|-----------|--------|-------------|-------------|
| **Temporal** | 0.25 | 4 | Burst intensity, adoption velocity, duration, Mann-Whitney U |
| **Participation** | 0.25 | 6 | Actor diversity, cross-media ratio, new entrant rate, growth pattern, network structure, network cohesion |
| **Convergence** | 0.25 | 4 | Semantic similarity (syndication-penalized), convergence trend, cross-media alignment, novelty decay |
| **Source** | 0.25 | 3 | Source diversity decline, messenger concentration, media coordination |

5 of the 17 sub-indices are embedding-based (convergence dimension + media coordination), contributing ~31% of total score weight.

**Semantic peak**: The cascade `peak_date` is determined by an **embedding-weighted P50** rather than the composite Z-score argmax. For each article, a weight is computed as `frame_signal × cosine_similarity(article_embedding, centroid)`, where the centroid is the frame-signal-weighted mean of all article embeddings. The weighted median date identifies the semantic center of mass, producing a peak date that reflects content convergence rather than just volume.

**Score adjustments**:

- **Media confidence factor**: The base score is multiplied by `min(1, log₂(n_media) / log₂(10))`, discounting cascades with few media outlets (1 outlet ≈ 0%, 5 outlets ≈ 66%, 10+ outlets = full weight).
- **Syndication penalty**: Semantic similarity is reduced by `(1 - 0.5 × syndication_ratio)` to prevent wire-service content from inflating convergence scores.
- **Network structure**: Uses a scale-free clustering-to-density ratio `min(1, avg_clustering / density / 20)` instead of raw density, neutral (0.5) for small networks (< 5 nodes).

### Step 3 — Classification

| Classification | Score threshold |
|----------------|---------------|
| **Strong cascade** | ≥ 0.65 |
| **Moderate cascade** | ≥ 0.40 |
| **Weak cascade** | ≥ 0.25 |
| **Not cascade** | < 0.25 |

### Steps 3.5 & 3.6 — Event occurrence detection and attribution

The `EventOccurrenceDetector` (`analysis/event_occurrence.py`) detects real-world event occurrences across **all** articles in the analysis period, independently of cascades. Events are detected first, then attributed to cascades.

**Architecture**: Events exist independently of cascades. The pipeline detects event occurrences on the full corpus, then attributes them to cascades via temporal and article overlap.

| Phase | Operation | Method |
|-------|-----------|--------|
| **Phase 1** | Seed selection | Composite score: `0.6 × evt_*_mean + 0.4 × event_mean`, P50 threshold |
| **Phase 2** | Per-type clustering | HAC on title+sentence embeddings (30/70 weight), 3D distance (semantic + temporal + entity) |
| **Phase 3** | Multi-type merge | Pool all occurrences, deduplicate (Jaccard > 0.5), HAC with silhouette-optimized k |
| **Phase 4** | Soft assignment | 4D distance (temporal + semantic + entity + signal), 2 iterations, self-adjusting threshold |
| **Phase 5** | Confidence scoring | 5 components: centroid tightness, coherence residual, media diversity, recruitment success, size adequacy |
| **Attribution** | Cascade linkage | Temporal overlap + shared articles → `CascadeAttribution` objects |

**Key data structures**:

- `EventOccurrence`: A dated event with soft article membership (belonging scores), confidence, and seed provenance
- `EventCluster`: A meta-event grouping multiple occurrences, with type structure analysis (dominant, constitutive, satellite) and 5-component strength scoring
- `CascadeAttribution`: Links occurrences to cascades via shared articles and temporal overlap

### Step 4 — Paradigm shift analysis

The `ParadigmShiftAnalyzer` detects transitions in the dominant frame composition and attributes them to specific cascades and events. It integrates the [CCF-paradigm](https://github.com/antoinelemor/CCF-paradigm) library's 4-method consensus (information theory, network analysis, causality, proportional) for computing frame dominance.

**Pipeline**: Weekly frame proportions → 12-week sliding window (daily step, parallelized across CPU cores) → paradigm state computation → shift detection → three-role cascade attribution → episode grouping

**Three-role cascade attribution**: Each cascade attributed to a paradigm shift is assigned a discursive role based on measured impact:

| Role | Condition | Theoretical basis |
|------|-----------|-------------------|
| **Amplification** | Cascade promotes its own frame (positive dominance lift) | Information cascade theory |
| **Déstabilisation** | Cascade disrupts paradigm structure without its own frame benefiting | Focusing events (Birkland, 1998) |
| **Dormante** | Active cascade with no measurable structural consequence | Dormant issues (Hilgartner & Bosk, 1988) |

Impact is measured using linear temporal decay weights ($w(t) = \max(0, 1 - t/42)$, zero after 6 weeks) applied to dominance lift and paradigm vector cosine distance. Only driver cascades (amplification + déstabilisation) contribute events to shift attribution.

Amplification cascades are further classified by **direction alignment** (`onset → end + 7 days` causal window):

| Score | Mechanism | Condition |
|-------|-----------|-----------|
| 1.0 | **Promotion** | Frame NOT dominant before, ENTERS dominant set during cascade |
| 0.7 | **Consolidation** | Frame ALREADY dominant, STAYS dominant during cascade |
| 0.3 | **Insufficient** | Frame never reaches dominant set despite positive lift |

**Two-level dynamics qualification**:

| Level | Metrics | Purpose |
|-------|---------|---------|
| **Shift-level** | `regime_duration_days` (continuous), `structural_change` (±n frames), `reversible` (local: does the next shift restore the previous state?) | Characterize each individual transition |
| **Episode-level** | `reversible` (global: same paradigm before/after?), `net_structural_change`, `max_complexity` (peak dominant frame count), `regime_after_duration_days` (stability after) | Characterize the net effect of temporally clustered shifts |

An **episode** is a cluster of shifts separated by less than 3 weeks. The causal chain traced by the framework is: **Events → Cascades → Paradigm Shifts → Durable or Ephemeral Change**.

### Step 5 — Stability selection impact analysis

The `StabSelImpactAnalyzer` (`analysis/stabsel_impact.py`) measures causal relationships between event clusters and media cascades using stability selection (Meinshausen & Bühlmann, 2010) and double-weighted treatment variables. This replaces the legacy `UnifiedImpactAnalyzer` (3-phase diff-in-diff approach, still importable for backward compatibility).

**Treatment variable** (per cluster $j$, per cascade):

$$D_j(t, l) = \sum_{a \in \text{articles}(t-l)} \text{belonging}(a,j) \times \text{frame\_signal}(a) \times \cos(\text{emb}(a), \text{centroid}_{\text{cascade}})$$

The cascade centroid is built from the cascade's own central articles (top-quartile frame signal within ±14 days of cascade window), weighted by frame signal.

**Pipeline**:

| Step | Method | Parameters |
|------|--------|------------|
| **Signal construction** | 5 two-sided Z-score signals → weighted composite | Same 5 signals as detection (temporal, participation, convergence, source, semantic), orthogonalized |
| **Variable selection** | Stability selection with ElasticNet | B=100 sub-samples, 50% sub-sampling, π ≥ 0.60 threshold |
| **Post-selection inference** | OLS on stable set + residual bootstrap | 500 bootstrap iterations, net β per cluster (sum across lags 0–3) |
| **Role classification** | Sign of net β + p-value | α = 0.10 significance threshold |

**Role classification**:

| Role | Condition | Interpretation |
|------|-----------|----------------|
| **driver** | net_β > 0, p < 0.10 | Event cluster amplifies cascade activity |
| **suppressor** | net_β < 0, p < 0.10 | Event cluster dampens cascade activity |
| **neutral** | p ≥ 0.10 | Stable selection but not statistically significant |

**Output**: `StabSelImpactResults` with a `cluster_cascade` DataFrame (one row per significant cluster-cascade pair), per-frame summary, and detailed `cascade_results` dict for diagnostic plots.

## Project structure

```
cascade_detector/
├── __init__.py                          # Public API
├── pipeline.py                          # CascadeDetectionPipeline orchestrator (7 steps)
├── core/
│   ├── config.py                        # DetectorConfig (all parameters)
│   ├── constants.py                     # Frames, messengers, events, thresholds, event cluster constants
│   ├── models.py                        # BurstResult, CascadeResult, EventOccurrence, EventCluster,
│   │                                    # CascadeAttribution, DetectionResults
│   └── exceptions.py                    # Custom exception hierarchy
├── data/
│   ├── connector.py                     # DatabaseConnector (PostgreSQL + SQLAlchemy)
│   └── processor.py                     # DataProcessor (cleaning, aggregation)
├── indexing/
│   ├── index_manager.py                 # IndexManager (orchestrates 6 indexers)
│   ├── temporal_indexer.py              # Daily/weekly time series per frame
│   ├── frame_indexer.py                 # Frame co-occurrence matrix, sequences
│   ├── emotion_indexer.py               # Sentiment tracking (3 tones)
│   ├── entity_indexer.py                # NER entity profiles with authority scores
│   ├── source_indexer.py                # Journalist/media messenger profiles
│   └── geographic_indexer.py            # Location focus, cascade spread patterns
├── detection/
│   ├── unified_detector.py              # UnifiedCascadeDetector (detection + scoring)
│   ├── signal_builder.py                # DailySignalBuilder (5 signals + composite)
│   ├── network_builder.py               # Co-coverage graph construction
│   └── networkit_worker.py              # NetworKit subprocess (SIGSEGV prevention)
├── analysis/
│   ├── __init__.py                      # Analysis module
│   ├── event_occurrence.py              # EventOccurrenceDetector (database-first event detection)
│   ├── stabsel_impact.py               # StabSelImpactAnalyzer (stability selection impact)
│   ├── unified_impact.py               # UnifiedImpactAnalyzer (legacy, backward compat)
│   ├── impact_analysis.py               # EventImpactAnalyzer (legacy, backward compat)
│   └── paradigm_shift.py               # ParadigmShiftAnalyzer (shift + episode detection)
├── embeddings/
│   ├── compute.py                       # Embedding computation (BAAI/bge-m3, titles + sentences)
│   ├── embedding_store.py               # EmbeddingStore (eager memmap access)
│   └── semantic_convergence.py          # SemanticConvergenceCalculator (4 metrics)
├── metrics/
│   ├── exhaustive_metrics_calculator.py # 73+ exact network metrics
│   ├── networkit_worker.py              # Batch NetworKit subprocess
│   └── geographical_data/
│       └── media_by_province.csv        # 21 outlets mapped to provinces
└── utils/
    ├── entity_resolver_fast.py          # Entity name deduplication (blocking + similarity)
    ├── media_geography.py               # Canadian media geographic mapping
    ├── author_resolver.py               # Journalist name resolution
    └── location_resolver_fast.py        # Location entity deduplication

scripts/
├── run/                                 # Pipeline execution
│   ├── run_production.py                # Full production pipeline (1978–2024)
│   ├── run_2018.py                      # Single-year pipeline runner (caches to results/cache/)
│   ├── run_2018_paradigm.py             # 2018 cascade + paradigm shift + figures
│   ├── run_2018_event_occ.py            # 2018 event occurrence detection
│   ├── augment_embeddings_titles.py     # Add title embeddings to existing store
│   └── calibrate_thresholds.py          # Threshold calibration utility
├── figures/                             # Visualization / publication figures
│   ├── fig_event_impact_composite.py    # Composite event impact figure (4 panels)
│   ├── fig_paradigm_analysis_2018.py    # Paradigm shift analysis figures
│   ├── fig_dominance_proportions_events.py  # 4-panel overview (2018)
│   ├── fig_top_clusters_2018.py         # Top event clusters visualization
│   ├── gen_validation_report_2018.py    # Per-cascade validation report
│   ├── plot_frame_proportions.py        # Daily frame proportions with cascade overlay
│   ├── plot_paradigm_overview.py        # Paradigm dynamics overview
│   └── visualize_impact_2018.py         # 6 impact analysis figures
└── analysis/                            # Post-production analysis
    ├── fig_paradigm_overview_full.py    # 6-panel 1978–2024 overview
    ├── fig_top10_cascades.py            # Top 10 strongest cascades
    ├── gen_validation_figures.py        # Per-cascade validation figures for LaTeX
    ├── recalculate_scores.py            # Post-production v2 score correction
    ├── eval_cluster_syndication.py      # Syndication audit (stratified sample)
    ├── qualitative_cluster_analysis.py  # Qualitative event cluster analysis
    └── top10_clusters_articles.py       # Detailed article breakdown

tests/
├── conftest.py                          # MockEmbeddingStore + ClusterableMockEmbeddingStore
├── test_signal_builder.py               # 14 unit tests (signal computation + orthogonalization)
├── test_unified_detector.py             # 23 unit tests (detection + scoring + semantic peak)
├── test_event_impact.py                 # 9 unit tests (legacy impact analysis)
├── test_unified_impact.py               # 68 unit tests (legacy 3-phase causal impact, backward compat)
├── test_event_occurrence.py             # 128 unit tests (event detection, clustering, attribution)
└── test_paradigm_shift.py              # 48 unit tests (paradigm shift detection + three-role attribution)
```

## Installation

```bash
git clone https://github.com/antoinelemor/CCF-media-cascade-detection.git
cd CCF-media-cascade-detection

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Dependencies

```
numpy, pandas, scipy, tqdm, pyarrow        # Core
psycopg2-binary, sqlalchemy                # Database
networkx, networkit, python-louvain        # Network analysis
sentence-transformers, torch               # Embeddings (optional, for computing embeddings)
matplotlib, adjustText                     # Visualization (optional, for scripts/)
```

### Prerequisites

- PostgreSQL database `CCF_Database_texts` with table `"CCF_processed_data"` (9.2M rows)
- Pre-computed sentence and title embeddings in `data/embeddings/` (auto-computed on first run; titles added via `scripts/run/augment_embeddings_titles.py`)

## Usage

### Quick start

```python
from cascade_detector import CascadeDetectionPipeline, DetectorConfig

config = DetectorConfig()
pipeline = CascadeDetectionPipeline(config)

results = pipeline.run(
    start_date="2018-01-01",
    end_date="2018-12-31"
)

print(results.summary())

# Event occurrences (detected database-first on all articles)
for cluster in sorted(results.event_clusters, key=lambda c: c.strength, reverse=True)[:5]:
    print(f"Cluster {cluster.cluster_id}: {cluster.dominant_type}, "
          f"strength={cluster.strength:.3f}, {cluster.n_occurrences} occurrences")

# StabSel impact analysis (included automatically)
impact = results.event_impact  # StabSelImpactResults
print(impact.cluster_cascade.head())   # cluster → cascade (driver/suppressor/neutral)
print(impact.summary)                  # per-frame {n_cascades, n_drivers, n_suppressors, median_r2}
# Detailed results per frame for diagnostic plots:
for frame, crs in impact.cascade_results.items():
    for cr in crs:
        print(f"  {cr.cascade_id}: R²={cr.r2:.3f}, {cr.n_drivers}D {cr.n_suppressors}S")

# Paradigm shift analysis (included automatically)
ps = results.paradigm_shifts
print(ps.summary())
for ep in ps.episodes:
    print(f"Episode {ep.episode_id}: {ep.start_date:%Y-%m-%d} to {ep.end_date:%Y-%m-%d}")
    print(f"  {ep.n_shifts} shifts, {'reversible' if ep.reversible else 'irreversible'}")
    print(f"  [{','.join(ep.dominant_frames_before)}] → [{','.join(ep.dominant_frames_after)}]")
```

### Production run (all years)

```bash
# Run detection on all years (1978-2024)
python scripts/run/run_production.py

# Or a single year
python scripts/run/run_production.py --year 2018

# Resume interrupted run
python scripts/run/run_production.py --resume
```

The production script exports year-by-year results in Parquet and JSON:

```
results/production/
├── run_manifest.json
├── cross_year_cascades.parquet
├── cross_year_summary.json
├── cross_year_paradigm_timeline.parquet
├── cross_year_paradigm_shifts.json
└── {year}/
    ├── cascades.json / cascades.parquet
    ├── bursts.parquet
    ├── year_metadata.json
    ├── time_series/         # Daily composite, signals, articles, journalists
    ├── networks/            # Edge lists, network metrics
    ├── signals/             # Per-frame daily Z-scores
    ├── indices/             # Temporal, frame, emotion, source, geographic
    ├── convergence/         # Semantic convergence, syndication stats
    ├── impact_analysis/     # StabSel impact analysis (cluster → cascade)
    ├── event_occurrences/   # Event detection results
    │   ├── occurrences.json     # All event occurrences with belonging scores
    │   ├── clusters.json        # EventClusters (multi-type meta-events)
    │   └── attributions.json    # Cascade ↔ occurrence linkage
    └── paradigm_shifts/     # Shifts, episodes, paradigm timeline
        ├── shifts.json      # All detected paradigm shifts with attribution
        ├── episodes.json    # Shift episodes with two-level dynamics
        └── paradigm_timeline.parquet  # 281 paradigm states (daily resolution)
```

### Inspecting results

```python
# Load results
results = pipeline.run("2018-01-01", "2018-12-31")

# DataFrame with one row per cascade
df = results.to_dataframe()

# Export to JSON
results.to_json("cascades_2018.json")

# Iterate cascades
for c in sorted(results.cascades, key=lambda x: x.total_score, reverse=True):
    print(f"[{c.classification}] {c.frame} "
          f"({c.onset_date:%Y-%m-%d} to {c.end_date:%Y-%m-%d}) "
          f"score={c.total_score:.3f}")
    print(f"  Articles: {c.n_articles}, Journalists: {c.n_journalists}, "
          f"Media: {c.n_media}")
    print(f"  Temporal={c.score_temporal:.3f}, "
          f"Participation={c.score_participation:.3f}, "
          f"Convergence={c.score_convergence:.3f}, "
          f"Source={c.score_source:.3f}")
```

## The eight climate frames

| Code | Full name | Database column |
|------|-----------|-----------------|
| Cult | Cultural | `cultural_frame` |
| Eco | Economic | `economic_frame` |
| Envt | Environmental | `environmental_frame` |
| Pbh | Public Health | `health_frame` |
| Just | Justice | `justice_frame` |
| Pol | Political | `political_frame` |
| Sci | Scientific | `scientific_frame` |
| Secu | Security | `security_frame` |

## Database requirements

PostgreSQL database with the following schema:

```sql
Table: "CCF_processed_data"
├── doc_id              BIGINT       -- Unique article identifier
├── sentence_id         BIGINT       -- Sentence sequence within article
├── sentences           TEXT         -- Sentence text content
├── date                DATE         -- Publication date (native PostgreSQL DATE)
├── author              VARCHAR      -- Journalist byline
├── media               VARCHAR      -- Media outlet name
├── language            VARCHAR      -- Article language (en/fr)
├── economic_frame      NUMERIC      -- Frame detection (binary: 0/1)
├── ...                              -- 8 frame columns total
├── msg_health          NUMERIC      -- Messenger type (binary: 0/1)
├── ...                              -- 9 messenger columns total
├── tone_positive       NUMERIC      -- Emotional tone (binary: 0/1)
├── tone_neutral        NUMERIC
├── tone_negative       NUMERIC
└── ner_entities        JSON         -- {"PER": [...], "ORG": [...], "LOC": [...]}
```

Environment variables for database configuration:

| Variable | Default |
|----------|---------|
| `DB_HOST` | `localhost` |
| `DB_PORT` | `5432` |
| `DB_USER` | `antoine` |
| `DB_PASSWORD` | (empty) |
| `DB_NAME_TEXTS` | `CCF_Database_texts` |
| `EMBEDDING_DIR` | `data/embeddings` |

## Scientific documentation

For detailed methodology, formulas, and interpretation guidelines, see [`docs/scientific_documentation.md`](docs/scientific_documentation.md).

## License

Part of the [CCF (Canadian Climate Framing)](https://github.com/antoinelemor/CCF-canadian-climate-framing) research project.
