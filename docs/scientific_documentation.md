# CCF Media Cascade Detection Framework — Scientific Documentation

**Version 0.6.0 — February 2026**
**Author: Antoine Lemor**

---

## Table of Contents

1. [Introduction and Conceptual Framework](#1-introduction-and-conceptual-framework)
2. [Data Infrastructure](#2-data-infrastructure)
3. [Indexing Layer: Six Specialized Indices](#3-indexing-layer-six-specialized-indices)
4. [Embedding Infrastructure](#4-embedding-infrastructure)
5. [Detection Layer: Five Daily Signals and Burst Identification](#5-detection-layer-five-daily-signals-and-burst-identification)
6. [Scoring Pipeline: Four Dimensions and Seventeen Sub-Indices](#6-scoring-pipeline-four-dimensions-and-seventeen-sub-indices)
7. [Network Construction and Metrics](#7-network-construction-and-metrics)
8. [Classification and Output](#8-classification-and-output)
9. [Event Occurrence Detection](#9-event-occurrence-detection)
10. [Unified Impact Analysis](#10-unified-impact-analysis)
11. [Paradigm Shift Detection and Dynamics Qualification](#11-paradigm-shift-detection-and-dynamics-qualification)
12. [Production Pipeline](#12-production-pipeline)
13. [Configuration Reference](#13-configuration-reference)
14. [Testing Strategy](#14-testing-strategy)

**Appendices**
- [A. Summary Table of All Sub-Indices](#appendix-a-summary-table-of-all-sub-indices)
- [B. Embedding Coverage Logging](#appendix-b-embedding-coverage-logging)
- [C. Network Construction Detail](#appendix-c-network-construction-detail)
- [D. Geographic Analysis Infrastructure](#appendix-d-geographic-analysis-infrastructure)
- [E. Entity Resolution Algorithms](#appendix-e-entity-resolution-algorithms)
- [F. StabSel Impact Analysis Output Schema](#appendix-f-stabsel-impact-analysis-output-schema)
- [G. Paradigm Shift Output Schema](#appendix-g-paradigm-shift-output-schema)
- [H. Event Occurrence Constants Reference](#appendix-h-event-occurrence-constants-reference)

---

## 1. Introduction and Conceptual Framework

### 1.1 Purpose

The CCF Media Cascade Detection Framework identifies and quantifies **information cascades** — coordinated surges of media coverage across multiple outlets and journalists around specific frames (economic, political, environmental, etc.) in Canadian media. It operates on 9.2 million sentence-level annotations from 266,271 articles published in 20 major Canadian newspapers between 1978 and 2024.

The framework answers a central research question: *when does a news topic shift from routine reporting to a self-reinforcing cascade where content converges, new journalists enter, and coverage amplifies across outlets?*

### 1.2 Key Design Principles

1. **Multi-signal detection**: Cascades are not detected from article volume alone. Five daily anomaly signals (temporal, participation, convergence, source, semantic) are combined into a composite indicator. A burst must show anomalies across multiple dimensions simultaneously.

2. **Continuous scoring, no binary gate**: Every detected burst receives a continuous score in [0, 1] across four dimensions. There is no validation step that discards bursts — the score itself determines the classification (strong ≥ 0.65, moderate ≥ 0.40, weak ≥ 0.25).

3. **Mandatory embeddings**: Sentence-level embeddings (BAAI/bge-m3, 1024 dimensions) are not optional. They drive the semantic anomaly signal during detection and four convergence sub-indices during scoring. Five of the seventeen sub-indices rely on embeddings, contributing approximately 32% of the final cascade score weight.

4. **Reproducibility**: All random operations use fixed seeds (`RandomState(42)`). The CUSUM algorithm is deterministic. Baseline windows are trailing (no lookahead bias). Subsampling operations are seeded.

5. **Bilingual support**: The BAAI/bge-m3 model supports cross-lingual alignment between English and French, enabling valid semantic comparison across Canada's bilingual media landscape without requiring translation.

### 1.3 What Is a Media Cascade?

A media cascade, as operationalized in this framework, is a period during which:

- **Temporal anomaly**: Coverage of a specific frame rises significantly above its 90-day trailing baseline.
- **Participation broadening**: New journalists and media outlets begin covering the topic, expanding beyond the usual pool of reporters.
- **Content convergence**: Articles become semantically more similar to each other over time, as measured via embedding cosine similarity.
- **Source coordination**: Messenger profiles (the types of experts and authorities cited) concentrate, and journalist-level content centroids align.

The framework distinguishes cascades from routine coverage spikes by requiring anomalies across multiple dimensions, not just volume. A burst in article count alone (e.g., a single breaking-news event) will produce a low composite score unless participation broadens, content converges, and sources align.

### 1.4 Theoretical Grounding

The cascade detection methodology draws on several theoretical traditions:

- **Information cascade theory** (Bikhchandani, Hirshleifer & Welch, 1992): Agents sequentially adopt behaviors based on predecessors' actions rather than private information. In the media context, journalists observe peer coverage patterns and make publication decisions accordingly.

- **Agenda-setting theory** (McCombs & Shaw, 1972): Media cascades represent moments when the agenda-setting function amplifies beyond normal levels — multiple outlets simultaneously elevate the same frame.

- **Framing theory** (Entman, 1993): The framework's 8 climate frames (economic, political, environmental, etc.) operationalize Entman's definition of framing as selecting and making salient particular aspects of a perceived reality.

- **Network diffusion models** (Watts & Dodds, 2007): The co-coverage network construction and cohesion metrics capture how information diffuses through journalist-media networks, consistent with the "big-seed" cascade model where influence spreads through connected communities.

---

## 2. Data Infrastructure

### 2.1 Database Schema

**Database**: PostgreSQL `CCF_Database_texts`
**Table**: `"CCF_processed_data"` — 9.2 million sentence-level rows, 77 columns

The CCF database contains sentence-level annotations produced by the upstream [CCF annotation pipeline](https://github.com/antoinelemor/CCF-canadian-climate-framing). Each sentence in every article is independently annotated with frame detections, messenger types, event classifications, emotional tones, and named entity recognition results.

**Key column groups:**

| Group | Columns | Type | Description |
|-------|---------|------|-------------|
| **Identifiers** | `doc_id`, `sentence_id` | BIGINT | Article and sentence identifiers |
| **Text** | `sentences` | TEXT | Raw sentence text |
| **Temporal** | `date` | DATE | Publication date (native PostgreSQL DATE) |
| **Metadata** | `author`, `media`, `language`, `news_type`, `title`, `words_count`, `page_number` | VARCHAR/INT | Article metadata |
| **Frames (8)** | `cultural_frame`, `economic_frame`, `environmental_frame`, `health_frame`, `justice_frame`, `political_frame`, `scientific_frame`, `security_frame` | NUMERIC | Binary (0/1) frame detection per sentence |
| **Messengers (9)** | `msg_health`, `msg_economic`, `msg_security`, `msg_legal`, `msg_cultural`, `msg_scientist`, `msg_social`, `msg_activist`, `msg_official` | NUMERIC | Binary (0/1) messenger type annotation per sentence |
| **Events (8)** | `evt_weather`, `evt_meeting`, `evt_publication`, `evt_election`, `evt_policy`, `evt_judiciary`, `evt_cultural`, `evt_protest` | NUMERIC | Binary (0/1) event classification |
| **Solutions (2)** | `sol_mitigation`, `sol_adaptation` | NUMERIC | Binary (0/1) solution type |
| **Tones (3)** | `tone_positive`, `tone_neutral`, `tone_negative` | NUMERIC | Binary (0/1) emotional tone |
| **Entities** | `ner_entities` | JSON | `{"PER": [...], "ORG": [...], "LOC": [...]}` |

### 2.2 The Eight Climate Frames

The framework analyzes 8 climate change frames, each representing a distinct perspective through which media coverage approaches the climate issue:

| Code | Full Name | Database Column | Description |
|------|-----------|-----------------|-------------|
| **Cult** | Cultural | `cultural_frame` | Climate as cultural identity, values, lifestyle issue |
| **Eco** | Economic | `economic_frame` | Climate as economic cost/benefit, market impact, industry concern |
| **Envt** | Environmental | `environmental_frame` | Climate as ecological, biodiversity, ecosystem issue |
| **Pbh** | Public Health | `health_frame` | Climate as public health threat, medical impact |
| **Just** | Justice | `justice_frame` | Climate as equity, fairness, rights issue |
| **Pol** | Political | `political_frame` | Climate as policy, governance, partisan issue |
| **Sci** | Scientific | `scientific_frame` | Climate as research, data, scientific consensus issue |
| **Secu** | Security | `security_frame` | Climate as national security, geopolitical risk |

Each frame is independently annotated at the sentence level. An article may be tagged with multiple frames across its sentences. The cascade detector analyzes each frame independently, detecting frame-specific surges.

### 2.3 The Nine Messenger Types

Messengers represent the types of authorities and experts cited in climate coverage:

| Column | Expertise Domain | Description |
|--------|-----------------|-------------|
| `msg_health` | Health expertise | Medical professionals, public health officials |
| `msg_economic` | Economic expertise | Economists, financial analysts, business leaders |
| `msg_security` | Security expertise | Defense/security officials, risk assessors |
| `msg_legal` | Legal expertise | Lawyers, judges, legal scholars |
| `msg_cultural` | Cultural expertise | Cultural commentators, arts figures |
| `msg_scientist` | Hard science | Climate scientists, researchers, academics |
| `msg_social` | Social science | Sociologists, psychologists, policy researchers |
| `msg_activist` | Activist | Environmental activists, advocacy organizations |
| `msg_official` | Public official | Politicians, government representatives |

### 2.4 DatabaseConnector

`DatabaseConnector` (`data/connector.py`) wraps SQLAlchemy with a connection pool:

- **Pool configuration**: 10 primary connections + 20 overflow, with pre-ping for stale connection detection
- **Connection string**: `postgresql://{user}:{password}@{host}:{port}/{db_name}`
- **Environment variable overrides**: `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`

**Primary SQL query** (`get_frame_data`):

```sql
SELECT * FROM "CCF_processed_data"
WHERE date >= %(start_date)s::date
  AND date <= %(end_date)s::date
ORDER BY date, doc_id, sentence_id
```

The `::date` cast ensures proper type handling with PostgreSQL's native DATE type. An optional `exclude_2025` flag (default: `True`) caps the end date to `2024-12-31`.

### 2.5 DataProcessor

`DataProcessor` (`data/processor.py`) transforms raw SQL output into analysis-ready DataFrames through a deterministic pipeline:

#### Stage 1: Date conversion

Detects native `datetime64` from PostgreSQL or parses string formats. Adds derived columns:
- `year` (int), `month` (int), `week` (ISO week number), `day_of_week` (0=Monday, 6=Sunday)

#### Stage 2: Frame binarization

Converts frame columns to strict binary (0/1) via `pd.to_numeric(errors='coerce').fillna(0)`, then `(col > 0).astype(int)`. Handles both current column names (`economic_frame`) and legacy names (`Eco_Detection`). Adds:
- `n_frames`: Count of active frames per sentence (0–8)
- `dominant_frame`: Frame with highest value, or `None` if no frames active

#### Stage 3: Messenger binarization

Same pattern as frames for 9 messenger columns. Tries current names (`msg_health`) first, falls back to legacy (`Messenger_1_SUB`). Adds `n_messengers` per sentence.

#### Stage 4: Event and solution cleaning

Binary columns for 8 event types and 2 solution types. Adds `n_events`, `dominant_event`, `n_solutions`.

#### Stage 5: Author normalization

Strips wire-service suffixes from bylines: `, Reuters`, `, AP`, `, Canadian Press`, `, Bloomberg`, `, Associated Press`, `, Staff`.

#### Stage 6: Article aggregation (`aggregate_by_article`)

Groups sentence-level rows by `doc_id`. For each frame and messenger column, computes both `_sum` (total sentences with that annotation) and `_mean` (proportion of sentences). Keeps first-occurrence values for metadata (date, media, author).

**Output**: An article-level DataFrame where each row is one article. Frame usage is captured in two forms:

- `economic_frame_sum` = number of sentences tagged with economic frame
- `economic_frame_mean` = proportion of sentences tagged with economic frame

**Data flow summary**:

```
Raw database (9.2M sentences × 77 columns)
    ↓ process_frame_data()
Processed DataFrame (9.2M sentences × ~100 columns)
    ↓ aggregate_by_article()
Article DataFrame (~266K articles × ~51 columns)
```

---

## 3. Indexing Layer: Six Specialized Indices

The `IndexManager` (`indexing/index_manager.py`) builds indices sequentially from the sentence-level DataFrame. All indexers follow a common pattern: no configuration in `__init__()`, processing via `build_index(df)`. Each indexer resets its internal state on each call, ensuring no leakage across years.

### 3.1 Temporal Index

**Source**: `indexing/temporal_indexer.py`

For each of the 8 frames, the `TemporalIndexer` builds:

| Key | Type | Description |
|-----|------|-------------|
| `daily_series` | `pd.Series` | Daily raw counts (sentences tagged with this frame per day) |
| `daily_proportions` | `pd.Series` | Daily proportions: sentences with frame / total sentences per day |
| `weekly_series` | `pd.Series` | Weekly aggregation of counts |
| `weekly_proportions` | `pd.Series` | Weekly aggregation of proportions |
| `articles_by_date` | `Dict[date, List[doc_id]]` | Fast article lookup by day |
| `statistics` | `Dict` | Mean, std, max, n_active_days, activity_rate |

**Why this matters**: The `daily_proportions` series is the **primary input** to Signal 1 (temporal Z-score). Using proportions rather than raw counts normalizes for variation in total daily coverage volume. The `articles_by_date` mapping enables the semantic signal to efficiently find doc_ids by day.

**Activity rate**: `n_active_days / total_days` — fraction of days with at least one article tagged with the frame.

### 3.2 Frame Index

**Source**: `indexing/frame_indexer.py`

| Key | Type | Description |
|-----|------|-------------|
| `article_frames` | `Dict[doc_id, Dict]` | Per-article: active frames, dominant frame, frame proportions |
| `cooccurrence_matrix` | `np.ndarray (8×8)` | Symmetric matrix of frame co-occurrence within articles |
| `temporal_cooccurrence` | `Dict[week, Dict]` | Week-level co-occurrence counts for frame pairs |
| `frame_sequences` | `List[Dict]` | Frequent frame transition sequences (length 2–4, min support 3) |
| `frame_statistics` | `Dict[frame, Dict]` | Per-frame: n_articles, prevalence, mean/std/max proportion |

**Normalization of co-occurrence matrix**: Each cell is normalized by the geometric mean of the diagonal:

$$
M_{ij}^{\text{norm}} = \frac{M_{ij}}{\sqrt{M_{ii} \cdot M_{jj}}}
$$

This produces a correlation-like measure in [0, 1] that corrects for frame frequency differences.

### 3.3 Emotion Index

**Source**: `indexing/emotion_indexer.py`

Tracks sentiment dynamics across three tones (`tone_positive`, `tone_neutral`, `tone_negative`):

| Key | Type | Description |
|-----|------|-------------|
| `article_emotions` | `Dict[doc_id, Dict]` | Per-article: positive/negative/neutral rates, sentiment_score, intensity |
| `temporal_emotion` | `Dict[week, Dict]` | Weekly sentiment aggregation |
| `media_emotion` | `Dict[media, Dict]` | Per-outlet sentiment profile |
| `author_emotion` | `Dict[author, Dict]` | Per-journalist sentiment profile |
| `emotion_statistics` | `Dict` | Global summary statistics |

**Sentiment score**: $\text{sentiment} = \text{positive\_rate} - \text{negative\_rate} \in [-1, +1]$

**Emotional intensity**: $|\text{sentiment\_score}| \in [0, 1]$ — magnitude regardless of direction.

### 3.4 Entity Index

**Source**: `indexing/entity_indexer.py`

Processes NER entities (PER, ORG, LOC) from the `ner_entities` JSON column. The indexer:

1. **Extracts** entities from JSON strings in parallel (16 workers, chunk-based)
2. **Resolves** name variations using `FastEntityResolver` (blocking + cosine similarity, threshold 0.75)
3. **Scores** authority for each entity

**Authority score formula**:

$$
\text{authority}(e) = \text{count}(e) \times \ln(1 + \text{diversity}(e))
$$

where:
- $\text{count}(e)$ = total citations of entity $e$ across all articles
- $\text{diversity}(e) = |\text{journalists}(e)| + |\text{media}(e)|$ = number of distinct journalists and outlets citing the entity

**Interpretation**: An entity cited 100 times by a single journalist has lower authority than one cited 50 times across 10 journalists and 5 outlets. The logarithmic dampening prevents extreme diversity values from dominating.

### 3.5 Source Index

**Source**: `indexing/source_indexer.py`

Builds journalist and media outlet profiles centered on messenger type usage:

| Key | Type | Description |
|-----|------|-------------|
| `article_profiles` | `Dict[doc_id, Dict]` | Per-article: messenger proportions, dominant messenger, Shannon entropy, date, media, author |
| `journalist_profiles` | `Dict[author, Dict]` | Average messenger proportions across all articles by journalist |
| `media_profiles` | `Dict[media, Dict]` | Same structure at the outlet level |
| `temporal_evolution` | `Dict[week, Dict]` | Weekly messenger type proportions |

**Shannon entropy of messenger distribution**:

$$
H = -\sum_{k=1}^{K} p_k \ln p_k, \quad p_k = \frac{c_k}{\sum_j c_j}
$$

where $c_k$ is the count of messenger type $k$ (across all sentences with messenger annotations in the article), and $K$ is the number of available messenger columns (up to 9).

**Normalized entropy**: $H_{\text{norm}} = H / \ln K \in [0, 1]$

**Why this matters**: The `article_profiles` serve as the source of `article_dates` and `article_media` mappings used by the convergence scorer. The messenger proportions feed sub-indices 4.1 (source diversity decline), 4.2 (messenger concentration), and indirectly Signal 4 (source anomaly).

### 3.6 Geographic Index

**Source**: `indexing/geographic_indexer.py`

Maps the spatial dimensions of media coverage using NER-extracted location entities:

| Key | Description |
|-----|-------------|
| `locations` | Per-location: occurrences, n_articles, n_media, n_journalists, media concentration (HHI) |
| `focus_metrics` | Geographic entropy, concentration (HHI), top location dominance, media/journalist focus alignment |
| `media_location_network` | Per-outlet: locations mentioned, focus score, entropy |
| `journalist_location_network` | Per-journalist: same structure |
| `location_cooccurrence` | NetworkX graph of co-mentioned locations |
| `temporal_focus` | Daily entropy values, sustained focus periods |
| `cascade_indicators` | Composite geographic cascade likelihood |
| `media_regional_spread` | Daily spread classification (regional/multi-regional/national) |
| `geographic_diffusion` | Cascade spread score, diffusion type |
| `frame_geographic_patterns` | Per-frame geographic tendency (national/regional/local) |
| `proximity_effects` | Geographic coherence between affected regions |
| `linguistic_barriers` | English/French barrier crossing analysis |

**Geographic focus score** per outlet/journalist:

$$
\text{focus} = 1 - \frac{H_{\text{locations}}}{\log_2(n_{\text{locations}})} \in [0, 1]
$$

where $H_{\text{locations}}$ is the Shannon entropy of the location mention distribution. A score of 1.0 means all mentions concentrate on a single location; 0 means perfectly uniform distribution.

**Cascade geographic spread classification**:
- **National**: ≥2 national media OR (≥1 national + ≥3 regions)
- **Multi-regional**: ≥2 distinct provinces represented
- **Regional**: Single province
- **National-only**: Only national outlets, no regional pickup

The geographic index uses `MediaGeography` (`utils/media_geography.py`) which maps 21 Canadian media outlets to 10 provinces with a province adjacency network for proximity analysis.

---

## 4. Embedding Infrastructure

### 4.1 Model: BAAI/bge-m3

Embeddings are computed using **BAAI/bge-m3** ([HuggingFace](https://huggingface.co/BAAI/bge-m3)), an XLM-RoBERTa-based multilingual embedding model:

| Property | Value |
|----------|-------|
| **Architecture** | XLM-RoBERTa backbone, fine-tuned with multi-stage contrastive learning |
| **Dimension** | 1024 |
| **Languages** | 100+ languages including English and French |
| **Normalization** | L2 normalization at encoding time (`normalize_embeddings=True`) |
| **Cross-lingual** | English and French sentences expressing the same meaning produce similar embeddings |

**Why BAAI/bge-m3**: The bilingual nature of Canadian media requires a model that can compare English-language articles (e.g., Globe and Mail) with French-language articles (e.g., Le Devoir) without translation. BGE-M3's cross-lingual alignment enables direct cosine similarity comparisons across languages, critical for detecting cascades that cross the English-French barrier.

**Why sentence-level**: Embeddings are computed per sentence rather than per article because the CCF annotation pipeline operates at the sentence level. Article-level embeddings are derived via mean pooling of their constituent sentence embeddings, preserving the fine-grained semantic information.

### 4.2 Embedding Computation

**Module**: `cascade_detector.embeddings.compute` (CLI: `scripts/run/augment_embeddings_titles.py`)

**SQL query**:
```sql
SELECT doc_id, sentence_id, sentences
FROM "CCF_processed_data"
WHERE sentences IS NOT NULL AND sentences != ''
ORDER BY doc_id, sentence_id
```

**Computation pipeline**:
1. Stream sentences from PostgreSQL via server-side cursor (batch size: 50,000)
2. Encode in device-adaptive batches (CUDA: 256, MPS: 32, CPU: 64)
3. Normalize embeddings (L2)
4. Write to memory-mapped NumPy file

**Output format**:

| File | Format | Content | Size (full corpus) |
|------|--------|---------|-------------------|
| `embeddings.npy` | NumPy memmap, float16 | `(n_sentences, 1024)` matrix | ~18.8 GB |
| `index.pkl` | Python pickle | `{(doc_id: int, sentence_id: int): row_index: int}` | ~500 MB |

**Device detection**: Auto-detected in order: CUDA > MPS (Apple Silicon) > CPU.

**Production integration**: `scripts/run/run_production.py` calls `ensure_embeddings()` before pipeline initialization, which checks that embeddings exist and cover ≥99% of database sentences. If coverage is insufficient, it triggers automatic recomputation.

### 4.3 EmbeddingStore

`EmbeddingStore` (`embeddings/embedding_store.py`) provides lazy-loaded, memory-mapped access to pre-computed embeddings:

**Lazy loading**: Neither the memmap array nor the index are loaded until first access. This allows the pipeline to initialize quickly and amortize the I/O cost across actual embedding requests.

**Float16 → float32 upcast**: All returned embeddings are converted from storage float16 to float32 via `np.array(..., dtype=np.float32)` for numerical stability in subsequent cosine similarity computations.

**`doc_id_to_rows` mapping**: A secondary index built lazily from `(doc_id, sentence_id)` tuples, grouping all sentence row indices by `doc_id`. This enables article-level embedding retrieval.

**Key methods**:

| Method | Description | Output |
|--------|-------------|--------|
| `get_article_embedding(doc_id)` | Mean-pool all sentence embeddings for this article | 1024-dim float32 vector, or `None` |
| `get_batch_article_embeddings(doc_ids)` | Batch retrieval with coverage tracking | `(n_found × 1024 array, found_ids list)` |
| `mean_pairwise_similarity(doc_ids)` | Mean pairwise cosine similarity | Float ∈ [0, 1] |
| `pairwise_cosine_similarity(embeddings)` | Full cosine similarity matrix | `(n × n)` float32 matrix |
| `deduplicate_embeddings(doc_ids, threshold=0.95)` | Remove near-duplicate articles | Deduplicated doc_id list |
| `cosine_similarity(emb1, emb2)` | Pairwise cosine similarity | Float ∈ [-1, 1] |

**Coverage logging**: When fewer than 50% of requested doc_ids are found (for batches ≥ 5), a WARNING is logged. This prevents silent degradation of embedding-based metrics.

### 4.4 Syndication Deduplication

**Problem**: Wire-service articles (Reuters, AP, Canadian Press) are republished verbatim across multiple outlets. These near-identical articles inflate cosine similarity scores, creating false convergence signals.

**Solution**: Before computing convergence metrics, `deduplicate_embeddings()` removes near-duplicates:

1. Compute full pairwise cosine similarity matrix
2. Greedy clustering: iterate through articles; for each non-removed article, mark all subsequent articles with similarity > 0.95 as duplicates
3. Keep one representative per cluster (first in document order)

**Threshold**: 0.95 (cosine similarity). Syndicated articles typically have similarity ~1.0; original articles covering the same topic typically score 0.60–0.85.

### 4.5 Title Embeddings

Article titles are stored alongside sentence embeddings using a reserved `sentence_id = 0` (constant `TITLE_SENTENCE_ID`). This allows the embedding store to provide both sentence-level and title-level representations from a single memmap.

**Title weight**: `TITLE_WEIGHT = 0.30`. When computing article-level embeddings for event occurrence clustering (Section 9), the title embedding is blended with the mean sentence embedding:

$$
\mathbf{e}_{\text{article}} = \alpha \cdot \mathbf{e}_{\text{title}} + (1 - \alpha) \cdot \bar{\mathbf{e}}_{\text{sentences}}, \quad \alpha = 0.30
$$

If the title embedding is not available (e.g., for articles ingested before title embedding augmentation), the system falls back to sentence-only embeddings.

**Augmentation script**: `scripts/run/augment_embeddings_titles.py` is an idempotent script that adds title embeddings to an existing memmap store, skipping entries already present.

### 4.6 Index Key Types and doc_id Matching

The PostgreSQL `doc_id` column is BIGINT. psycopg2 returns Python `int`. When article-level DataFrames call `.tolist()`, NumPy int64 converts to Python `int`. The pickle index also stores Python `int` keys. Hash-based lookup is consistent across all layers — no type mismatch.

---

## 5. Detection Layer: Five Daily Signals and Burst Identification

### 5.1 Rolling Z-Score (Shared Computation)

**Source**: `detection/signal_builder.py`

All five daily signals use the same rolling Z-score function. This standardization ensures that all signals are on the same scale and can be combined into a meaningful composite.

**Input**: A daily time series $x(t)$ (e.g., frame proportion, journalist count, similarity).

**Parameters**:
- Window $w = 90$ days (configurable via `config.baseline_window_days`)
- Minimum periods $= w/2 = 45$ (minimum observations for a valid window)
- Shift $= 1$ day (the window runs from $[t-90, t-1]$, never including day $t$ itself)

**Computation**:

$$
\bar{x}_{[t-90, t-1]} = \text{rolling\_mean}(x, w=90, \text{min\_periods}=45).\text{shift}(1)
$$

$$
\sigma_{[t-90, t-1]} = \text{rolling\_std}(x, w=90, \text{min\_periods}=45).\text{shift}(1)
$$

For the first ~45 days where the rolling window has insufficient observations, the global mean and std of the entire series are used as fallback:

$$
\bar{x}_{\text{fallback}} = \text{mean}(x), \quad \sigma_{\text{fallback}} = \text{std}(x)
$$

The rolling standard deviation is floored to prevent division by zero:

$$
\sigma_{\text{floor}} = \max(0.1 \cdot \sigma_{\text{global}}, 10^{-10})
$$

The Z-score is **one-sided** (clipped at zero — only positive anomalies are cascade signals):

$$
z(t) = \max\left(0, \frac{x(t) - \bar{x}_{[t-90, t-1]}}{\sigma_{[t-90, t-1]}}\right)
$$

**Design rationale**: The trailing window with shift(1) prevents lookahead bias — the baseline never includes the current day's data. This is critical for ensuring that cascade detection is not artificially inflated by including the anomalous day in its own baseline. The one-sided clipping reflects the theoretical framework: we are interested in positive anomalies (surges) only.

### 5.2 Signal 1: Temporal Anomaly ($z_{\text{temporal}}$)

**Raw input**: `daily_proportions` from `temporal_index[frame]`:

$$
p_{\text{frame}}(t) = \frac{\text{sentences tagged with frame on day } t}{\text{total sentences on day } t}
$$

**Computation**: `rolling_zscore(daily_proportions)`

**Interpretation**: Detects when a frame occupies an unusually large share of total coverage compared to its trailing 90-day baseline. Using proportions rather than raw counts normalizes for variation in total daily coverage volume.

**Example**: If the economic frame typically represents 15% of daily coverage (baseline mean = 0.15) but rises to 35% on a given day (after a carbon tax announcement), and the baseline standard deviation is 0.05, the Z-score would be $(0.35 - 0.15) / 0.05 = 4.0$ — a strong anomaly.

### 5.3 Signal 2: Participation Anomaly ($z_{\text{participation}}$)

**Raw input**: For each day $t$, count the number of unique journalists who published at least one article tagged with this frame.

**Data flow**:
1. Filter the article-level DataFrame to articles where the frame column (e.g., `economic_frame_sum` or `economic_frame_mean`) is > 0
2. Parse the date column, normalize to day
3. Group by day, count unique journalists (`nunique(author)`)
4. Reindex to the full date range, filling missing days with 0

**Computation**: `rolling_zscore(daily_unique_journalist_count)`

**Interpretation**: Detects when an unusual number of journalists engage with the topic. A surge from 5 journalists/day (baseline) to 20 journalists/day signals that the topic has broken out of its usual reporter pool — a key cascade indicator.

**Why unique counts, not article counts**: A single prolific journalist publishing 10 articles in a day does not indicate a cascade. What matters is the breadth of independent editorial decisions to cover the topic.

### 5.4 Signal 3: Convergence Anomaly ($z_{\text{convergence}}$)

**Raw input**: Daily dominance ratio — this frame's daily proportion relative to the sum of all 8 frames' daily proportions.

**Data flow**:
1. For each of the 8 frames, retrieve `daily_proportions` from `temporal_index[frame]`
2. Sum all frames' daily proportions into `all_sums(t) = \sum_{f \in \text{frames}} p_f(t)$
3. Compute dominance ratio:

$$
\text{dominance}(t) = \begin{cases}
\frac{p_{\text{target}}(t)}{\sum_{f \in \text{frames}} p_f(t)} & \text{if } \sum_f p_f(t) > 0 \\
0 & \text{otherwise}
\end{cases}
$$

**Computation**: `rolling_zscore(dominance)`

**Interpretation**: Detects when a single frame monopolizes the media agenda relative to all other frames. This is distinct from Signal 1 (which measures frame share of total coverage): Signal 3 measures frame dominance within the multi-frame landscape.

**Example**: If the environmental frame suddenly captures 60% of all frame-tagged coverage (dominance ratio = 0.60) when it normally captures 12% (baseline), this indicates strong frame convergence.

### 5.4.1 Convergence Orthogonalization

The convergence signal ($z_{\text{convergence}}$) and temporal signal ($z_{\text{temporal}}$) are approximately $\rho \approx 0.92$ correlated because both are functions of daily frame proportion. To ensure the convergence signal captures *additional* information about frame dominance (target frame / all frames) that is not already explained by the temporal signal, $z_{\text{convergence}}$ is orthogonalized with respect to $z_{\text{temporal}}$ before entering the composite.

**Computation**:

$$
\beta = \frac{\langle z_{\text{conv}}, z_{\text{temp}} \rangle}{\langle z_{\text{temp}}, z_{\text{temp}} \rangle}, \quad z_{\text{conv}}^{\perp} = \max\left(0, z_{\text{conv}} - \beta \cdot z_{\text{temp}}\right)
$$

The projection subtraction removes the component of convergence that is linearly predictable from the temporal signal. The result is re-clipped to $\geq 0$ (one-sided) so that only residual positive anomalies count.

**Effect**: When convergence closely tracks temporal (as in most routine coverage), the orthogonalized signal is near-zero. It only fires when the convergence/dominance ratio deviates from what temporal alone would predict — e.g., when one frame monopolizes the landscape even relative to its own elevated proportion.

If $z_{\text{temporal}}$ is all-zero (no temporal anomaly), the orthogonalization is skipped and $z_{\text{convergence}}$ passes through unchanged.

### 5.5 Signal 4: Source Anomaly ($z_{\text{source}}$)

**Raw input**: Daily messenger concentration, defined as 1 minus the normalized Shannon entropy of the messenger type distribution.

**Data flow**:
1. Filter articles to those tagged with the current frame
2. For each day $t$, sum messenger column values across that day's articles to get a messenger count vector $(c_1, c_2, \ldots, c_K)$ where $K \leq 9$ is the number of available messenger types
3. Compute concentration:

$$
p_k(t) = \frac{c_k(t)}{\sum_k c_k(t)}, \quad H(t) = -\sum_{k: p_k > 0} p_k(t) \ln p_k(t)
$$

$$
\text{concentration}(t) = 1 - \frac{H(t)}{\ln K}
$$

**Computation**: `rolling_zscore(concentration)`

**Interpretation**: High concentration (low entropy) means a few messenger types dominate coverage — sources are narrowing. In a normal news environment, multiple messenger types contribute (scientists, officials, economists); during a cascade, coverage often converges on a single authority type.

**Example**: During a health-framed cascade about climate impacts, `msg_health` and `msg_scientist` may dominate to the exclusion of other messenger types, producing high concentration and a positive Z-score.

### 5.6 Signal 5: Semantic Anomaly ($z_{\text{semantic}}$)

**Raw input**: Daily mean pairwise cosine similarity of article-level embeddings for articles tagged with this frame.

**Data flow**:
1. Filter articles to those tagged with the current frame
2. For each day $t$:
   - Get all `doc_id` values from articles published on day $t$
   - If fewer than 2 doc_ids: similarity = 0 (skip day)
   - Call `embedding_store.mean_pairwise_similarity(doc_ids)`:
     - Retrieve article-level embeddings (mean-pooled from sentence embeddings)
     - L2-normalize each embedding
     - Compute full cosine similarity matrix via `normalized @ normalized.T`
     - Extract upper triangle (exclude diagonal)
     - Return mean of upper-triangle values

$$
\text{sim}(t) = \frac{2}{n_t(n_t - 1)} \sum_{i < j} \cos(\mathbf{e}_i, \mathbf{e}_j)
$$

where $\mathbf{e}_i$ is the mean-pooled sentence embedding for article $i$ (1024-dim, float32), and $n_t$ is the number of articles with embeddings on day $t$.

**Computation**: `rolling_zscore(daily_similarity)`

**Interpretation**: High daily similarity = articles about this frame are saying similar things = content homogenization. This is a direct measure of the semantic convergence that characterizes cascades, complementing the structural signals (participation, source concentration) with actual content analysis.

**Why this signal is critical**: The other four signals measure structural and metadata-level anomalies. Signal 5 directly measures the content itself. A surge in article count (Signal 1) with broadening participation (Signal 2) could represent diverse coverage of a complex topic. But when combined with high semantic similarity (Signal 5), it strongly suggests cascade behavior — many journalists writing essentially the same story.

### 5.7 Composite Signal

The five Z-scores are combined into a single weighted daily composite:

$$
C(t) = w_1 z_{\text{temporal}}(t) + w_2 z_{\text{participation}}(t) + w_3 z_{\text{convergence}}(t) + w_4 z_{\text{source}}(t) + w_5 z_{\text{semantic}}(t)
$$

| Weight | Signal | Value |
|--------|--------|-------|
| $w_1$ | Temporal | 0.25 |
| $w_2$ | Participation | 0.20 |
| $w_3$ | Convergence | 0.20 |
| $w_4$ | Source | 0.15 |
| $w_5$ | Semantic | 0.20 |
| | **Total** | **1.00** |

**Design rationale**: Temporal receives the highest weight (0.25) because it is the most reliable indicator and typically the first signal to fire. Source receives the lowest weight (0.15) because messenger type annotations have inherently higher noise than structural measurements. The semantic signal (0.20) is weighted equally with convergence and higher than source, reflecting its importance as the only signal measuring actual article content.

**Important**: These are the **detection signal weights** (used to detect bursts). They are distinct from the **scoring dimension weights** (Section 6) used later to produce the final cascade score.

### 5.8 Burst Detection: Dual-Method Approach

**Source**: `detection/unified_detector.py`

Two complementary algorithms run on the composite signal $C(t)$:

#### Method 1: Z-Score Flagging

Days where $C(t) > \theta$ for at least `min_burst_days` consecutive days are flagged:

$$
\text{flag}_{\text{zscore}}(t) = \begin{cases}
1 & \text{if } C(t) > \theta \text{ and run length } \geq 3 \\
0 & \text{otherwise}
\end{cases}
$$

- Default threshold: $\theta = 2.0$ standard deviations
- Minimum consecutive days: 3

The consecutiveness requirement filters out isolated high-value days that represent noise rather than sustained anomalies.

#### Method 2: CUSUM Change-Point Detection (Page, 1954)

The Cumulative Sum (CUSUM) algorithm detects sustained shifts in the composite signal. Unlike Z-score flagging, which requires the signal to exceed threshold on every individual day, CUSUM accumulates deviations and can detect persistent but moderate shifts.

**Algorithm** (one-sided upper CUSUM):

$$
S_i = \max(0, S_{i-1} + C(t_i) - \bar{C} - k), \quad S_0 = 0
$$

$$
\text{flag}_{\text{cusum}}(t_i) = \begin{cases}
1 & \text{if } S_i > h \\
0 & \text{otherwise}
\end{cases}
$$

**Parameters** (computed from the full composite series):

| Parameter | Formula | Interpretation |
|-----------|---------|----------------|
| $\bar{C}$ | `composite.mean()` | Expected value under normal conditions |
| $\sigma_C$ | `composite.std()` | Standard deviation of composite |
| $h$ | $4.0 \times \sigma_C$ | Decision threshold: triggers at 4σ cumulative deviation |
| $k$ | $0.5 \times \sigma_C$ | Slack: only deviations > 0.5σ above mean contribute |

**Why dual methods**: Z-score flagging detects sharp, dramatic bursts that exceed 2σ on every day. CUSUM detects gradual, sustained shifts where the signal may not exceed 2σ on any single day but consistently runs above the mean. Together, they capture the full spectrum of cascade onset patterns.

#### Period Combination and Merging

Both methods produce boolean flag series converted to temporal periods $[start, end]$:

1. **Combine**: Periods from both methods are labeled by source (`composite_zscore`, `composite_cusum`, or `composite_both` for overlapping periods)
2. **Merge**: Adjacent periods within 3 days (`burst_merge_gap_days`) are merged — a 1-day dip in the middle of a cascade should not split it into two separate events
3. **Extend onsets**: Each period's start is extended backward up to 14 days (`onset_lookback_days`), walking back while the composite signal remains > 0. This captures the rising edge of cascades

### 5.9 BurstResult Properties

For each detected period $[t_{\text{onset}}, t_{\text{end}}]$, a `BurstResult` is constructed:

| Property | Formula | Description |
|----------|---------|-------------|
| `peak_date` | $\arg\max_{t \in \text{window}} C(t)$ | Date of maximum composite signal |
| `peak_proportion` | $\max_{t \in \text{window}} p_{\text{frame}}(t)$ | Maximum daily frame proportion |
| `baseline_mean` | Mean of $p_{\text{frame}}$ over $[t_{\text{onset}} - 90, t_{\text{onset}} - 1]$ | Pre-burst baseline level |
| `intensity` | `peak_proportion / baseline_mean` | Burst magnitude relative to baseline |
| `duration_days` | $(t_{\text{end}} - t_{\text{onset}}).\text{days} + 1$ | Total burst duration |
| `detection_method` | `composite_both`, `composite_zscore`, or `composite_cusum` | Detection provenance |

### 5.10 Semantic Peak Date

**Source**: `detection/unified_detector.py` — `_compute_semantic_peak()`

The legacy `peak_date` was the argmax of the composite Z-score $C(t)$, which is dominated by article volume. The **semantic peak** instead uses embedding-based weights to locate the date where content convergence is highest, producing a peak that reflects narrative concentration rather than just activity.

**Algorithm**:

1. For each article in the burst window, retrieve the article embedding and the frame signal value (e.g., `economic_frame_mean`).

2. Compute the **frame-signal-weighted centroid** of all article embeddings:

$$
\mathbf{c} = \frac{\sum_i s_i \cdot \mathbf{e}_i}{\sum_i s_i}, \quad \hat{\mathbf{c}} = \frac{\mathbf{c}}{|\mathbf{c}|}
$$

where $s_i$ is the frame signal and $\mathbf{e}_i$ is the article embedding.

3. For each article, compute a weight combining frame signal and semantic proximity to the centroid:

$$
w_i = s_i \times \max(0, \cos(\mathbf{e}_i, \hat{\mathbf{c}}))
$$

4. Aggregate weights by day to produce a **daily semantic mass** series:

$$
M(t) = \sum_{i : \text{date}(i) = t} w_i
$$

5. Compute the **weighted P50** (median) of the daily mass series:

$$
\text{peak\_date} = \text{weightedPercentile}_{50}\left(\{t : M(t) > 0\}, \{M(t)\}\right)
$$

The weighted percentile uses midpoint-centered interpolation to avoid edge bias.

**Fallback**: If fewer than 2 embeddings are found, or if the centroid is degenerate, the system falls back to the legacy composite-argmax peak date.

**Daily composite substitution**: The `daily_semantic_mass` series replaces the composite Z-score as the cascade's `daily_composite` when available, providing a content-weighted activity trace for downstream impact analysis.

---

## 6. Scoring Pipeline: Four Dimensions and Seventeen Sub-Indices

**Source**: `detection/unified_detector.py`

Every burst is scored by `_score_cascade()`. First, burst-level data is extracted: articles in the window tagged with the burst's frame, journalist/media counts, daily time series, and network structure. Then, four scoring functions compute sub-indices.

**Architecture**: Each dimension's score is the **arithmetic mean** of its sub-indices, clipped to [0, 1]. The total score is a **weighted sum** of dimension scores, also clipped to [0, 1].

**Hard filter**: If a burst has fewer than 3 articles (`MIN_ARTICLES_HARD`) or 0 journalists, the cascade receives a total score of 0 without computing any sub-indices. This prevents meaningless scores from noise.

### 6.1 Dimension 1 — Temporal Dynamics (Weight: 0.25)

Measures how strong, fast, sustained, and statistically significant the burst is.

#### Sub-index 1.1: Burst Intensity

**Raw input**: `cascade.burst_intensity = peak_proportion / baseline_mean`

**Normalization**:

$$
I_{\text{burst}} = \min\left(1, \frac{\text{burst\_intensity}}{5.0}\right)
$$

**Interpretation**: A ratio of 5× baseline is considered exceptional in Canadian media. A frame that normally occupies 10% of coverage rising to 50% (5× intensity) scores 1.0.

**Why 5.0**: Analysis of Canadian media coverage patterns shows that even the most dramatic cascades (e.g., pipeline debates, climate strikes) rarely exceed 5× their baseline frame proportion. This normalization constant ensures the sub-index uses the full [0, 1] range without saturating too early.

#### Sub-index 1.2: Adoption Velocity

**Raw input**: `cascade.adoption_velocity` — rate of new journalist adoption during the growth phase (onset to peak).

**Computation**:

$$
V = \frac{\text{cumulative\_journalists}(t_{\text{peak}}) - \text{cumulative\_journalists}(t_{\text{onset}})}{(t_{\text{peak}} - t_{\text{onset}}).\text{days}}
$$

**Normalization**:

$$
V_{\text{norm}} = \min\left(1, \frac{V}{3.0}\right)
$$

**Interpretation**: 3 new journalists per day is considered very fast adoption. This captures the "bandwagon effect" — how quickly new voices join the conversation.

#### Sub-index 1.3: Duration

**Raw input**: `cascade.duration_days = (t_{\text{end}} - t_{\text{onset}}).days + 1`

**Normalization**:

$$
D = \min\left(1, \frac{\text{duration\_days}}{30.0}\right)
$$

**Interpretation**: Cascades in Canadian media rarely exceed 30 days. Longer cascades indicate sustained societal engagement with the topic rather than a fleeting news cycle.

#### Sub-index 1.4: Mann-Whitney U Test

**Purpose**: Statistical validation that the burst represents a genuine shift from baseline, not random fluctuation.

**Computation**: Two-sample Mann-Whitney U test (non-parametric):
- **Baseline sample**: Daily proportions in $[t_{\text{onset}} - 90, t_{\text{onset}} - 1]$
- **Burst sample**: Daily proportions in $[t_{\text{onset}}, t_{\text{end}}]$
- **Hypothesis**: $H_1$: burst proportions are stochastically greater than baseline proportions (one-tailed)

$$
p = \text{mannwhitneyu}(\text{burst\_daily}, \text{baseline\_daily}, \text{alternative}=\text{'greater'})
$$

**Normalization** (gated):

| p-value | Sub-index |
|---------|-----------|
| $p < 0.01$ | 1.0 (highly significant) |
| $p < 0.05$ | 0.5 |
| $p < 0.10$ | 0.25 |
| $p \geq 0.10$ | 0.0 |

**Why Mann-Whitney**: Non-parametric tests are appropriate because daily proportions are not normally distributed — they have a lower bound at 0 and can have heavy tails during cascade events. Mann-Whitney is robust to non-normality and outliers.

**Edge cases**: Returns $p = 1.0$ (sub-index = 0) if baseline has < 5 days or burst has < 3 days.

**Dimension score**:

$$
S_{\text{temporal}} = \text{clip}\left(\frac{I_{\text{burst}} + V_{\text{norm}} + D + \text{MW}}{4}, 0, 1\right)
$$

### 6.2 Dimension 2 — Participation Breadth (Weight: 0.25)

Measures how broadly the cascade is adopted across journalists, outlets, and network structure.

#### Sub-index 2.1: Actor Diversity

**Raw input**: `cascade.top_journalists` — list of `(journalist_name, article_count)` tuples (top 10 by output).

**Computation**: Normalized Shannon entropy of the journalist article count distribution:

$$
p_i = \frac{\text{articles}_i}{\sum_j \text{articles}_j}, \quad H = -\sum_{i: p_i > 0} p_i \ln p_i, \quad H_{\text{norm}} = \frac{H}{\ln n}
$$

**Normalization**: $\min(1, H_{\text{norm}})$

**Interpretation**: A uniform distribution across journalists (high entropy) indicates diverse authorship — the cascade has genuinely spread beyond a single voice. Concentrated output from one or two journalists (low entropy) suggests limited participation despite high article counts.

#### Sub-index 2.2: Cross-Media Ratio

**Raw input**: `cascade.n_media` — unique media outlets in the burst window.

$$
R_{\text{media}} = \min\left(1, \frac{n_{\text{media}}}{20}\right)
$$

**Interpretation**: 20 reflects the approximate size of the Canadian major media landscape (21 outlets tracked in the geographic data). A cascade reaching 15 of 20 outlets scores 0.75.

#### Sub-index 2.3: New Entrant Rate

**Raw input**: Fraction of journalists in the burst who were NOT active in the 90-day baseline (same frame).

$$
E = \min\left(1, \frac{n_{\text{new\_journalists}}}{n_{\text{journalists}}}\right)
$$

**Interpretation**: High new entrant rate means the cascade attracted journalists who don't normally cover this frame — a key indicator that the topic has broken out of its specialist pool.

#### Sub-index 2.4: Growth Pattern

**Raw input**: `cumulative_journalists` time series during the burst window.

**Computation**: Fraction of days with new journalist adoption:

$$
\text{growth\_pattern} = \frac{|\{i : \text{cum}[i] > \text{cum}[i-1]\}|}{n - 1}
$$

**Interpretation**: A value of 1.0 means every day added at least one new journalist — perfectly monotonic growth. Low values indicate early saturation or sporadic participation.

#### Sub-index 2.5: Network Structure (Clustering-to-Density Ratio)

**Raw input**: Co-coverage network graph (see Section 7 for construction details).

**Computation**: A scale-free metric based on the ratio of average clustering coefficient to graph density:

$$
\text{structure} = \begin{cases}
\min\left(1, \frac{\bar{C} / \rho}{20}\right) & \text{if } \rho > 0 \text{ and } n \geq 5 \\
0.5 & \text{if } n < 5 \quad \text{(neutral)} \\
0 & \text{otherwise}
\end{cases}
$$

where $\bar{C}$ is the average clustering coefficient and $\rho$ is the graph density.

**Why clustering/density ratio**: Raw density $2m/n(n-1)$ decreases quadratically with graph size, making it unreliable across cascades of different sizes. The clustering-to-density ratio is scale-free: in random graphs, $\bar{C} \approx \rho$, so the ratio ≈ 1. In cascade networks with community structure, clustering greatly exceeds density, producing ratios of 5–20×. The normalization constant 20 maps this range to [0, 1]. Networks with < 5 nodes receive a neutral score (0.5) because clustering is undefined or unreliable on very small graphs.

#### Sub-index 2.6: Network Cohesion

**Raw input**: Modularity from Louvain community detection on the co-coverage graph.

$$
\text{cohesion} = \max(0, \min(1, 1 - Q))
$$

where $Q$ is the Newman-Girvan modularity.

**Interpretation**: Low modularity means the cascade crossed media boundaries — journalists from different outlets are interleaved in the network rather than forming separate clusters. Inverted because high modularity (fragmented communities) is the opposite of cascade behavior.

**Dimension score**:

$$
S_{\text{participation}} = \text{clip}\left(\frac{H_{\text{actors}} + R_{\text{media}} + E + \text{growth} + \text{structure} + \text{cohesion}}{6}, 0, 1\right)
$$

### 6.3 Dimension 3 — Content Convergence (Weight: 0.25)

**Entirely embedding-based.** Measures semantic homogenization of content during the cascade using the `SemanticConvergenceCalculator` (`embeddings/semantic_convergence.py`).

**Pre-processing before metric computation**:
1. Collect all frame-specific article doc_ids in the cascade window
2. Subsample to max 500 articles (seed 42) for computational tractability
3. Deduplicate near-duplicate articles (threshold 0.95) to remove syndicated content
4. Retrieve article dates and media outlet mappings from the source index

#### Sub-index 3.1: Semantic Similarity (Intra-Window)

**Computation**: Mean pairwise cosine similarity of all article embeddings in the cascade window:

$$
S_{\text{intra}} = \frac{2}{n(n-1)} \sum_{i < j} \cos(\mathbf{e}_i, \mathbf{e}_j)
$$

where $\mathbf{e}_i$ is the mean-pooled sentence embedding for article $i$.

**Normalization**: `clip(x, 0, 1)`, then adjusted by the syndication penalty (Section 6.5):

$$
S_{\text{intra}}^{\text{adjusted}} = S_{\text{intra}} \times (1 - 0.5 \times r_{\text{synd}})
$$

**Interpretation**: Measures the overall degree of semantic homogeneity. High values indicate that articles are saying similar things — the hallmark of a cascade where coverage converges on a shared narrative. The syndication penalty ensures that wire-service duplication does not inflate this measure.

**Typical values**: Routine coverage produces similarity in the 0.40–0.60 range. Strong cascades can push this to 0.70–0.85.

#### Sub-index 3.2: Convergence Trend (Temporal Slope)

**Computation**:
1. Sort articles chronologically
2. Split into 5 equal-size temporal sub-windows
3. For each sub-window: compute intra-window mean pairwise cosine similarity
4. Fit linear regression on `(window_index, similarity_value)`

**Normalization**:

$$
\text{convergence\_trend} = \text{clip}\left(\frac{\beta_1 + 0.05}{0.10}, 0, 1\right)
$$

This maps the slope range $[-0.05, +0.05]$ to $[0, 1]$:
- Slope = −0.05 (divergence) → sub-index = 0
- Slope = 0 (no trend) → sub-index = 0.5
- Slope = +0.05 (convergence) → sub-index = 1.0

**Interpretation**: A positive slope means articles within the cascade are becoming more similar over time — the cascade is actively producing increasingly homogeneous content. This distinguishes active convergence from static similarity.

#### Sub-index 3.3: Cross-Media Alignment

**Computation**:
1. Group articles by media outlet
2. For each outlet with ≥ 2 articles: compute centroid = mean of article embeddings
3. Compute mean pairwise cosine similarity between outlet centroids

$$
A_{\text{cross}} = \frac{2}{m(m-1)} \sum_{i < j} \cos(\mathbf{c}_i, \mathbf{c}_j), \quad \mathbf{c}_k = \frac{1}{|D_k|} \sum_{d \in D_k} \mathbf{e}_d
$$

**Normalization**: `clip(x, 0, 1)`

**Interpretation**: Measures whether different media outlets are converging on similar language at the organizational level, beyond individual article comparisons. High alignment means outlets that typically have distinct editorial voices are producing similar content — a strong cascade signal.

#### Sub-index 3.4: Novelty Decay

**Computation** (incremental centroid method):
1. Sort articles chronologically
2. For each article $i$ in order:
   - If $i = 0$: novelty = 1.0 (maximally novel). Initialize running sum = $\mathbf{e}_0$.
   - If $i > 0$: compute running centroid of all previous articles:

$$
\mathbf{c}_{i-1} = \frac{\text{running\_sum}}{n_{\text{added}}}, \quad \text{novelty}(i) = 1 - \cos(\mathbf{e}_i, \mathbf{c}_{i-1})
$$

3. Fit linear regression on novelty scores
4. Decay rate = $-\text{slope}$ (positive when novelty decreases)

**Normalization**:

$$
\text{novelty\_decay} = \text{clip}\left(\frac{\text{decay\_rate} \times n_{\text{articles}}}{0.5}, 0, 1\right)
$$

The multiplication by $n_{\text{articles}}$ makes the metric size-invariant: the total cumulative decay matters, not the per-article slope.

**Interpretation**: Measures how quickly new articles become redundant. In a cascade, later articles add progressively less novel information because the narrative has converged. The incremental centroid approach captures this directly: each article is compared against the evolving "consensus" of all prior articles.

**Additional metrics** (stored but not used in sub-index scoring):
- `final_novelty`: Mean novelty of the last 20% of articles
- `novelty_half_life`: Normalized position where novelty first drops below 50% of initial

**Dimension score**:

$$
S_{\text{convergence}} = \text{clip}\left(\frac{S_{\text{intra}} + T_{\text{conv}} + A_{\text{cross}} + D_{\text{novelty}}}{4}, 0, 1\right)
$$

### 6.4 Dimension 4 — Source Convergence (Weight: 0.25)

Measures whether sources, messenger profiles, and journalist content align.

#### Sub-index 4.1: Source Diversity Decline

**Computation**:
1. Split cascade articles at temporal midpoint
2. For each half: compute normalized Shannon entropy of the 9-type messenger distribution
3. Compute relative decline:

$$
\Delta H = \frac{H_{\text{first}} - H_{\text{second}}}{H_{\text{first}}}
$$

4. Apply log-scaled confidence damping for small cascades:

$$
\text{confidence} = \text{clip}\left(\frac{\log_2 n}{\log_2 100}, 0, 1\right)
$$

$$
\text{diversity\_decline} = \text{clip}(\Delta H \times \text{confidence}, 0, 1)
$$

**Interpretation**: A positive decline means the second half of the cascade uses fewer messenger types — sources are converging as the cascade matures. The confidence damping prevents small cascades (< 7 articles, where $\log_2 n < \log_2 100 \times 0.5$) from producing unreliable entropy estimates.

#### Sub-index 4.2: Messenger Concentration

**Computation**:

$$
\text{concentration} = 1 - H_{\text{norm}}
$$

where $H_{\text{norm}}$ is the normalized Shannon entropy across all 9 messenger types over the entire cascade window.

**Interpretation**: High concentration = few messenger types dominate. This complements sub-index 4.1: while 4.1 measures the *change* in diversity over the cascade's lifetime, 4.2 measures the *absolute level* of concentration. Both can be informative independently — a cascade might start with high concentration (4.2 high) without further decline (4.1 low).

#### Sub-index 4.3: Media Coordination (Journalist Embedding Similarity)

**Computation**:
1. For each journalist in the cascade: get all their article doc_ids
2. Retrieve embeddings and compute journalist centroid = mean of article embeddings
3. Compute mean pairwise cosine similarity between journalist centroids

$$
\text{coord} = \frac{2}{J(J-1)} \sum_{i < j} \cos(\mathbf{c}_i^{\text{journalist}}, \mathbf{c}_j^{\text{journalist}})
$$

where $\mathbf{c}_i^{\text{journalist}} = \frac{1}{|D_i|} \sum_{d \in D_i} \mathbf{e}_d$ is the centroid of journalist $i$'s articles.

**Interpretation**: Measures whether journalists covering this cascade are producing semantically similar content. This is a stronger signal than messenger-profile similarity (4.2) because it operates on the actual text content via embeddings rather than metadata annotations. When independent journalists from different outlets produce highly similar content, it indicates coordinated messaging or shared source material.

**Dimension score**:

$$
S_{\text{source}} = \text{clip}\left(\frac{\Delta H + \text{concentration} + \text{coord}}{3}, 0, 1\right)
$$

### 6.5 Score Adjustments

Before computing the total score, two adjustments are applied:

#### Syndication Penalty

Wire-service articles (Reuters, AP, Canadian Press) are republished verbatim across outlets, inflating semantic similarity without reflecting independent editorial convergence. After deduplication (Section 4.4) removes near-identical articles, the remaining syndication effect is handled via a soft penalty on the semantic similarity sub-index:

$$
S_{\text{semantic}}^{\text{adjusted}} = S_{\text{semantic}}^{\text{raw}} \times (1 - 0.5 \times r_{\text{synd}})
$$

where $r_{\text{synd}}$ is the syndication ratio (fraction of article pairs with cosine similarity > 0.95 before deduplication). This reduces semantic similarity by up to 50% for fully syndicated cascades.

#### Media Confidence Factor

The base score (weighted sum of dimensions) is multiplied by a media confidence factor that discounts cascades reaching few outlets:

$$
\text{conf}_{\text{media}} = \min\left(1, \frac{\log_2(\max(n_{\text{media}}, 1))}{\log_2(10)}\right)
$$

| Media outlets | Confidence |
|---------------|------------|
| 1 | 0.00 |
| 2 | 0.30 |
| 5 | 0.70 |
| 10+ | 1.00 |

This reflects the principle that a cascade requires multi-outlet participation: a single-outlet burst, regardless of intensity, is not a cascade.

### 6.6 Total Score Computation

$$
S_{\text{total}} = \text{clip}\left(\left(0.25 \cdot S_{\text{temporal}} + 0.25 \cdot S_{\text{participation}} + 0.25 \cdot S_{\text{convergence}} + 0.25 \cdot S_{\text{source}}\right) \times \text{conf}_{\text{media}}, 0, 1\right)
$$

**Weight rationale**: All four dimensions receive equal weight (0.25), reflecting the principle that each dimension captures a distinct and necessary aspect of cascade behavior. No single dimension should dominate the score.

---

## 7. Network Construction and Metrics

### 7.1 Co-Coverage Graph Definition

**Source**: `detection/network_builder.py`

The co-coverage network is an undirected weighted graph where:

- **Nodes**: `(journalist_name: str, media_outlet: str)` tuples. The same journalist at different outlets appears as separate nodes. Each article contributes one node.

- **Edges**: Two nodes are connected if they both published on the **same calendar day** using articles tagged with the **cascade's frame**. Edge weight = number of co-occurrence days.

### 7.2 Construction Algorithm

```
For each date in [onset_date, end_date]:
    actors = set()
    For each article on this date tagged with cascade frame:
        actor = (article.author, article.media)
        actors.add(actor)
        Add node(actor) with attributes {journalist, media}

    For each pair (a, b) in actors:
        If edge(a, b) exists:
            edge.weight += 1
        Else:
            Add edge(a, b, weight=1)
```

### 7.3 Metrics via NetworKit (Subprocess Isolation)

Metrics are computed via NetworKit in a separate subprocess to avoid a dual-libomp SIGSEGV crash. Both PyTorch (for embeddings) and NetworKit bundle their own `libomp.dylib` — running both in the same process causes segmentation faults in `__kmp_suspend_initialize_thread`. The solution is subprocess isolation with pickle serialization over stdin/stdout.

**Protocol**:
1. Main process serializes the graph (nodes + edges) via pickle
2. Subprocess (`detection/networkit_worker.py`) loads NetworKit, builds the graph, computes metrics
3. Results are pickled back to the main process
4. Fallback to pure NetworkX if the subprocess fails

| Metric | NetworKit Method | NetworkX Fallback | Range |
|--------|-----------------|-------------------|-------|
| Density | $2m / n(n-1)$ | `nx.density(G)` | [0, 1] |
| Modularity | PLM (`refine=True, gamma=1.0`) + `Modularity().getQuality()` | `community_louvain.best_partition(random_state=42)` + `nx.modularity()` | [0, 1] |
| Mean degree centrality | `DegreeCentrality(normalized=True).scores()` → mean | `nx.degree_centrality()` → mean | [0, 1] |
| Connected components | `ConnectedComponents().numberOfComponents()` | `nx.number_connected_components()` | [1, ∞) |

### 7.4 Exhaustive Metrics Calculator

**Source**: `metrics/exhaustive_metrics_calculator.py`

For deeper analysis, the `ExhaustiveMetricsCalculator` computes 73+ exact network metrics organized in 7 categories:

| Category | Metrics | Examples |
|----------|---------|---------|
| Centrality (13) | degree, betweenness, closeness, eigenvector, PageRank, Katz, harmonic, load | Identifies influential nodes |
| Clustering (6) | local, global, transitivity, average, square | Measures triadic closure |
| Community (7) | Louvain, label propagation, greedy modularity, coverage | Detects community structure |
| Structure (12) | density, diameter, radius, avg path length, efficiency, assortativity | Global topology |
| Connectivity (7) | node/edge connectivity, algebraic connectivity, spectral gap | Robustness measures |
| Robustness (7) | percolation threshold, attack robustness, k-core, degeneracy | Cascade resilience |
| Spectral (5) | eigenvalues, spectral radius, energy, Estrada index | Algebraic properties |

All metrics are computed exactly (no approximation or sampling) using NetworKit in batch subprocess mode for efficiency.

---

## 8. Classification and Output

### 8.1 Classification Thresholds

| Classification | Threshold | Interpretation |
|----------------|-----------|----------------|
| **`strong_cascade`** | $S_{\text{total}} \geq 0.65$ | Multi-dimensional cascade with strong signals across most dimensions |
| **`moderate_cascade`** | $S_{\text{total}} \geq 0.40$ | Clear cascade pattern but not all dimensions align strongly |
| **`weak_cascade`** | $S_{\text{total}} \geq 0.25$ | Some cascade features detected, below moderate threshold |
| **`not_cascade`** | $S_{\text{total}} < 0.25$ | Burst detected but insufficient cascade evidence |

Classification is determined by iterating thresholds from highest to lowest.

### 8.2 CascadeResult Structure

Each scored burst produces a `CascadeResult` dataclass containing:

**Identification**: `cascade_id` (format: `{frame}_{onset_YYYYMMDD}_{counter}`), `frame`, temporal boundaries (onset, peak, end, duration).

**Participation counts**: `n_articles`, `n_journalists`, `n_media`, `n_new_journalists`.

**4 dimension scores**: `score_temporal`, `score_participation`, `score_convergence`, `score_source`.

**17 sub-indices**: Stored in `sub_indices: Dict[str, float]`, keyed by dimension-prefixed names (e.g., `temporal_burst_intensity`, `convergence_semantic_similarity`, `participation_network_density`).

**Network metrics**: `network_density`, `network_modularity`, `network_mean_degree`, `network_n_components`.

**Semantic convergence raw values**: `semantic_similarity`, `convergence_trend` (raw slope), `cross_media_alignment`, `novelty_decay_rate` (raw rate).

**Source convergence**: `source_diversity_decline`, `messenger_concentration`, `media_coordination`.

**Time series for visualization**: `daily_articles`, `daily_journalists`, `cumulative_journalists`, `daily_composite` (composite signal), `daily_signals` (per-signal Z-scores).

**Context**: `top_journalists` (top 10 by output), `top_media` (top 10), `dominant_events`, `dominant_messengers`.

**Event occurrences**: `event_occurrences: List[EventOccurrence]` — all occurrences attributed to this cascade, with belonging scores and confidence metrics.

**Full data capture**: `network_edges` (full edge list for graph reconstruction), `convergence_metrics_full` (all raw convergence metrics including syndication stats).

**Statistical test**: `mann_whitney_p` (p-value from burst vs. baseline comparison).

### 8.3 DetectionResults

The top-level container aggregates all cascades from a pipeline run:

- `cascades: List[CascadeResult]` — All scored cascades
- `all_bursts: List[BurstResult]` — All raw burst detections
- `n_cascades_by_frame: Dict[str, int]` — Count per frame
- `n_cascades_by_classification: Dict[str, int]` — Count per classification
- `analysis_period: Tuple[str, str]` — Date range
- `n_articles_analyzed: int` — Total articles processed
- `runtime_seconds: float`
- `detection_parameters: Dict` — Snapshot of detection config
- `frame_signals: Dict[str, Dict]` — Per-frame daily Z-score signals
- `event_clusters: List[EventCluster]` — All detected meta-events (Section 9)
- `all_occurrences: List[EventOccurrence]` — All detected event occurrences
- `cascade_attributions: List[CascadeAttribution]` — Occurrence-cascade linkage
- `paradigm_shifts: Optional[ParadigmShiftResults]` — Paradigm shift analysis results (Section 11)
- `event_impact: Optional[UnifiedImpactResults]` — Unified 3-phase causal impact results (Section 10)

**Export methods**:
- `to_dataframe()` → One row per cascade (pandas DataFrame)
- `to_json(path)` → Full serialization to JSON
- `summary()` → Human-readable text summary

---

## 9. Event Occurrence Detection

### 9.1 Motivation and Architecture

The cascade detection pipeline (Sections 5–8) identifies frame-specific surges and scores them. However, it does not answer: **what specific real-world events are these cascades covering?** The CCF database annotates every sentence with 8 event types (`evt_weather`, `evt_meeting`, `evt_publication`, `evt_election`, `evt_policy`, `evt_judiciary`, `evt_cultural`, `evt_protest`), but these are sentence-level labels, not structured event instances.

The `EventOccurrenceDetector` (`analysis/event_occurrence.py`) bridges this gap by detecting **distinct event occurrences** — dated, semantically validated clusters of articles covering the same real-world event — across the entire analysis period.

**Key architectural decision**: Events are detected independently of cascades. Detection operates on **all** articles in the period, not just those within cascade windows. This database-first approach means:

1. Events exist as first-class entities, not cascade byproducts
2. The same event can be attributed to multiple cascades (or none)
3. Event detection quality is not contaminated by cascade boundary artifacts

After detection, a separate attribution step links occurrences to cascades via temporal overlap and shared articles.

### 9.2 Pipeline Overview

```
All articles with evt_*_mean > 0
    │
    ▼
Phase 1: SEED SELECTION
    Composite seed score → P50 threshold → seed pool
    │
    ▼
Phase 2: PER-TYPE HAC CLUSTERING (seeds only)
    Title+sentence embedding blend → 3D distance → agglomerative clustering
    One cluster set per event type
    │
    ▼
(intermediate): BUILD TEMPORARY OCCURRENCES
    Compute temporal bounds, centroid for Phase 3 input
    │
    ▼
Phase 3: MULTI-TYPE EVENT CLUSTER MERGE
    Pool all mono-type occurrences → deduplicate → HAC → silhouette cut
    EventCluster objects replace constituent mono-type _RawClusters
    │
    ▼
Phase 4: ITERATIVE 4D SOFT ASSIGNMENT (all articles → clusters)
    2 iterations: temporal + semantic + entity + signal distance
    Self-adjusting threshold, belonging ∈ [0, 1]
    │
    ▼
Phase 5: CONFIDENCE SCORING
    5-component quality assessment
    │
    ▼
Post-processing: CONNECTIVITY + INCLUSION + FRAGMENTATION
    Step 6b: seed-overlap connectivity (BFS split)
    Step 6c: seed inclusion (absorption if ≥80% subset)
    Step 6d: fragmentation safety net (high-overlap consolidation)
    │
    ▼
ATTRIBUTION: LINK TO CASCADES
    Temporal overlap + shared articles → CascadeAttribution
```

### 9.3 Phase 1: Seed Selection

Seeds are articles with a non-zero event signal that are likely to be at the core of an event occurrence.

**Composite seed score**: For each event type $e$ and each article $a$:

$$
\text{seed\_score}(a, e) = 0.6 \times \overline{\text{evt}_e}(a) + 0.4 \times \overline{\text{event}}(a)
$$

where $\overline{\text{evt}_e}(a)$ is the article's mean score for event type $e$ and $\overline{\text{event}}(a)$ is the overall event mean. If `event_mean` is absent, fallback to $\overline{\text{evt}_e}$ only.

**Seed threshold**: Articles with $\overline{\text{evt}_e} > 0$ and seed score ≥ P50 of all positive-signal articles become seeds for type $e$.

**Dominant ratio filter**: An article is seeded for type $e$ only if $\overline{\text{evt}_e} \geq 0.5 \times \max_k(\overline{\text{evt}_k})$, preventing minor annotations from generating spurious seeds.

### 9.4 Phase 2: Per-Type HAC Clustering

For each event type, seeds are clustered using agglomerative hierarchical clustering (HAC) with average linkage.

**Embedding blend**: Article embeddings combine title and sentence representations:

$$
\mathbf{e}_a = \alpha \cdot \mathbf{e}_{\text{title}} + (1 - \alpha) \cdot \bar{\mathbf{e}}_{\text{sentences}}, \quad \alpha = 0.30
$$

**3D compound distance**: Between two seed articles $a, b$:

$$
d(a, b) = 0.50 \cdot d_{\text{semantic}} + 0.30 \cdot d_{\text{temporal}} + 0.20 \cdot d_{\text{entity}}
$$

| Component | Formula | Scale |
|-----------|---------|-------|
| Semantic | $1 - \cos(\mathbf{e}_a, \mathbf{e}_b)$ | [0, 2] |
| Temporal | $1 - \exp(-|\Delta t| / \tau)$, $\tau = 14$ days | [0, 1] |
| Entity | $1 - J(E_a, E_b)$, Jaccard on entity sets | [0, 1] |

**Singleton preservation**: Seeds that form clusters of size 1 (below `MIN_CLUSTER_SIZE=2`) are preserved as **micro unique events** rather than discarded. These singletons carry the `is_singleton=True` flag on their `EventOccurrence` and are naturally flagged `low_confidence=True` by the confidence formula (size_adequacy and recruitment_success contribute ~0). Phase 4 can recruit additional articles into singletons via soft assignment, potentially growing them into larger occurrences. This preserves event signal from isolated articles (e.g., 57% of `evt_weather` seeds in 2018 were singletons) that would otherwise be lost.

### 9.5 Phase 3: Multi-Type Event Cluster Merge

All mono-type occurrences are pooled across event types to detect multi-type meta-events (e.g., a COP summit generating `evt_meeting` + `evt_policy` + `evt_publication` occurrences simultaneously).

**Deduplication**: Before HAC, occurrences with the same event type and Jaccard > 0.5 on `seed_doc_ids` are greedily merged (highest mass wins). This removes duplicates from cross-cascade pooling.

**5D inter-occurrence distance**:

$$
d(o_i, o_j) = 0.25 \cdot d_{\text{temporal}} + 0.20 \cdot d_{\text{semantic}} + 0.15 \cdot d_{\text{entity}} + 0.30 \cdot d_{\text{article}} + 0.10 \cdot d_{\text{type}}
$$

where $d_{\text{article}} = 1 - J(\text{seeds}_i, \text{seeds}_j)$ (Jaccard on seed doc_ids) and $d_{\text{type}}$ is binary (0 = same type, 1 = different type).

**Silhouette cut**: No fixed threshold. The algorithm searches for the optimal number of clusters $k$ by maximizing `silhouette_score` (scikit-learn). Falls back to $k = 1$ if all silhouette scores ≤ 0.

**Type structure analysis**: Each `EventCluster` is analyzed for:

- **Dominant type**: $\arg\max_t(0.6 \times \text{mass\_norm}_t + 0.4 \times \bar{J}_t)$ where $\bar{J}_t$ is mean Jaccard with other types
- **Constitutive types**: Types with Jaccard > 0 with at least one other type (genuinely co-occurring)
- **Satellite types**: Types with Jaccard = 0 with all others (temporally coincident but article-disjoint)

### 9.6 Phase 4: Iterative Soft Assignment

After Phase 3 produces EventClusters, Phase 4 assigns **all** articles (not just seeds) to clusters using a 4D distance and soft belonging scores.

**4D article-to-cluster distance**:

$$
d(a, c) = 0.25 \cdot d_{\text{temporal}} + 0.35 \cdot d_{\text{semantic}} + 0.15 \cdot d_{\text{entity}} + 0.25 \cdot d_{\text{signal}}
$$

where $d_{\text{signal}} = 1 - \overline{\text{evt}_e}(a)$ — articles with low event signal are naturally distant.

**Self-adjusting threshold**: The belonging threshold $\theta_c$ for each cluster $c$ is not fixed:

- **Iteration 0**: Bootstrapped from seed distances: $\theta_c = 2.0 \times \text{median}(\{d(a, c) : a \in \text{seeds}_c\})$
- **Iteration 1+**: Recalibrated from cluster core: $\theta_c = \text{P75}(\{d(a, c) : a \in c, \text{belonging}(a) > 0\})$
- Capped at 0.5, floored at 0.05

**Belonging score**:

$$
\text{belonging}(a, c) = \max\left(0, 1 - \frac{d(a, c)}{\theta_c}\right)
$$

**Iterative refinement**: 2 iterations (constant `PHASE4_N_ITERATIONS`). After each iteration, cluster centroids are recomputed as belonging-weighted × evt_mean-weighted averages.

### 9.7 Confidence Scoring

Each `EventOccurrence` receives a composite confidence score from 5 equally-weighted (0.20 each) components:

| Component | Formula | Measures |
|-----------|---------|----------|
| **Centroid tightness** | $\text{clip}(1 - 2 \times \bar{d}_{\text{centroid}}, 0, 1)$ | How close articles are to the centroid |
| **Coherence residual** | $\frac{\text{raw} - \text{baseline}}{1 - \text{baseline}}$ | Semantic coherence above corpus baseline |
| **Media diversity** | $1 - 1/n_{\text{media}}$ | Number of distinct media outlets |
| **Recruitment success** | $\text{clip}((n_{\text{total}} / n_{\text{seeds}} - 1) / 2, 0, 1)$ | Growth beyond initial seeds |
| **Size adequacy** | $\text{clip}(n_{\text{articles}} / 10, 0, 1)$ | Absolute cluster size |

### 9.8 Event Cluster Strength Scoring

Each `EventCluster` receives a composite strength score from 5 components:

| Component | Weight | Formula |
|-----------|--------|---------|
| **Mass** | 0.20 | $\log_2(1 + m) / \log_2(1 + 100)$ |
| **Coverage** | 0.25 | Mean of media, journalist, and geographic coverage scores |
| **Intensity** | 0.20 | $\text{effective\_mass} / \max(1, \text{core\_duration\_days})$ |
| **Coherence** | 0.15 | Residual above corpus baseline |
| **Media diversity** | 0.20 | $1 - 1/n_{\text{media}}$ |

### 9.8.1 Article-Level Temporal Bounds for EventClusters

**Source**: `analysis/event_occurrence.py` — `_compute_article_level_temporal_bounds()`

The temporal bounds of an `EventCluster` (`peak_date`, `core_start`, `core_end`) are computed from **individual article dates** weighted by a composite score, rather than from occurrence-level percentiles. This produces more precise bounds, especially for multi-type clusters where constituent occurrences may have different temporal profiles.

**Composite article weight**: For each article $a$ across all occurrences in the cluster:

$$
w(a) = b(a)^{0.40} \times \cos(a)^{0.30} \times s(a)^{0.30}
$$

where:
- $b(a) = \max_o(\text{belonging}(a, o))$ — maximum belonging score across all occurrences $o$ in the cluster
- $\cos(a) = \max(0, \cos(\mathbf{e}_a, \hat{\mathbf{c}}))$ — cosine similarity between the article embedding and the cluster centroid
- $s(a) = \max_e(\overline{\text{evt}_e}(a))$ — maximum event signal across all event types in the cluster

**Temporal bounds**: Weighted percentiles of article publication dates:

| Metric | Percentile | Description |
|--------|-----------|-------------|
| `core_start` | P10 | Early boundary of core activity |
| `peak_date` | P50 | Semantic center of mass |
| `core_end` | P90 | Late boundary of core activity |

**Fallback**: If fewer than 3 articles have valid dates and weights, the system falls back to occurrence-level temporal bounds (min/max of constituent occurrence dates).

### 9.9 Post-Processing

Three post-processing steps clean up cluster artifacts:

**Step 6b — Seed-overlap connectivity**: BFS on a graph where nodes are occurrences and edges connect occurrences with Jaccard > 0 on `seed_doc_ids`. Disconnected components are split into separate EventClusters.

**Step 6c — Seed inclusion**: If ≥ 80% of an occurrence's seeds are a subset of another occurrence's seeds (`SEED_INCLUSION_THRESHOLD = 0.80`), the smaller occurrence is absorbed into the larger.

**Step 6d — Fragmentation safety net**: Occurrences with `FRAGMENTATION_THRESHOLD = 0.60` overlap on seed doc_ids are consolidated, preventing high-overlap multi-type clusters from remaining fragmented.

### 9.10 Attribution to Cascades

`attribute_to_cascades()` links detected occurrences to cascades via:

1. **Temporal overlap**: The occurrence's `[core_start, core_end]` intersects with the cascade's `[onset_date, end_date]`
2. **Shared articles**: $|\text{occ\_doc\_ids} \cap \text{cascade\_doc\_ids}| > 0$

Each linkage produces a `CascadeAttribution` dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `cascade_id` | str | Cascade identifier |
| `occurrence_id` | int | Occurrence identifier |
| `shared_articles` | int | Number of articles in both |
| `temporal_overlap_days` | int | Days of temporal overlap |
| `overlap_ratio` | float | `shared_articles / occurrence.n_articles` |

### 9.11 Data Structures

**EventOccurrence** (`core/models.py`): A dated, semantically validated event instance with:

- Temporal bounds (first/last/core_start/core_end/peak_date)
- Soft membership: `belonging: Dict[doc_id, float]`, `doc_ids: List[int]`, `seed_doc_ids: List[int]`
- Quality: `confidence`, `confidence_components`, `semantic_coherence`
- Type validation: `type_confidence`, `type_scores`
- Strength metrics: `media_count`, `temporal_intensity`, `emotional_intensity`, `tone_coherence`

**EventCluster** (`core/models.py`): A multi-occurrence meta-event with:

- `occurrences: List[EventOccurrence]`, `event_types: Set[str]`
- Temporal bounds, `total_mass`, `centroid`
- Type analysis: `dominant_type`, `type_structure`, `type_overlap_graph`, `type_ranking`
- `strength`, `strength_components`

**DetectionResults** now includes:

- `event_clusters: List[EventCluster]` — All detected meta-events
- `all_occurrences: List[EventOccurrence]` — All detected occurrences
- `cascade_attributions: List[CascadeAttribution]` — Occurrence-cascade linkage

---

## 10. Stability Selection Impact Analysis

### 10.1 Motivation

The cascade detection pipeline (Sections 5–8) identifies *when* media cascades occur and *how strong* they are. The event occurrence detection (Section 9) identifies *which real-world events* drive coverage. But these layers do not measure the **causal relationships** between event clusters and cascades — specifically, whether a given event amplifies or dampens a cascade.

The `StabSelImpactAnalyzer` (`analysis/stabsel_impact.py`) addresses this using **stability selection** (Meinshausen & Bühlmann, 2010) with double-weighted treatment variables and OLS post-selection inference. The approach answers: for each cascade, which event clusters are statistically stable drivers or suppressors of the cascade's composite signal?

### 10.2 Dependent Variable: Two-Sided Composite Signal

For each frame, a composite signal $y(t)$ is constructed from five two-sided Z-scores computed against a 90-day trailing baseline (same signals as detection, Section 5):

$$
y(t) = 0.25 \cdot z_{\text{temporal}}(t) + 0.20 \cdot z_{\text{participation}}(t) + 0.20 \cdot z_{\text{convergence}}^{\perp}(t) + 0.15 \cdot z_{\text{source}}(t) + 0.20 \cdot z_{\text{semantic}}(t)
$$

where $z_{\text{convergence}}^{\perp}$ is the convergence signal orthogonalized with respect to the temporal signal (same projection as in detection). "Two-sided" means these Z-scores capture both positive and negative deviations, unlike the one-sided signals used for burst detection.

Each Z-score uses a rolling baseline:

$$
z(t) = \frac{x(t) - \bar{x}_{[t-90, t-1]}}{\sigma_{[t-90, t-1]}}
$$

with fallback to global mean/std for the first 45 days.

### 10.3 Treatment Variable: Double-Weighted Lagged Mass

For each event cluster $j$, a treatment variable is constructed that captures the cluster's temporal footprint weighted by its relevance to the cascade:

$$
D_j(t, l) = \sum_{a \in \text{articles}(t-l)} \underbrace{\text{belonging}(a, j)}_{\text{membership}} \times \underbrace{\text{frame\_signal}(a)}_{\text{frame relevance}} \times \underbrace{\cos\bigl(\text{emb}(a),\; \hat{c}_{\text{cascade}}\bigr)}_{\text{semantic alignment}}
$$

where:
- $\text{belonging}(a, j) \in [0, 1]$ is the soft membership score from Phase 4 assignment (max across occurrences)
- $\text{frame\_signal}(a)$ is the article's frame column value (e.g., `economic_frame_mean`)
- $\hat{c}_{\text{cascade}}$ is the **cascade centroid** — the L2-normalized, signal-weighted mean of article embeddings from the cascade's top-quartile frame-relevant articles within ±14 days of the cascade window
- $l \in \{0, 1, 2, 3\}$ are lag days (`MAX_LAG = 3`)

This triple weighting naturally eliminates frame-irrelevant clusters: a cluster with articles carrying zero frame signal or semantically distant from the cascade centroid will have $D_j \approx 0$ regardless of its belonging scores.

**Minimum treatment threshold**: Clusters with $\sum_t D_j(t, 0) < 0.01$ (`MIN_D_SUM`) are excluded.

### 10.4 Analysis Window

For each cascade with onset $t_0$, peak $t_p$, and end $t_e$:

$$
\text{window} = [\max(t_0 - 30, \text{Jan 1}),\; \min(t_e + 30, \text{Dec 31})]
$$

The 30-day margin (`MARGIN_DAYS`) captures pre-cascade and post-cascade dynamics. The window is clipped to the calendar year.

### 10.5 Variable Selection: Stability Selection

Given $n$ days and $p = K \times (L+1)$ treatment columns (where $K$ = number of clusters, $L$ = `MAX_LAG`), a high-dimensional regression problem is solved via stability selection:

1. **Calibration**: ElasticNetCV on the full (scaled) design matrix to estimate $\hat{\alpha}$ and $\hat{l_1}$. The regularization parameter used for sub-samples is $\alpha_{\text{base}} = \hat{\alpha} / 2$ (relaxed to increase selection power).

2. **Sub-sampling**: For $B = 100$ iterations (`N_SUBSAMPLES`):
   - Draw a random 50% sub-sample (`SUBSAMPLE_FRAC`) without replacement
   - Fit ElasticNet with $\alpha_{\text{base}}$ and $\hat{l_1}$
   - Record which treatment columns have $|\beta| > 10^{-10}$

3. **Selection**: A column is declared **stable** if its selection frequency $\hat{\pi}_j \geq 0.60$ (`PI_THRESHOLD`).

All random draws use `np.random.default_rng(42)` for reproducibility.

### 10.6 Post-Selection Inference: OLS + Residual Bootstrap

On the stable set of columns (plus a linear trend term), OLS is fit:

$$
\hat{\beta}_{\text{OLS}} = (X_{\text{stable}}^\top X_{\text{stable}})^{-1} X_{\text{stable}}^\top y
$$

**Residual bootstrap** (500 iterations, `N_BOOTSTRAP`): For each bootstrap draw $b$:

$$
y_b^* = \hat{y} + \hat{\varepsilon}_{\pi(b)}, \quad \hat{\beta}_b^* = (X^\top X)^{-1} X^\top y_b^*
$$

where $\pi(b)$ is a random permutation of residual indices (drawn with replacement).

**Net effect per cluster**: For each cluster $j$ with stable columns at lags $\{l_1, l_2, \ldots\}$:

$$
\hat{\beta}_j^{\text{net}} = \sum_{l \in \text{stable}} \hat{\beta}_{j,l}
$$

**P-value**: Two-sided from bootstrap distribution of $\hat{\beta}_j^{\text{net}}$:

$$
p_j = \begin{cases}
P(\beta_j^{*,\text{net}} \leq 0) & \text{if } \hat{\beta}_j^{\text{net}} > 0 \\
P(\beta_j^{*,\text{net}} \geq 0) & \text{if } \hat{\beta}_j^{\text{net}} < 0
\end{cases}
$$

**R²**: Standard OLS $R^2 = 1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$ on the full window.

### 10.7 Role Classification

Each cluster retained by stability selection is classified:

| Role | Condition | Interpretation |
|------|-----------|----------------|
| **driver** | $\hat{\beta}_j^{\text{net}} > 0$ and $p_j < 0.10$ | Event cluster amplifies cascade signal |
| **suppressor** | $\hat{\beta}_j^{\text{net}} < 0$ and $p_j < 0.10$ | Event cluster dampens cascade signal |
| **neutral** | $p_j \geq 0.10$ | Stable selection but not statistically significant |

The significance threshold is $\alpha = 0.10$ (`ALPHA_SIG`).

### 10.8 Cluster Profiling

For each cluster retained in the analysis, belonging-weighted profiles are computed across all its articles:

- **Frame profile**: Mean of each of the 8 frame columns, weighted by belonging
- **Messenger profile**: Mean of each of the 9 messenger columns, weighted by belonging
- **Event profile**: Mean of each of the 8 event columns, weighted by belonging

These profiles characterize *what kind of content* the cluster carries, enabling substantive interpretation of driver/suppressor roles.

### 10.9 Constants Reference

| Constant | Value | Description |
|----------|-------|-------------|
| `MARGIN_DAYS` | 30 | Days before/after cascade for analysis window |
| `MIN_D_SUM` | 0.01 | Minimum treatment mass for cluster inclusion |
| `MAX_LAG` | 3 | Maximum lag days for treatment variable |
| `BASELINE_WINDOW` | 90 | Rolling Z-score baseline window (days) |
| `CASCADE_CENTROID_MARGIN` | 14 | Days around cascade window for centroid articles |
| `N_SUBSAMPLES` | 100 | Number of stability selection sub-samples |
| `SUBSAMPLE_FRAC` | 0.50 | Fraction of observations per sub-sample |
| `PI_THRESHOLD` | 0.60 | Selection frequency threshold for stability |
| `N_BOOTSTRAP` | 500 | Number of residual bootstrap iterations |
| `ALPHA_SIG` | 0.10 | Significance threshold for role classification |
| `W_TEMPORAL` | 0.25 | Composite signal weight: temporal |
| `W_PARTICIPATION` | 0.20 | Composite signal weight: participation |
| `W_CONVERGENCE` | 0.20 | Composite signal weight: convergence |
| `W_SOURCE` | 0.15 | Composite signal weight: source |
| `W_SEMANTIC` | 0.20 | Composite signal weight: semantic |

### 10.10 Pipeline Integration

The stability selection impact analysis is integrated as **Step 5** of the `CascadeDetectionPipeline`, after paradigm shift analysis (Step 4):

```
Step 1: LOAD & PROCESS → Step 2: BUILD INDICES → Step 3: DETECT & SCORE → Step 3.5/3.6: EVENTS → Step 4: PARADIGM SHIFTS → Step 5: STABSEL IMPACT → RESULTS
```

The pipeline instantiates `StabSelImpactAnalyzer(embedding_store)` and calls `analyzer.run(results)`. The embedding store is required for computing cascade centroids and cosine similarities. The output `StabSelImpactResults` is stored in `results.event_impact`.

**Source**: `cascade_detector/analysis/stabsel_impact.py` — `StabSelImpactAnalyzer` class + `StabSelImpactResults` dataclass.

**Dependencies**: `numpy`, `pandas`, `scikit-learn` (ElasticNet, ElasticNetCV, StandardScaler).

### 10.11 Output Structure

`StabSelImpactResults` contains:

| Field | Type | Description |
|-------|------|-------------|
| `cluster_cascade` | DataFrame | One row per significant (cluster_id, cascade_id) pair |
| `summary` | Dict | Per-frame `{n_cascades, n_drivers, n_suppressors, median_r2}` |
| `cascade_results` | Dict | `frame → [StabSelCascadeResult]` for diagnostic plots and export |

Each `StabSelCascadeResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `cascade_id` | str | Cascade identifier |
| `frame` | str | Cascade frame abbreviation |
| `r2` | float | OLS R² on the analysis window |
| `n_clusters` | int | Total clusters with treatment mass > threshold |
| `n_stable` | int | Clusters passing stability selection |
| `n_drivers` / `n_suppressors` / `n_neutral` | int | Role counts |
| `roles` | List[ClusterRole] | Per-cluster `{cluster_id, net_beta, p_value, role, selection_freq, D_sum, lag_profile}` |
| `cluster_meta` | Dict | Per-cluster metadata (dominant_type, entities, strength, profiles) |

### 10.12 Legacy Compatibility

The `UnifiedImpactAnalyzer` (`analysis/unified_impact.py`) and `EventImpactAnalyzer` (`analysis/impact_analysis.py`) remain importable for backward compatibility but are no longer part of the pipeline. The `UnifiedImpactAnalyzer` provided a 3-phase analysis (cluster→cascade via diff-in-diff, cluster→dominance and cascade→dominance via Granger causality). The `EventImpactAnalyzer` computed annotation-level prevalence ratios using Fisher exact tests with Benjamini-Hochberg FDR correction

---

## 11. Paradigm Shift Detection and Dynamics Qualification

### 11.1 Motivation and Theoretical Framework

The cascade detection pipeline (Sections 5–9) identifies frame-specific surges in coverage, detects real-world event occurrences, and quantifies cascade characteristics. However, cascades are *mechanisms* — they describe how coverage amplifies. The paradigm shift layer answers the higher-order question: **do cascades actually change the dominant framing of climate change?**

This completes the causal chain:

```
Events → Cascades → Paradigm Shifts → Durable or Ephemeral Change
       (Section 9)  (Section 11)       (Section 11.6)
```

A **paradigm**, in this context, is the composition of dominant frames at a given time. For instance, if Political and Economic frames jointly dominate for several months, the paradigm is "Dual-paradigm (Pol, Eco)." A **paradigm shift** occurs when this composition changes — a new frame enters dominance, an existing frame exits, or the entire composition is replaced.

The key insight motivating the two-level qualification system is that not all shifts are equal: some produce lasting change while others are ephemeral oscillations that quickly reverse. The framework characterizes both levels without applying arbitrary filters.

### 11.2 Paradigm State Computation

**Source**: `cascade_detector/analysis/paradigm_shift.py` — `ParadigmStateComputer`

**External dependency**: The paradigm computation uses the [CCF-paradigm](https://github.com/antoinelemor/CCF-paradigm) library's `ParadigmDominanceAnalyzer`, which implements a 4-method consensus for determining dominant frames:

1. **Information theory**: Entropy-based dominance scoring
2. **Network analysis**: Frame co-occurrence network centrality
3. **Causality analysis**: Granger causality between frame time series
4. **Proportional analysis**: Frame proportion relative to total

A frame is considered dominant only if a majority of methods agree.

#### Sliding window with daily resolution

The analyzer uses a 12-week (84-day) window for statistical robustness — this is the minimum required for reliable results from all 4 methods (particularly Granger causality, which needs sufficient temporal samples). However, the window advances by **1 day** (not 1 week) for maximum temporal resolution.

**Interpolation procedure**:

1. Weekly frame proportions $p_{f,w}$ (from the temporal index) are interpolated to daily resolution using time-weighted linear interpolation
2. At each window position $t$, the daily data within $[t - 84\text{d}, t]$ is resampled back to weekly means
3. The resampled weekly data feeds the `ParadigmDominanceAnalyzer`

This produces one `ParadigmState` per day of the analysis period (after the initial 12-week warm-up). For a single year (e.g., 2018), this yields approximately 281 states.

#### Parallel execution

Each window is independent and can be analyzed in a separate process. The module-level function `_analyze_window_worker()` receives serializable arguments (numpy arrays, ISO timestamp strings, frame names) and returns a plain dict — this avoids pickling issues with class instances.

```python
# Worker receives: (window_values, window_index, date, window_start, window_end, frame_names)
# Worker returns:  dict with dominant_frames, paradigm_type, paradigm_vector, frame_scores, ...
```

`ProcessPoolExecutor` distributes windows across `n_workers` (default: `os.cpu_count()`). On a 16-core M4 Max, 281 windows complete in approximately 6 minutes.

#### ParadigmState data structure

Each state captures the full paradigm composition at time $t$:

| Field | Type | Description |
|-------|------|-------------|
| `date` | Timestamp | Window end date |
| `window_start`, `window_end` | Timestamp | Window boundaries |
| `dominant_frames` | List[str] | Frames classified as dominant by consensus |
| `paradigm_type` | str | `Mono-paradigm`, `Dual-paradigm`, `Tri-paradigm`, `Poly-paradigm` |
| `paradigm_vector` | ndarray(8) | Dominance score per frame (continuous, [0, 1]) |
| `frame_scores` | Dict[str, float] | Named dominance scores |
| `concentration` | float | Sum of dominant frame proportions |
| `coherence` | float | Mean pairwise correlation among dominant frames |

### 11.3 Shift Detection

**Source**: `ShiftDetector`

#### State comparison

Two consecutive states are compared on:

1. **Set change**: Does the set of dominant frames differ? (entering/exiting frames computed via set difference)
2. **Vector distance**: Cosine distance between paradigm vectors exceeds threshold ($\tau = 0.3$)

A shift is registered if either condition holds.

#### Shift magnitude

$$
M = 0.40 \cdot J(S_\text{before}, S_\text{after}) + 0.40 \cdot \min(d_\text{cos}, 1) + 0.20 \cdot \min(|\Delta C|, 1)
$$

Where:
- $J(A, B) = 1 - \frac{|A \cap B|}{|A \cup B|}$ is the Jaccard distance between dominant frame sets
- $d_\text{cos}$ is the cosine distance between paradigm vectors
- $\Delta C$ is the change in concentration

#### Shift type classification

| Type | Condition |
|------|-----------|
| `frame_entry` | New frame(s) join the dominant set, none leave |
| `frame_exit` | Frame(s) leave the dominant set, none join |
| `recomposition` | Both entering and exiting frames (or vector change without set change) |
| `full_replacement` | No overlap between before and after dominant sets |

#### Direction-aware merging

Consecutive shifts within `merge_window_weeks` (default: 2 weeks) are merged if and only if they go in the **same direction**. Two shifts $s_i$ and $s_j$ are considered same-direction when neither reverses the other:

$$
\text{reversal} = (\text{entering}_i \cap \text{exiting}_j \neq \emptyset) \lor (\text{exiting}_i \cap \text{entering}_j \neq \emptyset)
$$

If a frame that entered in $s_i$ exits in $s_j$ (or vice versa), the shifts represent distinct transitions (e.g., Dual→Triple→Dual) and are kept separate. This preserves the granularity of transient states rather than collapsing them into a single artificial shift.

When merging, the higher-magnitude shift is retained with combined entering/exiting frame sets and the `state_before` of the first shift.

### 11.4 Shift-Level Dynamics Qualification

**Source**: `ShiftDetector.qualify_shifts()`

After detection, each shift is qualified with three continuous metrics:

#### Regime duration

$$
d_i = \begin{cases}
t_{i+1} - t_i & \text{if } i < n \\
t_\text{end} - t_i & \text{if } i = n
\end{cases}
$$

Where $t_i$ is the shift date and $t_\text{end}$ is the analysis period end. This measures how long the post-shift paradigm persists before the next transition — a continuous proxy for stability without arbitrary categorization.

#### Structural change

$$
\Delta S_i = |D_\text{after}| - |D_\text{before}|
$$

Where $D$ is the set of dominant frames. Positive values indicate complexification (more frames enter dominance), negative values indicate simplification. A Mono→Dual transition gives $\Delta S = +1$; a Triple→Mono gives $\Delta S = -2$.

#### Local reversibility

$$
R_i = \begin{cases}
\text{True} & \text{if } D_\text{after}^{(i+1)} = D_\text{before}^{(i)} \\
\text{False} & \text{otherwise}
\end{cases}
$$

A shift is locally reversible if the *next* shift restores the state that existed before this shift. This is a local measure — it captures micro-oscillations but does not assess whether the paradigm returns to its original state over a longer period.

### 11.5 Episode Construction and Qualification

**Source**: `EpisodeAnalyzer`

#### Episode grouping

Shifts are grouped into episodes using a gap threshold:

$$
\text{group}(s_i, s_j) \iff |t_j - t_i| \leq \tau_\text{gap}
$$

Where $\tau_\text{gap} = 3$ weeks (default). This produces clusters of temporally adjacent shifts separated by stable periods. A single isolated shift forms a 1-shift episode.

#### Episode-level metrics

Each episode captures the **net effect** of all its constituent shifts:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `dominant_frames_before` | $D_\text{before}^{(s_1)}$ | Paradigm at episode onset |
| `dominant_frames_after` | $D_\text{after}^{(s_n)}$ | Paradigm at episode end |
| `reversible` | $D_\text{before}^{(s_1)} = D_\text{after}^{(s_n)}$ | Did the paradigm return to its original state? |
| `net_structural_change` | $|D_\text{after}^{(s_n)}| - |D_\text{before}^{(s_1)}|$ | Net change in paradigm complexity |
| `max_complexity` | $\max_i |D_\text{after}^{(s_i)}|$ | Peak number of dominant frames during turbulence |
| `duration_days` | $t_{s_n} - t_{s_1}$ | Episode duration |
| `n_shifts` | $n$ | Number of transitions in the episode |
| `regime_after_duration_days` | $t_{s_1}^{(\text{next episode})} - t_{s_n}$ | Stability after the episode ends |

The two-level system (shift-level + episode-level) resolves a key scope issue: a shift may appear "irreversible" locally (the next shift doesn't restore it) while being part of an episode that is globally reversible (the paradigm returns to its original composition). For example, in a chain Dual(Pol,Eco) → Triple(Pol,Eco,Envt) → Dual(Pol,Envt) → Dual(Pol,Eco), the middle shifts are locally irreversible but the episode is globally reversible.

### 11.6 Cascade-Shift Attribution — Three-Role Model

**Source**: `CascadeShiftAttributor`

For each paradigm shift, the attributor identifies which cascades may have driven the transition and assigns each one a **discursive role** based on measured impact.

#### Theoretical grounding

Empirical analysis reveals that cascade strength alone does not predict paradigm impact (Spearman $\rho \approx 0$ between `total_score` and measured dominance lift). Most cascades affect *other* frames rather than their own. The three-role model captures this by distinguishing three discursive functions:

| Role | Name | Definition | Theoretical basis |
|------|------|-----------|-------------------|
| R1 | **Amplification** | Cascade promotes its own frame toward dominance | Information cascade theory (Bikhchandani et al.) |
| R2 | **Déstabilisation** | Cascade disrupts paradigm structure without its own frame benefiting | Focusing events (Birkland, 1998) |
| R3 | **Dormante** | Cascade is active but without measurable structural consequence | Dormant issues (Hilgartner & Bosk, 1988) |

Only R1 (amplification) and R2 (déstabilisation) are considered **drivers** of paradigm shifts. R3 (dormante) cascades are documented but not causally linked to shifts.

#### Filtering criteria

A cascade is considered for attribution if:

1. **Temporal proximity**: The cascade overlaps with or precedes the shift within a lookback window ($\tau_\text{lookback} = 12$ weeks)
2. **Strength threshold**: The cascade's `total_score` ≥ 0.40 (moderate or strong)

#### Impact measurement

Two metrics are computed for each cascade using the paradigm timeline:

**Weighted dominance lift** (`own_lift`): Measures whether the cascade promotes its own frame. For each day $t$ from cascade onset to onset + $T_\text{decay}$ (default 42 days):

$$
\text{own\_lift} = \frac{\sum_{t=0}^{T} w(t) \cdot [\text{dom}(t) - \bar{\text{dom}}_\text{baseline}]}{\sum_{t=0}^{T} w(t)}
$$

Where $w(t) = \max(0, 1 - t/T_\text{decay})$ is a linear decay weight and $\bar{\text{dom}}_\text{baseline}$ is the mean dominance in the 14 days before cascade onset. The continuous decay avoids arbitrary cutoff effects.

**Structural impact** (`structural_impact`): Cosine distance between the pre-cascade paradigm vector (mean of 14 days before onset) and the decay-weighted post-cascade paradigm vector:

$$
\text{structural\_impact} = 1 - \cos(\vec{v}_\text{pre}, \vec{v}_\text{post})
$$

This measures paradigm disruption regardless of which specific frame moved.

#### Role assignment

1. **Amplification** if `own_lift > lift_threshold` (provisionally 0.05)
2. **Déstabilisation** if `structural_impact > structural_threshold` (provisionally 0.01) AND `own_lift ≤ lift_threshold`
3. **Dormante** otherwise

#### Attribution scoring (role-specific)

**Amplification**:
$$
A_\text{amp} = 0.30 \cdot T + 0.35 \cdot \hat{L} + 0.20 \cdot S + 0.15 \cdot D
$$
Where $\hat{L} = \min(|\text{own\_lift}| / 0.5, 1)$, and $D$ is the **direction alignment** score distinguishing three amplification mechanisms:

| $D$ | Condition | Mechanism |
|-----|-----------|-----------|
| 1.0 | Frame NOT dominant before cascade, ENTERS dominant set during cascade (+7 days) | **Promotion** — cascade pushes its frame into dominance |
| 0.7 | Frame ALREADY dominant before cascade, REMAINS dominant during cascade | **Consolidation** — cascade maintains an existing dominant frame |
| 0.3 | Frame NEVER reaches the dominant set during the causal window | **Insufficient** — positive lift but too weak for dominance |

The causal window for dominance assessment is `onset → end + 7 days` (short extension for media inertia). This is intentionally shorter than the 42-day lift decay horizon because dominance entry is a discrete event where causal attribution weakens rapidly with temporal distance.

**Déstabilisation**:
$$
A_\text{dest} = 0.30 \cdot T + 0.35 \cdot \hat{I} + 0.20 \cdot S + 0.15 \cdot \hat{C}
$$
Where $\hat{I} = \min(\text{structural\_impact} / 0.1, 1)$ and $\hat{C} = \min(|\Delta\text{concentration}| / 0.3, 1)$.

**Dormante**:
$$
A_\text{dorm} = 0.50 \cdot T + 0.50 \cdot S
$$
Simple co-occurrence score — no causal claim.

The temporal overlap score $T$ combines overlap fraction and recency:

$$
T = 0.5 \cdot \frac{\text{overlap\_days}}{\text{cascade\_duration}} + 0.5 \cdot \left(1 - \frac{\text{days\_before\_shift}}{\tau_\text{lookback}}\right)
$$

#### Output and sorting

Attributed cascades are sorted by role priority (amplification > déstabilisation > dormante), then by attribution score within each role. Each attribution includes: `role`, `attribution_score`, `own_lift`, `structural_impact`, plus role-specific fields (`direction_alignment` for amplification, `concentration_disruption` for déstabilisation).

**Event aggregation**: Only driver cascades (amplification + déstabilisation) contribute their `dominant_events` to the shift's `attributed_events`. Dormante cascades are documented but do not influence event aggregation.

#### Threshold calibration

The `lift_threshold` and `structural_threshold` are provisionally calibrated on 2018 data. After running the full corpus (1978–2024), these will be refined using the 75th percentile of the respective distributions across all years.

### 11.7 Data Flow Summary

```
Weekly frame proportions (from temporal index)
    │
    ├─ Interpolate to daily → resample per window
    │
    ▼
ParadigmStateComputer (12-week window, 1-day step, parallel)
    │  → 281 ParadigmState objects per year
    │
    ▼
ShiftDetector
    │  → Compare consecutive states
    │  → Direction-aware merge
    │  → 57 ParadigmShift objects (2018)
    │
    ▼
ShiftDetector.qualify_shifts()
    │  → regime_duration_days, structural_change, reversible
    │
    ▼
CascadeShiftAttributor (three-role model)
    │  → Compute own_lift + structural_impact per cascade
    │  → Assign role: amplification / déstabilisation / dormante
    │  → Aggregate events from drivers only (R1 + R2)
    │
    ▼
EpisodeAnalyzer
    │  → Group shifts by gap < 3 weeks
    │  → Compute episode-level dynamics
    │  → 5 ShiftEpisode objects (2018)
    │
    ▼
ParadigmShiftResults
    ├── shifts: List[ParadigmShift]    (with dynamics + attribution)
    ├── episodes: List[ShiftEpisode]    (with two-level qualification)
    └── paradigm_timeline: DataFrame    (281 daily paradigm states)
```

---

## 12. Production Pipeline

### 12.1 Production Script

**Source**: `scripts/run/run_production.py`

The production script runs cascade detection across all years (1978–2024) with fault tolerance:

```bash
python scripts/run/run_production.py                    # All years
python scripts/run/run_production.py --year 2018        # Single year
python scripts/run/run_production.py --start 2000 --end 2010  # Range
python scripts/run/run_production.py --resume           # Skip completed years
python scripts/run/run_production.py --skip-embeddings  # Skip embedding check
```

**Key design decisions**:

1. **Single pipeline initialization**: The pipeline is initialized once, and the embedding memmap is shared across all years. This avoids the overhead of reloading the ~18 GB embedding file for each year.

2. **Per-year isolation**: Each year is processed in a try/except block with full traceback logging. A failure on one year does not abort the entire run.

3. **Resume mode**: Checks for existing `year_metadata.json` files and skips completed years.

4. **Cascade counter reset**: `pipeline.detector._cascade_counter = 0` before each year ensures clean per-year cascade IDs.

5. **Embedding validation**: `ensure_embeddings()` verifies that pre-computed embeddings exist and cover ≥99% of database sentences before starting detection.

### 12.2 Output Structure

```
results/production/
├── run_manifest.json              # Run metadata
├── cross_year_cascades.parquet    # All cascades across all years
├── cross_year_cascades.csv        # Same in CSV
├── cross_year_summary.json        # Aggregate statistics
└── {year}/
    ├── cascades.json              # Full cascade objects
    ├── cascades.parquet           # Cascade DataFrame
    ├── bursts.parquet             # BurstResult objects
    ├── year_metadata.json         # Year-level summary
    ├── time_series/
    │   ├── daily_composite.parquet        # (cascade_id, date, value)
    │   ├── daily_signals.parquet          # (cascade_id, signal, date, value)
    │   ├── daily_articles.parquet
    │   ├── daily_journalists.parquet
    │   └── cumulative_journalists.parquet
    ├── networks/
    │   ├── edge_lists.csv         # (cascade_id, source_journalist, source_media,
    │   │                          #  target_journalist, target_media, weight)
    │   └── network_metrics.json
    ├── signals/
    │   └── frame_signals.parquet  # (frame, signal, date, value)
    ├── indices/
    │   ├── temporal_daily_proportions.parquet
    │   ├── temporal_daily_series.parquet
    │   ├── temporal_statistics.json
    │   ├── frame_cooccurrence.parquet
    │   ├── frame_statistics.json
    │   ├── emotion_statistics.json
    │   ├── emotion_temporal.parquet
    │   ├── source_metadata.json
    │   └── geographic_summary.json
    ├── convergence/
    │   ├── semantic_convergence_full.json
    │   └── syndication_stats.json
    ├── impact_analysis/
    │   ├── cluster_cascade.parquet     # StabSel: EventCluster → Cascade roles (driver/suppressor/neutral)
    │   ├── stabsel_results.pkl         # Full StabSelCascadeResult objects for figures/resume
    │   └── summary.json               # Per-frame role distributions and R² statistics
    └── paradigm_shifts/
        ├── shifts.json                     # All paradigm shifts with dynamics + attribution
        ├── episodes.json                   # Shift episodes with two-level qualification
        └── paradigm_timeline.parquet       # Daily paradigm states (dominance scores)
```

**Serialization strategy**:
- **Time series** (pd.Series): Converted to long-format DataFrames `(cascade_id, date, value)` for Parquet storage
- **Network edges**: Decomposed from `(journalist, media)` tuples into flat CSV columns
- **Indices**: Serializable parts extracted (JSON for statistics, Parquet for time series/matrices)
- **JSON types**: All numpy types converted via `_jsonify()` + `default=str`

---

## 13. Configuration Reference

### 13.1 DetectorConfig

All parameters are set via the `DetectorConfig` dataclass (`core/config.py`). Environment variables override defaults.

#### Database

| Parameter | Default | Env Variable |
|-----------|---------|-------------|
| `db_host` | `"localhost"` | `DB_HOST` |
| `db_port` | `5432` | `DB_PORT` |
| `db_name` | `"CCF_Database_texts"` | `DB_NAME` |
| `db_user` | `"antoine"` | `DB_USER` |
| `db_password` | `""` | `DB_PASSWORD` |
| `db_table` | `"CCF_processed_data"` | — |
| `embedding_dir` | `"data/embeddings"` | `EMBEDDING_DIR` |
| `embedding_dim` | `1024` | — |

#### Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `zscore_threshold` | 2.0 | Z-score threshold for initial flagging |
| `composite_threshold` | 2.0 | Threshold on composite signal for burst detection |
| `min_burst_days` | 3 | Minimum consecutive anomaly days for Z-score method |
| `baseline_window_days` | 90 | Trailing baseline window for rolling Z-scores |
| `burst_merge_gap_days` | 3 | Maximum gap (days) between periods to merge them |
| `onset_lookback_days` | 14 | Maximum backward extension of burst onset |
| `min_articles` | 10 | Minimum articles for cascade consideration |
| `min_journalists` | 3 | Minimum journalists |
| `min_media` | 2 | Minimum media outlets |

#### Scoring Dimension Weights (sum = 1.00)

| Dimension | Weight | Config Field |
|-----------|--------|-------------|
| Temporal | 0.25 | `weight_temporal` |
| Participation | 0.25 | `weight_participation` |
| Convergence | 0.25 | `weight_convergence` |
| Source | 0.25 | `weight_source` |

#### Composite Signal Weights (sum = 1.00)

| Signal | Weight | Config Field |
|--------|--------|-------------|
| Temporal | 0.25 | `signal_weight_temporal` |
| Participation | 0.20 | `signal_weight_participation` |
| Convergence | 0.20 | `signal_weight_convergence` |
| Source | 0.15 | `signal_weight_source` |
| Semantic | 0.20 | `signal_weight_semantic` |

#### Normalization Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `MAX_BURST_INTENSITY` | 5.0 | 5× baseline = exceptional burst |
| `MAX_ADOPTION_VELOCITY` | 3.0 | 3 new journalists/day = very fast adoption |
| `MAX_DURATION_DAYS` | 30.0 | Cascades rarely exceed 30 days |
| `MAX_MEDIA` | 20.0 | Canadian media landscape ~20 major outlets |
| `MIN_ARTICLES_HARD` | 3 | Hard filter: below this, score = 0 |

#### CUSUM Parameters (Computed at Runtime)

| Parameter | Formula | Meaning |
|-----------|---------|---------|
| $h$ | $4.0 \times \sigma_C$ | Decision threshold |
| $k$ | $0.5 \times \sigma_C$ | Slack variable |

### 13.2 Weight Validation

`DetectorConfig.validate()` asserts:
1. Scoring dimension weights sum to 1.0 (± 0.01)
2. Composite signal weights sum to 1.0 (± 0.01)
3. Z-score threshold > 0
4. Minimum burst days ≥ 1
5. Baseline window ≥ 30 days

---

## 14. Testing Strategy

### 14.1 Unit Tests (298 tests, 293 fast + 5 slow)

**`tests/test_signal_builder.py`** — 14 tests covering:
- Signal builder API (returns all keys, handles missing frames, insufficient data)
- Signal alignment (same DatetimeIndex, all non-negative)
- Composite weighting (weighted sum matches manual computation)
- Convergence orthogonalization (residual ≤ original, zero reference → unchanged)
- Flat series produces low composite; burst in one dimension raises composite
- Multi-signal bursts produce higher composite than single-signal
- Semantic signal present and non-negative; zero when doc_ids absent
- Rolling Z-score clipped at zero; spikes produce high Z

**`tests/test_unified_detector.py`** — 23 tests covering:
- API contracts (detect returns tuple, detect_all_frames iterates, requires embedding store)
- Burst detection (detected in synthetic data, valid fields, no detection in flat series)
- Cascade scoring (scores in [0, 1], sub-indices in [0, 1], total = weighted sum of dimensions, classification assigned)
- Semantic peak (embedding-weighted P50 replaces Z-score argmax for peak_date)
- New fields (composite_peak, daily_composite, daily_signals, detection_method)
- Sub-index count per dimension: temporal = 4, participation = 6, convergence = 4, source = 3
- Hard filter (< 3 articles → zero score)
- Weight validation (scoring weights sum to 1.0, signal weights sum to 1.0)
- Serialization (to_dict produces JSON-serializable output)

**`tests/test_event_impact.py`** — 9 tests covering:
- Prevalence ratio with known elevated counts yields PR > 2.0
- Fisher's exact test yields p < 0.05 for strongly elevated event
- Pre-onset surge detects elevated pre-cascade activity (median surge > 1.0)
- Strength correlation direction (positive rho for positively correlated event-score pairs)
- BH FDR correction: adjusted p-values ≥ raw p-values
- BH monotonicity: adjusted p-values non-decreasing when sorted by raw
- Empty cascades produce empty DataFrames without error
- Zero-occurrence event handled gracefully (rate = 0.0)
- `run()` returns all 4 expected DataFrames with correct column schemas

**`tests/test_unified_impact.py`** — 68 tests covering:
- Diff-in-diff: positive/zero/empty cases, signed output, pre/post window handling
- Cross-correlation (dose-response): positive/negative lag detection, short series fallback, zero-variance handling
- Granger causality: significant/insignificant detection, short series fallback, statsmodels dependency
- Temporal proximity: Gaussian decay, zero distance, large distance asymptotic behavior
- DID normalization: tanh scaling, zero/large input behavior
- Daily mass construction: belonging aggregation across occurrences, max per doc_id
- Dominance series extraction: correct column selection, empty timeline handling
- Phase 1 (cluster→cascade): impact score computation, article overlap weighting, proximity filtering, role assignment (driver/late_support/suppressor/neutral/unrelated)
- Phase 2 (cluster→dominance): Granger integration, role assignment (catalyst/disruptor/inert)
- Phase 3 (cascade→dominance): own-frame detection, role assignment (amplification/destabilisation/dormant), is_own_frame flag
- Impact classification: strong/moderate/weak/negligible thresholds
- Late support reclassification: post-peak suppressor → late_support (frame affinity threshold, embedding alignment check, NaN alignment passthrough, no-driver cascade, empty DataFrame)
- Cascade-level role aggregation: dominant role derivation from Phase 3
- Integration: `run()` from DetectionResults, `run_from_components()`, empty input handling
- Summary enrichment: role counts, label counts, cascade_roles in summary dict

**`tests/test_event_occurrence.py`** — 132 tests covering:
- Phase 1: seed selection (composite scoring, dominant ratio filter, percentile threshold, fallback)
- Phase 2: per-type HAC clustering (3D distance, title embedding blend, minimum cluster size)
- Phase 3: multi-type merge (deduplication, 5D distance, silhouette cut, type structure analysis)
- Phase 4: soft assignment (4D distance, self-adjusting threshold, belonging normalization, iterative refinement)
- Phase 5: confidence scoring (5 components, centroid tightness, coherence residual, media diversity, recruitment, size adequacy)
- Post-processing: seed-overlap connectivity (Step 6b), seed inclusion (Step 6c), fragmentation safety net (Step 6d)
- EventCluster strength scoring (5 components: mass, coverage, intensity, coherence, diversity)
- Article-level temporal bounds (composite-weighted P10/P50/P90)
- Attribution: temporal overlap, shared articles, overlap ratio
- Integration: `detect_events()`, `attribute_to_cascades()`, `detect()` and `detect_all()` wrappers

**`tests/test_paradigm_shift.py`** — 48 tests covering:
- ParadigmStateComputer: stable data → consistent dominant frames, known dominant correctly identified, single-window fallback, paradigm vector length, serialization
- ShiftDetector: no shift on stable paradigm, shift on frame change, shift type classification (entry, exit, full_replacement), magnitude proportional to change, nearby shift merging, distant shifts preserved, reversal not merged, serialization
- ShiftDetector.qualify_shifts: regime duration between shifts, structural change (+/-), reversible/irreversible shift detection
- CascadeShiftAttributor (three-role model): overlapping cascade → amplification role, distant cascade excluded, weak cascade filtered, events from drivers only, amplification role assignment, déstabilisation role assignment, dormante role assignment, dormante cascades excluded from events, linear decay weighting verification, no-timeline fallback (all dormante), role ordering (amplification > déstabilisation > dormante), direction alignment: promotion (frame enters dominance), consolidation (frame already dominant), insufficient (frame never dominant)
- EpisodeAnalyzer: single episode from close shifts, two episodes from distant shifts, episode reversibility/irreversibility, net structural change, max complexity, regime after duration, serialization, empty input
- ParadigmShiftAnalyzer: end-to-end with no cascades, end-to-end with cascades, results serialization (incl. episodes), timeline frame columns, detection from mock DetectionResults

**`tests/conftest.py`** provides `MockEmbeddingStore` and `ClusterableMockEmbeddingStore` with deterministic 64-dimensional embeddings seeded by doc_id hash for reproducible unit tests without requiring GPU or embedding files.

### 14.2 Key Invariants Tested

1. Every burst produces a CascadeResult (no filtering gate — every burst is scored)
2. All dimension scores ∈ [0, 1]
3. All sub-indices ∈ [0, 1]
4. `total_score = Σ(weight_i × score_i)` verified numerically
5. Convergence sub-indices always use embedding-based keys
6. `z_semantic` present in `daily_signals` for cascades with articles
7. Score distribution is not degenerate (range > 0.05 when ≥ 3 scored cascades)
8. Detection method contains `"composite"`
9. Dimension weights and signal weights each sum to 1.00

---

## Appendix A: Summary Table of All Sub-Indices

| # | Dimension | Sub-Index Key | Raw Input | Computation | Normalization | Embedding? |
|---|-----------|---------------|-----------|-------------|---------------|------------|
| 1 | Temporal | `burst_intensity` | Peak proportion / baseline mean | Ratio | $\min(1, x/5)$ | No |
| 2 | Temporal | `adoption_velocity` | Cumulative journalist growth / days | Slope | $\min(1, x/3)$ | No |
| 3 | Temporal | `duration` | End − onset + 1 | Count (days) | $\min(1, x/30)$ | No |
| 4 | Temporal | `mann_whitney` | Daily proportions: burst vs baseline | Mann-Whitney U (greater) | Gated by p-value | No |
| 5 | Participation | `actor_diversity` | Journalist article counts (top 10) | $H / \ln n$ (Shannon) | $\min(1, x)$ | No |
| 6 | Participation | `cross_media_ratio` | Unique media outlet count | Count | $\min(1, x/20)$ | No |
| 7 | Participation | `new_entrant_rate` | New vs total journalists | Ratio | $\min(1, x)$ | No |
| 8 | Participation | `growth_pattern` | Cumulative journalist time series | Fraction of growth days | $\text{clip}(x, 0, 1)$ | No |
| 9 | Participation | `network_structure` | Co-coverage graph | $\bar{C}/\rho$ (clustering/density ratio) | $\min(1, x/20)$ | No |
| 10 | Participation | `network_cohesion` | Louvain modularity | $1 - Q$ | $\text{clip}(x, 0, 1)$ | No |
| 11 | Convergence | `semantic_similarity` | All article embeddings | Mean pairwise cosine | $\text{clip}(x, 0, 1)$ | **Yes** |
| 12 | Convergence | `convergence_trend` | Temporal sub-window similarities | Linear regression slope | $(x + 0.05) / 0.10$ | **Yes** |
| 13 | Convergence | `cross_media_alignment` | Outlet centroid embeddings | Mean pairwise cosine | $\text{clip}(x, 0, 1)$ | **Yes** |
| 14 | Convergence | `novelty_decay` | Incremental centroid distance | $-\text{slope}(\text{novelty}) \times n$ | $x / 0.5$ | **Yes** |
| 15 | Source | `source_diversity_decline` | 1st/2nd half messenger entropy | $(H_1 - H_2) / H_1 \times \text{conf}$ | $\text{clip}(x, 0, 1)$ | No |
| 16 | Source | `messenger_concentration` | All messenger counts | $1 - H_{\text{norm}}$ | $\text{clip}(x, 0, 1)$ | No |
| 17 | Source | `media_coordination` | Journalist centroid embeddings | Mean pairwise cosine | $\text{clip}(x, 0, 1)$ | **Yes** |

**Total embedding-dependent sub-indices**: 5 out of 17 (indices 11–14 plus 17).

**Effective embedding weight contribution**: Convergence dimension (0.25, 4/4 sub-indices embedding-based) + 1/3 of source dimension (0.25/3 ≈ 0.083) = **~0.333 of 1.0, or ~31%** of the total score (before media confidence factor).

---

## Appendix B: Embedding Coverage Logging

The pipeline logs warnings at critical integration points when embedding coverage is insufficient:

| Component | Condition | Level | Message |
|-----------|-----------|-------|---------|
| `EmbeddingStore.get_batch_article_embeddings()` | < 50% of doc_ids found (batch ≥ 5) | WARNING | Coverage fraction and counts |
| `DailySignalBuilder._compute_semantic_z()` | 0 days produced similarity > 0 | WARNING | Embeddings may be missing |
| `UnifiedCascadeDetector._score_convergence()` | < 2 embeddings found for cascade | WARNING | Returning zero convergence |
| `UnifiedCascadeDetector._media_coordination()` | Journalists skipped (no embeddings) | INFO | Skip count / total |

These logs enable researchers to assess whether embedding-based scores are reliable for a given analysis period.

---

## Appendix C: Network Construction Detail

### Graph Definition

The co-coverage network is an undirected weighted graph:

- **Nodes**: `(journalist, media)` tuples — the same journalist at different outlets produces separate nodes
- **Edges**: Same-day co-coverage of the cascade frame; weight = number of co-occurrence days
- **Construction**: O(actors²) per day, O(days × actors²) total

### NetworKit Subprocess Protocol

```
Main process                          Subprocess (networkit_worker.py)
     │                                      │
     ├─ pickle.dumps({edges, nodes}) ──→ stdin
     │                                      ├─ Build nk.Graph
     │                                      ├─ PLM community detection
     │                                      ├─ DegreeCentrality
     │                                      ├─ ConnectedComponents
     │                                      ├─ Density computation
     │                                      │
     │                                 stdout ←─ pickle.dumps({density, modularity,
     │                                           mean_degree, n_components})
```

**Thread configuration in subprocess**: `min(4, cpu_count // 2)` threads to prevent oversubscription when multiple cascades are processed.

**Fallback**: If the subprocess fails (timeout, crash), pure NetworkX is used as fallback with `community_louvain.best_partition(random_state=42)`.

---

## Appendix D: Geographic Analysis Infrastructure

### Media Outlet to Province Mapping

21 Canadian media outlets are mapped to 10 provinces:

| Province | Outlets |
|----------|---------|
| National | Toronto Star, Globe and Mail, National Post |
| Quebec | Le Devoir, La Presse Plus, La Presse, Montreal Gazette, Journal de Montreal |
| Ontario | Le Droit, Toronto Sun |
| Alberta | Edmonton Journal, Calgary Herald |
| British Columbia | Vancouver Sun, Times Colonist |
| Manitoba | Winnipeg Free Press |
| Saskatchewan | Star Phoenix |
| Nova Scotia | Chronicle Herald |
| Newfoundland & Labrador | The Telegram |
| New Brunswick | Acadie Nouvelle |
| Yukon | Whitehorse Daily Star |

### Province Adjacency Network

Used for geographic coherence and proximity analysis:

```
Ontario — Quebec — New Brunswick — Nova Scotia — Prince Edward Island
   │         │                         │
Manitoba     Newfoundland & Labrador
   │
Saskatchewan — Alberta — British Columbia — Yukon — Northwest Territories — Nunavut
```

### Linguistic Barrier Analysis

The framework tracks whether cascades cross the English-French linguistic barrier:

- **Francophone outlets**: Le Devoir, La Presse, La Presse Plus, Journal de Montreal, Le Droit, Acadie Nouvelle
- **Anglophone outlets**: All others
- **Linguistic permeability**: $\min(p_{\text{franco}}, p_{\text{anglo}}) \times 2 \in [0, 1]$ (maximum at 50/50 split)
- **Barrier crossing**: Boolean — both language groups represented in cascade coverage

---

## Appendix E: Entity Resolution Algorithms

### FastEntityResolver

**Purpose**: Merge name variations (e.g., "Trudeau", "Justin Trudeau", "J. Trudeau") into canonical entities.

**Algorithm**: Blocking + similarity scoring + graph-based clustering

**Blocking strategy** (reduces O(n²) to O(n·k)):
- Last name blocking: `lastname:{last_word}`
- First-last combination: `fl:{first[0]}:{last_word}`
- Soundex phonetic: `sound:{phonetic_key}`
- Maximum block size: 100 entities (larger blocks skipped)

**Similarity scoring**:
- Substring containment: 0.95
- Initial match ("J. Trudeau" vs "Justin Trudeau"): 0.92
- Last name only with occurrence-based confidence:
  - ≥ 500 occurrences: 0.95
  - ≥ 100 occurrences: 0.92
  - ≥ 30 occurrences + 100 total: 0.88
- Co-occurrence boost (same article): 0.95–0.98

**Safety checks**:
- Must share at least one significant word (excludes stopwords: "the", "of", "de", "du", etc.)
- Different first names prevent merge ("Doug Ford" ≠ "Rob Ford")
- Hardcoded rules for prominent Canadian political figures

**Clustering**: NetworkX connected components with transitive-merge safety (zero-similarity pairs break clusters).

### FastLocationResolver

**Purpose**: Deduplicate geographic entities (e.g., "Fort McMurray" = "Fort Mac").

**Algorithm**: Ultra-fast string similarity without SequenceMatcher:
- Exact match: 1.0
- Length ratio < 0.5: immediate reject (0.0)
- Substring containment: 0.9
- Token-based Jaccard overlap (> 0.8 → 0.95, > 0.5 → 0.85)
- Threshold: 0.85

### AuthorResolver

**Purpose**: Merge journalist name variations across articles.

**Algorithm**: Blocking by surname prefix + context-aware scoring:
- Name similarity: exact (1.0), subset (0.95), initial match (0.92)
- Context similarity (weighted): 0.3 × temporal + 0.4 × topic + 0.3 × messenger pattern
- Suffix removal: Reuters, AP, Staff, Reporter, bureau locations, titles (Mr., Dr., Prof.)

---

## Appendix F: StabSel Impact Analysis Output Schema

### F.1 Cluster → Cascade DataFrame (`cluster_cascade.parquet`)

One row per (cluster, cascade) pair where the cluster was selected by stability selection or has a significant OLS coefficient.

| Column | Type | Description |
|--------|------|-------------|
| `cluster_id` | str | EventCluster identifier |
| `cascade_id` | str | Cascade identifier |
| `cascade_frame` | str | Cascade frame abbreviation |
| `role` | str | `driver` (β > 0, p < α), `suppressor` (β < 0, p < α), or `neutral` (p ≥ α) |
| `net_beta` | float | OLS coefficient from post-selection regression |
| `p_value` | float | Two-sided p-value from residual bootstrap (B = 500) |
| `selection_freq` | float | Stability selection frequency π ∈ [0, 1] (selected if π ≥ 0.60) |
| `D_sum` | float | Sum of double-weighted treatment variable across all lags |
| `lag_profile` | str | JSON-encoded list of D_sum per lag (lags 0 to MAX_LAG) |
| `r2` | float | R² of the post-selection OLS model for this cascade |
| `dominant_type` | str | Cluster dominant event type |
| `event_types` | str | All event types in the cluster (comma-separated) |
| `entities` | str | Top entities in the cluster (comma-separated) |
| `strength` | float | EventCluster strength score |
| `n_articles_weighted` | float | Sum of belonging scores for cluster articles |
| `n_articles_total` | int | Total articles assigned to the cluster |
| `peak_date` | str (ISO) | Cluster peak date |
| `frame_profile` | str | JSON-encoded {frame: mean_value} for cluster articles |
| `messenger_profile` | str | JSON-encoded {messenger: mean_value} for cluster articles |
| `event_profile` | str | JSON-encoded {event_type: mean_value} for cluster articles |

### F.2 Pickle Results (`stabsel_results.pkl`)

Python pickle (protocol 4) containing `Dict[str, List[StabSelCascadeResult]]` keyed by frame abbreviation. Each `StabSelCascadeResult` stores the full analysis for one cascade:

| Field | Type | Description |
|-------|------|-------------|
| `cascade_id` | str | Cascade identifier |
| `r2` | float | R² of post-selection OLS |
| `n_selected` | int | Number of clusters selected by stability selection |
| `n_candidates` | int | Total candidate clusters (D_sum > MIN_D_SUM) |
| `roles` | List[ClusterRole] | Per-cluster results with net_beta, p_value, role, selection_freq, D_sum, lag_profile |
| `composite_series` | np.ndarray | Daily two-sided composite signal for the cascade frame |
| `event_dates` | Dict[str, date] | Cluster peak dates {cluster_id: date} |

Each `ClusterRole` contains: `cluster_id`, `net_beta`, `p_value`, `role`, `selection_freq`, `D_sum`, `lag_profile`.

### F.3 Summary (`summary.json`)

| Key | Type | Description |
|-----|------|-------------|
| `n_frames` | int | Number of frames analyzed (up to 8) |
| `per_frame` | Dict | Per-frame statistics keyed by frame abbreviation |
| `per_frame.{frame}.n_cascades` | int | Number of cascades analyzed |
| `per_frame.{frame}.n_drivers` | int | Clusters with role = `driver` |
| `per_frame.{frame}.n_suppressors` | int | Clusters with role = `suppressor` |
| `per_frame.{frame}.n_neutral` | int | Clusters with role = `neutral` |
| `per_frame.{frame}.median_r2` | float | Median R² across cascade models |
| `per_frame.{frame}.mean_r2` | float | Mean R² across cascade models |

---

## Appendix G: Paradigm Shift Output Schema

### G.1 Shifts (`shifts.json`)

| Field | Type | Description |
|-------|------|-------------|
| `shift_id` | str | Unique identifier (e.g., `shift_a1b2c3d4`) |
| `shift_date` | str (ISO) | Date of the paradigm transition |
| `shift_type` | str | `frame_entry`, `frame_exit`, `recomposition`, `full_replacement` |
| `entering_frames` | List[str] | Frames entering the dominant set |
| `exiting_frames` | List[str] | Frames leaving the dominant set |
| `shift_magnitude` | float | Composite magnitude [0, 1] |
| `vector_distance` | float | Cosine distance between paradigm vectors |
| `set_jaccard_distance` | float | Jaccard distance between dominant frame sets |
| `concentration_change` | float | Change in dominant frame concentration |
| `regime_duration_days` | int | Days until the next shift (or period end) |
| `structural_change` | int | Change in number of dominant frames (e.g., +1, -2) |
| `reversible` | bool | Does the next shift restore the previous state? |
| `state_before` | ParadigmState | Full paradigm state before the shift |
| `state_after` | ParadigmState | Full paradigm state after the shift |
| `attributed_cascades` | List[Dict] | Cascades attributed to this shift with three-role classification (amplification/déstabilisation/dormante), sorted by role priority then score |
| `attributed_events` | List[Dict] | Aggregated events from driver cascades only (amplification + déstabilisation) |

### G.2 Episodes (`episodes.json`)

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | str | Unique identifier (e.g., `episode_a1b2c3d4`) |
| `start_date` | str (ISO) | First shift date in the episode |
| `end_date` | str (ISO) | Last shift date in the episode |
| `duration_days` | int | Episode duration (end - start) |
| `n_shifts` | int | Number of shifts in the episode |
| `shift_ids` | List[str] | IDs of constituent shifts |
| `dominant_frames_before` | List[str] | Paradigm at episode onset |
| `dominant_frames_after` | List[str] | Paradigm at episode end |
| `reversible` | bool | Same paradigm before and after? |
| `net_structural_change` | int | len(after) − len(before) |
| `max_complexity` | int | Peak number of dominant frames during the episode |
| `regime_after_duration_days` | int | Days until the next episode (stability measure) |

### G.3 Paradigm Timeline (`paradigm_timeline.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `date` | Timestamp | Window end date |
| `dominant_frames` | str | Comma-separated dominant frame codes |
| `paradigm_type` | str | `Mono-paradigm`, `Dual-paradigm`, etc. |
| `concentration` | float | Sum of dominant frame proportions |
| `coherence` | float | Mean pairwise correlation among dominant frames |
| `paradigm_Cult` | float | Cultural frame dominance score |
| `paradigm_Eco` | float | Economic frame dominance score |
| `paradigm_Envt` | float | Environmental frame dominance score |
| `paradigm_Pbh` | float | Public Health frame dominance score |
| `paradigm_Just` | float | Justice frame dominance score |
| `paradigm_Pol` | float | Political frame dominance score |
| `paradigm_Sci` | float | Scientific frame dominance score |
| `paradigm_Secu` | float | Security frame dominance score |

---

## Appendix H: Event Occurrence Constants Reference

All constants are defined in `cascade_detector/core/constants.py`.

### H.1 Title Embeddings

| Constant | Value | Description |
|----------|-------|-------------|
| `TITLE_SENTENCE_ID` | 0 | Convention: sentence_id=0 stores article title embedding |
| `TITLE_WEIGHT` | 0.30 | Blend weight for title vs sentence embeddings |

### H.2 Phase 2: Per-Type Clustering

| Constant | Value | Description |
|----------|-------|-------------|
| `SEED_PERCENTILE` | 50 | Percentile threshold for seed selection |
| `SEED_WEIGHT_TYPE` | 0.6 | Weight of evt_*_mean in composite seed score |
| `SEED_WEIGHT_GLOBAL` | 0.4 | Weight of event_mean in composite seed score |
| `SEED_DOMINANT_RATIO` | 0.5 | Minimum ratio evt_X_mean / max(evt_*_mean) for seeding |
| `MIN_CLUSTER_SIZE` | 2 | Minimum articles per cluster |
| `PHASE2_SEMANTIC_WEIGHT` | 0.50 | Semantic distance weight |
| `PHASE2_TEMPORAL_WEIGHT` | 0.30 | Temporal distance weight |
| `PHASE2_ENTITY_WEIGHT` | 0.20 | Entity Jaccard distance weight |

### H.3 Phase 3: Event Cluster Merge

| Constant | Value | Description |
|----------|-------|-------------|
| `EVENT_CLUSTER_TEMPORAL_WEIGHT` | 0.25 | Temporal distance weight |
| `EVENT_CLUSTER_SEMANTIC_WEIGHT` | 0.20 | Semantic distance weight |
| `EVENT_CLUSTER_ENTITY_WEIGHT` | 0.15 | Entity distance weight |
| `EVENT_CLUSTER_ARTICLE_WEIGHT` | 0.30 | Article overlap (Jaccard on seeds) weight |
| `EVENT_CLUSTER_TYPE_WEIGHT` | 0.10 | Type distance weight (binary) |
| `EVENT_CLUSTER_TEMPORAL_SCALE` | 14.0 | Days for temporal decay |
| `EVENT_CLUSTER_MIN_ENTITY_CITATIONS` | 3 | Minimum citations per entity |

### H.4 Phase 4: Soft Assignment

| Constant | Value | Description |
|----------|-------|-------------|
| `PHASE4_TEMPORAL_WEIGHT` | 0.25 | Temporal distance weight |
| `PHASE4_SEMANTIC_WEIGHT` | 0.35 | Semantic distance weight |
| `PHASE4_ENTITY_WEIGHT` | 0.15 | Entity distance weight |
| `PHASE4_SIGNAL_WEIGHT` | 0.25 | Event signal distance weight |
| `PHASE4_N_ITERATIONS` | 2 | Number of assignment iterations |
| `PHASE4_TEMPORAL_SCALE` | 14.0 | Days for temporal decay |

### H.5 Strength Scoring

| Constant | Value | Description |
|----------|-------|-------------|
| `EVENT_CLUSTER_STRENGTH_MASS_WEIGHT` | 0.20 | Mass component weight |
| `EVENT_CLUSTER_STRENGTH_COVERAGE_WEIGHT` | 0.25 | Coverage component weight |
| `EVENT_CLUSTER_STRENGTH_INTENSITY_WEIGHT` | 0.20 | Intensity component weight |
| `EVENT_CLUSTER_STRENGTH_COHERENCE_WEIGHT` | 0.15 | Coherence component weight |
| `EVENT_CLUSTER_STRENGTH_DIVERSITY_WEIGHT` | 0.20 | Media diversity component weight |

### H.6 Post-Processing

| Constant | Value | Description |
|----------|-------|-------------|
| `EVENT_CLUSTER_TITLE_SIM_THRESHOLD` | 0.50 | Min cosine similarity for title connectivity |
| `EVENT_CLUSTER_MAX_GAP_DAYS` | 30 | Max peak gap for title-only connectivity |
| `SEED_INCLUSION_THRESHOLD` | 0.80 | Min seed subset fraction for absorption |
| `FRAGMENTATION_THRESHOLD` | 0.60 | Overlap threshold for consolidation |
