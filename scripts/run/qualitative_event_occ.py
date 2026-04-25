#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
qualitative_event_occ.py

MAIN OBJECTIVE:
---------------
Deep qualitative analysis of event occurrences in the top 5 cascades (2018).
For each occurrence, queries the database for actual article text (sentences)
and compares event-filtered vs full-article embedding similarity to verify
semantic coherence.

Usage:
    EMBEDDING_DIR=data/embeddings-test python scripts/run/qualitative_event_occ.py

Author:
-------
Antoine Lemor
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# =========================================================================
# Constants
# =========================================================================
TOP_N_CASCADES = 5
ARTICLES_PER_OCC = 3          # representative articles to show per occurrence
SENTENCES_PER_ARTICLE = 5     # max sentences to display per article
MAX_SIM_ARTICLES = 100        # cap for pairwise similarity computation

# Coherence thresholds for verdict
COHERENCE_HIGH = 0.70
COHERENCE_LOW = 0.50


# =========================================================================
# Helpers
# =========================================================================

def get_occurrence_doc_ids(
    occ,
    articles: pd.DataFrame,
    date_col: str,
) -> List:
    """Recover doc_ids belonging to an occurrence from cascade articles.

    Since EventOccurrence does not store doc_ids directly, we approximate
    them by filtering cascade articles on:
      1. The event type column >= 0.10 (seed threshold)
      2. Date within [first_date, last_date] of the occurrence

    This is a heuristic; articles reassigned from other clusters or unlabeled
    articles captured during consolidation may be missed, but for the purpose
    of sampling representative texts this is sufficient.
    """
    onset = pd.Timestamp(occ.first_date)
    end = pd.Timestamp(occ.last_date)
    dates = pd.to_datetime(articles[date_col])
    date_mask = (dates >= onset) & (dates <= end)

    # Find the event column (could be evt_type_mean or evt_type_sum)
    evt_col = None
    for suffix in ['_mean', '_sum', '']:
        col = f'{occ.event_type}{suffix}'
        if col in articles.columns:
            evt_col = col
            break

    if evt_col is not None:
        evt_mask = articles[evt_col] >= 0.10
        mask = date_mask & evt_mask
    else:
        mask = date_mask

    matched = articles.loc[mask]
    return matched['doc_id'].tolist()


def compute_pairwise_similarity(
    embedding_store,
    doc_ids: list,
    max_articles: int = MAX_SIM_ARTICLES,
) -> Tuple[float, float, int]:
    """Compute mean pairwise cosine similarity for a set of doc_ids.

    Returns:
        (mean_sim, std_sim, n_found)
    """
    if len(doc_ids) < 2:
        return 0.0, 0.0, len(doc_ids)

    if len(doc_ids) > max_articles:
        rng = np.random.RandomState(42)
        doc_ids = list(rng.choice(doc_ids, max_articles, replace=False))

    embeddings, found_ids = embedding_store.get_batch_article_embeddings(doc_ids)
    n_found = len(found_ids)
    if n_found < 2:
        return 0.0, 0.0, n_found

    sim_matrix = embedding_store.pairwise_cosine_similarity(embeddings)
    n = len(embeddings)
    upper = np.triu_indices(n, k=1)
    sims = sim_matrix[upper]
    return float(np.mean(sims)), float(np.std(sims)), n_found


def compute_filtered_similarity(
    embedding_store,
    doc_ids: list,
    evt_type: str,
    evt_sentence_index: Optional[Dict] = None,
    max_articles: int = MAX_SIM_ARTICLES,
) -> Tuple[float, float, int]:
    """Compute mean pairwise similarity using event-filtered embeddings.

    For each article, only sentences where the event type is active are
    mean-pooled. Falls back to full article embedding if no event sentences.

    Returns:
        (mean_sim, std_sim, n_found)
    """
    if len(doc_ids) < 2:
        return 0.0, 0.0, len(doc_ids)

    if len(doc_ids) > max_articles:
        rng = np.random.RandomState(42)
        doc_ids = list(rng.choice(doc_ids, max_articles, replace=False))

    embeddings = []
    found_ids = []
    for doc_id in doc_ids:
        if evt_sentence_index is not None:
            key = (doc_id, evt_type)
            sentence_ids = evt_sentence_index.get(key)
            if sentence_ids:
                emb = embedding_store.get_filtered_article_embedding(
                    doc_id, sentence_ids
                )
            else:
                emb = embedding_store.get_article_embedding(doc_id)
        else:
            emb = embedding_store.get_article_embedding(doc_id)

        if emb is not None:
            embeddings.append(emb)
            found_ids.append(doc_id)

    n_found = len(found_ids)
    if n_found < 2:
        return 0.0, 0.0, n_found

    embeddings = np.array(embeddings, dtype=np.float32)
    sim_matrix = embedding_store.pairwise_cosine_similarity(embeddings)
    n = len(embeddings)
    upper = np.triu_indices(n, k=1)
    sims = sim_matrix[upper]
    return float(np.mean(sims)), float(np.std(sims)), n_found


def query_article_sentences(
    db_connector,
    doc_ids: list,
    max_sentences: int = SENTENCES_PER_ARTICLE,
) -> pd.DataFrame:
    """Query DB for sentences of given doc_ids.

    Returns DataFrame with doc_id, sentence_id, sentences, media, date.
    """
    if not doc_ids:
        return pd.DataFrame()

    # Cast to Python int for PostgreSQL compatibility
    doc_ids_int = [int(d) for d in doc_ids]

    query = (
        f'SELECT doc_id, sentence_id, sentences, media, date '
        f'FROM "{db_connector.config.db_table}" '
        f'WHERE doc_id = ANY(%(doc_ids)s) '
        f'ORDER BY doc_id, sentence_id'
    )
    params = {'doc_ids': doc_ids_int}

    try:
        df = pd.read_sql(query, db_connector.engine, params=params)
        return df
    except Exception as e:
        logger.error(f"DB query failed: {e}")
        return pd.DataFrame()


def coherence_verdict(filtered_sim: float, full_sim: float, coherence: float) -> str:
    """Generate a coherence verdict string."""
    if coherence >= COHERENCE_HIGH and filtered_sim >= 0.60:
        return "COHERENT -- high intra-cluster similarity"
    elif coherence >= COHERENCE_LOW:
        delta = filtered_sim - full_sim
        if delta > 0.05:
            return "COHERENT -- event-filtered sim exceeds full-article sim"
        elif abs(delta) <= 0.05:
            return "PLAUSIBLE -- similar filtered vs full-article similarity"
        else:
            return "WEAK -- full-article sim higher than event-filtered sim"
    else:
        return "INCOHERENT -- low semantic coherence, articles may be unrelated"


# =========================================================================
# Main
# =========================================================================

def main():
    from cascade_detector.core.config import DetectorConfig
    from cascade_detector.pipeline import CascadeDetectionPipeline
    from cascade_detector.data.connector import DatabaseConnector
    from cascade_detector.core.constants import EVENT_COLUMNS

    embedding_dir = os.environ.get('EMBEDDING_DIR', 'data/embeddings-test')
    config = DetectorConfig(embedding_dir=embedding_dir, verbose=True)

    # =====================================================================
    # Step 1: Run pipeline
    # =====================================================================
    logger.info("Creating pipeline...")
    pipeline = CascadeDetectionPipeline(config)

    logger.info("Running 2018...")
    results = pipeline.run('2018-01-01', '2018-12-31')

    # Access internals
    embedding_store = pipeline.detector.embedding_store
    articles = results._articles
    db_connector = pipeline.connector

    # Determine date column in articles
    if 'date_converted_first' in articles.columns:
        date_col = 'date_converted_first'
    elif 'date_converted' in articles.columns:
        date_col = 'date_converted'
    else:
        date_col = 'date'

    # Build event sentence index for filtered embeddings
    # (Re-use the pipeline's sentence-level data if available)
    evt_sentence_index = None
    if hasattr(pipeline.detector, '_event_occ_detector'):
        det = pipeline.detector._event_occ_detector
        if hasattr(det, '_evt_sentence_index') and det._evt_sentence_index:
            evt_sentence_index = det._evt_sentence_index
    if evt_sentence_index is None:
        # Rebuild from DB sentence data
        logger.info("Building event sentence index from DB...")
        sentence_df = db_connector.get_frame_data('2018-01-01', '2018-12-31')
        _idx = {}
        if 'sentence_id' in sentence_df.columns:
            for evt_type in EVENT_COLUMNS:
                if evt_type not in sentence_df.columns:
                    continue
                evt_mask = sentence_df[evt_type] == 1
                if not evt_mask.any():
                    continue
                for doc_id, grp in sentence_df.loc[evt_mask, ['doc_id', 'sentence_id']].groupby('doc_id'):
                    _idx[(doc_id, evt_type)] = grp['sentence_id'].tolist()
        evt_sentence_index = _idx if _idx else None
        logger.info(f"  Built event sentence index: {len(_idx)} entries")

    # =====================================================================
    # Step 2: Select top cascades
    # =====================================================================
    cascades = sorted(results.cascades, key=lambda c: c.total_score, reverse=True)
    top_cascades = cascades[:TOP_N_CASCADES]

    # =====================================================================
    # Step 3: Qualitative analysis
    # =====================================================================
    print("\n" + "=" * 90)
    print("QUALITATIVE EVENT OCCURRENCE ANALYSIS -- 2018")
    print("=" * 90)
    print(f"Top {TOP_N_CASCADES} cascades by score | {ARTICLES_PER_OCC} articles/occ | "
          f"{SENTENCES_PER_ARTICLE} sentences/article")

    # Collect summary rows for final table
    summary_rows = []

    for rank, cascade in enumerate(top_cascades, 1):
        print(f"\n{'=' * 90}")
        print(f"CASCADE #{rank}: {cascade.cascade_id}  "
              f"[{cascade.classification}, {cascade.total_score:.3f}, "
              f"{cascade.duration_days}d, {cascade.n_articles} articles]")
        print(f"{'=' * 90}")
        print(f"  Period: {cascade.onset_date.strftime('%Y-%m-%d')} -> "
              f"{cascade.end_date.strftime('%Y-%m-%d')}")
        print(f"  Events (flat): {cascade.dominant_events}")
        print(f"  Occurrences: {len(cascade.event_occurrences)}")

        if not cascade.event_occurrences:
            print("  (no event occurrences detected)")
            continue

        # Get cascade articles
        dates = pd.to_datetime(articles[date_col])
        onset = pd.Timestamp(cascade.onset_date)
        end = pd.Timestamp(cascade.end_date)
        cascade_mask = (dates >= onset) & (dates <= end)
        cascade_articles = articles[cascade_mask].copy()

        for occ in cascade.event_occurrences:
            print(f"\n  {'~' * 80}")
            print(f"  OCC#{occ.occurrence_id}: {occ.event_type}  "
                  f"({occ.n_articles} articles, coherence={occ.semantic_coherence:.4f})")
            print(f"    Period: {occ.first_date.strftime('%Y-%m-%d')} -> "
                  f"{occ.last_date.strftime('%Y-%m-%d')}  "
                  f"(core: {occ.core_start.strftime('%Y-%m-%d')} -- "
                  f"{occ.core_end.strftime('%Y-%m-%d')})")
            print(f"    Peak: {occ.peak_date.strftime('%Y-%m-%d')}  "
                  f"| Mass: {occ.effective_mass:.2f} (core: {occ.core_mass:.2f})  "
                  f"| Confidence: {occ.confidence:.4f}"
                  f"{'  [LOW]' if occ.low_confidence else ''}")

            # Recover doc_ids for this occurrence
            occ_doc_ids = get_occurrence_doc_ids(occ, cascade_articles, date_col)
            if not occ_doc_ids:
                print("    (could not recover doc_ids for this occurrence)")
                continue

            # --- Similarity comparison ---
            full_sim, full_std, n_full = compute_pairwise_similarity(
                embedding_store, occ_doc_ids
            )
            filt_sim, filt_std, n_filt = compute_filtered_similarity(
                embedding_store, occ_doc_ids, occ.event_type,
                evt_sentence_index
            )
            print(f"    Filtered sim: {filt_sim:.4f} (std={filt_std:.4f}, n={n_filt})  |  "
                  f"Full-article sim: {full_sim:.4f} (std={full_std:.4f}, n={n_full})")

            verdict = coherence_verdict(filt_sim, full_sim, occ.semantic_coherence)
            print(f"    VERDICT: {verdict}")

            summary_rows.append({
                'cascade': cascade.cascade_id,
                'occ': occ.occurrence_id,
                'event_type': occ.event_type,
                'n_articles': occ.n_articles,
                'coherence': occ.semantic_coherence,
                'filt_sim': filt_sim,
                'full_sim': full_sim,
                'delta': filt_sim - full_sim,
                'verdict': verdict.split(' -- ')[0],
            })

            # --- Sample articles from DB ---
            # Pick representative doc_ids (spread across the occurrence period)
            sample_ids = _pick_representative_docs(
                occ_doc_ids, cascade_articles, date_col, n=ARTICLES_PER_OCC
            )

            sentences_df = query_article_sentences(db_connector, sample_ids)
            if sentences_df.empty:
                print("    (no sentences found in DB for sampled articles)")
                continue

            print(f"    Representative articles:")
            for doc_id in sample_ids:
                doc_rows = sentences_df[sentences_df['doc_id'] == int(doc_id)]
                if doc_rows.empty:
                    continue
                media = doc_rows['media'].iloc[0] if 'media' in doc_rows.columns else '?'
                date_val = doc_rows['date'].iloc[0] if 'date' in doc_rows.columns else '?'
                if hasattr(date_val, 'strftime'):
                    date_str = date_val.strftime('%Y-%m-%d')
                else:
                    date_str = str(date_val)

                print(f"      [doc_id={doc_id}, media={media}, {date_str}]")
                shown = 0
                for _, row in doc_rows.iterrows():
                    if shown >= SENTENCES_PER_ARTICLE:
                        break
                    text = str(row.get('sentences', '')).strip()
                    if not text or text == 'nan':
                        continue
                    # Truncate long sentences
                    if len(text) > 200:
                        text = text[:200] + "..."
                    print(f"        \"{text}\"")
                    shown += 1
                if shown == 0:
                    print(f"        (no text available)")

    # =====================================================================
    # Step 4: Summary comparison table
    # =====================================================================
    if summary_rows:
        print(f"\n\n{'=' * 90}")
        print("SUMMARY COMPARISON TABLE")
        print(f"{'=' * 90}")
        print(f"{'Cascade':<25s} {'Occ':>3s} {'EventType':<18s} {'Arts':>5s} "
              f"{'Coh':>6s} {'FiltSim':>8s} {'FullSim':>8s} {'Delta':>7s} {'Verdict':<12s}")
        print(f"{'~' * 90}")

        for r in summary_rows:
            print(f"{r['cascade']:<25s} {r['occ']:>3d} {r['event_type']:<18s} "
                  f"{r['n_articles']:>5d} {r['coherence']:>6.4f} "
                  f"{r['filt_sim']:>8.4f} {r['full_sim']:>8.4f} "
                  f"{r['delta']:>+7.4f} {r['verdict']:<12s}")

        # Aggregate statistics
        all_coh = [r['coherence'] for r in summary_rows]
        all_filt = [r['filt_sim'] for r in summary_rows]
        all_full = [r['full_sim'] for r in summary_rows]
        all_delta = [r['delta'] for r in summary_rows]

        print(f"{'~' * 90}")
        print(f"{'MEAN':<25s} {'':>3s} {'':>18s} {'':>5s} "
              f"{np.mean(all_coh):>6.4f} {np.mean(all_filt):>8.4f} "
              f"{np.mean(all_full):>8.4f} {np.mean(all_delta):>+7.4f}")
        print(f"{'STD':<25s} {'':>3s} {'':>18s} {'':>5s} "
              f"{np.std(all_coh):>6.4f} {np.std(all_filt):>8.4f} "
              f"{np.std(all_full):>8.4f} {np.std(all_delta):>+7.4f}")

        # Count verdicts
        from collections import Counter
        verdict_counts = Counter(r['verdict'] for r in summary_rows)
        print(f"\n  Verdict distribution:")
        for v, count in verdict_counts.most_common():
            print(f"    {v:<15s}: {count}")

        # Flag incoherent occurrences
        incoherent = [r for r in summary_rows if r['verdict'] == 'INCOHERENT']
        if incoherent:
            print(f"\n  FLAGGED INCOHERENT OCCURRENCES ({len(incoherent)}):")
            for r in incoherent:
                print(f"    {r['cascade']} OCC#{r['occ']} ({r['event_type']}, "
                      f"coherence={r['coherence']:.4f})")
        else:
            print(f"\n  No incoherent occurrences flagged.")

    print(f"\n{'=' * 90}")
    print("Done.")


def _pick_representative_docs(
    doc_ids: list,
    articles: pd.DataFrame,
    date_col: str,
    n: int = 3,
) -> list:
    """Pick n doc_ids spread across the occurrence period.

    Selects early, middle, and late articles for temporal coverage.
    """
    if len(doc_ids) <= n:
        return doc_ids

    # Get dates for these doc_ids
    mask = articles['doc_id'].isin(doc_ids)
    sub = articles.loc[mask, ['doc_id', date_col]].copy()
    sub['_date'] = pd.to_datetime(sub[date_col])
    sub = sub.sort_values('_date')

    if len(sub) <= n:
        return sub['doc_id'].tolist()

    # Pick evenly spaced indices
    indices = np.linspace(0, len(sub) - 1, n, dtype=int)
    return [sub.iloc[i]['doc_id'] for i in indices]


if __name__ == '__main__':
    main()
