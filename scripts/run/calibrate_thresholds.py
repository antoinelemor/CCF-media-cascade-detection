#!/usr/bin/env python3
"""
Calibration script: measure actual pairwise cosine distances within each
event type to find the right COSINE_DISTANCE_THRESHOLD for splitting
within-type clusters.

Usage:
    EMBEDDING_DIR=data/embeddings-test python scripts/run/calibrate_thresholds.py
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    from cascade_detector.core.config import DetectorConfig
    from cascade_detector.core.constants import EVENT_COLUMNS
    from cascade_detector.pipeline import CascadeDetectionPipeline

    embedding_dir = os.environ.get('EMBEDDING_DIR', 'data/embeddings-test')
    config = DetectorConfig(embedding_dir=embedding_dir)
    pipeline = CascadeDetectionPipeline(config)

    logger.info("Running 2018 pipeline...")
    results = pipeline.run('2018-01-01', '2018-12-31')
    articles = results._articles
    embedding_store = pipeline.detector.embedding_store

    # Resolve date column
    date_col = 'date_converted_first' if 'date_converted_first' in articles.columns else 'date'

    print("\n" + "=" * 80)
    print("INTRA-TYPE DISTANCE ANALYSIS")
    print("=" * 80)

    # Pick a few representative cascades
    cascades = sorted(results.cascades, key=lambda c: c.total_score, reverse=True)

    for cascade in cascades[:5]:
        dates = pd.to_datetime(articles[date_col])
        onset = pd.Timestamp(cascade.onset_date)
        end = pd.Timestamp(cascade.end_date)
        mask = (dates >= onset) & (dates <= end)
        burst_articles = articles[mask].copy()

        print(f"\n{'─' * 80}")
        print(f"CASCADE: {cascade.cascade_id} ({cascade.duration_days}d, {len(burst_articles)} articles)")
        print(f"{'─' * 80}")

        for evt_type in EVENT_COLUMNS:
            # Resolve column
            evt_col = None
            for suffix in ['', '_mean', '_sum']:
                col = f'{evt_type}{suffix}'
                if col in burst_articles.columns:
                    evt_col = col
                    break
            if evt_col is None:
                continue

            seeds = burst_articles[burst_articles[evt_col] >= 0.10]
            if len(seeds) < 5:
                continue

            doc_ids = seeds['doc_id'].tolist()
            embeddings, found_ids = embedding_store.get_batch_article_embeddings(doc_ids)
            if len(found_ids) < 5:
                continue

            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            normalized = embeddings / norms

            # Pairwise cosine distances
            dists = pdist(normalized, metric='cosine')
            dists = np.clip(dists, 0, 2)

            sims = 1.0 - dists

            print(f"\n  {evt_type} ({len(found_ids)} seed articles):")
            print(f"    Pairwise cosine SIMILARITY: "
                  f"mean={np.mean(sims):.4f}  median={np.median(sims):.4f}  "
                  f"min={np.min(sims):.4f}  max={np.max(sims):.4f}")
            print(f"    Pairwise cosine DISTANCE:   "
                  f"mean={np.mean(dists):.4f}  median={np.median(dists):.4f}  "
                  f"min={np.min(dists):.4f}  max={np.max(dists):.4f}")

            # Show what we'd get at different thresholds
            print(f"    Cluster counts at different thresholds:")
            Z = linkage(dists, method='average')
            for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
                labels = fcluster(Z, t=t, criterion='distance')
                n_clusters = len(set(labels))
                sizes = [list(labels).count(l) for l in set(labels)]
                sizes_above_3 = [s for s in sizes if s >= 3]
                print(f"      t={t:.2f} (sim≥{1-t:.2f}): {n_clusters:>3d} clusters "
                      f"({len(sizes_above_3)} with ≥3 articles)  "
                      f"sizes={sorted(sizes, reverse=True)[:10]}")

            # Temporal analysis: do articles on different dates have higher distance?
            seed_dates = pd.to_datetime(seeds.set_index('doc_id').loc[found_ids, date_col])
            date_ordinals = seed_dates.map(lambda d: d.toordinal()).values
            date_diffs = squareform(pdist(date_ordinals.reshape(-1, 1), metric='euclidean'))
            dist_matrix = squareform(dists)

            # Correlation between temporal and semantic distance
            upper_idx = np.triu_indices(len(found_ids), k=1)
            temporal_flat = date_diffs[upper_idx]
            semantic_flat = dist_matrix[upper_idx]

            if len(temporal_flat) > 5:
                corr = np.corrcoef(temporal_flat, semantic_flat)[0, 1]
                print(f"    Temporal-semantic distance correlation: {corr:.4f}")

                # Bin by temporal distance
                for d_cutoff in [7, 14, 21]:
                    near_mask = temporal_flat <= d_cutoff
                    far_mask = temporal_flat > d_cutoff
                    if near_mask.sum() > 0 and far_mask.sum() > 0:
                        near_sim = np.mean(1 - semantic_flat[near_mask])
                        far_sim = np.mean(1 - semantic_flat[far_mask])
                        print(f"    Articles ≤{d_cutoff}d apart: sim={near_sim:.4f} "
                              f"(n={near_mask.sum()})  |  >{d_cutoff}d apart: sim={far_sim:.4f} "
                              f"(n={far_mask.sum()})  delta={near_sim - far_sim:.4f}")

    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    main()
