#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
run_2018.py

MAIN OBJECTIVE:
---------------
Run the full cascade detection pipeline on 2018 data (Jan 1 - Dec 31).
Caches the full DetectionResults to results/cache/results_2018.pkl so
subsequent runs load instantly when only analysis/visualization changes.

Use --no-cache to force a fresh pipeline run.
Use --cache-only to skip the pipeline and only print cached results.

Usage:
    python scripts/run/run_2018.py
    python scripts/run/run_2018.py --no-cache
    python scripts/run/run_2018.py --cache-only

Author:
-------
Antoine Lemor
"""

import argparse
import json
import logging
import os
import pickle
import sys
import statistics
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cascade_detector.core.constants import FRAMES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

CACHE_DIR = PROJECT_ROOT / 'results' / 'cache'
CACHE_FILE = CACHE_DIR / 'results_2018.pkl'


def run_pipeline():
    """Run the full 2018 pipeline and return DetectionResults."""
    from cascade_detector.core.config import DetectorConfig
    from cascade_detector.pipeline import CascadeDetectionPipeline

    config = DetectorConfig(
        embedding_dir=os.environ.get('EMBEDDING_DIR', 'data/embeddings'),
        verbose=True,
    )

    logger.info("Creating pipeline...")
    pipeline = CascadeDetectionPipeline(config)

    logger.info("Running cascade detection on 2018 data...")
    results = pipeline.run(
        '2018-01-01', '2018-12-31',
        checkpoint_dir=CACHE_DIR / 'checkpoints_2018',
    )
    return results


def save_cache(results):
    """Save results to pickle cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = CACHE_FILE.stat().st_size / 1024 / 1024
    logger.info(f"Cache saved: {CACHE_FILE} ({size_mb:.1f} MB)")


def load_cache():
    """Load results from pickle cache. Returns None if not found."""
    if not CACHE_FILE.exists():
        return None
    logger.info(f"Loading cached results from {CACHE_FILE}...")
    t0 = time.time()
    with open(CACHE_FILE, 'rb') as f:
        results = pickle.load(f)
    logger.info(f"Cache loaded in {time.time() - t0:.1f}s")
    return results


def export_results(results):
    """Export results to JSON and CSV."""
    output_dir = PROJECT_ROOT / 'results'
    output_dir.mkdir(exist_ok=True)

    json_path = str(output_dir / 'cascades_2018.json')
    results.to_json(json_path)
    logger.info(f"JSON exported to {json_path}")

    df = results.to_dataframe()
    csv_path = str(output_dir / 'cascades_2018.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV exported to {csv_path}")

    return json_path, csv_path


def print_cascade_detail(c, rank=None):
    """Print exhaustive detail for a single cascade."""
    header = f"#{rank} " if rank else ""
    print(f"\n{'─' * 80}")
    print(f"  {header}{c.cascade_id}")
    print(f"{'─' * 80}")
    print(f"  Frame:       {c.frame}")
    print(f"  Period:      {c.onset_date.strftime('%Y-%m-%d')} → {c.end_date.strftime('%Y-%m-%d')} ({c.duration_days} days)")
    print(f"  Peak:        {c.peak_date.strftime('%Y-%m-%d')}")
    print(f"  Detection:   {c.detection_method}")
    print()
    print(f"  PARTICIPATION:")
    print(f"    Articles:       {c.n_articles}")
    print(f"    Journalists:    {c.n_journalists} ({c.n_new_journalists} new)")
    print(f"    Media outlets:  {c.n_media}")
    print()
    print(f"  BURST METRICS:")
    print(f"    Intensity:      {c.burst_intensity:.4f} (peak={c.peak_proportion:.6f} / baseline={c.baseline_mean:.6f})")
    print(f"    Adoption vel.:  {c.adoption_velocity:.4f} journalists/day")
    print(f"    Composite peak: {c.composite_peak:.4f}")
    print(f"    Mann-Whitney p: {c.mann_whitney_p:.6f}")
    print()
    print(f"  SCORES (4 dimensions):")
    print(f"    Temporal:       {c.score_temporal:.4f}  (weight 0.25)")
    print(f"    Participation:  {c.score_participation:.4f}  (weight 0.25)")
    print(f"    Convergence:    {c.score_convergence:.4f}  (weight 0.25)")
    print(f"    Source:         {c.score_source:.4f}  (weight 0.25)")
    print(f"    ────────────────────────────────")
    print(f"    TOTAL:          {c.total_score:.4f}  [{c.classification}]")
    print()
    print(f"  ALL 17 SUB-INDICES:")
    for key in sorted(c.sub_indices.keys()):
        print(f"    {key:<45s} {c.sub_indices[key]:.4f}")
    print()
    print(f"  NETWORK METRICS:")
    print(f"    Density:        {c.network_density:.4f}")
    print(f"    Modularity:     {c.network_modularity:.4f}")
    print(f"    Mean degree:    {c.network_mean_degree:.4f}")
    print(f"    Components:     {c.network_n_components}")
    print()
    print(f"  SEMANTIC CONVERGENCE (raw embedding-based):")
    print(f"    Intra-window similarity:  {c.semantic_similarity:.4f}")
    print(f"    Convergence trend slope:  {c.convergence_trend:.6f}")
    print(f"    Cross-media alignment:    {c.cross_media_alignment:.4f}")
    print(f"    Novelty decay rate:       {c.novelty_decay_rate:.6f}")
    print()
    print(f"  SOURCE CONVERGENCE (raw):")
    print(f"    Diversity decline:        {c.source_diversity_decline:.4f}")
    print(f"    Messenger concentration:  {c.messenger_concentration:.4f}")
    print(f"    Media coordination:       {c.media_coordination:.4f}")
    if c.top_journalists:
        print(f"  TOP JOURNALISTS: {c.top_journalists[:5]}")
    if c.top_media:
        print(f"  TOP MEDIA: {c.top_media[:5]}")
    if c.dominant_events:
        print(f"  EVENTS: {c.dominant_events}")
    if c.dominant_messengers:
        print(f"  MESSENGERS: {c.dominant_messengers}")

    # Event occurrences (v3 attribution)
    if c.event_occurrences:
        print(f"  EVENT OCCURRENCES ({len(c.event_occurrences)}):")
        for occ in sorted(c.event_occurrences, key=lambda o: o.peak_date):
            conf_label = 'LOW' if occ.low_confidence else 'OK'
            print(f"    {occ.event_type:<20s} peak={occ.peak_date.strftime('%Y-%m-%d')} "
                  f"n={occ.n_articles:3d} mass={occ.effective_mass:.1f} "
                  f"conf={occ.confidence:.2f}[{conf_label}]")


def print_summary(results, json_path, csv_path):
    """Print full summary including v3 event occurrence info."""
    print("\n" + "=" * 80)
    print("CASCADE DETECTION RESULTS — 2018")
    print("=" * 80)
    print(f"\nAnalysis period: 2018-01-01 to 2018-12-31")
    print(f"Articles analyzed: {results.n_articles_analyzed:,}")
    print(f"Runtime: {results.runtime_seconds:.1f}s")
    print(f"Bursts detected: {len(results.all_bursts)}")
    print(f"Cascades scored: {len(results.cascades)}")

    # v3 event detection summary
    if hasattr(results, 'all_occurrences') and results.all_occurrences:
        print(f"Event occurrences (database-first): {len(results.all_occurrences)}")
    if hasattr(results, 'event_clusters') and results.event_clusters:
        print(f"Event clusters: {len(results.event_clusters)}")
    if hasattr(results, 'cascade_attributions') and results.cascade_attributions:
        print(f"Cascade attributions: {len(results.cascade_attributions)}")

    # By frame
    print("\n" + "─" * 80)
    print("CASCADES BY FRAME")
    print("─" * 80)
    for frame in FRAMES:
        count = results.n_cascades_by_frame.get(frame, 0)
        print(f"  {frame:<6s}: {count}")

    # By classification
    print("\n" + "─" * 80)
    print("CASCADES BY CLASSIFICATION")
    print("─" * 80)
    for cls in ['strong_cascade', 'moderate_cascade', 'weak_cascade', 'not_cascade']:
        count = results.n_cascades_by_classification.get(cls, 0)
        print(f"  {cls:<20s}: {count}")

    # Event occurrence summary
    if hasattr(results, 'all_occurrences') and results.all_occurrences:
        print("\n" + "─" * 80)
        print("EVENT OCCURRENCES (DATABASE-FIRST v3)")
        print("─" * 80)
        by_type = {}
        for occ in results.all_occurrences:
            by_type.setdefault(occ.event_type, []).append(occ)
        for et in sorted(by_type):
            occs = by_type[et]
            hi = sum(1 for o in occs if not o.low_confidence)
            print(f"  {et:<25s}: {len(occs):3d} occurrences ({hi} high confidence)")

        # Attribution statistics
        if hasattr(results, 'cascade_attributions') and results.cascade_attributions:
            n_with = sum(1 for c in results.cascades if c.event_occurrences)
            multi_attr = sum(1 for o in results.all_occurrences
                            if len(o.cascade_attributions) > 1)
            print(f"\n  Cascades with events:  {n_with}/{len(results.cascades)}")
            print(f"  Occurrences in 2+ cascades: {multi_attr}")
            print(f"  Total attributions: {len(results.cascade_attributions)}")

    # Event clusters
    if hasattr(results, 'event_clusters') and results.event_clusters:
        print("\n" + "─" * 80)
        print("EVENT CLUSTERS")
        print("─" * 80)
        ecs = results.event_clusters
        multi = sum(1 for ec in ecs if ec.is_multi_type)
        print(f"  Total: {len(ecs)} ({multi} multi-type, {len(ecs) - multi} mono-type)")
        print(f"  Mean strength: {np.mean([ec.strength for ec in ecs]):.3f}")
        print(f"\n  Top 10 by strength:")
        for ec in sorted(ecs, key=lambda x: x.strength, reverse=True)[:10]:
            types = ', '.join(sorted(ec.event_types))
            print(f"    C{ec.cluster_id:3d}: str={ec.strength:.3f} "
                  f"n_occ={ec.n_occurrences} mass={ec.total_mass:.0f} "
                  f"[{types}] peak={ec.peak_date.strftime('%Y-%m-%d')} "
                  f"{'MULTI' if ec.is_multi_type else ''}")

    # All cascades — full detail
    if results.cascades:
        sorted_cascades = sorted(results.cascades, key=lambda x: x.total_score, reverse=True)

        print("\n" + "=" * 80)
        print("ALL CASCADES — RANKED BY SCORE (FULL DETAIL)")
        print("=" * 80)
        for rank, c in enumerate(sorted_cascades, 1):
            print_cascade_detail(c, rank=rank)

        # Score statistics
        scored = [c for c in results.cascades if c.total_score > 0]
        if scored:
            scores = [c.total_score for c in scored]
            print("\n" + "─" * 80)
            print(f"SCORE STATISTICS (n={len(scored)} scored cascades)")
            print("─" * 80)
            print(f"  Mean:    {statistics.mean(scores):.4f}")
            print(f"  Median:  {statistics.median(scores):.4f}")
            print(f"  Min:     {min(scores):.4f}")
            print(f"  Max:     {max(scores):.4f}")
            if len(scores) > 1:
                print(f"  Stdev:   {statistics.stdev(scores):.4f}")

            print(f"\n  {'Dimension':<20s} {'Mean':>8s} {'Min':>8s} {'Max':>8s}")
            print(f"  {'─' * 50}")
            for dim_name, dim_attr in [
                ('Temporal', 'score_temporal'),
                ('Participation', 'score_participation'),
                ('Convergence', 'score_convergence'),
                ('Source', 'score_source'),
            ]:
                vals = [getattr(c, dim_attr) for c in scored]
                print(f"  {dim_name:<20s} {np.mean(vals):>8.4f} {min(vals):>8.4f} {max(vals):>8.4f}")

        # Sub-index statistics
        if scored:
            print(f"\n" + "─" * 80)
            print(f"SUB-INDEX STATISTICS (n={len(scored)})")
            print("─" * 80)
            all_keys = set()
            for c in scored:
                all_keys.update(c.sub_indices.keys())

            print(f"  {'Sub-Index':<45s} {'Mean':>8s} {'Min':>8s} {'Max':>8s} {'Std':>8s}")
            print(f"  {'─' * 75}")
            for key in sorted(all_keys):
                vals = [c.sub_indices.get(key, 0) for c in scored]
                print(f"  {key:<45s} {np.mean(vals):>8.4f} {min(vals):>8.4f} "
                      f"{max(vals):>8.4f} {np.std(vals):>8.4f}")

    print("\n" + "=" * 80)
    print(f"Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Run 2018 cascade detection')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force fresh pipeline run, ignore cache')
    parser.add_argument('--cache-only', action='store_true',
                        help='Only load and display cached results')
    args = parser.parse_args()

    # Try cache first
    results = None
    if not args.no_cache:
        results = load_cache()

    if results is None:
        if args.cache_only:
            logger.error(f"No cache found at {CACHE_FILE}. Run without --cache-only first.")
            sys.exit(1)
        results = run_pipeline()
        save_cache(results)

    # Export
    json_path, csv_path = export_results(results)

    # Summary
    print_summary(results, json_path, csv_path)


if __name__ == '__main__':
    main()
