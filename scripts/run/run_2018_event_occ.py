#!/usr/bin/env python3
"""
Diagnostic run: 2018 cascade detection + event occurrence analysis.
Prints detailed per-cascade occurrence output for threshold calibration.

Usage:
    EMBEDDING_DIR=data/embeddings-test python scripts/run/run_2018_event_occ.py
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    from cascade_detector.core.config import DetectorConfig
    from cascade_detector.pipeline import CascadeDetectionPipeline

    embedding_dir = os.environ.get('EMBEDDING_DIR', 'data/embeddings-test')
    config = DetectorConfig(embedding_dir=embedding_dir, verbose=True)

    logger.info("Creating pipeline...")
    pipeline = CascadeDetectionPipeline(config)

    logger.info("Running 2018...")
    results = pipeline.run('2018-01-01', '2018-12-31')

    # =========================================================================
    # Event occurrence diagnostic output
    # =========================================================================
    print("\n" + "=" * 80)
    print("EVENT OCCURRENCE DIAGNOSTIC — 2018")
    print("=" * 80)

    cascades = sorted(results.cascades, key=lambda c: c.total_score, reverse=True)
    n_with_occ = sum(1 for c in cascades if c.event_occurrences)
    total_occ = sum(len(c.event_occurrences) for c in cascades)

    print(f"\nCascades: {len(cascades)}  |  With occurrences: {n_with_occ}  |  Total occurrences: {total_occ}")

    # Per-cascade summary
    print(f"\n{'─' * 80}")
    print(f"{'Cascade':<25s} {'Score':>6s} {'Class':<12s} {'Days':>4s} {'Arts':>5s} "
          f"{'Occ':>3s} {'EvtTypes':>8s} {'MeanCoh':>8s} {'Overlap':>8s}")
    print(f"{'─' * 80}")

    for c in cascades:
        m = c.event_occurrence_metrics or {}
        n_occ = int(m.get('n_occurrences', 0))
        n_et = int(m.get('n_event_types', 0))
        coh = m.get('mean_coherence', 0.0)
        ovlp = m.get('temporal_overlap', 0.0)

        print(f"{c.cascade_id:<25s} {c.total_score:>6.3f} {c.classification:<12s} "
              f"{c.duration_days:>4d} {c.n_articles:>5d} "
              f"{n_occ:>3d} {n_et:>8d} {coh:>8.4f} {ovlp:>8.4f}")

    # Detailed per-occurrence
    print(f"\n{'=' * 80}")
    print("DETAILED OCCURRENCES (top 15 cascades by score)")
    print("=" * 80)

    for c in cascades[:15]:
        if not c.event_occurrences:
            print(f"\n  {c.cascade_id}: no occurrences")
            continue

        print(f"\n  {c.cascade_id}  [{c.classification}, score={c.total_score:.3f}, "
              f"{c.duration_days}d, {c.n_articles} articles]")
        print(f"  dominant_events (flat): {c.dominant_events}")
        print(f"  Occurrence metrics: {c.event_occurrence_metrics}")

        for occ in c.event_occurrences:
            print(f"    OCC#{occ.occurrence_id}: {occ.event_type}")
            print(f"      Period: {occ.first_date.strftime('%Y-%m-%d')} → {occ.last_date.strftime('%Y-%m-%d')} "
                  f"(core: {occ.core_start.strftime('%Y-%m-%d')}–{occ.core_end.strftime('%Y-%m-%d')})")
            print(f"      Peak: {occ.peak_date.strftime('%Y-%m-%d')}")
            print(f"      Articles: {occ.n_articles}  |  Effective mass: {occ.effective_mass:.2f}  |  Core mass: {occ.core_mass:.2f}")
            print(f"      Coherence: {occ.semantic_coherence:.4f}  |  Confidence: {occ.confidence:.4f}"
                  f"{'  [LOW]' if occ.low_confidence else ''}")

    # Calibration statistics
    print(f"\n{'=' * 80}")
    print("CALIBRATION STATISTICS")
    print("=" * 80)

    all_coherences = []
    all_n_articles = []
    all_durations = []
    occ_per_cascade = []

    for c in cascades:
        if c.event_occurrences:
            occ_per_cascade.append(len(c.event_occurrences))
        for occ in c.event_occurrences:
            all_coherences.append(occ.semantic_coherence)
            all_n_articles.append(occ.n_articles)
            dur = (occ.last_date - occ.first_date).days
            all_durations.append(dur)

    if all_coherences:
        print(f"\n  Semantic coherence:")
        print(f"    Mean:   {np.mean(all_coherences):.4f}")
        print(f"    Median: {np.median(all_coherences):.4f}")
        print(f"    Min:    {np.min(all_coherences):.4f}")
        print(f"    Max:    {np.max(all_coherences):.4f}")
        print(f"    P10:    {np.percentile(all_coherences, 10):.4f}")
        print(f"    P90:    {np.percentile(all_coherences, 90):.4f}")

        print(f"\n  Articles per occurrence:")
        print(f"    Mean:   {np.mean(all_n_articles):.1f}")
        print(f"    Median: {np.median(all_n_articles):.1f}")
        print(f"    Min:    {np.min(all_n_articles)}")
        print(f"    Max:    {np.max(all_n_articles)}")

        print(f"\n  Duration (days) per occurrence:")
        print(f"    Mean:   {np.mean(all_durations):.1f}")
        print(f"    Median: {np.median(all_durations):.1f}")
        print(f"    Min:    {np.min(all_durations)}")
        print(f"    Max:    {np.max(all_durations)}")

        print(f"\n  Occurrences per cascade:")
        print(f"    Mean:   {np.mean(occ_per_cascade):.1f}")
        print(f"    Median: {np.median(occ_per_cascade):.1f}")
        print(f"    Min:    {np.min(occ_per_cascade)}")
        print(f"    Max:    {np.max(occ_per_cascade)}")

    # Belonging / confidence stats
    all_eff_mass = [o.effective_mass for c in cascades for o in c.event_occurrences]
    all_confidence = [o.confidence for c in cascades for o in c.event_occurrences]
    n_low_conf = sum(1 for o in (occ for c in cascades for occ in c.event_occurrences) if o.low_confidence)

    if all_eff_mass:
        print(f"\n  Effective mass per occurrence:")
        print(f"    Mean:   {np.mean(all_eff_mass):.2f}")
        print(f"    Median: {np.median(all_eff_mass):.2f}")
        print(f"    Min:    {np.min(all_eff_mass):.2f}")
        print(f"    Max:    {np.max(all_eff_mass):.2f}")

        print(f"\n  Confidence per occurrence:")
        print(f"    Mean:   {np.mean(all_confidence):.4f}")
        print(f"    Median: {np.median(all_confidence):.4f}")
        print(f"    Min:    {np.min(all_confidence):.4f}")
        print(f"    Max:    {np.max(all_confidence):.4f}")
        print(f"    Low confidence: {n_low_conf}/{len(all_confidence)}")

    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    main()
