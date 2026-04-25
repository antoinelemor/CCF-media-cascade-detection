#!/usr/bin/env python3
"""
Re-run unified impact analysis on cached 2018 results.

Loads the cached DetectionResults, re-runs Phase 1/2/3 with the new
robust statistical attribution, exports JSON/CSV, and saves updated cache.
"""

import logging
import os
import pickle
import sys
import time
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

CACHE_DIR = PROJECT_ROOT / 'results' / 'cache'
CACHE_FILE = CACHE_DIR / 'results_2018.pkl'
IMPACT_DIR = PROJECT_ROOT / 'results' / 'impact_analysis'


def main():
    # Load cache
    logger.info(f"Loading cached results from {CACHE_FILE}...")
    t0 = time.time()
    with open(CACHE_FILE, 'rb') as f:
        results = pickle.load(f)
    logger.info(f"Loaded in {time.time() - t0:.1f}s: "
                f"{len(results.cascades)} cascades, "
                f"{len(results.event_clusters)} clusters")

    # Re-run unified impact analysis
    from cascade_detector.analysis.unified_impact import UnifiedImpactAnalyzer
    logger.info("Re-running unified impact analysis v2...")
    t0 = time.time()

    analyzer = UnifiedImpactAnalyzer()
    articles = getattr(results, '_articles', pd.DataFrame())
    paradigm_shifts = getattr(results, 'paradigm_shifts', None)
    paradigm_timeline = (
        getattr(paradigm_shifts, 'paradigm_timeline', None)
        if paradigm_shifts else None
    )

    impact_results = analyzer.run_from_components(
        clusters=results.event_clusters,
        cascades=results.cascades,
        articles=articles,
        paradigm_timeline=paradigm_timeline,
    )
    results.event_impact = impact_results
    elapsed = time.time() - t0
    logger.info(f"Impact analysis completed in {elapsed:.1f}s")

    # Print results
    df1 = impact_results.cluster_cascade
    df2 = impact_results.cluster_dominance
    df3 = impact_results.cascade_dominance

    print(f"\n{'='*80}")
    print("UNIFIED IMPACT ANALYSIS v2 — 2018")
    print(f"{'='*80}")
    print(f"Phase 1 (Cluster → Cascade): {len(df1)} pairs")
    print(f"Phase 2 (Cluster → Dominance): {len(df2)} pairs")
    print(f"Phase 3 (Cascade → Dominance): {len(df3)} pairs")

    if not df1.empty:
        print(f"\n{'─'*80}")
        print("Phase 1 Role Distribution:")
        print(df1['role'].value_counts().to_string())
        print(f"\nPhase 1 Impact Label Distribution:")
        print(df1['impact_label'].value_counts().to_string())
        print(f"\nPhase 1 Confidence Stats:")
        print(f"  Mean:   {df1['confidence'].mean():.3f}")
        print(f"  Median: {df1['confidence'].median():.3f}")
        print(f"  Min:    {df1['confidence'].min():.3f}")
        print(f"  Max:    {df1['confidence'].max():.3f}")

        # Statistical significance
        sig_did = (df1['did_p_value'] < 0.10).sum()
        sig_xcorr = (df1['xcorr_p_value'] < 0.10).sum()
        sig_granger = (df1['granger_p'] < 0.10).sum()
        sig_any = ((df1['did_p_value'] < 0.10) | (df1['xcorr_p_value'] < 0.10) | (df1['granger_p'] < 0.10)).sum()
        sig_perm = (df1['perm_p_adjusted'] < 0.10).sum()
        print(f"\nStatistical Significance (p < 0.10):")
        print(f"  DID:        {sig_did}/{len(df1)}")
        print(f"  Xcorr:      {sig_xcorr}/{len(df1)}")
        print(f"  Granger P1: {sig_granger}/{len(df1)}")
        print(f"  Any stat:   {sig_any}/{len(df1)}")
        print(f"  Permutation (BH-adjusted): {sig_perm}/{len(df1)}")

        # Content relevance stats
        print(f"\nContent Relevance Stats:")
        print(f"  Mean:   {df1['content_relevance'].mean():.3f}")
        print(f"  Median: {df1['content_relevance'].median():.3f}")

    # Export impact Parquet
    IMPACT_DIR.mkdir(parents=True, exist_ok=True)
    df1.to_parquet(IMPACT_DIR / 'cluster_cascade.parquet', index=False)
    df2.to_parquet(IMPACT_DIR / 'cluster_dominance.parquet', index=False)
    df3.to_parquet(IMPACT_DIR / 'cascade_dominance.parquet', index=False)

    import json
    summary_path = IMPACT_DIR / 'summary.json'
    with open(summary_path, 'w') as f:
        # Convert numpy int64 etc to native Python types
        summary = {}
        for k, v in impact_results.summary.items():
            if isinstance(v, dict):
                summary[k] = {str(kk): int(vv) if isinstance(vv, (np.integer,)) else vv
                              for kk, vv in v.items()}
            elif isinstance(v, (np.integer,)):
                summary[k] = int(v)
            else:
                summary[k] = v
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Impact Parquet/JSON saved to {IMPACT_DIR}/")

    # Save updated cache
    logger.info("Saving updated cache...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Export JSON and CSV (cascades)
    json_path = PROJECT_ROOT / 'results' / 'cascades_2018.json'
    results.to_json(str(json_path))
    df = results.to_dataframe()
    csv_path = PROJECT_ROOT / 'results' / 'cascades_2018.csv'
    df.to_csv(str(csv_path), index=False)
    logger.info(f"Cascade JSON/CSV exported to {json_path}")

    print(f"\n{'='*80}")
    print("All outputs saved:")
    print(f"  Impact:  {IMPACT_DIR}/")
    print(f"  JSON:    {json_path}")
    print(f"  CSV:     {csv_path}")
    print(f"  Cache:   {CACHE_FILE}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
