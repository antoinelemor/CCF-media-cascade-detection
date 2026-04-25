#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
compute_dominance_scores.py

MAIN OBJECTIVE:
---------------
Compute composite dominance scores using the exact 4-method consensus
from ParadigmDominanceAnalyzer (CCF-paradigm).

Input: PostgreSQL CCF_Database_texts (weekly frame proportions 1978-2024)
Output: tables/overall_dominance_scores.csv

Reproduces the analysis from CCF-paradigm/scripts/02_dominant_frames_analysis.py

Usage:
  python paper/papers/CCF_paradigm/scripts/compute_dominance_scores.py

Author:
-------
Antoine Lemor
"""
import sys
from pathlib import Path

# Add local modules
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd
import numpy as np
from database_access.db_connector import DatabaseConnector
from paradigm_dominance import ParadigmDominanceAnalyzer

OUT = SCRIPT_DIR.parent / 'tables'
OUT.mkdir(exist_ok=True)

FRAMES = ['Cult', 'Eco', 'Envt', 'Pbh', 'Just', 'Pol', 'Sci', 'Secu']
FRAME_NAMES = {
    'Cult': 'Culture', 'Eco': 'Economy', 'Envt': 'Environment',
    'Pbh': 'Health', 'Just': 'Justice', 'Pol': 'Politics',
    'Sci': 'Science', 'Secu': 'Security',
}


def load_weekly_proportions():
    """Load weekly mean frame proportions from database."""
    connector = DatabaseConnector()
    query = """
    SELECT
        DATE_TRUNC('week', date)::date as week,
        AVG(cultural_frame) as "Cult",
        AVG(economic_frame) as "Eco",
        AVG(environmental_frame) as "Envt",
        AVG(health_frame) as "Pbh",
        AVG(justice_frame) as "Just",
        AVG(political_frame) as "Pol",
        AVG(scientific_frame) as "Sci",
        AVG(security_frame) as "Secu"
    FROM "CCF_processed_data"
    WHERE date >= '1978-01-01' AND date <= '2024-12-31'
    GROUP BY DATE_TRUNC('week', date)::date
    ORDER BY week
    """
    with connector.get_connection() as conn:
        df = pd.read_sql(query, conn)
    df['date'] = pd.to_datetime(df['week'])
    df = df.drop(columns=['week'])
    print(f"Loaded {len(df)} weekly observations ({df['date'].min().date()} to {df['date'].max().date()})")
    return df


def main():
    # Load data
    df = load_weekly_proportions()

    # Run 4-method dominance analysis
    analyzer = ParadigmDominanceAnalyzer(frame_names=FRAMES, n_workers=4)
    print("Running 4-method dominance analysis...")
    result = analyzer.analyze_paradigm_composition(df, 'overall', show_progress=True)

    # Extract results
    dom_scores = result['dominance_scores']
    dominant_frames = result['dominant_frames']
    paradigm_type = result['paradigm_type']

    # Print summary
    print(f"\nParadigm type: {paradigm_type}")
    print(f"Dominant frames: {', '.join(FRAME_NAMES[f] for f in dominant_frames)}")
    print(f"\nDominance scores (4-method consensus):")
    for f in dom_scores.sort_values('dominance_score', ascending=False).index:
        s = dom_scores.loc[f]
        marker = ' *' if f in dominant_frames else ''
        print(f"  {FRAME_NAMES[f]:12s}: {s['dominance_score']:.4f}  "
              f"(IT={s.get('information_theory_score', 0):.3f}  "
              f"Net={s.get('network_analysis_score', 0):.3f}  "
              f"Caus={s.get('causality_score', 0):.3f}  "
              f"Prop={s.get('proportional_score', 0):.3f}){marker}")

    if dominant_frames:
        threshold = min(dom_scores.loc[f, 'dominance_score'] for f in dominant_frames)
        print(f"\nDominance threshold: {threshold:.4f}")

    # Save
    dom_scores.to_csv(OUT / 'overall_dominance_scores.csv')
    print(f"\nSaved to {OUT / 'overall_dominance_scores.csv'}")

    # Save analysis details
    details = result.get('analysis_details', {})
    import json
    with open(OUT / 'dominance_analysis_details.json', 'w') as f:
        json.dump({
            'dominant_frames': dominant_frames,
            'paradigm_type': paradigm_type,
            'n_dominant': result.get('n_dominant_frames', len(dominant_frames)),
            'threshold': float(threshold) if dominant_frames else None,
            'n_weeks': len(df),
        }, f, indent=2)


if __name__ == '__main__':
    main()
