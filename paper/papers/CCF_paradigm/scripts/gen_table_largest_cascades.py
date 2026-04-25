#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
gen_table_largest_cascades.py

MAIN OBJECTIVE:
---------------
Generate a table of the 10 largest cascades by article count, with their
paradigmatic impact, showing that the biggest cascades are paradigmatically
inert. Generates a LaTeX table for Supplementary Information.

Usage:
  python paper/papers/CCF_paradigm/scripts/gen_table_largest_cascades.py

Output:
  tables/table_largest_cascades.tex
  tables/largest_cascades.csv

Author:
-------
Antoine Lemor
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PROD = ROOT / 'results' / 'production'
OUT = Path(__file__).resolve().parent.parent / 'tables'

FL = {'Pol': 'Politics', 'Eco': 'Economy', 'Sci': 'Science',
      'Envt': 'Environment', 'Pbh': 'Health', 'Just': 'Justice',
      'Cult': 'Culture', 'Secu': 'Security'}


def load_data():
    cascades = []
    for yd in sorted(PROD.iterdir()):
        if not yd.is_dir():
            continue
        cj = yd / 'cascades.json'
        if not cj.exists():
            continue
        with open(cj) as f:
            for c in json.load(f):
                c['year'] = int(yd.name)
                cascades.append(c)
    cdf = pd.DataFrame(cascades)
    cdf['onset'] = pd.to_datetime(cdf['onset_date']).dt.strftime('%Y-%m-%d')

    cd_dfs = []
    for yd in sorted(PROD.iterdir()):
        if not yd.is_dir():
            continue
        f = yd / 'impact_analysis' / 'stabsel_cascade_dominance.parquet'
        if f.exists():
            cd_dfs.append(pd.read_parquet(f))
    cd = pd.concat(cd_dfs, ignore_index=True)
    active = cd[(cd['net_beta'].abs() < 100) & (cd['role'] != 'inert')]
    cascade_para = active.groupby('cascade_id').agg(
        paradigm_impact=('net_beta', lambda x: x.abs().mean()),
        n_effects=('role', 'count'),
    ).reset_index()
    return cdf.merge(cascade_para, on='cascade_id', how='left')


if __name__ == '__main__':
    merged = load_data()
    merged['paradigm_impact'] = merged['paradigm_impact'].fillna(0)

    top = merged.nlargest(10, 'n_articles')

    print("Top 10 largest cascades:")
    for _, r in top.iterrows():
        impact = f"{r['paradigm_impact']:.4f}" if r['paradigm_impact'] > 0 else 'none'
        print(f"  {r['onset']} | {FL.get(r['frame'], r['frame']):>10} | "
              f"{r['n_articles']:>5} art | {r['duration_days']:.0f}d | "
              f"score={r['total_score']:.3f} | impact={impact}")

    # LaTeX table
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{\textbf{Ten largest cascades by article count.}',
        r'The largest cascades are overwhelmingly \emph{Political},',
        r'yet their paradigmatic impact is negligible.',
        r'Copenhagen 2009 (rank 1) and the 2021 mega-cascade (rank 2)',
        r'produced no significant paradigmatic effects despite',
        r'mobilising thousands of articles across all outlets.}',
        r'\label{tab:si_largest_cascades}',
        r'\vspace{0.3em}',
        r'\small',
        r'\begin{tabular}{@{}r l l r r r r@{}}',
        r'\toprule',
        r'Rank & Onset & Frame & Articles & Days & Score & $|\beta|$ \\',
        r'\midrule',
    ]
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        impact = f"{r['paradigm_impact']:.4f}" if r['paradigm_impact'] > 0 else '--'
        lines.append(
            f"{rank} & {r['onset']} & {FL.get(r['frame'], r['frame'])} & "
            f"{int(r['n_articles']):,} & {int(r['duration_days'])} & "
            f"{r['total_score']:.3f} & {impact} \\\\"
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    OUT.mkdir(exist_ok=True)
    tex_path = OUT / 'table_largest_cascades.tex'
    tex_path.write_text('\n'.join(lines))
    print(f"\nSaved: {tex_path}")

    csv_path = OUT / 'largest_cascades.csv'
    top[['cascade_id', 'frame', 'onset', 'n_articles', 'duration_days',
         'total_score', 'paradigm_impact']].to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
