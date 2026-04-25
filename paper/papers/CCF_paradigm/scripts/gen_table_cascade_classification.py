#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
gen_table_cascade_classification.py

MAIN OBJECTIVE:
---------------
Compute cascade counts by frame and classification (strong, moderate, weak).
Generates a LaTeX table for Supplementary Information.

Usage:
  python paper/papers/CCF_paradigm/scripts/gen_table_cascade_classification.py

Output:
  tables/table_cascade_classification.tex
  tables/cascade_classification.csv

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

FRAMES = ['Pol', 'Eco', 'Sci', 'Envt', 'Pbh', 'Just', 'Cult', 'Secu']
FL = {'Pol': 'Politics', 'Eco': 'Economy', 'Sci': 'Science',
      'Envt': 'Environment', 'Pbh': 'Health', 'Just': 'Justice',
      'Cult': 'Culture', 'Secu': 'Security'}


def load_cascades():
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
    return pd.DataFrame(cascades)


if __name__ == '__main__':
    cdf = load_cascades()
    # Exclude not_cascade
    cdf = cdf[cdf['classification'] != 'not_cascade']
    print(f"Cascades (excl. not_cascade): {len(cdf)}")

    rows = []
    for frame in FRAMES:
        sub = cdf[cdf['frame'] == frame]
        n = len(sub)
        n_strong = (sub['classification'] == 'strong_cascade').sum()
        n_mod = (sub['classification'] == 'moderate_cascade').sum()
        n_weak = (sub['classification'] == 'weak_cascade').sum()
        med_art = sub['n_articles'].median()
        mean_art = sub['n_articles'].mean()
        rows.append({
            'Frame': FL[frame], 'n': n,
            'Strong': n_strong, 'Moderate': n_mod, 'Weak': n_weak,
            'pct_strong': round(100 * n_strong / n, 1) if n > 0 else 0,
            'med_art': med_art, 'mean_art': round(mean_art, 0),
        })

    # Total row
    n_all = len(cdf)
    rows.append({
        'Frame': 'Total', 'n': n_all,
        'Strong': (cdf['classification'] == 'strong_cascade').sum(),
        'Moderate': (cdf['classification'] == 'moderate_cascade').sum(),
        'Weak': (cdf['classification'] == 'weak_cascade').sum(),
        'pct_strong': round(100 * (cdf['classification'] == 'strong_cascade').sum() / n_all, 1),
        'med_art': cdf['n_articles'].median(),
        'mean_art': round(cdf['n_articles'].mean(), 0),
    })

    rdf = pd.DataFrame(rows)
    print(rdf.to_string(index=False))

    # LaTeX
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{\textbf{Cascade count by frame and classification.}',
        r'\emph{Economy} and \emph{Environment} produce zero strong cascades',
        r'despite being among the most frequent frames.',
        r'\emph{Justice} has the highest strong-cascade rate (10\%).}',
        r'\label{tab:si_cascade_class}',
        r'\vspace{0.3em}',
        r'\small',
        r'\begin{tabular}{@{}l r r r r r r r@{}}',
        r'\toprule',
        r'Frame & $n$ & Strong & Moderate & Weak & \% Strong & Med.\ art. & Mean art. \\',
        r'\midrule',
    ]
    for _, r in rdf.iterrows():
        if r['Frame'] == 'Total':
            lines.append(r'\midrule')
        lines.append(
            f"{r['Frame']} & {int(r['n'])} & {int(r['Strong'])} & "
            f"{int(r['Moderate'])} & {int(r['Weak'])} & "
            f"{r['pct_strong']:.1f} & {r['med_art']:.0f} & {r['mean_art']:.0f} \\\\"
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    OUT.mkdir(exist_ok=True)
    tex_path = OUT / 'table_cascade_classification.tex'
    tex_path.write_text('\n'.join(lines))
    print(f"\nSaved: {tex_path}")

    csv_path = OUT / 'cascade_classification.csv'
    rdf.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
