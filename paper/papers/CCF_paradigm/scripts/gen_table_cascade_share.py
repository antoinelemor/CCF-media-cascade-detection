#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
gen_table_cascade_share.py

MAIN OBJECTIVE:
---------------
Compute the share of cascade articles captured by each frame, by decade.
Generates a LaTeX table for Supplementary Information.

Usage:
  python paper/papers/CCF_paradigm/scripts/gen_table_cascade_share.py

Output:
  tables/table_cascade_share.tex
  tables/cascade_share_by_decade.csv

Author:
-------
Antoine Lemor
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PROD = ROOT / 'results' / 'production'
OUT = Path(__file__).resolve().parent.parent / 'tables'

FRAMES = ['Pol', 'Eco', 'Sci', 'Envt', 'Pbh', 'Just', 'Cult', 'Secu']
FRAME_SHORT = ['Pol', 'Eco', 'Sci', 'Env', 'Hea', 'Jus', 'Cul', 'Sec']

DECADES = {
    '1980s': (1984, 1989), '1990s': (1990, 1999),
    '2000s': (2000, 2009), '2010s': (2010, 2019),
    '2020s': (2020, 2024),
}


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
    print(f"Loaded {len(cdf)} cascades")

    rows = []
    for name, (y0, y1) in DECADES.items():
        sub = cdf[(cdf['year'] >= y0) & (cdf['year'] <= y1)]
        total = sub['n_articles'].sum()
        row = {'Decade': name, 'n_cascades': len(sub), 'total_articles': total}
        for frame in FRAMES:
            fsub = sub[sub['frame'] == frame]
            row[frame] = round(fsub['n_articles'].sum() / total * 100, 1) if total > 0 else 0
        rows.append(row)
    results = pd.DataFrame(rows)

    # Linear trends
    yearly = []
    for y in sorted(cdf['year'].unique()):
        sub = cdf[cdf['year'] == y]
        total = sub['n_articles'].sum()
        if total == 0:
            continue
        row = {'year': y}
        for frame in FRAMES:
            row[frame] = sub[sub['frame'] == frame]['n_articles'].sum() / total * 100
        yearly.append(row)
    ydf = pd.DataFrame(yearly)

    print("\nShares by decade:")
    print(results.to_string(index=False))

    print("\nLinear trends:")
    for frame in FRAMES:
        sl, ic, r, p, se = stats.linregress(ydf['year'], ydf[frame])
        if p < 0.05:
            print(f"  {frame}: {sl:+.2f} pp/yr, R2={r**2:.2f}, p={p:.2e}")

    # LaTeX table
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{\textbf{Share of cascade articles by frame and decade (\%).}',
        r'Each cell shows the percentage of all cascade articles in that decade',
        r'belonging to cascades of the given frame.}',
        r'\label{tab:si_cascade_share}',
        r'\vspace{0.3em}',
        r'\footnotesize',
        r'\begin{tabular}{@{}l r r ' + 'r ' * len(FRAMES) + r'@{}}',
        r'\toprule',
        r'Decade & $n$ & Articles & ' + ' & '.join(FRAME_SHORT) + r' \\',
        r'\midrule',
    ]
    for _, row in results.iterrows():
        cells = [row['Decade'], f"{int(row['n_cascades'])}", f"{int(row['total_articles']):,}"]
        for frame in FRAMES:
            cells.append(f"{row[frame]:.1f}")
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    OUT.mkdir(exist_ok=True)
    tex_path = OUT / 'table_cascade_share.tex'
    tex_path.write_text('\n'.join(lines))
    print(f"\nSaved: {tex_path}")

    csv_path = OUT / 'cascade_share_by_decade.csv'
    results.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
