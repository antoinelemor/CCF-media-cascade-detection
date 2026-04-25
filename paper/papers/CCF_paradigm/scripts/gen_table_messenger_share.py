#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
gen_table_messenger_share.py

MAIN OBJECTIVE:
---------------
Compute mean messenger share by cascade frame, excluding security and
legal messengers. Generates a LaTeX table for Supplementary Information.

Usage:
  python paper/papers/CCF_paradigm/scripts/gen_table_messenger_share.py

Output:
  tables/table_messenger_share.tex
  tables/messenger_share.csv

Author:
-------
Antoine Lemor
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PROD = ROOT / 'results' / 'production'
OUT = Path(__file__).resolve().parent.parent / 'tables'

FRAMES = ['Pol', 'Eco', 'Sci', 'Envt', 'Pbh', 'Just', 'Cult', 'Secu']
FL = {'Pol': 'Politics', 'Eco': 'Economy', 'Sci': 'Science',
      'Envt': 'Environment', 'Pbh': 'Health', 'Just': 'Justice',
      'Cult': 'Culture', 'Secu': 'Security'}

MESSENGERS = ['msg_official', 'msg_scientist', 'msg_economic', 'msg_activist',
              'msg_cultural', 'msg_health', 'msg_social']
ML = {'msg_official': 'Official', 'msg_scientist': 'Scientist',
      'msg_economic': 'Economic', 'msg_activist': 'Activist',
      'msg_cultural': 'Cultural', 'msg_health': 'Health',
      'msg_social': 'Social'}
MS = {'msg_official': 'Off', 'msg_scientist': 'Sci', 'msg_economic': 'Eco',
      'msg_activist': 'Act', 'msg_cultural': 'Cul', 'msg_health': 'Hea',
      'msg_social': 'Soc'}


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

    # Extract messenger shares per cascade
    rows = []
    for _, c in cdf.iterrows():
        dm = c.get('dominant_messengers', {})
        if not isinstance(dm, dict) or not dm:
            continue
        total = sum(dm.get(m, 0) for m in MESSENGERS)
        if total == 0:
            continue
        row = {'frame': c['frame'], 'year': c['year']}
        for msg in MESSENGERS:
            row[msg] = dm.get(msg, 0) / total * 100
        rows.append(row)
    mdf = pd.DataFrame(rows)
    print(f"Cascades with messenger data: {len(mdf)}")

    # Per-frame mean shares
    results = []
    for frame in FRAMES:
        sub = mdf[mdf['frame'] == frame]
        row = {'Frame': FL[frame], 'n': len(sub)}
        for msg in MESSENGERS:
            row[ML[msg]] = round(sub[msg].mean(), 1)
        results.append(row)

    # Overall
    row = {'Frame': 'All', 'n': len(mdf)}
    for msg in MESSENGERS:
        row[ML[msg]] = round(mdf[msg].mean(), 1)
    results.append(row)

    rdf = pd.DataFrame(results)
    print(rdf.to_string(index=False))

    # Temporal trends (rolling 5-year)
    print("\nTemporal trends (all cascades):")
    years = sorted(mdf['year'].unique())
    for msg in MESSENGERS:
        yearly = []
        for y in years:
            sub = mdf[mdf['year'] == y]
            if len(sub) >= 5:
                yearly.append({'year': y, 'val': sub[msg].mean()})
        if len(yearly) < 10:
            continue
        ydf = pd.DataFrame(yearly)
        sl, ic, r, p, se = sp_stats.linregress(ydf['year'], ydf['val'])
        if p < 0.05:
            print(f"  {ML[msg]:>12}: {sl:+.2f} pp/yr, R2={r**2:.2f}, p={p:.2e}")

    # LaTeX table
    msg_labels = [ML[m] for m in MESSENGERS]
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{\textbf{Mean messenger share (\%) by cascade frame.}',
        r'Each cell shows the mean percentage of messenger voice within',
        r'cascades of that frame, excluding security and legal messengers.',
        r'Official messengers dominate most frames. Economic messengers',
        r'account for only 19\% even within Economic cascades.}',
        r'\label{tab:si_messenger_share}',
        r'\vspace{0.3em}',
        r'\footnotesize',
        r'\begin{tabular}{@{}l r ' + 'r ' * len(MESSENGERS) + r'@{}}',
        r'\toprule',
        r'Frame & $n$ & ' + ' & '.join(MS[m] for m in MESSENGERS) + r' \\',
        r'\midrule',
    ]
    for _, r in rdf.iterrows():
        if r['Frame'] == 'All':
            lines.append(r'\midrule')
        cells = [r['Frame'], str(int(r['n']))]
        for msg in MESSENGERS:
            cells.append(f"{r[ML[msg]]:.1f}")
        lines.append(' & '.join(cells) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    OUT.mkdir(exist_ok=True)
    tex_path = OUT / 'table_messenger_share.tex'
    tex_path.write_text('\n'.join(lines))
    print(f"\nSaved: {tex_path}")

    csv_path = OUT / 'messenger_share.csv'
    rdf.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
