#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
gen_table_dominance_decades.py

MAIN OBJECTIVE:
---------------
Compute the percentage of days each frame appears as dominant, by decade,
using the 4-method consensus (dominant_frames column in paradigm_timeline).
Each paradigm_timeline row is one day, produced by a 12-week sliding window
of the ParadigmDominanceAnalyzer (4 scoring methods x 4 threshold methods,
consensus >= 3 votes). Generates a LaTeX table for Supplementary Information.

Usage:
  python paper/papers/CCF_paradigm/scripts/gen_table_dominance_decades.py

Output:
  tables/table_dominance_decades.tex
  tables/dominance_by_decade.csv

Author:
-------
Antoine Lemor
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PROD = ROOT / 'results' / 'production'
OUT = Path(__file__).resolve().parent.parent / 'tables'

# Short tags as stored in dominant_frames (comma-separated)
TAGS = ['Pol', 'Eco', 'Sci', 'Envt', 'Pbh', 'Just', 'Cult', 'Secu']
LABELS = ['Politics', 'Economy', 'Science', 'Environment',
          'Health', 'Justice', 'Culture', 'Security']
SHORT = ['Pol', 'Eco', 'Sci', 'Env', 'Hea', 'Jus', 'Cul', 'Sec']

DECADES = {
    '1980s': (1982, 1989),
    '1990s': (1990, 1999),
    '2000s': (2000, 2009),
    '2010s': (2010, 2019),
    '2020s': (2020, 2024),
}


def load_paradigm_timelines():
    """Load and concatenate all paradigm_timeline.parquet files."""
    dfs = []
    for year_dir in sorted(PROD.iterdir()):
        if not year_dir.is_dir():
            continue
        pt = year_dir / 'paradigm_shifts' / 'paradigm_timeline.parquet'
        if pt.exists():
            dfs.append(pd.read_parquet(pt))
    if not dfs:
        raise FileNotFoundError("No paradigm_timeline.parquet files found")
    full = pd.concat(dfs, ignore_index=True)
    full['date'] = pd.to_datetime(full['date'])
    full = full.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
    return full


def tag_present(series, tag):
    """Check if tag appears as whole token in comma-separated dominant_frames."""
    pattern = rf'(?:^|,){re.escape(tag)}(?:,|$)'
    return series.str.contains(pattern, na=False)


def compute_dominance_percentages(df):
    """Compute % of days each frame is dominant, by decade."""
    df['year'] = df['date'].dt.year
    rows = []
    for decade_name, (y_start, y_end) in DECADES.items():
        sub = df[(df['year'] >= y_start) & (df['year'] <= y_end)]
        n = len(sub)
        if n == 0:
            continue
        row = {'Decade': decade_name, 'n_days': n}
        for tag in TAGS:
            row[tag] = round(100 * tag_present(sub['dominant_frames'], tag).mean(), 1)
        row['Pol_or_Eco'] = round(100 * (
            tag_present(sub['dominant_frames'], 'Pol') |
            tag_present(sub['dominant_frames'], 'Eco')
        ).mean(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def generate_latex_table(results):
    """Generate LaTeX table for Supplementary Information."""
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{\textbf{Percentage of days each frame appears as dominant, by decade.}',
        r'Values indicate the share of days on which each frame is classified as dominant',
        r'by the 4-method consensus (see Methods). Multiple frames can be dominant',
        r'simultaneously. Pol\,$\cup$\,Eco: days where at least one of the two is dominant.}',
        r'\label{tab:si_dominance_decades}',
        r'\vspace{0.3em}',
        r'\footnotesize',
        r'\begin{tabular}{@{}l r rr r rrrr r r@{}}',
        r'\toprule',
        r'& & \multicolumn{2}{c}{\textit{Dominant}} & \textit{Chall.}',
        r'& \multicolumn{4}{c}{\textit{Minor frames}} & & \\',
        r'\cmidrule(lr){3-4} \cmidrule(lr){5-5} \cmidrule(lr){6-9}',
        r'Decade & $n$ & Pol & Eco & Sci & Env & Hea & Jus & Cul & Sec',
        r'& Pol\,$\cup$\,Eco \\',
        r'\midrule',
    ]
    for _, row in results.iterrows():
        cells = [row['Decade'], f"{int(row['n_days']):,}"]
        for tag in TAGS:
            cells.append(f"{row[tag]:.1f}")
        cells.append(f"{row['Pol_or_Eco']:.1f}")
        lines.append(' & '.join(cells) + r' \\')
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


if __name__ == '__main__':
    print(f"Loading paradigm timelines from {PROD}...")
    df = load_paradigm_timelines()
    print(f"  {len(df)} total days ({df['date'].min().date()} to {df['date'].max().date()})")

    results = compute_dominance_percentages(df)
    print("\nDominance percentages by decade (% of days):")
    print(results.to_string(index=False))

    OUT.mkdir(exist_ok=True)

    tex = generate_latex_table(results)
    tex_path = OUT / 'table_dominance_decades.tex'
    tex_path.write_text(tex)
    print(f"\nSaved: {tex_path}")

    csv_path = OUT / 'dominance_by_decade.csv'
    results.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
