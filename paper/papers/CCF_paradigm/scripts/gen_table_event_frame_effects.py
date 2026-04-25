#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
gen_table_event_frame_effects.py

MAIN OBJECTIVE:
---------------
Compute the suppressor rate (% of links that suppress rather than amplify)
for each event type x cascade frame combination, from production StabSel
Phase 1 results (cluster_cascade.parquet). Generates a LaTeX table for
Supplementary Information.

Usage:
  python paper/papers/CCF_paradigm/scripts/gen_table_event_frame_effects.py

Output:
  tables/table_event_frame_effects.tex
  tables/event_frame_effects.csv

Author:
-------
Antoine Lemor
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PROD = ROOT / 'results' / 'production'
OUT = Path(__file__).resolve().parent.parent / 'tables'

EVENT_TYPES = ['evt_weather', 'evt_election', 'evt_publication',
               'evt_meeting', 'evt_policy', 'evt_protest',
               'evt_cultural', 'evt_judiciary']
EVENT_LABELS = ['Weather', 'Election', 'Publication',
                'Meeting', 'Policy', 'Protest',
                'Cultural', 'Judiciary']

FRAMES = ['Pol', 'Eco', 'Sci', 'Envt', 'Pbh', 'Just', 'Cult', 'Secu']
FRAME_LABELS = ['Pol', 'Eco', 'Sci', 'Env', 'Hea', 'Jus', 'Cul', 'Sec']


def load_cluster_cascade():
    """Load all cluster_cascade.parquet files across years."""
    dfs = []
    for year_dir in sorted(PROD.iterdir()):
        if not year_dir.is_dir():
            continue
        f = year_dir / 'impact_analysis' / 'cluster_cascade.parquet'
        if f.exists():
            df = pd.read_parquet(f)
            df['year'] = int(year_dir.name)
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No cluster_cascade.parquet files found")
    return pd.concat(dfs, ignore_index=True)


def compute_suppressor_rates(cc):
    """Compute suppressor rate for each event type x cascade frame."""
    rows = []
    for evt, elbl in zip(EVENT_TYPES, EVENT_LABELS):
        sub = cc[cc['dominant_type'] == evt]
        n_total = len(sub)
        if n_total == 0:
            continue
        row = {'event_type': elbl, 'n_links': n_total}
        for frame in FRAMES:
            fsub = sub[sub['cascade_frame'] == frame]
            n = len(fsub)
            if n >= 3:
                sup_rate = (fsub['role'] == 'suppressor').mean() * 100
                row[frame] = round(sup_rate, 0)
                row[f'{frame}_n'] = n
            else:
                row[frame] = np.nan
                row[f'{frame}_n'] = n
        rows.append(row)
    return pd.DataFrame(rows)


def generate_latex_table(results):
    """Generate LaTeX table for SI."""
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{\textbf{Suppressor rate (\%) by event type and cascade frame.}',
        r'Each cell shows the percentage of event-to-cascade links classified',
        r'as suppressors (rather than drivers) by stability selection.',
        r'Cells with fewer than 3 links are marked --.',
        r'Dominant paradigm frames (Pol, Eco) are highlighted.}',
        r'\label{tab:si_event_frame}',
        r'\vspace{0.3em}',
        r'\footnotesize',
        r'\begin{tabular}{@{}l r r r r r r r r r@{}}',
        r'\toprule',
        r'Event type & $n$ & \textbf{Pol} & \textbf{Eco} & Sci & Env & Hea & Jus & Cul & Sec \\',
        r'\midrule',
    ]

    for _, row in results.iterrows():
        cells = [row['event_type'], f"{int(row['n_links'])}"]
        for i, frame in enumerate(FRAMES):
            v = row[frame]
            if np.isnan(v):
                cells.append('--')
            else:
                val = f"{v:.0f}"
                # Bold for dominant frames (Pol, Eco)
                if frame in ('Pol', 'Eco'):
                    cells.append(f"\\textbf{{{val}}}")
                else:
                    cells.append(val)
        lines.append(' & '.join(cells) + r' \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


if __name__ == '__main__':
    print(f"Loading cluster_cascade data from {PROD}...")
    cc = load_cluster_cascade()
    print(f"  {len(cc)} total links")

    results = compute_suppressor_rates(cc)

    print("\nSuppressor rates (%) by event type x cascade frame:")
    display_cols = ['event_type', 'n_links'] + FRAMES
    print(results[display_cols].to_string(index=False))

    OUT.mkdir(exist_ok=True)

    tex = generate_latex_table(results)
    tex_path = OUT / 'table_event_frame_effects.tex'
    tex_path.write_text(tex)
    print(f"\nSaved: {tex_path}")

    # CSV with full detail (including n per cell)
    csv_path = OUT / 'event_frame_effects.csv'
    results.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
