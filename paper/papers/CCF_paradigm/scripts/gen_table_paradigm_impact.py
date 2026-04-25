#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
gen_table_paradigm_impact.py

MAIN OBJECTIVE:
---------------
Compute the net paradigmatic effect (% catalyst - % disruptor) for each
event type x target frame combination, from production StabSel Phase 2
results (stabsel_cluster_dominance.parquet). This measures how events
affect paradigm dominance, which differs from their effect on cascades.
Generates a LaTeX table for Supplementary Information.

Usage:
  python paper/papers/CCF_paradigm/scripts/gen_table_paradigm_impact.py

Output:
  tables/table_paradigm_impact.tex
  tables/paradigm_impact_by_event.csv

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
               'evt_meeting', 'evt_policy', 'evt_judiciary']
EVENT_LABELS = ['Weather', 'Election', 'Publication',
                'Meeting', 'Policy', 'Judiciary']

FRAMES = ['Pol', 'Eco', 'Sci', 'Envt', 'Pbh', 'Just', 'Cult', 'Secu']
FRAME_LABELS_SHORT = ['Pol', 'Eco', 'Sci', 'Env', 'Hea', 'Jus', 'Cul', 'Sec']

MIN_N = 5


def load_cluster_dominance():
    """Load all stabsel_cluster_dominance.parquet files."""
    dfs = []
    for year_dir in sorted(PROD.iterdir()):
        if not year_dir.is_dir():
            continue
        f = year_dir / 'impact_analysis' / 'stabsel_cluster_dominance.parquet'
        if f.exists():
            df = pd.read_parquet(f)
            df['year'] = int(year_dir.name)
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No stabsel_cluster_dominance.parquet files found")
    return pd.concat(dfs, ignore_index=True)


def compute_net_scores(cdom):
    """Compute net paradigmatic score for each event type x frame."""
    data = cdom[(cdom['net_beta'].abs() < 100) & (cdom['role'] != 'inert')]
    rows = []
    for evt, lbl in zip(EVENT_TYPES, EVENT_LABELS):
        row = {'event_type': lbl}
        n_total = len(data[data['dominant_type'] == evt])
        row['n_effects'] = n_total
        for frame in FRAMES:
            sub = data[(data['dominant_type'] == evt) & (data['frame'] == frame)]
            n = len(sub)
            if n >= MIN_N:
                pct_cat = (sub['role'] == 'catalyst').mean() * 100
                net = pct_cat - (100 - pct_cat)
                row[frame] = round(net, 0)
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
        r'\caption{\textbf{Net paradigmatic effect by event type and target frame.}',
        r'Each cell shows \% catalyst $-$ \% disruptor among significant',
        r'event-to-paradigm effects identified by stability selection.',
        r'Positive values indicate that the event type reinforces',
        r'the target frame in the paradigm; negative values indicate erosion.',
        r'Cells with fewer than 5 effects are marked --.}',
        r'\label{tab:si_paradigm_impact}',
        r'\vspace{0.3em}',
        r'\footnotesize',
        r'\begin{tabular}{@{}l r r r r r r r r r@{}}',
        r'\toprule',
        r'Event type & $n$ & \textbf{Pol} & \textbf{Eco} & Sci & Env & Hea & Jus & Cul & Sec \\',
        r'\midrule',
    ]

    for _, row in results.iterrows():
        cells = [row['event_type'], f"{int(row['n_effects'])}"]
        for frame in FRAMES:
            v = row[frame]
            if np.isnan(v):
                cells.append('--')
            else:
                val = f"{v:+.0f}"
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
    print(f"Loading cluster_dominance data from {PROD}...")
    cdom = load_cluster_dominance()
    print(f"  {len(cdom)} total rows")

    results = compute_net_scores(cdom)

    print("\nNet paradigmatic effect (% catalyst - % disruptor):")
    display_cols = ['event_type', 'n_effects'] + FRAMES
    print(results[display_cols].to_string(index=False))

    OUT.mkdir(exist_ok=True)

    tex = generate_latex_table(results)
    tex_path = OUT / 'table_paradigm_impact.tex'
    tex_path.write_text(tex)
    print(f"\nSaved: {tex_path}")

    csv_path = OUT / 'paradigm_impact_by_event.csv'
    results.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
