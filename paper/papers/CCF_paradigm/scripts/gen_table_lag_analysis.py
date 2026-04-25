#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
gen_table_lag_analysis.py

MAIN OBJECTIVE:
---------------
Analyse the lag structure of event-to-paradigm effects from the raw
StabSel Phase 2 results. For each event type x target frame combination,
compute the mean lag, the lag distribution, and the sign pattern across
lags. This reveals whether paradigmatic effects are immediate or delayed,
and whether the sign of the effect changes across lags (e.g., suppression
at short lags followed by reinforcement at longer lags).

Usage:
  python paper/papers/CCF_paradigm/scripts/gen_table_lag_analysis.py

Output:
  tables/table_lag_analysis.tex
  tables/lag_analysis.csv

Author:
-------
Antoine Lemor
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PROD = ROOT / 'results' / 'production'
OUT = Path(__file__).resolve().parent.parent / 'tables'

EVENT_TYPES = ['evt_weather', 'evt_election', 'evt_publication',
               'evt_meeting', 'evt_policy', 'evt_judiciary']
EVENT_LABELS = {'evt_weather': 'Weather', 'evt_election': 'Election',
                'evt_publication': 'Publication', 'evt_meeting': 'Meeting',
                'evt_policy': 'Policy', 'evt_judiciary': 'Judiciary'}

FRAMES = ['Pol', 'Eco', 'Sci', 'Envt', 'Pbh', 'Just', 'Cult', 'Secu']
MAX_LAG = 7
MIN_N = 5


def load_all_lag_data():
    """Extract per-lag betas from all raw StabSel paradigm results."""
    records = []
    for year_dir in sorted(PROD.iterdir()):
        if not year_dir.is_dir():
            continue
        pkl_path = year_dir / 'impact_analysis' / 'stabsel_paradigm_results.pkl'
        if not pkl_path.exists():
            continue
        year = int(year_dir.name)
        with open(pkl_path, 'rb') as f:
            raw = pickle.load(f)

        model_a = raw.get('model_a', {})
        for frame, frame_res in model_a.items():
            if frame not in FRAMES:
                continue
            cluster_results = frame_res.get('cluster_results', {})
            for cid, cr in cluster_results.items():
                evt_type = cr.get('dominant_type', '')
                if evt_type not in EVENT_TYPES:
                    continue
                role = cr.get('role', 'inert')
                if role == 'inert':
                    continue
                betas_by_lag = cr.get('betas_by_lag', {})
                net_beta = cr.get('net_beta', 0)
                if abs(net_beta) >= 100:
                    continue
                for lag, beta in betas_by_lag.items():
                    records.append({
                        'year': year,
                        'frame': frame,
                        'cluster_id': cid,
                        'event_type': evt_type,
                        'role': role,
                        'lag': int(lag),
                        'beta': float(beta),
                        'net_beta': float(net_beta),
                    })
    return pd.DataFrame(records)


def compute_lag_summary(df):
    """Compute lag statistics by event type x frame."""
    rows = []
    for evt in EVENT_TYPES:
        for frame in FRAMES:
            sub = df[(df['event_type'] == evt) & (df['frame'] == frame)]
            n = len(sub)
            if n < MIN_N:
                continue

            mean_lag = sub['lag'].mean()
            median_lag = sub['lag'].median()

            # Lag distribution: count per lag
            lag_counts = sub['lag'].value_counts().sort_index()

            # Sign by lag: mean beta per lag
            sign_by_lag = sub.groupby('lag')['beta'].mean()

            # Weighted mean lag (weighted by |beta|)
            weights = sub['beta'].abs()
            if weights.sum() > 0:
                wmean_lag = np.average(sub['lag'], weights=weights)
            else:
                wmean_lag = mean_lag

            # Early (lag 0-2) vs late (lag 3-7) beta
            early = sub[sub['lag'] <= 2]['beta']
            late = sub[sub['lag'] >= 3]['beta']
            early_mean = early.mean() if len(early) > 0 else np.nan
            late_mean = late.mean() if len(late) > 0 else np.nan

            # Sign reversal: early negative, late positive (or vice versa)
            reversal = ''
            if not np.isnan(early_mean) and not np.isnan(late_mean):
                if early_mean < 0 and late_mean > 0:
                    reversal = '$-/+$'
                elif early_mean > 0 and late_mean < 0:
                    reversal = '$+/-$'
                elif early_mean > 0 and late_mean > 0:
                    reversal = '$+/+$'
                else:
                    reversal = '$-/-$'

            rows.append({
                'event_type': EVENT_LABELS[evt],
                'frame': frame,
                'n': n,
                'mean_lag': round(mean_lag, 1),
                'median_lag': round(median_lag, 1),
                'wmean_lag': round(wmean_lag, 1),
                'early_beta': round(early_mean, 3) if not np.isnan(early_mean) else np.nan,
                'late_beta': round(late_mean, 3) if not np.isnan(late_mean) else np.nan,
                'sign_pattern': reversal,
                'n_early': len(early),
                'n_late': len(late),
            })
    return pd.DataFrame(rows)


def generate_latex_table(summary):
    """Generate LaTeX table for SI."""
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{\textbf{Lag structure of event-to-paradigm effects.}',
        r'For each event type and target frame, the table shows the number',
        r'of significant per-lag coefficients ($n$), the beta-weighted mean',
        r'lag (in days), the mean coefficient at early lags (0--2 days)',
        r'and late lags (3--7 days), and the sign pattern across lags.',
        r'A $-/+$ pattern indicates suppression at short lags followed by',
        r'reinforcement at longer lags.}',
        r'\label{tab:si_lag_analysis}',
        r'\vspace{0.3em}',
        r'\footnotesize',
        r'\begin{tabular}{@{}l l r r r r l@{}}',
        r'\toprule',
        r'Event & Frame & $n$ & Wt.\ mean lag & $\bar{\beta}_{0\text{--}2}$',
        r'& $\bar{\beta}_{3\text{--}7}$ & Pattern \\',
        r'\midrule',
    ]

    prev_evt = ''
    for _, row in summary.iterrows():
        evt = row['event_type'] if row['event_type'] != prev_evt else ''
        prev_evt = row['event_type']
        early = f"{row['early_beta']:+.3f}" if not np.isnan(row['early_beta']) else '--'
        late = f"{row['late_beta']:+.3f}" if not np.isnan(row['late_beta']) else '--'
        lines.append(
            f"{evt} & {row['frame']} & {row['n']} & {row['wmean_lag']:.1f} "
            f"& {early} & {late} & {row['sign_pattern']} \\\\"
        )

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


if __name__ == '__main__':
    print(f"Loading raw StabSel results from {PROD}...")
    df = load_all_lag_data()
    print(f"  {len(df)} per-lag beta records from {df['year'].nunique()} years")

    summary = compute_lag_summary(df)
    print(f"\nLag summary ({len(summary)} event-frame combinations):")
    print(summary.to_string(index=False))

    # Highlight key finding: weather -> Pol
    wp = summary[(summary['event_type'] == 'Weather') & (summary['frame'] == 'Pol')]
    if not wp.empty:
        r = wp.iloc[0]
        print(f"\nKey: Weather -> Pol: n={r['n']}, wmean_lag={r['wmean_lag']}, "
              f"early={r['early_beta']}, late={r['late_beta']}, pattern={r['sign_pattern']}")

    OUT.mkdir(exist_ok=True)

    tex = generate_latex_table(summary)
    tex_path = OUT / 'table_lag_analysis.tex'
    tex_path.write_text(tex)
    print(f"\nSaved: {tex_path}")

    csv_path = OUT / 'lag_analysis.csv'
    summary.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
