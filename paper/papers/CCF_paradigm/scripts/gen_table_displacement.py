#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
gen_table_displacement.py

MAIN OBJECTIVE:
---------------
Compute frame displacement and driver ratio statistics by decade and
over the full period, from production StabSel Phase 1 results
(cluster_cascade.parquet). Generates a LaTeX table for Supplementary
Information and prints aggregate statistics cited in the main text.

Usage:
  python paper/papers/CCF_paradigm/scripts/gen_table_displacement.py

Output:
  tables/table_displacement.tex
  tables/displacement_by_decade.csv

Author:
-------
Antoine Lemor
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PROD = ROOT / 'results' / 'production'
OUT = Path(__file__).resolve().parent.parent / 'tables'

# Event type -> natural frame (domain logic)
NATURAL_FRAME = {
    'evt_weather': 'Envt',
    'evt_meeting': 'Pol',
    'evt_policy': 'Pol',
    'evt_publication': 'Sci',
    'evt_election': 'Pol',
    'evt_judiciary': 'Just',
    'evt_cultural': 'Cult',
    'evt_protest': 'Pol',
}

DECADES = {
    '1980s': (1987, 1989),
    '1990s': (1990, 1999),
    '2000s': (2000, 2009),
    '2010s': (2010, 2019),
    '2020s': (2020, 2024),
}


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


def compute_stats(cc, mask=None):
    """Compute displacement and driver ratio for a subset."""
    sub = cc if mask is None else cc[mask]
    n_links = len(sub)
    drivers = sub[sub['role'] == 'driver'].copy()
    drivers['natural'] = drivers['dominant_type'].map(NATURAL_FRAME)
    drivers['displaced'] = drivers['cascade_frame'] != drivers['natural']
    n_drivers = len(drivers)
    disp = drivers['displaced'].mean() * 100 if n_drivers > 0 else np.nan
    dr = n_drivers / n_links * 100 if n_links > 0 else np.nan
    return n_links, n_drivers, round(disp, 1), round(dr, 1)


def compute_regressions(cc):
    """Compute linear trends on yearly series."""
    years = sorted(cc['year'].unique())
    disp_arr, dr_arr, x_arr = [], [], []
    for y in years:
        n, nd, d, r = compute_stats(cc, cc['year'] == y)
        if n >= 5 and not np.isnan(d):
            disp_arr.append(d)
            dr_arr.append(r)
            x_arr.append(y)
    x = np.array(x_arr, dtype=float)
    disp = np.array(disp_arr)
    dr = np.array(dr_arr)

    sl_d, ic_d, r_d, _, _ = stats.linregress(x, disp)
    sl_r, ic_r, r_r, _, _ = stats.linregress(x, dr)

    crossing = (ic_r - ic_d) / (sl_d - sl_r) if sl_d != sl_r else np.nan

    return {
        'disp_start': round(sl_d * x[0] + ic_d, 1),
        'disp_end': round(sl_d * x[-1] + ic_d, 1),
        'disp_r2': round(r_d**2, 2),
        'dr_start': round(sl_r * x[0] + ic_r, 1),
        'dr_end': round(sl_r * x[-1] + ic_r, 1),
        'dr_r2': round(r_r**2, 2),
        'crossing': round(crossing, 1),
    }


def generate_latex_table(decades_data, full_data, reg):
    """Generate LaTeX table by decade with full-period summary."""
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{\textbf{Frame displacement and driver ratio by decade.}',
        r'Displacement: percentage of driver links where the event natural frame',
        r'differs from the cascade frame.',
        r'Driver ratio: percentage of all event-to-cascade links that amplify',
        r'(rather than suppress) the cascade.}',
        r'\label{tab:si_displacement}',
        r'\vspace{0.3em}',
        r'\small',
        r'\begin{tabular}{@{}l r r r r@{}}',
        r'\toprule',
        r'Period & Links & Drivers & Displacement (\%) & Driver ratio (\%) \\',
        r'\midrule',
    ]

    for name, (n, nd, d, r) in decades_data.items():
        lines.append(f"{name} & {n:,} & {nd:,} & {d:.1f} & {r:.1f} \\\\")

    n, nd, d, r = full_data
    lines += [
        r'\midrule',
        f"\\textbf{{1987--2024}} & \\textbf{{{n:,}}} & \\textbf{{{nd:,}}} "
        f"& \\textbf{{{d:.1f}}} & \\textbf{{{r:.1f}}} \\\\",
        r'\midrule',
        f"Linear trend & & & "
        f"{reg['disp_start']} to {reg['disp_end']} "
        f"($R^2={reg['disp_r2']}$) & "
        f"{reg['dr_start']} to {reg['dr_end']} "
        f"($R^2={reg['dr_r2']}$) \\\\",
        f"Crossing & \\multicolumn{{4}}{{l}}"
        f"{{$\\sim${int(round(reg['crossing']))}}} \\\\",
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


if __name__ == '__main__':
    print(f"Loading cluster_cascade data from {PROD}...")
    cc = load_cluster_cascade()
    print(f"  {len(cc)} total links, {cc['year'].nunique()} years")

    # By decade
    decades_data = {}
    for name, (y0, y1) in DECADES.items():
        mask = (cc['year'] >= y0) & (cc['year'] <= y1)
        decades_data[name] = compute_stats(cc, mask)

    # Full period
    full_data = compute_stats(cc)

    # Regressions (on yearly series)
    reg = compute_regressions(cc)

    print(f"\nAggregate statistics:")
    print(f"  Overall displacement rate: {full_data[2]:.1f}%")
    print(f"  Displacement trend: {reg['disp_start']}% to {reg['disp_end']}% (R2={reg['disp_r2']})")
    print(f"  Driver ratio trend: {reg['dr_start']}% to {reg['dr_end']}% (R2={reg['dr_r2']})")
    print(f"  Crossing year: ~{reg['crossing']}")

    print(f"\nBy decade:")
    for name, (n, nd, d, r) in decades_data.items():
        print(f"  {name}: {n} links, {nd} drivers, displacement={d}%, driver_ratio={r}%")

    OUT.mkdir(exist_ok=True)

    tex = generate_latex_table(decades_data, full_data, reg)
    tex_path = OUT / 'table_displacement.tex'
    tex_path.write_text(tex)
    print(f"\nSaved: {tex_path}")

    # CSV
    rows = []
    for name, (n, nd, d, r) in decades_data.items():
        rows.append({'period': name, 'n_links': n, 'n_drivers': nd,
                     'displacement_pct': d, 'driver_ratio_pct': r})
    rows.append({'period': 'Full', 'n_links': full_data[0], 'n_drivers': full_data[1],
                 'displacement_pct': full_data[2], 'driver_ratio_pct': full_data[3]})
    csv_path = OUT / 'displacement_by_decade.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
