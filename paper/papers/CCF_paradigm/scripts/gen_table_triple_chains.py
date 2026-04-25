#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
gen_table_triple_chains.py

MAIN OBJECTIVE:
---------------
Compute triple chain statistics: displacement rate, displaced vs aligned
impact, top pathways, and paradigmatic reinforcement by target frame.
Generates a LaTeX table for Supplementary Information.

Usage:
  python paper/papers/CCF_paradigm/scripts/gen_table_triple_chains.py

Output:
  tables/table_triple_chain_stats.tex
  tables/triple_chain_stats.csv

Author:
-------
Antoine Lemor
"""
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
NAT = {'evt_weather': 'Envt', 'evt_meeting': 'Pol', 'evt_policy': 'Pol',
       'evt_publication': 'Sci', 'evt_election': 'Pol', 'evt_judiciary': 'Just',
       'evt_cultural': 'Cult', 'evt_protest': 'Pol'}
EL = {'evt_weather': 'Weather', 'evt_meeting': 'Meeting',
      'evt_publication': 'Publication', 'evt_policy': 'Policy',
      'evt_election': 'Election', 'evt_judiciary': 'Judiciary',
      'evt_cultural': 'Cultural', 'evt_protest': 'Protest'}


def load_triple_chains():
    cc_dfs, cd_dfs = [], []
    for yd in sorted(PROD.iterdir()):
        if not yd.is_dir():
            continue
        try:
            yr = int(yd.name)
        except ValueError:
            continue
        for f, lst in [('impact_analysis/cluster_cascade.parquet', cc_dfs),
                        ('impact_analysis/stabsel_cascade_dominance.parquet', cd_dfs)]:
            p = yd / f
            if p.exists():
                df = pd.read_parquet(p)
                df['year'] = yr
                lst.append(df)
    cc = pd.concat(cc_dfs, ignore_index=True)
    cd = pd.concat(cd_dfs, ignore_index=True)
    cc_sig = cc[cc['role'].isin(['driver', 'suppressor'])]
    cd_sig = cd[(cd['net_beta'].abs() < 100) & (cd['role'] != 'inert')]

    triples = []
    for _, row in cc_sig.iterrows():
        cas, yr = row['cascade_id'], row['year']
        b_m = cd_sig[(cd_sig['cascade_id'] == cas) & (cd_sig['year'] == yr)]
        if b_m.empty:
            continue
        nat = NAT.get(row['dominant_type'], '?')
        for _, b in b_m.iterrows():
            triples.append({
                'event_type': row['dominant_type'],
                'natural_frame': nat,
                'cascade_frame': row['cascade_frame'],
                'paradigm_frame': b['target_frame'],
                'role_B': b['role'],
                'beta_B': b['net_beta'],
                'displaced': row['cascade_frame'] != nat,
            })
    tdf = pd.DataFrame(triples)
    evt_short = {'evt_weather': 'Weather', 'evt_meeting': 'Meeting',
                 'evt_publication': 'Publ.', 'evt_policy': 'Policy',
                 'evt_election': 'Election', 'evt_judiciary': 'Judic.',
                 'evt_cultural': 'Cultural', 'evt_protest': 'Protest'}
    tdf['evt_label'] = tdf['event_type'].map(evt_short).fillna('?')
    tdf['pathway'] = (tdf['evt_label'] + ' > ' +
                      tdf['cascade_frame'] + ' > ' +
                      tdf['paradigm_frame'])
    return tdf


if __name__ == '__main__':
    tdf = load_triple_chains()
    n_total = len(tdf)
    n_disp = tdf['displaced'].sum()
    disp = tdf[tdf['displaced']]
    alig = tdf[~tdf['displaced']]
    u, p_mw = sp_stats.mannwhitneyu(disp['beta_B'].abs(), alig['beta_B'].abs(),
                                     alternative='greater')

    print(f"Total triple chains: {n_total}")
    print(f"Displaced: {n_disp} ({100*n_disp/n_total:.1f}%)")
    print(f"Displaced median |b|: {disp['beta_B'].abs().median():.4f}")
    print(f"Aligned median |b|: {alig['beta_B'].abs().median():.4f}")
    print(f"Mann-Whitney p: {p_mw:.2e}")

    # Top pathways
    ps = tdf.groupby('pathway').agg(
        n=('beta_B', 'count'),
        mean_abs=('beta_B', lambda x: x.abs().mean()),
        pct_cat=('role_B', lambda x: (x == 'catalyst').mean() * 100),
    ).reset_index()

    print("\nTop 10 by frequency:")
    top_freq = ps.nlargest(10, 'n')
    for _, r in top_freq.iterrows():
        print(f"  {r['pathway']:>25}: n={int(r['n'])}, |b|={r['mean_abs']:.4f}, {r['pct_cat']:.0f}% cat")

    print("\nTop 10 by |b| (min n=5):")
    top_impact = ps[ps['n'] >= 5].nlargest(10, 'mean_abs')
    for _, r in top_impact.iterrows():
        print(f"  {r['pathway']:>25}: n={int(r['n'])}, |b|={r['mean_abs']:.4f}, {r['pct_cat']:.0f}% cat")

    # By target frame
    print("\n% catalyst by target frame:")
    for f in FRAMES:
        sub = tdf[tdf['paradigm_frame'] == f]
        pct = (sub['role_B'] == 'catalyst').mean() * 100
        roles = (sub['role_B'] == 'catalyst').values.astype(int)
        rng = np.random.default_rng(42)
        boot = np.array([rng.choice(roles, size=len(roles), replace=True).mean() * 100
                         for _ in range(5000)])
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        sig = '*' if not (ci_lo <= 50 <= ci_hi) else ' '
        print(f"  {FL[f]:>12}: {pct:.1f}% [{ci_lo:.1f}, {ci_hi:.1f}] n={len(sub)} {sig}")

    # LaTeX table: summary stats
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{\textbf{Triple chain statistics.}',
        r'Of ' + f'{n_total:,}' + r' complete triple chains,',
        f'{100*n_disp/n_total:.0f}' + r'\% involve frame displacement.',
        r'Displaced chains produce twice the paradigmatic impact of aligned ones',
        f'(median $|\\beta| = {disp["beta_B"].abs().median():.3f}$ vs',
        f'${alig["beta_B"].abs().median():.3f}$, $p < 10^{{-13}}$).',
        r'Bottom rows show reinforcement rate (\% catalyst)',
        r'by target paradigm frame with bootstrap 95\% CI.}',
        r'\label{tab:si_triple_stats}',
        r'\vspace{0.3em}',
        r'\small',
        r'\begin{tabular}{@{}l r r r@{}}',
        r'\toprule',
        r'Target frame & $n$ & \% catalyst & 95\% CI \\',
        r'\midrule',
    ]
    for f in ['Pol', 'Eco', 'Sci', 'Envt', 'Pbh', 'Just', 'Cult', 'Secu']:
        sub = tdf[tdf['paradigm_frame'] == f]
        pct = (sub['role_B'] == 'catalyst').mean() * 100
        roles = (sub['role_B'] == 'catalyst').values.astype(int)
        rng = np.random.default_rng(42)
        boot = np.array([rng.choice(roles, size=len(roles), replace=True).mean() * 100
                         for _ in range(5000)])
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        lines.append(
            f"{FL[f]} & {len(sub)} & {pct:.1f} & [{ci_lo:.1f}, {ci_hi:.1f}] \\\\"
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    OUT.mkdir(exist_ok=True)
    tex_path = OUT / 'table_triple_chain_stats.tex'
    tex_path.write_text('\n'.join(lines))
    print(f"\nSaved: {tex_path}")

    # CSV with top pathways
    csv_data = []
    for _, r in ps.nlargest(20, 'n').iterrows():
        csv_data.append({
            'pathway': r['pathway'], 'n': int(r['n']),
            'mean_abs_beta': round(r['mean_abs'], 4),
            'pct_catalyst': round(r['pct_cat'], 1),
        })
    csv_path = OUT / 'triple_chain_stats.csv'
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
