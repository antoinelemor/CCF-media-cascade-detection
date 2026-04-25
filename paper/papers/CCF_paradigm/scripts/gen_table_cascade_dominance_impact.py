#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
gen_table_cascade_dominance_impact.py

MAIN OBJECTIVE:
---------------
Analyse the relationship between cascade frame dominance and paradigmatic
impact. Tests whether non-dominant frames produce disproportionate
paradigmatic effects, controlling for cascade size.

Usage:
  python paper/papers/CCF_paradigm/scripts/gen_table_cascade_dominance_impact.py

Output:
  tables/table_cascade_dominance_impact.tex
  tables/cascade_dominance_impact.csv

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
DOM_SCORES = {'Pol': 0.925, 'Eco': 0.584, 'Sci': 0.560, 'Envt': 0.336,
              'Pbh': 0.322, 'Just': 0.273, 'Cult': 0.237, 'Secu': 0.129}


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

    cd_dfs = []
    for yd in sorted(PROD.iterdir()):
        if not yd.is_dir():
            continue
        f = yd / 'impact_analysis' / 'stabsel_cascade_dominance.parquet'
        if f.exists():
            cd_dfs.append(pd.read_parquet(f))
    cd = pd.concat(cd_dfs, ignore_index=True)
    data = cd[(cd['net_beta'].abs() < 100) & (cd['role'] != 'inert')]
    cascade_para = data.groupby('cascade_id').agg(
        paradigm_impact=('net_beta', lambda x: x.abs().mean()),
    ).reset_index()
    return cdf.merge(cascade_para, on='cascade_id', how='inner')


if __name__ == '__main__':
    merged = load_data()
    merged['dom_score'] = merged['frame'].map(DOM_SCORES)
    merged['is_dominant'] = merged['frame'].isin(['Pol', 'Eco'])
    print(f"Cascades with paradigmatic effects: {len(merged)}")

    # Per-frame summary
    rows = []
    for f in FRAMES:
        sub = merged[merged['frame'] == f]
        rows.append({
            'Frame': FL[f],
            'Dominance': DOM_SCORES[f],
            'n': len(sub),
            'med_articles': sub['n_articles'].median(),
            'med_impact': sub['paradigm_impact'].median(),
            'mean_impact': sub['paradigm_impact'].mean(),
        })
    results = pd.DataFrame(rows)
    print(results.to_string(index=False))

    # Correlations
    rho_raw, p_raw = sp_stats.spearmanr(merged['dom_score'],
                                         merged['paradigm_impact'])
    log_art = np.log10(merged['n_articles'].clip(lower=1))
    log_imp = np.log10(merged['paradigm_impact'])
    dom_s = merged['dom_score']
    res_dom = dom_s - np.polyval(np.polyfit(log_art, dom_s, 1), log_art)
    res_imp = log_imp - np.polyval(np.polyfit(log_art, log_imp, 1), log_art)
    rho_partial, p_partial = sp_stats.spearmanr(res_dom, res_imp)

    dom = merged[merged['is_dominant']]
    nondom = merged[~merged['is_dominant']]
    u, p_mw = sp_stats.mannwhitneyu(dom['paradigm_impact'],
                                     nondom['paradigm_impact'])

    print(f"\nDominance vs impact: rho={rho_raw:.3f}, p={p_raw:.2e}")
    print(f"Partial (controlling for size): rho={rho_partial:.3f}, p={p_partial:.2e}")
    print(f"Dominant vs non-dominant: U={u}, p={p_mw:.2e}")
    print(f"  Dominant: n={len(dom)}, median |b|={dom['paradigm_impact'].median():.4f}")
    print(f"  Non-dom:  n={len(nondom)}, median |b|={nondom['paradigm_impact'].median():.4f}")

    # LaTeX table
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{\textbf{Cascade frame dominance and paradigmatic impact.}',
        r'Frames ordered by composite dominance score.',
        r'Non-dominant frames produce significantly higher paradigmatic impact',
        r'(Spearman $\rho = ' + f'{rho_raw:.2f}' + r'$, $p < 10^{-46}$),',
        r'and the relationship holds after controlling for cascade size',
        r'(partial $\rho = ' + f'{rho_partial:.2f}' + r'$, $p < 10^{-42}$).}',
        r'\label{tab:si_dominance_impact}',
        r'\vspace{0.3em}',
        r'\small',
        r'\begin{tabular}{@{}l r r r r r@{}}',
        r'\toprule',
        r'Frame & Dominance & $n$ & Med.\ articles & Med.\ $|\beta|$ & Mean $|\beta|$ \\',
        r'\midrule',
    ]
    for _, r in results.iterrows():
        lines.append(
            f"{r['Frame']} & {r['Dominance']:.3f} & {r['n']} & "
            f"{r['med_articles']:.0f} & {r['med_impact']:.4f} & "
            f"{r['mean_impact']:.4f} \\\\"
        )
    lines += [
        r'\midrule',
        f"Dominant (Pol, Eco) & & {len(dom)} & {dom['n_articles'].median():.0f} & "
        f"{dom['paradigm_impact'].median():.4f} & {dom['paradigm_impact'].mean():.4f} \\\\",
        f"Non-dominant & & {len(nondom)} & {nondom['n_articles'].median():.0f} & "
        f"{nondom['paradigm_impact'].median():.4f} & {nondom['paradigm_impact'].mean():.4f} \\\\",
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    OUT.mkdir(exist_ok=True)
    tex_path = OUT / 'table_cascade_dominance_impact.tex'
    tex_path.write_text('\n'.join(lines))
    print(f"\nSaved: {tex_path}")

    csv_path = OUT / 'cascade_dominance_impact.csv'
    results.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
