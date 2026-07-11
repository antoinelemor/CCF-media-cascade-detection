#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection (paper: CCF_paradigm)

TITLE:
------
gen_table_sensitivity.py

MAIN OBJECTIVE:
---------------
Supplementary Table 13: robustness of the cascade set to the burst-admission
threshold. Reports (a) how many production cascades survive stricter final
elevation floors (peak proportion / local baseline >= 1.2, 1.35, 1.5), and
(b) the recovery rate of an INDEPENDENT strict re-run (admission ratio 1.5)
within the production set (same frame, peak within +/- 3 days).

Dependencies:
-------------
- pandas, numpy (parquet reading)

MAIN FEATURES:
--------------
1) Reads cascades.parquet from results/production (main set, admission 1.2)
   and results/production_strict (independent re-run, admission 1.5).
2) Computes the final elevation ratio peak_proportion / baseline_mean.
3) Emits tables/table_sensitivity.tex + sensitivity.csv.

Author:
-------
Antoine Lemor
"""
from pathlib import Path
import glob

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / 'tables'


def load(tree):
    dfs = []
    for p in sorted(glob.glob(str(ROOT / 'results' / tree / '*' / 'cascades.parquet'))):
        df = pd.read_parquet(p)
        if len(df) and 'frame' in df.columns:
            if 'classification' in df.columns:
                df = df[df['classification'] != 'not_cascade']
            dfs.append(df[['frame', 'peak_date', 'peak_proportion', 'baseline_mean']])
    out = pd.concat(dfs, ignore_index=True)
    out['peak_date'] = pd.to_datetime(out['peak_date'])
    out['ratio'] = out['peak_proportion'] / out['baseline_mean'].replace(0, np.nan)
    return out


def main():
    prod = load('production')
    strict = load('production_strict')

    n = len(prod)
    floors = [(1.2, (prod['ratio'] >= 1.2).sum()),
              (1.35, (prod['ratio'] >= 1.35).sum()),
              (1.5, (prod['ratio'] >= 1.5).sum())]

    by_frame = {f: d['peak_date'].values for f, d in prod.groupby('frame')}
    hit = 0
    for _, r in strict.iterrows():
        arr = by_frame.get(r['frame'])
        if arr is not None and (np.abs((arr - r['peak_date'].to_datetime64())
                                       .astype('timedelta64[D]').astype(int)) <= 3).any():
            hit += 1

    med = prod['ratio'].median()
    tex = [
        r'\begin{table}[H]', r'\centering',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{\textbf{Robustness of the cascade set to the admission threshold.}',
        r'Top: share of the production set (burst admission at $1.2\times$ the local',
        r'baseline) that survives stricter floors on the \emph{final} elevation ratio',
        r'(peak proportion / local baseline; median %.2f). Bottom: recovery of an' % med,
        r'independent strict re-run of the full pipeline (admission $1.5\times$)',
        r'within the production set (same frame, peak within $\pm 3$ days). The',
        r'admission threshold gates candidate windows \emph{before} clustering and',
        r'scoring, so the pipeline is not strictly monotone in this parameter;',
        r'recovery below 100\% reflects this non-monotonicity, not disagreement on',
        r'cascade strength.}',
        r'\label{tab:si_sensitivity}', r'\vspace{0.3em}', r'\small',
        r'\begin{tabular}{@{}l r r@{}}', r'\toprule',
        r'Criterion & $n$ & \% of set \\', r'\midrule',
        r'Production set (admission $\geq 1.2$) & %d & 100 \\' % n,
    ]
    for f, k in floors[1:]:
        tex.append(r'Final elevation $\geq %.2f$ & %d & %.0f \\' % (f, k, k / n * 100))
    tex += [
        r'\midrule',
        r'Independent strict re-run (admission $\geq 1.5$) & %d & \\' % len(strict),
        r'\quad recovered in production set ($\pm 3$ d, same frame) & %d & %.0f \\'
        % (hit, hit / len(strict) * 100),
        r'\bottomrule', r'\end{tabular}', r'\end{table}',
    ]
    OUT.mkdir(exist_ok=True)
    (OUT / 'table_sensitivity.tex').write_text('\n'.join(tex) + '\n')
    pd.DataFrame({'criterion': ['floor_1.2', 'floor_1.35', 'floor_1.5',
                                'strict_n', 'strict_recovered'],
                  'value': [floors[0][1], floors[1][1], floors[2][1],
                            len(strict), hit]}).to_csv(OUT / 'sensitivity.csv', index=False)
    print(f"production: {n} | >=1.35: {floors[1][1]} | >=1.5: {floors[2][1]} "
          f"| strict: {len(strict)} recouvrées {hit} ({hit/len(strict)*100:.0f}%)")
    print(f"Saved: {OUT / 'table_sensitivity.tex'}")


if __name__ == '__main__':
    main()
