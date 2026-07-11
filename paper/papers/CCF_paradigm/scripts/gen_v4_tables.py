#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection (paper: CCF_paradigm, v4)

TITLE:
------
gen_v4_tables.py

MAIN OBJECTIVE:
---------------
Supplementary Tables 18--21 for the v4 paper (two-regime analyses).
analysis_v4.py must have been run first (tables/v4_*.csv).

Author:
-------
Antoine Lemor
"""
from pathlib import Path
import pandas as pd

T = Path(__file__).resolve().parent.parent / 'tables'
FL = {'Pol': 'Politics', 'Eco': 'Economy', 'Sci': 'Science', 'Envt': 'Environment',
      'Pbh': 'Health', 'Just': 'Justice', 'Cult': 'Culture', 'Secu': 'Security'}
ERA_L = {'contest': 'Contest (1978--1995)',
         'consolidation': 'Consolidation (1996--2009)',
         'lockin': 'Lock-in (2010--2024)'}

# ---- Table 18 : production de cascades par cadre et période ----
prod = pd.read_csv(T / 'v4_cascade_production.csv', index_col=0)
periods = [('1978--1987', 1978, 1987), ('1988--1995', 1988, 1995),
           ('1996--2005', 1996, 2005), ('2006--2024', 2006, 2024)]
rows = [r'\begin{table}[H]', r'\centering', r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{\textbf{Cascade production by frame and period.} Number of',
        r'detected cascades peaking in each period. The dominant pair produced',
        r'its entire cascade record while dominance was still being contested,',
        r'and essentially none after the September 2004 structural break in the',
        r'Politics--Science dominance gap (PELT on the monthly series, single',
        r'break stable across penalties 8--20).}',
        r'\label{tab:si_production}', r'\vspace{0.3em}', r'\small',
        r'\begin{tabular}{@{}l r r r r r@{}}', r'\toprule',
        r'Frame & 1978--87 & 1988--95 & 1996--2005 & 2006--24 & Total \\',
        r'\midrule']
for f in ['Pol', 'Eco', 'Sci', 'Envt', 'Just', 'Pbh', 'Cult', 'Secu']:
    s = prod.get(f, pd.Series(dtype=float)).fillna(0)
    vals = [int(s.loc[a:b].sum()) for _, a, b in periods]
    rows.append(f"{FL[f]} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {sum(vals)} \\\\")
pair = prod.get('Pol', 0).fillna(0) + prod.get('Eco', 0).fillna(0)
vals = [int(pair.loc[a:b].sum()) for _, a, b in periods]
rows += [r'\midrule',
         f"Politics + Economy & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {sum(vals)} \\\\",
         r'\bottomrule', r'\end{tabular}', r'\end{table}']
(T / 'table_v4_production.tex').write_text('\n'.join(rows) + '\n')

# ---- Table 19 : demi-vies par ère ----
hl = pd.read_csv(T / 'v4_half_life_era.csv')
rows = [r'\begin{table}[H]', r'\centering', r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{\textbf{Half-life of dominance deviations by era.} Deviations',
        r'of each frame\textquotesingle s daily dominance index from its one-year centred',
        r'moving average; half-life implied by the first-order autocorrelation,',
        r'with 95\% moving-block bootstrap intervals (60-day blocks, 400 draws).',
        r'Perturbations survived seven to nine days during the contest era and',
        r'four to five days afterwards.}',
        r'\label{tab:si_halflife_era}', r'\vspace{0.3em}', r'\small',
        r'\begin{tabular}{@{}l l r l@{}}', r'\toprule',
        r'Era & Frame & Half-life (d) & 95\% CI \\', r'\midrule']
for era in ['contest', 'consolidation', 'lockin']:
    for _, r in hl[hl.era == era].iterrows():
        rows.append(f"{ERA_L[era]} & {FL[r['frame']]} & {r.half_life:.1f} & "
                    f"[{r.lo:.1f}, {max(r.hi, r.half_life):.1f}] \\\\")
    if era != 'lockin':
        rows.append(r'\addlinespace')
rows += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
(T / 'table_v4_halflife.tex').write_text('\n'.join(rows) + '\n')

# ---- Table 20 : IRF par ère ----
irf = pd.read_csv(T / 'v4_irf_era.csv')
rows = [r'\begin{table}[H]', r'\centering', r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{\textbf{Impulse responses by era (local projections, HAC).}',
        r'Response of the target frame\textquotesingle s daily dominance index (percentage',
        r'points) to a one-standard-deviation day of event mass, by horizon and',
        r'era. The polarity of both pathways inverts between the contest era and',
        r'the locked eras.}',
        r'\label{tab:si_irf_era}', r'\vspace{0.3em}', r'\footnotesize',
        r'\begin{tabular}{@{}l l r r r r@{}}', r'\toprule',
        r'Pathway & Era & $h=3$ & $h=7$ & $h=14$ & $h=21$ \\', r'\midrule']
for shock, tgt, lbl in [('evt_weather', 'Pol', 'Weather $\\to$ Politics'),
                        ('evt_publication', 'Sci', 'Publication $\\to$ Science')]:
    for era in ['contest', 'consolidation', 'lockin']:
        d = irf[(irf.shock == shock) & (irf.target == tgt) & (irf.era == era)]
        cells = []
        for h in (3, 7, 14, 21):
            r = d[d.h == h].iloc[0]
            star = '*' if r.p < 0.05 else ('$^\\dagger$' if r.p < 0.10 else '')
            cells.append(f"{r.beta:+.2f}{star}")
        rows.append(f"{lbl} & {ERA_L[era]} & " + ' & '.join(cells) + r' \\')
    rows.append(r'\addlinespace')
rows += [r'\bottomrule', r'\end{tabular}',
         r'\par\vspace{2pt}\footnotesize *$p<0.05$; $\dagger$ $p<0.10$.',
         r'\end{table}']
(T / 'table_v4_irf.tex').write_text('\n'.join(rows) + '\n')

# ---- Table 21 : premier ordre par ère ----
fo = pd.read_csv(T / 'v4_first_order_era.csv')
rows = [r'\begin{table}[H]', r'\centering', r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{\textbf{First-order links by era.} Share of significant',
        r'event-to-cascade links classified as drivers, and share involving frame',
        r'displacement. The amplifying role of events declines monotonically',
        r'across regimes while displacement remains a structural constant.}',
        r'\label{tab:si_first_order_era}', r'\vspace{0.3em}', r'\small',
        r'\begin{tabular}{@{}l r r r@{}}', r'\toprule',
        r'Era & $n$ & Driver (\%) & Displaced (\%) \\', r'\midrule']
for _, r in fo.iterrows():
    rows.append(f"{ERA_L[r.era]} & {int(r.n)} & {r.driver:.1f} & {r.displaced:.1f} \\\\")
rows += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
(T / 'table_v4_first_order.tex').write_text('\n'.join(rows) + '\n')

print('SI Tables 18-21 écrites')
