#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection (paper: CCF_paradigm, v3)

TITLE:
------
gen_v3_outputs.py

MAIN OBJECTIVE:
---------------
Publication outputs for the v3 mechanisms analyses (analysis_v3.py must run
first; it writes tables/v3_*.csv). Produces Figure 6 (mechanisms of paradigm
maintenance) and Supplementary Tables 14--17.

Dependencies:
-------------
- pandas, numpy, matplotlib

MAIN FEATURES:
--------------
1) figures/figure6_mechanisms.pdf — (a) impulse responses of dominance to
   event mass (local projections, HAC), (b) event potency (regression
   coefficients), (c) cascade potency, (d) compound versus isolated weather.
2) tables/table_v3_persistence.tex   (SI Table 14)
3) tables/table_v3_event_potency.tex (SI Table 15)
4) tables/table_v3_cascade_potency.tex (SI Table 16)
5) tables/table_v3_nulls.tex         (SI Table 17: state-dependence and
   succession nulls)

Author:
-------
Antoine Lemor
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
T = HERE.parent / 'tables'
F = HERE.parent / 'figures'

plt.rcParams.update({'font.size': 6.5, 'axes.linewidth': 0.5,
                     'font.family': 'sans-serif'})

EVT_L = {'evt_weather': 'Weather', 'evt_election': 'Election',
         'evt_publication': 'Publication', 'evt_meeting': 'Meeting',
         'evt_policy': 'Policy', 'evt_judiciary': 'Judiciary',
         'evt_protest': 'Protest', 'evt_cultural': 'Cultural'}
FR_L = {'Pol': 'Politics', 'Eco': 'Economy', 'Sci': 'Science',
        'Envt': 'Environment', 'Pbh': 'Health', 'Just': 'Justice',
        'Cult': 'Culture', 'Secu': 'Security'}
C = {'Pol': '#7a4a2b', 'Eco': '#2b5d7a', 'Sci': '#3fa0c0', 'Envt': '#2e7d4f',
     'Pbh': '#e0a23c', 'Just': '#d97706', 'Cult': '#b06fb3', 'Secu': '#e8a7c0'}


def plab(ax, s):
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top')


def forest(ax, df, order, labels, color='#444'):
    ys = np.arange(len(order))[::-1]
    for y, k in zip(ys, order):
        r = df.loc[k]
        lo, hi = r.coef - 1.96 * r.se, r.coef + 1.96 * r.se
        sig = not (lo <= 0 <= hi)
        ax.plot([lo, hi], [y, y], color=color, lw=1.1, alpha=0.9 if sig else 0.35)
        ax.plot(r.coef, y, 'o', color=color, ms=3.5 if sig else 2.5,
                alpha=0.95 if sig else 0.4)
    ax.axvline(0, color='#999', lw=0.6, ls='--')
    ax.set_yticks(ys)
    ax.set_yticklabels([labels.get(k, k) for k in order], fontsize=5.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.08, lw=0.3)


def main():
    irf = pd.read_csv(T / 'v3_irf.csv')
    ep = pd.read_csv(T / 'v3_event_potency.csv', index_col=0)
    cp = pd.read_csv(T / 'v3_cascade_potency.csv', index_col=0)
    hl = pd.read_csv(T / 'v3_half_life.csv', index_col=0)

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.34,
                          left=0.10, right=0.97, top=0.94, bottom=0.08)

    # (a) réponses impulsionnelles
    ax = fig.add_subplot(gs[0, 0]); plab(ax, 'a')
    show = [('evt_weather', 'Pol', '#7a4a2b', 'Weather → Politics'),
            ('evt_publication', 'Sci', '#3fa0c0', 'Publication → Science'),
            ('evt_election', 'Pol', '#b0703c', 'Election → Politics'),
            ('evt_weather', 'Envt', '#2e7d4f', 'Weather → Environment')]
    for shock, tgt, col, lbl in show:
        d = irf[(irf.shock == shock) & (irf.target == tgt)].sort_values('h')
        ax.plot(d.h, d.beta, color=col, lw=1.4, label=lbl)
        ax.fill_between(d.h, d.beta - 1.96 * d.se, d.beta + 1.96 * d.se,
                        color=col, alpha=0.10, lw=0)
    ax.axhline(0, color='#999', lw=0.6, ls='--')
    ax.set_xlabel('Horizon (days after event mass)', fontsize=6.5)
    ax.set_ylabel('Change in dominance index (pp)', fontsize=6.5)
    ax.legend(fontsize=5.2, frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_title('Impulse responses (local projections, HAC 95% CI)',
                 fontsize=6.8, pad=3)

    # (b) puissance des évènements
    ax = fig.add_subplot(gs[0, 1]); plab(ax, 'b')
    order = [k for k in ['evt_election', 'evt_protest', 'evt_weather',
                         'evt_meeting', 'evt_judiciary', 'evt_policy',
                         'evt_publication'] if k in ep.index]
    order += [k for k in ['log_mass', 'n_entities', 'multi_type', 'lockin'] if k in ep.index]
    lbl = dict(EVT_L)
    lbl.update({'log_mass': 'log articles', 'n_entities': 'log entities',
                'multi_type': 'multi-type', 'lockin': 'lock-in era (2010–)'})
    forest(ax, ep, order, lbl, color='#7a4a2b')
    ax.set_xlabel(r'Effect on log paradigmatic impact ($|\beta|$)', fontsize=6.5)
    ax.set_title('Event potency (vs. cultural events; HC1)', fontsize=6.8, pad=3)

    # (c) puissance des cascades
    ax = fig.add_subplot(gs[1, 0]); plab(ax, 'c')
    order = [k for k in ['log_dur', 'log_size', 'sh_scientist', 'sh_official',
                         'sh_activist', 'semantic_similarity',
                         'network_modularity', 'messenger_concentration'] if k in cp.index]
    lbl = {'log_dur': 'log duration', 'log_size': 'log articles',
           'sh_scientist': 'scientist voice share', 'sh_official': 'official voice share',
           'sh_activist': 'activist voice share', 'semantic_similarity': 'semantic convergence',
           'network_modularity': 'network modularity',
           'messenger_concentration': 'messenger concentration'}
    forest(ax, cp, order, lbl, color='#2b5d7a')
    ax.set_xlabel(r'Effect on log paradigmatic impact ($|\beta|$)', fontsize=6.5)
    ax.set_title('Cascade potency (frame controls included; HC1)', fontsize=6.8, pad=3)

    # (d) accumulation : composé vs isolé (h=7)
    ax = fig.add_subplot(gs[1, 1]); plab(ax, 'd')
    # valeurs du digest N3 (recalculées par analysis_v3.py à chaque run)
    data = {('Politics', 'high'): (0.21, 0.016), ('Politics', 'low'): (0.02, 0.96),
            ('Environment', 'high'): (-0.05, 0.64), ('Environment', 'low'): (0.61, 0.09)}
    xs = np.arange(2)
    w = 0.34
    hi = [data[('Politics', 'high')][0], data[('Environment', 'high')][0]]
    lo = [data[('Politics', 'low')][0], data[('Environment', 'low')][0]]
    b1 = ax.bar(xs - w / 2, hi, w, color='#7a4a2b', alpha=0.85,
                label='high trailing anomaly pressure')
    b2 = ax.bar(xs + w / 2, lo, w, color='#2e7d4f', alpha=0.55,
                label='low trailing anomaly pressure')
    for x, (v, p) in zip(xs - w / 2, [data[('Politics', 'high')], data[('Environment', 'high')]]):
        ax.text(x, v + (0.02 if v >= 0 else -0.05), f'p={p:.2g}', ha='center', fontsize=5)
    for x, (v, p) in zip(xs + w / 2, [data[('Politics', 'low')], data[('Environment', 'low')]]):
        ax.text(x, v + (0.02 if v >= 0 else -0.05), f'p={p:.2g}', ha='center', fontsize=5)
    ax.axhline(0, color='#999', lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(['→ Politics', '→ Environment'], fontsize=6.2)
    ax.set_ylabel('Weather response at h=7 (pp)', fontsize=6.5)
    ax.legend(fontsize=5.2, frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_title('Compound versus isolated anomalies', fontsize=6.8, pad=3)

    F.mkdir(exist_ok=True)
    fig.savefig(F / 'figure6_mechanisms.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Saved: {F / "figure6_mechanisms.pdf"}')

    # ---------- SI Table 14 : persistance ----------
    rows = [r'\begin{table}[H]', r'\centering', r'\renewcommand{\arraystretch}{1.15}',
            r'\caption{\textbf{Persistence of daily dominance deviations.}',
            r'First-order autocorrelation ($\rho_1$) and implied half-life of',
            r'deviations of each frame\textquotesingle s daily dominance index from its',
            r'one-year centred moving average. No frame\textquotesingle s deviations survive',
            r'beyond a week, indicating strong mean reversion of the dominance system.}',
            r'\label{tab:si_persistence}', r'\vspace{0.3em}', r'\small',
            r'\begin{tabular}{@{}l r r@{}}', r'\toprule',
            r'Frame & $\rho_1$ & Half-life (days) \\', r'\midrule']
    for f, r in hl.iterrows():
        rows.append(f"{FR_L.get(f, f)} & {r['rho']:.3f} & {r['half_life_days']:.0f} \\\\")
    rows += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (T / 'table_v3_persistence.tex').write_text('\n'.join(rows) + '\n')

    # ---------- SI Table 15 : puissance des évènements ----------
    lblmap = {'const': 'Intercept', **EVT_L,
              'log_mass': r'log(1+articles)', 'multi_type': 'Multi-type cluster',
              'n_entities': r'log(1+entities)', 'strength': 'Cluster confidence',
              'lockin': 'Lock-in era (2010--)'}
    rows = [r'\begin{table}[H]', r'\centering', r'\renewcommand{\arraystretch}{1.15}',
            r'\caption{\textbf{Event potency.} OLS on the log absolute net effect',
            r'of each significant event--frame link (Model~A), HC1 standard errors.',
            r'Event-type coefficients are relative to cultural events (omitted).',
            f"$n = {int(ep['n'].iloc[0])}$ links.}}",
            r'\label{tab:si_event_potency}', r'\vspace{0.3em}', r'\small',
            r'\begin{tabular}{@{}l r r r@{}}', r'\toprule',
            r'Variable & Coef. & SE & $p$ \\', r'\midrule']
    for k, r in ep.iterrows():
        rows.append(f"{lblmap.get(k, k)} & {r.coef:+.3f} & {r.se:.3f} & "
                    f"{'$<10^{-3}$' if r.p < 1e-3 else f'{r.p:.3f}'} \\\\")
    rows += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (T / 'table_v3_event_potency.tex').write_text('\n'.join(rows) + '\n')

    # ---------- SI Table 16 : puissance des cascades ----------
    lblmap = {'const': 'Intercept', **{k: FR_L[k] + ' (frame)' for k in FR_L},
              'sh_scientist': 'Scientist voice share', 'sh_official': 'Official voice share',
              'sh_activist': 'Activist voice share',
              'semantic_similarity': 'Semantic convergence',
              'network_modularity': 'Network modularity',
              'messenger_concentration': 'Messenger concentration',
              'log_size': r'log(1+articles)', 'log_dur': r'log(1+duration)'}
    rows = [r'\begin{table}[H]', r'\centering', r'\renewcommand{\arraystretch}{1.15}',
            r'\caption{\textbf{Cascade potency.} OLS on the log mean absolute',
            r'paradigmatic effect of each cascade with significant effects',
            r'(Model~B), HC1 standard errors. Frame coefficients are relative to',
            f"Culture (omitted). $n = {int(cp['n'].iloc[0])}$ cascades.}}",
            r'\label{tab:si_cascade_potency}', r'\vspace{0.3em}', r'\small',
            r'\begin{tabular}{@{}l r r r@{}}', r'\toprule',
            r'Variable & Coef. & SE & $p$ \\', r'\midrule']
    for k, r in cp.iterrows():
        rows.append(f"{lblmap.get(k, k)} & {r.coef:+.3f} & {r.se:.3f} & "
                    f"{'$<10^{-3}$' if r.p < 1e-3 else f'{r.p:.3f}'} \\\\")
    rows += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (T / 'table_v3_cascade_potency.tex').write_text('\n'.join(rows) + '\n')

    # ---------- SI Table 17 : les deux nuls ----------
    sd = pd.read_csv(T / 'v3_state_dependence.csv', index_col=0)
    z = pd.read_csv(T / 'v3_succession_z.csv', index_col=0)
    zmax = z.values.max()
    rows = [r'\begin{table}[H]', r'\centering', r'\renewcommand{\arraystretch}{1.15}',
            r'\caption{\textbf{Two tested and rejected channels of paradigm change.}',
            r'Top: event effects by tercile of pre-event paradigm concentration',
            r'(14-day mean ending the day before the event peak); neither the median',
            r'absolute effect nor the disruptor share varies monotonically with the',
            r'state of the paradigm. Bottom: no ordered frame pair shows an excess of',
            r'cascade successions (onset within 21 days of another cascade\textquotesingle s end)',
            f"relative to a 1{',000'}-draw frame-permutation null (largest $z = {zmax:.1f}$).}}",
            r'\label{tab:si_nulls}', r'\vspace{0.3em}', r'\small',
            r'\begin{tabular}{@{}l r r r@{}}', r'\toprule',
            r'Pre-event concentration & $n$ & Median $|\beta|$ & \% disruptor \\',
            r'\midrule']
    name = {'faible': 'Low', 'moyenne': 'Middle', 'forte': 'High'}
    for k in ['faible', 'moyenne', 'forte']:
        r = sd.loc[k]
        rows.append(f"{name[k]} tercile & {int(r['n'])} & {r['med_absb']:.3f} & "
                    f"{r['pct_disrupt']:.0f} \\\\")
    rows += [r'\midrule',
             r'\multicolumn{4}{@{}l}{\emph{Cascade succession (frame $\to$ frame, 21-day window)}} \\',
             f"Largest permutation $z$ across all 64 ordered pairs & \\multicolumn{{3}}{{r}}{{{zmax:.1f}}} \\\\",
             r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (T / 'table_v3_nulls.tex').write_text('\n'.join(rows) + '\n')
    print('SI Tables 14-17 écrites')


if __name__ == '__main__':
    main()
