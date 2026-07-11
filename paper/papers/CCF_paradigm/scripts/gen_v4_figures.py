#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection (paper: CCF_paradigm, v4)

TITLE:
------
gen_v4_figures.py

MAIN OBJECTIVE:
---------------
The four redesigned figures of the v4 paper (two-regime theory of discursive
competition). analysis_v3.py and analysis_v4.py must have been run first
(they write tables/v3_*.csv and tables/v4_*.csv).

  Figure 1  fig_v4_theory.pdf      — the two regimes (conceptual)
  Figure 2  fig_v4_lifecourse.pdf  — one transition and its closure
  Figure 3  fig_v4_polarity.pdf    — the polarity inversion of events
  Figure 4  fig_v4_mechanisms.pdf  — absorption-regime mechanics

Author:
-------
Antoine Lemor
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent.parent
R = ROOT / 'results' / 'production'
T = HERE.parent / 'tables'
F = HERE.parent / 'figures'

plt.rcParams.update({'font.size': 6.5, 'axes.linewidth': 0.5,
                     'font.family': 'sans-serif'})

FRAMES = ['Pol', 'Eco', 'Sci', 'Envt', 'Pbh', 'Just', 'Cult', 'Secu']
FL = {'Pol': 'Politics', 'Eco': 'Economy', 'Sci': 'Science', 'Envt': 'Environment',
      'Pbh': 'Health', 'Just': 'Justice', 'Cult': 'Culture', 'Secu': 'Security'}
C = {'Pol': '#8a4b23', 'Eco': '#2b5d7a', 'Sci': '#3fa0c0', 'Envt': '#2e7d4f',
     'Pbh': '#e0a23c', 'Just': '#d97706', 'Cult': '#b06fb3', 'Secu': '#c96f8f'}
ERA_C = {'contest': '#c96f2b', 'consolidation': '#8899aa', 'lockin': '#37474f'}
ERA_L = {'contest': 'Contest (1988–1995)', 'consolidation': 'Consolidation (1996–2009)',
         'lockin': 'Lock-in (2010–2024)'}


def plab(ax, s):
    ax.text(-0.13, 1.07, s, transform=ax.transAxes, fontsize=9.5,
            fontweight='bold', va='top')


def load_timeline():
    tl = pd.concat([pd.read_parquet(p) for p in
                    sorted(glob.glob(str(R / '*/paradigm_shifts/paradigm_timeline.parquet')))])
    tl['date'] = pd.to_datetime(tl['date'])
    return tl.drop_duplicates('date').sort_values('date').set_index('date').asfreq('D')


def load_cascades():
    rows = []
    for p in sorted(glob.glob(str(R / '*/cascades.json'))):
        with open(p) as f:
            for c in json.load(f):
                if c.get('classification') == 'not_cascade':
                    continue
                rows.append({'frame': c['frame'], 'peak': pd.to_datetime(c.get('peak_date')),
                             'n_articles': c.get('n_articles')})
    return pd.DataFrame(rows)


# ═══════════════════ Figure 1 : la théorie (deux régimes) ═══════════════════
def fig_theory():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    # ---- (a) régime de CONTESTATION ----
    ax = axes[0]; plab(ax, 'a')
    ax.set_title('Contest regime — cascades are the currency of ascent',
                 fontsize=7, pad=6)
    # barre de dominance disputée
    ax.add_patch(FancyBboxPatch((1.0, 8.2), 8.0, 1.1, boxstyle='round,pad=0.08',
                                fc='#f2ede6', ec='#999', lw=0.6))
    ax.text(5, 8.75, 'Paradigm dominance (contested)', ha='center', fontsize=6.2)
    # trois cadres qui cascadent vers la dominance
    for x, f, up in ((2.2, 'Sci', False), (5.0, 'Pol', True), (7.8, 'Eco', True)):
        ax.add_patch(Circle((x, 4.6), 0.95, fc=C[f], alpha=0.75, ec='white', lw=0.8))
        ax.text(x, 4.6, FL[f], ha='center', va='center', fontsize=6, color='white',
                fontweight='bold')
        style = '-' if up else '--'
        ax.add_patch(FancyArrowPatch((x, 5.7), (x, 8.1),
                                     arrowstyle='-|>', mutation_scale=9,
                                     lw=1.6 if up else 1.0, ls=style,
                                     color=C[f], alpha=0.9 if up else 0.5))
        ax.text(x + 0.15, 6.9, 'cascades' if up else 'fading',
                fontsize=5.2, rotation=90, va='center',
                color=C[f], alpha=0.9 if up else 0.6)
    # évènements comme munitions des challengers
    ax.add_patch(FancyBboxPatch((3.1, 1.0), 3.8, 1.15, boxstyle='round,pad=0.08',
                                fc='#e9edf2', ec='#999', lw=0.6))
    ax.text(5, 1.58, 'Focusing events', ha='center', fontsize=6.2)
    for x in (2.2, 5.0, 7.8):
        ax.add_patch(FancyArrowPatch((5, 2.3), (x, 3.5), arrowstyle='-|>',
                                     mutation_scale=7, lw=0.8, color='#777',
                                     connectionstyle='arc3,rad=0.12'))
    ax.text(5, 0.35, 'events amplify challengers (driver ratio 81%);\n'
                     'perturbations persist 7–9 days', ha='center', fontsize=5.4,
            style='italic', color='#555')

    # ---- (b) régime d'ABSORPTION ----
    ax = axes[1]; plab(ax, 'b')
    ax.set_title('Absorption regime — the locked pair no longer cascades',
                 fontsize=7, pad=6)
    # noyau verrouillé
    ax.add_patch(Circle((5, 6.3), 2.15, fc='none', ec='#8a4b23', lw=2.2))
    ax.add_patch(Circle((5, 6.3), 1.75, fc='#f2e8df', ec='none', alpha=0.9))
    ax.text(5, 6.75, 'Politics + Economy', ha='center', fontsize=6.4,
            fontweight='bold', color='#6a3a1b')
    ax.text(5, 6.1, 'dominant 99% of days', ha='center', fontsize=5.4, color='#6a3a1b')
    ax.text(5, 5.55, 'no cascade since 2005', ha='center', fontsize=5.4,
            style='italic', color='#6a3a1b')
    # évènements traduits
    ax.add_patch(FancyBboxPatch((0.6, 1.0), 3.2, 1.15, boxstyle='round,pad=0.08',
                                fc='#e9edf2', ec='#999', lw=0.6))
    ax.text(2.2, 1.58, 'Events, anomalies', ha='center', fontsize=6.2)
    ax.add_patch(FancyArrowPatch((2.6, 2.3), (4.1, 4.9), arrowstyle='-|>',
                                 mutation_scale=9, lw=1.4, color='#555'))
    ax.text(2.5, 3.9, 'translated into the\ndominant register', fontsize=5.4,
            style='italic', color='#555', ha='center')
    # cascades périphériques qui orbitent sans accès
    for ang, f in ((25, 'Secu'), (335, 'Pbh'), (155, 'Just'), (205, 'Cult')):
        x = 5 + 3.35 * np.cos(np.radians(ang))
        y = 6.3 + 2.9 * np.sin(np.radians(ang))
        ax.add_patch(Circle((x, y), 0.62, fc=C[f], alpha=0.7, ec='white', lw=0.6))
        ax.text(x, y, f, ha='center', va='center', fontsize=5.2, color='white',
                fontweight='bold')
        ax.add_patch(FancyArrowPatch((x, y), (5 + 2.35 * np.cos(np.radians(ang)),
                                              6.3 + 2.25 * np.sin(np.radians(ang))),
                                     arrowstyle='-|>', mutation_scale=6, lw=0.7,
                                     ls=':', color=C[f], alpha=0.8))
    ax.text(8.6, 1.55, 'peripheral cascades\ncarry the impact\nbut not the days',
            fontsize=5.4, style='italic', color='#555', ha='center')
    ax.text(5, 0.35, 'events feed the incumbent register (polarity inverted);\n'
                     'perturbations persist 4–5 days', ha='center', fontsize=5.4,
            style='italic', color='#555')

    fig.savefig(F / 'fig_v4_theory.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('fig_v4_theory.pdf')


# ═══════════════ Figure 2 : une transition et sa clôture ═══════════════
def fig_lifecourse(tl, CA):
    dom = tl[[f'paradigm_{f}' for f in FRAMES]].interpolate(limit=7)
    yr = dom.resample('YS').mean()
    yr.index = yr.index.year

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.46, wspace=0.30,
                          left=0.09, right=0.97, top=0.95, bottom=0.09)

    # (a) portrait de phase Politics × Science
    ax = fig.add_subplot(gs[0, 0]); plab(ax, 'a')
    x = yr['paradigm_Sci']; y = yr['paradigm_Pol']
    era_col = np.array([ERA_C['contest'] if yy < 1996 else
                        (ERA_C['consolidation'] if yy < 2010 else ERA_C['lockin'])
                        for yy in yr.index])
    for i in range(len(yr) - 1):
        ax.plot([x.iloc[i], x.iloc[i + 1]], [y.iloc[i], y.iloc[i + 1]],
                color=era_col[i], lw=1.1, alpha=0.8, solid_capstyle='round')
    # flèches de direction clairsemées (une tous les 4 ans, segments non nuls)
    for i in range(0, len(yr) - 1, 4):
        dx, dy = x.iloc[i + 1] - x.iloc[i], y.iloc[i + 1] - y.iloc[i]
        if abs(dx) + abs(dy) < 3e-3:
            continue
        ax.quiver(x.iloc[i], y.iloc[i], dx, dy, angles='xy', scale_units='xy',
                  scale=1, color=era_col[i], width=0.006, headwidth=6,
                  headlength=7, alpha=0.9)
    for yy, dx, dy in ((1979, -0.005, -0.02), (1988, -0.03, 0.012), (1992, 0.012, 0.01),
                       (2005, 0.012, 0.0), (2024, 0.008, -0.01)):
        if yy in yr.index:
            ax.plot(x.loc[yy], y.loc[yy], 'o', color='#222', ms=2.6)
            ax.annotate(str(yy), (x.loc[yy], y.loc[yy]),
                        xytext=(x.loc[yy] + dx, y.loc[yy] + dy), fontsize=5.6)
    lim = [min(x.min(), y.min()) - 0.02, max(x.max(), y.max()) + 0.02]
    ax.plot(lim, lim, ls=':', color='#aaa', lw=0.7)
    ax.text(lim[1] - 0.005, lim[1] - 0.022, 'parity', fontsize=5.2, color='#888',
            rotation=38, ha='right')
    ax.set_xlabel('Science dominance index', fontsize=6.5)
    ax.set_ylabel('Politics dominance index', fontsize=6.5)
    ax.set_title('Phase portrait, 1978–2024', fontsize=7, pad=3)
    for e, lbl in ERA_L.items():
        ax.plot([], [], color=ERA_C[e], lw=1.6, label=lbl)
    ax.legend(fontsize=5.0, frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # (b) raster des cascades + saturation
    ax = fig.add_subplot(gs[0, 1]); plab(ax, 'b')
    order = ['Pol', 'Eco', 'Sci', 'Envt', 'Just', 'Pbh', 'Cult', 'Secu']
    for k, f in enumerate(order):
        d = CA[CA.frame == f]
        ax.scatter(d.peak.dt.year + np.random.default_rng(k).uniform(-0.2, 0.2, len(d)),
                   np.full(len(d), len(order) - k),
                   s=np.clip(d.n_articles / 40, 2, 26), color=C[f], alpha=0.65, lw=0)
    ax.axvline(2004.75, color='#333', lw=0.8, ls='--')
    ax.text(2004.2, 0.35, 'structural break\nSep 2004', fontsize=5.2,
            color='#333', ha='right')
    ax.axvspan(1988, 1995.5, color=ERA_C['contest'], alpha=0.07)
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([FL[f] for f in order[::-1]], fontsize=5.6)
    ax2 = ax.twinx()
    dyr = tl['dominant_frames'].astype(str)
    sat = ((dyr.str.contains('Pol') | dyr.str.contains('Eco'))
           .groupby(tl.index.year).mean() * 100)
    sat = sat[sat.index >= 1982]   # densité de données suffisante à partir de 1982
    ax2.plot(sat.index, sat.values, color='#8a4b23', lw=1.3, alpha=0.85)
    ax2.set_ylabel('Politics ∪ Economy dominant (% of days)', fontsize=6,
                   color='#8a4b23')
    ax2.tick_params(labelsize=5, colors='#8a4b23')
    ax2.set_ylim(0, 105)
    ax.set_xlim(1977, 2025)
    ax.set_title('Cascade record and saturation', fontsize=7, pad=3)
    ax.tick_params(labelsize=5)
    ax.spines['top'].set_visible(False)

    # (c) le raidissement : demi-vies par ère
    ax = fig.add_subplot(gs[1, 0]); plab(ax, 'c')
    hl = pd.read_csv(T / 'v4_half_life_era.csv')
    eras = ['contest', 'consolidation', 'lockin']
    xs = np.arange(len(eras))
    off = {'Pol': -0.24, 'Eco': -0.08, 'Sci': 0.08, 'Envt': 0.24}
    for f in ['Pol', 'Eco', 'Sci', 'Envt']:
        d = hl[hl.frame == f].set_index('era').reindex(eras)
        hi = np.maximum(d.hi, d.half_life)   # percentile bootstrap parfois < point
        ax.errorbar(xs + off[f], d.half_life,
                    yerr=[np.clip(d.half_life - d.lo, 0, None),
                          np.clip(hi - d.half_life, 0, None)],
                    fmt='o', ms=3.2, lw=1.0, capsize=1.6, color=C[f], label=FL[f])
    ax.set_xticks(xs)
    ax.set_xticklabels(['Contest\n1978–95', 'Consolidation\n1996–2009',
                        'Lock-in\n2010–24'], fontsize=5.8)
    ax.set_ylabel('Half-life of dominance deviations (days)', fontsize=6.5)
    ax.legend(fontsize=5.2, frameon=False, ncol=2)
    ax.set_title('The stiffening of the system', fontsize=7, pad=3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.08, lw=0.3)

    # (d) premier ordre par ère : driver et déplacement
    ax = fig.add_subplot(gs[1, 1]); plab(ax, 'd')
    fo = pd.read_csv(T / 'v4_first_order_era.csv').set_index('era').reindex(eras)
    ax.plot(xs, fo.driver, 'o-', color='#2b5d7a', lw=1.4, ms=4, label='Driver ratio')
    ax.plot(xs, fo.displaced, 's--', color='#c0392b', lw=1.2, ms=3.6,
            label='Frame displacement')
    for xi, (dr, di, n) in enumerate(zip(fo.driver, fo.displaced, fo.n)):
        ax.annotate(f'{dr:.0f}%', (xi, dr), xytext=(xi + 0.06, dr + 1.5), fontsize=5.4,
                    color='#2b5d7a')
        ax.annotate(f'{di:.0f}%', (xi, di), xytext=(xi + 0.06, di + 1.5), fontsize=5.4,
                    color='#c0392b')
        ax.annotate(f'n={int(n)}', (xi, 46), fontsize=5.0, color='#777', ha='center')
    ax.set_xticks(xs)
    ax.set_xticklabels(['Contest\n1978–95', 'Consolidation\n1996–2009',
                        'Lock-in\n2010–24'], fontsize=5.8)
    ax.set_ylabel('Share of significant links (%)', fontsize=6.5)
    ax.set_ylim(44, 100)
    ax.legend(fontsize=5.4, frameon=False, loc='center left')
    ax.set_title('Events lose their amplifying role', fontsize=7, pad=3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.08, lw=0.3)

    fig.savefig(F / 'fig_v4_lifecourse.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('fig_v4_lifecourse.pdf')


# ═══════════════ Figure 3 : inversion de polarité ═══════════════
def fig_polarity():
    irf = pd.read_csv(T / 'v4_irf_era.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, (shock, tgt, ttl) in zip(axes, [
            ('evt_weather', 'Pol', 'Weather → Politics dominance'),
            ('evt_publication', 'Sci', 'Publications → Science dominance')]):
        for era in ['contest', 'consolidation', 'lockin']:
            d = irf[(irf.shock == shock) & (irf.target == tgt) &
                    (irf.era == era)].sort_values('h')
            ax.plot(d.h, d.beta, 'o-', color=ERA_C[era], lw=1.3, ms=2.8,
                    label=ERA_L[era])
            ax.fill_between(d.h, d.beta - 1.96 * d.se, d.beta + 1.96 * d.se,
                            color=ERA_C[era], alpha=0.10, lw=0)
        ax.axhline(0, color='#999', lw=0.6, ls='--')
        ax.set_xlabel('Horizon (days)', fontsize=6.5)
        ax.set_ylabel('Response (pp)', fontsize=6.5)
        ax.set_title(ttl, fontsize=7, pad=3)
        ax.legend(fontsize=5.0, frameon=False)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.08, lw=0.3)
    plab(axes[0], 'a'); plab(axes[1], 'b')
    fig.tight_layout()
    fig.savefig(F / 'fig_v4_polarity.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('fig_v4_polarity.pdf')


# ═══════════════ Figure 4 : mécanique du régime d'absorption ═══════════════
def fig_mechanisms():
    ep = pd.read_csv(T / 'v3_event_potency.csv', index_col=0)
    cp = pd.read_csv(T / 'v3_cascade_potency.csv', index_col=0)

    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 3, wspace=0.55, left=0.13, right=0.98,
                          top=0.88, bottom=0.16)

    def forest(ax, df, order, labels, color):
        ys = np.arange(len(order))[::-1]
        for y, k in zip(ys, order):
            r = df.loc[k]
            lo, hi = r.coef - 1.96 * r.se, r.coef + 1.96 * r.se
            sig = not (lo <= 0 <= hi)
            ax.plot([lo, hi], [y, y], color=color, lw=1.1, alpha=0.95 if sig else 0.3)
            ax.plot(r.coef, y, 'o', color=color, ms=3.4 if sig else 2.4,
                    alpha=0.95 if sig else 0.35)
        ax.axvline(0, color='#999', lw=0.6, ls='--')
        ax.set_yticks(ys)
        ax.set_yticklabels([labels[k] for k in order], fontsize=5.6)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.08, lw=0.3)

    ax = fig.add_subplot(gs[0]); plab(ax, 'a')
    order = ['evt_election', 'evt_weather', 'evt_protest', 'evt_meeting',
             'evt_policy', 'evt_publication', 'log_mass', 'n_entities', 'lockin']
    lbl = {'evt_election': 'Election', 'evt_weather': 'Weather', 'evt_protest': 'Protest',
           'evt_meeting': 'Meeting', 'evt_policy': 'Policy', 'evt_publication': 'Publication',
           'log_mass': 'log articles', 'n_entities': 'log entities',
           'lockin': 'lock-in era'}
    forest(ax, ep, [k for k in order if k in ep.index], lbl, '#8a4b23')
    ax.set_xlabel('Effect on log event impact', fontsize=6.2)
    ax.set_title('What makes an event potent', fontsize=6.8, pad=3)

    ax = fig.add_subplot(gs[1]); plab(ax, 'b')
    order = ['log_dur', 'log_size', 'sh_scientist', 'sh_official', 'sh_activist',
             'semantic_similarity']
    lbl = {'log_dur': 'log duration', 'log_size': 'log articles',
           'sh_scientist': 'scientist voices', 'sh_official': 'official voices',
           'sh_activist': 'activist voices', 'semantic_similarity': 'semantic conv.'}
    forest(ax, cp, [k for k in order if k in cp.index], lbl, '#2b5d7a')
    ax.set_xlabel('Effect on log cascade impact', fontsize=6.2)
    ax.set_title('What makes a cascade potent', fontsize=6.8, pad=3)

    ax = fig.add_subplot(gs[2]); plab(ax, 'c')
    data = {('Politics', 'high'): (0.21, 0.016), ('Politics', 'low'): (0.02, 0.96),
            ('Environment', 'high'): (-0.05, 0.64), ('Environment', 'low'): (0.61, 0.09)}
    xs = np.arange(2); w = 0.34
    ax.bar(xs - w / 2, [data[('Politics', 'high')][0], data[('Environment', 'high')][0]],
           w, color='#8a4b23', alpha=0.85, label='compound anomalies')
    ax.bar(xs + w / 2, [data[('Politics', 'low')][0], data[('Environment', 'low')][0]],
           w, color='#2e7d4f', alpha=0.55, label='isolated anomalies')
    for x, key in zip(list(xs - w / 2) + list(xs + w / 2),
                      [('Politics', 'high'), ('Environment', 'high'),
                       ('Politics', 'low'), ('Environment', 'low')]):
        v, p = data[key]
        ax.text(x, v + (0.03 if v >= 0 else -0.07), f'p={p:.2g}', ha='center', fontsize=4.8)
    ax.axhline(0, color='#999', lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(['→ Politics', '→ Environment'], fontsize=6)
    ax.set_ylabel('Weather response, h=7 (pp)', fontsize=6.2)
    ax.legend(fontsize=5.0, frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_title('Accumulation is politicised', fontsize=6.8, pad=3)

    fig.savefig(F / 'fig_v4_mechanisms.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('fig_v4_mechanisms.pdf')


if __name__ == '__main__':
    tl = load_timeline()
    CA = load_cascades()
    fig_theory()
    fig_lifecourse(tl, CA)
    fig_polarity()
    fig_mechanisms()
    print('Fini.')
