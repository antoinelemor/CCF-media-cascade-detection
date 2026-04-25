#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
gen_figure_second_order.py

MAIN OBJECTIVE:
---------------
Generate the second-order figure with 3 panels:
  a) Strip plot: paradigmatic impact by cascade frame (log scale)
  b) Scatter: cascade size vs paradigmatic impact (log-log, LOWESS)
  c) Messenger share evolution over time (rolling 5-year)

All values from production data. Nothing hardcoded.

Usage:
  python paper/papers/CCF_paradigm/scripts/gen_figure_second_order.py

Output:
  figures/figure4_second_order.pdf

Author:
-------
Antoine Lemor
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
import statsmodels.api as sm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PROD = ROOT / 'results' / 'production'
OUT = Path(__file__).resolve().parent.parent / 'figures'

FRAMES = ['Pol', 'Eco', 'Sci', 'Envt', 'Pbh', 'Just', 'Cult', 'Secu']
FL = {
    'Pol': 'Politics', 'Eco': 'Economy', 'Sci': 'Science',
    'Envt': 'Environ.', 'Pbh': 'Health', 'Just': 'Justice',
    'Cult': 'Culture', 'Secu': 'Security'
}
FC = {
    'Pol': '#795548', 'Eco': '#2980b9', 'Sci': '#00bcd4',
    'Envt': '#4caf50', 'Pbh': '#e91e63', 'Just': '#ff9800',
    'Cult': '#9c27b0', 'Secu': '#f44336'
}

MESSENGERS = [
    'msg_official', 'msg_scientist', 'msg_economic', 'msg_activist',
    'msg_cultural', 'msg_health', 'msg_social'
]
ML = {
    'msg_official': 'Official', 'msg_scientist': 'Scientist',
    'msg_economic': 'Economic', 'msg_activist': 'Activist',
    'msg_cultural': 'Cultural', 'msg_health': 'Health',
    'msg_social': 'Social'
}
MC = {
    'msg_official': '#c0392b', 'msg_scientist': '#2980b9',
    'msg_economic': '#27ae60', 'msg_activist': '#f39c12',
    'msg_cultural': '#8e44ad', 'msg_health': '#e91e63',
    'msg_social': '#1abc9c'
}

WINDOW = 5  # rolling window in years


def load_cascades():
    """Load all cascades from JSON."""
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
    return pd.DataFrame(cascades)


def load_paradigm_impact():
    """Load per-cascade paradigmatic impact from StabSel Phase 2."""
    dfs = []
    for yd in sorted(PROD.iterdir()):
        if not yd.is_dir():
            continue
        f = yd / 'impact_analysis' / 'stabsel_cascade_dominance.parquet'
        if f.exists():
            dfs.append(pd.read_parquet(f))
    cd = pd.concat(dfs, ignore_index=True)
    active = cd[(cd['net_beta'].abs() < 100) & (cd['role'] != 'inert')]
    return active.groupby('cascade_id').agg(
        paradigm_impact=('net_beta', lambda x: x.abs().mean()),
    ).reset_index()


def plab(ax, label):
    ax.text(-0.12, 1.06, label, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')


def main():
    print("Loading data...")
    cdf = load_cascades()
    cascade_para = load_paradigm_impact()
    merged = cdf.merge(cascade_para, on='cascade_id', how='left')
    merged['paradigm_impact'] = merged['paradigm_impact'].fillna(0)
    active = merged[merged['paradigm_impact'] > 0].copy()
    print(f"  {len(cdf)} cascades, {len(active)} with paradigmatic effects")

    fig = plt.figure(figsize=(7.2, 3.2))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[0.7, 1.0, 1.0],
                           wspace=0.35, left=0.06, right=0.97,
                           bottom=0.16, top=0.94)

    # ── Panel (a): Strip plot of paradigmatic impact by frame ──
    ax = fig.add_subplot(gs[0])
    plab(ax, 'a')

    medians = {f: active[active['frame'] == f]['paradigm_impact'].median()
               for f in FRAMES}
    frame_order = sorted(FRAMES, key=lambda f: medians[f], reverse=True)

    rng = np.random.default_rng(42)
    for i, f in enumerate(frame_order):
        vals = active[active['frame'] == f]['paradigm_impact'].values
        jitter = rng.uniform(-0.22, 0.22, size=len(vals))
        ax.scatter(vals, i + jitter, s=4, c='#888',
                   alpha=0.20, edgecolors='none', zorder=1)
        med = np.median(vals)
        q25, q75 = np.percentile(vals, [25, 75])
        ax.plot([med, med], [i - 0.28, i + 0.28], color='#222',
                lw=2.0, zorder=3)
        ax.plot([q25, q75], [i, i], color='#222', lw=0.8,
                zorder=2, alpha=0.5)
        n = len(vals)
        all_sub = cdf[cdf['frame'] == f]
        med_art = all_sub['n_articles'].median()
        ax.text(0.0015, i, f' n={n}, {med_art:.0f} art.',
                ha='left', va='center', fontsize=4, color='#555')

    ax.set_xscale('log')
    ax.set_yticks(range(len(frame_order)))
    ax.set_yticklabels([FL[f] for f in frame_order], fontsize=5.5)
    ax.set_xlabel('Paradigmatic impact\n(mean $|\\beta|$)', fontsize=6.5)
    ax.tick_params(labelsize=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.1, lw=0.3)
    ax.invert_yaxis()

    # ── Panel (b): Size vs impact scatter ──
    ax = fig.add_subplot(gs[1])
    plab(ax, 'b')

    log_art = np.log10(active['n_articles'].clip(lower=1).values)
    log_imp = np.log10(active['paradigm_impact'].values)

    for frame in FRAMES:
        sub = active[active['frame'] == frame]
        ax.scatter(np.log10(sub['n_articles'].clip(lower=1)),
                   np.log10(sub['paradigm_impact']),
                   s=6, c=FC[frame], alpha=0.25,
                   edgecolors='none', zorder=2)

    lowess = sm.nonparametric.lowess(log_imp, log_art, frac=0.5)
    ax.plot(lowess[:, 0], lowess[:, 1], color='#222', lw=1.5, zorder=3)

    rho, p = sp_stats.spearmanr(active['n_articles'], active['paradigm_impact'])
    ax.text(0.97, 0.97, f'$\\rho$ = {rho:.2f}\nn = {len(active)}',
            transform=ax.transAxes, fontsize=5.5, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.85, edgecolor='#ddd'))

    handles = [Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=FC[f], markersize=3,
                       label=FL[f]) for f in FRAMES]
    ax.legend(handles=handles, fontsize=3.8, frameon=False,
              loc='lower left', ncol=2, columnspacing=0.4,
              handletextpad=0.1)

    ax.set_xlabel('Cascade size (log$_{10}$ articles)', fontsize=6.5)
    ax.set_ylabel('Paradigmatic impact\n(log$_{10}$ mean $|\\beta|$)',
                  fontsize=6.5)
    ax.tick_params(labelsize=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.06, lw=0.3)

    # ── Panel (c): Messenger evolution ──
    ax = fig.add_subplot(gs[2])
    plab(ax, 'c')

    msg_rows = []
    for _, c in cdf.iterrows():
        dm = c.get('dominant_messengers', {})
        if not isinstance(dm, dict) or not dm:
            continue
        total = sum(dm.get(m, 0) for m in MESSENGERS)
        if total == 0:
            continue
        row = {'year': c['year']}
        for msg in MESSENGERS:
            row[msg] = dm.get(msg, 0) / total
        msg_rows.append(row)
    mdf = pd.DataFrame(msg_rows)

    years = sorted(mdf['year'].unique())
    roll_years = []
    roll = {m: [] for m in MESSENGERS}
    for y in range(min(years), max(years) + 1):
        y0 = y - WINDOW // 2
        y1 = y + WINDOW // 2
        sub = mdf[(mdf['year'] >= y0) & (mdf['year'] <= y1)]
        if len(sub) < 10:
            continue
        roll_years.append(y)
        for msg in MESSENGERS:
            roll[msg].append(sub[msg].mean() * 100)

    roll_years = np.array(roll_years)
    msg_order = sorted(MESSENGERS,
                       key=lambda m: roll[m][-1], reverse=True)

    for msg in msg_order:
        vals = np.array(roll[msg])
        ax.plot(roll_years, vals, color=MC[msg], lw=1.5, alpha=0.85)
        # Add regression line + R2 for significant trends
        sl, ic, r, p_reg, se = sp_stats.linregress(roll_years, vals)
        if p_reg < 0.05:
            y_fit = sl * roll_years + ic
            ax.plot(roll_years, y_fit, color=MC[msg], lw=0.7, ls='--',
                    alpha=0.5)
            # R2 label near end of line
            ax.text(roll_years[-1] + 0.5, y_fit[-1],
                    f'$R^2$={r**2:.2f}', fontsize=3.5, color=MC[msg],
                    va='center', alpha=0.7)

    handles = [Line2D([0], [0], color=MC[m], lw=1.5,
                       label=ML[m]) for m in msg_order]
    ax.legend(handles=handles, fontsize=4.5, frameon=False,
              loc='center right')

    ax.set_xlabel('Year', fontsize=6.5)
    ax.set_ylabel('Messenger share (%)', fontsize=6.5)
    ax.tick_params(labelsize=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.06, lw=0.3)
    ax.set_xlim(roll_years[0], roll_years[-1])

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'figure4_second_order.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {OUT / 'figure4_second_order.pdf'}")

    # Print key stats
    print(f"\nPanel a: {len(active)} cascades with effects")
    print(f"Panel b: rho = {rho:.3f}, p = {p:.2e}")
    print("Panel c messenger trends:")
    for msg in msg_order:
        vals = np.array(roll[msg])
        sl, ic, r, p_m, se = sp_stats.linregress(roll_years, vals)
        if p_m < 0.05:
            print(f"  {ML[msg]:>12}: {sl:+.2f} pp/yr, R2={r**2:.2f}")


if __name__ == '__main__':
    main()
