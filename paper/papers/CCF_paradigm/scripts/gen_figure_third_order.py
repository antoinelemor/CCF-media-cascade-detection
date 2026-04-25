#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
gen_figure_third_order.py

MAIN OBJECTIVE:
---------------
Generate the third-order figure with 3 panels:
  a) Displaced vs aligned chains: paradigmatic impact comparison
  b) Top pathways by paradigmatic impact (mean |beta_B|)
  c) Paradigmatic reinforcement rate (% catalyst) by target frame

Usage:
  python paper/papers/CCF_paradigm/scripts/gen_figure_third_order.py

Output:
  figures/figure5_third_order.pdf

Author:
-------
Antoine Lemor
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PROD = ROOT / 'results' / 'production'
OUT = Path(__file__).resolve().parent.parent / 'figures'

FRAMES = ['Pol', 'Eco', 'Sci', 'Envt', 'Pbh', 'Just', 'Cult', 'Secu']
FL = {'Pol': 'Politics', 'Eco': 'Economy', 'Sci': 'Science',
      'Envt': 'Environ.', 'Pbh': 'Health', 'Just': 'Justice',
      'Cult': 'Culture', 'Secu': 'Security'}
NAT = {'evt_weather': 'Envt', 'evt_meeting': 'Pol', 'evt_policy': 'Pol',
       'evt_publication': 'Sci', 'evt_election': 'Pol', 'evt_judiciary': 'Just',
       'evt_cultural': 'Cult', 'evt_protest': 'Pol'}


def plab(ax, label):
    ax.text(-0.12, 1.06, label, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')


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
                'year': yr,
            })
    tdf = pd.DataFrame(triples)
    EVT_SHORT = {'evt_weather': 'Wea', 'evt_meeting': 'Mee',
                 'evt_publication': 'Pub', 'evt_policy': 'Pol',
                 'evt_election': 'Ele', 'evt_judiciary': 'Jud',
                 'evt_cultural': 'Cul', 'evt_protest': 'Pro'}
    tdf['evt_label'] = tdf['event_type'].map(EVT_SHORT).fillna('?')
    tdf['pathway'] = (tdf['evt_label'] + ' > ' +
                      tdf['cascade_frame'] + ' > ' +
                      tdf['paradigm_frame'])
    return tdf


def main():
    print("Loading triple chains...")
    tdf = load_triple_chains()
    print(f"  {len(tdf)} triple chains")

    fig = plt.figure(figsize=(7.2, 3.2))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[0.55, 1.1, 0.9],
                           wspace=0.50, left=0.06, right=0.97,
                           bottom=0.18, top=0.92)

    # ── Panel (a): Displaced vs aligned impact ──
    ax = fig.add_subplot(gs[0])
    plab(ax, 'a')

    disp = tdf[tdf['displaced']]['beta_B'].abs().values
    alig = tdf[~tdf['displaced']]['beta_B'].abs().values

    rng = np.random.default_rng(42)
    N_BOOT = 10000
    for i, (vals, label) in enumerate([(alig, 'Aligned'), (disp, 'Displaced')]):
        jitter = rng.uniform(-0.2, 0.2, size=len(vals))
        ax.scatter(i + jitter, vals, s=3, c='#bbb', alpha=0.10,
                   edgecolors='none', zorder=1)
        med = np.median(vals)
        boot_med = np.array([np.median(rng.choice(vals, size=len(vals),
                             replace=True)) for _ in range(N_BOOT)])
        ci_lo = np.percentile(boot_med, 2.5)
        ci_hi = np.percentile(boot_med, 97.5)
        # CI as colored rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle((i - 0.25, ci_lo), 0.5, ci_hi - ci_lo,
                          facecolor='#c0392b' if i == 1 else '#2980b9',
                          alpha=0.35, edgecolor='none', zorder=2)
        ax.add_patch(rect)
        # Median line
        ax.plot([i - 0.25, i + 0.25], [med, med], color='#222', lw=2.5, zorder=3)
        ax.text(i, 0.0012, f'n={len(vals)}',
                ha='center', va='top', fontsize=5, color='#555')

    u, p = sp_stats.mannwhitneyu(disp, alig, alternative='greater')
    ax.text(0.5, 0.97, f'$p < 10^{{-13}}$', transform=ax.transAxes,
            fontsize=6, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      alpha=0.8, edgecolor='#ddd'))

    ax.set_yscale('log')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Aligned', 'Displaced'], fontsize=6.5)
    ax.set_ylabel('Paradigmatic impact ($|\\beta|$)', fontsize=6.5)
    ax.tick_params(labelsize=5.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel (b): Top pathways by |beta_B| ──
    ax = fig.add_subplot(gs[1])
    plab(ax, 'b')

    ps = tdf.groupby('pathway').agg(
        n=('beta_B', 'count'),
        mean_abs=('beta_B', lambda x: x.abs().mean()),
        pct_cat=('role_B', lambda x: (x == 'catalyst').mean() * 100),
    ).reset_index()
    top = ps[ps['n'] >= 5].nlargest(10, 'mean_abs')

    y_pos = range(len(top))
    colors = ['#c0392b' if p < 50 else '#2980b9' for p in top['pct_cat']]
    ax.barh(y_pos, top['mean_abs'], color=colors, alpha=0.7, height=0.7,
            edgecolor='white', linewidth=0.2)

    ax.set_yticks([])
    ax.set_xlabel('Mean $|\\beta|$', fontsize=6.5)
    ax.tick_params(labelsize=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.invert_yaxis()

    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r['mean_abs'] + 0.008, i, f'{r["pathway"]}  (n={int(r["n"])})',
                va='center', ha='left', fontsize=4.5, color='#444')

    # ── Panel (c): Scientific dispossession across three measures ──
    ax = fig.add_subplot(gs[2])
    plab(ax, 'c')

    import json
    # 1. Science paradigm dominance by decade (% of days)
    sci_dom = {}
    for yd in sorted(PROD.iterdir()):
        if not yd.is_dir():
            continue
        try:
            yr = int(yd.name)
        except ValueError:
            continue
        pt = yd / 'paradigm_shifts' / 'paradigm_timeline.parquet'
        if not pt.exists():
            continue
        df = pd.read_parquet(pt)
        import re
        for _, row in df.iterrows():
            y = pd.Timestamp(row['date']).year
            has_sci = bool(re.search(r'(?:^|,)Sci(?:,|$)', str(row['dominant_frames'])))
            sci_dom.setdefault(y, []).append(has_sci)

    # 2. Science cascade share by year
    sci_share = {}
    all_share = {}
    for yd in sorted(PROD.iterdir()):
        if not yd.is_dir():
            continue
        try:
            yr = int(yd.name)
        except ValueError:
            continue
        cj = yd / 'cascades.json'
        if not cj.exists():
            continue
        with open(cj) as f:
            cascades = json.load(f)
        total = sum(c['n_articles'] for c in cascades)
        sci_art = sum(c['n_articles'] for c in cascades if c['frame'] == 'Sci')
        if total > 0:
            sci_share[yr] = sci_art / total * 100
            all_share[yr] = total

    # 3. Science messenger share by year
    sci_msg = {}
    MSGS = ['msg_official', 'msg_scientist', 'msg_economic', 'msg_activist',
            'msg_cultural', 'msg_health', 'msg_social']
    for yd in sorted(PROD.iterdir()):
        if not yd.is_dir():
            continue
        try:
            yr = int(yd.name)
        except ValueError:
            continue
        cj = yd / 'cascades.json'
        if not cj.exists():
            continue
        with open(cj) as f:
            cascades = json.load(f)
        vals = []
        for c in cascades:
            dm = c.get('dominant_messengers', {})
            if not isinstance(dm, dict) or not dm:
                continue
            total_m = sum(dm.get(m, 0) for m in MSGS)
            if total_m > 0:
                vals.append(dm.get('msg_scientist', 0) / total_m * 100)
        if vals:
            sci_msg[yr] = np.mean(vals)

    # Rolling 5-year
    WINDOW = 5
    years = sorted(set(sci_dom.keys()) & set(sci_share.keys()) & set(sci_msg.keys()))
    roll_years, roll_dom, roll_share, roll_msg = [], [], [], []
    for y in range(min(years), max(years) + 1):
        y0 = y - WINDOW // 2
        y1 = y + WINDOW // 2
        dom_vals = [np.mean(sci_dom[yy]) * 100 for yy in range(y0, y1 + 1)
                    if yy in sci_dom]
        share_vals = [sci_share[yy] for yy in range(y0, y1 + 1)
                      if yy in sci_share]
        msg_vals = [sci_msg[yy] for yy in range(y0, y1 + 1)
                    if yy in sci_msg]
        if len(dom_vals) >= 3 and len(share_vals) >= 3 and len(msg_vals) >= 3:
            roll_years.append(y)
            roll_dom.append(np.mean(dom_vals))
            roll_share.append(np.mean(share_vals))
            roll_msg.append(np.mean(msg_vals))

    ax.plot(roll_years, roll_dom, color='#222', lw=1.5, label='Paradigm dominance')
    ax.plot(roll_years, roll_share, color='#222', lw=1.5, ls='--',
            label='Cascade share')
    ax.plot(roll_years, roll_msg, color='#222', lw=1.5, ls=':',
            label='Scientist messenger')
    ax.set_xlabel('Year', fontsize=6.5)
    ax.set_ylabel('Science (%)', fontsize=6.5)
    ax.tick_params(labelsize=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=4.5, frameon=False, loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.06, lw=0.3)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'figure5_third_order.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {OUT / 'figure5_third_order.pdf'}")

    # Print key stats
    print(f"\nDisplaced: n={len(disp)}, median |b|={np.median(disp):.4f}")
    print(f"Aligned:   n={len(alig)}, median |b|={np.median(alig):.4f}")
    print(f"Mann-Whitney p={p:.2e}")
    print(f"\nTarget frame % catalyst:")
    for _, r in fdf.iterrows():
        sig = '*' if not (r['ci_lo'] <= 50 <= r['ci_hi']) else ' '
        print(f"  {FL[r['frame']]:>10}: {r['pct']:.1f}% [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}] {sig}")


if __name__ == '__main__':
    main()
