#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
gen_figure_cascade_projection.py

MAIN OBJECTIVE:
---------------
UMAP projection of 1,235 cascades in a 20-dimensional feature space
(size, duration, scores, network, semantic, paradigmatic impact).
One panel per event type shows how each type distributes across the
cascade landscape. Point size = log(n_articles), colour = cascade frame.

Usage:
  python paper/papers/CCF_paradigm/scripts/gen_figure_cascade_projection.py

Output:
  figures/figure_cascade_projection.pdf

Author:
-------
Antoine Lemor
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import umap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PROD = ROOT / 'results' / 'production'
OUT = Path(__file__).resolve().parent.parent / 'figures'

FRAMES = ['Pol', 'Eco', 'Sci', 'Envt', 'Pbh', 'Just', 'Cult', 'Secu']
FRAME_COLORS = {
    'Pol': '#795548', 'Eco': '#2980b9', 'Sci': '#00bcd4',
    'Envt': '#4caf50', 'Pbh': '#e91e63', 'Just': '#ff9800',
    'Cult': '#9c27b0', 'Secu': '#f44336'
}
FRAME_LABELS = {
    'Pol': 'Politics', 'Eco': 'Economy', 'Sci': 'Science',
    'Envt': 'Environ.', 'Pbh': 'Health', 'Just': 'Justice',
    'Cult': 'Culture', 'Secu': 'Security'
}
EVT_ORDER = ['evt_weather', 'evt_meeting', 'evt_publication',
             'evt_policy', 'evt_election', 'evt_judiciary']
EVT_LABELS = {
    'evt_weather': 'Weather', 'evt_meeting': 'Meeting',
    'evt_publication': 'Publication', 'evt_policy': 'Policy',
    'evt_election': 'Election', 'evt_judiciary': 'Judiciary'
}


def load_and_project():
    """Load all cascade data, merge with events and paradigm, run UMAP."""
    # Cascades
    cascades = []
    for yd in sorted(PROD.iterdir()):
        if not yd.is_dir():
            continue
        cj = yd / 'cascades.json'
        if not cj.exists():
            continue
        with open(cj) as f:
            for c in json.load(f):
                c['year_dir'] = int(yd.name)
                cascades.append(c)
    cdf = pd.DataFrame(cascades)
    cdf['year'] = pd.to_datetime(cdf['onset_date']).dt.year

    # Event attributions (drivers)
    cc_dfs = []
    for yd in sorted(PROD.iterdir()):
        if not yd.is_dir():
            continue
        f = yd / 'impact_analysis' / 'cluster_cascade.parquet'
        if f.exists():
            df = pd.read_parquet(f)
            cc_dfs.append(df)
    cc = pd.concat(cc_dfs, ignore_index=True)
    drivers = cc[cc['role'] == 'driver']
    cascade_drivers = drivers.groupby('cascade_id').agg(
        main_event=('dominant_type', lambda x: x.value_counts().index[0]),
        n_drivers=('cluster_id', 'count'),
    ).reset_index()

    # Paradigm impact
    cd_dfs = []
    for yd in sorted(PROD.iterdir()):
        if not yd.is_dir():
            continue
        f = yd / 'impact_analysis' / 'stabsel_cascade_dominance.parquet'
        if f.exists():
            df = pd.read_parquet(f)
            cd_dfs.append(df)
    cd = pd.concat(cd_dfs, ignore_index=True)
    cd_active = cd[(cd['net_beta'].abs() < 100) & (cd['role'] != 'inert')]
    cascade_paradigm = cd_active.groupby('cascade_id').agg(
        paradigm_impact=('net_beta', lambda x: x.abs().mean()),
    ).reset_index()

    # Merge — keep ALL cascades (left join)
    merged = cdf.merge(cascade_drivers, on='cascade_id', how='left')
    merged = merged.merge(cascade_paradigm, on='cascade_id', how='left')
    merged['paradigm_impact'] = merged['paradigm_impact'].fillna(0)
    merged['main_event'] = merged['main_event'].fillna('none')

    # Feature matrix
    features = [
        'n_articles', 'duration_days', 'n_journalists', 'n_media',
        'burst_intensity', 'adoption_velocity', 'baseline_mean', 'peak_proportion',
        'score_temporal', 'score_participation', 'score_convergence', 'score_source',
        'semantic_similarity', 'cross_media_alignment', 'novelty_decay_rate',
        'media_coordination', 'messenger_concentration',
        'network_density', 'network_modularity',
        'paradigm_impact',
    ]
    X = merged[features].copy()
    for col in ['n_articles', 'duration_days', 'n_journalists', 'burst_intensity']:
        X[col] = np.log1p(X[col])
    X = X.fillna(0)

    X_scaled = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    merged['umap_x'] = embedding[:, 0]
    merged['umap_y'] = embedding[:, 1]

    return merged


def main():
    print("Computing UMAP projection...")
    df = load_and_project()
    n_attributed = (df['main_event'] != 'none').sum()
    print(f"  {len(df)} cascades projected ({n_attributed} with event attribution)")

    df['size'] = np.log1p(df['n_articles']) * 3

    fig = plt.figure(figsize=(7.2, 8.0))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.28, wspace=0.22,
                           left=0.06, right=0.97, bottom=0.06, top=0.95)

    xlim = (df['umap_x'].min() - 0.5, df['umap_x'].max() + 0.5)
    ylim = (df['umap_y'].min() - 0.5, df['umap_y'].max() + 0.5)

    def plab(ax, label):
        ax.text(-0.10, 1.06, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')

    # (a) All cascades by frame
    ax = fig.add_subplot(gs[0, 0])
    plab(ax, 'a')
    for frame in FRAMES:
        sub = df[df['frame'] == frame]
        ax.scatter(sub['umap_x'], sub['umap_y'], c=FRAME_COLORS[frame],
                   s=sub['size'], alpha=0.45, edgecolors='white', linewidths=0.1)
    ax.set_title('All cascades (by frame)', fontsize=7, pad=2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=4.5)
    ax.set_ylabel('UMAP 2', fontsize=6)

    # (b) All cascades by paradigmatic impact
    ax = fig.add_subplot(gs[0, 1])
    plab(ax, 'b')
    vmax = df['paradigm_impact'].quantile(0.95)
    sc = ax.scatter(df['umap_x'], df['umap_y'], c=df['paradigm_impact'],
                    s=df['size'], alpha=0.5, cmap='magma_r',
                    edgecolors='white', linewidths=0.1,
                    vmin=0, vmax=vmax)
    ax.set_title('Paradigmatic impact', fontsize=7, pad=2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=4.5)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.ax.tick_params(labelsize=4)
    cb.set_label(r'Mean $|\beta|$', fontsize=5)

    # (c) Size distribution by event type
    ax = fig.add_subplot(gs[0, 2])
    plab(ax, 'c')
    evt_colors = {
        'Weather': '#d62728', 'Meeting': '#2ca02c', 'Publication': '#ff7f0e',
        'Policy': '#9467bd', 'Election': '#1f77b4', 'Judiciary': '#8c564b',
    }
    for evt in EVT_ORDER:
        sub = df[df['main_event'] == evt]
        if len(sub) < 5:
            continue
        vals = np.log10(sub['n_articles'].clip(lower=1))
        ax.hist(vals, bins=20, alpha=0.35, density=True,
                label=EVT_LABELS[evt], color=evt_colors[EVT_LABELS[evt]])
    ax.set_xlabel(r'$\log_{10}$(articles)', fontsize=6)
    ax.set_ylabel('Density', fontsize=6)
    ax.set_title('Cascade size by trigger', fontsize=7, pad=2)
    ax.tick_params(labelsize=4.5)
    ax.legend(fontsize=4.5, frameon=False)

    # (d-i) One panel per event type
    for idx, evt in enumerate(EVT_ORDER):
        row = 1 + idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        label = chr(ord('d') + idx)
        plab(ax, label)

        # Grey background
        other = df[df['main_event'] != evt]
        ax.scatter(other['umap_x'], other['umap_y'],
                   c='#e0e0e0', s=1.5, alpha=0.25, zorder=1)

        # Highlighted
        sub = df[df['main_event'] == evt]
        for frame in FRAMES:
            fsub = sub[sub['frame'] == frame]
            if len(fsub) == 0:
                continue
            ax.scatter(fsub['umap_x'], fsub['umap_y'], c=FRAME_COLORS[frame],
                       s=fsub['size'] * 1.5, alpha=0.7,
                       edgecolors='white', linewidths=0.2, zorder=2)

        ax.set_title(f'{EVT_LABELS[evt]} (n={len(sub)})', fontsize=7, pad=2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=4.5)
        if col == 0:
            ax.set_ylabel('UMAP 2', fontsize=6)
        if row == 2:
            ax.set_xlabel('UMAP 1', fontsize=6)

    # Frame legend
    handles = [Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=FRAME_COLORS[f], markersize=4,
                       label=FRAME_LABELS[f]) for f in FRAMES]
    fig.legend(handles=handles, loc='lower center', ncol=8, fontsize=5.5,
               frameon=False, bbox_to_anchor=(0.5, -0.005))

    OUT.mkdir(exist_ok=True)
    outpath = OUT / 'figure_cascade_projection.pdf'
    fig.savefig(outpath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {outpath}")


if __name__ == '__main__':
    main()
