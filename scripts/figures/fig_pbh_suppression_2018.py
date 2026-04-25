#!/usr/bin/env python3
"""
PROJECT: CCF-media-cascade-detection
TITLE: fig_pbh_suppression_2018.py

Visualizes Phase 1 impact roles (driver / late_support / suppressor) for the
three 2018 Pbh (Public Health) cascades.

Panel A — Timeline: daily health_frame signal with cascade windows, cluster
          peaks annotated by role.
Panel B — Horizontal lollipop chart: impact score by cluster, coloured by role,
          with event descriptions from article content.

Author: Antoine Lemor
"""

import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.patheffects as PathEffects
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cascade_detector.analysis.unified_impact import UnifiedImpactAnalyzer

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
})

ROLE_COLORS = {
    'driver': '#27AE60',
    'late_support': '#F39C12',
    'suppressor': '#E74C3C',
}

CASCADE_LABELS = {
    'Pbh_20180227_1': 'Pbh #1 (Mar)',
    'Pbh_20180621_5': 'Pbh #2 (Jul)',
    'Pbh_20180803_9': 'Pbh #3 (Aug)',
}
CASCADE_COLORS = {
    'Pbh_20180227_1': '#2E86C1',
    'Pbh_20180621_5': '#D4AC0D',
    'Pbh_20180803_9': '#C0392B',
}
CASCADE_DARK = {
    'Pbh_20180227_1': '#1A5276',
    'Pbh_20180621_5': '#7D6608',
    'Pbh_20180803_9': '#922B21',
}

# Event descriptions — auto-generated from cluster metadata
# (Cluster IDs change across runs; manual descriptions removed)
EVENT_DESCRIPTIONS = {}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    cache_path = PROJECT_ROOT / 'results' / 'cache' / 'results_2018.pkl'
    with open(cache_path, 'rb') as f:
        results = pickle.load(f)

    # Use cached impact results (already computed by rerun_impact_2018.py)
    impact = results.event_impact
    if not hasattr(impact, 'cluster_cascade'):
        print("Cache contains legacy impact format — re-running...")
        analyzer = UnifiedImpactAnalyzer()
        impact = analyzer.run(results)
    cc = impact.cluster_cascade

    pbh = cc[cc['cascade_frame'] == 'Pbh'].copy()
    significant = pbh[pbh['role'].isin(['driver', 'late_support', 'suppressor'])]

    cluster_map = {c.cluster_id: c for c in results.event_clusters}
    cascade_map = {c.cascade_id: c for c in results.cascades}
    articles = getattr(results, '_articles', pd.DataFrame())

    return significant, cluster_map, cascade_map, articles


# ---------------------------------------------------------------------------
# Panel A — Timeline
# ---------------------------------------------------------------------------
def plot_timeline(ax, significant, cluster_map, cascade_map, articles):
    date_col = 'date_converted_first'
    frame_col = 'health_frame_mean'
    if frame_col not in articles.columns:
        frame_col = 'health_frame'

    arts = articles.copy()
    arts['_date'] = pd.to_datetime(arts[date_col], errors='coerce')
    daily = arts.groupby('_date')[frame_col].mean()
    daily = daily.loc['2018-02-01':'2018-10-01']
    smoothed = daily.rolling(7, center=True, min_periods=1).mean()

    ax.fill_between(smoothed.index, 0, smoothed.values,
                    color='#3C5488', alpha=0.10, zorder=1)
    ax.plot(smoothed.index, smoothed.values,
            color='#3C5488', lw=1.4, alpha=0.6, zorder=2)

    # Cascade windows
    cascade_order = ['Pbh_20180227_1', 'Pbh_20180621_5', 'Pbh_20180803_9']
    for cid in cascade_order:
        cascade = cascade_map[cid]
        onset = pd.Timestamp(cascade.onset_date)
        end = pd.Timestamp(cascade.end_date)
        peak = pd.Timestamp(cascade.peak_date)
        col = CASCADE_COLORS[cid]
        dark = CASCADE_DARK[cid]

        ax.axvspan(onset, end, alpha=0.07, color=col, zorder=0)
        ax.axvline(peak, color=col, ls='--', lw=1.2, alpha=0.5, zorder=3)

        ymax = smoothed.max() * 1.08
        ax.text(peak, ymax, f"peak {peak.strftime('%b %d')}",
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color=dark,
                path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white')])

    # Cluster peaks
    for _, row in significant.iterrows():
        cl = cluster_map[row['cluster_id']]
        peak_dt = pd.Timestamp(cl.peak_date)
        role = row['role']
        color = ROLE_COLORS[role]

        if peak_dt in smoothed.index:
            y_base = smoothed.loc[peak_dt]
        else:
            idx_pos = smoothed.index.searchsorted(peak_dt)
            idx_pos = min(idx_pos, len(smoothed) - 1)
            y_base = smoothed.iloc[idx_pos]

        marker = {'driver': '^', 'late_support': 'D', 'suppressor': 'v'}[role]
        ax.scatter(peak_dt, y_base, marker=marker, s=65, color=color,
                   edgecolor='white', linewidth=0.8, zorder=5)

    ax.set_ylabel('Mean health frame score')
    ax.set_xlim(pd.Timestamp('2018-02-01'), pd.Timestamp('2018-10-01'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(
        'A.  Daily health frame signal with Pbh cascade windows and event cluster peaks',
        loc='left', fontweight='bold', fontsize=11,
    )

    handles = [
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=ROLE_COLORS['driver'],
                   markersize=9, label='Driver'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=ROLE_COLORS['late_support'],
                   markersize=8, label='Late Support'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor=ROLE_COLORS['suppressor'],
                   markersize=9, label='Suppressor'),
        mpatches.Patch(facecolor='#3C5488', alpha=0.15, label='Health frame (7d avg)'),
    ]
    ax.legend(handles=handles, loc='upper right', frameon=True, fancybox=True,
              edgecolor='#BDC3C7', fontsize=8.5)


# ---------------------------------------------------------------------------
# Panel B — Lollipop
# ---------------------------------------------------------------------------
def plot_lollipop(ax, significant, cluster_map, cascade_map):
    cascade_order = ['Pbh_20180227_1', 'Pbh_20180621_5', 'Pbh_20180803_9']

    rows = []
    for cid in cascade_order:
        sub = significant[significant['cascade_id'] == cid].sort_values(
            'impact_score', ascending=True
        )
        for _, r in sub.iterrows():
            rows.append(r)
    if not rows:
        return

    n = len(rows)

    # Group boundaries
    group_bounds = {}
    current_cascade = None
    for i, row in enumerate(rows):
        cid = row['cascade_id']
        if cid != current_cascade:
            if current_cascade is not None:
                group_bounds[current_cascade] = (group_bounds[current_cascade][0], i - 1)
            group_bounds[cid] = (i, None)
            current_cascade = cid
    group_bounds[current_cascade] = (group_bounds[current_cascade][0], n - 1)

    # Background bands
    for cid in cascade_order:
        if cid in group_bounds:
            y0, y1 = group_bounds[cid]
            ax.axhspan(y0 - 0.5, y1 + 0.5, alpha=0.05,
                       color=CASCADE_COLORS[cid], zorder=0)

    # Lollipops
    labels = []
    for i, row in enumerate(rows):
        cl = cluster_map[row['cluster_id']]
        role = row['role']
        color = ROLE_COLORS[role]
        impact = row['impact_score']

        # Stem
        ax.plot([0, impact], [i, i], color=color, lw=2.0, alpha=0.55, zorder=2,
                solid_capstyle='round')

        # Dot
        marker = {'driver': '^', 'late_support': 'D', 'suppressor': 'v'}[role]
        ax.scatter(impact, i, marker=marker, s=90, color=color,
                   edgecolor='white', linewidth=1.0, zorder=3)

        # Label — auto-generated from cluster metadata
        desc = EVENT_DESCRIPTIONS.get(row['cluster_id'])
        if not desc:
            parts = []
            if hasattr(cl, 'event_types') and cl.event_types:
                types = ', '.join(
                    t.replace('evt_', '').replace('_', ' ').title()
                    for t in sorted(cl.event_types)[:2]
                )
                parts.append(types)
            if hasattr(cl, 'entities') and cl.entities:
                ents = list(cl.entities.keys())[:3] if isinstance(cl.entities, dict) else list(cl.entities)[:3]
                parts.append(', '.join(str(e) for e in ents))
            desc = ' — '.join(parts) if parts else f'Cluster {cl.cluster_id}'
            if len(desc) > 62:
                desc = desc[:59] + '...'
        peak_str = cl.peak_date.strftime('%b %d')
        label = f"C{row['cluster_id']}  {desc}  [{peak_str}]"
        labels.append(label)

        # Right annotation: DID + health affinity
        did_sign = '+' if row['diff_in_diff'] > 0 else ''
        ann = f"DID {did_sign}{row['diff_in_diff']:.3f}   health={row['frame_affinity']:.3f}"
        ax.text(impact + 0.004, i, ann, ha='left', va='center', fontsize=7.5,
                color='#566573', fontstyle='italic',
                path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')])

    # Y-axis labels with role-aware coloring
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    for i, tick_label in enumerate(ax.get_yticklabels()):
        role = rows[i]['role']
        # Darken the role color slightly for text readability
        text_colors = {
            'driver': '#1E8449',
            'late_support': '#B7950B',
            'suppressor': '#C0392B',
        }
        tick_label.set_color(text_colors.get(role, '#2C3E50'))
        tick_label.set_fontweight('bold')

    # Cascade group labels
    ax.set_xlim(-0.002, None)
    xmax = ax.get_xlim()[1]

    for cid in cascade_order:
        if cid in group_bounds:
            y0, y1 = group_bounds[cid]
            y_mid = (y0 + y1) / 2
            ax.text(xmax * 0.97, y_mid, CASCADE_LABELS[cid],
                    ha='right', va='center', fontsize=9, fontweight='bold',
                    fontstyle='italic', color=CASCADE_DARK[cid],
                    path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white')])

    ax.set_xlabel('Impact score')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(
        'B.  Phase 1 causal impact of event clusters on Pbh cascades',
        loc='left', fontweight='bold', fontsize=11,
    )

    handles = [
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=ROLE_COLORS['driver'],
                   markersize=9, label='Driver  (pre-peak, DID > 0)'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=ROLE_COLORS['late_support'],
                   markersize=8, label='Late Support  (post-peak, high health affinity)'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor=ROLE_COLORS['suppressor'],
                   markersize=9, label='Suppressor  (post-peak, low health affinity)'),
    ]
    ax.legend(handles=handles, loc='lower right', frameon=True, fancybox=True,
              edgecolor='#BDC3C7', fontsize=8.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data and computing impact...")
    significant, cluster_map, cascade_map, articles = load_data()

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 14),
        gridspec_kw={'height_ratios': [0.7, 1.3]},
    )
    fig.subplots_adjust(hspace=0.30, left=0.32, right=0.87, top=0.93, bottom=0.05)

    plot_timeline(axes[0], significant, cluster_map, cascade_map, articles)
    plot_lollipop(axes[1], significant, cluster_map, cascade_map)

    fig.suptitle(
        'Phase 1 Impact Analysis:  Event Clusters on Public Health Cascades  (2018)',
        fontsize=14, fontweight='bold', y=0.97,
    )

    out_path = PROJECT_ROOT / 'results' / 'figures' / 'fig_pbh_suppression_2018.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
