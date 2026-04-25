#!/usr/bin/env python3
"""
Publication figure: Event cluster impact on cascades (2018).

Three-panel composite:
  A — Impact role distribution by frame (stacked bars)
  B — Event type × frame driver profile (bubble matrix)
  C — Top 30 event-cascade interactions (lollipop chart)

Reads from: results/cache/results_2018.pkl
Outputs to: results/figures/fig_cluster_cascade_impact_2018.{png,pdf}
"""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cascade_detector.core.constants import FRAME_COLORS

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 8.5,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

ROLE_COLORS = {
    'driver': '#27AE60',
    'late_support': '#F39C12',
    'suppressor': '#E74C3C',
}

ROLE_LABELS = {
    'driver': 'Driver',
    'late_support': 'Late support',
    'suppressor': 'Suppressor',
}

FRAME_LABELS = {
    'Cult': 'Cultural', 'Eco': 'Economic', 'Envt': 'Environmental',
    'Just': 'Justice', 'Pbh': 'Public Health', 'Pol': 'Political',
    'Sci': 'Scientific', 'Secu': 'Security',
}

EVENT_LABELS = {
    'evt_policy': 'Policy',
    'evt_meeting': 'Meeting',
    'evt_publication': 'Publication',
    'evt_weather': 'Weather',
    'evt_election': 'Election',
    'evt_judiciary': 'Judiciary',
    'evt_protest': 'Protest',
    'evt_cultural': 'Cultural',
}

EVENT_ORDER = [
    'evt_policy', 'evt_meeting', 'evt_publication', 'evt_weather',
    'evt_election', 'evt_judiciary', 'evt_protest', 'evt_cultural',
]

ROLE_ORDER = ['driver', 'late_support', 'suppressor']

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    cache_path = PROJECT_ROOT / 'results' / 'cache' / 'results_2018.pkl'
    if not cache_path.exists():
        print(f"ERROR: Cache not found at {cache_path}")
        sys.exit(1)

    with open(cache_path, 'rb') as f:
        results = pickle.load(f)

    df = results.event_impact.cluster_cascade.copy()
    cluster_map = {c.cluster_id: c for c in results.event_clusters}
    n_cascades_per_frame = df.groupby('cascade_frame')['cascade_id'].nunique()

    # Filter significant pairs
    sig = df[df['role'].isin(ROLE_ORDER)].copy()
    sig['dominant_type'] = sig['cluster_id'].map(
        lambda cid: cluster_map[cid].dominant_type if cid in cluster_map else 'unknown'
    )

    return sig, cluster_map, n_cascades_per_frame


# ---------------------------------------------------------------------------
# Panel A: Impact role distribution by frame
# ---------------------------------------------------------------------------

def panel_a(ax, sig, n_cascades_per_frame):
    counts = (
        sig.groupby(['cascade_frame', 'role'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ROLE_ORDER, fill_value=0)
    )
    # Sort by total significant count descending
    counts['_total'] = counts.sum(axis=1)
    counts = counts.sort_values('_total', ascending=True)  # ascending for horizontal bars
    totals = counts['_total'].copy()
    counts = counts.drop(columns='_total')

    frames = counts.index.tolist()
    y = np.arange(len(frames))
    bar_h = 0.55

    left = np.zeros(len(frames))
    for role in ROLE_ORDER:
        vals = counts[role].values.astype(float)
        bars = ax.barh(y, vals, height=bar_h, left=left,
                       color=ROLE_COLORS[role], edgecolor='white', linewidth=0.3,
                       label=ROLE_LABELS[role], zorder=3)
        # Annotate counts inside segments
        for i, (bar, v) in enumerate(zip(bars, vals)):
            if v >= 8:
                cx = left[i] + v / 2
                ax.text(cx, y[i], str(int(v)), ha='center', va='center',
                        fontsize=7, fontweight='bold', color='white', zorder=4)
        left += vals

    # Right-side annotation: total (n cascades)
    x_max = totals.max()
    for i, frame in enumerate(frames):
        n_casc = n_cascades_per_frame.get(frame, 0)
        ax.text(totals[frame] + x_max * 0.02, y[i],
                f'{int(totals[frame])}  ({n_casc} casc.)',
                ha='left', va='center', fontsize=7, color='#444444')

    # Y-axis: frame labels with color patches
    ax.set_yticks(y)
    ax.set_yticklabels([FRAME_LABELS.get(f, f) for f in frames])
    for i, frame in enumerate(frames):
        ax.get_yticklabels()[i].set_color(FRAME_COLORS.get(frame, '#333333'))
        ax.get_yticklabels()[i].set_fontweight('bold')

    ax.set_xlabel('Number of significant cluster-cascade pairs')
    ax.set_title('A.  Impact role distribution by frame', loc='left', fontweight='bold')
    ax.set_xlim(0, x_max * 1.35)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='#cccccc',
              ncol=3, columnspacing=1.0)
    ax.grid(axis='x', alpha=0.15, linewidth=0.4)


# ---------------------------------------------------------------------------
# Panel B: Event type × frame bubble matrix
# ---------------------------------------------------------------------------

def panel_b(ax, sig):
    # Focus on driver + late_support for positive impact
    positive = sig[sig['role'].isin(['driver', 'late_support'])]
    suppressors = sig[sig['role'] == 'suppressor']

    # Aggregate: (frame, event_type) → count, mean impact
    agg_pos = (
        positive.groupby(['cascade_frame', 'dominant_type'])
        .agg(count=('impact_score', 'size'), mean_score=('impact_score', 'mean'))
        .reset_index()
    )

    # Suppressor presence per cell
    sup_cells = set(
        zip(suppressors['cascade_frame'], suppressors['dominant_type'])
    )

    # Sort frames by total positive count
    frame_totals = agg_pos.groupby('cascade_frame')['count'].sum().sort_values(ascending=False)
    frame_order = frame_totals.index.tolist()
    # Ensure all 8 frames present
    for f in FRAME_LABELS:
        if f not in frame_order:
            frame_order.append(f)

    # Filter to event types that appear in data
    evt_types = [e for e in EVENT_ORDER if e in agg_pos['dominant_type'].values]

    y_map = {f: i for i, f in enumerate(reversed(frame_order))}
    x_map = {e: i for i, e in enumerate(evt_types)}

    # Size scaling
    max_count = agg_pos['count'].max() if len(agg_pos) > 0 else 1
    min_size = 40
    max_size = 600

    # Color normalization
    if len(agg_pos) > 0:
        vmin = agg_pos['mean_score'].min()
        vmax = agg_pos['mean_score'].max()
    else:
        vmin, vmax = 0, 1
    cmap = plt.cm.OrRd

    for _, row in agg_pos.iterrows():
        frame = row['cascade_frame']
        etype = row['dominant_type']
        if etype not in x_map or frame not in y_map:
            continue
        x = x_map[etype]
        y = y_map[frame]
        size = min_size + (row['count'] / max_count) * (max_size - min_size)
        # Map to 0.15–0.95 range to avoid white at low end
        color_val = 0.15 + 0.80 * ((row['mean_score'] - vmin) / (vmax - vmin) if vmax > vmin else 0.5)
        color = cmap(color_val)

        ax.scatter(x, y, s=size, c=[color], edgecolors='#555555', linewidth=0.5, zorder=3)

        # Red ring if suppressors exist for this cell
        if (frame, etype) in sup_cells:
            ax.scatter(x, y, s=size * 1.5, facecolors='none',
                       edgecolors='#E74C3C', linewidth=1.2, zorder=2)

    # Axes
    ax.set_xticks(range(len(evt_types)))
    ax.set_xticklabels([EVENT_LABELS.get(e, e) for e in evt_types], rotation=35, ha='right')
    ax.set_yticks(range(len(frame_order)))
    ax.set_yticklabels([FRAME_LABELS.get(f, f) for f in reversed(frame_order)])
    for i, frame in enumerate(reversed(frame_order)):
        ax.get_yticklabels()[i].set_color(FRAME_COLORS.get(frame, '#333333'))
        ax.get_yticklabels()[i].set_fontweight('bold')

    ax.set_xlim(-0.5, len(evt_types) - 0.5)
    ax.set_ylim(-0.5, len(frame_order) - 0.5)
    ax.set_title('B.  Event type × frame driver profile', loc='left', fontweight='bold')

    # Grid
    for i in range(len(frame_order)):
        ax.axhline(i, color='#eeeeee', linewidth=0.3, zorder=0)
    for i in range(len(evt_types)):
        ax.axvline(i, color='#eeeeee', linewidth=0.3, zorder=0)

    # Size legend — pick round representative values
    candidates = [1, 5, 10, 25, 50, 100, 150, 200]
    size_vals = [c for c in candidates if c <= max_count]
    if max_count not in size_vals:
        size_vals.append(int(max_count))
    # Keep at most 4 entries
    if len(size_vals) > 4:
        size_vals = [size_vals[0], size_vals[len(size_vals)//3],
                     size_vals[2*len(size_vals)//3], size_vals[-1]]
    legend_handles = []
    for v in size_vals:
        s = min_size + (v / max_count) * (max_size - min_size)
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#aaaaaa',
                   markeredgecolor='#555555', markersize=np.sqrt(s) * 0.55,
                   label=str(v), linewidth=0)
        )
    # Add red ring legend entry
    legend_handles.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='#E74C3C', markeredgewidth=1.2,
               markersize=8, label='Suppressors\npresent', linewidth=0)
    )
    leg = ax.legend(handles=legend_handles, title='Pairs', loc='lower left',
                    frameon=True, framealpha=0.9, edgecolor='#cccccc',
                    fontsize=7, title_fontsize=7.5, handletextpad=0.5,
                    labelspacing=0.8)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, aspect=20)
    cbar.set_label('Mean impact score', fontsize=8)
    cbar.ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Panel C: Top 30 event-cascade interactions (lollipop)
# ---------------------------------------------------------------------------

def panel_c(ax, sig, cluster_map):
    # Top 30 by impact_score
    top = sig.nlargest(30, 'impact_score').copy()
    top = top.sort_values('impact_score', ascending=True).reset_index(drop=True)

    y = np.arange(len(top))
    markers = {'driver': '^', 'late_support': 'D', 'suppressor': 'v'}

    for i, (_, row) in enumerate(top.iterrows()):
        role = row['role']
        frame = row['cascade_frame']
        color = ROLE_COLORS[role]
        frame_color = FRAME_COLORS.get(frame, '#888888')

        # Stem
        ax.plot([0, row['impact_score']], [y[i], y[i]],
                color=color, linewidth=1.0, alpha=0.7, zorder=2)

        # Marker
        ax.scatter(row['impact_score'], y[i], marker=markers[role],
                   s=50, c=frame_color, edgecolors=color, linewidth=1.0, zorder=4)

    # Y labels
    labels = []
    for _, row in top.iterrows():
        cid = row['cluster_id']
        cluster = cluster_map.get(cid)
        etype_short = EVENT_LABELS.get(
            cluster.dominant_type if cluster else '', '?'
        )
        cascade_short = row['cascade_id']
        # Shorten cascade_id: e.g. "Pol_20180101_44" → "Pol #44"
        parts = cascade_short.split('_')
        if len(parts) >= 3:
            cascade_label = f"{parts[0]} #{parts[-1]}"
        else:
            cascade_label = cascade_short
        labels.append(f"{etype_short} C{cid} \u2192 {cascade_label}")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5, fontfamily='monospace')
    ax.set_xlabel('Impact score')
    ax.set_title('C.  Top 30 event-cascade interactions', loc='left', fontweight='bold')
    ax.set_xlim(0, top['impact_score'].max() * 1.35)
    ax.grid(axis='x', alpha=0.15, linewidth=0.4)

    # Right-side annotations (use axes transform blend for x position)
    x_max = top['impact_score'].max()
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for i, (_, row) in enumerate(top.iterrows()):
        did_str = f"DID={row['diff_in_diff']:.2f}"
        aff_str = f"aff={row['frame_affinity']:.2f}" if pd.notna(row['frame_affinity']) else ""
        annotation = f"{did_str}  {aff_str}".strip()
        ax.text(0.78, y[i], annotation, transform=trans,
                ha='left', va='center', fontsize=5.5, color='#666666')

    # Legend
    legend_elements = []
    for role in ROLE_ORDER:
        legend_elements.append(
            Line2D([0], [0], marker=markers[role], color='w',
                   markerfacecolor=ROLE_COLORS[role], markeredgecolor=ROLE_COLORS[role],
                   markersize=7, label=ROLE_LABELS[role], linewidth=0)
        )
    legend_elements.append(
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#999999',
               markersize=6, label='Fill = frame color', linewidth=0)
    )
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              framealpha=0.9, edgecolor='#cccccc', fontsize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    sig, cluster_map, n_cascades_per_frame = load_data()

    print(f"  {len(sig)} significant pairs: "
          f"{(sig['role']=='driver').sum()} drivers, "
          f"{(sig['role']=='late_support').sum()} late_support, "
          f"{(sig['role']=='suppressor').sum()} suppressors")

    # Create figure
    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1.2, 1.5],
                           hspace=0.30, figure=fig)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    print("Drawing Panel A...")
    panel_a(ax_a, sig, n_cascades_per_frame)

    print("Drawing Panel B...")
    panel_b(ax_b, sig)

    print("Drawing Panel C...")
    panel_c(ax_c, sig, cluster_map)

    # Suptitle
    fig.suptitle(
        'Event cluster impact on media cascades — 2018',
        fontsize=14, fontweight='bold', y=0.995
    )

    # Output
    outdir = PROJECT_ROOT / 'results' / 'figures'
    outdir.mkdir(parents=True, exist_ok=True)

    for ext in ('png', 'pdf'):
        outpath = outdir / f'fig_cluster_cascade_impact_2018.{ext}'
        fig.savefig(outpath)
        print(f"  Saved: {outpath}")

    plt.close(fig)
    print("Done.")


if __name__ == '__main__':
    main()
