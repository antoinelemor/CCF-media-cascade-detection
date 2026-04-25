#!/usr/bin/env python3
"""
Causal chain visualization: Events → Cascades → Paradigm Dominance (2018).

Shows the most significant causal pathways from event clusters through media
cascades to shifts in frame dominance, using unified impact analysis results.

Layout: 3-column flow diagram with:
  Left: Event clusters (source nodes)
  Center: Cascades (intermediary nodes)
  Right: Dominance frames (target nodes)
  Links: Weighted by impact score, colored by role

Usage:
    python scripts/figures/fig_causal_chains_2018.py
"""

import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
})

FRAME_COLORS = {
    'Cult': '#E64B35', 'Eco': '#4DBBD5', 'Envt': '#00A087',
    'Pbh': '#3C5488', 'Just': '#F39B7F', 'Pol': '#8491B4',
    'Sci': '#91D1C2', 'Secu': '#B09C85',
}

FRAME_LABELS = {
    'Cult': 'Cultural', 'Eco': 'Economic', 'Envt': 'Environmental',
    'Pbh': 'Public Health', 'Just': 'Justice', 'Pol': 'Political',
    'Sci': 'Scientific', 'Secu': 'Security',
}

EVENT_TYPE_LABELS = {
    'evt_weather': 'Weather', 'evt_meeting': 'Meeting',
    'evt_publication': 'Publication', 'evt_election': 'Election',
    'evt_policy': 'Policy', 'evt_judiciary': 'Judiciary',
    'evt_cultural': 'Cultural', 'evt_protest': 'Protest',
}

ROLE_COLORS = {
    # Phase 1
    'driver': '#27AE60', 'suppressor': '#E74C3C', 'late_support': '#F39C12',
    'neutral': '#95A5A6',
    # Phase 3
    'amplification': '#27AE60', 'destabilisation': '#E74C3C', 'dormant': '#BDC3C7',
}

ROLE_LABELS = {
    'driver': 'Driver', 'suppressor': 'Suppressor', 'late_support': 'Late Support',
    'neutral': 'Neutral',
    'amplification': 'Amplification', 'destabilisation': 'Destabilisation',
}


def load_data():
    """Load cached 2018 results with unified impact."""
    cache_path = PROJECT_ROOT / 'results' / 'cache' / 'results_2018.pkl'
    logger.info(f"Loading cached results from {cache_path}...")
    with open(cache_path, 'rb') as f:
        results = pickle.load(f)

    impact = results.event_impact
    if not hasattr(impact, 'cluster_cascade'):
        raise ValueError("Cache contains legacy impact format. Re-run unified impact first.")

    return results, impact


def build_chains(results, impact, top_n_cascades=6, top_n_clusters_per=3):
    """Build the set of causal chains to visualize.

    Selects cascades with non-dormant Phase 3 roles, plus their top
    feeding event clusters from Phase 1.
    """
    p1 = impact.cluster_cascade
    p3 = impact.cascade_dominance
    cascade_roles = impact.summary['phase3_cascade_roles']

    # Select active cascades (non-dormant), sorted by max Phase 3 impact
    active_ids = [cid for cid, role in cascade_roles.items() if role != 'dormant']
    if not active_ids:
        logger.warning("No active cascades found")
        return [], [], []

    cascade_max_impact = {}
    for cid in active_ids:
        p3_rows = p3[(p3['cascade_id'] == cid) & (p3['role'] != 'dormant')]
        cascade_max_impact[cid] = p3_rows['impact_score'].max() if not p3_rows.empty else 0.0

    active_ids.sort(key=lambda x: cascade_max_impact[x], reverse=True)
    selected_cascades = active_ids[:top_n_cascades]

    # Build chain data
    chains = []
    cluster_ids_used = set()

    for cid in selected_cascades:
        cascade = [c for c in results.cascades if c.cascade_id == cid][0]

        # Top clusters feeding this cascade (Phase 1, non-unrelated)
        p1_links = p1[(p1['cascade_id'] == cid) & (p1['role'] != 'unrelated')]
        p1_links = p1_links.sort_values('impact_score', ascending=False).head(top_n_clusters_per)

        # Phase 3 links (non-dormant)
        p3_links = p3[(p3['cascade_id'] == cid) & (p3['role'] != 'dormant')]
        p3_links = p3_links.sort_values('impact_score', ascending=False)

        for _, r1 in p1_links.iterrows():
            cl_id = r1['cluster_id']
            cluster_ids_used.add(cl_id)

            for _, r3 in p3_links.iterrows():
                chains.append({
                    'cluster_id': cl_id,
                    'cascade_id': cid,
                    'cascade_frame': cascade.frame,
                    'target_frame': r3['frame'],
                    'p1_score': r1['impact_score'],
                    'p1_role': r1['role'],
                    'p3_score': r3['impact_score'],
                    'p3_role': r3['role'],
                    'cascade_score': cascade.total_score,
                    'cascade_role': cascade_roles[cid],
                })

    # Gather cluster info
    cluster_info = {}
    for cl in results.event_clusters:
        if cl.cluster_id in cluster_ids_used:
            types = sorted(cl.event_types) if hasattr(cl, 'event_types') else []
            type_labels = [EVENT_TYPE_LABELS.get(t, t.replace('evt_', '').title()) for t in types]
            cluster_info[cl.cluster_id] = {
                'types': type_labels,
                'strength': cl.strength,
                'peak_date': cl.peak_date,
            }

    return chains, selected_cascades, cluster_info


def draw_figure(chains, selected_cascades, cluster_info, results, impact):
    """Draw the 3-column causal chain figure."""
    if not chains:
        logger.warning("No chains to draw")
        return

    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # ─── Column positions ─────────────────────────────────────────────
    x_cluster = 0.10   # left column: event clusters
    x_cascade = 0.45   # center column: cascades
    x_frame = 0.80     # right column: dominance frames

    # ─── Collect unique nodes ─────────────────────────────────────────
    cluster_ids = sorted(set(c['cluster_id'] for c in chains))
    cascade_ids = selected_cascades
    target_frames = sorted(set(c['target_frame'] for c in chains))

    # ─── Position nodes vertically ────────────────────────────────────
    margin = 0.06
    usable = 1.0 - 2 * margin

    def space_nodes(n_nodes, start=margin, end=1.0 - margin):
        if n_nodes <= 1:
            return [(start + end) / 2]
        step = (end - start) / (n_nodes - 1)
        return [start + i * step for i in range(n_nodes)]

    cluster_y = {cid: y for cid, y in zip(cluster_ids, space_nodes(len(cluster_ids)))}
    cascade_y = {cid: y for cid, y in zip(cascade_ids, space_nodes(len(cascade_ids)))}
    frame_y = {f: y for f, y in zip(target_frames, space_nodes(len(target_frames)))}

    # ─── Draw nodes ───────────────────────────────────────────────────
    node_h = min(0.05, usable / max(len(cluster_ids), len(cascade_ids), len(target_frames)) * 0.6)
    node_w_cluster = 0.14
    node_w_cascade = 0.14
    node_w_frame = 0.10

    # Draw cluster nodes (left)
    for cid, y in cluster_y.items():
        info = cluster_info.get(cid, {'types': ['?'], 'strength': 0, 'peak_date': None})
        types_str = ', '.join(info['types'][:2])
        if len(info['types']) > 2:
            types_str += f' +{len(info["types"])-2}'
        label = f'C{cid}\n{types_str}'
        peak = info['peak_date']
        if peak:
            label += f'\n{peak.strftime("%b %d")}'

        box = FancyBboxPatch(
            (x_cluster - node_w_cluster/2, y - node_h/2),
            node_w_cluster, node_h,
            boxstyle='round,pad=0.005',
            facecolor='#F0F4F8', edgecolor='#34495E', linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x_cluster, y, label, ha='center', va='center',
                fontsize=7, fontweight='bold', linespacing=1.2)

    # Draw cascade nodes (center)
    cascade_roles = impact.summary['phase3_cascade_roles']
    for cid, y in cascade_y.items():
        cascade = [c for c in results.cascades if c.cascade_id == cid][0]
        frame = cascade.frame
        role = cascade_roles.get(cid, 'dormant')
        color = FRAME_COLORS.get(frame, '#888888')

        # Role badge color
        role_edge = ROLE_COLORS.get(role, '#888888')

        # Use frame abbreviation + date for compact label
        frame_label = FRAME_LABELS.get(frame, frame)
        label = f'{frame_label}\n{cascade.peak_date.strftime("%b %d")}\n{cascade.total_score:.2f}'

        box = FancyBboxPatch(
            (x_cascade - node_w_cascade/2, y - node_h/2),
            node_w_cascade, node_h,
            boxstyle='round,pad=0.005',
            facecolor=color, edgecolor=role_edge, linewidth=3.0,
            alpha=0.90,
        )
        ax.add_patch(box)

        # Determine text color based on background luminance
        hex_color = color.lstrip('#')
        r_c, g_c, b_c = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        lum = (0.299 * r_c + 0.587 * g_c + 0.114 * b_c) / 255
        text_color = '#FFFFFF' if lum < 0.55 else '#1A1A1A'

        txt = ax.text(x_cascade, y, label, ha='center', va='center',
                      fontsize=7.5, fontweight='bold', color=text_color, linespacing=1.2)
        if lum < 0.55:
            txt.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='#1A1A1A')])

        # Role badge
        role_label = ROLE_LABELS.get(role, role.title())
        ax.text(x_cascade + node_w_cascade/2 + 0.005, y + node_h/2 - 0.005,
                role_label, fontsize=6, color=role_edge, fontweight='bold',
                va='top', ha='left')

    # Draw frame nodes (right)
    for frame, y in frame_y.items():
        color = FRAME_COLORS.get(frame, '#888888')
        label = FRAME_LABELS.get(frame, frame)

        box = FancyBboxPatch(
            (x_frame - node_w_frame/2, y - node_h/2),
            node_w_frame, node_h,
            boxstyle='round,pad=0.005',
            facecolor=color, edgecolor='#2C3E50', linewidth=1.5,
            alpha=0.92,
        )
        ax.add_patch(box)

        # Text color by luminance
        hex_color = color.lstrip('#')
        r_c, g_c, b_c = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        lum = (0.299 * r_c + 0.587 * g_c + 0.114 * b_c) / 255
        text_color = '#FFFFFF' if lum < 0.55 else '#1A1A1A'

        txt = ax.text(x_frame, y, label, ha='center', va='center',
                      fontsize=9, fontweight='bold', color=text_color)
        if lum < 0.55:
            txt.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='#1A1A1A')])

    # ─── Draw links ───────────────────────────────────────────────────
    # Aggregate links for width calculation
    max_p1_score = max(c['p1_score'] for c in chains) if chains else 1
    max_p3_score = max(c['p3_score'] for c in chains) if chains else 1

    # Phase 1 links: cluster → cascade
    p1_drawn = set()
    for c in chains:
        key = (c['cluster_id'], c['cascade_id'])
        if key in p1_drawn:
            continue
        p1_drawn.add(key)

        y_start = cluster_y[c['cluster_id']]
        y_end = cascade_y[c['cascade_id']]

        width = 0.5 + 3.0 * (c['p1_score'] / max_p1_score)
        alpha = 0.3 + 0.5 * (c['p1_score'] / max_p1_score)
        color = ROLE_COLORS.get(c['p1_role'], '#95A5A6')

        ax.annotate('',
            xy=(x_cascade - node_w_cascade/2, y_end),
            xytext=(x_cluster + node_w_cluster/2, y_start),
            arrowprops=dict(
                arrowstyle='->', color=color, alpha=alpha,
                lw=width, connectionstyle='arc3,rad=0.05',
                shrinkA=2, shrinkB=2,
            ))

    # Phase 3 links: cascade → dominance
    p3_drawn = set()
    for c in chains:
        key = (c['cascade_id'], c['target_frame'])
        if key in p3_drawn:
            continue
        p3_drawn.add(key)

        y_start = cascade_y[c['cascade_id']]
        y_end = frame_y[c['target_frame']]

        width = 0.8 + 3.5 * (c['p3_score'] / max_p3_score)
        alpha = 0.4 + 0.5 * (c['p3_score'] / max_p3_score)
        color = ROLE_COLORS.get(c['p3_role'], '#888888')

        ax.annotate('',
            xy=(x_frame - node_w_frame/2, y_end),
            xytext=(x_cascade + node_w_cascade/2, y_start),
            arrowprops=dict(
                arrowstyle='->', color=color, alpha=alpha,
                lw=width, connectionstyle='arc3,rad=0.05',
                shrinkA=2, shrinkB=2,
            ))

        # Impact label on the link
        mid_x = (x_cascade + node_w_cascade/2 + x_frame - node_w_frame/2) / 2
        mid_y = (y_start + y_end) / 2
        ax.text(mid_x, mid_y, f'{c["p3_score"]:.2f}',
                fontsize=5.5, color=color, ha='center', va='center',
                fontweight='bold', alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                          edgecolor='none', alpha=0.7))

    # ─── Column headers ──────────────────────────────────────────────
    ax.text(x_cluster, 0.97, 'Event Clusters', ha='center', va='top',
            fontsize=13, fontweight='bold', color='#2C3E50')
    ax.text(x_cluster, 0.94, '(real-world events)', ha='center', va='top',
            fontsize=9, color='#7F8C8D', fontstyle='italic')

    ax.text(x_cascade, 0.97, 'Media Cascades', ha='center', va='top',
            fontsize=13, fontweight='bold', color='#2C3E50')
    ax.text(x_cascade, 0.94, '(amplified coverage bursts)', ha='center', va='top',
            fontsize=9, color='#7F8C8D', fontstyle='italic')

    ax.text(x_frame, 0.97, 'Paradigm Dominance', ha='center', va='top',
            fontsize=13, fontweight='bold', color='#2C3E50')
    ax.text(x_frame, 0.94, '(frame influence shifts)', ha='center', va='top',
            fontsize=9, color='#7F8C8D', fontstyle='italic')

    # ─── Legend ───────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=ROLE_COLORS['driver'], label='Driver (event amplifies cascade)',
                       alpha=0.7),
        mpatches.Patch(facecolor=ROLE_COLORS['late_support'], label='Late Support (post-peak, thematically aligned)',
                       alpha=0.7),
        mpatches.Patch(facecolor=ROLE_COLORS['suppressor'], label='Suppressor (event reduces cascade)',
                       alpha=0.7),
        mpatches.Patch(facecolor=ROLE_COLORS['neutral'], label='Neutral (significant, no direction)',
                       alpha=0.7),
        mpatches.Patch(facecolor=ROLE_COLORS['amplification'], label='Amplification (cascade → dominance ↑)',
                       alpha=0.7),
        mpatches.Patch(facecolor=ROLE_COLORS['destabilisation'], label='Destabilisation (cascade → dominance ↓)',
                       alpha=0.7),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=3,
              fontsize=8, frameon=True, fancybox=True,
              bbox_to_anchor=(0.5, -0.02))

    # ─── Title ────────────────────────────────────────────────────────
    fig.suptitle('Causal Impact Chains: Events → Cascades → Paradigm Shifts (2018)',
                 fontsize=15, fontweight='bold', y=0.99, color='#2C3E50')

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def main():
    results, impact = load_data()
    chains, selected_cascades, cluster_info = build_chains(results, impact)

    if not chains:
        logger.error("No causal chains found. Exiting.")
        return

    logger.info(f"Found {len(chains)} chain segments across {len(selected_cascades)} active cascades")

    fig = draw_figure(chains, selected_cascades, cluster_info, results, impact)

    # Save
    fig_dir = PROJECT_ROOT / 'results' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    for ext in ('png', 'pdf'):
        out = fig_dir / f'fig_causal_chains_2018.{ext}'
        fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved: {out}")

    plt.close(fig)

    # Print chain summary
    print("\n" + "=" * 70)
    print("CAUSAL CHAIN SUMMARY")
    print("=" * 70)
    cascade_roles = impact.summary['phase3_cascade_roles']
    for cid in selected_cascades:
        cascade = [c for c in results.cascades if c.cascade_id == cid][0]
        role = cascade_roles.get(cid, 'dormant')
        print(f"\n  {cid} [{cascade.frame}] score={cascade.total_score:.3f} role={role}")

        # Phase 1 top clusters
        p1 = impact.cluster_cascade
        p1_top = p1[(p1['cascade_id'] == cid) & (p1['role'] != 'unrelated')]
        p1_top = p1_top.sort_values('impact_score', ascending=False).head(3)
        for _, r in p1_top.iterrows():
            cl_id = r['cluster_id']
            info = cluster_info.get(cl_id, {'types': ['?'], 'strength': 0})
            types_str = ', '.join(info['types'][:2])
            print(f"    <- C{cl_id} ({types_str}, str={info['strength']:.3f}) "
                  f"impact={r['impact_score']:.3f} [{r['role']}]")

        # Phase 3 non-dormant
        p3 = impact.cascade_dominance
        p3_active = p3[(p3['cascade_id'] == cid) & (p3['role'] != 'dormant')]
        p3_active = p3_active.sort_values('impact_score', ascending=False)
        for _, r in p3_active.iterrows():
            label = FRAME_LABELS.get(r['frame'], r['frame'])
            print(f"    -> {label} dominance: impact={r['impact_score']:.3f} "
                  f"[{r['role']}] DID={r['diff_in_diff']:+.3f}")


if __name__ == '__main__':
    main()
