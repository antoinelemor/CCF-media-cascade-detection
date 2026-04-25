#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
run_2018_paradigm.py

MAIN OBJECTIVE:
---------------
Run cascade detection + paradigm shift analysis on 2018 data, then produce
publication-quality figures showing:
  1. Paradigm composition timeline (stacked area)
  2. Shift events with cascade attribution (annotated timeline)
  3. Cascade–shift–event attribution Sankey-style chord diagram
  4. Shift magnitude + cascade strength scatter
  5. Per-shift detail panels

Uses test embeddings (data/embeddings-test).

Usage:
    python scripts/run_2018_paradigm.py

Author:
-------
Antoine Lemor
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.constants import FRAMES
from cascade_detector.pipeline import CascadeDetectionPipeline
from cascade_detector.analysis.paradigm_shift import ParadigmShiftAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / 'results' / 'paradigm_2018'

# =============================================================================
# Colour palette — one per frame, consistent across all figures
# =============================================================================
FRAME_COLORS = {
    'Cult': '#E64B35',   # Vermillion
    'Eco':  '#4DBBD5',   # Cyan
    'Envt': '#00A087',   # Teal green
    'Pbh':  '#3C5488',   # Steel blue
    'Just': '#F39B7F',   # Salmon
    'Pol':  '#8491B4',   # Lavender grey
    'Sci':  '#91D1C2',   # Mint
    'Secu': '#DC9A2E',   # Gold
}

FRAME_LABELS = {
    'Cult': 'Cultural',
    'Eco':  'Economic',
    'Envt': 'Environmental',
    'Pbh':  'Public Health',
    'Just': 'Justice',
    'Pol':  'Political',
    'Sci':  'Scientific',
    'Secu': 'Security',
}

SHIFT_TYPE_MARKERS = {
    'frame_entry':      ('^', '#2ca02c'),   # Green triangle up
    'frame_exit':       ('v', '#d62728'),   # Red triangle down
    'recomposition':    ('D', '#9467bd'),   # Purple diamond
    'full_replacement': ('s', '#1f77b4'),   # Blue square
}


def setup_style():
    """Apply publication-quality style settings."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


# =============================================================================
# Figure 1: Paradigm composition timeline (stacked area + dominant overlay)
# =============================================================================

def fig_paradigm_timeline(timeline: pd.DataFrame, shifts, output_path: Path):
    """Stacked area chart of paradigm composition with shift markers."""
    fig, (ax_area, ax_dom) = plt.subplots(
        2, 1, figsize=(14, 7), height_ratios=[3, 1],
        sharex=True, gridspec_kw={'hspace': 0.05},
    )

    dates = pd.to_datetime(timeline['date'])
    frame_cols = [f'paradigm_{f}' for f in FRAMES]

    # Stacked area
    values = timeline[frame_cols].values.T  # (8, T)
    # Normalize to sum=1 per timestep for stacking
    totals = values.sum(axis=0)
    totals[totals == 0] = 1
    normed = values / totals

    ax_area.stackplot(
        dates, normed,
        labels=[FRAME_LABELS[f] for f in FRAMES],
        colors=[FRAME_COLORS[f] for f in FRAMES],
        alpha=0.85,
    )

    # Shift markers on top
    for s in shifts:
        sd = pd.Timestamp(s.shift_date)
        ax_area.axvline(sd, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
        marker, color = SHIFT_TYPE_MARKERS.get(s.shift_type, ('o', 'black'))
        ax_area.plot(sd, 1.02, marker=marker, color=color, markersize=8,
                     transform=ax_area.get_xaxis_transform(), clip_on=False, zorder=5)

    ax_area.set_ylabel('Paradigm composition\n(dominance-weighted)')
    ax_area.set_ylim(0, 1)
    ax_area.set_xlim(dates.iloc[0], dates.iloc[-1])

    # Legend
    handles = [mpatches.Patch(facecolor=FRAME_COLORS[f], label=FRAME_LABELS[f])
               for f in FRAMES]
    # Add shift type markers to legend
    for stype, (marker, color) in SHIFT_TYPE_MARKERS.items():
        label = stype.replace('_', ' ').title()
        handles.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
                               markersize=8, label=f'Shift: {label}'))
    ax_area.legend(handles=handles, loc='upper left', ncol=3, framealpha=0.9,
                   fontsize=8)

    # Bottom panel: dominant frame type over time
    dom_map = {row['date']: row['dominant_frames']
               for _, row in timeline.iterrows()}
    for i, row in timeline.iterrows():
        d = pd.Timestamp(row['date'])
        dom_frames = str(row['dominant_frames']).split(',')
        n = len(dom_frames)
        for j, f in enumerate(dom_frames):
            f = f.strip()
            if f in FRAME_COLORS:
                ax_dom.barh(j / n, 7, left=mdates.date2num(d) - 3.5,
                            height=1.0 / n,
                            color=FRAME_COLORS[f], edgecolor='none')

    ax_dom.set_yticks([0.5])
    ax_dom.set_yticklabels(['Dominant\nframes'])
    ax_dom.set_ylim(0, 1)
    ax_dom.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax_dom.xaxis.set_major_locator(mdates.MonthLocator())
    ax_dom.set_xlabel('2018')
    ax_dom.grid(False)

    fig.suptitle('Paradigm Composition & Shifts — 2018', fontsize=14, fontweight='bold', y=0.98)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# =============================================================================
# Figure 2: Cascade–Shift attribution timeline
# =============================================================================

def fig_cascade_shift_timeline(cascades, shifts, timeline, output_path: Path):
    """Annotated timeline: cascades as horizontal bars, shifts as vertical lines,
    attribution shown with connecting arrows."""
    n_cascades = len(cascades)
    if n_cascades == 0:
        logger.warning("No cascades to plot in timeline")
        return

    fig, ax = plt.subplots(figsize=(14, max(6, 2 + n_cascades * 0.35)))

    # Sort cascades by onset date
    sorted_c = sorted(cascades, key=lambda c: c.onset_date)

    # Plot cascades as horizontal bars
    y_positions = {}
    for i, c in enumerate(sorted_c):
        y = i
        y_positions[c.cascade_id] = y
        onset = mdates.date2num(c.onset_date)
        duration = (c.end_date - c.onset_date).days
        color = FRAME_COLORS.get(c.frame, '#999999')
        alpha = 0.5 + 0.5 * min(c.total_score, 1.0)  # Stronger = more opaque

        ax.barh(y, duration, left=onset, height=0.6,
                color=color, alpha=alpha, edgecolor=color, linewidth=0.8)

        # Label
        label = f"{c.frame} ({c.total_score:.2f})"
        ax.text(onset + duration + 2, y, label,
                va='center', ha='left', fontsize=7.5, color=color,
                fontweight='bold' if c.classification == 'strong_cascade' else 'normal')

    # Plot shifts as vertical lines
    for s in shifts:
        sd = mdates.date2num(pd.Timestamp(s.shift_date))
        marker, color = SHIFT_TYPE_MARKERS.get(s.shift_type, ('o', 'black'))
        ax.axvline(sd, color='grey', linewidth=1, linestyle='--', alpha=0.5, zorder=1)

        # Draw connecting arrows from shift to attributed cascades
        for ac in s.attributed_cascades[:5]:  # Top 5
            cid = ac['cascade_id']
            if cid in y_positions:
                y_c = y_positions[cid]
                ax.annotate(
                    '', xy=(sd, n_cascades + 0.5), xytext=(sd, y_c),
                    arrowprops=dict(
                        arrowstyle='->', color=color,
                        alpha=0.4 + 0.4 * ac['attribution_score'],
                        linewidth=1 + ac['attribution_score'] * 2,
                        connectionstyle='arc3,rad=0.1',
                    ),
                )

        # Shift marker at top
        ax.plot(sd, n_cascades + 0.5, marker=marker, color=color,
                markersize=10, zorder=5, clip_on=False)

        # Label shift
        entering = ', '.join(s.entering_frames) if s.entering_frames else ''
        exiting = ', '.join(s.exiting_frames) if s.exiting_frames else ''
        shift_label = f"+{entering}" if entering else ''
        if exiting:
            shift_label += f" -{exiting}" if shift_label else f"-{exiting}"
        ax.text(sd, n_cascades + 1.0, shift_label,
                ha='center', va='bottom', fontsize=7, rotation=30,
                fontweight='bold', color=color)

    ax.set_yticks(range(n_cascades))
    ax.set_yticklabels([f"{c.cascade_id}" for c in sorted_c], fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_ylim(-0.5, n_cascades + 2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cascades')
    ax.set_title('Cascade–Paradigm Shift Attribution Timeline — 2018',
                 fontweight='bold', pad=20)

    # Legend
    legend_handles = []
    for f in FRAMES:
        legend_handles.append(mpatches.Patch(color=FRAME_COLORS[f],
                                              label=FRAME_LABELS[f]))
    for stype, (marker, color) in SHIFT_TYPE_MARKERS.items():
        legend_handles.append(Line2D([0], [0], marker=marker, color='w',
                                      markerfacecolor=color, markersize=8,
                                      label=f'Shift: {stype.replace("_", " ")}'))
    ax.legend(handles=legend_handles, loc='lower right', ncol=2,
              fontsize=7, framealpha=0.9)

    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# =============================================================================
# Figure 3: Per-shift detail panels
# =============================================================================

def fig_shift_detail_panels(shifts, output_path: Path):
    """One panel per shift: before/after paradigm vectors + attributed cascades."""
    n_shifts = len(shifts)
    if n_shifts == 0:
        logger.warning("No shifts to plot detail panels for")
        return

    fig, axes = plt.subplots(n_shifts, 2, figsize=(14, 3.5 * n_shifts),
                             gridspec_kw={'width_ratios': [1, 1.5]})
    if n_shifts == 1:
        axes = [axes]

    for i, s in enumerate(shifts):
        ax_bar, ax_attr = axes[i]

        # Left: before/after paradigm vectors as grouped bar chart
        x = np.arange(len(FRAMES))
        width = 0.35

        before_vals = [s.state_before.frame_scores.get(f, 0) for f in FRAMES]
        after_vals = [s.state_after.frame_scores.get(f, 0) for f in FRAMES]

        bars_before = ax_bar.bar(x - width/2, before_vals, width, label='Before',
                                  color=[FRAME_COLORS[f] for f in FRAMES], alpha=0.4,
                                  edgecolor='grey', linewidth=0.5)
        bars_after = ax_bar.bar(x + width/2, after_vals, width, label='After',
                                 color=[FRAME_COLORS[f] for f in FRAMES], alpha=0.9,
                                 edgecolor='black', linewidth=0.5)

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(FRAMES, fontsize=8)
        ax_bar.set_ylabel('Dominance score')
        shift_date_str = s.shift_date.strftime('%Y-%m-%d')
        ax_bar.set_title(
            f"Shift #{i+1}: {s.shift_type.replace('_', ' ')} "
            f"({shift_date_str}, mag={s.shift_magnitude:.2f})",
            fontsize=10, fontweight='bold',
        )

        # Highlight entering/exiting frames
        for j, f in enumerate(FRAMES):
            if f in s.entering_frames:
                ax_bar.annotate('+', (x[j] + width/2, after_vals[j]),
                               ha='center', va='bottom', fontsize=14,
                               color='green', fontweight='bold')
            if f in s.exiting_frames:
                ax_bar.annotate('-', (x[j] - width/2, before_vals[j]),
                               ha='center', va='bottom', fontsize=14,
                               color='red', fontweight='bold')

        ax_bar.legend(fontsize=8, loc='upper right')

        # Right: attributed cascades + events
        attributed = s.attributed_cascades[:8]  # Top 8
        if attributed:
            cascade_labels = []
            attr_scores = []
            bar_colors = []
            for ac in reversed(attributed):
                cascade_labels.append(
                    f"{ac['cascade_id']}\n({ac['frame']}, {ac['classification'][:3]})"
                )
                attr_scores.append(ac['attribution_score'])
                bar_colors.append(FRAME_COLORS.get(ac['frame'], '#999'))

            y_pos = np.arange(len(cascade_labels))
            ax_attr.barh(y_pos, attr_scores, color=bar_colors, alpha=0.8,
                         edgecolor='black', linewidth=0.5)
            ax_attr.set_yticks(y_pos)
            ax_attr.set_yticklabels(cascade_labels, fontsize=7)
            ax_attr.set_xlabel('Attribution score')
            ax_attr.set_xlim(0, 1)

            # Add events as text
            if s.attributed_events:
                top_events = s.attributed_events[:5]
                event_text = 'Key events: ' + ', '.join(
                    f"{e['event']} ({e['count']})" for e in top_events
                )
                ax_attr.text(0.02, 0.98, event_text, transform=ax_attr.transAxes,
                            va='top', ha='left', fontsize=7,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                      alpha=0.8, edgecolor='grey'))
        else:
            ax_attr.text(0.5, 0.5, 'No cascades attributed',
                         transform=ax_attr.transAxes, ha='center', va='center',
                         fontsize=10, color='grey')

        ax_attr.set_title('Cascade Attribution', fontsize=10)

    fig.suptitle('Paradigm Shift Details — 2018', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# =============================================================================
# Figure 4: Shift magnitude vs. cascade density heatmap
# =============================================================================

def fig_shift_cascade_heatmap(shifts, cascades, output_path: Path):
    """Matrix: rows = shifts, columns = frames. Cell color = attribution strength."""
    n_shifts = len(shifts)
    if n_shifts == 0:
        return

    # Build attribution matrix: shifts × frames
    matrix = np.zeros((n_shifts, len(FRAMES)))
    shift_labels = []

    for i, s in enumerate(shifts):
        date_str = s.shift_date.strftime('%b %d')
        stype = s.shift_type.replace('_', ' ')
        shift_labels.append(f"S{i+1}: {date_str}\n({stype})")

        for ac in s.attributed_cascades:
            frame = ac['frame']
            if frame in FRAMES:
                j = FRAMES.index(frame)
                matrix[i, j] = max(matrix[i, j], ac['attribution_score'])

    fig, ax = plt.subplots(figsize=(10, max(3, 1.5 + n_shifts * 0.8)))

    cmap = LinearSegmentedColormap.from_list('cascade_attr',
                                              ['#f7f7f7', '#fee090', '#d73027'])
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(FRAMES)))
    ax.set_xticklabels([FRAME_LABELS[f] for f in FRAMES], rotation=45, ha='right',
                       fontsize=9)
    ax.set_yticks(range(n_shifts))
    ax.set_yticklabels(shift_labels, fontsize=8)

    # Annotate cells
    for i in range(n_shifts):
        for j in range(len(FRAMES)):
            val = matrix[i, j]
            if val > 0.01:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=8, color='white' if val > 0.5 else 'black')

    # Mark entering/exiting
    for i, s in enumerate(shifts):
        for j, f in enumerate(FRAMES):
            if f in s.entering_frames:
                ax.plot(j, i - 0.35, marker='^', color='green', markersize=6)
            if f in s.exiting_frames:
                ax.plot(j, i - 0.35, marker='v', color='red', markersize=6)

    plt.colorbar(im, ax=ax, label='Max attribution score', shrink=0.8)
    ax.set_title('Shift × Frame Attribution Matrix — 2018',
                 fontsize=12, fontweight='bold')

    # Add legend for arrows
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
               markersize=8, label='Entering frame'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red',
               markersize=8, label='Exiting frame'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# =============================================================================
# Figure 5: Event attribution summary
# =============================================================================

def fig_event_attribution(shifts, output_path: Path):
    """Bar chart: aggregated event counts across all shifts."""
    event_totals = {}
    for s in shifts:
        for e in s.attributed_events:
            event_totals[e['event']] = event_totals.get(e['event'], 0) + e['count']

    if not event_totals:
        logger.warning("No events to plot")
        return

    # Sort by count
    sorted_events = sorted(event_totals.items(), key=lambda x: -x[1])
    events = [e[0] for e in sorted_events]
    counts = [e[1] for e in sorted_events]

    # Clean names
    clean_names = [e.replace('evt_', '').replace('_', ' ').title() for e in events]

    fig, ax = plt.subplots(figsize=(10, max(4, len(events) * 0.4)))

    y_pos = np.arange(len(events))
    bars = ax.barh(y_pos, counts, color='#4DBBD5', alpha=0.85,
                   edgecolor='#2a7f94', linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_names, fontsize=9)
    ax.set_xlabel('Total occurrence count across attributed cascades')
    ax.set_title('Events Driving Paradigm Shifts — 2018',
                 fontsize=12, fontweight='bold')

    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=9)

    ax.invert_yaxis()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# =============================================================================
# Figure 6: Comprehensive panel — the "money figure"
# =============================================================================

def fig_comprehensive_panel(timeline, shifts, cascades, weekly_props, output_path: Path):
    """3-row panel combining proportions, paradigm state, and cascade bars."""
    dates = pd.to_datetime(timeline['date'])

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2.5, 1, 0.8, 2.5], hspace=0.15)

    # --- Row 1: Weekly proportions (raw time series) ---
    ax1 = fig.add_subplot(gs[0])
    for frame in FRAMES:
        if frame in weekly_props.columns:
            ax1.plot(weekly_props.index, weekly_props[frame],
                     color=FRAME_COLORS[frame], linewidth=1.2, alpha=0.9,
                     label=FRAME_LABELS[frame])

    # Shade shift periods
    for s in shifts:
        ax1.axvspan(s.state_before.window_start, s.shift_date,
                     color='grey', alpha=0.08)
        ax1.axvline(s.shift_date, color='black', linewidth=1.2, linestyle='--', alpha=0.5)

    ax1.set_ylabel('Weekly frame proportion')
    ax1.legend(loc='upper right', ncol=4, fontsize=8, framealpha=0.9)
    ax1.set_title('Frame Proportions, Paradigm Shifts & Cascade Attribution — 2018',
                  fontsize=13, fontweight='bold')
    ax1.set_xlim(dates.iloc[0], dates.iloc[-1])

    # --- Row 2: Paradigm composition stacked ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    frame_cols = [f'paradigm_{f}' for f in FRAMES]
    values = timeline[frame_cols].values.T
    totals = values.sum(axis=0)
    totals[totals == 0] = 1
    normed = values / totals

    ax2.stackplot(dates, normed,
                  colors=[FRAME_COLORS[f] for f in FRAMES], alpha=0.85)
    for s in shifts:
        ax2.axvline(s.shift_date, color='black', linewidth=1.2, linestyle='--', alpha=0.5)
    ax2.set_ylabel('Paradigm\ncomposition')
    ax2.set_ylim(0, 1)

    # --- Row 3: Dominant frame strip ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for _, row in timeline.iterrows():
        d = pd.Timestamp(row['date'])
        dom_frames = str(row['dominant_frames']).split(',')
        n = len(dom_frames)
        for j, f in enumerate(dom_frames):
            f = f.strip()
            if f in FRAME_COLORS:
                ax3.barh(j / n, 7, left=mdates.date2num(d) - 3.5,
                         height=1.0 / n,
                         color=FRAME_COLORS[f], edgecolor='none')
    for s in shifts:
        ax3.axvline(s.shift_date, color='black', linewidth=1.5, linestyle='-', alpha=0.7)
        marker, color = SHIFT_TYPE_MARKERS.get(s.shift_type, ('o', 'black'))
        ax3.plot(s.shift_date, 1.15, marker=marker, color=color, markersize=9,
                 transform=ax3.get_xaxis_transform(), clip_on=False, zorder=5)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.5])
    ax3.set_yticklabels(['Dominant'], fontsize=8)
    ax3.grid(False)

    # --- Row 4: Cascade bars with attribution lines ---
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    sorted_c = sorted(cascades, key=lambda c: c.onset_date)
    for i, c in enumerate(sorted_c):
        onset = c.onset_date
        duration = (c.end_date - c.onset_date).days
        color = FRAME_COLORS.get(c.frame, '#999')
        alpha = 0.4 + 0.5 * min(c.total_score, 1.0)

        ax4.barh(i, duration, left=mdates.date2num(onset), height=0.6,
                 color=color, alpha=alpha, edgecolor=color, linewidth=0.8)

        # Classification badge
        badge = {'strong_cascade': 'S', 'moderate_cascade': 'M',
                 'weak_cascade': 'W'}.get(c.classification, '?')
        ax4.text(mdates.date2num(onset) - 5, i, badge,
                 ha='right', va='center', fontsize=8, fontweight='bold',
                 color=color)

    # Attribution arrows
    for s in shifts:
        sd = mdates.date2num(pd.Timestamp(s.shift_date))
        for ac in s.attributed_cascades[:5]:
            cid = ac['cascade_id']
            for j, c in enumerate(sorted_c):
                if c.cascade_id == cid:
                    ax4.annotate(
                        '', xy=(sd, len(sorted_c) + 0.3), xytext=(sd, j),
                        arrowprops=dict(
                            arrowstyle='->', color='grey',
                            alpha=0.3 + 0.5 * ac['attribution_score'],
                            linewidth=0.8 + ac['attribution_score'] * 2,
                        ),
                    )
                    break

    ax4.set_yticks(range(len(sorted_c)))
    ax4.set_yticklabels(
        [f"{c.frame} {c.cascade_id.split('_')[-1]}" for c in sorted_c],
        fontsize=7,
    )
    ax4.set_ylim(-0.5, len(sorted_c) + 1)
    ax4.set_ylabel('Cascades')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator())
    ax4.set_xlabel('2018')

    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# =============================================================================
# Figure 7: Three-role attribution analysis
# =============================================================================

ROLE_COLORS = {
    'amplification':   '#2ca02c',   # Green
    'destabilisation': '#d62728',   # Red
    'dormante':        '#7f7f7f',   # Grey
}

ROLE_LABELS = {
    'amplification':   'Amplification',
    'destabilisation': 'Déstabilisation',
    'dormante':        'Dormante',
}


def fig_role_analysis(shifts, cascades, output_path: Path):
    """Three-panel figure: role distribution, lift vs structural impact,
    role-stratified event contribution."""
    # Collect all attributed cascades across all shifts
    all_attr = []
    for s in shifts:
        for ac in s.attributed_cascades:
            ac_copy = dict(ac)
            ac_copy['shift_id'] = s.shift_id
            ac_copy['shift_date'] = s.shift_date
            all_attr.append(ac_copy)

    if not all_attr:
        logger.warning("No attributed cascades for role analysis")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_pie, ax_scatter, ax_bar = axes

    # --- Panel A: Role distribution (pie chart + counts) ---
    role_counts = {}
    for ac in all_attr:
        role = ac.get('role', 'unknown')
        role_counts[role] = role_counts.get(role, 0) + 1

    roles = ['amplification', 'destabilisation', 'dormante']
    counts = [role_counts.get(r, 0) for r in roles]
    colors = [ROLE_COLORS[r] for r in roles]
    labels = [f"{ROLE_LABELS[r]}\n(n={c})" for r, c in zip(roles, counts)]

    total = sum(counts)
    if total > 0:
        wedges, texts, autotexts = ax_pie.pie(
            counts, labels=labels, colors=colors, autopct='%1.0f%%',
            startangle=90, textprops={'fontsize': 10},
            pctdistance=0.75, labeldistance=1.15,
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight('bold')
    ax_pie.set_title('A. Role Distribution', fontweight='bold', fontsize=12)

    # --- Panel B: Own lift vs structural impact scatter ---
    for ac in all_attr:
        role = ac.get('role', 'dormante')
        color = ROLE_COLORS.get(role, '#999')
        marker = {'amplification': '^', 'destabilisation': 's', 'dormante': 'o'}.get(role, 'o')
        ax_scatter.scatter(
            ac.get('own_lift', 0), ac.get('structural_impact', 0),
            c=color, marker=marker, s=60 + 80 * ac.get('total_score', 0),
            alpha=0.7, edgecolors='white', linewidths=0.5,
        )

    # Draw threshold lines
    from cascade_detector.analysis.paradigm_shift import CascadeShiftAttributor
    default_attr = CascadeShiftAttributor()
    ax_scatter.axvline(default_attr.lift_threshold, color='green', linestyle='--',
                       alpha=0.5, label=f'lift_threshold={default_attr.lift_threshold}')
    ax_scatter.axhline(default_attr.structural_threshold, color='red', linestyle='--',
                       alpha=0.5, label=f'struct_threshold={default_attr.structural_threshold}')

    # Role region labels
    ax_scatter.text(0.95, 0.95, 'Amplification', transform=ax_scatter.transAxes,
                    ha='right', va='top', fontsize=9, color=ROLE_COLORS['amplification'],
                    fontweight='bold', alpha=0.7)
    ax_scatter.text(0.05, 0.95, 'Déstabilisation', transform=ax_scatter.transAxes,
                    ha='left', va='top', fontsize=9, color=ROLE_COLORS['destabilisation'],
                    fontweight='bold', alpha=0.7)
    ax_scatter.text(0.05, 0.05, 'Dormante', transform=ax_scatter.transAxes,
                    ha='left', va='bottom', fontsize=9, color=ROLE_COLORS['dormante'],
                    fontweight='bold', alpha=0.7)

    ax_scatter.set_xlabel('Own lift (frame dominance change)')
    ax_scatter.set_ylabel('Structural impact (cosine distance)')
    ax_scatter.set_title('B. Impact Space', fontweight='bold', fontsize=12)
    ax_scatter.legend(fontsize=8, loc='center right')

    # --- Panel C: Events by role ---
    event_by_role = {}
    for s in shifts:
        for ac in s.attributed_cascades:
            role = ac.get('role', 'unknown')
            cid = ac['cascade_id']
            for c in cascades:
                if c.cascade_id == cid:
                    for evt, cnt in c.dominant_events.items():
                        key = (role, evt)
                        event_by_role[key] = event_by_role.get(key, 0) + cnt
                    break

    # Get top events
    event_totals = {}
    for (role, evt), cnt in event_by_role.items():
        event_totals[evt] = event_totals.get(evt, 0) + cnt

    top_events = sorted(event_totals.items(), key=lambda x: -x[1])[:12]
    event_names = [e[0] for e in top_events]
    clean_names = [e.replace('evt_', '').replace('msg_', '').replace('_', ' ').title()
                   for e in event_names]

    y_pos = np.arange(len(event_names))
    bar_width = 0.25

    for ri, role in enumerate(roles):
        role_vals = [event_by_role.get((role, evt), 0) for evt in event_names]
        ax_bar.barh(y_pos + ri * bar_width, role_vals, height=bar_width,
                    color=ROLE_COLORS[role], alpha=0.85, label=ROLE_LABELS[role],
                    edgecolor='white', linewidth=0.5)

    ax_bar.set_yticks(y_pos + bar_width)
    ax_bar.set_yticklabels(clean_names, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('Occurrence count')
    ax_bar.set_title('C. Events by Cascade Role', fontweight='bold', fontsize=12)
    ax_bar.legend(fontsize=9)

    fig.suptitle('Three-Role Cascade Attribution Analysis — 2018',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# =============================================================================
# Cache loading helpers
# =============================================================================

def _load_cached_results(timeline_path, shifts_path):
    """Rebuild ParadigmShiftResults from cached files."""
    from cascade_detector.analysis.paradigm_shift import (
        ParadigmShiftResults, ParadigmShift, ParadigmState,
    )

    timeline = pd.read_parquet(timeline_path)

    with open(shifts_path) as f:
        shifts_data = json.load(f)

    shifts = []
    for sd in shifts_data.get('shifts', []):
        before = sd.get('state_before', {})
        after = sd.get('state_after', {})

        def _build_state(d):
            return ParadigmState(
                date=pd.Timestamp(d['date']),
                window_start=pd.Timestamp(d['window_start']),
                window_end=pd.Timestamp(d['window_end']),
                dominant_frames=d.get('dominant_frames', []),
                paradigm_type=d.get('paradigm_type', ''),
                paradigm_vector=np.array(d.get('paradigm_vector', [])),
                frame_scores=d.get('frame_scores', {}),
                concentration=d.get('concentration', 0),
                coherence=d.get('coherence', 0),
            )

        shifts.append(ParadigmShift(
            shift_id=sd.get('shift_id', ''),
            shift_date=pd.Timestamp(sd['shift_date']),
            shift_type=sd.get('shift_type', ''),
            entering_frames=sd.get('entering_frames', []),
            exiting_frames=sd.get('exiting_frames', []),
            state_before=_build_state(before),
            state_after=_build_state(after),
            shift_magnitude=sd.get('shift_magnitude', 0),
            vector_distance=sd.get('vector_distance', 0),
            set_jaccard_distance=sd.get('set_jaccard_distance', 0),
            concentration_change=sd.get('concentration_change', 0),
            attributed_cascades=sd.get('attributed_cascades', []),
            attributed_events=sd.get('attributed_events', []),
        ))

    return ParadigmShiftResults(
        shifts=shifts, paradigm_timeline=timeline,
        analysis_period=shifts_data.get('analysis_period', ('', '')),
    )


def _load_cached_cascades(cascades_path):
    """Load cascades from JSON as SimpleNamespace objects."""
    from types import SimpleNamespace

    with open(cascades_path) as f:
        data = json.load(f)

    cascades = []
    for cd in data.get('cascades', []):
        cascades.append(SimpleNamespace(
            cascade_id=cd.get('cascade_id', ''),
            frame=cd.get('frame', ''),
            onset_date=pd.Timestamp(cd.get('onset_date', '')),
            peak_date=pd.Timestamp(cd.get('peak_date', '')),
            end_date=pd.Timestamp(cd.get('end_date', '')),
            total_score=cd.get('total_score', 0),
            classification=cd.get('classification', ''),
            dominant_events=cd.get('dominant_events', {}),
        ))
    return cascades


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--figures-only', action='store_true',
                        help='Skip pipeline, load cached results and regenerate figures')
    args = parser.parse_args()

    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cache_props = OUTPUT_DIR / 'weekly_props.parquet'
    cache_cascades = OUTPUT_DIR / 'cascades_2018.json'
    cache_shifts = OUTPUT_DIR / 'paradigm_shifts.json'
    cache_timeline = OUTPUT_DIR / 'paradigm_timeline.parquet'

    if args.figures_only and cache_timeline.exists() and cache_props.exists():
        logger.info("Loading cached results (--figures-only mode)...")
        weekly_props = pd.read_parquet(cache_props)
        ps_results = _load_cached_results(cache_timeline, cache_shifts)
        cascades = _load_cached_cascades(cache_cascades)
    else:
        # ---------------------------------------------------------------------
        # 1. Run cascade detection pipeline on 2018
        # ---------------------------------------------------------------------
        logger.info("=" * 70)
        logger.info("Running cascade detection on 2018 (test embeddings)...")
        logger.info("=" * 70)

        config = DetectorConfig(
            embedding_dir='data/embeddings-test',
            verbose=True,
        )
        pipeline = CascadeDetectionPipeline(config)
        results = pipeline.run('2018-01-01', '2018-12-31')

        logger.info(f"Cascades: {len(results.cascades)}")
        logger.info(f"Bursts: {len(results.all_bursts)}")

        # Use the paradigm_shifts already computed by the pipeline (Step 5)
        ps_results = results.paradigm_shifts
        cascades = results.cascades

        # Extract weekly proportions
        logger.info("Extracting weekly proportions...")
        indices = getattr(results, '_indices', {})
        temporal_idx = indices.get('temporal', {})

        series_dict = {}
        for frame in FRAMES:
            frame_data = temporal_idx.get(frame, {})
            if isinstance(frame_data, dict):
                wp = frame_data.get('weekly_proportions')
                if wp is not None and not wp.empty:
                    series_dict[frame] = wp

        weekly_props = pd.DataFrame(series_dict)
        for f in FRAMES:
            if f not in weekly_props.columns:
                weekly_props[f] = 0.0
        weekly_props = weekly_props[list(FRAMES)].fillna(0.0)

        # Save caches
        with open(OUTPUT_DIR / 'paradigm_shifts.json', 'w') as f:
            json.dump(ps_results.to_dict(), f, indent=2, default=str)
        if not ps_results.paradigm_timeline.empty:
            ps_results.paradigm_timeline.to_parquet(cache_timeline, index=False)
        weekly_props.to_parquet(cache_props)
        # Save cascades for figure-only reloads
        results.to_json(str(cache_cascades))
        logger.info(f"Cached results to {OUTPUT_DIR}")

    logger.info(f"\n{ps_results.summary()}")

    # -------------------------------------------------------------------------
    # 4. Print detailed shift report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PARADIGM SHIFT REPORT — 2018")
    print("=" * 80)

    for i, s in enumerate(ps_results.shifts):
        print(f"\n{'─' * 80}")
        print(f"  SHIFT #{i+1}: {s.shift_type.replace('_', ' ').upper()}")
        print(f"{'─' * 80}")
        print(f"  Date:         {s.shift_date.strftime('%Y-%m-%d')}")
        print(f"  Magnitude:    {s.shift_magnitude:.4f}")
        print(f"  Vector dist:  {s.vector_distance:.4f}")
        print(f"  Jaccard dist: {s.set_jaccard_distance:.4f}")
        print(f"  Entering:     {', '.join(s.entering_frames) or 'none'}")
        print(f"  Exiting:      {', '.join(s.exiting_frames) or 'none'}")
        print(f"  Before:       {', '.join(s.state_before.dominant_frames)} ({s.state_before.paradigm_type})")
        print(f"  After:        {', '.join(s.state_after.dominant_frames)} ({s.state_after.paradigm_type})")

        if s.attributed_cascades:
            # Group by role
            by_role = {}
            for ac in s.attributed_cascades:
                role = ac.get('role', 'unknown')
                by_role.setdefault(role, []).append(ac)

            print(f"\n  ATTRIBUTED CASCADES ({len(s.attributed_cascades)}):")
            for role in ['amplification', 'destabilisation', 'dormante']:
                role_cascades = by_role.get(role, [])
                if role_cascades:
                    print(f"    [{role.upper()}] ({len(role_cascades)}):")
                    for ac in role_cascades[:5]:
                        extras = ''
                        if role == 'amplification':
                            extras = f" lift={ac.get('own_lift', 0):.4f} dir={ac.get('direction_alignment', 0):.1f}"
                        elif role == 'destabilisation':
                            extras = f" struct={ac.get('structural_impact', 0):.4f} conc_d={ac.get('concentration_disruption', 0):.4f}"
                        elif role == 'dormante':
                            extras = f" lift={ac.get('own_lift', 0):.4f} struct={ac.get('structural_impact', 0):.4f}"
                        print(f"      {ac['cascade_id']} [{ac['frame']}] "
                              f"score={ac['total_score']:.3f} "
                              f"attr={ac['attribution_score']:.3f} "
                              f"({ac['classification']}){extras}")

        if s.attributed_events:
            print(f"\n  TRIGGERING EVENTS:")
            for e in s.attributed_events[:10]:
                print(f"    {e['event']}: {e['count']} occurrences")

    # -------------------------------------------------------------------------
    # 5. Generate figures
    # -------------------------------------------------------------------------
    logger.info("\nGenerating figures...")

    fig_paradigm_timeline(
        ps_results.paradigm_timeline, ps_results.shifts,
        OUTPUT_DIR / 'fig1_paradigm_timeline.png',
    )

    fig_cascade_shift_timeline(
        cascades, ps_results.shifts, ps_results.paradigm_timeline,
        OUTPUT_DIR / 'fig2_cascade_shift_timeline.png',
    )

    fig_shift_detail_panels(
        ps_results.shifts,
        OUTPUT_DIR / 'fig3_shift_details.png',
    )

    fig_shift_cascade_heatmap(
        ps_results.shifts, cascades,
        OUTPUT_DIR / 'fig4_shift_cascade_heatmap.png',
    )

    fig_event_attribution(
        ps_results.shifts,
        OUTPUT_DIR / 'fig5_event_attribution.png',
    )

    fig_comprehensive_panel(
        ps_results.paradigm_timeline, ps_results.shifts,
        cascades, weekly_props,
        OUTPUT_DIR / 'fig6_comprehensive_panel.png',
    )

    fig_role_analysis(
        ps_results.shifts, cascades,
        OUTPUT_DIR / 'fig7_role_analysis.png',
    )

    # -------------------------------------------------------------------------
    # 6. Print role distribution summary
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("THREE-ROLE ATTRIBUTION SUMMARY")
    print(f"{'=' * 80}")

    role_counts = {'amplification': 0, 'destabilisation': 0, 'dormante': 0}
    role_cascades = {'amplification': [], 'destabilisation': [], 'dormante': []}
    for s in ps_results.shifts:
        for ac in s.attributed_cascades:
            role = ac.get('role', 'unknown')
            if role in role_counts:
                role_counts[role] += 1
                role_cascades[role].append(ac)

    total_attr = sum(role_counts.values())
    for role in ['amplification', 'destabilisation', 'dormante']:
        n = role_counts[role]
        pct = 100 * n / total_attr if total_attr > 0 else 0
        print(f"\n  {role.upper()}: {n} ({pct:.0f}%)")
        # Show top 3
        sorted_rc = sorted(role_cascades[role],
                          key=lambda x: x['attribution_score'], reverse=True)
        for ac in sorted_rc[:3]:
            extras = ''
            if role == 'amplification':
                extras = f"  lift={ac.get('own_lift', 0):.4f}"
            elif role == 'destabilisation':
                extras = f"  struct_impact={ac.get('structural_impact', 0):.4f}"
            print(f"    {ac['cascade_id']} [{ac['frame']}] "
                  f"attr={ac['attribution_score']:.3f}{extras}")

    print(f"\n{'=' * 80}")
    print(f"ALL DONE — {len(ps_results.shifts)} shifts detected, 7 figures generated")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
