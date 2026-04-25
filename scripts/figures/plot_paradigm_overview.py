#!/usr/bin/env python3
"""
Generate a publication-quality overview of 2018 paradigm dynamics.

Layout (4 rows, shared x-axis):
  Row A (tall):   8 smoothed frame-dominance curves with episode shading
  Row B (thin):   Dominant-frame regime strip (color-coded by frame)
  Row C (medium): Shift-level dynamics — structural change direction
                  (complexification vs simplification) + reversibility markers
  Row D (medium): Cascade attribution — horizontal bars by frame,
                  strong cascades with thick borders

Right margin: legends + episode summary table with two-level dynamics.

Reads data from results/production/2018/.

Usage:
    python scripts/plot_paradigm_overview.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

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
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

FRAME_COLORS = {
    'Cult': '#E64B35', 'Eco': '#4DBBD5', 'Envt': '#00A087',
    'Pbh': '#3C5488', 'Just': '#F39B7F', 'Pol': '#8491B4',
    'Sci': '#91D1C2', 'Secu': '#B09C85',
}
FRAME_NAMES = {
    'Cult': 'Cultural', 'Eco': 'Economic', 'Envt': 'Environmental',
    'Pbh': 'Public Health', 'Just': 'Justice', 'Pol': 'Political',
    'Sci': 'Scientific', 'Secu': 'Security',
}
FRAMES = ['Cult', 'Eco', 'Envt', 'Pbh', 'Just', 'Pol', 'Sci', 'Secu']
PTYPE_ABBREV = {
    'Mono-paradigm': 'M', 'Dual-paradigm': 'D',
    'Triple-paradigm': 'T', 'Quad-paradigm': 'Q',
}

DATA_DIR = PROJECT_ROOT / 'results' / 'production' / '2018'
OUT_PATH = PROJECT_ROOT / 'results' / 'fig_paradigm_overview_2018.png'


def load_data():
    """Load all data from production 2018 results."""
    timeline = pd.read_parquet(DATA_DIR / 'paradigm_shifts' / 'paradigm_timeline.parquet')
    with open(DATA_DIR / 'paradigm_shifts' / 'shifts.json') as f:
        shifts = json.load(f)
    with open(DATA_DIR / 'paradigm_shifts' / 'episodes.json') as f:
        episodes = json.load(f)
    with open(DATA_DIR / 'cascades.json') as f:
        cascades_raw = json.load(f)
    # Handle both list and dict formats
    if isinstance(cascades_raw, dict):
        cascades = cascades_raw.get('cascades', [])
    else:
        cascades = cascades_raw
    return timeline, shifts, episodes, cascades


def _episode_span(episode_data):
    """Get start/end timestamps from an episode dict."""
    return (pd.Timestamp(episode_data['start_date']),
            pd.Timestamp(episode_data['end_date']))


def main():
    timeline, shifts, episodes, all_cascades = load_data()
    logger.info(f"Loaded: {len(timeline)} paradigm states, {len(shifts)} shifts, "
                f"{len(episodes)} episodes, {len(all_cascades)} cascades")

    # ── Build cascade lookup ──────────────────────────────────────────────
    cascade_by_id = {c['cascade_id']: c for c in all_cascades}

    # ── Attributed cascade IDs (any shift attribution) ────────────────────
    attributed_ids = set()
    for s in shifts:
        for ac in s.get('attributed_cascades', []):
            attributed_ids.add(ac['cascade_id'])
    attr_cascades = [c for c in all_cascades if c['cascade_id'] in attributed_ids]
    logger.info(f"Shift-attributed cascades: {len(attr_cascades)}")

    # ── X-axis limits ─────────────────────────────────────────────────────
    xlim = (pd.Timestamp('2018-03-20'), pd.Timestamp('2019-01-10'))

    # ── Episode colors (muted, consistent) ────────────────────────────────
    ep_colors = ['#C0392B', '#2471A3', '#1E8449', '#7D3C98', '#D4AC0D']

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE LAYOUT
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(4, 1, height_ratios=[5, 1.0, 2.5, 3.5],
                          hspace=0.04, left=0.065, right=0.68,
                          top=0.925, bottom=0.05)
    ax_a = fig.add_subplot(gs[0])                  # dominance curves
    ax_b = fig.add_subplot(gs[1], sharex=ax_a)     # regime strip
    ax_c = fig.add_subplot(gs[2], sharex=ax_a)     # shift dynamics
    ax_d = fig.add_subplot(gs[3], sharex=ax_a)     # cascade bars

    all_axes = [ax_a, ax_b, ax_c, ax_d]

    # ══════════════════════════════════════════════════════════════════════
    # EPISODE SHADING (across all panels)
    # ══════════════════════════════════════════════════════════════════════
    for ei, ep in enumerate(episodes):
        color = ep_colors[ei % len(ep_colors)]
        d0, d1 = _episode_span(ep)
        pad = pd.Timedelta(days=2)
        for ax in all_axes:
            ax.axvspan(d0 - pad, d1 + pad,
                       facecolor=color, alpha=0.07, zorder=0)

    # ══════════════════════════════════════════════════════════════════════
    # PANEL A — Frame dominance scores (from paradigm timeline)
    # ══════════════════════════════════════════════════════════════════════
    tl = timeline.sort_values('date').reset_index(drop=True)
    dates = pd.to_datetime(tl['date'])

    for frame in FRAMES:
        col = f'paradigm_{frame}'
        if col not in tl.columns:
            continue
        vals = tl[col].values.astype(float)
        # Light smoothing (7-day centered rolling mean)
        smooth = pd.Series(vals).rolling(7, center=True, min_periods=1).mean().values
        lw = 2.2 if frame in ('Pol', 'Eco') else 1.4
        alpha = 0.92 if frame in ('Pol', 'Eco') else 0.72
        ax_a.plot(dates, smooth, color=FRAME_COLORS[frame],
                  linewidth=lw, alpha=alpha,
                  label=f'{FRAME_NAMES[frame]} ({frame})', zorder=2)

    # Episode labels at top
    y_top = ax_a.get_ylim()[1] if ax_a.get_ylim()[1] > 0 else 1.0
    for ei, ep in enumerate(episodes):
        color = ep_colors[ei % len(ep_colors)]
        d0, d1 = _episode_span(ep)
        mid = d0 + (d1 - d0) / 2
        n_sh = ep['n_shifts']
        ax_a.annotate(f'E{ei+1} ({n_sh})', xy=(mid, y_top * 0.97),
                      ha='center', va='top', fontsize=9,
                      fontweight='bold', color=color,
                      bbox=dict(boxstyle='round,pad=0.2',
                                facecolor='white', alpha=0.85,
                                edgecolor=color, linewidth=1.0),
                      zorder=5)

    ax_a.set_ylabel('Dominance score\n(4-method consensus)', fontsize=10)
    ax_a.set_xlim(xlim)
    ax_a.grid(True, alpha=0.10, linewidth=0.3)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.tick_params(labelbottom=False)

    # Frame legend
    ax_a.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                fontsize=9, framealpha=0.95, edgecolor='#cccccc',
                title='Frames', title_fontsize=10)

    # ══════════════════════════════════════════════════════════════════════
    # PANEL B — Dominant-frame regime strip
    # ══════════════════════════════════════════════════════════════════════
    for idx in range(len(dates)):
        d = dates.iloc[idx]
        d_next = dates.iloc[idx + 1] if idx < len(dates) - 1 \
            else d + pd.Timedelta(days=1)
        dom_str = str(tl.iloc[idx]['dominant_frames'])
        dom_list = [f.strip() for f in dom_str.split(',')]
        n = len(dom_list)
        ptype = str(tl.iloc[idx]['paradigm_type'])
        abbrev = PTYPE_ABBREV.get(ptype, '?')

        for j, f in enumerate(dom_list):
            if f in FRAME_COLORS:
                ax_b.axvspan(d, d_next, ymin=j / n, ymax=(j + 1) / n,
                             facecolor=FRAME_COLORS[f], alpha=0.85,
                             edgecolor='white', linewidth=0.2)

        # Paradigm type label for non-Dual
        seg_width = (d_next - d).days
        if abbrev != 'D' and seg_width >= 2:
            mid_d = d + (d_next - d) / 2
            ax_b.text(mdates.date2num(mid_d), 0.5, abbrev,
                      ha='center', va='center', fontsize=6.5,
                      fontweight='bold', color='white', zorder=6)

    ax_b.set_ylabel('Dominant\nframes', fontsize=9, fontweight='bold',
                     rotation=0, labelpad=40, va='center')
    ax_b.set_yticks([])
    ax_b.tick_params(labelbottom=False)
    for sp in ax_b.spines.values():
        sp.set_visible(False)

    # Paradigm type legend
    ptype_legend = [
        mpatches.Patch(facecolor='#999999', label='D = Dual'),
        mpatches.Patch(facecolor='#777777', label='T = Triple'),
        mpatches.Patch(facecolor='#555555', label='Q = Quad'),
    ]
    ax_b.legend(handles=ptype_legend, loc='upper left',
                bbox_to_anchor=(1.02, 1.5), fontsize=8,
                framealpha=0.95, edgecolor='#cccccc',
                title='Paradigm type', title_fontsize=9, ncol=3)

    # ══════════════════════════════════════════════════════════════════════
    # PANEL C — Shift-level dynamics (NEW)
    # Vertical stems: height = regime_duration_days (capped for readability)
    # Color: structural_change direction (blue=complexification, red=simplification)
    # Marker: circle=non-reversible, X=reversible
    # ══════════════════════════════════════════════════════════════════════
    from matplotlib.colors import TwoSlopeNorm

    shift_dates = [pd.Timestamp(s['shift_date']) for s in shifts]
    struct_changes = [s.get('structural_change', 0) for s in shifts]
    regime_durs = [max(s.get('regime_duration_days', 1), 0) for s in shifts]
    reversibles = [s.get('reversible', False) for s in shifts]
    magnitudes = [s.get('shift_magnitude', 0.15) for s in shifts]

    # Colormap: blue for complexification (+), red for simplification (-)
    cmap = plt.cm.RdBu_r
    max_abs = max(abs(sc) for sc in struct_changes) if struct_changes else 1
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    for i, (sd, sc, rd, rev, mag) in enumerate(
            zip(shift_dates, struct_changes, regime_durs, reversibles, magnitudes)):
        color = cmap(norm(sc))
        marker = 'X' if rev else 'o'
        size = 25 + mag * 300

        # Stem from y=0 to y=regime_duration
        ax_c.plot([sd, sd], [0, min(rd, 40)], color=color,
                  linewidth=1.0, alpha=0.5, zorder=1)
        ax_c.scatter(sd, min(rd, 40), color=color, marker=marker,
                     s=size, edgecolors='#333333' if rev else 'none',
                     linewidths=0.7, alpha=0.85, zorder=3)

    ax_c.set_ylabel('Regime duration\n(days)', fontsize=9)
    ax_c.set_ylim(-1, 42)
    ax_c.axhline(0, color='#888888', linewidth=0.5, zorder=0)
    ax_c.grid(True, axis='y', alpha=0.10, linewidth=0.3)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.tick_params(labelbottom=False)

    # Shift dynamics legend
    shift_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2166AC',
               markersize=7, label='Complexification (+frames)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#B2182B',
               markersize=7, label='Simplification (-frames)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
               markersize=7, label='Recomposition (0)'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#555555',
               markeredgecolor='#333333', markersize=8,
               label='Locally reversible'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555',
               markersize=7, label='Non-reversible'),
    ]
    ax_c.legend(handles=shift_legend, loc='upper left',
                bbox_to_anchor=(1.02, 1.05), fontsize=8,
                framealpha=0.95, edgecolor='#cccccc',
                title='Shift dynamics', title_fontsize=9)

    # ══════════════════════════════════════════════════════════════════════
    # PANEL D — Cascade attribution bars
    # ══════════════════════════════════════════════════════════════════════
    frame_y = {f: i for i, f in enumerate(FRAMES)}

    # Alternating row shading
    for i in range(len(FRAMES)):
        if i % 2 == 0:
            ax_d.axhspan(i - 0.45, i + 0.45, facecolor='#f5f5f5', zorder=0)

    for c in attr_cascades:
        frame = c['frame']
        if frame not in frame_y:
            continue
        y = frame_y[frame]
        onset = pd.Timestamp(c['onset_date'])
        end = pd.Timestamp(c['end_date'])
        dur = max((end - onset).days, 3)
        score = c['total_score']
        strong = c['classification'] == 'strong_cascade'

        h = 0.55 if strong else 0.35
        ec = '#222222' if strong else FRAME_COLORS[frame]
        lw = 2.0 if strong else 0.6
        al = 0.92 if strong else 0.55

        ax_d.barh(y, dur, left=mdates.date2num(onset),
                  height=h, color=FRAME_COLORS[frame],
                  alpha=al, edgecolor=ec, linewidth=lw, zorder=3)

        # Score label inside wide bars
        mid = mdates.date2num(onset) + dur / 2
        if dur >= 14:
            ax_d.text(mid, y, f'.{int(score * 100):02d}',
                      ha='center', va='center', fontsize=7,
                      fontweight='bold', color='white', zorder=4)

    ax_d.set_yticks(range(len(FRAMES)))
    ax_d.set_yticklabels(FRAMES, fontsize=9, fontweight='bold')
    for tl_label, frame in zip(ax_d.get_yticklabels(), FRAMES):
        tl_label.set_color(FRAME_COLORS[frame])
    ax_d.set_ylim(len(FRAMES) - 0.5, -0.6)
    ax_d.set_ylabel('Attributed\ncascades', fontsize=9, fontweight='bold')
    ax_d.grid(True, axis='x', alpha=0.10, linewidth=0.3)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax_d.xaxis.set_major_locator(mdates.MonthLocator())
    ax_d.set_xlabel('2018', fontsize=11, fontweight='bold')

    # Cascade legend
    cascade_legend = [
        mpatches.Patch(facecolor='#888888', alpha=0.92, edgecolor='#222222',
                       linewidth=2.0, label='Strong (>=0.65)'),
        mpatches.Patch(facecolor='#888888', alpha=0.50,
                       label='Moderate (0.40-0.65)'),
    ]
    ax_d.legend(handles=cascade_legend, loc='upper left',
                bbox_to_anchor=(1.02, 1.0), fontsize=8,
                framealpha=0.95, edgecolor='#cccccc',
                title='Cascade classification', title_fontsize=9)

    # ══════════════════════════════════════════════════════════════════════
    # RIGHT MARGIN — Episode summary with two-level dynamics
    # ══════════════════════════════════════════════════════════════════════
    lines = ['Episode dynamics (two-level qualification)', '']
    for ei, ep in enumerate(episodes):
        d0, d1 = _episode_span(ep)
        n_sh = ep['n_shifts']
        dur = ep['duration_days']
        rev = 'Yes' if ep['reversible'] else 'No'
        max_c = ep['max_complexity']
        stab = ep['regime_after_duration_days']
        stab_str = f'{stab}d stable' if stab > 0 else 'unstable'
        net_sc = ep['net_structural_change']

        # Frames before/after
        fb = ','.join(ep['dominant_frames_before'])
        fa = ','.join(ep['dominant_frames_after'])

        if dur <= 7:
            date_str = d0.strftime('%b %d')
        else:
            date_str = f"{d0.strftime('%b %d')} -- {d1.strftime('%b %d')}"

        lines.append(f"E{ei+1}  {date_str}  ({dur}d, {n_sh} shifts)")
        lines.append(f"    [{fb}] -> [{fa}]")
        lines.append(f"    Reversible: {rev}  |  Max complexity: {max_c}")
        lines.append(f"    Net structural: {net_sc:+d}  |  After: {stab_str}")
        lines.append('')

    # Global summary
    n_rev = sum(1 for ep in episodes if ep['reversible'])
    lines.append(f"Global: {n_rev}/{len(episodes)} episodes reversible")
    lines.append(f"57 shifts: 19 complexif. / 20 simplif. / 18 recomp.")
    lines.append(f"33% locally reversible, 100% episode-reversible")

    summary_text = '\n'.join(lines)

    fig.text(0.695, 0.40, summary_text, fontsize=7.8,
             va='top', ha='left', linespacing=1.5,
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fafafa',
                       edgecolor='#aaaaaa', alpha=0.95, linewidth=0.8))

    # ── Title ─────────────────────────────────────────────────────────────
    fig.suptitle(
        'Media Frame Dynamics & Cascade-Driven Paradigm Shifts -- 2018',
        fontsize=15, fontweight='bold', y=0.97)
    fig.text(0.5, 0.945,
             '9,754 articles  |  38 cascades (4 strong)  |  281 paradigm states  |  57 shifts  |  5 episodes',
             ha='center', fontsize=9, color='#666666')

    # ── Save ──────────────────────────────────────────────────────────────
    fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    logger.info(f"Saved: {OUT_PATH}")


if __name__ == '__main__':
    main()
