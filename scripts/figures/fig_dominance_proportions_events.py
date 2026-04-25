#!/usr/bin/env python3
"""
5-panel overview of 2018 climate change framing dynamics.

Layout (5 rows, shared x-axis):
  Row A: Frame dominance index (paradigm_* from 4-method consensus)
  Row B: Mean daily frame proportions (sentence-level)
  Row C: Cascade periods (horizontal bars by frame)
  Row D: Event occurrence peaks (all occurrences, colored by cascade attribution)
  Row E: Event clusters (meta-events) as horizontal spans with labels

Reads paradigm + temporal data from results/production/2018/.
Loads cached pipeline results from results/cache/results_2018.pkl
(run scripts/run/run_2018.py first to populate the cache).

Usage:
    python scripts/figures/fig_dominance_proportions_events.py
"""

import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

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
    'legend.fontsize': 8,
})

FRAMES = ['Cult', 'Eco', 'Envt', 'Pbh', 'Just', 'Pol', 'Sci', 'Secu']
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

EVENT_COLORS = {
    'evt_weather': '#E74C3C',
    'evt_meeting': '#3498DB',
    'evt_publication': '#2ECC71',
    'evt_election': '#9B59B6',
    'evt_policy': '#F39C12',
    'evt_judiciary': '#1ABC9C',
    'evt_cultural': '#E67E22',
    'evt_protest': '#C0392B',
}
EVENT_NAMES = {
    'evt_weather': 'Weather',
    'evt_meeting': 'Meeting',
    'evt_publication': 'Publication',
    'evt_election': 'Election',
    'evt_policy': 'Policy',
    'evt_judiciary': 'Judiciary',
    'evt_cultural': 'Cultural',
    'evt_protest': 'Protest',
}

RESULTS_DIR = PROJECT_ROOT / 'results' / 'production' / '2018'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'figures'
CACHE_FILE = PROJECT_ROOT / 'results' / 'cache' / 'results_2018.pkl'


def load_paradigm_timeline():
    """Load paradigm dominance timeline from production results."""
    path = RESULTS_DIR / 'paradigm_shifts' / 'paradigm_timeline.parquet'
    tl = pd.read_parquet(path)
    tl['date'] = pd.to_datetime(tl['date'])
    return tl.sort_values('date')


def load_daily_proportions():
    """Load daily frame proportions from production results."""
    path = RESULTS_DIR / 'indices' / 'temporal_daily_proportions.parquet'
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_results():
    """Load cached pipeline results, or run pipeline if no cache."""
    if CACHE_FILE.exists():
        logger.info(f"Loading cached results from {CACHE_FILE}...")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    logger.info("No cache found, running pipeline...")
    from cascade_detector.core.config import DetectorConfig
    from cascade_detector.pipeline import CascadeDetectionPipeline

    embedding_dir = os.environ.get('EMBEDDING_DIR', 'data/embeddings')
    config = DetectorConfig(embedding_dir=embedding_dir, verbose=False)
    pipeline = CascadeDetectionPipeline(config)
    results = pipeline.run('2018-01-01', '2018-12-31')

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Cache saved: {CACHE_FILE}")
    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info("Loading paradigm timeline...")
    timeline = load_paradigm_timeline()

    logger.info("Loading daily proportions...")
    proportions = load_daily_proportions()

    logger.info("Loading pipeline results...")
    results = load_results()
    cascades = results.cascades
    event_clusters = results.event_clusters
    all_occurrences = getattr(results, 'all_occurrences', [])
    cascade_attributions = getattr(results, 'cascade_attributions', [])

    # Sort cascades by frame then onset
    cascades = sorted(cascades, key=lambda c: (FRAMES.index(c.frame), c.onset_date))

    # Build occurrence → cascade frame mapping from attributions
    occ_id_to_frames = {}
    for attr in cascade_attributions:
        cid = attr.cascade_id
        oid = attr.occurrence_id
        # Find the cascade frame
        for c in cascades:
            if c.cascade_id == cid:
                occ_id_to_frames.setdefault(oid, []).append(c.frame)
                break

    # Collect all event occurrence peaks (database-first: all occurrences)
    occ_peaks = []
    for occ in all_occurrences:
        frames = occ_id_to_frames.get(occ.occurrence_id, [])
        occ_peaks.append({
            'occurrence_id': occ.occurrence_id,
            'event_type': occ.event_type,
            'peak_date': occ.peak_date,
            'n_articles': occ.n_articles,
            'effective_mass': occ.effective_mass,
            'confidence': occ.confidence,
            'low_confidence': occ.low_confidence,
            'attributed_frames': frames,
            'n_cascades': len(frames),
        })

    n_attributed = sum(1 for o in occ_peaks if o['n_cascades'] > 0)
    logger.info(f"  {len(occ_peaks)} total occurrences, "
                f"{n_attributed} attributed to cascades")

    # Save event cluster data as JSON for analysis
    cluster_json_path = OUTPUT_DIR / 'event_clusters_2018.json'
    cluster_data = []
    for ec in event_clusters:
        cd = ec.to_dict()
        cd['occurrence_details'] = []
        for occ in ec.occurrences:
            cd['occurrence_details'].append({
                'occurrence_id': occ.occurrence_id,
                'event_type': occ.event_type,
                'peak_date': occ.peak_date.isoformat(),
                'core_start': occ.core_start.isoformat(),
                'core_end': occ.core_end.isoformat(),
                'n_articles': occ.n_articles,
                'effective_mass': occ.effective_mass,
                'media_count': occ.media_count,
                'temporal_intensity': occ.temporal_intensity,
                'doc_ids': [str(d) for d in occ.doc_ids[:20]],
            })
        cluster_data.append(cd)
    with open(cluster_json_path, 'w') as f:
        json.dump(cluster_data, f, indent=2, default=str)
    logger.info(f"Saved cluster data: {cluster_json_path}")

    # ── Date range ──────────────────────────────────────────────────────────
    date_min = pd.Timestamp('2018-01-01')
    date_max = pd.Timestamp('2018-12-31')

    # ── Create figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 20))
    gs = gridspec.GridSpec(
        5, 1,
        height_ratios=[2.5, 2.5, 1.8, 2.2, 4.0],
        hspace=0.08,
        left=0.06, right=0.88, top=0.96, bottom=0.03,
    )

    ax_a = fig.add_subplot(gs[0])   # Dominance index
    ax_b = fig.add_subplot(gs[1], sharex=ax_a)  # Frame proportions
    ax_c = fig.add_subplot(gs[2], sharex=ax_a)  # Cascades
    ax_d = fig.add_subplot(gs[3], sharex=ax_a)  # Event peaks
    ax_e = fig.add_subplot(gs[4], sharex=ax_a)  # Event clusters

    # ── ROW A: Frame Dominance Index ──────────────────────────────────────
    for frame in FRAMES:
        col = f'paradigm_{frame}'
        if col not in timeline.columns:
            continue
        vals = timeline[col].values.astype(float)
        dates = timeline['date'].values
        smooth = pd.Series(vals).rolling(7, center=True, min_periods=1).mean()
        ax_a.plot(dates, smooth.values, color=FRAME_COLORS[frame],
                  linewidth=2.0, alpha=0.9, label=FRAME_NAMES[frame])

    ax_a.set_ylabel('Dominance index')
    ax_a.set_title('A. Frame dominance index (12-week window, 4-method consensus)',
                    loc='left', fontweight='bold')
    ax_a.set_xlim(date_min, date_max)
    ax_a.legend(loc='upper left', ncol=4, framealpha=0.9, fontsize=8)
    ax_a.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax_a.axvspan(date_min, pd.Timestamp('2018-03-31'), alpha=0.05, color='gray')
    ax_a.text(pd.Timestamp('2018-02-15'), ax_a.get_ylim()[1] * 0.5,
              '12-week\nwindow\nwarm-up', ha='center', va='center',
              fontsize=7, color='gray', fontstyle='italic')
    plt.setp(ax_a.get_xticklabels(), visible=False)

    # ── ROW B: Mean Daily Frame Proportions ──────────────────────────────
    for frame in FRAMES:
        fdf = proportions[proportions['frame'] == frame].sort_values('date')
        dates = fdf['date'].values
        vals = fdf['proportion'].values

        smooth = pd.Series(vals).rolling(7, center=True, min_periods=1).mean()
        ax_b.plot(dates, smooth.values, color=FRAME_COLORS[frame],
                  linewidth=1.8, alpha=0.85, label=FRAME_NAMES[frame])

    ax_b.set_ylabel('Mean sentence proportion')
    ax_b.set_title('B. Daily mean frame proportion (7-day rolling average)',
                    loc='left', fontweight='bold')
    ax_b.grid(axis='y', alpha=0.3, linewidth=0.5)
    plt.setp(ax_b.get_xticklabels(), visible=False)

    # ── ROW C: Cascade Periods ────────────────────────────────────────────
    frame_y = {f: i for i, f in enumerate(FRAMES)}

    for c in cascades:
        y = frame_y[c.frame]
        onset = mdates.date2num(pd.Timestamp(c.onset_date))
        end = mdates.date2num(pd.Timestamp(c.end_date))
        dur = end - onset

        if c.classification == 'strong_cascade':
            h, edge_lw, alpha = 0.60, 1.5, 0.75
        elif c.classification == 'moderate_cascade':
            h, edge_lw, alpha = 0.45, 0.8, 0.55
        else:
            h, edge_lw, alpha = 0.30, 0.5, 0.40

        ax_c.barh(y, dur, left=onset, height=h,
                  color=FRAME_COLORS[c.frame], alpha=alpha,
                  edgecolor='black', linewidth=edge_lw)

        if c.classification == 'strong_cascade':
            mid_x = onset + dur / 2
            ax_c.text(mid_x, y + 0.35, f'{c.total_score:.2f}',
                      ha='center', va='bottom', fontsize=7,
                      fontweight='bold', color='black')

    ax_c.set_yticks(list(range(len(FRAMES))))
    ax_c.set_yticklabels([FRAME_NAMES[f] for f in FRAMES], fontsize=8)
    ax_c.set_ylim(-0.5, len(FRAMES) - 0.5)
    ax_c.set_title('C. Detected cascades (bar width = duration, label = score)',
                    loc='left', fontweight='bold')
    ax_c.grid(axis='x', alpha=0.3, linewidth=0.5)
    plt.setp(ax_c.get_xticklabels(), visible=False)

    legend_elements = [
        mpatches.Patch(facecolor='gray', alpha=0.75, edgecolor='black',
                       linewidth=1.5, label='Strong'),
        mpatches.Patch(facecolor='gray', alpha=0.55, edgecolor='black',
                       linewidth=0.8, label='Moderate'),
        mpatches.Patch(facecolor='gray', alpha=0.40, edgecolor='black',
                       linewidth=0.5, label='Weak'),
    ]
    ax_c.legend(handles=legend_elements, loc='upper right', fontsize=7,
                title='Classification', title_fontsize=7, framealpha=0.9)

    # ── ROW D: Event Occurrence Peaks ─────────────────────────────────────
    evt_types = sorted(set(o['event_type'] for o in occ_peaks)) if occ_peaks else sorted(EVENT_COLORS.keys())
    evt_y = {e: i for i, e in enumerate(evt_types)}

    # Generate cluster color palette for visual grouping
    n_clusters = len(event_clusters)
    cluster_cmap = matplotlib.colormaps.get_cmap('tab20').resampled(max(n_clusters, 1))
    cluster_colors = {ec.cluster_id: cluster_cmap(i % 20) for i, ec in enumerate(event_clusters)}

    # Draw cluster spans as background bands on Row D
    for ec in event_clusters:
        if ec.n_occurrences < 2:
            continue
        cluster_evt_types = set()
        cluster_dates = []
        for occ in ec.occurrences:
            cluster_evt_types.add(occ.event_type)
            cluster_dates.append(occ.peak_date)
        if not cluster_dates:
            continue
        x_min = mdates.date2num(min(cluster_dates) - pd.Timedelta(days=3))
        x_max = mdates.date2num(max(cluster_dates) + pd.Timedelta(days=3))
        y_positions = [evt_y[et] for et in cluster_evt_types if et in evt_y]
        if not y_positions:
            continue
        y_min = min(y_positions) - 0.4
        y_max = max(y_positions) + 0.4
        color = cluster_colors[ec.cluster_id]
        rect = mpatches.FancyBboxPatch(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            boxstyle='round,pad=0.02',
            facecolor=color, alpha=0.15, edgecolor=color,
            linewidth=1.5, linestyle='--', zorder=1,
        )
        ax_d.add_patch(rect)

    # Plot all occurrences (database-first v3)
    # Color by event type; marker shape = attribution; edge = confidence
    for o in occ_peaks:
        y = evt_y.get(o['event_type'])
        if y is None:
            continue
        x = mdates.date2num(pd.Timestamp(o['peak_date']))
        size = max(12, min(80, 4 * np.sqrt(o['effective_mass'])))
        evt_color = EVENT_COLORS.get(o['event_type'], '#888888')
        alpha = min(0.95, max(0.3, o['confidence']))
        conf = o['confidence']

        if o['low_confidence']:
            marker = 'X'
        elif o['n_cascades'] > 0:
            marker = 'D'  # diamond = attributed to cascade
        else:
            marker = 'o'  # circle = unattributed

        # Edge encodes confidence level
        edge_width = 0.3 + 1.5 * conf
        if conf >= 0.6:
            edge_color = 'black'
        elif conf >= 0.4:
            edge_color = '#555555'
        else:
            edge_color = '#AAAAAA'

        ax_d.scatter(x, y, s=size, c=evt_color, alpha=alpha, marker=marker,
                     edgecolors=edge_color, linewidths=edge_width, zorder=3)

    ax_d.set_yticks(list(range(len(evt_types))))
    ax_d.set_yticklabels([EVENT_NAMES.get(e, e) for e in evt_types], fontsize=8)
    ax_d.set_ylim(-0.5, len(evt_types) - 0.5)
    n_total = len(occ_peaks)
    n_attr = sum(1 for o in occ_peaks if o['n_cascades'] > 0)
    ax_d.set_title(f'D. Event occurrences (database-first): {n_total} total, '
                    f'{n_attr} attributed to cascades',
                    loc='left', fontweight='bold')
    ax_d.grid(axis='x', alpha=0.3, linewidth=0.5)
    plt.setp(ax_d.get_xticklabels(), visible=False)

    # Legend for Row D: markers, mass scale, confidence scale
    d_legend = [
        # Marker shapes
        plt.scatter([], [], c='gray', marker='D', edgecolors='black',
                    linewidths=1.2, s=30, label='Cascade-attributed'),
        plt.scatter([], [], c='gray', marker='o', edgecolors='#555555',
                    linewidths=0.8, s=30, label='Unattributed'),
        plt.scatter([], [], c='gray', marker='X', edgecolors='#AAAAAA',
                    linewidths=0.5, s=30, label='Low confidence'),
        # Mass scale (effective_mass → size)
        plt.scatter([], [], c='white', edgecolors='black', linewidths=0.5,
                    s=max(12, 4 * np.sqrt(2)), label='mass=2'),
        plt.scatter([], [], c='white', edgecolors='black', linewidths=0.5,
                    s=max(12, 4 * np.sqrt(10)), label='mass=10'),
        plt.scatter([], [], c='white', edgecolors='black', linewidths=0.5,
                    s=max(12, 4 * np.sqrt(50)), label='mass=50'),
        # Confidence scale (edge color + width)
        plt.scatter([], [], c='white', edgecolors='black', linewidths=1.8,
                    s=25, marker='s', label='High conf (≥0.6)'),
        plt.scatter([], [], c='white', edgecolors='#555555', linewidths=0.9,
                    s=25, marker='s', label='Med conf (0.4–0.6)'),
        plt.scatter([], [], c='white', edgecolors='#AAAAAA', linewidths=0.5,
                    s=25, marker='s', label='Low conf (<0.4)'),
    ]
    ax_d.legend(handles=d_legend, loc='upper right', fontsize=6.5,
                framealpha=0.9, ncol=3, columnspacing=1.0)

    # ── ROW E: Event Clusters (meta-events) ───────────────────────────────
    # Sort by strength descending (hierarchy of force)
    sorted_clusters = sorted(event_clusters, key=lambda ec: ec.strength, reverse=True)

    notable_clusters = [ec for ec in sorted_clusters
                        if ec.n_occurrences >= 2 or ec.strength >= 0.3]
    if not notable_clusters:
        notable_clusters = sorted_clusters[:15]
    # Cap display to top 40 for readability (already sorted by strength desc)
    MAX_DISPLAY = 40
    if len(notable_clusters) > MAX_DISPLAY:
        notable_clusters = notable_clusters[:MAX_DISPLAY]
    n_hidden = len(event_clusters) - len(notable_clusters)

    # Confidence colormap (red→yellow→green)
    conf_cmap = matplotlib.colormaps['RdYlGn']

    for i, ec in enumerate(notable_clusters):
        y = i
        x_start = mdates.date2num(ec.core_start)
        x_end = mdates.date2num(ec.core_end)
        x_peak = mdates.date2num(ec.peak_date)
        width = max(x_end - x_start, 2)

        # Use type_ranking if available, else fallback to type counts
        if ec.type_ranking:
            dominant_type = ec.type_ranking[0][0]
        else:
            type_counts = {}
            for occ in ec.occurrences:
                type_counts[occ.event_type] = type_counts.get(occ.event_type, 0) + 1
            dominant_type = max(type_counts, key=type_counts.get) if type_counts else 'evt_weather'
        color = EVENT_COLORS.get(dominant_type, '#888888')

        h = 0.3 + 0.5 * ec.strength
        alpha = 0.4 + 0.4 * ec.strength

        # Border encodes strength level
        if ec.strength >= 0.5:
            bar_ec, bar_lw = 'black', 1.5
        elif ec.strength >= 0.3:
            bar_ec, bar_lw = '#666666', 1.0
        else:
            bar_ec, bar_lw = '#AAAAAA', 0.5

        ax_e.barh(y, width, left=x_start, height=h,
                  color=color, alpha=alpha,
                  edgecolor=bar_ec, linewidth=bar_lw)

        ax_e.plot(x_peak, y, 'k|', markersize=8, markeredgewidth=1.5, zorder=4)

        # Confidence indicator: small colored square after bar
        mean_conf = np.mean([occ.confidence for occ in ec.occurrences]) if ec.occurrences else 0
        conf_color = conf_cmap(mean_conf)
        ax_e.plot(x_end + 0.8, y, 's', color=conf_color, markersize=6,
                  markeredgecolor='black', markeredgewidth=0.3, zorder=5)

        # Label: types + type ranking colored squares + s= c=
        types_str = ', '.join(
            EVENT_NAMES.get(t, t) for t in sorted(ec.event_types)
        )
        multi_marker = ' [M]' if ec.is_multi_type else ''
        label = f'C{ec.cluster_id}: {types_str}{multi_marker} (s={ec.strength:.2f}, c={mean_conf:.2f})'

        # Text style by strength level
        if ec.strength >= 0.5:
            txt_color, txt_weight = 'black', 'bold'
        elif ec.strength >= 0.3:
            txt_color, txt_weight = 'black', 'normal'
        else:
            txt_color, txt_weight = '#888888', 'normal'

        ax_e.text(x_end + 2.5, y, label, va='center', ha='left',
                  fontsize=6.5, color=txt_color, fontweight=txt_weight,
                  bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                            edgecolor='gray', alpha=0.8, linewidth=0.3))

        # Type ranking colored squares in label area (for multi-type clusters)
        if ec.is_multi_type and ec.type_ranking and len(ec.type_ranking) > 1:
            for j, (t, _score) in enumerate(ec.type_ranking[1:4]):  # up to 3 secondary types
                sq_color = EVENT_COLORS.get(t, '#888888')
                # Place small squares just before the text label
                ax_e.plot(x_end + 1.5 + j * 0.7, y, 's', color=sq_color,
                          markersize=4, markeredgecolor='white',
                          markeredgewidth=0.2, zorder=5)

    ax_e.set_ylim(-0.5, max(len(notable_clusters) - 0.5, 0.5))
    ax_e.set_yticks([])
    ax_e.set_title(
        f'E. Event clusters / meta-events — sorted by strength '
        f'({len(event_clusters)} total, '
        f'{sum(1 for ec in event_clusters if ec.is_multi_type)} multi-type)',
        loc='left', fontweight='bold')
    ax_e.grid(axis='x', alpha=0.3, linewidth=0.5)

    # Note for hidden clusters
    if n_hidden > 0:
        ax_e.text(0.01, -0.02, f'{n_hidden} clusters with strength < 0.3 and n_occ < 2 not shown',
                  transform=ax_e.transAxes, fontsize=7, color='gray',
                  fontstyle='italic', va='top')

    # Legend: event types + strength scale + confidence colorbar
    evt_handles = [
        mpatches.Patch(color=EVENT_COLORS[e], label=EVENT_NAMES[e])
        for e in sorted(EVENT_COLORS.keys())
    ]
    # Add strength scale
    evt_handles.append(mpatches.Patch(facecolor='gray', alpha=0.8,
                       edgecolor='black', linewidth=1.5, label='Strong (≥0.5)'))
    evt_handles.append(mpatches.Patch(facecolor='gray', alpha=0.6,
                       edgecolor='#666666', linewidth=1.0, label='Moderate (0.3–0.5)'))
    evt_handles.append(mpatches.Patch(facecolor='gray', alpha=0.4,
                       edgecolor='#AAAAAA', linewidth=0.5, label='Weak (<0.3)'))
    # Add confidence indicator samples
    evt_handles.append(plt.scatter([], [], c=[conf_cmap(0.8)], marker='s',
                       s=36, edgecolors='black', linewidths=0.3,
                       label='■ High conf'))
    evt_handles.append(plt.scatter([], [], c=[conf_cmap(0.5)], marker='s',
                       s=36, edgecolors='black', linewidths=0.3,
                       label='■ Med conf'))
    evt_handles.append(plt.scatter([], [], c=[conf_cmap(0.2)], marker='s',
                       s=36, edgecolors='black', linewidths=0.3,
                       label='■ Low conf'))
    ax_e.legend(handles=evt_handles, loc='lower right', ncol=4, fontsize=6,
                title='Event types / Strength / Confidence', title_fontsize=6.5,
                framealpha=0.9)

    # X-axis formatting
    ax_e.xaxis.set_major_locator(mdates.MonthLocator())
    ax_e.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax_e.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))

    # Frame color legend (right margin)
    frame_handles = [
        mpatches.Patch(color=FRAME_COLORS[f], label=FRAME_NAMES[f])
        for f in FRAMES
    ]
    fig.legend(handles=frame_handles, loc='center right',
               bbox_to_anchor=(0.98, 0.55), fontsize=8,
               title='Frames', title_fontsize=9, framealpha=0.9)

    # ── Title ──────────────────────────────────────────────────────────────
    fig.suptitle('Climate Change Media Dynamics — 2018',
                 fontsize=14, fontweight='bold', y=0.97)

    # ── Save ───────────────────────────────────────────────────────────────
    outpath = OUTPUT_DIR / 'fig_dominance_proportions_events_2018.png'
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    logger.info(f"Saved: {outpath}")

    outpath_pdf = OUTPUT_DIR / 'fig_dominance_proportions_events_2018.pdf'
    fig.savefig(outpath_pdf, bbox_inches='tight')
    logger.info(f"Saved: {outpath_pdf}")

    plt.close(fig)

    # ── Summary stats ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Cascades: {len(cascades)}")
    print(f"Event occurrences (all): {len(occ_peaks)}")
    print(f"  Attributed to cascades: {n_attr}")
    print(f"  Unattributed: {n_total - n_attr}")
    for et in evt_types:
        n = sum(1 for o in occ_peaks if o['event_type'] == et)
        n_a = sum(1 for o in occ_peaks if o['event_type'] == et and o['n_cascades'] > 0)
        print(f"  {EVENT_NAMES.get(et, et):15s}: {n:3d} total, {n_a:3d} attributed")
    print(f"\nHigh confidence: {sum(1 for o in occ_peaks if not o['low_confidence'])}/{len(occ_peaks)}")
    print(f"\nEvent clusters: {len(event_clusters)}")
    print(f"  Multi-type: {sum(1 for ec in event_clusters if ec.is_multi_type)}")
    print(f"  Single-type: {sum(1 for ec in event_clusters if not ec.is_multi_type)}")
    if event_clusters:
        print(f"  Mean strength: {np.mean([ec.strength for ec in event_clusters]):.3f}")
        print(f"  Total mass: {sum(ec.total_mass for ec in event_clusters):.1f}")

    print(f"\nTop 10 event clusters by strength:")
    for ec in sorted(event_clusters, key=lambda x: x.strength, reverse=True)[:10]:
        types = ', '.join(sorted(ec.event_types))
        print(f"  C{ec.cluster_id:3d}: strength={ec.strength:.3f} "
              f"n_occ={ec.n_occurrences} mass={ec.total_mass:.1f} "
              f"types=[{types}] peak={ec.peak_date.strftime('%Y-%m-%d')} "
              f"{'MULTI' if ec.is_multi_type else 'single'}")

    print(f"\nFigures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
