#!/usr/bin/env python3
"""
PROJECT: CCF-media-cascade-detection
TITLE: fig_phase1_impact_all_cascades_2018.py

For each frame: generates a Phase 1 impact figure showing ALL cascades
(including those with no attributed event clusters), then exports
comprehensive data (cluster metrics, article titles, texts) to JSON.

Output directory: results/phase1_impact_2018/
  - fig_{frame}_phase1_impact_2018.png   (one per frame)
  - {frame}_impact_data.json             (full export per frame)

Author: Antoine Lemor
"""

import json
import pickle
import sys
import warnings
from datetime import date, datetime
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

from cascade_detector.core.constants import FRAMES, FRAME_COLUMNS, FRAME_COLORS
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
    'neutral': '#95A5A6',
    'unrelated': '#D5D8DC',
}

FRAME_FULL_NAMES = {
    'Cult': 'Cultural',
    'Eco': 'Economic',
    'Envt': 'Environmental',
    'Pbh': 'Public Health',
    'Just': 'Justice',
    'Pol': 'Political',
    'Sci': 'Scientific',
    'Secu': 'Security',
}

# Palette for cascade bands — enough for 10+ cascades
CASCADE_PALETTE = [
    ('#2E86C1', '#1A5276'),
    ('#D4AC0D', '#7D6608'),
    ('#C0392B', '#922B21'),
    ('#27AE60', '#1E8449'),
    ('#8E44AD', '#6C3483'),
    ('#E67E22', '#A04000'),
    ('#1ABC9C', '#148F77'),
    ('#2C3E50', '#1B2631'),
    ('#D35400', '#A04000'),
    ('#16A085', '#0E6655'),
]

# Hatching for cascades with no attributed events
NO_EVENT_HATCH = '///'

# Max clusters to show per cascade in the lollipop panel
MAX_CLUSTERS_PER_CASCADE = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _json_serializable(obj):
    """Convert numpy/pandas types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(str(x) for x in obj)
    if pd.isna(obj):
        return None
    return obj


def _build_cascade_style(cascade_ids):
    """Assign colors and labels to cascades."""
    labels, colors, dark = {}, {}, {}
    for i, cid in enumerate(sorted(cascade_ids)):
        parts = cid.split('_')
        try:
            month = pd.Timestamp(parts[1]).strftime('%b')
        except Exception:
            month = f"#{i+1}"
        frame_abbr = parts[0]
        labels[cid] = f"{frame_abbr} #{i+1} ({month})"
        idx = i % len(CASCADE_PALETTE)
        colors[cid] = CASCADE_PALETTE[idx][0]
        dark[cid] = CASCADE_PALETTE[idx][1]
    return labels, colors, dark


def _get_cluster_desc(cluster):
    """Short description from cluster event types + entities."""
    parts = []
    if cluster.event_types:
        types = ', '.join(
            t.replace('evt_', '').replace('_', ' ').title()
            for t in sorted(cluster.event_types)[:2]
        )
        parts.append(types)
    if cluster.entities:
        ents = list(cluster.entities)[:3] if isinstance(cluster.entities, set) else list(cluster.entities.keys())[:3]
        parts.append(', '.join(str(e) for e in ents))
    desc = ' — '.join(parts) if parts else f"Cluster {cluster.cluster_id}"
    return desc[:62] + '...' if len(desc) > 65 else desc


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    cache_path = PROJECT_ROOT / 'results' / 'cache' / 'results_2018.pkl'
    print("Loading cached results...")
    with open(cache_path, 'rb') as f:
        results = pickle.load(f)

    # Use cached impact results (already computed by rerun_impact_2018.py)
    impact = results.event_impact
    if not hasattr(impact, 'cluster_cascade'):
        print("Cache contains legacy impact format — re-running...")
        analyzer = UnifiedImpactAnalyzer()
        impact = analyzer.run(results)

    cluster_map = {c.cluster_id: c for c in results.event_clusters}
    cascade_map = {c.cascade_id: c for c in results.cascades}
    articles = getattr(results, '_articles', pd.DataFrame())

    # Build attribution lookup: cascade_id → set of cluster_ids (via occurrences)
    occ_to_cluster = {}
    for cl in results.event_clusters:
        for occ in cl.occurrences:
            occ_to_cluster[occ.occurrence_id] = cl.cluster_id

    cascade_attributed_clusters = {}
    for attr in results.cascade_attributions:
        cid = attr.cascade_id
        oid = attr.occurrence_id
        if oid in occ_to_cluster:
            cascade_attributed_clusters.setdefault(cid, set()).add(occ_to_cluster[oid])

    return impact, cluster_map, cascade_map, articles, results, cascade_attributed_clusters


# ---------------------------------------------------------------------------
# Fetch article titles from database
# ---------------------------------------------------------------------------
def fetch_titles(doc_ids):
    """Fetch article titles from the 'title' column in the database."""
    try:
        from cascade_detector.data.connector import DatabaseConnector
        from cascade_detector.core.config import DetectorConfig
        config = DetectorConfig()
        db = DatabaseConnector(config)
        # Use DISTINCT ON to get one title per doc_id
        id_list = ', '.join(str(int(d)) for d in doc_ids)
        query = (
            f'SELECT DISTINCT ON (doc_id) doc_id, title '
            f'FROM "{config.db_table}" '
            f'WHERE doc_id IN ({id_list}) AND title IS NOT NULL'
        )
        df = pd.read_sql(query, db.engine)
        titles = df.set_index('doc_id')['title'].to_dict()
        db.close()
        return titles
    except Exception as e:
        print(f"  Warning: could not fetch titles from DB: {e}")
        return {}


# ---------------------------------------------------------------------------
# Panel A — Timeline with ALL cascades
# ---------------------------------------------------------------------------
def plot_timeline(ax, frame_abbr, all_cascade_ids, cascades_with_events,
                  significant, cluster_map, cascade_map, articles,
                  cascade_labels, cascade_colors, cascade_dark):
    frame_col_name = FRAME_COLUMNS[frame_abbr]
    frame_col_mean = f'{frame_col_name}_mean'
    frame_color = FRAME_COLORS[frame_abbr]

    date_col = 'date_converted_first'
    use_col = frame_col_mean if frame_col_mean in articles.columns else frame_col_name
    if use_col not in articles.columns:
        ax.text(0.5, 0.5, f'Column {frame_col_name} not found',
                transform=ax.transAxes, ha='center')
        return

    arts = articles.copy()
    arts['_date'] = pd.to_datetime(arts[date_col], errors='coerce')
    daily = arts.groupby('_date')[use_col].mean()

    # Extend x-range to cover all cascades
    all_dates = []
    for cid in all_cascade_ids:
        c = cascade_map[cid]
        all_dates.extend([pd.Timestamp(c.onset_date), pd.Timestamp(c.end_date)])
    xmin = min(all_dates) - pd.Timedelta(days=30)
    xmax = max(all_dates) + pd.Timedelta(days=30)
    # Clamp to 2018
    xmin = max(xmin, pd.Timestamp('2018-01-01'))
    xmax = min(xmax, pd.Timestamp('2018-12-31'))

    daily = daily.loc[xmin:xmax]
    smoothed = daily.rolling(7, center=True, min_periods=1).mean()

    ax.fill_between(smoothed.index, 0, smoothed.values,
                    color=frame_color, alpha=0.10, zorder=1)
    ax.plot(smoothed.index, smoothed.values,
            color=frame_color, lw=1.4, alpha=0.6, zorder=2)

    # Draw ALL cascade windows
    cascade_order = sorted(all_cascade_ids)
    for cid in cascade_order:
        cascade = cascade_map[cid]
        onset = pd.Timestamp(cascade.onset_date)
        end = pd.Timestamp(cascade.end_date)
        peak = pd.Timestamp(cascade.peak_date)
        col = cascade_colors[cid]
        dark = cascade_dark[cid]
        has_events = cid in cascades_with_events

        # Window band — hatched if no events
        if has_events:
            ax.axvspan(onset, end, alpha=0.07, color=col, zorder=0)
        else:
            ax.axvspan(onset, end, alpha=0.04, color=col, zorder=0,
                       hatch=NO_EVENT_HATCH, edgecolor=col, linewidth=0.5)

        ax.axvline(peak, color=col, ls='--' if has_events else ':',
                   lw=1.2 if has_events else 0.8, alpha=0.5, zorder=3)

        ymax_val = smoothed.max() * 1.08 if len(smoothed) > 0 else 0.01
        label_str = f"peak {peak.strftime('%b %d')}"
        if not has_events:
            label_str += " (no events)"
        ax.text(peak, ymax_val, label_str,
                ha='center', va='bottom', fontsize=7, fontweight='bold',
                color=dark, alpha=0.9 if has_events else 0.6,
                path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white')])

    # Cluster peaks on the curve (only significant ones)
    for _, row in significant.iterrows():
        cl = cluster_map.get(row['cluster_id'])
        if cl is None:
            continue
        peak_dt = pd.Timestamp(cl.peak_date)
        role = row['role']
        color = ROLE_COLORS.get(role, '#95A5A6')

        if peak_dt in smoothed.index:
            y_base = smoothed.loc[peak_dt]
        else:
            idx_pos = smoothed.index.searchsorted(peak_dt)
            idx_pos = min(idx_pos, len(smoothed) - 1)
            y_base = smoothed.iloc[idx_pos] if len(smoothed) > 0 else 0

        marker = {'driver': '^', 'late_support': 'D', 'suppressor': 'v'}.get(role, 'o')
        ax.scatter(peak_dt, y_base, marker=marker, s=65, color=color,
                   edgecolor='white', linewidth=0.8, zorder=5)

    frame_label = frame_col_name.replace('_', ' ')
    ax.set_ylabel(f'Mean {frame_label} score')
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    n_total = len(all_cascade_ids)
    n_with = len(cascades_with_events)
    n_without = n_total - n_with
    ax.set_title(
        f'A.  Daily {frame_label} signal — {n_total} {frame_abbr} cascades '
        f'({n_with} with events, {n_without} without)',
        loc='left', fontweight='bold', fontsize=11,
    )

    handles = [
        plt.Line2D([0], [0], marker='^', color='w',
                   markerfacecolor=ROLE_COLORS['driver'],
                   markersize=9, label='Driver'),
        plt.Line2D([0], [0], marker='D', color='w',
                   markerfacecolor=ROLE_COLORS['late_support'],
                   markersize=8, label='Late Support'),
        plt.Line2D([0], [0], marker='v', color='w',
                   markerfacecolor=ROLE_COLORS['suppressor'],
                   markersize=9, label='Suppressor'),
        mpatches.Patch(facecolor=frame_color, alpha=0.15,
                       label=f'{frame_label.title()} (7d avg)'),
        mpatches.Patch(facecolor='#BDC3C7', alpha=0.15, hatch=NO_EVENT_HATCH,
                       edgecolor='#95A5A6', label='No attributed events'),
    ]
    ax.legend(handles=handles, loc='upper right', frameon=True, fancybox=True,
              edgecolor='#BDC3C7', fontsize=8)


# ---------------------------------------------------------------------------
# Panel B — Lollipop (significant clusters grouped by cascade)
# ---------------------------------------------------------------------------
def plot_lollipop(ax, frame_abbr, all_cascade_ids, cascades_with_events,
                  significant, cluster_map, cascade_map,
                  cascade_labels, cascade_colors, cascade_dark):
    frame_col_name = FRAME_COLUMNS[frame_abbr]
    frame_short = frame_col_name.split('_')[0]
    cascade_order = sorted(all_cascade_ids)

    # Build rows: top clusters per cascade + placeholders for empty / overflow
    rows = []
    row_cascade = []
    row_is_placeholder = []
    row_is_overflow = []  # "... and N more" lines

    for cid in cascade_order:
        sub = significant[significant['cascade_id'] == cid].sort_values(
            'impact_score', ascending=False  # highest first for selection
        )
        if len(sub) == 0:
            # Placeholder row for cascade with no significant clusters
            rows.append(None)
            row_cascade.append(cid)
            row_is_placeholder.append(True)
            row_is_overflow.append(False)
        else:
            n_total_cascade = len(sub)
            top = sub.head(MAX_CLUSTERS_PER_CASCADE).sort_values(
                'impact_score', ascending=True  # ascending for visual stacking
            )
            for _, r in top.iterrows():
                rows.append(r)
                row_cascade.append(cid)
                row_is_placeholder.append(False)
                row_is_overflow.append(False)
            n_omitted = n_total_cascade - len(top)
            if n_omitted > 0:
                rows.append(n_omitted)
                row_cascade.append(cid)
                row_is_placeholder.append(False)
                row_is_overflow.append(True)

    if not rows:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
        return

    n = len(rows)

    # Group boundaries
    group_bounds = {}
    current_cascade = None
    for i in range(n):
        cid = row_cascade[i]
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
            has_events = cid in cascades_with_events
            alpha = 0.05 if has_events else 0.03
            ax.axhspan(y0 - 0.5, y1 + 0.5, alpha=alpha,
                       color=cascade_colors[cid], zorder=0,
                       hatch=None if has_events else NO_EVENT_HATCH)

    # Lollipops
    labels = []
    for i in range(n):
        if row_is_placeholder[i]:
            # Placeholder for cascade with no events
            cid = row_cascade[i]
            cascade = cascade_map[cid]
            onset_str = pd.Timestamp(cascade.onset_date).strftime('%b %d')
            end_str = pd.Timestamp(cascade.end_date).strftime('%b %d')
            label = f"  No significant event clusters  [{onset_str}–{end_str}]"
            labels.append(label)
            ax.text(0.002, i, f"score={cascade.total_score:.3f}  {cascade.classification}",
                    ha='left', va='center', fontsize=7.5,
                    color='#95A5A6', fontstyle='italic',
                    path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')])
            continue

        if row_is_overflow[i]:
            n_omitted = rows[i]
            label = f"  ... and {n_omitted} more clusters (see JSON export)"
            labels.append(label)
            ax.text(0.002, i, '', ha='left', va='center', fontsize=7.5,
                    color='#95A5A6', fontstyle='italic')
            continue

        row = rows[i]
        cl = cluster_map.get(row['cluster_id'])
        role = row['role']
        color = ROLE_COLORS.get(role, '#95A5A6')
        impact = row['impact_score']

        # Stem
        ax.plot([0, impact], [i, i], color=color, lw=2.0, alpha=0.55, zorder=2,
                solid_capstyle='round')

        # Dot
        marker = {'driver': '^', 'late_support': 'D', 'suppressor': 'v'}.get(role, 'o')
        ax.scatter(impact, i, marker=marker, s=90, color=color,
                   edgecolor='white', linewidth=1.0, zorder=3)

        # Label
        if cl is not None:
            desc = _get_cluster_desc(cl)
            peak_str = cl.peak_date.strftime('%b %d')
        else:
            desc = '?'
            peak_str = '?'
        label = f"C{row['cluster_id']}  {desc}  [{peak_str}]"
        labels.append(label)

        # Annotation
        did_sign = '+' if row['diff_in_diff'] > 0 else ''
        ann = (f"DID {did_sign}{row['diff_in_diff']:.3f}   "
               f"{frame_short}={row['frame_affinity']:.3f}")
        ax.text(impact + 0.004, i, ann, ha='left', va='center', fontsize=7.5,
                color='#566573', fontstyle='italic',
                path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')])

    # Y labels
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    text_colors = {
        'driver': '#1E8449', 'late_support': '#B7950B',
        'suppressor': '#C0392B',
    }
    for i, tick_label in enumerate(ax.get_yticklabels()):
        if row_is_placeholder[i] or row_is_overflow[i]:
            tick_label.set_color('#95A5A6')
            tick_label.set_fontstyle('italic')
        else:
            role = rows[i]['role']
            tick_label.set_color(text_colors.get(role, '#2C3E50'))
        tick_label.set_fontweight('bold')

    # Cascade group labels
    ax.set_xlim(-0.002, None)
    xmax = ax.get_xlim()[1]
    for cid in cascade_order:
        if cid in group_bounds:
            y0, y1 = group_bounds[cid]
            y_mid = (y0 + y1) / 2
            has_events = cid in cascades_with_events
            lbl = cascade_labels[cid]
            if not has_events:
                lbl += ' *'
            ax.text(xmax * 0.97, y_mid, lbl,
                    ha='right', va='center', fontsize=9, fontweight='bold',
                    fontstyle='italic', color=cascade_dark[cid],
                    alpha=0.9 if has_events else 0.5,
                    path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white')])

    ax.set_xlabel('Impact score')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(
        f'B.  Phase 1 causal impact of event clusters on {frame_abbr} cascades'
        f'  (* = no attributed events)',
        loc='left', fontweight='bold', fontsize=11,
    )

    handles = [
        plt.Line2D([0], [0], marker='^', color='w',
                   markerfacecolor=ROLE_COLORS['driver'],
                   markersize=9, label='Driver  (pre-peak, DID > 0)'),
        plt.Line2D([0], [0], marker='D', color='w',
                   markerfacecolor=ROLE_COLORS['late_support'],
                   markersize=8,
                   label=f'Late Support  (post-peak, high {frame_short} affinity)'),
        plt.Line2D([0], [0], marker='v', color='w',
                   markerfacecolor=ROLE_COLORS['suppressor'],
                   markersize=9,
                   label=f'Suppressor  (post-peak, low {frame_short} affinity)'),
        mpatches.Patch(facecolor='#BDC3C7', alpha=0.1, hatch=NO_EVENT_HATCH,
                       edgecolor='#95A5A6', label='No events attributed'),
    ]
    ax.legend(handles=handles, loc='lower right', frameon=True, fancybox=True,
              edgecolor='#BDC3C7', fontsize=8.5)


# ---------------------------------------------------------------------------
# Fetch article sentences (full text) from database
# ---------------------------------------------------------------------------
def fetch_sentences(doc_ids):
    """Fetch all sentences for given doc_ids. Returns {doc_id: [sent1, sent2, ...]}."""
    if not doc_ids:
        return {}
    try:
        from cascade_detector.data.connector import DatabaseConnector
        from cascade_detector.core.config import DetectorConfig
        config = DetectorConfig()
        db = DatabaseConnector(config)
        df = db.get_sentences(doc_ids=list(doc_ids))
        db.close()
        # Group by doc_id, order by sentence_id
        result = {}
        for doc_id, group in df.sort_values('sentence_id').groupby('doc_id'):
            result[doc_id] = group['sentences'].tolist()
        return result
    except Exception as e:
        print(f"    Warning: could not fetch sentences from DB: {e}")
        return {}


# ---------------------------------------------------------------------------
# Export JSON with ALL clusters + full article texts
# ---------------------------------------------------------------------------
def export_frame_data(frame_abbr, cc_frame, all_cascade_ids, cascades_with_events,
                      cluster_map, cascade_map, articles, results, out_dir):
    """Export comprehensive JSON: all clusters (attributed or not), full article texts."""
    frame_col_name = FRAME_COLUMNS[frame_abbr]

    # --- Identify ALL unique cluster IDs that have impact data for this frame ---
    all_cluster_ids = sorted(cc_frame['cluster_id'].unique())

    # --- Collect ALL doc_ids from those clusters ---
    all_doc_ids = set()
    for clid in all_cluster_ids:
        cl = cluster_map.get(clid)
        if cl:
            for occ in cl.occurrences:
                all_doc_ids.update(occ.seed_doc_ids)

    # --- Fetch titles + sentences from DB ---
    print(f"    Fetching titles for {len(all_doc_ids)} articles...")
    titles = fetch_titles(list(all_doc_ids)) if all_doc_ids else {}
    print(f"    Fetching full texts for {len(all_doc_ids)} articles...")
    sentences = fetch_sentences(list(all_doc_ids)) if all_doc_ids else {}
    n_with_text = sum(1 for d in all_doc_ids if d in sentences)
    print(f"    Texts retrieved: {n_with_text}/{len(all_doc_ids)}")

    # --- Build article lookup (indexed for fast access) ---
    art_index = {}
    if 'doc_id' in articles.columns:
        art_index = articles.set_index('doc_id')

    # --- Build output structure ---
    # Role counts per cascade
    sig = cc_frame[cc_frame['role'].isin(['driver', 'late_support', 'suppressor'])]

    output = {
        'frame': frame_abbr,
        'frame_full_name': FRAME_FULL_NAMES[frame_abbr],
        'frame_column': frame_col_name,
        'n_cascades': len(all_cascade_ids),
        'n_cascades_with_events': len(cascades_with_events),
        'n_clusters_total': len(all_cluster_ids),
        'n_clusters_attributed': len(
            set(cc_frame[cc_frame['role'] != 'unrelated']['cluster_id'].unique())
        ),
        'cascades': [],
        'clusters': [],
    }

    # --- Cascade-level data ---
    for cid in sorted(all_cascade_ids):
        cascade = cascade_map[cid]
        has_events = cid in cascades_with_events
        cc_cascade = cc_frame[cc_frame['cascade_id'] == cid]
        c_data = {
            'cascade_id': cid,
            'onset_date': _json_serializable(cascade.onset_date),
            'peak_date': _json_serializable(cascade.peak_date),
            'end_date': _json_serializable(cascade.end_date),
            'duration_days': cascade.duration_days,
            'n_articles': cascade.n_articles,
            'n_journalists': cascade.n_journalists,
            'n_media': cascade.n_media,
            'total_score': cascade.total_score,
            'classification': cascade.classification,
            'score_temporal': cascade.score_temporal,
            'score_participation': cascade.score_participation,
            'score_convergence': cascade.score_convergence,
            'score_source': cascade.score_source,
            'burst_intensity': cascade.burst_intensity,
            'has_attributed_events': has_events,
            'role_counts': cc_cascade['role'].value_counts().to_dict(),
        }
        output['cascades'].append(c_data)

    # --- Helper to build article info ---
    def _article_info(doc_id, belonging_val):
        info = {
            'doc_id': _json_serializable(doc_id),
            'title': titles.get(doc_id, None),
            'sentences': sentences.get(doc_id, None),
            'belonging': _json_serializable(belonging_val),
        }
        if doc_id in art_index.index:
            row = art_index.loc[doc_id]
            info['date'] = _json_serializable(row.get('date_converted_first', None))
            info['media'] = _json_serializable(row.get('media_first', None))
            info['author'] = _json_serializable(row.get('author_first', None))
            for f_abbr, f_col in FRAME_COLUMNS.items():
                mean_col = f'{f_col}_mean'
                if mean_col in art_index.columns:
                    info[f'{f_abbr}_score'] = _json_serializable(row.get(mean_col, None))
        return info

    # --- Helper to build occurrence details ---
    def _occurrence_info(occ):
        occ_articles = [
            _article_info(doc_id, occ.belonging.get(doc_id, 0))
            for doc_id in occ.seed_doc_ids
        ]
        return {
            'occurrence_id': occ.occurrence_id,
            'event_type': occ.event_type,
            'peak_date': _json_serializable(occ.peak_date),
            'core_start': _json_serializable(occ.core_start),
            'core_end': _json_serializable(occ.core_end),
            'n_articles': occ.n_articles,
            'n_seed_articles': len(occ.seed_doc_ids),
            'effective_mass': _json_serializable(occ.effective_mass),
            'semantic_coherence': _json_serializable(occ.semantic_coherence),
            'confidence': _json_serializable(occ.confidence),
            'confidence_components': {
                k: _json_serializable(v)
                for k, v in occ.confidence_components.items()
            },
            'is_singleton': occ.is_singleton,
            'media_count': occ.media_count,
            'temporal_intensity': _json_serializable(occ.temporal_intensity),
            'emotional_intensity': _json_serializable(occ.emotional_intensity),
            'tone_coherence': _json_serializable(occ.tone_coherence),
            'entities': _json_serializable(occ.entities),
            'seed_articles': occ_articles,
        }

    # --- Cluster-level data: ALL clusters, grouped by unique cluster_id ---
    # Build per-cluster cascade attributions from cc_frame
    cluster_attributions = {}
    for _, row in cc_frame.iterrows():
        clid = row['cluster_id']
        cluster_attributions.setdefault(clid, []).append({
            'cascade_id': row['cascade_id'],
            'role': row['role'],
            'impact_score': _json_serializable(row['impact_score']),
            'impact_label': _json_serializable(row.get('impact_label', None)),
            'diff_in_diff': _json_serializable(row['diff_in_diff']),
            'dose_response_corr': _json_serializable(row.get('dose_response_corr', None)),
            'dose_response_lag': _json_serializable(row.get('dose_response_lag', None)),
            'frame_affinity': _json_serializable(row['frame_affinity']),
            'embedding_alignment': _json_serializable(row.get('embedding_alignment', None)),
            'is_post_peak': _json_serializable(row.get('is_post_peak', None)),
            'proximity': _json_serializable(row.get('proximity', None)),
        })

    for clid in all_cluster_ids:
        cl = cluster_map.get(clid)
        if cl is None:
            continue

        # Occurrence details with full article texts
        occ_details = [_occurrence_info(occ) for occ in cl.occurrences]

        # Cascade attributions for this cluster
        attribs = cluster_attributions.get(clid, [])
        has_significant = any(
            a['role'] in ('driver', 'late_support', 'suppressor') for a in attribs
        )

        cluster_data = {
            'cluster_id': _json_serializable(clid),
            'has_significant_impact': has_significant,
            # Cluster-level metrics
            'peak_date': _json_serializable(cl.peak_date),
            'core_start': _json_serializable(cl.core_start),
            'core_end': _json_serializable(cl.core_end),
            'total_mass': _json_serializable(cl.total_mass),
            'strength': _json_serializable(cl.strength),
            'strength_components': {
                k: _json_serializable(v) for k, v in cl.strength_components.items()
            },
            'n_occurrences': cl.n_occurrences,
            'is_multi_type': cl.is_multi_type,
            'event_types': _json_serializable(cl.event_types),
            'dominant_type': cl.dominant_type,
            'type_structure': cl.type_structure,
            'entities': _json_serializable(cl.entities),
            # Per-cascade impact (ALL roles including unrelated/neutral)
            'cascade_impacts': attribs,
            # Occurrences with full article data
            'occurrences': occ_details,
        }
        output['clusters'].append(cluster_data)

    # Write JSON
    json_path = out_dir / f'{frame_abbr.lower()}_impact_data.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=_json_serializable)

    return json_path


# ---------------------------------------------------------------------------
# Generate figure + export for one frame
# ---------------------------------------------------------------------------
def process_frame(frame_abbr, cc, cluster_map, cascade_map, articles,
                  results, cascade_attributed_clusters, out_dir):
    # All cascades for this frame
    all_cascade_ids = sorted(
        c.cascade_id for c in results.cascades if c.frame == frame_abbr
    )
    if not all_cascade_ids:
        print(f"  {frame_abbr}: no cascades — skipping")
        return None, None

    # Filter impact data to this frame
    cc_frame = cc[cc['cascade_frame'] == frame_abbr].copy()
    significant = cc_frame[
        cc_frame['role'].isin(['driver', 'late_support', 'suppressor'])
    ]

    # Which cascades have significant event clusters?
    cascades_with_events = set(significant['cascade_id'].unique())
    cascades_without = set(all_cascade_ids) - cascades_with_events

    n_total = len(all_cascade_ids)
    n_sig = len(significant)
    print(f"  {frame_abbr}: {n_total} cascades ({len(cascades_with_events)} with events, "
          f"{len(cascades_without)} without), {n_sig} significant clusters")

    cascade_labels, cascade_colors, cascade_dark = _build_cascade_style(all_cascade_ids)

    # --- Filter top clusters per cascade for the figure ---
    sig_for_fig_parts = []
    for cid in all_cascade_ids:
        sub = significant[significant['cascade_id'] == cid].sort_values(
            'impact_score', ascending=False
        )
        sig_for_fig_parts.append(sub.head(MAX_CLUSTERS_PER_CASCADE))
    sig_for_fig = pd.concat(sig_for_fig_parts) if sig_for_fig_parts else significant.iloc[:0]

    n_shown = len(sig_for_fig)
    n_omitted_total = n_sig - n_shown
    # Number of visible rows: top clusters + overflow placeholders + eventless placeholders
    n_cascades_with_overflow = sum(
        1 for cid in all_cascade_ids
        if len(significant[significant['cascade_id'] == cid]) > MAX_CLUSTERS_PER_CASCADE
    )
    n_fig_rows = n_shown + n_cascades_with_overflow + len(cascades_without)

    # --- Figure ---
    panel_b_height = max(4, n_fig_rows * 0.45 + 1.5)
    panel_a_height = 4.5
    fig_height = panel_a_height + panel_b_height + 1.5

    fig, axes = plt.subplots(
        2, 1, figsize=(16, fig_height),
        gridspec_kw={'height_ratios': [panel_a_height, panel_b_height]},
    )
    fig.subplots_adjust(hspace=0.30, left=0.32, right=0.87, top=0.93, bottom=0.05)

    plot_timeline(axes[0], frame_abbr, all_cascade_ids, cascades_with_events,
                  sig_for_fig, cluster_map, cascade_map, articles,
                  cascade_labels, cascade_colors, cascade_dark)
    plot_lollipop(axes[1], frame_abbr, all_cascade_ids, cascades_with_events,
                  significant, cluster_map, cascade_map,
                  cascade_labels, cascade_colors, cascade_dark)

    frame_full = FRAME_FULL_NAMES[frame_abbr]
    fig.suptitle(
        f'Phase 1 Impact Analysis:  Event Clusters on {frame_full} Cascades  (2018)',
        fontsize=14, fontweight='bold', y=0.97,
    )

    fig_path = out_dir / f'fig_{frame_abbr.lower()}_phase1_impact_2018.png'
    fig.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # --- JSON export ---
    json_path = export_frame_data(
        frame_abbr, cc_frame, all_cascade_ids, cascades_with_events,
        cluster_map, cascade_map, articles, results, out_dir,
    )

    return fig_path, json_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    impact, cluster_map, cascade_map, articles, results, cascade_attributed_clusters = load_data()
    cc = impact.cluster_cascade

    out_dir = PROJECT_ROOT / 'results' / 'phase1_impact_2018'
    out_dir.mkdir(parents=True, exist_ok=True)

    available_frames = sorted(cc['cascade_frame'].unique())
    print(f"Frames with cascade data: {available_frames}\n")

    fig_paths = []
    json_paths = []

    for frame_abbr in FRAMES:
        if frame_abbr not in available_frames:
            print(f"  {frame_abbr}: not in results — skipping")
            continue
        fig_path, json_path = process_frame(
            frame_abbr, cc, cluster_map, cascade_map, articles,
            results, cascade_attributed_clusters, out_dir,
        )
        if fig_path:
            fig_paths.append(fig_path)
        if json_path:
            json_paths.append(json_path)

    print(f"\n{'='*60}")
    print(f"Done — {len(fig_paths)} figures + {len(json_paths)} JSON exports")
    print(f"Output: {out_dir}/")
    for p in fig_paths:
        print(f"  {p.name}")
    for p in json_paths:
        print(f"  {p.name}")


if __name__ == '__main__':
    main()
