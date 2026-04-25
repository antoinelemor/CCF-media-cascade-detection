#!/usr/bin/env python3
"""
Multi-panel publication figure: Event–cascade dynamics across the full database.

Narrative: Events redistribute media framing attention. The dynamics shift
structurally around 2007 as climate coverage becomes multi-framed.

Layout (5 rows, 8 panels):
  A (full width) — Timeline: total volume + D/(D+S) ratio + structural break 2007
  B (left) — Diverging bars: n × median |β| per frame (robust impact)
  C (right) — Horizontal dot plot: median |β| by event type, split by role
  D (left) — Self vs cross-frame suppression by frame (stacked bars)
  E (right) — R² model fit over time with LOESS
  G (left) — Dual-role cluster redistribution flows (net beneficiary/target)
  H (right) — Dual-role event profile vs single-role (overrepresentation)
  F (full width) — Driver event-type composition (smoothed stacked area)

Automatically reads all completed years from results/stabsel_production/.

Usage:
    python scripts/figures/fig_stabsel_dynamics.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'axes.titlesize': 11,
    'axes.labelsize': 9.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 7.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

FRAMES = ['Cult', 'Eco', 'Envt', 'Pbh', 'Just', 'Pol', 'Sci', 'Secu']
FRAME_COLORS = {
    'Cult': '#E64B35', 'Eco': '#4DBBD5', 'Envt': '#00A087',
    'Pbh': '#3C5488', 'Just': '#F39B7F', 'Pol': '#8491B4',
    'Sci': '#91D1C2', 'Secu': '#B09C85',
}
FRAME_LABELS = {
    'Cult': 'Cultural', 'Eco': 'Economic', 'Envt': 'Environmental',
    'Pbh': 'Public health', 'Just': 'Justice', 'Pol': 'Political',
    'Sci': 'Scientific', 'Secu': 'Security',
}
EVENT_TYPES = [
    'evt_weather', 'evt_meeting', 'evt_publication', 'evt_election',
    'evt_policy', 'evt_judiciary', 'evt_cultural', 'evt_protest',
]
EVENT_LABELS = {
    'evt_weather': 'Weather', 'evt_meeting': 'Summit',
    'evt_publication': 'Publication', 'evt_election': 'Election',
    'evt_policy': 'Policy', 'evt_judiciary': 'Judiciary',
    'evt_cultural': 'Cultural', 'evt_protest': 'Protest',
}
EVENT_COLORS = {
    'evt_policy': '#9b59b6', 'evt_meeting': '#3498db',
    'evt_weather': '#e74c3c', 'evt_publication': '#2ecc71',
    'evt_election': '#f39c12', 'evt_judiciary': '#1abc9c',
    'evt_cultural': '#e67e22', 'evt_protest': '#e91e63',
}
ROLE_COLORS = {'driver': '#27AE60', 'suppressor': '#E74C3C'}


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_all_data(results_dir: Path):
    """Load all completed year data."""
    year_summaries = {}
    all_clusters = []

    for year_dir in sorted(results_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue

        summary_path = year_dir / 'year_summary.json'
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            year_summaries[year] = json.load(f)

        for frame in FRAMES:
            cluster_path = year_dir / f'significant_clusters_{frame}.json'
            if not cluster_path.exists():
                continue
            with open(cluster_path) as f:
                clusters = json.load(f)
            for c in clusters:
                c['year'] = year
                all_clusters.append(c)

    logger.info(f"Loaded {len(year_summaries)} years, {len(all_clusters)} significant clusters")
    return year_summaries, pd.DataFrame(all_clusters) if all_clusters else pd.DataFrame()


def prepare_data(year_summaries, df):
    """Prepare derived DataFrames."""
    rows = []
    for year, summary in sorted(year_summaries.items()):
        for frame, stats in summary['frames'].items():
            rows.append({
                'year': year, 'frame': frame,
                'n_cascades': stats['n_cascades'],
                'n_drivers': stats['n_drivers'],
                'n_suppressors': stats['n_suppressors'],
                'median_r2': stats['median_r2'],
            })
    df_summary = pd.DataFrame(rows)

    if not df.empty:
        df['abs_beta'] = df['net_beta'].abs()
        df['impact'] = df['abs_beta'] * df['D_sum']
        df['self_suppress'] = df.apply(
            lambda r: r['frame_profile'].get(r['frame'], 0) > 0.3
            if r['role'] == 'suppressor' else False, axis=1)

        # Tag dual-role clusters (both driver AND suppressor across cascades)
        df['cid_year'] = df['year'].astype(str) + '_' + df['cluster_id'].astype(str)
        role_by_cid = df.groupby('cid_year')['role'].apply(set)
        dual_cids = set(role_by_cid[role_by_cid.apply(
            lambda s: 'driver' in s and 'suppressor' in s)].index)
        df['is_dual'] = df['cid_year'].isin(dual_cids)

    return df_summary, df


def loess_smooth(x, y, frac=0.3, num_points=200, n_boot=500):
    """Gaussian-kernel smoothing with bootstrap 95% CI."""
    sort_idx = np.argsort(x)
    xs, ys = np.array(x, dtype=float)[sort_idx], np.array(y, dtype=float)[sort_idx]
    if len(xs) < 5:
        return xs, ys, ys, ys

    x_fine = np.linspace(xs.min(), xs.max(), num_points)
    y_interp = np.interp(x_fine, xs, ys)
    sigma = max(1.5, len(x_fine) * frac / 4)
    y_smooth = gaussian_filter1d(y_interp, sigma=sigma)

    rng = np.random.default_rng(42)
    boot = np.zeros((n_boot, len(x_fine)))
    for b in range(n_boot):
        idx = rng.choice(len(xs), size=len(xs), replace=True)
        xb, yb = xs[idx], ys[idx]
        sb = np.argsort(xb)
        boot[b] = gaussian_filter1d(np.interp(x_fine, xb[sb], yb[sb]), sigma=sigma)

    return x_fine, y_smooth, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


# ─── Panel A: Timeline ───────────────────────────────────────────────────────

def panel_a_timeline(ax, df_summary):
    """Dual-axis: bars = total significant, line = D/(D+S)."""
    yearly = df_summary.groupby('year').agg(
        drivers=('n_drivers', 'sum'),
        suppressors=('n_suppressors', 'sum'),
        cascades=('n_cascades', 'sum'),
    ).reset_index()
    yearly = yearly[yearly['cascades'] > 0].copy()
    years = yearly['year'].values
    total = (yearly['drivers'] + yearly['suppressors']).values
    ratio = yearly['drivers'].values / np.where(total > 0, total, 1)

    # Bars colored by majority
    bar_colors = [ROLE_COLORS['driver'] if r >= 0.5 else ROLE_COLORS['suppressor']
                  for r in ratio]
    ax.bar(years, total, 0.78, color=bar_colors, alpha=0.55, zorder=2,
           edgecolor='none')

    # Right axis: ratio
    ax2 = ax.twinx()
    valid_mask = total > 0
    if valid_mask.sum() >= 5:
        xf, ys, lo, hi = loess_smooth(years[valid_mask].astype(float),
                                        ratio[valid_mask], frac=0.35)
        ax2.plot(xf, ys, color='#2C3E50', lw=2.5, zorder=4)
        ax2.fill_between(xf, lo, hi, color='#2C3E50', alpha=0.12, zorder=3)
    ax2.scatter(years[valid_mask], ratio[valid_mask], s=20, color='#2C3E50',
                alpha=0.5, zorder=5, edgecolors='none')
    ax2.axhline(0.5, color='grey', lw=0.8, ls=':', zorder=1)
    ax2.set_ylabel('Driver proportion D/(D+S)', fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.spines['top'].set_visible(False)

    # 2007 annotation
    ax2.annotate('', xy=(2007, 0.15), xytext=(2007, 0.92),
                 arrowprops=dict(arrowstyle='->', color='#888', lw=1.2))
    ax2.text(2007.5, 0.07, '2007: IPCC AR4 + Nobel + Live Earth',
             fontsize=7.5, color='#555', fontstyle='italic', va='bottom')

    ax.set_ylabel('Total significant event–cascade pairs', fontsize=9)
    ax.set_title('A.  Event–cascade interaction volume and driver proportion',
                 fontweight='bold', loc='left', fontsize=11.5)

    handles = [
        mpatches.Patch(color=ROLE_COLORS['driver'], alpha=0.55, label='Driver-majority year'),
        mpatches.Patch(color=ROLE_COLORS['suppressor'], alpha=0.55, label='Suppressor-majority year'),
        Line2D([0], [0], color='#2C3E50', lw=2.5, label='D/(D+S) ratio (LOESS ± 95% CI)'),
    ]
    ax.legend(handles=handles, loc='upper left', framealpha=0.92, fontsize=7.5)


# ─── Panel B: Frame impact (diverging bars, median-weighted) ──────────────────

def panel_b_frame_bars(ax, df):
    """Diverging horizontal bars per frame. Length = n × median |β| (robust
    proxy for total impact without outlier sensitivity). Drivers right,
    suppressors left. Annotated with n, median |β|, and top event type."""
    if df.empty:
        return

    frame_stats = {}
    for frame in FRAMES:
        sub_d = df[(df['frame'] == frame) & (df['role'] == 'driver')]
        sub_s = df[(df['frame'] == frame) & (df['role'] == 'suppressor')]
        nd, ns = len(sub_d), len(sub_s)
        med_d = sub_d['abs_beta'].median() if nd >= 3 else 0
        med_s = sub_s['abs_beta'].median() if ns >= 3 else 0
        # Robust total impact: n × median (less sensitive to outliers than sum)
        robust_d = nd * med_d
        robust_s = ns * med_s
        # Top event type by count (most common driver/suppressor)
        d_top = (sub_d['dominant_type'].value_counts().index[0] if nd > 0 else '')
        s_top = (sub_s['dominant_type'].value_counts().index[0] if ns > 0 else '')
        frame_stats[frame] = {
            'robust_d': robust_d, 'robust_s': robust_s,
            'nd': nd, 'ns': ns, 'med_d': med_d, 'med_s': med_s,
            'net': robust_d - robust_s,
            'd_top': d_top, 's_top': s_top,
        }

    frame_order = sorted(FRAMES, key=lambda f: frame_stats[f]['net'])
    y_pos = np.arange(len(frame_order))

    for i, frame in enumerate(frame_order):
        st = frame_stats[frame]
        # Driver bar (right, saturated)
        ax.barh(i, st['robust_d'], 0.65, color=FRAME_COLORS[frame], alpha=0.80,
                zorder=2, edgecolor='white', linewidth=0.3)
        # Suppressor bar (left, faded)
        ax.barh(i, -st['robust_s'], 0.65, color=FRAME_COLORS[frame], alpha=0.40,
                zorder=2, edgecolor='white', linewidth=0.3)

        # Inside: n × med |β|
        if st['robust_d'] > 40:
            ax.text(st['robust_d'] / 2, i,
                    f'n={st["nd"]}\nmed={st["med_d"]:.1f}',
                    ha='center', va='center', fontsize=6, color='white',
                    fontweight='bold', linespacing=0.9)
        if st['robust_s'] > 40:
            ax.text(-st['robust_s'] / 2, i,
                    f'n={st["ns"]}\nmed={st["med_s"]:.1f}',
                    ha='center', va='center', fontsize=6, color='white',
                    fontweight='bold', linespacing=0.9)

        # Top event type annotations outside
        d_label = EVENT_LABELS.get(st['d_top'], '')
        s_label = EVENT_LABELS.get(st['s_top'], '')
        if d_label:
            ax.text(st['robust_d'] + 8, i, d_label, va='center', fontsize=6.5,
                    color=EVENT_COLORS.get(st['d_top'], '#666'), fontstyle='italic')
        if s_label and st['robust_s'] > 15:
            ax.text(-st['robust_s'] - 8, i, s_label, va='center', fontsize=6.5,
                    ha='right', color=EVENT_COLORS.get(st['s_top'], '#666'),
                    fontstyle='italic')

    ax.axvline(0, color='#333', lw=0.8, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([FRAME_LABELS[f] for f in frame_order], fontsize=9)
    ax.set_xlabel('← Suppressor effect   n × median |β|   Driver effect →')
    ax.set_title('B.  Event impact on frame cascades (robust)',
                 fontweight='bold', loc='left')

    max_val = max(max(st['robust_d'] for st in frame_stats.values()),
                  max(st['robust_s'] for st in frame_stats.values()))
    ax.set_xlim(-max_val * 1.4, max_val * 1.4)

    handles = [mpatches.Patch(facecolor='#666', alpha=0.80, label='Driver (n × med |β|)'),
               mpatches.Patch(facecolor='#666', alpha=0.40, label='Suppressor (n × med |β|)')]
    ax.legend(handles=handles, loc='upper left', framealpha=0.92, fontsize=7)


# ─── Panel C: Event type median effect (dot + CI) ─────────────────────────────

def panel_c_event_effect(ax, df):
    """Horizontal dot plot: median |β| with bootstrap 95% CI per event type,
    split by driver/suppressor. Robust central tendency, not cumulative."""
    if df.empty:
        return

    et_order = []
    for et in EVENT_TYPES:
        sub = df[df['dominant_type'] == et]
        if len(sub) >= 10:
            et_order.append((et, sub[sub['role'] == 'driver']['abs_beta'].median()
                             if (sub['role'] == 'driver').sum() >= 5 else 0))
    et_order.sort(key=lambda x: x[1])
    et_order = [et for et, _ in et_order]

    y_pos = np.arange(len(et_order))
    offset = 0.20
    rng = np.random.default_rng(42)

    for role, color, dy in [('driver', ROLE_COLORS['driver'], offset),
                             ('suppressor', ROLE_COLORS['suppressor'], -offset)]:
        for i, et in enumerate(et_order):
            sub = df[(df['dominant_type'] == et) & (df['role'] == role)]['abs_beta']
            if len(sub) < 5:
                continue

            med = sub.median()
            # Bootstrap 95% CI on median
            boot_meds = [np.median(rng.choice(sub.values, size=len(sub), replace=True))
                         for _ in range(1000)]
            ci_lo = np.percentile(boot_meds, 2.5)
            ci_hi = np.percentile(boot_meds, 97.5)

            ax.errorbar(med, i + dy, xerr=[[med - ci_lo], [ci_hi - med]],
                        fmt='o', color=color, markersize=6, capsize=3,
                        capthick=1.2, elinewidth=1.2, markeredgecolor='white',
                        markeredgewidth=0.5, zorder=3)

            # n annotation
            ax.text(ci_hi + 0.08, i + dy, f'n={len(sub)}',
                    fontsize=6, color='#777', va='center')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([EVENT_LABELS.get(et, et) for et in et_order], fontsize=9)
    ax.set_xlabel('Median |β| (95% bootstrap CI)')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.set_title('C.  Typical effect size by event type (median ± 95% CI)',
                 fontweight='bold', loc='left')

    handles = [Line2D([0], [0], marker='o', color=ROLE_COLORS['driver'], lw=0,
                       markersize=7, label='Driver'),
               Line2D([0], [0], marker='o', color=ROLE_COLORS['suppressor'], lw=0,
                       markersize=7, label='Suppressor')]
    ax.legend(handles=handles, loc='lower right', framealpha=0.92, fontsize=7.5)


# ─── Panel D: Self suppression ───────────────────────────────────────────────

def panel_d_self_suppression(ax, df):
    """Stacked horizontal bars: self vs cross suppression per frame."""
    if df.empty:
        return

    suppressors = df[df['role'] == 'suppressor'].copy()
    if suppressors.empty:
        return

    frame_order = (suppressors.groupby('frame').size()
                   .sort_values(ascending=True).index.tolist())

    y_pos = np.arange(len(frame_order))
    self_counts, cross_counts = [], []
    for frame in frame_order:
        sub = suppressors[suppressors['frame'] == frame]
        self_s = int(sub['self_suppress'].sum())
        self_counts.append(self_s)
        cross_counts.append(len(sub) - self_s)

    ax.barh(y_pos, self_counts, 0.65, color='#E74C3C', alpha=0.65,
            label='Self-suppression (same frame)', zorder=2)
    ax.barh(y_pos, cross_counts, 0.65, left=self_counts, color='#3498DB',
            alpha=0.65, label='Cross-frame', zorder=2)

    for i, frame in enumerate(frame_order):
        total = self_counts[i] + cross_counts[i]
        if total > 0:
            pct = 100 * self_counts[i] / total
            ax.text(total + 8, i, f'{pct:.0f}%',
                    fontsize=7.5, va='center', color='#E74C3C',
                    fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([FRAME_LABELS[f] for f in frame_order], fontsize=8.5)
    ax.set_xlabel('Number of suppressor event–cascade pairs')
    ax.set_title('D.  Self vs. cross-frame suppression',
                 fontweight='bold', loc='left')
    ax.legend(loc='lower right', framealpha=0.92, fontsize=7)


# ─── Panel E: R² evolution ───────────────────────────────────────────────────

def panel_e_r2(ax, df_summary):
    """R² over time with LOESS + individual frame-year scatter."""
    df = df_summary[(df_summary['n_cascades'] > 0) & (df_summary['median_r2'] > 0)].copy()

    yearly_r2 = []
    for year, grp in df.groupby('year'):
        w = grp['n_cascades'].values
        r2s = grp['median_r2'].values
        yearly_r2.append({'year': year, 'r2': np.average(r2s, weights=w)})
    yr2 = pd.DataFrame(yearly_r2)

    # Individual points
    for frame in FRAMES:
        fd = df[df['frame'] == frame]
        if fd.empty:
            continue
        ax.scatter(fd['year'], fd['median_r2'], s=14,
                   color=FRAME_COLORS[frame], alpha=0.35,
                   edgecolors='none', zorder=1)

    # LOESS on weighted annual median
    if len(yr2) >= 5:
        xf, ys, lo, hi = loess_smooth(yr2['year'].values, yr2['r2'].values, frac=0.40)
        ax.plot(xf, ys, color='#2C3E50', lw=2.5, zorder=4, label='Weighted mean (LOESS)')
        ax.fill_between(xf, lo, hi, color='#2C3E50', alpha=0.10, zorder=3)

    ax.set_ylabel('Median $R^2$')
    ax.set_xlabel('Year')
    ax.set_title('E.  Model explanatory power ($R^2$) over time',
                 fontweight='bold', loc='left')

    # Frame color legend (small)
    handles = [Line2D([0], [0], marker='o', color=FRAME_COLORS[f], lw=0,
                       markersize=5, alpha=0.5, label=f) for f in FRAMES]
    handles.append(Line2D([0], [0], color='#2C3E50', lw=2.5, label='LOESS'))
    ax.legend(handles=handles, loc='upper left', ncol=3, framealpha=0.92,
              fontsize=6, columnspacing=0.4, handletextpad=0.3)


# ─── Panel F: Composition timeline ───────────────────────────────────────────

def panel_f_composition(ax, df):
    """Stacked area: driver event-type composition over time."""
    if df.empty:
        return

    drivers = df[df['role'] == 'driver'].copy()
    if drivers.empty:
        return

    pivot = drivers.groupby(['year', 'dominant_type']).size().unstack(fill_value=0)
    totals = pivot.sum(axis=1)
    pivot_pct = pivot.div(totals, axis=0)
    for et in EVENT_TYPES:
        if et not in pivot_pct.columns:
            pivot_pct[et] = 0.0

    years = pivot_pct.index.values
    type_order = pivot_pct[EVENT_TYPES].sum().sort_values(ascending=False).index.tolist()

    sigma = 1.5 if len(years) >= 10 else 0.8
    smoothed = {}
    for et in type_order:
        vals = pivot_pct[et].values.astype(float)
        smoothed[et] = np.clip(gaussian_filter1d(vals, sigma=sigma), 0, None) if len(years) >= 5 else vals

    total = sum(smoothed[et] for et in type_order)
    total[total == 0] = 1
    for et in type_order:
        smoothed[et] /= total

    bottom = np.zeros(len(years))
    for et in type_order:
        ax.fill_between(years, bottom, bottom + smoothed[et],
                        color=EVENT_COLORS[et], alpha=0.80,
                        label=EVENT_LABELS[et], zorder=2)
        bottom += smoothed[et]

    ax.set_ylabel('Proportion of drivers')
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Year')
    ax.set_title('F.  Driver event-type composition over time',
                 fontweight='bold', loc='left', fontsize=11.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', ncol=4, framealpha=0.92,
              fontsize=7, columnspacing=0.6, handlelength=1.5)


# ─── Panel G: Dual-role redistribution flows ──────────────────────────────────

def panel_g_dual_flows(ax, df):
    """Arrow plot showing how dual-role clusters redistribute attention:
    which frames they drive vs suppress. Net flow matrix."""
    if df.empty or 'is_dual' not in df.columns:
        return

    dual = df[df['is_dual']].copy()
    if len(dual) < 10:
        ax.text(0.5, 0.5, 'Insufficient dual-role data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='#999')
        return

    # Build flow matrix: for each dual cluster, find which frames it drives
    # and which it suppresses → count directed flows
    from collections import Counter
    flows = Counter()  # (driven_frame, suppressed_frame) → count
    for cid, grp in dual.groupby('cid_year'):
        driven = grp[grp['role'] == 'driver']['frame'].unique()
        suppressed = grp[grp['role'] == 'suppressor']['frame'].unique()
        for d_f in driven:
            for s_f in suppressed:
                flows[(d_f, s_f)] += 1

    if not flows:
        return

    # Build matrix
    flow_matrix = pd.DataFrame(0, index=FRAMES, columns=FRAMES, dtype=float)
    for (d_f, s_f), cnt in flows.items():
        if d_f in FRAMES and s_f in FRAMES:
            flow_matrix.loc[d_f, s_f] = cnt

    # Net balance per frame: total driven by dual clusters minus total suppressed
    net_driven = flow_matrix.sum(axis=1)  # times this frame was driven
    net_suppressed = flow_matrix.sum(axis=0)  # times this frame was suppressed
    net_balance = net_driven - net_suppressed

    # Sort by net balance
    frame_order = net_balance.sort_values().index.tolist()
    y_pos = np.arange(len(frame_order))

    for i, frame in enumerate(frame_order):
        bal = net_balance[frame]
        color = ROLE_COLORS['driver'] if bal >= 0 else ROLE_COLORS['suppressor']
        ax.barh(i, bal, 0.65, color=FRAME_COLORS[frame], alpha=0.75,
                edgecolor='white', linewidth=0.3, zorder=2)
        # Top source/target annotation
        if bal >= 0 and net_driven[frame] > 0:
            top_target = flow_matrix.loc[frame].idxmax()
            cnt = int(flow_matrix.loc[frame, top_target])
            ax.text(bal + 1, i, f'→ {top_target} ({cnt}×)',
                    fontsize=6.5, va='center', color='#666', fontstyle='italic')
        elif bal < 0 and net_suppressed[frame] > 0:
            top_source = flow_matrix[frame].idxmax()
            cnt = int(flow_matrix.loc[top_source, frame])
            ax.text(bal - 1, i, f'{top_source} → ({cnt}×)',
                    fontsize=6.5, va='center', ha='right', color='#666',
                    fontstyle='italic')

    ax.axvline(0, color='#333', lw=0.8, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([FRAME_LABELS[f] for f in frame_order], fontsize=8.5)
    ax.set_xlabel('Net balance (driven − suppressed by dual-role clusters)')

    n_dual = df['is_dual'].sum()
    n_unique = df[df['is_dual']]['cid_year'].nunique()
    ax.set_title(f'G.  Dual-role cluster redistribution ({n_unique} clusters, '
                 f'{n_dual} entries)',
                 fontweight='bold', loc='left', fontsize=11)

    handles = [mpatches.Patch(facecolor=ROLE_COLORS['driver'], alpha=0.6,
                              label='Net beneficiary'),
               mpatches.Patch(facecolor=ROLE_COLORS['suppressor'], alpha=0.6,
                              label='Net target')]
    ax.legend(handles=handles, loc='lower right', framealpha=0.92, fontsize=7)


# ─── Panel H: Dual-role event profile & asymmetry ────────────────────────────

def panel_h_dual_profile(ax, df):
    """Compare dual-role vs single-role clusters: event-type profile and
    driver/suppressor β asymmetry. Grouped bars + scatter overlay."""
    if df.empty or 'is_dual' not in df.columns:
        return

    # Event-type prevalence: dual vs single-role
    for_et = df.drop_duplicates(subset='cid_year')  # one row per cluster
    dual_mask = for_et['is_dual']

    if dual_mask.sum() < 5 or (~dual_mask).sum() < 5:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='#999')
        return

    et_types = [et for et in EVENT_TYPES
                if (for_et['dominant_type'] == et).sum() >= 5]

    # Compute overrepresentation ratio: % in dual / % in single
    dual_total = dual_mask.sum()
    single_total = (~dual_mask).sum()

    ratios = []
    for et in et_types:
        pct_dual = (for_et[dual_mask]['dominant_type'] == et).sum() / dual_total
        pct_single = (for_et[~dual_mask]['dominant_type'] == et).sum() / single_total
        ratio = pct_dual / pct_single if pct_single > 0 else 0
        n_dual_et = (for_et[dual_mask]['dominant_type'] == et).sum()
        ratios.append((et, ratio, pct_dual * 100, pct_single * 100, n_dual_et))

    ratios.sort(key=lambda x: x[1])
    et_sorted = [r[0] for r in ratios]
    ratio_vals = [r[1] for r in ratios]
    pct_dual_vals = [r[2] for r in ratios]
    pct_single_vals = [r[3] for r in ratios]
    n_dual_vals = [r[4] for r in ratios]

    y_pos = np.arange(len(et_sorted))
    bar_h = 0.35

    # Grouped bars: dual vs single prevalence
    ax.barh(y_pos + bar_h/2, pct_dual_vals, bar_h, color='#E67E22', alpha=0.80,
            label=f'Dual-role (n={dual_total})', zorder=2, edgecolor='white',
            linewidth=0.3)
    ax.barh(y_pos - bar_h/2, pct_single_vals, bar_h, color='#95A5A6', alpha=0.70,
            label=f'Single-role (n={single_total})', zorder=2, edgecolor='white',
            linewidth=0.3)

    # Overrepresentation ratio on secondary axis
    ax2 = ax.twiny()
    ax2.scatter(ratio_vals, y_pos, s=50, color='#2C3E50', marker='D',
                zorder=5, edgecolors='white', linewidths=0.5)
    ax2.axvline(1.0, color='#999', lw=0.8, ls=':', zorder=1)
    ax2.set_xlabel('Overrepresentation ratio (diamond)', fontsize=8, color='#2C3E50')
    ax2.spines['top'].set_visible(True)
    ax2.spines['top'].set_color('#2C3E50')
    ax2.spines['top'].set_linewidth(0.5)
    ax2.tick_params(axis='x', colors='#2C3E50', labelsize=7.5)

    # Annotations
    for i, (et, r, n_d) in enumerate(zip(et_sorted, ratio_vals, n_dual_vals)):
        label = f'{r:.1f}× (n={n_d})'
        ax2.text(r + 0.05, i, label, fontsize=6.5, va='center', color='#2C3E50')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([EVENT_LABELS.get(et, et) for et in et_sorted], fontsize=8.5)
    ax.set_xlabel('Prevalence (%)')
    ax.set_title('H.  Dual-role cluster event profile (vs. single-role)',
                 fontweight='bold', loc='left', fontsize=11)
    ax.legend(loc='lower right', framealpha=0.92, fontsize=7)


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_figure(df_summary, df, years_range, out_path, fmt='png'):
    """Build 8-panel figure (A–H)."""
    fig = plt.figure(figsize=(18, 32))
    gs = gridspec.GridSpec(5, 2,
                           height_ratios=[0.7, 1.0, 0.85, 0.85, 0.7],
                           hspace=0.38, wspace=0.30,
                           left=0.07, right=0.95, top=0.965, bottom=0.025)

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])
    ax_g = fig.add_subplot(gs[3, 0])
    ax_h = fig.add_subplot(gs[3, 1])
    ax_f = fig.add_subplot(gs[4, :])

    panel_a_timeline(ax_a, df_summary)
    panel_b_frame_bars(ax_b, df)
    panel_c_event_effect(ax_c, df)
    panel_d_self_suppression(ax_d, df)
    panel_e_r2(ax_e, df_summary)
    panel_g_dual_flows(ax_g, df)
    panel_h_dual_profile(ax_h, df)
    panel_f_composition(ax_f, df)

    active_years = df_summary[df_summary['n_cascades'] > 0]['year']
    x_min, x_max = active_years.min() - 1, active_years.max() + 1
    for ax in [ax_a, ax_e, ax_f]:
        ax.set_xlim(x_min, x_max)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))

    n_clust = len(df)
    n_drv = (df['role'] == 'driver').sum() if not df.empty else 0
    n_sup = (df['role'] == 'suppressor').sum() if not df.empty else 0
    n_dual = df['cid_year'].nunique() if not df.empty and 'is_dual' in df.columns else 0
    n_dual_c = df[df['is_dual']]['cid_year'].nunique() if not df.empty and 'is_dual' in df.columns else 0
    fig.suptitle(
        f'Event–cascade dynamics in Canadian climate change media '
        f'({years_range[0]}–{years_range[-1]})\n'
        f'{len(years_range)} years  ·  {n_clust:,} significant event clusters '
        f'({n_drv:,} drivers, {n_sup:,} suppressors, {n_dual_c} dual-role)  ·  '
        f'Stability selection ($B$=100, $\\pi$≥0.60)',
        fontsize=12.5, fontweight='bold', y=0.99,
    )

    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', format=fmt)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def main():
    results_dir = PROJECT_ROOT / 'results' / 'stabsel_production'
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    year_summaries, df_clusters = load_all_data(results_dir)
    if not year_summaries:
        logger.error("No completed years found")
        sys.exit(1)

    df_summary, df_clusters = prepare_data(year_summaries, df_clusters)
    years_range = sorted(year_summaries.keys())
    logger.info(f"Years: {years_range[0]}–{years_range[-1]} ({len(years_range)} years)")
    logger.info(f"Significant clusters: {len(df_clusters)}")

    build_figure(df_summary, df_clusters, years_range,
                 results_dir / 'fig_stabsel_dynamics.png', fmt='png')
    build_figure(df_summary, df_clusters, years_range,
                 results_dir / 'fig_stabsel_dynamics.pdf', fmt='pdf')


if __name__ == '__main__':
    main()
