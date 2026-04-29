#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
gen_figures.py

MAIN OBJECTIVE:
---------------
Generate all publication figures for Nature Climate Change submission.
All values computed from production data. Nothing hardcoded.

Figures:
  1. Schema — Three orders of discursive change (matplotlib)
  2. Paradigmatic structure — Secular trends + composite dominance scores
  3. First order — Frame displacement, driver ratio, paradigmatic impact
  4. Structural residue — Quartile impact, destructive ratio, triple chains

Usage:
  python paper/papers/CCF_paradigm/scripts/gen_figures.py

Author:
-------
Antoine Lemor
"""

import json, glob, warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PROD = ROOT / 'results' / 'production'
OUT  = Path(__file__).resolve().parent.parent / 'figures'

FRAMES = ['Cult', 'Eco', 'Envt', 'Pbh', 'Just', 'Pol', 'Sci', 'Secu']
FL = {'Cult':'Culture','Eco':'Economy','Envt':'Environment',
      'Pbh':'Health','Just':'Justice','Pol':'Politics',
      'Sci':'Science','Secu':'Security'}
C = {'Cult':'#d62728','Eco':'#1f77b4','Envt':'#2ca02c',
     'Pbh':'#ff7f0e','Just':'#9467bd','Pol':'#8c564b',
     'Sci':'#17becf','Secu':'#e377c2'}

def setup():
    plt.rcParams.update({
        'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':7,'axes.labelsize':8,'axes.titlesize':9,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6,
        'axes.linewidth':0.5,'lines.linewidth':1.0,
        'xtick.major.width':0.5,'ytick.major.width':0.5,
        'xtick.major.size':3,'ytick.major.size':3,
        'axes.spines.top':False,'axes.spines.right':False,
        'figure.dpi':150,'savefig.dpi':600,
        'savefig.bbox':'tight','savefig.pad_inches':0.08,
        'figure.facecolor':'white','axes.facecolor':'white',
        'pdf.fonttype':42,'ps.fonttype':42,
    })

def plab(ax, s, x=-0.10, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

def smooth(y, s=1.5):
    return gaussian_filter1d(np.asarray(y, dtype=float), sigma=s)

# ── Loaders ──
def load_tl():
    df = pd.read_parquet(PROD/'cross_year_paradigm_timeline.parquet')
    df['date'] = pd.to_datetime(df['date']); return df

def load_casc():
    df = pd.read_parquet(PROD/'cross_year_cascades.parquet')
    df['onset_date'] = pd.to_datetime(df['onset_date']); return df

def load_pq(pat):
    fs = []
    for fp in sorted(glob.glob(str(PROD/'*/impact_analysis'/pat))):
        df = pd.read_parquet(fp)
        if df.shape[0] > 0:
            df['year'] = int(Path(fp).parts[-3]); fs.append(df)
    return pd.concat(fs, ignore_index=True) if fs else pd.DataFrame()

def load_cc(): return load_pq('cluster_cascade.parquet')
def load_cdom(): return load_pq('stabsel_cluster_dominance.parquet')
def load_casdom():
    df = load_pq('stabsel_cascade_dominance.parquet')
    if 'is_own_frame' not in df.columns and 'cascade_frame' in df.columns:
        df['is_own_frame'] = df['cascade_frame'] == df['target_frame']
    return df
def load_ab(): return load_pq('stabsel_alignment_b.parquet')
def load_aa(): return load_pq('stabsel_alignment_a.parquet')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Schema
# ══════════════════════════════════════════════════════════════════════════
def fig_schema():
    print("Fig 1: Schema...")
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    ax.set_xlim(-0.5, 12.0); ax.set_ylim(-1.3, 9.0); ax.axis('off')
    ax.set_aspect('equal', adjustable='datalim')

    cx, cy = 5.0, 4.5

    # ── Two rings: paradigm (outer) and cascade (middle) ──
    for r, col, alpha in [(4.4, '#B0BEC5', 0.10), (2.9, '#90A4AE', 0.12)]:
        ax.add_patch(plt.Circle((cx, cy), r, facecolor=col, alpha=alpha,
                     edgecolor=col, lw=1.0, zorder=1))

    # ── Large star at center (focusing event) ──
    n_points = 12
    outer_r, inner_r = 1.8, 1.0
    star_x, star_y = [], []
    for i in range(2 * n_points):
        angle = np.pi / 2 + i * np.pi / n_points
        r = outer_r if i % 2 == 0 else inner_r
        star_x.append(cx + r * np.cos(angle))
        star_y.append(cy + r * np.sin(angle))
    star_x.append(star_x[0]); star_y.append(star_y[0])
    ax.fill(star_x, star_y, color='#546E7A', alpha=0.18, zorder=3)
    ax.plot(star_x, star_y, color='#455A64', lw=0.8, alpha=0.5, zorder=3)

    # ── Text INSIDE the star ──
    ax.text(cx, cy + 0.45, 'Focusing event', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#263238', fontfamily='serif', zorder=5)
    ax.text(cx, cy - 0.25, 'An event focuses attention\nwithout determining\nits interpretation',
            ha='center', va='center', fontsize=6, color='#37474F',
            fontfamily='serif', linespacing=1.3, zorder=5)

    # ── Text INSIDE the cascade ring (upper band between star and outer ring) ──
    ax.text(cx, cy + 2.45, 'Media cascade', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#263238', fontfamily='serif', zorder=2)
    ax.text(cx, cy + 2.0, 'Outlets converge on a frame\nthat may differ from the event\'s own',
            ha='center', va='center', fontsize=6, color='#607D8B',
            fontfamily='serif', linespacing=1.3, zorder=2)

    # ── Text INSIDE the paradigm ring (top band) ──
    ax.text(cx, cy + 3.85, 'Paradigm dominance', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='#263238', fontfamily='serif', zorder=2)
    ax.text(cx, cy + 3.35, 'Discursive structure that persists\nafter events and cascades are forgotten',
            ha='center', va='center', fontsize=6, color='#607D8B',
            fontfamily='serif', linespacing=1.3, zorder=2)

    # ── Order labels — outside the outer circle, dotted lines to mid-band ──
    # Mid-band radii: paradigm band (2.9–4.4)=3.65, cascade band (1.8–2.9)=2.35, star=0.9
    for txt, y_lbl, mid_r in [('Third order', cy + 3.0, 3.65),
                               ('Second order', cy + 1.0, 2.35),
                               ('First order', cy - 1.0, 0.9)]:
        dy = y_lbl - cy
        x_mid = cx + np.sqrt(max(mid_r**2 - dy**2, 0)) if abs(dy) < mid_r else cx + mid_r
        ax.text(11.5, y_lbl, txt, ha='right', va='center', fontsize=8,
                color='#999', fontfamily='serif', fontstyle='italic')
        ax.plot([11.52, x_mid], [y_lbl, y_lbl], color='#bbb', lw=0.8, ls=':', zorder=0)

    # ── Structural persistence arrows on outer ring (larger arcs) ──
    persistence_color = '#666666'
    for angle in [125, 220, 330]:
        rad = np.radians(angle); dr = 0.40
        x1 = cx + 4.25 * np.cos(rad); y1 = cy + 4.25 * np.sin(rad)
        x2 = cx + 4.25 * np.cos(rad + dr); y2 = cy + 4.25 * np.sin(rad + dr)
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', lw=1.0, color=persistence_color,
                                    connectionstyle='arc3,rad=0.30', mutation_scale=8, alpha=0.7))

    # ── Compute safe arrow targets using polar coordinates ──
    arr_kw = dict(arrowstyle='-|>', mutation_scale=9, lw=1.0)
    star_r_val = 1.8; cascade_r_val = 2.9; paradigm_r_val = 4.4

    # A: center → mid-cascade band, at 35° angle (lower-right, below text)
    angle_a = np.radians(35)
    a_r = (star_r_val + cascade_r_val) / 2  # 2.35 — middle of cascade band
    a_end = (cx + a_r * np.cos(angle_a), cy + a_r * np.sin(angle_a))

    ax.annotate('',
        xy=a_end, xytext=(cx, cy),
        arrowprops=dict(**arr_kw, color='#455A64', connectionstyle='arc3,rad=-0.12'))
    ax.text(a_end[0] + 0.25, a_end[1] - 0.3, 'A', ha='center', va='center', fontsize=7,
            fontweight='bold', color='#455A64', fontfamily='serif')

    # B: from where A ends → mid-paradigm band, at 40° angle
    angle_b = np.radians(40)
    b_r = (cascade_r_val + paradigm_r_val) / 2  # 3.65 — middle of paradigm band
    b_end = (cx + b_r * np.cos(angle_b), cy + b_r * np.sin(angle_b))

    ax.annotate('',
        xy=b_end, xytext=a_end,
        arrowprops=dict(**arr_kw, color='#455A64', connectionstyle='arc3,rad=-0.12'))
    ax.text(b_end[0] + 0.25, b_end[1] - 0.3, 'B', ha='center', va='center', fontsize=7,
            fontweight='bold', color='#455A64', fontfamily='serif')

    # C: center → mid-paradigm band, at ~130° angle (upper-left, dashed)
    angle_c = np.radians(130)
    c_end = (cx + b_r * np.cos(angle_c), cy + b_r * np.sin(angle_c))

    ax.annotate('',
        xy=c_end, xytext=(cx, cy),
        arrowprops=dict(**arr_kw, color='#455A64', ls=(0, (4, 3)),
                        connectionstyle='arc3,rad=-0.30'))
    ax.text(c_end[0] - 0.3, c_end[1] - 0.5, 'C', ha='center', va='center', fontsize=7,
            fontweight='bold', color='#455A64', fontfamily='serif')

    # ── Legend OUTSIDE the outer ring, well below ──
    leg_x, leg_y = cx, cy - 5.5
    spacing = 0.30
    arr_leg = dict(arrowstyle='-|>', mutation_scale=7, lw=0.9)

    ax.annotate('', xy=(leg_x - 1.5, leg_y + spacing * 2), xytext=(leg_x - 2.2, leg_y + spacing * 2),
                arrowprops=dict(**arr_leg, color='#455A64'))
    ax.text(leg_x - 1.35, leg_y + spacing * 2, 'A + B  Cascade-mediated paradigmatic effect',
            va='center', fontsize=5.5, color='#455A64', fontfamily='serif')

    ax.annotate('', xy=(leg_x - 1.5, leg_y + spacing), xytext=(leg_x - 2.2, leg_y + spacing),
                arrowprops=dict(**arr_leg, color='#455A64', ls=(0, (4, 3))))
    ax.text(leg_x - 1.35, leg_y + spacing, 'C  Direct paradigmatic effect (need not align with A + B)',
            va='center', fontsize=5.5, color='#455A64', fontfamily='serif')

    ax.annotate('', xy=(leg_x - 1.5, leg_y), xytext=(leg_x - 2.2, leg_y),
                arrowprops=dict(arrowstyle='-|>', mutation_scale=6, lw=0.8,
                                color=persistence_color, connectionstyle='arc3,rad=0.3'))
    ax.text(leg_x - 1.35, leg_y, 'Structural persistence (economic interests, institutions)',
            va='center', fontsize=5.5, color=persistence_color, fontfamily='serif')

    fig.savefig(OUT / 'figure1_schema.pdf')
    plt.close(fig)
    print("  -> figure1_schema.pdf")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Paradigmatic structure
#   a) Secular trends with R2 and crossing
#   b) Dominance heatmap 8x47
#   c) Catalyst/disruptor balance Model A vs B
# ══════════════════════════════════════════════════════════════════════════
def fig_paradigm():
    print("Fig 2: Paradigm...")
    tl = load_tl(); tl['yr'] = tl['date'].dt.year

    fig = plt.figure(figsize=(7.2, 4.0))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.5, 0.8],
                           wspace=0.25, left=0.08, right=0.96, bottom=0.12, top=0.93)

    # (a) Secular trends: all frames visible, trends for all significant
    ax = fig.add_subplot(gs[0, 0]); plab(ax, 'a')
    yr = tl.groupby('yr')[[f'paradigm_{f}' for f in FRAMES]].mean()
    hl = {'Pol','Eco','Sci','Envt'}
    for f in FRAMES:
        s = smooth(yr[f'paradigm_{f}'].values)
        ax.plot(yr.index, s, color=C[f], lw=2.0 if f in hl else 0.8,
                alpha=0.90 if f in hl else 0.40, label=FL[f], solid_capstyle='round')

    # Linear trends for ALL frames with labels
    label_positions = []
    for f in FRAMES:
        yv = yr[f'paradigm_{f}'].values; xv = yr.index.values.astype(float)
        sl, ic, r, p, se = stats.linregress(xv, yv)
        sig = p < 0.05
        ax.plot(xv, sl*xv+ic, color=C[f], lw=0.7 if sig else 0.3,
                ls='--', alpha=0.30 if sig else 0.10)
        y_end = sl*xv[-1]+ic
        pct = (y_end - (sl*xv[0]+ic)) / (sl*xv[0]+ic) * 100
        ns = '' if sig else '\u2020'
        label_positions.append((y_end, pct, f, ns))
    # Avoid label overlap
    label_positions.sort(key=lambda x: x[0])
    min_gap = 0.022
    for i in range(1, len(label_positions)):
        if label_positions[i][0] - label_positions[i-1][0] < min_gap:
            y_old, pct, f, ns = label_positions[i]
            label_positions[i] = (label_positions[i-1][0] + min_gap, pct, f, ns)
    for y_end, pct, f, ns in label_positions:
        ax.text(2024.5, y_end, f'{pct:+.0f}%{ns}', fontsize=4.5, color=C[f], va='center',
                alpha=0.9 if not ns else 0.5)

    # Pol × Sci crossing
    sp, ip = stats.linregress(yr.index.values.astype(float), yr['paradigm_Pol'].values)[:2]
    ss, si = stats.linregress(yr.index.values.astype(float), yr['paradigm_Sci'].values)[:2]
    if abs(sp-ss) > 1e-10:
        cx_val = (si-ip)/(sp-ss)
        if 1978 <= cx_val <= 2024:
            ax.axvline(cx_val, color='grey', lw=0.5, ls=':', alpha=0.4)
            ax.annotate(f'~{cx_val:.0f}', xy=(cx_val, sp*cx_val+ip),
                       xytext=(cx_val+3, sp*cx_val+ip-0.06),
                       fontsize=6, color='grey',
                       arrowprops=dict(arrowstyle='->', color='grey', lw=0.4))

    ax.set_ylabel('Mean dominance index', fontsize=7)
    ax.set_xlabel('Year', fontsize=7)
    ax.set_xlim(1978, 2024); ax.set_ylim(0.15, 0.85)
    ax.tick_params(labelsize=6)
    ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.20), frameon=False, fontsize=5.5)

    # (b) Composite dominance score by frame (4-method consensus) — vertical bars
    ax2 = fig.add_subplot(gs[0, 1]); plab(ax2, 'b')

    dom_csv = Path(__file__).resolve().parent.parent / 'tables' / 'overall_dominance_scores.csv'
    if dom_csv.exists():
        dom_df = pd.read_csv(dom_csv, index_col=0)
        dom_scores_s = dom_df['dominance_score']
        sorted_f = dom_scores_s.sort_values(ascending=False).index.tolist()
        scores = [dom_scores_s[f] for f in sorted_f]

        import json
        details_path = dom_csv.parent / 'dominance_analysis_details.json'
        if details_path.exists():
            with open(details_path) as jf:
                details = json.load(jf)
            threshold = details.get('threshold', dom_scores_s.median())
            dominant_set = set(details.get('dominant_frames', []))
        else:
            threshold = dom_scores_s.median()
            dominant_set = set()
    else:
        dom_scores_s = pd.Series({f: yr[f'paradigm_{f}'].mean() for f in FRAMES})
        sorted_f = dom_scores_s.sort_values(ascending=False).index.tolist()
        scores = [dom_scores_s[f] for f in sorted_f]
        threshold = dom_scores_s.median()
        dominant_set = set()

    x_pos = np.arange(len(sorted_f))
    for i, f in enumerate(sorted_f):
        is_dom = f in dominant_set if dominant_set else scores[i] >= threshold
        ax2.bar(x_pos[i], scores[i], color=C.get(f, '#777'),
               alpha=0.85 if is_dom else 0.45,
               edgecolor='white', lw=0.3, width=0.7)
        ax2.text(x_pos[i], scores[i] + 0.015, f'{scores[i]:.2f}',
                ha='center', va='bottom', fontsize=4.5,
                color='black' if is_dom else 'gray')

    ax2.axhline(threshold, color='black', ls='--', lw=0.8, alpha=0.5)
    ax2.text(2.5, threshold + 0.03, f'threshold\n{threshold:.2f}',
            ha='center', va='bottom', fontsize=4.5, color='black')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([FL.get(f, f)[:4] for f in sorted_f], fontsize=5.5, rotation=45, ha='right')
    ax2.set_ylabel('Dominance score', fontsize=6.5)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(labelsize=5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.savefig(OUT/'figure2_paradigm.pdf')
    plt.close(fig)
    print("  -> figure2_paradigm.pdf")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — First order: frame displacement
#   a) Displacement rate by decade (% of driver links where cascade frame != event natural frame)
#   b) Driver ratio by decade
#   c) Driver ratio heatmap event_type x cascade_frame
# ══════════════════════════════════════════════════════════════════════════
def fig_displacement():
    print("Fig 3: Displacement...")
    cc = load_cc()

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0],
                           wspace=0.30, left=0.07, right=0.97, bottom=0.22, top=0.94)

    # Natural frame mapping (domain logic)
    natural = {'evt_weather':'Envt', 'evt_meeting':'Pol', 'evt_policy':'Pol',
               'evt_publication':'Sci', 'evt_election':'Pol', 'evt_judiciary':'Just',
               'evt_cultural':'Cult', 'evt_protest':'Pol'}

    # (a) Combined: displacement rate + driver ratio as smoothed yearly curves
    ax = fig.add_subplot(gs[0]); plab(ax, 'a')
    if not cc.empty and 'dominant_type' in cc.columns:
        drivers_all = cc[cc['role']=='driver'].copy()
        drivers_all['natural'] = drivers_all['dominant_type'].map(natural)
        drivers_all['displaced'] = drivers_all['cascade_frame'] != drivers_all['natural']

        # Compute yearly values — keep all years, use heavier smoothing
        years_all = sorted(cc['year'].unique())
        disp_by_year = []
        dr_by_year = []
        for y in years_all:
            drv = drivers_all[drivers_all['year']==y]
            all_y = cc[cc['year']==y]
            if len(drv) > 0:
                disp_by_year.append(drv['displaced'].mean() * 100)
            else:
                disp_by_year.append(np.nan)
            if len(all_y) > 0:
                dr_by_year.append(all_y[all_y['role']=='driver'].shape[0] / len(all_y) * 100)
            else:
                dr_by_year.append(np.nan)

        disp_arr = np.array(disp_by_year, dtype=float)
        dr_arr = np.array(dr_by_year, dtype=float)
        years_arr = np.array(years_all)

        # Interpolate NaNs for smoothing, then smooth with sigma=3
        for arr in [disp_arr, dr_arr]:
            nans = np.isnan(arr)
            if nans.any() and not nans.all():
                arr[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], arr[~nans])

        x_arr = years_arr

        # Plot linear regressions with CI bands (no smoothed curves)
        reg_lines = {}
        for y_data, color, label in [(disp_arr, '#d62728', 'Frame displacement'),
                                      (dr_arr, '#1f77b4', 'Driver ratio')]:
            sl, ic, r, p, se = stats.linregress(x_arr, y_data)
            y_pred = sl * x_arr + ic
            n_pts = len(x_arr)
            x_mean = x_arr.mean()
            residuals = y_data - y_pred
            mse = np.sum(residuals**2) / (n_pts - 2)
            se_pred = np.sqrt(mse * (1/n_pts + (x_arr - x_mean)**2 / np.sum((x_arr - x_mean)**2)))
            t_val = stats.t.ppf(0.975, n_pts - 2)

            ax.plot(x_arr, y_pred, color=color, lw=2.0, solid_capstyle='round')
            ax.fill_between(x_arr, y_pred - t_val*se_pred, y_pred + t_val*se_pred,
                           color=color, alpha=0.12)
            ax.text(2018, y_pred[-3] + (3 if color == '#d62728' else -3),
                    f'R\u00b2={r**2:.2f}', fontsize=5.5, color=color, va='center')
            reg_lines[label] = y_pred

        # Crossing point from regression lines
        y_disp = reg_lines['Frame displacement']
        y_dr = reg_lines['Driver ratio']
        for i in range(len(x_arr)-1):
            if y_disp[i] <= y_dr[i] and y_disp[i+1] >= y_dr[i+1]:
                frac = (y_dr[i] - y_disp[i]) / ((y_disp[i+1] - y_disp[i]) - (y_dr[i+1] - y_dr[i]))
                cx_val = x_arr[i] + frac * (x_arr[i+1] - x_arr[i])
                cy_val = (y_disp[i] + y_dr[i]) / 2
                ax.axvline(cx_val, color='grey', lw=0.5, ls=':', alpha=0.3)
                ax.annotate(f'~{cx_val:.0f}', xy=(cx_val, cy_val), xytext=(cx_val+2, cy_val+8),
                            fontsize=6, color='grey',
                            arrowprops=dict(arrowstyle='->', color='grey', lw=0.4))
                break

        ax.set_ylabel('Rate (%)', fontsize=7)
        ax.set_xlabel('Year', fontsize=7)
        ax.set_xlim(1983, 2024); ax.set_ylim(30, 100)
        ax.tick_params(labelsize=6)
        ax.grid(axis='y', alpha=0.1, ls=':', lw=0.3)

        # Legend below x-axis, close to "Year" label
        ax.plot([1987, 1990], [22, 22], color='#d62728', lw=2.0, clip_on=False)
        ax.text(1991, 22, 'Displacement', fontsize=5.5, color='#d62728', va='center', clip_on=False)
        ax.plot([2006, 2009], [22, 22], color='#1f77b4', lw=2.0, clip_on=False)
        ax.text(2010, 22, 'Driver ratio', fontsize=5.5, color='#1f77b4', va='center', clip_on=False)

    # (b) Grouped bar chart in a single axis
    ax = fig.add_subplot(gs[0, 1]); plab(ax, 'b')

    if not cc.empty and 'dominant_type' in cc.columns:
        evt_types = ['evt_weather','evt_meeting','evt_policy','evt_publication','evt_election','evt_judiciary']
        evt_labels = ['Weather','Meeting','Policy','Publication','Election','Judiciary']

        frame_scale = {}
        for f in FRAMES:
            vals = cc[cc['cascade_frame']==f]['net_beta'].abs()
            frame_scale[f] = vals.median() if len(vals) > 0 else 1.0

        n_evt = len(evt_types)
        n_frm = len(FRAMES)
        bar_h = 0.08
        group_size = n_frm * bar_h + 0.15

        xlim = (-1.8, 1.2)
        group_centers = []

        for idx_e, (et, el) in enumerate(zip(evt_types, evt_labels)):
            sub = cc[cc['dominant_type']==et]
            vals = []
            for f in FRAMES:
                fsub = sub[sub['cascade_frame']==f]
                n = len(fsub)
                if n >= 5:
                    v = fsub['net_beta'].median() / frame_scale[f]
                    vals.append(np.clip(v, xlim[0], xlim[1]))
                else:
                    vals.append(0)
            vals = np.array(vals)

            order = np.argsort(vals)
            group_bottom = idx_e * group_size

            # Frame bars
            for k_idx, k in enumerate(order):
                y_pos = group_bottom + k_idx * bar_h
                ax.barh(y_pos, vals[k], height=bar_h * 0.85, color=C[FRAMES[k]],
                       edgecolor='white', lw=0.2, alpha=0.8)

            group_centers.append(group_bottom + (n_frm - 1) * bar_h / 2)

        ax.axvline(0, color='#333', lw=0.5, alpha=0.4)
        ax.set_xlim(xlim)
        ax.set_xlabel('Normalised effect', fontsize=7)
        ax.tick_params(axis='x', labelsize=6)

        # Dotted separator lines between groups
        for idx_e in range(n_evt - 1):
            sep_y = (idx_e + 1) * group_size - (group_size - n_frm * bar_h) / 2
            ax.axhline(sep_y, color='#bbb', lw=0.4, ls=':', zorder=0)

        # Event type labels inside the plot, left-aligned
        for idx_e, el in enumerate(evt_labels):
            ax.text(xlim[0] + 0.03, group_centers[idx_e], el, ha='left', va='center',
                   fontsize=6.5, color='#444')

        # No y ticks
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Frame color legend
        import matplotlib.lines as mlines
        handles = [mlines.Line2D([], [], color=C[f], marker='s', linestyle='', ms=4,
                                  label=FL[f], alpha=0.8) for f in FRAMES]
        ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.18),
                  ncol=4, frameon=False, fontsize=5.5, columnspacing=0.8)

    # (c) Scatter: paradigmatic impact vs directional asymmetry by event type
    ax = fig.add_subplot(gs[0, 2]); plab(ax, 'c')

    cdom_a = load_cdom()
    align_a = load_aa()
    if not cdom_a.empty and not align_a.empty and 'dominant_type' in cdom_a.columns:
        data_c = cdom_a[(cdom_a['net_beta'].abs() < 100) & (cdom_a['role'] != 'inert')]

        evt_types_c = ['evt_weather','evt_meeting','evt_policy','evt_publication','evt_election','evt_judiciary']
        evt_labels_c = ['Weather','Meeting','Policy','Publication','Election','Judiciary']

        # Scatter: X = paradigmatic impact, Y = net score, by frame
        MIN_N = 5
        frame_markers = {'Pol': 'o', 'Eco': 'D', 'Sci': 's'}
        frame_colors_c = {'Pol': '#795548', 'Eco': '#2980b9', 'Sci': '#00bcd4'}
        frame_names_c = {'Pol': 'Pol', 'Eco': 'Eco', 'Sci': 'Sci'}
        evt_colors = {
            'Weather': '#d62728', 'Meeting': '#2ca02c', 'Policy': '#9467bd',
            'Publication': '#ff7f0e', 'Election': '#1f77b4', 'Judiciary': '#8c564b',
        }

        scatter_data = []
        for et, el in zip(evt_types_c, evt_labels_c):
            for frame in ['Pol', 'Eco', 'Sci']:
                sub = data_c[(data_c['dominant_type']==et) & (data_c['frame']==frame)]
                if len(sub) < MIN_N: continue
                # Impact specific to this event type × frame
                n_all_frame = len(cdom_a[(cdom_a['dominant_type']==et) & (cdom_a['frame']==frame)])
                impact = sub['net_beta'].abs().sum() / n_all_frame if n_all_frame > 0 else 0
                pct_cat = (sub['role']=='catalyst').mean() * 100
                net = pct_cat - (100 - pct_cat)
                scatter_data.append((el, frame, impact, net, len(sub)))

        # Plot — size proportional to n
        for el, frame, impact, net, n in scatter_data:
            sz = max(20, n * 0.8)  # scale: n=5 → 20, n=130 → 104
            ax.scatter(impact, net, marker=frame_markers[frame],
                      color=evt_colors[el], s=sz,
                      alpha=0.7, edgecolors='white', lw=0.3, zorder=3)

        # No inline labels — legend handles both dimensions

        ax.axhline(0, color='#333', lw=0.5, alpha=0.3)
        ax.set_xlabel('Paradigmatic impact\n($\\Sigma|\\beta|$ / $n$ effects)', fontsize=6.5)
        ax.set_ylabel('Net effect\n(erodes $\\leftarrow$ / $\\rightarrow$ reinforces)', fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend below x-axis, two rows
        from matplotlib.lines import Line2D
        handles = []
        # Row 1: frame shapes
        for f in ['Pol','Eco','Sci']:
            handles.append(Line2D([0],[0], marker=frame_markers[f], color='w',
                                  markerfacecolor='#777', markersize=4,
                                  markeredgecolor='white', markeredgewidth=0.3,
                                  label=frame_names_c[f]))
        # Row 2: event type colors
        for et, el in zip(evt_types_c, evt_labels_c):
            handles.append(Line2D([0],[0], marker='o', color='w',
                                  markerfacecolor=evt_colors[el], markersize=3.5,
                                  markeredgecolor='white', markeredgewidth=0.3,
                                  label=el))
        ax.legend(handles=handles, fontsize=4.5, frameon=False,
                 ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.12),
                 columnspacing=0.5, handletextpad=0.2)

    fig.savefig(OUT/'figure3_first_order.pdf')
    plt.close(fig)
    print("  -> figure3_first_order.pdf")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Structural residue
#   a) Paradigmatic impact by cascade score quartile
#   b) Destructive ratio by decade
#   c) Triple chain divergence by target frame
# ══════════════════════════════════════════════════════════════════════════
def fig_residue():
    print("Fig 4: Structural residue...")
    casdom = load_casdom(); ab = load_ab()
    cc_data = load_cc(); cdom_a_data = load_cdom()

    fig = plt.figure(figsize=(7.2, 7.5))
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[0.9, 0.9, 1.0],
                           hspace=0.50, left=0.12, right=0.92, bottom=0.06, top=0.96)

    # (a) Impact by cascade score quartile
    ax = fig.add_subplot(gs[0]); plab(ax, 'a')
    if not ab.empty and 'total_score' in ab.columns and 'impact_magnitude' in ab.columns:
        valid = ab.dropna(subset=['total_score','impact_magnitude'])
        valid = valid[valid['total_score'] > 0].copy()
        valid['quartile'] = pd.qcut(valid['total_score'], 4, labels=['Q1\n(weakest)','Q2','Q3','Q4\n(strongest)'])
        q_means = valid.groupby('quartile', observed=True)['impact_magnitude'].mean()
        bars = ax.bar(range(len(q_means)), q_means.values, color=['#d62728','#ff7f0e','#2196F3','#1f77b4'],
                      width=0.55, edgecolor='white', lw=0.3)
        ax.set_xticks(range(len(q_means)))
        ax.set_xticklabels(q_means.index, fontsize=6.5)
        ax.set_ylabel('Mean paradigmatic impact')
        for b, v in zip(bars, q_means.values):
            ax.text(b.get_x()+b.get_width()/2, v+q_means.max()*0.02, f'{v:.4f}', ha='center', fontsize=6)
        # Ratio annotation
        if len(q_means) == 4:
            ratio = q_means.iloc[0] / q_means.iloc[3]
            ax.text(0.97, 0.95, f'Q1/Q4 = {ratio:.1f}x', transform=ax.transAxes, ha='right', va='top',
                    fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.9))
    ax.set_title('Paradigmatic impact by cascade strength quartile', fontsize=7.5, pad=4)

    # (b) Destructive ratio by decade
    ax = fig.add_subplot(gs[1]); plab(ax, 'b')
    if not casdom.empty:
        casdom_c = casdom.copy()
        casdom_c['decade'] = (casdom_c['year']//10)*10
        decades = sorted(casdom_c['decade'].unique())
        ratios = []
        for d in decades:
            g = casdom_c[casdom_c['decade']==d]
            nd = (g['role']=='disruptor').sum(); nc = (g['role']=='catalyst').sum()
            ratios.append(nd/nc if nc > 0 else 0)
        colors = ['#2196F3' if r <= 1.0 else '#d62728' for r in ratios]
        bars = ax.bar(range(len(decades)), ratios, color=colors, width=0.55, edgecolor='white', lw=0.3)
        ax.set_xticks(range(len(decades)))
        ax.set_xticklabels([f'{d}s' for d in decades], fontsize=6)
        ax.set_ylabel('Disruptor / catalyst ratio')
        ax.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.5)
        ax.text(len(decades)-0.5, 1.02, 'parity', fontsize=5.5, ha='right', color='grey')
        for b, v in zip(bars, ratios):
            ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.2f}', ha='center', fontsize=6, fontweight='bold')
    ax.set_title('Cascade destructive ratio by decade', fontsize=7.5, pad=4)

    # (c) Triple chain divergence by frame
    ax = fig.add_subplot(gs[2]); plab(ax, 'c')
    cc_sig = cc_data[cc_data['role'].isin(['driver','suppressor'])]
    a_sig = cdom_a_data[cdom_a_data['p_value_hac']<0.10] if 'p_value_hac' in cdom_a_data.columns else cdom_a_data[cdom_a_data['role']!='inert']
    b_sig = casdom[casdom['p_value_hac']<0.10] if 'p_value_hac' in casdom.columns else casdom[casdom['role']!='inert']

    triples_list = []
    for _, row in cc_sig.iterrows():
        clu, cas, yr = row['cluster_id'], row['cascade_id'], row['year']
        a_m = a_sig[(a_sig['cluster_id']==clu)&(a_sig['year']==yr)]
        b_m = b_sig[(b_sig['cascade_id']==cas)&(b_sig['year']==yr)]
        if a_m.empty or b_m.empty: continue
        for _, a in a_m.iterrows():
            for _, b in b_m.iterrows():
                if a['frame'] == b['target_frame']:
                    triples_list.append({'frame': a['frame'], 'a_beta': a['net_beta'], 'b_beta': b['net_beta']})
    if triples_list:
        tdf = pd.DataFrame(triples_list)
        tdf['divergent'] = (tdf['a_beta'] * tdf['b_beta']) < 0
        conv_counts = []; div_counts = []
        for f in FRAMES:
            sub = tdf[tdf['frame']==f]
            conv_counts.append((~sub['divergent']).sum())
            div_counts.append(sub['divergent'].sum())
        conv_counts = np.array(conv_counts); div_counts = np.array(div_counts)
        y = np.arange(len(FRAMES)); w = 0.35
        ax.barh(y-w/2, conv_counts, w, color='#1f77b4', label='Convergent')
        ax.barh(y+w/2, div_counts, w, color='#d62728', label='Divergent')
        for i in range(len(FRAMES)):
            tot = conv_counts[i] + div_counts[i]
            if tot > 0:
                pct = div_counts[i] / tot * 100
                ax.text(max(conv_counts[i], div_counts[i]) + 0.5, y[i],
                        f'{pct:.0f}% div. (n={tot})', ha='left', va='center', fontsize=5)
        ax.set_yticks(y); ax.set_yticklabels([FL[f] for f in FRAMES], fontsize=6.5)
        ax.set_xlabel('Same-frame triple chains')
        ax.legend(frameon=False, fontsize=5.5, loc='lower right')
        total_div = tdf['divergent'].sum(); total_n = len(tdf)
        ax.text(0.97, 0.97, f'Total: {total_n} chains\n{total_div} divergent ({total_div/total_n*100:.0f}%)',
                transform=ax.transAxes, ha='right', va='top', fontsize=6,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.9))
    ax.set_title('Triple chain divergence by target frame', fontsize=7.5, pad=4)

    fig.savefig(OUT/'figure5_third_order.pdf')
    plt.close(fig)
    print("  -> figure5_third_order.pdf")


# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    setup()
    print(f"Data: {PROD}\nOutput: {OUT}\n")
    fig_schema()
    fig_paradigm()
    fig_displacement()
    # fig_residue() — replaced by gen_figure_third_order.py
    print("\nDone.")
