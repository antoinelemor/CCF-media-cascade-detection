#!/usr/bin/env python3
# DEPRECATED: Uses legacy EventImpactAnalyzer output format (prevalence ratios).
# Kept for backward compatibility with existing pickle/parquet results.
# New analyses should use UnifiedImpactAnalyzer from cascade_detector.analysis.
"""
Single SOTA figure: Which events drive media cascades? (2018, all frames)

Layout: 2x2 grid with fig.text captions beneath each panel.
  A. Parallel coordinates -- 3 metric dimensions per event type
  B. Heatmap -- per-frame prevalence ratios with significance overlay
  C. Dot plot -- effect on cascade strength (median split, 95% CI)
  D. Composite impact score -- final ranking with decomposition

Style: Nature Communications / PNAS conventions.

Author: Antoine Lemor
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

# -- Style -----------------------------------------------------------------

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8.5,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
})

EVENT_COLORS = {
    'evt_election':    '#1B9E77',
    'evt_judiciary':   '#D95F02',
    'evt_weather':     '#7570B3',
    'evt_policy':      '#E7298A',
    'evt_meeting':     '#66A61E',
    'evt_publication': '#E6AB02',
    'evt_protest':     '#A6761D',
    'evt_cultural':    '#666666',
}

EVENT_LABELS_SHORT = {
    'evt_weather':     'Weather',
    'evt_meeting':     'Meetings',
    'evt_publication': 'Publications',
    'evt_election':    'Elections',
    'evt_policy':      'Policy',
    'evt_judiciary':   'Judiciary',
    'evt_cultural':    'Cultural',
    'evt_protest':     'Protests',
}

FRAME_ORDER = ['Cult', 'Eco', 'Envt', 'Just', 'Pbh', 'Pol', 'Sci', 'Secu']
FRAME_LABELS = {
    'Cult': 'Cultural', 'Eco': 'Economic', 'Envt': 'Environ.',
    'Pbh': 'Health', 'Just': 'Justice', 'Pol': 'Political',
    'Sci': 'Scientific', 'Secu': 'Security',
}

OUTDIR = Path('results/figures_2018')
OUTDIR.mkdir(parents=True, exist_ok=True)

# -- Load data -------------------------------------------------------------

with open('results/test_2018_results.pkl', 'rb') as f:
    results = pickle.load(f)

impact = results.event_impact
prev_df = impact['prevalence_ratios']
surge_df = impact['pre_onset_surge']
corr_df = impact['strength_correlations']

prev_evt = prev_df[prev_df['type'] == 'event'].copy()
surge_evt = surge_df[surge_df['type'] == 'event'].copy()
corr_evt = corr_df[corr_df['type'] == 'event'].copy()

events = sorted(prev_evt['annotation'].unique())

# -- Compute aggregates ----------------------------------------------------

agg = pd.DataFrame(index=events)
agg['mean_pr'] = prev_evt.groupby('annotation').apply(
    lambda g: np.average(g['prevalence_ratio'].clip(upper=10), weights=g['n_cascade']))
agg['mean_surge'] = surge_evt.groupby('annotation')['median_surge'].mean()
agg['mean_rho'] = corr_evt.groupby('annotation')['spearman_rho'].mean()
agg['n_sig'] = (
    prev_evt.groupby('annotation').apply(lambda g: (g['p_value_adjusted'] < 0.05).sum()) +
    surge_evt.groupby('annotation').apply(lambda g: (g['p_value_adjusted'] < 0.05).sum()) +
    corr_evt.groupby('annotation').apply(lambda g: (g['p_value_adjusted'] < 0.05).sum())
)

for col in ['mean_pr', 'mean_surge', 'mean_rho']:
    vals = agg[col]
    agg[f'{col}_norm'] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)

agg['composite'] = (
    agg['mean_pr_norm'] * 0.35 +
    agg['mean_surge_norm'] * 0.30 +
    agg['mean_rho_norm'] * 0.35
)

rank_order = agg.sort_values('composite', ascending=False).index.tolist()

# -- Compute per-cascade event prevalence (for Panel C) --------------------

articles = results._articles
cascades = results.cascades
date_col = 'date_converted_first'
evt_sum_cols = [c for c in articles.columns if c.startswith('evt_') and c.endswith('_sum')]

cascade_rows = []
for c in cascades:
    onset = pd.Timestamp(c.onset_date)
    end = pd.Timestamp(c.end_date)
    mask = (articles[date_col] >= onset) & (articles[date_col] <= end)
    casc_art = articles[mask]
    if len(casc_art) == 0:
        continue
    row = {'cascade_id': c.cascade_id, 'frame': c.frame,
           'total_score': c.total_score, 'n_articles': len(casc_art),
           'classification': c.classification}
    for ec in evt_sum_cols:
        row[ec.replace('_sum', '')] = (casc_art[ec] > 0).mean()
    cascade_rows.append(row)

cascade_evt_df = pd.DataFrame(cascade_rows)

# -- Compute Panel C data (median split) -----------------------------------

from scipy import stats as sp_stats

evt_cols_in_df = [c for c in cascade_evt_df.columns if c.startswith('evt_')]

diff_data = []
for evt in rank_order:
    if evt not in evt_cols_in_df:
        continue
    med = cascade_evt_df[evt].median()
    high_mask = cascade_evt_df[evt] > med
    low_mask = cascade_evt_df[evt] <= med

    scores_high = cascade_evt_df.loc[high_mask, 'total_score']
    scores_low = cascade_evt_df.loc[low_mask, 'total_score']

    if len(scores_high) < 2 or len(scores_low) < 2:
        continue

    mean_diff = scores_high.mean() - scores_low.mean()

    n1, n2 = len(scores_high), len(scores_low)
    v1, v2 = scores_high.var(), scores_low.var()
    se = np.sqrt(v1 / n1 + v2 / n2)
    df_ws = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1)) \
            if (v1/n1 + v2/n2) > 0 else 2
    t_crit = sp_stats.t.ppf(0.975, df_ws)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se

    _, p_val = sp_stats.ttest_ind(scores_high, scores_low, equal_var=False)

    diff_data.append({
        'event': evt,
        'mean_diff': mean_diff,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'se': se,
        'p_value': p_val,
        'n_high': n1,
        'n_low': n2,
    })

diff_df = pd.DataFrame(diff_data)
diff_df = diff_df.sort_values('mean_diff', ascending=True).reset_index(drop=True)

# -- Prepare dynamic example strings for captions --------------------------

# Panel A: find top event per metric for accurate caption
top_pr_evt = agg['mean_pr'].idxmax()
top_surge_evt = agg['mean_surge'].idxmax()
top_rho_evt = agg['mean_rho'].idxmax()
top_pr_label = EVENT_LABELS_SHORT[top_pr_evt]
top_surge_label = EVENT_LABELS_SHORT[top_surge_evt]
top_rho_label = EVENT_LABELS_SHORT[top_rho_evt]

best_pr_row = prev_evt.sort_values('prevalence_ratio', ascending=False).iloc[0]
best_pr_evt = EVENT_LABELS_SHORT[best_pr_row['annotation']]
best_pr_frame = FRAME_LABELS[best_pr_row['frame']]
best_pr_val = best_pr_row['prevalence_ratio']

top_c = diff_df.iloc[-1]
top_c_label = EVENT_LABELS_SHORT[top_c['event']]
top_c_diff = top_c['mean_diff']

top_d_label = EVENT_LABELS_SHORT[rank_order[0]]
top_d_score = agg.loc[rank_order[0], 'composite']
bot_d_label = EVENT_LABELS_SHORT[rank_order[-1]]
bot_d_score = agg.loc[rank_order[-1], 'composite']

# =========================================================================
# Build figure: simple 2x2 grid, captions via fig.text below each panel
# =========================================================================

fig = plt.figure(figsize=(16, 15))

gs = gridspec.GridSpec(2, 2, width_ratios=[1.1, 1.0],
                       height_ratios=[1.0, 1.0],
                       hspace=0.42, wspace=0.30)

# ==========================================================================
# Panel A: Slope chart (parallel coordinates) -- 3 metrics
# ==========================================================================

ax_a = fig.add_subplot(gs[0, 0])

metrics = ['mean_pr', 'mean_surge', 'mean_rho']
metric_labels = ['Prevalence\nRatio', 'Pre-Onset\nSurge', 'Strength\nCorrelation']

norm_vals = {}
for m in metrics:
    vals = agg[m]
    lo, hi = vals.min(), vals.max()
    norm_vals[m] = (vals - lo) / (hi - lo + 1e-10)

x_positions = [0, 1, 2]
label_positions = []

for evt in rank_order:
    color = EVENT_COLORS[evt]
    y_vals = [norm_vals[m][evt] for m in metrics]
    rank_pos = rank_order.index(evt)
    lw = max(2.5 - rank_pos * 0.2, 1.0)
    alpha = max(0.95 - rank_pos * 0.07, 0.4)

    ax_a.plot(x_positions, y_vals, color=color, linewidth=lw,
              alpha=alpha, zorder=10 - rank_pos, solid_capstyle='round')
    for xi, yi in zip(x_positions, y_vals):
        ax_a.scatter(xi, yi, color=color, s=35, zorder=11 - rank_pos,
                     edgecolors='white', linewidth=0.5)
    label_positions.append((evt, y_vals[-1], color, rank_pos, alpha))

label_positions.sort(key=lambda x: x[1])
adjusted_y = [d[1] for d in label_positions]
min_gap = 0.085
for _ in range(30):
    for i in range(1, len(adjusted_y)):
        if adjusted_y[i] - adjusted_y[i-1] < min_gap:
            overlap = min_gap - (adjusted_y[i] - adjusted_y[i-1])
            adjusted_y[i-1] -= overlap / 2
            adjusted_y[i] += overlap / 2

for i, (evt, orig_y, color, rank_pos, alpha) in enumerate(label_positions):
    ax_a.annotate(
        EVENT_LABELS_SHORT[evt],
        xy=(2.0, orig_y), xytext=(2.15, adjusted_y[i]),
        fontsize=7.5, va='center', ha='left', color=color,
        fontweight='bold' if rank_pos < 3 else 'normal', alpha=alpha,
        arrowprops=dict(arrowstyle='-', color=color, linewidth=0.4,
                        alpha=0.3) if abs(adjusted_y[i] - orig_y) > 0.02 else None,
    )

ax_a.set_xticks(x_positions)
ax_a.set_xticklabels(metric_labels, fontsize=8)
ax_a.set_xlim(-0.15, 2.9)
ax_a.set_ylim(-0.12, 1.08)
ax_a.set_ylabel('Normalized Score', fontsize=9)
for x in x_positions:
    ax_a.axvline(x, color='#e0e0e0', linewidth=0.5, zorder=0)
ax_a.axhline(0.5, color='#cccccc', linewidth=0.4, linestyle=':', zorder=0)

for m_idx, m in enumerate(metrics):
    ax_a.text(x_positions[m_idx], -0.08,
              f'[{agg[m].min():.2f} -- {agg[m].max():.2f}]',
              fontsize=6, ha='center', va='top', color='#888888')

ax_a.set_title('A  Three Dimensions of Event Impact', fontsize=10,
               fontweight='bold', loc='left', pad=8)
ax_a.spines['bottom'].set_visible(False)
ax_a.tick_params(axis='x', length=0)

# ==========================================================================
# Panel B: Heatmap -- per-frame prevalence ratios
# ==========================================================================

ax_b = fig.add_subplot(gs[0, 1])

pivot_pr = prev_evt.pivot(index='annotation', columns='frame', values='prevalence_ratio')
pivot_sig = prev_evt.pivot(index='annotation', columns='frame', values='p_value_adjusted')

frame_cols = [f for f in FRAME_ORDER if f in pivot_pr.columns]
pivot_pr = pivot_pr.reindex(index=rank_order, columns=frame_cols)
pivot_sig = pivot_sig.reindex(index=rank_order, columns=frame_cols)

log2_pr = np.log2(pivot_pr.clip(lower=0.01).values)
vmax = max(abs(np.nanmin(log2_pr)), abs(np.nanmax(log2_pr)), 0.8)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax_b.imshow(log2_pr, cmap='RdBu_r', norm=norm, aspect='auto',
                 interpolation='nearest')

for i in range(len(rank_order)):
    for j in range(len(frame_cols)):
        pr_val = pivot_pr.iloc[i, j]
        p_val = pivot_sig.iloc[i, j]
        if pd.notna(p_val) and p_val < 0.05:
            ax_b.text(j, i, f'{pr_val:.2f}', ha='center', va='center',
                      fontsize=6.5, fontweight='bold', color='black')
        elif pd.notna(p_val) and p_val < 0.10:
            ax_b.text(j, i, f'{pr_val:.2f}', ha='center', va='center',
                      fontsize=6, color='#555555')

ax_b.set_xticks(range(len(frame_cols)))
ax_b.set_xticklabels([FRAME_LABELS[f] for f in frame_cols], rotation=45,
                      ha='right', fontsize=7.5)
ax_b.set_yticks(range(len(rank_order)))
ax_b.set_yticklabels([EVENT_LABELS_SHORT[e] for e in rank_order], fontsize=7.5)

cbar = fig.colorbar(im, ax=ax_b, shrink=0.75, pad=0.03, aspect=25)
cbar.ax.tick_params(labelsize=6.5)
cbar.ax.text(0.5, 1.06, 'Over-\nrepresented', transform=cbar.ax.transAxes,
             fontsize=5.5, ha='center', va='bottom', color='#B2182B')
cbar.ax.text(0.5, -0.06, 'Under-\nrepresented', transform=cbar.ax.transAxes,
             fontsize=5.5, ha='center', va='top', color='#2166AC')

ax_b.set_title('B  Prevalence Ratio by Frame', fontsize=10,
               fontweight='bold', loc='left', pad=8)

# ==========================================================================
# Panel C: Dot plot -- Effect on Cascade Strength
# ==========================================================================

ax_c = fig.add_subplot(gs[1, 0])

y_pos = np.arange(len(diff_df))

ax_c.axvline(0, color='#aaaaaa', linewidth=0.8, linestyle='-', zorder=1)

for i, row in diff_df.iterrows():
    color = EVENT_COLORS[row['event']]
    sig = row['p_value'] < 0.05
    marker_size = 70 if sig else 45

    ax_c.plot([row['ci_low'], row['ci_high']], [i, i],
              color=color, linewidth=2.0, alpha=0.7, solid_capstyle='round',
              zorder=4)
    ax_c.scatter(row['mean_diff'], i, c=color, s=marker_size, zorder=5,
                 edgecolors='white', linewidth=0.6,
                 marker='D' if sig else 'o')

    label = EVENT_LABELS_SHORT[row['event']]
    sig_marker = ' *' if sig else ''
    ax_c.text(row['ci_high'] + 0.003, i,
              f'{label}  ({row["mean_diff"]:+.3f}){sig_marker}',
              va='center', ha='left', fontsize=7.5, color=color,
              fontweight='bold' if sig else 'normal')

ax_c.set_yticks([])
ax_c.set_xlabel('Difference in Mean Cascade Strength\n(high prevalence group minus low prevalence group)',
                fontsize=8.5)

xlim = ax_c.get_xlim()
ax_c.axvspan(0, max(xlim[1], 0.05), color='#E8F5E9', alpha=0.3, zorder=0)
ax_c.axvspan(min(xlim[0], -0.05), 0, color='#FFEBEE', alpha=0.3, zorder=0)
ax_c.text(0.003, len(diff_df) - 0.3, 'Amplifies', fontsize=7,
          color='#2E7D32', fontstyle='italic', va='top')
ax_c.text(-0.003, len(diff_df) - 0.3, 'Attenuates', fontsize=7,
          color='#C62828', fontstyle='italic', va='top', ha='right')

ax_c.set_title('C  Effect on Cascade Strength', fontsize=10,
               fontweight='bold', loc='left', pad=8)

# ==========================================================================
# Panel D: Composite ranking -- horizontal bar with decomposition
# ==========================================================================

ax_d = fig.add_subplot(gs[1, 1])

y_pos_d = np.arange(len(rank_order))
bar_height_d = 0.65

pr_contrib = agg.loc[rank_order, 'mean_pr_norm'] * 0.35
surge_contrib = agg.loc[rank_order, 'mean_surge_norm'] * 0.30
rho_contrib = agg.loc[rank_order, 'mean_rho_norm'] * 0.35

colors_stack = ['#2166AC', '#4393C3', '#92C5DE']

ax_d.barh(y_pos_d, pr_contrib.values, height=bar_height_d,
          color=colors_stack[0], edgecolor='white', linewidth=0.3,
          label='Over-representation (35%)')
ax_d.barh(y_pos_d, surge_contrib.values, left=pr_contrib.values,
          height=bar_height_d, color=colors_stack[1], edgecolor='white',
          linewidth=0.3, label='Pre-onset signal (30%)')
ax_d.barh(y_pos_d, rho_contrib.values,
          left=(pr_contrib + surge_contrib).values,
          height=bar_height_d, color=colors_stack[2], edgecolor='white',
          linewidth=0.3, label='Amplification (35%)')

for i, evt in enumerate(rank_order):
    total = agg.loc[evt, 'composite']
    n_s = agg.loc[evt, 'n_sig']
    ax_d.text(total + 0.015, i, f'{total:.2f}', va='center', ha='left',
              fontsize=7, fontweight='bold', color='#333333')
    if n_s > 0:
        ax_d.text(total + 0.08, i + 0.01, f'({int(n_s)}*)',
                  va='center', ha='left', fontsize=5.5, color='#888888')

ax_d.set_yticks(y_pos_d)
ax_d.set_yticklabels([EVENT_LABELS_SHORT[e] for e in rank_order], fontsize=7.5)
ax_d.set_xlabel('Composite Impact Score', fontsize=8.5)
ax_d.set_xlim(0, 1.18)
ax_d.invert_yaxis()

ax_d.legend(loc='lower right', fontsize=6.5, framealpha=0.9,
            edgecolor='#cccccc', handlelength=1.2, handleheight=0.8)

ax_d.set_title('D  Composite Ranking', fontsize=10,
               fontweight='bold', loc='left', pad=8)

for i, evt in enumerate(rank_order):
    ax_d.text(-0.02, i, f'#{i+1}', va='center', ha='right',
              fontsize=7, fontweight='bold',
              color=EVENT_COLORS[evt], transform=ax_d.get_yaxis_transform())

# -- Title -----------------------------------------------------------------

fig.suptitle(
    'Which Events Drive Media Cascades?\n'
    'Canadian Climate Change Coverage, 2018  (n = 40 cascades, 9,754 articles)',
    fontsize=13, fontweight='bold', y=0.98
)

# -- Captions below the figure (fig.text in figure coordinates) ------------

cap_style = dict(fontsize=6.8, color='#555555', style='italic',
                 linespacing=1.4, va='top', ha='left',
                 transform=fig.transFigure,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAFAF5',
                           edgecolor='#DDDDDD', linewidth=0.4))

fig.text(0.03, -0.02,
    f'A: Each line tracks one event type across 3 metrics. Higher = stronger association with cascades. '
    f'For example, {top_pr_label} leads on prevalence ratio, {top_surge_label} on pre-onset surge, '
    f'and {top_rho_label} on strength correlation -- no single event dominates all three.',
    **cap_style)

fig.text(0.03, -0.05,
    f'B: Red = over-represented during cascades; Blue = under-represented. Bold = significant (p < 0.05, BH-corrected). '
    f'For example, {best_pr_evt} in {best_pr_frame} cascades has PR = {best_pr_val:.2f} ({(best_pr_val-1)*100:.0f}% more frequent).',
    **cap_style)

fig.text(0.03, -0.08,
    f'C: Cascades split at median event prevalence. Dot = difference in mean cascade score (high minus low group). '
    f'Bars = 95% CI. For example, {top_c_label} shows +{top_c_diff:.3f}: cascades rich in {top_c_label.lower()} events are stronger.',
    **cap_style)

fig.text(0.03, -0.11,
    f'D: Composite score = over-representation (35%) + pre-onset surge (30%) + amplification (35%). '
    f'For example, {top_d_label} ranks #1 ({top_d_score:.2f}) while {bot_d_label} ranks last ({bot_d_score:.2f}).',
    **cap_style)

# Method footnote
fig.text(0.03, -0.15,
         'PR = P(event | cascade) / P(event | baseline).  '
         'Surge = 7-day pre-onset rate / 30-day baseline.  '
         'Strength = Spearman rho (event prevalence vs total_score).  '
         'All p-values BH-corrected.',
         fontsize=6, color='#888888', style='italic',
         transform=fig.transFigure)

# -- Save ------------------------------------------------------------------

path = OUTDIR / 'fig_event_impact_composite.png'
fig.savefig(path)
fig.savefig(OUTDIR / 'fig_event_impact_composite.pdf')
plt.close(fig)
print(f'Saved: {path}')
print(f'Saved: {OUTDIR / "fig_event_impact_composite.pdf"}')
