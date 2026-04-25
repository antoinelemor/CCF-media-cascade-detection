#!/usr/bin/env python3
"""Generate publication-quality figures for LaTeX report (1978-2024)."""

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#cccccc',
    'grid.linestyle': '-',
    'axes.axisbelow': True,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Colorblind-friendly palette (Okabe-Ito inspired)
C_STRONG = '#D55E00'    # vermillion / red
C_MODERATE = '#E69F00'  # orange
C_WEAK = '#F0E442'      # yellow
C_NOT = '#999999'        # gray
C_BLUE = '#0072B2'
C_GREEN = '#009E73'
C_PINK = '#CC79A7'
C_SKY = '#56B4E9'

FRAME_COLORS = {
    'Cult': '#0072B2',
    'Eco': '#E69F00',
    'Envt': '#009E73',
    'Pbh': '#CC79A7',
    'Just': '#D55E00',
    'Pol': '#56B4E9',
    'Sci': '#F0E442',
    'Secu': '#999999',
}

BASE = Path('/Users/antoine/Documents/GitHub/CCF-media-cascade-detection/results/production')
OUTDIR = BASE / 'figures'
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_summary():
    with open(BASE / 'cross_year_summary.json') as f:
        return json.load(f)


def load_cascades():
    return pd.read_parquet(BASE / 'cross_year_cascades.parquet')


def load_paradigm_timeline():
    return pd.read_parquet(BASE / 'cross_year_paradigm_timeline.parquet')


def load_shifts():
    with open(BASE / 'cross_year_paradigm_shifts.json') as f:
        return json.load(f)


def load_stabsel_across_years(filename):
    """Load a stabsel parquet from each year's impact_analysis/ and concat."""
    frames = []
    for y in range(1978, 2025):
        p = BASE / str(y) / 'impact_analysis' / filename
        if p.exists():
            df = pd.read_parquet(p)
            df['year'] = y
            frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Cascade volume and classification over time
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_cascade_volume():
    summary = load_summary()
    by_year = summary['by_year']

    years = sorted(int(y) for y in by_year.keys())
    classes = ['strong_cascade', 'moderate_cascade', 'weak_cascade', 'not_cascade']
    labels_map = {'strong_cascade': 'Strong', 'moderate_cascade': 'Moderate',
                  'weak_cascade': 'Weak', 'not_cascade': 'Not cascade'}
    colors_map = {'strong_cascade': C_STRONG, 'moderate_cascade': C_MODERATE,
                  'weak_cascade': C_WEAK, 'not_cascade': C_NOT}

    data = {c: [] for c in classes}
    articles = []
    plot_years = []

    for y in years:
        bc = by_year[str(y)].get('by_classification', {})
        n_total = sum(bc.values()) if bc else 0
        if n_total == 0:
            continue
        plot_years.append(y)
        for c in classes:
            data[c].append(bc.get(c, 0))
        articles.append(by_year[str(y)].get('n_articles', 0))

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(plot_years))
    width = 0.8

    bottom = np.zeros(len(plot_years))
    for c in classes:
        vals = np.array(data[c])
        ax1.bar(x, vals, width, bottom=bottom, color=colors_map[c],
                label=labels_map[c], edgecolor='white', linewidth=0.3)
        bottom += vals

    ax1.set_ylabel('Number of cascades')
    ax1.set_xticks(x[::5])
    ax1.set_xticklabels([str(y) for y in plot_years[::5]], rotation=45, ha='right')

    ax2 = ax1.twinx()
    ax2.plot(x, articles, color=C_BLUE, linewidth=1.5, alpha=0.8, label='Articles')
    ax2.set_ylabel('Number of articles', color=C_BLUE)
    ax2.tick_params(axis='y', labelcolor=C_BLUE)

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc='upper left', fontsize=7, framealpha=0.9)

    fig.savefig(OUTDIR / 'fig1_cascade_volume.pdf')
    plt.close(fig)
    print('  fig1 done')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Score distribution by decade
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_score_distribution():
    df = load_cascades()
    df['decade'] = (df['year'] // 10) * 10
    # Filter to decades with enough data
    decades = sorted(df['decade'].unique())
    decades = [d for d in decades if d >= 1980]

    decade_labels = {d: f'{d}s' for d in decades}

    fig, ax = plt.subplots(figsize=(7, 4.5))

    positions = []
    data_groups = []
    tick_labels = []

    for i, d in enumerate(decades):
        subset = df[df['decade'] == d]['total_score'].dropna()
        if len(subset) < 5:
            continue
        data_groups.append(subset.values)
        positions.append(i)
        tick_labels.append(decade_labels[d])

    parts = ax.violinplot(data_groups, positions=positions, showmedians=True,
                          showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(C_SKY)
        pc.set_alpha(0.6)
    parts['cmedians'].set_color('black')

    # Threshold lines
    thresholds = [(0.25, 'Weak', '#888888'), (0.40, 'Moderate', C_MODERATE),
                  (0.65, 'Strong', C_STRONG)]
    for val, label, color in thresholds:
        ax.axhline(val, color=color, linestyle='--', linewidth=1, alpha=0.7)
        ax.text(len(positions) - 0.5, val + 0.01, label, fontsize=7,
                color=color, va='bottom', ha='right')

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('Total score')
    ax.set_xlabel('Decade')

    fig.savefig(OUTDIR / 'fig2_score_distribution.pdf')
    plt.close(fig)
    print('  fig2 done')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Top 20 cascades - score decomposition
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_top20_decomposition():
    df = load_cascades()
    dims = ['score_temporal', 'score_participation', 'score_convergence', 'score_source']
    dim_labels = ['Temporal', 'Participation', 'Convergence', 'Source']
    dim_colors = [C_BLUE, C_GREEN, C_MODERATE, C_PINK]

    top20 = df.nlargest(20, 'total_score').iloc[::-1]  # reverse for horizontal bar
    labels = [f"{row['cascade_id']} ({row['year']})" for _, row in top20.iterrows()]
    # Shorten labels
    labels = [l.replace('_cascade', '') for l in labels]

    fig, ax = plt.subplots(figsize=(7, 6))
    y = np.arange(len(top20))

    left = np.zeros(len(top20))
    for dim, label, color in zip(dims, dim_labels, dim_colors):
        vals = top20[dim].values * 0.25  # weight: each dimension contributes 0.25
        ax.barh(y, vals, left=left, color=color, label=label,
                edgecolor='white', linewidth=0.3, height=0.7)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel('Total score (weighted sum of 4 dimensions)')
    ax.legend(loc='lower right', fontsize=7, framealpha=0.9)
    ax.set_xlim(0, None)

    fig.savefig(OUTDIR / 'fig3_top20_decomposition.pdf')
    plt.close(fig)
    print('  fig3 done')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Paradigm timeline coverage - first date per year
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_paradigm_coverage():
    ptl = load_paradigm_timeline()
    ptl['date'] = pd.to_datetime(ptl['date'])

    first_dates = ptl.groupby('year')['date'].min().reset_index()
    first_dates['day_of_year'] = first_dates['date'].dt.dayofyear

    # "Before" baseline: March 30 = day 89 (roughly 12-week window start)
    baseline_day = 89

    fig, ax = plt.subplots(figsize=(7, 4.5))
    years = first_dates['year'].values
    actual_days = first_dates['day_of_year'].values

    x = np.arange(len(years))
    width = 0.35

    ax.bar(x - width/2, [baseline_day] * len(years), width,
           color=C_NOT, alpha=0.6, label='Before (12-week window)')
    ax.bar(x + width/2, actual_days, width,
           color=C_BLUE, alpha=0.8, label='After (actual first date)')

    ax.set_ylabel('Day of year of first paradigm state')
    ax.set_xticks(x[::5])
    ax.set_xticklabels([str(y) for y in years[::5]], rotation=45, ha='right')
    ax.legend(fontsize=7, framealpha=0.9)
    ax.axhline(31, color='black', linestyle=':', linewidth=0.5, alpha=0.5)  # Feb 1
    ax.text(0, 33, 'Feb 1', fontsize=6, alpha=0.5)

    fig.savefig(OUTDIR / 'fig4_paradigm_coverage.pdf')
    plt.close(fig)
    print('  fig4 done')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Paradigm shift dynamics (2018 example + shift density)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_paradigm_dynamics():
    ptl = load_paradigm_timeline()
    shifts_all = load_shifts()

    frames = ['Cult', 'Eco', 'Envt', 'Pbh', 'Just', 'Pol', 'Sci', 'Secu']
    frame_cols = [f'paradigm_{f}' for f in frames]

    # Top panel: 2018 stacked area
    ptl_2018 = ptl[ptl['year'] == 2018].copy()
    ptl_2018['date'] = pd.to_datetime(ptl_2018['date'])
    ptl_2018 = ptl_2018.sort_values('date')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={'height_ratios': [1.2, 1]})

    dates = ptl_2018['date'].values
    vals = ptl_2018[frame_cols].values.T
    colors = [FRAME_COLORS[f] for f in frames]

    for i, (f, col) in enumerate(zip(frames, frame_cols)):
        ax1.plot(dates, ptl_2018[col].values, color=colors[i], label=f,
                 linewidth=1.0, alpha=0.85)
    ax1.set_ylabel('Frame dominance index')
    ax1.set_xlabel('Date (2018)')
    ax1.legend(loc='upper right', fontsize=6, ncol=4, framealpha=0.9)
    ax1.set_xlim(dates[0], dates[-1])
    ax1.set_ylim(0, None)

    # Bottom panel: shift density per year
    shift_years = [s.get('year', pd.to_datetime(s['shift_date']).year) for s in shifts_all]
    year_counts = pd.Series(shift_years).value_counts().sort_index()
    all_years = range(year_counts.index.min(), year_counts.index.max() + 1)
    counts = [year_counts.get(y, 0) for y in all_years]

    ax2.bar(list(all_years), counts, color=C_BLUE, alpha=0.7, edgecolor='white', linewidth=0.3)
    ax2.set_ylabel('Number of paradigm shifts')
    ax2.set_xlabel('Year')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig5_paradigm_dynamics.pdf')
    plt.close(fig)
    print('  fig5 done')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: StabSel Model validation (R² full vs R² test)
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_stabsel_validation():
    df = load_stabsel_across_years('stabsel_validation.parquet')
    if df.empty:
        print('  fig6 SKIPPED: no stabsel_validation data')
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model, color, marker in [('A', C_BLUE, 'o'), ('B', C_STRONG, 's')]:
        sub = df[df['model'] == model]
        # Filter out extreme negative R² test values for visualization
        sub = sub[(sub['r2_test'] > -1) & (sub['r2_full'] > -1)]
        ax.scatter(sub['r2_full'], sub['r2_test'], c=color, marker=marker,
                   alpha=0.5, s=25, label=f'Model {model}', edgecolors='white',
                   linewidth=0.3)

    # Diagonal
    lims = [0, 1]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5, label='Perfect generalization')
    ax.set_xlim(-0.1, 1.05)
    ax.set_ylim(-0.5, 1.05)
    ax.set_xlabel(r'$R^2$ (full sample)')
    ax.set_ylabel(r'$R^2$ (test set)')
    ax.legend(fontsize=7, framealpha=0.9)

    fig.savefig(OUTDIR / 'fig6_stabsel_validation.pdf')
    plt.close(fig)
    print('  fig6 done')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Event type roles in paradigm (Model A)
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_event_type_roles():
    df = load_stabsel_across_years('stabsel_cluster_dominance.parquet')
    if df.empty:
        print('  fig7 SKIPPED: no stabsel_cluster_dominance data')
        return

    # Extract dominant_type
    roles_of_interest = ['catalyst', 'disruptor', 'inert']
    role_colors = {'catalyst': C_GREEN, 'disruptor': C_STRONG, 'inert': C_NOT}

    # Count by dominant_type × role
    counts = df.groupby(['dominant_type', 'role']).size().unstack(fill_value=0)
    # Keep only known roles
    for r in roles_of_interest:
        if r not in counts.columns:
            counts[r] = 0
    counts = counts[roles_of_interest]
    counts = counts.sort_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(counts))
    width = 0.25

    for i, role in enumerate(roles_of_interest):
        ax.bar(x + i * width, counts[role].values, width,
               color=role_colors[role], label=role.capitalize(),
               edgecolor='white', linewidth=0.3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Count (across all years)')
    ax.legend(fontsize=7, framealpha=0.9)

    fig.savefig(OUTDIR / 'fig7_event_type_roles.pdf')
    plt.close(fig)
    print('  fig7 done')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Cascade-paradigm cross-frame matrix (Model B)
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_cross_frame_matrix():
    df = load_stabsel_across_years('stabsel_cascade_dominance.parquet')
    if df.empty:
        print('  fig8 SKIPPED: no stabsel_cascade_dominance data')
        return

    # Keep only significant pairs (p < 0.10)
    sig = df[df['p_value_hac'] < 0.10]

    frames = ['Cult', 'Eco', 'Envt', 'Pbh', 'Just', 'Pol', 'Sci', 'Secu']
    matrix = pd.DataFrame(0, index=frames, columns=frames)

    for _, row in sig.iterrows():
        cf = row.get('cascade_frame', '')
        tf = row.get('target_frame', '')
        if cf in frames and tf in frames:
            matrix.loc[cf, tf] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(frames)))
    ax.set_xticklabels(frames, rotation=45, ha='right')
    ax.set_yticks(range(len(frames)))
    ax.set_yticklabels(frames)
    ax.set_xlabel('Target frame (paradigm dominance)')
    ax.set_ylabel('Cascade frame')

    # Annotate cells
    for i in range(len(frames)):
        for j in range(len(frames)):
            val = matrix.values[i, j]
            color = 'white' if val > matrix.values.max() * 0.6 else 'black'
            ax.text(j, i, str(int(val)), ha='center', va='center',
                    fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Significant pairs', fontsize=8)

    fig.savefig(OUTDIR / 'fig8_cross_frame_matrix.pdf')
    plt.close(fig)
    print('  fig8 done')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: Paradigm frame volatility
# ═══════════════════════════════════════════════════════════════════════════════
def fig9_frame_volatility():
    shifts_all = load_shifts()
    frames = ['Cult', 'Eco', 'Envt', 'Pbh', 'Just', 'Pol', 'Sci', 'Secu']

    entering = {f: 0 for f in frames}
    exiting = {f: 0 for f in frames}

    for s in shifts_all:
        for f in s.get('entering_frames', []):
            if f in entering:
                entering[f] += 1
        for f in s.get('exiting_frames', []):
            if f in exiting:
                exiting[f] += 1

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(frames))
    width = 0.35

    ax.bar(x - width/2, [entering[f] for f in frames], width,
           color=C_GREEN, label='Entering dominance', edgecolor='white', linewidth=0.3)
    ax.bar(x + width/2, [exiting[f] for f in frames], width,
           color=C_STRONG, label='Exiting dominance', edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(frames)
    ax.set_ylabel('Count (1978-2024)')
    ax.legend(fontsize=7, framealpha=0.9)

    fig.savefig(OUTDIR / 'fig9_frame_volatility.pdf')
    plt.close(fig)
    print('  fig9 done')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Q1 coverage validation
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_q1_coverage():
    casc_df = load_stabsel_across_years('stabsel_cascade_dominance.parquet')
    cascades = load_cascades()

    if casc_df.empty:
        print('  fig10 SKIPPED: no stabsel_cascade_dominance data')
        return

    # Merge to get onset_date
    cascades['onset_date'] = pd.to_datetime(cascades['onset_date'])
    cascades['onset_month'] = cascades['onset_date'].dt.month

    # Merge on cascade_id and year
    merged = casc_df.merge(
        cascades[['cascade_id', 'year', 'onset_month']],
        on=['cascade_id', 'year'],
        how='left'
    )
    # Handle missing cascade_id matches by trying without year
    if merged['onset_month'].isna().sum() > len(merged) * 0.5:
        merged2 = casc_df.merge(
            cascades[['cascade_id', 'onset_month']].drop_duplicates('cascade_id'),
            on='cascade_id', how='left'
        )
        if merged2['onset_month'].isna().sum() < merged['onset_month'].isna().sum():
            merged = merged2

    merged = merged.dropna(subset=['onset_month'])
    merged['quarter'] = np.where(merged['onset_month'] <= 3, 'Q1', 'Q2-Q4')
    merged['significant'] = merged['p_value_hac'] < 0.10

    # Group by decade
    merged['decade'] = (merged['year'] // 10) * 10
    decades = sorted(merged['decade'].unique())
    decades = [d for d in decades if d >= 1980]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(decades))
    width = 0.35

    q1_pcts = []
    q24_pcts = []

    for d in decades:
        sub = merged[merged['decade'] == d]
        q1 = sub[sub['quarter'] == 'Q1']
        q24 = sub[sub['quarter'] == 'Q2-Q4']
        q1_pct = q1['significant'].mean() * 100 if len(q1) > 0 else 0
        q24_pct = q24['significant'].mean() * 100 if len(q24) > 0 else 0
        q1_pcts.append(q1_pct)
        q24_pcts.append(q24_pct)

    ax.bar(x - width/2, q1_pcts, width, color=C_SKY, label='Q1 cascades',
           edgecolor='white', linewidth=0.3)
    ax.bar(x + width/2, q24_pcts, width, color=C_BLUE, label='Q2-Q4 cascades',
           edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{d}s' for d in decades])
    ax.set_ylabel('% significant cascade-dominance pairs')
    ax.set_xlabel('Decade')
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_ylim(0, 100)

    fig.savefig(OUTDIR / 'fig10_q1_coverage.pdf')
    plt.close(fig)
    print('  fig10 done')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.chdir('/Users/antoine/Documents/GitHub/CCF-media-cascade-detection')
    print('Generating publication figures...')

    fig1_cascade_volume()
    fig2_score_distribution()
    fig3_top20_decomposition()
    fig4_paradigm_coverage()
    fig5_paradigm_dynamics()
    fig6_stabsel_validation()
    fig7_event_type_roles()
    fig8_cross_frame_matrix()
    fig9_frame_volatility()
    fig10_q1_coverage()

    print(f'\nAll figures saved to {OUTDIR}/')
