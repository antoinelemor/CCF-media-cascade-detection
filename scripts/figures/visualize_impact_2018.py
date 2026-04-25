#!/usr/bin/env python3
# DEPRECATED: Uses legacy EventImpactAnalyzer output format (prevalence ratios).
# Kept for backward compatibility with existing pickle/parquet results.
# New analyses should use UnifiedImpactAnalyzer from cascade_detector.analysis.
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
visualize_impact_2018.py

MAIN OBJECTIVE:
---------------
Generate publication-quality (SOTA) figures from the 2018 impact analysis.

6 figures:
  1. Forest plot — Prevalence ratios with 95% CI (significant annotations)
  2. Heatmap — Prevalence ratios across frames x annotations
  3. Volcano plot — Effect size vs statistical significance
  4. Pre-onset surge lollipop chart
  5. Strength correlation dot plot
  6. Summary dashboard — multi-metric evidence panel

Each figure includes a pedagogical annotation box explaining what it shows
and how to interpret it.

Style: Nature/Science conventions, colorblind-safe palette, LaTeX-ready.

Author:
-------
Antoine Lemor
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
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm

# ============================================================================
# Style
# ============================================================================

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.5,
})

# Colorblind-safe palette (Wong 2011, Nature Methods)
COLORS = {
    'event': '#0072B2',       # blue
    'messenger': '#D55E00',   # vermillion
    'solution': '#009E73',    # green
    'significant': '#CC79A7', # reddish purple
    'ns': '#999999',          # grey
}

TYPE_MARKERS = {
    'event': 'o',
    'messenger': 's',
    'solution': 'D',
}

# Human-readable annotation labels
LABEL_MAP = {
    'evt_weather': 'Weather events',
    'evt_meeting': 'Institutional meetings',
    'evt_publication': 'Publications',
    'evt_election': 'Elections',
    'evt_policy': 'Policy changes',
    'evt_judiciary': 'Judicial events',
    'evt_cultural': 'Cultural events',
    'evt_protest': 'Protests',
    'msg_health': 'Health experts',
    'msg_economic': 'Economic experts',
    'msg_security': 'Security experts',
    'msg_legal': 'Legal experts',
    'msg_cultural': 'Cultural voices',
    'msg_scientist': 'Scientists',
    'msg_social': 'Social scientists',
    'msg_activist': 'Activists',
    'msg_official': 'Public officials',
    'sol_mitigation': 'Mitigation solutions',
    'sol_adaptation': 'Adaptation solutions',
}

FRAME_ORDER = ['Cult', 'Eco', 'Envt', 'Pbh', 'Just', 'Pol', 'Sci', 'Secu']
FRAME_LABELS = {
    'Cult': 'Cultural', 'Eco': 'Economic', 'Envt': 'Environmental',
    'Pbh': 'Health', 'Just': 'Justice', 'Pol': 'Political',
    'Sci': 'Scientific', 'Secu': 'Security',
}

OUTDIR = Path('results/figures_2018')
OUTDIR.mkdir(parents=True, exist_ok=True)


def _label(ann: str) -> str:
    return LABEL_MAP.get(ann, ann)


def _frame_label(f: str) -> str:
    return FRAME_LABELS.get(f, f)


def _add_interpretation_box(ax, text, loc='lower left', fontsize=7, width=None):
    """Add a pedagogical interpretation box to an axes."""
    positions = {
        'lower left':   (0.02, 0.02, 'left', 'bottom'),
        'lower right':  (0.98, 0.02, 'right', 'bottom'),
        'upper left':   (0.02, 0.98, 'left', 'top'),
        'upper right':  (0.98, 0.98, 'right', 'top'),
        'lower center': (0.50, 0.02, 'center', 'bottom'),
        'upper center': (0.50, 0.98, 'center', 'top'),
    }
    x, y, ha, va = positions.get(loc, positions['lower left'])

    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fontsize, va=va, ha=ha, color='#444444',
            style='italic', linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F0',
                      edgecolor='#CCCCCC', linewidth=0.5, alpha=0.92))


def _footnote(fig, text):
    """Add a method footnote at the bottom of the figure."""
    fig.text(0.02, -0.02, text, fontsize=6.5, color='#666666',
             style='italic', transform=fig.transFigure, va='top')


# ============================================================================
# Load data
# ============================================================================

print("Loading 2018 results...")
with open('results/test_2018_results.pkl', 'rb') as f:
    results = pickle.load(f)

impact = results.event_impact
prev_df = impact['prevalence_ratios'].copy()
surge_df = impact['pre_onset_surge'].copy()
corr_df = impact['strength_correlations'].copy()
summary_df = impact['summary'].copy()

print(f"  Prevalence ratios: {len(prev_df)} rows")
print(f"  Pre-onset surge:   {len(surge_df)} rows")
print(f"  Strength corr:     {len(corr_df)} rows")
print(f"  Summary:           {len(summary_df)} rows")

# ============================================================================
# Figure 1: Forest Plot — Prevalence Ratios with 95% CI
# ============================================================================

def fig1_forest_plot():
    """Forest plot of odds ratios with 95% CI for significant annotations."""
    df = prev_df.copy()
    df['label'] = df['annotation'].map(_label) + '  (' + df['frame'].map(_frame_label) + ')'

    df = df.dropna(subset=['odds_ratio', 'ci_low', 'ci_high'])
    sig = df[df['p_value_adjusted'] < 0.10].copy()
    if sig.empty:
        sig = df.nsmallest(20, 'p_value_adjusted').copy()
    sig = sig.sort_values('odds_ratio', ascending=True).tail(30)

    fig, ax = plt.subplots(figsize=(8, max(5, len(sig) * 0.30)))

    for i, (_, row) in enumerate(sig.iterrows()):
        color = COLORS[row['type']]
        marker = TYPE_MARKERS[row['type']]
        alpha = 1.0 if row['p_value_adjusted'] < 0.05 else 0.5

        ax.plot([row['ci_low'], row['ci_high']], [i, i],
                color=color, linewidth=1.5, alpha=alpha, solid_capstyle='round')
        ax.scatter(row['odds_ratio'], i, color=color, marker=marker,
                   s=40, zorder=3, alpha=alpha, edgecolors='white', linewidth=0.3)

    ax.axvline(x=1.0, color='#333333', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.set_yticks(np.arange(len(sig)))
    ax.set_yticklabels(sig['label'].values, fontsize=7.5)
    ax.set_xlabel('Odds Ratio (95% CI)')
    ax.set_title('Cascade-Associated Events & Messengers\n'
                 '(Fisher exact test, Benjamini-Hochberg adjusted)', fontsize=10)
    ax.set_xlim(left=0)

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['event'],
               markersize=6, label='Events'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['messenger'],
               markersize=6, label='Messengers'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['solution'],
               markersize=6, label='Solutions'),
        Line2D([0], [0], color='#333333', linewidth=0.8, linestyle='--',
               label='Null (OR = 1)', alpha=0.5),
    ]
    ax.legend(handles=handles, loc='lower right', framealpha=0.9,
              edgecolor='#cccccc', fontsize=7.5)

    _add_interpretation_box(ax,
        'How to read: Each point is an odds ratio (OR) comparing\n'
        'how frequently an event/messenger appears during cascade\n'
        'periods vs baseline. OR > 1 (right of dashed line) = more\n'
        'frequent during cascades. Horizontal lines show the 95%\n'
        'confidence interval. Full opacity = p < 0.05; faded = p < 0.10.',
        loc='upper left', fontsize=6.5)

    _footnote(fig, 'Method: Fisher exact test on 2x2 contingency table '
              '(cascade vs baseline x event present vs absent). '
              'CI: Woolf log method. Multiple testing: BH correction.')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig1_forest_plot.png')
    fig.savefig(OUTDIR / 'fig1_forest_plot.pdf')
    plt.close(fig)
    print(f"  Fig 1 saved")


# ============================================================================
# Figure 2: Heatmap — Prevalence Ratios x Frames
# ============================================================================

def fig2_heatmap():
    """Heatmap of log2(prevalence ratio) across frames and annotations."""
    df = prev_df.copy()

    pivot = df.pivot_table(index='annotation', columns='frame', values='prevalence_ratio')
    frame_cols = [f for f in FRAME_ORDER if f in pivot.columns]
    pivot = pivot[frame_cols]

    log2_pivot = np.log2(pivot.clip(lower=0.01))

    ann_order = (
        sorted([a for a in pivot.index if a.startswith('evt_')]) +
        sorted([a for a in pivot.index if a.startswith('msg_')]) +
        sorted([a for a in pivot.index if a.startswith('sol_')])
    )
    log2_pivot = log2_pivot.reindex(ann_order)
    pivot = pivot.reindex(ann_order)

    sig_pivot = df.pivot_table(
        index='annotation', columns='frame', values='p_value_adjusted'
    ).reindex(index=ann_order, columns=frame_cols)

    fig, ax = plt.subplots(figsize=(8, max(6, len(ann_order) * 0.35)))

    vmax = max(abs(log2_pivot.min().min()), abs(log2_pivot.max().max()), 1.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(log2_pivot.values, cmap='RdBu_r', norm=norm,
                   aspect='auto', interpolation='nearest')

    # Show PR value in cells: bold for p < 0.05, normal for p < 0.10
    for i in range(len(ann_order)):
        for j in range(len(frame_cols)):
            pr_val = pivot.iloc[i, j]
            p = sig_pivot.iloc[i, j]
            if pd.notna(p) and p < 0.05:
                ax.text(j, i, f'{pr_val:.2f}', ha='center', va='center',
                        fontsize=6.5, fontweight='bold', color='black')
            elif pd.notna(p) and p < 0.10:
                ax.text(j, i, f'{pr_val:.2f}', ha='center', va='center',
                        fontsize=6, color='#555555')

    ax.set_xticks(range(len(frame_cols)))
    ax.set_xticklabels([_frame_label(f) for f in frame_cols], rotation=45, ha='right')
    ax.set_yticks(range(len(ann_order)))
    ax.set_yticklabels([_label(a) for a in ann_order], fontsize=7.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('log$_2$(Prevalence Ratio)', fontsize=8)
    cbar.ax.text(0.5, 1.04, 'Over-represented\nin cascades',
                 transform=cbar.ax.transAxes, fontsize=5.5,
                 ha='center', va='bottom', color='#B2182B')
    cbar.ax.text(0.5, -0.04, 'Under-represented\nin cascades',
                 transform=cbar.ax.transAxes, fontsize=5.5,
                 ha='center', va='top', color='#2166AC')

    evt_count = sum(1 for a in ann_order if a.startswith('evt_'))
    msg_count = sum(1 for a in ann_order if a.startswith('msg_'))
    ax.axhline(y=evt_count - 0.5, color='white', linewidth=1.5)
    ax.axhline(y=evt_count + msg_count - 0.5, color='white', linewidth=1.5)

    ax.set_title('Prevalence Ratios: Cascade vs Baseline Periods', fontsize=10)

    # Find a concrete example for the caption
    best = df.sort_values('prevalence_ratio', ascending=False).iloc[0]
    best_label = _label(best['annotation'])
    best_frame = _frame_label(best['frame'])
    best_val = best['prevalence_ratio']

    fig.tight_layout()

    # Caption below the figure
    fig.text(0.03, -0.01,
        f'Each cell shows whether an annotation (row) appears more or less '
        f'often during cascades of a given frame (column). '
        f'Red = over-represented; Blue = under-represented. '
        f'Bold values = significant (p < 0.05, BH-corrected); '
        f'light values = marginal (p < 0.10); empty = not significant. '
        f'For example, {best_label} in {best_frame} cascades has '
        f'PR = {best_val:.2f}, meaning it is {(best_val-1)*100:.0f}% more frequent than baseline.',
        fontsize=6.5, color='#555555', style='italic',
        transform=fig.transFigure, va='top', wrap=True)

    fig.text(0.03, -0.05,
        'PR = P(annotation | cascade articles) / P(annotation | baseline articles). '
        'White dividers separate events, messengers, and solutions.',
        fontsize=6, color='#888888', style='italic',
        transform=fig.transFigure, va='top')

    fig.savefig(OUTDIR / 'fig2_heatmap.png')
    fig.savefig(OUTDIR / 'fig2_heatmap.pdf')
    plt.close(fig)
    print(f"  Fig 2 saved")


# ============================================================================
# Figure 3: Volcano Plot
# ============================================================================

def fig3_volcano():
    """Volcano plot: effect size (log2 PR) vs significance (-log10 p)."""
    from adjustText import adjust_text

    df = prev_df.copy()
    df['log2_pr'] = np.log2(df['prevalence_ratio'].clip(lower=0.01))
    df['neg_log10_p'] = -np.log10(df['p_value_adjusted'].clip(lower=1e-20))

    fig, ax = plt.subplots(figsize=(9, 6.5))

    for ann_type, marker in TYPE_MARKERS.items():
        subset = df[df['type'] == ann_type]
        sig = subset[subset['p_value_adjusted'] < 0.05]
        ns = subset[subset['p_value_adjusted'] >= 0.05]

        ax.scatter(ns['log2_pr'], ns['neg_log10_p'],
                   c=COLORS['ns'], marker=marker, s=25, alpha=0.35,
                   edgecolors='white', linewidth=0.3)
        ax.scatter(sig['log2_pr'], sig['neg_log10_p'],
                   c=COLORS[ann_type], marker=marker, s=45, alpha=0.85,
                   edgecolors='white', linewidth=0.3, zorder=3)

    sig_df = df[df['p_value_adjusted'] < 0.05].copy()
    sig_df['short_label'] = sig_df.apply(
        lambda r: f"{_label(r['annotation'])} ({_frame_label(r['frame'])})", axis=1
    )

    texts = []
    for _, row in sig_df.iterrows():
        texts.append(ax.text(
            row['log2_pr'], row['neg_log10_p'], row['short_label'],
            fontsize=6, alpha=0.9, color='#222222',
        ))

    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle='-', color='#888888', linewidth=0.4, alpha=0.6),
        expand=(1.8, 2.0), force_text=(1.5, 2.0), force_points=(0.8, 1.0),
        max_move=50, only_move='xy', ensure_inside_axes=True,
    )

    if (df['p_value_adjusted'] > 0).any():
        ax.axhline(-np.log10(0.05), color='#CC79A7', linewidth=0.7,
                   linestyle='--', alpha=0.6)
    ax.axvline(0, color='#333333', linewidth=0.5, linestyle='-', alpha=0.3)

    ax.set_xlabel('log$_2$(Prevalence Ratio)')
    ax.set_ylabel('$-$log$_{10}$(p$_{adj}$)')
    ax.set_title('Volcano Plot: Event & Messenger Cascade Association', fontsize=10)

    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.5, xlim[1] + 0.8)

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['event'],
               markersize=6, label='Events'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['messenger'],
               markersize=6, label='Messengers'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['solution'],
               markersize=6, label='Solutions'),
        Line2D([0], [0], color='#CC79A7', linewidth=0.7, linestyle='--',
               label='p$_{adj}$ = 0.05'),
    ]
    ax.legend(handles=handles, loc='upper left', framealpha=0.9,
              edgecolor='#cccccc')

    _add_interpretation_box(ax,
        'How to read: Each point is a (annotation, frame) pair.\n'
        'X-axis = effect size: how much more/less frequent during\n'
        'cascades (right = over-represented, left = under-represented).\n'
        'Y-axis = statistical significance: higher = more significant.\n'
        'Colored points above the dashed line are significant (p < 0.05).\n'
        'Grey points below are not statistically significant.',
        loc='lower right', fontsize=6.5)

    _footnote(fig, 'Each point: one (annotation, frame) pair. '
              'Effect size: log2(PR). Significance: -log10(BH-adjusted p-value). '
              'Only significant points are labeled.')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig3_volcano.png')
    fig.savefig(OUTDIR / 'fig3_volcano.pdf')
    plt.close(fig)
    print(f"  Fig 3 saved")


# ============================================================================
# Figure 4: Pre-Onset Surge Lollipop Chart
# ============================================================================

def fig4_surge_lollipop():
    """Lollipop chart of median pre-onset surge ratios."""
    df = surge_df.copy()
    df = df.dropna(subset=['median_surge'])
    df['label'] = df['annotation'].map(_label) + '  (' + df['frame'].map(_frame_label) + ')'

    df = df[df['n_cascades'] >= 2]
    df = df.sort_values('median_surge', ascending=True)

    n_show = min(15, len(df) // 2)
    if n_show < 3:
        show = df
    else:
        show = pd.concat([df.head(n_show), df.tail(n_show)]).drop_duplicates()

    fig, ax = plt.subplots(figsize=(7.5, max(5, len(show) * 0.28)))

    for i, (_, row) in enumerate(show.iterrows()):
        color = COLORS[row['type']]
        alpha = 1.0 if pd.notna(row.get('p_value_adjusted')) and row['p_value_adjusted'] < 0.05 else 0.5

        ax.hlines(i, 1.0, row['median_surge'], color=color, linewidth=1.5, alpha=alpha)
        ax.scatter(row['median_surge'], i, color=color,
                   marker=TYPE_MARKERS[row['type']], s=35, zorder=3,
                   alpha=alpha, edgecolors='white', linewidth=0.3)

    ax.axvline(1.0, color='#333333', linewidth=0.8, linestyle='--', alpha=0.4)

    ax.set_yticks(np.arange(len(show)))
    ax.set_yticklabels(show['label'].values, fontsize=7)
    ax.set_xlabel('Median Surge Ratio (7-day pre-onset / 30-day baseline)')
    ax.set_title('Pre-Onset Surge: Activity Before Cascade Onset\n'
                 '(Wilcoxon signed-rank test, BH-adjusted)', fontsize=10)

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['event'],
               markersize=6, label='Events'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['messenger'],
               markersize=6, label='Messengers'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['solution'],
               markersize=6, label='Solutions'),
        Line2D([0], [0], color='#333333', linewidth=0.8, linestyle='--',
               label='No change (ratio = 1)', alpha=0.4),
    ]
    ax.legend(handles=handles, loc='lower right', framealpha=0.9,
              edgecolor='#cccccc', fontsize=7.5)

    _add_interpretation_box(ax,
        'How to read: Each lollipop shows whether an annotation\n'
        'appears at elevated rates in the 7 days before a cascade\n'
        'starts, compared to its rate in the preceding 30 days.\n'
        'Right of dashed line (ratio > 1) = pre-cascade surge.\n'
        'Left (ratio < 1) = decline before cascade onset.\n'
        'Full opacity = significant (p < 0.05); faded = not significant.',
        loc='upper left', fontsize=6.5)

    _footnote(fig, 'Surge = rate(7-day pre-onset window) / rate(30-day baseline). '
              'Median across all cascades of each frame. '
              'Wilcoxon test: is the surge distribution systematically > 1?')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig4_surge_lollipop.png')
    fig.savefig(OUTDIR / 'fig4_surge_lollipop.pdf')
    plt.close(fig)
    print(f"  Fig 4 saved")


# ============================================================================
# Figure 5: Strength Correlation Dot Plot
# ============================================================================

def fig5_correlation_dots():
    """Dot plot of Spearman rho between annotation prevalence and cascade score."""
    df = corr_df.copy()
    df = df.dropna(subset=['spearman_rho'])
    df['label'] = df['annotation'].map(_label) + '  (' + df['frame'].map(_frame_label) + ')'

    df['abs_rho'] = df['spearman_rho'].abs()
    df = df.sort_values('abs_rho', ascending=True).tail(30)

    fig, ax = plt.subplots(figsize=(7.5, max(5, len(df) * 0.28)))

    for i, (_, row) in enumerate(df.iterrows()):
        color = COLORS[row['type']]
        alpha = 1.0 if pd.notna(row.get('p_value_adjusted')) and row['p_value_adjusted'] < 0.05 else 0.45

        ax.scatter(row['spearman_rho'], i, color=color,
                   marker=TYPE_MARKERS[row['type']], s=45, alpha=alpha,
                   edgecolors='white', linewidth=0.3, zorder=3)

    ax.axvline(0, color='#333333', linewidth=0.8, linestyle='--', alpha=0.4)

    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['label'].values, fontsize=7)
    ax.set_xlabel('Spearman $\\rho$ (prevalence vs cascade strength)')
    ax.set_title('Strength Correlation:\n'
                 'Do Annotations Predict Cascade Intensity?', fontsize=10)
    ax.set_xlim(-1.15, 1.15)

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['event'],
               markersize=6, label='Events'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['messenger'],
               markersize=6, label='Messengers'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['solution'],
               markersize=6, label='Solutions'),
        Line2D([0], [0], color='#333333', linewidth=0.8, linestyle='--',
               label='No correlation (ρ = 0)', alpha=0.4),
    ]
    ax.legend(handles=handles, loc='lower left', framealpha=0.9,
              edgecolor='#cccccc', fontsize=7.5)

    _add_interpretation_box(ax,
        'How to read: Each point shows the Spearman correlation\n'
        'between an annotation\'s prevalence across cascades and\n'
        'cascade strength (total_score). Right (ρ > 0) = when this\n'
        'annotation is more prevalent, cascades are stronger.\n'
        'Left (ρ < 0) = more prevalent in weaker cascades.\n'
        'Full opacity = significant (p < 0.05); faded = not significant.',
        loc='upper right', fontsize=6.5)

    _footnote(fig, 'Spearman rank correlation between event/messenger prevalence '
              '(proportion of articles) and cascade total_score, computed across '
              'all cascades of each frame. Minimum 3 cascades per frame.')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig5_correlation_dots.png')
    fig.savefig(OUTDIR / 'fig5_correlation_dots.pdf')
    plt.close(fig)
    print(f"  Fig 5 saved")


# ============================================================================
# Figure 6: Multi-Metric Evidence Panel
# ============================================================================

def fig6_evidence_panel():
    """Three-panel summary: combined evidence across all 3 metrics."""
    df = summary_df.copy()
    df['label'] = df['annotation'].map(_label)

    agg = df.groupby(['annotation', 'type']).agg(
        max_pr=('prevalence_ratio', lambda x: x.max() if x.notna().any() else np.nan),
        max_surge=('median_surge', lambda x: x.max() if x.notna().any() else np.nan),
        max_abs_rho=('spearman_rho', lambda x: x.abs().max() if x.notna().any() else np.nan),
        total_sig=('n_significant_metrics', 'sum'),
        n_frames=('n_significant_metrics', 'count'),
    ).reset_index()

    agg['label'] = agg['annotation'].map(_label)
    agg = agg.sort_values('total_sig', ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, max(5.5, len(agg) * 0.32)),
                             sharey=True, gridspec_kw={'wspace': 0.12},
                             layout='constrained')

    y_pos = np.arange(len(agg))

    # Panel A: Max Prevalence Ratio
    ax = axes[0]
    for i, (_, row) in enumerate(agg.iterrows()):
        color = COLORS[row['type']]
        val = row['max_pr']
        if pd.isna(val):
            continue
        ax.barh(i, np.log2(max(val, 0.01)), color=color, alpha=0.7, height=0.6,
                edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='#333333', linewidth=0.6, linestyle='-', alpha=0.3)
    ax.set_xlabel('Max log$_2$(PR) across frames')
    ax.set_title('A. Over-Representation', fontsize=9, fontweight='bold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(agg['label'].values, fontsize=7.5)

    _add_interpretation_box(ax,
        'Max prevalence ratio (best\n'
        'frame for this annotation).\n'
        'Right of 0 = over-represented\n'
        'during cascades.',
        loc='lower right', fontsize=6)

    # Panel B: Max Surge
    ax = axes[1]
    for i, (_, row) in enumerate(agg.iterrows()):
        color = COLORS[row['type']]
        val = row['max_surge']
        if pd.isna(val):
            continue
        ax.barh(i, val - 1.0, left=1.0, color=color, alpha=0.7, height=0.6,
                edgecolor='white', linewidth=0.3)
    ax.axvline(1.0, color='#333333', linewidth=0.6, linestyle='--', alpha=0.3)
    ax.set_xlabel('Max median surge across frames')
    ax.set_title('B. Pre-Onset Signal', fontsize=9, fontweight='bold')

    _add_interpretation_box(ax,
        'Max pre-onset surge (best\n'
        'frame). Right of 1.0 = more\n'
        'active before cascade onset\n'
        'than during baseline.',
        loc='lower right', fontsize=6)

    # Panel C: Max |rho|
    ax = axes[2]
    for i, (_, row) in enumerate(agg.iterrows()):
        color = COLORS[row['type']]
        val = row['max_abs_rho']
        if pd.isna(val):
            continue
        ax.barh(i, val, color=color, alpha=0.7, height=0.6,
                edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Max |Spearman $\\rho$| across frames')
    ax.set_title('C. Amplification', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 1.15)

    _add_interpretation_box(ax,
        'Max |correlation| between\n'
        'prevalence and cascade strength.\n'
        'Higher = stronger association\n'
        'with cascade intensity.',
        loc='lower right', fontsize=6)

    # Shared legend
    handles = [
        mpatches.Patch(color=COLORS['event'], alpha=0.7, label='Events'),
        mpatches.Patch(color=COLORS['messenger'], alpha=0.7, label='Messengers'),
        mpatches.Patch(color=COLORS['solution'], alpha=0.7, label='Solutions'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=3, framealpha=0.9,
               edgecolor='#cccccc', fontsize=8, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle(
        'Multi-Metric Evidence: Event & Messenger Impact on Cascades (2018)\n'
        'Each bar shows the strongest signal across all 8 frames for each annotation',
        fontsize=11, fontweight='bold', y=1.03)

    fig.savefig(OUTDIR / 'fig6_evidence_panel.png')
    fig.savefig(OUTDIR / 'fig6_evidence_panel.pdf')
    plt.close(fig)
    print(f"  Fig 6 saved")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\nGenerating figures...")
    fig1_forest_plot()
    fig2_heatmap()
    fig3_volcano()
    fig4_surge_lollipop()
    fig5_correlation_dots()
    fig6_evidence_panel()
    print(f"\nAll figures saved to {OUTDIR}/")
