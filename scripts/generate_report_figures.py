#!/usr/bin/env python3
"""Generate all 25 publication-quality figures for the CCF report."""

import json
import os
import warnings
import ast
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path('/Users/antoine/Documents/GitHub/CCF-media-cascade-detection/results/production')
OUT = Path('/Users/antoine/Documents/GitHub/CCF-media-cascade-detection/results/report/figures')
STATS_PATH = Path('/Users/antoine/Documents/GitHub/CCF-media-cascade-detection/results/report/verified_stats.json')
OUT.mkdir(parents=True, exist_ok=True)

# ── Frame colours ──────────────────────────────────────────────────────
FRAME_ORDER = ['Pol', 'Eco', 'Envt', 'Sci', 'Pbh', 'Just', 'Cult', 'Secu']
FRAME_COLORS = {
    'Pol': '#E41A1C', 'Eco': '#377EB8', 'Envt': '#4DAF4A', 'Sci': '#984EA3',
    'Pbh': '#FF7F00', 'Just': '#A65628', 'Cult': '#F781BF', 'Secu': '#999999'
}
FRAME_LABELS = {
    'Pol': 'Political', 'Eco': 'Economic', 'Envt': 'Environmental', 'Sci': 'Scientific',
    'Pbh': 'Public Health', 'Just': 'Justice', 'Cult': 'Cultural', 'Secu': 'Security'
}
CLASS_COLORS = {
    'strong_cascade': '#E41A1C', 'moderate_cascade': '#377EB8',
    'weak_cascade': '#FF7F00', 'not_cascade': '#999999'
}
CLASS_ORDER = ['strong_cascade', 'moderate_cascade', 'weak_cascade', 'not_cascade']
CLASS_LABELS = {'strong_cascade': 'Strong', 'moderate_cascade': 'Moderate',
                'weak_cascade': 'Weak', 'not_cascade': 'Not cascade'}

EVENT_TYPES = ['evt_weather', 'evt_meeting', 'evt_publication', 'evt_legal',
               'evt_protest', 'evt_disaster', 'evt_election', 'evt_policy']
EVENT_LABELS = {
    'evt_weather': 'Weather', 'evt_meeting': 'Meeting', 'evt_publication': 'Publication',
    'evt_legal': 'Legal', 'evt_protest': 'Protest', 'evt_disaster': 'Disaster',
    'evt_election': 'Election', 'evt_policy': 'Policy'
}

DECADE_LABELS = {1980: '1980s', 1990: '1990s', 2000: '2000s', 2010: '2010s', 2020: '2020s'}

# ── Load data ──────────────────────────────────────────────────────────
print("Loading data...")
with open(BASE / 'cross_year_summary.json') as f:
    summary = json.load(f)
with open(STATS_PATH) as f:
    stats = json.load(f)

cascades = pd.read_parquet(BASE / 'cross_year_cascades.parquet')
cascades['onset_date'] = pd.to_datetime(cascades['onset_date'])
cascades['peak_date'] = pd.to_datetime(cascades['peak_date'])
cascades['end_date'] = pd.to_datetime(cascades['end_date'])

paradigm_timeline = pd.read_parquet(BASE / 'cross_year_paradigm_timeline.parquet')
paradigm_timeline['date'] = pd.to_datetime(paradigm_timeline['date'])

with open(BASE / 'cross_year_paradigm_shifts.json') as f:
    shifts = json.load(f)

# Parse sub_indices
def parse_si(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str) and x != '{}':
        try:
            return json.loads(x)
        except:
            try:
                return ast.literal_eval(x)
            except:
                return {}
    return {}

cascades['sub_indices_parsed'] = cascades['sub_indices'].apply(parse_si)

# Load all stabsel data across years
def load_all_parquet(filename):
    dfs = []
    for y in sorted(os.listdir(BASE)):
        if not y.isdigit():
            continue
        fp = BASE / y / 'impact_analysis' / filename
        if fp.exists():
            df = pd.read_parquet(fp)
            df['year'] = int(y)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

print("Loading stabsel data...")
cluster_dom = load_all_parquet('stabsel_cluster_dominance.parquet')
cascade_dom = load_all_parquet('stabsel_cascade_dominance.parquet')
validation = load_all_parquet('stabsel_validation.parquet')
alignment_a = load_all_parquet('stabsel_alignment_a.parquet')
alignment_b = load_all_parquet('stabsel_alignment_b.parquet')

# Normalize cascade_dom roles (old → new)
role_map = {'amplification': 'catalyst', 'destabilisation': 'disruptor', 'dormant': 'inert'}
cascade_dom['role'] = cascade_dom['role'].replace(role_map)

# Assign decades
cascades['decade'] = (cascades['year'] // 10) * 10
cluster_dom['decade'] = (cluster_dom['year'] // 10) * 10
cascade_dom['decade'] = (cascade_dom['year'] // 10) * 10

print(f"Loaded: {len(cascades)} cascades, {len(cluster_dom)} cluster_dom, {len(cascade_dom)} cascade_dom")


def savefig(fig, name):
    fig.savefig(OUT / name, format='pdf')
    plt.close(fig)
    print(f"  Saved {name}")


# ═══════════════════════════════════════════════════════════════════════
# Fig 01: Cascade volume by year
# ═══════════════════════════════════════════════════════════════════════
print("Fig 01: Cascade volume...")
fig, ax1 = plt.subplots(figsize=(12, 4.5))

by_year = summary['by_year']
years_sorted = sorted([int(y) for y in by_year.keys()])

# Stacked bars by classification
bottom = np.zeros(len(years_sorted))
for cls in CLASS_ORDER:
    vals = []
    for y in years_sorted:
        bc = by_year[str(y)].get('by_classification', {})
        vals.append(bc.get(cls, 0))
    ax1.bar(years_sorted, vals, bottom=bottom, color=CLASS_COLORS[cls],
            label=CLASS_LABELS[cls], width=0.8, edgecolor='white', linewidth=0.3)
    bottom += np.array(vals)

ax1.set_xlabel('Year')
ax1.set_ylabel('Number of cascades')
ax1.legend(loc='upper left', framealpha=0.9)

ax2 = ax1.twinx()
articles = [by_year[str(y)].get('n_articles', 0) for y in years_sorted]
ax2.plot(years_sorted, articles, color='black', linewidth=1.5, alpha=0.7, linestyle='--')
ax2.set_ylabel('Articles')
ax2.spines['top'].set_visible(False)

savefig(fig, 'fig01_cascade_volume.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 02: Score distribution by decade
# ═══════════════════════════════════════════════════════════════════════
print("Fig 02: Score distribution...")
fig, ax = plt.subplots(figsize=(8, 5))

decade_groups = []
decade_names = []
for d in sorted(DECADE_LABELS.keys()):
    mask = cascades['decade'] == d
    scores = cascades.loc[mask, 'total_score'].dropna()
    if len(scores) > 0:
        decade_groups.append(scores.values)
        decade_names.append(DECADE_LABELS[d])

parts = ax.violinplot(decade_groups, positions=range(len(decade_names)),
                      showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('#377EB8')
    pc.set_alpha(0.6)

ax.axhline(0.65, color='#E41A1C', linestyle='--', alpha=0.7, label='Strong (0.65)')
ax.axhline(0.40, color='#FF7F00', linestyle='--', alpha=0.7, label='Moderate (0.40)')
ax.axhline(0.25, color='#999999', linestyle='--', alpha=0.7, label='Weak (0.25)')

ax.set_xticks(range(len(decade_names)))
ax.set_xticklabels(decade_names)
ax.set_ylabel('Cascade score')
ax.legend(loc='upper left')

savefig(fig, 'fig02_score_distribution.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 03: Top 20 decomposition
# ═══════════════════════════════════════════════════════════════════════
print("Fig 03: Top 20 decomposition...")
top20 = cascades.nlargest(20, 'total_score').copy()
top20 = top20.sort_values('total_score', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
dims = ['score_temporal', 'score_participation', 'score_convergence', 'score_source']
dim_labels = ['Temporal', 'Participation', 'Convergence', 'Source']
dim_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']

labels = [f"{r['frame']} ({r['year']})" for _, r in top20.iterrows()]
left = np.zeros(len(top20))
for dim, lbl, col in zip(dims, dim_labels, dim_colors):
    vals = top20[dim].values * 0.25
    ax.barh(range(len(top20)), vals, left=left, color=col, label=lbl, height=0.7)
    left += vals

ax.set_yticks(range(len(top20)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Cascade score (weighted)')
ax.legend(loc='lower right')

savefig(fig, 'fig03_top20_decomposition.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 04: Paradigm coverage
# ═══════════════════════════════════════════════════════════════════════
print("Fig 04: Paradigm coverage...")
fig, ax = plt.subplots(figsize=(10, 4.5))

# First date per year from paradigm timeline
first_dates = paradigm_timeline.groupby('year')['date'].min()
first_doy = first_dates.dt.dayofyear

ax.bar(first_doy.index, 365 - first_doy.values, bottom=first_doy.values,
       color='#377EB8', alpha=0.7, label='Actual coverage')
ax.axhline(85, color='#E41A1C', linestyle='--', alpha=0.7,
           label='Constant start (day 85)')
ax.fill_between(first_doy.index, 85, 365, alpha=0.1, color='#E41A1C')

ax.set_xlabel('Year')
ax.set_ylabel('Day of year')
ax.set_ylim(0, 370)
ax.legend()

savefig(fig, 'fig04_paradigm_coverage.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 05: Paradigm dynamics (2018 example + shift density)
# ═══════════════════════════════════════════════════════════════════════
print("Fig 05: Paradigm dynamics...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [2, 1]})

# 2018 frame dominance
t2018 = paradigm_timeline[paradigm_timeline['year'] == 2018].copy()
t2018 = t2018.sort_values('date')
for frame in FRAME_ORDER:
    col = f'paradigm_{frame}'
    if col in t2018.columns:
        ax1.plot(t2018['date'], t2018[col], color=FRAME_COLORS[frame],
                 label=FRAME_LABELS[frame], linewidth=1.2)

ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax1.set_ylabel('Dominance index')
ax1.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.15))
ax1.set_xlim(t2018['date'].min(), t2018['date'].max())

# Shift density by year
shift_years = [int(s['shift_date'][:4]) for s in shifts]
year_counts = Counter(shift_years)
syears = sorted(year_counts.keys())
ax2.bar(syears, [year_counts[y] for y in syears], color='#377EB8', alpha=0.7)
ax2.set_xlabel('Year')
ax2.set_ylabel('Paradigm shifts')

plt.tight_layout()
savefig(fig, 'fig05_paradigm_dynamics.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 06: StabSel validation (R² scatter)
# ═══════════════════════════════════════════════════════════════════════
print("Fig 06: StabSel validation...")
fig, ax = plt.subplots(figsize=(7, 6))

for model, marker, color in [('A', 'o', '#E41A1C'), ('B', 's', '#377EB8')]:
    mask = validation['model'] == model
    subset = validation[mask].dropna(subset=['r2_full', 'r2_test'])
    ax.scatter(subset['r2_full'], subset['r2_test'], c=color, marker=marker,
               alpha=0.4, s=20, label=f'Model {model}')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
ax.set_xlabel(r'$R^2$ (full model)')
ax.set_ylabel(r'$R^2$ (test set)')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.5, 1.05)
ax.legend()

savefig(fig, 'fig06_stabsel_validation.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 07: Event type roles (Model A)
# ═══════════════════════════════════════════════════════════════════════
print("Fig 07: Event type roles...")
fig, ax = plt.subplots(figsize=(10, 5))

role_colors = {'catalyst': '#4DAF4A', 'disruptor': '#E41A1C', 'inert': '#999999'}
role_order = ['catalyst', 'disruptor', 'inert']

# Parse event types from cluster_dom
type_role_counts = defaultdict(lambda: defaultdict(int))
for _, row in cluster_dom.iterrows():
    dt = row.get('dominant_type', '')
    if dt and isinstance(dt, str):
        type_role_counts[dt][row['role']] += 1

etypes = sorted(type_role_counts.keys())
x = np.arange(len(etypes))
width = 0.25

for i, role in enumerate(role_order):
    vals = [type_role_counts[et][role] for et in etypes]
    ax.bar(x + i * width, vals, width, label=role.capitalize(), color=role_colors[role])

ax.set_xticks(x + width)
elabels = [EVENT_LABELS.get(et, et.replace('evt_', '').capitalize()) for et in etypes]
ax.set_xticklabels(elabels, rotation=30, ha='right')
ax.set_ylabel('Count')
ax.legend()

savefig(fig, 'fig07_event_type_roles.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 08: Cross-frame matrix (cascade_frame × target_frame)
# ═══════════════════════════════════════════════════════════════════════
print("Fig 08: Cross-frame matrix...")
fig, ax = plt.subplots(figsize=(7, 6))

# Build 8×8 matrix: mean |net_beta| for cascade_frame × target_frame
sig = cascade_dom[cascade_dom['role'].isin(['catalyst', 'disruptor'])].copy()
matrix = np.zeros((8, 8))
counts = np.zeros((8, 8))

for _, row in sig.iterrows():
    cf = row.get('cascade_frame', '')
    tf = row.get('target_frame', '')
    if cf in FRAME_ORDER and tf in FRAME_ORDER:
        i = FRAME_ORDER.index(cf)
        j = FRAME_ORDER.index(tf)
        matrix[i, j] += abs(row['net_beta'])
        counts[i, j] += 1

with np.errstate(divide='ignore', invalid='ignore'):
    matrix = np.where(counts > 0, matrix / counts, 0)

im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(8))
ax.set_xticklabels([FRAME_LABELS[f] for f in FRAME_ORDER], rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(8))
ax.set_yticklabels([FRAME_LABELS[f] for f in FRAME_ORDER], fontsize=8)
ax.set_xlabel('Target frame')
ax.set_ylabel('Cascade frame')
plt.colorbar(im, ax=ax, label=r'Mean $|\beta|$')

savefig(fig, 'fig08_cross_frame_matrix.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 09: Frame volatility (entering vs exiting)
# ═══════════════════════════════════════════════════════════════════════
print("Fig 09: Frame volatility...")
fig, ax = plt.subplots(figsize=(9, 5))

entering_counts = defaultdict(int)
exiting_counts = defaultdict(int)
for s in shifts:
    for f in s.get('entering_frames', []):
        entering_counts[f] += 1
    for f in s.get('exiting_frames', []):
        exiting_counts[f] += 1

x = np.arange(len(FRAME_ORDER))
width = 0.35
ax.bar(x - width/2, [entering_counts.get(f, 0) for f in FRAME_ORDER],
       width, label='Entering dominance', color='#4DAF4A')
ax.bar(x + width/2, [exiting_counts.get(f, 0) for f in FRAME_ORDER],
       width, label='Exiting dominance', color='#E41A1C')

ax.set_xticks(x)
ax.set_xticklabels([FRAME_LABELS[f] for f in FRAME_ORDER], rotation=30, ha='right')
ax.set_ylabel('Count')
ax.legend()

savefig(fig, 'fig09_frame_volatility.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 10: Q1 coverage (significance rates)
# ═══════════════════════════════════════════════════════════════════════
print("Fig 10: Q1 coverage...")
fig, ax = plt.subplots(figsize=(8, 5))

# Q1 = months 1-3, Q2-Q4 = months 4-12
cluster_dom['month'] = pd.to_datetime(cluster_dom['peak_date'], errors='coerce').dt.month
q1 = cluster_dom[cluster_dom['month'].between(1, 3)]
q234 = cluster_dom[cluster_dom['month'].between(4, 12)]

q1_sig = q1[q1['role'] != 'inert'].shape[0] / max(1, len(q1)) * 100
q234_sig = q234[q234['role'] != 'inert'].shape[0] / max(1, len(q234)) * 100

bars = ax.bar(['Q1 (Jan-Mar)', 'Q2-Q4 (Apr-Dec)'], [q1_sig, q234_sig],
              color=['#E41A1C', '#377EB8'], alpha=0.7)
ax.set_ylabel('Significance rate (%)')
for bar, val in zip(bars, [q1_sig, q234_sig]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom')

savefig(fig, 'fig10_q1_coverage.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 11: Lag asymmetry (driver vs suppressor)
# ═══════════════════════════════════════════════════════════════════════
print("Fig 11: Lag asymmetry...")
fig, ax = plt.subplots(figsize=(8, 5))

# Drivers = catalyst, suppressors = disruptor (Model A)
drivers_a = cluster_dom[cluster_dom['role'] == 'catalyst']
suppressors_a = cluster_dom[cluster_dom['role'] == 'disruptor']

# Use ar_order as a proxy for lag structure
bins = np.arange(0, 8) - 0.5
if len(drivers_a) > 0:
    ax.hist(drivers_a['ar_order'].dropna(), bins=bins, alpha=0.6,
            color='#4DAF4A', label=f'Drivers (n={len(drivers_a)})', density=True)
if len(suppressors_a) > 0:
    ax.hist(suppressors_a['ar_order'].dropna(), bins=bins, alpha=0.6,
            color='#E41A1C', label=f'Suppressors (n={len(suppressors_a)})', density=True)

ax.set_xlabel('AR order (lag)')
ax.set_ylabel('Density')
ax.legend()

savefig(fig, 'fig11_lag_asymmetry.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 12: Driver ratio evolution by decade
# ═══════════════════════════════════════════════════════════════════════
print("Fig 12: Driver ratio evolution...")
fig, ax = plt.subplots(figsize=(8, 5))

sig_a = cluster_dom[cluster_dom['role'] != 'inert']
decade_driver = sig_a.groupby('decade').apply(
    lambda g: (g['role'] == 'catalyst').sum() / max(1, len(g)) * 100
)

decades_present = sorted(decade_driver.index)
ax.bar([DECADE_LABELS.get(d, str(d)) for d in decades_present],
       [decade_driver[d] for d in decades_present],
       color='#4DAF4A', alpha=0.7)
ax.set_ylabel('% Drivers among significant')
ax.axhline(50, color='gray', linestyle=':', alpha=0.5)

savefig(fig, 'fig12_driver_ratio_evolution.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 13: Lag by frame (driver/suppressor)
# ═══════════════════════════════════════════════════════════════════════
print("Fig 13: Lag by frame...")
fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(FRAME_ORDER))
width = 0.35

driver_lags = []
suppr_lags = []
for f in FRAME_ORDER:
    d = cluster_dom[(cluster_dom['frame'] == f) & (cluster_dom['role'] == 'catalyst')]
    s = cluster_dom[(cluster_dom['frame'] == f) & (cluster_dom['role'] == 'disruptor')]
    driver_lags.append(d['ar_order'].mean() if len(d) > 0 else 0)
    suppr_lags.append(s['ar_order'].mean() if len(s) > 0 else 0)

ax.barh(x - width/2, driver_lags, width, label='Driver lag', color='#4DAF4A')
ax.barh(x + width/2, suppr_lags, width, label='Suppressor lag', color='#E41A1C')
ax.set_yticks(x)
ax.set_yticklabels([FRAME_LABELS[f] for f in FRAME_ORDER])
ax.set_xlabel('Mean AR order')
ax.legend()

savefig(fig, 'fig13_lag_by_frame.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 14: Driver ratio heatmap (frame × event type)
# ═══════════════════════════════════════════════════════════════════════
print("Fig 14: Driver ratio heatmap...")
fig, ax = plt.subplots(figsize=(10, 6))

# Get unique event types present
all_etypes = sorted(set(cluster_dom['dominant_type'].dropna().unique()))
etypes_present = [et for et in all_etypes if et in EVENT_LABELS or et.startswith('evt_')]

matrix = np.full((8, len(etypes_present)), np.nan)
for i, frame in enumerate(FRAME_ORDER):
    for j, et in enumerate(etypes_present):
        subset = cluster_dom[(cluster_dom['frame'] == frame) &
                            (cluster_dom['dominant_type'] == et) &
                            (cluster_dom['role'] != 'inert')]
        if len(subset) >= 3:
            matrix[i, j] = (subset['role'] == 'catalyst').sum() / len(subset) * 100

im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(len(etypes_present)))
ax.set_xticklabels([EVENT_LABELS.get(et, et.replace('evt_', '').capitalize())
                     for et in etypes_present], rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(8))
ax.set_yticklabels([FRAME_LABELS[f] for f in FRAME_ORDER], fontsize=8)
plt.colorbar(im, ax=ax, label='% Driver')

# Mark NaN cells
for i in range(8):
    for j in range(len(etypes_present)):
        if np.isnan(matrix[i, j]):
            ax.text(j, i, '–', ha='center', va='center', color='gray', fontsize=8)

savefig(fig, 'fig14_driver_ratio_heatmap.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 15: Selectivity matrix (|β| mean, Model A)
# ═══════════════════════════════════════════════════════════════════════
print("Fig 15: Selectivity matrix...")
fig, ax = plt.subplots(figsize=(10, 6))

matrix = np.zeros((len(etypes_present), 8))
counts = np.zeros((len(etypes_present), 8))
for _, row in cluster_dom.iterrows():
    dt = row.get('dominant_type', '')
    fr = row.get('frame', '')
    if dt in etypes_present and fr in FRAME_ORDER:
        i = etypes_present.index(dt)
        j = FRAME_ORDER.index(fr)
        matrix[i, j] += abs(row['net_beta'])
        counts[i, j] += 1

with np.errstate(divide='ignore', invalid='ignore'):
    matrix = np.where(counts > 0, matrix / counts, 0)

im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
ax.set_yticks(range(len(etypes_present)))
ax.set_yticklabels([EVENT_LABELS.get(et, et.replace('evt_', '').capitalize())
                     for et in etypes_present], fontsize=8)
ax.set_xticks(range(8))
ax.set_xticklabels([FRAME_LABELS[f] for f in FRAME_ORDER], rotation=45, ha='right', fontsize=8)
ax.set_xlabel('Target frame')
ax.set_ylabel('Event type')
plt.colorbar(im, ax=ax, label=r'Mean $|\beta|$')

savefig(fig, 'fig15_selectivity_matrix.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 16: Predator/prey scores (recomputed)
# ═══════════════════════════════════════════════════════════════════════
print("Fig 16: Predator/prey scores...")
fig, ax = plt.subplots(figsize=(9, 5))

# Predator = net driver of other frames, Prey = net suppressed by other frames
# For each frame: predator_score = sum of catalyst interactions targeting OTHER frames
#                 prey_score = sum of disruptor interactions targeting THIS frame
predator_scores = {}
prey_scores = {}

for f in FRAME_ORDER:
    # As source frame: how many times do cascades of this frame catalyze other frames?
    src = cascade_dom[(cascade_dom['cascade_frame'] == f) & (cascade_dom['role'] == 'catalyst')]
    cross_src = src[src['target_frame'] != f]
    predator_scores[f] = len(cross_src)

    # As target frame: how many times is this frame disrupted by other frames?
    tgt = cascade_dom[(cascade_dom['target_frame'] == f) & (cascade_dom['role'] == 'disruptor')]
    cross_tgt = tgt[tgt['cascade_frame'] != f]
    prey_scores[f] = len(cross_tgt)

# Net score: predator - prey (normalized)
max_val = max(max(predator_scores.values(), default=1), max(prey_scores.values(), default=1))
net_scores = {f: (predator_scores[f] - prey_scores[f]) / max(1, max_val) for f in FRAME_ORDER}

sorted_frames = sorted(FRAME_ORDER, key=lambda f: net_scores[f])
colors = ['#4DAF4A' if net_scores[f] > 0 else '#E41A1C' for f in sorted_frames]

ax.barh(range(len(sorted_frames)),
        [net_scores[f] for f in sorted_frames],
        color=colors, alpha=0.7)
ax.set_yticks(range(len(sorted_frames)))
ax.set_yticklabels([FRAME_LABELS[f] for f in sorted_frames])
ax.set_xlabel('Net predator score (normalized)')
ax.axvline(0, color='gray', linewidth=0.8)

savefig(fig, 'fig16_predator_prey.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 17: Paradigm complexity by decade
# ═══════════════════════════════════════════════════════════════════════
print("Fig 17: Paradigm complexity...")
fig, ax = plt.subplots(figsize=(9, 5))

complexity_types = ['Mono-paradigm', 'Dual-paradigm', 'Triple-paradigm', 'Quad-paradigm']
complexity_colors = ['#377EB8', '#4DAF4A', '#FF7F00', '#E41A1C']

paradigm_timeline['decade'] = (paradigm_timeline['year'] // 10) * 10
decade_complexity = {}
for d in sorted(paradigm_timeline['decade'].unique()):
    sub = paradigm_timeline[paradigm_timeline['decade'] == d]
    total = len(sub)
    decade_complexity[d] = {ct: (sub['paradigm_type'] == ct).sum() / max(1, total) * 100
                           for ct in complexity_types}

decades_sorted = sorted(decade_complexity.keys())
x = np.arange(len(decades_sorted))
bottom = np.zeros(len(decades_sorted))

for ct, col in zip(complexity_types, complexity_colors):
    vals = [decade_complexity[d].get(ct, 0) for d in decades_sorted]
    ax.bar(x, vals, bottom=bottom, color=col, label=ct.replace('-paradigm', ''))
    bottom += np.array(vals)

ax.set_xticks(x)
ax.set_xticklabels([DECADE_LABELS.get(d, f'{d}s') for d in decades_sorted])
ax.set_ylabel('% of days')
ax.legend(loc='upper right')

savefig(fig, 'fig17_paradigm_complexity.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 18: Secular trends (yearly dominance)
# ═══════════════════════════════════════════════════════════════════════
print("Fig 18: Secular trends...")
fig, ax = plt.subplots(figsize=(12, 5))

yearly_dom = paradigm_timeline.groupby('year')[[f'paradigm_{f}' for f in FRAME_ORDER]].mean()

focus_frames = ['Pol', 'Sci', 'Eco', 'Envt']
for f in focus_frames:
    col = f'paradigm_{f}'
    if col in yearly_dom.columns:
        ax.plot(yearly_dom.index, yearly_dom[col], color=FRAME_COLORS[f],
                label=FRAME_LABELS[f], linewidth=1.5, alpha=0.7)
        # Trend line
        valid = yearly_dom[col].dropna()
        if len(valid) > 2:
            z = np.polyfit(valid.index, valid.values, 1)
            p = np.poly1d(z)
            ax.plot(valid.index, p(valid.index), color=FRAME_COLORS[f],
                    linestyle='--', linewidth=0.8, alpha=0.5)

ax.set_xlabel('Year')
ax.set_ylabel('Mean dominance index')
ax.legend()

savefig(fig, 'fig18_secular_trends.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 19: Auto-suppression (using is_own_frame)
# ═══════════════════════════════════════════════════════════════════════
print("Fig 19: Auto-suppression...")
fig, ax = plt.subplots(figsize=(9, 5))

own = cascade_dom[cascade_dom['is_own_frame'] == True].copy()
x = np.arange(len(FRAME_ORDER))
width = 0.35

auto_cat = []
auto_dis = []
for f in FRAME_ORDER:
    sub = own[own['cascade_frame'] == f]
    auto_cat.append((sub['role'] == 'catalyst').sum())
    auto_dis.append((sub['role'] == 'disruptor').sum())

ax.bar(x - width/2, auto_cat, width, label='Auto-amplification', color='#4DAF4A')
ax.bar(x + width/2, auto_dis, width, label='Auto-suppression', color='#E41A1C')

ax.set_xticks(x)
ax.set_xticklabels([FRAME_LABELS[f] for f in FRAME_ORDER], rotation=30, ha='right')
ax.set_ylabel('Count')
ax.legend()

savefig(fig, 'fig19_auto_suppression.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 20: Strength vs impact scatter
# ═══════════════════════════════════════════════════════════════════════
print("Fig 20: Strength vs impact...")
fig, ax = plt.subplots(figsize=(8, 6))

# Use alignment_b for cascade-level impact
if len(alignment_b) > 0:
    valid = alignment_b.dropna(subset=['total_score', 'impact_magnitude'])
    if 'total_score' not in valid.columns:
        # Fallback: merge with cascades
        valid = alignment_b.merge(
            cascades[['cascade_id', 'total_score', 'frame']],
            on='cascade_id', how='left', suffixes=('', '_casc')
        ).dropna(subset=['total_score', 'impact_magnitude'])

    for f in FRAME_ORDER:
        sub = valid[valid['cascade_frame'] == f]
        if len(sub) > 0:
            ax.scatter(sub['total_score'], sub['impact_magnitude'],
                      c=FRAME_COLORS[f], label=FRAME_LABELS[f], alpha=0.5, s=30)

    ax.set_xlabel('Cascade score')
    ax.set_ylabel('Paradigm impact magnitude')
    ax.legend(fontsize=7, ncol=2)

savefig(fig, 'fig20_strength_vs_impact.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 21: Concentration + coherence over time
# ═══════════════════════════════════════════════════════════════════════
print("Fig 21: Concentration & coherence...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

yearly_stats = paradigm_timeline.groupby('year').agg(
    concentration=('concentration', 'mean'),
    coherence=('coherence', 'mean')
)

ax1.plot(yearly_stats.index, yearly_stats['concentration'], color='#377EB8', linewidth=1.5)
ax1.fill_between(yearly_stats.index, yearly_stats['concentration'], alpha=0.2, color='#377EB8')
ax1.set_ylabel('Mean concentration')

ax2.plot(yearly_stats.index, yearly_stats['coherence'], color='#E41A1C', linewidth=1.5)
ax2.fill_between(yearly_stats.index, yearly_stats['coherence'], alpha=0.2, color='#E41A1C')
ax2.set_ylabel('Mean coherence')
ax2.set_xlabel('Year')

plt.tight_layout()
savefig(fig, 'fig21_concentration_coherence.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 22: Regime duration histogram
# ═══════════════════════════════════════════════════════════════════════
print("Fig 22: Regime duration...")
fig, ax = plt.subplots(figsize=(8, 5))

durations = [s['regime_duration_days'] for s in shifts if s.get('regime_duration_days', 0) > 0]
ax.hist(durations, bins=50, color='#377EB8', alpha=0.7, edgecolor='white')
ax.axvline(np.median(durations), color='#E41A1C', linestyle='--',
           label=f'Median: {np.median(durations):.0f} days')
ax.set_xlabel('Regime duration (days)')
ax.set_ylabel('Count')
ax.legend()

savefig(fig, 'fig22_regime_duration.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 23: Reversibility by decade
# ═══════════════════════════════════════════════════════════════════════
print("Fig 23: Reversibility...")
fig, ax = plt.subplots(figsize=(8, 5))

shift_df = pd.DataFrame(shifts)
shift_df['year'] = shift_df['shift_date'].str[:4].astype(int)
shift_df['decade'] = (shift_df['year'] // 10) * 10

rev_by_decade = shift_df.groupby('decade').apply(
    lambda g: g['reversible'].sum() / max(1, len(g)) * 100
)

decades_present = sorted(rev_by_decade.index)
ax.bar([DECADE_LABELS.get(d, f'{d}s') for d in decades_present],
       [rev_by_decade[d] for d in decades_present],
       color='#FF7F00', alpha=0.7)
ax.set_ylabel('% Reversible shifts')
ax.axhline(50, color='gray', linestyle=':', alpha=0.5)

savefig(fig, 'fig23_reversibility.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 24: Cannibalization flows
# ═══════════════════════════════════════════════════════════════════════
print("Fig 24: Cannibalization flows...")
fig, ax = plt.subplots(figsize=(10, 6))

# Count co-occurrences of entering/exiting frames in shifts
flows = defaultdict(int)
for s in shifts:
    entering = s.get('entering_frames', [])
    exiting = s.get('exiting_frames', [])
    for ef in exiting:
        for nf in entering:
            if ef != nf:
                flows[(ef, nf)] += 1

# Top flows
top_flows = sorted(flows.items(), key=lambda x: -x[1])[:20]
labels = [f'{FRAME_LABELS.get(f[0], f[0])} → {FRAME_LABELS.get(f[1], f[1])}' for f, _ in top_flows]
vals = [v for _, v in top_flows]
colors_bar = [FRAME_COLORS.get(f[0], '#999999') for f, _ in top_flows]

ax.barh(range(len(labels)), vals, color=colors_bar, alpha=0.7)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Number of transitions')
ax.invert_yaxis()

savefig(fig, 'fig24_cannibalization_flows.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 25: Copenhagen timeline (Nov-Dec 2009)
# ═══════════════════════════════════════════════════════════════════════
print("Fig 25: Copenhagen timeline...")
fig, ax = plt.subplots(figsize=(12, 5))

t2009 = paradigm_timeline[
    (paradigm_timeline['year'] == 2009) &
    (paradigm_timeline['date'] >= '2009-11-01') &
    (paradigm_timeline['date'] <= '2009-12-31')
].copy()
t2009 = t2009.sort_values('date')

for f in FRAME_ORDER:
    col = f'paradigm_{f}'
    if col in t2009.columns:
        ax.plot(t2009['date'], t2009[col], color=FRAME_COLORS[f],
                label=FRAME_LABELS[f], linewidth=1.5)

ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
# Mark COP15 dates
cop15_start = pd.Timestamp('2009-12-07')
cop15_end = pd.Timestamp('2009-12-18')
ax.axvspan(cop15_start, cop15_end, alpha=0.1, color='blue', label='COP15')
ax.set_ylabel('Dominance index')
ax.set_xlabel('Date')
ax.legend(ncol=3, fontsize=8)
fig.autofmt_xdate()

savefig(fig, 'fig25_copenhagen_timeline.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

all_ok = True
for i in range(1, 26):
    fname = f'fig{i:02d}_*.pdf'
    matches = list(OUT.glob(fname))
    if matches:
        size = matches[0].stat().st_size
        status = "OK" if size > 0 else "EMPTY"
        if size == 0:
            all_ok = False
        print(f"  {matches[0].name}: {size:,} bytes [{status}]")
    else:
        print(f"  fig{i:02d}_*.pdf: MISSING")
        all_ok = False

print(f"\n{'All 25 figures generated successfully!' if all_ok else 'Some figures failed!'}")
