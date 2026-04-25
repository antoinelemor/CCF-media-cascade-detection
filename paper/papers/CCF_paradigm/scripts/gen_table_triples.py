#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
gen_table_triples.py

MAIN OBJECTIVE:
---------------
Generate LaTeX table of complete triple chains (event to cascade to paradigm)
from production data. All values computed from data, nothing hardcoded.

Selection criteria for theoretical interest:
- Divergent chains: ranked by product of |direct| * |indirect| beta,
  so that both effects are substantively large (not just one extreme).
- Convergent chains: same ranking criterion.
This ensures each row illustrates a case where all three orders
are meaningfully engaged, not dominated by a single extreme coefficient.

Usage:
  python paper/papers/CCF_paradigm/scripts/gen_table_triples.py

Output:
  tables/table_triples.tex

Author:
-------
Antoine Lemor
"""
import pandas as pd, glob, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PROD = ROOT / 'results' / 'production'
OUT  = Path(__file__).resolve().parent.parent / 'tables'
OUT.mkdir(exist_ok=True)

# Load data
print("Loading causal layers...")
cc = pd.concat([pd.read_parquet(fp).assign(year=int(Path(fp).parts[-3]))
                for fp in sorted(glob.glob(str(PROD / '*/impact_analysis/cluster_cascade.parquet')))
                if pd.read_parquet(fp).shape[0] > 0], ignore_index=True)
cdom_a = pd.concat([pd.read_parquet(fp).assign(year=int(Path(fp).parts[-3]))
                    for fp in sorted(glob.glob(str(PROD / '*/impact_analysis/stabsel_cluster_dominance.parquet')))
                    if pd.read_parquet(fp).shape[0] > 0], ignore_index=True)
cdom_b = pd.concat([pd.read_parquet(fp).assign(year=int(Path(fp).parts[-3]))
                    for fp in sorted(glob.glob(str(PROD / '*/impact_analysis/stabsel_cascade_dominance.parquet')))
                    if pd.read_parquet(fp).shape[0] > 0], ignore_index=True)

cc_sig = cc[cc['role'].isin(['driver', 'suppressor'])]
a_sig = cdom_a[cdom_a['p_value_hac'] < 0.10]
b_sig = cdom_b[cdom_b['p_value_hac'] < 0.10]

# Build same-frame triple chains
print("Building triple chains...")
triples = []
for _, row in cc_sig.iterrows():
    clu, cas, yr = row['cluster_id'], row['cascade_id'], row['year']
    a_m = a_sig[(a_sig['cluster_id']==clu) & (a_sig['year']==yr)]
    b_m = b_sig[(b_sig['cascade_id']==cas) & (b_sig['year']==yr)]
    if a_m.empty or b_m.empty:
        continue
    for _, a in a_m.iterrows():
        for _, b in b_m.iterrows():
            triples.append({
                'year': yr, 'cluster_id': clu, 'cascade_id': cas,
                'cascade_frame': row['cascade_frame'],
                'event_type': str(row.get('dominant_type', '')).replace('evt_', '').title(),
                'a_frame': a['frame'], 'a_beta': a['net_beta'],
                'b_frame': b.get('target_frame', ''), 'b_beta': b.get('net_beta', 0),
                'same_frame': a['frame'] == b.get('target_frame', ''),
            })

tdf = pd.DataFrame(triples)
tdf['divergent'] = (tdf['a_beta'] * tdf['b_beta']) < 0

# Summary stats
total = len(tdf)
same = tdf[tdf['same_frame']].copy()
n_same = len(same)
n_div = int(same['divergent'].sum())
pct_div = n_div / n_same * 100

print(f"Total triple chain links: {total}")
print(f"Same-frame: {n_same}, divergent: {n_div} ({pct_div:.1f}%)")

# Selection: rank by product of absolute betas (both effects must be large)
same['joint_strength'] = same['a_beta'].abs() * same['b_beta'].abs()

# Top divergent chains
div = same[same['divergent']].copy()
# Filter extreme outliers
div_clean = div[div['a_beta'].abs() < 100]
top_div = div_clean.nlargest(6, 'joint_strength')

# Top convergent chains
conv = same[~same['divergent']].copy()
conv_clean = conv[conv['a_beta'].abs() < 100]
top_conv = conv_clean.nlargest(4, 'joint_strength')

FL = {'Cult':'Culture','Eco':'Economy','Envt':'Environment',
      'Pbh':'Health','Just':'Justice','Pol':'Politics',
      'Sci':'Science','Secu':'Security'}

# Generate LaTeX — clean 5-column format, no arrows in headers
lines = []
lines.append(r'\begin{table}[H]')
lines.append(r'\caption{Complete causal chains linking all three orders. ')
lines.append(f'Of {n_same} same-frame triple chains, {n_div} ({pct_div:.0f}\\%) show divergent effects ')
lines.append(r'between the direct (event to paradigm) and indirect (event to cascade to paradigm) pathways. ')
lines.append(r'Chains are ranked by joint strength (product of $|\beta_\mathrm{direct}| \times |\beta_\mathrm{indirect}|$), ')
lines.append(r'so that each row represents a case where all three orders are substantively engaged. ')
lines.append(r'$+$ = reinforcement, $-$ = erosion.}')
lines.append(r'\label{tab:triples}')
lines.append(r'\centering\footnotesize')
lines.append(r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}')
lines.append(r'\toprule')
lines.append(r'\textbf{Year} & \textbf{Event type} & \textbf{Cascade frame} & \textbf{Paradigm dimension} & \textbf{Direct / Indirect $\beta$} \\')
lines.append(r'\midrule')
lines.append(r'\multicolumn{5}{l}{\textit{A. Divergent chains (opposing effects)}} \\[3pt]')

for _, r in top_div.iterrows():
    lines.append(f"{int(r['year'])} & {r['event_type']} & {FL.get(r['cascade_frame'], r['cascade_frame'])} & "
                 f"{FL.get(r['a_frame'], r['a_frame'])} & ${r['a_beta']:+.2f}$ / ${r['b_beta']:+.3f}$ \\\\")

lines.append(r'\midrule')
lines.append(r'\multicolumn{5}{l}{\textit{B. Convergent chains (aligned effects)}} \\[3pt]')

for _, r in top_conv.iterrows():
    lines.append(f"{int(r['year'])} & {r['event_type']} & {FL.get(r['cascade_frame'], r['cascade_frame'])} & "
                 f"{FL.get(r['a_frame'], r['a_frame'])} & ${r['a_beta']:+.2f}$ / ${r['b_beta']:+.3f}$ \\\\")

lines.append(r'\bottomrule')
lines.append(r'\end{tabular*}')
lines.append(r'\end{table}')

tex = '\n'.join(lines)
(OUT / 'table_triples.tex').write_text(tex)
print(f"\nWritten to {OUT / 'table_triples.tex'}")
print(tex)
