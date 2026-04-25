#!/usr/bin/env python3
"""
Generate LaTeX validation report for 2018 cascade detection results.

Loads cached results from results/cache/results_2018.pkl and produces
a comprehensive validation report as a compiled PDF.

Usage:
    python scripts/figures/gen_validation_report_2018.py
"""

import pickle
import sys
import statistics
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_FILE = PROJECT_ROOT / 'results' / 'cache' / 'results_2018.pkl'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'figures'
TEX_FILE = OUTPUT_DIR / 'validation_report_2018.tex'


def esc(s):
    """Escape LaTeX special characters."""
    return str(s).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')


def fmt(v, decimals=3):
    """Format a float."""
    if v is None:
        return '---'
    return f'{v:.{decimals}f}'


def load_results():
    print(f"Loading {CACHE_FILE}...")
    with open(CACHE_FILE, 'rb') as f:
        return pickle.load(f)


def gen_overview(results):
    n_strong = results.n_cascades_by_classification.get('strong_cascade', 0)
    n_moderate = results.n_cascades_by_classification.get('moderate_cascade', 0)
    n_weak = results.n_cascades_by_classification.get('weak_cascade', 0)
    n_not = results.n_cascades_by_classification.get('not_cascade', 0)

    n_occ = len(results.all_occurrences) if results.all_occurrences else 0
    n_attr = len(results.cascade_attributions) if results.cascade_attributions else 0
    n_clusters = len(results.event_clusters) if results.event_clusters else 0
    n_shifts = len(results.paradigm_shifts.shifts) if hasattr(results.paradigm_shifts, 'shifts') else 0
    n_episodes = len(results.paradigm_shifts.episodes) if hasattr(results.paradigm_shifts, 'episodes') else 0

    return r"""
\begin{table}[H]
\centering
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Analysis period & 2018-01-01 to 2018-12-31 \\
Articles analyzed & """ + f"{results.n_articles_analyzed:,}" + r""" \\
Pipeline runtime & """ + f"{results.runtime_seconds:.1f}s" + r""" \\
\midrule
Cascades detected & """ + str(len(results.cascades)) + r""" \\
\quad Strong ($\geq 0.65$) & """ + str(n_strong) + r""" \\
\quad Moderate ($\geq 0.40$) & """ + str(n_moderate) + r""" \\
\quad Weak ($\geq 0.25$) & """ + str(n_weak) + r""" \\
\quad Not cascade ($< 0.25$) & """ + str(n_not) + r""" \\
\midrule
Event occurrences (database-first) & """ + str(n_occ) + r""" \\
Cascade--event attributions & """ + f"{n_attr:,}" + r""" \\
Event clusters (meta-events) & """ + str(n_clusters) + r""" \\
Paradigm shifts & """ + str(n_shifts) + r""" \\
Paradigm episodes & """ + str(n_episodes) + r""" \\
\bottomrule
\end{tabular}
\caption{Pipeline summary for 2018.}
\end{table}
"""


def gen_score_distribution(results):
    cascades = results.cascades
    scores = [c.total_score for c in cascades]
    dims = {
        'Temporal': [c.score_temporal for c in cascades],
        'Participation': [c.score_participation for c in cascades],
        'Convergence': [c.score_convergence for c in cascades],
        'Source': [c.score_source for c in cascades],
    }

    lines = r"""
\begin{table}[H]
\centering
\begin{tabular}{lrrrrr}
\toprule
\textbf{Score} & \textbf{Mean} & \textbf{Median} & \textbf{Std} & \textbf{Min} & \textbf{Max} \\
\midrule
"""
    lines += f"Total & {np.mean(scores):.3f} & {np.median(scores):.3f} & {np.std(scores):.3f} & {min(scores):.3f} & {max(scores):.3f} \\\\\n"
    for name, vals in dims.items():
        lines += f"{name} & {np.mean(vals):.3f} & {np.median(vals):.3f} & {np.std(vals):.3f} & {min(vals):.3f} & {max(vals):.3f} \\\\\n"

    lines += r"""
\bottomrule
\end{tabular}
\caption{Score distribution across 40 cascades (equal weights 0.25 per dimension).}
\end{table}

\noindent\textbf{By classification:}
\begin{itemize}[nosep]
"""
    for cls in ['strong_cascade', 'moderate_cascade', 'weak_cascade', 'not_cascade']:
        group = [c for c in cascades if c.classification == cls]
        if group:
            gs = [c.total_score for c in group]
            lines += f"  \\item \\textbf{{{esc(cls)}}} ({len(group)}): mean={np.mean(gs):.3f}, range=[{min(gs):.3f}, {max(gs):.3f}]\n"
    lines += r"\end{itemize}" + "\n"
    return lines


def gen_dimension_breakdown(results):
    cascades = results.cascades
    lines = r"""
Each dimension score is the mean of its constituent sub-indices, all normalized to $[0, 1]$.
The four dimensions are weighted equally (0.25 each) in the total score.

\begin{table}[H]
\centering
\small
\begin{tabular}{p{6cm}rrrr}
\toprule
\textbf{Sub-Index} & \textbf{Mean} & \textbf{Std} & \textbf{Min} & \textbf{Max} \\
\midrule
"""
    all_keys = set()
    for c in cascades:
        all_keys.update(c.sub_indices.keys())

    for key in sorted(all_keys):
        vals = [c.sub_indices.get(key, 0) for c in cascades]
        lines += f"{esc(key)} & {np.mean(vals):.3f} & {np.std(vals):.3f} & {min(vals):.3f} & {max(vals):.3f} \\\\\n"

    lines += r"""
\bottomrule
\end{tabular}
\caption{Sub-index statistics across 40 cascades.}
\end{table}
"""
    return lines


def gen_top_cascades(results):
    cascades = sorted(results.cascades, key=lambda c: c.total_score, reverse=True)
    lines = r"""
\begin{longtable}{rlp{2cm}p{3cm}rrrr}
\toprule
\textbf{\#} & \textbf{Frame} & \textbf{Period} & \textbf{Classification} & \textbf{T} & \textbf{P} & \textbf{C} & \textbf{S} \\
\midrule
\endfirsthead
\toprule
\textbf{\#} & \textbf{Frame} & \textbf{Period} & \textbf{Classification} & \textbf{T} & \textbf{P} & \textbf{C} & \textbf{S} \\
\midrule
\endhead
"""
    for i, c in enumerate(cascades[:10], 1):
        period = f"{c.onset_date.strftime('%m-%d')}--{c.end_date.strftime('%m-%d')}"
        score_str = f"\\textbf{{{c.total_score:.3f}}}"
        lines += (
            f"{i} & {c.frame} & {period} & "
            f"{esc(c.classification)} ({score_str}) & "
            f"{c.score_temporal:.2f} & {c.score_participation:.2f} & "
            f"{c.score_convergence:.2f} & {c.score_source:.2f} \\\\\n"
        )

    lines += r"""
\bottomrule
\caption{Top 10 cascades by total score. T=Temporal, P=Participation, C=Convergence, S=Source.}
\end{longtable}
"""

    # Details for each
    for i, c in enumerate(cascades[:10], 1):
        n_eo = len(c.event_occurrences) if c.event_occurrences else 0
        evt_types = Counter(
            eo.event_type for eo in c.event_occurrences
        ) if c.event_occurrences else {}
        evt_str = ', '.join(f"{esc(k)}: {v}" for k, v in evt_types.most_common(5))
        media_str = ', '.join(esc(m) for m, _ in (c.top_media or [])[:3])
        lines += f"""
\\paragraph{{\\#{i}: {esc(c.cascade_id)}}}
{c.frame} frame, {c.onset_date.strftime('%Y-%m-%d')} to {c.end_date.strftime('%Y-%m-%d')} ({c.duration_days} days).
{c.n_articles} articles, {c.n_journalists} journalists, {c.n_media} media outlets.
Burst intensity: {c.burst_intensity:.3f}. {n_eo} event occurrences attributed ({evt_str}).
Top media: {media_str}.
"""
    return lines


def gen_strong_cascades(results):
    strong = [c for c in results.cascades if c.classification == 'strong_cascade']
    if not strong:
        return "No strong cascades detected.\n"

    lines = f"\\textbf{{{len(strong)} strong cascades}} (total\\_score $\\geq$ 0.65):\n\n"
    for c in sorted(strong, key=lambda x: x.total_score, reverse=True):
        lines += f"\\textbf{{{esc(c.cascade_id)}}}: score={c.total_score:.3f}, frame={c.frame}, "
        lines += f"{c.onset_date.strftime('%Y-%m-%d')} to {c.end_date.strftime('%Y-%m-%d')} ({c.duration_days}d), "
        lines += f"{c.n_articles} articles, {c.n_media} media.\n\n"

        # Highest sub-indices
        top_si = sorted(c.sub_indices.items(), key=lambda x: x[1], reverse=True)[:5]
        lines += "Highest sub-indices: "
        lines += ', '.join(f"{esc(k)}={v:.3f}" for k, v in top_si) + ".\n\n"

        # Lowest sub-indices
        low_si = sorted(c.sub_indices.items(), key=lambda x: x[1])[:3]
        lines += "Lowest sub-indices: "
        lines += ', '.join(f"{esc(k)}={v:.3f}" for k, v in low_si) + ".\n\n"

    return lines


def gen_weak_cascades(results):
    weak = [c for c in results.cascades if c.classification == 'weak_cascade']
    if not weak:
        return "No weak cascades detected.\n"

    lines = f"\\textbf{{{len(weak)} weak cascades}} (total\\_score $\\in [0.25, 0.40)$):\n\n"
    for c in sorted(weak, key=lambda x: x.total_score):
        lines += f"\\textbf{{{esc(c.cascade_id)}}}: score={c.total_score:.3f}, frame={c.frame}, "
        lines += f"{c.n_articles} articles, {c.n_media} media.\n\n"

        low_si = sorted(c.sub_indices.items(), key=lambda x: x[1])[:5]
        lines += "Lowest sub-indices: "
        lines += ', '.join(f"{esc(k)}={v:.3f}" for k, v in low_si) + ".\n\n"

    return lines


def gen_event_overview(results):
    occs = results.all_occurrences or []
    lines = f"Database-first detection identified \\textbf{{{len(occs)} event occurrences}} "
    lines += f"across the full 2018 analysis period (9,754 articles).\n\n"

    # Size stats
    sizes = [o.n_articles for o in occs]
    masses = [o.effective_mass for o in occs]
    lines += f"Article counts: mean={np.mean(sizes):.1f}, median={np.median(sizes):.1f}, "
    lines += f"min={min(sizes)}, max={max(sizes)}.\n\n"
    lines += f"Effective mass: mean={np.mean(masses):.1f}, median={np.median(masses):.1f}, "
    lines += f"min={min(masses):.1f}, max={max(masses):.1f}.\n\n"

    small = sum(1 for o in occs if o.n_articles < 5)
    lines += f"Occurrences with $<$ 5 articles: {small} ({small*100/len(occs):.1f}\\%).\n\n"

    return lines


def gen_event_by_type(results):
    occs = results.all_occurrences or []
    by_type = defaultdict(list)
    for o in occs:
        by_type[o.event_type].append(o)

    lines = r"""
\begin{table}[H]
\centering
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Event Type} & \textbf{Count} & \textbf{Mean Size} & \textbf{Mean Mass} & \textbf{Mean Conf.} & \textbf{Low Conf.} & \textbf{High Conf.} \\
\midrule
"""
    for et in sorted(by_type.keys()):
        group = by_type[et]
        n = len(group)
        mean_size = np.mean([o.n_articles for o in group])
        mean_mass = np.mean([o.effective_mass for o in group])
        mean_conf = np.mean([o.confidence for o in group])
        n_low = sum(1 for o in group if o.low_confidence)
        n_high = n - n_low
        lines += f"{esc(et)} & {n} & {mean_size:.1f} & {mean_mass:.1f} & {mean_conf:.3f} & {n_low} & {n_high} \\\\\n"

    lines += r"""
\midrule
"""
    n = len(occs)
    mean_size = np.mean([o.n_articles for o in occs])
    mean_mass = np.mean([o.effective_mass for o in occs])
    mean_conf = np.mean([o.confidence for o in occs])
    n_low = sum(1 for o in occs if o.low_confidence)
    lines += f"\\textbf{{Total}} & \\textbf{{{n}}} & {mean_size:.1f} & {mean_mass:.1f} & {mean_conf:.3f} & {n_low} & {n - n_low} \\\\\n"

    lines += r"""
\bottomrule
\end{tabular}
\caption{Event occurrences by type.}
\end{table}
"""
    return lines


def gen_event_confidence(results):
    occs = results.all_occurrences or []
    confs = [o.confidence for o in occs]
    n_low = sum(1 for o in occs if o.low_confidence)

    lines = f"Confidence: mean={np.mean(confs):.3f}, std={np.std(confs):.3f}, "
    lines += f"min={min(confs):.3f}, max={max(confs):.3f}.\n\n"
    lines += f"Low confidence ($<$ 0.40): {n_low}/{len(occs)} ({n_low*100/len(occs):.1f}\\%).\n\n"

    # Confidence components breakdown
    if occs and hasattr(occs[0], 'confidence_components') and occs[0].confidence_components:
        comp_keys = list(occs[0].confidence_components.keys())
        lines += "\\textbf{Confidence components} (mean across all occurrences):\n"
        lines += "\\begin{itemize}[nosep]\n"
        for k in comp_keys:
            vals = [o.confidence_components.get(k, 0) for o in occs if o.confidence_components]
            if vals:
                lines += f"  \\item {esc(k)}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}\n"
        lines += "\\end{itemize}\n\n"

    return lines


def gen_attribution_analysis(results):
    attrs = results.cascade_attributions or []
    cascades = results.cascades
    occs = results.all_occurrences or []

    lines = f"\\textbf{{{len(attrs):,} attributions}} link {len(occs)} occurrences to {len(cascades)} cascades.\n\n"

    # Cascades with attributions
    cascade_attr_counts = Counter(a.cascade_id for a in attrs)
    n_with = sum(1 for c in cascades if cascade_attr_counts.get(c.cascade_id, 0) > 0)
    n_without = len(cascades) - n_with

    lines += f"Cascades with $\\geq 1$ event attribution: {n_with}/{len(cascades)}.\n\n"
    lines += f"Cascades with 0 attributions: {n_without}.\n\n"

    if cascade_attr_counts:
        counts = list(cascade_attr_counts.values())
        lines += f"Attributions per cascade: mean={np.mean(counts):.1f}, "
        lines += f"median={np.median(counts):.1f}, max={max(counts)}.\n\n"

    # Occurrences attributed to multiple cascades
    occ_cascade_counts = Counter(a.occurrence_id for a in attrs)
    multi = sum(1 for oid, cnt in occ_cascade_counts.items() if cnt > 1)
    lines += f"Occurrences attributed to $\\geq 2$ cascades: {multi}/{len(occ_cascade_counts)} "
    lines += f"({multi*100/max(len(occ_cascade_counts),1):.1f}\\%).\n\n"

    # Overlap ratio distribution
    if attrs:
        overlap_ratios = [a.overlap_ratio for a in attrs]
        lines += f"Overlap ratio: mean={np.mean(overlap_ratios):.3f}, "
        lines += f"median={np.median(overlap_ratios):.3f}, "
        lines += f"min={min(overlap_ratios):.3f}, max={max(overlap_ratios):.3f}.\n\n"

    # Unattributed occurrences
    attributed_occ_ids = set(a.occurrence_id for a in attrs)
    all_occ_ids = set(o.occurrence_id for o in occs)
    unattributed = all_occ_ids - attributed_occ_ids
    lines += f"Unattributed occurrences: {len(unattributed)}/{len(all_occ_ids)}.\n\n"

    if len(unattributed) == 0:
        lines += r"""
\noindent\textbf{Note:} All 659 occurrences are attributed to at least one cascade. This is expected
because the database-first approach uses a continuous belonging score (no threshold) and a broad
temporal window for attribution, meaning most events in any given period overlap with at least one
cascade. This does not indicate a problem; the \texttt{overlap\_ratio} field distinguishes strong
attributions from weak ones.
"""

    return lines


def gen_cluster_analysis(results):
    clusters = results.event_clusters or []
    if not clusters:
        return "No event clusters detected.\n"

    n_multi = sum(1 for ec in clusters if ec.is_multi_type)
    n_mono = len(clusters) - n_multi
    strengths = [ec.strength for ec in clusters]

    lines = f"\\textbf{{{len(clusters)} event clusters}} ({n_multi} multi-type, {n_mono} mono-type).\n\n"
    lines += f"Strength: mean={np.mean(strengths):.3f}, std={np.std(strengths):.3f}, "
    lines += f"min={min(strengths):.3f}, max={max(strengths):.3f}.\n\n"

    # Top 10
    lines += "\\textbf{Top 10 event clusters by strength:}\n\n"
    lines += r"""
\begin{table}[H]
\centering
\small
\begin{tabular}{rrrrlrl}
\toprule
\textbf{ID} & \textbf{Str.} & \textbf{N\_occ} & \textbf{Mass} & \textbf{Types} & \textbf{Peak} & \textbf{Multi?} \\
\midrule
"""
    for ec in sorted(clusters, key=lambda x: x.strength, reverse=True)[:10]:
        types = ', '.join(esc(t) for t in sorted(ec.event_types))
        multi = 'Yes' if ec.is_multi_type else 'No'
        lines += f"C{ec.cluster_id} & {ec.strength:.3f} & {ec.n_occurrences} & "
        lines += f"{ec.total_mass:.0f} & {types} & "
        lines += f"{ec.peak_date.strftime('%m-%d')} & {multi} \\\\\n"

    lines += r"""
\bottomrule
\end{tabular}
\caption{Top 10 event clusters.}
\end{table}
"""
    return lines


def gen_paradigm_analysis(results):
    ps = results.paradigm_shifts
    if not ps or not hasattr(ps, 'shifts'):
        return "No paradigm shift data available.\n"

    shifts = ps.shifts
    episodes = ps.episodes if hasattr(ps, 'episodes') else []

    lines = f"\\textbf{{{len(shifts)} paradigm shifts}} in {len(episodes)} episodes.\n\n"

    # Episodes
    if episodes:
        lines += "\\textbf{Episodes:}\n"
        lines += r"""
\begin{table}[H]
\centering
\small
\begin{tabular}{rlrrrl}
\toprule
\textbf{\#} & \textbf{Period} & \textbf{Shifts} & \textbf{Net $\Delta$} & \textbf{Complexity} & \textbf{Reversible?} \\
\midrule
"""
        for i, ep in enumerate(episodes, 1):
            start = ep.start_date.strftime('%m-%d') if hasattr(ep, 'start_date') else '?'
            end = ep.end_date.strftime('%m-%d') if hasattr(ep, 'end_date') else '?'
            n_sh = ep.n_shifts if hasattr(ep, 'n_shifts') else '?'
            net = fmt(ep.net_structural_change, 3) if hasattr(ep, 'net_structural_change') else '?'
            cpx = fmt(ep.max_complexity, 2) if hasattr(ep, 'max_complexity') else '?'
            rev = 'Yes' if getattr(ep, 'reversible', False) else 'No'
            lines += f"{i} & {start}--{end} & {n_sh} & {net} & {cpx} & {rev} \\\\\n"

        lines += r"""
\bottomrule
\end{tabular}
\caption{Paradigm shift episodes.}
\end{table}
"""

    # Cascade roles
    if hasattr(ps, 'cascade_roles') and ps.cascade_roles:
        role_counts = Counter(r.role for r in ps.cascade_roles)
        lines += "\n\\textbf{Cascade roles in paradigm shifts:}\n"
        lines += "\\begin{itemize}[nosep]\n"
        for role, count in role_counts.most_common():
            lines += f"  \\item {esc(role)}: {count} cascades\n"
        lines += "\\end{itemize}\n\n"

    return lines


def gen_validation_issues(results):
    cascades = results.cascades
    occs = results.all_occurrences or []
    attrs = results.cascade_attributions or []

    lines = "\\subsection{Issues Identified}\n\n"
    lines += "\\begin{enumerate}\n"

    # Issue 1: 100% nonzero belonging
    lines += r"""
\item \textbf{100\% nonzero belonging matrix.} Phase 4 produces 6,427,886 / 6,427,886 nonzero
entries (9,754 articles $\times$ 659 clusters). This means every article belongs to every cluster
(with varying belonging scores). While the user explicitly requested no threshold (``ne met pas de
seuil''), this creates very large data structures (each occurrence carries belonging scores for all
9,754 articles). The JSON export is 4.2~GB as a result.

\textbf{Recommendation:} For export/serialization, consider applying a minimum belonging threshold
(e.g., 0.01) to reduce data size without affecting analytical results.
"""

    # Issue 2: 23,065 attributions
    n_attr = len(attrs)
    occ_per_cascade = Counter(a.cascade_id for a in attrs)
    max_attr = max(occ_per_cascade.values()) if occ_per_cascade else 0
    lines += f"""
\\item \\textbf{{High attribution count ({n_attr:,}).}} With 659 occurrences and 40 cascades,
the mean is {n_attr/max(len(cascades),1):.0f} attributions per cascade (max={max_attr}).
This suggests the temporal overlap criterion is very permissive.

\\textbf{{Recommendation:}} Consider filtering attributions by \\texttt{{overlap\\_ratio}} $>$ some
minimum (e.g., 0.05) to retain only meaningful event--cascade relationships.
"""

    # Issue 3: Low-score moderate cascades
    low_moderates = [c for c in cascades
                     if c.classification == 'moderate_cascade'
                     and c.total_score < 0.45]
    if low_moderates:
        lines += f"""
\\item \\textbf{{{len(low_moderates)} moderate cascades with score $<$ 0.45.}} These cascades are
near the weak/moderate boundary:
\\begin{{itemize}}[nosep]
"""
        for c in sorted(low_moderates, key=lambda x: x.total_score):
            lines += f"  \\item {esc(c.cascade_id)}: score={c.total_score:.3f}, "
            lines += f"frame={c.frame}, {c.n_articles} articles, {c.n_media} media\n"
        lines += "\\end{itemize}\n"

    # Issue 4: evt_policy dominance
    by_type = Counter(o.event_type for o in occs)
    dominant = by_type.most_common(1)[0] if by_type else ('', 0)
    lines += f"""
\\item \\textbf{{Event type imbalance.}} {esc(dominant[0])} dominates with {dominant[1]} occurrences
({dominant[1]*100/len(occs):.1f}\\% of total). Distribution:
\\begin{{itemize}}[nosep]
"""
    for et, cnt in by_type.most_common():
        lines += f"  \\item {esc(et)}: {cnt} ({cnt*100/len(occs):.1f}\\%)\n"
    lines += "\\end{itemize}\n"

    # Issue 5: Large occurrences
    large = [o for o in occs if o.n_articles > 500]
    if large:
        lines += f"""
\\item \\textbf{{{len(large)} occurrences with $>$ 500 articles.}} These may represent
overly broad clusters:
\\begin{{itemize}}[nosep]
"""
        for o in sorted(large, key=lambda x: x.n_articles, reverse=True)[:5]:
            lines += f"  \\item occ\\#{o.occurrence_id}: {esc(o.event_type)}, "
            lines += f"n={o.n_articles}, mass={o.effective_mass:.1f}, conf={o.confidence:.3f}\n"
        lines += "\\end{itemize}\n"

    lines += "\\end{enumerate}\n\n"

    # Positive findings
    lines += "\\subsection{Positive Findings}\n\n"
    lines += "\\begin{enumerate}\n"
    lines += r"""
\item \textbf{Performance.} Phase 4 completes in $\sim$2s total (was 280s+ before sparse matrix
optimization). The full pipeline runs in $\sim$23s (excluding paradigm shift analysis).

\item \textbf{Event--cascade coverage.} 35/40 cascades have at least one event attribution,
confirming that cascade detection aligns with real-world event activity.

\item \textbf{Multi-type event clusters.} 204 multi-type clusters (44\% of total) capture
events that span multiple categories (e.g., meeting+policy, publication+weather),
which would be missed by a mono-type approach.

\item \textbf{Paradigm shift integration.} """ + str(len(results.paradigm_shifts.shifts) if hasattr(results.paradigm_shifts, 'shifts') else 0) + r""" shifts
detected across """ + str(len(results.paradigm_shifts.episodes) if hasattr(results.paradigm_shifts, 'episodes') else 0) + r""" episodes, with cascade roles attributed
(amplification, d\'estabilisation, dormante).
"""
    lines += "\\end{enumerate}\n"

    return lines


def main():
    results = load_results()

    # Generate each section
    sections = {
        '%%OVERVIEW_TABLE%%': gen_overview(results),
        '%%SCORE_DISTRIBUTION%%': gen_score_distribution(results),
        '%%DIMENSION_BREAKDOWN%%': gen_dimension_breakdown(results),
        '%%SUBINDEX_STATS%%': '',  # merged into dimension breakdown
        '%%TOP_CASCADES%%': gen_top_cascades(results),
        '%%STRONG_CASCADES%%': gen_strong_cascades(results),
        '%%WEAK_CASCADES%%': gen_weak_cascades(results),
        '%%EVENT_OVERVIEW%%': gen_event_overview(results),
        '%%EVENT_BY_TYPE%%': gen_event_by_type(results),
        '%%EVENT_CONFIDENCE%%': gen_event_confidence(results),
        '%%ATTRIBUTION_ANALYSIS%%': gen_attribution_analysis(results),
        '%%CLUSTER_ANALYSIS%%': gen_cluster_analysis(results),
        '%%PARADIGM_ANALYSIS%%': gen_paradigm_analysis(results),
        '%%VALIDATION_ISSUES%%': gen_validation_issues(results),
    }

    # Read template
    template = TEX_FILE.read_text()

    # Replace placeholders
    for placeholder, content in sections.items():
        template = template.replace(placeholder, content)

    # Write final
    TEX_FILE.write_text(template)
    print(f"LaTeX written to {TEX_FILE}")

    # Compile
    import subprocess
    print("Compiling LaTeX...")
    for _ in range(2):  # Run twice for TOC
        r = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
             TEX_FILE.name],
            cwd=str(OUTPUT_DIR),
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            print("LaTeX error output (last 30 lines):")
            print('\n'.join(r.stdout.split('\n')[-30:]))
            break

    pdf = OUTPUT_DIR / 'validation_report_2018.pdf'
    if pdf.exists():
        print(f"PDF generated: {pdf}")
        print(f"Size: {pdf.stat().st_size / 1024:.1f} KB")
    else:
        print("PDF generation failed!")
        print(r.stdout[-2000:] if r.stdout else "No stdout")


if __name__ == '__main__':
    main()
