#!/usr/bin/env python3
"""Top event clusters visualizations across full database (1978-2024).

Produces three publication-quality figures:
  1. Top 10 clusters by strength — horizontal stacked bar
  2. Top 5 clusters per event type — multi-panel (7 types)
  3. Strength vs. coherence scatter — all clusters, top 10 labelled

Saves to results/figures/top/
"""

import json
import pathlib
import textwrap

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[2]
CLUSTER_JSON = ROOT / "results" / "production" / "cross_year_event_clusters.json"
OUT_DIR = ROOT / "results" / "figures" / "top"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COMP_COLORS = {
    "mass_score":       "#2166AC",
    "coverage_score":   "#4393C3",
    "intensity_score":  "#F4A582",
    "coherence_score":  "#D6604D",
    "media_diversity":  "#92C5DE",
}
COMP_LABELS = {
    "mass_score":       "Mass",
    "coverage_score":   "Coverage",
    "intensity_score":  "Intensity",
    "coherence_score":  "Coherence",
    "media_diversity":  "Media div.",
}

TYPE_COLORS = {
    "evt_policy":      "#1B7837",
    "evt_meeting":     "#762A83",
    "evt_publication": "#2166AC",
    "evt_weather":     "#B2182B",
    "evt_judiciary":   "#E08214",
    "evt_election":    "#542788",
    "evt_protest":     "#D6604D",
}
TYPE_LABELS = {
    "evt_policy":      "Policy",
    "evt_meeting":     "Meeting",
    "evt_publication": "Publication",
    "evt_weather":     "Weather",
    "evt_judiciary":   "Judiciary",
    "evt_election":    "Election",
    "evt_protest":     "Protest",
}

# ── Hand-curated names for top clusters ──────────────────────────────────────
# key = (year, cluster_id)
CLUSTER_NAMES = {
    # Top 10
    (2017, "2017-02-16"): "CETA Ratification, EU-Canada",
    (2021, "2021-04-23"): "Biden Climate Summit & Canada Pledge",
    (2008, "2008-07-09"): "G8 Summit Climate Commitments, Hokkaido",
    (2015, "2015-11-23"): "Pre-COP21 Federal-Provincial Summit",
    (2010, "2010-11-18"): "Cancun COP16 & Clean Energy Act",
    (2021, "2021-01-06"): "U.S. Capitol & Georgia Runoff",
    (2016, "2016-10-04"): "Federal Carbon Price Announcement",
    (2011, "2011-12-13"): "Canada Exits Kyoto Protocol (COP17 Durban)",
    (2007, "2007-04-13"): "IPCC AR4 & Conference Board Report",
    (2018, "2018-12-17"): "COP24 Outcomes, Katowice",
    # Top per-type extras
    (2007, "2007-06-22"): "Harper Clean Air Act & Kyoto",
    (2019, "2019-12-06"): "Throne Speech & COP25",
    (2007, "2007-06-08"): "G8 Summit Heiligendamm (Merkel-Bush)",
    (2007, "2007-09-08"): "APEC Summit Sydney",
    (2020, "2020-05-19"): "Keystone XL Cancellation Debate",
    (2019, "2019-06-19"): "Trans Mountain Re-Approval",
    (2018, "2018-06-08"): "G7 Summit, Charlevoix",
    (2003, "2003-08-12"): "2003 Blackout & Energy Policy",
    (2016, "2016-12-10"): "First Ministers' Climate Framework",
    (2007, "2007-06-01"): "Pre-G8 Climate Debate (Bush)",
    (2009, "2009-12-18"): "Copenhagen Accord (COP15)",
    (2006, "2006-10-20"): "Stern Review & Clean Air Act",
    (2015, "2015-12-01"): "COP21 Paris Agreement",
    (2019, "2019-09-27"): "Global Climate Strike",
    (2021, "2021-11-12"): "COP26 Glasgow Outcomes",
    (2015, "2015-06-08"): "G7 Summit Schloss Elmau",
    (2012, "2012-06-22"): "Rio+20 Summit",
    (2022, "2022-11-18"): "COP27 Sharm el-Sheikh",
    (2009, "2009-06-05"): "Obama Clean Energy Bill",
    (2014, "2014-09-25"): "UN Climate Summit, New York",
}


def _get_name(c: dict) -> str:
    """Return human-readable cluster name from entities + manual overrides."""
    year = c.get("year", 0)
    peak = c.get("peak_date", "")[:10]
    key = (year, peak)

    if key in CLUSTER_NAMES:
        return CLUSTER_NAMES[key]

    # Auto-generate from entities
    pers = [e.replace("PER:", "") for e in c.get("entities", []) if e.startswith("PER:")][:2]
    orgs = [e.replace("ORG:", "") for e in c.get("entities", []) if e.startswith("ORG:")][:2]
    locs = [e.replace("LOC:", "") for e in c.get("entities", []) if e.startswith("LOC:")][:2]

    parts = []
    if pers:
        parts.append(", ".join(pers))
    if orgs:
        parts.append(", ".join(orgs))
    if locs:
        parts.append(", ".join(locs))

    types = [TYPE_LABELS.get(t, t.replace("evt_", "").title()) for t in c["event_types"]]
    type_str = " + ".join(types)

    if parts:
        name = f'{" / ".join(parts[:2])} ({type_str})'
    else:
        name = f'{type_str} Event'

    return f'{name}, {peak}'


def _type_tag(event_types: list) -> str:
    return " + ".join(TYPE_LABELS.get(t, t) for t in event_types)


# ── Load data ────────────────────────────────────────────────────────────────
with open(CLUSTER_JSON) as f:
    clusters = json.load(f)

clusters_sorted = sorted(clusters, key=lambda c: c["strength"], reverse=True)

COMP_KEYS = ["mass_score", "coverage_score", "intensity_score",
             "coherence_score", "media_diversity"]


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Top 10 by Strength
# ══════════════════════════════════════════════════════════════════════════════
def fig_top10_strength():
    top10 = clusters_sorted[:10][::-1]

    fig, (ax_bar, ax_meta) = plt.subplots(
        1, 2, figsize=(14, 7), width_ratios=[3, 1.3],
        gridspec_kw={"wspace": 0.02},
    )

    y = np.arange(len(top10))
    bar_h = 0.55

    for i, c in enumerate(top10):
        sc = c["strength_components"]
        total = sum(sc[k] for k in COMP_KEYS)
        strength = c["strength"]
        left = 0.0
        for k in COMP_KEYS:
            width = (sc[k] / total) * strength if total > 0 else 0
            ax_bar.barh(y[i], width, left=left, height=bar_h,
                        color=COMP_COLORS[k], edgecolor="white", linewidth=0.5,
                        zorder=2)
            left += width

    for i, c in enumerate(top10):
        ax_bar.plot(c["strength"], y[i], "o", color="#222222", markersize=7.5, zorder=5)
        ax_bar.text(c["strength"] + 0.009, y[i], f'{c["strength"]:.3f}',
                    va="center", ha="left", fontsize=9, fontweight="bold", color="#222222")

    labels = [textwrap.fill(_get_name(c), width=35) for c in top10]
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=8.5, linespacing=1.15)
    ax_bar.set_xlim(0, 0.82)
    ax_bar.set_xlabel("Composite Strength Score", fontsize=10)
    ax_bar.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_bar.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax_bar.set_axisbelow(True)

    # Metadata panel
    ax_meta.set_xlim(0, 1)
    ax_meta.set_ylim(ax_bar.get_ylim())
    ax_meta.set_yticks([])
    ax_meta.spines["left"].set_visible(False)
    ax_meta.spines["bottom"].set_visible(False)
    ax_meta.set_xticks([])

    for i, c in enumerate(top10):
        tag = _type_tag(c["event_types"])
        year = c.get("year", "")
        date = c["peak_date"][:10]
        mass = int(c["total_mass"])
        multi = "*" if c["is_multi_type"] else "-"
        color = TYPE_COLORS.get(c["dominant_type"], "#666")

        ax_meta.text(0.05, y[i] + 0.12, f'{year}  {date}', fontsize=8, va="center",
                     color="#444444", fontweight="bold")
        ax_meta.text(0.05, y[i] - 0.18, f'{multi} {tag}  (m={mass})',
                     fontsize=7.2, va="center", color=color)

    ax_meta.text(0.05, y[-1] + 0.75, "Year / Peak date / Type (mass)",
                 fontsize=8, fontweight="bold", color="#666666", va="center")

    handles = [mpatches.Patch(facecolor=COMP_COLORS[k], edgecolor="white",
                              label=COMP_LABELS[k]) for k in COMP_KEYS]
    handles.append(plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="#222", markersize=7, label="Total"))
    ax_bar.legend(handles=handles, loc="lower right", frameon=True,
                  framealpha=0.92, edgecolor="#cccccc", ncol=3, fontsize=8)

    fig.suptitle(
        "Top 10 Event Clusters by Strength\n"
        "Canadian Climate Media Coverage, 1978-2024 (20,782 clusters total)",
        fontweight="bold", fontsize=13, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"top10_clusters_strength_all_years.{ext}")
    print(f"  [1/3] Saved top10_clusters_strength_all_years")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Top 5 per Event Type
# ══════════════════════════════════════════════════════════════════════════════
def fig_top_by_type():
    display_types = ["evt_policy", "evt_meeting", "evt_publication",
                     "evt_weather", "evt_judiciary", "evt_election", "evt_protest"]
    display_types = [t for t in display_types
                     if sum(1 for c in clusters if t in c["event_types"]) >= 5]

    n_types = len(display_types)
    n_cols = 2
    n_rows = (n_types + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.2 * n_rows))
    axes = axes.flatten()

    for idx, etype in enumerate(display_types):
        ax = axes[idx]
        typed = [c for c in clusters if etype in c["event_types"]]
        top5 = sorted(typed, key=lambda c: c["strength"], reverse=True)[:5][::-1]

        y_pos = np.arange(len(top5))
        bar_h = 0.50

        for i, c in enumerate(top5):
            sc = c["strength_components"]
            total = sum(sc[k] for k in COMP_KEYS)
            strength = c["strength"]
            left = 0.0
            for k in COMP_KEYS:
                width = (sc[k] / total) * strength if total > 0 else 0
                ax.barh(y_pos[i], width, left=left, height=bar_h,
                        color=COMP_COLORS[k], edgecolor="white", linewidth=0.3, zorder=2)
                left += width

        for i, c in enumerate(top5):
            ax.plot(c["strength"], y_pos[i], "o",
                    color=TYPE_COLORS.get(etype, "#333"), markersize=6, zorder=5)
            ax.text(c["strength"] + 0.006, y_pos[i], f'{c["strength"]:.3f}',
                    va="center", ha="left", fontsize=7.5, fontweight="bold", color="#333333")

        labels = []
        for c in top5:
            name = _get_name(c)
            wrapped = textwrap.fill(name, width=28)
            labels.append(f'{wrapped}\n({c.get("year", "")})')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7.5, linespacing=1.1)

        n_total = len(typed)
        ax.set_title(f'{TYPE_LABELS[etype]}  (n = {n_total:,})',
                     fontweight="bold", fontsize=10,
                     color=TYPE_COLORS.get(etype, "#333"))
        ax.set_xlim(0, 0.82)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.xaxis.grid(True, alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Strength")

    for idx in range(n_types, len(axes)):
        axes[idx].set_visible(False)

    handles = [mpatches.Patch(facecolor=COMP_COLORS[k], edgecolor="white",
                              label=COMP_LABELS[k]) for k in COMP_KEYS]
    fig.legend(handles=handles, loc="lower center", frameon=True,
               framealpha=0.92, edgecolor="#cccccc", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Top 5 Event Clusters per Type\n"
        "Canadian Climate Media Coverage, 1978-2024",
        fontweight="bold", fontsize=13, y=1.02,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"top5_clusters_by_type_all_years.{ext}")
    print(f"  [2/3] Saved top5_clusters_by_type_all_years")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Strength vs. Coherence scatter
# ══════════════════════════════════════════════════════════════════════════════
def fig_strength_coherence_scatter():
    fig, ax = plt.subplots(figsize=(12, 8))

    strengths = np.array([c["strength"] for c in clusters])
    coherences = np.array([c["strength_components"]["coherence_score"] for c in clusters])
    masses = np.array([c["total_mass"] for c in clusters])
    colors_bg = [TYPE_COLORS.get(c["dominant_type"], "#aaaaaa") for c in clusters]

    sizes_bg = np.clip(masses * 1.5, 6, 150)
    ax.scatter(strengths, coherences, s=sizes_bg, c=colors_bg,
               alpha=0.12, edgecolors="none", zorder=2)

    top10 = clusters_sorted[:10]
    label_offsets = [
        (-0.18, -0.08),  # #1
        (0.015, 0.04),   # #2
        (-0.20, -0.04),  # #3
        (0.015, -0.06),  # #4
        (-0.20, 0.02),   # #5
        (0.015, 0.04),   # #6
        (-0.18, -0.08),  # #7
        (0.015, 0.02),   # #8
        (-0.20, 0.04),   # #9
        (-0.18, -0.04),  # #10
    ]

    for idx, c in enumerate(top10):
        s = c["strength"]
        coh = c["strength_components"]["coherence_score"]
        m = c["total_mass"]
        color = TYPE_COLORS.get(c["dominant_type"], "#333333")

        ax.scatter(s, coh, s=max(50, m * 3), c=color,
                   edgecolors="#333333", linewidths=1.2, alpha=0.9, zorder=4)

        name = _get_name(c)
        name_short = textwrap.fill(name, width=28)

        dx, dy = label_offsets[idx] if idx < len(label_offsets) else (0.015, 0.015)
        ax.annotate(
            name_short,
            xy=(s, coh),
            xytext=(s + dx, coh + dy),
            fontsize=6.8, fontweight="bold", color="#222222",
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.7,
                            connectionstyle="arc3,rad=0.15"),
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#bbbbbb", alpha=0.9),
            verticalalignment="center",
        )

    ax.set_xlabel("Composite Strength", fontsize=11)
    ax.set_ylabel("Coherence Score (residual above corpus baseline)", fontsize=11)
    ax.set_title(
        "Event Cluster Landscape: Strength vs. Coherence, 1978-2024\n"
        f"20,782 clusters; bubble size = article mass; colour = dominant event type",
        fontweight="bold", fontsize=12, pad=12,
    )

    type_handles = [
        mpatches.Patch(facecolor=TYPE_COLORS[t], label=TYPE_LABELS[t], alpha=0.75)
        for t in ["evt_policy", "evt_meeting", "evt_publication",
                  "evt_weather", "evt_judiciary", "evt_election", "evt_protest"]
    ]
    ax.legend(handles=type_handles, loc="lower left", frameon=True,
              framealpha=0.92, edgecolor="#cccccc", fontsize=7.5, ncol=2)

    ax.xaxis.grid(True, alpha=0.2, linestyle="--")
    ax.yaxis.grid(True, alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xlim(0.15, 0.80)
    ax.set_ylim(0.10, 1.05)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"top10_strength_vs_coherence_all_years.{ext}")
    print(f"  [3/3] Saved top10_strength_vs_coherence_all_years")
    plt.close(fig)


if __name__ == "__main__":
    print(f"Generating top cluster figures for full database ({len(clusters):,} clusters)...")
    fig_top10_strength()
    fig_top_by_type()
    fig_strength_coherence_scatter()
    print("Done — all figures saved to results/figures/top/")
