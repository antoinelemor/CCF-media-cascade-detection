#!/usr/bin/env python3
"""Top event clusters visualizations for 2018.

Produces three publication-quality figures:
  1. Top 10 clusters by strength — horizontal stacked bar with component decomposition
  2. Top 5 clusters per event type — grouped panel (7 types)
  3. Strength vs. coherence scatter — all 611 clusters, top 10 labelled

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
CLUSTER_JSON = ROOT / "results" / "figures" / "event_clusters_2018.json"
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

# Component colours — scientifically inspired, colourblind-friendly
COMP_COLORS = {
    "mass_score":       "#2166AC",  # steel blue
    "coverage_score":   "#4393C3",  # mid blue
    "intensity_score":  "#F4A582",  # salmon
    "coherence_score":  "#D6604D",  # brick red
    "media_diversity":  "#92C5DE",  # sky blue
}
COMP_LABELS = {
    "mass_score":       "Mass",
    "coverage_score":   "Coverage",
    "intensity_score":  "Intensity",
    "coherence_score":  "Coherence",
    "media_diversity":  "Media div.",
}

# Event type aesthetics
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

# Cluster hand-curated names (from entity/date/type inspection)
CLUSTER_NAMES = {
    258: "G7 Summit, Charlevoix",
    586: "COP24 Outcomes, Katowice",
    187: "Trans Mountain Expansion Crisis",
    173: "CETA Debate, French Parliament",
    496: "Carbon Tax Confrontation (Ford vs. Trudeau)",
    402: "Trans Mountain Court Ruling",
    309: "UNESCO & Athabasca Oil Sands",
    323: "Summer Pipeline Hearings",
    520: "Nunavut Climate Policy Report",
    294: "Arctic Shipping & Northwest Passage Study",
    569: "Pre-COP24 Ministerial Meetings",
    591: "CIBC Energy Forum, Whistler",
    202: "Kinder Morgan Shareholder Vote",
    324: "Provincial Health & Climate Ministers' Meeting",
    448: "LNG Canada Investment Decision",
    338: "Ford Climate Rollback & Courts",
    121: "Burnaby Mountain Pipeline Protest",
    607: "Year-End Extreme Weather Review",
    588: "Winter Storm & Policy Response",
    485: "IPCC 1.5\u00b0C Special Report",
    11:  "Early 2018 Cold Snap Studies",
    342: "Ontario & Alberta Wildfire Season",
    510: "House of Commons Carbon Debate",
    603: "Green Party Surge Polling",
    364: "Bernier & Conservative Climate Split",
    59:  "Provincial Election Climate Platforms",
    383: "Australia Climate Politics Spillover",
    567: "Gilets Jaunes & Carbon Tax Backlash",
    552: "Paris Gilets Jaunes Escalation",
    42:  "Early Pipeline Protests",
    276: "Summer Pipeline Protests",
    164: "Ontario Climate Plan Rollback",
}


def _get_name(cluster_id: int, fallback_types: list, peak_date: str) -> str:
    if cluster_id in CLUSTER_NAMES:
        return CLUSTER_NAMES[cluster_id]
    types_str = " + ".join(TYPE_LABELS.get(t, t.replace("evt_", "")) for t in fallback_types)
    return f"{types_str} ({peak_date[:10]})"


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
    top10 = clusters_sorted[:10][::-1]  # bottom-to-top

    fig, (ax_bar, ax_meta) = plt.subplots(
        1, 2, figsize=(13, 6.5), width_ratios=[3, 1.2],
        gridspec_kw={"wspace": 0.02},
    )

    y = np.arange(len(top10))
    bar_h = 0.55

    # ── Stacked bars ──
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

    # Lollipop dots + values
    for i, c in enumerate(top10):
        ax_bar.plot(c["strength"], y[i], "o", color="#222222", markersize=7.5,
                    zorder=5)
        ax_bar.text(c["strength"] + 0.009, y[i], f'{c["strength"]:.3f}',
                    va="center", ha="left", fontsize=9, fontweight="bold",
                    color="#222222")

    # Y-axis: cluster names
    labels = [_get_name(c["cluster_id"], c["event_types"], c["peak_date"])
              for c in top10]
    # Wrap long names
    labels = [textwrap.fill(l, width=30) for l in labels]
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=9, linespacing=1.15)

    ax_bar.set_xlim(0, 0.78)
    ax_bar.set_xlabel("Composite Strength Score", fontsize=10)
    ax_bar.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_bar.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax_bar.set_axisbelow(True)

    # ── Metadata panel (right): date + type + mass ──
    ax_meta.set_xlim(0, 1)
    ax_meta.set_ylim(ax_bar.get_ylim())
    ax_meta.set_yticks([])
    ax_meta.spines["left"].set_visible(False)
    ax_meta.spines["bottom"].set_visible(False)
    ax_meta.set_xticks([])

    for i, c in enumerate(top10):
        tag = _type_tag(c["event_types"])
        date = c["peak_date"][:10]
        mass = int(c["total_mass"])
        multi = "*" if c["is_multi_type"] else "-"
        color = TYPE_COLORS.get(c["dominant_type"], "#666")

        ax_meta.text(0.05, y[i] + 0.12, date, fontsize=8, va="center",
                     color="#444444", fontweight="bold")
        ax_meta.text(0.05, y[i] - 0.18, f'{multi} {tag}  (m={mass})',
                     fontsize=7.2, va="center", color=color)

    # Header for meta panel
    ax_meta.text(0.05, y[-1] + 0.75, "Peak date / Type (mass)",
                 fontsize=8, fontweight="bold", color="#666666", va="center")

    # Legend
    handles = [mpatches.Patch(facecolor=COMP_COLORS[k], edgecolor="white",
                              label=COMP_LABELS[k]) for k in COMP_KEYS]
    handles.append(plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="#222", markersize=7,
                              label="Total"))
    ax_bar.legend(handles=handles, loc="lower right", frameon=True,
                  framealpha=0.92, edgecolor="#cccccc", ncol=3, fontsize=8)

    fig.suptitle("Top 10 Event Clusters by Strength\nCanada Climate Media Coverage, 2018",
                 fontweight="bold", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"top10_clusters_strength.{ext}")
    print(f"  [1/3] Saved top10_clusters_strength")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Top 5 per Event Type
# ══════════════════════════════════════════════════════════════════════════════
def fig_top_by_type():
    display_types = ["evt_policy", "evt_meeting", "evt_publication",
                     "evt_weather", "evt_judiciary", "evt_election", "evt_protest"]
    display_types = [t for t in display_types
                     if sum(1 for c in clusters if t in c["event_types"]) >= 3]

    n_types = len(display_types)
    n_cols = 2
    n_rows = (n_types + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.0 * n_rows))
    axes = axes.flatten()

    for idx, etype in enumerate(display_types):
        ax = axes[idx]
        typed = [c for c in clusters if etype in c["event_types"]]
        top5 = sorted(typed, key=lambda c: c["strength"], reverse=True)[:5][::-1]

        y = np.arange(len(top5))
        bar_h = 0.50

        for i, c in enumerate(top5):
            sc = c["strength_components"]
            total = sum(sc[k] for k in COMP_KEYS)
            strength = c["strength"]
            left = 0.0
            for k in COMP_KEYS:
                width = (sc[k] / total) * strength if total > 0 else 0
                ax.barh(y[i], width, left=left, height=bar_h,
                        color=COMP_COLORS[k], edgecolor="white", linewidth=0.3,
                        zorder=2)
                left += width

        # Dots + values
        for i, c in enumerate(top5):
            ax.plot(c["strength"], y[i], "o",
                    color=TYPE_COLORS.get(etype, "#333"), markersize=6, zorder=5)
            ax.text(c["strength"] + 0.006, y[i], f'{c["strength"]:.3f}',
                    va="center", ha="left", fontsize=7.5, fontweight="bold",
                    color="#333333")

        # Y labels: name + date
        labels = []
        for c in top5:
            name = _get_name(c["cluster_id"], c["event_types"], c["peak_date"])
            wrapped = textwrap.fill(name, width=25)
            labels.append(f'{wrapped}\n({c["peak_date"][:10]})')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5, linespacing=1.1)

        n_total = len(typed)
        type_color = TYPE_COLORS.get(etype, "#333")
        ax.set_title(f'{TYPE_LABELS[etype]}  (n = {n_total})',
                     fontweight="bold", fontsize=10, color=type_color)
        ax.set_xlim(0, 0.80)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.xaxis.grid(True, alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Strength")

    # Hide extra axes
    for idx in range(n_types, len(axes)):
        axes[idx].set_visible(False)

    # Legend
    handles = [mpatches.Patch(facecolor=COMP_COLORS[k], edgecolor="white",
                              label=COMP_LABELS[k]) for k in COMP_KEYS]
    fig.legend(handles=handles, loc="lower center", frameon=True,
               framealpha=0.92, edgecolor="#cccccc", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Top 5 Event Clusters per Type\nCanada Climate Media Coverage, 2018",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"top5_clusters_by_type.{ext}")
    print(f"  [2/3] Saved top5_clusters_by_type")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Strength vs. Coherence scatter (all clusters, top 10 labelled)
# ══════════════════════════════════════════════════════════════════════════════
def fig_strength_coherence_scatter():
    fig, ax = plt.subplots(figsize=(11, 7.5))

    strengths = np.array([c["strength"] for c in clusters])
    coherences = np.array([c["strength_components"]["coherence_score"] for c in clusters])
    masses = np.array([c["total_mass"] for c in clusters])
    colors_bg = [TYPE_COLORS.get(c["dominant_type"], "#aaaaaa") for c in clusters]

    # Background — all clusters
    sizes_bg = np.clip(masses * 2, 10, 200)
    ax.scatter(strengths, coherences, s=sizes_bg, c=colors_bg,
               alpha=0.18, edgecolors="none", zorder=2)

    # Top 10 — highlighted with labels
    top10 = clusters_sorted[:10]

    # Hand-tuned label positions to avoid overlap
    # (x_offset, y_offset) in data coords relative to point
    label_offsets = {
        258: (-0.18, -0.12),   # G7 Summit — bottom-left, away from cluster
        586: (-0.22, 0.02),    # COP24 — left
        187: (0.02, 0.05),     # Trans Mtn Crisis — top right
        173: (-0.22, -0.04),   # CETA — left
        496: (0.02, -0.07),    # Carbon Tax — below right
        402: (0.02, 0.04),     # Trans Mtn Court — top right
        309: (-0.22, -0.01),   # UNESCO — left
        323: (-0.20, -0.08),   # Summer Hearings — bottom left
        520: (-0.24, 0.04),    # Nunavut — far left
        294: (-0.18, 0.04),    # Arctic Shipping — left up
    }

    for c in top10:
        s = c["strength"]
        coh = c["strength_components"]["coherence_score"]
        m = c["total_mass"]
        color = TYPE_COLORS.get(c["dominant_type"], "#333333")
        cid = c["cluster_id"]

        ax.scatter(s, coh, s=max(50, m * 3.5), c=color,
                   edgecolors="#333333", linewidths=1.2, alpha=0.9, zorder=4)

        name = _get_name(cid, c["event_types"], c["peak_date"])
        name_short = textwrap.fill(name, width=28)

        dx, dy = label_offsets.get(cid, (0.015, 0.015))
        ax.annotate(
            name_short,
            xy=(s, coh),
            xytext=(s + dx, coh + dy),
            fontsize=7.2, fontweight="bold", color="#222222",
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
        "Event Cluster Landscape — Strength vs. Coherence, 2018\n"
        "Bubble size = total article mass; colour = dominant event type",
        fontweight="bold", fontsize=12, pad=12,
    )

    # Type legend
    type_handles = [
        mpatches.Patch(facecolor=TYPE_COLORS[t], label=TYPE_LABELS[t], alpha=0.75)
        for t in ["evt_policy", "evt_meeting", "evt_publication",
                  "evt_weather", "evt_judiciary", "evt_election", "evt_protest"]
    ]
    # Size legend
    for sz_val, sz_label in [(5, "5"), (20, "20"), (45, "45")]:
        type_handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#cccccc", markeredgecolor="#888888",
                       markersize=np.sqrt(sz_val * 3.5),
                       label=f"mass = {sz_label}")
        )

    ax.legend(handles=type_handles, loc="lower left", frameon=True,
              framealpha=0.92, edgecolor="#cccccc", fontsize=7.5, ncol=2,
              columnspacing=1.0, handletextpad=0.5)

    ax.xaxis.grid(True, alpha=0.2, linestyle="--")
    ax.yaxis.grid(True, alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xlim(0.15, 0.78)
    ax.set_ylim(0.15, 1.05)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"top10_strength_vs_coherence.{ext}")
    print(f"  [3/3] Saved top10_strength_vs_coherence")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating top cluster figures for 2018...")
    fig_top10_strength()
    fig_top_by_type()
    fig_strength_coherence_scatter()
    print("Done — all figures saved to results/figures/top/")
