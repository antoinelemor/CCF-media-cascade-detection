#!/usr/bin/env python3
"""
Publication-quality figures for the 2018 paradigm shift analysis.

Generates 3 figures:
  1. paradigm_timeline_2018.png  -- Multi-panel paradigm overview
  2. episode_dynamics_2018.png   -- Episode-level summary
  3. shift_dynamics_2018.png     -- Shift-level dynamics scatter

Data sources:
  - results/production/2018/paradigm_shifts/shifts.json
  - results/production/2018/paradigm_shifts/episodes.json
  - results/production/2018/paradigm_shifts/paradigm_timeline.parquet
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "results" / "production" / "2018" / "paradigm_shifts"
OUT_DIR = DATA_DIR  # Save figures alongside the data

SHIFTS_PATH = DATA_DIR / "shifts.json"
EPISODES_PATH = DATA_DIR / "episodes.json"
TIMELINE_PATH = DATA_DIR / "paradigm_timeline.parquet"

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
FRAMES = ["Cult", "Eco", "Envt", "Pbh", "Just", "Pol", "Sci", "Secu"]

# Professional 8-color palette (colorblind-friendly, muted tones)
FRAME_COLORS = {
    "Cult": "#E07B39",   # burnt orange
    "Eco":  "#2D8E5F",   # forest green
    "Envt": "#5BAE5B",   # leaf green
    "Pbh":  "#C75B7A",   # rose
    "Just": "#8B6DB0",   # lavender purple
    "Pol":  "#3578B2",   # steel blue
    "Sci":  "#D4A933",   # goldenrod
    "Secu": "#7A7A7A",   # slate grey
}

PARADIGM_TYPE_COLORS = {
    "Mono-paradigm":   "#1b9e77",
    "Dual-paradigm":   "#d95f02",
    "Triple-paradigm": "#7570b3",
    "Quad-paradigm":   "#e7298a",
}

SHIFT_TYPE_COLORS = {
    "frame_entry":      "#2ca02c",   # green
    "frame_exit":       "#d62728",   # red
    "recomposition":    "#1f77b4",   # blue
    "full_replacement": "#9467bd",   # purple
}

EPISODE_FILL = "#d0d0d0"
EPISODE_ALPHA = 0.18

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_data():
    """Load all three data sources and return (timeline_df, shifts, episodes)."""
    timeline = pd.read_parquet(TIMELINE_PATH)
    timeline["date"] = pd.to_datetime(timeline["date"])
    timeline.sort_values("date", inplace=True)

    with open(SHIFTS_PATH) as f:
        shifts = json.load(f)
    for s in shifts:
        s["shift_date"] = pd.Timestamp(s["shift_date"])

    with open(EPISODES_PATH) as f:
        episodes = json.load(f)
    for ep in episodes:
        ep["start_date"] = pd.Timestamp(ep["start_date"])
        ep["end_date"] = pd.Timestamp(ep["end_date"])

    return timeline, shifts, episodes


def add_episode_shading(ax, episodes, label_y=None):
    """Add semi-transparent episode regions + labels to an axes."""
    for i, ep in enumerate(episodes):
        ax.axvspan(
            ep["start_date"], ep["end_date"],
            color=EPISODE_FILL, alpha=EPISODE_ALPHA, zorder=0,
            linewidth=0,
        )
        # Label at the top
        mid = ep["start_date"] + (ep["end_date"] - ep["start_date"]) / 2
        if label_y is not None:
            ax.text(
                mid, label_y, f"E{i+1}",
                ha="center", va="bottom", fontsize=7.5,
                fontstyle="italic", color="#555555",
            )


# ===========================================================================
# Figure 1: Multi-panel paradigm overview
# ===========================================================================
def figure_1_paradigm_timeline(timeline, shifts, episodes):
    """Stacked area + paradigm type bar + shift magnitude markers."""
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 10), dpi=300,
        gridspec_kw={"height_ratios": [5, 1, 2.5], "hspace": 0.12},
        sharex=True,
    )
    ax_area, ax_type, ax_mag = axes
    dates = timeline["date"]

    # --- Panel A: Stacked area chart of paradigm dominance scores -----------
    cols = [f"paradigm_{f}" for f in FRAMES]
    values = timeline[cols].values  # (281, 8)

    # Stacked area
    ax_area.stackplot(
        dates, values.T,
        labels=FRAMES,
        colors=[FRAME_COLORS[f] for f in FRAMES],
        alpha=0.85,
        linewidth=0.3,
        edgecolor="white",
    )
    add_episode_shading(ax_area, episodes, label_y=values.sum(axis=1).max() * 0.97)
    ax_area.set_ylabel("Paradigm dominance score")
    ax_area.set_title("A.  Paradigm composition over time (2018)", loc="left", fontweight="bold")
    # Legend: two rows at top-right
    handles, labels = ax_area.get_legend_handles_labels()
    ax_area.legend(handles, labels, loc="upper right", ncol=4, frameon=True,
                   framealpha=0.9, edgecolor="#cccccc", fancybox=False)
    ax_area.set_xlim(dates.iloc[0], dates.iloc[-1])

    # --- Panel B: Paradigm type horizontal color bar -----------------------
    type_map = PARADIGM_TYPE_COLORS
    ptype_series = timeline["paradigm_type"]
    # Draw colored rectangles
    for idx in range(len(dates)):
        ptype = ptype_series.iloc[idx]
        color = type_map.get(ptype, "#cccccc")
        d = dates.iloc[idx]
        width = timedelta(days=1)
        ax_type.barh(0, width, left=d, height=1.0, color=color, linewidth=0, align="center")

    ax_type.set_ylim(-0.6, 0.6)
    ax_type.set_yticks([])
    ax_type.set_title("B.  Paradigm type", loc="left", fontweight="bold")
    add_episode_shading(ax_type, episodes)

    # Legend for paradigm types
    ptype_handles = [
        mpatches.Patch(color=c, label=t.replace("-paradigm", ""))
        for t, c in type_map.items()
        if t in ptype_series.values
    ]
    ax_type.legend(handles=ptype_handles, loc="upper right", ncol=4,
                   frameon=True, framealpha=0.9, edgecolor="#cccccc", fancybox=False)

    # Remove top/right spine for type bar; also remove left
    ax_type.spines["left"].set_visible(False)
    ax_type.spines["bottom"].set_visible(False)

    # --- Panel C: Shift magnitude markers ----------------------------------
    for s in shifts:
        color = SHIFT_TYPE_COLORS.get(s["shift_type"], "#333333")
        ax_mag.scatter(
            s["shift_date"], s["shift_magnitude"],
            c=color, s=40, edgecolors="white", linewidths=0.4,
            zorder=5, alpha=0.9,
        )
    add_episode_shading(ax_mag, episodes, label_y=max(s["shift_magnitude"] for s in shifts) * 0.95)
    ax_mag.set_ylabel("Shift magnitude")
    ax_mag.set_title("C.  Paradigm shifts", loc="left", fontweight="bold")

    # Shift type legend
    # Only include types that appear in the data
    present_types = set(s["shift_type"] for s in shifts)
    st_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=7,
               label=t.replace("_", " ").title(), linewidth=0)
        for t, c in SHIFT_TYPE_COLORS.items()
        if t in present_types
    ]
    ax_mag.legend(handles=st_handles, loc="upper right", ncol=4,
                  frameon=True, framealpha=0.9, edgecolor="#cccccc", fancybox=False)

    # X-axis formatting
    ax_mag.xaxis.set_major_locator(mdates.MonthLocator())
    ax_mag.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_mag.tick_params(axis="x", rotation=0)

    fig.suptitle("Paradigm Shift Analysis -- 2018", fontsize=14, fontweight="bold", y=0.995)
    fig.align_ylabels(axes)

    outpath = OUT_DIR / "paradigm_timeline_2018.png"
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ===========================================================================
# Figure 2: Episode-level summary
# ===========================================================================
def figure_2_episode_dynamics(episodes, shifts):
    """Horizontal timeline + annotations for each episode."""
    n_ep = len(episodes)
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

    y_positions = list(range(n_ep - 1, -1, -1))  # E1 at top
    bar_height = 0.45

    # Global date range for axis
    date_min = min(ep["start_date"] for ep in episodes) - timedelta(days=15)
    date_max = max(ep["end_date"] for ep in episodes) + timedelta(days=65)

    for i, ep in enumerate(episodes):
        y = y_positions[i]
        start = ep["start_date"]
        end = ep["end_date"]
        duration = ep["duration_days"]
        n_shifts_ep = ep["n_shifts"]

        # Duration bar
        ax.barh(
            y, (end - start).days, left=start, height=bar_height,
            color="#4a90d9", alpha=0.7, edgecolor="#2c5f8a", linewidth=0.8,
            zorder=3,
        )

        # Stability-after indicator (lighter shade to the right)
        stab = ep["regime_after_duration_days"]
        if stab > 0:
            stab_start = end
            stab_end = end + timedelta(days=stab)
            ax.barh(
                y, stab, left=stab_start, height=bar_height * 0.5,
                color="#a0c4e8", alpha=0.5, edgecolor="#6699cc", linewidth=0.5,
                zorder=2,
            )
            ax.text(
                stab_end + timedelta(days=1), y,
                f"stable {stab}d",
                va="center", ha="left", fontsize=7.5, color="#446688",
            )
        else:
            ax.text(
                end + timedelta(days=2), y,
                f"unstable ({stab}d)",
                va="center", ha="left", fontsize=7.5, color="#cc4444",
            )

        # Episode label on the left
        ax.text(
            start - timedelta(days=5), y,
            f"E{i+1}",
            va="center", ha="right", fontsize=11, fontweight="bold", color="#333333",
        )

        # Duration + shifts inside bar
        mid_date = start + (end - start) / 2
        # For short episodes, text might not fit -- adjust fontsize
        text_fs = 8.5 if duration > 10 else 7
        ax.text(
            mid_date, y + 0.01,
            f"{duration}d / {n_shifts_ep} shifts",
            va="center", ha="center", fontsize=text_fs, fontweight="bold",
            color="white", zorder=5,
        )

        # Before/after frames
        frames_before = ep["dominant_frames_before"]
        frames_after = ep["dominant_frames_after"]
        before_label = "[" + ",".join(frames_before) + "]"
        after_label = "[" + ",".join(frames_after) + "]"

        ax.text(
            start, y + bar_height / 2 + 0.12,
            before_label,
            va="bottom", ha="left", fontsize=7.5,
            color="#333333", fontstyle="italic",
        )
        ax.text(
            end, y + bar_height / 2 + 0.12,
            after_label,
            va="bottom", ha="right", fontsize=7.5,
            color="#333333", fontstyle="italic",
        )
        # Arrow between before and after labels
        arrow_y = y + bar_height / 2 + 0.22
        if duration > 5:
            ax.annotate(
                "",
                xy=(end - timedelta(days=1), arrow_y),
                xytext=(start + timedelta(days=1), arrow_y),
                arrowprops=dict(
                    arrowstyle="->", color="#666666", lw=1.0,
                ),
            )

        # Reversibility + complexity below bar
        rev_marker = "Yes" if ep["reversible"] else "No"
        rev_color = "#2ca02c" if ep["reversible"] else "#d62728"
        ax.text(
            mid_date, y - bar_height / 2 - 0.08,
            f"Rev: ",
            va="top", ha="right", fontsize=7.5, color="#555555",
        )
        # Colored checkmark/X
        ax.text(
            mid_date, y - bar_height / 2 - 0.08,
            rev_marker,
            va="top", ha="left", fontsize=7.5, color=rev_color, fontweight="bold",
        )
        # Max complexity after the marker
        offset_date = mid_date + timedelta(days=4)
        ax.text(
            offset_date, y - bar_height / 2 - 0.08,
            f"   Max complexity: {ep['max_complexity']}",
            va="top", ha="left", fontsize=7.5, color="#555555",
        )

    # Axes
    ax.set_xlim(date_min, date_max)
    ax.set_ylim(-0.8, n_ep - 0.2)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=0)
    ax.set_xlabel("Date")
    ax.set_title("Episode Dynamics -- 2018 Paradigm Shift Analysis",
                  fontweight="bold", fontsize=13)

    # Legend for bar elements
    legend_handles = [
        mpatches.Patch(color="#4a90d9", alpha=0.7, label="Episode duration"),
        mpatches.Patch(color="#a0c4e8", alpha=0.5, label="Post-episode stability"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True,
              framealpha=0.9, edgecolor="#cccccc", fancybox=False)

    ax.spines["left"].set_visible(False)

    outpath = OUT_DIR / "episode_dynamics_2018.png"
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ===========================================================================
# Figure 3: Shift-level dynamics scatter
# ===========================================================================
def figure_3_shift_dynamics(shifts, episodes):
    """Scatter plot: date vs regime_duration_days, colored by structural_change."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

    dates = [s["shift_date"] for s in shifts]
    durations = [s["regime_duration_days"] for s in shifts]
    magnitudes = [s["shift_magnitude"] for s in shifts]
    struct_change = [s["structural_change"] for s in shifts]
    reversible = [s["reversible"] for s in shifts]

    # Normalize magnitudes to reasonable marker sizes (50-250)
    mag_arr = np.array(magnitudes)
    mag_norm = (mag_arr - mag_arr.min()) / (mag_arr.max() - mag_arr.min() + 1e-9)
    sizes = 50 + mag_norm * 200

    # Diverging colormap: red for negative, blue for positive
    cmap = plt.cm.RdBu
    sc_arr = np.array(struct_change, dtype=float)
    abs_max = max(abs(sc_arr.min()), abs(sc_arr.max()))
    if abs_max == 0:
        abs_max = 1
    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)

    # Episode shading first (so scatter is on top)
    add_episode_shading(ax, episodes)

    # Plot each point; use distinct markers for reversible vs not
    for idx in range(len(shifts)):
        marker = "o" if not reversible[idx] else "X"
        edgecolor = "white" if not reversible[idx] else "#333333"
        lw = 0.4 if not reversible[idx] else 0.8
        ax.scatter(
            dates[idx], durations[idx],
            c=[struct_change[idx]], cmap=cmap, norm=norm,
            s=sizes[idx], marker=marker,
            edgecolors=edgecolor, linewidths=lw,
            zorder=5, alpha=0.88,
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30, shrink=0.8)
    cbar.set_label("Structural change", fontsize=10)
    cbar.outline.set_visible(False)

    # Reference line at y=0
    ax.axhline(0, color="#cccccc", linewidth=0.6, zorder=1)

    ax.set_ylabel("Regime duration (days)")
    ax.set_xlabel("Shift date")
    ax.set_title("Shift Dynamics -- 2018 Paradigm Shift Analysis",
                  fontweight="bold", fontsize=13)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=0)

    # Episode labels
    ymax_final = ax.get_ylim()[1]
    for i, ep in enumerate(episodes):
        mid = ep["start_date"] + (ep["end_date"] - ep["start_date"]) / 2
        ax.text(
            mid, ymax_final * 0.93,
            f"E{i+1}",
            ha="center", va="top", fontsize=8,
            fontstyle="italic", color="#555555",
            zorder=10,
        )

    # Legend for marker types and sizes
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markersize=8, label="Non-reversible", linewidth=0,
               markeredgecolor="white", markeredgewidth=0.4),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#888888",
               markersize=9, label="Reversible", linewidth=0,
               markeredgecolor="#333333", markeredgewidth=0.8),
    ]
    # Size legend (small, medium, large)
    for label, frac in [("Low magnitude", 0.0), ("Med magnitude", 0.5), ("High magnitude", 1.0)]:
        s = 50 + frac * 200
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa",
                   markersize=np.sqrt(s) / 1.8, label=label, linewidth=0,
                   markeredgecolor="white", markeredgewidth=0.3),
        )

    ax.legend(handles=legend_handles, loc="upper left", frameon=True,
              framealpha=0.9, edgecolor="#cccccc", fancybox=False, ncol=1)

    outpath = OUT_DIR / "shift_dynamics_2018.png"
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ===========================================================================
# Figure 4: Three-role cascade attribution
# ===========================================================================

ROLE_COLORS = {
    "amplification":   "#2ca02c",
    "destabilisation": "#d62728",
    "dormante":        "#7f7f7f",
}

ROLE_LABELS = {
    "amplification":   "Amplification",
    "destabilisation": "Déstabilisation",
    "dormante":        "Dormante",
}


def figure_4_role_attribution(shifts, cascades_path, episodes):
    """Three-panel figure: impact space scatter, role pie, direction alignment bar.

    Parameters:
        shifts: list of shift dicts (from shifts.json)
        cascades_path: Path to cascades.json for article counts
        episodes: list of episode dicts (for reference)
    """
    # Collect unique attributed cascades
    seen = set()
    unique_attr = []
    for s in shifts:
        for ac in s.get("attributed_cascades", []):
            cid = ac.get("cascade_id", "")
            if cid not in seen:
                seen.add(cid)
                unique_attr.append(dict(ac))

    if not unique_attr:
        print("  [SKIP] No attributed cascades for role figure")
        return

    # Load cascade details for article counts
    cascade_lookup = {}
    if cascades_path.exists():
        with open(cascades_path) as f:
            for c in json.load(f):
                cascade_lookup[f"{c['frame']}_{c['onset_date']}"] = c

    fig = plt.figure(figsize=(16, 10), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.35, wspace=0.30)
    ax_scatter = fig.add_subplot(gs[0, :])
    ax_pie = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])

    # --- Panel A: Impact space scatter (own_lift vs structural_impact) ---
    # Separate amplification markers by direction_alignment
    AMP_DA_MARKERS = {1.0: ("D", "Promotion"), 0.7: ("^", "Consolidation"), 0.3: ("v", "Insufficient")}
    AMP_DA_COLORS = {1.0: "#1a7a1a", 0.7: "#2ca02c", 0.3: "#7fbf7f"}

    for ac in unique_attr:
        role = ac.get("role", "dormante")
        own_lift = ac.get("own_lift", 0)
        struct = ac.get("structural_impact", 0)
        score = ac.get("total_score", 0)
        size = 60 + 120 * score

        if role == "amplification":
            da = ac.get("direction_alignment", 0.3)
            marker = AMP_DA_MARKERS.get(da, AMP_DA_MARKERS[0.3])[0]
            color = AMP_DA_COLORS.get(da, AMP_DA_COLORS[0.3])
            ax_scatter.scatter(own_lift, struct, c=color, marker=marker, s=size,
                               alpha=0.85, edgecolors="white", linewidths=0.6, zorder=6)
        elif role == "destabilisation":
            ax_scatter.scatter(own_lift, struct, c=ROLE_COLORS["destabilisation"],
                               marker="s", s=size, alpha=0.7,
                               edgecolors="white", linewidths=0.5, zorder=5)
        else:
            ax_scatter.scatter(own_lift, struct, c=ROLE_COLORS["dormante"],
                               marker="o", s=size, alpha=0.5,
                               edgecolors="white", linewidths=0.5, zorder=4)

    # Threshold lines
    ax_scatter.axvline(0.05, color="#2ca02c", linestyle="--", alpha=0.4, linewidth=1)
    ax_scatter.axhline(0.01, color="#d62728", linestyle="--", alpha=0.4, linewidth=1)

    # Threshold region labels
    lifts = [ac.get("own_lift", 0) for ac in unique_attr]
    structs = [ac.get("structural_impact", 0) for ac in unique_attr]
    lift_margin = max(0.02, (max(lifts) - min(lifts)) * 0.15) if lifts else 0.02
    struct_margin = max(0.002, max(structs) * 0.15) if structs else 0.002
    ax_scatter.set_xlim(min(lifts) - lift_margin, max(lifts) + lift_margin)
    ax_scatter.set_ylim(-struct_margin, max(structs) + struct_margin)

    # Region labels
    xmax = max(lifts) + lift_margin
    ymax = max(structs) + struct_margin
    ax_scatter.text(xmax * 0.85, ymax * 0.92, "AMPLIFICATION",
                    ha="center", va="center", fontsize=8, color="#2ca02c",
                    alpha=0.5, fontstyle="italic")
    ax_scatter.text(min(lifts) * 0.5, ymax * 0.92, "DESTABILISATION",
                    ha="center", va="center", fontsize=8, color="#d62728",
                    alpha=0.5, fontstyle="italic")
    ax_scatter.text(0.0, -struct_margin * 0.3, "DORMANTE",
                    ha="center", va="center", fontsize=8, color="#7f7f7f",
                    alpha=0.5, fontstyle="italic")

    # Legend
    from matplotlib.lines import Line2D as L2D
    role_handles = [
        L2D([0], [0], marker="D", color="w", markerfacecolor=AMP_DA_COLORS[1.0],
            markersize=9, label="Amplification: promotion", linewidth=0),
        L2D([0], [0], marker="^", color="w", markerfacecolor=AMP_DA_COLORS[0.7],
            markersize=9, label="Amplification: consolidation", linewidth=0),
        L2D([0], [0], marker="v", color="w", markerfacecolor=AMP_DA_COLORS[0.3],
            markersize=9, label="Amplification: insufficient", linewidth=0),
        L2D([0], [0], marker="s", color="w", markerfacecolor=ROLE_COLORS["destabilisation"],
            markersize=8, label="Destabilisation", linewidth=0),
        L2D([0], [0], marker="o", color="w", markerfacecolor=ROLE_COLORS["dormante"],
            markersize=8, label="Dormante", linewidth=0),
    ]
    ax_scatter.legend(handles=role_handles, loc="upper left", frameon=True,
                      framealpha=0.9, edgecolor="#cccccc", ncol=2)

    ax_scatter.set_xlabel("Own lift (frame dominance change)")
    ax_scatter.set_ylabel("Structural impact (cosine distance)")
    ax_scatter.set_title("A.  Impact space — cascade role assignment", loc="left", fontweight="bold")

    # --- Panel B: Role distribution pie ---
    role_counts = {}
    for ac in unique_attr:
        role = ac.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1

    roles = ["amplification", "destabilisation", "dormante"]
    counts = [role_counts.get(r, 0) for r in roles]
    colors = [ROLE_COLORS[r] for r in roles]
    labels = [f"{ROLE_LABELS[r]}\n(n={c})" for r, c in zip(roles, counts)]

    total = sum(counts)
    if total > 0:
        wedges, texts, autotexts = ax_pie.pie(
            counts, labels=labels, colors=colors, autopct="%1.0f%%",
            startangle=90, textprops={"fontsize": 10},
            pctdistance=0.75, labeldistance=1.15,
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight("bold")

    ax_pie.set_title("B.  Role distribution (unique cascades)", loc="left", fontweight="bold")

    # --- Panel C: Direction alignment bar for amplification cascades ---
    amp_cascades = [ac for ac in unique_attr if ac.get("role") == "amplification"]
    if amp_cascades:
        amp_cascades.sort(key=lambda x: -x.get("own_lift", 0))
        bar_labels = []
        bar_lifts = []
        bar_colors = []
        bar_da = []
        for ac in amp_cascades:
            da = ac.get("direction_alignment", 0.3)
            cid = ac.get("cascade_id", "?")
            # Get article count
            short_id = "_".join(cid.split("_")[:2])
            cascade_data = cascade_lookup.get(short_id, {})
            n_art = cascade_data.get("n_articles", "?")
            bar_labels.append(f"{short_id}\n({n_art} art.)")
            bar_lifts.append(ac.get("own_lift", 0))
            bar_colors.append(AMP_DA_COLORS.get(da, AMP_DA_COLORS[0.3]))
            da_name = {1.0: "Promotion", 0.7: "Consolidation", 0.3: "Insufficient"}.get(da, "?")
            bar_da.append(da_name)

        y_pos = range(len(amp_cascades))
        bars = ax_bar.barh(y_pos, bar_lifts, color=bar_colors, edgecolor="white",
                           linewidth=0.5, height=0.6, zorder=3)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(bar_labels, fontsize=8)
        ax_bar.axvline(0.05, color="#2ca02c", linestyle="--", alpha=0.4, linewidth=1)

        # Add direction_alignment label on bars
        for i, (bar, da_name) in enumerate(zip(bars, bar_da)):
            x_pos = bar.get_width() + 0.003
            ax_bar.text(x_pos, i, da_name, va="center", ha="left", fontsize=8,
                        fontstyle="italic", color="#444444")

        ax_bar.set_xlabel("Own lift")
        ax_bar.invert_yaxis()
    else:
        ax_bar.text(0.5, 0.5, "No amplification cascades", ha="center", va="center",
                    transform=ax_bar.transAxes, fontsize=11, color="#999")

    ax_bar.set_title("C.  Amplification mechanisms", loc="left", fontweight="bold")

    fig.suptitle("Three-Role Cascade Attribution -- 2018", fontsize=14,
                 fontweight="bold", y=0.995)

    outpath = OUT_DIR / "role_attribution_2018.png"
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading data...")
    timeline, shifts, episodes = load_data()
    print(f"  Timeline: {len(timeline)} rows, Shifts: {len(shifts)}, Episodes: {len(episodes)}")

    print("\nGenerating Figure 1: Paradigm timeline...")
    figure_1_paradigm_timeline(timeline, shifts, episodes)

    print("Generating Figure 2: Episode dynamics...")
    figure_2_episode_dynamics(episodes, shifts)

    print("Generating Figure 3: Shift dynamics...")
    figure_3_shift_dynamics(shifts, episodes)

    print("Generating Figure 4: Role attribution...")
    cascades_path = PROJECT_ROOT / "results" / "production" / "2018" / "cascades.json"
    figure_4_role_attribution(shifts, cascades_path, episodes)

    print("\nAll figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
