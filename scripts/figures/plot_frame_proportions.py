#!/usr/bin/env python3
"""
Generate fig_frame_proportions_cascades.png — Daily Mean Frame Proportion
per Article with cascade overlay zones (red).

Reads cascade results from results/cascades_2018.json and frame proportions
from the database. Produces an 8-panel figure (one per frame).

Usage:
    python scripts/plot_frame_proportions.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.constants import FRAMES, FRAME_COLUMNS, FRAME_COLORS
from cascade_detector.data.connector import DatabaseConnector
from cascade_detector.data.processor import DataProcessor
from cascade_detector.indexing.index_manager import IndexManager

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # =========================================================================
    # 1. Load cascade results
    # =========================================================================
    json_path = PROJECT_ROOT / 'results' / 'cascades_2018.json'
    if not json_path.exists():
        raise FileNotFoundError(
            f"Results not found: {json_path}. Run 'python scripts/run_2018.py' first."
        )

    with open(json_path) as f:
        raw = json.load(f)

    cascades = raw['cascades']
    logger.info(f"Loaded {len(cascades)} cascades from JSON")

    # =========================================================================
    # 2. Load data and build temporal index
    # =========================================================================
    config = DetectorConfig(embedding_dir='data/embeddings')
    connector = DatabaseConnector(config)
    processor = DataProcessor()
    index_manager = IndexManager()

    logger.info("Loading data from database...")
    df = connector.get_frame_data('2018-01-01', '2018-12-31')
    df = processor.process_frame_data(df)
    articles = processor.aggregate_by_article(df)
    logger.info(f"  {len(articles):,} articles")

    logger.info("Building temporal index...")
    indices = index_manager.build_all_indices(df)
    temporal_index = indices.get('temporal', {})

    # =========================================================================
    # 3. Build figure — 8 subplots (one per frame)
    # =========================================================================
    fig, axes = plt.subplots(8, 1, figsize=(22, 28), sharex=True)
    fig.suptitle(
        'Daily Mean Frame Proportion per Article — 2018\n'
        'Red zones = cascades (onset → first composite > 0, merged)',
        fontsize=14, fontweight='bold', y=0.995
    )

    frame_full_names = {
        'Cult': 'Cultural', 'Eco': 'Economic', 'Envt': 'Environmental',
        'Pbh': 'Public Health', 'Just': 'Justice', 'Pol': 'Political',
        'Sci': 'Scientific', 'Secu': 'Security',
    }

    for ax, frame in zip(axes, FRAMES):
        color = FRAME_COLORS[frame]
        full_name = frame_full_names.get(frame, frame)

        # Get daily proportions from temporal index
        frame_data = temporal_index.get(frame, {})
        daily_props = frame_data.get('daily_proportions', None)

        if daily_props is None or daily_props.empty:
            ax.set_ylabel(f'{frame}\n({full_name})', fontsize=9,
                          color=color, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            continue

        # Plot raw daily proportions (light)
        ax.plot(daily_props.index, daily_props.values,
                color=color, alpha=0.25, linewidth=0.5)

        # Smoothed 7-day rolling average
        smoothed = daily_props.rolling(7, center=True, min_periods=1).mean()
        ax.plot(smoothed.index, smoothed.values,
                color=color, linewidth=1.8, label=f'{frame} (7d avg)')

        # Overlay cascades for this frame
        frame_cascades = [c for c in cascades if c['frame'] == frame]
        frame_cascades.sort(key=lambda c: c['onset_date'])

        for c in frame_cascades:
            onset = pd.Timestamp(c['onset_date'])
            end = pd.Timestamp(c['end_date'])
            score = c['total_score']
            classification = c['classification']

            # Shade cascade zone
            alpha = 0.15 + 0.25 * min(score, 1.0)
            ax.axvspan(onset, end, alpha=alpha, color='#E74C3C',
                       zorder=0)

            # Label with score
            mid = onset + (end - onset) / 2
            y_top = ax.get_ylim()[1]
            label = f"S={score:.2f}"
            ax.text(mid, y_top * 0.85, label,
                    fontsize=6.5, ha='center', va='top',
                    color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='#C0392B', alpha=0.85,
                              edgecolor='none'))

        # Formatting
        ax.set_ylabel(f'{frame}\n({full_name})', fontsize=9,
                      color=color, fontweight='bold')
        ax.tick_params(axis='y', labelsize=7)
        ax.set_xlim(pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-31'))
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add cascade count annotation
        n_cascades = len(frame_cascades)
        if n_cascades > 0:
            ax.text(0.995, 0.92, f'{n_cascades} cascade{"s" if n_cascades > 1 else ""}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=8, color='#C0392B',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.8,
                              edgecolor='#C0392B'))

    # X-axis formatting on bottom subplot
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].set_xlabel('2018', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = PROJECT_ROOT / 'results' / 'fig_frame_proportions_cascades.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    logger.info(f"Figure saved to {out_path}")


if __name__ == '__main__':
    main()
