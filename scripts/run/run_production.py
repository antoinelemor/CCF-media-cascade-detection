#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
run_production.py

MAIN OBJECTIVE:
---------------
Production run of cascade detection across all years (1978–2024).
Saves everything year-by-year in Parquet + JSON.

Pipeline is initialized ONCE (embedding memmap shared across years).
Each year is isolated with try/except for fault tolerance.
Resume mode skips years where year_metadata.json already exists.

Usage:
    python scripts/run_production.py                     # all years
    python scripts/run_production.py --year 2018         # single year
    python scripts/run_production.py --start 2000 --end 2010
    python scripts/run_production.py --resume             # skip completed
    python scripts/run_production.py --verbose

Author:
-------
Antoine Lemor
"""

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.constants import FRAMES
from cascade_detector.core.models import _jsonify
from cascade_detector.pipeline import CascadeDetectionPipeline

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / 'results' / 'production'


# =============================================================================
# Embedding computation (delegated to cascade_detector.embeddings.compute)
# =============================================================================

from cascade_detector.embeddings.compute import ensure_embeddings


# =============================================================================
# Serialization helpers
# =============================================================================

def _series_to_long_df(series: pd.Series, cascade_id: str,
                       col_name: str = 'value') -> pd.DataFrame:
    """Convert a pd.Series (date-indexed) to long-format DataFrame."""
    if series is None or series.empty:
        return pd.DataFrame(columns=['cascade_id', 'date', col_name])
    df = series.reset_index()
    df.columns = ['date', col_name]
    df.insert(0, 'cascade_id', cascade_id)
    return df


def _signals_to_long_df(signals: Dict[str, pd.Series],
                         cascade_id: str) -> pd.DataFrame:
    """Convert per-signal dict of Series to long-format DataFrame."""
    if not signals:
        return pd.DataFrame(columns=['cascade_id', 'signal', 'date', 'value'])
    rows = []
    for signal_name, series in signals.items():
        if series is None or (hasattr(series, 'empty') and series.empty):
            continue
        for date, value in series.items():
            rows.append({
                'cascade_id': cascade_id,
                'signal': signal_name,
                'date': date,
                'value': float(value) if pd.notna(value) else 0.0,
            })
    return pd.DataFrame(rows)


def _safe_parquet(df: pd.DataFrame, path: Path,
                  fallback_columns: list = None) -> None:
    """Write DataFrame to Parquet, handling empty DataFrames and Arrow errors."""
    if df.empty and len(df.columns) == 0 and fallback_columns:
        # Create empty DF with schema so parquet has typed columns
        df = pd.DataFrame({c: pd.Series(dtype='object') for c in fallback_columns})
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        if 'Arrow' in type(e).__name__:
            # Fallback: stringify all object columns to avoid struct/type errors
            df = df.copy()
            for col in df.select_dtypes(include='object').columns:
                df[col] = df[col].apply(
                    lambda x: json.dumps(x, default=str) if x is not None else '{}'
                )
            df.to_parquet(path, index=False)
        else:
            raise


def _safe_json(data: Any, path: Path) -> None:
    """Write JSON with numpy type handling."""
    with open(path, 'w') as f:
        json.dump(_jsonify(data), f, indent=2, default=str)


# =============================================================================
# Per-year export
# =============================================================================

def export_year(results, year: int, year_dir: Path) -> Dict[str, Any]:
    """Export all results for a single year.

    Returns year_metadata dict.
    """
    year_dir.mkdir(parents=True, exist_ok=True)

    cascades = results.cascades
    bursts = results.all_bursts

    # -------------------------------------------------------------------------
    # 1. Year metadata
    # -------------------------------------------------------------------------
    year_metadata = {
        'year': year,
        'analysis_period': results.analysis_period,
        'n_articles_analyzed': results.n_articles_analyzed,
        'n_bursts': len(bursts),
        'n_cascades': len(cascades),
        'n_cascades_by_frame': results.n_cascades_by_frame,
        'n_cascades_by_classification': results.n_cascades_by_classification,
        'runtime_seconds': results.runtime_seconds,
        'detection_parameters': results.detection_parameters,
        'exported_at': datetime.now().isoformat(),
    }

    # -------------------------------------------------------------------------
    # 2. Cascades JSON + Parquet + CSV
    # -------------------------------------------------------------------------
    _safe_json([c.to_dict() for c in cascades], year_dir / 'cascades.json')

    if cascades:
        cascade_df = results.to_dataframe()
        _safe_parquet(cascade_df, year_dir / 'cascades.parquet')
    else:
        _safe_parquet(pd.DataFrame(), year_dir / 'cascades.parquet')

    # -------------------------------------------------------------------------
    # 3. Bursts Parquet
    # -------------------------------------------------------------------------
    if bursts:
        burst_rows = [b.to_dict() for b in bursts]
        _safe_parquet(pd.DataFrame(burst_rows), year_dir / 'bursts.parquet')
    else:
        _safe_parquet(pd.DataFrame(), year_dir / 'bursts.parquet')

    # -------------------------------------------------------------------------
    # 4. Time series
    # -------------------------------------------------------------------------
    ts_dir = year_dir / 'time_series'
    ts_dir.mkdir(exist_ok=True)

    composite_dfs = []
    signals_dfs = []
    articles_dfs = []
    journalists_dfs = []
    cumulative_dfs = []

    for c in cascades:
        cid = c.cascade_id

        if c.daily_composite is not None:
            composite_dfs.append(_series_to_long_df(c.daily_composite, cid))

        if c.daily_signals:
            signals_dfs.append(_signals_to_long_df(c.daily_signals, cid))

        if c.daily_articles is not None:
            articles_dfs.append(_series_to_long_df(c.daily_articles, cid))

        if c.daily_journalists is not None:
            journalists_dfs.append(_series_to_long_df(c.daily_journalists, cid))

        if c.cumulative_journalists is not None:
            cumulative_dfs.append(_series_to_long_df(c.cumulative_journalists, cid))

    ts_3col = ['cascade_id', 'date', 'value']
    ts_4col = ['cascade_id', 'signal', 'date', 'value']

    _safe_parquet(
        pd.concat(composite_dfs, ignore_index=True) if composite_dfs else pd.DataFrame(),
        ts_dir / 'daily_composite.parquet', fallback_columns=ts_3col,
    )
    _safe_parquet(
        pd.concat(signals_dfs, ignore_index=True) if signals_dfs else pd.DataFrame(),
        ts_dir / 'daily_signals.parquet', fallback_columns=ts_4col,
    )
    _safe_parquet(
        pd.concat(articles_dfs, ignore_index=True) if articles_dfs else pd.DataFrame(),
        ts_dir / 'daily_articles.parquet', fallback_columns=ts_3col,
    )
    _safe_parquet(
        pd.concat(journalists_dfs, ignore_index=True) if journalists_dfs else pd.DataFrame(),
        ts_dir / 'daily_journalists.parquet', fallback_columns=ts_3col,
    )
    _safe_parquet(
        pd.concat(cumulative_dfs, ignore_index=True) if cumulative_dfs else pd.DataFrame(),
        ts_dir / 'cumulative_journalists.parquet', fallback_columns=ts_3col,
    )

    # -------------------------------------------------------------------------
    # 5. Networks
    # -------------------------------------------------------------------------
    net_dir = year_dir / 'networks'
    net_dir.mkdir(exist_ok=True)

    edge_cols = ['cascade_id', 'source_journalist', 'source_media',
                 'target_journalist', 'target_media', 'weight']
    edge_rows = []
    network_metrics = {}
    for c in cascades:
        cid = c.cascade_id
        for u, v, w in c.network_edges:
            # Nodes are (journalist, media) tuples
            src_j = u[0] if isinstance(u, tuple) else str(u)
            src_m = u[1] if isinstance(u, tuple) else ''
            tgt_j = v[0] if isinstance(v, tuple) else str(v)
            tgt_m = v[1] if isinstance(v, tuple) else ''
            edge_rows.append({
                'cascade_id': cid,
                'source_journalist': src_j,
                'source_media': src_m,
                'target_journalist': tgt_j,
                'target_media': tgt_m,
                'weight': w,
            })
        network_metrics[cid] = {
            'density': c.network_density,
            'modularity': c.network_modularity,
            'mean_degree': c.network_mean_degree,
            'n_components': c.network_n_components,
            'n_edges': len(c.network_edges),
        }

    if edge_rows:
        pd.DataFrame(edge_rows).to_csv(net_dir / 'edge_lists.csv', index=False)
    else:
        pd.DataFrame(columns=edge_cols).to_csv(
            net_dir / 'edge_lists.csv', index=False
        )

    _safe_json(network_metrics, net_dir / 'network_metrics.json')

    # -------------------------------------------------------------------------
    # 6. Frame signals
    # -------------------------------------------------------------------------
    sig_dir = year_dir / 'signals'
    sig_dir.mkdir(exist_ok=True)

    frame_signal_rows = []
    frame_signals = results.frame_signals
    for frame, signals in frame_signals.items():
        for signal_name, series in signals.items():
            if series is None or (hasattr(series, 'empty') and series.empty):
                continue
            for date, value in series.items():
                frame_signal_rows.append({
                    'frame': frame,
                    'signal': signal_name,
                    'date': date,
                    'value': float(value) if pd.notna(value) else 0.0,
                })

    _safe_parquet(
        pd.DataFrame(frame_signal_rows) if frame_signal_rows else pd.DataFrame(),
        sig_dir / 'frame_signals.parquet',
        fallback_columns=['frame', 'signal', 'date', 'value'],
    )

    # -------------------------------------------------------------------------
    # 7. Indices (serializable parts)
    # -------------------------------------------------------------------------
    idx_dir = year_dir / 'indices'
    idx_dir.mkdir(exist_ok=True)

    indices = getattr(results, '_indices', {})

    # Temporal index
    temporal_idx = indices.get('temporal', {})
    temporal_props_rows = []
    temporal_series_rows = []
    temporal_stats = {}

    for frame, frame_data in temporal_idx.items():
        if not isinstance(frame_data, dict):
            continue

        # Daily proportions
        props = frame_data.get('daily_proportions')
        if props is not None and not props.empty:
            for date, val in props.items():
                temporal_props_rows.append({
                    'frame': frame,
                    'date': date,
                    'proportion': float(val),
                })

        # Daily series (counts)
        series = frame_data.get('daily_series')
        if series is not None and not series.empty:
            for date, val in series.items():
                temporal_series_rows.append({
                    'frame': frame,
                    'date': date,
                    'count': int(val),
                })

        # Statistics
        stats = frame_data.get('statistics')
        if stats:
            temporal_stats[frame] = _jsonify(stats)

    _safe_parquet(
        pd.DataFrame(temporal_props_rows) if temporal_props_rows else pd.DataFrame(),
        idx_dir / 'temporal_daily_proportions.parquet',
        fallback_columns=['frame', 'date', 'proportion'],
    )
    _safe_parquet(
        pd.DataFrame(temporal_series_rows) if temporal_series_rows else pd.DataFrame(),
        idx_dir / 'temporal_daily_series.parquet',
        fallback_columns=['frame', 'date', 'count'],
    )
    _safe_json(temporal_stats, idx_dir / 'temporal_statistics.json')

    # Frame index
    frames_idx = indices.get('frames', {})
    if isinstance(frames_idx, dict):
        cooc = frames_idx.get('cooccurrence_matrix')
        if cooc is not None and hasattr(cooc, 'shape'):
            cooc_df = pd.DataFrame(cooc, index=FRAMES, columns=FRAMES)
            _safe_parquet(cooc_df.reset_index().rename(columns={'index': 'frame'}),
                          idx_dir / 'frame_cooccurrence.parquet')

        frame_stats = frames_idx.get('frame_statistics', {})
        _safe_json(frame_stats, idx_dir / 'frame_statistics.json')

    # Emotion index
    emotion_idx = indices.get('emotions', {})
    if isinstance(emotion_idx, dict):
        emotion_stats = emotion_idx.get('emotion_statistics', {})
        _safe_json(emotion_stats, idx_dir / 'emotion_statistics.json')

        temporal_emotion = emotion_idx.get('temporal_emotion', {})
        emotion_temporal_rows = []
        for date, data in temporal_emotion.items():
            if isinstance(data, dict):
                row = {'date': date}
                row.update({k: v for k, v in data.items()})
                emotion_temporal_rows.append(row)
        _safe_parquet(
            pd.DataFrame(emotion_temporal_rows) if emotion_temporal_rows else pd.DataFrame(),
            idx_dir / 'emotion_temporal.parquet',
            fallback_columns=['date', 'sentences', 'positive', 'negative', 'neutral'],
        )

    # Source index
    source_idx = indices.get('sources', {})
    if isinstance(source_idx, dict):
        source_meta = {}
        journalist_profiles = source_idx.get('journalist_profiles', {})
        media_profiles = source_idx.get('media_profiles', {})
        source_meta['n_journalists'] = len(journalist_profiles)
        source_meta['n_media'] = len(media_profiles)
        _safe_json(source_meta, idx_dir / 'source_metadata.json')

    # Geographic index (complex nested structure — serialize top-level keys)
    geo_idx = indices.get('geographic', {})
    if isinstance(geo_idx, dict) and geo_idx:
        # Extract serializable summary keys (focus_metrics, cascade_indicators,
        # geographic_diffusion, frame_geographic_patterns, linguistic_barriers)
        geo_safe_keys = [
            'focus_metrics', 'cascade_indicators', 'geographic_diffusion',
            'frame_geographic_patterns', 'linguistic_barriers', 'proximity_effects',
        ]
        geo_summary = {k: geo_idx[k] for k in geo_safe_keys if k in geo_idx}
        _safe_json(geo_summary, idx_dir / 'geographic_summary.json')

    # -------------------------------------------------------------------------
    # 8. Convergence
    # -------------------------------------------------------------------------
    conv_dir = year_dir / 'convergence'
    conv_dir.mkdir(exist_ok=True)

    convergence_full = {}
    for c in cascades:
        if c.convergence_metrics_full:
            convergence_full[c.cascade_id] = c.convergence_metrics_full

    _safe_json(convergence_full, conv_dir / 'semantic_convergence_full.json')

    # Syndication stats from source index
    syndication = {}
    if isinstance(source_idx, dict):
        for media, profile in source_idx.get('media_profiles', {}).items():
            if isinstance(profile, dict):
                syndication[str(media)] = {
                    'n_articles': len(profile.get('articles', [])),
                    'avg_proportions': _jsonify(profile.get('avg_proportions', {})),
                    'consistency': float(profile.get('consistency', 0.0)),
                }
    _safe_json(syndication, conv_dir / 'syndication_stats.json')

    # -------------------------------------------------------------------------
    # 9. Impact analysis (StabSel Phase 1: cluster → cascade)
    # -------------------------------------------------------------------------
    event_impact = getattr(results, 'event_impact', None)
    if event_impact is not None:
        impact_dir = year_dir / 'impact_analysis'
        impact_dir.mkdir(exist_ok=True)

        if hasattr(event_impact, 'cluster_cascade'):
            df = getattr(event_impact, 'cluster_cascade', None)
            if df is not None and isinstance(df, pd.DataFrame):
                _safe_parquet(df, impact_dir / 'cluster_cascade.parquet')

            summary = getattr(event_impact, 'summary', {})
            if summary:
                _safe_json(summary, impact_dir / 'summary.json')

            cascade_results = getattr(event_impact, 'cascade_results', None)
            if cascade_results:
                pkl_path = impact_dir / 'stabsel_results.pkl'
                with open(pkl_path, 'wb') as f:
                    pickle.dump(cascade_results, f, protocol=4)
            year_metadata['has_impact_analysis'] = True

        else:
            logger.warning(f"  event_impact has unexpected type: {type(event_impact).__name__}, skipping export")
            year_metadata['has_impact_analysis'] = False
    else:
        year_metadata['has_impact_analysis'] = False

    # -------------------------------------------------------------------------
    # 9b. Paradigm impact (StabSel v2: cluster/cascade → paradigm dominance)
    # -------------------------------------------------------------------------
    paradigm_impact = getattr(results, 'paradigm_impact', None)
    if paradigm_impact is not None:
        impact_dir = year_dir / 'impact_analysis'
        impact_dir.mkdir(exist_ok=True)

        for attr, fname in [
            ('cluster_dominance', 'stabsel_cluster_dominance.parquet'),
            ('cascade_dominance', 'stabsel_cascade_dominance.parquet'),
            ('alignment_a', 'stabsel_alignment_a.parquet'),
            ('alignment_b', 'stabsel_alignment_b.parquet'),
            ('validation', 'stabsel_validation.parquet'),
        ]:
            df = getattr(paradigm_impact, attr, None)
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                _safe_parquet(df, impact_dir / fname)

        summary = getattr(paradigm_impact, 'summary', {})
        if summary:
            _safe_json(summary, impact_dir / 'stabsel_paradigm_summary.json')

        raw = getattr(paradigm_impact, 'raw_results', None)
        if raw:
            pkl_path = impact_dir / 'stabsel_paradigm_results.pkl'
            with open(pkl_path, 'wb') as f:
                pickle.dump(raw, f, protocol=4)

        year_metadata['has_paradigm_impact'] = True
    else:
        year_metadata['has_paradigm_impact'] = False

    # -------------------------------------------------------------------------
    # 10. Paradigm shifts
    # -------------------------------------------------------------------------
    paradigm_shifts = getattr(results, 'paradigm_shifts', None)
    if paradigm_shifts is not None and hasattr(paradigm_shifts, 'shifts'):
        ps_dir = year_dir / 'paradigm_shifts'
        ps_dir.mkdir(exist_ok=True)

        # Timeline parquet
        if not paradigm_shifts.paradigm_timeline.empty:
            _safe_parquet(
                paradigm_shifts.paradigm_timeline,
                ps_dir / 'paradigm_timeline.parquet',
            )

        # Shifts JSON
        _safe_json(
            [s.to_dict() for s in paradigm_shifts.shifts],
            ps_dir / 'shifts.json',
        )

        # Episodes JSON
        if hasattr(paradigm_shifts, 'episodes') and paradigm_shifts.episodes:
            _safe_json(
                [e.to_dict() for e in paradigm_shifts.episodes],
                ps_dir / 'episodes.json',
            )

        year_metadata['n_paradigm_shifts'] = len(paradigm_shifts.shifts)
        year_metadata['n_paradigm_episodes'] = len(
            getattr(paradigm_shifts, 'episodes', []) or []
        )
        year_metadata['has_paradigm_shifts'] = True
    else:
        year_metadata['n_paradigm_shifts'] = 0
        year_metadata['has_paradigm_shifts'] = False

    # -------------------------------------------------------------------------
    # 11. Event occurrences
    # -------------------------------------------------------------------------
    has_occurrences = any(
        getattr(c, 'event_occurrences', None) for c in cascades
    )
    if has_occurrences:
        occ_dir = year_dir / 'event_occurrences'
        occ_dir.mkdir(exist_ok=True)

        # Occurrences parquet (one row per occurrence, flat)
        occ_rows = []
        for c in cascades:
            for occ in getattr(c, 'event_occurrences', []):
                row = occ.to_dict()
                row['cascade_id'] = c.cascade_id
                occ_rows.append(row)

        if occ_rows:
            _safe_parquet(pd.DataFrame(occ_rows), occ_dir / 'occurrences.parquet')

        # Occurrence metrics parquet (one row per cascade)
        metric_rows = []
        for c in cascades:
            metrics = getattr(c, 'event_occurrence_metrics', {})
            if metrics:
                row = dict(metrics)
                row['cascade_id'] = c.cascade_id
                metric_rows.append(row)

        if metric_rows:
            _safe_parquet(pd.DataFrame(metric_rows), occ_dir / 'occurrence_metrics.parquet')

        # Full JSON
        occ_json = {}
        for c in cascades:
            occs = getattr(c, 'event_occurrences', [])
            if occs:
                occ_json[c.cascade_id] = {
                    'occurrences': [o.to_dict() for o in occs],
                    'metrics': getattr(c, 'event_occurrence_metrics', {}),
                }
        _safe_json(occ_json, occ_dir / 'event_occurrences.json')

        # Daily event profile to time_series/
        profile_dfs = []
        for c in cascades:
            profile = getattr(c, 'daily_event_profile', None)
            if profile is not None and not profile.empty:
                df = profile.reset_index()
                df.insert(0, 'cascade_id', c.cascade_id)
                profile_dfs.append(df)

        if profile_dfs:
            _safe_parquet(
                pd.concat(profile_dfs, ignore_index=True),
                ts_dir / 'daily_event_profile.parquet',
            )

        year_metadata['has_event_occurrences'] = True
        year_metadata['n_event_occurrences'] = len(occ_rows)
    else:
        year_metadata['has_event_occurrences'] = False
        year_metadata['n_event_occurrences'] = 0

    # -------------------------------------------------------------------------
    # 12. Top-level event clusters, occurrences, attributions
    # -------------------------------------------------------------------------
    event_clusters = getattr(results, 'event_clusters', [])
    all_occurrences = getattr(results, 'all_occurrences', [])
    cascade_attributions = getattr(results, 'cascade_attributions', [])

    events_dir = year_dir / 'events'
    events_dir.mkdir(exist_ok=True)

    # Event clusters JSON (full structure with occurrence IDs, types, strength)
    if event_clusters:
        _safe_json(
            [ec.to_dict() for ec in event_clusters],
            events_dir / 'event_clusters.json',
        )

        # Event clusters Parquet (flat summary, one row per cluster)
        ec_rows = []
        for ec in event_clusters:
            d = ec.to_dict()
            # Flatten lists/dicts for Parquet
            d['event_types'] = json.dumps(d['event_types'])
            d['occurrence_ids'] = json.dumps(d['occurrence_ids'])
            d['strength_components'] = json.dumps(d['strength_components'])
            d['entities'] = json.dumps(d['entities'])
            d['type_structure'] = json.dumps(d['type_structure'])
            d['type_overlap_graph'] = json.dumps(d['type_overlap_graph'])
            ec_rows.append(d)
        _safe_parquet(pd.DataFrame(ec_rows), events_dir / 'event_clusters.parquet')

    # All occurrences Parquet (one row per occurrence, independent of cascades)
    if all_occurrences:
        occ_rows = []
        for occ in all_occurrences:
            d = occ.to_dict()
            # Flatten lists for Parquet
            d['doc_ids'] = json.dumps(d['doc_ids'])
            d['seed_doc_ids'] = json.dumps(d['seed_doc_ids'])
            d['entities'] = json.dumps(d['entities'])
            d['confidence_components'] = json.dumps(d['confidence_components'])
            d['type_scores'] = json.dumps(d['type_scores'])
            occ_rows.append(d)
        _safe_parquet(pd.DataFrame(occ_rows), events_dir / 'all_occurrences.parquet')

    # Cascade attributions Parquet (occurrence → cascade links)
    if cascade_attributions:
        attr_rows = [a.to_dict() for a in cascade_attributions]
        _safe_parquet(pd.DataFrame(attr_rows), events_dir / 'cascade_attributions.parquet')

    year_metadata['n_event_clusters'] = len(event_clusters)
    year_metadata['n_all_occurrences'] = len(all_occurrences)
    year_metadata['n_cascade_attributions'] = len(cascade_attributions)
    n_multi = sum(1 for ec in event_clusters if ec.is_multi_type)
    year_metadata['n_multi_type_clusters'] = n_multi

    # -------------------------------------------------------------------------
    # Write year_metadata last (signals completion)
    # -------------------------------------------------------------------------
    _safe_json(year_metadata, year_dir / 'year_metadata.json')

    return year_metadata


# =============================================================================
# Cross-year aggregation
# =============================================================================

def aggregate_cross_year(output_dir: Path, year_metadatas: List[Dict]) -> None:
    """Aggregate all year-level cascade parquets into cross-year files."""
    logger.info("Aggregating cross-year results...")

    all_cascade_dfs = []
    for meta in year_metadatas:
        year = meta['year']
        pq_path = output_dir / str(year) / 'cascades.parquet'
        if pq_path.exists():
            df = pd.read_parquet(pq_path)
            if not df.empty:
                df.insert(0, 'year', year)
                all_cascade_dfs.append(df)

    if all_cascade_dfs:
        combined = pd.concat(all_cascade_dfs, ignore_index=True)
        _safe_parquet(combined, output_dir / 'cross_year_cascades.parquet')
        combined.to_csv(output_dir / 'cross_year_cascades.csv', index=False)
        logger.info(f"  Cross-year cascades: {len(combined)} rows")
    else:
        logger.info("  No cascades found across all years")

    # Cross-year summary
    summary = {
        'total_years': len(year_metadatas),
        'years_with_cascades': sum(1 for m in year_metadatas if m.get('n_cascades', 0) > 0),
        'total_cascades': sum(m.get('n_cascades', 0) for m in year_metadatas),
        'total_bursts': sum(m.get('n_bursts', 0) for m in year_metadatas),
        'total_articles': sum(m.get('n_articles_analyzed', 0) for m in year_metadatas),
        'total_event_clusters': sum(m.get('n_event_clusters', 0) for m in year_metadatas),
        'total_occurrences': sum(m.get('n_all_occurrences', 0) for m in year_metadatas),
        'total_attributions': sum(m.get('n_cascade_attributions', 0) for m in year_metadatas),
        'total_runtime_seconds': sum(m.get('runtime_seconds', 0) for m in year_metadatas),
        'by_year': {m['year']: {
            'n_cascades': m.get('n_cascades', 0),
            'n_bursts': m.get('n_bursts', 0),
            'n_articles': m.get('n_articles_analyzed', 0),
            'by_frame': m.get('n_cascades_by_frame', {}),
            'by_classification': m.get('n_cascades_by_classification', {}),
        } for m in year_metadatas},
        'generated_at': datetime.now().isoformat(),
    }
    _safe_json(summary, output_dir / 'cross_year_summary.json')
    logger.info(f"  Cross-year summary saved")

    # Cross-year paradigm timeline
    all_timeline_dfs = []
    all_shifts = []
    for meta in year_metadatas:
        year = meta['year']
        tl_path = output_dir / str(year) / 'paradigm_shifts' / 'paradigm_timeline.parquet'
        if tl_path.exists():
            df = pd.read_parquet(tl_path)
            if not df.empty:
                df.insert(0, 'year', year)
                all_timeline_dfs.append(df)

        shifts_path = output_dir / str(year) / 'paradigm_shifts' / 'shifts.json'
        if shifts_path.exists():
            import json as _json
            with open(shifts_path) as f:
                year_shifts = _json.load(f)
            for s in year_shifts:
                s['year'] = year
                all_shifts.append(s)

    if all_timeline_dfs:
        combined_tl = pd.concat(all_timeline_dfs, ignore_index=True)
        _safe_parquet(combined_tl, output_dir / 'cross_year_paradigm_timeline.parquet')
        logger.info(f"  Cross-year paradigm timeline: {len(combined_tl)} rows")

    if all_shifts:
        _safe_json(all_shifts, output_dir / 'cross_year_paradigm_shifts.json')
        logger.info(f"  Cross-year paradigm shifts: {len(all_shifts)} shifts")

    # Cross-year event clusters
    all_event_clusters = []
    all_attributions_dfs = []
    for meta in year_metadatas:
        year = meta['year']

        ec_path = output_dir / str(year) / 'events' / 'event_clusters.json'
        if ec_path.exists():
            with open(ec_path) as f:
                year_clusters = json.load(f)
            for ec in year_clusters:
                ec['year'] = year
                all_event_clusters.append(ec)

        attr_path = output_dir / str(year) / 'events' / 'cascade_attributions.parquet'
        if attr_path.exists():
            df = pd.read_parquet(attr_path)
            if not df.empty:
                df.insert(0, 'year', year)
                all_attributions_dfs.append(df)

    if all_event_clusters:
        _safe_json(all_event_clusters, output_dir / 'cross_year_event_clusters.json')
        logger.info(f"  Cross-year event clusters: {len(all_event_clusters)} clusters")

    if all_attributions_dfs:
        combined_attr = pd.concat(all_attributions_dfs, ignore_index=True)
        _safe_parquet(combined_attr, output_dir / 'cross_year_cascade_attributions.parquet')
        logger.info(f"  Cross-year cascade attributions: {len(combined_attr)} rows")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run cascade detection across all years (1978-2024).'
    )
    parser.add_argument('--start', type=int, default=1978,
                        help='Start year (default: 1978)')
    parser.add_argument('--end', type=int, default=2024,
                        help='End year (default: 2024)')
    parser.add_argument('--year', type=int, default=None,
                        help='Run a single year only')
    parser.add_argument('--resume', action='store_true',
                        help='Skip years where year_metadata.json exists')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--skip-embeddings', action='store_true',
                        help='Skip embedding coverage check (assume already computed)')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    if args.year:
        years = [args.year]
    else:
        years = list(range(args.start, args.end + 1))

    logger.info(f"Production run: {len(years)} year(s) from {years[0]} to {years[-1]}")

    # Output directory
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Ensure embeddings are computed and complete
    embedding_dir = os.environ.get('EMBEDDING_DIR', 'data/embeddings')
    ensure_embeddings(embedding_dir, skip=args.skip_embeddings)

    # Initialize pipeline ONCE (embedding memmap shared across years)
    logger.info("Initializing pipeline (loading embeddings)...")
    config = DetectorConfig(
        embedding_dir=embedding_dir,
        verbose=args.verbose,
    )
    pipeline = CascadeDetectionPipeline(config)
    logger.info("Pipeline ready.")

    # Run manifest
    manifest = {
        'start_time': datetime.now().isoformat(),
        'years': years,
        'resume': args.resume,
        'config': _jsonify(config.to_dict()),
    }

    year_metadatas = []
    failed_years = []

    for year in years:
        year_dir = output_dir / str(year)

        # Resume check
        if args.resume and (year_dir / 'year_metadata.json').exists():
            try:
                with open(year_dir / 'year_metadata.json') as f:
                    year_metadatas.append(json.load(f))
                logger.info(f"[{year}] Skipping (already completed)")
                continue
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[{year}] Corrupt metadata, re-running: {e}")

        logger.info(f"\n{'=' * 70}")
        logger.info(f"[{year}] Starting cascade detection...")
        logger.info(f"{'=' * 70}")

        try:
            start_date = f'{year}-01-01'
            end_date = f'{year}-12-31'
            # Extend 6 weeks into next year to capture boundary cascades
            if year < args.end:
                extended_end = f'{year + 1}-02-15'
            else:
                extended_end = end_date

            # Reset cascade counter for clean per-year IDs
            pipeline.detector._cascade_counter = 0

            # Per-step checkpointing
            checkpoint_dir = year_dir / '.checkpoint'
            results = pipeline.run(start_date, extended_end,
                                   target_end_date=end_date,
                                   checkpoint_dir=checkpoint_dir)

            # Flag low-activity years
            if results.n_articles_analyzed < 50:
                logger.warning(
                    f"[{year}] Low activity: only {results.n_articles_analyzed} articles"
                )

            year_meta = export_year(results, year, year_dir)
            year_metadatas.append(year_meta)

            # Clean up checkpoint files after successful export
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)

            n_strong = results.n_cascades_by_classification.get('strong_cascade', 0)
            n_moderate = results.n_cascades_by_classification.get('moderate_cascade', 0)
            logger.info(
                f"[{year}] Done: {results.n_articles_analyzed:,} articles, "
                f"{len(results.cascades)} cascades "
                f"({n_strong} strong, {n_moderate} moderate) "
                f"in {results.runtime_seconds:.1f}s"
            )

        except Exception as e:
            logger.error(f"[{year}] FAILED: {e}")
            logger.error(traceback.format_exc())
            failed_years.append({'year': year, 'error': str(e)})
            continue

    # Cross-year aggregation
    if year_metadatas:
        aggregate_cross_year(output_dir, year_metadatas)

    # Update manifest
    manifest['end_time'] = datetime.now().isoformat()
    manifest['completed_years'] = [m['year'] for m in year_metadatas]
    manifest['failed_years'] = failed_years
    manifest['total_cascades'] = sum(m.get('n_cascades', 0) for m in year_metadatas)
    _safe_json(manifest, output_dir / 'run_manifest.json')

    # Final summary
    total_time = sum(m.get('runtime_seconds', 0) for m in year_metadatas)
    total_cascades = sum(m.get('n_cascades', 0) for m in year_metadatas)
    total_articles = sum(m.get('n_articles_analyzed', 0) for m in year_metadatas)

    print(f"\n{'=' * 70}")
    print(f"PRODUCTION RUN COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Years processed:  {len(year_metadatas)}/{len(years)}")
    print(f"  Failed years:     {len(failed_years)}")
    print(f"  Total articles:   {total_articles:,}")
    print(f"  Total cascades:   {total_cascades}")
    print(f"  Total runtime:    {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"  Output dir:       {output_dir}")

    if failed_years:
        print(f"\n  FAILED YEARS:")
        for fy in failed_years:
            print(f"    {fy['year']}: {fy['error']}")

    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
