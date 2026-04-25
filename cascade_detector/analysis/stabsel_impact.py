"""
PROJECT: CCF-media-cascade-detection
TITLE: stabsel_impact.py — Stability Selection impact analysis (Phase 1: cluster → cascade)

Treatment variable (per cluster j, per cascade):
  D_j(t,l) = Σ_a∈articles(t-l) belonging(a,j) × frame_signal(a) × cosine_sim(a, cascade_centroid)

The cascade centroid is built from the cascade's own central articles.
This double-weighting naturally eliminates frame-irrelevant clusters.

Variable selection: Stability Selection (Meinshausen & Bühlmann, 2010)
  B=100 sub-samples, ElasticNet, π≥0.60
Post-selection: OLS + residual bootstrap → p-values

Author: Antoine Lemor
"""

import logging
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler

from cascade_detector.core.constants import (
    FRAMES, FRAME_COLUMNS, FRAME_COLORS, MESSENGERS, EVENT_COLUMNS,
)
from cascade_detector.core.models import EventCluster

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ── Constants (exact copy from sandbox L57-75) ────────────────────────────
MARGIN_DAYS = 30
MIN_D_SUM = 0.01
MAX_LAG = 3
BASELINE_WINDOW = 90
CASCADE_CENTROID_MARGIN = 14

# Stability Selection
N_SUBSAMPLES = 100
SUBSAMPLE_FRAC = 0.5
PI_THRESHOLD = 0.60
N_BOOTSTRAP = 500
ALPHA_SIG = 0.10

# Signal weights
W_TEMPORAL = 0.25
W_PARTICIPATION = 0.20
W_CONVERGENCE = 0.20
W_SOURCE = 0.15
W_SEMANTIC = 0.20

# Phase 2 occurrence wrapping
OCCURRENCE_ID_OFFSET = 100_000  # wrapped occurrence IDs = 100_000 + occurrence_id

FRAME_FULL = {
    'Cult': 'Cultural', 'Eco': 'Economic', 'Envt': 'Environmental',
    'Pbh': 'Public Health', 'Just': 'Justice', 'Pol': 'Political',
    'Sci': 'Scientific', 'Secu': 'Security',
}
ROLE_COLORS = {
    'driver': '#27AE60', 'suppressor': '#E74C3C', 'neutral': '#BDC3C7',
}


# ── Dataclasses ───────────────────────────────────────────────────────────

@dataclass
class ClusterRole:
    cluster_id: int
    net_beta: float
    p_value: float
    role: str
    lag_profile: np.ndarray
    selection_freq: float = 0.0
    D_sum: float = 0.0


@dataclass
class StabSelCascadeResult:
    cascade_id: str
    frame: str
    r2: float
    n_clusters: int
    n_stable: int
    n_days: int
    n_drivers: int
    n_suppressors: int
    n_neutral: int
    roles: List[ClusterRole]
    cluster_meta: Dict = field(default_factory=dict)
    y_obs: Optional[np.ndarray] = None
    date_index: Optional[pd.DatetimeIndex] = None
    onset: Optional[pd.Timestamp] = None
    peak: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None


@dataclass
class StabSelImpactResults:
    """Container for StabSel impact analysis results."""
    cluster_cascade: pd.DataFrame          # flat (cluster_id, cascade_id, role, net_beta, ...)
    summary: Dict[str, Any]                # per-frame {n_cascades, n_drivers, n_suppressors, median_r2}
    cascade_results: Dict[str, List]       # frame → [StabSelCascadeResult] for figures/pickle


# ── Two-sided rolling z-score ─────────────────────────────────────────────

def rolling_zscore_twosided(series, window=BASELINE_WINDOW):
    rolling_mean = series.rolling(window=window, min_periods=window // 2).mean().shift(1)
    rolling_std = series.rolling(window=window, min_periods=window // 2).std().shift(1)
    global_mean = series.mean()
    global_std = series.std()
    rolling_mean = rolling_mean.fillna(global_mean)
    rolling_std = rolling_std.fillna(global_std)
    rolling_std = rolling_std.clip(lower=max(global_std * 0.1, 1e-10))
    return (series - rolling_mean) / rolling_std


# ── Signal builders ───────────────────────────────────────────────────────

def _resolve_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _filter_frame_articles(articles, frame_col):
    if frame_col is None:
        return articles
    for col in [frame_col, f'{frame_col}_sum', f'{frame_col}_mean']:
        if col in articles.columns:
            return articles[articles[col] > 0]
    return articles

def build_z_temporal(frame, temporal_index):
    series = temporal_index[frame]['daily_proportions'].sort_index().astype(float)
    return rolling_zscore_twosided(series)

def build_z_participation(frame, articles, date_range):
    raw = pd.Series(0.0, index=date_range)
    date_col = _resolve_col(articles, ['date', 'date_converted_first'])
    author_col = _resolve_col(articles, ['author', 'author_first', 'author_clean_first'])
    if date_col is None or author_col is None:
        return raw
    frame_col = FRAME_COLUMNS.get(frame)
    fa = _filter_frame_articles(articles, frame_col)
    if fa.empty:
        return raw
    fa = fa.copy()
    fa['_date'] = pd.to_datetime(fa[date_col], errors='coerce').dt.normalize()
    daily = fa.dropna(subset=['_date', author_col]).groupby('_date')[author_col].nunique()
    raw = raw.add(daily, fill_value=0).fillna(0).reindex(date_range, fill_value=0)
    return rolling_zscore_twosided(raw)

def build_z_convergence(frame, temporal_index, date_range):
    all_sums = pd.Series(0.0, index=date_range)
    target = pd.Series(0.0, index=date_range)
    for f in FRAMES:
        if f not in temporal_index:
            continue
        fs = temporal_index[f].get('daily_proportions')
        if fs is None:
            continue
        fs = fs.reindex(date_range, fill_value=0).astype(float)
        all_sums = all_sums + fs
        if f == frame:
            target = fs
    dominance = pd.Series(0.0, index=date_range)
    nz = all_sums > 0
    dominance[nz] = target[nz] / all_sums[nz]
    return rolling_zscore_twosided(dominance)

def build_z_source(frame, articles, date_range):
    raw = pd.Series(0.0, index=date_range)
    date_col = _resolve_col(articles, ['date', 'date_converted_first'])
    if date_col is None:
        return raw
    frame_col = FRAME_COLUMNS.get(frame)
    fa = _filter_frame_articles(articles, frame_col)
    if fa.empty:
        return raw
    fa = fa.copy()
    fa['_date'] = pd.to_datetime(fa[date_col], errors='coerce').dt.normalize()
    msg_cols = []
    for mc in MESSENGERS:
        for col in [mc, f'{mc}_sum', f'{mc}_mean']:
            if col in fa.columns:
                msg_cols.append(col)
                break
    if not msg_cols:
        return raw
    for date in date_range:
        day = fa[fa['_date'] == date]
        if day.empty:
            continue
        counts = np.array([day[c].sum() for c in msg_cols], dtype=float)
        total = counts.sum()
        if total == 0:
            continue
        probs = counts / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        max_ent = np.log(len(msg_cols))
        if max_ent > 0:
            raw[date] = 1.0 - (entropy / max_ent)
    return rolling_zscore_twosided(raw)

def build_z_semantic_proximity(frame, articles, date_range, embedding_store):
    raw = pd.Series(0.0, index=date_range)
    date_col = _resolve_col(articles, ['date', 'date_converted_first'])
    if date_col is None or 'doc_id' not in articles.columns:
        return raw
    frame_col = FRAME_COLUMNS.get(frame)
    fa = _filter_frame_articles(articles, frame_col)
    if fa.empty or len(fa) < 10:
        return raw
    score_col = None
    for c in [frame_col, f'{frame_col}_mean', f'{frame_col}_sum']:
        if c and c in fa.columns:
            score_col = c
            break
    if score_col:
        threshold = fa[score_col].quantile(0.75)
        top_fa = fa[fa[score_col] >= threshold]
    else:
        top_fa = fa
    top_doc_ids = top_fa['doc_id'].dropna().unique().tolist()
    centroid_embs, centroid_ids = embedding_store.get_batch_article_embeddings(top_doc_ids)
    if len(centroid_embs) < 5:
        return raw
    centroid = centroid_embs.mean(axis=0)
    centroid_norm = centroid / max(np.linalg.norm(centroid), 1e-10)
    art = articles.copy()
    art['_date'] = pd.to_datetime(art[date_col], errors='coerce').dt.normalize()
    daily_groups = art.dropna(subset=['_date', 'doc_id']).groupby('_date')['doc_id'].apply(list)
    for date in date_range:
        if date not in daily_groups.index:
            continue
        doc_ids = daily_groups[date]
        if len(doc_ids) < 1:
            continue
        embs, _ = embedding_store.get_batch_article_embeddings(doc_ids)
        if len(embs) == 0:
            continue
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = embs / norms
        sims = normalized @ centroid_norm
        raw[date] = float(np.mean(sims))
    return rolling_zscore_twosided(raw)

def orthogonalize(target, reference):
    ref = reference.values.astype(float)
    tgt = target.values.astype(float)
    ref_dot = np.dot(ref, ref)
    if ref_dot < 1e-10:
        return target
    beta = np.dot(tgt, ref) / ref_dot
    residual = tgt - beta * ref
    return pd.Series(residual, index=target.index)


# ── Composite builder ─────────────────────────────────────────────────────

def build_twosided_composite(frame, articles, temporal_index, embedding_store):
    date_range = temporal_index[frame]['daily_proportions'].sort_index().index
    logger.info(f"  Building 5 two-sided signals for {frame}...")
    z_temporal = build_z_temporal(frame, temporal_index)
    z_participation = build_z_participation(frame, articles, date_range)
    z_convergence_raw = build_z_convergence(frame, temporal_index, date_range)
    z_source = build_z_source(frame, articles, date_range)
    z_semantic = build_z_semantic_proximity(frame, articles, date_range, embedding_store)
    z_convergence = orthogonalize(z_convergence_raw, z_temporal)
    composite = (
        W_TEMPORAL * z_temporal
        + W_PARTICIPATION * z_participation
        + W_CONVERGENCE * z_convergence
        + W_SOURCE * z_source
        + W_SEMANTIC * z_semantic
    )
    return composite, {
        'z_temporal': z_temporal, 'z_participation': z_participation,
        'z_convergence': z_convergence, 'z_source': z_source,
        'z_semantic': z_semantic, 'composite': composite,
    }


# ── Cascade-specific centroid ──────────────────────────────────────────────

def build_cascade_centroid(cascade, frame, articles, embedding_store):
    """Build embedding centroid from cascade's own central articles."""
    onset = pd.Timestamp(cascade.onset_date)
    end = pd.Timestamp(cascade.end_date)
    margin = pd.Timedelta(days=CASCADE_CENTROID_MARGIN)

    frame_col_base = FRAME_COLUMNS.get(frame)
    frame_col = None
    for candidate in [f"{frame_col_base}_mean", f"{frame_col_base}_sum", frame_col_base]:
        if candidate in articles.columns:
            frame_col = candidate
            break
    if frame_col is None:
        return None, None

    mask = (articles['date'] >= onset - margin) & (articles['date'] <= end + margin)
    window_articles = articles[mask].copy()
    frame_articles = window_articles[window_articles[frame_col] > 0]

    if len(frame_articles) < 5:
        frame_articles = articles[articles[frame_col] > 0].copy()
        if len(frame_articles) < 5:
            return None, frame_col

    threshold = frame_articles[frame_col].quantile(0.75)
    top_articles = frame_articles[frame_articles[frame_col] >= threshold]
    if len(top_articles) < 3:
        top_articles = frame_articles

    doc_ids = top_articles['doc_id'].dropna().unique().tolist()
    embs, found_ids = embedding_store.get_batch_article_embeddings(doc_ids)
    if len(embs) < 3:
        return None, frame_col

    signals = np.array([
        float(top_articles.set_index('doc_id')[frame_col].get(did, 0.0))
        if did in top_articles['doc_id'].values else 0.0
        for did in found_ids
    ])
    signal_sum = signals.sum()
    if signal_sum > 1e-10:
        centroid = (embs * signals[:, np.newaxis]).sum(axis=0) / signal_sum
    else:
        centroid = embs.mean(axis=0)

    c_norm = np.linalg.norm(centroid)
    if c_norm < 1e-10:
        return None, frame_col
    return centroid / c_norm, frame_col


# ── Double-weighted lagged mass matrix ─────────────────────────────────────

def build_weighted_lagged_mass(clusters, articles, date_index, frame_col,
                                cascade_centroid, embedding_store):
    """Build double-weighted lagged mass matrix for StabSel.

    Returns:
        X: (n_days, n_clusters × (MAX_LAG+1)) matrix
        cids: unique cluster IDs
        lag_labels: list of (cid, lag) tuples
        cluster_meta: metadata dict
    """
    n = len(date_index)
    win_start = date_index[0]
    win_end = date_index[-1]

    art_frame = articles.set_index('doc_id')[frame_col].to_dict()
    art_date = articles.set_index('doc_id')['date'].to_dict()

    # Pre-build lookups for all frame / messenger / event columns (profiling)
    # Columns have _mean or _sum suffix in the DataFrame
    art_indexed = articles.set_index('doc_id')

    def _resolve(base):
        for suffix in ['_mean', '_sum', '']:
            c = f'{base}{suffix}'
            if c in articles.columns:
                return c
        return None

    all_frame_cols = {}   # {frame_key: resolved_col}
    all_frame_lookup = {}
    for f in FRAMES:
        resolved = _resolve(FRAME_COLUMNS[f])
        if resolved:
            all_frame_cols[f] = resolved
            all_frame_lookup[resolved] = art_indexed[resolved].to_dict()

    msg_cols_avail = []   # [(raw_name, resolved_col)]
    msg_lookup = {}
    for m in MESSENGERS:
        resolved = _resolve(m)
        if resolved:
            msg_cols_avail.append((m, resolved))
            msg_lookup[m] = art_indexed[resolved].to_dict()

    evt_cols_avail = []   # [(raw_name, resolved_col)]
    evt_lookup = {}
    for e in EVENT_COLUMNS:
        resolved = _resolve(e)
        if resolved:
            evt_cols_avail.append((e, resolved))
            evt_lookup[e] = art_indexed[resolved].to_dict()

    all_doc_ids = articles['doc_id'].dropna().unique().tolist()
    all_embs, all_found_ids = embedding_store.get_batch_article_embeddings(all_doc_ids)
    emb_norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    emb_norms = np.maximum(emb_norms, 1e-10)
    all_embs_normed = all_embs / emb_norms
    cosine_sims = all_embs_normed @ cascade_centroid
    cosine_lookup = dict(zip(all_found_ids, cosine_sims.tolist()))

    raw_weighted = {}
    cluster_meta = {}

    for cluster in clusters:
        cid = cluster.cluster_id
        if cluster.core_start and cluster.core_end:
            cs = pd.Timestamp(cluster.core_start)
            ce = pd.Timestamp(cluster.core_end)
            if ce < win_start - pd.Timedelta(days=MAX_LAG) or cs > win_end:
                continue

        combined = {}
        for occ in cluster.occurrences:
            for doc_id, belonging in occ.belonging.items():
                combined[doc_id] = max(combined.get(doc_id, 0.0), belonging)
        if not combined:
            continue

        daily_weighted = {}
        article_details = []
        # Accumulators for belonging-weighted profiles (keyed by canonical name)
        profile_frame_sum = {f: 0.0 for f in all_frame_cols}       # frame key → sum
        profile_msg_sum = {raw: 0.0 for raw, _ in msg_cols_avail}  # raw name → sum
        profile_evt_sum = {raw: 0.0 for raw, _ in evt_cols_avail}  # raw name → sum
        belonging_total = 0.0

        for doc_id, belonging in combined.items():
            date = art_date.get(doc_id)
            if date is None or pd.isna(date):
                continue
            date = pd.Timestamp(date)
            frame_signal = art_frame.get(doc_id, 0.0)
            if pd.isna(frame_signal):
                frame_signal = 0.0
            cos_sim = cosine_lookup.get(doc_id, 0.0)
            weight = belonging * frame_signal * max(cos_sim, 0.0)

            # Accumulate belonging-weighted profiles (all articles, not just weight>0)
            belonging_total += belonging
            for f_key, resolved_col in all_frame_cols.items():
                val = all_frame_lookup[resolved_col].get(doc_id, 0.0)
                if pd.isna(val):
                    val = 0.0
                profile_frame_sum[f_key] += belonging * float(val)
            for raw, _ in msg_cols_avail:
                val = msg_lookup[raw].get(doc_id, 0.0)
                if pd.isna(val):
                    val = 0.0
                profile_msg_sum[raw] += belonging * float(val)
            for raw, _ in evt_cols_avail:
                val = evt_lookup[raw].get(doc_id, 0.0)
                if pd.isna(val):
                    val = 0.0
                profile_evt_sum[raw] += belonging * float(val)

            if weight > 0:
                daily_weighted[date] = daily_weighted.get(date, 0.0) + weight
                article_details.append({
                    'doc_id': int(doc_id) if isinstance(doc_id, (int, np.integer)) else doc_id,
                    'belonging': round(float(belonging), 4),
                    'frame_signal': round(float(frame_signal), 4),
                    'cosine_sim': round(float(cos_sim), 4),
                    'weight': round(float(weight), 6),
                })

        if not daily_weighted:
            continue

        dm = pd.Series(daily_weighted)
        aligned = dm.reindex(date_index, fill_value=0.0).values
        if aligned.sum() < MIN_D_SUM:
            continue

        raw_weighted[cid] = aligned

        # Compute belonging-weighted mean profiles (keyed by canonical short names)
        if belonging_total > 0:
            frame_profile = {f_key: round(profile_frame_sum[f_key] / belonging_total, 4)
                             for f_key in all_frame_cols}
            msg_profile = {raw: round(profile_msg_sum[raw] / belonging_total, 4)
                           for raw, _ in msg_cols_avail}
            evt_profile = {raw: round(profile_evt_sum[raw] / belonging_total, 4)
                           for raw, _ in evt_cols_avail}
        else:
            frame_profile = {f_key: 0.0 for f_key in all_frame_cols}
            msg_profile = {raw: 0.0 for raw, _ in msg_cols_avail}
            evt_profile = {raw: 0.0 for raw, _ in evt_cols_avail}

        entities = getattr(cluster, 'entities', set()) or set()
        cluster_meta[cid] = {
            'cluster_id': int(cid),
            'source_level': 'occurrence' if cid >= OCCURRENCE_ID_OFFSET else 'cluster',
            'dominant_type': getattr(cluster, 'dominant_type', 'unknown'),
            'event_types': list(getattr(cluster, 'event_types', set()) or set()),
            'entities': sorted(entities)[:15] if entities else [],
            'strength': round(float(getattr(cluster, 'strength', 0.0)), 4),
            'n_articles_weighted': len(article_details),
            'n_articles_total': len(combined),
            'D_sum': round(float(aligned.sum()), 4),
            'peak_date': str(dm.idxmax().date()) if not dm.empty else None,
            'article_details': sorted(article_details, key=lambda x: -x['weight'])[:20],
            'frame_profile': frame_profile,
            'messenger_profile': msg_profile,
            'event_profile': evt_profile,
        }

    if not raw_weighted:
        return np.empty((n, 0)), [], [], {}

    # Build lagged columns
    cids = list(raw_weighted.keys())
    cols = []
    lag_labels = []
    for cid in cids:
        m = raw_weighted[cid]
        for lag in range(MAX_LAG + 1):
            if lag == 0:
                cols.append(m.copy())
            else:
                lagged = np.zeros(n)
                lagged[lag:] = m[:-lag]
                cols.append(lagged)
            lag_labels.append((cid, lag))
    X = np.column_stack(cols)
    return X, cids, lag_labels, cluster_meta


# ── Stability Selection ──────────────────────────────────────────────────

def stability_selection(y, X_with_trend, n_subsamples=N_SUBSAMPLES,
                        subsample_frac=SUBSAMPLE_FRAC, pi_thr=PI_THRESHOLD,
                        n_controls=1):
    """Stability Selection (Meinshausen & Bühlmann, 2010).

    Args:
        n_controls: Number of leading control columns in X_with_trend
            that should be excluded from the selection mask. Default 1
            (trend only) for backward compatibility with Phase 1.
    """
    n, p_full = X_with_trend.shape
    p = p_full - n_controls
    n_sub = max(int(n * subsample_frac), 10)

    selection_counts = np.zeros(p)
    rng = np.random.default_rng(42)

    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X_with_trend)
    l1_ratios = [0.7, 0.9, 0.95, 0.99] if p > n else [0.5, 0.7, 0.9, 0.95]
    enet_cal = ElasticNetCV(
        l1_ratio=l1_ratios, n_alphas=50,
        cv=min(5, max(3, n // 5)), max_iter=10000,
        random_state=42, n_jobs=-1,
    )
    enet_cal.fit(X_scaled_full, y)
    alpha_base = enet_cal.alpha_ / 2.0
    l1_ratio_base = enet_cal.l1_ratio_

    for b in range(n_subsamples):
        idx = rng.choice(n, size=n_sub, replace=False)
        y_sub = y[idx]
        X_sub = X_with_trend[idx]
        scaler = StandardScaler()
        X_sub_scaled = scaler.fit_transform(X_sub)
        enet = ElasticNet(
            alpha=alpha_base, l1_ratio=l1_ratio_base,
            max_iter=10000, random_state=b,
        )
        enet.fit(X_sub_scaled, y_sub)
        coefs = enet.coef_[n_controls:]
        selected = np.abs(coefs) > 1e-10
        selection_counts += selected.astype(float)

    selection_freqs = selection_counts / n_subsamples
    stable_mask = selection_freqs >= pi_thr
    return stable_mask, selection_freqs


# ── OLS post-selection + bootstrap ────────────────────────────────────────

def ols_post_selection(y, X_with_trend, stable_mask, lag_labels,
                        n_bootstrap=N_BOOTSTRAP):
    n = len(y)
    stable_indices = np.where(stable_mask)[0]
    n_stable = len(stable_indices)

    if n_stable == 0:
        return {'cluster_results': {}, 'r2': 0.0, 'n_stable': 0,
                'y_pred': np.full(n, y.mean()), 'residuals': y - y.mean()}

    trend = X_with_trend[:, 0:1]
    X_stable = X_with_trend[:, 1:][:, stable_indices]
    X_ols = np.column_stack([np.ones(n), trend, X_stable])

    try:
        beta_hat, _, _, _ = np.linalg.lstsq(X_ols, y, rcond=None)
    except np.linalg.LinAlgError:
        return {'cluster_results': {}, 'r2': 0.0, 'n_stable': 0,
                'y_pred': np.full(n, y.mean()), 'residuals': y - y.mean()}

    y_pred = X_ols @ beta_hat
    resid = y - y_pred
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    rng = np.random.default_rng(42)
    try:
        XtX_inv = np.linalg.inv(X_ols.T @ X_ols)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X_ols.T @ X_ols)
    hat_matrix = XtX_inv @ X_ols.T

    boot_betas = np.empty((n_bootstrap, len(beta_hat)))
    for b in range(n_bootstrap):
        boot_resid = resid[rng.integers(0, n, size=n)]
        y_boot = y_pred + boot_resid
        boot_betas[b] = hat_matrix @ y_boot

    stable_lag_labels = [lag_labels[i] for i in stable_indices]
    cluster_betas = {}
    cluster_boot = {}
    for j, (cid, lag) in enumerate(stable_lag_labels):
        col = j + 2
        if cid not in cluster_betas:
            cluster_betas[cid] = {}
            cluster_boot[cid] = {}
        cluster_betas[cid][lag] = float(beta_hat[col])
        cluster_boot[cid][lag] = boot_betas[:, col]

    cluster_results = {}
    for cid in cluster_betas:
        betas_by_lag = cluster_betas[cid]
        boots_by_lag = cluster_boot[cid]
        net_beta = sum(betas_by_lag.values())
        net_boot = sum(boots_by_lag.values())
        if net_beta > 0:
            p_val = float(np.mean(net_boot <= 0))
        elif net_beta < 0:
            p_val = float(np.mean(net_boot >= 0))
        else:
            p_val = 1.0
        lag_profile = np.zeros(MAX_LAG + 1)
        for lag, b in betas_by_lag.items():
            lag_profile[lag] = b
        cluster_results[cid] = {
            'net_beta': net_beta, 'p_value': p_val,
            'lag_profile': lag_profile, 'betas_by_lag': betas_by_lag,
        }

    return {'cluster_results': cluster_results, 'r2': r2,
            'n_stable': n_stable, 'y_pred': y_pred, 'residuals': resid}


# ── Per-cascade analysis ─────────────────────────────────────────────────

def analyze_cascade(cascade, frame, articles, cluster_map, composite_signal,
                    embedding_store):
    onset = pd.Timestamp(cascade.onset_date)
    peak = pd.Timestamp(cascade.peak_date)
    end = pd.Timestamp(cascade.end_date)

    year = onset.year
    win_start = max(onset - pd.Timedelta(days=MARGIN_DAYS), pd.Timestamp(f'{year}-01-01'))
    win_end = min(end + pd.Timedelta(days=MARGIN_DAYS), pd.Timestamp(f'{year}-12-31'))
    date_index = pd.date_range(win_start, win_end, freq='D')
    n = len(date_index)
    y = composite_signal.reindex(date_index, fill_value=0.0).values

    cascade_centroid, frame_col = build_cascade_centroid(
        cascade, frame, articles, embedding_store)
    if cascade_centroid is None or frame_col is None:
        return None

    clusters = list(cluster_map.values())
    X, cids, lag_labels, cluster_meta = build_weighted_lagged_mass(
        clusters, articles, date_index, frame_col, cascade_centroid, embedding_store)

    if X.shape[1] == 0:
        return None

    trend = np.arange(n, dtype=np.float64) / n
    X_with_trend = np.column_stack([trend, X])

    stable_mask, sel_freqs = stability_selection(y, X_with_trend)
    n_stable = int(stable_mask.sum())

    if n_stable == 0:
        return StabSelCascadeResult(
            cascade_id=cascade.cascade_id, frame=frame,
            r2=0.0, n_clusters=len(cids), n_stable=0, n_days=n,
            n_drivers=0, n_suppressors=0, n_neutral=0, roles=[],
            cluster_meta=cluster_meta,
            y_obs=y, date_index=date_index, onset=onset, peak=peak, end=end,
        )

    result = ols_post_selection(y, X_with_trend, stable_mask, lag_labels)

    roles = []
    n_d = n_s = n_n = 0

    cluster_sel_freq = {}
    for j, (cid, lag) in enumerate(lag_labels):
        freq = sel_freqs[j]
        cluster_sel_freq[cid] = max(cluster_sel_freq.get(cid, 0.0), freq)

    for cid, cr in result['cluster_results'].items():
        net_beta = cr['net_beta']
        p_val = cr['p_value']
        d_sum = cluster_meta.get(cid, {}).get('D_sum', 0.0)

        if p_val >= ALPHA_SIG:
            role = 'neutral'
            n_n += 1
        elif net_beta > 0:
            role = 'driver'
            n_d += 1
        else:
            role = 'suppressor'
            n_s += 1

        roles.append(ClusterRole(
            cluster_id=cid, net_beta=net_beta, p_value=p_val,
            role=role, lag_profile=cr['lag_profile'],
            selection_freq=cluster_sel_freq.get(cid, 0.0),
            D_sum=d_sum,
        ))

    return StabSelCascadeResult(
        cascade_id=cascade.cascade_id, frame=frame,
        r2=result['r2'], n_clusters=len(cids),
        n_stable=n_stable, n_days=n,
        n_drivers=n_d, n_suppressors=n_s, n_neutral=n_n,
        roles=roles, cluster_meta=cluster_meta,
        y_obs=y, date_index=date_index, onset=onset, peak=peak, end=end,
    )


# ── Profile helpers ────────────────────────────────────────────────────────

# Short labels for frame columns (strip suffix)
FRAME_SHORT = {v: k for k, v in FRAME_COLUMNS.items()}   # e.g. 'cultural_frame' → 'Cult'
MSG_SHORT = {m: m.replace('msg_', '').title() for m in MESSENGERS}
EVT_SHORT = {e: e.replace('evt_', '').title() for e in EVENT_COLUMNS}


def _collect_profiles(cascade_results, role_filter=None):
    """Collect frame/msg/evt profiles from significant clusters.

    Returns dict of lists keyed by canonical names:
      frame_vals: {frame_key: [values...]}  e.g. {'Cult': [...], 'Eco': [...]}
      msg_vals:   {msg_raw: [values...]}    e.g. {'msg_health': [...]}
      evt_vals:   {evt_raw: [values...]}    e.g. {'evt_weather': [...]}
    """
    frame_vals = {f: [] for f in FRAMES}
    msg_vals = {m: [] for m in MESSENGERS}
    evt_vals = {e: [] for e in EVENT_COLUMNS}

    for cr in cascade_results:
        for r in cr.roles:
            if r.role == 'neutral':
                continue
            if role_filter and r.role != role_filter:
                continue
            meta = cr.cluster_meta.get(r.cluster_id, {})
            fp = meta.get('frame_profile', {})
            mp = meta.get('messenger_profile', {})
            ep = meta.get('event_profile', {})
            for f in FRAMES:
                frame_vals[f].append(fp.get(f, 0.0))
            for m in MESSENGERS:
                msg_vals[m].append(mp.get(m, 0.0))
            for e in EVENT_COLUMNS:
                evt_vals[e].append(ep.get(e, 0.0))

    return frame_vals, msg_vals, evt_vals


# ── Orchestrator class ────────────────────────────────────────────────────

class StabSelImpactAnalyzer:
    """Stability Selection impact analysis (Phase 1: cluster → cascade)."""

    def __init__(self, embedding_store):
        self.embedding_store = embedding_store

    def run(self, results) -> StabSelImpactResults:
        """Run StabSel analysis on DetectionResults.

        Reproduces the analyze_year() loop from run_stabsel_production.py.
        """
        articles = results._articles.copy()
        # Resolve date column
        for date_col in ['date', 'date_converted_first']:
            if date_col in articles.columns:
                articles['date'] = pd.to_datetime(articles[date_col], errors='coerce')
                break

        cluster_map = {c.cluster_id: c for c in results.event_clusters}

        # Enrich cluster_map with Phase 2 occurrences from multi-occurrence clusters.
        # This gives StabSel access to both the merged Phase 3 cluster AND its
        # constituent Phase 2 occurrences as separate treatment variables.
        # Stability selection naturally handles the redundancy.
        #
        # IMPORTANT: use seed-only belonging (not Phase 4 soft-assigned) so that
        # the wrapped occurrences match the standalone's Phase 2-only behavior.
        # Phase 4 expands belonging ~4× which changes the D_j signal shape.
        n_added = 0
        for cluster in results.event_clusters:
            if len(cluster.occurrences) <= 1:
                continue  # mono-occurrence cluster — already represented 1:1
            for occ in cluster.occurrences:
                wrapped_id = OCCURRENCE_ID_OFFSET + occ.occurrence_id
                if wrapped_id in cluster_map:
                    continue
                # Restrict belonging to seed articles only (Phase 2 core)
                seed_set = set(occ.seed_doc_ids)
                seed_belonging = {d: b for d, b in occ.belonging.items()
                                  if d in seed_set} if seed_set else occ.belonging
                seed_occ = replace(occ, belonging=seed_belonging)
                cluster_map[wrapped_id] = EventCluster(
                    cluster_id=wrapped_id,
                    occurrences=[seed_occ],
                    event_types={occ.event_type},
                    peak_date=occ.peak_date,
                    core_start=occ.core_start,
                    core_end=occ.core_end,
                    total_mass=occ.effective_mass,
                    centroid=occ.centroid,
                    n_occurrences=1,
                    is_multi_type=False,
                    strength=occ.confidence,
                    entities=occ.entities,
                    dominant_type=occ.event_type,
                )
                n_added += 1
        logger.info(f"  Enriched cluster_map: {len(results.event_clusters)} Phase 3 clusters "
                    f"+ {n_added} Phase 2 occurrences = {len(cluster_map)} treatment variables")

        cascade_map = {c.cascade_id: c for c in results.cascades}
        temporal_index = results._indices.get('temporal', {})

        if not cascade_map:
            return StabSelImpactResults(
                cluster_cascade=pd.DataFrame(),
                summary={},
                cascade_results={},
            )

        all_results_by_frame = self._analyze_all_frames(
            cascade_map, cluster_map, articles, temporal_index, self.embedding_store
        )

        cluster_cascade_df = self._build_cluster_cascade_df(all_results_by_frame)
        summary = self._build_summary(all_results_by_frame)

        return StabSelImpactResults(
            cluster_cascade=cluster_cascade_df,
            summary=summary,
            cascade_results=all_results_by_frame,
        )

    @staticmethod
    def _analyze_all_frames(cascade_map, cluster_map, articles, temporal_index,
                            embedding_store) -> Dict[str, List]:
        """Run StabSel for each frame, each cascade."""
        import time as _time

        valid_frames = [f for f in FRAMES if f in temporal_index
                        and 'daily_proportions' in temporal_index.get(f, {})]
        if not valid_frames:
            logger.warning("  No valid temporal index — skipping StabSel")
            return {}

        # Build composite signals
        composites = {}
        for frame in valid_frames:
            try:
                composite, signals = build_twosided_composite(
                    frame, articles, temporal_index, embedding_store)
                composites[frame] = composite
            except Exception as e:
                logger.warning(f"  Signal build failed for {frame}: {e}")
                continue

        all_results_by_frame = {}

        for frame in valid_frames:
            if frame not in composites:
                continue

            frame_cascades = sorted(
                [c for c in cascade_map.values() if c.frame == frame],
                key=lambda c: c.cascade_id,
            )
            if not frame_cascades:
                all_results_by_frame[frame] = []
                continue

            logger.info(f"  StabSel {frame} ({FRAME_FULL[frame]}): "
                        f"{len(frame_cascades)} cascades")

            cascade_results = []
            for cascade in frame_cascades:
                t1 = _time.time()
                try:
                    cr = analyze_cascade(cascade, frame, articles, cluster_map,
                                         composites[frame], embedding_store)
                except Exception as e:
                    logger.warning(f"    {cascade.cascade_id}: FAILED — {e}")
                    continue
                dt = _time.time() - t1
                if cr is not None:
                    cascade_results.append(cr)
                    if cr.n_drivers + cr.n_suppressors > 0:
                        logger.info(f"    {cascade.cascade_id}: R²={cr.r2:.3f}, "
                                    f"{cr.n_stable} stable → {cr.n_drivers}D "
                                    f"{cr.n_suppressors}S ({dt:.1f}s)")

            all_results_by_frame[frame] = cascade_results

        return all_results_by_frame

    @staticmethod
    def _build_cluster_cascade_df(all_results_by_frame) -> pd.DataFrame:
        """Flatten StabSel results into a cluster_cascade DataFrame."""
        rows = []
        for frame, cascade_results in all_results_by_frame.items():
            for cr in cascade_results:
                for r in cr.roles:
                    if r.role == 'neutral':
                        continue
                    meta = cr.cluster_meta.get(r.cluster_id, {})
                    rows.append({
                        'cluster_id': int(r.cluster_id),
                        'cascade_id': cr.cascade_id,
                        'cascade_frame': frame,
                        'source_level': meta.get('source_level', 'cluster'),
                        'role': r.role,
                        'net_beta': round(r.net_beta, 6),
                        'p_value': round(r.p_value, 6),
                        'selection_freq': round(r.selection_freq, 4),
                        'D_sum': round(r.D_sum, 4),
                        'lag_profile': r.lag_profile.tolist(),
                        'r2': round(cr.r2, 6),
                        'dominant_type': meta.get('dominant_type', 'unknown'),
                        'event_types': meta.get('event_types', []),
                        'entities': meta.get('entities', []),
                        'strength': meta.get('strength', 0.0),
                        'n_articles_weighted': meta.get('n_articles_weighted', 0),
                        'n_articles_total': meta.get('n_articles_total', 0),
                        'peak_date': meta.get('peak_date'),
                        'frame_profile': meta.get('frame_profile', {}),
                        'messenger_profile': meta.get('messenger_profile', {}),
                        'event_profile': meta.get('event_profile', {}),
                    })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    @staticmethod
    def _build_summary(all_results_by_frame) -> Dict[str, Any]:
        """Build per-frame summary dict."""
        summary = {}
        for frame in FRAMES:
            crs = all_results_by_frame.get(frame, [])
            n_d = sum(cr.n_drivers for cr in crs)
            n_s = sum(cr.n_suppressors for cr in crs)
            summary[frame] = {
                'n_cascades': len(crs),
                'n_drivers': n_d,
                'n_suppressors': n_s,
                'median_r2': float(np.median([cr.r2 for cr in crs])) if crs else 0.0,
            }
        return summary
