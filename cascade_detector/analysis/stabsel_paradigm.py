"""
PROJECT: CCF-media-cascade-detection
TITLE: stabsel_paradigm.py — StabSel v2 paradigm impact analysis (cluster/cascade → dominance)

Two models:
  Model A — Event Cluster → Paradigm Dominance (per frame × 8)
    y_t = μ + Σ φ_k·y_{t-k} + trend + Σ_j Σ_{l=0}^{7} β_{j,l}·D_j(t-l) + ε_t
    where D_j(t) = Σ_a belonging(a,j) × frame_signal(a) × cos(a, frame_centroid)

  Model B — Cascade → Paradigm Dominance (per frame × 8)
    y_t = μ + Σ φ_k·y_{t-k} + trend + Σ_c Σ_{l=0}^{7} β_{c,l}·composite_c(t-l) + ε_t

Variable selection: Stability Selection (Meinshausen & Bühlmann, 2010)
  B=100, π≥0.60, ElasticNet
Post-selection: OLS + HAC Newey-West inference (primary) + residual bootstrap (secondary)
Validation: expanding-window temporal CV (3 folds) + train/test 70/30

Classification:
  Per-pair: catalyst/disruptor/inert (Model A), amplification/destabilisation/dormant (Model B)
  Global: reinforcer/challenger/neutral (cosine alignment with paradigm vector)

Author: Antoine Lemor
"""

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats

from cascade_detector.analysis.stabsel_impact import stability_selection
from cascade_detector.core.constants import (
    FRAMES, FRAME_COLUMNS, MESSENGERS, EVENT_COLUMNS,
)
from cascade_detector.core.models import EventCluster

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────��──────────────────────

MAX_LAG_PARADIGM = 7              # 1 week of lags (vs 3 days in Phase 1)
MAX_AR_ORDER = 5                  # max AR order to test
MIN_D_SUM = 0.01                  # minimum total weighted mass to keep a cluster
FRAME_CENTROID_QUANTILE = 0.75    # top-quartile paradigm days for centroid
ALPHA_SIG = 0.10                  # significance threshold
N_BOOTSTRAP = 500                 # bootstrap replications
TRAIN_FRAC = 0.70                 # train/test split fraction
N_CV_FOLDS = 3                    # expanding-window CV folds


# ── Dataclasses ────────────────────────────────────���─────────────────────────

@dataclass
class StabSelParadigmResults:
    """Container for StabSel paradigm analysis outputs (Models A & B)."""
    cluster_dominance: pd.DataFrame    # Model A: cluster × frame (flat)
    cascade_dominance: pd.DataFrame    # Model B: cascade × frame (flat)
    alignment_a: pd.DataFrame          # Model A: global roles per cluster
    alignment_b: pd.DataFrame          # Model B: global roles per cascade
    validation: pd.DataFrame           # R² summary per model × frame
    summary: Dict[str, Any] = field(default_factory=dict)
    raw_results: Dict[str, Any] = field(default_factory=dict)


# ── AR order selection ───────────────────────────────────────────────────────

def select_ar_order(y, max_p=MAX_AR_ORDER):
    """Select AR order by BIC on pure AR(p) + trend model.

    Tests AR(1)..AR(max_p), returns optimal p (minimum 1).
    """
    n = len(y)
    best_bic = np.inf
    best_p = 1

    for p in range(1, max_p + 1):
        if p + 2 >= n:
            break
        Y_trim = y[p:]
        n_trim = len(Y_trim)
        cols = [np.ones(n_trim), np.linspace(0, 1, n_trim)]
        for k in range(1, p + 1):
            cols.append(y[p - k: n - k])
        X_ar = np.column_stack(cols)

        try:
            beta, _, _, _ = np.linalg.lstsq(X_ar, Y_trim, rcond=None)
        except np.linalg.LinAlgError:
            continue

        resid = Y_trim - X_ar @ beta
        ss_res = np.sum(resid ** 2)
        sigma2 = ss_res / n_trim
        if sigma2 <= 0:
            continue
        k_params = p + 2  # intercept + trend + p AR terms
        bic = n_trim * np.log(sigma2) + k_params * np.log(n_trim)

        if bic < best_bic:
            best_bic = bic
            best_p = p

    return best_p


def build_ar_columns(y, p):
    """Build AR(p) lag matrix from y.

    Returns (n, p) array where column k = y_{t-k-1} (1-indexed lag).
    The caller must truncate the first p rows from y and X.
    """
    n = len(y)
    ar_cols = np.zeros((n, p))
    for k in range(1, p + 1):
        ar_cols[k:, k - 1] = y[:-k]
    return ar_cols


# ── OLS post-selection v2 (HAC Newey-West + bootstrap) ────────��─────────────

def ols_post_selection_v2(y, X_with_controls, stable_mask, lag_labels,
                          n_controls, n_bootstrap=N_BOOTSTRAP):
    """OLS on stable variables with AR controls + HAC inference.

    Args:
        y: target series (already trimmed for AR)
        X_with_controls: [controls | treatment_columns] matrix
        stable_mask: boolean mask over treatment_columns only (len = n_treatment)
        lag_labels: list of (cluster_id, lag) for each treatment column
        n_controls: number of control columns (trend + AR)
        n_bootstrap: number of bootstrap replications

    Returns dict with:
        cluster_results: {cid: {net_beta, p_value_hac, p_value_boot, lag_profile, ...}}
        r2: full-sample R²
        n_stable: number of stable variables
        y_pred, residuals: fitted values and residuals
    """
    n = len(y)
    stable_indices = np.where(stable_mask)[0]
    n_stable = len(stable_indices)

    if n_stable == 0:
        return {'cluster_results': {}, 'r2': 0.0, 'n_stable': 0,
                'y_pred': np.full(n, y.mean()), 'residuals': y - y.mean()}

    # Build OLS design: intercept + controls + stable treatment columns
    controls = X_with_controls[:, :n_controls]
    X_treatment = X_with_controls[:, n_controls:]
    X_stable = X_treatment[:, stable_indices]
    X_ols = np.column_stack([np.ones(n), controls, X_stable])

    n_intercept_controls = 1 + n_controls
    n_params = X_ols.shape[1]

    # ── HAC OLS via statsmodels ──
    try:
        model = sm.OLS(y, X_ols)
        result = model.fit(cov_type='HAC', cov_kwds={'maxlags': None})
        beta_hat = result.params
        hac_cov = result.cov_params()
        y_pred = result.fittedvalues
        resid = result.resid
        r2 = result.rsquared
    except Exception as e:
        logger.warning(f"HAC OLS failed: {e}, falling back to numpy")
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
        hac_cov = None

    # ── Residual bootstrap (secondary) ──
    rng = np.random.default_rng(42)
    try:
        XtX_inv = np.linalg.inv(X_ols.T @ X_ols)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X_ols.T @ X_ols)
    hat_matrix = XtX_inv @ X_ols.T

    boot_betas = np.empty((n_bootstrap, n_params))
    for b in range(n_bootstrap):
        boot_resid = resid[rng.integers(0, n, size=n)]
        y_boot = y_pred + boot_resid
        boot_betas[b] = hat_matrix @ y_boot

    # ── Aggregate per cluster ──
    stable_lag_labels = [lag_labels[i] for i in stable_indices]
    cluster_betas = {}
    cluster_boot = {}
    cluster_col_indices = {}

    for j, (cid, lag) in enumerate(stable_lag_labels):
        col = j + n_intercept_controls
        if cid not in cluster_betas:
            cluster_betas[cid] = {}
            cluster_boot[cid] = {}
            cluster_col_indices[cid] = []
        cluster_betas[cid][lag] = float(beta_hat[col])
        cluster_boot[cid][lag] = boot_betas[:, col]
        cluster_col_indices[cid].append(col)

    cluster_results = {}
    for cid in cluster_betas:
        betas_by_lag = cluster_betas[cid]
        boots_by_lag = cluster_boot[cid]
        net_beta = sum(betas_by_lag.values())
        net_boot = sum(boots_by_lag.values())

        # HAC p-value via Wald test: H0: Σ β_lag = 0
        p_hac = 1.0
        if hac_cov is not None:
            cols = cluster_col_indices[cid]
            R = np.zeros(n_params)
            R[cols] = 1.0
            var_net = R @ hac_cov @ R
            if var_net > 0:
                t_stat = net_beta / np.sqrt(var_net)
                p_two = 2.0 * (1.0 - sp_stats.t.cdf(abs(t_stat), df=n - n_params))
                p_hac = p_two / 2.0  # one-tailed in direction of net_beta

        # Bootstrap p-value
        if net_beta > 0:
            p_boot = float(np.mean(net_boot <= 0))
        elif net_beta < 0:
            p_boot = float(np.mean(net_boot >= 0))
        else:
            p_boot = 1.0

        lag_profile = np.zeros(MAX_LAG_PARADIGM + 1)
        for lag, b in betas_by_lag.items():
            if lag < len(lag_profile):
                lag_profile[lag] = b

        cluster_results[cid] = {
            'net_beta': net_beta,
            'p_value_hac': p_hac,
            'p_value_boot': p_boot,
            'p_value': p_hac,  # primary = HAC
            'lag_profile': lag_profile,
            'betas_by_lag': betas_by_lag,
        }

    return {'cluster_results': cluster_results, 'r2': r2,
            'n_stable': n_stable, 'y_pred': y_pred, 'residuals': resid}


# ── Temporal cross-validation ────────────────���───────────────────────────────

def temporal_cross_validation(y, X_with_controls, lag_labels, n_controls,
                              n_folds=N_CV_FOLDS):
    """Expanding-window temporal CV with honest StabSel on each fold.

    Folds divide the data into n_folds+1 equal segments.
    Fold k: train on [0 : (k+1)*seg], test on [(k+1)*seg : (k+2)*seg].
    """
    n = len(y)
    seg = n // (n_folds + 1)
    if seg < 30:
        logger.warning(f"CV segments too small ({seg} days), skipping CV")
        return {'r2_per_fold': [], 'r2_cv': np.nan, 'stable_per_fold': []}

    r2_folds = []
    stable_folds = []

    for fold in range(n_folds):
        train_end = (fold + 1) * seg + seg
        test_end = min(train_end + seg, n)
        if test_end <= train_end:
            break

        y_train = y[:train_end]
        y_test = y[train_end:test_end]
        X_train = X_with_controls[:train_end]
        X_test = X_with_controls[train_end:test_end]

        stable_mask, _ = stability_selection(y_train, X_train)
        n_stable = stable_mask.sum()
        stable_folds.append(n_stable)

        if n_stable == 0:
            y_pred_test = np.full(len(y_test), y_train.mean())
        else:
            controls_train = X_train[:, :n_controls]
            X_treat_train = X_train[:, n_controls:][:, np.where(stable_mask)[0]]
            X_ols_train = np.column_stack([np.ones(len(y_train)),
                                           controls_train, X_treat_train])
            try:
                beta, _, _, _ = np.linalg.lstsq(X_ols_train, y_train, rcond=None)
            except np.linalg.LinAlgError:
                y_pred_test = np.full(len(y_test), y_train.mean())
                ss_res = np.sum((y_test - y_pred_test) ** 2)
                ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                r2_folds.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
                continue

            controls_test = X_test[:, :n_controls]
            X_treat_test = X_test[:, n_controls:][:, np.where(stable_mask)[0]]
            X_ols_test = np.column_stack([np.ones(len(y_test)),
                                          controls_test, X_treat_test])
            y_pred_test = X_ols_test @ beta

        ss_res = np.sum((y_test - y_pred_test) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2_test = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_folds.append(r2_test)

    return {
        'r2_per_fold': r2_folds,
        'r2_cv': float(np.mean(r2_folds)) if r2_folds else np.nan,
        'stable_per_fold': stable_folds,
    }


# ── Train/test split evaluation ───────────���──────────────────────────────────

def train_test_evaluation(y, X_with_controls, lag_labels, n_controls,
                          train_frac=TRAIN_FRAC):
    """Single train/test split (70/30)."""
    n = len(y)
    split = int(n * train_frac)

    y_train, y_test = y[:split], y[split:]
    X_train, X_test = X_with_controls[:split], X_with_controls[split:]

    if len(y_test) < 10:
        return {'r2_train': np.nan, 'r2_test': np.nan, 'n_stable_train': 0}

    stable_mask, _ = stability_selection(y_train, X_train)
    n_stable = stable_mask.sum()

    if n_stable == 0:
        y_pred_train = np.full(len(y_train), y_train.mean())
        y_pred_test = np.full(len(y_test), y_train.mean())
    else:
        controls_train = X_train[:, :n_controls]
        X_treat_train = X_train[:, n_controls:][:, np.where(stable_mask)[0]]
        X_ols_train = np.column_stack([np.ones(len(y_train)),
                                       controls_train, X_treat_train])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_ols_train, y_train, rcond=None)
        except np.linalg.LinAlgError:
            return {'r2_train': np.nan, 'r2_test': np.nan, 'n_stable_train': 0}

        y_pred_train = X_ols_train @ beta

        controls_test = X_test[:, :n_controls]
        X_treat_test = X_test[:, n_controls:][:, np.where(stable_mask)[0]]
        X_ols_test = np.column_stack([np.ones(len(y_test)),
                                      controls_test, X_treat_test])
        y_pred_test = X_ols_test @ beta

    ss_res_tr = np.sum((y_train - y_pred_train) ** 2)
    ss_tot_tr = np.sum((y_train - y_train.mean()) ** 2)
    r2_train = 1.0 - ss_res_tr / ss_tot_tr if ss_tot_tr > 0 else 0.0

    ss_res_te = np.sum((y_test - y_pred_test) ** 2)
    ss_tot_te = np.sum((y_test - y_test.mean()) ** 2)
    r2_test = 1.0 - ss_res_te / ss_tot_te if ss_tot_te > 0 else 0.0

    return {'r2_train': r2_train, 'r2_test': r2_test, 'n_stable_train': n_stable}


# ── Impact metrics ───────────────────────────────────────────��───────────────

def compute_impact_magnitude(beta_vec, source_weight):
    """Impact magnitude = ||β|| × source_weight."""
    return float(np.linalg.norm(beta_vec) * source_weight)


def compute_shift_contribution(beta_vec, paradigm_indexed, peak_date, frames,
                                window=7):
    """How much of the observed paradigm change is attributable.

    shift_contribution = Σ_f |β_f| × |Δparadigm_f(peak ± window)|
    """
    peak_dt = pd.Timestamp(peak_date)
    dists = abs(paradigm_indexed.index - peak_dt)
    nearest_idx = dists.argmin()

    before_idx = max(0, nearest_idx - window)
    after_idx = min(len(paradigm_indexed) - 1, nearest_idx + window)

    contribution = 0.0
    for i, f in enumerate(frames):
        col = f'paradigm_{f}'
        if col in paradigm_indexed.columns:
            delta = abs(paradigm_indexed.iloc[after_idx][col] -
                       paradigm_indexed.iloc[before_idx][col])
            contribution += abs(beta_vec[i]) * delta
    return float(contribution)


# ── Frame centroid construction ─────────��────────────────────────────────────

def build_frame_centroid(frame, articles, paradigm_timeline, embedding_store):
    """Build embedding centroid for 'what discourse looks like when frame dominates'.

    Uses top-quartile paradigm dominance days to select representative articles,
    then builds signal-weighted centroid from their embeddings.
    """
    paradigm_col = f'paradigm_{frame}'
    if paradigm_col not in paradigm_timeline.columns:
        logger.warning(f"No {paradigm_col} in paradigm timeline")
        return None, None

    frame_col_base = FRAME_COLUMNS.get(frame)
    frame_col = None
    for candidate in [f"{frame_col_base}_mean", f"{frame_col_base}_sum", frame_col_base]:
        if candidate in articles.columns:
            frame_col = candidate
            break
    if frame_col is None:
        logger.warning(f"No frame column for {frame} in articles")
        return None, None

    threshold = paradigm_timeline[paradigm_col].quantile(FRAME_CENTROID_QUANTILE)
    high_days = paradigm_timeline[paradigm_timeline[paradigm_col] >= threshold]['date']
    high_days_set = set(pd.to_datetime(high_days).dt.date)

    if not high_days_set:
        logger.warning(f"No high-dominance days for {frame}")
        return None, frame_col

    articles_dated = articles.copy()
    articles_dated['_date_key'] = pd.to_datetime(articles_dated['date']).dt.date
    mask = articles_dated['_date_key'].isin(high_days_set) & (articles_dated[frame_col] > 0)
    frame_articles = articles_dated[mask]

    if len(frame_articles) < 5:
        frame_articles = articles[articles[frame_col] > 0]
        if len(frame_articles) < 5:
            logger.warning(f"Not enough frame-active articles for {frame}")
            return None, frame_col

    q75 = frame_articles[frame_col].quantile(0.75)
    top_articles = frame_articles[frame_articles[frame_col] >= q75]
    if len(top_articles) < 3:
        top_articles = frame_articles

    doc_ids = top_articles['doc_id'].dropna().unique().tolist()
    embs, found_ids = embedding_store.get_batch_article_embeddings(doc_ids)
    if len(embs) < 3:
        logger.warning(f"Not enough embeddings for {frame} centroid")
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


# ── Weighted lagged mass matrix (Model A, MAX_LAG=7) ────────────────────────

def build_weighted_lagged_mass_paradigm(clusters, articles, date_index, frame_col,
                                         frame_centroid, embedding_store):
    """Build triple-weighted lagged mass matrix for paradigm StabSel.

    Returns:
        X: (n_days, n_clusters × (MAX_LAG_PARADIGM+1)) matrix
        cids: unique cluster IDs retained
        lag_labels: list of (cid, lag) tuples
        cluster_meta: metadata dict per cluster
    """
    n = len(date_index)
    win_start = date_index[0]
    win_end = date_index[-1]

    art_frame = articles.set_index('doc_id')[frame_col].to_dict()
    art_date = articles.set_index('doc_id')['date'].to_dict()

    all_doc_ids = articles['doc_id'].dropna().unique().tolist()
    all_embs, all_found_ids = embedding_store.get_batch_article_embeddings(all_doc_ids)
    emb_norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    emb_norms = np.maximum(emb_norms, 1e-10)
    all_embs_normed = all_embs / emb_norms
    cosine_sims = all_embs_normed @ frame_centroid
    cosine_lookup = dict(zip(all_found_ids, cosine_sims.tolist()))

    raw_weighted = {}
    cluster_meta = {}

    for cluster in clusters:
        cid = cluster.cluster_id
        if cluster.core_start and cluster.core_end:
            cs = pd.Timestamp(cluster.core_start)
            ce = pd.Timestamp(cluster.core_end)
            if ce < win_start - pd.Timedelta(days=MAX_LAG_PARADIGM) or cs > win_end:
                continue

        combined = {}
        for occ in cluster.occurrences:
            for doc_id, belonging in occ.belonging.items():
                combined[doc_id] = max(combined.get(doc_id, 0.0), belonging)
        if not combined:
            continue

        daily_weighted = {}
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
            if weight > 0:
                daily_weighted[date] = daily_weighted.get(date, 0.0) + weight

        if not daily_weighted:
            continue

        dm = pd.Series(daily_weighted)
        aligned = dm.reindex(date_index, fill_value=0.0).values
        if aligned.sum() < MIN_D_SUM:
            continue

        raw_weighted[cid] = aligned
        cluster_meta[cid] = {
            'cluster_id': int(cid),
            'dominant_type': getattr(cluster, 'dominant_type', 'unknown'),
            'event_types': list(getattr(cluster, 'event_types', set()) or set()),
            'strength': round(float(getattr(cluster, 'strength', 0.0)), 4),
            'n_articles': len(combined),
            'D_sum': round(float(aligned.sum()), 4),
            'peak_date': str(dm.idxmax().date()) if not dm.empty else None,
        }

    if not raw_weighted:
        return np.empty((n, 0)), [], [], {}

    cids = list(raw_weighted.keys())
    cols = []
    lag_labels = []
    for cid in cids:
        m = raw_weighted[cid]
        for lag in range(MAX_LAG_PARADIGM + 1):
            if lag == 0:
                cols.append(m.copy())
            else:
                lagged = np.zeros(n)
                lagged[lag:] = m[:-lag]
                cols.append(lagged)
            lag_labels.append((cid, lag))
    X = np.column_stack(cols)
    return X, cids, lag_labels, cluster_meta


# ── Build cascade lagged matrix (Model B) ───────────────────────────────────

def build_cascade_lagged_matrix(cascades, date_index):
    """Build lagged daily_composite matrix for all cascades."""
    n = len(date_index)
    raw = {}

    for cascade in cascades:
        cid = cascade.cascade_id
        dc = cascade.daily_composite
        if dc is None or dc.empty:
            continue
        aligned = dc.reindex(date_index, fill_value=0.0).values
        if aligned.sum() < MIN_D_SUM:
            continue
        raw[cid] = aligned

    if not raw:
        return np.empty((n, 0)), [], []

    cascade_ids = list(raw.keys())
    cols = []
    lag_labels = []
    for cid in cascade_ids:
        m = raw[cid]
        for lag in range(MAX_LAG_PARADIGM + 1):
            if lag == 0:
                cols.append(m.copy())
            else:
                lagged = np.zeros(n)
                lagged[lag:] = m[:-lag]
                cols.append(lagged)
            lag_labels.append((cid, lag))
    X = np.column_stack(cols)
    return X, cascade_ids, lag_labels


# ── Role classification (per-pair) ──────────────────────────────────────────

def classify_model_a_roles(ols_results):
    """Classify cluster roles: catalyst/disruptor/inert (per frame pair)."""
    roles = {}
    for cid, res in ols_results.get('cluster_results', {}).items():
        nb = res['net_beta']
        pv = res['p_value']
        if pv < ALPHA_SIG:
            roles[cid] = 'catalyst' if nb > 0 else 'disruptor'
        else:
            roles[cid] = 'inert'
    return roles


def classify_model_b_roles(ols_results, target_frame, cascades_by_id):
    """Classify cascade roles: catalyst/disruptor/inert (per frame pair).

    Aligned with Model A taxonomy.  The ``is_own_frame`` flag allows
    downstream code to derive the legacy labels:

    - amplification  = catalyst  + is_own_frame
    - auto-suppression = disruptor + is_own_frame
    - catalysis      = catalyst  + not is_own_frame
    - destabilisation = disruptor + not is_own_frame
    - dormant        = inert

    Returns:
        Tuple[dict, dict]: (roles, own_frame_flags) keyed by cascade id.
    """
    roles = {}
    own_frame_flags = {}
    for cid, res in ols_results.get('cluster_results', {}).items():
        nb = res['net_beta']
        pv = res['p_value']
        cascade = cascades_by_id.get(cid)
        is_own = (cascade.frame == target_frame) if cascade else False
        own_frame_flags[cid] = is_own

        if pv < ALPHA_SIG:
            roles[cid] = 'catalyst' if nb > 0 else 'disruptor'
        else:
            roles[cid] = 'inert'
    return roles, own_frame_flags


# ── Model runners ────────────────────────────────────────���───────────────────

def run_model(model_type, frame, y_full, X_treatment, lag_labels,
              cascades_by_id=None, cluster_meta=None, entity_ids=None):
    """Unified model runner for both Model A and Model B.

    Pipeline: AR selection → trim → StabSel → HAC OLS → train/test → CV → classify
    """
    n_treat_cols = X_treatment.shape[1]

    if n_treat_cols == 0:
        return None

    # 1. AR order selection
    ar_order = select_ar_order(y_full)
    ar_cols = build_ar_columns(y_full, ar_order)

    # Trim first ar_order rows
    y = y_full[ar_order:]
    X_treat_trimmed = X_treatment[ar_order:]
    ar_trimmed = ar_cols[ar_order:]
    n = len(y)

    # 2. Build control + treatment matrix
    trend = np.linspace(0, 1, n)
    controls = np.column_stack([trend, ar_trimmed])
    n_controls = controls.shape[1]  # 1 (trend) + ar_order

    X_with_controls = np.column_stack([controls, X_treat_trimmed])

    # 3. Stability Selection (protect controls)
    logger.info(f"Model {model_type} [{frame}]: AR({ar_order}), "
                f"StabSel on {n_treat_cols} treatment cols...")
    t0 = time.time()
    full_stable_mask, selection_freqs = stability_selection(y, X_with_controls)

    full_stable_mask[:n_controls] = False
    treatment_stable_mask = full_stable_mask[n_controls:]
    n_stable = treatment_stable_mask.sum()
    elapsed = time.time() - t0
    logger.info(f"Model {model_type} [{frame}]: {n_stable} stable "
                f"(of {n_treat_cols}) in {elapsed:.1f}s")

    # 4. OLS with HAC + bootstrap
    if n_stable == 0:
        ols_results = {'cluster_results': {}, 'r2': 0.0, 'n_stable': 0,
                       'y_pred': np.full(n, y.mean()), 'residuals': y - y.mean()}
    else:
        ols_results = ols_post_selection_v2(
            y, X_with_controls, treatment_stable_mask, lag_labels, n_controls)

    # 5. Train/test split
    tt = train_test_evaluation(y, X_with_controls, lag_labels, n_controls)

    # 6. Cross-validation
    cv = temporal_cross_validation(y, X_with_controls, lag_labels, n_controls)

    # 7. Classify roles
    if model_type == 'A':
        roles = classify_model_a_roles(ols_results)
        for cid, res in ols_results['cluster_results'].items():
            res['role'] = roles.get(cid, 'inert')
            if cluster_meta and cid in cluster_meta:
                res.update(cluster_meta[cid])
    else:
        roles, own_frame_flags = classify_model_b_roles(
            ols_results, frame, cascades_by_id)
        for cid, res in ols_results['cluster_results'].items():
            res['role'] = roles.get(cid, 'inert')
            res['is_own_frame'] = own_frame_flags.get(cid, False)
            cascade = cascades_by_id.get(cid) if cascades_by_id else None
            if cascade:
                res['cascade_frame'] = cascade.frame
                res['cascade_id'] = cid

    n_entities = len(entity_ids) if entity_ids else 0
    result = {
        'frame': frame,
        'model': model_type,
        'ar_order': ar_order,
        'n_entities': n_entities,
        'n_columns': n_treat_cols,
        'n_stable': n_stable,
        'r2_full': ols_results['r2'],
        'r2_train': tt['r2_train'],
        'r2_test': tt['r2_test'],
        'r2_cv': cv['r2_cv'],
        'r2_cv_folds': cv['r2_per_fold'],
        'stable_cv_folds': cv['stable_per_fold'],
        'roles': roles,
        'cluster_results': ols_results.get('cluster_results', {}),
    }
    if cluster_meta:
        result['cluster_meta'] = cluster_meta

    return result


def run_model_a(frame, clusters, articles, paradigm_timeline, embedding_store):
    """Run Model A: Cluster -> Paradigm for a single frame."""
    paradigm_col = f'paradigm_{frame}'
    paradigm_series = paradigm_timeline.set_index('date')[paradigm_col]
    date_index = pd.DatetimeIndex(paradigm_series.index)
    y_full = paradigm_series.values.astype(float)

    frame_centroid, frame_col = build_frame_centroid(
        frame, articles, paradigm_timeline, embedding_store)
    if frame_centroid is None:
        logger.warning(f"Model A [{frame}]: no frame centroid, skipping")
        return None

    logger.info(f"Model A [{frame}]: building weighted lagged mass (MAX_LAG={MAX_LAG_PARADIGM})...")
    t0 = time.time()
    X, cids, lag_labels, cluster_meta = build_weighted_lagged_mass_paradigm(
        clusters, articles, date_index, frame_col, frame_centroid, embedding_store)
    elapsed = time.time() - t0
    logger.info(f"Model A [{frame}]: {len(cids)} clusters, X {X.shape}, {elapsed:.1f}s")

    if X.shape[1] == 0:
        logger.warning(f"Model A [{frame}]: no clusters with mass, skipping")
        return None

    return run_model(
        'A', frame, y_full, X, lag_labels,
        cluster_meta=cluster_meta, entity_ids=cids)


def run_model_b(frame, cascades, paradigm_timeline, cascades_by_id):
    """Run Model B: Cascade -> Paradigm for a single frame."""
    paradigm_col = f'paradigm_{frame}'
    paradigm_series = paradigm_timeline.set_index('date')[paradigm_col]
    date_index = pd.DatetimeIndex(paradigm_series.index)
    y_full = paradigm_series.values.astype(float)

    X, cascade_ids, lag_labels = build_cascade_lagged_matrix(cascades, date_index)
    logger.info(f"Model B [{frame}]: {len(cascade_ids)} cascades, X {X.shape}")

    if X.shape[1] == 0:
        logger.warning(f"Model B [{frame}]: no cascades with composite, skipping")
        return None

    return run_model(
        'B', frame, y_full, X, lag_labels,
        cascades_by_id=cascades_by_id, entity_ids=cascade_ids)


# ── Results export helpers ──��────────────────────────────────────────────────

def results_to_dataframe_a(all_results):
    """Convert Model A results to a flat DataFrame."""
    rows = []
    for res in all_results:
        if res is None:
            continue
        frame = res['frame']
        for cid, cr in res.get('cluster_results', {}).items():
            rows.append({
                'frame': frame,
                'cluster_id': cid,
                'net_beta': cr['net_beta'],
                'p_value_hac': cr.get('p_value_hac', np.nan),
                'p_value_boot': cr.get('p_value_boot', np.nan),
                'role': cr.get('role', 'inert'),
                'dominant_type': cr.get('dominant_type', ''),
                'event_types': str(cr.get('event_types', [])),
                'strength': cr.get('strength', 0.0),
                'D_sum': cr.get('D_sum', 0.0),
                'peak_date': cr.get('peak_date', ''),
                'ar_order': res.get('ar_order', 0),
                'r2_full': res.get('r2_full', 0.0),
                'r2_test': res.get('r2_test', np.nan),
                'r2_cv': res.get('r2_cv', np.nan),
            })
    return pd.DataFrame(rows)


def results_to_dataframe_b(all_results):
    """Convert Model B results to a flat DataFrame."""
    rows = []
    for res in all_results:
        if res is None:
            continue
        frame = res['frame']
        for cid, cr in res.get('cluster_results', {}).items():
            rows.append({
                'target_frame': frame,
                'cascade_id': cid,
                'cascade_frame': cr.get('cascade_frame', ''),
                'net_beta': cr['net_beta'],
                'p_value_hac': cr.get('p_value_hac', np.nan),
                'p_value_boot': cr.get('p_value_boot', np.nan),
                'role': cr.get('role', 'inert'),
                'is_own_frame': cr.get('is_own_frame', False),
                'ar_order': res.get('ar_order', 0),
                'r2_full': res.get('r2_full', 0.0),
                'r2_test': res.get('r2_test', np.nan),
                'r2_cv': res.get('r2_cv', np.nan),
            })
    return pd.DataFrame(rows)


# ── Orchestrator class ───────────────────────────────────────────────────────

class StabSelParadigmAnalyzer:
    """Stability Selection paradigm impact analysis (cluster/cascade -> paradigm dominance)."""

    def __init__(self, embedding_store):
        self.embedding_store = embedding_store

    def run(self, results) -> StabSelParadigmResults:
        """Run Model A + Model B on all frames.

        Requires results.paradigm_shifts (Step 4 completed).
        """
        # 1. Validate inputs
        if results.paradigm_shifts is None:
            raise ValueError("paradigm_shifts is None — run Step 4 first")

        paradigm_timeline = results.paradigm_shifts.paradigm_timeline
        if paradigm_timeline is None or paradigm_timeline.empty:
            raise ValueError("paradigm_timeline is empty")

        paradigm_timeline = paradigm_timeline.copy()
        paradigm_timeline['date'] = pd.to_datetime(paradigm_timeline['date'])

        # 2. Prepare data
        articles = results._articles.copy()
        for date_col in ['date', 'date_converted_first']:
            if date_col in articles.columns:
                articles['date'] = pd.to_datetime(articles[date_col], errors='coerce')
                break

        clusters = results.event_clusters
        cascades = results.cascades
        cascades_by_id = {c.cascade_id: c for c in cascades}

        # 3. Run Model A (cluster -> paradigm)
        logger.info("  Model A: Cluster -> Paradigm Dominance")
        results_a = []
        validation_rows = []

        for frame in FRAMES:
            res = run_model_a(frame, clusters, articles, paradigm_timeline,
                             self.embedding_store)
            results_a.append(res)
            if res:
                validation_rows.append({
                    'model': 'A', 'frame': frame,
                    'ar_order': res['ar_order'],
                    'n_entities': res['n_entities'],
                    'n_stable': res['n_stable'],
                    'r2_full': res['r2_full'],
                    'r2_train': res['r2_train'],
                    'r2_test': res['r2_test'],
                    'r2_cv': res['r2_cv'],
                })

        # 4. Run Model B (cascade -> paradigm)
        logger.info("  Model B: Cascade -> Paradigm Dominance")
        results_b = []
        for frame in FRAMES:
            res = run_model_b(frame, cascades, paradigm_timeline, cascades_by_id)
            results_b.append(res)
            if res:
                validation_rows.append({
                    'model': 'B', 'frame': frame,
                    'ar_order': res['ar_order'],
                    'n_entities': res['n_entities'],
                    'n_stable': res['n_stable'],
                    'r2_full': res['r2_full'],
                    'r2_train': res['r2_train'],
                    'r2_test': res['r2_test'],
                    'r2_cv': res['r2_cv'],
                })

        # 5. Build flat DataFrames
        df_a = results_to_dataframe_a(results_a)
        df_b = results_to_dataframe_b(results_b)
        df_validation = pd.DataFrame(validation_rows)

        # 6. Build alignment tables
        paradigm_indexed = paradigm_timeline.set_index('date')
        df_align_a = self._build_alignment_a(df_a, paradigm_indexed, clusters)
        df_align_b = self._build_alignment_b(df_b, paradigm_indexed, cascades_by_id)

        # 7. Build summary
        summary = self._build_summary(results_a, results_b, df_a, df_b,
                                       df_align_a, df_align_b)

        # 8. Raw results for pickle
        raw_results = {
            'model_a': {res['frame']: res for res in results_a if res},
            'model_b': {res['frame']: res for res in results_b if res},
        }

        logger.info(f"  Model A: {len(df_a)} cluster-frame pairs, "
                    f"Model B: {len(df_b)} cascade-frame pairs")

        return StabSelParadigmResults(
            cluster_dominance=df_a,
            cascade_dominance=df_b,
            alignment_a=df_align_a,
            alignment_b=df_align_b,
            validation=df_validation,
            summary=summary,
            raw_results=raw_results,
        )

    @staticmethod
    def _build_alignment_a(df_a, paradigm_indexed, clusters):
        """Build Model A alignment table (cluster-level global roles)."""
        if df_a.empty:
            return pd.DataFrame()

        rows = []
        sig_a = df_a[df_a['p_value_hac'] < ALPHA_SIG]
        for cid, grp in sig_a.groupby('cluster_id'):
            beta_vec = np.zeros(len(FRAMES))
            for _, row in grp.iterrows():
                fidx = FRAMES.index(row['frame'])
                beta_vec[fidx] = row['net_beta']

            peak = grp.iloc[0].get('peak_date', '')
            if peak and peak != '':
                peak_dt = pd.Timestamp(peak)
            else:
                peak_dt = paradigm_indexed.index[len(paradigm_indexed) // 2]

            dists = abs(paradigm_indexed.index - peak_dt)
            nearest_idx = dists.argmin()
            paradigm_vec = np.array([
                paradigm_indexed.iloc[nearest_idx][f'paradigm_{f}']
                for f in FRAMES
            ])

            norm_b = np.linalg.norm(beta_vec)
            norm_p = np.linalg.norm(paradigm_vec)
            cos_sim = (np.dot(beta_vec, paradigm_vec) / (norm_b * norm_p)
                      if norm_b > 0 and norm_p > 0 else 0.0)

            if cos_sim > 0.1:
                global_role = 'reinforcer'
            elif cos_sim < -0.1:
                global_role = 'challenger'
            else:
                global_role = 'neutral'

            strength = grp.iloc[0].get('strength', 0.0)
            magnitude = compute_impact_magnitude(beta_vec, strength)
            shift_contrib = compute_shift_contribution(
                beta_vec, paradigm_indexed, peak_dt, FRAMES)

            rows.append({
                'cluster_id': cid,
                'n_sig_frames': len(grp),
                'beta_norm': round(norm_b, 4),
                'cos_alignment': round(cos_sim, 4),
                'global_role': global_role,
                'impact_magnitude': round(magnitude, 4),
                'shift_contribution': round(shift_contrib, 6),
                'dominant_type': grp.iloc[0].get('dominant_type', ''),
                'strength': strength,
                'peak_date': str(peak),
                'sig_frames': ', '.join(
                    f"{r['frame']}({r['net_beta']:+.2f})" for _, r in grp.iterrows()),
            })

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values('impact_magnitude', ascending=False)

    @staticmethod
    def _build_alignment_b(df_b, paradigm_indexed, cascades_by_id):
        """Build Model B alignment table (cascade-level global roles)."""
        if df_b.empty:
            return pd.DataFrame()

        rows = []
        sig_b = df_b[df_b['p_value_hac'] < ALPHA_SIG]
        for cid, grp in sig_b.groupby('cascade_id'):
            beta_vec = np.zeros(len(FRAMES))
            for _, row in grp.iterrows():
                fidx = FRAMES.index(row['target_frame'])
                beta_vec[fidx] = row['net_beta']

            cascade = cascades_by_id.get(cid)
            if cascade and cascade.peak_date is not None:
                peak_dt = pd.Timestamp(cascade.peak_date)
            else:
                peak_dt = paradigm_indexed.index[len(paradigm_indexed) // 2]

            dists = abs(paradigm_indexed.index - peak_dt)
            nearest_idx = dists.argmin()
            paradigm_vec = np.array([
                paradigm_indexed.iloc[nearest_idx][f'paradigm_{f}']
                for f in FRAMES
            ])

            norm_b = np.linalg.norm(beta_vec)
            norm_p = np.linalg.norm(paradigm_vec)
            cos_sim = (np.dot(beta_vec, paradigm_vec) / (norm_b * norm_p)
                      if norm_b > 0 and norm_p > 0 else 0.0)

            if cos_sim > 0.1:
                global_role = 'reinforcer'
            elif cos_sim < -0.1:
                global_role = 'challenger'
            else:
                global_role = 'neutral'

            source_weight = cascade.total_score if cascade else 0.0
            magnitude = compute_impact_magnitude(beta_vec, source_weight)
            shift_contrib = compute_shift_contribution(
                beta_vec, paradigm_indexed, peak_dt, FRAMES)

            rows.append({
                'cascade_id': cid,
                'cascade_frame': cascade.frame if cascade else '',
                'n_sig_frames': len(grp),
                'beta_norm': round(norm_b, 4),
                'cos_alignment': round(cos_sim, 4),
                'global_role': global_role,
                'impact_magnitude': round(magnitude, 4),
                'shift_contribution': round(shift_contrib, 6),
                'peak_date': str(cascade.peak_date.date()) if cascade and cascade.peak_date else '',
                'total_score': round(source_weight, 4),
                'sig_frames': ', '.join(
                    f"{r['target_frame']}({r['net_beta']:+.2f})" for _, r in grp.iterrows()),
            })

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values('impact_magnitude', ascending=False)

    @staticmethod
    def _build_summary(results_a, results_b, df_a, df_b, df_align_a, df_align_b):
        """Build per-model, per-frame summary."""
        summary = {'model_a': {}, 'model_b': {}, 'global': {}}

        # Model A per-frame
        for res in results_a:
            if res is None:
                continue
            frame = res['frame']
            frame_df = df_a[df_a['frame'] == frame] if not df_a.empty else pd.DataFrame()
            n_sig = int((frame_df['p_value_hac'] < ALPHA_SIG).sum()) if not frame_df.empty else 0
            summary['model_a'][frame] = {
                'n_clusters': res['n_entities'],
                'n_stable': res['n_stable'],
                'n_significant_hac': n_sig,
                'n_catalyst': int((frame_df['role'] == 'catalyst').sum()) if not frame_df.empty else 0,
                'n_disruptor': int((frame_df['role'] == 'disruptor').sum()) if not frame_df.empty else 0,
                'ar_order': res['ar_order'],
                'r2_full': round(res['r2_full'], 4),
                'r2_test': round(res['r2_test'], 4) if not np.isnan(res['r2_test']) else None,
                'r2_cv': round(res['r2_cv'], 4) if not np.isnan(res['r2_cv']) else None,
            }

        # Model B per-frame
        for res in results_b:
            if res is None:
                continue
            frame = res['frame']
            frame_df = df_b[df_b['target_frame'] == frame] if not df_b.empty else pd.DataFrame()
            n_sig = int((frame_df['p_value_hac'] < ALPHA_SIG).sum()) if not frame_df.empty else 0
            summary['model_b'][frame] = {
                'n_cascades': res['n_entities'],
                'n_stable': res['n_stable'],
                'n_significant_hac': n_sig,
                'n_catalyst': int((frame_df['role'] == 'catalyst').sum()) if not frame_df.empty else 0,
                'n_disruptor': int((frame_df['role'] == 'disruptor').sum()) if not frame_df.empty else 0,
                'ar_order': res['ar_order'],
                'r2_full': round(res['r2_full'], 4),
                'r2_test': round(res['r2_test'], 4) if not np.isnan(res['r2_test']) else None,
                'r2_cv': round(res['r2_cv'], 4) if not np.isnan(res['r2_cv']) else None,
            }

        # Global
        summary['global'] = {
            'n_cluster_frame_pairs': len(df_a),
            'n_cascade_frame_pairs': len(df_b),
            'n_sig_cluster_pairs': int((df_a['p_value_hac'] < ALPHA_SIG).sum()) if not df_a.empty else 0,
            'n_sig_cascade_pairs': int((df_b['p_value_hac'] < ALPHA_SIG).sum()) if not df_b.empty else 0,
            'n_reinforcers_a': int((df_align_a['global_role'] == 'reinforcer').sum()) if not df_align_a.empty else 0,
            'n_challengers_a': int((df_align_a['global_role'] == 'challenger').sum()) if not df_align_a.empty else 0,
            'n_reinforcers_b': int((df_align_b['global_role'] == 'reinforcer').sum()) if not df_align_b.empty else 0,
            'n_challengers_b': int((df_align_b['global_role'] == 'challenger').sum()) if not df_align_b.empty else 0,
        }

        return summary
