"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
semantic_convergence.py

MAIN OBJECTIVE:
---------------
Novel embedding-based semantic convergence metric for cascade detection.
Measures whether media coverage is converging semantically during cascade periods
using pre-computed BAAI/bge-m3 sentence embeddings.

Dependencies:
-------------
- numpy
- scipy
- logging
- cascade_detector.embeddings.embedding_store

MAIN FEATURES:
--------------
1) Intra-window cosine similarity: mean pairwise similarity of article embeddings
2) Temporal convergence trend: is similarity increasing over time?
3) Cross-media semantic alignment: are different outlets converging on similar language?
4) Semantic novelty decay: are new articles adding less new information?

Author:
-------
Antoine Lemor
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SemanticConvergenceCalculator:
    """
    Computes embedding-based semantic convergence metrics for cascade detection.

    This is a novel scientific contribution that measures whether media coverage
    is converging semantically during cascade periods, providing evidence of
    information homogenization in media cascades.
    """

    def __init__(self, embedding_store):
        """
        Initialize with an EmbeddingStore.

        Args:
            embedding_store: EmbeddingStore instance for accessing pre-computed embeddings
        """
        self.store = embedding_store

    def compute_all_metrics(self,
                            article_ids: List[str],
                            article_dates: Optional[Dict[str, datetime]] = None,
                            article_media: Optional[Dict[str, str]] = None,
                            n_sub_windows: int = 5,
                            max_articles: int = 500) -> Dict[str, float]:
        """
        Compute all semantic convergence metrics for a set of articles.

        Args:
            article_ids: List of document IDs
            article_dates: Optional mapping doc_id -> publication date
            article_media: Optional mapping doc_id -> media outlet
            n_sub_windows: Number of sub-windows for temporal analysis
            max_articles: Maximum articles to consider

        Returns:
            Dictionary of semantic convergence metrics
        """
        metrics = {}

        if len(article_ids) < 2:
            return self._empty_metrics()

        # Subsample if needed
        if len(article_ids) > max_articles:
            rng = np.random.RandomState(42)
            article_ids = list(rng.choice(article_ids, max_articles, replace=False))

        # Deduplicate near-identical articles (syndication detection)
        # Keeps one representative per cluster of cosine > 0.95
        deduped_ids = self.store.deduplicate_embeddings(article_ids)
        n_syndicated = len(article_ids) - len(deduped_ids)
        if n_syndicated > 0:
            metrics['n_syndicated_removed'] = n_syndicated
            metrics['syndication_ratio'] = n_syndicated / len(article_ids)
        else:
            metrics['n_syndicated_removed'] = 0
            metrics['syndication_ratio'] = 0.0

        # 1. Intra-window cosine similarity (on deduplicated set)
        intra_sim = self.intra_window_similarity(deduped_ids)
        metrics['intra_window_similarity'] = intra_sim

        # 2. Temporal convergence trend
        if article_dates:
            trend_metrics = self.temporal_convergence_trend(
                deduped_ids, article_dates, n_sub_windows=n_sub_windows
            )
            metrics.update(trend_metrics)
        else:
            metrics['convergence_trend_slope'] = 0.0
            metrics['convergence_trend_r2'] = 0.0
            metrics['convergence_trend_p_value'] = 1.0

        # 3. Cross-media semantic alignment (deduped per outlet)
        if article_media:
            alignment = self.cross_media_alignment(deduped_ids, article_media)
            metrics['cross_media_alignment'] = alignment
        else:
            metrics['cross_media_alignment'] = 0.0

        # 4. Semantic novelty decay (on deduplicated set)
        if article_dates:
            novelty_metrics = self.semantic_novelty_decay(
                deduped_ids, article_dates
            )
            metrics.update(novelty_metrics)
        else:
            metrics['novelty_decay_rate'] = 0.0
            metrics['final_novelty'] = 0.0
            metrics['novelty_half_life'] = 0.0

        return metrics

    def intra_window_similarity(self, article_ids: List[str]) -> float:
        """
        Compute mean pairwise cosine similarity of article embeddings.

        High similarity = articles saying similar things = cascade signal.

        Args:
            article_ids: List of document IDs

        Returns:
            Mean pairwise cosine similarity in [0, 1]
        """
        return self.store.mean_pairwise_similarity(article_ids)

    def temporal_convergence_trend(self,
                                   article_ids: List[str],
                                   article_dates: Dict[str, datetime],
                                   n_sub_windows: int = 5) -> Dict[str, float]:
        """
        Measure whether semantic similarity is increasing over time.

        Splits articles into temporal sub-windows and computes intra-window
        similarity for each. A positive slope = convergence (cascade signal).

        Args:
            article_ids: List of document IDs
            article_dates: Mapping doc_id -> publication date
            n_sub_windows: Number of temporal sub-windows

        Returns:
            Dictionary with trend slope, r2, p_value
        """
        # Sort articles by date
        dated_articles = [
            (doc_id, article_dates[doc_id])
            for doc_id in article_ids
            if doc_id in article_dates
        ]
        if len(dated_articles) < n_sub_windows * 2:
            return {
                'convergence_trend_slope': 0.0,
                'convergence_trend_r2': 0.0,
                'convergence_trend_p_value': 1.0,
            }

        dated_articles.sort(key=lambda x: x[1])

        # Split into sub-windows
        chunk_size = len(dated_articles) // n_sub_windows
        sub_window_sims = []

        for i in range(n_sub_windows):
            start = i * chunk_size
            end = start + chunk_size if i < n_sub_windows - 1 else len(dated_articles)
            sub_ids = [doc_id for doc_id, _ in dated_articles[start:end]]

            if len(sub_ids) >= 2:
                sim = self.store.mean_pairwise_similarity(sub_ids)
                sub_window_sims.append(sim)

        if len(sub_window_sims) < 3:
            return {
                'convergence_trend_slope': 0.0,
                'convergence_trend_r2': 0.0,
                'convergence_trend_p_value': 1.0,
            }

        # Linear regression on sub-window similarities
        from scipy import stats
        x = np.arange(len(sub_window_sims))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, sub_window_sims)

        return {
            'convergence_trend_slope': float(slope),
            'convergence_trend_r2': float(r_value ** 2),
            'convergence_trend_p_value': float(p_value),
        }

    def cross_media_alignment(self,
                               article_ids: List[str],
                               article_media: Dict[str, str]) -> float:
        """
        Measure whether different media outlets are converging on similar language.

        Computes mean cosine similarity between outlet centroids.
        High cross-media similarity = outlets saying similar things = cascade signal.

        Args:
            article_ids: List of document IDs
            article_media: Mapping doc_id -> media outlet

        Returns:
            Mean cross-media cosine similarity in [0, 1]
        """
        # Group articles by media
        media_articles = {}
        for doc_id in article_ids:
            media = article_media.get(doc_id)
            if media:
                if media not in media_articles:
                    media_articles[media] = []
                media_articles[media].append(doc_id)

        # Need at least 2 media outlets
        if len(media_articles) < 2:
            return 0.0

        # Compute centroid for each outlet
        centroids = {}
        for media, ids in media_articles.items():
            if len(ids) < 2:
                continue
            embeddings, found_ids = self.store.get_batch_article_embeddings(ids)
            if len(embeddings) > 0:
                centroids[media] = embeddings.mean(axis=0)

        if len(centroids) < 2:
            return 0.0

        # Pairwise cosine similarity between centroids
        media_list = list(centroids.keys())
        similarities = []

        for i in range(len(media_list)):
            for j in range(i + 1, len(media_list)):
                sim = self.store.cosine_similarity(
                    centroids[media_list[i]],
                    centroids[media_list[j]]
                )
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.0

    def semantic_novelty_decay(self,
                                article_ids: List[str],
                                article_dates: Dict[str, datetime]) -> Dict[str, float]:
        """
        Measure whether new articles add less new information over time.

        Uses incremental centroid distance: as the cascade develops, new articles
        are increasingly similar to the running centroid (lower novelty).

        Args:
            article_ids: List of document IDs
            article_dates: Mapping doc_id -> publication date

        Returns:
            Dictionary with novelty decay metrics
        """
        # Sort by date
        dated_articles = [
            (doc_id, article_dates[doc_id])
            for doc_id in article_ids
            if doc_id in article_dates
        ]
        if len(dated_articles) < 5:
            return {
                'novelty_decay_rate': 0.0,
                'final_novelty': 0.0,
                'novelty_half_life': 0.0,
            }

        dated_articles.sort(key=lambda x: x[1])

        # Compute incremental centroid distances
        running_sum = np.zeros(self.store.embedding_dim, dtype=np.float64)
        n_added = 0
        novelty_scores = []

        for doc_id, _ in dated_articles:
            emb = self.store.get_article_embedding(doc_id)
            if emb is None:
                continue

            emb = emb.astype(np.float64)

            if n_added == 0:
                running_sum = emb.copy()
                n_added = 1
                novelty_scores.append(1.0)  # First article is maximally novel
                continue

            # Current centroid
            centroid = running_sum / n_added

            # Novelty = distance from centroid (1 - cosine_similarity)
            norm_emb = np.linalg.norm(emb)
            norm_cent = np.linalg.norm(centroid)
            if norm_emb > 0 and norm_cent > 0:
                cos_sim = np.dot(emb, centroid) / (norm_emb * norm_cent)
                novelty = 1.0 - cos_sim
            else:
                novelty = 0.0

            novelty_scores.append(float(novelty))

            # Update running sum
            running_sum += emb
            n_added += 1

        if len(novelty_scores) < 3:
            return {
                'novelty_decay_rate': 0.0,
                'final_novelty': 0.0,
                'novelty_half_life': 0.0,
            }

        novelty_array = np.array(novelty_scores)

        # Decay rate: slope of novelty over time
        from scipy import stats
        x = np.arange(len(novelty_array))
        slope, _, _, _, _ = stats.linregress(x, novelty_array)
        decay_rate = float(-slope)  # Positive = novelty decreasing = cascade

        # Final novelty: average of last 20% of articles
        last_chunk = max(1, len(novelty_array) // 5)
        final_novelty = float(np.mean(novelty_array[-last_chunk:]))

        # Half-life: when does novelty drop below 50% of initial?
        initial_novelty = novelty_array[0] if novelty_array[0] > 0 else 1.0
        half_threshold = initial_novelty * 0.5
        half_life_idx = np.where(novelty_array < half_threshold)[0]
        if len(half_life_idx) > 0:
            novelty_half_life = float(half_life_idx[0] / len(novelty_array))
        else:
            novelty_half_life = 1.0  # Never reached half

        return {
            'novelty_decay_rate': decay_rate,
            'final_novelty': final_novelty,
            'novelty_half_life': novelty_half_life,
        }

    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        """Return empty metrics dict."""
        return {
            'intra_window_similarity': 0.0,
            'convergence_trend_slope': 0.0,
            'convergence_trend_r2': 0.0,
            'convergence_trend_p_value': 1.0,
            'cross_media_alignment': 0.0,
            'novelty_decay_rate': 0.0,
            'final_novelty': 0.0,
            'novelty_half_life': 0.0,
        }
