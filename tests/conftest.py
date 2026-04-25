"""
Shared pytest fixtures for cascade_detector tests.

Provides MockEmbeddingStore -- a deterministic, lightweight stand-in for
EmbeddingStore that requires no GPU, embedding files, or external dependencies.
"""

import hashlib
import numpy as np
import pytest
from typing import Dict, List, Tuple, Optional

MOCK_EMBEDDING_DIM = 64


class MockEmbeddingStore:
    """
    Deterministic mock of cascade_detector.embeddings.EmbeddingStore.

    Generates reproducible 64-dimensional embeddings seeded by a hash of the
    doc_id (and sentence_id where applicable).  No files or GPU required.
    """

    def __init__(self, embedding_dim: int = MOCK_EMBEDDING_DIM):
        self.embedding_dim = embedding_dim

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_from_key(key: str) -> int:
        """Derive a deterministic 32-bit seed from an arbitrary string key."""
        return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)

    def _make_embedding(self, key: str) -> np.ndarray:
        """Return a deterministic unit-norm embedding for *key*."""
        rng = np.random.RandomState(self._seed_from_key(key))
        vec = rng.randn(self.embedding_dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # ------------------------------------------------------------------
    # Public API (mirrors EmbeddingStore)
    # ------------------------------------------------------------------

    def get_sentence_embedding(self, doc_id: str, sentence_id: int) -> Optional[np.ndarray]:
        """Return a deterministic embedding for (doc_id, sentence_id)."""
        key = f"{doc_id}::{sentence_id}"
        return self._make_embedding(key)

    def get_article_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Return a deterministic article-level embedding for *doc_id*."""
        return self._make_embedding(doc_id)

    def get_filtered_article_embedding(
        self, doc_id, sentence_ids: List[int]
    ) -> Optional[np.ndarray]:
        """Return a deterministic embedding filtered by sentence IDs.

        Mean-pools per-sentence embeddings for the given sentence_ids,
        mirroring the real EmbeddingStore behaviour.
        """
        if not sentence_ids:
            return None
        embeddings = []
        for sid in sentence_ids:
            embeddings.append(self.get_sentence_embedding(doc_id, sid))
        return np.mean(embeddings, axis=0).astype(np.float32)

    def get_batch_article_embeddings(self, doc_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Return embeddings for all *doc_ids* (always 100 % coverage)."""
        if not doc_ids:
            return np.empty((0, self.embedding_dim), dtype=np.float32), []
        embeddings = np.stack([self.get_article_embedding(d) for d in doc_ids])
        return embeddings, list(doc_ids)

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (n1 * n2))

    def pairwise_cosine_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Pairwise cosine similarity matrix."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = embeddings / norms
        return normalized @ normalized.T

    def mean_pairwise_similarity(self, doc_ids: List[str], max_articles: int = 500) -> float:
        """Mean pairwise cosine similarity for a set of articles."""
        if len(doc_ids) < 2:
            return 0.0
        if len(doc_ids) > max_articles:
            rng = np.random.RandomState(42)
            doc_ids = list(rng.choice(doc_ids, max_articles, replace=False))
        embeddings, _ = self.get_batch_article_embeddings(doc_ids)
        sim = self.pairwise_cosine_similarity(embeddings)
        n = len(embeddings)
        upper = np.triu_indices(n, k=1)
        return float(np.mean(sim[upper]))

    def compute_corpus_baseline(self, n_sample: int = 1000, seed: int = 42) -> float:
        """Return 0.0 so residual similarity equals raw similarity in tests."""
        return 0.0

    def centroid_distance(self, doc_ids: List[str]) -> Tuple[np.ndarray, float]:
        """Centroid and mean distance to centroid."""
        embeddings, _ = self.get_batch_article_embeddings(doc_ids)
        if len(embeddings) == 0:
            return np.zeros(self.embedding_dim), 0.0
        centroid = embeddings.mean(axis=0)
        dists = [1.0 - self.cosine_similarity(e, centroid) for e in embeddings]
        return centroid, float(np.mean(dists))

    def deduplicate_embeddings(self, doc_ids: List[str],
                               threshold: float = 0.95) -> List[str]:
        """Remove near-duplicate articles above *threshold*."""
        embeddings, found_ids = self.get_batch_article_embeddings(doc_ids)
        if len(embeddings) < 2:
            return found_ids
        sim = self.pairwise_cosine_similarity(embeddings)
        keep = [True] * len(found_ids)
        for i in range(len(found_ids)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(found_ids)):
                if keep[j] and sim[i, j] > threshold:
                    keep[j] = False
        return [fid for fid, k in zip(found_ids, keep) if k]


class ClusterableMockEmbeddingStore(MockEmbeddingStore):
    """Mock embedding store that produces clusterable embeddings.

    Articles with doc_id starting with ``cluster_N_`` (e.g. ``cluster_0_art1``)
    get embeddings near centroid N plus small noise, making clustering
    deterministic and testable.

    Centroids are placed at orthogonal directions in embedding space.
    """

    def __init__(self, embedding_dim: int = MOCK_EMBEDDING_DIM,
                 noise_scale: float = 0.05, n_centroids: int = 8):
        super().__init__(embedding_dim)
        self.noise_scale = noise_scale
        # Generate deterministic orthogonal-ish centroids
        rng = np.random.RandomState(12345)
        raw = rng.randn(n_centroids, embedding_dim).astype(np.float32)
        # Make each centroid unit-norm
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        self._centroids = raw / np.maximum(norms, 1e-10)

    def _clusterable_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Return embedding near centroid N if doc_id matches cluster_N_*."""
        if doc_id.startswith('cluster_'):
            parts = doc_id.split('_')
            if len(parts) >= 3:
                try:
                    cluster_idx = int(parts[1])
                except ValueError:
                    return None
                if cluster_idx < len(self._centroids):
                    rng = np.random.RandomState(self._seed_from_key(doc_id))
                    noise = rng.randn(self.embedding_dim).astype(np.float32)
                    vec = self._centroids[cluster_idx] + self.noise_scale * noise
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec /= norm
                    return vec
        return None

    def get_article_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Return clusterable embedding if doc_id matches pattern."""
        emb = self._clusterable_embedding(doc_id)
        if emb is not None:
            return emb
        return super().get_article_embedding(doc_id)

    def get_sentence_embedding(self, doc_id: str, sentence_id: int) -> Optional[np.ndarray]:
        """Return clusterable sentence embedding for cluster_N_* doc_ids.

        Each sentence gets a small per-sentence noise offset from the
        cluster centroid, ensuring belonging score computation works
        with clusterable phrase-level embeddings.
        """
        emb = self._clusterable_embedding(doc_id)
        if emb is not None:
            seed_str = f"{doc_id}::sent::{sentence_id}"
            rng = np.random.RandomState(self._seed_from_key(seed_str))
            noise = rng.randn(self.embedding_dim).astype(np.float32)
            vec = emb + 0.03 * noise
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            return vec
        return super().get_sentence_embedding(doc_id, sentence_id)

    def get_filtered_article_embedding(
        self, doc_id, sentence_ids: List[int]
    ) -> Optional[np.ndarray]:
        """Return clusterable filtered embedding for cluster_N_* doc_ids.

        Uses the same centroid + noise strategy as get_article_embedding
        (with a different seed incorporating sentence_ids) so that
        event-filtered embeddings still cluster deterministically.
        """
        emb = self._clusterable_embedding(doc_id)
        if emb is not None:
            # Add small extra noise seeded by sentence_ids for variation
            seed_str = f"{doc_id}::filtered::{'_'.join(str(s) for s in sorted(sentence_ids))}"
            rng = np.random.RandomState(self._seed_from_key(seed_str))
            noise = rng.randn(self.embedding_dim).astype(np.float32)
            vec = emb + 0.02 * noise
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            return vec
        return super().get_filtered_article_embedding(doc_id, sentence_ids)


# -----------------------------------------------------------------------
# Pytest fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def mock_embedding_store():
    """Provide a MockEmbeddingStore instance to any test."""
    return MockEmbeddingStore()


@pytest.fixture
def clusterable_embedding_store():
    """Provide a ClusterableMockEmbeddingStore instance to any test."""
    return ClusterableMockEmbeddingStore()
