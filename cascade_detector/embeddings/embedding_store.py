"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
embedding_store.py

MAIN OBJECTIVE:
---------------
Memory-mapped embedding store for efficient access to pre-computed sentence embeddings.
Supports batch retrieval, article-level mean pooling, and cosine similarity computation.

Dependencies:
-------------
- numpy
- pickle
- pathlib
- logging

MAIN FEATURES:
--------------
1) Memory-mapped numpy array for 9.2M x 1024 embeddings (~37.5 GB)
2) Index mapping (doc_id, sentence_id) -> row index
3) Article-level mean pooling of sentence embeddings
4) Efficient batch cosine similarity computation
5) Lazy loading - only reads data when accessed

Author:
-------
Antoine Lemor
"""

import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """
    Memory-mapped embedding store for pre-computed sentence embeddings.

    Uses numpy memmap for efficient access without loading entire array into RAM.
    Supports ~9.2M sentences x 1024 dimensions (~37.5 GB on disk).
    """

    def __init__(self,
                 embedding_dir: str,
                 embedding_dim: int = 1024,
                 dtype: str = 'float32'):
        """
        Initialize embedding store.

        Args:
            embedding_dir: Directory containing embeddings.npy and index.pkl
            embedding_dim: Embedding dimension (1024 for bge-m3)
            dtype: Data type for embeddings
        """
        self.embedding_dir = Path(embedding_dir)
        self.embedding_dim = embedding_dim
        self.dtype = dtype

        # File paths
        self.embeddings_path = self.embedding_dir / "embeddings.npy"
        self.index_path = self.embedding_dir / "index.pkl"

        # Lazy-loaded resources
        self._embeddings = None  # memmap array
        self._index = None  # (doc_id, sentence_id) -> row_index
        self._doc_id_to_rows = None  # doc_id -> [row_indices]

        # Verify files exist
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        # Eager-load index and doc_id mapping to avoid thread-safety issues
        # with lazy loading (multiple threads may trigger concurrent loads)
        _ = self.index
        _ = self.doc_id_to_rows
        _ = self.embeddings

        logger.info(f"EmbeddingStore initialized: {self.embedding_dir}")

    @property
    def embeddings(self) -> np.ndarray:
        """Lazy-load memory-mapped embeddings."""
        if self._embeddings is None:
            try:
                self._embeddings = np.load(
                    str(self.embeddings_path),
                    mmap_mode='r'
                )
            except ValueError:
                # Raw memmap without .npy header (created by np.memmap mode='w+')
                n_rows = len(self.index)
                self._embeddings = np.memmap(
                    str(self.embeddings_path),
                    dtype=np.float16, mode='r',
                    shape=(n_rows, 1024),
                )
                logger.info("Loaded raw memmap (no .npy header)")
            logger.info(
                f"Loaded embeddings memmap: {self._embeddings.shape} "
                f"({self._embeddings.nbytes / 1e9:.1f} GB on disk)"
            )
        return self._embeddings

    @property
    def index(self) -> Dict[Tuple, int]:
        """Lazy-load index mapping."""
        if self._index is None:
            with open(self.index_path, 'rb') as f:
                raw = pickle.load(f)
            # Handle wrapped format: {'index': {(doc_id, sent_id): row, ...}}
            if isinstance(raw, dict) and 'index' in raw and isinstance(raw['index'], dict):
                self._index = raw['index']
            else:
                self._index = raw
            logger.info(f"Loaded index: {len(self._index):,} entries")
            # Log key types for debugging
            if self._index:
                sample = next(iter(self._index.keys()))
                logger.info(f"Index key sample: {sample!r} (types: {type(sample[0]).__name__}, {type(sample[1]).__name__})")
        return self._index

    @property
    def doc_id_to_rows(self) -> Dict:
        """Build doc_id -> row indices mapping (lazy)."""
        if self._doc_id_to_rows is None:
            self._doc_id_to_rows = {}
            for (doc_id, sent_id), row_idx in self.index.items():
                if doc_id not in self._doc_id_to_rows:
                    self._doc_id_to_rows[doc_id] = []
                self._doc_id_to_rows[doc_id].append(row_idx)
            logger.info(f"Built doc_id index: {len(self._doc_id_to_rows):,} articles")
        return self._doc_id_to_rows

    def __len__(self) -> int:
        """Number of embeddings stored."""
        return len(self.index)

    def get_sentence_embedding(self, doc_id: str, sentence_id: int) -> Optional[np.ndarray]:
        """
        Get embedding for a specific sentence.

        Args:
            doc_id: Document ID
            sentence_id: Sentence ID

        Returns:
            1D numpy array of shape (embedding_dim,) or None if not found
        """
        key = (doc_id, sentence_id)
        row_idx = self.index.get(key)
        if row_idx is None:
            return None
        return np.array(self.embeddings[row_idx], dtype=np.float32)

    def get_article_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """
        Get article-level embedding by mean pooling sentence embeddings.

        Args:
            doc_id: Document ID

        Returns:
            Mean-pooled embedding or None if no sentences found
        """
        row_indices = self.doc_id_to_rows.get(doc_id)
        if not row_indices:
            return None

        # Gather sentence embeddings and mean pool
        sentence_embeddings = np.array(self.embeddings[row_indices], dtype=np.float32)
        return sentence_embeddings.mean(axis=0)

    def get_batch_article_embeddings(self, doc_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Get embeddings for multiple articles.

        Optimized for large batches: sorts by memmap row for sequential I/O,
        pre-allocates output array, and uses parallel mean-pooling.

        Args:
            doc_ids: List of document IDs

        Returns:
            Tuple of (embeddings array [n_found, dim], list of found doc_ids)
        """
        # Filter to doc_ids that exist in the index
        valid = []
        for doc_id in doc_ids:
            rows = self.doc_id_to_rows.get(doc_id)
            if rows:
                valid.append((doc_id, rows))

        if not valid:
            if len(doc_ids) >= 5:
                logger.warning(f"Embedding coverage: 0/{len(doc_ids)} doc_ids found in index")
            return np.empty((0, self.embedding_dim), dtype=np.float32), []

        # Sort by minimum row index for sequential memmap access
        valid.sort(key=lambda x: min(x[1]))

        # Pre-allocate output
        n = len(valid)
        result = np.empty((n, self.embedding_dim), dtype=np.float32)
        found_ids = []

        # Batch mean-pool: read memmap rows and compute means
        # For large batches, gather all unique rows first for sequential read
        if n > 500:
            # Collect all row indices, read once, then scatter
            all_rows = []
            row_ranges = []  # (start_idx_in_all_rows, count) per doc
            for doc_id, rows in valid:
                start = len(all_rows)
                all_rows.extend(rows)
                row_ranges.append((start, len(rows)))
                found_ids.append(doc_id)

            # Sort all_rows for sequential memmap access
            sorted_indices = sorted(range(len(all_rows)), key=lambda i: all_rows[i])
            sorted_rows = [all_rows[i] for i in sorted_indices]

            # Build reverse mapping: original position → position in sorted read
            reverse_map = [0] * len(all_rows)
            for new_pos, orig_pos in enumerate(sorted_indices):
                reverse_map[orig_pos] = new_pos

            # Single sorted read from memmap
            all_embs = np.array(
                self.embeddings[sorted_rows], dtype=np.float32
            )

            # Mean-pool per document using the reverse mapping
            for i, (start, count) in enumerate(row_ranges):
                mapped_indices = [reverse_map[start + k] for k in range(count)]
                result[i] = all_embs[mapped_indices].mean(axis=0)
        else:
            for i, (doc_id, rows) in enumerate(valid):
                embs = np.array(self.embeddings[rows], dtype=np.float32)
                result[i] = embs.mean(axis=0)
                found_ids.append(doc_id)

        coverage = n / len(doc_ids)
        if coverage < 0.5 and len(doc_ids) >= 5:
            logger.warning(
                f"Embedding coverage: {n}/{len(doc_ids)} "
                f"({coverage:.0%}) doc_ids found in index"
            )

        return result, found_ids

    def get_filtered_article_embedding(
        self, doc_id, sentence_ids: List[int]
    ) -> Optional[np.ndarray]:
        """
        Get article embedding by mean pooling only specified sentences.

        Useful for event-filtered embeddings: pass only sentence IDs where
        a specific event type is active, to get an event-specific article
        representation instead of a generic mean-pool of all sentences.

        Args:
            doc_id: Document ID (int or str)
            sentence_ids: List of sentence IDs to include

        Returns:
            Mean-pooled embedding of specified sentences, or None if none found
        """
        if not sentence_ids:
            return None

        embeddings = []
        for sid in sentence_ids:
            key = (doc_id, int(sid))
            row_idx = self.index.get(key)
            if row_idx is not None:
                embeddings.append(
                    np.array(self.embeddings[row_idx], dtype=np.float32)
                )

        if not embeddings:
            return None

        return np.mean(embeddings, axis=0)

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Cosine similarity in [-1, 1]
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def pairwise_cosine_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.

        Args:
            embeddings: Array of shape (n, dim)

        Returns:
            Similarity matrix of shape (n, n)
        """
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = embeddings / norms

        # Compute similarity matrix
        return normalized @ normalized.T

    def deduplicate_embeddings(self, doc_ids: List[str],
                               threshold: float = 0.95) -> List[str]:
        """
        Remove near-duplicate articles (cosine similarity > threshold).

        Syndicated articles (same text republished in multiple outlets) have
        cosine similarity ~1.0 and inflate convergence metrics. This method
        keeps one representative per near-duplicate cluster.

        Args:
            doc_ids: List of document IDs
            threshold: Cosine similarity above which articles are considered duplicates

        Returns:
            Deduplicated list of doc_ids
        """
        embeddings, found_ids = self.get_batch_article_embeddings(doc_ids)
        if len(embeddings) < 2:
            return found_ids

        sim_matrix = self.pairwise_cosine_similarity(embeddings)
        keep = [True] * len(found_ids)
        for i in range(len(found_ids)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(found_ids)):
                if keep[j] and sim_matrix[i, j] > threshold:
                    keep[j] = False

        deduped = [fid for fid, k in zip(found_ids, keep) if k]
        n_removed = len(found_ids) - len(deduped)
        if n_removed > 0:
            logger.info(
                f"Deduplication: removed {n_removed}/{len(found_ids)} "
                f"near-duplicate articles (threshold={threshold})"
            )
        return deduped

    def mean_pairwise_similarity(self, doc_ids: List[str], max_articles: int = 500) -> float:
        """
        Compute mean pairwise cosine similarity for a set of articles.

        Args:
            doc_ids: List of document IDs
            max_articles: Maximum articles to consider (for performance)

        Returns:
            Mean pairwise cosine similarity in [0, 1]
        """
        if len(doc_ids) < 2:
            return 0.0

        # Subsample if too many articles
        if len(doc_ids) > max_articles:
            rng = np.random.RandomState(42)
            doc_ids = list(rng.choice(doc_ids, max_articles, replace=False))

        embeddings, _ = self.get_batch_article_embeddings(doc_ids)
        if len(embeddings) < 2:
            return 0.0

        sim_matrix = self.pairwise_cosine_similarity(embeddings)

        # Extract upper triangle (excluding diagonal)
        n = len(embeddings)
        upper_indices = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[upper_indices]

        return float(np.mean(pairwise_sims))

    def compute_corpus_baseline(self, n_sample: int = 1000, seed: int = 42) -> float:
        """Compute mean pairwise cosine similarity on a random sample.

        Cached on first call. Returns the corpus-level semantic baseline.
        """
        if hasattr(self, '_corpus_baseline'):
            return self._corpus_baseline

        all_doc_ids = list(self.doc_id_to_rows.keys())
        rng = np.random.RandomState(seed)
        sample_ids = list(rng.choice(
            all_doc_ids, min(n_sample, len(all_doc_ids)), replace=False
        ))
        self._corpus_baseline = self.mean_pairwise_similarity(
            sample_ids, max_articles=n_sample
        )
        logger.info(f"Corpus semantic baseline: {self._corpus_baseline:.4f}")
        return self._corpus_baseline

    def centroid_distance(self, doc_ids: List[str]) -> Tuple[np.ndarray, float]:
        """
        Compute centroid and average distance to centroid.

        Args:
            doc_ids: List of document IDs

        Returns:
            Tuple of (centroid embedding, mean distance to centroid)
        """
        embeddings, _ = self.get_batch_article_embeddings(doc_ids)
        if len(embeddings) == 0:
            return np.zeros(self.embedding_dim), 0.0

        centroid = embeddings.mean(axis=0)

        # Compute distances to centroid
        distances = []
        for emb in embeddings:
            distances.append(1.0 - self.cosine_similarity(emb, centroid))

        return centroid, float(np.mean(distances))
