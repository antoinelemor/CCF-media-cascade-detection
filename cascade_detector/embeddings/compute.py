"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
compute.py

MAIN OBJECTIVE:
---------------
Compute BAAI/bge-m3 sentence embeddings from the PostgreSQL database and save
as memory-mapped .npy + index.pkl for use by EmbeddingStore.

Supports resumable checkpoints, device-adaptive batch sizing (MPS/CUDA/CPU),
and periodic MPS memory cleanup to prevent fragmentation.

Output:
    - embeddings.npy : float16 memmap array (n_sentences x 1024)
    - index.pkl : dict mapping (doc_id, sentence_id) -> row_index

Author:
-------
Antoine Lemor
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import psycopg2
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default batch sizes per device (MPS cannot handle large batches for bge-m3)
DEVICE_BATCH_SIZES = {
    'cuda': 256,
    'mps': 32,
    'cpu': 64,
}

# How often to save checkpoint (number of sentences)
CHECKPOINT_INTERVAL = 100_000


def _get_connection():
    """Connect to PostgreSQL using env vars or defaults."""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', '5432')),
        dbname=os.getenv('DB_NAME', 'CCF_Database_texts'),
        user=os.getenv('DB_USER', 'antoine'),
        password=os.getenv('DB_PASSWORD', ''),
    )


def count_sentences(conn=None, start_date: str = None,
                    end_date: str = None) -> int:
    """Count total sentences + titles in the database, optionally filtered by date range.

    Counts non-null sentences, plus one title per distinct doc_id that has a
    non-empty title. This matches _fetch_sentences_batched which emits titles
    as sentence_id=0.

    Args:
        conn: Optional psycopg2 connection. If None, creates one.
        start_date: Optional lower bound (YYYY-MM-DD).
        end_date: Optional upper bound (YYYY-MM-DD).

    Returns:
        Total number of rows to embed (sentences + titles).
    """
    close_conn = False
    if conn is None:
        conn = _get_connection()
        close_conn = True

    # Count sentences
    query_sent = ('SELECT COUNT(*) FROM "CCF_processed_data" '
                  'WHERE sentences IS NOT NULL AND sentences != \'\'')
    # Count distinct doc_ids with non-empty titles (one title embedding each)
    query_titles = (
        'SELECT COUNT(DISTINCT doc_id) FROM "CCF_processed_data" '
        'WHERE sentences IS NOT NULL AND sentences != \'\' '
        'AND title IS NOT NULL AND title != \'\''
    )
    params = {}
    if start_date is not None:
        query_sent += " AND date >= %(start_date)s::date"
        query_titles += " AND date >= %(start_date)s::date"
        params['start_date'] = start_date
    if end_date is not None:
        query_sent += " AND date <= %(end_date)s::date"
        query_titles += " AND date <= %(end_date)s::date"
        params['end_date'] = end_date

    with conn.cursor() as cur:
        cur.execute(query_sent, params)
        n_sentences = cur.fetchone()[0]
        cur.execute(query_titles, params)
        n_titles = cur.fetchone()[0]

    if close_conn:
        conn.close()

    return n_sentences + n_titles


def _fetch_sentences_batched(conn, batch_size=50000,
                             start_date=None, end_date=None):
    """Yield batches of (doc_id, sentence_id, sentence_text) from the database.

    Titles are emitted as sentence_id=0 for each doc_id before its sentences.
    This ensures a full run natively produces title embeddings.
    """
    query = """
        SELECT doc_id, sentence_id, sentences, title
        FROM "CCF_processed_data"
        WHERE sentences IS NOT NULL AND sentences != ''
    """
    params = {}
    if start_date is not None:
        query += " AND date >= %(start_date)s::date"
        params['start_date'] = start_date
    if end_date is not None:
        query += " AND date <= %(end_date)s::date"
        params['end_date'] = end_date
    query += " ORDER BY doc_id, sentence_id"

    with conn.cursor(name='embedding_cursor') as cur:
        cur.itersize = batch_size
        cur.execute(query, params)

        batch = []
        last_doc_id = None
        for row in cur:
            doc_id, sentence_id, sentence_text, title = row

            # Emit title as sentence_id=0 on first encounter of each doc_id
            if doc_id != last_doc_id:
                last_doc_id = doc_id
                if title and str(title).strip():
                    batch.append((doc_id, 0, str(title).strip()))
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

            batch.append((doc_id, sentence_id, sentence_text))
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


def _detect_device() -> str:
    """Detect best available compute device."""
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _save_checkpoint(index: dict, index_path: str, row_idx: int) -> None:
    """Save index checkpoint atomically."""
    checkpoint_path = str(index_path) + '.checkpoint'
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'index': index, 'row_idx': row_idx}, f,
                     protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(checkpoint_path, str(index_path) + '.resume')


def _load_checkpoint(index_path: str):
    """Load checkpoint if it exists. Returns (index, row_idx) or (None, 0)."""
    resume_path = str(index_path) + '.resume'
    if os.path.exists(resume_path):
        try:
            with open(resume_path, 'rb') as f:
                data = pickle.load(f)
            return data['index'], data['row_idx']
        except Exception as e:
            logger.warning(f"Cannot load checkpoint ({e}). Starting fresh.")
    return None, 0


def ensure_embeddings(embedding_dir: str,
                      skip: bool = False) -> None:
    """Ensure embeddings cover the full database. Compute if missing or incomplete.

    Checks:
    1. Do embeddings.npy and index.pkl exist?
    2. Does the index cover >= 99% of DB sentences?

    If not, runs the full embedding computation pipeline.

    Args:
        embedding_dir: Directory for embeddings.npy and index.pkl.
        skip: If True, skip check entirely (assume already computed).
    """
    if skip:
        logger.info("Skipping embedding check (skip=True)")
        return

    emb_path = Path(embedding_dir)
    embeddings_file = emb_path / 'embeddings.npy'
    index_file = emb_path / 'index.pkl'

    # Step 1: Check files exist
    if not embeddings_file.exists() or not index_file.exists():
        logger.info("Embeddings not found. Computing from scratch...")
        compute_embeddings(output_dir=embedding_dir, resume=True)
        return

    # Step 2: Check coverage
    logger.info("Checking embedding coverage...")
    try:
        with open(index_file, 'rb') as f:
            index = pickle.load(f)
        n_embedded = len(index)
    except Exception as e:
        logger.warning(f"Cannot read index.pkl ({e}). Recomputing embeddings...")
        compute_embeddings(output_dir=embedding_dir, resume=True)
        return

    n_db = count_sentences()
    coverage = n_embedded / n_db if n_db > 0 else 0.0
    logger.info(
        f"Embedding coverage: {n_embedded:,} / {n_db:,} sentences ({coverage:.1%})"
    )

    if coverage >= 0.99:
        logger.info("Embeddings are complete. Skipping computation.")
        return

    logger.info(
        f"Embeddings incomplete ({coverage:.1%}). "
        f"Recomputing for full database..."
    )
    compute_embeddings(output_dir=embedding_dir, resume=True)


def compute_embeddings(output_dir: str = 'data/embeddings',
                       batch_size: Optional[int] = None,
                       db_batch_size: int = 50000,
                       dtype: str = 'float16',
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       resume: bool = False) -> None:
    """Compute BAAI/bge-m3 sentence embeddings for the full database.

    Args:
        output_dir: Directory for embeddings.npy and index.pkl.
        batch_size: Encoding batch size. If None, auto-detected per device
                    (cuda=256, mps=32, cpu=64).
        db_batch_size: Database fetch batch size.
        dtype: 'float16' or 'float32'.
        start_date: Optional start date filter (YYYY-MM-DD).
        end_date: Optional end date filter (YYYY-MM-DD).
        resume: If True, resume from last checkpoint.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers torch"
        )
        raise

    import torch

    # Detect device
    device = _detect_device()
    logger.info(f"Using device: {device}")

    # Auto-select batch size if not specified
    if batch_size is None:
        batch_size = DEVICE_BATCH_SIZES.get(device, 64)
    logger.info(f"Encoding batch size: {batch_size}")

    # Load model
    logger.info("Loading BAAI/bge-m3 model...")
    model = SentenceTransformer('BAAI/bge-m3', device=device)

    # Cap max_seq_length: bge-m3 defaults to 8192 tokens, but our data is
    # sentence-level (typically <100 tokens). SDPA attention allocates
    # O(batch × heads × seq_len²) — at 8192 tokens this can reach >100 GiB
    # on a single batch, crashing MPS. 512 tokens covers >99.9% of sentences
    # with no quality loss and caps SDPA memory at ~0.8 GiB per batch.
    model.max_seq_length = 512

    embedding_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Model loaded. Embedding dimension: {embedding_dim}, "
                f"max_seq_length: {model.max_seq_length}")

    # Connect to database
    conn = _get_connection()
    total_sentences = count_sentences(conn, start_date=start_date, end_date=end_date)
    if start_date or end_date:
        logger.info(f"Date filter: {start_date or 'beginning'} to {end_date or 'end'}")
    logger.info(f"Total sentences to embed: {total_sentences:,}")

    if total_sentences == 0:
        logger.warning("No sentences found. Exiting.")
        conn.close()
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    embeddings_path = os.path.join(output_dir, 'embeddings.npy')
    index_path = os.path.join(output_dir, 'index.pkl')

    np_dtype = np.float16 if dtype == 'float16' else np.float32

    # Check for resume
    index = {}
    row_idx = 0
    skip_count = 0

    if resume:
        loaded_index, loaded_row_idx = _load_checkpoint(index_path)
        if loaded_index is not None:
            index = loaded_index
            row_idx = loaded_row_idx
            skip_count = row_idx
            logger.info(f"Resuming from checkpoint: {row_idx:,} sentences already done")

    # Create or open memory-mapped file
    if row_idx == 0:
        embeddings = np.lib.format.open_memmap(
            embeddings_path, dtype=np_dtype, mode='w+',
            shape=(total_sentences, embedding_dim),
        )
    else:
        embeddings = np.lib.format.open_memmap(
            embeddings_path, dtype=np_dtype, mode='r+',
        )

    logger.info(f"Memmap: {embeddings_path} "
                f"({total_sentences * embedding_dim * np.dtype(np_dtype).itemsize / 1e9:.1f} GB)")

    # Main progress bar
    pbar = tqdm(
        total=total_sentences,
        initial=row_idx,
        desc="Encoding",
        unit="sent",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        dynamic_ncols=True,
    )

    last_checkpoint = row_idx
    sentences_since_cache_clear = 0

    for db_batch in _fetch_sentences_batched(conn, batch_size=db_batch_size,
                                             start_date=start_date,
                                             end_date=end_date):
        doc_ids = [row[0] for row in db_batch]
        sentence_ids = [row[1] for row in db_batch]
        texts = [str(row[2]) for row in db_batch]

        # Skip already-processed sentences when resuming
        if skip_count > 0:
            if skip_count >= len(texts):
                skip_count -= len(texts)
                continue
            else:
                doc_ids = doc_ids[skip_count:]
                sentence_ids = sentence_ids[skip_count:]
                texts = texts[skip_count:]
                skip_count = 0

        # Encode in sub-batches
        for i in range(0, len(texts), batch_size):
            sub_texts = texts[i:i + batch_size]
            sub_doc_ids = doc_ids[i:i + batch_size]
            sub_sentence_ids = sentence_ids[i:i + batch_size]

            embs = model.encode(
                sub_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            n = len(sub_texts)
            embeddings[row_idx:row_idx + n] = embs.astype(np_dtype)

            for j in range(n):
                index[(sub_doc_ids[j], sub_sentence_ids[j])] = row_idx + j

            row_idx += n
            pbar.update(n)
            sentences_since_cache_clear += n

            # Periodic MPS memory cleanup to prevent fragmentation
            if device == 'mps' and sentences_since_cache_clear >= 10_000:
                torch.mps.empty_cache()
                sentences_since_cache_clear = 0

            # Periodic checkpoint
            if row_idx - last_checkpoint >= CHECKPOINT_INTERVAL:
                embeddings.flush()
                _save_checkpoint(index, index_path, row_idx)
                last_checkpoint = row_idx

    pbar.close()

    # Flush memmap
    embeddings.flush()
    del embeddings

    # Save final index
    logger.info("Saving index...")
    with open(index_path, 'wb') as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Remove checkpoint file
    resume_file = str(index_path) + '.resume'
    if os.path.exists(resume_file):
        os.remove(resume_file)

    conn.close()

    file_size_gb = os.path.getsize(embeddings_path) / (1024 ** 3)
    index_size_mb = os.path.getsize(index_path) / (1024 ** 2)

    logger.info(f"Done! Processed {row_idx:,} sentences")
    logger.info(f"Embeddings: {embeddings_path} ({file_size_gb:.2f} GB)")
    logger.info(f"Index: {index_path} ({index_size_mb:.1f} MB, {len(index):,} entries)")
