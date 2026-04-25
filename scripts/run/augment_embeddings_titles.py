"""
Augment existing embedding store with title embeddings (sentence_id=0).

Queries all distinct (doc_id, title) pairs from the database, skips doc_ids
that already have a (doc_id, 0) entry in the index, encodes titles with
bge-m3, expands the memmap, and updates the index.

Idempotent: safe to re-run; already-processed titles are skipped.

Usage:
    python scripts/run/augment_embeddings_titles.py --embedding-dir data/embeddings/
    python scripts/run/augment_embeddings_titles.py --embedding-dir data/embeddings-test/

After augmenting the full embeddings, regenerate the test index:
    python scripts/run/augment_embeddings_titles.py --embedding-dir data/embeddings/ --regenerate-test-index data/embeddings-test/
"""

import argparse
import logging
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import psycopg2
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# Batch sizes per device
DEVICE_BATCH_SIZES = {'cuda': 256, 'mps': 32, 'cpu': 64}


def _get_connection():
    """Connect to PostgreSQL using env vars or defaults."""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', '5432')),
        dbname=os.getenv('DB_NAME', 'CCF_Database_texts'),
        user=os.getenv('DB_USER', 'antoine'),
        password=os.getenv('DB_PASSWORD', ''),
    )


def _detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def fetch_titles(conn) -> list:
    """Fetch all (doc_id, title) pairs with non-empty titles."""
    query = """
        SELECT DISTINCT doc_id, title
        FROM "CCF_processed_data"
        WHERE title IS NOT NULL AND title != ''
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    logger.info(f"Fetched {len(rows):,} (doc_id, title) pairs from database")
    return rows


def augment(embedding_dir: str, batch_size: int = None) -> int:
    """Add title embeddings to existing embedding store.

    Returns:
        Number of new title embeddings added.
    """
    from sentence_transformers import SentenceTransformer

    emb_dir = Path(embedding_dir)
    embeddings_path = emb_dir / 'embeddings.npy'
    index_path = emb_dir / 'index.pkl'

    if not embeddings_path.exists() or not index_path.exists():
        logger.error(f"Embeddings not found in {emb_dir}. Run compute.py first.")
        sys.exit(1)

    # Load existing index
    with open(index_path, 'rb') as f:
        raw = pickle.load(f)
    if isinstance(raw, dict) and 'index' in raw and isinstance(raw['index'], dict):
        index = raw['index']
    else:
        index = raw
    logger.info(f"Loaded index: {len(index):,} entries")

    # Load existing memmap (read shape)
    existing = np.load(str(embeddings_path), mmap_mode='r')
    old_rows, embedding_dim = existing.shape
    np_dtype = existing.dtype
    logger.info(f"Existing memmap: {old_rows:,} x {embedding_dim}, dtype={np_dtype}")
    del existing

    # Fetch titles from database
    conn = _get_connection()
    title_rows = fetch_titles(conn)
    conn.close()

    # Filter out doc_ids that already have sentence_id=0 in index
    new_titles = []
    for doc_id, title in title_rows:
        if (doc_id, 0) not in index:
            new_titles.append((doc_id, title))

    logger.info(
        f"Titles to add: {len(new_titles):,} "
        f"(skipped {len(title_rows) - len(new_titles):,} already present)"
    )

    if not new_titles:
        logger.info("Nothing to do. All titles already in index.")
        return 0

    # Detect device and load model
    device = _detect_device()
    if batch_size is None:
        batch_size = DEVICE_BATCH_SIZES.get(device, 64)
    logger.info(f"Device: {device}, batch_size: {batch_size}")

    logger.info("Loading BAAI/bge-m3 model...")
    model = SentenceTransformer('BAAI/bge-m3', device=device)
    model.max_seq_length = 512

    n_new = len(new_titles)

    # Create expanded memmap
    new_total = old_rows + n_new
    temp_path = str(embeddings_path) + '.augment_tmp'
    logger.info(f"Creating expanded memmap: {new_total:,} x {embedding_dim}")

    # Copy existing data to new file
    shutil.copy2(str(embeddings_path), temp_path)

    # Reopen as writable with new shape using np.lib.format
    new_memmap = np.lib.format.open_memmap(
        temp_path, dtype=np_dtype, mode='r+',
        shape=(new_total, embedding_dim),
    )

    # If the copy didn't extend, we need to create fresh and copy
    if new_memmap.shape[0] != new_total:
        del new_memmap
        os.remove(temp_path)
        new_memmap = np.lib.format.open_memmap(
            temp_path, dtype=np_dtype, mode='w+',
            shape=(new_total, embedding_dim),
        )
        # Copy old data in chunks
        old_memmap = np.load(str(embeddings_path), mmap_mode='r')
        chunk_size = 100_000
        for i in range(0, old_rows, chunk_size):
            end = min(i + chunk_size, old_rows)
            new_memmap[i:end] = old_memmap[i:end]
        del old_memmap
        new_memmap.flush()
        logger.info("Copied existing embeddings to expanded memmap")

    # Encode titles in batches
    row_idx = old_rows
    pbar = tqdm(total=n_new, desc="Encoding titles", unit="title")

    import torch

    sentences_since_clear = 0
    for i in range(0, n_new, batch_size):
        batch = new_titles[i:i + batch_size]
        texts = [str(title).strip() for _, title in batch]

        embs = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        n = len(texts)
        new_memmap[row_idx:row_idx + n] = embs.astype(np_dtype)

        for j, (doc_id, _) in enumerate(batch):
            index[(doc_id, 0)] = row_idx + j

        row_idx += n
        pbar.update(n)
        sentences_since_clear += n

        if device == 'mps' and sentences_since_clear >= 5000:
            torch.mps.empty_cache()
            sentences_since_clear = 0

    pbar.close()
    new_memmap.flush()
    del new_memmap

    # Atomic replace: move new file over old
    os.replace(temp_path, str(embeddings_path))

    # Save updated index
    with open(index_path, 'wb') as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Done! Added {n_new:,} title embeddings")
    logger.info(f"New index size: {len(index):,} entries")
    logger.info(f"New memmap: {new_total:,} x {embedding_dim}")

    return n_new


def regenerate_test_index(full_embedding_dir: str, test_embedding_dir: str):
    """Regenerate test index by filtering full index to 2018 doc_ids.

    The test embeddings share the same memmap via symlink; only the index
    needs to be regenerated to include title entries (sentence_id=0).
    """
    full_index_path = Path(full_embedding_dir) / 'index.pkl'
    test_index_path = Path(test_embedding_dir) / 'index.pkl'

    with open(full_index_path, 'rb') as f:
        raw = pickle.load(f)
    if isinstance(raw, dict) and 'index' in raw and isinstance(raw['index'], dict):
        full_index = raw['index']
    else:
        full_index = raw

    # Load existing test index to get the set of 2018 doc_ids
    if test_index_path.exists():
        with open(test_index_path, 'rb') as f:
            raw_test = pickle.load(f)
        if isinstance(raw_test, dict) and 'index' in raw_test and isinstance(raw_test['index'], dict):
            old_test = raw_test['index']
        else:
            old_test = raw_test
        test_doc_ids = set(k[0] for k in old_test.keys())
    else:
        logger.error(f"Test index not found: {test_index_path}")
        return

    # Filter full index to test doc_ids
    new_test_index = {
        k: v for k, v in full_index.items()
        if k[0] in test_doc_ids
    }

    n_titles = sum(1 for k in new_test_index if k[1] == 0)
    logger.info(
        f"Regenerated test index: {len(new_test_index):,} entries "
        f"(including {n_titles:,} titles) for {len(test_doc_ids):,} doc_ids"
    )

    with open(test_index_path, 'wb') as f:
        pickle.dump(new_test_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Saved: {test_index_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Augment embedding store with title embeddings'
    )
    parser.add_argument(
        '--embedding-dir', required=True,
        help='Directory containing embeddings.npy and index.pkl'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Encoding batch size (auto-detected if not set)'
    )
    parser.add_argument(
        '--regenerate-test-index', default=None, metavar='TEST_DIR',
        help='After augmenting, regenerate test index from full index'
    )
    args = parser.parse_args()

    n_added = augment(args.embedding_dir, batch_size=args.batch_size)

    if args.regenerate_test_index and n_added > 0:
        regenerate_test_index(args.embedding_dir, args.regenerate_test_index)
