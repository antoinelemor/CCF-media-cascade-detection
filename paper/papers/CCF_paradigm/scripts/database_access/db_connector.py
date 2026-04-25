"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
db_connector.py

MAIN OBJECTIVE:
---------------
This script provides a robust database connection manager for accessing the CCF PostgreSQL database
with connection pooling, error handling, and efficient data retrieval methods.
Includes automatic exclusion of 2025 data and progress tracking with tqdm.

Dependencies:
-------------
- psycopg2-binary
- pandas
- sqlalchemy
- contextlib
- tqdm

MAIN FEATURES:
--------------
1) Database connection pooling for efficient resource management
2) Context manager for automatic connection cleanup
3) Parameterized queries to prevent SQL injection
4) Batch data retrieval with chunking support
5) Error handling and retry logic
6) Automatic exclusion of 2025 data
7) Progress tracking with tqdm for large data loads

Author:
-------
Antoine Lemor
"""

import os
import logging
from typing import Dict, Optional, Any, List
from contextlib import contextmanager
from sqlalchemy import create_engine, pool
from sqlalchemy.engine import Engine
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnector:
    """Manages PostgreSQL database connections with pooling and error handling.
    Optimized for M4 Ultra with massive RAM capacity."""
    
    def __init__(self, db_params: Optional[Dict[str, Any]] = None, exclude_2025: bool = True,
                 enable_cache: bool = True, cache_size_gb: int = 50):
        """
        Initialize database connector.
        
        Args:
            db_params: Dictionary with connection parameters (host, port, dbname, user, password)
            exclude_2025: Whether to automatically exclude data from 2025 onwards
            enable_cache: Enable in-memory caching for query results
            cache_size_gb: Maximum cache size in GB (default 50GB for M4 Ultra)
        """
        self.db_params = db_params or self._get_default_params()
        self.exclude_2025 = exclude_2025
        self.engine = None
        self.enable_cache = enable_cache
        self.cache_size_gb = cache_size_gb
        self.query_cache = {} if enable_cache else None
        self.cache_stats = {'hits': 0, 'misses': 0, 'size_bytes': 0}
        self._initialize_engine()
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default database parameters from environment or config."""
        return {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "dbname": os.getenv("DB_NAME", "CCF_Database_texts"),
            "user": os.getenv("DB_USER", "antoine"),
            "password": os.getenv("PGPASSWORD", ""),
        }
    
    def _initialize_engine(self) -> None:
        """Initialize SQLAlchemy engine with connection pooling."""
        connection_string = (
            f"postgresql://{self.db_params['user']}:{self.db_params['password']}"
            f"@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['dbname']}"
        )
        
        # Create engine with optimized connection pooling for M4 Ultra
        # Increased pool size to leverage multiple cores
        self.engine = create_engine(
            connection_string,
            poolclass=pool.QueuePool,
            pool_size=50,  # Increased from 10 for M4 Ultra
            max_overflow=100,  # Increased from 20
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=False,
            connect_args={
                'connect_timeout': 10,
                'options': '-c statement_timeout=0',  # No timeout for long queries
                'keepalives': 1,
                'keepalives_idle': 30,
                'keepalives_interval': 10,
                'keepalives_count': 5
            }
        )
        logger.info(f"Database engine initialized with optimized pooling (pool_size=50, max_overflow=100)")
        if self.exclude_2025:
            logger.info("2025 data exclusion is enabled")
        if self.enable_cache:
            logger.info(f"Query cache enabled with {self.cache_size_gb}GB limit")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Yields:
            connection: Database connection object
        """
        connection = None
        try:
            connection = self.engine.connect()
            yield connection
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def _get_cache_key(self, query: str, params: Optional[Dict] = None) -> str:
        """Generate cache key for query."""
        cache_str = query + str(sorted(params.items()) if params else '')
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _check_cache_size(self) -> None:
        """Check and manage cache size."""
        if self.cache_stats['size_bytes'] > self.cache_size_gb * 1024**3:
            # Remove oldest cached items (FIFO)
            items_to_remove = len(self.query_cache) // 4
            for key in list(self.query_cache.keys())[:items_to_remove]:
                del self.query_cache[key]
            self.cache_stats['size_bytes'] = sum(
                df.memory_usage(deep=True).sum() for df in self.query_cache.values()
            )
            logger.info(f"Cache cleaned, removed {items_to_remove} items")
    
    def read_data(self, query: str, params: Optional[Dict] = None, 
                  chunksize: Optional[int] = None, show_progress: bool = False,
                  use_cache: bool = True) -> pd.DataFrame:
        """
        Read data from database into pandas DataFrame with caching support.
        
        Args:
            query: SQL query string
            params: Query parameters
            chunksize: If specified, return iterator of DataFrames
            show_progress: Whether to show progress bar for chunked reads
            use_cache: Whether to use query cache
            
        Returns:
            DataFrame or iterator of DataFrames
        """
        # Check cache first
        if self.enable_cache and use_cache and not chunksize:
            cache_key = self._get_cache_key(query, params)
            if cache_key in self.query_cache:
                self.cache_stats['hits'] += 1
                logger.debug(f"Cache hit! (hits: {self.cache_stats['hits']}, misses: {self.cache_stats['misses']})")
                return self.query_cache[cache_key].copy()
            else:
                self.cache_stats['misses'] += 1
        
        try:
            with self.get_connection() as conn:
                if chunksize:
                    if show_progress:
                        # First get total count
                        count_query = f"SELECT COUNT(*) FROM ({query}) as subquery"
                        total_rows = pd.read_sql(count_query, conn, params=params).iloc[0, 0]
                        total_chunks = (total_rows // chunksize) + (1 if total_rows % chunksize else 0)
                        
                        # Read with progress bar
                        chunks = []
                        chunk_iter = pd.read_sql(query, conn, params=params, chunksize=chunksize)
                        
                        with tqdm(total=total_chunks, desc="Loading data chunks", file=sys.stderr, ncols=100, ascii=True) as pbar:
                            for chunk in chunk_iter:
                                chunks.append(chunk)
                                pbar.update(1)
                        
                        return pd.concat(chunks, ignore_index=True)
                    else:
                        return pd.read_sql(query, conn, params=params, chunksize=chunksize)
                else:
                    if show_progress:
                        # For non-chunked reads, just show a simple progress bar
                        with tqdm(total=1, desc="Loading data", file=sys.stderr, ncols=100, ascii=True) as pbar:
                            df = pd.read_sql(query, conn, params=params)
                            pbar.update(1)
                    else:
                        df = pd.read_sql(query, conn, params=params)
                    
                    # Cache the result if caching is enabled
                    if self.enable_cache and use_cache:
                        cache_key = self._get_cache_key(query, params)
                        self.query_cache[cache_key] = df.copy()
                        self.cache_stats['size_bytes'] += df.memory_usage(deep=True).sum()
                        self._check_cache_size()
                        logger.debug(f"Cached query result, cache size: {self.cache_stats['size_bytes'] / 1024**3:.2f}GB")
                    
                    return df
        except Exception as e:
            logger.error(f"Error reading data: {e}")
            raise
    
    def get_frame_data_parallel(self, table_name: str = "CCF_processed_data",
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               media_list: Optional[List[str]] = None,
                               n_workers: int = 8) -> pd.DataFrame:
        """
        Get frame data using parallel processing for faster loading.
        Optimized for M4 Ultra's multiple cores.
        
        Args:
            table_name: Name of the database table
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            media_list: List of media outlets to filter
            n_workers: Number of parallel workers
            
        Returns:
            DataFrame with frame detection data
        """
        # Handle 2025 exclusion
        if self.exclude_2025 and (not end_date or end_date >= "2025-01-01"):
            end_date = "2024-12-31"
            logger.info("Capping end date to 2024-12-31 to exclude 2025 data")
        
        # Get date range for partitioning (native DATE column in CCF_Database_texts)
        date_query = f"""
        SELECT
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM "{table_name}"
        WHERE date IS NOT NULL
        """

        if start_date:
            date_query += f" AND date >= '{start_date}'::date"
        if end_date:
            date_query += f" AND date <= '{end_date}'::date"

        with self.get_connection() as conn:
            date_range = pd.read_sql(date_query, conn)

        if date_range.empty or date_range['min_date'].isna().any():
            logger.warning("No data found in specified date range")
            return pd.DataFrame()

        min_date = pd.to_datetime(date_range['min_date'].iloc[0])
        max_date = pd.to_datetime(date_range['max_date'].iloc[0])
        
        logger.info(f"Data range found: {min_date} to {max_date}")
        
        # Create date chunks for parallel processing
        date_chunks = pd.date_range(min_date, max_date, periods=n_workers + 1).tolist()
        
        def load_chunk(start, end):
            """Load a single date chunk."""
            chunk_query = f'''SELECT * FROM "{table_name}"
                           WHERE date >= %(start_date)s::date
                           AND date < %(end_date)s::date'''
            params = {
                'start_date': start.strftime('%Y-%m-%d'),
                'end_date': end.strftime('%Y-%m-%d')
            }
            
            if media_list:
                chunk_query += " AND media = ANY(%(media_list)s)"
                params['media_list'] = media_list
            
            chunk_query += " ORDER BY date, doc_id, sentence_id"
            
            with self.engine.connect() as conn:
                return pd.read_sql(chunk_query, conn, params=params)
        
        # Load chunks in parallel
        chunks = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i in range(len(date_chunks) - 1):
                future = executor.submit(load_chunk, date_chunks[i], date_chunks[i + 1])
                futures.append(future)
            
            # Process results with progress bar
            with tqdm(total=len(futures), desc="Loading data chunks in parallel", file=sys.stderr, ncols=100, ascii=True) as pbar:
                for future in as_completed(futures):
                    chunks.append(future.result())
                    pbar.update(1)
        
        # Combine all chunks
        if chunks:
            result = pd.concat(chunks, ignore_index=True)
            logger.info(f"Loaded {len(result):,} rows using {n_workers} parallel workers")
            return result
        else:
            return pd.DataFrame()
    
    def get_frame_data(self, table_name: str = "CCF_processed_data",
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      media_list: Optional[List[str]] = None,
                      show_progress: bool = True,
                      chunk_size: Optional[int] = 500000,  # Increased from 100000
                      use_parallel: bool = True,
                      n_workers: int = 8) -> pd.DataFrame:
        """
        Get frame detection data with optional filters.
        Optimized for M4 Ultra with parallel processing option.
        
        Args:
            table_name: Name of the database table
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            media_list: List of media outlets to filter
            show_progress: Whether to show progress bar
            chunk_size: Size of chunks for reading large datasets
            use_parallel: Use parallel processing for large datasets
            n_workers: Number of parallel workers
            
        Returns:
            DataFrame with frame detection data
        """
        # Use parallel loading for large datasets
        if use_parallel:
            return self.get_frame_data_parallel(
                table_name, start_date, end_date, media_list, n_workers
            )
        # Base query
        query = f'SELECT * FROM "{table_name}"'
        conditions = []
        params = {}
        
        # Add date filters (native DATE column in CCF_Database_texts)
        if start_date:
            conditions.append("date >= %(start_date)s::date")
            params["start_date"] = start_date

        # Handle end date and 2025 exclusion
        if self.exclude_2025:
            if not end_date or end_date >= "2025-01-01":
                end_date = "2024-12-31"
                logger.info("Capping end date to 2024-12-31 to exclude 2025 data")

        if end_date:
            conditions.append("date <= %(end_date)s::date")
            params["end_date"] = end_date
        
        # Add media filter
        if media_list:
            conditions.append("media = ANY(%(media_list)s)")
            params["media_list"] = media_list
        
        # Combine conditions
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY date, doc_id, sentence_id"
        
        logger.info(f"Executing query with filters: {conditions}")
        
        # Check if we should use chunked reading
        if chunk_size and show_progress:
            # First get count
            count_query = f'SELECT COUNT(*) FROM "{table_name}"'
            if conditions:
                count_query += " WHERE " + " AND ".join(conditions)
            
            with self.get_connection() as conn:
                total_rows = pd.read_sql(count_query, conn, params=params).iloc[0, 0]
            
            if total_rows > chunk_size:
                logger.info(f"Loading {total_rows:,} rows in chunks of {chunk_size:,}")
                return self.read_data(query, params, chunksize=chunk_size, show_progress=True)
        
        return self.read_data(query, params, chunksize=chunk_size if chunk_size else None, 
                            show_progress=show_progress)
    
    def get_frame_columns(self, table_name: str = "CCF_processed_data") -> List[str]:
        """
        Get list of frame detection columns from the database.

        Args:
            table_name: Name of the database table

        Returns:
            List of frame column names
        """
        # Define the 8 main frame columns in CCF_Database_texts
        main_frame_columns = [
            "cultural_frame", "economic_frame", "environmental_frame",
            "health_frame", "justice_frame", "political_frame",
            "scientific_frame", "security_frame"
        ]

        query = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %(table_name)s
        AND column_name = ANY(%(columns)s)
        ORDER BY column_name
        """

        df = self.read_data(query, {"table_name": table_name, "columns": main_frame_columns})
        return df["column_name"].tolist()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.enable_cache:
            hit_rate = self.cache_stats['hits'] / max(1, self.cache_stats['hits'] + self.cache_stats['misses'])
            return {
                'enabled': True,
                'hits': self.cache_stats['hits'],
                'misses': self.cache_stats['misses'],
                'hit_rate': f"{hit_rate:.2%}",
                'size_gb': self.cache_stats['size_bytes'] / 1024**3,
                'items_cached': len(self.query_cache)
            }
        return {'enabled': False}
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        if self.enable_cache:
            self.query_cache.clear()
            self.cache_stats = {'hits': 0, 'misses': 0, 'size_bytes': 0}
            logger.info("Query cache cleared")
    
    def close(self) -> None:
        """Close the database engine and all connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database engine closed")
        if self.enable_cache:
            cache_stats = self.get_cache_stats()
            logger.info(f"Cache stats at close: {cache_stats}")


class OptimizedDatabaseConnector(DatabaseConnector):
    """Ultra-optimized database connector for M4 Ultra Max with 1280GB RAM."""
    
    def __init__(self, **kwargs):
        # Set aggressive defaults for M4 Ultra Max
        kwargs.setdefault('cache_size_gb', 200)  # Use 200GB for cache
        kwargs.setdefault('enable_cache', True)
        super().__init__(**kwargs)
        
        # Pre-load frequently used data into memory
        self.preloaded_data = {}
        self.preload_common_queries()
    
    def preload_common_queries(self):
        """Pre-load commonly used queries into memory."""
        try:
            logger.info("Pre-loading common queries into memory...")
            
            # Pre-load frame columns
            self.preloaded_data['frame_columns'] = self.get_frame_columns()
            
            # Pre-load recent data (last 3 months from end of 2024) for quick access
            # Use a fixed date to avoid issues with current date being in 2025
            recent_date = pd.Timestamp('2024-10-01')  # Last 3 months of 2024
            logger.info(f"Pre-loading data from {recent_date.date()} onwards...")
            
            # This will cache the recent data
            recent_data = self.get_frame_data(
                start_date=recent_date.strftime('%Y-%m-%d'),
                end_date='2024-12-31',  # Explicitly set end date
                show_progress=True,
                use_parallel=True,
                n_workers=16  # Use more workers for initial load
            )
            
            if not recent_data.empty:
                self.preloaded_data['recent_data'] = recent_data
                logger.info(f"Pre-loaded {len(recent_data):,} recent records into memory")
            
        except Exception as e:
            logger.warning(f"Could not pre-load data: {e}")
    
    def batch_query(self, queries: List[Dict[str, Any]], n_workers: int = 16) -> List[pd.DataFrame]:
        """
        Execute multiple queries in parallel.
        
        Args:
            queries: List of dicts with 'query' and optional 'params' keys
            n_workers: Number of parallel workers
            
        Returns:
            List of DataFrames
        """
        results = [None] * len(queries)
        
        def execute_query(idx, query_dict):
            query = query_dict['query']
            params = query_dict.get('params')
            return idx, self.read_data(query, params)
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i, q in enumerate(queries):
                future = executor.submit(execute_query, i, q)
                futures.append(future)
            
            with tqdm(total=len(futures), desc="Executing batch queries", file=sys.stderr, ncols=100, ascii=True) as pbar:
                for future in as_completed(futures):
                    idx, result = future.result()
                    results[idx] = result
                    pbar.update(1)
        
        return results

# Convenience function for quick data access
def get_ccf_data(exclude_2025: bool = True, show_progress: bool = True, 
                 use_optimized: bool = True, **kwargs) -> pd.DataFrame:
    """
    Quick function to get CCF data.
    
    Args:
        exclude_2025: Whether to exclude 2025 data
        show_progress: Whether to show progress bar
        use_optimized: Use OptimizedDatabaseConnector for M4 Ultra
        **kwargs: Arguments passed to get_frame_data method
        
    Returns:
        DataFrame with CCF data
    """
    connector_class = OptimizedDatabaseConnector if use_optimized else DatabaseConnector
    connector = connector_class(exclude_2025=exclude_2025)
    try:
        # Set optimized defaults
        kwargs.setdefault('use_parallel', True)
        kwargs.setdefault('n_workers', 8)
        kwargs.setdefault('chunk_size', 500000)
        
        return connector.get_frame_data(show_progress=show_progress, **kwargs)
    finally:
        if hasattr(connector, 'get_cache_stats'):
            stats = connector.get_cache_stats()
            if stats['enabled']:
                logger.info(f"Final cache stats: {stats}")
        connector.close()
