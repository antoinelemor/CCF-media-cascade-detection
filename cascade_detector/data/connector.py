"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
connector.py

MAIN OBJECTIVE:
---------------
This script manages PostgreSQL database connections for the cascade detection framework, providing
optimized batch data loading from the CCF database with date filtering and frame selection.

Dependencies:
-------------
- os
- logging
- pandas
- sqlalchemy
- contextlib

MAIN FEATURES:
--------------
1) SQLAlchemy engine initialization with connection pooling
2) Context manager for safe database connections
3) Frame data retrieval with date and frame filtering
4) Automatic handling of date format conversion (MM-DD-YYYY to ISO)
5) Query optimization for large-scale data loading

Author:
-------
Antoine Lemor
"""

import os
import logging
from typing import Dict, Optional, Any, List
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from contextlib import contextmanager

from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.exceptions import DatabaseConnectionError

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    Manages PostgreSQL database connections for cascade detection.
    Optimized for batch data loading.
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize database connector.
        
        Args:
            config: Detector configuration
        """
        self.config = config or DetectorConfig()
        self.engine = None
        self._initialize_engine()
    
    def _initialize_engine(self) -> None:
        """Initialize SQLAlchemy engine."""
        connection_string = (
            f"postgresql://{self.config.db_user}:{self.config.db_password}"
            f"@{self.config.db_host}:{self.config.db_port}/{self.config.db_name}"
        )
        
        try:
            self.engine = create_engine(
                connection_string,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False
            )
            logger.info(f"Database engine initialized for {self.config.db_name}")
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to initialize database engine: {e}")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Yields:
            Database connection
        """
        connection = None
        try:
            connection = self.engine.connect()
            yield connection
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise DatabaseConnectionError(f"Connection failed: {e}")
        finally:
            if connection:
                connection.close()
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            True if connection successful
        """
        try:
            from sqlalchemy import text
            with self.get_connection() as conn:
                result = conn.execute(text("SELECT 1"))
                return result.fetchone()[0] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_frame_data(self,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      frames: Optional[List[str]] = None,
                      limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get frame detection data from database.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frames: List of frames to include
            limit: Maximum rows to return
            
        Returns:
            DataFrame with frame data
        """
        # Base query
        query = f'SELECT * FROM "{self.config.db_table}"'
        conditions = []
        params = {}
        
        # Add date filters (handle MM-DD-YYYY format)
        if start_date:
            conditions.append(
                "SUBSTRING(date, 7, 4)||'-'||SUBSTRING(date, 1, 2)||'-'||SUBSTRING(date, 4, 2) >= %(start_date)s"
            )
            params["start_date"] = start_date
        
        # Handle end date and 2025 exclusion
        if self.config.exclude_2025:
            if not end_date or end_date >= "2025-01-01":
                end_date = "2024-12-31"
                logger.info("Capping end date to 2024-12-31 to exclude 2025 data")
        
        if end_date:
            conditions.append(
                "SUBSTRING(date, 7, 4)||'-'||SUBSTRING(date, 1, 2)||'-'||SUBSTRING(date, 4, 2) <= %(end_date)s"
            )
            params["end_date"] = end_date
        
        # Combine conditions
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # Add ordering
        query += " ORDER BY date, doc_id, sentence_id"
        
        # Add limit if specified
        if limit:
            query += f" LIMIT {limit}"
        
        logger.info(f"Executing query with filters: {conditions}")
        
        # Execute query
        try:
            with self.get_connection() as conn:
                df = pd.read_sql(query, conn, params=params)
                logger.info(f"Loaded {len(df):,} rows from database")
                return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise DatabaseConnectionError(f"Failed to load data: {e}")
    
    def get_row_count(self) -> int:
        """
        Get total row count in the table.
        
        Returns:
            Number of rows
        """
        from sqlalchemy import text
        query = f'SELECT COUNT(*) FROM "{self.config.db_table}"'
        
        # Add 2025 exclusion if enabled
        if self.config.exclude_2025:
            query += " WHERE SUBSTRING(date, 7, 4)||'-'||SUBSTRING(date, 1, 2)||'-'||SUBSTRING(date, 4, 2) < '2025-01-01'"
        
        try:
            with self.get_connection() as conn:
                result = conn.execute(text(query))
                count = result.fetchone()[0]
                return count
        except Exception as e:
            logger.error(f"Error getting row count: {e}")
            return 0
    
    def get_date_range(self) -> Dict[str, str]:
        """
        Get date range of data in database.
        
        Returns:
            Dictionary with 'start' and 'end' dates
        """
        query = f"""
        SELECT 
            MIN(SUBSTRING(date, 7, 4)||'-'||SUBSTRING(date, 1, 2)||'-'||SUBSTRING(date, 4, 2)) as min_date,
            MAX(SUBSTRING(date, 7, 4)||'-'||SUBSTRING(date, 1, 2)||'-'||SUBSTRING(date, 4, 2)) as max_date
        FROM "{self.config.db_table}"
        WHERE date IS NOT NULL
        """
        
        # Add 2025 exclusion if enabled
        if self.config.exclude_2025:
            query += " AND SUBSTRING(date, 7, 4)||'-'||SUBSTRING(date, 1, 2)||'-'||SUBSTRING(date, 4, 2) < '2025-01-01'"
        
        try:
            with self.get_connection() as conn:
                result = pd.read_sql(query, conn)
                if not result.empty:
                    return {
                        'start': result['min_date'].iloc[0],
                        'end': result['max_date'].iloc[0]
                    }
                return {'start': None, 'end': None}
        except Exception as e:
            logger.error(f"Error getting date range: {e}")
            return {'start': None, 'end': None}
    
    def close(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database engine closed")