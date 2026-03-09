"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
processor.py

MAIN OBJECTIVE:
---------------
This script handles data cleaning and preprocessing for the cascade detection framework, converting
raw database data into analysis-ready formats with proper date handling and column normalization.

Dependencies:
-------------
- pandas
- numpy
- logging
- typing
- tqdm

MAIN FEATURES:
--------------
1) Native DATE type handling (CCF_Database uses proper DATE columns)
2) Frame detection column cleaning and binarization
3) Messenger column processing and normalization
4) Derived column generation (year, month, week, frame counts)
5) Invalid row removal and data quality checks

Author:
-------
Antoine Lemor
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any
from tqdm import tqdm

from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.constants import (
    FRAMES, MESSENGERS, FRAME_COLUMNS,
    EVENT_MAIN, EVENT_COLUMNS, SOLUTION_MAIN, SOLUTION_COLUMNS
)

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Processes and prepares data for cascade detection.
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize data processor.
        
        Args:
            config: Detector configuration
        """
        self.config = config or DetectorConfig()
        self.frames = self.config.frames
        self.messengers = self.config.messengers
    
    def process_frame_data(self, df: pd.DataFrame, 
                          show_progress: bool = False) -> pd.DataFrame:
        """
        Process frame detection data.
        
        Args:
            df: Raw DataFrame from database
            show_progress: Show progress bar
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Processing {len(df):,} rows...")
        
        # Convert dates
        df = self._convert_dates(df)
        
        # Clean frame columns
        df = self._clean_frame_columns(df, show_progress)
        
        # Clean messenger columns
        df = self._clean_messenger_columns(df)

        # Clean event columns
        df = self._clean_event_columns(df)

        # Clean solution columns
        df = self._clean_solution_columns(df)

        # Add derived columns
        df = self._add_derived_columns(df)
        
        # Remove invalid rows
        initial_rows = len(df)
        df = self._remove_invalid_rows(df)
        removed = initial_rows - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} invalid rows")
        
        logger.info(f"Processing complete: {len(df):,} rows")
        
        return df
    
    def _convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process date column from CCF_Database.

        CCF_Database uses native PostgreSQL DATE type, so dates are already
        in proper format (YYYY-MM-DD). We just need to ensure pandas datetime.
        """
        if 'date_converted' not in df.columns:
            logger.info("Processing dates...")

            # CCF_Database has native DATE type - pandas reads it as datetime64
            if pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date_converted'] = df['date']
            else:
                # Fallback: parse as date string (YYYY-MM-DD format)
                df['date_converted'] = pd.to_datetime(
                    df['date'],
                    format='%Y-%m-%d',
                    errors='coerce'
                )

            # Check for invalid dates
            invalid_dates = df['date_converted'].isna().sum()
            if invalid_dates > 0:
                logger.warning(f"Found {invalid_dates} invalid dates")

        # Add time-based columns
        df['year'] = df['date_converted'].dt.year
        df['month'] = df['date_converted'].dt.month
        df['week'] = df['date_converted'].dt.isocalendar().week
        df['day_of_week'] = df['date_converted'].dt.dayofweek

        return df
    
    def _clean_frame_columns(self, df: pd.DataFrame,
                            show_progress: bool = False) -> pd.DataFrame:
        """
        Clean and convert frame detection columns.

        CCF_Database uses new column names (economic_frame, health_frame, etc.)
        instead of old format (Eco_Detection, Pbh_Detection, etc.)
        """
        logger.info("Cleaning frame columns...")

        # Get frame column names from FRAME_COLUMNS mapping
        # FRAME_COLUMNS maps internal name (Eco) -> DB column (economic_frame)
        frame_cols = list(FRAME_COLUMNS.values())
        existing_cols = [col for col in frame_cols if col in df.columns]

        # Fallback: check for legacy column names if new ones not found
        if not existing_cols:
            legacy_cols = [f"{frame}_Detection" for frame in self.frames]
            existing_cols = [col for col in legacy_cols if col in df.columns]
            if existing_cols:
                logger.info("Using legacy column names (old database format)")

        if show_progress:
            iterator = tqdm(existing_cols, desc="Cleaning frames")
        else:
            iterator = existing_cols

        for col in iterator:
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Ensure binary (0 or 1)
            df[col] = (df[col] > 0).astype(int)

        # Add frame count column
        if existing_cols:
            df['n_frames'] = df[existing_cols].sum(axis=1)
        else:
            df['n_frames'] = 0
            logger.warning("No frame columns found in data")

        return df
    
    def _clean_messenger_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and convert messenger columns.

        CCF_Database uses new column names (msg_health, msg_scientist, etc.)
        instead of old format (Messenger_1_SUB, Messenger_6_SUB, etc.)
        """
        logger.info("Cleaning messenger columns...")

        # MESSENGERS constant now contains new column names
        existing_cols = [col for col in self.messengers if col in df.columns]

        # Fallback: check for legacy column names if new ones not found
        if not existing_cols:
            legacy_cols = [f'Messenger_{i}_SUB' for i in range(1, 10)]
            existing_cols = [col for col in legacy_cols if col in df.columns]
            if existing_cols:
                logger.info("Using legacy messenger column names (old database format)")

        for col in existing_cols:
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Ensure binary
            df[col] = (df[col] > 0).astype(int)

        # Add messenger count column
        if existing_cols:
            df['n_messengers'] = df[existing_cols].sum(axis=1)
        else:
            df['n_messengers'] = 0

        return df
    
    def _clean_event_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and convert event detection columns.

        CCF_Database uses column names: event (main), evt_weather, evt_meeting, etc.
        """
        logger.info("Cleaning event columns...")

        # Clean main event column
        if EVENT_MAIN in df.columns:
            df[EVENT_MAIN] = pd.to_numeric(df[EVENT_MAIN], errors='coerce').fillna(0)
            df[EVENT_MAIN] = (df[EVENT_MAIN] > 0).astype(int)

        # Clean sub-type columns
        existing_cols = [col for col in EVENT_COLUMNS if col in df.columns]

        for col in existing_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df[col] = (df[col] > 0).astype(int)

        # Add event count and dominant event columns
        if existing_cols:
            df['n_events'] = df[existing_cols].sum(axis=1)
            # Dominant event type (only when at least one event detected)
            df['dominant_event'] = df[existing_cols].idxmax(axis=1)
            df.loc[df['n_events'] == 0, 'dominant_event'] = None
        else:
            df['n_events'] = 0
            df['dominant_event'] = None

        return df

    def _clean_solution_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and convert solution detection columns.

        CCF_Database uses column names: solution (main), sol_mitigation, sol_adaptation.
        """
        logger.info("Cleaning solution columns...")

        # Clean main solution column
        if SOLUTION_MAIN in df.columns:
            df[SOLUTION_MAIN] = pd.to_numeric(df[SOLUTION_MAIN], errors='coerce').fillna(0)
            df[SOLUTION_MAIN] = (df[SOLUTION_MAIN] > 0).astype(int)

        # Clean sub-type columns
        existing_cols = [col for col in SOLUTION_COLUMNS if col in df.columns]

        for col in existing_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df[col] = (df[col] > 0).astype(int)

        # Add solution count
        if existing_cols:
            df['n_solutions'] = df[existing_cols].sum(axis=1)
        else:
            df['n_solutions'] = 0

        return df

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful derived columns."""
        # Clean author names
        if 'author' in df.columns:
            df['author_clean'] = df['author'].apply(self._clean_author_name)

        # Add article length (sentences per article)
        if 'doc_id' in df.columns:
            article_lengths = df.groupby('doc_id').size()
            df['article_length'] = df['doc_id'].map(article_lengths)

        # Add dominant frame per sentence
        # Try new column names first, then legacy
        frame_cols = [col for col in FRAME_COLUMNS.values() if col in df.columns]
        if not frame_cols:
            frame_cols = [f"{frame}_Detection" for frame in self.frames
                         if f"{frame}_Detection" in df.columns]

        if frame_cols:
            df['dominant_frame'] = df[frame_cols].idxmax(axis=1)
            # Convert to internal frame name
            # e.g., 'economic_frame' -> 'Eco' or 'Eco_Detection' -> 'Eco'
            df['dominant_frame'] = df['dominant_frame'].apply(self._normalize_frame_name)

            # Set to None if no frame detected
            df.loc[df['n_frames'] == 0, 'dominant_frame'] = None

        return df

    def _normalize_frame_name(self, col_name: str) -> str:
        """Convert column name to internal frame name."""
        if col_name is None:
            return None

        # Check new format (economic_frame -> Eco)
        from cascade_detector.core.constants import FRAME_COLUMNS_REVERSE
        if col_name in FRAME_COLUMNS_REVERSE:
            return FRAME_COLUMNS_REVERSE[col_name]

        # Check legacy format (Eco_Detection -> Eco)
        if col_name.endswith('_Detection'):
            return col_name.replace('_Detection', '')

        return col_name
    
    def _clean_author_name(self, name: str) -> str:
        """Clean author name for better matching."""
        if pd.isna(name):
            return ""
        
        name = str(name).strip()
        
        # Remove common suffixes
        suffixes = [', Staff', ', Reuters', ', AP', ', Canadian Press', 
                   ', Bloomberg', ', Associated Press']
        for suffix in suffixes:
            name = name.replace(suffix, '')
        
        return name.strip()
    
    def _remove_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove invalid rows."""
        initial_len = len(df)
        
        # Remove rows with invalid dates
        if 'date_converted' in df.columns:
            df = df[df['date_converted'].notna()]
        
        # Remove rows with missing doc_id
        if 'doc_id' in df.columns:
            df = df[df['doc_id'].notna()]
        
        # Remove rows from 2025 if configured
        if self.config.exclude_2025 and 'year' in df.columns:
            df = df[df['year'] < 2025]
        
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} invalid rows")
        
        return df
    
    def aggregate_by_article(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data at article level.

        Args:
            df: Sentence-level DataFrame

        Returns:
            Article-level DataFrame
        """
        logger.info("Aggregating data by article...")

        if 'doc_id' not in df.columns:
            logger.error("No doc_id column found")
            return pd.DataFrame()

        # Prepare aggregation dict
        agg_dict = {
            'date_converted': 'first',
            'media': 'first',
            'author': 'first',
            'author_clean': 'first',
            'year': 'first',
            'month': 'first',
            'week': 'first'
        }

        # Add frame aggregations - try new column names first
        frame_cols = [col for col in FRAME_COLUMNS.values() if col in df.columns]
        if not frame_cols:
            frame_cols = [f"{frame}_Detection" for frame in self.frames
                         if f"{frame}_Detection" in df.columns]

        for col in frame_cols:
            agg_dict[col] = ['sum', 'mean']

        # Add messenger aggregations
        for messenger in self.messengers:
            if messenger in df.columns:
                agg_dict[messenger] = ['sum', 'mean']

        # Add event aggregations (main event flag + per-type)
        if EVENT_MAIN in df.columns:
            agg_dict[EVENT_MAIN] = ['sum', 'mean']
        for evt_col in EVENT_COLUMNS:
            if evt_col in df.columns:
                agg_dict[evt_col] = ['sum', 'mean']

        # Add solution aggregations
        for sol_col in SOLUTION_COLUMNS:
            if sol_col in df.columns:
                agg_dict[sol_col] = ['sum', 'mean']

        # Aggregate
        article_df = df.groupby('doc_id').agg(agg_dict)

        # Flatten column names
        article_df.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                              for col in article_df.columns.values]

        # Add article statistics
        article_stats = df.groupby('doc_id').agg({
            'sentence_id': 'count',  # Number of sentences
            'n_frames': 'mean',      # Average frames per sentence
            'n_messengers': 'mean'   # Average messengers per sentence
        })

        article_stats.columns = ['n_sentences', 'avg_frames', 'avg_messengers']
        article_df = article_df.join(article_stats)

        # Reset index
        article_df = article_df.reset_index()

        logger.info(f"Aggregated to {len(article_df):,} articles")

        return article_df
    
    def calculate_frame_proportions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate frame proportions for each article.

        Args:
            df: DataFrame with frame columns

        Returns:
            DataFrame with proportion columns added
        """
        # Try new column names first
        for frame, db_col in FRAME_COLUMNS.items():
            sum_col = f"{db_col}_sum"
            if sum_col in df.columns and 'n_sentences' in df.columns:
                prop_col = f"{frame}_proportion"
                df[prop_col] = df[sum_col] / df['n_sentences']

        # Fallback: legacy column names
        for frame in self.frames:
            sum_col = f"{frame}_Detection_sum"
            if sum_col in df.columns and 'n_sentences' in df.columns:
                prop_col = f"{frame}_proportion"
                if prop_col not in df.columns:  # Don't overwrite if already set
                    df[prop_col] = df[sum_col] / df['n_sentences']

        return df
    
    def filter_by_activity(self, df: pd.DataFrame,
                          min_articles: int = 10,
                          min_journalists: int = 3,
                          min_media: int = 2) -> pd.DataFrame:
        """
        Filter data to periods with sufficient activity.
        
        Args:
            df: Input DataFrame
            min_articles: Minimum articles per period
            min_journalists: Minimum journalists per period
            min_media: Minimum media outlets per period
            
        Returns:
            Filtered DataFrame
        """
        # Group by week
        weekly_stats = df.groupby('week').agg({
            'doc_id': 'nunique',
            'author': 'nunique',
            'media': 'nunique'
        })
        
        weekly_stats.columns = ['n_articles', 'n_journalists', 'n_media']
        
        # Find active weeks
        active_weeks = weekly_stats[
            (weekly_stats['n_articles'] >= min_articles) &
            (weekly_stats['n_journalists'] >= min_journalists) &
            (weekly_stats['n_media'] >= min_media)
        ].index
        
        # Filter data
        filtered = df[df['week'].isin(active_weeks)]
        
        logger.info(f"Filtered to {len(filtered):,} rows in {len(active_weeks)} active weeks")
        
        return filtered