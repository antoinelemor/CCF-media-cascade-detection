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
1) Date conversion from MM-DD-YYYY to ISO format with validation
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
from cascade_detector.core.constants import FRAMES, MESSENGERS

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
        """Convert dates from MM-DD-YYYY to ISO format."""
        if 'date_converted' not in df.columns:
            logger.info("Converting dates to ISO format...")
            
            # Handle MM-DD-YYYY format
            df['date_converted'] = pd.to_datetime(
                df['date'].str[6:10] + '-' +  # Year
                df['date'].str[0:2] + '-' +   # Month
                df['date'].str[3:5],          # Day
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
        """Clean and convert frame detection columns."""
        logger.info("Cleaning frame columns...")
        
        frame_cols = [f"{frame}_Detection" for frame in self.frames]
        existing_cols = [col for col in frame_cols if col in df.columns]
        
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
        df['n_frames'] = df[existing_cols].sum(axis=1)
        
        return df
    
    def _clean_messenger_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert messenger columns."""
        logger.info("Cleaning messenger columns...")
        
        existing_cols = [col for col in self.messengers if col in df.columns]
        
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
        frame_cols = [f"{frame}_Detection" for frame in self.frames 
                     if f"{frame}_Detection" in df.columns]
        
        if frame_cols:
            df['dominant_frame'] = df[frame_cols].idxmax(axis=1)
            df['dominant_frame'] = df['dominant_frame'].str.replace('_Detection', '')
            
            # Set to None if no frame detected
            df.loc[df['n_frames'] == 0, 'dominant_frame'] = None
        
        return df
    
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
        
        # Add frame aggregations
        for frame in self.frames:
            col = f"{frame}_Detection"
            if col in df.columns:
                agg_dict[col] = ['sum', 'mean']
        
        # Add messenger aggregations
        for messenger in self.messengers:
            if messenger in df.columns:
                agg_dict[messenger] = ['sum', 'mean']
        
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
        for frame in self.frames:
            sum_col = f"{frame}_Detection_sum"
            if sum_col in df.columns and 'n_sentences' in df.columns:
                prop_col = f"{frame}_proportion"
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