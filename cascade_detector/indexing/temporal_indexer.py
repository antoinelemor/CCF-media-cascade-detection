"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
temporal_indexer.py

MAIN OBJECTIVE:
---------------
This script creates optimized temporal indices for time series analysis of frame detection data,
providing multi-resolution temporal aggregations and efficient date-based lookups for cascade detection.

Dependencies:
-------------
- pandas
- numpy
- typing
- collections
- datetime
- logging

MAIN FEATURES:
--------------
1) Multi-resolution time series creation (daily, weekly, monthly)
2) Date format conversion from MM-DD-YYYY to ISO format
3) Temporal mapping for efficient date-based queries
4) Statistical summaries per time period
5) Frame-specific temporal indices with activity tracking

Author:
-------
Antoine Lemor
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import logging

from cascade_detector.indexing.base_indexer import AbstractIndexer
from cascade_detector.core.constants import FRAMES, DATE_FORMAT_DB, DATE_FORMAT_ISO
from cascade_detector.core.models import IndexEntry

logger = logging.getLogger(__name__)


class TemporalIndexer(AbstractIndexer):
    """
    Creates optimized temporal indices for each frame.
    Handles date conversion from MM-DD-YYYY to ISO format.
    """
    
    def __init__(self):
        """Initialize temporal indexer."""
        super().__init__(name="TemporalIndexer")
        self.frames = FRAMES
        
    def get_required_columns(self) -> List[str]:
        """Get required columns."""
        # Only require base columns, frames are optional
        base_cols = ['date', 'doc_id', 'sentence_id', 'media', 'author']
        return base_cols
    
    def build_index(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Build temporal index from data.
        
        Creates structure: {
            frame: {
                'daily_series': pd.Series,
                'weekly_series': pd.Series,
                'mapping': {date: [row_indices]},
                'statistics': {...}
            }
        }
        """
        logger.info(f"Building temporal index for {len(data):,} rows...")
        self.validate_data(data)
        
        # Convert dates from MM-DD-YYYY to ISO format
        data = self._convert_dates(data)
        
        # Initialize index structure
        self.index = {}
        
        # Process each frame
        for frame in self.frames:
            frame_col = f"{frame}_Detection"
            
            if frame_col not in data.columns:
                logger.warning(f"Frame column {frame_col} not found, skipping")
                continue
            
            # Convert to numeric and filter
            data[frame_col] = pd.to_numeric(data[frame_col], errors='coerce').fillna(0)
            
            # Build frame index
            frame_index = self._build_frame_index(data, frame, frame_col)
            self.index[frame] = frame_index
            
            logger.info(f"  {frame}: {len(frame_index['daily_series'])} days indexed")
        
        # Update metadata
        self.metadata['created'] = datetime.now().isoformat()
        self.metadata['n_entries'] = len(data)
        self.metadata['date_range'] = {
            'start': data['date_converted'].min().isoformat(),
            'end': data['date_converted'].max().isoformat()
        }
        
        logger.info(f"Temporal index built: {len(self.index)} frames")
        return self.index
    
    def _convert_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert dates from MM-DD-YYYY to ISO format."""
        if 'date_converted' not in data.columns:
            # Parse MM-DD-YYYY format
            data['date_converted'] = pd.to_datetime(
                data['date'].str[6:10] + '-' +  # Year
                data['date'].str[0:2] + '-' +   # Month  
                data['date'].str[3:5],          # Day
                format='%Y-%m-%d',
                errors='coerce'
            )
            
            # Check for parsing errors
            invalid_dates = data['date_converted'].isna().sum()
            if invalid_dates > 0:
                logger.warning(f"Found {invalid_dates} invalid dates")
        
        return data
    
    def _build_frame_index(self, data: pd.DataFrame, frame: str, frame_col: str) -> Dict:
        """Build index for a single frame."""
        # Group by date
        frame_data = data[data[frame_col] == 1].copy()
        
        # Daily mapping: date -> row indices
        daily_mapping = defaultdict(list)
        # NEW: Also create articles_by_date mapping: date -> doc_ids
        articles_by_date = defaultdict(set)
        for idx, row in frame_data.iterrows():
            daily_mapping[row['date_converted']].append(idx)
            # Store doc_id for articles_by_date
            articles_by_date[row['date_converted']].add(row['doc_id'])
        
        # Create time series
        all_dates = pd.date_range(
            start=data['date_converted'].min(),
            end=data['date_converted'].max(),
            freq='D'
        )
        
        # Daily counts
        daily_counts = pd.Series(
            [len(daily_mapping.get(date, [])) for date in all_dates],
            index=all_dates
        )
        
        # Daily proportions (normalized by total daily volume)
        daily_totals = data.groupby('date_converted').size()
        daily_props = pd.Series(index=all_dates, dtype=float)
        
        for date in all_dates:
            if date in daily_totals.index and daily_totals[date] > 0:
                count = len(daily_mapping.get(date, []))
                daily_props[date] = count / daily_totals[date]
            else:
                daily_props[date] = 0.0
        
        # Weekly aggregates
        weekly_counts = daily_counts.resample('W').sum()
        weekly_props = daily_props.resample('W').mean()
        
        # Calculate statistics
        statistics = self._calculate_statistics(daily_counts, daily_props)
        
        # Convert articles_by_date sets to lists
        articles_by_date_final = {}
        for date, doc_ids in articles_by_date.items():
            articles_by_date_final[date] = list(doc_ids)
        
        return {
            'daily_series': daily_counts,
            'daily_proportions': daily_props,
            'weekly_series': weekly_counts,
            'weekly_proportions': weekly_props,
            'mapping': dict(daily_mapping),
            'articles_by_date': articles_by_date_final,  # NEW: Add articles_by_date
            'statistics': statistics,
            'frame': frame
        }
    
    def _calculate_statistics(self, counts: pd.Series, proportions: pd.Series) -> Dict:
        """Calculate time series statistics."""
        return {
            'mean_daily_count': float(counts.mean()),
            'std_daily_count': float(counts.std()),
            'max_daily_count': float(counts.max()),
            'mean_proportion': float(proportions.mean()),
            'std_proportion': float(proportions.std()),
            'max_proportion': float(proportions.max()),
            'n_active_days': int((counts > 0).sum()),
            'activity_rate': float((counts > 0).mean())
        }
    
    def update_index(self, new_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Update index with new data."""
        logger.info(f"Updating temporal index with {len(new_data):,} new rows...")
        
        # Convert dates
        new_data = self._convert_dates(new_data)
        
        # Update each frame
        for frame in self.frames:
            if frame not in self.index:
                continue
            
            frame_col = f"{frame}_Detection"
            if frame_col not in new_data.columns:
                continue
            
            # Update mappings
            frame_data = new_data[new_data[frame_col] == 1]
            for idx, row in frame_data.iterrows():
                date = row['date_converted']
                if date not in self.index[frame]['mapping']:
                    self.index[frame]['mapping'][date] = []
                self.index[frame]['mapping'][date].append(idx)
            
            # Rebuild time series (more efficient to rebuild than update)
            self._rebuild_time_series(frame)
        
        # Update metadata
        self.metadata['updated'] = datetime.now().isoformat()
        self.metadata['n_entries'] += len(new_data)
        
        return self.index
    
    def _rebuild_time_series(self, frame: str) -> None:
        """Rebuild time series for a frame after update."""
        mapping = self.index[frame]['mapping']
        
        # Get date range
        all_dates = sorted(mapping.keys())
        if not all_dates:
            return
        
        date_range = pd.date_range(start=min(all_dates), end=max(all_dates), freq='D')
        
        # Rebuild daily series
        daily_counts = pd.Series(
            [len(mapping.get(date, [])) for date in date_range],
            index=date_range
        )
        
        # Update index
        self.index[frame]['daily_series'] = daily_counts
        self.index[frame]['weekly_series'] = daily_counts.resample('W').sum()
        self.index[frame]['statistics'] = self._calculate_statistics(
            daily_counts, 
            self.index[frame].get('daily_proportions', daily_counts)
        )
    
    def query_index(self, criteria: Dict[str, Any]) -> List[Any]:
        """
        Query temporal index.
        
        Criteria:
            - frame: Frame name
            - start_date: Start date
            - end_date: End date
            - min_activity: Minimum daily activity
        """
        frame = criteria.get('frame')
        start_date = criteria.get('start_date')
        end_date = criteria.get('end_date')
        min_activity = criteria.get('min_activity', 0)
        
        results = []
        
        if frame and frame in self.index:
            frame_index = self.index[frame]
            series = frame_index['daily_series']
            
            # Filter by date range
            if start_date:
                series = series[series.index >= pd.to_datetime(start_date)]
            if end_date:
                series = series[series.index <= pd.to_datetime(end_date)]
            
            # Filter by activity
            if min_activity > 0:
                series = series[series >= min_activity]
            
            # Get indices for matching dates
            for date in series.index:
                if date in frame_index['mapping']:
                    results.extend(frame_index['mapping'][date])
        
        return results
    
    def get_time_series(self, frame: str, 
                       frequency: str = 'daily',
                       metric: str = 'count') -> pd.Series:
        """
        Get time series for a frame.
        
        Args:
            frame: Frame name
            frequency: 'daily' or 'weekly'
            metric: 'count' or 'proportion'
            
        Returns:
            Time series
        """
        if frame not in self.index:
            return pd.Series()
        
        frame_index = self.index[frame]
        
        if frequency == 'daily':
            if metric == 'count':
                return frame_index['daily_series']
            else:
                return frame_index['daily_proportions']
        else:  # weekly
            if metric == 'count':
                return frame_index['weekly_series']
            else:
                return frame_index['weekly_proportions']
    
    def find_peaks(self, frame: str, 
                  min_prominence: float = 0.1) -> List[Tuple[pd.Timestamp, float]]:
        """
        Find peaks in time series.
        
        Args:
            frame: Frame name
            min_prominence: Minimum prominence for peak detection
            
        Returns:
            List of (date, value) tuples for peaks
        """
        from scipy.signal import find_peaks
        
        series = self.get_time_series(frame, 'daily', 'proportion')
        if series.empty:
            return []
        
        # Find peaks
        peaks, properties = find_peaks(
            series.values,
            prominence=min_prominence,
            distance=7  # Minimum 7 days between peaks
        )
        
        # Extract peak dates and values
        peak_list = []
        for idx in peaks:
            date = series.index[idx]
            value = series.iloc[idx]
            peak_list.append((date, value))
        
        return peak_list
    
    def calculate_burst_periods(self, frame: str, 
                               method: str = 'percentile',
                               threshold: float = 0.75) -> List[Dict]:
        """
        Calculate burst periods for a frame.
        
        Args:
            frame: Frame name
            method: 'percentile' or 'std'
            threshold: Threshold for burst detection
            
        Returns:
            List of burst periods
        """
        series = self.get_time_series(frame, 'daily', 'proportion')
        if series.empty:
            return []
        
        # Calculate threshold
        if method == 'percentile':
            burst_threshold = series.quantile(threshold)
        else:  # std method
            burst_threshold = series.mean() + threshold * series.std()
        
        # Find burst periods
        bursts = []
        in_burst = False
        burst_start = None
        
        for date, value in series.items():
            if value >= burst_threshold and not in_burst:
                burst_start = date
                in_burst = True
            elif value < burst_threshold and in_burst:
                bursts.append({
                    'start': burst_start,
                    'end': date,
                    'peak_value': series[burst_start:date].max(),
                    'duration_days': (date - burst_start).days
                })
                in_burst = False
        
        # Handle ongoing burst
        if in_burst:
            bursts.append({
                'start': burst_start,
                'end': series.index[-1],
                'peak_value': series[burst_start:].max(),
                'duration_days': (series.index[-1] - burst_start).days
            })
        
        return bursts