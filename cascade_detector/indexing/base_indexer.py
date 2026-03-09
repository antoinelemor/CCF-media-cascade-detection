"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
base_indexer.py

MAIN OBJECTIVE:
---------------
This script provides the abstract base class for all indexing strategies in the cascade detection
framework, defining the common interface and shared functionality for index creation, querying,
and persistence.

Dependencies:
-------------
- abc
- typing
- pandas
- pickle
- json
- pathlib
- logging

MAIN FEATURES:
--------------
1) Abstract interface for index building, updating, and querying
2) Multiple serialization formats (pickle, JSON, parquet)
3) Index persistence and loading from disk
4) Statistics and memory usage tracking
5) Data validation and required column checking

Author:
-------
Antoine Lemor
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
import pickle
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AbstractIndexer(ABC):
    """
    Abstract base class for all indexing strategies.
    Provides common functionality and interface.
    """
    
    def __init__(self, name: str = "indexer"):
        """
        Initialize indexer.
        
        Args:
            name: Name of the indexer for logging
        """
        self.name = name
        self.index = {}
        self.metadata = {
            'created': None,
            'updated': None,
            'n_entries': 0,
            'version': '1.0.0'
        }
        
    @abstractmethod
    def build_index(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Build index from data.
        
        Args:
            data: DataFrame with cascade data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing the index
        """
        pass
    
    @abstractmethod
    def update_index(self, new_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Update existing index with new data.
        
        Args:
            new_data: New data to add to index
            **kwargs: Additional parameters
            
        Returns:
            Updated index
        """
        pass
    
    @abstractmethod
    def query_index(self, criteria: Dict[str, Any]) -> List[Any]:
        """
        Query the index based on criteria.
        
        Args:
            criteria: Query criteria
            
        Returns:
            List of matching entries
        """
        pass
    
    def save_index(self, path: str, format: str = 'pickle') -> None:
        """
        Save index to disk.
        
        Args:
            path: Path to save the index
            format: Format ('pickle', 'json', 'parquet')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump({
                    'index': self.index,
                    'metadata': self.metadata
                }, f)
        elif format == 'json':
            # Convert to JSON-serializable format
            json_data = self._to_json_serializable()
            with open(path, 'w') as f:
                json.dump(json_data, f, indent=2)
        elif format == 'parquet':
            # Convert to DataFrame and save as parquet
            df = self._to_dataframe()
            df.to_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"{self.name} index saved to {path} ({format})")
    
    def load_index(self, path: str, format: str = 'pickle') -> Dict[str, Any]:
        """
        Load index from disk.
        
        Args:
            path: Path to load the index from
            format: Format of the saved index
            
        Returns:
            Loaded index
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        
        if format == 'pickle':
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.index = data['index']
                self.metadata = data['metadata']
        elif format == 'json':
            with open(path, 'r') as f:
                data = json.load(f)
                self.index = self._from_json_serializable(data['index'])
                self.metadata = data['metadata']
        elif format == 'parquet':
            df = pd.read_parquet(path)
            self.index = self._from_dataframe(df)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"{self.name} index loaded from {path}")
        return self.index
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'name': self.name,
            'n_entries': self.metadata['n_entries'],
            'created': self.metadata['created'],
            'updated': self.metadata['updated'],
            'memory_size_mb': self._get_memory_size() / (1024 * 1024)
        }
    
    def clear_index(self) -> None:
        """Clear the index."""
        self.index = {}
        self.metadata['n_entries'] = 0
        logger.info(f"{self.name} index cleared")
    
    def _get_memory_size(self) -> int:
        """Get approximate memory size of index in bytes."""
        import sys
        return sys.getsizeof(self.index)
    
    def _to_json_serializable(self) -> Dict:
        """Convert index to JSON-serializable format."""
        # Override in subclasses for custom serialization
        return {
            'index': self.index,
            'metadata': self.metadata
        }
    
    def _from_json_serializable(self, data: Dict) -> Dict:
        """Convert from JSON-serializable format to index."""
        # Override in subclasses for custom deserialization
        return data
    
    def _to_dataframe(self) -> pd.DataFrame:
        """Convert index to DataFrame."""
        # Override in subclasses
        raise NotImplementedError("DataFrame conversion not implemented")
    
    def _from_dataframe(self, df: pd.DataFrame) -> Dict:
        """Convert DataFrame to index."""
        # Override in subclasses
        raise NotImplementedError("DataFrame loading not implemented")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data has required columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if valid
        """
        required_cols = self.get_required_columns()
        missing = set(required_cols) - set(data.columns)
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return True
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Get list of required columns for this indexer."""
        pass