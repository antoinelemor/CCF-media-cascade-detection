"""
Unit tests for Phase 1: Infrastructure and Indexing.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cascade_detector.core.config import DetectorConfig
from cascade_detector.core.models import (
    TimeWindow, IndexEntry, CascadeCandidate, 
    SubIndex, Dimension, MediaCascade
)
from cascade_detector.core.constants import FRAMES, MESSENGERS
from cascade_detector.indexing import (
    TemporalIndexer, EntityIndexer, SourceIndexer, 
    FrameIndexer, IndexManager
)
from cascade_detector.data import DatabaseConnector, DataProcessor


class TestConfig:
    """Test configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = DetectorConfig()
        
        # Check defaults
        assert config.db_host == "localhost"
        assert config.db_name == "CCF"
        assert config.n_workers == 64
        assert config.dimension_weight == 0.20
        assert config.subindex_weight == 0.05
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = DetectorConfig()
        
        # Should validate successfully
        assert config.validate()
        
        # Test invalid weights
        config.dimension_weight = 0.25  # 5 * 0.25 = 1.25 != 1.0
        with pytest.raises(AssertionError):
            config.validate()
    
    def test_config_to_dict(self):
        """Test configuration export."""
        config = DetectorConfig()
        config_dict = config.to_dict()
        
        assert 'database' in config_dict
        assert 'frames' in config_dict
        assert 'performance' in config_dict
        assert config_dict['database']['host'] == "localhost"


class TestModels:
    """Test data models."""
    
    def test_time_window(self):
        """Test TimeWindow model."""
        start = pd.Timestamp('2024-01-01')
        end = pd.Timestamp('2024-01-31')
        
        window = TimeWindow(
            start=start,
            end=end,
            size_days=30,
            variance=0.5,
            data_points=100
        )
        
        # Test properties
        assert window.duration == pd.Timedelta(days=30)
        assert window.contains(pd.Timestamp('2024-01-15'))
        assert not window.contains(pd.Timestamp('2024-02-01'))
    
    def test_subindex_validation(self):
        """Test SubIndex validation."""
        # Valid sub-index
        si = SubIndex(
            name="test",
            value=0.5,
            weight=0.05,
            components={'a': 0.3, 'b': 0.7}
        )
        assert si.validate()
        assert si.weighted_value == 0.025  # 0.5 * 0.05
        
        # Invalid value
        si_invalid = SubIndex(name="test", value=1.5, weight=0.05)
        with pytest.raises(AssertionError):
            si_invalid.validate()
    
    def test_dimension_validation(self):
        """Test Dimension validation."""
        # Create 4 sub-indices (required)
        sub_indices = [
            SubIndex(f"si_{i}", 0.25, 0.05) 
            for i in range(4)
        ]
        
        dim = Dimension(
            name="test_dimension",
            score=0.0,
            weight=0.20,
            sub_indices=sub_indices
        )
        
        # Should validate
        assert dim.validate()
        
        # Calculate score
        score = dim.calculate_score()
        assert score == 0.25  # Average of 4 * 0.25
        
        # Test with wrong number of sub-indices
        dim_invalid = Dimension(
            name="test",
            score=0.0,
            weight=0.20,
            sub_indices=sub_indices[:3]  # Only 3
        )
        with pytest.raises(AssertionError):
            dim_invalid.validate()


class TestIndexing:
    """Test indexing functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        data = []
        for i, date in enumerate(dates):
            for j in range(10):  # 10 sentences per day
                data.append({
                    'date': date.strftime('%m-%d-%Y'),  # MM-DD-YYYY format
                    'doc_id': f"doc_{i}_{j//3}",  # ~3 sentences per article
                    'sentence_id': j,
                    'media': f"media_{i % 5}",
                    'author': f"author_{i % 10}",
                    'Cult_Detection': np.random.choice([0, 1], p=[0.8, 0.2]),
                    'Eco_Detection': np.random.choice([0, 1], p=[0.9, 0.1]),
                    'Pol_Detection': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'Messenger_1_SUB': np.random.choice([0, 1], p=[0.95, 0.05]),
                    'NER_entities': '{"PER": ["John Doe"], "ORG": ["UN"], "LOC": ["Paris"]}'
                })
        
        return pd.DataFrame(data)
    
    def test_temporal_indexer(self, sample_data):
        """Test temporal indexer."""
        indexer = TemporalIndexer()
        index = indexer.build_index(sample_data)
        
        # Check structure
        assert 'Cult' in index
        assert 'Pol' in index
        
        # Check time series
        cult_series = index['Cult']['daily_series']
        assert isinstance(cult_series, pd.Series)
        assert len(cult_series) == 100  # 100 days
        
        # Check statistics
        assert 'statistics' in index['Cult']
        assert 'mean_daily_count' in index['Cult']['statistics']
    
    def test_entity_indexer(self, sample_data):
        """Test entity indexer."""
        indexer = EntityIndexer()
        index = indexer.build_index(sample_data)
        
        # Check entities were extracted
        assert len(index) > 0
        assert 'PER:John Doe' in index
        assert 'ORG:UN' in index
        
        # Check entity structure
        entity = index['PER:John Doe']
        assert entity['type'] == 'PER'
        assert entity['name'] == 'John Doe'
        assert entity['count'] > 0
        assert 'authority_score' in entity
    
    def test_source_indexer(self, sample_data):
        """Test source indexer."""
        indexer = SourceIndexer()
        index = indexer.build_index(sample_data)
        
        # Check structure
        assert 'article_profiles' in index
        assert 'journalist_profiles' in index
        assert 'media_profiles' in index
        
        # Check profiles exist
        assert len(index['article_profiles']) > 0
        assert len(index['journalist_profiles']) > 0
    
    def test_frame_indexer(self, sample_data):
        """Test frame indexer."""
        indexer = FrameIndexer()
        index = indexer.build_index(sample_data)
        
        # Check structure
        assert 'cooccurrence_matrix' in index
        assert 'article_frames' in index
        assert 'frame_statistics' in index
        
        # Check co-occurrence matrix
        matrix = index['cooccurrence_matrix']
        assert matrix.shape == (8, 8)  # 8 frames
    
    def test_index_manager(self, sample_data):
        """Test index manager."""
        config = DetectorConfig(n_workers=2)  # Reduce for testing
        manager = IndexManager(config)
        
        # Build all indices
        indices = manager.build_all_indices(sample_data, parallel=False)
        
        # Check all indices built
        assert 'temporal' in indices
        assert 'entities' in indices
        assert 'sources' in indices
        assert 'frames' in indices
        
        # Test query
        results = manager.query('temporal', {
            'frame': 'Pol',
            'min_activity': 1
        })
        assert isinstance(results, list)


class TestDataProcessing:
    """Test data processing."""
    
    @pytest.fixture
    def raw_data(self):
        """Create raw data similar to database format."""
        data = {
            'date': ['01-15-2024', '01-16-2024', '01-17-2024'],
            'doc_id': ['doc1', 'doc1', 'doc2'],
            'sentence_id': [1, 2, 1],
            'media': ['CNN', 'CNN', 'BBC'],
            'author': ['John Doe, Staff', 'John Doe, Staff', 'Jane Smith'],
            'Cult_Detection': ['1', '0', '1'],
            'Eco_Detection': ['0', '1', '0'],
            'Messenger_1_SUB': ['1', '0', '1']
        }
        return pd.DataFrame(data)
    
    def test_data_processor(self, raw_data):
        """Test data processor."""
        processor = DataProcessor()
        
        # Process data
        processed = processor.process_frame_data(raw_data)
        
        # Check date conversion
        assert 'date_converted' in processed.columns
        assert processed['date_converted'].dtype == 'datetime64[ns]'
        
        # Check frame cleaning
        assert processed['Cult_Detection'].dtype == 'int64'
        assert processed['Eco_Detection'].dtype == 'int64'
        
        # Check derived columns
        assert 'n_frames' in processed.columns
        assert 'year' in processed.columns
        assert 'author_clean' in processed.columns
        
        # Check author cleaning
        assert processed['author_clean'].iloc[0] == 'John Doe'
    
    def test_article_aggregation(self, raw_data):
        """Test article-level aggregation."""
        processor = DataProcessor()
        processed = processor.process_frame_data(raw_data)
        
        # Aggregate by article
        articles = processor.aggregate_by_article(processed)
        
        # Check aggregation
        assert len(articles) == 2  # 2 unique doc_ids
        assert 'n_sentences' in articles.columns
        assert articles['n_sentences'].iloc[0] == 2  # doc1 has 2 sentences


class TestIntegration:
    """Integration tests for Phase 1."""
    
    def test_full_pipeline(self):
        """Test full Phase 1 pipeline."""
        # Create config
        config = DetectorConfig(n_workers=2)
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        data = []
        for date in dates:
            for i in range(5):
                data.append({
                    'date': date.strftime('%m-%d-%Y'),
                    'doc_id': f"doc_{date.strftime('%Y%m%d')}_{i}",
                    'sentence_id': i,
                    'media': f"media_{i % 3}",
                    'author': f"author_{i % 5}",
                    'Cult_Detection': np.random.choice([0, 1]),
                    'Pol_Detection': np.random.choice([0, 1]),
                    'Messenger_1_SUB': np.random.choice([0, 1]),
                    'NER_entities': '{"PER": ["Test Person"]}'
                })
        df = pd.DataFrame(data)
        
        # Process data
        processor = DataProcessor(config)
        processed = processor.process_frame_data(df)
        
        # Build indices
        manager = IndexManager(config)
        indices = manager.build_all_indices(processed, parallel=False)
        
        # Verify all components work
        assert len(processed) > 0
        assert all(key in indices for key in ['temporal', 'entities', 'sources', 'frames'])
        
        # Test statistics
        stats = manager.get_statistics()
        assert 'metadata' in stats
        assert 'indices' in stats


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])