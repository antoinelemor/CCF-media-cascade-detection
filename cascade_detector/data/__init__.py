"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
__init__.py (data module)

MAIN OBJECTIVE:
---------------
This script initializes the data module of the cascade detector, providing access to database
connectivity and data processing components.

Dependencies:
-------------
- cascade_detector.data.connector
- cascade_detector.data.processor

MAIN FEATURES:
--------------
1) Exports DatabaseConnector for PostgreSQL connections
2) Exports DataProcessor for data cleaning and preparation
3) Provides clean API for data access components

Author:
-------
Antoine Lemor
"""

from cascade_detector.data.connector import DatabaseConnector
from cascade_detector.data.processor import DataProcessor

__all__ = [
    'DatabaseConnector',
    'DataProcessor'
]