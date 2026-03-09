"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
exceptions.py

MAIN OBJECTIVE:
---------------
This script defines custom exception classes for the cascade detection framework, providing 
structured error handling for different components of the system.

Dependencies:
-------------
None

MAIN FEATURES:
--------------
1) Base CascadeDetectorError exception class
2) Specialized exceptions for configuration, database, indexing, and detection errors
3) Validation and data insufficiency exceptions
4) Dimension calculation error handling

Author:
-------
Antoine Lemor
"""


class CascadeDetectorError(Exception):
    """Base exception for cascade detector."""
    pass


class ConfigurationError(CascadeDetectorError):
    """Configuration-related errors."""
    pass


class DatabaseConnectionError(CascadeDetectorError):
    """Database connection errors."""
    pass


class IndexingError(CascadeDetectorError):
    """Indexing-related errors."""
    pass


class DetectionError(CascadeDetectorError):
    """Detection-related errors."""
    pass


class ValidationError(CascadeDetectorError):
    """Validation errors."""
    pass


class InsufficientDataError(CascadeDetectorError):
    """Not enough data for analysis."""
    pass


class DimensionError(CascadeDetectorError):
    """Dimension calculation errors."""
    pass