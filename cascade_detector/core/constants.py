"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
constants.py

MAIN OBJECTIVE:
---------------
This script defines global constants used throughout the cascade detection framework, including 
frame definitions, detection thresholds, and analysis parameters.

Dependencies:
-------------
None

MAIN FEATURES:
--------------
1) Frame definitions and color mappings for visualization
2) Detection thresholds and statistical parameters
3) Time window configurations for analysis
4) Messenger type definitions
5) Database column mappings

Author:
-------
Antoine Lemor
"""

# Main frames to analyze (8 frames)
FRAMES = ["Cult", "Eco", "Envt", "Pbh", "Just", "Pol", "Sci", "Secu"]

# Frame colors for visualization
FRAME_COLORS = {
    "Cult": "#E64B35",
    "Eco": "#4DBBD5",
    "Envt": "#00A087",
    "Pbh": "#3C5488",
    "Just": "#F39B7F",
    "Pol": "#8491B4",
    "Sci": "#91D1C2",
    "Secu": "#B09C85",
}

# Messenger detection hierarchy
# Messenger_Detection = 1 when sentence contains a messenger (global indicator)
# Messenger_X_SUB = 1 only when Messenger_Detection = 1 (specific type)
MESSENGER_MAIN = 'Messenger_Detection'  # Global messenger indicator

# Messenger sub-types (only active when MESSENGER_MAIN = 1)
MESSENGERS = [
    'Messenger_1_SUB',
    'Messenger_2_SUB', 
    'Messenger_3_SUB',
    'Messenger_4_SUB',
    'Messenger_5_SUB',
    'Messenger_6_SUB',
    'Messenger_7_SUB',
    'Messenger_8_SUB',
    'Messenger_9_SUB'
]

# Messenger type mapping
# These can be cross-referenced with NER entities for precise identification
MESSENGER_TYPES = {
    'Messenger_1_SUB': 'health_expertise',
    'Messenger_2_SUB': 'economic_expertise',
    'Messenger_3_SUB': 'security_expertise',
    'Messenger_4_SUB': 'law_expertise',
    'Messenger_5_SUB': 'culture_expertise',
    'Messenger_6_SUB': 'hard_science',
    'Messenger_7_SUB': 'social_science',
    'Messenger_8_SUB': 'activist',
    'Messenger_9_SUB': 'public_official'
}

# Entity types from NER
ENTITY_TYPES = ['PER', 'ORG', 'LOC']

# Emotion/Sentiment columns
EMOTION_COLUMNS = {
    'Emotion:_Positive': 'positive',
    'Emotion:_Neutral': 'neutral', 
    'Emotion:_Negative': 'negative'
}

# Database constants
DB_TABLE = "CCF_processed_data"
DATE_FORMAT_DB = "%m-%d-%Y"  # Format in database
DATE_FORMAT_ISO = "%Y-%m-%d"  # ISO format for processing

# Performance constants (Optimized for M4 Max)
DEFAULT_BATCH_SIZE = 500000  # Larger batches for 128GB RAM
DEFAULT_N_WORKERS = 16  # Optimal for M4 Max (proven effective)
DEFAULT_CACHE_SIZE_GB = 100  # Generous cache for large datasets

# Detection thresholds (data-driven, these are just fallbacks)
MIN_CASCADE_DAYS = 7
MAX_CASCADE_DAYS = 180
MIN_ARTICLES = 10
MIN_JOURNALISTS = 3
MIN_MEDIA = 2

# Window parameters
WINDOW_MIN_DAYS = 7
WINDOW_MAX_DAYS = 90
WINDOW_OVERLAP = 0.5

# Scoring weights
DIMENSION_WEIGHT = 0.20  # 20% per dimension
SUBINDEX_WEIGHT = 0.05   # 5% per sub-index