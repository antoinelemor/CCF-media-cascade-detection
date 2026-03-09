"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
constants.py

MAIN OBJECTIVE:
---------------
Global constants: frame definitions, column mappings, detection thresholds.
No system resource detection, no GPU/parallel constants.

Author:
-------
Antoine Lemor
"""

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

DB_NAME = "CCF_Database_texts"
DB_NAME_TEXTS = DB_NAME
DB_TABLE = "CCF_processed_data"

DATE_FORMAT_DB = "%Y-%m-%d"
DATE_FORMAT_ISO = "%Y-%m-%d"

# =============================================================================
# FRAME DEFINITIONS
# =============================================================================

FRAMES = ["Cult", "Eco", "Envt", "Pbh", "Just", "Pol", "Sci", "Secu"]

FRAME_COLUMNS = {
    "Cult": "cultural_frame",
    "Eco": "economic_frame",
    "Envt": "environmental_frame",
    "Pbh": "health_frame",
    "Just": "justice_frame",
    "Pol": "political_frame",
    "Sci": "scientific_frame",
    "Secu": "security_frame",
}

FRAME_COLUMNS_REVERSE = {v: k for k, v in FRAME_COLUMNS.items()}

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

# =============================================================================
# MESSENGER DEFINITIONS
# =============================================================================

MESSENGER_MAIN = 'messenger'

MESSENGERS = [
    'msg_health',
    'msg_economic',
    'msg_security',
    'msg_legal',
    'msg_cultural',
    'msg_scientist',
    'msg_social',
    'msg_activist',
    'msg_official'
]

MESSENGER_TYPES = {
    'msg_health': 'health_expertise',
    'msg_economic': 'economic_expertise',
    'msg_security': 'security_expertise',
    'msg_legal': 'law_expertise',
    'msg_cultural': 'culture_expertise',
    'msg_scientist': 'hard_science',
    'msg_social': 'social_science',
    'msg_activist': 'activist',
    'msg_official': 'public_official'
}

MESSENGER_LEGACY_MAPPING = {
    'Messenger_Detection': 'messenger',
    'Messenger_1_SUB': 'msg_health',
    'Messenger_2_SUB': 'msg_economic',
    'Messenger_3_SUB': 'msg_security',
    'Messenger_4_SUB': 'msg_legal',
    'Messenger_5_SUB': 'msg_cultural',
    'Messenger_6_SUB': 'msg_scientist',
    'Messenger_7_SUB': 'msg_social',
    'Messenger_8_SUB': 'msg_activist',
    'Messenger_9_SUB': 'msg_official'
}

# =============================================================================
# EMOTION/TONE DEFINITIONS
# =============================================================================

EMOTION_COLUMNS = {
    'tone_positive': 'positive',
    'tone_neutral': 'neutral',
    'tone_negative': 'negative'
}

EMOTION_COLUMNS_REVERSE = {v: k for k, v in EMOTION_COLUMNS.items()}

EMOTION_LEGACY_MAPPING = {
    'Emotion:_Positive': 'tone_positive',
    'Emotion:_Neutral': 'tone_neutral',
    'Emotion:_Negative': 'tone_negative'
}

# =============================================================================
# OTHER DETECTION COLUMNS
# =============================================================================

LOCATION_COLUMN = 'canada'
URGENCY_COLUMN = 'urgency'

EVENT_MAIN = 'event'
EVENT_COLUMNS = [
    'evt_weather',
    'evt_meeting',
    'evt_publication',
    'evt_election',
    'evt_policy',
    'evt_judiciary',
    'evt_cultural',
    'evt_protest'
]

EVENT_TYPES = {
    'evt_weather': 'natural_event',
    'evt_meeting': 'institutional_event',
    'evt_publication': 'knowledge_event',
    'evt_election': 'political_event',
    'evt_policy': 'policy_event',
    'evt_judiciary': 'legal_event',
    'evt_cultural': 'cultural_event',
    'evt_protest': 'social_mobilization'
}

SOLUTION_MAIN = 'solution'
SOLUTION_COLUMNS = ['sol_mitigation', 'sol_adaptation']

SOLUTION_TYPES = {
    'sol_mitigation': 'mitigation',
    'sol_adaptation': 'adaptation'
}

# =============================================================================
# ENTITY DEFINITIONS
# =============================================================================

ENTITY_TYPES = ['PER', 'ORG', 'LOC']
NER_COLUMN = 'ner_entities'

# =============================================================================
# METADATA COLUMNS
# =============================================================================

METADATA_COLUMNS = [
    'doc_id',
    'news_type',
    'title',
    'author',
    'media',
    'words_count',
    'date',
    'language',
    'page_number',
    'sentence_id'
]

# =============================================================================
# DETECTION THRESHOLDS
# =============================================================================

MIN_CASCADE_DAYS = 7
MAX_CASCADE_DAYS = 180
MIN_ARTICLES = 10
MIN_JOURNALISTS = 3
MIN_MEDIA = 2

# Cascade classification thresholds
CASCADE_THRESHOLDS = {
    'strong_cascade': 0.65,
    'moderate_cascade': 0.40,
    'weak_cascade': 0.25,
    'not_cascade': 0.0,
}

# =============================================================================
# TITLE EMBEDDING CONSTANTS
# =============================================================================

TITLE_SENTENCE_ID = 0      # convention: sentence_id=0 = title embedding
TITLE_WEIGHT = 0.30        # α — weight of title vs body phrases in clustering & belonging

# =============================================================================
# EVENT CLUSTER CONSTANTS (Phase 3)
# =============================================================================

# Distance weights (5 dimensions, sum to 1.0)
EVENT_CLUSTER_TEMPORAL_WEIGHT = 0.25
EVENT_CLUSTER_SEMANTIC_WEIGHT = 0.20
EVENT_CLUSTER_ENTITY_WEIGHT = 0.15
EVENT_CLUSTER_ARTICLE_WEIGHT = 0.30    # article overlap (Jaccard on seed_doc_ids)
EVENT_CLUSTER_TYPE_WEIGHT = 0.10       # event type: 0=same, 1=different
EVENT_CLUSTER_TEMPORAL_SCALE = 14.0    # days for temporal distance decay (unified with Phase 4)
EVENT_CLUSTER_MIN_ENTITY_CITATIONS = 3 # min citations per entity within occurrence

# Post-processing: title+temporal connectivity (Step 6b)
EVENT_CLUSTER_TITLE_SIM_THRESHOLD = 0.50   # min cosine sim between title centroids
EVENT_CLUSTER_MAX_GAP_DAYS = 30            # max peak gap for title-only connectivity

# Strength weights (5 dimensions, sum to 1.0)
EVENT_CLUSTER_STRENGTH_MASS_WEIGHT = 0.20
EVENT_CLUSTER_STRENGTH_COVERAGE_WEIGHT = 0.25
EVENT_CLUSTER_STRENGTH_INTENSITY_WEIGHT = 0.20
EVENT_CLUSTER_STRENGTH_COHERENCE_WEIGHT = 0.15
EVENT_CLUSTER_STRENGTH_DIVERSITY_WEIGHT = 0.20

# =============================================================================
# EVENT OCCURRENCE CONSTANTS (Phase 2/4)
# =============================================================================

SEED_PERCENTILE = 50               # P50 of seed_score among articles with evt_*_mean > 0
SEED_WEIGHT_TYPE = 0.6             # weight of evt_*_mean in composite seed score
SEED_WEIGHT_GLOBAL = 0.4           # weight of event_mean in composite seed score
MIN_CLUSTER_SIZE = 2               # minimum cluster size (confidence qualifies)

# Phase 4: 4D distance weights (article → cluster assignment)
PHASE4_TEMPORAL_WEIGHT = 0.25
PHASE4_SEMANTIC_WEIGHT = 0.35
PHASE4_ENTITY_WEIGHT = 0.15
PHASE4_SIGNAL_WEIGHT = 0.25
PHASE4_N_ITERATIONS = 2
PHASE4_TEMPORAL_SCALE = 14.0      # days for temporal distance decay

# Phase 2 compound distance weights (3D, sum to 1.0)
PHASE2_SEMANTIC_WEIGHT = 0.50     # was (1 - TEMPORAL_WEIGHT) = 0.70
PHASE2_TEMPORAL_WEIGHT = 0.30     # unchanged
PHASE2_ENTITY_WEIGHT = 0.20      # entity Jaccard dimension

# Seed dominant ratio: seed for evt_X only if evt_X_mean >= ratio × max(evt_*_mean)
SEED_DOMINANT_RATIO = 0.5
