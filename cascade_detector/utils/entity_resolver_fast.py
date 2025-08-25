"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
entity_resolver_fast.py

MAIN OBJECTIVE:
---------------
This script provides ultra-fast entity resolution using blocking and parallel processing, optimized
for Apple M4 Max architecture to intelligently merge entity name variations.

Dependencies:
-------------
- re
- logging
- typing
- collections
- numpy
- difflib
- pandas
- sklearn
- networkx
- concurrent.futures
- multiprocessing
- functools
- pickle

MAIN FEATURES:
--------------
1) Blocking strategies to reduce comparison space
2) Parallel similarity computation across multiple cores
3) Context-aware entity matching
4) Graph-based transitive closure for entity merging
5) Optimized for M4 Max with intelligent worker allocation

Author:
-------
Antoine Lemor
"""

import re
import logging
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import numpy as np
from difflib import SequenceMatcher
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import pickle

logger = logging.getLogger(__name__)


class FastEntityResolver:
    """
    Ultra-fast entity resolution using blocking and parallel processing.
    Optimized for M4 Ultra Max (128 cores, 128GB RAM).
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.80,
                 context_weight: float = 0.3,
                 min_occurrences: int = 2,
                 n_workers: int = 14):  # M4 Max cores
        """
        Initialize fast entity resolver.
        
        Args:
            similarity_threshold: Minimum similarity for merging (0-1)
            context_weight: Weight for context similarity (0-1)
            min_occurrences: Minimum occurrences to consider entity
            n_workers: Number of parallel workers
        """
        self.similarity_threshold = similarity_threshold
        self.context_weight = context_weight
        self.min_occurrences = min_occurrences
        # M4 Max optimization: don't exceed actual cores for CPU-bound tasks
        self.n_workers = min(n_workers, mp.cpu_count())
        
        # Precompiled patterns for speed
        self.title_pattern = re.compile(r'^(M\.|Mme|Dr|Prof\.|Me|Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+', re.IGNORECASE)
        self.suffix_pattern = re.compile(r'(,\s*(Reuters|AP|Staff|Ottawa|Toronto|Montreal)).*$', re.IGNORECASE)
        
        logger.info(f"FastEntityResolver initialized with {self.n_workers} workers")
    
    def _clean_entity_name(self, name: str) -> str:
        """
        Clean entity names to remove malformed punctuation.
        E.g., "Stephen Harper," -> "Stephen Harper"
              "Charles, Trump" -> "Charles Trump"
        """
        # Remove trailing punctuation
        name = re.sub(r'[,.\-;:!?]+$', '', name).strip()
        
        # Fix internal punctuation issues (comma between names)
        # But keep legitimate punctuation like "J." for initials
        if ',' in name:
            # Check if it's a malformed name like "Charles, Trump"
            parts = name.split(',')
            if len(parts) == 2:
                # If both parts look like names, join them
                part1, part2 = parts[0].strip(), parts[1].strip()
                if part1 and part2:
                    # Check if this looks like "Last, First" format
                    if part2 and part2[0].isupper() and not (part1 and part1[0].islower()):
                        # Likely "Charles, Trump" format - join them
                        name = f"{part1} {part2}"
        
        # Remove leading punctuation like ". Trudeau"
        name = re.sub(r'^[,.\-;:!?]+', '', name).strip()
        
        return name
    
    def resolve_entities(self, 
                         entity_index: Dict[str, Dict],
                         use_context: bool = True,
                         use_blocking: bool = True) -> Dict[str, Dict]:
        """
        Resolve entities using fast blocking and parallel processing.
        
        Args:
            entity_index: Entity index from EntityIndexer
            use_context: Whether to use context-based resolution
            use_blocking: Use blocking to reduce comparisons
            
        Returns:
            Resolved entity index with merged entities
        """
        logger.info(f"Fast entity resolution for {len(entity_index)} entities...")
        start_time = pd.Timestamp.now()
        
        # Clean entity names first
        cleaned_index = {}
        for key, data in entity_index.items():
            if isinstance(data, dict) and 'name' in data:
                clean_name = self._clean_entity_name(data['name'])
                data = data.copy()
                data['name'] = clean_name
                # Update key if it contains the entity name
                if ':' in key:
                    prefix, _ = key.split(':', 1)
                    new_key = f"{prefix}:{clean_name}"
                    cleaned_index[new_key] = data
                else:
                    cleaned_index[key] = data
            else:
                cleaned_index[key] = data
        
        # Filter entities with minimum occurrences
        eligible_entities = {
            k: v for k, v in cleaned_index.items()
            if v.get('occurrences', 0) >= self.min_occurrences
        }
        
        logger.info(f"Processing {len(eligible_entities)} entities with >= {self.min_occurrences} occurrences")
        
        if use_blocking:
            # Use blocking to dramatically reduce comparisons
            resolved_index = self._resolve_with_blocking(eligible_entities, cleaned_index)
        else:
            # Fall back to full resolution (slower)
            resolved_index = self._resolve_full(eligible_entities, cleaned_index)
        
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        reduction = (len(cleaned_index) - len(resolved_index)) / len(cleaned_index) * 100
        
        logger.info(f"Entity resolution complete in {elapsed:.1f}s: "
                   f"{len(cleaned_index)} → {len(resolved_index)} (reduced by {reduction:.1f}%)")
        
        return resolved_index
    
    def _resolve_with_blocking(self, 
                              eligible_entities: Dict,
                              full_index: Dict) -> Dict:
        """
        Resolve using blocking to reduce comparisons from O(n²) to O(n*k).
        """
        # Create blocks based on entity characteristics
        blocks = self._create_blocks(eligible_entities)
        
        logger.info(f"Created {len(blocks)} blocks for parallel processing")
        
        # Process blocks in parallel
        all_clusters = []
        
        # Split blocks into chunks for parallel processing
        block_items = list(blocks.items())
        chunk_size = max(1, len(block_items) // self.n_workers)
        block_chunks = [
            dict(block_items[i:i + chunk_size])
            for i in range(0, len(block_items), chunk_size)
        ]
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for chunk in block_chunks:
                future = executor.submit(self._process_block_chunk, chunk)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    chunk_clusters = future.result()
                    all_clusters.extend(chunk_clusters)
                except Exception as e:
                    logger.error(f"Error processing block chunk: {e}")
        
        # Merge clusters across blocks (handle entities in multiple blocks)
        merged_clusters = self._merge_cross_block_clusters(all_clusters)
        
        # Create resolved index
        resolved_index = self._create_resolved_index(merged_clusters, full_index)
        
        return resolved_index
    
    def _create_blocks(self, entities: Dict, debug: bool = False) -> Dict[str, List]:
        """
        Create blocks of potentially matching entities.
        Uses multiple blocking keys for better recall.
        """
        blocks = defaultdict(list)
        
        for entity_key, entity_data in entities.items():
            entity_type = entity_data.get('type', 'UNK')
            name = entity_data.get('name', entity_key.split(':', 1)[1] if ':' in entity_key else entity_key)
            
            # Normalize name for blocking
            normalized = self._normalize_for_blocking(name)
            
            if not normalized:
                continue
            
            # Create multiple blocking keys
            blocking_keys = []
            
            # 1. For names - use last word as primary key (catches "Trudeau" with "Justin Trudeau")
            parts = normalized.split()
            if parts:
                # IMPORTANT: Use last word for blocking (last names are key)
                if len(parts) >= 1:
                    # Last word is the key blocking element
                    last_word = parts[-1]
                    if len(last_word) >= 3:  # Skip very short words
                        key = f"{entity_type}:lastname:{last_word}"
                        blocking_keys.append(key)
                        
                        # Debug specific cases (disabled for production)
                        # if 'trudeau' in last_word.lower():
                        #     print(f"[BLOCKING DEBUG] Trudeau variant: '{name}' (normalized: '{normalized}') -> key '{key}'", flush=True)
                
                # Also add first word for single-word entities
                if len(parts) == 1:
                    key = f"{entity_type}:single:{parts[0]}"
                    blocking_keys.append(key)
                
                # For multi-word, also create first-last combo
                if len(parts) >= 2:
                    key = f"{entity_type}:fl:{parts[0][0]}:{parts[-1]}"
                    blocking_keys.append(key)
            
            # 2. Soundex-like key for phonetic matching
            phonetic_key = self._get_phonetic_key(normalized)
            if phonetic_key:
                blocking_keys.append(f"{entity_type}:sound:{phonetic_key}")
            
            # Add entity to all relevant blocks
            for key in blocking_keys:
                blocks[key].append((entity_key, entity_data))
        
        # Filter out blocks with only one entity
        blocks = {k: v for k, v in blocks.items() if len(v) > 1}
        
        return dict(blocks)
    
    def _normalize_for_blocking(self, name: str) -> str:
        """Fast normalization for blocking."""
        # Remove titles and suffixes
        normalized = self.title_pattern.sub('', name)
        normalized = self.suffix_pattern.sub('', normalized)
        
        # Basic cleaning
        normalized = re.sub(r'[^\w\s-]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.lower().strip()
    
    def _get_phonetic_key(self, name: str) -> str:
        """
        Simple phonetic key generation (simplified Soundex).
        """
        if not name:
            return ""
        
        # Keep first letter and encode rest
        parts = name.split()
        if not parts:
            return ""
        
        # Use first and last word for key
        if len(parts) >= 2:
            first = parts[0][:3] if len(parts[0]) >= 3 else parts[0]
            last = parts[-1][:3] if len(parts[-1]) >= 3 else parts[-1]
            return f"{first}{last}"
        else:
            return parts[0][:4] if len(parts[0]) >= 4 else parts[0]
    
    def _process_block_chunk(self, block_chunk: Dict) -> List[Set]:
        """
        Process a chunk of blocks in parallel.
        """
        all_clusters = []
        
        for block_key, entities in block_chunk.items():
            if len(entities) <= 1:
                continue
            
            # Build similarity matrix for this block only
            n = len(entities)
            
            # Skip very large blocks (likely too generic)
            if n > 100:
                logger.warning(f"Skipping large block {block_key} with {n} entities")
                continue
            
            similarity_matrix = np.zeros((n, n))
            
            # Calculate similarities within block
            for i in range(n):
                for j in range(i + 1, n):
                    sim = self._fast_similarity(entities[i], entities[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
            
            np.fill_diagonal(similarity_matrix, 1.0)
            
            # Find clusters in this block
            clusters = self._find_clusters_fast(similarity_matrix)
            
            # Convert to entity keys
            for cluster in clusters:
                if len(cluster) > 1:
                    cluster_keys = {entities[i][0] for i in cluster}
                    all_clusters.append(cluster_keys)
        
        return all_clusters
    
    def _fast_similarity(self, entity1: Tuple, entity2: Tuple) -> float:
        """
        Intelligent similarity calculation based on name analysis and co-occurrence.
        KEY INSIGHT: If "Trudeau" and "Justin Trudeau" appear in the same article,
        they're likely the SAME person (journalist using short/long form).
        """
        key1, data1 = entity1
        key2, data2 = entity2
        
        # Different types never match
        if data1.get('type') != data2.get('type'):
            return 0.0
        
        name1 = data1.get('name', key1.split(':', 1)[1] if ':' in key1 else key1)
        name2 = data2.get('name', key2.split(':', 1)[1] if ':' in key2 else key2)
        
        # Debug for Ford cases (disabled for production)
        # if 'ford' in name1.lower() and 'ford' in name2.lower():
        #     print(f"[SIMILARITY DEBUG] Comparing '{name1}' with '{name2}'", flush=True)
        
        # Quick exact match check
        if name1.lower() == name2.lower():
            return 1.0
        
        # Get article overlap
        articles1 = set(data1.get('articles', []))
        articles2 = set(data2.get('articles', []))
        shared_articles = articles1 & articles2
        
        # Normalize names for comparison
        norm1 = self._normalize_for_blocking(name1)
        norm2 = self._normalize_for_blocking(name2)
        
        if norm1 == norm2:
            return 0.95
        
        # INTELLIGENT NAME ANALYSIS
        # Clean punctuation from names before splitting to avoid "Harper," vs "Harper" issues
        clean_name1 = re.sub(r'[,.\-;:!?]', ' ', name1).strip()
        clean_name2 = re.sub(r'[,.\-;:!?]', ' ', name2).strip()
        words1 = clean_name1.split()
        words2 = clean_name2.split()
        
        # CRITICAL: Names that share NO words should NEVER merge
        # (e.g., "Sean Fraser" and "Catherine McKenna" share no words)
        if not self._shares_any_word(words1, words2):
            return 0.0  # No shared words = different people
        
        # For PERSONS: Check if it's a name pattern
        if data1.get('type') == 'PER':
            # CRITICAL: Check if both have full first and last names that conflict
            if len(words1) >= 2 and len(words2) >= 2:
                first1 = words1[0].lower()
                first2 = words2[0].lower()
                last1 = words1[-1].lower()
                last2 = words2[-1].lower()
                
                # If last names match but first names are clearly different
                if last1 == last2 and len(first1) > 2 and len(first2) > 2:
                    if first1 != first2 and not self._could_be_nickname(first1, first2):
                        # Debug output (disabled for production)
                        # if 'ford' in last1:
                        #     print(f"  [DEBUG] Different first names: {first1} vs {first2} - returning 0.0", flush=True)
                        return 0.0  # Different people with same last name
            
            if self._shares_significant_words(words1, words2):
                if self._is_name_pattern_match(name1, name2, words1, words2):
                    # If they appear in same articles, VERY HIGH confidence
                    # (journalist using both forms to refer to same person)
                    if shared_articles:
                        return 0.98  # Very high confidence - same person in same article
                    else:
                        # Without co-occurrence, use structural analysis
                        confidence = self._get_name_confidence(name1, name2, words1, words2)
                        return confidence
        
        # For ORGANIZATIONS: Check acronyms and variations
        # Note: Organizations may not share words (e.g., "NDP" vs "New Democratic Party")
        elif data1.get('type') == 'ORG':
            if self._is_org_variation(name1, name2, words1, words2):
                # Check for acronym confidence
                is_acronym = self._is_acronym_match(name1, name2)
                
                if is_acronym:
                    # Acronyms are high confidence
                    if shared_articles:
                        return 0.95  # Very high confidence with co-occurrence
                    else:
                        return 0.88  # High confidence even without co-occurrence
                else:
                    # Other variations need co-occurrence
                    if shared_articles:
                        return 0.92  # Same org referenced differently in same article
                    else:
                        # Without co-occurrence, need very strong match
                        return 0.75  # Below threshold - orgs need co-occurrence or exact match
        
        # Special case: Check if they NEVER share words but co-occur
        # These are likely DIFFERENT entities
        if shared_articles and not self._shares_significant_words(words1, words2):
            # Different names in same article = different entities
            return 0.0
        
        # Typo detection - very similar strings
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        if similarity > 0.95:  # Almost identical (typos)
            return similarity
        
        return 0.0  # No match
    
    def _shares_any_word(self, words1: List[str], words2: List[str]) -> bool:
        """
        Check if two name lists share ANY words at all.
        Used to quickly filter out completely different names.
        """
        if not words1 or not words2:
            return False
        
        # Convert to lowercase sets
        set1 = {w.lower() for w in words1}
        set2 = {w.lower() for w in words2}
        
        # Check if they share any words
        return bool(set1 & set2)
    
    def _shares_significant_words(self, words1: List[str], words2: List[str]) -> bool:
        """
        Check if two name lists share significant words.
        """
        if not words1 or not words2:
            return False
        
        # Convert to lowercase sets
        set1 = {w.lower() for w in words1}
        set2 = {w.lower() for w in words2}
        
        # Find shared words
        shared = set1 & set2
        
        # No shared words = no match
        if not shared:
            return False
        
        # Check if shared words are significant (not just common words)
        insignificant = {'the', 'of', 'and', 'de', 'du', 'la', 'le', 'et', 'for', 'in', 'on', 'at'}
        significant_shared = shared - insignificant
        
        # Need at least one significant shared word
        if not significant_shared:
            return False
        
        # Check if one is subset of other
        if set1.issubset(set2) or set2.issubset(set1):
            return True
        
        # Check if they share a substantial portion
        min_set_size = min(len(set1), len(set2))
        if len(significant_shared) >= min_set_size * 0.5:
            return True
        
        return False
    
    def _get_name_confidence(self, name1: str, name2: str, words1: List[str], words2: List[str]) -> float:
        """
        Get intelligent confidence score for name matching without co-occurrence.
        Uses structural analysis rather than hardcoded lists.
        """
        # Sort by length
        if len(words1) < len(words2):
            shorter, longer = words1, words2
            short_name, long_name = name1, name2
        else:
            shorter, longer = words2, words1
            short_name, long_name = name2, name1
        
        # Single last name pattern (e.g., "Trudeau" vs "Justin Trudeau")
        if len(shorter) == 1 and len(longer) >= 2:
            last_name = shorter[0].lower()
            
            # Check if it matches the last name in the longer version
            if longer and last_name == longer[-1].lower():
                # Check for parsing errors with punctuation
                # Names shouldn't contain these characters except for valid initials
                suspicious_chars = [',', '"', ';', ':', '/', '\\', '|', '[', ']', '{', '}']
                if any(char in long_name for char in suspicious_chars):
                    return 0.0  # Parsing error
                
                # Check for period but allow valid initials (e.g., "J. Trudeau")
                if '.' in long_name:
                    # Periods should only appear after single letters (initials)
                    parts = long_name.split()
                    for part in parts:
                        if '.' in part and not (len(part) <= 2 and part[0].isalpha()):
                            # Period not part of an initial
                            return 0.0
                
                # Intelligent structural analysis:
                # If the longer name has 3+ words, check if middle words could be surnames
                if len(longer) >= 3:
                    # Middle words that are capitalized and substantial (4+ chars)
                    # could indicate concatenated names
                    middle_words = longer[1:-1]
                    suspicious_middle = False
                    for word in middle_words:
                        # Substantial capitalized middle word could be another surname
                        if len(word) >= 4 and word[0].isupper():
                            # But allow common middle name patterns (van, de, la, etc.)
                            if word.lower() not in ['van', 'von', 'de', 'la', 'le', 'du', 'della', 'degli']:
                                suspicious_middle = True
                                break
                    
                    if suspicious_middle:
                        # Likely concatenation of different people
                        return 0.0
                
                # Key insight: A single last name needs strong evidence to match a full name
                # Without co-occurrence, we should be conservative
                # This prevents "Ford" from matching both "Doug Ford" and "Rob Ford"
                return 0.70  # Below threshold - needs co-occurrence
                
        # Initial + last name pattern (e.g., "J. Trudeau" and "Justin Trudeau")
        
        # Initial + last name pattern (e.g., "J. Trudeau" and "Justin Trudeau")
        if len(shorter) == 2 and len(longer) >= 2:
            if shorter[0].endswith('.') and len(shorter[0]) <= 2:
                # Check if initials and last names match
                if shorter[-1].lower() == longer[-1].lower():
                    if shorter[0][0].lower() == longer[0][0].lower():
                        return 0.90  # High confidence for initial matches
        
        # Default: not enough confidence without co-occurrence
        return 0.80  # Below threshold
    
    def _is_name_pattern_match(self, name1: str, name2: str, words1: List[str], words2: List[str]) -> bool:
        """
        Check if two person names match common patterns.
        E.g., "Trudeau" and "Justin Trudeau", "J. Trudeau" and "Justin Trudeau"
        
        IMPORTANT: Must avoid merging different people with same last name
        E.g., "Catherine McKenna" should NOT merge with "Heather McKenna"
        E.g., "Doug Ford" should NOT merge with "Rob Ford"
        """
        # Sort by length
        if len(words1) < len(words2):
            shorter, longer = words1, words2
            short_name, long_name = name1, name2
        elif len(words2) < len(words1):
            shorter, longer = words2, words1
            short_name, long_name = name2, name1
        else:
            # Same length - check if they're the same person
            if len(words1) >= 2:
                # Both have at least first and last name
                first1, last1 = words1[0].lower(), words1[-1].lower()
                first2, last2 = words2[0].lower(), words2[-1].lower()
                
                # If last names match but first names differ completely, they're different people
                if last1 == last2 and first1 != first2:
                    # Check if first names are clearly different (not initials)
                    if len(first1) > 2 and len(first2) > 2:
                        # Both are full first names, not initials
                        # Check for common nicknames
                        if not self._are_name_variations(first1, first2):
                            return False  # Different people with same last name
                    # Even if one is an initial, be conservative
                    elif len(first1) > 2 or len(first2) > 2:
                        # One is full name, other might be initial - check compatibility
                        if len(first1) == 1 or first1.endswith('.'):
                            # first1 is initial, check if it matches first2
                            if first1[0] != first2[0]:
                                return False
                        elif len(first2) == 1 or first2.endswith('.'):
                            # first2 is initial, check if it matches first1
                            if first2[0] != first1[0]:
                                return False
                        else:
                            # Neither is initial but they don't match
                            return False
            return False
        
        # CRITICAL CHECK: If both names have full first and last names, verify compatibility
        if len(shorter) >= 2 and len(longer) >= 2:
            short_first = shorter[0].lower()
            short_last = shorter[-1].lower()
            long_first = longer[0].lower()
            long_last = longer[-1].lower()
            
            # Last names must match
            if short_last != long_last:
                return False
            
            # Check first name compatibility
            # If both have full first names (not initials)
            if len(short_first) > 2 and len(long_first) > 2:
                # They must be the same or variations
                if short_first != long_first:
                    if not self._are_name_variations(short_first, long_first):
                        return False  # Different first names = different people
        
        # Pattern 1: Last name only (e.g., "Trudeau")
        if len(shorter) == 1:
            last_name = shorter[0].lower()
            
            # Check if it's the last name in the longer version
            if longer and last_name == longer[-1].lower():
                # Verify it's not a common first name
                common_first_names = {
                    'justin', 'elizabeth', 'john', 'robert', 'michael',
                    'david', 'james', 'william', 'mary', 'jennifer',
                    'andrew', 'doug', 'jason', 'catherine', 'joe', 'ralph',
                    'heather', 'sean', 'emmett', 'rob', 'randy', 'renata'
                }
                if last_name not in common_first_names:
                    return True
        
        # Pattern 2: Initial + last name (e.g., "J. Trudeau")
        if len(shorter) == 2:
            first = shorter[0]
            last = shorter[1].lower()
            
            # Check if first is an initial
            if len(first) <= 2 and (first.endswith('.') or len(first) == 1):
                # Check if it matches the pattern in longer name
                if len(longer) >= 2:
                    if longer[-1].lower() == last:  # Last names match
                        # Check if initial matches
                        if longer[0][0].lower() == first[0].lower():
                            return True
        
        # Pattern 3: Middle name variations
        if len(shorter) == 2 and len(longer) == 3:
            # Check first and last match
            if shorter[0].lower() == longer[0].lower() and shorter[-1].lower() == longer[-1].lower():
                return True
        
        return False
    
    def _are_name_variations(self, name1: str, name2: str) -> bool:
        """
        Check if two first names are variations of each other using intelligent patterns.
        """
        # If they're the same, they're variations
        if name1 == name2:
            return True
        
        # Check if one is a substring of the other (common nickname pattern)
        # e.g., "Rob" in "Robert", "Doug" in "Douglas", "Chris" in "Christopher"
        if len(name1) >= 3 and len(name2) >= 3:
            if name1 in name2 or name2 in name1:
                # But avoid false positives like "Ann" in "Joanne"
                # The shorter name should start at the beginning
                if name1 in name2:
                    return name2.startswith(name1)
                else:
                    return name1.startswith(name2)
        
        # Check common phonetic patterns for nicknames
        # Pattern: First letter same, similar sound
        if name1[0] == name2[0]:
            # Check if they share significant phonetic similarity
            # This is a simplified check - could be enhanced with proper phonetic algorithms
            shorter = min(name1, name2, key=len)
            longer = max(name1, name2, key=len)
            
            # If the shorter is less than half the longer, probably not a nickname
            if len(shorter) < len(longer) // 2:
                return False
            
            # Check character overlap
            common_chars = sum(1 for c in shorter if c in longer)
            if common_chars >= len(shorter) * 0.75:  # 75% character overlap
                return True
        
        return False
    
    def _could_be_nickname(self, name1: str, name2: str) -> bool:
        """
        More lenient check if two names could possibly be nicknames.
        Used when we have other evidence (like co-occurrence).
        """
        # Use the variations check
        if self._are_name_variations(name1, name2):
            return True
        
        # Additional lenient checks for co-occurrence context
        # Same first letter and reasonable length difference
        if name1[0] == name2[0]:
            len_ratio = len(min(name1, name2)) / len(max(name1, name2))
            if len_ratio >= 0.4:  # At least 40% of the length
                return True
        
        return False
    
    def _is_acronym_match(self, name1: str, name2: str) -> bool:
        """
        Check if one name is an acronym of the other.
        """
        shorter = name1 if len(name1) < len(name2) else name2
        longer = name2 if shorter == name1 else name1
        
        # Is shorter an acronym?
        if shorter.isupper() and 2 <= len(shorter) <= 6:
            # Generate acronym from longer name
            acronym = self._generate_org_acronym(longer)
            if acronym == shorter:
                return True
        return False
    
    def _is_org_variation(self, name1: str, name2: str, words1: List[str], words2: List[str]) -> bool:
        """
        Check if two organization names are variations of each other.
        More conservative to avoid incorrect merges.
        E.g., "NDP" and "New Democratic Party" should merge
        But "Berlin Energy" and "Devon Energy" should NOT
        """
        # Check for acronym match FIRST (high confidence)
        shorter = name1 if len(name1) < len(name2) else name2
        longer = name2 if shorter == name1 else name1
        
        # Is shorter an acronym?
        if shorter.isupper() and 2 <= len(shorter) <= 6:
            # Generate acronym from longer name
            acronym = self._generate_org_acronym(longer)
            if acronym == shorter:
                return True  # High confidence - acronym match
        
        # For non-acronym cases, be more conservative
        norm1 = name1.lower()
        norm2 = name2.lower()
        
        # Exact match after normalization
        if norm1 == norm2:
            return True
        
        # Check for suffix variations (Inc, Ltd, etc.)
        suffixes = ['inc', 'corp', 'ltd', 'limited', 'company', 'co', 'llc', 'plc', 
                   'group', 'international', 'global']
        
        # Remove suffixes
        core1 = norm1
        core2 = norm2
        for suffix in suffixes:
            core1 = core1.replace(f' {suffix}', '').replace(f'.{suffix}', '')
            core2 = core2.replace(f' {suffix}', '').replace(f'.{suffix}', '')
        
        core1 = core1.strip()
        core2 = core2.strip()
        
        # Check if cores match exactly
        if core1 == core2 and core1:  # Cores must be non-empty
            return True
        
        # DO NOT merge based on partial containment
        # "Energy" should NOT match "Berlin Energy" or "Devon Energy"
        # This was the problematic logic causing bad merges
        
        # Only merge if it's a clear subset with qualifiers
        # E.g., "Ministry of Energy" and "Energy Ministry" 
        if len(words1) > 1 and len(words2) > 1:
            # Both are multi-word - check if they're reorderings
            if set(words1) == set(words2):
                return True  # Same words, different order
        
        return False  # Default: don't merge
    
    def _generate_org_acronym(self, org_name: str) -> str:
        """
        Generate acronym from organization name.
        """
        # Split into words
        words = org_name.split()
        
        # Skip common words
        skip_words = {'the', 'of', 'and', 'for', 'de', 'du', 'la', 'le'}
        significant_words = [w for w in words if w.lower() not in skip_words]
        
        if not significant_words:
            return ""
        
        # Take first letter of each significant word
        acronym = ''.join(w[0].upper() for w in significant_words if w)
        return acronym
    
    def _is_same_person_smart(self, name1: str, name2: str, norm1: str, norm2: str) -> bool:
        """
        Smart person name matching (e.g., 'Trudeau' with 'Justin Trudeau').
        """
        words1 = norm1.split()
        words2 = norm2.split()
        
        if not words1 or not words2:
            return False
        
        # Common first names to avoid merging
        common_first_names = {
            'justin', 'elizabeth', 'john', 'robert', 'michael', 
            'david', 'james', 'william', 'mary', 'jennifer',
            'andrew', 'doug', 'jason', 'catherine', 'joe', 
            'ralph', 'donald', 'stephen', 'peter', 'paul'
        }
        
        # Case 1: Last name match (e.g., "Trudeau" and "Justin Trudeau")
        if len(words1) == 1 and len(words2) > 1:
            # Check if single word matches last name
            if words1[0] == words2[-1]:
                # Don't merge if it's a common first name
                if words1[0] not in common_first_names:
                    return True
        elif len(words2) == 1 and len(words1) > 1:
            # Check reverse case
            if words2[0] == words1[-1]:
                if words2[0] not in common_first_names:
                    return True
        
        # Case 2: Initial match (e.g., "J. Trudeau" and "Justin Trudeau")
        if len(words1) == 2 and words1[0].endswith('.'):
            if len(words1[0]) <= 2:  # It's an initial
                if words2 and words1[0][0].lower() == words2[0][0].lower():
                    if words1[-1] == words2[-1]:  # Last names match
                        return True
        elif len(words2) == 2 and words2[0].endswith('.'):
            if len(words2[0]) <= 2:  # It's an initial
                if words1 and words2[0][0].lower() == words1[0][0].lower():
                    if words2[-1] == words1[-1]:  # Last names match
                        return True
        
        return False
    
    def _fast_string_similarity(self, s1: str, s2: str) -> float:
        """
        Fast approximate string similarity.
        """
        # Length-based quick rejection
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        len_ratio = min(len1, len2) / max(len1, len2)
        if len_ratio < 0.5:  # Too different in length
            return len_ratio
        
        # Character-based similarity (faster than SequenceMatcher)
        chars1 = set(s1)
        chars2 = set(s2)
        char_overlap = len(chars1 & chars2) / len(chars1 | chars2)
        
        # Weighted average of length ratio and character overlap
        return 0.3 * len_ratio + 0.7 * char_overlap
    
    def _find_clusters_fast(self, similarity_matrix: np.ndarray) -> List[Set[int]]:
        """
        Smart clustering that prevents transitive merging of incompatible entities.
        E.g., prevents "Doug Ford" and "Rob Ford" from merging through "Ford".
        """
        n = len(similarity_matrix)
        edges = []
        zero_pairs = set()  # Pairs with 0 similarity (must never merge)
        
        # Find edges above threshold AND track zero similarity pairs
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    edges.append((i, j))
                elif similarity_matrix[i, j] == 0.0:
                    zero_pairs.add((min(i, j), max(i, j)))
        
        if not edges:
            return [{i} for i in range(n)]
        
        # Build initial clusters using connected components
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        initial_clusters = list(nx.connected_components(G))
        
        # Split clusters that contain entities with 0 similarity
        final_clusters = []
        for cluster in initial_clusters:
            # Check if this cluster contains any zero-similarity pairs
            needs_split = False
            problematic_pairs = []
            for i in cluster:
                for j in cluster:
                    if i < j and (i, j) in zero_pairs:
                        needs_split = True
                        problematic_pairs.append((i, j))
            
            # Debug output (disabled for production)
            # if needs_split and len(cluster) > 2:
            #     print(f"[CLUSTER DEBUG] Cluster {sorted(cluster)} has zero-similarity pairs: {problematic_pairs}", flush=True)
            
            if not needs_split:
                final_clusters.append(cluster)
            else:
                # Split the cluster to separate zero-similarity entities
                # Build a new graph without zero-similarity connections
                sub_G = nx.Graph()
                sub_G.add_nodes_from(cluster)
                
                # Only add edges between nodes that don't have zero similarity
                for node1 in cluster:
                    for node2 in cluster:
                        if node1 < node2:
                            pair = (node1, node2)
                            # Check if they have positive similarity AND not zero
                            if pair not in zero_pairs and similarity_matrix[node1, node2] >= self.similarity_threshold:
                                sub_G.add_edge(node1, node2)
                
                # Get connected components of the filtered graph
                sub_clusters = list(nx.connected_components(sub_G))
                
                # Debug output (disabled for production)
                # if len(sub_clusters) > 1:
                #     print(f"  [CLUSTER DEBUG] Split into {len(sub_clusters)} sub-clusters: {[sorted(sc) for sc in sub_clusters]}", flush=True)
                
                final_clusters.extend(sub_clusters)
        
        return final_clusters
    
    def _merge_cross_block_clusters(self, all_clusters: List[Set]) -> List[Set]:
        """
        Merge clusters that share entities (from different blocks).
        """
        if not all_clusters:
            return []
        
        # Build graph of cluster relationships
        cluster_graph = nx.Graph()
        cluster_graph.add_nodes_from(range(len(all_clusters)))
        
        # Find overlapping clusters
        for i in range(len(all_clusters)):
            for j in range(i + 1, len(all_clusters)):
                if all_clusters[i] & all_clusters[j]:  # Have common entities
                    cluster_graph.add_edge(i, j)
        
        # Merge connected cluster groups
        merged = []
        for component in nx.connected_components(cluster_graph):
            # Union all clusters in component
            merged_cluster = set()
            for cluster_idx in component:
                merged_cluster.update(all_clusters[cluster_idx])
            merged.append(merged_cluster)
        
        return merged
    
    def _create_resolved_index(self, 
                              clusters: List[Set],
                              full_index: Dict) -> Dict:
        """
        Create resolved index from clusters.
        """
        resolved = {}
        processed = set()
        
        for cluster in clusters:
            if len(cluster) == 1:
                # Single entity
                entity_key = list(cluster)[0]
                if entity_key in full_index:
                    resolved[entity_key] = full_index[entity_key]
                    processed.add(entity_key)
            else:
                # Merge cluster
                cluster_entities = [(k, full_index[k]) for k in cluster if k in full_index]
                if cluster_entities:
                    canonical_key, merged_data = self._merge_cluster_fast(cluster_entities)
                    resolved[canonical_key] = merged_data
                    processed.update(cluster)
        
        # Add unprocessed entities
        for key, data in full_index.items():
            if key not in processed:
                resolved[key] = data
        
        return resolved
    
    def _merge_cluster_fast(self, cluster_entities: List[Tuple[str, Dict]]) -> Tuple[str, Dict]:
        """
        Fast cluster merging with final incompatibility check.
        """
        # CRITICAL: Check if cluster contains obviously different people
        # This is a safety net to prevent wrong merges
        if len(cluster_entities) > 1:
            names = [(e[0], e[1].get('name', '')) for e in cluster_entities]
            
            # Check for person entities with conflicting full names
            for i, (key1, name1) in enumerate(names):
                if 'PER:' in key1:
                    words1 = name1.lower().split()
                    if len(words1) >= 2:  # Full name with first and last
                        for j, (key2, name2) in enumerate(names):
                            if i < j and 'PER:' in key2:
                                words2 = name2.lower().split()
                                if len(words2) >= 2:  # Also full name
                                    # Same last name but different first names?
                                    if words1[-1] == words2[-1]:  # Same last name
                                        first1 = words1[0]
                                        first2 = words2[0]
                                        # Check if first names are incompatible
                                        if len(first1) > 2 and len(first2) > 2 and first1 != first2:
                                            # Not initials, different names
                                            # Check if they could be nicknames
                                            if not self._could_be_nickname(first1, first2):
                                                # Split the cluster - return individual entities
                                                # Debug output (disabled for production)
                                                # print(f"[MERGE SAFETY] Preventing merge of '{name1}' and '{name2}' - different people", flush=True)
                                                # Return just the canonical entity, don't merge
                                                canonical_key, canonical_data = max(
                                                    cluster_entities,
                                                    key=lambda x: x[1].get('authority_score', 0) * x[1].get('occurrences', 1)
                                                )
                                                return canonical_key, canonical_data
        
        # Select canonical entity (highest authority score)
        canonical_key, canonical_data = max(
            cluster_entities,
            key=lambda x: x[1].get('authority_score', 0) * x[1].get('occurrences', 1)
        )
        
        # Fast merge
        merged = canonical_data.copy()
        merged['merged_from'] = []
        
        total_occurrences = canonical_data.get('occurrences', 0)
        all_articles = set(canonical_data.get('articles', []))
        
        for key, data in cluster_entities:
            if key != canonical_key:
                total_occurrences += data.get('occurrences', 0)
                all_articles.update(data.get('articles', []))
                merged['merged_from'].append({
                    'key': key,
                    'name': data.get('name'),
                    'occurrences': data.get('occurrences', 0)
                })
        
        merged['occurrences'] = total_occurrences
        merged['articles'] = list(all_articles)
        
        return canonical_key, merged