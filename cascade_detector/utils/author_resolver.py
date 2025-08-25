"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
author_resolver.py

MAIN OBJECTIVE:
---------------
This script provides intelligent author and journalist name resolution, handling common variations
in bylines and author names to accurately track journalist influence across media outlets.

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
- time

MAIN FEATURES:
--------------
1) Name variation detection and standardization
2) Context-based similarity matching
3) Media outlet association tracking
4) Graph-based transitive closure for name merging
5) Handles special cases (wire services, multiple authors)

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
import time

logger = logging.getLogger(__name__)


class AuthorResolver:
    """
    Intelligent author/journalist name resolution.
    Handles common variations in bylines and author names.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.85,  # Lowered slightly for better matching
                 context_weight: float = 0.3,  # Reduced context weight
                 min_articles: int = 2,
                 n_workers: int = None):
        """
        Initialize author resolver.
        
        Args:
            similarity_threshold: Minimum similarity for merging (0-1)
            context_weight: Weight for context similarity (0-1)
            min_articles: Minimum articles to consider author
        """
        self.similarity_threshold = similarity_threshold
        self.context_weight = context_weight
        self.min_articles = min_articles
        
        # Add parallel processing support
        import multiprocessing as mp
        self.n_workers = n_workers or min(mp.cpu_count(), 14)  # M4 Max optimization
        
        # Common suffixes to remove from author names
        self.suffixes_to_remove = [
            # News agencies
            ', Reuters', ', AP', ', Associated Press', ', Bloomberg',
            ', Canadian Press', ', The Canadian Press', ', CP',
            ', Agence France-Presse', ', AFP',
            
            # Roles
            ', Staff', ', Staff Writer', ', Reporter', ', Correspondent',
            ', Editor', ', Contributing Editor', ', Senior Editor',
            ', Columnist', ', Bureau Chief', ', News Editor',
            
            # Locations
            ', Ottawa', ', Toronto', ', Montreal', ', Vancouver',
            ', Calgary', ', Edmonton', ', Quebec City',
            ', Washington', ', New York', ', London', ', Paris',
            
            # Special cases
            ' - Special to.*', ' - .*Bureau$', r' \(.*\)$'
        ]
        
        # Common title variations
        self.titles = {
            'english': ['Mr.', 'Ms.', 'Mrs.', 'Dr.', 'Prof.', 'Sir', 'Dame'],
            'french': ['M.', 'Mme', 'Mlle', 'Dr', 'Prof.', 'Me']
        }
        
        # Known name variations (can be extended)
        self.known_aliases = {
            'Robert': 'Robert',
        }
        
        self._resolution_cache = {}
        
    def resolve_authors(self, 
                       journalist_profiles: Dict[str, Dict],
                       article_profiles: Dict[str, Dict] = None,
                       use_context: bool = True) -> Tuple[Dict[str, Dict], Dict[str, str]]:
        """
        Resolve author name variations.
        
        Args:
            journalist_profiles: Journalist profiles from SourceIndexer
            article_profiles: Article profiles for context (optional)
            use_context: Whether to use context-based resolution
            
        Returns:
            Tuple of (resolved_profiles, name_mapping)
        """
        logger.info(f"Starting author resolution for {len(journalist_profiles)} journalists...")
        
        # Filter journalists with minimum articles
        eligible_journalists = {
            name: profile for name, profile in journalist_profiles.items()
            if len(profile.get('articles', [])) >= self.min_articles
        }
        
        logger.info(f"Processing {len(eligible_journalists)} journalists with >= {self.min_articles} articles")
        
        # Build similarity matrix with progress tracking
        from tqdm import tqdm
        journalists = list(eligible_journalists.keys())
        
        with tqdm(total=1, desc="Building author similarity matrix", unit="stage", leave=False) as pbar:
            similarity_matrix = self._build_similarity_matrix(
                journalists, eligible_journalists, article_profiles, use_context
            )
            pbar.update(1)
        
        # Find clusters
        clusters = self._find_clusters(journalists, similarity_matrix)
        
        # Create name mapping and resolved profiles
        name_mapping = {}
        resolved_profiles = {}
        
        for cluster in clusters:
            if len(cluster) == 1:
                # Single journalist, no merging
                journalist = journalists[list(cluster)[0]]
                resolved_profiles[journalist] = journalist_profiles[journalist]
                name_mapping[journalist] = journalist
            else:
                # Merge cluster
                cluster_journalists = [journalists[i] for i in cluster]
                canonical_name, merged_profile = self._merge_journalist_cluster(
                    cluster_journalists, journalist_profiles
                )
                resolved_profiles[canonical_name] = merged_profile
                
                # Update mapping
                for journalist in cluster_journalists:
                    name_mapping[journalist] = canonical_name
        
        # Add journalists that didn't meet minimum threshold
        for name, profile in journalist_profiles.items():
            if name not in name_mapping:
                resolved_profiles[name] = profile
                name_mapping[name] = name
        
        # Log statistics
        original_count = len(journalist_profiles)
        resolved_count = len(resolved_profiles)
        reduction = (original_count - resolved_count) / original_count * 100
        
        logger.info(f"Author resolution complete: {original_count} → {resolved_count} "
                   f"(reduced by {reduction:.1f}%)")
        
        return resolved_profiles, name_mapping
    
    def _build_similarity_matrix(self,
                                 journalists: List[str],
                                 profiles: Dict[str, Dict],
                                 article_profiles: Optional[Dict] = None,
                                 use_context: bool = True) -> np.ndarray:
        """
        Build similarity matrix using blocking to reduce comparisons.
        Uses intelligent filtering to avoid computing all O(n²) pairs.
        """
        n = len(journalists)
        similarity_matrix = np.zeros((n, n))
        
        # Create blocks to reduce comparisons
        logger.info(f"Building similarity matrix for {n} journalists using blocking...")
        
        # Step 1: Create blocks based on name characteristics
        blocks = self._create_name_blocks(journalists)
        
        # Count actual pairs to compute
        total_pairs = 0
        for block_key, block_indices in blocks.items():
            block_size = len(block_indices)
            if block_size > 1:
                total_pairs += block_size * (block_size - 1) // 2
        
        logger.info(f"Reduced comparisons from {n*(n-1)//2:,} to {total_pairs:,} pairs using {len(blocks)} blocks")
        
        if total_pairs == 0:
            # No similar names found
            np.fill_diagonal(similarity_matrix, 1.0)
            return similarity_matrix
        
        # Step 2: Compute similarities only within blocks
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        from tqdm import tqdm
        
        # Prepare block pairs for parallel processing
        block_pairs = []
        for block_key, block_indices in blocks.items():
            if len(block_indices) > 1:
                # Create pairs within this block
                for i_idx, i in enumerate(block_indices):
                    for j in block_indices[i_idx + 1:]:
                        block_pairs.append((i, j))
        
        # Process in MUCH smaller chunks for better parallelization
        # With 58M pairs, we need thousands of small chunks
        # Aim for ~1000-5000 pairs per chunk for fast processing
        chunk_size = 2000  # Fixed small chunk size for consistent performance
        
        # For very small sets, use smaller chunks
        if len(block_pairs) < 10000:
            chunk_size = max(100, len(block_pairs) // (self.n_workers * 4))
        chunks = [block_pairs[i:i + chunk_size] for i in range(0, len(block_pairs), chunk_size)]
        
        # Use MORE workers for similarity computation (CPU-bound task)
        # For M4 Max with 16 cores, we can use all of them
        similarity_workers = min(mp.cpu_count(), max(self.n_workers, 16))  # Use all 16 cores
        
        logger.info(f"Processing {len(block_pairs):,} pairs in {len(chunks)} chunks using {similarity_workers} workers")
        logger.info(f"Chunk size: ~{chunk_size} pairs per chunk")
        
        # Process chunks in parallel with more workers
        with ProcessPoolExecutor(max_workers=similarity_workers, mp_context=mp.get_context('spawn')) as executor:
            futures = []
            for chunk in chunks:
                future = executor.submit(
                    _compute_similarity_chunk,
                    chunk,
                    journalists,
                    profiles,
                    article_profiles,
                    use_context,
                    self.context_weight,
                    self.similarity_threshold
                )
                futures.append(future)
            
            # Collect results with better progress tracking
            completed = 0
            start_time = time.time()
            with tqdm(total=len(futures), desc="Computing similarities", unit="chunks", 
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                     smoothing=0.1) as pbar:
                for future in as_completed(futures, timeout=300):  # 5 min total timeout
                    try:
                        chunk_results = future.result(timeout=10)  # 10s per chunk max
                        for (i, j), sim in chunk_results:
                            similarity_matrix[i, j] = sim
                            similarity_matrix[j, i] = sim
                        completed += 1
                        pbar.update(1)
                        
                        # Update with statistics every 10 chunks
                        if completed % 10 == 0:
                            elapsed = time.time() - start_time
                            rate = completed / elapsed if elapsed > 0 else 0
                            pbar.set_postfix({'rate': f'{rate:.1f} chunks/s', 'workers': similarity_workers})
                    except Exception as e:
                        logger.error(f"Error computing similarity chunk: {e}")
                        pbar.update(1)
        
        # Set diagonal to 1
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix
    
    def _create_name_blocks(self, journalists: List[str]) -> Dict[str, List[int]]:
        """
        Create blocks of potentially similar names to reduce comparisons.
        Uses CONSERVATIVE blocking to avoid creating too many pairs.
        """
        from collections import defaultdict
        blocks = defaultdict(set)  # Use set to avoid duplicates
        
        for idx, name in enumerate(journalists):
            # Normalize for blocking
            normalized = self._normalize_author_name(name).lower()
            
            if not normalized:
                continue
            
            parts = normalized.split()
            if not parts:
                continue
            
            # MAIN BLOCKING: Use last name as primary block
            # This is the most reliable indicator of same person
            if len(parts) >= 2:
                # Last word is likely surname
                surname = parts[-1]
                if len(surname) >= 3:
                    # Create block by first 3 chars of surname
                    block_key = f"surname_{surname[:3]}"
                    blocks[block_key].add(idx)
            elif len(parts) == 1:
                # Single name - use first 3 chars
                if len(parts[0]) >= 3:
                    block_key = f"single_{parts[0][:3]}"
                    blocks[block_key].add(idx)
        
        # Convert to regular dict and filter
        filtered_blocks = {}
        total_pairs = 0
        
        for key, indices in blocks.items():
            if len(indices) > 1 and len(indices) < 1000:  # Cap block size to avoid explosion
                indices_list = list(indices)
                filtered_blocks[key] = indices_list
                # Count pairs in this block
                n = len(indices_list)
                total_pairs += n * (n - 1) // 2
        
        # If we still have too many pairs, be more aggressive
        if total_pairs > 100000:  # Cap at 100k pairs
            logger.warning(f"Too many pairs ({total_pairs:,}), using more restrictive blocking")
            # Only keep smaller blocks
            restricted_blocks = {}
            restricted_pairs = 0
            
            for key, indices in sorted(filtered_blocks.items(), key=lambda x: len(x[1])):
                n = len(indices)
                block_pairs = n * (n - 1) // 2
                if restricted_pairs + block_pairs <= 100000:
                    restricted_blocks[key] = indices
                    restricted_pairs += block_pairs
                else:
                    break
            
            return restricted_blocks
        
        return filtered_blocks
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two author names.
        """
        # Normalize names
        norm1 = self._normalize_author_name(name1)
        norm2 = self._normalize_author_name(name2)
        
        # Exact match after normalization
        if norm1 == norm2:
            return 1.0
        
        # Check for subset relationship (one name contains the other)
        if self._is_name_subset(norm1, norm2):
            return 0.95
        
        # Check for initials match (e.g., "J. Smith" vs "John Smith")
        if self._match_author_initials(norm1, norm2):
            return 0.92
        
        # Check for known aliases
        if self._check_aliases(norm1, norm2):
            return 0.90
        
        # Check for last name match with different first names
        if self._last_name_match(norm1, norm2):
            # Could be same person with nickname or middle name
            # But needs more evidence, so lower score
            return 0.70
        
        # Standard fuzzy matching
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def _normalize_author_name(self, name: str) -> str:
        """
        Normalize author name by removing suffixes and titles.
        """
        if not name:
            return ""
        
        normalized = name.strip()
        
        # Remove common suffixes
        for suffix in self.suffixes_to_remove:
            if suffix.startswith(' - '):
                # Regex pattern
                normalized = re.sub(suffix, '', normalized, flags=re.IGNORECASE)
            else:
                # Simple string replacement
                normalized = normalized.replace(suffix, '')
        
        # Remove titles
        for lang_titles in self.titles.values():
            for title in lang_titles:
                normalized = re.sub(rf'^{re.escape(title)}\s+', '', normalized, flags=re.IGNORECASE)
        
        # Remove extra spaces and punctuation
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.strip(' ,.-')
        
        return normalized.lower()
    
    def _is_name_subset(self, name1: str, name2: str) -> bool:
        """
        Check if one name is a subset of another.
        """
        parts1 = set(name1.split())
        parts2 = set(name2.split())
        
        # Require at least one meaningful part (not just initials)
        meaningful1 = [p for p in parts1 if len(p) > 2]
        meaningful2 = [p for p in parts2 if len(p) > 2]
        
        if not meaningful1 or not meaningful2:
            return False
        
        # Check if all parts of shorter name are in longer name
        if len(parts1) < len(parts2):
            return parts1.issubset(parts2) and len(parts1) >= 2
        elif len(parts2) < len(parts1):
            return parts2.issubset(parts1) and len(parts2) >= 2
        
        return False
    
    def _match_author_initials(self, name1: str, name2: str) -> bool:
        """
        Check if names match when considering initials.
        Examples: "J. Smith" vs "John Smith", "J. P. Morgan" vs "John Pierpont Morgan"
        """
        parts1 = name1.split()
        parts2 = name2.split()
        
        # Allow different number of parts for initials
        # e.g., "J. Smith" (2 parts) vs "John Smith" (2 parts)
        # or "J. Smith" (2 parts) vs "John Q. Smith" (3 parts)
        
        # Must have at least last name in common
        if not parts1 or not parts2:
            return False
        
        # Check if last names match
        if parts1[-1] != parts2[-1]:
            return False
        
        # Check first names/initials
        if len(parts1) >= 2 and len(parts2) >= 2:
            # Check if first part is initial match
            first1, first2 = parts1[0], parts2[0]
            if first1 != first2:
                # Check if one is initial of the other
                if not ((len(first1) == 1 and first2.startswith(first1)) or
                       (len(first2) == 1 and first1.startswith(first2))):
                    return False
        
        return True
    
    def _check_aliases(self, name1: str, name2: str) -> bool:
        """
        Check for known name aliases (Bob/Robert, Bill/William, etc.)
        """
        parts1 = name1.split()
        parts2 = name2.split()
        
        if len(parts1) != len(parts2):
            return False
        
        alias_matches = 0
        for p1, p2 in zip(parts1, parts2):
            if p1 == p2:
                continue
            
            # Check aliases in both directions
            if p1 in self.known_aliases and self.known_aliases[p1].lower() == p2:
                alias_matches += 1
            elif p2 in self.known_aliases and self.known_aliases[p2].lower() == p1:
                alias_matches += 1
            else:
                # If not an alias and not the same, not a match
                return False
        
        return alias_matches > 0
    
    def _last_name_match(self, name1: str, name2: str) -> bool:
        """
        Check if last names match (useful for identifying potential same person).
        """
        parts1 = name1.split()
        parts2 = name2.split()
        
        if len(parts1) < 2 or len(parts2) < 2:
            return False
        
        # Assume last part is last name
        return parts1[-1] == parts2[-1]
    
    def _calculate_context_similarity(self,
                                     profile1: Dict,
                                     profile2: Dict,
                                     article_profiles: Dict) -> float:
        """
        Calculate context similarity based on writing patterns and topics.
        """
        articles1 = set(profile1.get('articles', []))
        articles2 = set(profile2.get('articles', []))
        
        if not articles1 or not articles2:
            return 0.0
        
        # Check temporal overlap (do they write at similar times?)
        temporal_sim = self._calculate_temporal_similarity(profile1, profile2)
        
        # Check topic similarity (do they write about similar things?)
        topic_sim = self._calculate_topic_similarity(
            articles1, articles2, article_profiles
        )
        
        # Check messenger pattern similarity
        messenger_sim = self._calculate_messenger_similarity(profile1, profile2)
        
        # Weighted combination
        return 0.3 * temporal_sim + 0.4 * topic_sim + 0.3 * messenger_sim
    
    def _calculate_temporal_similarity(self, profile1: Dict, profile2: Dict) -> float:
        """
        Calculate temporal overlap between two journalists.
        """
        # This would need date information from articles
        # For now, return a default value
        return 0.5
    
    def _calculate_topic_similarity(self,
                                   articles1: Set[str],
                                   articles2: Set[str],
                                   article_profiles: Dict) -> float:
        """
        Calculate topic similarity based on frames and entities.
        """
        if not article_profiles:
            return 0.0
        
        # Get frames for each journalist's articles
        frames1 = []
        frames2 = []
        
        for article_id in articles1:
            if article_id in article_profiles:
                profile = article_profiles[article_id]
                if 'dominant_frame' in profile:
                    frames1.append(profile['dominant_frame'])
        
        for article_id in articles2:
            if article_id in article_profiles:
                profile = article_profiles[article_id]
                if 'dominant_frame' in profile:
                    frames2.append(profile['dominant_frame'])
        
        if not frames1 or not frames2:
            return 0.0
        
        # Calculate frame distribution similarity
        from collections import Counter
        dist1 = Counter(frames1)
        dist2 = Counter(frames2)
        
        # Get all frames
        all_frames = set(dist1.keys()) | set(dist2.keys())
        
        # Calculate cosine similarity of distributions
        vec1 = np.array([dist1.get(f, 0) for f in all_frames])
        vec2 = np.array([dist2.get(f, 0) for f in all_frames])
        
        if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        return 0.0
    
    def _calculate_messenger_similarity(self, profile1: Dict, profile2: Dict) -> float:
        """
        Calculate similarity in messenger usage patterns.
        """
        props1 = profile1.get('avg_proportions', {})
        props2 = profile2.get('avg_proportions', {})
        
        if not props1 or not props2:
            return 0.0
        
        # Get common messenger types
        common_types = set(props1.keys()) & set(props2.keys())
        
        if not common_types:
            return 0.0
        
        # Calculate cosine similarity
        vec1 = np.array([props1.get(t, 0) for t in common_types])
        vec2 = np.array([props2.get(t, 0) for t in common_types])
        
        if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        return 0.0
    
    def _find_clusters(self, journalists: List[str], similarity_matrix: np.ndarray) -> List[Set[int]]:
        """
        Find journalist clusters using graph-based community detection.
        Enhanced with better thresholding and logging.
        """
        # Build graph from similarity matrix
        G = nx.Graph()
        
        for i in range(len(journalists)):
            G.add_node(i)
        
        # Count edges at different thresholds for debugging
        edge_counts = {}
        thresholds_to_test = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        for thresh in thresholds_to_test:
            count = np.sum(similarity_matrix > thresh) // 2  # Divide by 2 for upper triangle
            edge_counts[thresh] = count
        
        logger.debug(f"Edge counts at different thresholds: {edge_counts}")
        
        # Use adaptive threshold if too few edges
        effective_threshold = self.similarity_threshold
        min_edges = max(10, len(journalists) // 100)  # At least 10 edges or 1% of nodes
        
        if edge_counts.get(self.similarity_threshold, 0) < min_edges:
            # LOWER threshold to get more edges (not higher!)
            # Sort ascending to try lower thresholds first
            for thresh in sorted(thresholds_to_test):
                if thresh < self.similarity_threshold and edge_counts.get(thresh, 0) >= min_edges:
                    effective_threshold = thresh
                    logger.info(f"Lowered similarity threshold from {self.similarity_threshold} to {effective_threshold} to find more matches")
                    break
            
            # If still no good threshold found, use a reasonable default
            if effective_threshold == self.similarity_threshold:
                effective_threshold = max(0.7, self.similarity_threshold - 0.15)
                logger.info(f"Using fallback threshold {effective_threshold} (original: {self.similarity_threshold})")
        
        # Add edges for similarities above threshold
        edges_added = 0
        for i in range(len(journalists)):
            for j in range(i + 1, len(journalists)):
                if similarity_matrix[i, j] >= effective_threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i, j])
                    edges_added += 1
                    
                    # Log some high-similarity pairs for debugging
                    if edges_added <= 10 and similarity_matrix[i, j] >= 0.9:
                        logger.debug(f"High similarity ({similarity_matrix[i, j]:.3f}): '{journalists[i]}' <-> '{journalists[j]}'")
        
        logger.info(f"Added {edges_added} edges to similarity graph (threshold: {effective_threshold})")
        
        # Find connected components (clusters)
        clusters = list(nx.connected_components(G))
        
        # Log cluster statistics
        cluster_sizes = [len(c) for c in clusters]
        if any(size > 1 for size in cluster_sizes):
            multi_node_clusters = sum(1 for size in cluster_sizes if size > 1)
            max_cluster = max(cluster_sizes)
            logger.info(f"Found {multi_node_clusters} multi-journalist clusters (largest: {max_cluster} journalists)")
        
        return clusters
    
    def _merge_journalist_cluster(self,
                                 cluster_journalists: List[str],
                                 profiles: Dict[str, Dict]) -> Tuple[str, Dict]:
        """
        Merge a cluster of journalists into a single canonical profile.
        """
        # Select canonical name (most articles or longest name)
        canonical_name = max(
            cluster_journalists,
            key=lambda x: (len(profiles[x].get('articles', [])), len(x))
        )
        
        # Start with canonical profile
        merged_profile = profiles[canonical_name].copy()
        merged_profile['merged_from'] = []
        
        # Merge data from all journalists
        all_articles = set(merged_profile.get('articles', []))
        
        for journalist in cluster_journalists:
            if journalist != canonical_name:
                profile = profiles[journalist]
                
                # Merge articles
                all_articles.update(profile.get('articles', []))
                
                # Track merged names
                merged_profile['merged_from'].append({
                    'name': journalist,
                    'articles': len(profile.get('articles', []))
                })
                
                # Merge messenger proportions (weighted average)
                # This would require more sophisticated merging logic
        
        merged_profile['articles'] = list(all_articles)
        merged_profile['n_articles'] = len(all_articles)
        
        # Recalculate consistency and other metrics
        # This would need to be done based on the merged article set
        
        return canonical_name, merged_profile


# Helper function for parallel similarity computation (must be at module level for pickling)
def _compute_similarity_chunk(pairs, journalists, profiles, article_profiles, 
                             use_context, context_weight, similarity_threshold):
    """
    Compute similarity for a chunk of journalist pairs.
    Optimized for speed and accuracy.
    
    Args:
        pairs: List of (i, j) index pairs to compute
        journalists: List of journalist names
        profiles: Journalist profiles
        article_profiles: Article profiles for context
        use_context: Whether to use context
        context_weight: Weight for context similarity
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        List of ((i, j), similarity) tuples
    """
    from difflib import SequenceMatcher
    import re
    import numpy as np
    
    # Comprehensive suffix list for normalization
    suffixes_to_remove = [
        # News agencies
        ', reuters', ', ap', ', associated press', ', bloomberg',
        ', canadian press', ', the canadian press', ', cp',
        ', agence france-presse', ', afp',
        # Roles
        ', staff', ', staff writer', ', reporter', ', correspondent',
        ', editor', ', contributing editor', ', senior editor',
        ', columnist', ', bureau chief', ', news editor',
        # Locations
        ', ottawa', ', toronto', ', montreal', ', vancouver',
        ', calgary', ', edmonton', ', quebec city',
        ', washington', ', new york', ', london', ', paris'
    ]
    
    def normalize_name(name):
        """Enhanced name normalization."""
        name = name.lower().strip()
        
        # Remove suffixes
        for suffix in suffixes_to_remove:
            if name.endswith(suffix):
                name = name[:-len(suffix)].strip()
        
        # Remove parenthetical content
        name = re.sub(r'\([^)]*\)', '', name).strip()
        
        # Remove special characters but keep spaces and hyphens
        name = re.sub(r'[^a-z0-9\s\-]', '', name).strip()
        
        # Normalize whitespace
        name = ' '.join(name.split())
        
        return name
    
    def calculate_enhanced_similarity(name1, name2, prof1, prof2):
        """Calculate enhanced similarity with multiple factors."""
        # Normalize names
        norm1 = normalize_name(name1)
        norm2 = normalize_name(name2)
        
        # Quick check: if normalized names are identical
        if norm1 == norm2:
            return 1.0
        
        # Check if one name is a subset of the other (common for initials)
        if norm1 in norm2 or norm2 in norm1:
            length_ratio = min(len(norm1), len(norm2)) / max(len(norm1), len(norm2))
            if length_ratio > 0.5:  # At least half the length
                return 0.95  # High similarity for subsets
        
        # Calculate base name similarity
        name_sim = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Check for last name match (important for journalists)
        parts1 = norm1.split()
        parts2 = norm2.split()
        
        if len(parts1) >= 2 and len(parts2) >= 2:
            # Compare last names
            if parts1[-1] == parts2[-1]:  # Same last name
                name_sim = min(1.0, name_sim + 0.3)  # Boost similarity
            
            # Check for first initial match
            if parts1[0][0] == parts2[0][0]:  # Same first initial
                name_sim = min(1.0, name_sim + 0.1)
        
        # Context similarity if profiles available
        if use_context and prof1 and prof2:
            # Media overlap
            media1 = set(prof1.get('media', []))
            media2 = set(prof2.get('media', []))
            
            if media1 and media2:
                media_overlap = len(media1 & media2) / len(media1 | media2)
            else:
                media_overlap = 0.0
            
            # Time overlap (check if they were active in similar periods)
            # This would require date analysis from articles
            
            # Weighted combination
            similarity = (1 - context_weight) * name_sim + context_weight * media_overlap
        else:
            similarity = name_sim
        
        return similarity
    
    # Process pairs
    results = []
    for i, j in pairs:
        name1 = journalists[i]
        name2 = journalists[j]
        
        # Get profiles
        prof1 = profiles.get(name1, {})
        prof2 = profiles.get(name2, {})
        
        # Calculate enhanced similarity
        similarity = calculate_enhanced_similarity(name1, name2, prof1, prof2)
        
        # Store if above threshold (use 70% of threshold for initial filtering)
        if similarity > similarity_threshold * 0.7:
            results.append(((i, j), similarity))
    
    return results