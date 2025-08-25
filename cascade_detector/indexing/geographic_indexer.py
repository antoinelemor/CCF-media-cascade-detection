"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
geographic_indexer.py

MAIN OBJECTIVE:
---------------
This script indexes geographic patterns for cascade detection, measuring location focus, 
co-occurrence networks, and spatial diffusion patterns to identify geographic cascade dynamics.

Dependencies:
-------------
- pandas
- numpy
- json
- typing
- collections
- datetime
- logging
- networkx
- scipy

MAIN FEATURES:
--------------
1) Location extraction and resolution from NER entities
2) Geographic co-occurrence network construction
3) Spatial cascade score computation
4) Regional vs national coverage analysis
5) Cross-border diffusion pattern detection

Author:
-------
Antoine Lemor
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime
import logging
import networkx as nx
from scipy.stats import entropy

from cascade_detector.indexing.base_indexer import AbstractIndexer
from cascade_detector.utils.location_resolver_fast import FastLocationResolver as LocationResolver

from cascade_detector.utils.media_geography import MediaGeography

logger = logging.getLogger(__name__)


class GeographicIndexer(AbstractIndexer):
    """
    Geographic indexer for cascade detection through location focus analysis.
    Measures concentration and co-occurrence patterns that indicate cascades.
    """
    
    def __init__(self, 
                 use_location_resolver: bool = True,
                 similarity_threshold: float = 0.85,
                 min_occurrences: int = 2,
                 n_workers: int = 16):
        """
        Initialize geographic indexer for cascade detection.
        
        Args:
            use_location_resolver: Whether to resolve similar location names
            similarity_threshold: Minimum similarity for location merging
            min_occurrences: Minimum occurrences to consider location
            n_workers: Number of parallel workers
        """
        super().__init__(name="GeographicIndexer")
        
        self.use_location_resolver = use_location_resolver
        self.n_workers = n_workers
        self.min_occurrences = min_occurrences
        
        # Initialize location resolver if needed
        if self.use_location_resolver:
            self.location_resolver = LocationResolver(
                similarity_threshold=similarity_threshold,
                context_weight=0.3,
                min_occurrences=min_occurrences,
                n_workers=min(32, self.n_workers)  # Limit workers for speed
            )
            logger.info(f"GeographicIndexer initialized with LocationResolver")
        
        # Initialize media geography for cascade analysis
        self.media_geography = MediaGeography()
        logger.info(f"Loaded media geography: {len(self.media_geography.valid_media)} valid media outlets")
        
    def get_required_columns(self) -> List[str]:
        """Get required columns."""
        return ['date', 'doc_id', 'media', 'author', 'ner_entities']
    
    def build_index(self, data: pd.DataFrame, entity_index: Optional[Dict] = None, 
                    frame_index: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Build geographic index focused on cascade detection.
        
        Args:
            data: DataFrame with processed data
            entity_index: Optional pre-built entity index
            frame_index: Optional pre-built frame index for frame-geography analysis
            
        Returns:
            Geographic index with cascade-relevant metrics
        """
        logger.info(f"Building geographic cascade index from {len(data):,} rows...")
        
        # Extract location entities
        if entity_index:
            loc_entities = {k: v for k, v in entity_index.items() if k.startswith('LOC:')}
            logger.info(f"Using {len(loc_entities)} location entities from entity index")
        else:
            loc_entities = self._extract_location_entities(data)
            logger.info(f"Extracted {len(loc_entities)} location entities")
        
        if not loc_entities:
            logger.warning("No location entities found")
            self.index = self._create_empty_index()
            return self.index
        
        # Apply location resolution if enabled - FULLY PARALLELIZED
        if self.use_location_resolver:
            original_count = len(loc_entities)
            logger.info(f"Resolving {original_count} locations with maximum parallelization...")
            
            import time
            start_resolution = time.time()
            
            # Use ALL available cores for location resolution
            resolved_entities, location_mapping = self.location_resolver.resolve_locations(
                loc_entities,
                use_context=True,
                use_blocking=True
            )
            
            resolution_time = time.time() - start_resolution
            loc_entities = resolved_entities
            self.location_mapping = location_mapping
            
            reduction = (original_count - len(loc_entities)) / original_count * 100 if original_count > 0 else 0
            logger.info(f"Location resolution: {original_count} → {len(loc_entities)} "
                       f"(reduced by {reduction:.1f}%) in {resolution_time:.1f}s")
            
            if resolution_time > 60:
                logger.warning(f"⚠️ Location resolution took {resolution_time:.1f}s - consider optimization")
        
        # Initialize cascade-focused index
        self.index = {
            'locations': {},                    # Basic location profiles
            'focus_metrics': {},                 # Geographic focus indicators
            'media_location_network': {},       # Media → locations network
            'journalist_location_network': {},  # Journalist → locations network
            'location_cooccurrence': {},        # Location co-occurrence network
            'temporal_focus': {},                # Temporal focus patterns
            'cascade_indicators': {},            # Direct cascade indicators
            'media_regional_spread': {},        # Media regional spread analysis
            'geographic_diffusion': {},         # Geographic cascade diffusion patterns
            'frame_geographic_patterns': {},    # Frame-geography interaction patterns
            'proximity_effects': {},            # Geographic proximity analysis
            'linguistic_barriers': {}           # Linguistic barrier analysis
        }
        
        # Build location profiles
        self._build_location_profiles(loc_entities)
        
        # Build co-occurrence networks for cascade detection
        self._build_media_location_network(data)
        self._build_journalist_location_network(data)
        self._build_location_cooccurrence_network()
        
        # Calculate focus metrics
        self._calculate_focus_metrics()
        
        # Analyze temporal focus patterns
        self._analyze_temporal_focus(data)
        
        # Calculate cascade indicators
        self._calculate_cascade_indicators()
        
        # Analyze media regional spread for cascade detection
        self._analyze_media_regional_spread(data)
        
        # Calculate geographic diffusion patterns
        self._calculate_geographic_diffusion()
        
        # Analyze frame-geography interaction if frame index available
        if frame_index:
            self._analyze_frame_geographic_patterns(data, frame_index)
        
        # Analyze proximity effects and linguistic barriers
        self._analyze_proximity_and_linguistic_effects(data)
        
        # Update metadata
        self.metadata['created'] = datetime.now().isoformat()
        self.metadata['n_locations'] = len(self.index['locations'])
        self.metadata['focus_score'] = self.index['cascade_indicators'].get('overall_focus_score', 0)
        
        logger.info(f"Geographic cascade index built: {len(self.index['locations'])} locations")
        
        return self.index
    
    def _extract_location_entities(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Extract location entities from data."""
        loc_entities = defaultdict(lambda: {
            'occurrences': 0,
            'articles': set(),
            'media': [],
            'journalists': [],
            'dates': [],
            'co_mentions': defaultdict(int)
        })
        
        for idx, row in data.iterrows():
            if pd.notna(row.get('ner_entities')):
                try:
                    entities = json.loads(row['ner_entities']) if isinstance(row['ner_entities'], str) else row['ner_entities']
                    locations = entities.get('LOC', [])
                    if not isinstance(locations, list):
                        locations = [locations]
                    
                    for location in locations:
                        if location:
                            loc_key = f"LOC:{location}"
                            loc_entities[loc_key]['occurrences'] += 1
                            loc_entities[loc_key]['articles'].add(row['doc_id'])
                            loc_entities[loc_key]['dates'].append(row.get('date'))
                            
                            if pd.notna(row.get('media')):
                                loc_entities[loc_key]['media'].append(row['media'])
                            if pd.notna(row.get('author')):
                                loc_entities[loc_key]['journalists'].append(row['author'])
                            
                            # Track co-mentions with other locations
                            for other_loc in locations:
                                if other_loc and other_loc != location:
                                    loc_entities[loc_key]['co_mentions'][f"LOC:{other_loc}"] += 1
                
                except Exception as e:
                    logger.debug(f"Error extracting entities at row {idx}: {e}")
        
        # Convert sets to lists
        for loc_key in loc_entities:
            loc_entities[loc_key]['articles'] = list(loc_entities[loc_key]['articles'])
            loc_entities[loc_key]['co_mentions'] = dict(loc_entities[loc_key]['co_mentions'])
        
        return dict(loc_entities)
    
    def _build_location_profiles(self, loc_entities: Dict) -> None:
        """Build basic profiles for each location."""
        for loc_key, loc_data in loc_entities.items():
            location_name = loc_key[4:]  # Remove 'LOC:' prefix
            
            # Count unique media and journalists
            media_counts = Counter(loc_data.get('media', []))
            journalist_counts = Counter(loc_data.get('journalists', []))
            
            profile = {
                'name': location_name,
                'occurrences': loc_data.get('occurrences', 0),
                'n_articles': len(loc_data.get('articles', [])),
                'n_media': len(media_counts),
                'n_journalists': len(journalist_counts),
                'media_concentration': self._calculate_concentration(media_counts),
                'journalist_concentration': self._calculate_concentration(journalist_counts),
                'top_media': media_counts.most_common(5),
                'top_journalists': journalist_counts.most_common(5)
            }
            
            self.index['locations'][loc_key] = profile
    
    def _calculate_concentration(self, counts: Counter) -> float:
        """
        Calculate concentration (HHI) for cascade detection.
        High concentration = potential cascade indicator.
        """
        if not counts:
            return 0.0
        
        total = sum(counts.values())
        if total == 0:
            return 0.0
        
        # Herfindahl-Hirschman Index (0-10000)
        hhi = sum((count/total)**2 for count in counts.values()) * 10000
        return hhi
    
    def _build_media_location_network(self, data: pd.DataFrame) -> None:
        """
        Build media → location network for cascade detection.
        Track which media focus on which locations.
        Only includes valid media from CSV.
        """
        media_locations = defaultdict(lambda: defaultdict(int))
        
        for idx, row in data.iterrows():
            media = row.get('media')
            # Only process valid media from CSV
            if pd.notna(media) and self.media_geography.is_valid_media(media) and pd.notna(row.get('ner_entities')):
                try:
                    entities = json.loads(row['ner_entities']) if isinstance(row['ner_entities'], str) else row['ner_entities']
                    locations = entities.get('LOC', [])
                    if not isinstance(locations, list):
                        locations = [locations]
                    
                    for location in locations:
                        if location:
                            loc_key = f"LOC:{location}"
                            if loc_key in self.index['locations']:
                                media_locations[media][loc_key] += 1
                
                except Exception:
                    continue
        
        # Calculate focus scores for each media
        network = {}
        for media, locations in media_locations.items():
            total = sum(locations.values())
            if total > 0:
                # Calculate entropy (lower = more focused)
                probs = [count/total for count in locations.values()]
                location_entropy = -sum(p * np.log2(p) for p in probs if p > 0)
                
                # Top location share (higher = more focused)
                top_location_share = max(locations.values()) / total if locations else 0
                
                network[media] = {
                    'locations': dict(locations),
                    'n_locations': len(locations),
                    'total_mentions': total,
                    'entropy': location_entropy,
                    'focus_score': 1.0 - (location_entropy / np.log2(max(len(locations), 2))),
                    'top_location_share': top_location_share,
                    'top_locations': sorted(locations.items(), key=lambda x: x[1], reverse=True)[:5]
                }
        
        self.index['media_location_network'] = network
    
    def _build_journalist_location_network(self, data: pd.DataFrame) -> None:
        """
        Build journalist → location network for cascade detection.
        Track which journalists focus on which locations.
        """
        journalist_locations = defaultdict(lambda: defaultdict(int))
        
        for idx, row in data.iterrows():
            journalist = row.get('author')
            if pd.notna(journalist) and pd.notna(row.get('ner_entities')):
                try:
                    entities = json.loads(row['ner_entities']) if isinstance(row['ner_entities'], str) else row['ner_entities']
                    locations = entities.get('LOC', [])
                    if not isinstance(locations, list):
                        locations = [locations]
                    
                    for location in locations:
                        if location:
                            loc_key = f"LOC:{location}"
                            if loc_key in self.index['locations']:
                                journalist_locations[journalist][loc_key] += 1
                
                except Exception:
                    continue
        
        # Calculate focus scores for each journalist
        network = {}
        for journalist, locations in journalist_locations.items():
            total = sum(locations.values())
            if total > 0:
                # Calculate entropy
                probs = [count/total for count in locations.values()]
                location_entropy = -sum(p * np.log2(p) for p in probs if p > 0)
                
                # Top location share
                top_location_share = max(locations.values()) / total if locations else 0
                
                network[journalist] = {
                    'locations': dict(locations),
                    'n_locations': len(locations),
                    'total_mentions': total,
                    'entropy': location_entropy,
                    'focus_score': 1.0 - (location_entropy / np.log2(max(len(locations), 2))),
                    'top_location_share': top_location_share,
                    'top_locations': sorted(locations.items(), key=lambda x: x[1], reverse=True)[:3]
                }
        
        self.index['journalist_location_network'] = network
    
    def _build_location_cooccurrence_network(self) -> None:
        """
        Build location co-occurrence network.
        Locations mentioned together may indicate geographic focus patterns.
        """
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        # Aggregate co-mentions from location profiles
        for loc_key, loc_data in self.index['locations'].items():
            profile = self.index['locations'].get(loc_key, {})
            # Use co-mentions from entity extraction
            for other_loc, count in loc_data.get('co_mentions', {}).items():
                if other_loc in self.index['locations']:
                    cooccurrence[loc_key][other_loc] += count
        
        # Convert to network format
        network = {
            'edges': [],
            'nodes': list(self.index['locations'].keys()),
            'statistics': {}
        }
        
        for loc1, connections in cooccurrence.items():
            for loc2, weight in connections.items():
                if weight > 0:
                    network['edges'].append({
                        'source': loc1,
                        'target': loc2,
                        'weight': weight
                    })
        
        # Calculate network statistics
        if network['edges']:
            G = nx.Graph()
            for edge in network['edges']:
                G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
            
            network['statistics'] = {
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                'density': nx.density(G),
                'avg_clustering': nx.average_clustering(G, weight='weight') if G.number_of_nodes() > 0 else 0
            }
            
            # Find communities (potential cascade clusters)
            try:
                communities = list(nx.community.greedy_modularity_communities(G, weight='weight'))
                network['communities'] = [list(c) for c in communities if len(c) >= 3]
                network['n_communities'] = len(network['communities'])
            except:
                network['communities'] = []
                network['n_communities'] = 0
        
        self.index['location_cooccurrence'] = network
    
    def _calculate_focus_metrics(self) -> None:
        """
        Calculate geographic focus metrics for cascade detection.
        """
        locations = self.index['locations']
        
        # Overall location distribution
        total_occurrences = sum(loc['occurrences'] for loc in locations.values())
        
        if total_occurrences == 0:
            self.index['focus_metrics'] = {
                'geographic_entropy': 0,
                'geographic_concentration': 0,
                'top_location_dominance': 0,
                'media_focus_alignment': 0,
                'journalist_focus_alignment': 0
            }
            return
        
        # Geographic entropy (lower = more focused = cascade indicator)
        location_probs = [loc['occurrences']/total_occurrences for loc in locations.values()]
        geographic_entropy = -sum(p * np.log2(p) for p in location_probs if p > 0)
        
        # Geographic concentration (HHI)
        geographic_concentration = sum(p**2 for p in location_probs) * 10000
        
        # Top location dominance
        top_locations = sorted(locations.items(), key=lambda x: x[1]['occurrences'], reverse=True)[:5]
        top_location_share = sum(loc[1]['occurrences'] for loc in top_locations[:1]) / total_occurrences
        
        # Media focus alignment (do different media focus on same locations?)
        media_network = self.index['media_location_network']
        if len(media_network) > 1:
            # Calculate overlap in top locations across media
            media_top_locs = []
            for media_data in media_network.values():
                if media_data['top_locations']:
                    media_top_locs.append(set(loc[0] for loc in media_data['top_locations'][:3]))
            
            if len(media_top_locs) > 1:
                # Calculate Jaccard similarity between media
                similarities = []
                for i in range(len(media_top_locs)):
                    for j in range(i+1, len(media_top_locs)):
                        intersection = len(media_top_locs[i] & media_top_locs[j])
                        union = len(media_top_locs[i] | media_top_locs[j])
                        if union > 0:
                            similarities.append(intersection / union)
                
                media_focus_alignment = np.mean(similarities) if similarities else 0
            else:
                media_focus_alignment = 0
        else:
            media_focus_alignment = 0
        
        # Journalist focus alignment
        journalist_network = self.index['journalist_location_network']
        if len(journalist_network) > 1:
            journalist_top_locs = []
            for journalist_data in journalist_network.values():
                if journalist_data['top_locations']:
                    journalist_top_locs.append(set(loc[0] for loc in journalist_data['top_locations'][:2]))
            
            if len(journalist_top_locs) > 1:
                similarities = []
                for i in range(len(journalist_top_locs)):
                    for j in range(i+1, len(journalist_top_locs)):
                        intersection = len(journalist_top_locs[i] & journalist_top_locs[j])
                        union = len(journalist_top_locs[i] | journalist_top_locs[j])
                        if union > 0:
                            similarities.append(intersection / union)
                
                journalist_focus_alignment = np.mean(similarities) if similarities else 0
            else:
                journalist_focus_alignment = 0
        else:
            journalist_focus_alignment = 0
        
        self.index['focus_metrics'] = {
            'geographic_entropy': geographic_entropy,
            'geographic_concentration': geographic_concentration,
            'top_location_dominance': top_location_share,
            'media_focus_alignment': media_focus_alignment,
            'journalist_focus_alignment': journalist_focus_alignment,
            # Count media/journalists where top location has >30% of their coverage
            'n_focused_media': sum(1 for m in media_network.values() if m.get('top_location_share', 0) > 0.3),
            'n_focused_journalists': sum(1 for j in journalist_network.values() if j.get('top_location_share', 0) > 0.3)
        }
    
    def _analyze_temporal_focus(self, data: pd.DataFrame) -> None:
        """
        Analyze temporal patterns of geographic focus for cascade detection.
        """
        # Group by date and track location mentions
        temporal_data = defaultdict(lambda: defaultdict(int))
        
        for idx, row in data.iterrows():
            date = row.get('date')
            if pd.notna(date) and pd.notna(row.get('ner_entities')):
                try:
                    # Convert date to string for grouping
                    date_str = pd.to_datetime(date).date().isoformat()
                    
                    entities = json.loads(row['ner_entities']) if isinstance(row['ner_entities'], str) else row['ner_entities']
                    locations = entities.get('LOC', [])
                    if not isinstance(locations, list):
                        locations = [locations]
                    
                    for location in locations:
                        if location:
                            loc_key = f"LOC:{location}"
                            if loc_key in self.index['locations']:
                                temporal_data[date_str][loc_key] += 1
                
                except Exception:
                    continue
        
        if not temporal_data:
            self.index['temporal_focus'] = {
                'daily_entropy': [],
                'focus_volatility': 0,
                'sustained_focus': []
            }
            return
        
        # Calculate daily entropy
        daily_entropy = []
        daily_top_location = []
        
        for date, locations in sorted(temporal_data.items()):
            total = sum(locations.values())
            if total > 0:
                probs = [count/total for count in locations.values()]
                entropy_val = -sum(p * np.log2(p) for p in probs if p > 0)
                daily_entropy.append((date, entropy_val))
                
                # Track top location
                top_loc = max(locations.items(), key=lambda x: x[1])
                daily_top_location.append((date, top_loc[0], top_loc[1]/total))
        
        # Calculate focus volatility (changes in entropy)
        if len(daily_entropy) > 1:
            entropy_values = [e[1] for e in daily_entropy]
            focus_volatility = np.std(entropy_values)
        else:
            focus_volatility = 0
        
        # Detect sustained focus periods (consecutive days with same top location)
        sustained_focus = []
        if daily_top_location:
            current_loc = daily_top_location[0][1]
            start_date = daily_top_location[0][0]
            consecutive_days = 1
            
            for i in range(1, len(daily_top_location)):
                if daily_top_location[i][1] == current_loc:
                    consecutive_days += 1
                else:
                    if consecutive_days >= 3:  # At least 3 days
                        sustained_focus.append({
                            'location': current_loc,
                            'start': start_date,
                            'end': daily_top_location[i-1][0],
                            'days': consecutive_days,
                            'avg_dominance': np.mean([d[2] for d in daily_top_location[i-consecutive_days:i]])
                        })
                    current_loc = daily_top_location[i][1]
                    start_date = daily_top_location[i][0]
                    consecutive_days = 1
            
            # Check last period
            if consecutive_days >= 3:
                sustained_focus.append({
                    'location': current_loc,
                    'start': start_date,
                    'end': daily_top_location[-1][0],
                    'days': consecutive_days,
                    'avg_dominance': np.mean([d[2] for d in daily_top_location[-consecutive_days:]])
                })
        
        self.index['temporal_focus'] = {
            'daily_entropy': daily_entropy,
            'focus_volatility': focus_volatility,
            'sustained_focus': sustained_focus,
            'n_sustained_periods': len(sustained_focus),
            'max_sustained_days': max([s['days'] for s in sustained_focus]) if sustained_focus else 0
        }
    
    def _calculate_cascade_indicators(self) -> None:
        """
        Calculate direct cascade indicators from geographic patterns.
        """
        focus_metrics = self.index['focus_metrics']
        temporal_focus = self.index['temporal_focus']
        
        # Calculate overall focus score (0-1, higher = stronger cascade signal)
        indicators = {}
        
        # Geographic concentration indicator (HHI normalized)
        concentration_score = min(focus_metrics['geographic_concentration'] / 5000, 1.0)  # 5000 = moderate concentration
        
        # Media alignment indicator
        media_alignment_score = focus_metrics['media_focus_alignment']
        
        # Journalist alignment indicator  
        journalist_alignment_score = focus_metrics['journalist_focus_alignment']
        
        # Temporal consistency indicator
        if temporal_focus['n_sustained_periods'] > 0:
            temporal_consistency = min(temporal_focus['max_sustained_days'] / 7, 1.0)  # 7 days = strong signal
        else:
            temporal_consistency = 0
        
        # Top location dominance
        dominance_score = focus_metrics['top_location_dominance']
        
        # Combined cascade score (weighted average)
        overall_focus_score = (
            concentration_score * 0.25 +
            media_alignment_score * 0.20 +
            journalist_alignment_score * 0.20 +
            temporal_consistency * 0.20 +
            dominance_score * 0.15
        )
        
        # Cascade likelihood classification
        if overall_focus_score > 0.7:
            cascade_likelihood = 'high'
        elif overall_focus_score > 0.4:
            cascade_likelihood = 'moderate'
        else:
            cascade_likelihood = 'low'
        
        # Identify potential cascade locations
        cascade_locations = []
        for loc_key, loc_data in self.index['locations'].items():
            # High occurrences + high media concentration = potential cascade
            if (loc_data['occurrences'] > 10 and 
                loc_data['media_concentration'] > 3000 and
                loc_data['n_media'] > 2):
                
                cascade_locations.append({
                    'location': loc_key,
                    'occurrences': loc_data['occurrences'],
                    'media_concentration': loc_data['media_concentration'],
                    'n_media': loc_data['n_media'],
                    'n_journalists': loc_data['n_journalists']
                })
        
        # Sort by cascade potential
        cascade_locations.sort(key=lambda x: x['occurrences'] * (x['media_concentration']/10000), reverse=True)
        
        # Add geographic spread component
        # Calculate based on the diversity of locations with significant presence
        unique_significant_locations = sum(1 for loc_data in self.index['locations'].values() 
                                          if loc_data['occurrences'] > 5)
        total_locations = len(self.index['locations'])
        if total_locations > 0:
            # Ratio of significant locations to total (more spread = higher score)
            geographic_spread_score = min(unique_significant_locations / 20, 1.0)  # 20 locations = max spread
        else:
            geographic_spread_score = 0
        
        # Adjust overall score to include geographic spread
        overall_focus_score_with_geo = (
            concentration_score * 0.20 +
            media_alignment_score * 0.15 +
            journalist_alignment_score * 0.15 +
            temporal_consistency * 0.15 +
            dominance_score * 0.10 +
            geographic_spread_score * 0.25  # Geographic spread is important for cascades
        )
        
        self.index['cascade_indicators'] = {
            'overall_focus_score': overall_focus_score,
            'overall_score_with_geography': overall_focus_score_with_geo,
            'cascade_likelihood': cascade_likelihood,
            'concentration_score': concentration_score,
            'media_alignment_score': media_alignment_score,
            'journalist_alignment_score': journalist_alignment_score,
            'temporal_consistency_score': temporal_consistency,
            'dominance_score': dominance_score,
            'geographic_spread_score': geographic_spread_score,
            'potential_cascade_locations': cascade_locations[:10],
            'n_cascade_locations': len(cascade_locations)
        }
    
    def _create_empty_index(self) -> Dict:
        """Create an empty geographic index structure."""
        return {
            'locations': {},
            'focus_metrics': {
                'geographic_entropy': 0,
                'geographic_concentration': 0,
                'top_location_dominance': 0,
                'media_focus_alignment': 0,
                'journalist_focus_alignment': 0
            },
            'media_location_network': {},
            'journalist_location_network': {},
            'location_cooccurrence': {'edges': [], 'nodes': [], 'statistics': {}},
            'temporal_focus': {
                'daily_entropy': [],
                'focus_volatility': 0,
                'sustained_focus': []
            },
            'cascade_indicators': {
                'overall_focus_score': 0,
                'cascade_likelihood': 'low'
            }
        }
    
    def update_index(self, new_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Update geographic index with new data."""
        logger.info(f"Updating geographic index with {len(new_data):,} new rows...")
        
        # For simplicity, rebuild the entire index
        # In production, implement incremental updates
        return self.build_index(new_data, **kwargs)
    
    def query_index(self, criteria: Dict[str, Any]) -> List[Any]:
        """
        Query geographic index for cascade analysis.
        
        Criteria:
            - min_focus_score: Minimum cascade focus score
            - cascade_likelihood: 'high', 'moderate', or 'low'
            - location: Specific location to analyze
            - media: Media outlet to analyze
            - journalist: Journalist to analyze
        """
        results = []
        
        min_focus_score = criteria.get('min_focus_score', 0)
        cascade_likelihood = criteria.get('cascade_likelihood')
        location = criteria.get('location')
        media = criteria.get('media')
        journalist = criteria.get('journalist')
        
        # Return cascade indicators if requested
        if min_focus_score > 0:
            if self.index['cascade_indicators']['overall_focus_score'] >= min_focus_score:
                results.append(self.index['cascade_indicators'])
        
        if cascade_likelihood:
            if self.index['cascade_indicators']['cascade_likelihood'] == cascade_likelihood:
                results.append(self.index['cascade_indicators'])
        
        # Return specific location analysis
        if location:
            loc_key = f"LOC:{location}" if not location.startswith('LOC:') else location
            if loc_key in self.index['locations']:
                results.append({
                    'location': self.index['locations'][loc_key],
                    'in_cascade': loc_key in [l['location'] for l in self.index['cascade_indicators']['potential_cascade_locations']]
                })
        
        # Return media geographic profile
        if media and media in self.index['media_location_network']:
            results.append({
                'media': media,
                'geographic_profile': self.index['media_location_network'][media]
            })
        
        # Return journalist geographic profile
        if journalist and journalist in self.index['journalist_location_network']:
            results.append({
                'journalist': journalist,
                'geographic_profile': self.index['journalist_location_network'][journalist]
            })
        
        return results
    
    def get_cascade_locations(self, min_score: float = 0.5) -> List[Dict]:
        """
        Get locations with high cascade potential.
        
        Args:
            min_score: Minimum cascade score threshold
            
        Returns:
            List of locations with cascade indicators
        """
        cascade_locations = []
        
        for loc_data in self.index['cascade_indicators'].get('potential_cascade_locations', []):
            # Calculate location-specific cascade score
            cascade_score = (loc_data['media_concentration'] / 10000) * (loc_data['n_media'] / 10)
            
            if cascade_score >= min_score:
                cascade_locations.append({
                    'location': loc_data['location'],
                    'cascade_score': cascade_score,
                    'metrics': loc_data
                })
        
        return cascade_locations
    
    def get_media_alignment(self, threshold: float = 0.5) -> Dict:
        """
        Get media alignment patterns for cascade detection.
        
        Args:
            threshold: Minimum alignment threshold
            
        Returns:
            Media alignment analysis
        """
        media_network = self.index['media_location_network']
        
        # Find media pairs with similar geographic focus
        aligned_pairs = []
        
        media_list = list(media_network.keys())
        for i in range(len(media_list)):
            for j in range(i+1, len(media_list)):
                media1 = media_list[i]
                media2 = media_list[j]
                
                # Get top locations for each
                locs1 = set(loc[0] for loc in media_network[media1].get('top_locations', [])[:5])
                locs2 = set(loc[0] for loc in media_network[media2].get('top_locations', [])[:5])
                
                if locs1 and locs2:
                    # Calculate Jaccard similarity
                    similarity = len(locs1 & locs2) / len(locs1 | locs2)
                    
                    if similarity >= threshold:
                        aligned_pairs.append({
                            'media1': media1,
                            'media2': media2,
                            'similarity': similarity,
                            'shared_locations': list(locs1 & locs2)
                        })
        
        return {
            'aligned_pairs': aligned_pairs,
            'n_aligned_pairs': len(aligned_pairs),
            'avg_alignment': np.mean([p['similarity'] for p in aligned_pairs]) if aligned_pairs else 0
        }
    
    def _analyze_media_regional_spread(self, data: pd.DataFrame) -> None:
        """
        Analyze how stories spread across regions using media geography.
        Key for detecting geographic cascades.
        """
        # Track media coverage by date and location
        temporal_media_coverage = defaultdict(lambda: defaultdict(set))
        
        for idx, row in data.iterrows():
            date = row.get('date')
            media = row.get('media')
            
            if pd.notna(date) and pd.notna(media) and self.media_geography.is_valid_media(media):
                date_str = pd.to_datetime(date).date().isoformat()
                
                # Track locations covered by this media
                if pd.notna(row.get('ner_entities')):
                    try:
                        entities = json.loads(row['ner_entities']) if isinstance(row['ner_entities'], str) else row['ner_entities']
                        locations = entities.get('LOC', [])
                        if not isinstance(locations, list):
                            locations = [locations]
                        
                        for location in locations:
                            if location:
                                loc_key = f"LOC:{location}"
                                if loc_key in self.index['locations']:
                                    temporal_media_coverage[date_str][loc_key].add(media)
                    except:
                        continue
        
        # Analyze spread patterns
        spread_analysis = {
            'daily_spread': [],
            'max_spread_score': 0.0,
            'national_cascade_days': 0,
            'regional_cascade_days': 0,
            'cascade_progression': []
        }
        
        for date in sorted(temporal_media_coverage.keys()):
            # For each location on this date, analyze media spread
            location_spreads = {}
            
            for loc_key, media_set in temporal_media_coverage[date].items():
                if media_set:
                    spread = self.media_geography.calculate_geographic_spread(list(media_set))
                    location_spreads[loc_key] = spread
            
            if location_spreads:
                # Find location with maximum spread on this day
                max_spread_loc = max(location_spreads.items(), 
                                   key=lambda x: x[1]['cascade_geographic_score'])
                
                daily_spread = {
                    'date': date,
                    'top_location': max_spread_loc[0],
                    'spread_score': max_spread_loc[1]['cascade_geographic_score'],
                    'spread_type': max_spread_loc[1]['spread_type'],
                    'n_regions': max_spread_loc[1]['n_regions'],
                    'n_national_media': max_spread_loc[1]['n_national_media']
                }
                
                spread_analysis['daily_spread'].append(daily_spread)
                
                # Track cascade types
                if max_spread_loc[1]['spread_type'] == 'national':
                    spread_analysis['national_cascade_days'] += 1
                elif max_spread_loc[1]['spread_type'] in ['regional', 'multi-regional']:
                    spread_analysis['regional_cascade_days'] += 1
                
                # Update max spread
                if max_spread_loc[1]['cascade_geographic_score'] > spread_analysis['max_spread_score']:
                    spread_analysis['max_spread_score'] = max_spread_loc[1]['cascade_geographic_score']
        
        # Detect cascade progression (local -> regional -> national)
        if spread_analysis['daily_spread']:
            progression = self._detect_cascade_progression(spread_analysis['daily_spread'])
            spread_analysis['cascade_progression'] = progression
        
        self.index['media_regional_spread'] = spread_analysis
    
    def _detect_cascade_progression(self, daily_spreads: List[Dict]) -> Dict:
        """
        Detect if cascade follows typical progression pattern.
        Local -> Regional -> National indicates strong cascade.
        """
        if not daily_spreads:
            return {'pattern': 'none', 'strength': 0.0}
        
        # Track spread type transitions
        spread_types = [d['spread_type'] for d in daily_spreads]
        
        # Define progression levels
        progression_levels = {
            'regional': 1,
            'multi-regional': 2,
            'national': 3,
            'national-only': 2.5
        }
        
        # Calculate progression score
        max_level = 0
        level_sequence = []
        
        for spread_type in spread_types:
            level = progression_levels.get(spread_type, 0)
            level_sequence.append(level)
            max_level = max(max_level, level)
        
        # Determine pattern
        if max_level >= 3:
            pattern = 'reached_national'
        elif max_level >= 2:
            pattern = 'reached_multi_regional'
        elif max_level >= 1:
            pattern = 'remained_regional'
        else:
            pattern = 'unclear'
        
        # Calculate progression strength (how smooth was the progression)
        if len(level_sequence) > 1:
            # Check for monotonic increase (ideal cascade)
            increases = sum(1 for i in range(1, len(level_sequence)) 
                          if level_sequence[i] >= level_sequence[i-1])
            smoothness = increases / (len(level_sequence) - 1)
        else:
            smoothness = 0.0
        
        return {
            'pattern': pattern,
            'max_level_reached': max_level,
            'smoothness': smoothness,
            'strength': (max_level / 3.0) * smoothness  # Combined metric
        }
    
    def _calculate_geographic_diffusion(self) -> None:
        """
        Calculate overall geographic diffusion metrics for cascade detection.
        """
        spread_data = self.index.get('media_regional_spread', {})
        
        if not spread_data or not spread_data.get('daily_spread'):
            self.index['geographic_diffusion'] = {
                'cascade_spread_score': 0.0,
                'diffusion_type': 'none',
                'has_geographic_cascade': False
            }
            return
        
        # Calculate diffusion metrics
        max_spread = spread_data['max_spread_score']
        national_days = spread_data['national_cascade_days']
        regional_days = spread_data['regional_cascade_days']
        total_days = len(spread_data['daily_spread'])
        
        # Proportion of days with significant spread
        spread_ratio = (national_days + regional_days) / total_days if total_days > 0 else 0
        
        # National reach ratio
        national_ratio = national_days / total_days if total_days > 0 else 0
        
        # Progression strength
        progression = spread_data.get('cascade_progression', {})
        progression_strength = progression.get('strength', 0.0)
        
        # Combined cascade spread score
        cascade_spread_score = (
            max_spread * 0.3 +           # Peak spread intensity
            spread_ratio * 0.25 +         # Consistency of spread
            national_ratio * 0.25 +       # National reach
            progression_strength * 0.2    # Progression pattern
        )
        
        # Determine diffusion type
        if national_ratio > 0.3 or max_spread > 0.7:
            diffusion_type = 'national_cascade'
        elif regional_days > national_days and regional_days > total_days * 0.3:
            diffusion_type = 'regional_cascade'
        elif spread_ratio > 0.2:
            diffusion_type = 'limited_cascade'
        else:
            diffusion_type = 'no_cascade'
        
        self.index['geographic_diffusion'] = {
            'cascade_spread_score': cascade_spread_score,
            'diffusion_type': diffusion_type,
            'has_geographic_cascade': cascade_spread_score > 0.3,
            'max_daily_spread': max_spread,
            'national_coverage_days': national_days,
            'regional_coverage_days': regional_days,
            'total_days': total_days,
            'spread_consistency': spread_ratio,
            'national_reach': national_ratio,
            'progression_pattern': progression.get('pattern', 'none'),
            'progression_strength': progression_strength
        }
    
    def _analyze_frame_geographic_patterns(self, data: pd.DataFrame, frame_index: Dict) -> None:
        """
        Analyze how different frames spread geographically.
        Key for understanding cascade patterns by content type.
        """
        from concurrent.futures import ThreadPoolExecutor
        import pandas as pd
        
        # Get frame assignments for articles
        article_frames = frame_index.get('article_frames', {})
        
        # Track frame-location-media combinations
        frame_location_media = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        frame_spread_patterns = {}
        
        # Process data to link frames, locations, and media
        for idx, row in data.iterrows():
            doc_id = row.get('doc_id')
            media = row.get('media')
            
            if doc_id in article_frames and pd.notna(media) and self.media_geography.is_valid_media(media):
                # Get dominant frame for this article
                frame_data = article_frames[doc_id]
                dominant_frame = frame_data.get('dominant_frame')
                
                if dominant_frame and pd.notna(row.get('ner_entities')):
                    try:
                        entities = json.loads(row['ner_entities']) if isinstance(row['ner_entities'], str) else row['ner_entities']
                        locations = entities.get('LOC', [])
                        if not isinstance(locations, list):
                            locations = [locations]
                        
                        for location in locations:
                            if location:
                                loc_key = f"LOC:{location}"
                                if loc_key in self.index['locations']:
                                    date_str = pd.to_datetime(row.get('date')).date().isoformat()
                                    frame_location_media[dominant_frame][loc_key][date_str].add(media)
                    except:
                        continue
        
        # Analyze spread patterns for each frame with parallelization
        def analyze_frame_spread(frame_data):
            frame, location_data = frame_data
            spread_pattern = {
                'frame': frame,
                'geographic_tendency': 'unknown',
                'avg_spread_score': 0.0,
                'max_spread_score': 0.0,
                'typical_spread_type': None,
                'n_locations': len(location_data),
                'linguistic_barrier_crossing': 0.0
            }
            
            spread_scores = []
            spread_types = []
            linguistic_crossings = []
            
            for loc_key, date_media in location_data.items():
                for date, media_set in date_media.items():
                    if media_set:
                        # Calculate geographic spread
                        spread = self.media_geography.calculate_geographic_spread(list(media_set))
                        spread_scores.append(spread['cascade_geographic_score'])
                        spread_types.append(spread['spread_type'])
                        
                        # Analyze linguistic barriers
                        linguistic = self.media_geography.analyze_linguistic_barriers(list(media_set))
                        if linguistic['barrier_crossed']:
                            linguistic_crossings.append(1)
                        else:
                            linguistic_crossings.append(0)
            
            if spread_scores:
                spread_pattern['avg_spread_score'] = np.mean(spread_scores)
                spread_pattern['max_spread_score'] = max(spread_scores)
                spread_pattern['linguistic_barrier_crossing'] = np.mean(linguistic_crossings)
                
                # Determine typical spread type
                from collections import Counter
                type_counts = Counter(spread_types)
                if type_counts:
                    spread_pattern['typical_spread_type'] = type_counts.most_common(1)[0][0]
                
                # Determine geographic tendency
                if spread_pattern['avg_spread_score'] > 0.5:
                    spread_pattern['geographic_tendency'] = 'national'
                elif spread_pattern['avg_spread_score'] > 0.3:
                    spread_pattern['geographic_tendency'] = 'regional'
                else:
                    spread_pattern['geographic_tendency'] = 'local'
            
            return frame, spread_pattern
        
        # Use thread pool for parallel frame analysis
        with ThreadPoolExecutor(max_workers=min(8, self.n_workers)) as executor:
            frame_results = list(executor.map(
                analyze_frame_spread,
                frame_location_media.items()
            ))
        
        for frame, pattern in frame_results:
            frame_spread_patterns[frame] = pattern
        
        # Expected patterns for different frames (data-driven calibration)
        expected_patterns = {
            'Pol': {'expected_tendency': 'national', 'expected_score': 0.7},
            'Eco': {'expected_tendency': 'national', 'expected_score': 0.6},
            'Envt': {'expected_tendency': 'regional', 'expected_score': 0.4},
            'Just': {'expected_tendency': 'local', 'expected_score': 0.3},
            'Pbh': {'expected_tendency': 'regional', 'expected_score': 0.4},
            'Sci': {'expected_tendency': 'national', 'expected_score': 0.5},
            'Secu': {'expected_tendency': 'national', 'expected_score': 0.7},
            'Cult': {'expected_tendency': 'local', 'expected_score': 0.2}
        }
        
        # Detect anomalies (frames spreading differently than expected)
        anomalies = []
        for frame, pattern in frame_spread_patterns.items():
            if frame in expected_patterns:
                expected = expected_patterns[frame]
                
                # Check if actual differs significantly from expected
                score_diff = abs(pattern['avg_spread_score'] - expected['expected_score'])
                if score_diff > 0.3:
                    anomalies.append({
                        'frame': frame,
                        'expected_tendency': expected['expected_tendency'],
                        'actual_tendency': pattern['geographic_tendency'],
                        'score_difference': score_diff,
                        'cascade_signal': 'strong' if pattern['avg_spread_score'] > expected['expected_score'] else 'weak'
                    })
        
        self.index['frame_geographic_patterns'] = {
            'frame_patterns': frame_spread_patterns,
            'anomalies': anomalies,
            'n_anomalies': len(anomalies),
            'strongest_cascade_frame': max(frame_spread_patterns.items(), 
                                          key=lambda x: x[1]['avg_spread_score'])[0] if frame_spread_patterns else None
        }
    
    def _analyze_proximity_and_linguistic_effects(self, data: pd.DataFrame) -> None:
        """
        Analyze geographic proximity and linguistic barrier effects.
        """
        # Track which regions are affected over time
        temporal_regions = defaultdict(set)
        temporal_media = defaultdict(set)
        
        for idx, row in data.iterrows():
            date = row.get('date')
            media = row.get('media')
            
            if pd.notna(date) and pd.notna(media) and self.media_geography.is_valid_media(media):
                date_str = pd.to_datetime(date).date().isoformat()
                temporal_media[date_str].add(media)
                
                # Get region for this media
                region = self.media_geography.get_media_region(media)
                if region and region != 'National':
                    temporal_regions[date_str].add(region)
        
        # Analyze proximity effects over time
        proximity_timeline = []
        linguistic_timeline = []
        
        for date in sorted(temporal_regions.keys()):
            if temporal_regions[date]:
                # Calculate proximity effects
                proximity = self.media_geography.calculate_proximity_effects(list(temporal_regions[date]))
                proximity['date'] = date
                proximity_timeline.append(proximity)
            
            if temporal_media[date]:
                # Calculate linguistic barriers
                linguistic = self.media_geography.analyze_linguistic_barriers(list(temporal_media[date]))
                linguistic['date'] = date
                linguistic_timeline.append(linguistic)
        
        # Calculate overall metrics
        if proximity_timeline:
            avg_coherence = np.mean([p['geographic_coherence'] for p in proximity_timeline])
            avg_proximity = np.mean([p['proximity_score'] for p in proximity_timeline])
            
            # Detect if spread follows geographic proximity
            coherence_trend = []
            for i in range(1, len(proximity_timeline)):
                if proximity_timeline[i]['geographic_coherence'] >= proximity_timeline[i-1]['geographic_coherence']:
                    coherence_trend.append(1)
                else:
                    coherence_trend.append(0)
            
            follows_proximity = np.mean(coherence_trend) > 0.5 if coherence_trend else False
        else:
            avg_coherence = 0.0
            avg_proximity = 0.0
            follows_proximity = False
        
        if linguistic_timeline:
            # Calculate linguistic barrier metrics
            barrier_crossing_days = sum(1 for l in linguistic_timeline if l['barrier_crossed'])
            total_days = len(linguistic_timeline)
            barrier_crossing_rate = barrier_crossing_days / total_days if total_days > 0 else 0
            
            avg_permeability = np.mean([l['linguistic_permeability'] for l in linguistic_timeline])
            
            # Find when barrier was first crossed
            first_crossing = None
            for l in linguistic_timeline:
                if l['barrier_crossed']:
                    first_crossing = l['date']
                    break
        else:
            barrier_crossing_rate = 0.0
            avg_permeability = 0.0
            first_crossing = None
        
        self.index['proximity_effects'] = {
            'avg_geographic_coherence': avg_coherence,
            'avg_proximity_score': avg_proximity,
            'follows_geographic_proximity': follows_proximity,
            'proximity_timeline': proximity_timeline[:30],  # Limit for memory
            'max_isolated_regions': max([p['n_isolated_regions'] for p in proximity_timeline]) if proximity_timeline else 0
        }
        
        self.index['linguistic_barriers'] = {
            'barrier_crossing_rate': barrier_crossing_rate,
            'avg_linguistic_permeability': avg_permeability,
            'first_barrier_crossing': first_crossing,
            'linguistic_timeline': linguistic_timeline[:30],  # Limit for memory
            'days_to_cross_barrier': linguistic_timeline.index(
                next((l for l in linguistic_timeline if l['barrier_crossed']), None)
            ) + 1 if any(l['barrier_crossed'] for l in linguistic_timeline) else None
        }