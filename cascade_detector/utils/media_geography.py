"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
media_geography.py

MAIN OBJECTIVE:
---------------
This script manages media geographic associations for cascade detection, determining if cascades
are local, regional, or national by mapping media outlets to Canadian provinces and regions.

Dependencies:
-------------
- pandas
- numpy
- logging
- typing
- pathlib

MAIN FEATURES:
--------------
1) Media-province mapping from CSV configuration
2) National vs regional media classification
3) Language-based media grouping (francophone/anglophone)
4) Provincial adjacency for proximity analysis
5) Cross-regional cascade pattern detection

Author:
-------
Antoine Lemor
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Set, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MediaGeography:
    """
    Manages media geographic associations for cascade analysis.
    Determines if cascades are local, regional, or national.
    """
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize media geography mapping.
        
        Args:
            csv_path: Path to media_by_province.csv
        """
        if csv_path is None:
            # Default path relative to this file
            csv_path = Path(__file__).parent.parent / "metrics" / "geographical_data" / "media_by_province.csv"
        
        self.csv_path = Path(csv_path)
        self.media_regions = {}
        self.valid_media = set()
        self.national_media = set()
        self.regional_media = {}  # region -> set of media
        self.francophone_media = set()
        self.anglophone_media = set()
        
        # Provincial adjacency for proximity analysis
        self.province_adjacency = {
            'Ontario': ['Quebec', 'Manitoba'],
            'Quebec': ['Ontario', 'New Brunswick', 'Newfoundland and Labrador'],
            'British Columbia': ['Alberta', 'Yukon'],
            'Alberta': ['British Columbia', 'Saskatchewan', 'Northwest Territories'],
            'Saskatchewan': ['Alberta', 'Manitoba'],
            'Manitoba': ['Saskatchewan', 'Ontario', 'Nunavut'],
            'New Brunswick': ['Quebec', 'Nova Scotia', 'Prince Edward Island'],
            'Nova Scotia': ['New Brunswick', 'Prince Edward Island'],
            'Prince Edward Island': ['New Brunswick', 'Nova Scotia'],
            'Newfoundland and Labrador': ['Quebec'],
            'Yukon': ['British Columbia', 'Northwest Territories'],
            'Northwest Territories': ['Yukon', 'Alberta', 'Saskatchewan', 'Nunavut'],
            'Nunavut': ['Northwest Territories', 'Manitoba']
        }
        
        self._load_media_mappings()
        self._identify_linguistic_groups()
    
    def _load_media_mappings(self) -> None:
        """Load media-region mappings from CSV."""
        if not self.csv_path.exists():
            logger.warning(f"Media geography CSV not found at {self.csv_path}")
            return
        
        try:
            df = pd.read_csv(self.csv_path)
            
            for _, row in df.iterrows():
                media = row['media'].strip()
                region = row['region'].strip()
                
                self.media_regions[media] = region
                self.valid_media.add(media)
                
                if region == 'National':
                    self.national_media.add(media)
                else:
                    if region not in self.regional_media:
                        self.regional_media[region] = set()
                    self.regional_media[region].add(media)
            
            logger.info(f"Loaded {len(self.valid_media)} media outlets: "
                       f"{len(self.national_media)} national, "
                       f"{len(self.valid_media) - len(self.national_media)} regional")
            
            # Log regions found
            regions = sorted(self.regional_media.keys())
            logger.info(f"Regions: {', '.join(regions)}")
            
        except Exception as e:
            logger.error(f"Error loading media geography CSV: {e}")
    
    def is_valid_media(self, media: str) -> bool:
        """Check if media is in the valid list."""
        return media in self.valid_media
    
    def get_media_region(self, media: str) -> Optional[str]:
        """Get the region for a media outlet."""
        return self.media_regions.get(media)
    
    def is_national_media(self, media: str) -> bool:
        """Check if media is national."""
        return media in self.national_media
    
    def get_region_media(self, region: str) -> Set[str]:
        """Get all media for a region."""
        if region == 'National':
            return self.national_media
        return self.regional_media.get(region, set())
    
    def get_all_regions(self) -> List[str]:
        """Get all unique regions (including National)."""
        regions = list(self.regional_media.keys())
        if self.national_media:
            regions.append('National')
        return sorted(regions)
    
    def filter_valid_media(self, media_list: List[str]) -> List[str]:
        """Filter a list to only include valid media."""
        return [m for m in media_list if self.is_valid_media(m)]
    
    def calculate_geographic_spread(self, media_list: List[str]) -> Dict:
        """
        Calculate geographic spread metrics for cascade detection.
        
        Args:
            media_list: List of media outlets covering a story
            
        Returns:
            Dictionary with spread metrics
        """
        # Filter to valid media only
        valid_media = self.filter_valid_media(media_list)
        
        if not valid_media:
            return {
                'spread_type': 'none',
                'n_regions': 0,
                'n_national_media': 0,
                'national_coverage_ratio': 0.0,
                'regional_diversity': 0.0,
                'cascade_geographic_score': 0.0
            }
        
        # Count by type
        national_count = sum(1 for m in valid_media if self.is_national_media(m))
        
        # Count unique regions covered
        regions_covered = set()
        for media in valid_media:
            region = self.get_media_region(media)
            if region and region != 'National':
                regions_covered.add(region)
        
        # Calculate metrics
        n_regions = len(regions_covered)
        total_regions = len(self.regional_media)  # Total possible regions
        
        # National coverage ratio (proportion of national media involved)
        national_ratio = national_count / len(self.national_media) if self.national_media else 0
        
        # Regional diversity (how many regions are covered)
        regional_diversity = n_regions / total_regions if total_regions > 0 else 0
        
        # Determine spread type
        if national_count >= 2 or (national_count >= 1 and n_regions >= 3):
            spread_type = 'national'
        elif n_regions >= 2:
            spread_type = 'multi-regional'
        elif n_regions == 1:
            spread_type = 'regional'
        else:
            spread_type = 'national-only' if national_count > 0 else 'unknown'
        
        # Cascade geographic score (0-1, higher = wider spread = stronger cascade)
        cascade_score = (
            national_ratio * 0.5 +  # National media involvement
            regional_diversity * 0.3 +  # Regional spread
            (min(n_regions, 5) / 5) * 0.2  # Raw region count
        )
        
        return {
            'spread_type': spread_type,
            'n_regions': n_regions,
            'regions_covered': list(regions_covered),
            'n_national_media': national_count,
            'n_regional_media': len(valid_media) - national_count,
            'national_coverage_ratio': national_ratio,
            'regional_diversity': regional_diversity,
            'cascade_geographic_score': cascade_score,
            'total_valid_media': len(valid_media)
        }
    
    def analyze_cascade_pattern(self, 
                                media_timeline: Dict[str, List[str]],
                                min_spread_score: float = 0.3) -> Dict:
        """
        Analyze temporal cascade pattern across regions.
        
        Args:
            media_timeline: Dict of date -> list of media outlets
            min_spread_score: Minimum score to consider as cascade
            
        Returns:
            Cascade pattern analysis
        """
        cascade_events = []
        
        for date in sorted(media_timeline.keys()):
            media_list = media_timeline[date]
            spread = self.calculate_geographic_spread(media_list)
            
            if spread['cascade_geographic_score'] >= min_spread_score:
                cascade_events.append({
                    'date': date,
                    'spread_score': spread['cascade_geographic_score'],
                    'spread_type': spread['spread_type'],
                    'n_regions': spread['n_regions']
                })
        
        # Analyze pattern
        if not cascade_events:
            return {
                'has_geographic_cascade': False,
                'cascade_events': [],
                'max_spread_score': 0.0,
                'cascade_duration_days': 0
            }
        
        # Calculate cascade metrics
        max_score = max(e['spread_score'] for e in cascade_events)
        first_date = cascade_events[0]['date']
        last_date = cascade_events[-1]['date']
        
        # Convert dates and calculate duration
        from datetime import datetime
        try:
            first = datetime.fromisoformat(first_date)
            last = datetime.fromisoformat(last_date)
            duration = (last - first).days + 1
        except:
            duration = len(cascade_events)
        
        return {
            'has_geographic_cascade': True,
            'cascade_events': cascade_events,
            'max_spread_score': max_score,
            'cascade_duration_days': duration,
            'peak_spread_type': max(cascade_events, key=lambda x: x['spread_score'])['spread_type']
        }
    
    def _identify_linguistic_groups(self) -> None:
        """
        Identify francophone and anglophone media outlets.
        """
        # Francophone media (French names or known French outlets)
        francophone_indicators = ['Le ', 'La ', 'Journal', 'Devoir', 'Droit', 'Acadie', 'Nouvelle']
        
        # Explicitly francophone
        known_francophone = {'Le Devoir', 'La Presse', 'La Presse Plus', 'Journal de Montreal', 
                            'Le Droit', 'Acadie Nouvelle'}
        
        for media in self.valid_media:
            if media in known_francophone or any(indicator in media for indicator in francophone_indicators):
                self.francophone_media.add(media)
            else:
                self.anglophone_media.add(media)
        
        logger.info(f"Identified {len(self.francophone_media)} francophone and "
                   f"{len(self.anglophone_media)} anglophone media outlets")
    
    def calculate_proximity_effects(self, affected_regions: List[str]) -> Dict:
        """
        Calculate geographic proximity effects for cascade spread.
        
        Args:
            affected_regions: List of regions currently affected
            
        Returns:
            Proximity metrics including coherence score
        """
        if not affected_regions:
            return {
                'geographic_coherence': 0.0,
                'proximity_score': 0.0,
                'isolated_regions': [],
                'connected_regions': []
            }
        
        # Count adjacent connections
        proximity_connections = 0
        possible_connections = 0
        isolated_regions = []
        connected_regions = []
        
        for region in affected_regions:
            if region == 'National' or region not in self.province_adjacency:
                continue
            
            neighbors = self.province_adjacency[region]
            has_affected_neighbor = False
            
            for neighbor in neighbors:
                possible_connections += 1
                if neighbor in affected_regions:
                    proximity_connections += 1
                    has_affected_neighbor = True
            
            if has_affected_neighbor:
                connected_regions.append(region)
            else:
                isolated_regions.append(region)
        
        # Calculate coherence (how geographically connected the spread is)
        geographic_coherence = (proximity_connections / possible_connections 
                               if possible_connections > 0 else 0.0)
        
        # Proximity score (penalizes isolated regions)
        proximity_score = 1.0 - (len(isolated_regions) / len(affected_regions) 
                                if affected_regions else 0.0)
        
        return {
            'geographic_coherence': geographic_coherence,
            'proximity_score': proximity_score,
            'n_isolated_regions': len(isolated_regions),
            'n_connected_regions': len(connected_regions),
            'isolated_regions': isolated_regions,
            'connected_regions': connected_regions
        }
    
    def analyze_linguistic_barriers(self, media_list: List[str]) -> Dict:
        """
        Analyze linguistic barrier crossing in media coverage.
        
        Args:
            media_list: List of media outlets covering a story
            
        Returns:
            Linguistic barrier metrics
        """
        valid_media = self.filter_valid_media(media_list)
        
        if not valid_media:
            return {
                'linguistic_diversity': 0.0,
                'barrier_crossed': False,
                'n_francophone': 0,
                'n_anglophone': 0,
                'linguistic_permeability': 0.0
            }
        
        # Count by language
        francophone_count = sum(1 for m in valid_media if m in self.francophone_media)
        anglophone_count = sum(1 for m in valid_media if m in self.anglophone_media)
        
        # Barrier is crossed if both languages present
        barrier_crossed = francophone_count > 0 and anglophone_count > 0
        
        # Linguistic diversity (balance between languages)
        total = francophone_count + anglophone_count
        if total > 0:
            # Use entropy for diversity
            p_franco = francophone_count / total
            p_anglo = anglophone_count / total
            
            if p_franco > 0 and p_anglo > 0:
                linguistic_diversity = -(p_franco * np.log2(p_franco) + p_anglo * np.log2(p_anglo))
            else:
                linguistic_diversity = 0.0
            
            # Permeability (how balanced the crossing is)
            linguistic_permeability = min(p_franco, p_anglo) * 2  # Max 1.0 when 50/50
        else:
            linguistic_diversity = 0.0
            linguistic_permeability = 0.0
        
        return {
            'linguistic_diversity': linguistic_diversity,
            'barrier_crossed': barrier_crossed,
            'n_francophone': francophone_count,
            'n_anglophone': anglophone_count,
            'linguistic_permeability': linguistic_permeability,
            'francophone_ratio': francophone_count / total if total > 0 else 0,
            'anglophone_ratio': anglophone_count / total if total > 0 else 0
        }