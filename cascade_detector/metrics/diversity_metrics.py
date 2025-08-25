"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
diversity_metrics.py

MAIN OBJECTIVE:
---------------
This script calculates diversity metrics for cascade detection using Phase 1 indices, measuring
source, viewpoint, geographic, and temporal diversity in media coverage.

Dependencies:
-------------
- numpy
- pandas
- scipy
- typing
- collections
- logging
- json

MAIN FEATURES:
--------------
1) Shannon entropy calculation for diversity measurement
2) Source diversity across journalists and media outlets
3) Viewpoint diversity through frame distribution
4) Geographic diversity in coverage patterns
5) Temporal diversity and concentration metrics

Author:
-------
Antoine Lemor
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import defaultdict, Counter
import logging
import json

logger = logging.getLogger(__name__)


class DiversityMetrics:
    """
    Calculate diversity metrics using Phase 1 indices.
    Enhanced with geographical, temporal, and viewpoint diversity.
    """
    
    def __init__(self,
                 frame_index: Dict[str, Any],
                 source_index: Dict[str, Any],
                 emotion_index: Dict[str, Any],
                 entity_index: Optional[Dict[str, Any]] = None,
                 geographic_index: Optional[Dict[str, Any]] = None):
        """
        Initialize with Phase 1 indices.
        
        Args:
            frame_index: Output from FrameIndexer.build_index()
            source_index: Output from SourceIndexer.build_index()  
            emotion_index: Output from EmotionIndexer.build_index()
            entity_index: Optional entity index for entity diversity
            geographic_index: Optional geographic index for data-driven geographic analysis
        """
        self.frame_index = frame_index
        self.source_index = source_index
        self.emotion_index = emotion_index
        self.entity_index = entity_index or {}
        self.geographic_index = geographic_index or {}
        self._cache = {}
    
    def calculate_frame_diversity(self, 
                                 article_ids: List[str],
                                 method: str = 'shannon') -> float:
        """
        Calculate frame diversity using various entropy measures.
        
        Args:
            article_ids: List of article IDs
            method: 'shannon', 'simpson', or 'gini'
            
        Returns:
            Diversity score (higher = more diverse)
        """
        cache_key = f"frame_div_{method}_{len(article_ids)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        frame_counts = defaultdict(float)
        
        # Aggregate frame distributions from articles
        article_frames = self.frame_index.get('article_frames', {})
        
        for doc_id in article_ids:
            if doc_id in article_frames:
                frame_dist = article_frames[doc_id].get('frame_distribution', {})
                for frame, weight in frame_dist.items():
                    frame_counts[frame] += weight
        
        if not frame_counts:
            return 0.0
        
        # Calculate diversity based on method
        diversity = self._calculate_diversity(frame_counts, method)
        
        self._cache[cache_key] = diversity
        return diversity
    
    def _calculate_diversity(self, counts: Dict, method: str = 'shannon') -> float:
        """
        Calculate diversity using specified method.
        
        Args:
            counts: Dictionary of counts/weights
            method: Diversity calculation method
            
        Returns:
            Diversity score
        """
        if not counts:
            return 0.0
        
        # Convert to probabilities
        total = sum(counts.values())
        if total == 0:
            return 0.0
        
        probs = np.array([count/total for count in counts.values()])
        
        if method == 'shannon':
            # Shannon entropy
            return entropy(probs, base=2)
        
        elif method == 'simpson':
            # Simpson's diversity index (1 - sum(p^2))
            return 1 - np.sum(probs ** 2)
        
        elif method == 'gini':
            # Gini coefficient (inequality measure, inverted for diversity)
            sorted_probs = np.sort(probs)
            n = len(sorted_probs)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_probs)) / (n * np.sum(sorted_probs)) - (n + 1) / n
            return 1 - gini  # Invert so higher = more diverse
        
        else:
            return entropy(probs)
    
    def calculate_source_diversity(self, 
                                  article_ids: List[str],
                                  include_entities: bool = True) -> Dict[str, Any]:
        """
        Calculate diversity of sources (media, journalists, entities).
        Enhanced with additional metrics.
        
        Args:
            article_ids: List of article IDs
            include_entities: Whether to include entity diversity
            
        Returns:
            Dictionary with diversity metrics
        """
        diversity = {
            'media_diversity': 0.0,
            'journalist_diversity': 0.0,
            'entity_diversity': 0.0,
            'cross_media_score': 0.0,
            'unique_media': set(),
            'unique_journalists': set(),
            'unique_entities': set(),
            'concentration_index': {}
        }
        
        media_counts = defaultdict(int)
        journalist_counts = defaultdict(int)
        entity_counts = defaultdict(int)
        
        # Media-journalist pairs for cross-media analysis
        media_journalist_pairs = set()
        
        article_profiles = self.source_index.get('article_profiles', {})
        
        for doc_id in article_ids:
            if doc_id in article_profiles:
                profile = article_profiles[doc_id]
                
                # Media diversity
                media = profile.get('media')
                if media and media != 'Unknown':
                    media_counts[media] += 1
                    diversity['unique_media'].add(media)
                
                # Journalist diversity
                journalist = profile.get('author')
                if journalist and journalist != 'Unknown':
                    journalist_counts[journalist] += 1
                    diversity['unique_journalists'].add(journalist)
                    
                    # Track media-journalist pairs
                    if media and media != 'Unknown':
                        media_journalist_pairs.add((media, journalist))
                
                # Entity diversity
                if include_entities:
                    entities = profile.get('entities', [])
                    for entity in entities:
                        entity_counts[entity] += 1
                        diversity['unique_entities'].add(entity)
        
        # Calculate diversity scores
        if media_counts:
            diversity['media_diversity'] = self._calculate_diversity(media_counts, 'shannon')
            diversity['media_concentration'] = self._calculate_concentration(media_counts)
        
        if journalist_counts:
            diversity['journalist_diversity'] = self._calculate_diversity(journalist_counts, 'shannon')
            diversity['journalist_concentration'] = self._calculate_concentration(journalist_counts)
        
        if entity_counts:
            diversity['entity_diversity'] = self._calculate_diversity(entity_counts, 'shannon')
            diversity['entity_concentration'] = self._calculate_concentration(entity_counts)
        
        # Cross-media score (journalists working across multiple media)
        if media_journalist_pairs:
            journalists_per_media = defaultdict(set)
            media_per_journalist = defaultdict(set)
            
            for media, journalist in media_journalist_pairs:
                journalists_per_media[media].add(journalist)
                media_per_journalist[journalist].add(media)
            
            # Calculate cross-pollination
            cross_pollination = []
            for journalist, media_set in media_per_journalist.items():
                if len(media_set) > 1:
                    cross_pollination.append(len(media_set))
            
            if cross_pollination:
                diversity['cross_media_score'] = np.mean(cross_pollination) / len(diversity['unique_media'])
        
        # Convert sets to lists for JSON serialization
        diversity['unique_media'] = list(diversity['unique_media'])
        diversity['unique_journalists'] = list(diversity['unique_journalists'])
        diversity['unique_entities'] = list(diversity['unique_entities'])[:100]  # Limit for memory
        
        # Add summary statistics
        diversity['n_unique_sources'] = (len(diversity['unique_media']) + 
                                        len(diversity['unique_journalists']))
        diversity['source_richness'] = diversity['n_unique_sources'] / len(article_ids) if article_ids else 0
        
        return diversity
    
    def _calculate_concentration(self, counts: Dict) -> Dict[str, float]:
        """
        Calculate market concentration indices.
        
        Args:
            counts: Dictionary of counts
            
        Returns:
            Concentration metrics
        """
        if not counts:
            return {'hhi': 0, 'cr4': 0}
        
        total = sum(counts.values())
        if total == 0:
            return {'hhi': 0, 'cr4': 0}
        
        # Calculate market shares
        shares = [count/total for count in counts.values()]
        shares.sort(reverse=True)
        
        # Herfindahl-Hirschman Index (HHI)
        hhi = sum(s**2 for s in shares) * 10000  # Scale to 0-10000
        
        # Concentration Ratio (CR4) - top 4 share
        cr4 = sum(shares[:4]) if len(shares) >= 4 else sum(shares)
        
        return {
            'hhi': hhi,
            'cr4': cr4,
            'top_share': shares[0] if shares else 0
        }
    
    def calculate_emotional_diversity(self, 
                                     article_ids: List[str],
                                     include_intensity: bool = True) -> Dict[str, Any]:
        """
        Calculate diversity of emotional tones.
        Enhanced with intensity and polarity measures.
        
        Args:
            article_ids: List of article IDs
            include_intensity: Include emotional intensity analysis
            
        Returns:
            Dictionary with emotional diversity metrics
        """
        emotion_distribution = defaultdict(int)
        sentiments = []
        intensities = []
        
        article_emotions = self.emotion_index.get('article_emotions', {})
        
        for doc_id in article_ids:
            if doc_id in article_emotions:
                emotions = article_emotions[doc_id]
                
                # Dominant emotion
                dominant = emotions.get('dominant_emotion', 'neutral')
                emotion_distribution[dominant] += 1
                
                # Sentiment score
                sentiment = emotions.get('sentiment_score', 0)
                sentiments.append(sentiment)
                
                # Emotional intensity
                if include_intensity:
                    intensity = emotions.get('emotional_intensity', 0)
                    intensities.append(intensity)
        
        # Calculate metrics
        results = {
            'emotion_entropy': 0.0,
            'sentiment_variance': 0.0,
            'sentiment_polarity': 0.0,
            'emotion_distribution': dict(emotion_distribution),
            'mean_sentiment': 0.0,
            'emotional_consensus': 0.0
        }
        
        # Emotion entropy
        if emotion_distribution:
            results['emotion_entropy'] = self._calculate_diversity(emotion_distribution, 'shannon')
            
            # Emotional consensus (inverse of entropy, normalized)
            max_entropy = np.log2(3)  # Max entropy for 3 emotions
            if max_entropy > 0:
                results['emotional_consensus'] = 1 - (results['emotion_entropy'] / max_entropy)
        
        # Sentiment statistics
        if sentiments:
            results['mean_sentiment'] = float(np.mean(sentiments))
            results['sentiment_variance'] = float(np.var(sentiments))
            results['sentiment_std'] = float(np.std(sentiments))
            
            # Polarity (how extreme the sentiments are)
            results['sentiment_polarity'] = float(np.mean(np.abs(sentiments)))
            
            # Sentiment distribution
            results['sentiment_distribution'] = {
                'positive': sum(1 for s in sentiments if s > 0.1),
                'neutral': sum(1 for s in sentiments if -0.1 <= s <= 0.1),
                'negative': sum(1 for s in sentiments if s < -0.1)
            }
        
        # Intensity analysis
        if include_intensity and intensities:
            results['mean_intensity'] = float(np.mean(intensities))
            results['intensity_variance'] = float(np.var(intensities))
            results['high_intensity_ratio'] = sum(1 for i in intensities if i > 0.5) / len(intensities)
        
        return results
    
    def calculate_viewpoint_diversity(self, 
                                    article_data: pd.DataFrame,
                                    use_messengers: bool = True) -> Dict[str, float]:
        """
        Calculate diversity of viewpoints using messenger types and frames.
        
        Args:
            article_data: DataFrame with article sentences
            use_messengers: Use messenger type analysis
            
        Returns:
            Viewpoint diversity metrics
        """
        if use_messengers:
            # Import here to avoid circular dependency
            from cascade_detector.analysis.messenger_ner_cross import MessengerNERCrossReference
            
            cross_ref = MessengerNERCrossReference()
            messenger_type_counts = defaultdict(int)
            source_type_counts = defaultdict(int)
            
            # Analyze messengers by article
            for doc_id, article in article_data.groupby('doc_id'):
                analysis = cross_ref.analyze_article(article)
                
                # Count messenger types
                for msg_type, sources in analysis.get('identified_sources', {}).items():
                    if sources:
                        messenger_type_counts[msg_type] += len(sources)
                
                # Track source diversity score from article
                if 'source_diversity' in analysis:
                    source_type_counts[doc_id] = analysis['source_diversity']
        
            results = {
                'messenger_diversity': 0.0,
                'n_messenger_types': len(messenger_type_counts),
                'dominant_messenger': None,
                'source_entropy': 0.0
            }
            
            if messenger_type_counts:
                results['messenger_diversity'] = self._calculate_diversity(messenger_type_counts, 'shannon')
                results['dominant_messenger'] = max(messenger_type_counts, key=messenger_type_counts.get)
            
            if source_type_counts:
                results['avg_article_source_diversity'] = np.mean(list(source_type_counts.values()))
        
        else:
            # Use frame-based viewpoint diversity
            results = {
                'frame_viewpoint_diversity': self.calculate_frame_diversity(
                    list(article_data['doc_id'].unique())
                )
            }
        
        return results
    
    def calculate_geographic_diversity(self, 
                                      article_ids: List[str],
                                      use_geographic_index: bool = True) -> Dict[str, Any]:
        """
        Calculate geographic diversity of mentioned locations.
        
        High diversity = discourse is spread across many locations (low cascade likelihood)
        Low diversity = discourse focused on few locations (high cascade likelihood)
        
        Args:
            article_ids: List of article IDs
            use_geographic_index: Whether to use GeographicIndexer results
            
        Returns:
            Geographic diversity metrics for cascade detection
        """
        # Use geographic index if available for data-driven analysis
        if use_geographic_index and self.geographic_index:
            return self._calculate_geographic_diversity_datadriven(article_ids)
        
        # Fallback to entity-based analysis
        location_counts = defaultdict(int)
        
        # Extract locations from articles
        article_profiles = self.source_index.get('article_profiles', {})
        
        for doc_id in article_ids:
            if doc_id in article_profiles:
                entities = article_profiles[doc_id].get('entities', [])
                
                for entity in entities:
                    # Check if it's a location entity
                    if entity.startswith('LOC:'):
                        location = entity[4:]  # Remove 'LOC:' prefix
                        location_counts[location] += 1
        
        results = {
            'geographic_diversity': 0.0,
            'n_unique_locations': len(location_counts),
            'location_entropy': 0.0,
            'top_locations': [],
            'geographic_concentration': 0.0,
            'data_driven': False  # Indicate this is not fully data-driven
        }
        
        if location_counts:
            # Calculate diversity
            results['location_entropy'] = self._calculate_diversity(location_counts, 'shannon')
            results['geographic_diversity'] = results['location_entropy']
            
            # Top locations
            top_locs = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            results['top_locations'] = [{'location': loc, 'count': count} for loc, count in top_locs]
            
            # Concentration
            concentration = self._calculate_concentration(location_counts)
            results['geographic_concentration'] = concentration['hhi']
        
        return results
    
    def _calculate_geographic_diversity_datadriven(self, article_ids: List[str]) -> Dict[str, Any]:
        """
        Calculate geographic diversity using enriched geographic index.
        
        Key metrics:
        - Location entropy: Shannon entropy of location mentions (high = diverse)
        - Geographic concentration: HHI of locations (high = focused)
        - Cascade focus score: Likelihood of geographic cascade
        
        Args:
            article_ids: List of article IDs
            
        Returns:
            Data-driven geographic diversity metrics
        """
        # Get cascade indicators from geographic index
        cascade_indicators = self.geographic_index.get('cascade_indicators', {})
        focus_metrics = self.geographic_index.get('focus_metrics', {})
        locations_index = self.geographic_index.get('locations', {})
        
        # Filter locations mentioned in these articles
        article_locations = defaultdict(int)
        article_profiles = self.source_index.get('article_profiles', {})
        
        for doc_id in article_ids:
            if doc_id in article_profiles:
                entities = article_profiles[doc_id].get('entities', [])
                for entity in entities:
                    if entity.startswith('LOC:') and entity in locations_index:
                        article_locations[entity] += 1
        
        if not article_locations:
            return {
                'geographic_diversity': 0.0,
                'n_unique_locations': 0,
                'location_entropy': 0.0,
                'geographic_concentration': 0.0,
                'cascade_focus_score': 0.0,
                'cascade_likelihood': 'low',
                'top_locations': [],
                'data_driven': True
            }
        
        # Calculate diversity metrics
        results = {
            'data_driven': True,
            'n_unique_locations': len(article_locations)
        }
        
        # Location entropy (lower = more focused = potential cascade)
        results['location_entropy'] = self._calculate_diversity(article_locations, 'shannon')
        # Invert for diversity score (high diversity = low cascade risk)
        max_entropy = np.log2(len(article_locations)) if len(article_locations) > 1 else 1
        results['geographic_diversity'] = results['location_entropy'] / max_entropy if max_entropy > 0 else 0
        
        # Geographic concentration (HHI)
        concentration = self._calculate_concentration(article_locations)
        results['geographic_concentration'] = concentration['hhi']
        
        # Cascade indicators
        results['cascade_focus_score'] = cascade_indicators.get('overall_focus_score', 0.0)
        results['cascade_likelihood'] = cascade_indicators.get('cascade_likelihood', 'low')
        results['media_alignment'] = focus_metrics.get('media_focus_alignment', 0.0)
        results['journalist_alignment'] = focus_metrics.get('journalist_focus_alignment', 0.0)
        
        # Top locations with cascade potential
        top_locs = sorted(article_locations.items(), key=lambda x: x[1], reverse=True)[:10]
        cascade_locations = {l['location'] for l in cascade_indicators.get('potential_cascade_locations', [])}
        
        results['top_locations'] = []
        for loc_key, count in top_locs:
            loc_info = locations_index.get(loc_key, {})
            results['top_locations'].append({
                'location': loc_key[4:],  # Remove 'LOC:' prefix
                'count': count,
                'occurrences': loc_info.get('occurrences', 0),
                'n_media': loc_info.get('n_media', 0),
                'media_concentration': loc_info.get('media_concentration', 0),
                'is_cascade_location': loc_key in cascade_locations
            })
        
        # Temporal focus indicators
        temporal_focus = self.geographic_index.get('temporal_focus', {})
        if temporal_focus:
            results['temporal_consistency'] = min(temporal_focus.get('max_sustained_days', 0) / 7, 1.0)
            results['n_sustained_focus_periods'] = temporal_focus.get('n_sustained_periods', 0)
        
        return results
    
    def calculate_temporal_diversity(self, 
                                   article_ids: List[str],
                                   window: str = 'daily') -> Dict[str, Any]:
        """
        Calculate temporal diversity of articles.
        
        Args:
            article_ids: List of article IDs
            window: 'daily', 'weekly', or 'monthly'
            
        Returns:
            Temporal diversity metrics
        """
        temporal_distribution = defaultdict(int)
        
        article_profiles = self.source_index.get('article_profiles', {})
        
        dates = []
        for doc_id in article_ids:
            if doc_id in article_profiles:
                date = article_profiles[doc_id].get('date')
                if date:
                    dates.append(pd.Timestamp(date))
                    
                    # Categorize by window
                    if window == 'daily':
                        key = date.date()
                    elif window == 'weekly':
                        key = pd.Timestamp(date).to_period('W')
                    else:  # monthly
                        key = pd.Timestamp(date).to_period('M')
                    
                    temporal_distribution[str(key)] += 1
        
        results = {
            'temporal_diversity': 0.0,
            'time_span_days': 0,
            'temporal_entropy': 0.0,
            'temporal_concentration': 0.0,
            'burst_ratio': 0.0
        }
        
        if dates:
            # Time span
            results['time_span_days'] = (max(dates) - min(dates)).days + 1
            
            # Temporal entropy
            if temporal_distribution:
                results['temporal_entropy'] = self._calculate_diversity(temporal_distribution, 'shannon')
                results['temporal_diversity'] = results['temporal_entropy']
                
                # Concentration
                concentration = self._calculate_concentration(temporal_distribution)
                results['temporal_concentration'] = concentration['hhi']
                
                # Burst ratio (how concentrated in time)
                max_period = max(temporal_distribution.values())
                total_periods = len(temporal_distribution)
                results['burst_ratio'] = max_period / len(article_ids) if article_ids else 0
                
                # Temporal regularity (coefficient of variation)
                period_counts = list(temporal_distribution.values())
                if len(period_counts) > 1:
                    results['temporal_regularity'] = 1 - (np.std(period_counts) / np.mean(period_counts))
                else:
                    results['temporal_regularity'] = 1.0
        
        return results
    
    def calculate_comprehensive_diversity(self, 
                                        article_ids: List[str],
                                        article_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate all diversity metrics comprehensively.
        
        Args:
            article_ids: List of article IDs
            article_data: Optional DataFrame for messenger analysis
            
        Returns:
            Comprehensive diversity report
        """
        logger.info(f"Calculating comprehensive diversity for {len(article_ids)} articles")
        
        diversity_report = {
            'frame': self.calculate_frame_diversity(article_ids),
            'source': self.calculate_source_diversity(article_ids),
            'emotion': self.calculate_emotional_diversity(article_ids),
            'geographic': self.calculate_geographic_diversity(article_ids),
            'temporal': self.calculate_temporal_diversity(article_ids)
        }
        
        # Use data-driven geographic analysis if geographic index available
        if self.geographic_index:
            diversity_report['geographic'] = self.calculate_geographic_diversity(
                article_ids, 
                use_geographic_index=True
            )
        else:
            diversity_report['geographic'] = self.calculate_geographic_diversity(
                article_ids,
                use_geographic_index=False
            )
        
        # Add viewpoint diversity if data available
        if article_data is not None and not article_data.empty:
            diversity_report['viewpoint'] = self.calculate_viewpoint_diversity(article_data)
        
        # Calculate overall diversity score (weighted average)
        weights = {
            'frame': 0.25,
            'source': 0.25,
            'emotion': 0.15,
            'geographic': 0.15,
            'temporal': 0.20
        }
        
        overall_score = 0
        for dimension, weight in weights.items():
            if dimension in diversity_report:
                # Get the main diversity metric for each dimension
                if dimension == 'frame':
                    score = diversity_report[dimension]
                elif dimension == 'source':
                    score = diversity_report[dimension].get('media_diversity', 0)
                elif dimension == 'emotion':
                    score = diversity_report[dimension].get('emotion_entropy', 0)
                elif dimension == 'geographic':
                    score = diversity_report[dimension].get('geographic_diversity', 0)
                elif dimension == 'temporal':
                    score = diversity_report[dimension].get('temporal_diversity', 0)
                else:
                    score = 0
                
                overall_score += score * weight
        
        diversity_report['overall_diversity_score'] = overall_score
        
        # Add interpretation
        if overall_score < 0.3:
            diversity_report['diversity_level'] = 'low'
        elif overall_score < 0.6:
            diversity_report['diversity_level'] = 'moderate'
        else:
            diversity_report['diversity_level'] = 'high'
        
        return diversity_report