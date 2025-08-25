"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
messenger_ner_cross.py

MAIN OBJECTIVE:
---------------
This script cross-references messenger types with Named Entity Recognition (NER) entities to provide
precise source identification, determining which specific individuals or organizations are behind
messenger categories like "health expert" or "public official".

Dependencies:
-------------
- pandas
- numpy
- typing
- collections
- json
- logging
- concurrent.futures
- tqdm
- multiprocessing
- networkx

MAIN FEATURES:
--------------
1) Cross-reference messenger types with NER entities (PER, ORG)
2) Entity disambiguation using context and co-mentions
3) Authority scoring based on messenger type and frequency
4) Parallel processing for large-scale cross-referencing
5) Network analysis of messenger-entity relationships

Author:
-------
Antoine Lemor
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp
import networkx as nx

from cascade_detector.core.constants import MESSENGER_TYPES, MESSENGER_MAIN, MESSENGERS
from cascade_detector.utils.entity_resolver_fast import FastEntityResolver

logger = logging.getLogger(__name__)


class MessengerNERCrossReference:
    """
    Cross-references messenger types with NER entities to identify
    specific sources (e.g., which health expert, which public official).
    """
    
    def __init__(self, n_workers: int = 16):
        """Initialize cross-referencer.
        
        Args:
            n_workers: Number of parallel workers for entity resolution
        """
        self.messenger_types = MESSENGER_TYPES
        self.messenger_main = MESSENGER_MAIN
        
        # Initialize fast entity resolver with intelligent merging
        self.entity_resolver = FastEntityResolver(
            similarity_threshold=0.85,
            n_workers=n_workers
        )
        
        # Track entity history for disambiguation
        self.entity_history = defaultdict(lambda: {
            'messenger_types': defaultdict(int),
            'co_mentions': defaultdict(int),
            'contexts': []
        })
        
    def cross_reference_sentence(self, 
                                sentence_data: pd.Series,
                                ner_entities: Optional[str]) -> Dict:
        """
        Cross-reference messengers and NER entities for a single sentence.
        
        Args:
            sentence_data: Row from dataframe with messenger columns
            ner_entities: JSON string of NER entities
            
        Returns:
            Dictionary with identified sources
        """
        result = {
            'has_messenger': False,
            'messenger_types': [],
            'entities': {},
            'identified_sources': []
        }
        
        # Check if sentence has messengers
        if self.messenger_main in sentence_data:
            has_messenger = pd.to_numeric(sentence_data[self.messenger_main], errors='coerce')
            result['has_messenger'] = bool(has_messenger == 1)
        
        if not result['has_messenger']:
            return result
        
        # Identify messenger types
        for messenger_col, messenger_type in self.messenger_types.items():
            if messenger_col in sentence_data:
                value = pd.to_numeric(sentence_data[messenger_col], errors='coerce')
                if value == 1:
                    result['messenger_types'].append(messenger_type)
        
        # Parse NER entities if available
        if ner_entities and pd.notna(ner_entities):
            try:
                entities = json.loads(ner_entities) if isinstance(ner_entities, str) else ner_entities
                result['entities'] = entities
                
                # Cross-reference to identify specific sources
                result['identified_sources'] = self._identify_sources(
                    result['messenger_types'],
                    entities
                )
            except (json.JSONDecodeError, TypeError):
                pass
        
        return result
    
    def _identify_sources(self, 
                         messenger_types: List[str],
                         entities: Dict[str, List[str]]) -> List[Dict]:
        """
        Identify specific sources by matching messenger types with entities.
        Uses contextual validation rather than fixed confidence scores.
        
        Args:
            messenger_types: List of active messenger types
            entities: NER entities (PER, ORG, LOC)
            
        Returns:
            List of identified sources with type and name
        """
        identified = []
        
        # Get person and organization entities
        persons = entities.get('PER', [])
        organizations = entities.get('ORG', [])
        
        for msg_type in messenger_types:
            # Match messenger types with entity types based on context
            if msg_type in ['health_expertise', 'economic_expertise', 
                           'security_expertise', 'law_expertise', 
                           'culture_expertise']:
                # Experts can be persons or validated organizations
                for person in persons:
                    identified.append({
                        'type': msg_type,
                        'entity_type': 'PER',
                        'name': person,
                        'validated': True  # Person in expert context
                    })
                for org in organizations:
                    if self._is_expert_org(org, msg_type):
                        identified.append({
                            'type': msg_type,
                            'entity_type': 'ORG',
                            'name': org,
                            'validated': True  # Contextually validated
                        })
            
            elif msg_type in ['hard_science', 'social_science']:
                # Scientists and research institutions
                for person in persons:
                    identified.append({
                        'type': msg_type,
                        'entity_type': 'PER',
                        'name': person,
                        'validated': True
                    })
                for org in organizations:
                    if self._is_research_org(org):
                        identified.append({
                            'type': msg_type,
                            'entity_type': 'ORG',
                            'name': org,
                            'validated': True
                        })
            
            elif msg_type == 'activist':
                # Activists can be persons or organizations
                for person in persons:
                    identified.append({
                        'type': msg_type,
                        'entity_type': 'PER',
                        'name': person,
                        'validated': True
                    })
                for org in organizations:
                    if self._is_activist_org(org):
                        identified.append({
                            'type': msg_type,
                            'entity_type': 'ORG',
                            'name': org,
                            'validated': True
                        })
            
            elif msg_type == 'public_official':
                # Public officials and government bodies
                for person in persons:
                    identified.append({
                        'type': msg_type,
                        'entity_type': 'PER',
                        'name': person,
                        'validated': True
                    })
                for org in organizations:
                    if self._is_government_org(org):
                        identified.append({
                            'type': msg_type,
                            'entity_type': 'ORG',
                            'name': org,
                            'validated': True
                        })
        
        return identified
    
    def _is_expert_org(self, org_name: str, expertise_type: str) -> bool:
        """Check if organization matches expertise type."""
        org_lower = org_name.lower()
        
        if expertise_type == 'health_expertise':
            keywords = ['health', 'medical', 'hospital', 'clinic', 'pharma', 
                       'disease', 'institute', 'research']
        elif expertise_type == 'economic_expertise':
            keywords = ['bank', 'economic', 'finance', 'monetary', 'trade', 
                       'commerce', 'business', 'market']
        elif expertise_type == 'security_expertise':
            keywords = ['security', 'defense', 'military', 'police', 'intelligence']
        elif expertise_type == 'law_expertise':
            keywords = ['law', 'legal', 'court', 'justice', 'attorney', 'judicial']
        elif expertise_type == 'culture_expertise':
            keywords = ['culture', 'art', 'museum', 'heritage', 'academy']
        else:
            return False
        
        return any(keyword in org_lower for keyword in keywords)
    
    def _is_research_org(self, org_name: str) -> bool:
        """Check if organization is a research institution."""
        keywords = ['university', 'institute', 'research', 'laboratory', 
                   'college', 'academy', 'center', 'centre']
        return any(keyword in org_name.lower() for keyword in keywords)
    
    def _is_activist_org(self, org_name: str) -> bool:
        """Check if organization is an activist group."""
        keywords = ['foundation', 'association', 'coalition', 'alliance',
                   'network', 'movement', 'action', 'watch', 'advocacy']
        return any(keyword in org_name.lower() for keyword in keywords)
    
    def _is_government_org(self, org_name: str) -> bool:
        """Check if organization is a government body."""
        keywords = ['government', 'ministry', 'department', 'administration',
                   'parliament', 'congress', 'senate', 'council', 'commission',
                   'agency', 'authority', 'bureau']
        return any(keyword in org_name.lower() for keyword in keywords)
    
    def _disambiguate_entity(self, entity_name: str, entity_type: str, 
                            messenger_types: List[str], 
                            co_entities: List[str]) -> Dict[str, Any]:
        """
        Disambiguate entity role using contextual information.
        
        Args:
            entity_name: Name of the entity
            entity_type: PER or ORG
            messenger_types: Messenger types detected in the same context
            co_entities: Other entities mentioned together
            
        Returns:
            Disambiguation information including most likely role
        """
        # Update entity history
        for msg_type in messenger_types:
            self.entity_history[entity_name]['messenger_types'][msg_type] += 1
        
        for co_entity in co_entities:
            if co_entity != entity_name:
                self.entity_history[entity_name]['co_mentions'][co_entity] += 1
        
        # Analyze historical patterns
        history = self.entity_history[entity_name]
        
        # Find dominant messenger type based on frequency
        if history['messenger_types']:
            dominant_type = max(history['messenger_types'].items(), 
                              key=lambda x: x[1])[0]
            consistency = history['messenger_types'][dominant_type] / sum(history['messenger_types'].values())
        else:
            dominant_type = None
            consistency = 0
        
        # Analyze co-mention patterns
        frequent_companions = []
        if history['co_mentions']:
            # Get top 3 most frequent co-mentions
            sorted_mentions = sorted(history['co_mentions'].items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
            frequent_companions = [entity for entity, count in sorted_mentions]
        
        return {
            'entity_name': entity_name,
            'entity_type': entity_type,
            'dominant_role': dominant_type,
            'role_consistency': consistency,
            'frequent_companions': frequent_companions,
            'total_mentions': sum(history['messenger_types'].values())
        }
    
    def analyze_article(self, article_data: pd.DataFrame) -> Dict:
        """
        Analyze all messengers and sources in an article.
        
        Args:
            article_data: DataFrame with article sentences
            
        Returns:
            Dictionary with messenger and source analysis
        """
        results = {
            'doc_id': article_data['doc_id'].iloc[0] if 'doc_id' in article_data.columns else None,
            'total_sentences': len(article_data),
            'messenger_sentences': 0,
            'identified_sources': defaultdict(list),
            'source_diversity': 0,
            'dominant_source_type': None,
            'unique_persons': set(),
            'unique_organizations': set()
        }
        
        # Analyze each sentence
        for idx, row in article_data.iterrows():
            ner_entities = row.get('ner_entities')
            cross_ref = self.cross_reference_sentence(row, ner_entities)
            
            if cross_ref['has_messenger']:
                results['messenger_sentences'] += 1
                
                # Collect identified sources
                for source in cross_ref['identified_sources']:
                    results['identified_sources'][source['type']].append(source)
                    
                    # Track unique entities
                    if source['entity_type'] == 'PER':
                        results['unique_persons'].add(source['name'])
                    elif source['entity_type'] == 'ORG':
                        results['unique_organizations'].add(source['name'])
            
            # Also collect ALL organizations from raw NER (not just identified sources)
            # Organizations are messengers too, even if not matched to specific types
            entities = row.get('ner_entities')
            if entities and pd.notna(entities):
                try:
                    parsed = json.loads(entities) if isinstance(entities, str) else entities
                    # Add all organizations as potential messengers
                    for org in parsed.get('ORG', []):
                        results['unique_organizations'].add(org)
                    # Keep tracking persons too
                    for per in parsed.get('PER', []):
                        results['unique_persons'].add(per)
                except (json.JSONDecodeError, TypeError):
                    pass
        
        # Calculate statistics
        results['unique_persons'] = list(results['unique_persons'])
        results['unique_organizations'] = list(results['unique_organizations'])
        results['identified_sources'] = dict(results['identified_sources'])
        
        # Find dominant source type
        if results['identified_sources']:
            source_counts = {k: len(v) for k, v in results['identified_sources'].items()}
            results['dominant_source_type'] = max(source_counts, key=source_counts.get)
        
        # Calculate source diversity
        total_sources = len(results['unique_persons']) + len(results['unique_organizations'])
        if total_sources > 0:
            # Shannon entropy over source types
            type_counts = [len(v) for v in results['identified_sources'].values()]
            if type_counts:
                proportions = np.array(type_counts) / sum(type_counts)
                proportions = proportions[proportions > 0]
                results['source_diversity'] = float(-np.sum(proportions * np.log(proportions)))
        
        return results
    
    def analyze_articles_parallel(self, 
                                grouped_data: pd.core.groupby.DataFrameGroupBy,
                                n_workers: Optional[int] = None,
                                show_progress: bool = True) -> List[Dict]:
        """
        Analyze multiple articles in parallel for messenger-entity relationships.
        Optimized for M4 Max with maximum CPU utilization.
        
        Args:
            grouped_data: DataFrame grouped by doc_id
            n_workers: Number of parallel workers (None for auto-detection)
            show_progress: Whether to show progress bar
            
        Returns:
            List of article analysis results
        """
        if n_workers is None:
            # Use all cores for M4 Max optimization
            n_workers = min(mp.cpu_count(), 16)
        
        # Convert groups to list for processing
        articles_data = [(doc_id, group) for doc_id, group in grouped_data]
        total_articles = len(articles_data)
        
        if total_articles == 0:
            return []
        
        logger.info(f"Analyzing {total_articles} articles in parallel with {n_workers} workers")
        
        # Function to process a single article
        def process_article(args):
            doc_id, article_data = args
            return self.analyze_article(article_data)
        
        article_analyses = []
        
        # Use ThreadPoolExecutor for I/O-bound operations (article analysis)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_article, article_data): idx 
                for idx, article_data in enumerate(articles_data)
            }
            
            # Process results with progress bar
            if show_progress:
                with tqdm(total=total_articles, desc="  Analyzing articles", unit="article", leave=False) as pbar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            article_analyses.append(result)
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Error analyzing article: {e}")
                            pbar.update(1)
            else:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        article_analyses.append(result)
                    except Exception as e:
                        logger.error(f"Error analyzing article: {e}")
        
        logger.info(f"Analyzed {len(article_analyses)} articles successfully")
        return article_analyses
    
    def find_epistemic_authorities(self, 
                                  articles: List[Dict],
                                  min_citations: int = 5) -> List[Dict]:
        """
        Find the most cited epistemic authorities across articles.
        Uses intelligent entity resolution to merge variants.
        
        Args:
            articles: List of article analysis results
            min_citations: Minimum citations to be considered an authority
            
        Returns:
            List of authorities with citation counts and contexts
        """
        # First, build entity index for resolution
        entity_index = {}
        
        for article in articles:
            doc_id = article.get('doc_id')
            if not doc_id:
                continue
            
            # Build index for persons
            for person in article.get('unique_persons', []):
                key = f"PER:{person}"
                if key not in entity_index:
                    entity_index[key] = {
                        'name': person,
                        'type': 'PER',
                        'occurrences': 0,
                        'articles': []
                    }
                entity_index[key]['occurrences'] += 1
                entity_index[key]['articles'].append(doc_id)
            
            # Build index for organizations
            for org in article.get('unique_organizations', []):
                key = f"ORG:{org}"
                if key not in entity_index:
                    entity_index[key] = {
                        'name': org,
                        'type': 'ORG',
                        'occurrences': 0,
                        'articles': []
                    }
                entity_index[key]['occurrences'] += 1
                entity_index[key]['articles'].append(doc_id)
        
        # Resolve entities using fast resolver
        resolved_entities = self.entity_resolver.resolve_entities(entity_index)
        
        # Build canonical mapping: maps all entity keys (original and merged) to canonical keys
        canonical_map = {}
        for canonical_key, data in resolved_entities.items():
            # Map canonical to itself
            canonical_map[canonical_key] = canonical_key
            # Map all merged entities to canonical
            if 'merged_from' in data:
                for merged in data['merged_from']:
                    canonical_map[merged['key']] = canonical_key
        
        # Now build authority citations using resolved entities
        authority_citations = defaultdict(lambda: {
            'count': 0,
            'articles': [],
            'messenger_types': set(),
            'entity_type': None,
            'variants': set()
        })
        
        for article in articles:
            doc_id = article.get('doc_id')
            
            # Process persons with resolution
            for person in article.get('unique_persons', []):
                # Get canonical form using the mapping
                key = f"PER:{person}"
                canonical_key = canonical_map.get(key, key)
                
                # Use the canonical KEY as the authority key to prevent type mixing
                # This ensures PER:Ford and ORG:Ford remain separate
                authority_key = canonical_key
                
                # Get the display name from the resolved entity
                if canonical_key in resolved_entities:
                    display_name = resolved_entities[canonical_key].get('name', person)
                else:
                    display_name = person
                
                authority_citations[authority_key]['count'] += 1
                authority_citations[authority_key]['articles'].append(doc_id)
                authority_citations[authority_key]['entity_type'] = 'PER'
                authority_citations[authority_key]['display_name'] = display_name
                authority_citations[authority_key]['variants'].add(person)
                
                # Add messenger types
                for msg_type, sources in article.get('identified_sources', {}).items():
                    for source in sources:
                        if source['name'] == person:
                            authority_citations[authority_key]['messenger_types'].add(msg_type)
            
            # Process organizations with resolution
            for org in article.get('unique_organizations', []):
                # Get canonical form using the mapping
                key = f"ORG:{org}"
                canonical_key = canonical_map.get(key, key)
                
                # Use the canonical KEY as the authority key to prevent type mixing
                # This ensures PER:Ford and ORG:Ford remain separate
                authority_key = canonical_key
                
                # Get the display name from the resolved entity
                if canonical_key in resolved_entities:
                    display_name = resolved_entities[canonical_key].get('name', org)
                else:
                    display_name = org
                
                authority_citations[authority_key]['count'] += 1
                authority_citations[authority_key]['articles'].append(doc_id)
                authority_citations[authority_key]['entity_type'] = 'ORG'
                authority_citations[authority_key]['display_name'] = display_name
                authority_citations[authority_key]['variants'].add(org)
                
                # Add messenger types
                for msg_type, sources in article.get('identified_sources', {}).items():
                    for source in sources:
                        if source['name'] == org:
                            authority_citations[authority_key]['messenger_types'].add(msg_type)
        
        # Filter and format authorities using disambiguation
        authorities = []
        for name, data in authority_citations.items():
            if data['count'] >= min_citations:
                # Get disambiguation info for this entity
                disambiguation = self.entity_history.get(name, {})
                
                # Calculate authority based on multiple factors (not fixed confidence)
                # Authority = citations × role_diversity × consistency
                role_diversity = len(data['messenger_types']) if data['messenger_types'] else 1
                
                # Get role consistency from history
                if disambiguation and 'messenger_types' in disambiguation:
                    type_counts = disambiguation['messenger_types']
                    if type_counts:
                        dominant_count = max(type_counts.values())
                        total_count = sum(type_counts.values())
                        consistency = dominant_count / total_count if total_count > 0 else 0
                    else:
                        consistency = 0.5  # neutral if no history
                else:
                    consistency = 0.5  # neutral if no history
                
                # Authority score based on actual patterns, not arbitrary confidence
                authority_score = data['count'] * role_diversity * (0.5 + consistency)
                
                authorities.append({
                    'name': data.get('display_name', name.split(':', 1)[1] if ':' in name else name),
                    'entity_type': data['entity_type'],
                    'citation_count': data['count'],
                    'n_articles': len(set(data['articles'])),
                    'messenger_types': list(data['messenger_types']),
                    'role_consistency': consistency,
                    'role_diversity': role_diversity,
                    'authority_score': authority_score,
                    'variants': list(data['variants']) if data['variants'] else [data.get('display_name', name)]
                })
        
        # Sort by authority score
        authorities.sort(key=lambda x: x['authority_score'], reverse=True)
        
        return authorities
    
    def build_authority_network_vectorized(self, 
                                          authorities: List[Dict], 
                                          article_analyses: List[Dict],
                                          top_k: int = 50) -> nx.Graph:
        """
        Build co-mention network of authorities using vectorized operations.
        Optimized for speed with batch processing and numpy operations.
        
        Args:
            authorities: List of authorities with metadata
            article_analyses: List of article analysis results
            top_k: Number of top authorities to include in network
            
        Returns:
            NetworkX graph with authority co-mention relationships
        """
        from collections import defaultdict
        
        G = nx.Graph()
        
        # Limit to top authorities for visualization
        top_authorities = authorities[:top_k]
        
        # Add nodes with attributes
        for auth in top_authorities:
            # Convert messenger_types list to string for GEXF compatibility
            messenger_types_str = ', '.join(auth.get('messenger_types', [])) if auth.get('messenger_types') else ''
            
            G.add_node(
                auth['name'],
                entity_type=auth['entity_type'],
                citation_count=auth['citation_count'],
                authority_score=auth['authority_score'],
                messenger_types=messenger_types_str  # Store as string instead of list
            )
        
        # Create authority name set for fast lookup
        authority_names = {auth['name'] for auth in top_authorities}
        
        # Use vectorized approach to build co-mention counts
        co_mention_counts = defaultdict(int)
        
        # Process articles in batches for better cache efficiency
        batch_size = 100
        for i in range(0, len(article_analyses), batch_size):
            batch = article_analyses[i:i+batch_size]
            
            for analysis in batch:
                # Get all entities in this article
                entities = set()
                entities.update(analysis.get('unique_persons', []))
                entities.update(analysis.get('unique_organizations', []))
                
                # Find authorities mentioned in this article
                mentioned_authorities = entities.intersection(authority_names)
                
                # Convert to list for indexing
                mentioned_list = list(mentioned_authorities)
                n = len(mentioned_list)
                
                # Vectorized edge counting using combinations
                if n > 1:
                    # Generate all pairs efficiently
                    for i in range(n):
                        for j in range(i+1, n):
                            auth1, auth2 = mentioned_list[i], mentioned_list[j]
                            # Ensure consistent ordering for undirected graph
                            if auth1 > auth2:
                                auth1, auth2 = auth2, auth1
                            co_mention_counts[(auth1, auth2)] += 1
        
        # Add edges in batch (more efficient than one-by-one)
        edges_to_add = []
        for (auth1, auth2), weight in co_mention_counts.items():
            edges_to_add.append((auth1, auth2, {'weight': weight}))
        
        G.add_edges_from(edges_to_add)
        
        logger.info(f"Built network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def analyze_co_mention_networks(self, articles: List[Dict]) -> Dict[str, Any]:
        """
        Analyze co-mention patterns to understand entity relationships.
        This helps with disambiguation by revealing typical entity clusters.
        
        Args:
            articles: List of article analysis results
            
        Returns:
            Co-mention network analysis
        """
        co_mention_graph = defaultdict(lambda: defaultdict(int))
        entity_contexts = defaultdict(set)
        
        for article in articles:
            # Get all entities in the article
            all_entities = []
            all_entities.extend(article.get('unique_persons', []))
            all_entities.extend(article.get('unique_organizations', []))
            
            # Build co-mention graph
            for i, entity1 in enumerate(all_entities):
                for entity2 in all_entities[i+1:]:
                    co_mention_graph[entity1][entity2] += 1
                    co_mention_graph[entity2][entity1] += 1
                
                # Track contexts (messenger types) for each entity
                for msg_type, sources in article.get('identified_sources', {}).items():
                    for source in sources:
                        if source['name'] == entity1:
                            entity_contexts[entity1].add(msg_type)
        
        # Identify entity clusters based on co-mentions
        clusters = self._identify_entity_clusters(co_mention_graph)
        
        # Calculate network metrics
        network_metrics = {
            'total_entities': len(co_mention_graph),
            'total_co_mentions': sum(sum(mentions.values()) for mentions in co_mention_graph.values()) // 2,
            'clusters': clusters,
            'entity_contexts': {k: list(v) for k, v in entity_contexts.items()},
            'hub_entities': self._identify_hub_entities(co_mention_graph)
        }
        
        return network_metrics
    
    def _identify_entity_clusters(self, co_mention_graph: Dict) -> List[Dict]:
        """
        Identify clusters of entities that are frequently mentioned together.
        These clusters often share similar roles or contexts.
        """
        clusters = []
        processed = set()
        
        for entity, co_mentions in co_mention_graph.items():
            if entity not in processed:
                # Find entities strongly connected to this one
                cluster_members = [entity]
                processed.add(entity)
                
                # Add entities with strong co-mention relationships
                for co_entity, count in co_mentions.items():
                    if count >= 3 and co_entity not in processed:  # Threshold for strong relationship
                        cluster_members.append(co_entity)
                        processed.add(co_entity)
                
                if len(cluster_members) > 1:
                    clusters.append({
                        'members': cluster_members,
                        'size': len(cluster_members),
                        'cohesion': self._calculate_cluster_cohesion(cluster_members, co_mention_graph)
                    })
        
        return sorted(clusters, key=lambda x: x['size'], reverse=True)
    
    def _calculate_cluster_cohesion(self, members: List[str], co_mention_graph: Dict) -> float:
        """
        Calculate how tightly connected cluster members are.
        Higher cohesion means entities are more consistently mentioned together.
        """
        if len(members) < 2:
            return 1.0
        
        total_possible = len(members) * (len(members) - 1) / 2
        actual_connections = 0
        
        for i, entity1 in enumerate(members):
            for entity2 in members[i+1:]:
                if entity2 in co_mention_graph.get(entity1, {}):
                    actual_connections += 1
        
        return actual_connections / total_possible if total_possible > 0 else 0
    
    def _identify_hub_entities(self, co_mention_graph: Dict, top_n: int = 10) -> List[Dict]:
        """
        Identify hub entities that are frequently mentioned with many others.
        These are often key authorities or central figures.
        """
        hub_scores = {}
        
        for entity, co_mentions in co_mention_graph.items():
            # Hub score = number of connections × total co-mention strength
            n_connections = len(co_mentions)
            total_strength = sum(co_mentions.values())
            hub_scores[entity] = n_connections * np.log1p(total_strength)
        
        # Get top hubs
        top_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return [
            {
                'entity': entity,
                'hub_score': score,
                'n_connections': len(co_mention_graph.get(entity, {})),
                'total_co_mentions': sum(co_mention_graph.get(entity, {}).values())
            }
            for entity, score in top_hubs
        ]