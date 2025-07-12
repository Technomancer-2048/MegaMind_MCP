"""
Realm-Aware Vector Search Engine for MegaMind Context Database
Provides semantic search using embedding similarity with realm prioritization
"""

import os
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .embedding_service import get_embedding_service


class SearchType(Enum):
    """Search type enumeration"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword" 
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    """Container for search result with realm awareness"""
    chunk_id: str
    content: str
    source_document: str
    section_path: str
    realm_id: str
    similarity_score: float
    keyword_score: Optional[float] = None
    final_score: float = 0.0
    access_count: int = 0
    tags: Optional[List[str]] = None
    embedding: Optional[List[float]] = None


class RealmAwareVectorSearchEngine:
    """Realm-aware semantic search using embedding similarity"""
    
    def __init__(self, project_realm: str, global_realm: str = 'GLOBAL'):
        """
        Initialize realm-aware vector search engine
        
        Args:
            project_realm: Name of the project realm (from environment)
            global_realm: Name of the global realm (default: 'GLOBAL')
        """
        self.project_realm = project_realm
        self.global_realm = global_realm
        self.embedding_service = get_embedding_service()
        
        # Configuration from environment
        self.semantic_threshold = float(os.getenv('SEMANTIC_SEARCH_THRESHOLD', '0.7'))
        self.project_priority = float(os.getenv('REALM_PRIORITY_PROJECT', '1.2'))
        self.global_priority = float(os.getenv('REALM_PRIORITY_GLOBAL', '1.0'))
        self.cross_realm_enabled = os.getenv('CROSS_REALM_SEARCH_ENABLED', 'true').lower() == 'true'
        
        logging.info(f"Initialized RealmAwareVectorSearchEngine: project={project_realm}, global={global_realm}")
    
    def dual_realm_semantic_search(self, 
                                  query: str, 
                                  chunks_data: List[Dict[str, Any]], 
                                  limit: int = 10, 
                                  threshold: float = None) -> List[SearchResult]:
        """
        Primary semantic search across Global + Project realms
        
        Args:
            query: Search query text
            chunks_data: Raw chunk data from database
            limit: Maximum number of results
            threshold: Minimum similarity threshold (uses default if None)
            
        Returns:
            List of SearchResult objects ranked by relevance and realm priority
        """
        if not self.embedding_service.is_available():
            logging.warning("Embedding service not available, returning empty results")
            return []
        
        if threshold is None:
            threshold = self.semantic_threshold
        
        # Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(query)
        if query_embedding is None:
            logging.error("Failed to generate query embedding")
            return []
        
        # Calculate similarities and create search results
        results = []
        for chunk in chunks_data:
            # Skip chunks without embeddings
            if not chunk.get('embedding'):
                continue
            
            # Parse embedding from JSON string if needed
            chunk_embedding = self._parse_embedding(chunk['embedding'])
            if chunk_embedding is None:
                continue
            
            # Calculate realm-aware similarity
            similarity = self._calculate_realm_aware_similarity(
                query_embedding, 
                chunk_embedding, 
                chunk['realm_id']
            )
            
            # Apply threshold filter
            if similarity < threshold:
                continue
            
            # Create search result
            result = SearchResult(
                chunk_id=chunk['chunk_id'],
                content=chunk['content'],
                source_document=chunk['source_document'],
                section_path=chunk['section_path'],
                realm_id=chunk['realm_id'],
                similarity_score=similarity,
                final_score=similarity,
                access_count=chunk.get('access_count', 0),
                tags=chunk.get('tags', []),
                embedding=chunk_embedding
            )
            results.append(result)
        
        # Sort by final score (descending) and return top results
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:limit]
    
    def realm_aware_hybrid_search(self, 
                                 query: str, 
                                 chunks_data: List[Dict[str, Any]], 
                                 semantic_weight: float = 0.7,
                                 keyword_weight: float = 0.3,
                                 limit: int = 10) -> List[SearchResult]:
        """
        Combine semantic and keyword search with realm prioritization
        
        Args:
            query: Search query text
            chunks_data: Raw chunk data from database
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
            limit: Maximum number of results
            
        Returns:
            List of SearchResult objects with hybrid scoring
        """
        if not self.embedding_service.is_available():
            # Fallback to keyword-only search
            return self._keyword_only_search(query, chunks_data, limit)
        
        # Normalize weights
        total_weight = semantic_weight + keyword_weight
        if total_weight > 0:
            semantic_weight /= total_weight
            keyword_weight /= total_weight
        else:
            semantic_weight, keyword_weight = 0.7, 0.3
        
        # Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(query)
        if query_embedding is None:
            # Fallback to keyword-only search
            return self._keyword_only_search(query, chunks_data, limit)
        
        # Calculate hybrid scores
        results = []
        query_terms = query.lower().split()
        
        for chunk in chunks_data:
            # Calculate semantic score
            semantic_score = 0.0
            if chunk.get('embedding'):
                chunk_embedding = self._parse_embedding(chunk['embedding'])
                if chunk_embedding is not None:
                    semantic_score = self._calculate_cosine_similarity(query_embedding, chunk_embedding)
            
            # Calculate keyword score
            keyword_score = self._calculate_keyword_score(query_terms, chunk)
            
            # Combine scores
            combined_score = (semantic_score * semantic_weight) + (keyword_score * keyword_weight)
            
            # Apply realm priority
            final_score = self._apply_realm_priority(combined_score, chunk['realm_id'])
            
            # Skip low-scoring results
            if final_score < 0.1:
                continue
            
            # Create search result
            result = SearchResult(
                chunk_id=chunk['chunk_id'],
                content=chunk['content'],
                source_document=chunk['source_document'],
                section_path=chunk['section_path'],
                realm_id=chunk['realm_id'],
                similarity_score=semantic_score,
                keyword_score=keyword_score,
                final_score=final_score,
                access_count=chunk.get('access_count', 0),
                tags=chunk.get('tags', [])
            )
            results.append(result)
        
        # Sort by final score and return top results
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:limit]
    
    def find_similar_chunks(self, 
                           reference_chunk: Dict[str, Any], 
                           chunks_data: List[Dict[str, Any]], 
                           limit: int = 10,
                           threshold: float = None) -> List[SearchResult]:
        """
        Find chunks similar to a reference chunk using embeddings
        
        Args:
            reference_chunk: Reference chunk data with embedding
            chunks_data: Raw chunk data from database to search
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of SearchResult objects similar to reference
        """
        if not self.embedding_service.is_available():
            return []
        
        if threshold is None:
            threshold = self.semantic_threshold
        
        # Get reference embedding
        reference_embedding = self._parse_embedding(reference_chunk.get('embedding'))
        if reference_embedding is None:
            return []
        
        reference_chunk_id = reference_chunk['chunk_id']
        
        # Calculate similarities
        results = []
        for chunk in chunks_data:
            # Skip the reference chunk itself
            if chunk['chunk_id'] == reference_chunk_id:
                continue
            
            # Skip chunks without embeddings
            if not chunk.get('embedding'):
                continue
            
            chunk_embedding = self._parse_embedding(chunk['embedding'])
            if chunk_embedding is None:
                continue
            
            # Calculate realm-aware similarity
            similarity = self._calculate_realm_aware_similarity(
                reference_embedding,
                chunk_embedding,
                chunk['realm_id']
            )
            
            # Apply threshold
            if similarity < threshold:
                continue
            
            # Create search result
            result = SearchResult(
                chunk_id=chunk['chunk_id'],
                content=chunk['content'],
                source_document=chunk['source_document'],
                section_path=chunk['section_path'],
                realm_id=chunk['realm_id'],
                similarity_score=similarity,
                final_score=similarity,
                access_count=chunk.get('access_count', 0),
                tags=chunk.get('tags', [])
            )
            results.append(result)
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:limit]
    
    def _calculate_realm_aware_similarity(self, 
                                        query_embedding: List[float], 
                                        chunk_embedding: List[float], 
                                        chunk_realm: str) -> float:
        """
        Calculate cosine similarity with realm-based weighting
        
        Args:
            query_embedding: Query vector
            chunk_embedding: Chunk vector
            chunk_realm: Realm ID of the chunk
            
        Returns:
            Realm-weighted similarity score
        """
        # Calculate base cosine similarity
        base_similarity = self._calculate_cosine_similarity(query_embedding, chunk_embedding)
        
        # Apply realm priority weighting
        weighted_similarity = self._apply_realm_priority(base_similarity, chunk_realm)
        
        return weighted_similarity
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        # Avoid division by zero
        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, similarity))
    
    def _apply_realm_priority(self, base_score: float, chunk_realm: str) -> float:
        """Apply realm-based priority weighting to score"""
        if chunk_realm == self.project_realm:
            return base_score * self.project_priority
        elif chunk_realm == self.global_realm:
            return base_score * self.global_priority
        else:
            # Unknown realm gets lower priority
            return base_score * 0.8
    
    def _calculate_keyword_score(self, query_terms: List[str], chunk: Dict[str, Any]) -> float:
        """Calculate keyword matching score"""
        if not query_terms:
            return 0.0
        
        content = chunk.get('content', '').lower()
        source_doc = chunk.get('source_document', '').lower()
        section_path = chunk.get('section_path', '').lower()
        
        # Count term matches with different weights
        matches = 0
        total_terms = len(query_terms)
        
        for term in query_terms:
            term_score = 0.0
            
            # Content matches (highest weight)
            if term in content:
                term_score += 1.0
            
            # Source document matches (medium weight)
            if term in source_doc:
                term_score += 0.5
            
            # Section path matches (lower weight)
            if term in section_path:
                term_score += 0.3
            
            matches += min(term_score, 1.0)  # Cap at 1.0 per term
        
        return matches / total_terms if total_terms > 0 else 0.0
    
    def _keyword_only_search(self, query: str, chunks_data: List[Dict[str, Any]], limit: int) -> List[SearchResult]:
        """Fallback keyword-only search when embeddings unavailable"""
        query_terms = query.lower().split()
        results = []
        
        for chunk in chunks_data:
            keyword_score = self._calculate_keyword_score(query_terms, chunk)
            
            if keyword_score > 0.0:
                final_score = self._apply_realm_priority(keyword_score, chunk['realm_id'])
                
                result = SearchResult(
                    chunk_id=chunk['chunk_id'],
                    content=chunk['content'],
                    source_document=chunk['source_document'],
                    section_path=chunk['section_path'],
                    realm_id=chunk['realm_id'],
                    similarity_score=0.0,
                    keyword_score=keyword_score,
                    final_score=final_score,
                    access_count=chunk.get('access_count', 0),
                    tags=chunk.get('tags', [])
                )
                results.append(result)
        
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:limit]
    
    def _parse_embedding(self, embedding_data: Any) -> Optional[List[float]]:
        """Parse embedding from various formats (JSON string, list, etc.)"""
        if embedding_data is None:
            return None
        
        if isinstance(embedding_data, list):
            return embedding_data
        
        if isinstance(embedding_data, str):
            try:
                import json
                return json.loads(embedding_data)
            except (json.JSONDecodeError, ValueError):
                return None
        
        return None
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            'project_realm': self.project_realm,
            'global_realm': self.global_realm,
            'semantic_threshold': self.semantic_threshold,
            'project_priority': self.project_priority,
            'global_priority': self.global_priority,
            'cross_realm_enabled': self.cross_realm_enabled,
            'embedding_service_available': self.embedding_service.is_available()
        }