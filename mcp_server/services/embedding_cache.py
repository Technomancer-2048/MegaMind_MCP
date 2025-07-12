#!/usr/bin/env python3
"""
Phase 4 Performance Optimization: Embedding Cache
LRU cache for frequently accessed embeddings with content-based deduplication
"""

import hashlib
import time
import logging
from typing import Optional, List, Dict, Any
from collections import OrderedDict
from threading import Lock
import json

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """
    LRU cache for embedding vectors with content-based deduplication.
    Thread-safe implementation for concurrent MCP server access.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize embedding cache with LRU eviction policy.
        
        Args:
            max_size: Maximum number of embeddings to cache
            ttl_seconds: Time-to-live for cached embeddings in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.info(f"Initialized EmbeddingCache: max_size={max_size}, ttl={ttl_seconds}s")
    
    def _generate_content_hash(self, text: str, realm_context: Optional[str] = None) -> str:
        """
        Generate deterministic hash for text content with optional realm context.
        
        Args:
            text: Input text content
            realm_context: Optional realm context for differentiation
            
        Returns:
            SHA-256 hash of normalized content
        """
        # Normalize text for consistent hashing
        normalized_text = text.strip().lower()
        
        # Include realm context if provided
        if realm_context:
            content = f"{normalized_text}|realm:{realm_context}"
        else:
            content = normalized_text
        
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get_embedding(self, text: str, realm_context: Optional[str] = None) -> Optional[List[float]]:
        """
        Retrieve cached embedding by content hash.
        
        Args:
            text: Input text content
            realm_context: Optional realm context
            
        Returns:
            Cached embedding vector or None if not found/expired
        """
        content_hash = self._generate_content_hash(text, realm_context)
        
        with self._lock:
            if content_hash not in self._cache:
                self._misses += 1
                return None
            
            cached_item = self._cache[content_hash]
            current_time = time.time()
            
            # Check TTL expiration
            if current_time - cached_item['timestamp'] > self.ttl_seconds:
                del self._cache[content_hash]
                self._misses += 1
                logger.debug(f"Cache entry expired for hash {content_hash[:12]}...")
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(content_hash)
            self._hits += 1
            
            embedding = cached_item['embedding']
            logger.debug(f"Cache hit for hash {content_hash[:12]}... (size: {len(embedding)})")
            return embedding
    
    def store_embedding(self, text: str, embedding: List[float], realm_context: Optional[str] = None):
        """
        Store embedding with content hash key and LRU eviction.
        
        Args:
            text: Input text content
            embedding: Generated embedding vector
            realm_context: Optional realm context
        """
        if not embedding:
            return
        
        content_hash = self._generate_content_hash(text, realm_context)
        current_time = time.time()
        
        with self._lock:
            # Store embedding with metadata
            self._cache[content_hash] = {
                'embedding': embedding,
                'timestamp': current_time,
                'text_length': len(text),
                'realm_context': realm_context
            }
            
            # Move to end (most recently used)
            self._cache.move_to_end(content_hash)
            
            # Evict oldest entries if over capacity
            while len(self._cache) > self.max_size:
                oldest_hash, oldest_item = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug(f"Evicted oldest entry: {oldest_hash[:12]}...")
            
            logger.debug(f"Stored embedding for hash {content_hash[:12]}... (size: {len(embedding)})")
    
    def batch_get_embeddings(self, texts: List[str], realm_context: Optional[str] = None) -> Dict[str, Optional[List[float]]]:
        """
        Retrieve multiple embeddings in batch operation.
        
        Args:
            texts: List of input text content
            realm_context: Optional realm context for all texts
            
        Returns:
            Dictionary mapping text to cached embedding (None if not cached)
        """
        results = {}
        
        for text in texts:
            embedding = self.get_embedding(text, realm_context)
            results[text] = embedding
        
        return results
    
    def batch_store_embeddings(self, text_embeddings: Dict[str, List[float]], realm_context: Optional[str] = None):
        """
        Store multiple embeddings in batch operation.
        
        Args:
            text_embeddings: Dictionary mapping text to embedding vector
            realm_context: Optional realm context for all texts
        """
        for text, embedding in text_embeddings.items():
            self.store_embedding(text, embedding, realm_context)
    
    def invalidate_realm(self, realm_context: str):
        """
        Remove all cached embeddings for a specific realm.
        
        Args:
            realm_context: Realm context to invalidate
        """
        with self._lock:
            keys_to_remove = []
            
            for key, item in self._cache.items():
                if item.get('realm_context') == realm_context:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._cache[key]
            
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries for realm: {realm_context}")
    
    def clear_expired(self):
        """Remove all expired entries from cache."""
        current_time = time.time()
        
        with self._lock:
            keys_to_remove = []
            
            for key, item in self._cache.items():
                if current_time - item['timestamp'] > self.ttl_seconds:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._cache[key]
            
            if keys_to_remove:
                logger.info(f"Cleared {len(keys_to_remove)} expired cache entries")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'cache_size': len(self._cache),
                'max_size': self.max_size,
                'utilization_percent': (len(self._cache) / self.max_size * 100),
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'hit_rate_percent': hit_rate,
                'total_requests': total_requests
            }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get detailed cache information for debugging.
        
        Returns:
            Dictionary with detailed cache information
        """
        with self._lock:
            current_time = time.time()
            realm_distribution = {}
            size_distribution = {'small': 0, 'medium': 0, 'large': 0}
            
            for item in self._cache.values():
                # Realm distribution
                realm = item.get('realm_context', 'unknown')
                realm_distribution[realm] = realm_distribution.get(realm, 0) + 1
                
                # Size distribution
                text_length = item.get('text_length', 0)
                if text_length < 100:
                    size_distribution['small'] += 1
                elif text_length < 500:
                    size_distribution['medium'] += 1
                else:
                    size_distribution['large'] += 1
            
            return {
                'cache_size': len(self._cache),
                'realm_distribution': realm_distribution,
                'size_distribution': size_distribution,
                'statistics': self.get_statistics()
            }
    
    def optimize_cache(self):
        """Perform cache optimization by removing expired entries and defragmenting."""
        logger.info("Starting cache optimization...")
        
        with self._lock:
            initial_size = len(self._cache)
            current_time = time.time()
            
            # Remove expired entries
            expired_keys = []
            for key, item in self._cache.items():
                if current_time - item['timestamp'] > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
            
            # Rebuild cache to defragment
            items = list(self._cache.items())
            self._cache.clear()
            
            for key, value in items:
                self._cache[key] = value
            
            final_size = len(self._cache)
            removed_count = initial_size - final_size
            
            logger.info(f"Cache optimization complete: removed {removed_count} expired entries, "
                       f"size: {final_size}/{self.max_size}")


# Global cache instance for singleton pattern
_embedding_cache_instance: Optional[EmbeddingCache] = None

def get_embedding_cache(max_size: int = 1000, ttl_seconds: int = 3600) -> EmbeddingCache:
    """
    Get singleton embedding cache instance.
    
    Args:
        max_size: Maximum cache size (used only on first call)
        ttl_seconds: TTL for cache entries (used only on first call)
        
    Returns:
        Singleton EmbeddingCache instance
    """
    global _embedding_cache_instance
    
    if _embedding_cache_instance is None:
        _embedding_cache_instance = EmbeddingCache(max_size=max_size, ttl_seconds=ttl_seconds)
    
    return _embedding_cache_instance

def clear_embedding_cache():
    """Clear and reset the global embedding cache instance."""
    global _embedding_cache_instance
    
    if _embedding_cache_instance:
        with _embedding_cache_instance._lock:
            _embedding_cache_instance._cache.clear()
            _embedding_cache_instance._hits = 0
            _embedding_cache_instance._misses = 0
            _embedding_cache_instance._evictions = 0
        
        logger.info("Global embedding cache cleared")