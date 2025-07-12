"""
Embedding Service for MegaMind Context Database
Provides text-to-vector embedding generation using sentence-transformers
"""

import os
import hashlib
import logging
from typing import List, Optional, Dict, Any
from functools import lru_cache
import json

# Conditional imports for graceful degradation
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Embedding generation will be disabled.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class EmbeddingService:
    """Singleton service for generating text embeddings with realm context support"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.model = None
        self.model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        self.device = os.getenv('EMBEDDING_DEVICE', 'cpu')
        self.batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '50'))
        self.embedding_dimension = 384  # all-MiniLM-L6-v2 dimension
        self.cache_size = int(os.getenv('EMBEDDING_CACHE_SIZE', '1000'))
        
        # Initialize cache for embeddings
        self._embedding_cache = {}
        self._cache_order = []
        
        # Try to initialize model
        self._initialize_model()
        self._initialized = True
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logging.warning("Cannot initialize embedding model: sentence-transformers not available")
            return
            
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logging.info(f"Initialized embedding model: {self.model_name} on {self.device}")
        except Exception as e:
            logging.error(f"Failed to initialize embedding model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if embedding service is available"""
        return self.model is not None and SENTENCE_TRANSFORMERS_AVAILABLE
    
    def generate_embedding(self, text: str, realm_context: Optional[str] = None) -> Optional[List[float]]:
        """
        Generate embedding for single text chunk with optional realm context
        
        Args:
            text: Input text to embed
            realm_context: Optional realm context for enhanced embedding
            
        Returns:
            List of float values representing the embedding, or None if service unavailable
        """
        if not self.is_available():
            logging.warning("Embedding service not available, returning None")
            return None
            
        if not text or not text.strip():
            return None
        
        # Preprocess text
        processed_text = self.preprocess_text(text, realm_context)
        
        # Check cache first
        content_hash = self._get_content_hash(processed_text)
        cached_embedding = self._get_cached_embedding(content_hash)
        if cached_embedding is not None:
            return cached_embedding
        
        try:
            # Generate embedding
            embedding = self.model.encode(processed_text, convert_to_tensor=False)
            embedding_list = embedding.tolist()
            
            # Cache the result
            self._cache_embedding(content_hash, embedding_list)
            
            return embedding_list
            
        except Exception as e:
            logging.error(f"Failed to generate embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str], realm_contexts: Optional[List[str]] = None) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple chunks efficiently
        
        Args:
            texts: List of input texts to embed
            realm_contexts: Optional list of realm contexts (same length as texts)
            
        Returns:
            List of embeddings (same length as input texts)
        """
        if not self.is_available():
            logging.warning("Embedding service not available, returning None list")
            return [None] * len(texts)
        
        if not texts:
            return []
        
        # Ensure realm_contexts has same length as texts
        if realm_contexts is None:
            realm_contexts = [None] * len(texts)
        elif len(realm_contexts) != len(texts):
            realm_contexts.extend([None] * (len(texts) - len(realm_contexts)))
        
        results = []
        texts_to_process = []
        indices_to_process = []
        
        # Preprocess and check cache
        for i, (text, realm_context) in enumerate(zip(texts, realm_contexts)):
            if not text or not text.strip():
                results.append(None)
                continue
                
            processed_text = self.preprocess_text(text, realm_context)
            content_hash = self._get_content_hash(processed_text)
            cached_embedding = self._get_cached_embedding(content_hash)
            
            if cached_embedding is not None:
                results.append(cached_embedding)
            else:
                results.append(None)  # Placeholder
                texts_to_process.append(processed_text)
                indices_to_process.append(i)
        
        # Process uncached texts in batches
        if texts_to_process:
            try:
                embeddings = self.model.encode(texts_to_process, convert_to_tensor=False, batch_size=self.batch_size)
                
                # Store results and cache
                for i, embedding in enumerate(embeddings):
                    embedding_list = embedding.tolist()
                    original_index = indices_to_process[i]
                    results[original_index] = embedding_list
                    
                    # Cache the result
                    processed_text = texts_to_process[i]
                    content_hash = self._get_content_hash(processed_text)
                    self._cache_embedding(content_hash, embedding_list)
                    
            except Exception as e:
                logging.error(f"Failed to generate batch embeddings: {e}")
                # Fill remaining with None
                for i in indices_to_process:
                    if results[i] is None:
                        results[i] = None
        
        return results
    
    def preprocess_text(self, text: str, realm_context: Optional[str] = None) -> str:
        """
        Clean and normalize text for embedding with optional realm context
        
        Args:
            text: Input text to preprocess
            realm_context: Optional realm context to prepend
            
        Returns:
            Preprocessed text ready for embedding
        """
        if not text:
            return ""
        
        # Basic text cleaning
        processed = text.strip()
        
        # Remove excessive whitespace
        processed = ' '.join(processed.split())
        
        # Optionally prepend realm context for enhanced embedding
        if realm_context and realm_context != 'GLOBAL':
            processed = f"[{realm_context}] {processed}"
        
        return processed
    
    def _get_content_hash(self, text: str) -> str:
        """Generate hash for text content"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cached_embedding(self, content_hash: str) -> Optional[List[float]]:
        """Retrieve cached embedding"""
        if content_hash in self._embedding_cache:
            # Move to end (LRU)
            self._cache_order.remove(content_hash)
            self._cache_order.append(content_hash)
            return self._embedding_cache[content_hash]
        return None
    
    def _cache_embedding(self, content_hash: str, embedding: List[float]):
        """Store embedding in cache with LRU eviction"""
        if content_hash in self._embedding_cache:
            # Update existing
            self._cache_order.remove(content_hash)
        elif len(self._embedding_cache) >= self.cache_size:
            # Evict oldest
            oldest = self._cache_order.pop(0)
            del self._embedding_cache[oldest]
        
        self._embedding_cache[content_hash] = embedding
        self._cache_order.append(content_hash)
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'embedding_dimension': self.embedding_dimension,
            'cache_size': len(self._embedding_cache),
            'cache_limit': self.cache_size,
            'batch_size': self.batch_size,
            'available': self.is_available(),
            'dependencies': {
                'sentence_transformers': SENTENCE_TRANSFORMERS_AVAILABLE,
                'torch': TORCH_AVAILABLE
            }
        }
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self._embedding_cache.clear()
        self._cache_order.clear()
        logging.info("Embedding cache cleared")


# Global instance
embedding_service = EmbeddingService()


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance"""
    return embedding_service