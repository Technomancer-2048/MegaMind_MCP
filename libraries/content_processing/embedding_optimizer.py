#!/usr/bin/env python3
"""
Embedding Optimization Library for Enhanced Multi-Embedding Entry System
Optimizes text for embedding generation and manages the embedding process
"""

import re
import logging
import hashlib
import unicodedata
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

from intelligent_chunker import Chunk, ChunkType

logger = logging.getLogger(__name__)

class EmbeddingModel(Enum):
    """Supported embedding models"""
    ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"  # 384 dimensions, 512 tokens
    ALL_MPNET_BASE_V2 = "all-mpnet-base-v2"  # 768 dimensions, 512 tokens
    E5_SMALL_V2 = "e5-small-v2"  # 384 dimensions, 512 tokens
    E5_BASE_V2 = "e5-base-v2"  # 768 dimensions, 512 tokens
    INSTRUCTOR_BASE = "instructor-base"  # 768 dimensions, 512 tokens

class TextCleaningLevel(Enum):
    """Levels of text cleaning"""
    MINIMAL = "minimal"  # Preserve most formatting
    STANDARD = "standard"  # Remove markdown, normalize spaces
    AGGRESSIVE = "aggressive"  # Strip all formatting, punctuation normalization

@dataclass
class Embedding:
    """Represents an embedding vector with metadata"""
    embedding_id: str
    chunk_id: str
    vector: List[float]
    model: EmbeddingModel
    dimension: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_time_ms: float = 0.0
    quality_score: float = 0.0

@dataclass
class OptimizedText:
    """Text optimized for embedding generation"""
    original_text: str
    cleaned_text: str
    cleaning_level: TextCleaningLevel
    tokens_removed: int
    formatting_preserved: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SimilarityMatrix:
    """Matrix of similarities between embeddings"""
    embedding_ids: List[str]
    matrix: np.ndarray
    threshold: float = 0.7
    
    def get_similar_pairs(self) -> List[Tuple[str, str, float]]:
        """Get pairs of similar embeddings above threshold"""
        pairs = []
        n = len(self.embedding_ids)
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self.matrix[i, j]
                if similarity >= self.threshold:
                    pairs.append((
                        self.embedding_ids[i],
                        self.embedding_ids[j],
                        similarity
                    ))
        return sorted(pairs, key=lambda x: x[2], reverse=True)

class EmbeddingOptimizer:
    """
    Optimizes text for embedding generation and manages the embedding process
    """
    
    def __init__(self, 
                 model: EmbeddingModel = EmbeddingModel.ALL_MINILM_L6_V2,
                 cleaning_level: TextCleaningLevel = TextCleaningLevel.STANDARD):
        self.model = model
        self.cleaning_level = cleaning_level
        
        # Model specifications
        self.model_specs = {
            EmbeddingModel.ALL_MINILM_L6_V2: {
                'dimensions': 384,
                'max_tokens': 512,
                'optimal_tokens': 256
            },
            EmbeddingModel.ALL_MPNET_BASE_V2: {
                'dimensions': 768,
                'max_tokens': 512,
                'optimal_tokens': 256
            },
            EmbeddingModel.E5_SMALL_V2: {
                'dimensions': 384,
                'max_tokens': 512,
                'optimal_tokens': 256
            },
            EmbeddingModel.E5_BASE_V2: {
                'dimensions': 768,
                'max_tokens': 512,
                'optimal_tokens': 256
            },
            EmbeddingModel.INSTRUCTOR_BASE: {
                'dimensions': 768,
                'max_tokens': 512,
                'optimal_tokens': 256
            }
        }
        
        # Text cleaning patterns
        self.cleaning_patterns = {
            'markdown_headers': re.compile(r'^#{1,6}\s+', re.MULTILINE),
            'markdown_bold': re.compile(r'\*\*([^*]+)\*\*|__([^_]+)__'),
            'markdown_italic': re.compile(r'\*([^*]+)\*|_([^_]+)_'),
            'markdown_code': re.compile(r'`([^`]+)`'),
            'markdown_links': re.compile(r'\[([^\]]+)\]\([^)]+\)'),
            'markdown_images': re.compile(r'!\[([^\]]*)\]\([^)]+\)'),
            'html_tags': re.compile(r'<[^>]+>'),
            'urls': re.compile(r'https?://[^\s]+'),
            'emails': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            'multiple_spaces': re.compile(r'\s+'),
            'multiple_newlines': re.compile(r'\n{3,}'),
            'special_chars': re.compile(r'[^\w\s\.\,\!\?\-\(\)\[\]\{\}\'\"]+'),
            'emoji': re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "]+", 
                flags=re.UNICODE
            )
        }
        
        logger.info(f"EmbeddingOptimizer initialized with model: {model.value}, cleaning: {cleaning_level.value}")
    
    def prepare_text_for_embedding(self, chunk: Chunk) -> OptimizedText:
        """
        Prepare chunk text for embedding generation
        
        Args:
            chunk: Chunk to prepare
            
        Returns:
            OptimizedText object
        """
        original_text = chunk.content
        
        # Apply cleaning based on level
        if self.cleaning_level == TextCleaningLevel.MINIMAL:
            cleaned_text = self._minimal_cleaning(original_text, chunk.chunk_type)
        elif self.cleaning_level == TextCleaningLevel.STANDARD:
            cleaned_text = self._standard_cleaning(original_text, chunk.chunk_type)
        else:  # AGGRESSIVE
            cleaned_text = self._aggressive_cleaning(original_text)
        
        # Calculate tokens removed
        original_tokens = len(original_text.split())
        cleaned_tokens = len(cleaned_text.split())
        tokens_removed = original_tokens - cleaned_tokens
        
        # Preserve important formatting information
        formatting_preserved = self._extract_formatting_info(original_text, chunk)
        
        # Add chunk-specific optimizations
        if chunk.chunk_type == ChunkType.CODE_BLOCK:
            cleaned_text = self._optimize_code_for_embedding(cleaned_text, chunk)
        elif chunk.chunk_type == ChunkType.LIST_SECTION:
            cleaned_text = self._optimize_list_for_embedding(cleaned_text)
        
        return OptimizedText(
            original_text=original_text,
            cleaned_text=cleaned_text,
            cleaning_level=self.cleaning_level,
            tokens_removed=tokens_removed,
            formatting_preserved=formatting_preserved,
            metadata={
                'chunk_type': chunk.chunk_type.value,
                'original_length': len(original_text),
                'cleaned_length': len(cleaned_text),
                'compression_ratio': len(cleaned_text) / len(original_text) if original_text else 0
            }
        )
    
    def strip_formatting_preserve_semantics(self, text: str) -> str:
        """
        Strip formatting while preserving semantic meaning
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Replace markdown formatting with semantic equivalents
        cleaned = text
        
        # Headers - preserve hierarchy
        def replace_header(match):
            level = len(match.group(1))
            header_text = match.group(2)
            return f"{'Section' if level <= 2 else 'Subsection'}: {header_text}"
        
        cleaned = re.sub(r'^(#{1,6})\s+(.+)$', replace_header, cleaned, flags=re.MULTILINE)
        
        # Bold text - preserve emphasis
        cleaned = self.cleaning_patterns['markdown_bold'].sub(r'\1\2', cleaned)
        
        # Links - preserve link text
        cleaned = self.cleaning_patterns['markdown_links'].sub(r'\1', cleaned)
        
        # Code - preserve with markers
        cleaned = self.cleaning_patterns['markdown_code'].sub(r'code: \1', cleaned)
        
        # Lists - preserve structure
        cleaned = re.sub(r'^[-*+]\s+', '• ', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^\d+\.\s+', lambda m: f"{m.group(0)}", cleaned, flags=re.MULTILINE)
        
        # Clean up whitespace
        cleaned = self.cleaning_patterns['multiple_spaces'].sub(' ', cleaned)
        cleaned = self.cleaning_patterns['multiple_newlines'].sub('\n\n', cleaned)
        
        return cleaned.strip()
    
    def generate_optimized_embeddings(self, chunks: List[Chunk], embedding_service=None) -> List[Embedding]:
        """
        Generate embeddings for a list of chunks
        
        Args:
            chunks: List of chunks to embed
            embedding_service: Optional embedding service to use
            
        Returns:
            List of Embedding objects
        """
        embeddings = []
        
        for chunk in chunks:
            # Prepare text
            optimized_text = self.prepare_text_for_embedding(chunk)
            
            # Generate embedding (placeholder - would use actual embedding service)
            if embedding_service:
                try:
                    import time
                    start_time = time.time()
                    
                    vector = embedding_service.generate_embedding(
                        optimized_text.cleaned_text,
                        model=self.model.value
                    )
                    
                    generation_time = (time.time() - start_time) * 1000
                    
                except Exception as e:
                    logger.error(f"Failed to generate embedding for chunk {chunk.chunk_id}: {e}")
                    continue
            else:
                # Placeholder vector for testing
                dimensions = self.model_specs[self.model]['dimensions']
                vector = [0.0] * dimensions
                generation_time = 0.0
            
            # Create embedding object
            embedding = Embedding(
                embedding_id=self._generate_embedding_id(chunk.chunk_id),
                chunk_id=chunk.chunk_id,
                vector=vector,
                model=self.model,
                dimension=len(vector),
                metadata={
                    'optimized_text_length': len(optimized_text.cleaned_text),
                    'cleaning_level': self.cleaning_level.value,
                    'chunk_type': chunk.chunk_type.value,
                    'quality_score': chunk.quality_score,
                    'tokens_removed': optimized_text.tokens_removed
                },
                generation_time_ms=generation_time,
                quality_score=self._calculate_embedding_quality(chunk, optimized_text)
            )
            
            embeddings.append(embedding)
        
        return embeddings
    
    def calculate_semantic_similarity(self, embeddings: List[Embedding]) -> SimilarityMatrix:
        """
        Calculate semantic similarity between embeddings
        
        Args:
            embeddings: List of embeddings to compare
            
        Returns:
            SimilarityMatrix object
        """
        if len(embeddings) < 2:
            return SimilarityMatrix(
                embedding_ids=[e.embedding_id for e in embeddings],
                matrix=np.array([[1.0]]) if embeddings else np.array([])
            )
        
        # Convert to numpy array
        vectors = np.array([e.vector for e in embeddings])
        
        # Calculate cosine similarity
        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / (norms + 1e-8)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
        
        return SimilarityMatrix(
            embedding_ids=[e.embedding_id for e in embeddings],
            matrix=similarity_matrix
        )
    
    def _minimal_cleaning(self, text: str, chunk_type: ChunkType) -> str:
        """Minimal text cleaning - preserve most formatting"""
        cleaned = text
        
        # Remove only the most problematic elements
        cleaned = self.cleaning_patterns['html_tags'].sub('', cleaned)
        cleaned = self.cleaning_patterns['emoji'].sub('', cleaned)
        
        # Normalize unicode
        cleaned = unicodedata.normalize('NFKD', cleaned)
        
        # Clean whitespace
        cleaned = self.cleaning_patterns['multiple_spaces'].sub(' ', cleaned)
        cleaned = self.cleaning_patterns['multiple_newlines'].sub('\n\n', cleaned)
        
        return cleaned.strip()
    
    def _standard_cleaning(self, text: str, chunk_type: ChunkType) -> str:
        """Standard text cleaning - remove markdown, normalize"""
        cleaned = text
        
        # Skip cleaning for code blocks
        if chunk_type == ChunkType.CODE_BLOCK:
            return self._minimal_cleaning(text, chunk_type)
        
        # Remove markdown formatting
        cleaned = self.cleaning_patterns['markdown_headers'].sub('', cleaned)
        cleaned = self.cleaning_patterns['markdown_bold'].sub(r'\1\2', cleaned)
        cleaned = self.cleaning_patterns['markdown_italic'].sub(r'\1\2', cleaned)
        cleaned = self.cleaning_patterns['markdown_links'].sub(r'\1', cleaned)
        cleaned = self.cleaning_patterns['markdown_images'].sub(r'\1', cleaned)
        cleaned = self.cleaning_patterns['markdown_code'].sub(r'\1', cleaned)
        
        # Remove HTML and emojis
        cleaned = self.cleaning_patterns['html_tags'].sub('', cleaned)
        cleaned = self.cleaning_patterns['emoji'].sub('', cleaned)
        
        # Normalize URLs and emails
        cleaned = self.cleaning_patterns['urls'].sub('URL', cleaned)
        cleaned = self.cleaning_patterns['emails'].sub('EMAIL', cleaned)
        
        # Normalize unicode
        cleaned = unicodedata.normalize('NFKD', cleaned)
        
        # Clean whitespace
        cleaned = self.cleaning_patterns['multiple_spaces'].sub(' ', cleaned)
        cleaned = self.cleaning_patterns['multiple_newlines'].sub(' ', cleaned)
        
        return cleaned.strip()
    
    def _aggressive_cleaning(self, text: str) -> str:
        """Aggressive text cleaning - strip all formatting"""
        # Start with standard cleaning
        cleaned = self._standard_cleaning(text, ChunkType.PARAGRAPH)
        
        # Remove all special characters except basic punctuation
        cleaned = self.cleaning_patterns['special_chars'].sub(' ', cleaned)
        
        # Convert to lowercase
        cleaned = cleaned.lower()
        
        # Remove extra punctuation
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        
        # Final whitespace cleanup
        cleaned = self.cleaning_patterns['multiple_spaces'].sub(' ', cleaned)
        
        return cleaned.strip()
    
    def _optimize_code_for_embedding(self, text: str, chunk: Chunk) -> str:
        """Optimize code content for embedding"""
        # For code, we want to preserve structure but remove comments
        lines = text.split('\n')
        optimized_lines = []
        
        language = chunk.metadata.get('language', '')
        
        for line in lines:
            # Remove comments based on language
            if language in ['python', 'ruby', 'bash', 'sh']:
                line = re.sub(r'#.*$', '', line)
            elif language in ['javascript', 'java', 'c', 'cpp', 'go', 'rust']:
                line = re.sub(r'//.*$', '', line)
            
            # Remove empty lines
            if line.strip():
                optimized_lines.append(line)
        
        # Add language context
        if language:
            return f"{language} code:\n{chr(10).join(optimized_lines)}"
        
        return '\n'.join(optimized_lines)
    
    def _optimize_list_for_embedding(self, text: str) -> str:
        """Optimize list content for embedding"""
        # Convert list markers to consistent format
        lines = text.split('\n')
        optimized_lines = []
        
        for line in lines:
            # Unify list markers
            line = re.sub(r'^[-*+]\s+', '• ', line)
            line = re.sub(r'^\d+\.\s+', '• ', line)
            optimized_lines.append(line)
        
        return '\n'.join(optimized_lines)
    
    def _extract_formatting_info(self, text: str, chunk: Chunk) -> Dict[str, Any]:
        """Extract important formatting information to preserve"""
        formatting = {
            'has_headers': bool(self.cleaning_patterns['markdown_headers'].search(text)),
            'has_code': bool(self.cleaning_patterns['markdown_code'].search(text)),
            'has_links': bool(self.cleaning_patterns['markdown_links'].search(text)),
            'has_lists': bool(re.search(r'^[-*+\d]+\.\s+', text, re.MULTILINE)),
            'has_emphasis': bool(
                self.cleaning_patterns['markdown_bold'].search(text) or 
                self.cleaning_patterns['markdown_italic'].search(text)
            ),
            'chunk_type': chunk.chunk_type.value
        }
        
        # Extract header levels if present
        headers = self.cleaning_patterns['markdown_headers'].findall(text)
        if headers:
            formatting['header_levels'] = [len(h[0]) for h in headers]
        
        # Extract code languages
        code_blocks = re.findall(r'```(\w+)', text)
        if code_blocks:
            formatting['code_languages'] = list(set(code_blocks))
        
        return formatting
    
    def _calculate_embedding_quality(self, chunk: Chunk, optimized_text: OptimizedText) -> float:
        """Calculate quality score for the embedding"""
        score = 0.0
        
        # Base score from chunk quality
        score += chunk.quality_score * 0.4
        
        # Text optimization score
        compression_ratio = optimized_text.metadata.get('compression_ratio', 1.0)
        if 0.7 <= compression_ratio <= 0.95:
            score += 0.3  # Good compression
        elif compression_ratio < 0.5:
            score += 0.1  # Too much removed
        else:
            score += 0.2
        
        # Length score (prefer optimal length)
        text_length = len(optimized_text.cleaned_text.split())
        optimal_length = self.model_specs[self.model]['optimal_tokens']
        if text_length <= optimal_length:
            length_score = text_length / optimal_length
        else:
            max_length = self.model_specs[self.model]['max_tokens']
            length_score = 1.0 - (text_length - optimal_length) / (max_length - optimal_length)
        score += length_score * 0.3
        
        return min(score, 1.0)
    
    def _generate_embedding_id(self, chunk_id: str) -> str:
        """Generate unique ID for embedding"""
        return f"emb_{hashlib.md5(f"{chunk_id}_{self.model.value}".encode()).hexdigest()[:12]}"
    
    def batch_optimize_embeddings(self, chunks: List[Chunk], batch_size: int = 32) -> List[List[Chunk]]:
        """
        Organize chunks into optimal batches for embedding generation
        
        Args:
            chunks: List of chunks to batch
            batch_size: Maximum batch size
            
        Returns:
            List of chunk batches
        """
        # Sort chunks by type and size for better batching
        sorted_chunks = sorted(chunks, key=lambda c: (c.chunk_type.value, c.token_count))
        
        batches = []
        current_batch = []
        current_tokens = 0
        max_batch_tokens = self.model_specs[self.model]['max_tokens'] * batch_size
        
        for chunk in sorted_chunks:
            if len(current_batch) >= batch_size or current_tokens + chunk.token_count > max_batch_tokens:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [chunk]
                current_tokens = chunk.token_count
            else:
                current_batch.append(chunk)
                current_tokens += chunk.token_count
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def create_query_optimized_text(self, query: str, context: Optional[str] = None) -> str:
        """
        Optimize a query for embedding generation
        
        Args:
            query: User query
            context: Optional context to include
            
        Returns:
            Optimized query text
        """
        # Clean the query
        cleaned_query = self.strip_formatting_preserve_semantics(query)
        
        # Add context if provided
        if context:
            cleaned_context = self.strip_formatting_preserve_semantics(context)
            # Combine with clear separation
            optimized = f"Query: {cleaned_query}\nContext: {cleaned_context}"
        else:
            optimized = cleaned_query
        
        # Ensure within token limits
        words = optimized.split()
        max_words = int(self.model_specs[self.model]['max_tokens'] / 1.3)
        if len(words) > max_words:
            optimized = ' '.join(words[:max_words])
        
        return optimized