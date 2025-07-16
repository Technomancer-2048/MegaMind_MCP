"""
Content Processing Library for Enhanced Multi-Embedding Entry System
Phase 1 Foundation Components
"""

from .content_analyzer import (
    ContentAnalyzer,
    ContentType,
    MarkdownElementType,
    MarkdownElement,
    SemanticBoundary,
    DocumentStructure
)

from .intelligent_chunker import (
    IntelligentChunker,
    ChunkingStrategy,
    ChunkType,
    Chunk,
    ChunkingConfig
)

from .embedding_optimizer import (
    EmbeddingOptimizer,
    EmbeddingModel,
    TextCleaningLevel,
    Embedding,
    OptimizedText,
    SimilarityMatrix
)

from .ai_quality_assessor import (
    AIQualityAssessor,
    QualityDimension,
    QualityScore,
    QualityIssue
)

from .sentence_splitter import (
    SentenceSplitter
)

__all__ = [
    # Content Analyzer
    'ContentAnalyzer',
    'ContentType',
    'MarkdownElementType',
    'MarkdownElement',
    'SemanticBoundary',
    'DocumentStructure',
    
    # Intelligent Chunker
    'IntelligentChunker',
    'ChunkingStrategy',
    'ChunkType',
    'Chunk',
    'ChunkingConfig',
    
    # Embedding Optimizer
    'EmbeddingOptimizer',
    'EmbeddingModel',
    'TextCleaningLevel',
    'Embedding',
    'OptimizedText',
    'SimilarityMatrix',
    
    # AI Quality Assessor
    'AIQualityAssessor',
    'QualityDimension',
    'QualityScore',
    'QualityIssue',
    
    # Sentence Splitter
    'SentenceSplitter'
]

__version__ = '1.0.0'