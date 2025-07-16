#!/usr/bin/env python3
"""
Knowledge Management System for Enhanced Multi-Embedding Entry System
Phase 3: Implements document ingestion, quality-driven chunking, and retrieval optimization
"""

import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import Phase 1 components
from content_processing import (
    ContentAnalyzer, DocumentStructure,
    IntelligentChunker, ChunkingConfig, ChunkingStrategy, Chunk,
    EmbeddingOptimizer, EmbeddingModel, Embedding,
    AIQualityAssessor, QualityScore
)

logger = logging.getLogger(__name__)

class KnowledgeType(Enum):
    """Types of knowledge in the system"""
    DOCUMENTATION = "documentation"
    CODE_PATTERN = "code_pattern"
    BEST_PRACTICE = "best_practice"
    TROUBLESHOOTING = "troubleshooting"
    ARCHITECTURAL = "architectural"
    CONFIGURATION = "configuration"
    GENERAL = "general"

class RelationshipType(Enum):
    """Types of relationships between knowledge chunks"""
    PARENT_CHILD = "parent_child"
    SIBLING = "sibling"
    CROSS_REFERENCE = "cross_reference"
    PREREQUISITE = "prerequisite"
    ALTERNATIVE = "alternative"
    EXAMPLE_OF = "example_of"
    IMPLEMENTS = "implements"
    DEPENDS_ON = "depends_on"

@dataclass
class KnowledgeDocument:
    """Represents a knowledge document"""
    document_id: str
    source_path: str
    title: str
    knowledge_type: KnowledgeType
    metadata: Dict[str, Any] = field(default_factory=dict)
    ingested_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    version: int = 1
    tags: List[str] = field(default_factory=list)

@dataclass
class KnowledgeChunk:
    """Enhanced chunk with knowledge-specific metadata"""
    chunk_id: str
    document_id: str
    content: str
    chunk_type: str
    knowledge_type: KnowledgeType
    quality_score: float
    importance_score: float = 0.5
    confidence_score: float = 0.8
    relationships: List[Tuple[str, RelationshipType]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Dict[str, List[float]] = field(default_factory=dict)
    usage_count: int = 0
    last_accessed: Optional[datetime] = None

@dataclass
class RelationshipGraph:
    """Graph of relationships between knowledge chunks"""
    nodes: Dict[str, KnowledgeChunk] = field(default_factory=dict)
    edges: List[Tuple[str, str, RelationshipType]] = field(default_factory=list)
    clusters: Dict[str, List[str]] = field(default_factory=dict)
    
    def add_relationship(self, chunk1_id: str, chunk2_id: str, relationship_type: RelationshipType):
        """Add a relationship between two chunks"""
        self.edges.append((chunk1_id, chunk2_id, relationship_type))
        
    def get_related_chunks(self, chunk_id: str, relationship_types: Optional[List[RelationshipType]] = None) -> List[str]:
        """Get chunks related to a given chunk"""
        related = []
        for edge in self.edges:
            if edge[0] == chunk_id:
                if relationship_types is None or edge[2] in relationship_types:
                    related.append(edge[1])
            elif edge[1] == chunk_id and edge[2] != RelationshipType.PARENT_CHILD:
                if relationship_types is None or edge[2] in relationship_types:
                    related.append(edge[0])
        return list(set(related))

@dataclass
class RetrievalOptimization:
    """Optimization strategies for knowledge retrieval"""
    hot_chunks: List[str] = field(default_factory=list)  # Frequently accessed
    chunk_clusters: Dict[str, List[str]] = field(default_factory=dict)  # Semantic clusters
    access_patterns: Dict[str, List[str]] = field(default_factory=dict)  # Common access sequences
    cache_recommendations: List[str] = field(default_factory=list)  # Chunks to cache
    prefetch_patterns: Dict[str, List[str]] = field(default_factory=dict)  # Predictive prefetching

class KnowledgeManagementSystem:
    """
    Main system for managing knowledge ingestion, organization, and retrieval
    """
    
    def __init__(self, 
                 db_connection=None,
                 content_analyzer: Optional[ContentAnalyzer] = None,
                 chunker: Optional[IntelligentChunker] = None,
                 quality_assessor: Optional[AIQualityAssessor] = None,
                 embedding_optimizer: Optional[EmbeddingOptimizer] = None):
        self.db = db_connection
        self.content_analyzer = content_analyzer or ContentAnalyzer()
        self.quality_assessor = quality_assessor or AIQualityAssessor()
        
        # Default chunking configuration for knowledge
        self.default_chunking_config = ChunkingConfig(
            max_tokens=300,  # Smaller chunks for better granularity
            min_tokens=50,
            overlap_tokens=30,
            respect_boundaries=True,
            preserve_code_blocks=True,
            preserve_headings=True,
            quality_threshold=0.75,  # Higher threshold for knowledge
            strategy=ChunkingStrategy.SEMANTIC_AWARE
        )
        self.chunker = chunker or IntelligentChunker(self.default_chunking_config)
        
        self.embedding_optimizer = embedding_optimizer or EmbeddingOptimizer(
            model=EmbeddingModel.ALL_MINILM_L6_V2,
            cleaning_level=TextCleaningLevel.STANDARD
        )
        
        # Knowledge graph for relationship tracking
        self.knowledge_graph = RelationshipGraph()
        
        # Retrieval optimization data
        self.retrieval_optimization = RetrievalOptimization()
        
        logger.info("KnowledgeManagementSystem initialized")
    
    async def ingest_knowledge_document(self, 
                                      doc_path: str,
                                      title: Optional[str] = None,
                                      knowledge_type: Optional[KnowledgeType] = None,
                                      tags: Optional[List[str]] = None,
                                      session_id: Optional[str] = None) -> KnowledgeDocument:
        """
        Ingest a knowledge document and create quality-driven chunks
        
        Args:
            doc_path: Path to the document
            title: Optional document title
            knowledge_type: Type of knowledge
            tags: Optional tags for categorization
            session_id: Optional session ID for tracking
            
        Returns:
            KnowledgeDocument object
        """
        try:
            # Read document content
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Analyze document structure
            structure = self.content_analyzer.analyze_document_structure(content)
            
            # Determine knowledge type if not provided
            if knowledge_type is None:
                knowledge_type = self._infer_knowledge_type(structure, doc_path)
            
            # Create document record
            document = KnowledgeDocument(
                document_id=self._generate_document_id(doc_path),
                source_path=doc_path,
                title=title or Path(doc_path).stem,
                knowledge_type=knowledge_type,
                metadata={
                    'content_type': structure.content_type.value,
                    'total_elements': len(structure.elements),
                    'statistics': structure.statistics,
                    'session_id': session_id
                },
                tags=tags or []
            )
            
            # Create quality-driven chunks
            chunks = await self._create_knowledge_chunks(document, structure, session_id)
            
            # Discover relationships
            await self._discover_relationships(chunks)
            
            # Store in database if available
            if self.db:
                await self._store_document_and_chunks(document, chunks)
            
            logger.info(f"Ingested knowledge document: {document.title} with {len(chunks)} chunks")
            return document
            
        except Exception as e:
            logger.error(f"Failed to ingest document {doc_path}: {e}")
            raise
    
    async def apply_quality_assessment(self, 
                                     chunk: KnowledgeChunk,
                                     context_chunks: Optional[List[KnowledgeChunk]] = None) -> QualityScore:
        """
        Apply quality assessment to a knowledge chunk
        
        Args:
            chunk: Knowledge chunk to assess
            context_chunks: Optional surrounding chunks for context
            
        Returns:
            QualityScore object
        """
        # Convert to regular chunk for assessment
        regular_chunk = Chunk(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            chunk_type=chunk.chunk_type,
            line_start=0,
            line_end=0,
            token_count=len(chunk.content.split()),
            elements=[],
            metadata=chunk.metadata,
            quality_score=chunk.quality_score
        )
        
        # Get context if available
        context = None
        if context_chunks:
            context = [Chunk(
                chunk_id=c.chunk_id,
                content=c.content,
                chunk_type=c.chunk_type,
                line_start=0,
                line_end=0,
                token_count=len(c.content.split()),
                elements=[],
                metadata=c.metadata,
                quality_score=c.quality_score
            ) for c in context_chunks]
        
        # Assess quality
        quality_score = self.quality_assessor.assess_chunk_quality(regular_chunk, context)
        
        # Update chunk quality scores
        chunk.quality_score = quality_score.overall_score
        chunk.confidence_score = quality_score.confidence
        
        # Add knowledge-specific quality factors
        chunk.importance_score = self._calculate_importance_score(chunk, quality_score)
        
        return quality_score
    
    async def establish_cross_references(self, 
                                       chunks: List[KnowledgeChunk],
                                       similarity_threshold: float = 0.7) -> RelationshipGraph:
        """
        Establish cross-references between knowledge chunks
        
        Args:
            chunks: List of knowledge chunks
            similarity_threshold: Threshold for semantic similarity
            
        Returns:
            Updated RelationshipGraph
        """
        # Generate embeddings if not present
        chunks_needing_embeddings = [c for c in chunks if not c.embeddings]
        if chunks_needing_embeddings:
            await self._generate_embeddings(chunks_needing_embeddings)
        
        # Calculate similarities
        for i, chunk1 in enumerate(chunks):
            self.knowledge_graph.nodes[chunk1.chunk_id] = chunk1
            
            for j, chunk2 in enumerate(chunks[i+1:], i+1):
                # Semantic similarity
                similarity = self._calculate_semantic_similarity(chunk1, chunk2)
                
                if similarity >= similarity_threshold:
                    # Determine relationship type
                    rel_type = self._determine_relationship_type(chunk1, chunk2, similarity)
                    self.knowledge_graph.add_relationship(chunk1.chunk_id, chunk2.chunk_id, rel_type)
                    
                    # Update chunk relationships
                    chunk1.relationships.append((chunk2.chunk_id, rel_type))
                    chunk2.relationships.append((chunk1.chunk_id, rel_type))
        
        # Discover clusters
        self._discover_clusters()
        
        return self.knowledge_graph
    
    async def optimize_retrieval_patterns(self, 
                                        usage_data: Dict[str, Any]) -> RetrievalOptimization:
        """
        Optimize retrieval patterns based on usage data
        
        Args:
            usage_data: Dictionary containing access patterns and metrics
            
        Returns:
            RetrievalOptimization object
        """
        # Analyze access patterns
        access_logs = usage_data.get('access_logs', [])
        
        # Identify hot chunks
        chunk_access_counts = {}
        for log in access_logs:
            chunk_id = log.get('chunk_id')
            if chunk_id:
                chunk_access_counts[chunk_id] = chunk_access_counts.get(chunk_id, 0) + 1
        
        # Sort by access frequency
        hot_chunks = sorted(chunk_access_counts.items(), key=lambda x: x[1], reverse=True)
        self.retrieval_optimization.hot_chunks = [chunk_id for chunk_id, _ in hot_chunks[:20]]
        
        # Analyze access sequences
        self._analyze_access_sequences(access_logs)
        
        # Generate cache recommendations
        self._generate_cache_recommendations()
        
        # Create prefetch patterns
        self._create_prefetch_patterns()
        
        return self.retrieval_optimization
    
    # Private helper methods
    
    def _generate_document_id(self, doc_path: str) -> str:
        """Generate unique document ID"""
        return f"doc_{hashlib.md5(doc_path.encode()).hexdigest()[:12]}"
    
    def _infer_knowledge_type(self, structure: DocumentStructure, doc_path: str) -> KnowledgeType:
        """Infer knowledge type from document structure and path"""
        path_lower = doc_path.lower()
        
        if 'readme' in path_lower or 'doc' in path_lower:
            return KnowledgeType.DOCUMENTATION
        elif 'config' in path_lower or 'settings' in path_lower:
            return KnowledgeType.CONFIGURATION
        elif 'troubleshoot' in path_lower or 'debug' in path_lower:
            return KnowledgeType.TROUBLESHOOTING
        elif 'architecture' in path_lower or 'design' in path_lower:
            return KnowledgeType.ARCHITECTURAL
        elif structure.statistics.get('code_block_count', 0) > 5:
            return KnowledgeType.CODE_PATTERN
        elif 'best' in path_lower or 'practice' in path_lower:
            return KnowledgeType.BEST_PRACTICE
        else:
            return KnowledgeType.GENERAL
    
    async def _create_knowledge_chunks(self, 
                                     document: KnowledgeDocument,
                                     structure: DocumentStructure,
                                     session_id: Optional[str]) -> List[KnowledgeChunk]:
        """Create knowledge chunks from document structure"""
        # Use intelligent chunker
        regular_chunks = self.chunker.chunk_document(structure)
        
        # Convert to knowledge chunks
        knowledge_chunks = []
        for i, chunk in enumerate(regular_chunks):
            # Apply quality assessment
            quality_score = await self.apply_quality_assessment(chunk)
            
            # Only include high-quality chunks
            if quality_score.overall_score >= self.default_chunking_config.quality_threshold:
                knowledge_chunk = KnowledgeChunk(
                    chunk_id=f"kc_{document.document_id}_{i:04d}",
                    document_id=document.document_id,
                    content=chunk.content,
                    chunk_type=chunk.chunk_type.value,
                    knowledge_type=document.knowledge_type,
                    quality_score=quality_score.overall_score,
                    confidence_score=quality_score.confidence,
                    tags=document.tags.copy(),
                    metadata={
                        **chunk.metadata,
                        'line_start': chunk.line_start,
                        'line_end': chunk.line_end,
                        'token_count': chunk.token_count,
                        'session_id': session_id
                    }
                )
                
                knowledge_chunks.append(knowledge_chunk)
        
        return knowledge_chunks
    
    async def _discover_relationships(self, chunks: List[KnowledgeChunk]):
        """Discover relationships between chunks"""
        # Structural relationships (parent-child, siblings)
        for i, chunk in enumerate(chunks):
            # Previous chunk is potential parent or sibling
            if i > 0:
                prev_chunk = chunks[i-1]
                if self._is_parent_child(prev_chunk, chunk):
                    chunk.relationships.append((prev_chunk.chunk_id, RelationshipType.PARENT_CHILD))
                else:
                    chunk.relationships.append((prev_chunk.chunk_id, RelationshipType.SIBLING))
            
            # Next chunk is potential child or sibling
            if i < len(chunks) - 1:
                next_chunk = chunks[i+1]
                if self._is_parent_child(chunk, next_chunk):
                    chunk.relationships.append((next_chunk.chunk_id, RelationshipType.PARENT_CHILD))
    
    def _is_parent_child(self, chunk1: KnowledgeChunk, chunk2: KnowledgeChunk) -> bool:
        """Determine if two chunks have parent-child relationship"""
        # Check if chunk1 is a heading and chunk2 is content
        return (chunk1.chunk_type in ['heading', 'section_header'] and 
                chunk2.chunk_type in ['paragraph', 'list_section', 'code_block'])
    
    async def _generate_embeddings(self, chunks: List[KnowledgeChunk]):
        """Generate embeddings for chunks"""
        # Convert to regular chunks for embedding generation
        regular_chunks = []
        for kc in chunks:
            regular_chunk = Chunk(
                chunk_id=kc.chunk_id,
                content=kc.content,
                chunk_type=kc.chunk_type,
                line_start=0,
                line_end=0,
                token_count=len(kc.content.split()),
                elements=[],
                metadata=kc.metadata,
                quality_score=kc.quality_score
            )
            regular_chunks.append(regular_chunk)
        
        # Generate embeddings
        embeddings = self.embedding_optimizer.generate_optimized_embeddings(regular_chunks)
        
        # Store embeddings in knowledge chunks
        for kc, embedding in zip(chunks, embeddings):
            kc.embeddings[embedding.model.value] = embedding.vector
    
    def _calculate_semantic_similarity(self, chunk1: KnowledgeChunk, chunk2: KnowledgeChunk) -> float:
        """Calculate semantic similarity between two chunks"""
        # Get embeddings
        model_name = list(chunk1.embeddings.keys())[0] if chunk1.embeddings else None
        if not model_name or model_name not in chunk2.embeddings:
            return 0.0
        
        vec1 = chunk1.embeddings[model_name]
        vec2 = chunk2.embeddings[model_name]
        
        # Cosine similarity
        import numpy as np
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _determine_relationship_type(self, 
                                   chunk1: KnowledgeChunk, 
                                   chunk2: KnowledgeChunk,
                                   similarity: float) -> RelationshipType:
        """Determine the type of relationship between chunks"""
        # Check for specific patterns
        content1_lower = chunk1.content.lower()
        content2_lower = chunk2.content.lower()
        
        if 'example' in content1_lower and similarity > 0.8:
            return RelationshipType.EXAMPLE_OF
        elif 'implements' in content1_lower or 'implementation' in content1_lower:
            return RelationshipType.IMPLEMENTS
        elif 'depends on' in content1_lower or 'requires' in content1_lower:
            return RelationshipType.DEPENDS_ON
        elif 'alternative' in content1_lower or 'instead of' in content1_lower:
            return RelationshipType.ALTERNATIVE
        elif 'prerequisite' in content1_lower or 'before' in content1_lower:
            return RelationshipType.PREREQUISITE
        else:
            return RelationshipType.CROSS_REFERENCE
    
    def _calculate_importance_score(self, 
                                  chunk: KnowledgeChunk, 
                                  quality_score: QualityScore) -> float:
        """Calculate importance score for a knowledge chunk"""
        score = 0.0
        
        # Base score from quality
        score += quality_score.overall_score * 0.4
        
        # Knowledge type weighting
        type_weights = {
            KnowledgeType.BEST_PRACTICE: 0.9,
            KnowledgeType.ARCHITECTURAL: 0.85,
            KnowledgeType.TROUBLESHOOTING: 0.8,
            KnowledgeType.CODE_PATTERN: 0.75,
            KnowledgeType.CONFIGURATION: 0.7,
            KnowledgeType.DOCUMENTATION: 0.65,
            KnowledgeType.GENERAL: 0.5
        }
        score += type_weights.get(chunk.knowledge_type, 0.5) * 0.3
        
        # Chunk type weighting
        chunk_type_weights = {
            'heading': 0.8,
            'section_header': 0.75,
            'code_block': 0.7,
            'list_section': 0.65,
            'paragraph': 0.6
        }
        score += chunk_type_weights.get(chunk.chunk_type, 0.5) * 0.3
        
        return min(score, 1.0)
    
    def _discover_clusters(self):
        """Discover clusters of related chunks"""
        # Simple clustering based on high similarity connections
        visited = set()
        cluster_id = 0
        
        for node_id in self.knowledge_graph.nodes:
            if node_id not in visited:
                cluster = self._dfs_cluster(node_id, visited)
                if len(cluster) > 1:
                    self.knowledge_graph.clusters[f"cluster_{cluster_id}"] = cluster
                    cluster_id += 1
    
    def _dfs_cluster(self, start_node: str, visited: Set[str]) -> List[str]:
        """DFS to find connected components"""
        stack = [start_node]
        cluster = []
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                cluster.append(node)
                
                # Get strongly connected neighbors
                for edge in self.knowledge_graph.edges:
                    if edge[0] == node and edge[2] in [RelationshipType.CROSS_REFERENCE, RelationshipType.EXAMPLE_OF]:
                        if edge[1] not in visited:
                            stack.append(edge[1])
        
        return cluster
    
    def _analyze_access_sequences(self, access_logs: List[Dict[str, Any]]):
        """Analyze access sequences for patterns"""
        # Group by session
        session_sequences = {}
        for log in access_logs:
            session_id = log.get('session_id', 'default')
            chunk_id = log.get('chunk_id')
            if chunk_id:
                if session_id not in session_sequences:
                    session_sequences[session_id] = []
                session_sequences[session_id].append(chunk_id)
        
        # Find common sequences
        sequence_counts = {}
        for sequence in session_sequences.values():
            for i in range(len(sequence) - 1):
                seq_key = f"{sequence[i]}->{sequence[i+1]}"
                sequence_counts[seq_key] = sequence_counts.get(seq_key, 0) + 1
        
        # Store common patterns
        common_sequences = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for seq, count in common_sequences:
            chunk1, chunk2 = seq.split('->')
            if chunk1 not in self.retrieval_optimization.access_patterns:
                self.retrieval_optimization.access_patterns[chunk1] = []
            self.retrieval_optimization.access_patterns[chunk1].append(chunk2)
    
    def _generate_cache_recommendations(self):
        """Generate cache recommendations based on usage"""
        # Recommend hot chunks for caching
        self.retrieval_optimization.cache_recommendations = self.retrieval_optimization.hot_chunks[:10]
        
        # Add entry points of common access sequences
        for chunk_id, next_chunks in self.retrieval_optimization.access_patterns.items():
            if chunk_id not in self.retrieval_optimization.cache_recommendations:
                self.retrieval_optimization.cache_recommendations.append(chunk_id)
    
    def _create_prefetch_patterns(self):
        """Create prefetch patterns for predictive loading"""
        # Based on access patterns and relationships
        for chunk_id in self.retrieval_optimization.hot_chunks[:5]:
            # Get related chunks
            related = self.knowledge_graph.get_related_chunks(
                chunk_id, 
                [RelationshipType.CROSS_REFERENCE, RelationshipType.EXAMPLE_OF]
            )
            if related:
                self.retrieval_optimization.prefetch_patterns[chunk_id] = related[:3]
    
    async def _store_document_and_chunks(self, 
                                       document: KnowledgeDocument, 
                                       chunks: List[KnowledgeChunk]):
        """Store document and chunks in database"""
        # This would integrate with the database layer
        # For now, just log
        logger.info(f"Would store document {document.document_id} with {len(chunks)} chunks")