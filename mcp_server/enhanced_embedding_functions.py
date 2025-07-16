#!/usr/bin/env python3
"""
Enhanced Multi-Embedding Entry System - MCP Function Integration
Phase 2: Session-aware MCP functions for content processing
"""

import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import Phase 1 components
from libraries.content_processing import (
    ContentAnalyzer, DocumentStructure,
    IntelligentChunker, ChunkingConfig, ChunkingStrategy, Chunk,
    EmbeddingOptimizer, EmbeddingModel, TextCleaningLevel,
    AIQualityAssessor, QualityScore
)

# Import session management
try:
    from .session_manager import SessionManager, SessionType, OperationType
except ImportError:
    from session_manager import SessionManager, SessionType, OperationType

logger = logging.getLogger(__name__)

class EnhancedEmbeddingFunctions:
    """
    MCP function implementations for enhanced embedding entry system
    Integrates Phase 1 components with session management
    """
    
    def __init__(self, db_manager, session_manager: SessionManager):
        self.db = db_manager
        self.session_manager = session_manager
        
        # Initialize Phase 1 components
        self.content_analyzer = ContentAnalyzer()
        self.quality_assessor = AIQualityAssessor()
        
        # Default configurations
        self.default_chunking_config = ChunkingConfig(
            max_tokens=512,
            min_tokens=50,
            overlap_tokens=50,
            respect_boundaries=True,
            preserve_code_blocks=True,
            preserve_headings=True,
            quality_threshold=0.7,
            strategy=ChunkingStrategy.SEMANTIC_AWARE
        )
        
        self.default_embedding_model = EmbeddingModel.ALL_MINILM_L6_V2
        self.default_cleaning_level = TextCleaningLevel.STANDARD
        
        logger.info("Enhanced embedding functions initialized")
    
    async def content_analyze_document(self,
                                     content: str,
                                     document_name: Optional[str] = None,
                                     session_id: Optional[str] = None,
                                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze document structure and content
        
        Args:
            content: Document content to analyze
            document_name: Optional document name
            session_id: Optional session ID for tracking
            metadata: Optional metadata
            
        Returns:
            Analysis results with structure information
        """
        try:
            # Perform analysis
            structure = self.content_analyzer.analyze_document_structure(content)
            
            # Prepare results
            results = {
                'document_name': document_name or 'unnamed',
                'content_type': structure.content_type.value,
                'total_elements': len(structure.elements),
                'semantic_boundaries': len(structure.semantic_boundaries),
                'statistics': structure.statistics,
                'metadata': {
                    **structure.metadata,
                    'content_hash': structure.content_hash,
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'element_summary': self._summarize_elements(structure.elements),
                'boundary_summary': self._summarize_boundaries(structure.semantic_boundaries)
            }
            
            # Track in session if provided
            if session_id:
                document_id = self._generate_document_id(document_name or content[:100])
                
                # Add document to session
                await self.session_manager.add_document_to_session(
                    session_id, document_id, document_name or 'unnamed',
                    structure.content_hash
                )
                
                # Track analysis operation
                await self.session_manager.track_chunk_operation(
                    session_id, document_id, OperationType.ANALYZED,
                    metadata={'analysis_results': results}
                )
                
                # Record metrics
                await self.session_manager.record_metric(
                    session_id, 'document_analyzed', 1.0,
                    metadata={'document_id': document_id, 'elements': len(structure.elements)}
                )
                
                results['session_id'] = session_id
                results['document_id'] = document_id
            
            return results
            
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            raise
    
    async def content_create_chunks(self,
                                  content: str,
                                  document_name: Optional[str] = None,
                                  session_id: Optional[str] = None,
                                  strategy: Optional[str] = None,
                                  max_tokens: Optional[int] = None,
                                  min_tokens: Optional[int] = None,
                                  target_realm: Optional[str] = None) -> Dict[str, Any]:
        """
        Create optimized chunks from content
        
        Args:
            content: Content to chunk
            document_name: Optional document name
            session_id: Optional session ID for tracking
            strategy: Chunking strategy (semantic_aware, markdown_structure, hybrid)
            max_tokens: Maximum tokens per chunk
            min_tokens: Minimum tokens per chunk
            target_realm: Target realm for chunks
            
        Returns:
            Chunking results with chunk information
        """
        try:
            # Analyze document structure first
            structure = self.content_analyzer.analyze_document_structure(content)
            
            # Configure chunking
            config = ChunkingConfig(
                max_tokens=max_tokens or self.default_chunking_config.max_tokens,
                min_tokens=min_tokens or self.default_chunking_config.min_tokens,
                overlap_tokens=self.default_chunking_config.overlap_tokens,
                respect_boundaries=True,
                preserve_code_blocks=True,
                preserve_headings=True,
                quality_threshold=self.default_chunking_config.quality_threshold,
                strategy=ChunkingStrategy[strategy.upper()] if strategy else ChunkingStrategy.SEMANTIC_AWARE
            )
            
            # Create chunks
            chunker = IntelligentChunker(config)
            chunks = chunker.chunk_document(structure)
            
            # Generate document ID
            document_id = self._generate_document_id(document_name or content[:100])
            
            # Process chunks
            created_chunks = []
            for i, chunk in enumerate(chunks):
                # Generate chunk ID
                chunk_id = f"chunk_{hashlib.md5(f'{document_id}_{i}_{chunk.content[:50]}'.encode()).hexdigest()[:12]}"
                
                # Create chunk in database
                if self.db and hasattr(self.db, 'create_chunk'):
                    db_chunk_id = await self._create_chunk_in_db(
                        chunk, document_id, target_realm or 'PROJECT', session_id
                    )
                    if db_chunk_id:
                        chunk_id = db_chunk_id
                
                # Track in session
                if session_id:
                    await self.session_manager.track_chunk_operation(
                        session_id, chunk_id, OperationType.CREATED,
                        metadata={
                            'chunk_type': chunk.chunk_type.value,
                            'token_count': chunk.token_count,
                            'quality_score': chunk.quality_score,
                            'line_range': f"{chunk.line_start}-{chunk.line_end}"
                        },
                        quality_score=chunk.quality_score
                    )
                
                created_chunks.append({
                    'chunk_id': chunk_id,
                    'chunk_type': chunk.chunk_type.value,
                    'content_preview': chunk.content[:100] + '...' if len(chunk.content) > 100 else chunk.content,
                    'token_count': chunk.token_count,
                    'quality_score': chunk.quality_score,
                    'line_start': chunk.line_start,
                    'line_end': chunk.line_end,
                    'metadata': chunk.metadata
                })
            
            # Update document status in session
            if session_id:
                await self.session_manager.update_document_status(
                    session_id, document_id, 'completed',
                    chunks_created=len(created_chunks)
                )
                
                # Record metrics
                await self.session_manager.record_metric(
                    session_id, 'chunks_created', len(created_chunks),
                    metadata={'document_id': document_id, 'strategy': config.strategy.value}
                )
            
            results = {
                'document_id': document_id,
                'document_name': document_name or 'unnamed',
                'chunking_strategy': config.strategy.value,
                'total_chunks': len(created_chunks),
                'chunks': created_chunks,
                'statistics': {
                    'avg_tokens_per_chunk': sum(c['token_count'] for c in created_chunks) / len(created_chunks) if created_chunks else 0,
                    'avg_quality_score': sum(c['quality_score'] for c in created_chunks) / len(created_chunks) if created_chunks else 0,
                    'content_types': list(set(c['chunk_type'] for c in created_chunks))
                }
            }
            
            if session_id:
                results['session_id'] = session_id
            
            return results
            
        except Exception as e:
            logger.error(f"Chunk creation failed: {e}")
            if session_id and 'document_id' in locals():
                await self.session_manager.update_document_status(
                    session_id, document_id, 'failed',
                    error_details=str(e)
                )
            raise
    
    async def content_assess_quality(self,
                                   chunk_ids: List[str],
                                   session_id: Optional[str] = None,
                                   include_context: bool = False) -> Dict[str, Any]:
        """
        Assess quality of chunks
        
        Args:
            chunk_ids: List of chunk IDs to assess
            session_id: Optional session ID for tracking
            include_context: Whether to include surrounding chunks for context
            
        Returns:
            Quality assessment results
        """
        try:
            assessment_results = []
            
            for chunk_id in chunk_ids:
                # Get chunk from database
                chunk_data = await self._get_chunk_from_db(chunk_id)
                if not chunk_data:
                    continue
                
                # Convert to Chunk object
                chunk = self._db_chunk_to_chunk_object(chunk_data)
                
                # Get context if requested
                context = None
                if include_context:
                    context = await self._get_chunk_context(chunk_id)
                
                # Assess quality
                quality_score = self.quality_assessor.assess_chunk_quality(chunk, context)
                
                # Track in session
                if session_id:
                    await self.session_manager.track_chunk_operation(
                        session_id, chunk_id, OperationType.QUALITY_ASSESSED,
                        metadata={
                            'overall_score': quality_score.overall_score,
                            'quality_level': quality_score.assessment_metadata['quality_level'],
                            'issues_found': len(quality_score.assessment_metadata.get('issues_found', []))
                        },
                        quality_score=quality_score.overall_score
                    )
                
                assessment_results.append({
                    'chunk_id': chunk_id,
                    'overall_score': quality_score.overall_score,
                    'quality_level': quality_score.assessment_metadata['quality_level'],
                    'dimension_scores': {
                        dim.value: score 
                        for dim, score in quality_score.dimension_scores.items()
                    },
                    'confidence': quality_score.confidence,
                    'issues': [
                        {
                            'dimension': issue.dimension.value,
                            'severity': issue.severity,
                            'description': issue.description,
                            'suggested_fix': issue.suggested_fix
                        }
                        for issue in quality_score.assessment_metadata.get('issues_found', [])
                    ],
                    'improvement_suggestions': quality_score.assessment_metadata.get('improvement_suggestions', [])
                })
            
            # Record metrics
            if session_id and assessment_results:
                avg_score = sum(r['overall_score'] for r in assessment_results) / len(assessment_results)
                await self.session_manager.record_metric(
                    session_id, 'chunks_assessed', len(assessment_results),
                    metadata={'average_quality_score': avg_score}
                )
            
            return {
                'total_assessed': len(assessment_results),
                'assessments': assessment_results,
                'summary': {
                    'average_quality': sum(r['overall_score'] for r in assessment_results) / len(assessment_results) if assessment_results else 0,
                    'quality_distribution': self._calculate_quality_distribution(assessment_results),
                    'common_issues': self._identify_common_issues(assessment_results)
                },
                'session_id': session_id
            } if session_id else {
                'total_assessed': len(assessment_results),
                'assessments': assessment_results,
                'summary': {
                    'average_quality': sum(r['overall_score'] for r in assessment_results) / len(assessment_results) if assessment_results else 0,
                    'quality_distribution': self._calculate_quality_distribution(assessment_results),
                    'common_issues': self._identify_common_issues(assessment_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            raise
    
    async def content_optimize_embeddings(self,
                                        chunk_ids: List[str],
                                        session_id: Optional[str] = None,
                                        model: Optional[str] = None,
                                        cleaning_level: Optional[str] = None,
                                        batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize chunks for embedding generation
        
        Args:
            chunk_ids: List of chunk IDs to optimize
            session_id: Optional session ID for tracking
            model: Embedding model name
            cleaning_level: Text cleaning level (minimal, standard, aggressive)
            batch_size: Batch size for processing
            
        Returns:
            Optimization results
        """
        try:
            # Configure optimizer
            embedding_model = EmbeddingModel[model.upper()] if model else self.default_embedding_model
            cleaning = TextCleaningLevel[cleaning_level.upper()] if cleaning_level else self.default_cleaning_level
            optimizer = EmbeddingOptimizer(embedding_model, cleaning)
            
            # Process chunks
            optimized_chunks = []
            chunks_to_process = []
            
            for chunk_id in chunk_ids:
                # Get chunk from database
                chunk_data = await self._get_chunk_from_db(chunk_id)
                if not chunk_data:
                    continue
                
                # Convert to Chunk object
                chunk = self._db_chunk_to_chunk_object(chunk_data)
                chunks_to_process.append((chunk_id, chunk))
            
            # Batch processing
            batch_size = batch_size or 32
            batches = optimizer.batch_optimize_embeddings(
                [chunk for _, chunk in chunks_to_process], 
                batch_size
            )
            
            # Process batches
            for batch_idx, batch in enumerate(batches):
                # Optimize text for each chunk in batch
                for chunk_idx, chunk in enumerate(batch):
                    chunk_id = chunks_to_process[batch_idx * batch_size + chunk_idx][0]
                    
                    # Optimize text
                    optimized_text = optimizer.prepare_text_for_embedding(chunk)
                    
                    # Track in session
                    if session_id:
                        await self.session_manager.track_chunk_operation(
                            session_id, chunk_id, OperationType.EMBEDDED,
                            metadata={
                                'model': embedding_model.value,
                                'cleaning_level': cleaning.value,
                                'original_length': len(optimized_text.original_text),
                                'optimized_length': len(optimized_text.cleaned_text),
                                'tokens_removed': optimized_text.tokens_removed
                            }
                        )
                    
                    optimized_chunks.append({
                        'chunk_id': chunk_id,
                        'original_length': len(optimized_text.original_text),
                        'optimized_length': len(optimized_text.cleaned_text),
                        'tokens_removed': optimized_text.tokens_removed,
                        'compression_ratio': optimized_text.metadata['compression_ratio'],
                        'cleaning_level': optimized_text.cleaning_level.value,
                        'formatting_preserved': optimized_text.formatting_preserved
                    })
            
            # Generate embeddings if service available
            embeddings_generated = 0
            if hasattr(self.db, 'embedding_service') and self.db.embedding_service:
                embeddings = optimizer.generate_optimized_embeddings(
                    [chunk for _, chunk in chunks_to_process],
                    self.db.embedding_service
                )
                embeddings_generated = len(embeddings)
                
                # Store embeddings
                for embedding in embeddings:
                    if session_id:
                        await self.session_manager.track_chunk_operation(
                            session_id, embedding.chunk_id, OperationType.EMBEDDED,
                            metadata={'embedding_id': embedding.embedding_id},
                            embedding_id=embedding.embedding_id
                        )
            
            # Record metrics
            if session_id:
                await self.session_manager.record_metric(
                    session_id, 'chunks_optimized', len(optimized_chunks),
                    metadata={
                        'model': embedding_model.value,
                        'embeddings_generated': embeddings_generated
                    }
                )
            
            return {
                'total_optimized': len(optimized_chunks),
                'embeddings_generated': embeddings_generated,
                'model': embedding_model.value,
                'cleaning_level': cleaning.value,
                'optimizations': optimized_chunks,
                'summary': {
                    'avg_compression_ratio': sum(c['compression_ratio'] for c in optimized_chunks) / len(optimized_chunks) if optimized_chunks else 0,
                    'total_tokens_removed': sum(c['tokens_removed'] for c in optimized_chunks),
                    'batch_count': len(batches)
                },
                'session_id': session_id
            } if session_id else {
                'total_optimized': len(optimized_chunks),
                'embeddings_generated': embeddings_generated,
                'model': embedding_model.value,
                'cleaning_level': cleaning.value,
                'optimizations': optimized_chunks,
                'summary': {
                    'avg_compression_ratio': sum(c['compression_ratio'] for c in optimized_chunks) / len(optimized_chunks) if optimized_chunks else 0,
                    'total_tokens_removed': sum(c['tokens_removed'] for c in optimized_chunks),
                    'batch_count': len(batches)
                }
            }
            
        except Exception as e:
            logger.error(f"Embedding optimization failed: {e}")
            raise
    
    # Session management functions
    
    async def session_create(self,
                           session_type: str,
                           created_by: str,
                           description: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new embedding session
        
        Args:
            session_type: Type of session (analysis, ingestion, curation, mixed)
            created_by: User/system creating the session
            description: Optional session description
            metadata: Optional session metadata
            
        Returns:
            Session creation result
        """
        try:
            # Get realm from context
            realm_id = self.db.project_realm if hasattr(self.db, 'project_realm') else 'PROJECT'
            
            # Add description to metadata
            if metadata is None:
                metadata = {}
            if description:
                metadata['description'] = description
            
            # Create session
            session_id = await self.session_manager.create_session(
                SessionType[session_type.upper()],
                realm_id,
                created_by,
                metadata
            )
            
            return {
                'session_id': session_id,
                'session_type': session_type,
                'realm_id': realm_id,
                'created_by': created_by,
                'status': 'active',
                'created_at': datetime.now().isoformat(),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            raise
    
    async def session_get_state(self, session_id: str) -> Dict[str, Any]:
        """
        Get current session state and progress
        
        Args:
            session_id: Session ID
            
        Returns:
            Session state and progress information
        """
        try:
            # Get session state
            state = await self.session_manager.get_session_state(session_id)
            if not state:
                return {'error': 'Session not found'}
            
            # Get progress
            progress = await self.session_manager.get_session_progress(session_id)
            
            return {
                'session_id': session_id,
                'state': {
                    'current_document': state.current_document,
                    'current_chunk_index': state.current_chunk_index,
                    'processed_chunks': len(state.processed_chunks),
                    'failed_chunks': len(state.failed_chunks),
                    'last_saved': state.last_saved.isoformat(),
                    'metrics': state.metrics,
                    'checkpoints': state.checkpoints
                },
                'progress': progress
            }
            
        except Exception as e:
            logger.error(f"Failed to get session state: {e}")
            raise
    
    async def session_complete(self, session_id: str) -> Dict[str, Any]:
        """
        Complete and finalize a session
        
        Args:
            session_id: Session ID
            
        Returns:
            Completion result
        """
        try:
            # Get final progress
            progress = await self.session_manager.get_session_progress(session_id)
            
            # Complete session
            success = await self.session_manager.complete_session(session_id)
            
            return {
                'session_id': session_id,
                'completed': success,
                'final_stats': {
                    'total_documents': progress.get('total_documents', 0),
                    'total_chunks': progress.get('total_chunks', 0),
                    'avg_quality_score': progress.get('avg_quality_score', 0),
                    'processing_duration': progress.get('processing_duration_ms', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to complete session: {e}")
            raise
    
    # Helper methods
    
    def _generate_document_id(self, identifier: str) -> str:
        """Generate unique document ID"""
        return f"doc_{hashlib.md5(identifier.encode()).hexdigest()[:12]}"
    
    def _summarize_elements(self, elements) -> Dict[str, int]:
        """Summarize element types and counts"""
        summary = {}
        for element in elements:
            element_type = element.element_type.value
            summary[element_type] = summary.get(element_type, 0) + 1
        return summary
    
    def _summarize_boundaries(self, boundaries) -> Dict[str, int]:
        """Summarize boundary types and counts"""
        summary = {}
        for boundary in boundaries:
            boundary_type = boundary.boundary_type
            summary[boundary_type] = summary.get(boundary_type, 0) + 1
        return summary
    
    async def _create_chunk_in_db(self, chunk: Chunk, document_id: str, 
                                 target_realm: str, session_id: Optional[str]) -> Optional[str]:
        """Create chunk in database"""
        # This would integrate with the existing database methods
        # For now, return None to use generated ID
        return None
    
    async def _get_chunk_from_db(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk from database"""
        # This would integrate with existing database methods
        # For now, return mock data
        return {
            'chunk_id': chunk_id,
            'content': 'Mock chunk content for testing',
            'chunk_type': 'paragraph',
            'token_count': 100,
            'quality_score': 0.8,
            'metadata': {}
        }
    
    def _db_chunk_to_chunk_object(self, db_chunk: Dict[str, Any]) -> Chunk:
        """Convert database chunk to Chunk object"""
        from libraries.content_processing.intelligent_chunker import ChunkType
        
        return Chunk(
            chunk_id=db_chunk['chunk_id'],
            content=db_chunk['content'],
            chunk_type=ChunkType.PARAGRAPH,  # Simplified
            line_start=1,
            line_end=10,
            token_count=db_chunk.get('token_count', 100),
            elements=[],  # Simplified
            metadata=db_chunk.get('metadata', {}),
            quality_score=db_chunk.get('quality_score', 0.5)
        )
    
    async def _get_chunk_context(self, chunk_id: str, context_size: int = 2) -> List[Chunk]:
        """Get surrounding chunks for context"""
        # This would get neighboring chunks from database
        return []
    
    def _calculate_quality_distribution(self, assessments: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of quality levels"""
        distribution = {'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0}
        for assessment in assessments:
            level = assessment['quality_level']
            if level in distribution:
                distribution[level] += 1
        return distribution
    
    def _identify_common_issues(self, assessments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify most common quality issues"""
        issue_counts = {}
        for assessment in assessments:
            for issue in assessment.get('issues', []):
                dimension = issue['dimension']
                issue_counts[dimension] = issue_counts.get(dimension, 0) + 1
        
        # Sort by frequency
        common_issues = [
            {'dimension': dim, 'frequency': count}
            for dim, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return common_issues[:5]  # Top 5 issues