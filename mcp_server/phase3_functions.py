#!/usr/bin/env python3
"""
Phase 3 MCP Functions for Enhanced Multi-Embedding Entry System
Implements knowledge management and session tracking functions
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import Phase 3 components
from libraries.knowledge_management import (
    KnowledgeManagementSystem,
    KnowledgeType,
    RelationshipType
)
from libraries.session_tracking import (
    SessionTrackingSystem,
    ActionType,
    ActionPriority
)

logger = logging.getLogger(__name__)

class Phase3Functions:
    """
    MCP function implementations for Phase 3 knowledge management and session tracking
    """
    
    def __init__(self, db_manager):
        self.db = db_manager
        
        # Initialize systems
        self.knowledge_system = KnowledgeManagementSystem(db_connection=db_manager)
        self.session_tracker = SessionTrackingSystem(db_connection=db_manager)
        
        logger.info("Phase 3 functions initialized")
    
    # Knowledge Management Functions
    
    async def knowledge_ingest_document(self,
                                      document_path: str,
                                      title: Optional[str] = None,
                                      knowledge_type: Optional[str] = None,
                                      tags: Optional[List[str]] = None,
                                      session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Ingest a knowledge document into the system
        
        Args:
            document_path: Path to the document to ingest
            title: Optional document title
            knowledge_type: Type of knowledge (documentation, code_pattern, etc.)
            tags: Optional tags for categorization
            session_id: Optional session ID for tracking
            
        Returns:
            Ingestion results
        """
        try:
            # Validate knowledge type
            k_type = None
            if knowledge_type:
                try:
                    k_type = KnowledgeType[knowledge_type.upper()]
                except KeyError:
                    logger.warning(f"Invalid knowledge type: {knowledge_type}, using auto-detection")
            
            # Track action if session provided
            if session_id and hasattr(self, 'session_tracker'):
                await self.session_tracker.track_session_action(
                    session_id=session_id,
                    action_type=ActionType.ANALYZE_DOCUMENT,
                    description=f"Ingesting knowledge document: {Path(document_path).name}",
                    details={
                        'document_path': document_path,
                        'knowledge_type': knowledge_type
                    },
                    priority=ActionPriority.HIGH
                )
            
            # Ingest document
            document = await self.knowledge_system.ingest_knowledge_document(
                doc_path=document_path,
                title=title,
                knowledge_type=k_type,
                tags=tags,
                session_id=session_id
            )
            
            # Get chunk count
            chunk_count = len(self.knowledge_system.knowledge_graph.nodes)
            
            return {
                'document_id': document.document_id,
                'title': document.title,
                'knowledge_type': document.knowledge_type.value,
                'chunks_created': chunk_count,
                'tags': document.tags,
                'metadata': document.metadata,
                'ingested_date': document.ingested_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest document: {e}")
            raise
    
    async def knowledge_discover_relationships(self,
                                             chunk_ids: Optional[List[str]] = None,
                                             similarity_threshold: float = 0.7,
                                             session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Discover relationships between knowledge chunks
        
        Args:
            chunk_ids: Optional list of chunk IDs to analyze (None for all)
            similarity_threshold: Threshold for semantic similarity
            session_id: Optional session ID for tracking
            
        Returns:
            Discovered relationships
        """
        try:
            # Get chunks to analyze
            if chunk_ids:
                chunks = [
                    self.knowledge_system.knowledge_graph.nodes.get(chunk_id)
                    for chunk_id in chunk_ids
                    if chunk_id in self.knowledge_system.knowledge_graph.nodes
                ]
                chunks = [c for c in chunks if c is not None]
            else:
                chunks = list(self.knowledge_system.knowledge_graph.nodes.values())
            
            if not chunks:
                return {
                    'error': 'No chunks found to analyze',
                    'chunk_ids': chunk_ids
                }
            
            # Track action
            if session_id and hasattr(self, 'session_tracker'):
                await self.session_tracker.track_session_action(
                    session_id=session_id,
                    action_type=ActionType.DISCOVER_RELATIONSHIP,
                    description=f"Discovering relationships for {len(chunks)} chunks",
                    details={
                        'chunk_count': len(chunks),
                        'similarity_threshold': similarity_threshold
                    }
                )
            
            # Discover relationships
            graph = await self.knowledge_system.establish_cross_references(
                chunks,
                similarity_threshold
            )
            
            # Format relationships
            relationships = []
            for edge in graph.edges:
                relationships.append({
                    'source_chunk': edge[0],
                    'target_chunk': edge[1],
                    'relationship_type': edge[2].value
                })
            
            # Format clusters
            clusters = {}
            for cluster_id, chunk_ids in graph.clusters.items():
                clusters[cluster_id] = {
                    'chunk_ids': chunk_ids,
                    'size': len(chunk_ids)
                }
            
            return {
                'total_relationships': len(relationships),
                'relationships': relationships[:50],  # Limit to first 50
                'total_clusters': len(clusters),
                'clusters': clusters,
                'chunks_analyzed': len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Failed to discover relationships: {e}")
            raise
    
    async def knowledge_optimize_retrieval(self,
                                         usage_window_days: int = 7,
                                         session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize knowledge retrieval based on usage patterns
        
        Args:
            usage_window_days: Days of usage data to analyze
            session_id: Optional session ID for tracking
            
        Returns:
            Retrieval optimization recommendations
        """
        try:
            # Get usage data from database
            usage_data = {}
            if self.db and hasattr(self.db, 'get_usage_analytics'):
                # This would fetch real usage data
                usage_data = {
                    'access_logs': [],  # Would be populated from DB
                    'window_days': usage_window_days
                }
            
            # Track action
            if session_id and hasattr(self, 'session_tracker'):
                await self.session_tracker.track_session_action(
                    session_id=session_id,
                    action_type=ActionType.CUSTOM,
                    description=f"Optimizing retrieval patterns for {usage_window_days} days",
                    details={'usage_window_days': usage_window_days}
                )
            
            # Optimize retrieval
            optimization = await self.knowledge_system.optimize_retrieval_patterns(usage_data)
            
            return {
                'hot_chunks': optimization.hot_chunks[:20],
                'cache_recommendations': optimization.cache_recommendations,
                'chunk_clusters': {
                    cluster_id: chunks[:5]  # Limit chunks per cluster
                    for cluster_id, chunks in optimization.chunk_clusters.items()
                },
                'access_patterns': {
                    chunk_id: next_chunks[:3]  # Limit next chunks
                    for chunk_id, next_chunks in optimization.access_patterns.items()
                },
                'prefetch_patterns': optimization.prefetch_patterns
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize retrieval: {e}")
            raise
    
    async def knowledge_get_related(self,
                                  chunk_id: str,
                                  relationship_types: Optional[List[str]] = None,
                                  max_depth: int = 2,
                                  session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get chunks related to a given chunk
        
        Args:
            chunk_id: Source chunk ID
            relationship_types: Optional filter for relationship types
            max_depth: Maximum depth to traverse
            session_id: Optional session ID for tracking
            
        Returns:
            Related chunks with relationships
        """
        try:
            # Validate relationship types
            rel_types = None
            if relationship_types:
                rel_types = []
                for rt in relationship_types:
                    try:
                        rel_types.append(RelationshipType[rt.upper()])
                    except KeyError:
                        logger.warning(f"Invalid relationship type: {rt}")
            
            # Track action
            if session_id and hasattr(self, 'session_tracker'):
                await self.session_tracker.track_session_action(
                    session_id=session_id,
                    action_type=ActionType.RETRIEVE_KNOWLEDGE,
                    description=f"Getting related chunks for {chunk_id}",
                    details={
                        'chunk_id': chunk_id,
                        'max_depth': max_depth
                    }
                )
            
            # Get related chunks
            related_chunks = {}
            visited = set()
            to_visit = [(chunk_id, 0)]
            
            while to_visit and len(related_chunks) < 50:  # Limit total results
                current_id, depth = to_visit.pop(0)
                
                if current_id in visited or depth > max_depth:
                    continue
                
                visited.add(current_id)
                
                # Get directly related chunks
                related = self.knowledge_system.knowledge_graph.get_related_chunks(
                    current_id, rel_types
                )
                
                for rel_id in related:
                    if rel_id not in related_chunks:
                        # Get chunk details
                        chunk = self.knowledge_system.knowledge_graph.nodes.get(rel_id)
                        if chunk:
                            # Find relationship type
                            rel_type = None
                            for edge in self.knowledge_system.knowledge_graph.edges:
                                if (edge[0] == current_id and edge[1] == rel_id) or \
                                   (edge[1] == current_id and edge[0] == rel_id):
                                    rel_type = edge[2].value
                                    break
                            
                            related_chunks[rel_id] = {
                                'chunk_id': rel_id,
                                'content_preview': chunk.content[:200] + '...' if len(chunk.content) > 200 else chunk.content,
                                'knowledge_type': chunk.knowledge_type.value,
                                'quality_score': chunk.quality_score,
                                'importance_score': chunk.importance_score,
                                'relationship_type': rel_type,
                                'depth': depth + 1
                            }
                            
                            # Add to visit queue if not at max depth
                            if depth + 1 < max_depth:
                                to_visit.append((rel_id, depth + 1))
            
            return {
                'source_chunk': chunk_id,
                'total_related': len(related_chunks),
                'related_chunks': list(related_chunks.values()),
                'max_depth_reached': max_depth
            }
            
        except Exception as e:
            logger.error(f"Failed to get related chunks: {e}")
            raise
    
    # Session Tracking Functions
    
    async def session_create_operational(self,
                                       session_type: str = "general",
                                       user_id: Optional[str] = None,
                                       description: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new operational tracking session
        
        Args:
            session_type: Type of session
            user_id: Optional user identifier
            description: Optional session description
            
        Returns:
            Session creation result
        """
        try:
            result = await self.session_tracker.create_operational_session(
                session_type=session_type,
                user_id=user_id,
                description=description
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create operational session: {e}")
            raise
    
    async def session_track_action(self,
                                 session_id: str,
                                 action_type: str,
                                 description: str,
                                 details: Optional[Dict[str, Any]] = None,
                                 priority: Optional[str] = None) -> Dict[str, Any]:
        """
        Track an action in the operational session
        
        Args:
            session_id: Session identifier
            action_type: Type of action
            description: Human-readable description
            details: Optional action details
            priority: Optional priority level
            
        Returns:
            Tracking result
        """
        try:
            # Validate action type
            try:
                a_type = ActionType[action_type.upper()]
            except KeyError:
                a_type = ActionType.CUSTOM
            
            # Validate priority
            a_priority = None
            if priority:
                try:
                    a_priority = ActionPriority[priority.upper()]
                except KeyError:
                    logger.warning(f"Invalid priority: {priority}")
            
            # Track action
            entry = await self.session_tracker.track_session_action(
                session_id=session_id,
                action_type=a_type,
                description=description,
                details=details,
                priority=a_priority
            )
            
            return {
                'entry_id': entry.entry_id,
                'action_id': entry.action.action_id,
                'timestamp': entry.action.timestamp.isoformat(),
                'action_type': entry.action.action_type.value,
                'description': entry.action.description,
                'priority': entry.action.priority.value
            }
            
        except Exception as e:
            logger.error(f"Failed to track action: {e}")
            raise
    
    async def session_get_recap(self, session_id: str) -> Dict[str, Any]:
        """
        Get a recap of the operational session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session recap
        """
        try:
            recap = await self.session_tracker.generate_session_recap(session_id)
            
            return {
                'session_id': recap.session_id,
                'start_time': recap.start_time.isoformat(),
                'end_time': recap.end_time.isoformat() if recap.end_time else None,
                'total_actions': recap.total_actions,
                'accomplishments': recap.accomplishments,
                'pending_tasks': recap.pending_tasks,
                'context_summary': recap.context_summary,
                'suggested_next_steps': recap.suggested_next_steps,
                'recap_prompt': recap.to_prompt(),
                'key_actions': [
                    {
                        'timestamp': action.timestamp.isoformat(),
                        'description': action.description,
                        'type': action.action_type.value,
                        'priority': action.priority.value
                    }
                    for action in recap.key_actions[:5]
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get session recap: {e}")
            raise
    
    async def session_prime_context(self, session_id: str) -> Dict[str, Any]:
        """
        Prime context for session resumption
        
        Args:
            session_id: Session identifier
            
        Returns:
            Context primer information
        """
        try:
            primer = await self.session_tracker.prime_context_from_session(session_id)
            context = primer.to_context()
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to prime context: {e}")
            raise
    
    async def session_list_recent(self,
                                user_id: Optional[str] = None,
                                limit: int = 10) -> Dict[str, Any]:
        """
        List recent operational sessions
        
        Args:
            user_id: Optional filter by user
            limit: Maximum number of sessions to return
            
        Returns:
            List of recent sessions
        """
        try:
            sessions = await self.session_tracker.get_recent_sessions(
                user_id=user_id,
                limit=limit
            )
            
            return {
                'total_sessions': len(sessions),
                'sessions': sessions
            }
            
        except Exception as e:
            logger.error(f"Failed to list recent sessions: {e}")
            raise
    
    async def session_close(self, session_id: str) -> Dict[str, Any]:
        """
        Close an operational session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Final session recap
        """
        try:
            recap = await self.session_tracker.close_session(session_id)
            
            return {
                'session_id': recap.session_id,
                'final_recap': recap.to_prompt(),
                'total_actions': recap.total_actions,
                'accomplishments': recap.accomplishments,
                'duration_minutes': int((recap.end_time - recap.start_time).total_seconds() / 60) if recap.end_time else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to close session: {e}")
            raise