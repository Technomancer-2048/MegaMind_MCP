"""
Consolidated MCP Functions - Phase 1 Implementation
GitHub Issue #19: Function Name Standardization

This module implements the new standardized function architecture with master functions
that intelligently route to existing subfunctions while maintaining backward compatibility.

Total Functions: 44 → 19 (57% reduction)
Function Classes: SEARCH, CONTENT, PROMOTION, SESSION, AI, ANALYTICS
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class ConsolidatedMCPFunctions:
    """
    Consolidated MCP Functions implementing the standardized architecture
    with master functions that intelligently route to existing subfunctions.
    """
    
    def __init__(self, db_manager, session_manager=None):
        """
        Initialize with database manager and optional session manager.
        
        Args:
            db_manager: RealmAwareMegaMindDatabase instance
            session_manager: Optional session manager for session operations
        """
        self.db = db_manager
        self.session_manager = session_manager
        
    # ========================================
    # 🔍 SEARCH CLASS - 3 Master Functions
    # ========================================
    
    async def search_query(self, query: str, search_type: str = "hybrid", 
                          limit: int = 10, threshold: float = 0.7,
                          reference_chunk_id: str = None) -> Dict[str, Any]:
        """
        Master search function with intelligent routing.
        
        Routes to appropriate search function based on parameters:
        - Default: hybrid search across realms
        - search_type="semantic": pure semantic search
        - search_type="similarity": similarity search (requires reference_chunk_id)
        - search_type="keyword": keyword-based search
        
        Args:
            query: Search query text
            search_type: "hybrid", "semantic", "similarity", "keyword"
            limit: Maximum number of results
            threshold: Similarity threshold (for semantic/similarity searches)
            reference_chunk_id: Required for similarity search
            
        Returns:
            Dict with search results and metadata
        """
        try:
            logger.info(f"Master search_query: type={search_type}, query='{query[:50]}...'")
            
            if search_type == "similarity":
                if not reference_chunk_id:
                    raise ValueError("reference_chunk_id required for similarity search")
                
                # Route to similarity search
                results = self.db.search_chunks_by_similarity_dual_realm(
                    reference_chunk_id=reference_chunk_id,
                    limit=limit,
                    threshold=threshold
                )
                
            elif search_type == "semantic":
                # Route to pure semantic search
                results = self.db.search_chunks_semantic_dual_realm(
                    query=query,
                    limit=limit,
                    threshold=threshold
                )
                
            elif search_type == "keyword":
                # Route to keyword search (implemented as hybrid with keyword focus)
                results = self.db.search_chunks_dual_realm(
                    query=query,
                    limit=limit,
                    search_type="keyword"
                )
                
            else:  # hybrid (default)
                # Route to main hybrid search
                results = self.db.search_chunks_dual_realm(
                    query=query,
                    limit=limit,
                    search_type="hybrid"
                )
            
            return {
                "success": True,
                "search_type": search_type,
                "query": query,
                "results": results,
                "count": len(results) if results else 0,
                "routed_to": f"search_chunks_{search_type}_dual_realm" if search_type != "hybrid" else "search_chunks_dual_realm"
            }
            
        except Exception as e:
            logger.error(f"Master search_query error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "search_type": search_type,
                "query": query
            }
    
    async def search_related(self, chunk_id: str, max_depth: int = 2,
                           include_hot_contexts: bool = False,
                           model_type: str = "sonnet") -> Dict[str, Any]:
        """
        Master function for finding related chunks and contexts.
        
        Routes to:
        - get_related_chunks_dual_realm for relationship traversal
        - get_hot_contexts_dual_realm for frequently accessed chunks
        
        Args:
            chunk_id: Reference chunk ID
            max_depth: Relationship traversal depth
            include_hot_contexts: Also include hot contexts
            model_type: Model type for hot contexts
            
        Returns:
            Dict with related chunks and optional hot contexts
        """
        try:
            logger.info(f"Master search_related: chunk_id={chunk_id}, depth={max_depth}")
            
            # Get related chunks
            related_chunks = self.db.get_related_chunks_dual_realm(
                chunk_id=chunk_id,
                max_depth=max_depth
            )
            
            result = {
                "success": True,
                "chunk_id": chunk_id,
                "related_chunks": related_chunks,
                "max_depth": max_depth
            }
            
            # Optionally include hot contexts
            if include_hot_contexts:
                hot_contexts = self.db.get_hot_contexts_dual_realm(
                    model_type=model_type,
                    limit=20
                )
                result["hot_contexts"] = hot_contexts
                result["model_type"] = model_type
            
            return result
            
        except Exception as e:
            logger.error(f"Master search_related error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "chunk_id": chunk_id
            }
    
    async def search_retrieve(self, chunk_id: str, include_relationships: bool = True,
                            track_access: bool = True, query_context: str = "") -> Dict[str, Any]:
        """
        Master function for retrieving specific chunks by ID.
        
        Routes to:
        - get_chunk_dual_realm for chunk retrieval
        - track_access_dual_realm for usage tracking
        
        Args:
            chunk_id: Unique chunk identifier
            include_relationships: Include related chunks
            track_access: Track access for analytics
            query_context: Context for access tracking
            
        Returns:
            Dict with chunk data and metadata
        """
        try:
            logger.info(f"Master search_retrieve: chunk_id={chunk_id}")
            
            # Retrieve the chunk
            chunk = self.db.get_chunk_dual_realm(
                chunk_id=chunk_id,
                include_relationships=include_relationships
            )
            
            if not chunk:
                return {
                    "success": False,
                    "error": "Chunk not found",
                    "chunk_id": chunk_id
                }
            
            # Track access if enabled
            if track_access:
                self.db.track_access_dual_realm(
                    chunk_id=chunk_id,
                    query_context=query_context or f"retrieve_{chunk_id}"
                )
            
            return {
                "success": True,
                "chunk_id": chunk_id,
                "chunk": chunk,
                "include_relationships": include_relationships,
                "access_tracked": track_access
            }
            
        except Exception as e:
            logger.error(f"Master search_retrieve error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "chunk_id": chunk_id
            }
    
    # ========================================
    # 📝 CONTENT CLASS - 4 Master Functions
    # ========================================
    
    async def content_create(self, content: str, source_document: str,
                           section_path: str = "", session_id: str = "",
                           target_realm: str = "PROJECT", 
                           create_relationships: bool = True,
                           relationship_targets: List[str] = None,
                           relationship_type: str = "related") -> Dict[str, Any]:
        """
        Master function for creating new chunks and relationships.
        
        Routes to:
        - create_chunk_dual_realm for chunk creation
        - add_relationship_dual_realm for relationship creation
        - batch_generate_embeddings_dual_realm for embedding generation
        
        Args:
            content: Chunk content
            source_document: Source document name
            section_path: Section path within document
            session_id: Session ID for tracking
            target_realm: Target realm for creation
            create_relationships: Create relationships with existing chunks
            relationship_targets: Specific chunk IDs to relate to
            relationship_type: Type of relationship to create
            
        Returns:
            Dict with creation results and metadata
        """
        try:
            logger.info(f"Master content_create: source={source_document}, session={session_id}")
            
            # Create the chunk
            chunk_result = self.db.create_chunk_dual_realm(
                content=content,
                source_document=source_document,
                section_path=section_path,
                session_id=session_id,
                target_realm=target_realm
            )
            
            if not chunk_result.get("success"):
                return chunk_result
            
            chunk_id = chunk_result["chunk_id"]
            
            # Create relationships if requested
            relationships_created = []
            if create_relationships and relationship_targets:
                for target_id in relationship_targets:
                    try:
                        rel_result = self.db.add_relationship_dual_realm(
                            chunk_id_1=chunk_id,
                            chunk_id_2=target_id,
                            relationship_type=relationship_type,
                            session_id=session_id
                        )
                        if rel_result.get("success"):
                            relationships_created.append(target_id)
                    except Exception as e:
                        logger.warning(f"Failed to create relationship {chunk_id} -> {target_id}: {str(e)}")
            
            # Generate embeddings for the new chunk
            embedding_result = self.db.batch_generate_embeddings_dual_realm(
                chunk_ids=[chunk_id],
                realm_id=target_realm
            )
            
            return {
                "success": True,
                "chunk_id": chunk_id,
                "chunk_result": chunk_result,
                "relationships_created": relationships_created,
                "embedding_result": embedding_result,
                "target_realm": target_realm
            }
            
        except Exception as e:
            logger.error(f"Master content_create error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "source_document": source_document
            }
    
    async def content_update(self, chunk_id: str, new_content: str,
                           session_id: str = "", update_embeddings: bool = True,
                           update_relationships: bool = False,
                           new_relationships: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Master function for modifying existing chunks.
        
        Routes to:
        - update_chunk_dual_realm for content updates
        - batch_generate_embeddings_dual_realm for embedding updates
        - add_relationship_dual_realm for new relationships
        
        Args:
            chunk_id: Chunk ID to update
            new_content: Updated content
            session_id: Session ID for tracking
            update_embeddings: Regenerate embeddings after update
            update_relationships: Update relationships
            new_relationships: New relationships to add [{"target_id": "...", "type": "..."}]
            
        Returns:
            Dict with update results and metadata
        """
        try:
            logger.info(f"Master content_update: chunk_id={chunk_id}, session={session_id}")
            
            # Update the chunk content
            update_result = self.db.update_chunk_dual_realm(
                chunk_id=chunk_id,
                new_content=new_content,
                session_id=session_id
            )
            
            if not update_result.get("success"):
                return update_result
            
            # Update embeddings if requested
            embedding_result = None
            if update_embeddings:
                embedding_result = self.db.batch_generate_embeddings_dual_realm(
                    chunk_ids=[chunk_id],
                    realm_id=""  # Use current realm
                )
            
            # Add new relationships if requested
            relationships_added = []
            if update_relationships and new_relationships:
                for rel in new_relationships:
                    try:
                        rel_result = self.db.add_relationship_dual_realm(
                            chunk_id_1=chunk_id,
                            chunk_id_2=rel["target_id"],
                            relationship_type=rel.get("type", "related"),
                            session_id=session_id
                        )
                        if rel_result.get("success"):
                            relationships_added.append(rel)
                    except Exception as e:
                        logger.warning(f"Failed to add relationship: {str(e)}")
            
            return {
                "success": True,
                "chunk_id": chunk_id,
                "update_result": update_result,
                "embedding_result": embedding_result,
                "relationships_added": relationships_added,
                "embeddings_updated": update_embeddings
            }
            
        except Exception as e:
            logger.error(f"Master content_update error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "chunk_id": chunk_id
            }
    
    async def content_process(self, content: str, document_name: str,
                            session_id: str = "", strategy: str = "auto",
                            max_tokens: int = 150, target_realm: str = "PROJECT",
                            analyze_first: bool = True, optimize_after: bool = True,
                            batch_size: int = 10) -> Dict[str, Any]:
        """
        Master document processing function.
        
        Routes to:
        - content_analyze_document for document analysis
        - content_create_chunks for intelligent chunking
        - content_optimize_embeddings for optimization
        - content_assess_quality for quality assessment
        
        Args:
            content: Document content to process
            document_name: Name of the document
            session_id: Session ID for tracking
            strategy: Chunking strategy ("auto", "semantic", "fixed")
            max_tokens: Maximum tokens per chunk
            target_realm: Target realm for chunks
            analyze_first: Analyze document structure first
            optimize_after: Optimize embeddings after creation
            batch_size: Batch size for processing
            
        Returns:
            Dict with processing results and metadata
        """
        try:
            logger.info(f"Master content_process: document={document_name}, strategy={strategy}")
            
            result = {
                "success": True,
                "document_name": document_name,
                "strategy": strategy,
                "target_realm": target_realm
            }
            
            # Phase 1: Document analysis
            if analyze_first:
                analysis_result = self.db.content_analyze_document_dual_realm(
                    content=content,
                    document_name=document_name,
                    session_id=session_id,
                    metadata={"strategy": strategy}
                )
                result["analysis"] = analysis_result
            
            # Phase 2: Create chunks
            chunking_result = self.db.content_create_chunks_dual_realm(
                content=content,
                document_name=document_name,
                session_id=session_id,
                strategy=strategy,
                max_tokens=max_tokens,
                target_realm=target_realm
            )
            result["chunking"] = chunking_result
            
            if not chunking_result.get("success"):
                return chunking_result
            
            chunk_ids = chunking_result.get("chunk_ids", [])
            
            # Phase 3: Optimize embeddings
            if optimize_after and chunk_ids:
                optimization_result = self.db.content_optimize_embeddings_dual_realm(
                    chunk_ids=chunk_ids,
                    session_id=session_id,
                    model="default",
                    cleaning_level="standard",
                    batch_size=batch_size
                )
                result["optimization"] = optimization_result
            
            # Phase 4: Quality assessment
            if chunk_ids:
                quality_result = self.db.content_assess_quality_dual_realm(
                    chunk_ids=chunk_ids,
                    session_id=session_id,
                    include_context=True
                )
                result["quality_assessment"] = quality_result
            
            result["chunks_created"] = len(chunk_ids)
            result["chunk_ids"] = chunk_ids
            
            return result
            
        except Exception as e:
            logger.error(f"Master content_process error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "document_name": document_name
            }
    
    async def content_manage(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Master content management function.
        
        Routes to:
        - knowledge_ingest_document for document ingestion
        - knowledge_discover_relationships for relationship discovery
        - knowledge_optimize_retrieval for retrieval optimization
        - knowledge_get_related for related content retrieval
        
        Args:
            action: Action to perform ("ingest", "discover", "optimize", "get_related")
            **kwargs: Action-specific parameters
            
        Returns:
            Dict with action results and metadata
        """
        try:
            logger.info(f"Master content_manage: action={action}")
            
            if action == "ingest":
                # Route to document ingestion
                return self.db.knowledge_ingest_document_dual_realm(
                    document_path=kwargs.get("document_path"),
                    session_id=kwargs.get("session_id", ""),
                    processing_options=kwargs.get("processing_options", {})
                )
                
            elif action == "discover":
                # Route to relationship discovery
                return self.db.knowledge_discover_relationships_dual_realm(
                    chunk_ids=kwargs.get("chunk_ids", []),
                    session_id=kwargs.get("session_id", ""),
                    discovery_method=kwargs.get("discovery_method", "semantic")
                )
                
            elif action == "optimize":
                # Route to retrieval optimization
                return self.db.knowledge_optimize_retrieval_dual_realm(
                    target_queries=kwargs.get("target_queries", []),
                    session_id=kwargs.get("session_id", ""),
                    optimization_strategy=kwargs.get("optimization_strategy", "performance")
                )
                
            elif action == "get_related":
                # Route to related content retrieval
                return self.db.knowledge_get_related_dual_realm(
                    chunk_id=kwargs.get("chunk_id"),
                    session_id=kwargs.get("session_id", ""),
                    relation_types=kwargs.get("relation_types", [])
                )
                
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "available_actions": ["ingest", "discover", "optimize", "get_related"]
                }
                
        except Exception as e:
            logger.error(f"Master content_manage error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "action": action
            }
    
    # ========================================
    # 🚀 PROMOTION CLASS - 3 Master Functions
    # ========================================
    
    async def promotion_request(self, chunk_id: str, target_realm: str,
                              justification: str, session_id: str = "",
                              auto_analyze: bool = True) -> Dict[str, Any]:
        """
        Master function for creating and managing promotion requests.
        
        Routes to:
        - create_promotion_request_dual_realm for request creation
        - get_promotion_impact_dual_realm for impact analysis
        
        Args:
            chunk_id: Chunk ID to promote
            target_realm: Target realm for promotion
            justification: Justification for promotion
            session_id: Session ID for tracking
            auto_analyze: Automatically analyze impact
            
        Returns:
            Dict with promotion request and impact analysis
        """
        try:
            logger.info(f"Master promotion_request: chunk_id={chunk_id}, target={target_realm}")
            
            # Create the promotion request
            request_result = self.db.create_promotion_request_dual_realm(
                chunk_id=chunk_id,
                target_realm=target_realm,
                justification=justification,
                session_id=session_id
            )
            
            if not request_result.get("success"):
                return request_result
            
            promotion_id = request_result["promotion_id"]
            
            # Auto-analyze impact if requested
            impact_result = None
            if auto_analyze:
                impact_result = self.db.get_promotion_impact_dual_realm(
                    promotion_id=promotion_id
                )
            
            return {
                "success": True,
                "promotion_id": promotion_id,
                "request_result": request_result,
                "impact_analysis": impact_result,
                "auto_analyzed": auto_analyze
            }
            
        except Exception as e:
            logger.error(f"Master promotion_request error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "chunk_id": chunk_id
            }
    
    async def promotion_review(self, promotion_id: str, action: str,
                             reason: str, session_id: str = "",
                             analyze_before: bool = True) -> Dict[str, Any]:
        """
        Master function for reviewing promotions (approve/reject).
        
        Routes to:
        - get_promotion_impact_dual_realm for impact analysis
        - approve_promotion_request_dual_realm for approval
        - reject_promotion_request_dual_realm for rejection
        
        Args:
            promotion_id: Promotion ID to review
            action: "approve" or "reject"
            reason: Reason for action
            session_id: Session ID for tracking
            analyze_before: Analyze impact before action
            
        Returns:
            Dict with review results and impact analysis
        """
        try:
            logger.info(f"Master promotion_review: id={promotion_id}, action={action}")
            
            result = {
                "success": True,
                "promotion_id": promotion_id,
                "action": action
            }
            
            # Analyze impact before action if requested
            if analyze_before:
                impact_result = self.db.get_promotion_impact_dual_realm(
                    promotion_id=promotion_id
                )
                result["impact_analysis"] = impact_result
            
            # Perform the action
            if action == "approve":
                action_result = self.db.approve_promotion_request_dual_realm(
                    promotion_id=promotion_id,
                    approval_reason=reason,
                    session_id=session_id
                )
            elif action == "reject":
                action_result = self.db.reject_promotion_request_dual_realm(
                    promotion_id=promotion_id,
                    rejection_reason=reason,
                    session_id=session_id
                )
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "available_actions": ["approve", "reject"]
                }
            
            result["action_result"] = action_result
            return result
            
        except Exception as e:
            logger.error(f"Master promotion_review error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "promotion_id": promotion_id
            }
    
    async def promotion_monitor(self, filter_status: str = "",
                              filter_realm: str = "", limit: int = 20,
                              include_summary: bool = True) -> Dict[str, Any]:
        """
        Master function for monitoring promotion queue.
        
        Routes to:
        - get_promotion_requests_dual_realm for request listing
        - get_promotion_queue_summary_dual_realm for queue summary
        
        Args:
            filter_status: Filter by status ("pending", "approved", "rejected")
            filter_realm: Filter by target realm
            limit: Maximum number of requests to return
            include_summary: Include queue summary statistics
            
        Returns:
            Dict with promotion requests and queue summary
        """
        try:
            logger.info(f"Master promotion_monitor: status={filter_status}, realm={filter_realm}")
            
            # Get promotion requests
            requests_result = self.db.get_promotion_requests_dual_realm(
                filter_status=filter_status,
                filter_realm=filter_realm,
                limit=limit
            )
            
            result = {
                "success": True,
                "requests": requests_result,
                "filter_status": filter_status,
                "filter_realm": filter_realm,
                "limit": limit
            }
            
            # Include summary if requested
            if include_summary:
                summary_result = self.db.get_promotion_queue_summary_dual_realm(
                    filter_realm=filter_realm
                )
                result["queue_summary"] = summary_result
            
            return result
            
        except Exception as e:
            logger.error(f"Master promotion_monitor error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "filter_status": filter_status
            }
    
    # ========================================
    # 🔄 SESSION CLASS - 4 Master Functions
    # ========================================
    
    async def session_create(self, session_type: str, created_by: str,
                           description: str = "", metadata: Dict[str, Any] = None,
                           auto_prime: bool = True) -> Dict[str, Any]:
        """
        Master function for creating sessions.
        
        Routes to:
        - session_create_dual_realm for session creation
        - session_create_operational_dual_realm for operational sessions
        - session_prime_context_dual_realm for context priming
        
        Args:
            session_type: Type of session ("processing", "operational", "general")
            created_by: Creator identifier
            description: Session description
            metadata: Session metadata
            auto_prime: Automatically prime context
            
        Returns:
            Dict with session creation results
        """
        try:
            logger.info(f"Master session_create: type={session_type}, created_by={created_by}")
            
            if session_type == "operational":
                # Route to operational session creation
                session_result = self.db.session_create_operational_dual_realm(
                    created_by=created_by,
                    description=description,
                    metadata=metadata or {}
                )
            else:
                # Route to regular session creation
                session_result = self.db.session_create_dual_realm(
                    session_type=session_type,
                    created_by=created_by,
                    description=description,
                    metadata=metadata or {}
                )
            
            if not session_result.get("success"):
                return session_result
            
            session_id = session_result["session_id"]
            
            # Auto-prime context if requested
            if auto_prime:
                prime_result = self.db.session_prime_context_dual_realm(
                    session_id=session_id,
                    context_type="auto"
                )
                session_result["context_primed"] = prime_result
            
            return session_result
            
        except Exception as e:
            logger.error(f"Master session_create error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "session_type": session_type
            }
    
    async def session_manage(self, session_id: str, action: str,
                           **kwargs) -> Dict[str, Any]:
        """
        Master function for session management.
        
        Routes to:
        - session_get_state_dual_realm for state tracking
        - session_track_action_dual_realm for action tracking
        - session_prime_context_dual_realm for context priming
        
        Args:
            session_id: Session ID to manage
            action: Action to perform ("get_state", "track_action", "prime_context")
            **kwargs: Action-specific parameters
            
        Returns:
            Dict with management results
        """
        try:
            logger.info(f"Master session_manage: session_id={session_id}, action={action}")
            
            if action == "get_state":
                return self.db.session_get_state_dual_realm(
                    session_id=session_id
                )
                
            elif action == "track_action":
                return self.db.session_track_action_dual_realm(
                    session_id=session_id,
                    action_type=kwargs.get("action_type", "unknown"),
                    action_details=kwargs.get("action_details", {})
                )
                
            elif action == "prime_context":
                return self.db.session_prime_context_dual_realm(
                    session_id=session_id,
                    context_type=kwargs.get("context_type", "auto")
                )
                
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "available_actions": ["get_state", "track_action", "prime_context"]
                }
                
        except Exception as e:
            logger.error(f"Master session_manage error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    async def session_review(self, session_id: str, include_recap: bool = True,
                           include_pending: bool = True,
                           include_recent: bool = False) -> Dict[str, Any]:
        """
        Master function for session review.
        
        Routes to:
        - session_get_recap_dual_realm for session recap
        - get_pending_changes_dual_realm for pending changes
        - session_list_recent_dual_realm for recent sessions
        
        Args:
            session_id: Session ID to review
            include_recap: Include session recap
            include_pending: Include pending changes
            include_recent: Include recent sessions list
            
        Returns:
            Dict with review results
        """
        try:
            logger.info(f"Master session_review: session_id={session_id}")
            
            result = {
                "success": True,
                "session_id": session_id
            }
            
            # Get session recap
            if include_recap:
                recap_result = self.db.session_get_recap_dual_realm(
                    session_id=session_id
                )
                result["recap"] = recap_result
            
            # Get pending changes
            if include_pending:
                pending_result = self.db.get_pending_changes_dual_realm(
                    session_id=session_id
                )
                result["pending_changes"] = pending_result
            
            # Get recent sessions
            if include_recent:
                recent_result = self.db.session_list_recent_dual_realm(
                    limit=10
                )
                result["recent_sessions"] = recent_result
            
            return result
            
        except Exception as e:
            logger.error(f"Master session_review error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    async def session_commit(self, session_id: str, approved_changes: List[str] = None,
                           close_session: bool = True,
                           completion_status: str = "completed") -> Dict[str, Any]:
        """
        Master function for session commitment and closure.
        
        Routes to:
        - commit_session_changes_dual_realm for change commitment
        - session_close_dual_realm for session closure
        - session_complete_dual_realm for completion
        
        Args:
            session_id: Session ID to commit
            approved_changes: List of approved change IDs
            close_session: Close session after commit
            completion_status: Completion status
            
        Returns:
            Dict with commit results
        """
        try:
            logger.info(f"Master session_commit: session_id={session_id}")
            
            result = {
                "success": True,
                "session_id": session_id
            }
            
            # Commit changes
            if approved_changes:
                commit_result = self.db.commit_session_changes_dual_realm(
                    session_id=session_id,
                    approved_changes=approved_changes
                )
                result["commit_result"] = commit_result
            
            # Close session if requested
            if close_session:
                close_result = self.db.session_close_dual_realm(
                    session_id=session_id
                )
                result["close_result"] = close_result
            
            # Complete session
            complete_result = self.db.session_complete_dual_realm(
                session_id=session_id
            )
            result["complete_result"] = complete_result
            result["completion_status"] = completion_status
            
            return result
            
        except Exception as e:
            logger.error(f"Master session_commit error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    # ========================================
    # 🤖 AI CLASS - 3 Master Functions
    # ========================================
    
    async def ai_enhance(self, chunk_ids: List[str], session_id: str = "",
                        enhancement_type: str = "comprehensive",
                        include_curation: bool = True,
                        include_optimization: bool = True) -> Dict[str, Any]:
        """
        Master AI enhancement function.
        
        Routes to:
        - ai_improve_chunk_quality_dual_realm for quality improvement
        - ai_curate_chunks_dual_realm for curation
        - ai_optimize_performance_dual_realm for optimization
        
        Args:
            chunk_ids: List of chunk IDs to enhance
            session_id: Session ID for tracking
            enhancement_type: Type of enhancement ("quality", "curation", "optimization", "comprehensive")
            include_curation: Include curation workflow
            include_optimization: Include performance optimization
            
        Returns:
            Dict with enhancement results
        """
        try:
            logger.info(f"Master ai_enhance: {len(chunk_ids)} chunks, type={enhancement_type}")
            
            result = {
                "success": True,
                "chunk_ids": chunk_ids,
                "enhancement_type": enhancement_type,
                "session_id": session_id
            }
            
            # Quality improvement
            if enhancement_type in ["quality", "comprehensive"]:
                quality_results = []
                for chunk_id in chunk_ids:
                    quality_result = self.db.ai_improve_chunk_quality_dual_realm(
                        chunk_id=chunk_id,
                        session_id=session_id,
                        improvement_type="comprehensive"
                    )
                    quality_results.append(quality_result)
                result["quality_improvements"] = quality_results
            
            # Curation
            if include_curation and enhancement_type in ["curation", "comprehensive"]:
                curation_result = self.db.ai_curate_chunks_dual_realm(
                    chunk_ids=chunk_ids,
                    session_id=session_id,
                    curation_type="automated"
                )
                result["curation_result"] = curation_result
            
            # Performance optimization
            if include_optimization and enhancement_type in ["optimization", "comprehensive"]:
                optimization_result = self.db.ai_optimize_performance_dual_realm(
                    chunk_ids=chunk_ids,
                    session_id=session_id,
                    optimization_type="retrieval"
                )
                result["optimization_result"] = optimization_result
            
            return result
            
        except Exception as e:
            logger.error(f"Master ai_enhance error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "chunk_ids": chunk_ids
            }
    
    async def ai_learn(self, feedback_data: Dict[str, Any], session_id: str = "",
                      update_strategy: bool = True) -> Dict[str, Any]:
        """
        Master AI learning function.
        
        Routes to:
        - ai_record_user_feedback_dual_realm for feedback recording
        - ai_get_adaptive_strategy_dual_realm for strategy updates
        
        Args:
            feedback_data: User feedback data
            session_id: Session ID for tracking
            update_strategy: Update adaptive strategy based on feedback
            
        Returns:
            Dict with learning results
        """
        try:
            logger.info(f"Master ai_learn: session_id={session_id}")
            
            # Record user feedback
            feedback_result = self.db.ai_record_user_feedback_dual_realm(
                feedback_data=feedback_data,
                session_id=session_id
            )
            
            result = {
                "success": True,
                "feedback_result": feedback_result,
                "session_id": session_id
            }
            
            # Update adaptive strategy if requested
            if update_strategy:
                strategy_result = self.db.ai_get_adaptive_strategy_dual_realm(
                    session_id=session_id,
                    strategy_type="learning"
                )
                result["strategy_update"] = strategy_result
            
            return result
            
        except Exception as e:
            logger.error(f"Master ai_learn error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    async def ai_analyze(self, analysis_type: str, session_id: str = "",
                        target_chunks: List[str] = None,
                        include_insights: bool = True,
                        include_report: bool = True) -> Dict[str, Any]:
        """
        Master AI analysis function.
        
        Routes to:
        - ai_get_performance_insights_dual_realm for performance insights
        - ai_generate_enhancement_report_dual_realm for enhancement reporting
        
        Args:
            analysis_type: Type of analysis ("performance", "enhancement", "comprehensive")
            session_id: Session ID for tracking
            target_chunks: Specific chunks to analyze
            include_insights: Include performance insights
            include_report: Include enhancement report
            
        Returns:
            Dict with analysis results
        """
        try:
            logger.info(f"Master ai_analyze: type={analysis_type}, session_id={session_id}")
            
            result = {
                "success": True,
                "analysis_type": analysis_type,
                "session_id": session_id,
                "target_chunks": target_chunks
            }
            
            # Performance insights
            if include_insights and analysis_type in ["performance", "comprehensive"]:
                insights_result = self.db.ai_get_performance_insights_dual_realm(
                    session_id=session_id,
                    insight_type="comprehensive"
                )
                result["performance_insights"] = insights_result
            
            # Enhancement report
            if include_report and analysis_type in ["enhancement", "comprehensive"]:
                report_result = self.db.ai_generate_enhancement_report_dual_realm(
                    session_id=session_id,
                    report_type="comprehensive"
                )
                result["enhancement_report"] = report_result
            
            return result
            
        except Exception as e:
            logger.error(f"Master ai_analyze error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "analysis_type": analysis_type
            }
    
    # ========================================
    # 📊 ANALYTICS CLASS - 2 Master Functions
    # ========================================
    
    async def analytics_track(self, chunk_id: str, query_context: str = "",
                            track_type: str = "access",
                            metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Master analytics tracking function.
        
        Routes to:
        - track_access_dual_realm for access tracking
        
        Args:
            chunk_id: Chunk ID to track
            query_context: Context for the access
            track_type: Type of tracking ("access", "usage", "performance")
            metadata: Additional tracking metadata
            
        Returns:
            Dict with tracking results
        """
        try:
            logger.info(f"Master analytics_track: chunk_id={chunk_id}, type={track_type}")
            
            # Route to access tracking
            tracking_result = self.db.track_access_dual_realm(
                chunk_id=chunk_id,
                query_context=query_context
            )
            
            return {
                "success": True,
                "chunk_id": chunk_id,
                "track_type": track_type,
                "query_context": query_context,
                "tracking_result": tracking_result,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Master analytics_track error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "chunk_id": chunk_id
            }
    
    async def analytics_insights(self, insight_type: str = "hot_contexts",
                               model_type: str = "sonnet", limit: int = 20,
                               include_metrics: bool = True) -> Dict[str, Any]:
        """
        Master analytics insights function.
        
        Routes to:
        - get_hot_contexts_dual_realm for hot contexts
        - Additional analytics functions as they're implemented
        
        Args:
            insight_type: Type of insights ("hot_contexts", "usage_patterns", "performance")
            model_type: Model type for insights
            limit: Maximum number of insights
            include_metrics: Include detailed metrics
            
        Returns:
            Dict with insights results
        """
        try:
            logger.info(f"Master analytics_insights: type={insight_type}, model={model_type}")
            
            if insight_type == "hot_contexts":
                # Route to hot contexts
                insights_result = self.db.get_hot_contexts_dual_realm(
                    model_type=model_type,
                    limit=limit
                )
                
                return {
                    "success": True,
                    "insight_type": insight_type,
                    "model_type": model_type,
                    "limit": limit,
                    "insights": insights_result,
                    "include_metrics": include_metrics
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown insight type: {insight_type}",
                    "available_types": ["hot_contexts"]
                }
                
        except Exception as e:
            logger.error(f"Master analytics_insights error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "insight_type": insight_type
            }
    
    # ========================================
    # 🏗️ APPROVAL CLASS - 4 Functions (GitHub Issue #26)
    # ========================================
    
    async def approval_get_pending(self, limit: int = 20, realm_filter: str = None) -> Dict[str, Any]:
        """
        Get all pending chunks across the system.
        
        Args:
            limit: Maximum number of chunks to return
            realm_filter: Optional realm filter (e.g., "PROJECT", "GLOBAL")
            
        Returns:
            Dict with pending chunks list and metadata
        """
        try:
            logger.info(f"Master approval_get_pending: limit={limit}, realm_filter={realm_filter}")
            
            result = self.db.get_pending_chunks_dual_realm(
                limit=limit,
                realm_filter=realm_filter
            )
            
            return {
                "success": result["success"],
                "chunks": result.get("chunks", []),
                "count": result.get("count", 0),
                "realm_filter": realm_filter,
                "limit": limit,
                "routed_to": "get_pending_chunks_dual_realm"
            }
            
        except Exception as e:
            logger.error(f"Master approval_get_pending error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "chunks": [],
                "count": 0
            }
    
    async def approval_approve(self, chunk_id: str, approved_by: str, approval_notes: str = None) -> Dict[str, Any]:
        """
        Approve a chunk by updating its approval status.
        
        Args:
            chunk_id: Chunk ID to approve
            approved_by: User performing the approval
            approval_notes: Optional approval notes
            
        Returns:
            Dict with approval results and metadata
        """
        try:
            logger.info(f"Master approval_approve: chunk_id={chunk_id}, approved_by={approved_by}")
            
            result = self.db.approve_chunk_dual_realm(
                chunk_id=chunk_id,
                approved_by=approved_by,
                approval_notes=approval_notes
            )
            
            return {
                "success": result["success"],
                "chunk_id": chunk_id,
                "approval_status": result.get("approval_status"),
                "approved_by": approved_by,
                "approved_at": result.get("approved_at"),
                "message": result.get("message"),
                "routed_to": "approve_chunk_dual_realm"
            }
            
        except Exception as e:
            logger.error(f"Master approval_approve error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "chunk_id": chunk_id
            }
    
    async def approval_reject(self, chunk_id: str, rejected_by: str, rejection_reason: str) -> Dict[str, Any]:
        """
        Reject a chunk by updating its approval status.
        
        Args:
            chunk_id: Chunk ID to reject
            rejected_by: User performing the rejection
            rejection_reason: Reason for rejection
            
        Returns:
            Dict with rejection results and metadata
        """
        try:
            logger.info(f"Master approval_reject: chunk_id={chunk_id}, rejected_by={rejected_by}")
            
            result = self.db.reject_chunk_dual_realm(
                chunk_id=chunk_id,
                rejected_by=rejected_by,
                rejection_reason=rejection_reason
            )
            
            return {
                "success": result["success"],
                "chunk_id": chunk_id,
                "approval_status": result.get("approval_status"),
                "rejected_by": rejected_by,
                "rejection_reason": rejection_reason,
                "rejected_at": result.get("rejected_at"),
                "routed_to": "reject_chunk_dual_realm"
            }
            
        except Exception as e:
            logger.error(f"Master approval_reject error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "chunk_id": chunk_id
            }
    
    async def approval_bulk_approve(self, chunk_ids: List[str], approved_by: str) -> Dict[str, Any]:
        """
        Approve multiple chunks in bulk.
        
        Args:
            chunk_ids: List of chunk IDs to approve
            approved_by: User performing the bulk approval
            
        Returns:
            Dict with bulk approval results and metadata
        """
        try:
            logger.info(f"Master approval_bulk_approve: {len(chunk_ids)} chunks, approved_by={approved_by}")
            
            result = self.db.bulk_approve_chunks_dual_realm(
                chunk_ids=chunk_ids,
                approved_by=approved_by
            )
            
            return {
                "success": result["success"],
                "approved_count": result.get("approved_count", 0),
                "failed_count": result.get("failed_count", 0),
                "approved_chunks": result.get("approved_chunks", []),
                "failed_chunks": result.get("failed_chunks", []),
                "approved_by": approved_by,
                "approved_at": result.get("approved_at"),
                "routed_to": "bulk_approve_chunks_dual_realm"
            }
            
        except Exception as e:
            logger.error(f"Master approval_bulk_approve error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "approved_count": 0,
                "failed_count": len(chunk_ids)
            }