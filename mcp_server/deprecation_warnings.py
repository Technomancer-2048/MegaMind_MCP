"""
Deprecation Warnings Module for MegaMind MCP Functions
Phase 2: Function Consolidation Cleanup Plan

This module provides deprecation warnings and function routing
for the original 20 MCP function names to new consolidated functions.
"""

import logging
import warnings
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import functools

try:
    from .removal_scheduler import RemovalScheduler
except ImportError:
    from removal_scheduler import RemovalScheduler

logger = logging.getLogger(__name__)

class DeprecationWarning(UserWarning):
    """Custom deprecation warning for MegaMind MCP functions"""
    pass

def deprecated_function(old_name: str, new_name: str, removal_version: str = "v2.0"):
    """
    Decorator to mark functions as deprecated and log usage.
    
    Args:
        old_name: Name of the deprecated function
        new_name: Name of the replacement function
        removal_version: Version when function will be removed
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if function has been removed
            if hasattr(self, 'removal_scheduler') and self.removal_scheduler.is_function_removed(old_name.replace("mcp__megamind__", "")):
                error_message = f"Function '{old_name}' has been removed. Use '{new_name}' instead."
                logger.error(f"REMOVED FUNCTION CALL: {error_message}")
                raise RuntimeError(error_message)
            
            # Get progressive warning message
            if hasattr(self, 'removal_scheduler'):
                warning_message = self.removal_scheduler.get_removal_alert_message(old_name.replace("mcp__megamind__", ""))
            else:
                warning_message = (
                    f"Function '{old_name}' is deprecated and will be removed in {removal_version}. "
                    f"Use '{new_name}' instead."
                )
            
            logger.warning(f"DEPRECATION: {warning_message}")
            warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
            
            # Call the actual function
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

class DeprecatedFunctionRouter:
    """
    Routes deprecated function calls to new consolidated functions
    while maintaining backward compatibility.
    """
    
    def __init__(self, db_manager, consolidated_functions):
        self.db_manager = db_manager
        self.consolidated_functions = consolidated_functions
        self.usage_stats = {}
        self.removal_scheduler = RemovalScheduler()
        
        # Initialize removal schedule on first run
        if not self.removal_scheduler.schedule.get("deployment_date"):
            self.removal_scheduler.initialize_removal_schedule()
    
    def track_usage(self, function_name: str):
        """Track usage of deprecated functions for monitoring"""
        if function_name not in self.usage_stats:
            self.usage_stats[function_name] = {
                'count': 0,
                'first_used': datetime.now(),
                'last_used': None
            }
        
        self.usage_stats[function_name]['count'] += 1
        self.usage_stats[function_name]['last_used'] = datetime.now()
        
        # Also track in removal scheduler
        self.removal_scheduler.record_usage(function_name)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get deprecation usage statistics"""
        return self.usage_stats
    
    # ========================================
    # 🔍 SEARCH FUNCTION DEPRECATION ROUTING
    # ========================================
    
    @deprecated_function("mcp__megamind__search_chunks", "mcp__megamind__search_query")
    def search_chunks(self, query: str, limit: int = 10, search_type: str = "hybrid") -> Dict[str, Any]:
        """DEPRECATED: Use search_query instead"""
        self.track_usage("search_chunks")
        return self.consolidated_functions.search_query(
            query=query,
            limit=limit,
            search_type=search_type
        )
    
    @deprecated_function("mcp__megamind__get_chunk", "mcp__megamind__search_retrieve")
    def get_chunk(self, chunk_id: str, include_relationships: bool = True) -> Dict[str, Any]:
        """DEPRECATED: Use search_retrieve instead"""
        self.track_usage("get_chunk")
        return self.consolidated_functions.search_retrieve(
            chunk_id=chunk_id,
            include_relationships=include_relationships
        )
    
    @deprecated_function("mcp__megamind__get_related_chunks", "mcp__megamind__search_related")
    def get_related_chunks(self, chunk_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """DEPRECATED: Use search_related instead"""
        self.track_usage("get_related_chunks")
        return self.consolidated_functions.search_related(
            chunk_id=chunk_id,
            max_depth=max_depth
        )
    
    @deprecated_function("mcp__megamind__search_chunks_semantic", "mcp__megamind__search_query")
    def search_chunks_semantic(self, query: str, limit: int = 10, threshold: float = 0.7) -> Dict[str, Any]:
        """DEPRECATED: Use search_query with search_type='semantic' instead"""
        self.track_usage("search_chunks_semantic")
        return self.consolidated_functions.search_query(
            query=query,
            limit=limit,
            search_type="semantic",
            threshold=threshold
        )
    
    @deprecated_function("mcp__megamind__search_chunks_by_similarity", "mcp__megamind__search_query")
    def search_chunks_by_similarity(self, reference_chunk_id: str, limit: int = 10, threshold: float = 0.7) -> Dict[str, Any]:
        """DEPRECATED: Use search_query with search_type='similarity' instead"""
        self.track_usage("search_chunks_by_similarity")
        return self.consolidated_functions.search_query(
            query="",
            limit=limit,
            search_type="similarity",
            threshold=threshold,
            reference_chunk_id=reference_chunk_id
        )
    
    # ========================================
    # 📝 CONTENT FUNCTION DEPRECATION ROUTING
    # ========================================
    
    @deprecated_function("mcp__megamind__create_chunk", "mcp__megamind__content_create")
    def create_chunk(self, content: str, source_document: str, section_path: str = "", 
                    session_id: str = "", target_realm: str = "PROJECT") -> Dict[str, Any]:
        """DEPRECATED: Use content_create instead"""
        self.track_usage("create_chunk")
        return self.consolidated_functions.content_create(
            content=content,
            source_document=source_document,
            session_id=session_id,
            target_realm=target_realm
        )
    
    @deprecated_function("mcp__megamind__update_chunk", "mcp__megamind__content_update")
    def update_chunk(self, chunk_id: str, new_content: str, session_id: str = "") -> Dict[str, Any]:
        """DEPRECATED: Use content_update instead"""
        self.track_usage("update_chunk")
        return self.consolidated_functions.content_update(
            chunk_id=chunk_id,
            new_content=new_content,
            session_id=session_id
        )
    
    @deprecated_function("mcp__megamind__add_relationship", "mcp__megamind__content_process")
    def add_relationship(self, chunk_id_1: str, chunk_id_2: str, relationship_type: str = "related", 
                        session_id: str = "") -> Dict[str, Any]:
        """DEPRECATED: Use content_process for relationship management instead"""
        self.track_usage("add_relationship")
        return self.consolidated_functions.content_process(
            action="add_relationship",
            chunk_id_1=chunk_id_1,
            chunk_id_2=chunk_id_2,
            relationship_type=relationship_type,
            session_id=session_id
        )
    
    @deprecated_function("mcp__megamind__batch_generate_embeddings", "mcp__megamind__content_manage")
    def batch_generate_embeddings(self, chunk_ids: List[str] = None, realm_id: str = "") -> Dict[str, Any]:
        """DEPRECATED: Use content_manage for embedding operations instead"""
        self.track_usage("batch_generate_embeddings")
        return self.consolidated_functions.content_manage(
            action="batch_generate_embeddings",
            chunk_ids=chunk_ids or [],
            realm_id=realm_id
        )
    
    # ========================================
    # 🚀 PROMOTION FUNCTION DEPRECATION ROUTING
    # ========================================
    
    @deprecated_function("mcp__megamind__create_promotion_request", "mcp__megamind__promotion_request")
    def create_promotion_request(self, chunk_id: str, target_realm: str, justification: str, 
                                session_id: str = "") -> Dict[str, Any]:
        """DEPRECATED: Use promotion_request instead"""
        self.track_usage("create_promotion_request")
        return self.consolidated_functions.promotion_request(
            chunk_id=chunk_id,
            target_realm=target_realm,
            justification=justification,
            session_id=session_id
        )
    
    @deprecated_function("mcp__megamind__get_promotion_requests", "mcp__megamind__promotion_monitor")
    def get_promotion_requests(self, filter_status: str = "", filter_realm: str = "", 
                              limit: int = 20) -> Dict[str, Any]:
        """DEPRECATED: Use promotion_monitor instead"""
        self.track_usage("get_promotion_requests")
        return self.consolidated_functions.promotion_monitor(
            filter_status=filter_status,
            filter_realm=filter_realm,
            limit=limit
        )
    
    @deprecated_function("mcp__megamind__approve_promotion_request", "mcp__megamind__promotion_review")
    def approve_promotion_request(self, promotion_id: str, approval_reason: str, 
                                 session_id: str = "") -> Dict[str, Any]:
        """DEPRECATED: Use promotion_review with action='approve' instead"""
        self.track_usage("approve_promotion_request")
        return self.consolidated_functions.promotion_review(
            promotion_id=promotion_id,
            action="approve",
            reason=approval_reason,
            session_id=session_id
        )
    
    @deprecated_function("mcp__megamind__reject_promotion_request", "mcp__megamind__promotion_review")
    def reject_promotion_request(self, promotion_id: str, rejection_reason: str, 
                                session_id: str = "") -> Dict[str, Any]:
        """DEPRECATED: Use promotion_review with action='reject' instead"""
        self.track_usage("reject_promotion_request")
        return self.consolidated_functions.promotion_review(
            promotion_id=promotion_id,
            action="reject",
            reason=rejection_reason,
            session_id=session_id
        )
    
    @deprecated_function("mcp__megamind__get_promotion_impact", "mcp__megamind__promotion_review")
    def get_promotion_impact(self, promotion_id: str) -> Dict[str, Any]:
        """DEPRECATED: Use promotion_review with analyze_before=True instead"""
        self.track_usage("get_promotion_impact")
        return self.consolidated_functions.promotion_review(
            promotion_id=promotion_id,
            action="analyze",
            analyze_before=True
        )
    
    @deprecated_function("mcp__megamind__get_promotion_queue_summary", "mcp__megamind__promotion_monitor")
    def get_promotion_queue_summary(self, filter_realm: str = "") -> Dict[str, Any]:
        """DEPRECATED: Use promotion_monitor with include_summary=True instead"""
        self.track_usage("get_promotion_queue_summary")
        return self.consolidated_functions.promotion_monitor(
            filter_realm=filter_realm,
            include_summary=True
        )
    
    # ========================================
    # 📊 SESSION FUNCTION DEPRECATION ROUTING
    # ========================================
    
    @deprecated_function("mcp__megamind__get_session_primer", "mcp__megamind__session_create")
    def get_session_primer(self, last_session_data: str = "") -> Dict[str, Any]:
        """DEPRECATED: Use session_create for session initialization instead"""
        self.track_usage("get_session_primer")
        return self.consolidated_functions.session_create(
            session_type="operational",
            created_by="claude-code",
            description="Session primer migration",
            auto_prime=True
        )
    
    @deprecated_function("mcp__megamind__get_pending_changes", "mcp__megamind__session_manage")
    def get_pending_changes(self, session_id: str) -> Dict[str, Any]:
        """DEPRECATED: Use session_manage with action='get_pending' instead"""
        self.track_usage("get_pending_changes")
        return self.consolidated_functions.session_manage(
            session_id=session_id,
            action="get_pending"
        )
    
    @deprecated_function("mcp__megamind__commit_session_changes", "mcp__megamind__session_commit")
    def commit_session_changes(self, session_id: str, approved_changes: List[str]) -> Dict[str, Any]:
        """DEPRECATED: Use session_commit instead"""
        self.track_usage("commit_session_changes")
        return self.consolidated_functions.session_commit(
            session_id=session_id,
            approved_changes=approved_changes
        )
    
    # ========================================
    # 📈 ANALYTICS FUNCTION DEPRECATION ROUTING
    # ========================================
    
    @deprecated_function("mcp__megamind__track_access", "mcp__megamind__analytics_track")
    def track_access(self, chunk_id: str, query_context: str = "") -> Dict[str, Any]:
        """DEPRECATED: Use analytics_track instead"""
        self.track_usage("track_access")
        return self.consolidated_functions.analytics_track(
            chunk_id=chunk_id,
            track_type="access",
            metadata={"query_context": query_context}
        )
    
    @deprecated_function("mcp__megamind__get_hot_contexts", "mcp__megamind__analytics_insights")
    def get_hot_contexts(self, model_type: str = "sonnet", limit: int = 20) -> Dict[str, Any]:
        """DEPRECATED: Use analytics_insights instead"""
        self.track_usage("get_hot_contexts")
        return self.consolidated_functions.analytics_insights(
            insight_type="hot_contexts",
            model_type=model_type,
            limit=limit
        )