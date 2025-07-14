#!/usr/bin/env python3
"""
Advanced Session MCP Functions - Phase 5 Implementation
Implements 6 core + 4 semantic functions for advanced session management
"""

import json
import logging
import uuid
import csv
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict

# Import session management components
from session_manager import SessionManager, SessionEntry, EntryType, SessionPriority, SessionState, SessionMetadata
from content_processor import ContentProcessor, EmbeddingType, TruncationStrategy
from session_embedding_service import SessionEmbeddingService

logger = logging.getLogger(__name__)

class AdvancedSessionMCPFunctions:
    """
    Advanced Session MCP Functions for Phase 5
    Implements 6 core + 4 semantic functions for sophisticated session management
    """
    
    def __init__(self, session_manager: SessionManager, session_extension, db_manager):
        self.session_manager = session_manager
        self.session_extension = session_extension
        self.db_manager = db_manager
        
        # Get services from session extension
        self.content_processor = getattr(session_extension, 'content_processor', None)
        self.session_embedding_service = getattr(session_extension, 'session_embedding_service', None)
        
        logger.info("AdvancedSessionMCPFunctions initialized with Phase 5 capabilities")
    
    def get_advanced_tools_list(self) -> List[Dict[str, Any]]:
        """Get the list of advanced session MCP tools (Phase 5)"""
        return [
            # ================================================================
            # CORE SESSION FUNCTIONS (6)
            # ================================================================
            {
                "name": "mcp__megamind__session_list_user_sessions",
                "description": "List sessions for a user with advanced filtering and pagination",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User identifier"},
                        "realm_id": {"type": "string", "default": "PROJECT", "description": "Realm identifier"},
                        "state_filter": {"type": "string", "description": "Filter by session state (open, active, archived)"},
                        "priority_filter": {"type": "string", "description": "Filter by priority (low, medium, high, critical)"},
                        "date_from": {"type": "string", "description": "Filter sessions from date (ISO format)"},
                        "date_to": {"type": "string", "description": "Filter sessions to date (ISO format)"},
                        "limit": {"type": "integer", "default": 50, "description": "Maximum results"},
                        "offset": {"type": "integer", "default": 0, "description": "Results offset for pagination"}
                    },
                    "required": ["user_id"]
                }
            },
            {
                "name": "mcp__megamind__session_bulk_archive",
                "description": "Archive multiple sessions with bulk operation support",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User identifier"},
                        "session_ids": {"type": "array", "items": {"type": "string"}, "description": "List of session IDs to archive"},
                        "archive_reason": {"type": "string", "default": "Bulk archive operation", "description": "Reason for archiving"},
                        "older_than_days": {"type": "integer", "description": "Archive sessions older than N days"},
                        "state_filter": {"type": "string", "description": "Only archive sessions in specific state"}
                    },
                    "required": ["user_id"]
                }
            },
            {
                "name": "mcp__megamind__session_get_entries_filtered",
                "description": "Get session entries with advanced filtering and search",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session identifier"},
                        "entry_types": {"type": "array", "items": {"type": "string"}, "description": "Filter by entry types"},
                        "operation_types": {"type": "array", "items": {"type": "string"}, "description": "Filter by operation types"},
                        "content_search": {"type": "string", "description": "Search in entry content"},
                        "quality_threshold": {"type": "number", "description": "Minimum quality score"},
                        "date_from": {"type": "string", "description": "Filter entries from date"},
                        "date_to": {"type": "string", "description": "Filter entries to date"},
                        "limit": {"type": "integer", "default": 100, "description": "Maximum results"},
                        "offset": {"type": "integer", "default": 0, "description": "Results offset"}
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "mcp__megamind__session_analytics_dashboard",
                "description": "Generate comprehensive session analytics and usage patterns",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User identifier"},
                        "realm_id": {"type": "string", "default": "PROJECT", "description": "Realm identifier"},
                        "time_period": {"type": "string", "default": "30d", "description": "Analysis period (7d, 30d, 90d, 1y)"},
                        "include_performance": {"type": "boolean", "default": True, "description": "Include performance metrics"},
                        "include_patterns": {"type": "boolean", "default": True, "description": "Include usage patterns"},
                        "include_recommendations": {"type": "boolean", "default": True, "description": "Include optimization recommendations"}
                    },
                    "required": ["user_id"]
                }
            },
            {
                "name": "mcp__megamind__session_export",
                "description": "Export session data in various formats with comprehensive options",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_ids": {"type": "array", "items": {"type": "string"}, "description": "Session IDs to export"},
                        "user_id": {"type": "string", "description": "User identifier for bulk export"},
                        "export_format": {"type": "string", "default": "json", "description": "Export format (json, csv, markdown, xml)"},
                        "include_entries": {"type": "boolean", "default": True, "description": "Include session entries"},
                        "include_embeddings": {"type": "boolean", "default": False, "description": "Include embedding data"},
                        "include_metadata": {"type": "boolean", "default": True, "description": "Include session metadata"},
                        "compression": {"type": "string", "description": "Compression format (gzip, zip)"},
                        "date_range": {"type": "object", "description": "Date range filter for export"}
                    },
                    "required": []
                }
            },
            {
                "name": "mcp__megamind__session_relationship_tracking",
                "description": "Track and analyze relationships between sessions and chunks",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Primary session identifier"},
                        "analysis_type": {"type": "string", "default": "comprehensive", "description": "Analysis type (basic, comprehensive, network)"},
                        "include_chunk_relationships": {"type": "boolean", "default": True, "description": "Include chunk relationships"},
                        "include_session_clusters": {"type": "boolean", "default": True, "description": "Include session clustering"},
                        "max_depth": {"type": "integer", "default": 3, "description": "Maximum relationship depth"},
                        "similarity_threshold": {"type": "number", "default": 0.7, "description": "Similarity threshold for relationships"}
                    },
                    "required": ["session_id"]
                }
            },
            
            # ================================================================
            # SEMANTIC SESSION FUNCTIONS (4)
            # ================================================================
            {
                "name": "mcp__megamind__session_semantic_similarity",
                "description": "Find sessions similar to a reference session using semantic analysis",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "reference_session_id": {"type": "string", "description": "Reference session identifier"},
                        "user_id": {"type": "string", "description": "User identifier for scoping"},
                        "similarity_threshold": {"type": "number", "default": 0.6, "description": "Minimum similarity threshold"},
                        "max_results": {"type": "integer", "default": 10, "description": "Maximum similar sessions"},
                        "include_archived": {"type": "boolean", "default": False, "description": "Include archived sessions"},
                        "analysis_depth": {"type": "string", "default": "content", "description": "Analysis depth (content, summary, full)"}
                    },
                    "required": ["reference_session_id"]
                }
            },
            {
                "name": "mcp__megamind__session_semantic_clustering",
                "description": "Group sessions into semantic clusters for pattern analysis",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User identifier"},
                        "realm_id": {"type": "string", "default": "PROJECT", "description": "Realm identifier"},
                        "clustering_method": {"type": "string", "default": "kmeans", "description": "Clustering method (kmeans, hierarchical, dbscan)"},
                        "num_clusters": {"type": "integer", "default": 5, "description": "Number of clusters (for kmeans)"},
                        "feature_type": {"type": "string", "default": "embeddings", "description": "Feature type (embeddings, content, hybrid)"},
                        "min_sessions": {"type": "integer", "default": 3, "description": "Minimum sessions per cluster"},
                        "time_period": {"type": "string", "description": "Time period for analysis"}
                    },
                    "required": ["user_id"]
                }
            },
            {
                "name": "mcp__megamind__session_semantic_insights",
                "description": "Generate semantic insights and patterns from session content",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_ids": {"type": "array", "items": {"type": "string"}, "description": "Session IDs for analysis"},
                        "user_id": {"type": "string", "description": "User identifier for bulk analysis"},
                        "insight_types": {"type": "array", "items": {"type": "string"}, "default": ["topics", "trends", "anomalies"], "description": "Types of insights to generate"},
                        "topic_modeling": {"type": "boolean", "default": True, "description": "Enable topic modeling"},
                        "trend_analysis": {"type": "boolean", "default": True, "description": "Enable trend analysis"},
                        "anomaly_detection": {"type": "boolean", "default": True, "description": "Enable anomaly detection"},
                        "time_granularity": {"type": "string", "default": "daily", "description": "Time granularity (hourly, daily, weekly)"}
                    },
                    "required": []
                }
            },
            {
                "name": "mcp__megamind__session_semantic_recommendations",
                "description": "Generate intelligent recommendations based on session patterns",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User identifier"},
                        "session_id": {"type": "string", "description": "Current session for context"},
                        "recommendation_types": {"type": "array", "items": {"type": "string"}, "default": ["next_actions", "related_content", "optimization"], "description": "Types of recommendations"},
                        "include_chunk_suggestions": {"type": "boolean", "default": True, "description": "Include chunk recommendations"},
                        "include_workflow_optimization": {"type": "boolean", "default": True, "description": "Include workflow optimization"},
                        "personalization_level": {"type": "string", "default": "medium", "description": "Personalization level (low, medium, high)"},
                        "confidence_threshold": {"type": "number", "default": 0.7, "description": "Minimum confidence for recommendations"}
                    },
                    "required": ["user_id"]
                }
            }
        ]
    
    # ================================================================
    # CORE SESSION FUNCTION HANDLERS (6)
    # ================================================================
    
    def handle_session_list_user_sessions(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List sessions for a user with advanced filtering"""
        try:
            user_id = args.get('user_id')
            realm_id = args.get('realm_id', 'PROJECT')
            state_filter = args.get('state_filter')
            priority_filter = args.get('priority_filter')
            date_from = args.get('date_from')
            date_to = args.get('date_to')
            limit = args.get('limit', 50)
            offset = args.get('offset', 0)
            
            # Convert string filters to enums if provided
            state_enum = None
            if state_filter:
                try:
                    state_enum = SessionState(state_filter.lower())
                except ValueError:
                    logger.warning(f"Invalid state filter: {state_filter}")
            
            priority_enum = None
            if priority_filter:
                try:
                    priority_enum = SessionPriority(priority_filter.lower())
                except ValueError:
                    logger.warning(f"Invalid priority filter: {priority_filter}")
            
            # Get sessions with basic filtering
            sessions = self.session_manager.list_user_sessions(
                user_id=user_id,
                realm_id=realm_id,
                state_filter=state_enum,
                limit=limit + offset  # Get more for offset handling
            )
            
            # Apply additional filtering
            filtered_sessions = []
            for session in sessions:
                # Priority filter
                if priority_enum and session.priority != priority_enum:
                    continue
                
                # Date filters
                if date_from:
                    try:
                        from_date = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
                        if session.created_at and session.created_at < from_date:
                            continue
                    except ValueError:
                        logger.warning(f"Invalid date_from format: {date_from}")
                
                if date_to:
                    try:
                        to_date = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
                        if session.created_at and session.created_at > to_date:
                            continue
                    except ValueError:
                        logger.warning(f"Invalid date_to format: {date_to}")
                
                filtered_sessions.append(session)
            
            # Apply pagination
            paginated_sessions = filtered_sessions[offset:offset + limit]
            
            # Convert to dict format
            session_data = []
            for session in paginated_sessions:
                session_dict = {
                    "session_id": session.session_id,
                    "session_name": session.session_name,
                    "user_id": session.user_id,
                    "realm_id": session.realm_id,
                    "session_state": session.session_state.value,
                    "priority": session.priority.value,
                    "total_entries": session.total_entries,
                    "total_operations": session.total_operations,
                    "performance_score": session.performance_score,
                    "created_at": session.created_at.isoformat() if session.created_at else None,
                    "last_activity": session.last_activity.isoformat() if session.last_activity else None,
                    "archived_at": session.archived_at.isoformat() if session.archived_at else None
                }
                session_data.append(session_dict)
            
            return {
                "success": True,
                "user_id": user_id,
                "realm_id": realm_id,
                "total_sessions": len(filtered_sessions),
                "returned_sessions": len(paginated_sessions),
                "offset": offset,
                "limit": limit,
                "filters_applied": {
                    "state": state_filter,
                    "priority": priority_filter,
                    "date_from": date_from,
                    "date_to": date_to
                },
                "sessions": session_data
            }
            
        except Exception as e:
            logger.error(f"Failed to list user sessions: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_session_bulk_archive(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Archive multiple sessions with bulk operation support"""
        try:
            user_id = args.get('user_id')
            session_ids = args.get('session_ids', [])
            archive_reason = args.get('archive_reason', 'Bulk archive operation')
            older_than_days = args.get('older_than_days')
            state_filter = args.get('state_filter')
            
            archived_sessions = []
            failed_sessions = []
            
            # Handle bulk archive by age
            if older_than_days:
                cutoff_date = datetime.now() - timedelta(days=older_than_days)
                sessions = self.session_manager.list_user_sessions(user_id, limit=1000)
                
                for session in sessions:
                    # Check age criteria
                    if session.last_activity and session.last_activity < cutoff_date:
                        # Check state filter if provided
                        if state_filter:
                            try:
                                state_enum = SessionState(state_filter.lower())
                                if session.session_state != state_enum:
                                    continue
                            except ValueError:
                                logger.warning(f"Invalid state filter: {state_filter}")
                                continue
                        
                        session_ids.append(session.session_id)
            
            # Archive sessions
            for session_id in session_ids:
                try:
                    success = self.session_manager.archive_session(
                        session_id=session_id,
                        user_id=user_id,
                        archive_reason=f"{archive_reason} (bulk operation)"
                    )
                    
                    if success:
                        archived_sessions.append(session_id)
                    else:
                        failed_sessions.append({"session_id": session_id, "error": "Archive operation failed"})
                        
                except Exception as e:
                    failed_sessions.append({"session_id": session_id, "error": str(e)})
            
            return {
                "success": True,
                "user_id": user_id,
                "archive_reason": archive_reason,
                "total_requested": len(session_ids),
                "successfully_archived": len(archived_sessions),
                "failed_operations": len(failed_sessions),
                "archived_sessions": archived_sessions,
                "failed_sessions": failed_sessions,
                "operation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to bulk archive sessions: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_session_get_entries_filtered(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get session entries with advanced filtering"""
        try:
            session_id = args.get('session_id')
            entry_types = args.get('entry_types', [])
            operation_types = args.get('operation_types', [])
            content_search = args.get('content_search')
            quality_threshold = args.get('quality_threshold')
            date_from = args.get('date_from')
            date_to = args.get('date_to')
            limit = args.get('limit', 100)
            offset = args.get('offset', 0)
            
            # Get all entries for the session
            all_entries = self.session_manager.get_session_entries(session_id, limit=10000)
            
            # Apply filters
            filtered_entries = []
            for entry in all_entries:
                # Entry type filter
                if entry_types and entry.entry_type.value not in entry_types:
                    continue
                
                # Operation type filter
                if operation_types and entry.operation_type not in operation_types:
                    continue
                
                # Content search filter
                if content_search and content_search.lower() not in entry.entry_content.lower():
                    continue
                
                # Quality threshold filter
                if quality_threshold and entry.quality_score < quality_threshold:
                    continue
                
                # Date filters
                if date_from:
                    try:
                        from_date = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
                        if entry.created_at and entry.created_at < from_date:
                            continue
                    except ValueError:
                        logger.warning(f"Invalid date_from format: {date_from}")
                
                if date_to:
                    try:
                        to_date = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
                        if entry.created_at and entry.created_at > to_date:
                            continue
                    except ValueError:
                        logger.warning(f"Invalid date_to format: {date_to}")
                
                filtered_entries.append(entry)
            
            # Apply pagination
            paginated_entries = filtered_entries[offset:offset + limit]
            
            # Convert to dict format
            entry_data = []
            for entry in paginated_entries:
                entry_dict = {
                    "entry_id": entry.entry_id,
                    "session_id": entry.session_id,
                    "entry_type": entry.entry_type.value,
                    "operation_type": entry.operation_type,
                    "entry_content": entry.entry_content,
                    "content_tokens": entry.content_tokens,
                    "content_truncated": entry.content_truncated,
                    "entry_sequence": entry.entry_sequence,
                    "conversation_turn": entry.conversation_turn,
                    "success_indicator": entry.success_indicator,
                    "quality_score": entry.quality_score,
                    "context_relevance_score": entry.context_relevance_score,
                    "created_at": entry.created_at.isoformat() if entry.created_at else None
                }
                entry_data.append(entry_dict)
            
            return {
                "success": True,
                "session_id": session_id,
                "total_entries": len(filtered_entries),
                "returned_entries": len(paginated_entries),
                "offset": offset,
                "limit": limit,
                "filters_applied": {
                    "entry_types": entry_types,
                    "operation_types": operation_types,
                    "content_search": content_search,
                    "quality_threshold": quality_threshold,
                    "date_from": date_from,
                    "date_to": date_to
                },
                "entries": entry_data
            }
            
        except Exception as e:
            logger.error(f"Failed to get filtered entries: {e}")
            return {"success": False, "error": str(e)}