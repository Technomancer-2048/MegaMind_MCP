#!/usr/bin/env python3
"""
Session-Aware MCP Integration for Enhanced Session System
Extends existing MCP server with session management and awareness
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

# Import session management components
from session_manager import SessionManager, SessionEntry, EntryType, SessionPriority, SessionState
from content_processor import ContentProcessor, EmbeddingType, TruncationStrategy
from session_embedding_service import SessionEmbeddingService

logger = logging.getLogger(__name__)

class SessionAwareMCPExtension:
    """
    Extension for the MCP server to provide session-aware functionality
    """
    
    def __init__(self, db_manager, session_manager: SessionManager, embedding_service):
        self.db_manager = db_manager
        self.session_manager = session_manager
        self.embedding_service = embedding_service
        
        # Initialize content processing pipeline
        self.content_processor = ContentProcessor(
            embedding_service=embedding_service,
            session_manager=session_manager
        )
        
        # Initialize session embedding service
        self.session_embedding_service = SessionEmbeddingService(
            session_manager=session_manager,
            database_connection=db_manager
        )
        
        logger.info("SessionAwareMCPExtension initialized with embedding integration")
    
    def get_session_tools_list(self) -> List[Dict[str, Any]]:
        """Get the list of session management MCP tools"""
        return [
            {
                "name": "mcp__megamind__session_create",
                "description": "Create a new session with automatic state management",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User identifier"},
                        "realm_id": {"type": "string", "default": "PROJECT", "description": "Realm identifier"},
                        "session_name": {"type": "string", "description": "Optional session name"},
                        "project_context": {"type": "string", "description": "Project context description"},
                        "priority": {"type": "string", "default": "medium", "description": "Session priority (low, medium, high, critical)"}
                    },
                    "required": ["user_id"]
                }
            },
            {
                "name": "mcp__megamind__session_activate",
                "description": "Activate a session and make it the current working session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session identifier"},
                        "user_id": {"type": "string", "description": "User identifier"}
                    },
                    "required": ["session_id", "user_id"]
                }
            },
            {
                "name": "mcp__megamind__session_archive",
                "description": "Archive a session and preserve its context",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session identifier"},
                        "user_id": {"type": "string", "description": "User identifier"},
                        "archive_reason": {"type": "string", "default": "User requested", "description": "Reason for archiving"}
                    },
                    "required": ["session_id", "user_id"]
                }
            },
            {
                "name": "mcp__megamind__session_get_active",
                "description": "Get the currently active session for a user",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User identifier"},
                        "realm_id": {"type": "string", "default": "PROJECT", "description": "Realm identifier"}
                    },
                    "required": ["user_id"]
                }
            },
            {
                "name": "mcp__megamind__session_add_entry",
                "description": "Add an entry to a session with automatic embedding generation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session identifier"},
                        "entry_type": {"type": "string", "description": "Entry type (query, operation, result, context_switch, error, system_event)"},
                        "content": {"type": "string", "description": "Entry content"},
                        "operation_type": {"type": "string", "description": "Optional operation type"},
                        "related_chunk_ids": {"type": "array", "items": {"type": "string"}, "description": "Related chunk IDs"},
                        "auto_embed": {"type": "boolean", "default": True, "description": "Automatically generate embedding"}
                    },
                    "required": ["session_id", "entry_type", "content"]
                }
            },
            {
                "name": "mcp__megamind__session_search_semantic",
                "description": "Search session entries using semantic similarity",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "session_id": {"type": "string", "description": "Optional session ID filter"},
                        "embedding_types": {"type": "array", "items": {"type": "string"}, "description": "Embedding types to search"},
                        "limit": {"type": "integer", "default": 10, "description": "Maximum results"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "mcp__megamind__session_get_summary",
                "description": "Get comprehensive session summary with embeddings",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session identifier"},
                        "include_embeddings": {"type": "boolean", "default": True, "description": "Include embedding summary"}
                    },
                    "required": ["session_id"]
                }
            }
        ]
    
    # ================================================================
    # SESSION MANAGEMENT MCP TOOL HANDLERS
    # ================================================================
    
    def handle_session_create(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle session creation with automatic conflict resolution"""
        try:
            user_id = args.get('user_id')
            realm_id = args.get('realm_id', 'PROJECT')
            session_name = args.get('session_name')
            project_context = args.get('project_context')
            priority_str = args.get('priority', 'medium')
            
            # Convert priority string to enum
            priority = SessionPriority(priority_str.lower())
            
            # Create session (will automatically handle active session conflicts)
            session = self.session_manager.create_session(
                user_id=user_id,
                realm_id=realm_id,
                session_name=session_name,
                project_context=project_context,
                priority=priority
            )
            
            # Add creation entry
            self.session_manager.add_session_entry(
                session_id=session.session_id,
                entry_type=EntryType.SYSTEM_EVENT,
                content=f"Session created by {user_id} in realm {realm_id}",
                operation_type="create_session"
            )
            
            return {
                "success": True,
                "session_id": session.session_id,
                "session_name": session.session_name,
                "realm_id": session.realm_id,
                "session_state": session.session_state.value,
                "priority": session.priority.value,
                "created_at": session.created_at.isoformat() if session.created_at else None
            }
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_session_activate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle session activation with state management"""
        try:
            session_id = args.get('session_id')
            user_id = args.get('user_id')
            
            # Activate session (will automatically deactivate others)
            session = self.session_manager.activate_session(session_id, user_id)
            
            return {
                "success": True,
                "session_id": session.session_id,
                "session_name": session.session_name,
                "session_state": session.session_state.value,
                "last_activity": session.last_activity.isoformat() if session.last_activity else None
            }
            
        except Exception as e:
            logger.error(f"Failed to activate session: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_session_archive(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle session archival"""
        try:
            session_id = args.get('session_id')
            user_id = args.get('user_id')
            archive_reason = args.get('archive_reason', 'User requested')
            
            # Archive session
            success = self.session_manager.archive_session(session_id, user_id, archive_reason)
            
            return {
                "success": success,
                "session_id": session_id,
                "archive_reason": archive_reason,
                "archived_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to archive session: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_session_get_active(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get active session for user"""
        try:
            user_id = args.get('user_id')
            realm_id = args.get('realm_id', 'PROJECT')
            
            # Get active session
            session = self.session_manager.get_active_session(user_id, realm_id)
            
            if session:
                return {
                    "success": True,
                    "has_active_session": True,
                    "session_id": session.session_id,
                    "session_name": session.session_name,
                    "session_state": session.session_state.value,
                    "priority": session.priority.value,
                    "last_activity": session.last_activity.isoformat() if session.last_activity else None,
                    "total_entries": session.total_entries
                }
            else:
                return {
                    "success": True,
                    "has_active_session": False,
                    "message": f"No active session found for user {user_id} in realm {realm_id}"
                }
                
        except Exception as e:
            logger.error(f"Failed to get active session: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_session_add_entry(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Add entry to session with optional embedding generation"""
        try:
            session_id = args.get('session_id')
            entry_type_str = args.get('entry_type')
            content = args.get('content')
            operation_type = args.get('operation_type')
            related_chunk_ids = args.get('related_chunk_ids')
            auto_embed = args.get('auto_embed', True)
            
            # Convert entry type string to enum
            entry_type = EntryType(entry_type_str.lower())
            
            # Add entry to session
            entry = self.session_manager.add_session_entry(
                session_id=session_id,
                entry_type=entry_type,
                content=content,
                operation_type=operation_type,
                related_chunk_ids=related_chunk_ids
            )
            
            # Generate embedding if requested and available
            embedding_id = None
            if auto_embed and self.embedding_service:
                try:
                    embedding_id = self.session_embedding_service.process_and_embed_session_entry(
                        session_id=session_id,
                        entry=entry
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for entry: {e}")
            
            return {
                "success": True,
                "entry_id": entry.entry_id,
                "session_id": session_id,
                "entry_type": entry.entry_type.value,
                "content_tokens": entry.content_tokens,
                "content_truncated": entry.content_truncated,
                "entry_sequence": entry.entry_sequence,
                "embedding_id": embedding_id,
                "created_at": entry.created_at.isoformat() if entry.created_at else None
            }
            
        except Exception as e:
            logger.error(f"Failed to add session entry: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_session_search_semantic(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search session entries using semantic similarity"""
        try:
            query = args.get('query')
            session_id = args.get('session_id')
            embedding_types_str = args.get('embedding_types', [])
            limit = args.get('limit', 10)
            
            # Convert embedding type strings to enums
            embedding_types = []
            for et_str in embedding_types_str:
                try:
                    embedding_types.append(EmbeddingType(et_str))
                except ValueError:
                    logger.warning(f"Unknown embedding type: {et_str}")
            
            # Perform semantic search
            results = self.session_embedding_service.search_session_embeddings(
                query=query,
                session_id=session_id,
                embedding_types=embedding_types if embedding_types else None,
                limit=limit
            )
            
            return {
                "success": True,
                "query": query,
                "total_results": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Failed to search session embeddings: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_session_get_summary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        try:
            session_id = args.get('session_id')
            include_embeddings = args.get('include_embeddings', True)
            
            # Get session metadata
            session = self.session_manager.get_session(session_id)
            if not session:
                return {"success": False, "error": f"Session {session_id} not found"}
            
            # Get session entries
            entries = self.session_manager.get_session_entries(session_id, limit=1000)
            
            # Get embedding summary if requested
            embedding_summary = None
            if include_embeddings:
                embedding_summary = self.session_embedding_service.get_session_embedding_summary(session_id)
            
            # Generate content summary
            content_summary = None
            if self.content_processor:
                try:
                    content_summary = self.content_processor.create_session_summary_embedding(session_id)
                except Exception as e:
                    logger.warning(f"Failed to generate content summary: {e}")
            
            return {
                "success": True,
                "session_id": session_id,
                "session_metadata": {
                    "session_name": session.session_name,
                    "user_id": session.user_id,
                    "realm_id": session.realm_id,
                    "session_state": session.session_state.value,
                    "priority": session.priority.value,
                    "total_entries": session.total_entries,
                    "total_operations": session.total_operations,
                    "performance_score": session.performance_score,
                    "created_at": session.created_at.isoformat() if session.created_at else None,
                    "last_activity": session.last_activity.isoformat() if session.last_activity else None
                },
                "entry_statistics": {
                    "total_entries": len(entries),
                    "entry_types": {},
                    "operation_types": {}
                },
                "embedding_summary": embedding_summary,
                "content_summary": {
                    "summary_content": content_summary.summary_content if content_summary else None,
                    "key_operations": content_summary.key_operations if content_summary else [],
                    "major_decisions": content_summary.major_decisions if content_summary else [],
                    "outcomes_discoveries": content_summary.outcomes_discoveries if content_summary else [],
                    "quality_score": content_summary.quality_score if content_summary else 0.0
                } if content_summary else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get session summary: {e}")
            return {"success": False, "error": str(e)}
    
    # ================================================================
    # SESSION-AWARE CHUNK OPERATIONS
    # ================================================================
    
    def handle_session_aware_search_chunks(self, args: Dict[str, Any], session_context: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced search chunks with session context awareness"""
        try:
            query = args.get('query', '')
            limit = args.get('limit', 10)
            search_type = args.get('search_type', 'hybrid')
            session_id = args.get('session_id')
            
            # Get session context if session_id provided
            if session_id and not session_context:
                session = self.session_manager.get_session(session_id)
                if session:
                    session_context = f"Session: {session.session_name}, Context: {session.project_context or 'None'}"
            
            # Perform dual-realm search with session context
            results = self.db_manager.search_chunks_dual_realm(
                query=query,
                limit=limit,
                search_type=search_type
            )
            
            # Add session entry if session provided
            if session_id:
                try:
                    self.session_manager.add_session_entry(
                        session_id=session_id,
                        entry_type=EntryType.QUERY,
                        content=f"Search query: {query}",
                        operation_type="search_chunks",
                        related_chunk_ids=[r.get('chunk_id') for r in results if r.get('chunk_id')]
                    )
                except Exception as e:
                    logger.warning(f"Failed to add session entry for search: {e}")
            
            return {
                "success": True,
                "query": query,
                "search_type": search_type,
                "session_context": session_context,
                "total_results": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Failed to perform session-aware search: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_session_aware_create_chunk(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced create chunk with session integration"""
        try:
            content = args.get('content', '')
            source_document = args.get('source_document', '')
            section_path = args.get('section_path', '')
            session_id = args.get('session_id', '')
            target_realm = args.get('target_realm', 'PROJECT')
            
            # Use existing create_chunk with session awareness
            result = self.db_manager.create_chunk(
                content=content,
                source_document=source_document,
                section_path=section_path,
                session_id=session_id,
                target_realm=target_realm
            )
            
            # Add detailed session entry
            if session_id:
                try:
                    self.session_manager.add_session_entry(
                        session_id=session_id,
                        entry_type=EntryType.OPERATION,
                        content=f"Created new chunk in {target_realm}: {content[:100]}...",
                        operation_type="create_chunk",
                        related_chunk_ids=[result.get('chunk_id')] if result.get('chunk_id') else []
                    )
                except Exception as e:
                    logger.warning(f"Failed to add session entry for chunk creation: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create chunk with session awareness: {e}")
            return {"success": False, "error": str(e)}