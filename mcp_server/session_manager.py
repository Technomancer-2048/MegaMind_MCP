#!/usr/bin/env python3
"""
Enhanced Session Manager for MegaMind Context Database
Implements core session logic with state management, lifecycle operations, and entry tracking
"""

import json
import logging
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

class SessionState(Enum):
    """Session state enumeration"""
    OPEN = "open"
    ACTIVE = "active"  
    ARCHIVED = "archived"

class SessionPriority(Enum):
    """Session priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EntryType(Enum):
    """Session entry types"""
    QUERY = "query"
    OPERATION = "operation"
    RESULT = "result"
    CONTEXT_SWITCH = "context_switch"
    ERROR = "error"
    SYSTEM_EVENT = "system_event"

@dataclass
class SessionMetadata:
    """Session metadata structure"""
    session_id: str
    session_name: Optional[str] = None
    user_id: Optional[str] = None
    realm_id: str = "PROJECT"
    project_context: Optional[str] = None
    session_state: SessionState = SessionState.OPEN
    priority: SessionPriority = SessionPriority.MEDIUM
    enable_semantic_indexing: bool = True
    content_token_limit: int = 128
    session_config: Optional[Dict[str, Any]] = None
    session_tags: Optional[List[str]] = None
    total_entries: int = 0
    total_chunks_accessed: int = 0
    total_operations: int = 0
    performance_score: float = 0.0
    context_quality_score: float = 0.0
    created_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    archived_at: Optional[datetime] = None

@dataclass
class SessionEntry:
    """Session entry structure"""
    entry_id: str
    session_id: str
    entry_type: EntryType
    entry_content: str
    operation_type: Optional[str] = None
    content_tokens: int = 0
    content_truncated: bool = False
    original_content_hash: Optional[str] = None
    semantic_summary: Optional[str] = None
    related_chunk_ids: Optional[List[str]] = None
    context_relevance_score: float = 0.0
    parent_entry_id: Optional[str] = None
    entry_sequence: int = 1
    conversation_turn: int = 1
    processing_time_ms: int = 0
    success_indicator: bool = True
    quality_score: float = 0.0
    entry_metadata: Optional[Dict[str, Any]] = None
    user_feedback: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

class SessionValidationError(Exception):
    """Session validation error"""
    pass

class SessionStateError(Exception):
    """Session state management error"""
    pass

class SessionManager:
    """
    Core session management logic with state tracking, lifecycle operations, and entry management
    """
    
    def __init__(self, database_connection):
        self.db = database_connection
        self._active_session_cache = {}  # Cache for performance
        self._session_locks = {}  # Simple session locking mechanism
        
        # Configuration
        self.max_active_sessions_per_user = 1
        self.session_timeout_hours = 24
        self.max_entries_per_session = 10000
        self.auto_archive_after_days = 30
        
        logger.info("SessionManager initialized with enhanced session logic")
    
    # ================================================================
    # CORE SESSION LIFECYCLE METHODS
    # ================================================================
    
    def create_session(self, 
                      user_id: str,
                      realm_id: str = "PROJECT",
                      session_name: Optional[str] = None,
                      project_context: Optional[str] = None,
                      priority: SessionPriority = SessionPriority.MEDIUM,
                      config: Optional[Dict[str, Any]] = None) -> SessionMetadata:
        """Create a new session with validation and state management"""
        
        try:
            # Generate session ID
            session_id = f"session_{uuid.uuid4().hex[:12]}"
            
            # Validate session creation
            self._validate_session_creation(user_id, realm_id)
            
            # Check if user has active sessions (enforce one active session rule)
            active_sessions = self._get_active_sessions_for_user(user_id, realm_id)
            if active_sessions:
                logger.warning(f"User {user_id} has active sessions. Archiving them before creating new session.")
                for active_session in active_sessions:
                    self._archive_session_internal(active_session['session_id'], "New session created")
            
            # Create session metadata
            session_metadata = SessionMetadata(
                session_id=session_id,
                session_name=session_name or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                user_id=user_id,
                realm_id=realm_id,
                project_context=project_context,
                session_state=SessionState.OPEN,
                priority=priority,
                session_config=config or {},
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            # Insert into database
            self._insert_session_to_database(session_metadata)
            
            # Update cache
            self._active_session_cache[session_id] = session_metadata
            
            logger.info(f"Created new session: {session_id} for user {user_id} in realm {realm_id}")
            return session_metadata
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise SessionValidationError(f"Session creation failed: {str(e)}")
    
    def activate_session(self, session_id: str, user_id: str) -> SessionMetadata:
        """Activate an open session and make it the current working session"""
        
        try:
            # Get session metadata
            session = self._get_session_from_database(session_id)
            if not session:
                raise SessionValidationError(f"Session {session_id} not found")
            
            # Validate session ownership and state
            if session.user_id != user_id:
                raise SessionValidationError(f"Session {session_id} does not belong to user {user_id}")
            
            if session.session_state not in [SessionState.OPEN, SessionState.ACTIVE]:
                raise SessionStateError(f"Cannot activate session in state: {session.session_state}")
            
            # Deactivate other active sessions for this user
            active_sessions = self._get_active_sessions_for_user(user_id, session.realm_id)
            for active_session in active_sessions:
                if active_session['session_id'] != session_id:
                    self._set_session_state(active_session['session_id'], SessionState.OPEN)
            
            # Activate this session
            session.session_state = SessionState.ACTIVE
            session.last_activity = datetime.now()
            
            # Update database
            self._update_session_in_database(session)
            
            # Update cache
            self._active_session_cache[session_id] = session
            
            logger.info(f"Activated session: {session_id} for user {user_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to activate session {session_id}: {e}")
            raise SessionStateError(f"Session activation failed: {str(e)}")
    
    def archive_session(self, session_id: str, user_id: str, archive_reason: str = "User requested") -> bool:
        """Archive a session and preserve its context"""
        
        try:
            # Get session metadata
            session = self._get_session_from_database(session_id)
            if not session:
                raise SessionValidationError(f"Session {session_id} not found")
            
            # Validate session ownership
            if session.user_id != user_id:
                raise SessionValidationError(f"Session {session_id} does not belong to user {user_id}")
            
            if session.session_state == SessionState.ARCHIVED:
                logger.info(f"Session {session_id} is already archived")
                return True
            
            # Archive the session
            return self._archive_session_internal(session_id, archive_reason)
            
        except Exception as e:
            logger.error(f"Failed to archive session {session_id}: {e}")
            raise SessionStateError(f"Session archival failed: {str(e)}")
    
    def resume_session(self, session_id: str, user_id: str) -> SessionMetadata:
        """Resume an archived session (move back to open state)"""
        
        try:
            # Get session metadata
            session = self._get_session_from_database(session_id)
            if not session:
                raise SessionValidationError(f"Session {session_id} not found")
            
            # Validate session ownership
            if session.user_id != user_id:
                raise SessionValidationError(f"Session {session_id} does not belong to user {user_id}")
            
            if session.session_state != SessionState.ARCHIVED:
                raise SessionStateError(f"Can only resume archived sessions. Current state: {session.session_state}")
            
            # Resume session
            session.session_state = SessionState.OPEN
            session.last_activity = datetime.now()
            session.archived_at = None
            
            # Update database
            self._update_session_in_database(session)
            
            # Add resume entry
            self.add_session_entry(
                session_id=session_id,
                entry_type=EntryType.SYSTEM_EVENT,
                content=f"Session resumed by {user_id}",
                operation_type="resume_session"
            )
            
            logger.info(f"Resumed session: {session_id} for user {user_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to resume session {session_id}: {e}")
            raise SessionStateError(f"Session resume failed: {str(e)}")
    
    # ================================================================
    # SESSION ENTRY MANAGEMENT
    # ================================================================
    
    def add_session_entry(self,
                         session_id: str,
                         entry_type: EntryType,
                         content: str,
                         operation_type: Optional[str] = None,
                         related_chunk_ids: Optional[List[str]] = None,
                         parent_entry_id: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> SessionEntry:
        """Add a new entry to a session with automatic state management"""
        
        try:
            # Get session and validate
            session = self._get_session_from_database(session_id)
            if not session:
                raise SessionValidationError(f"Session {session_id} not found")
            
            if session.session_state == SessionState.ARCHIVED:
                raise SessionStateError("Cannot add entries to archived session")
            
            # Generate entry ID and calculate sequence
            entry_id = f"entry_{uuid.uuid4().hex[:12]}"
            entry_sequence = self._get_next_entry_sequence(session_id)
            conversation_turn = self._get_current_conversation_turn(session_id)
            
            # Calculate content tokens and hash
            content_tokens = len(content.split())
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            # Apply token limits if needed
            truncated_content = content
            content_truncated = False
            if content_tokens > session.content_token_limit:
                # Simple truncation strategy - take first N tokens
                words = content.split()
                truncated_content = ' '.join(words[:session.content_token_limit])
                content_truncated = True
                content_tokens = session.content_token_limit
            
            # Create session entry
            entry = SessionEntry(
                entry_id=entry_id,
                session_id=session_id,
                entry_type=entry_type,
                entry_content=truncated_content,
                operation_type=operation_type,
                content_tokens=content_tokens,
                content_truncated=content_truncated,
                original_content_hash=content_hash,
                related_chunk_ids=related_chunk_ids,
                parent_entry_id=parent_entry_id,
                entry_sequence=entry_sequence,
                conversation_turn=conversation_turn,
                entry_metadata=metadata,
                created_at=datetime.now()
            )
            
            # Insert into database
            self._insert_entry_to_database(entry)
            
            # Update session statistics
            self._update_session_stats(session_id, entry_added=True)
            
            # Auto-activate session if it's open and we're adding operational entries
            if session.session_state == SessionState.OPEN and entry_type in [EntryType.QUERY, EntryType.OPERATION]:
                session.session_state = SessionState.ACTIVE
                self._update_session_in_database(session)
            
            logger.debug(f"Added entry {entry_id} to session {session_id}")
            return entry
            
        except Exception as e:
            logger.error(f"Failed to add entry to session {session_id}: {e}")
            raise SessionValidationError(f"Entry addition failed: {str(e)}")
    
    # ================================================================
    # SESSION QUERY AND RETRIEVAL
    # ================================================================
    
    def get_session(self, session_id: str, user_id: Optional[str] = None) -> Optional[SessionMetadata]:
        """Get session metadata with optional user validation"""
        
        try:
            session = self._get_session_from_database(session_id)
            if not session:
                return None
            
            # Validate user ownership if specified
            if user_id and session.user_id != user_id:
                logger.warning(f"User {user_id} attempted to access session {session_id} owned by {session.user_id}")
                return None
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    def get_active_session(self, user_id: str, realm_id: str) -> Optional[SessionMetadata]:
        """Get the currently active session for a user in a realm"""
        
        try:
            active_sessions = self._get_active_sessions_for_user(user_id, realm_id)
            if active_sessions:
                session_id = active_sessions[0]['session_id']
                return self._get_session_from_database(session_id)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get active session for user {user_id}: {e}")
            return None
    
    def get_session_entries(self, session_id: str, limit: int = 100, offset: int = 0) -> List[SessionEntry]:
        """Get session entries with pagination"""
        
        try:
            return self._get_entries_from_database(session_id, limit, offset)
            
        except Exception as e:
            logger.error(f"Failed to get entries for session {session_id}: {e}")
            return []
    
    def list_user_sessions(self, user_id: str, realm_id: str, 
                          state_filter: Optional[SessionState] = None,
                          limit: int = 50) -> List[SessionMetadata]:
        """List sessions for a user with optional state filtering"""
        
        try:
            return self._list_sessions_from_database(user_id, realm_id, state_filter, limit)
            
        except Exception as e:
            logger.error(f"Failed to list sessions for user {user_id}: {e}")
            return []
    
    # ================================================================
    # INTERNAL HELPER METHODS
    # ================================================================
    
    def _validate_session_creation(self, user_id: str, realm_id: str):
        """Validate session creation parameters"""
        if not user_id or not user_id.strip():
            raise SessionValidationError("User ID is required")
        
        if not realm_id or not realm_id.strip():
            raise SessionValidationError("Realm ID is required")
        
        # Additional validation logic can be added here
    
    def _get_active_sessions_for_user(self, user_id: str, realm_id: str) -> List[Dict[str, Any]]:
        """Get active sessions for a user in a realm"""
        try:
            connection = self.db.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            query = """
            SELECT session_id, session_state FROM megamind_sessions 
            WHERE user_id = %s AND realm_id = %s AND session_state = 'active'
            ORDER BY last_activity DESC
            """
            
            cursor.execute(query, (user_id, realm_id))
            result = cursor.fetchall()
            cursor.close()
            connection.close()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get active sessions: {e}")
            return []
    
    def _archive_session_internal(self, session_id: str, reason: str) -> bool:
        """Internal method to archive a session"""
        try:
            # Add archive entry BEFORE archiving (while session is still active)
            try:
                self.add_session_entry(
                    session_id=session_id,
                    entry_type=EntryType.SYSTEM_EVENT,
                    content=f"Session archived: {reason}",
                    operation_type="archive_session"
                )
            except Exception as entry_error:
                logger.warning(f"Could not add archive entry: {entry_error}")
            
            # Now archive the session
            connection = self.db.get_connection()
            cursor = connection.cursor()
            
            archive_query = """
            UPDATE megamind_sessions 
            SET session_state = 'archived', archived_at = CURRENT_TIMESTAMP, last_activity = CURRENT_TIMESTAMP
            WHERE session_id = %s
            """
            
            cursor.execute(archive_query, (session_id,))
            connection.commit()
            cursor.close()
            connection.close()
            
            # Remove from cache
            self._active_session_cache.pop(session_id, None)
            
            logger.info(f"Archived session {session_id}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive session {session_id}: {e}")
            return False
    
    def _set_session_state(self, session_id: str, new_state: SessionState) -> bool:
        """Update session state in database"""
        try:
            connection = self.db.get_connection()
            cursor = connection.cursor()
            
            update_query = """
            UPDATE megamind_sessions 
            SET session_state = %s, last_activity = CURRENT_TIMESTAMP
            WHERE session_id = %s
            """
            
            cursor.execute(update_query, (new_state.value, session_id))
            connection.commit()
            cursor.close()
            connection.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session state: {e}")
            return False
    
    def _get_next_entry_sequence(self, session_id: str) -> int:
        """Get the next entry sequence number for a session"""
        try:
            connection = self.db.get_connection()
            cursor = connection.cursor()
            
            query = "SELECT MAX(entry_sequence) FROM megamind_session_entries WHERE session_id = %s"
            cursor.execute(query, (session_id,))
            result = cursor.fetchone()
            
            cursor.close()
            connection.close()
            
            return (result[0] or 0) + 1
            
        except Exception as e:
            logger.error(f"Failed to get next entry sequence: {e}")
            return 1
    
    def _get_current_conversation_turn(self, session_id: str) -> int:
        """Get current conversation turn for a session"""
        try:
            connection = self.db.get_connection()
            cursor = connection.cursor()
            
            query = "SELECT MAX(conversation_turn) FROM megamind_session_entries WHERE session_id = %s"
            cursor.execute(query, (session_id,))
            result = cursor.fetchone()
            
            cursor.close()
            connection.close()
            
            return result[0] or 1
            
        except Exception as e:
            logger.error(f"Failed to get conversation turn: {e}")
            return 1
    
    def _update_session_stats(self, session_id: str, entry_added: bool = False, chunks_accessed: int = 0):
        """Update session statistics"""
        try:
            connection = self.db.get_connection()
            cursor = connection.cursor()
            
            if entry_added:
                update_query = """
                UPDATE megamind_sessions 
                SET total_entries = total_entries + 1, 
                    total_chunks_accessed = total_chunks_accessed + %s,
                    total_operations = total_operations + 1,
                    last_activity = CURRENT_TIMESTAMP
                WHERE session_id = %s
                """
                cursor.execute(update_query, (chunks_accessed, session_id))
            
            connection.commit()
            cursor.close()
            connection.close()
            
        except Exception as e:
            logger.error(f"Failed to update session stats: {e}")
    
    # ================================================================
    # DATABASE INTERACTION METHODS
    # ================================================================
    
    def _insert_session_to_database(self, session: SessionMetadata):
        """Insert session to database"""
        try:
            connection = self.db.get_connection()
            cursor = connection.cursor()
            
            insert_query = """
            INSERT INTO megamind_sessions (
                session_id, session_name, user_id, realm_id, project_context,
                session_state, priority, enable_semantic_indexing, content_token_limit,
                session_config, session_tags, total_entries, total_chunks_accessed,
                total_operations, performance_score, context_quality_score,
                created_at, last_activity
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            session_config_json = json.dumps(session.session_config) if session.session_config else None
            session_tags_json = json.dumps(session.session_tags) if session.session_tags else None
            
            cursor.execute(insert_query, (
                session.session_id, session.session_name, session.user_id, session.realm_id,
                session.project_context, session.session_state.value, session.priority.value,
                session.enable_semantic_indexing, session.content_token_limit,
                session_config_json, session_tags_json, session.total_entries,
                session.total_chunks_accessed, session.total_operations,
                session.performance_score, session.context_quality_score,
                session.created_at, session.last_activity
            ))
            
            connection.commit()
            cursor.close()
            connection.close()
            
        except Exception as e:
            logger.error(f"Failed to insert session to database: {e}")
            raise
    
    def _update_session_in_database(self, session: SessionMetadata):
        """Update session in database"""
        try:
            connection = self.db.get_connection()
            cursor = connection.cursor()
            
            update_query = """
            UPDATE megamind_sessions SET
                session_name = %s, session_state = %s, priority = %s,
                enable_semantic_indexing = %s, content_token_limit = %s,
                session_config = %s, session_tags = %s, total_entries = %s,
                total_chunks_accessed = %s, total_operations = %s,
                performance_score = %s, context_quality_score = %s,
                last_activity = %s, archived_at = %s
            WHERE session_id = %s
            """
            
            session_config_json = json.dumps(session.session_config) if session.session_config else None
            session_tags_json = json.dumps(session.session_tags) if session.session_tags else None
            
            cursor.execute(update_query, (
                session.session_name, session.session_state.value, session.priority.value,
                session.enable_semantic_indexing, session.content_token_limit,
                session_config_json, session_tags_json, session.total_entries,
                session.total_chunks_accessed, session.total_operations,
                session.performance_score, session.context_quality_score,
                session.last_activity, session.archived_at, session.session_id
            ))
            
            connection.commit()
            cursor.close()
            connection.close()
            
        except Exception as e:
            logger.error(f"Failed to update session in database: {e}")
            raise
    
    def _get_session_from_database(self, session_id: str) -> Optional[SessionMetadata]:
        """Get session from database"""
        try:
            connection = self.db.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            query = """
            SELECT * FROM megamind_sessions WHERE session_id = %s
            """
            
            cursor.execute(query, (session_id,))
            result = cursor.fetchone()
            cursor.close()
            connection.close()
            
            if not result:
                return None
            
            # Convert database result to SessionMetadata
            session_config = json.loads(result['session_config']) if result['session_config'] else None
            session_tags = json.loads(result['session_tags']) if result['session_tags'] else None
            
            return SessionMetadata(
                session_id=result['session_id'],
                session_name=result['session_name'],
                user_id=result['user_id'],
                realm_id=result['realm_id'],
                project_context=result['project_context'],
                session_state=SessionState(result['session_state']),
                priority=SessionPriority(result['priority']),
                enable_semantic_indexing=bool(result['enable_semantic_indexing']),
                content_token_limit=result['content_token_limit'],
                session_config=session_config,
                session_tags=session_tags,
                total_entries=result['total_entries'],
                total_chunks_accessed=result['total_chunks_accessed'],
                total_operations=result['total_operations'],
                performance_score=float(result['performance_score']),
                context_quality_score=float(result['context_quality_score']),
                created_at=result['created_at'],
                last_activity=result['last_activity'],
                archived_at=result['archived_at']
            )
            
        except Exception as e:
            logger.error(f"Failed to get session from database: {e}")
            return None
    
    def _insert_entry_to_database(self, entry: SessionEntry):
        """Insert session entry to database"""
        try:
            # First, check if session_entries table exists, if not skip for now
            connection = self.db.get_connection()
            cursor = connection.cursor()
            
            # Check if table exists
            cursor.execute("SHOW TABLES LIKE 'megamind_session_entries'")
            table_exists = cursor.fetchone()
            
            if not table_exists:
                logger.warning("megamind_session_entries table does not exist, skipping entry insertion")
                cursor.close()
                connection.close()
                return
            
            insert_query = """
            INSERT INTO megamind_session_entries (
                entry_id, session_id, entry_type, operation_type, entry_content,
                content_tokens, content_truncated, original_content_hash,
                semantic_summary, related_chunk_ids, context_relevance_score,
                parent_entry_id, entry_sequence, conversation_turn,
                processing_time_ms, success_indicator, quality_score,
                entry_metadata, user_feedback, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            related_chunks_json = json.dumps(entry.related_chunk_ids) if entry.related_chunk_ids else None
            entry_metadata_json = json.dumps(entry.entry_metadata) if entry.entry_metadata else None
            user_feedback_json = json.dumps(entry.user_feedback) if entry.user_feedback else None
            
            cursor.execute(insert_query, (
                entry.entry_id, entry.session_id, entry.entry_type.value,
                entry.operation_type, entry.entry_content, entry.content_tokens,
                entry.content_truncated, entry.original_content_hash,
                entry.semantic_summary, related_chunks_json, entry.context_relevance_score,
                entry.parent_entry_id, entry.entry_sequence, entry.conversation_turn,
                entry.processing_time_ms, entry.success_indicator, entry.quality_score,
                entry_metadata_json, user_feedback_json, entry.created_at
            ))
            
            connection.commit()
            cursor.close()
            connection.close()
            
        except Exception as e:
            logger.error(f"Failed to insert entry to database: {e}")
            # Don't raise - allow session operations to continue even if entry logging fails
    
    def _get_entries_from_database(self, session_id: str, limit: int, offset: int) -> List[SessionEntry]:
        """Get session entries from database"""
        try:
            connection = self.db.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Check if table exists
            cursor.execute("SHOW TABLES LIKE 'megamind_session_entries'")
            table_exists = cursor.fetchone()
            
            if not table_exists:
                logger.warning("megamind_session_entries table does not exist")
                cursor.close()
                connection.close()
                return []
            
            query = """
            SELECT * FROM megamind_session_entries 
            WHERE session_id = %s 
            ORDER BY entry_sequence DESC 
            LIMIT %s OFFSET %s
            """
            
            cursor.execute(query, (session_id, limit, offset))
            results = cursor.fetchall()
            cursor.close()
            connection.close()
            
            entries = []
            for result in results:
                related_chunk_ids = json.loads(result['related_chunk_ids']) if result['related_chunk_ids'] else None
                entry_metadata = json.loads(result['entry_metadata']) if result['entry_metadata'] else None
                user_feedback = json.loads(result['user_feedback']) if result['user_feedback'] else None
                
                entry = SessionEntry(
                    entry_id=result['entry_id'],
                    session_id=result['session_id'],
                    entry_type=EntryType(result['entry_type']),
                    entry_content=result['entry_content'],
                    operation_type=result['operation_type'],
                    content_tokens=result['content_tokens'],
                    content_truncated=bool(result['content_truncated']),
                    original_content_hash=result['original_content_hash'],
                    semantic_summary=result['semantic_summary'],
                    related_chunk_ids=related_chunk_ids,
                    context_relevance_score=float(result['context_relevance_score']),
                    parent_entry_id=result['parent_entry_id'],
                    entry_sequence=result['entry_sequence'],
                    conversation_turn=result['conversation_turn'],
                    processing_time_ms=result['processing_time_ms'],
                    success_indicator=bool(result['success_indicator']),
                    quality_score=float(result['quality_score']),
                    entry_metadata=entry_metadata,
                    user_feedback=user_feedback,
                    created_at=result['created_at']
                )
                entries.append(entry)
            
            return entries
            
        except Exception as e:
            logger.error(f"Failed to get entries from database: {e}")
            return []
    
    def _list_sessions_from_database(self, user_id: str, realm_id: str, 
                                   state_filter: Optional[SessionState], limit: int) -> List[SessionMetadata]:
        """List sessions from database"""
        try:
            connection = self.db.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            if state_filter:
                query = """
                SELECT * FROM megamind_sessions 
                WHERE user_id = %s AND realm_id = %s AND session_state = %s
                ORDER BY last_activity DESC 
                LIMIT %s
                """
                cursor.execute(query, (user_id, realm_id, state_filter.value, limit))
            else:
                query = """
                SELECT * FROM megamind_sessions 
                WHERE user_id = %s AND realm_id = %s
                ORDER BY last_activity DESC 
                LIMIT %s
                """
                cursor.execute(query, (user_id, realm_id, limit))
            
            results = cursor.fetchall()
            cursor.close()
            connection.close()
            
            sessions = []
            for result in results:
                session_config = json.loads(result['session_config']) if result['session_config'] else None
                session_tags = json.loads(result['session_tags']) if result['session_tags'] else None
                
                session = SessionMetadata(
                    session_id=result['session_id'],
                    session_name=result['session_name'],
                    user_id=result['user_id'],
                    realm_id=result['realm_id'],
                    project_context=result['project_context'],
                    session_state=SessionState(result['session_state']),
                    priority=SessionPriority(result['priority']),
                    enable_semantic_indexing=bool(result['enable_semantic_indexing']),
                    content_token_limit=result['content_token_limit'],
                    session_config=session_config,
                    session_tags=session_tags,
                    total_entries=result['total_entries'],
                    total_chunks_accessed=result['total_chunks_accessed'],
                    total_operations=result['total_operations'],
                    performance_score=float(result['performance_score']),
                    context_quality_score=float(result['context_quality_score']),
                    created_at=result['created_at'],
                    last_activity=result['last_activity'],
                    archived_at=result['archived_at']
                )
                sessions.append(session)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to list sessions from database: {e}")
            return []