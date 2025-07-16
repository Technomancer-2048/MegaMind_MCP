#!/usr/bin/env python3
"""
Session Management Layer for Enhanced Multi-Embedding Entry System
Handles session lifecycle, state tracking, and session-aware operations
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class SessionType(Enum):
    """Types of embedding sessions"""
    ANALYSIS = "analysis"
    INGESTION = "ingestion"
    CURATION = "curation"
    MIXED = "mixed"

class SessionStatus(Enum):
    """Session lifecycle states"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class OperationType(Enum):
    """Types of chunk operations"""
    CREATED = "created"
    UPDATED = "updated"
    ANALYZED = "analyzed"
    EMBEDDED = "embedded"
    QUALITY_ASSESSED = "quality_assessed"

@dataclass
class SessionConfig:
    """Configuration for embedding sessions"""
    timeout_minutes: int = 120
    auto_save_interval_seconds: int = 300
    chunk_batch_size: int = 100
    embedding_batch_size: int = 32
    quality_threshold: float = 0.7
    max_concurrent_operations: int = 5

@dataclass
class SessionState:
    """Current state of a session"""
    session_id: str
    current_document: Optional[str] = None
    current_chunk_index: int = 0
    processed_chunks: List[str] = field(default_factory=list)
    failed_chunks: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoints: Dict[str, Any] = field(default_factory=dict)
    last_saved: datetime = field(default_factory=datetime.now)

@dataclass
class SessionMetrics:
    """Metrics tracked for a session"""
    total_chunks_processed: int = 0
    total_embeddings_generated: int = 0
    average_quality_score: float = 0.0
    processing_time_ms: int = 0
    error_count: int = 0
    retry_count: int = 0

class SessionManager:
    """
    Manages embedding session lifecycle and operations
    """
    
    def __init__(self, database_connection, config: Optional[SessionConfig] = None):
        self.db = database_connection
        self.config = config or SessionConfig()
        self._active_sessions: Dict[str, SessionState] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._auto_save_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("SessionManager initialized with config: %s", self.config)
    
    async def create_session(self, 
                           session_type: SessionType,
                           realm_id: str,
                           created_by: str,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new embedding session
        
        Args:
            session_type: Type of session (analysis, ingestion, etc.)
            realm_id: Target realm for the session
            created_by: User/system creating the session
            metadata: Optional session metadata
            
        Returns:
            Session ID
        """
        session_id = self._generate_session_id()
        
        # Create session in database
        async with self.db.connection() as conn:
            await conn.execute(
                """
                INSERT INTO megamind_embedding_sessions 
                (session_id, session_type, realm_id, created_by, metadata)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, session_type.value, realm_id, created_by, 
                 json.dumps(metadata or {}))
            )
            await conn.commit()
        
        # Initialize session state
        session_state = SessionState(session_id=session_id)
        self._active_sessions[session_id] = session_state
        self._session_locks[session_id] = asyncio.Lock()
        
        # Start auto-save task
        self._auto_save_tasks[session_id] = asyncio.create_task(
            self._auto_save_session(session_id)
        )
        
        logger.info(f"Created session {session_id} of type {session_type.value}")
        return session_id
    
    async def get_session_state(self, session_id: str) -> Optional[SessionState]:
        """Get current session state"""
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]
        
        # Try to load from database
        state_data = await self._load_session_state(session_id)
        if state_data:
            session_state = SessionState(
                session_id=session_id,
                current_document=state_data.get('current_document'),
                current_chunk_index=state_data.get('current_chunk_index', 0),
                processed_chunks=state_data.get('processed_chunks', []),
                failed_chunks=state_data.get('failed_chunks', []),
                metrics=state_data.get('metrics', {}),
                checkpoints=state_data.get('checkpoints', {})
            )
            self._active_sessions[session_id] = session_state
            self._session_locks[session_id] = asyncio.Lock()
            return session_state
        
        return None
    
    async def update_session_state(self, 
                                 session_id: str,
                                 updates: Dict[str, Any]) -> bool:
        """Update session state"""
        async with self._get_session_lock(session_id):
            state = await self.get_session_state(session_id)
            if not state:
                return False
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)
            
            # Save to database
            await self._save_session_state(session_id, state)
            
            return True
    
    async def track_chunk_operation(self,
                                  session_id: str,
                                  chunk_id: str,
                                  operation_type: OperationType,
                                  metadata: Optional[Dict[str, Any]] = None,
                                  quality_score: Optional[float] = None,
                                  embedding_id: Optional[str] = None) -> bool:
        """
        Track a chunk operation within a session
        
        Args:
            session_id: Session ID
            chunk_id: Chunk ID
            operation_type: Type of operation performed
            metadata: Optional operation metadata
            quality_score: Optional quality score
            embedding_id: Optional embedding ID
            
        Returns:
            Success status
        """
        try:
            async with self.db.connection() as conn:
                # Generate tracking ID
                tracking_id = f"sc_{hashlib.md5(f'{session_id}{chunk_id}{datetime.now()}'.encode()).hexdigest()[:12]}"
                
                # Insert tracking record
                await conn.execute(
                    """
                    INSERT INTO megamind_session_chunks
                    (session_chunk_id, session_id, chunk_id, operation_type,
                     operation_metadata, quality_score, embedding_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (tracking_id, session_id, chunk_id, operation_type.value,
                     json.dumps(metadata or {}), quality_score, embedding_id)
                )
                
                # Update session metrics
                await conn.execute(
                    """
                    UPDATE megamind_embedding_sessions
                    SET total_chunks_processed = total_chunks_processed + 1,
                        total_embeddings_generated = total_embeddings_generated + %s
                    WHERE session_id = %s
                    """,
                    (1 if embedding_id else 0, session_id)
                )
                
                await conn.commit()
            
            # Update in-memory state
            async with self._get_session_lock(session_id):
                state = await self.get_session_state(session_id)
                if state:
                    state.processed_chunks.append(chunk_id)
                    state.metrics['last_operation'] = operation_type.value
                    state.metrics['last_operation_time'] = datetime.now().isoformat()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to track chunk operation: {e}")
            return False
    
    async def add_document_to_session(self,
                                    session_id: str,
                                    document_id: str,
                                    document_path: str,
                                    document_hash: Optional[str] = None) -> bool:
        """Add a document to session for processing"""
        try:
            tracking_id = f"sd_{hashlib.md5(f'{session_id}{document_id}'.encode()).hexdigest()[:12]}"
            
            async with self.db.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO megamind_session_documents
                    (session_document_id, session_id, document_id, 
                     document_path, document_hash, processing_status)
                    VALUES (%s, %s, %s, %s, %s, 'pending')
                    """,
                    (tracking_id, session_id, document_id, document_path, document_hash)
                )
                await conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document to session: {e}")
            return False
    
    async def update_document_status(self,
                                   session_id: str,
                                   document_id: str,
                                   status: str,
                                   chunks_created: Optional[int] = None,
                                   embeddings_created: Optional[int] = None,
                                   error_details: Optional[str] = None) -> bool:
        """Update document processing status"""
        try:
            async with self.db.connection() as conn:
                query_parts = [
                    "UPDATE megamind_session_documents",
                    "SET processing_status = %s"
                ]
                params = [status]
                
                if chunks_created is not None:
                    query_parts.append(", chunks_created = %s")
                    params.append(chunks_created)
                
                if embeddings_created is not None:
                    query_parts.append(", embeddings_created = %s")
                    params.append(embeddings_created)
                
                if error_details:
                    query_parts.append(", error_details = %s")
                    params.append(error_details)
                
                if status == 'processing':
                    query_parts.append(", started_at = NOW()")
                elif status in ['completed', 'failed']:
                    query_parts.append(", completed_at = NOW()")
                
                query_parts.append("WHERE session_id = %s AND document_id = %s")
                params.extend([session_id, document_id])
                
                await conn.execute(" ".join(query_parts), params)
                await conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document status: {e}")
            return False
    
    async def record_metric(self,
                          session_id: str,
                          metric_type: str,
                          metric_value: float,
                          metric_unit: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Record a session metric"""
        try:
            metric_id = f"metric_{hashlib.md5(f'{session_id}{metric_type}{datetime.now()}'.encode()).hexdigest()[:12]}"
            
            async with self.db.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO megamind_session_metrics
                    (metric_id, session_id, metric_type, metric_value, 
                     metric_unit, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (metric_id, session_id, metric_type, metric_value,
                     metric_unit, json.dumps(metadata or {}))
                )
                await conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
            return False
    
    async def get_session_progress(self, session_id: str) -> Dict[str, Any]:
        """Get detailed session progress"""
        async with self.db.connection() as conn:
            # Get session info
            cursor = await conn.execute(
                """
                SELECT s.*, 
                       COUNT(DISTINCT sd.document_id) as total_documents,
                       COUNT(DISTINCT sc.chunk_id) as total_chunks,
                       AVG(sc.quality_score) as avg_quality_score
                FROM megamind_embedding_sessions s
                LEFT JOIN megamind_session_documents sd ON s.session_id = sd.session_id
                LEFT JOIN megamind_session_chunks sc ON s.session_id = sc.session_id
                WHERE s.session_id = %s
                GROUP BY s.session_id
                """,
                (session_id,)
            )
            session_info = await cursor.fetchone()
            
            if not session_info:
                return {}
            
            # Get document progress
            cursor = await conn.execute(
                """
                SELECT processing_status, COUNT(*) as count,
                       SUM(chunks_created) as chunks,
                       SUM(embeddings_created) as embeddings
                FROM megamind_session_documents
                WHERE session_id = %s
                GROUP BY processing_status
                """,
                (session_id,)
            )
            doc_progress = {row['processing_status']: {
                'count': row['count'],
                'chunks': row['chunks'] or 0,
                'embeddings': row['embeddings'] or 0
            } for row in await cursor.fetchall()}
            
            # Get recent metrics
            cursor = await conn.execute(
                """
                SELECT metric_type, metric_value, metric_unit, recorded_at
                FROM megamind_session_metrics
                WHERE session_id = %s
                ORDER BY recorded_at DESC
                LIMIT 10
                """,
                (session_id,)
            )
            recent_metrics = await cursor.fetchall()
            
            return {
                'session_id': session_id,
                'status': session_info['status'],
                'created_date': session_info['created_date'],
                'last_activity': session_info['last_activity'],
                'total_documents': session_info['total_documents'],
                'total_chunks': session_info['total_chunks'],
                'avg_quality_score': float(session_info['avg_quality_score'] or 0),
                'document_progress': doc_progress,
                'recent_metrics': [dict(m) for m in recent_metrics]
            }
    
    async def complete_session(self, session_id: str) -> bool:
        """Mark session as completed"""
        try:
            # Stop auto-save task
            if session_id in self._auto_save_tasks:
                self._auto_save_tasks[session_id].cancel()
                del self._auto_save_tasks[session_id]
            
            # Final save of state
            if session_id in self._active_sessions:
                await self._save_session_state(session_id, self._active_sessions[session_id])
            
            # Update database
            async with self.db.connection() as conn:
                await conn.execute(
                    """
                    UPDATE megamind_embedding_sessions
                    SET status = 'completed',
                        processing_duration_ms = TIMESTAMPDIFF(MICROSECOND, created_date, NOW()) / 1000
                    WHERE session_id = %s
                    """,
                    (session_id,)
                )
                await conn.commit()
            
            # Clean up in-memory state
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
            if session_id in self._session_locks:
                del self._session_locks[session_id]
            
            logger.info(f"Completed session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete session: {e}")
            return False
    
    async def pause_session(self, session_id: str) -> bool:
        """Pause an active session"""
        try:
            # Save current state
            if session_id in self._active_sessions:
                await self._save_session_state(session_id, self._active_sessions[session_id])
            
            # Update status
            async with self.db.connection() as conn:
                await conn.execute(
                    "UPDATE megamind_embedding_sessions SET status = 'paused' WHERE session_id = %s",
                    (session_id,)
                )
                await conn.commit()
            
            # Stop auto-save but keep state in memory
            if session_id in self._auto_save_tasks:
                self._auto_save_tasks[session_id].cancel()
                del self._auto_save_tasks[session_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause session: {e}")
            return False
    
    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused session"""
        try:
            # Update status
            async with self.db.connection() as conn:
                await conn.execute(
                    "UPDATE megamind_embedding_sessions SET status = 'active' WHERE session_id = %s",
                    (session_id,)
                )
                await conn.commit()
            
            # Ensure state is loaded
            state = await self.get_session_state(session_id)
            if state:
                # Restart auto-save
                self._auto_save_tasks[session_id] = asyncio.create_task(
                    self._auto_save_session(session_id)
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resume session: {e}")
            return False
    
    @asynccontextmanager
    async def session_operation(self, session_id: str, operation_name: str):
        """Context manager for session operations with automatic tracking"""
        start_time = datetime.now()
        
        try:
            # Record operation start
            await self.record_metric(
                session_id, f"{operation_name}_started", 1.0,
                metadata={'timestamp': start_time.isoformat()}
            )
            
            yield
            
            # Record operation success
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            await self.record_metric(
                session_id, f"{operation_name}_duration_ms", duration_ms, "ms"
            )
            
        except Exception as e:
            # Record operation failure
            await self.record_metric(
                session_id, f"{operation_name}_failed", 1.0,
                metadata={'error': str(e), 'timestamp': datetime.now().isoformat()}
            )
            raise
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=self.config.timeout_minutes)
            
            async with self.db.connection() as conn:
                # Find expired sessions
                cursor = await conn.execute(
                    """
                    SELECT session_id FROM megamind_embedding_sessions
                    WHERE status = 'active' AND last_activity < %s
                    """,
                    (cutoff_time,)
                )
                expired_sessions = [row['session_id'] for row in await cursor.fetchall()]
                
                # Cancel them
                for session_id in expired_sessions:
                    await self.complete_session(session_id)
                
                return len(expired_sessions)
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    # Private helper methods
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{hashlib.md5(f'{datetime.now()}'.encode()).hexdigest()[:12]}"
    
    async def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get lock for session operations"""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]
    
    async def _save_session_state(self, session_id: str, state: SessionState) -> bool:
        """Save session state to database"""
        try:
            state_data = {
                'current_document': state.current_document,
                'current_chunk_index': state.current_chunk_index,
                'processed_chunks': state.processed_chunks,
                'failed_chunks': state.failed_chunks,
                'metrics': state.metrics,
                'checkpoints': state.checkpoints
            }
            
            async with self.db.connection() as conn:
                # Save main state
                await conn.execute(
                    """
                    INSERT INTO megamind_session_state 
                    (state_id, session_id, state_key, state_value)
                    VALUES (%s, %s, 'main_state', %s)
                    ON DUPLICATE KEY UPDATE 
                        state_value = VALUES(state_value),
                        updated_timestamp = CURRENT_TIMESTAMP
                    """,
                    (f"state_{session_id}_main", session_id, json.dumps(state_data))
                )
                await conn.commit()
            
            state.last_saved = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")
            return False
    
    async def _load_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session state from database"""
        try:
            async with self.db.connection() as conn:
                cursor = await conn.execute(
                    """
                    SELECT state_value FROM megamind_session_state
                    WHERE session_id = %s AND state_key = 'main_state'
                    ORDER BY updated_timestamp DESC
                    LIMIT 1
                    """,
                    (session_id,)
                )
                row = await cursor.fetchone()
                
                if row:
                    return json.loads(row['state_value'])
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to load session state: {e}")
            return None
    
    async def _auto_save_session(self, session_id: str):
        """Auto-save session state periodically"""
        while session_id in self._active_sessions:
            try:
                await asyncio.sleep(self.config.auto_save_interval_seconds)
                
                if session_id in self._active_sessions:
                    state = self._active_sessions[session_id]
                    await self._save_session_state(session_id, state)
                    logger.debug(f"Auto-saved session {session_id}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-save for session {session_id}: {e}")