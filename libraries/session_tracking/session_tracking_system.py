#!/usr/bin/env python3
"""
Session Tracking System for Enhanced Multi-Embedding Entry System
Phase 3: Implements operational tracking for session resumption and context priming
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
import re

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of actions tracked in sessions"""
    SEARCH = "search"
    CREATE_CHUNK = "create_chunk"
    UPDATE_CHUNK = "update_chunk"
    DELETE_CHUNK = "delete_chunk"
    ANALYZE_DOCUMENT = "analyze_document"
    GENERATE_EMBEDDING = "generate_embedding"
    ASSESS_QUALITY = "assess_quality"
    DISCOVER_RELATIONSHIP = "discover_relationship"
    RETRIEVE_KNOWLEDGE = "retrieve_knowledge"
    PROMOTE_CHUNK = "promote_chunk"
    CUSTOM = "custom"

class ActionPriority(Enum):
    """Priority levels for actions"""
    CRITICAL = "critical"  # Must include in recap
    HIGH = "high"         # Important for context
    MEDIUM = "medium"     # Useful for understanding
    LOW = "low"           # Optional detail

@dataclass
class Action:
    """Represents a tracked action in a session"""
    action_id: str
    action_type: ActionType
    timestamp: datetime
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    priority: ActionPriority = ActionPriority.MEDIUM
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    
    def to_summary(self) -> str:
        """Convert action to a summary string"""
        summary = f"[{self.timestamp.strftime('%H:%M:%S')}] {self.description}"
        if self.result:
            summary += f" → {self.result}"
        if self.error:
            summary += f" ❌ Error: {self.error}"
        return summary

@dataclass
class SessionEntry:
    """Entry in the session tracking log"""
    entry_id: str
    session_id: str
    action: Action
    context_before: Optional[Dict[str, Any]] = None
    context_after: Optional[Dict[str, Any]] = None
    related_entries: List[str] = field(default_factory=list)

@dataclass
class SessionRecap:
    """Recap of a session for context priming"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_actions: int
    key_actions: List[Action]  # High priority actions
    accomplishments: List[str]
    pending_tasks: List[str]
    context_summary: Dict[str, Any]
    suggested_next_steps: List[str]
    
    def to_prompt(self) -> str:
        """Convert recap to a context prompt"""
        prompt_parts = [
            f"## Session Recap: {self.session_id}",
            f"**Duration**: {self.start_time.strftime('%Y-%m-%d %H:%M')} - {self.end_time.strftime('%H:%M') if self.end_time else 'ongoing'}",
            f"**Total Actions**: {self.total_actions}",
            ""
        ]
        
        if self.accomplishments:
            prompt_parts.extend([
                "### Accomplishments:",
                *[f"- {acc}" for acc in self.accomplishments],
                ""
            ])
        
        if self.key_actions:
            prompt_parts.extend([
                "### Key Actions:",
                *[f"- {action.to_summary()}" for action in self.key_actions[:5]],
                ""
            ])
        
        if self.pending_tasks:
            prompt_parts.extend([
                "### Pending Tasks:",
                *[f"- {task}" for task in self.pending_tasks],
                ""
            ])
        
        if self.suggested_next_steps:
            prompt_parts.extend([
                "### Suggested Next Steps:",
                *[f"- {step}" for step in self.suggested_next_steps],
                ""
            ])
        
        return "\n".join(prompt_parts)

@dataclass
class ContextPrimer:
    """Context primer for session resumption"""
    session_recap: SessionRecap
    relevant_chunks: List[str]
    active_documents: List[str]
    search_history: List[str]
    modification_history: List[Tuple[str, str]]  # (chunk_id, action)
    
    def to_context(self) -> Dict[str, Any]:
        """Convert to context dictionary"""
        return {
            'session_id': self.session_recap.session_id,
            'recap_prompt': self.session_recap.to_prompt(),
            'relevant_chunks': self.relevant_chunks[:10],
            'active_documents': self.active_documents,
            'recent_searches': self.search_history[-5:],
            'recent_modifications': [
                f"{chunk_id}: {action}" 
                for chunk_id, action in self.modification_history[-5:]
            ]
        }

class SessionTrackingSystem:
    """
    Lightweight operational tracking system for session management
    """
    
    def __init__(self, db_connection=None, max_history_size: int = 1000):
        self.db = db_connection
        self.max_history_size = max_history_size
        
        # In-memory tracking structures
        self._active_sessions: Dict[str, List[SessionEntry]] = {}
        self._session_metadata: Dict[str, Dict[str, Any]] = {}
        self._action_history: deque = deque(maxlen=max_history_size)
        
        # Pattern matchers for accomplishment detection
        self._accomplishment_patterns = [
            (re.compile(r'created (\d+) chunks?'), "Created {} chunks"),
            (re.compile(r'analyzed document (.+)'), "Analyzed document: {}"),
            (re.compile(r'generated (\d+) embeddings?'), "Generated {} embeddings"),
            (re.compile(r'discovered (\d+) relationships?'), "Discovered {} relationships"),
            (re.compile(r'assessed quality.*score: ([\d.]+)'), "Quality assessment completed (score: {})"),
            (re.compile(r'promoted (\d+) chunks? to (.+)'), "Promoted {} chunks to {}"),
        ]
        
        logger.info("SessionTrackingSystem initialized")
    
    async def create_operational_session(self, 
                                       session_type: str = "general",
                                       user_id: Optional[str] = None,
                                       description: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new operational session
        
        Args:
            session_type: Type of session
            user_id: Optional user identifier
            description: Optional session description
            
        Returns:
            Session information
        """
        session_id = self._generate_session_id(user_id)
        
        self._active_sessions[session_id] = []
        self._session_metadata[session_id] = {
            'session_id': session_id,
            'session_type': session_type,
            'user_id': user_id,
            'description': description,
            'start_time': datetime.now(),
            'end_time': None,
            'total_actions': 0,
            'accomplishments': [],
            'pending_tasks': [],
            'active_documents': set(),
            'search_history': deque(maxlen=20),
            'modification_history': deque(maxlen=20)
        }
        
        logger.info(f"Created operational session: {session_id}")
        return {
            'session_id': session_id,
            'status': 'active',
            'created_at': self._session_metadata[session_id]['start_time'].isoformat()
        }
    
    async def track_session_action(self,
                                 session_id: str,
                                 action_type: ActionType,
                                 description: str,
                                 details: Optional[Dict[str, Any]] = None,
                                 priority: Optional[ActionPriority] = None) -> SessionEntry:
        """
        Track an action in the session
        
        Args:
            session_id: Session identifier
            action_type: Type of action
            description: Human-readable description
            details: Optional action details
            priority: Optional priority level
            
        Returns:
            SessionEntry object
        """
        if session_id not in self._active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        # Create action
        action = Action(
            action_id=self._generate_action_id(session_id),
            action_type=action_type,
            timestamp=datetime.now(),
            description=description,
            details=details or {},
            priority=priority or self._infer_priority(action_type)
        )
        
        # Create session entry
        entry = SessionEntry(
            entry_id=f"entry_{action.action_id}",
            session_id=session_id,
            action=action
        )
        
        # Track the action
        self._active_sessions[session_id].append(entry)
        self._action_history.append(entry)
        self._session_metadata[session_id]['total_actions'] += 1
        
        # Update session metadata based on action
        await self._update_session_metadata(session_id, action)
        
        # Check for accomplishments
        self._detect_accomplishments(session_id, action)
        
        logger.debug(f"Tracked action in session {session_id}: {description}")
        return entry
    
    async def generate_session_recap(self, session_id: str) -> SessionRecap:
        """
        Generate a recap of the session
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionRecap object
        """
        if session_id not in self._active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        metadata = self._session_metadata[session_id]
        entries = self._active_sessions[session_id]
        
        # Extract key actions (high priority)
        key_actions = [
            entry.action for entry in entries 
            if entry.action.priority in [ActionPriority.CRITICAL, ActionPriority.HIGH]
        ]
        
        # If not enough key actions, add some medium priority ones
        if len(key_actions) < 5:
            medium_actions = [
                entry.action for entry in entries 
                if entry.action.priority == ActionPriority.MEDIUM
            ]
            key_actions.extend(medium_actions[:5-len(key_actions)])
        
        # Generate context summary
        context_summary = {
            'documents_processed': len(metadata['active_documents']),
            'searches_performed': len(metadata['search_history']),
            'modifications_made': len(metadata['modification_history']),
            'session_type': metadata['session_type']
        }
        
        # Suggest next steps
        suggested_next_steps = self._suggest_next_steps(metadata, entries)
        
        recap = SessionRecap(
            session_id=session_id,
            start_time=metadata['start_time'],
            end_time=metadata['end_time'],
            total_actions=metadata['total_actions'],
            key_actions=key_actions[:10],
            accomplishments=metadata['accomplishments'],
            pending_tasks=metadata['pending_tasks'],
            context_summary=context_summary,
            suggested_next_steps=suggested_next_steps
        )
        
        return recap
    
    async def prime_context_from_session(self, session_id: str) -> ContextPrimer:
        """
        Prime context for session resumption
        
        Args:
            session_id: Session identifier
            
        Returns:
            ContextPrimer object
        """
        # Generate recap
        recap = await self.generate_session_recap(session_id)
        
        metadata = self._session_metadata[session_id]
        entries = self._active_sessions[session_id]
        
        # Extract relevant chunks from actions
        relevant_chunks = []
        for entry in entries[-20:]:  # Last 20 actions
            if 'chunk_id' in entry.action.details:
                relevant_chunks.append(entry.action.details['chunk_id'])
            if 'chunk_ids' in entry.action.details:
                relevant_chunks.extend(entry.action.details['chunk_ids'])
        
        # Remove duplicates while preserving order
        seen = set()
        relevant_chunks = [x for x in relevant_chunks if not (x in seen or seen.add(x))]
        
        primer = ContextPrimer(
            session_recap=recap,
            relevant_chunks=relevant_chunks,
            active_documents=list(metadata['active_documents']),
            search_history=list(metadata['search_history']),
            modification_history=list(metadata['modification_history'])
        )
        
        return primer
    
    async def close_session(self, session_id: str) -> SessionRecap:
        """
        Close a session and return final recap
        
        Args:
            session_id: Session identifier
            
        Returns:
            Final SessionRecap
        """
        if session_id not in self._active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        # Mark session as ended
        self._session_metadata[session_id]['end_time'] = datetime.now()
        
        # Generate final recap
        recap = await self.generate_session_recap(session_id)
        
        # Store session data if database available
        if self.db:
            await self._store_session_data(session_id, recap)
        
        # Clean up active session
        del self._active_sessions[session_id]
        
        logger.info(f"Closed session {session_id}")
        return recap
    
    # Private helper methods
    
    def _generate_session_id(self, user_id: Optional[str]) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        base = f"{user_id or 'anonymous'}_{timestamp}"
        return f"session_{hashlib.md5(base.encode()).hexdigest()[:12]}"
    
    def _generate_action_id(self, session_id: str) -> str:
        """Generate unique action ID"""
        timestamp = datetime.now().isoformat()
        count = self._session_metadata[session_id]['total_actions']
        return f"{session_id}_{count:04d}_{hashlib.md5(timestamp.encode()).hexdigest()[:8]}"
    
    def _infer_priority(self, action_type: ActionType) -> ActionPriority:
        """Infer priority from action type"""
        priority_map = {
            ActionType.CREATE_CHUNK: ActionPriority.HIGH,
            ActionType.UPDATE_CHUNK: ActionPriority.HIGH,
            ActionType.DELETE_CHUNK: ActionPriority.CRITICAL,
            ActionType.ANALYZE_DOCUMENT: ActionPriority.HIGH,
            ActionType.PROMOTE_CHUNK: ActionPriority.CRITICAL,
            ActionType.DISCOVER_RELATIONSHIP: ActionPriority.MEDIUM,
            ActionType.GENERATE_EMBEDDING: ActionPriority.LOW,
            ActionType.ASSESS_QUALITY: ActionPriority.MEDIUM,
            ActionType.RETRIEVE_KNOWLEDGE: ActionPriority.MEDIUM,
            ActionType.SEARCH: ActionPriority.MEDIUM,
            ActionType.CUSTOM: ActionPriority.LOW
        }
        return priority_map.get(action_type, ActionPriority.MEDIUM)
    
    async def _update_session_metadata(self, session_id: str, action: Action):
        """Update session metadata based on action"""
        metadata = self._session_metadata[session_id]
        
        # Track documents
        if 'document_id' in action.details:
            metadata['active_documents'].add(action.details['document_id'])
        
        # Track searches
        if action.action_type == ActionType.SEARCH and 'query' in action.details:
            metadata['search_history'].append(action.details['query'])
        
        # Track modifications
        if action.action_type in [ActionType.CREATE_CHUNK, ActionType.UPDATE_CHUNK, ActionType.DELETE_CHUNK]:
            if 'chunk_id' in action.details:
                metadata['modification_history'].append(
                    (action.details['chunk_id'], action.action_type.value)
                )
    
    def _detect_accomplishments(self, session_id: str, action: Action):
        """Detect accomplishments from action descriptions"""
        description_lower = action.description.lower()
        
        for pattern, template in self._accomplishment_patterns:
            match = pattern.search(description_lower)
            if match:
                accomplishment = template.format(*match.groups())
                if accomplishment not in self._session_metadata[session_id]['accomplishments']:
                    self._session_metadata[session_id]['accomplishments'].append(accomplishment)
                break
    
    def _suggest_next_steps(self, 
                          metadata: Dict[str, Any], 
                          entries: List[SessionEntry]) -> List[str]:
        """Suggest next steps based on session activity"""
        suggestions = []
        
        # Check for incomplete workflows
        recent_actions = [e.action for e in entries[-10:]]
        recent_types = [a.action_type for a in recent_actions]
        
        # Suggest based on patterns
        if ActionType.ANALYZE_DOCUMENT in recent_types and ActionType.CREATE_CHUNK not in recent_types:
            suggestions.append("Create chunks from the analyzed document")
        
        if ActionType.CREATE_CHUNK in recent_types and ActionType.GENERATE_EMBEDDING not in recent_types:
            suggestions.append("Generate embeddings for the created chunks")
        
        if ActionType.ASSESS_QUALITY in recent_types:
            # Check if any chunks had low quality
            for action in recent_actions:
                if action.action_type == ActionType.ASSESS_QUALITY:
                    score = action.details.get('quality_score', 1.0)
                    if score < 0.7:
                        suggestions.append("Improve quality of low-scoring chunks")
                        break
        
        # If many searches but no chunk creation
        search_count = sum(1 for a in recent_actions if a.action_type == ActionType.SEARCH)
        create_count = sum(1 for a in recent_actions if a.action_type == ActionType.CREATE_CHUNK)
        if search_count > 3 and create_count == 0:
            suggestions.append("Consider creating new knowledge chunks for frequently searched topics")
        
        # Always suggest session review if many actions
        if metadata['total_actions'] > 20:
            suggestions.append("Review session accomplishments and pending tasks")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    async def _store_session_data(self, session_id: str, recap: SessionRecap):
        """Store session data in database"""
        # This would integrate with the database layer
        logger.info(f"Would store session data for {session_id}")
    
    # Additional utility methods
    
    async def get_recent_sessions(self, 
                                user_id: Optional[str] = None, 
                                limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sessions, optionally filtered by user"""
        recent = []
        
        for session_id, metadata in self._session_metadata.items():
            if user_id is None or metadata.get('user_id') == user_id:
                recent.append({
                    'session_id': session_id,
                    'start_time': metadata['start_time'],
                    'end_time': metadata.get('end_time'),
                    'total_actions': metadata['total_actions'],
                    'session_type': metadata['session_type'],
                    'description': metadata.get('description', '')
                })
        
        # Sort by start time descending
        recent.sort(key=lambda x: x['start_time'], reverse=True)
        
        return recent[:limit]
    
    async def get_session_timeline(self, session_id: str) -> List[Dict[str, Any]]:
        """Get timeline of actions in a session"""
        if session_id not in self._active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        timeline = []
        for entry in self._active_sessions[session_id]:
            timeline.append({
                'timestamp': entry.action.timestamp.isoformat(),
                'action_type': entry.action.action_type.value,
                'description': entry.action.description,
                'priority': entry.action.priority.value,
                'duration_ms': entry.action.duration_ms
            })
        
        return timeline