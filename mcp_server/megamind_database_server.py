#!/usr/bin/env python3
"""
MegaMind Context Database MCP Server
Full MCP protocol implementation
"""

import json
import logging
import os
import sys
import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any, Union

import mysql.connector
from mysql.connector import pooling

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MegaMindDatabase:
    """Manages database connections and operations for MegaMind context system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection_pool = None
        self._setup_connection_pool()
    
    def _setup_connection_pool(self):
        """Setup MySQL connection pool"""
        try:
            pool_config = {
                'pool_name': 'megamind_mcp_pool',
                'pool_size': int(self.config.get('pool_size', 10)),
                'host': self.config['host'],
                'port': int(self.config['port']),
                'database': self.config['database'],
                'user': self.config['user'],
                'password': self.config['password'],
                'autocommit': False,
                'charset': 'utf8mb4',
                'use_unicode': True
            }
            self.connection_pool = pooling.MySQLConnectionPool(**pool_config)
            logger.info("Database connection pool established")
        except Exception as e:
            logger.error(f"Failed to setup database connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get a connection from the pool"""
        return self.connection_pool.get_connection()
    
    def search_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search chunks using simple text search"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Simple LIKE search for now
            search_query = """
            SELECT chunk_id, content, source_document, section_path, chunk_type,
                   line_count, token_count, access_count, last_accessed
            FROM megamind_chunks
            WHERE content LIKE %s OR source_document LIKE %s OR section_path LIKE %s
            ORDER BY access_count DESC
            LIMIT %s
            """
            
            like_query = f"%{query}%"
            cursor.execute(search_query, (like_query, like_query, like_query, limit))
            results = cursor.fetchall()
            
            # Convert datetime objects to strings
            for row in results:
                if row['last_accessed']:
                    row['last_accessed'] = row['last_accessed'].isoformat()
                else:
                    row['last_accessed'] = ''
            
            logger.info(f"Found {len(results)} chunks matching query: {query[:50]}")
            return results
            
        except Exception as e:
            logger.error(f"Search query failed: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def get_chunk(self, chunk_id: str, include_relationships: bool = True) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by ID with optional relationships"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get chunk data
            chunk_query = """
            SELECT chunk_id, content, source_document, section_path, chunk_type,
                   line_count, token_count, access_count, last_accessed
            FROM megamind_chunks
            WHERE chunk_id = %s
            """
            cursor.execute(chunk_query, (chunk_id,))
            chunk = cursor.fetchone()
            
            if not chunk:
                return None
            
            # Convert datetime to string
            if chunk['last_accessed']:
                chunk['last_accessed'] = chunk['last_accessed'].isoformat()
            else:
                chunk['last_accessed'] = ''
            
            # Get relationships if requested
            if include_relationships:
                rel_query = """
                SELECT related_chunk_id, relationship_type, strength
                FROM megamind_chunk_relationships
                WHERE chunk_id = %s
                """
                cursor.execute(rel_query, (chunk_id,))
                relationships = cursor.fetchall()
                # Convert Decimal to float for JSON serialization
                for rel in relationships:
                    rel['strength'] = float(rel['strength']) if rel['strength'] else 0.0
                chunk['relationships'] = relationships
            
            return chunk
            
        except Exception as e:
            logger.error(f"Get chunk failed: {e}")
            return None
        finally:
            if connection:
                connection.close()
    
    def get_related_chunks(self, chunk_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Get related chunks up to max_depth levels"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Recursive relationship query with depth limit
            related_query = """
            WITH RECURSIVE chunk_relations AS (
                SELECT related_chunk_id, relationship_type, strength, 1 as depth
                FROM megamind_chunk_relationships
                WHERE chunk_id = %s
                
                UNION ALL
                
                SELECT cr.related_chunk_id, cr.relationship_type, cr.strength, depth + 1
                FROM megamind_chunk_relationships cr
                JOIN chunk_relations r ON cr.chunk_id = r.related_chunk_id
                WHERE depth < %s
            )
            SELECT DISTINCT c.chunk_id, c.content, c.source_document, c.section_path,
                   c.chunk_type, c.line_count, c.token_count, r.relationship_type, r.strength
            FROM chunk_relations r
            JOIN megamind_chunks c ON r.related_chunk_id = c.chunk_id
            ORDER BY r.strength DESC
            """
            
            cursor.execute(related_query, (chunk_id, max_depth))
            results = cursor.fetchall()
            
            # Convert Decimal to float for JSON serialization
            for result in results:
                if 'strength' in result and result['strength'] is not None:
                    result['strength'] = float(result['strength'])
            
            return results
            
        except Exception as e:
            logger.error(f"Get related chunks failed: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def get_session_primer(self, last_session_data: Optional[str] = None) -> Dict[str, Any]:
        """Generate lightweight context for session continuity"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get recent high-access chunks for context priming
            primer_query = """
            SELECT chunk_id, content, source_document, section_path, access_count
            FROM megamind_chunks
            WHERE access_count > 5
            ORDER BY last_accessed DESC, access_count DESC
            LIMIT 10
            """
            cursor.execute(primer_query)
            recent_contexts = cursor.fetchall()
            
            # Get active sessions with pending changes
            session_query = """
            SELECT s.session_id, s.user_context, s.project_context, s.pending_changes_count
            FROM megamind_session_metadata s
            WHERE s.is_active = TRUE AND s.pending_changes_count > 0
            ORDER BY s.last_activity DESC
            LIMIT 5
            """
            cursor.execute(session_query)
            active_sessions = cursor.fetchall()
            
            return {
                "recent_contexts": recent_contexts,
                "active_sessions": active_sessions,
                "last_session_data": last_session_data
            }
            
        except Exception as e:
            logger.error(f"Get session primer failed: {e}")
            return {"recent_contexts": [], "active_sessions": [], "last_session_data": None}
        finally:
            if connection:
                connection.close()
    
    def track_access(self, chunk_id: str, query_context: str = "") -> bool:
        """Update access analytics for optimization"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Update chunk access count and timestamp
            update_query = """
            UPDATE megamind_chunks
            SET access_count = access_count + 1,
                last_accessed = NOW()
            WHERE chunk_id = %s
            """
            cursor.execute(update_query, (chunk_id,))
            connection.commit()
            
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"Track access failed: {e}")
            return False
        finally:
            if connection:
                connection.close()
    
    def get_hot_contexts(self, model_type: str = "sonnet", limit: int = 20) -> List[Dict[str, Any]]:
        """Get frequently accessed chunks prioritized by usage patterns"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Adjust for model type - Opus gets more selective, high-value chunks
            if model_type.lower() == "opus":
                hot_query = """
                SELECT chunk_id, content, source_document, section_path, 
                       chunk_type, access_count, last_accessed
                FROM megamind_chunks
                WHERE access_count >= 10
                ORDER BY access_count DESC, token_count ASC
                LIMIT %s
                """
            else:  # Sonnet and others
                hot_query = """
                SELECT chunk_id, content, source_document, section_path,
                       chunk_type, access_count, last_accessed
                FROM megamind_chunks
                WHERE access_count >= 3
                ORDER BY last_accessed DESC, access_count DESC
                LIMIT %s
                """
            
            cursor.execute(hot_query, (limit,))
            results = cursor.fetchall()
            
            # Convert datetime objects to strings
            for row in results:
                if row['last_accessed']:
                    row['last_accessed'] = row['last_accessed'].isoformat()
                else:
                    row['last_accessed'] = ''
            
            return results
            
        except Exception as e:
            logger.error(f"Get hot contexts failed: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def update_chunk(self, chunk_id: str, new_content: str, session_id: str) -> str:
        """Buffer chunk modifications for review"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get current chunk for impact assessment
            current_query = "SELECT access_count FROM megamind_chunks WHERE chunk_id = %s"
            cursor.execute(current_query, (chunk_id,))
            current_chunk = cursor.fetchone()
            
            if not current_chunk:
                return ""
            
            # Calculate impact score based on access count
            access_count = current_chunk['access_count']
            impact_score = min(1.0, access_count / 100.0)  # Scale to 0-1
            
            # Create change record
            change_id = str(uuid.uuid4())
            change_data = {
                "original_content": None,  # Will be filled during commit
                "new_content": new_content,
                "modification_type": "content_update"
            }
            
            insert_query = """
            INSERT INTO megamind_session_changes 
            (change_id, session_id, change_type, chunk_id, change_data, impact_score)
            VALUES (%s, %s, 'update', %s, %s, %s)
            """
            cursor.execute(insert_query, (
                change_id, session_id, chunk_id, 
                json.dumps(change_data), impact_score
            ))
            
            # Update session metadata
            self._update_session_metadata(cursor, session_id)
            
            connection.commit()
            return change_id
            
        except Exception as e:
            logger.error(f"Update chunk failed: {e}")
            return ""
        finally:
            if connection:
                connection.close()
    
    def create_chunk(self, content: str, source_document: str, section_path: str, session_id: str) -> str:
        """Buffer new knowledge creation"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Generate new chunk ID
            new_chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"
            
            # Create change record
            change_id = str(uuid.uuid4())
            change_data = {
                "chunk_id": new_chunk_id,
                "content": content,
                "source_document": source_document,
                "section_path": section_path,
                "chunk_type": "section",  # Valid ENUM value: 'rule', 'function', 'section', 'example'
                "line_count": len(content.split('\n')),
                "token_count": len(content.split()) * 1.3  # Rough estimate
            }
            
            insert_query = """
            INSERT INTO megamind_session_changes 
            (change_id, session_id, change_type, change_data, impact_score)
            VALUES (%s, %s, 'create', %s, 0.5)
            """
            cursor.execute(insert_query, (
                change_id, session_id, json.dumps(change_data)
            ))
            
            # Update session metadata
            self._update_session_metadata(cursor, session_id)
            
            connection.commit()
            return change_id
            
        except Exception as e:
            logger.error(f"Create chunk failed: {e}")
            return ""
        finally:
            if connection:
                connection.close()
    
    def add_relationship(self, chunk_id_1: str, chunk_id_2: str, relationship_type: str, session_id: str) -> str:
        """Create cross-references"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Create change record
            change_id = str(uuid.uuid4())
            change_data = {
                "source_chunk_id": chunk_id_1,
                "target_chunk_id": chunk_id_2,
                "relationship_type": relationship_type,
                "strength": 0.8  # Default strength
            }
            
            insert_query = """
            INSERT INTO megamind_session_changes 
            (change_id, session_id, change_type, chunk_id, target_chunk_id, change_data, impact_score)
            VALUES (%s, %s, 'relate', %s, %s, %s, 0.3)
            """
            cursor.execute(insert_query, (
                change_id, session_id, chunk_id_1, chunk_id_2, json.dumps(change_data)
            ))
            
            # Update session metadata
            self._update_session_metadata(cursor, session_id)
            
            connection.commit()
            return change_id
            
        except Exception as e:
            logger.error(f"Add relationship failed: {e}")
            return ""
        finally:
            if connection:
                connection.close()
    
    def get_pending_changes(self, session_id: str) -> List[Dict[str, Any]]:
        """Get pending changes with smart highlighting"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            changes_query = """
            SELECT sc.change_id, sc.change_type, sc.chunk_id, sc.target_chunk_id,
                   sc.change_data, sc.impact_score, sc.timestamp,
                   c.source_document, c.access_count
            FROM megamind_session_changes sc
            LEFT JOIN megamind_chunks c ON sc.chunk_id = c.chunk_id
            WHERE sc.session_id = %s AND sc.status = 'pending'
            ORDER BY sc.impact_score DESC, sc.timestamp ASC
            """
            cursor.execute(changes_query, (session_id,))
            changes = cursor.fetchall()
            
            # Add smart highlighting
            for change in changes:
                change['timestamp'] = change['timestamp'].isoformat()
                change['change_data'] = json.loads(change['change_data'])
                
                # Convert Decimal to float for JSON serialization
                change['impact_score'] = float(change['impact_score']) if change['impact_score'] else 0.0
                
                # Assign priority based on impact score and access count
                impact = change['impact_score']
                access_count = change.get('access_count', 0) or 0
                
                if impact >= 0.7 or access_count >= 50:
                    change['priority'] = 'CRITICAL'
                    change['priority_emoji'] = '🔴'
                elif impact >= 0.4 or access_count >= 20:
                    change['priority'] = 'IMPORTANT'
                    change['priority_emoji'] = '🟡'
                else:
                    change['priority'] = 'STANDARD'
                    change['priority_emoji'] = '🟢'
            
            return changes
            
        except Exception as e:
            logger.error(f"Get pending changes failed: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def commit_session_changes(self, session_id: str, approved_changes: List[str]) -> Dict[str, Any]:
        """Commit approved changes and track contributions"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            stats = {"chunks_modified": 0, "chunks_created": 0, "relationships_added": 0}
            rollback_data = []
            
            for change_id in approved_changes:
                # Get change details
                change_query = """
                SELECT change_type, chunk_id, target_chunk_id, change_data
                FROM megamind_session_changes
                WHERE change_id = %s AND session_id = %s AND status = 'pending'
                """
                cursor.execute(change_query, (change_id, session_id))
                change = cursor.fetchone()
                
                if not change:
                    continue
                
                change_data = json.loads(change['change_data'])
                
                if change['change_type'] == 'update':
                    # Store original for rollback
                    orig_query = "SELECT content FROM megamind_chunks WHERE chunk_id = %s"
                    cursor.execute(orig_query, (change['chunk_id'],))
                    original = cursor.fetchone()
                    
                    if original:
                        rollback_data.append({
                            "type": "update",
                            "chunk_id": change['chunk_id'],
                            "original_content": original['content']
                        })
                        
                        # Apply update
                        update_query = "UPDATE megamind_chunks SET content = %s WHERE chunk_id = %s"
                        cursor.execute(update_query, (change_data['new_content'], change['chunk_id']))
                        stats["chunks_modified"] += 1
                
                elif change['change_type'] == 'create':
                    # Create new chunk
                    insert_query = """
                    INSERT INTO megamind_chunks 
                    (chunk_id, content, source_document, section_path, chunk_type, line_count, token_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, (
                        change_data['chunk_id'], change_data['content'],
                        change_data['source_document'], change_data['section_path'],
                        change_data['chunk_type'], change_data['line_count'], change_data['token_count']
                    ))
                    stats["chunks_created"] += 1
                    
                    rollback_data.append({
                        "type": "create",
                        "chunk_id": change_data['chunk_id']
                    })
                
                elif change['change_type'] == 'relate':
                    # Create relationship
                    rel_insert = """
                    INSERT IGNORE INTO megamind_chunk_relationships 
                    (chunk_id, related_chunk_id, relationship_type, strength)
                    VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(rel_insert, (
                        change_data['source_chunk_id'], change_data['target_chunk_id'],
                        change_data['relationship_type'], change_data['strength']
                    ))
                    stats["relationships_added"] += 1
                
                # Mark change as approved
                approve_query = "UPDATE megamind_session_changes SET status = 'approved' WHERE change_id = %s"
                cursor.execute(approve_query, (change_id,))
            
            # Create contribution record
            contribution_id = str(uuid.uuid4())
            contrib_query = """
            INSERT INTO megamind_knowledge_contributions
            (contribution_id, session_id, chunks_modified, chunks_created, relationships_added, rollback_data)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(contrib_query, (
                contribution_id, session_id, stats["chunks_modified"],
                stats["chunks_created"], stats["relationships_added"], json.dumps(rollback_data)
            ))
            
            # Update session metadata
            reset_query = """
            UPDATE megamind_session_metadata 
            SET pending_changes_count = 0, last_activity = NOW()
            WHERE session_id = %s
            """
            cursor.execute(reset_query, (session_id,))
            
            connection.commit()
            
            return {
                "contribution_id": contribution_id,
                "changes_applied": len(approved_changes),
                **stats
            }
            
        except Exception as e:
            logger.error(f"Commit session changes failed: {e}")
            if connection:
                connection.rollback()
            return {"error": str(e)}
        finally:
            if connection:
                connection.close()
    
    def _update_session_metadata(self, cursor, session_id: str):
        """Helper to update session metadata"""
        # Ensure session exists
        session_check = "SELECT session_id FROM megamind_session_metadata WHERE session_id = %s"
        cursor.execute(session_check, (session_id,))
        
        if not cursor.fetchone():
            # Create new session
            create_session = """
            INSERT INTO megamind_session_metadata (session_id, user_context, project_context)
            VALUES (%s, 'development', 'megamind_system')
            """
            cursor.execute(create_session, (session_id,))
        
        # Update pending changes count
        update_session = """
        UPDATE megamind_session_metadata 
        SET pending_changes_count = (
            SELECT COUNT(*) FROM megamind_session_changes 
            WHERE session_id = %s AND status = 'pending'
        ),
        last_activity = NOW()
        WHERE session_id = %s
        """
        cursor.execute(update_session, (session_id, session_id))

def load_config():
    """Load configuration from environment variables"""
    return {
        'host': os.getenv('MEGAMIND_DB_HOST', '10.255.250.22'),
        'port': os.getenv('MEGAMIND_DB_PORT', '3309'),
        'database': os.getenv('MEGAMIND_DB_NAME', 'megamind_database'),
        'user': os.getenv('MEGAMIND_DB_USER', 'megamind_user'),
        'password': os.getenv('MEGAMIND_DB_PASSWORD', ''),
        'pool_size': os.getenv('CONNECTION_POOL_SIZE', '10')
    }

class MCPServer:
    """MCP Server implementation for MegaMind Context Database"""
    
    def __init__(self, db_manager: MegaMindDatabase):
        self.db_manager = db_manager
        self.request_id = 0
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP requests"""
        try:
            method = request.get('method', '')
            params = request.get('params', {})
            request_id = request.get('id')
            
            if method == 'initialize':
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "megamind-database",
                            "version": "1.0.0"
                        }
                    }
                }
            
            elif method == 'tools/list':
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "mcp__context_db__search_chunks",
                                "description": "Search context chunks with optional model optimization",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "Search query"},
                                        "limit": {"type": "integer", "default": 10, "description": "Maximum results"},
                                        "model_type": {"type": "string", "default": "sonnet", "description": "Model optimization"}
                                    },
                                    "required": ["query"]
                                }
                            },
                            {
                                "name": "mcp__context_db__get_chunk",
                                "description": "Get specific chunk by ID with relationships",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "chunk_id": {"type": "string", "description": "Chunk identifier"},
                                        "include_relationships": {"type": "boolean", "default": True, "description": "Include relationships"}
                                    },
                                    "required": ["chunk_id"]
                                }
                            },
                            {
                                "name": "mcp__context_db__get_related_chunks",
                                "description": "Get chunks related to specified chunk",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "chunk_id": {"type": "string", "description": "Source chunk identifier"},
                                        "max_depth": {"type": "integer", "default": 2, "description": "Maximum relationship depth"}
                                    },
                                    "required": ["chunk_id"]
                                }
                            },
                            {
                                "name": "mcp__context_db__get_session_primer",
                                "description": "Generate lightweight context for session continuity",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "last_session_data": {"type": "string", "description": "Previous session data (optional)"}
                                    },
                                    "required": []
                                }
                            },
                            {
                                "name": "mcp__context_db__track_access",
                                "description": "Update access analytics for optimization",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "chunk_id": {"type": "string", "description": "Chunk identifier"},
                                        "query_context": {"type": "string", "default": "", "description": "Query context (optional)"}
                                    },
                                    "required": ["chunk_id"]
                                }
                            },
                            {
                                "name": "mcp__context_db__get_hot_contexts",
                                "description": "Get frequently accessed chunks prioritized by usage patterns",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "model_type": {"type": "string", "default": "sonnet", "description": "Model optimization"},
                                        "limit": {"type": "integer", "default": 20, "description": "Maximum results"}
                                    },
                                    "required": []
                                }
                            },
                            {
                                "name": "mcp__context_db__update_chunk",
                                "description": "Buffer chunk modifications for review",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "chunk_id": {"type": "string", "description": "Chunk identifier"},
                                        "new_content": {"type": "string", "description": "Updated content"},
                                        "session_id": {"type": "string", "description": "Session identifier"}
                                    },
                                    "required": ["chunk_id", "new_content", "session_id"]
                                }
                            },
                            {
                                "name": "mcp__context_db__create_chunk",
                                "description": "Buffer new knowledge creation",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string", "description": "Chunk content"},
                                        "source_document": {"type": "string", "description": "Source document"},
                                        "section_path": {"type": "string", "description": "Section path"},
                                        "session_id": {"type": "string", "description": "Session identifier"}
                                    },
                                    "required": ["content", "source_document", "section_path", "session_id"]
                                }
                            },
                            {
                                "name": "mcp__context_db__add_relationship",
                                "description": "Create cross-references between chunks",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "chunk_id_1": {"type": "string", "description": "Source chunk identifier"},
                                        "chunk_id_2": {"type": "string", "description": "Target chunk identifier"},
                                        "relationship_type": {"type": "string", "description": "Relationship type"},
                                        "session_id": {"type": "string", "description": "Session identifier"}
                                    },
                                    "required": ["chunk_id_1", "chunk_id_2", "relationship_type", "session_id"]
                                }
                            },
                            {
                                "name": "mcp__context_db__get_pending_changes",
                                "description": "Get pending changes with smart highlighting",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "session_id": {"type": "string", "description": "Session identifier"}
                                    },
                                    "required": ["session_id"]
                                }
                            },
                            {
                                "name": "mcp__context_db__commit_session_changes",
                                "description": "Commit approved changes and track contributions",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "session_id": {"type": "string", "description": "Session identifier"},
                                        "approved_changes": {"type": "array", "items": {"type": "string"}, "description": "List of approved change IDs"}
                                    },
                                    "required": ["session_id", "approved_changes"]
                                }
                            }
                        ]
                    }
                }
            
            elif method == 'tools/call':
                tool_name = params.get('name', '')
                tool_args = params.get('arguments', {})
                
                if tool_name == 'mcp__context_db__search_chunks':
                    results = self.db_manager.search_chunks(
                        query=tool_args.get('query', ''),
                        limit=tool_args.get('limit', 10)
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(results, indent=2)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__context_db__get_chunk':
                    result = self.db_manager.get_chunk(
                        chunk_id=tool_args.get('chunk_id', ''),
                        include_relationships=tool_args.get('include_relationships', True)
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(result, indent=2) if result else "Chunk not found"
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__context_db__get_related_chunks':
                    results = self.db_manager.get_related_chunks(
                        chunk_id=tool_args.get('chunk_id', ''),
                        max_depth=tool_args.get('max_depth', 2)
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(results, indent=2)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__context_db__get_session_primer':
                    result = self.db_manager.get_session_primer(
                        last_session_data=tool_args.get('last_session_data')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(result, indent=2)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__context_db__track_access':
                    success = self.db_manager.track_access(
                        chunk_id=tool_args.get('chunk_id', ''),
                        query_context=tool_args.get('query_context', '')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps({"success": success}, indent=2)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__context_db__get_hot_contexts':
                    results = self.db_manager.get_hot_contexts(
                        model_type=tool_args.get('model_type', 'sonnet'),
                        limit=tool_args.get('limit', 20)
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(results, indent=2)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__context_db__update_chunk':
                    change_id = self.db_manager.update_chunk(
                        chunk_id=tool_args.get('chunk_id', ''),
                        new_content=tool_args.get('new_content', ''),
                        session_id=tool_args.get('session_id', '')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps({"change_id": change_id}, indent=2)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__context_db__create_chunk':
                    change_id = self.db_manager.create_chunk(
                        content=tool_args.get('content', ''),
                        source_document=tool_args.get('source_document', ''),
                        section_path=tool_args.get('section_path', ''),
                        session_id=tool_args.get('session_id', '')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps({"change_id": change_id}, indent=2)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__context_db__add_relationship':
                    change_id = self.db_manager.add_relationship(
                        chunk_id_1=tool_args.get('chunk_id_1', ''),
                        chunk_id_2=tool_args.get('chunk_id_2', ''),
                        relationship_type=tool_args.get('relationship_type', ''),
                        session_id=tool_args.get('session_id', '')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps({"change_id": change_id}, indent=2)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__context_db__get_pending_changes':
                    results = self.db_manager.get_pending_changes(
                        session_id=tool_args.get('session_id', '')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(results, indent=2)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__context_db__commit_session_changes':
                    result = self.db_manager.commit_session_changes(
                        session_id=tool_args.get('session_id', ''),
                        approved_changes=tool_args.get('approved_changes', [])
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(result, indent=2)
                                }
                            ]
                        }
                    }
                
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Unknown tool: {tool_name}"
                        }
                    }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown method: {method}"
                    }
                }
        
        except Exception as e:
            logger.error(f"Request handling error: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request.get('id'),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    async def run(self):
        """Run the MCP server on stdin/stdout"""
        logger.info("MCP Server starting on stdin/stdout...")
        
        try:
            while True:
                # Read JSON-RPC request from stdin
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    break
                
                try:
                    request = json.loads(line.strip())
                    response = await self.handle_request(request)
                    
                    # Write JSON-RPC response to stdout
                    print(json.dumps(response), flush=True)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        }
                    }
                    print(json.dumps(error_response), flush=True)
        
        except Exception as e:
            logger.error(f"Server error: {e}")
        
        logger.info("MCP Server shutting down")

async def main():
    """Main entry point for the MCP server"""
    try:
        # Load configuration
        db_config = load_config()
        
        if not db_config['password']:
            logger.error("Database password not configured. Set MEGAMIND_DB_PASSWORD environment variable.")
            return 1
        
        # Initialize database
        db_manager = MegaMindDatabase(db_config)
        
        # Test database connection
        test_results = db_manager.search_chunks("test", limit=1)
        logger.info(f"Database connection successful. Found {len(test_results)} test results.")
        
        # Start MCP server
        mcp_server = MCPServer(db_manager)
        await mcp_server.run()
            
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))