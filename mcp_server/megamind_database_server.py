#!/usr/bin/env python3
"""
MegaMind Context Database MCP Server
Realm-aware MCP protocol implementation with semantic search
"""

import json
import logging
import os
import sys
import asyncio
import uuid
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Optional, Any, Union

import mysql.connector
from mysql.connector import pooling

# Import Phase 2 components
try:
    from .session_manager import SessionManager
    from .enhanced_embedding_functions import EnhancedEmbeddingFunctions
except ImportError:
    from session_manager import SessionManager
    from enhanced_embedding_functions import EnhancedEmbeddingFunctions

class MegaMindJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles MySQL data types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):  # Other datetime-like objects
            return obj.isoformat()
        return super().default(obj)

def clean_decimal_objects(data):
    """Recursively clean Decimal objects from nested data structures"""
    from decimal import Decimal
    
    if isinstance(data, dict):
        return {key: clean_decimal_objects(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_decimal_objects(item) for item in data]
    elif isinstance(data, Decimal):
        return float(data)
    elif hasattr(data, 'isoformat'):  # datetime objects
        return data.isoformat()
    else:
        return data

# Import realm-aware database implementation
try:
    from .realm_aware_database import RealmAwareMegaMindDatabase
except ImportError:
    from realm_aware_database import RealmAwareMegaMindDatabase

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
                'host': self.config.get('db_host', self.config.get('host')),
                'port': int(self.config.get('db_port', self.config.get('port'))),
                'database': self.config.get('db_database', self.config.get('database')),
                'user': self.config.get('db_user', self.config.get('user')),
                'password': self.config.get('db_password', self.config.get('password')),
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
        """Search chunks using multi-word text search with improved logic"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Clean and split query into words
            query = query.strip()
            if not query:
                return []
            
            # For single word, use simple search
            if ' ' not in query:
                search_query = """
                SELECT chunk_id, content, source_document, section_path, chunk_type,
                       line_count, token_count, access_count, last_accessed
                FROM megamind_chunks
                WHERE LOWER(content) LIKE LOWER(%s) 
                   OR LOWER(source_document) LIKE LOWER(%s) 
                   OR LOWER(section_path) LIKE LOWER(%s)
                ORDER BY access_count DESC
                LIMIT %s
                """
                like_pattern = f"%{query}%"
                cursor.execute(search_query, (like_pattern, like_pattern, like_pattern, limit))
            else:
                # Multi-word search: all words must match somewhere in content/document/path
                words = [word.strip().lower() for word in query.split() if word.strip()]
                
                # Build a more reliable query using multiple LIKE conditions
                search_query = """
                SELECT chunk_id, content, source_document, section_path, chunk_type,
                       line_count, token_count, access_count, last_accessed
                FROM megamind_chunks
                WHERE """ + " AND ".join([
                    "(LOWER(content) LIKE %s OR LOWER(source_document) LIKE %s OR LOWER(section_path) LIKE %s)"
                    for _ in words
                ]) + """
                ORDER BY access_count DESC
                LIMIT %s
                """
                
                # Build parameters list
                params = []
                for word in words:
                    like_pattern = f"%{word}%"
                    params.extend([like_pattern, like_pattern, like_pattern])
                params.append(limit)
                
                cursor.execute(search_query, params)
            
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
            
            # Auto-track access when chunk is retrieved
            try:
                track_query = """
                UPDATE megamind_chunks
                SET access_count = access_count + 1,
                    last_accessed = NOW()
                WHERE chunk_id = %s
                """
                cursor.execute(track_query, (chunk_id,))
                connection.commit()
                
                # Update the chunk data with new access count
                chunk['access_count'] += 1
            except Exception as e:
                logger.warning(f"Failed to track access for {chunk_id}: {e}")
            
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
                WHERE access_count >= 2
                ORDER BY access_count DESC, token_count ASC
                LIMIT %s
                """
            else:  # Sonnet and others
                hot_query = """
                SELECT chunk_id, content, source_document, section_path,
                       chunk_type, access_count, last_accessed
                FROM megamind_chunks
                WHERE access_count >= 1
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
                    # Create new chunk with access_count = 1 (creation counts as first access)
                    insert_query = """
                    INSERT INTO megamind_chunks 
                    (chunk_id, content, source_document, section_path, chunk_type, line_count, token_count, access_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 1)
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

def setup_environment_paths():
    """Setup environment paths from MEGAMIND_ROOT with intelligent defaults"""
    # Get root path from environment or use intelligent default
    root = os.getenv('MEGAMIND_ROOT')
    
    if not root:
        # Try to determine root from current script location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # If we're in mcp_server directory, go up one level
        if current_dir.endswith('mcp_server'):
            root = os.path.dirname(current_dir)
        else:
            root = current_dir
        logger.info(f"MEGAMIND_ROOT not set, using inferred root: {root}")
    else:
        logger.info(f"Using MEGAMIND_ROOT: {root}")
    
    # Setup derived paths
    models_path = os.path.join(root, 'models')
    server_path = os.path.join(root, 'mcp_server')
    
    # Set model cache environment variables
    os.environ['HF_HOME'] = models_path
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = models_path
    os.environ['TRANSFORMERS_CACHE'] = models_path
    
    # Set Python path for imports
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if server_path not in current_pythonpath:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = f"{server_path}:{current_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = server_path
    
    # Create models directory if it doesn't exist
    os.makedirs(models_path, exist_ok=True)
    
    logger.info(f"Environment paths configured:")
    logger.info(f"  Models cache: {models_path}")
    logger.info(f"  Python path: {server_path}")
    logger.info(f"  HF_HOME: {os.environ['HF_HOME']}")
    
    return {
        'root': root,
        'models_path': models_path,
        'server_path': server_path
    }

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

class DatabaseConnectionAdapter:
    """Adapter to provide async context manager interface for SessionManager"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.project_realm = getattr(db_manager, 'project_realm', 'PROJECT')
    
    def connection(self):
        """Return async context manager for database connections"""
        return AsyncConnectionWrapper(self.db_manager)

class AsyncConnectionWrapper:
    """Async context manager wrapper for synchronous database connections"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.connection = None
        self._cursor = None
    
    async def __aenter__(self):
        """Enter async context - get connection"""
        self.connection = self.db_manager.get_connection()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context - close connection"""
        if self._cursor:
            self._cursor.close()
        if self.connection:
            self.connection.close()
    
    async def execute(self, query, params=None):
        """Execute query with cursor"""
        if not self.connection:
            raise RuntimeError("Connection not established")
        
        if not self._cursor:
            self._cursor = self.connection.cursor(dictionary=True)
        
        self._cursor.execute(query, params or ())
        return self._cursor
    
    async def fetchone(self):
        """Fetch one row"""
        if self._cursor:
            return self._cursor.fetchone()
        return None
    
    async def fetchall(self):
        """Fetch all rows"""
        if self._cursor:
            return self._cursor.fetchall()
        return []
    
    async def commit(self):
        """Commit transaction"""
        if self.connection:
            self.connection.commit()
    
    async def rollback(self):
        """Rollback transaction"""
        if self.connection:
            self.connection.rollback()

class MCPServer:
    """MCP Server implementation for MegaMind Context Database"""
    
    def __init__(self, db_manager: MegaMindDatabase):
        self.db_manager = db_manager
        self.request_id = 0
        self.default_realm = os.getenv('MEGAMIND_PROJECT_REALM', 'PROJECT')
        
        # Initialize Phase 2 components
        self.session_manager = None
        self.enhanced_functions = None
        self._init_phase2_components()
        
        # Initialize Phase 3 components
        self.phase3_functions = None
        self._init_phase3_components()
        
        # Initialize Phase 4 components
        self.phase4_functions = None
        self._init_phase4_components()
    
    def _init_phase2_components(self):
        """Initialize Phase 2 components for enhanced embedding functions"""
        try:
            # Create database adapter for async compatibility
            db_adapter = DatabaseConnectionAdapter(self.db_manager)
            
            # Initialize session manager with adapter
            self.session_manager = SessionManager(db_adapter)
            logger.info("Session manager initialized")
            
            # Initialize enhanced embedding functions
            self.enhanced_functions = EnhancedEmbeddingFunctions(
                self.db_manager, 
                self.session_manager
            )
            logger.info("Enhanced embedding functions initialized")
            
        except Exception as e:
            logger.warning(f"Phase 2 components initialization failed: {e}")
            logger.warning("Enhanced embedding functions will not be available")
            self.session_manager = None
            self.enhanced_functions = None
    
    def _init_phase3_components(self):
        """Initialize Phase 3 components for knowledge management and session tracking"""
        try:
            # Import Phase 3 functions
            from phase3_functions import Phase3Functions
            
            # Initialize Phase 3 functions
            self.phase3_functions = Phase3Functions(self.db_manager)
            logger.info("Phase 3 knowledge management and session tracking functions initialized")
            
        except Exception as e:
            logger.warning(f"Phase 3 components initialization failed: {e}")
            logger.warning("Knowledge management functions will not be available")
            self.phase3_functions = None
    
    def _init_phase4_components(self):
        """Initialize Phase 4 components for AI enhancement"""
        try:
            # Import Phase 4 functions
            from phase4_functions import Phase4Functions
            
            # Initialize Phase 4 functions with database manager
            self.phase4_functions = Phase4Functions(self.db_manager)
            logger.info("Phase 4 AI enhancement functions initialized")
            
        except Exception as e:
            logger.warning(f"Phase 4 components initialization failed: {e}")
            logger.warning("AI enhancement functions will not be available")
            self.phase4_functions = None
    
    def get_tools_list(self) -> List[Dict[str, Any]]:
        """Get the complete list of MCP tools for both initialize and tools/list responses"""
        return [
            {
                "name": "mcp__megamind__search_chunks",
                "description": "Enhanced dual-realm search with hybrid semantic capabilities",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "default": 10, "description": "Maximum results"},
                        "search_type": {"type": "string", "default": "hybrid", "description": "Search type: semantic, keyword, or hybrid"},
                        "realm_id": {"type": "string", "description": "Target realm identifier (optional, defaults to server realm)"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "mcp__megamind__get_chunk",
                "description": "Get specific chunk by ID with relationships",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "Chunk identifier"},
                        "include_relationships": {"type": "boolean", "default": True, "description": "Include relationships"},
                        "realm_id": {"type": "string", "description": "Target realm identifier (optional, defaults to server realm)"}
                    },
                    "required": ["chunk_id"]
                }
            },
            {
                "name": "mcp__megamind__get_related_chunks",
                "description": "Get chunks related to specified chunk",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "Source chunk identifier"},
                        "max_depth": {"type": "integer", "default": 2, "description": "Maximum relationship depth"},
                        "realm_id": {"type": "string", "description": "Target realm identifier (optional, defaults to server realm)"}
                    },
                    "required": ["chunk_id"]
                }
            },
            {
                "name": "mcp__megamind__get_session_primer",
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
                "name": "mcp__megamind__track_access",
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
                "name": "mcp__megamind__get_hot_contexts",
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
                "name": "mcp__megamind__update_chunk",
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
                "name": "mcp__megamind__create_chunk",
                "description": "Buffer new knowledge creation with realm targeting and embedding generation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Chunk content"},
                        "source_document": {"type": "string", "description": "Source document"},
                        "section_path": {"type": "string", "description": "Section path"},
                        "session_id": {"type": "string", "description": "Session identifier"},
                        "target_realm": {"type": "string", "description": "Target realm (optional, defaults to PROJECT)"}
                    },
                    "required": ["content", "source_document", "section_path", "session_id"]
                }
            },
            {
                "name": "mcp__megamind__add_relationship",
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
                "name": "mcp__megamind__get_pending_changes",
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
                "name": "mcp__megamind__commit_session_changes",
                "description": "Commit approved changes and track contributions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session identifier"},
                        "approved_changes": {"type": "array", "items": {"type": "string"}, "description": "List of approved change IDs"}
                    },
                    "required": ["session_id", "approved_changes"]
                }
            },
            {
                "name": "mcp__megamind__search_chunks_semantic",
                "description": "Pure semantic search across Global + Project realms",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "default": 10, "description": "Maximum results"},
                        "threshold": {"type": "number", "default": 0.7, "description": "Minimum similarity threshold"},
                        "realm_id": {"type": "string", "description": "Target realm identifier (optional, defaults to server realm)"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "mcp__megamind__search_chunks_by_similarity",
                "description": "Find chunks similar to a reference chunk using embeddings",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "reference_chunk_id": {"type": "string", "description": "Reference chunk identifier"},
                        "limit": {"type": "integer", "default": 10, "description": "Maximum results"},
                        "threshold": {"type": "number", "default": 0.7, "description": "Minimum similarity threshold"},
                        "realm_id": {"type": "string", "description": "Target realm identifier (optional, defaults to server realm)"}
                    },
                    "required": ["reference_chunk_id"]
                }
            },
            {
                "name": "mcp__megamind__batch_generate_embeddings",
                "description": "Generate embeddings for existing chunks in batch",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_ids": {"type": "array", "items": {"type": "string"}, "description": "List of chunk IDs (optional)"},
                        "realm_id": {"type": "string", "description": "Realm ID to process (optional)"}
                    },
                    "required": []
                }
            },
            {
                "name": "mcp__megamind__create_promotion_request",
                "description": "Create a promotion request to move knowledge between realms",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source_chunk_id": {"type": "string", "description": "Source chunk identifier"},
                        "target_realm_id": {"type": "string", "description": "Target realm identifier"},
                        "promotion_type": {"type": "string", "enum": ["copy", "move", "reference"], "default": "copy", "description": "Type of promotion"},
                        "justification": {"type": "string", "description": "Justification for promotion"},
                        "business_impact": {"type": "string", "enum": ["low", "medium", "high", "critical"], "default": "medium", "description": "Business impact level"},
                        "requested_by": {"type": "string", "description": "User requesting promotion"},
                        "session_id": {"type": "string", "description": "Session identifier"}
                    },
                    "required": ["source_chunk_id", "target_realm_id", "justification", "requested_by", "session_id"]
                }
            },
            {
                "name": "mcp__megamind__get_promotion_requests",
                "description": "Get promotion requests with optional filtering",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["pending", "approved", "rejected", "processing", "completed"], "description": "Filter by status (optional)"},
                        "user_id": {"type": "string", "description": "Filter by user (optional)"},
                        "limit": {"type": "integer", "default": 50, "description": "Maximum results"}
                    },
                    "required": []
                }
            },
            {
                "name": "mcp__megamind__approve_promotion_request",
                "description": "Approve a pending promotion request",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "promotion_id": {"type": "string", "description": "Promotion request identifier"},
                        "reviewed_by": {"type": "string", "description": "User approving the request"},
                        "review_notes": {"type": "string", "description": "Review notes (optional)"}
                    },
                    "required": ["promotion_id", "reviewed_by"]
                }
            },
            {
                "name": "mcp__megamind__reject_promotion_request",
                "description": "Reject a pending promotion request",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "promotion_id": {"type": "string", "description": "Promotion request identifier"},
                        "reviewed_by": {"type": "string", "description": "User rejecting the request"},
                        "review_notes": {"type": "string", "description": "Reason for rejection"}
                    },
                    "required": ["promotion_id", "reviewed_by", "review_notes"]
                }
            },
            {
                "name": "mcp__megamind__get_promotion_impact",
                "description": "Get impact analysis for a promotion request",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "promotion_id": {"type": "string", "description": "Promotion request identifier"}
                    },
                    "required": ["promotion_id"]
                }
            },
            {
                "name": "mcp__megamind__get_promotion_queue_summary",
                "description": "Get summary of promotion queue status",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "realm_id": {"type": "string", "description": "Filter by realm (optional)"}
                    },
                    "required": []
                }
            },
            # Phase 2: Enhanced Embedding Functions
            {
                "name": "mcp__megamind__content_analyze_document",
                "description": "Analyze document structure and content with Phase 1 components",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Document content to analyze"},
                        "document_name": {"type": "string", "description": "Optional document name"},
                        "session_id": {"type": "string", "description": "Optional session ID for tracking"},
                        "metadata": {"type": "object", "description": "Optional metadata"}
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "mcp__megamind__content_create_chunks",
                "description": "Create optimized chunks from content using intelligent chunking",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to chunk"},
                        "document_name": {"type": "string", "description": "Optional document name"},
                        "session_id": {"type": "string", "description": "Optional session ID for tracking"},
                        "strategy": {"type": "string", "enum": ["semantic_aware", "markdown_structure", "hybrid"], "description": "Chunking strategy"},
                        "max_tokens": {"type": "integer", "description": "Maximum tokens per chunk"},
                        "min_tokens": {"type": "integer", "description": "Minimum tokens per chunk"},
                        "target_realm": {"type": "string", "description": "Target realm for chunks"}
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "mcp__megamind__content_assess_quality",
                "description": "Assess quality of chunks using 8-dimensional scoring",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_ids": {"type": "array", "items": {"type": "string"}, "description": "List of chunk IDs to assess"},
                        "session_id": {"type": "string", "description": "Optional session ID for tracking"},
                        "include_context": {"type": "boolean", "description": "Include surrounding chunks for context"}
                    },
                    "required": ["chunk_ids"]
                }
            },
            {
                "name": "mcp__megamind__content_optimize_embeddings",
                "description": "Optimize chunks for embedding generation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_ids": {"type": "array", "items": {"type": "string"}, "description": "List of chunk IDs to optimize"},
                        "session_id": {"type": "string", "description": "Optional session ID for tracking"},
                        "model": {"type": "string", "description": "Embedding model name"},
                        "cleaning_level": {"type": "string", "enum": ["minimal", "standard", "aggressive"], "description": "Text cleaning level"},
                        "batch_size": {"type": "integer", "description": "Batch size for processing"}
                    },
                    "required": ["chunk_ids"]
                }
            },
            {
                "name": "mcp__megamind__session_create",
                "description": "Create a new embedding session for tracking operations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_type": {"type": "string", "enum": ["analysis", "ingestion", "curation", "mixed"], "description": "Type of session"},
                        "created_by": {"type": "string", "description": "User creating the session"},
                        "description": {"type": "string", "description": "Optional session description"},
                        "metadata": {"type": "object", "description": "Optional session metadata"}
                    },
                    "required": ["session_type", "created_by"]
                }
            },
            {
                "name": "mcp__megamind__session_get_state",
                "description": "Get current session state and progress",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID"}
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "mcp__megamind__session_complete",
                "description": "Complete and finalize a session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID"}
                    },
                    "required": ["session_id"]
                }
            },
            # Phase 3: Knowledge Management Functions
            {
                "name": "mcp__megamind__knowledge_ingest_document",
                "description": "Ingest a knowledge document into the system",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_path": {"type": "string", "description": "Path to the document to ingest"},
                        "title": {"type": "string", "description": "Optional document title"},
                        "knowledge_type": {"type": "string", "description": "Type of knowledge (documentation, code_pattern, etc.)"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags for categorization"},
                        "session_id": {"type": "string", "description": "Optional session ID for tracking"}
                    },
                    "required": ["document_path"]
                }
            },
            {
                "name": "mcp__megamind__knowledge_discover_relationships",
                "description": "Discover relationships between knowledge chunks",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional list of chunk IDs to analyze"},
                        "similarity_threshold": {"type": "number", "default": 0.7, "description": "Threshold for semantic similarity"},
                        "session_id": {"type": "string", "description": "Optional session ID for tracking"}
                    },
                    "required": []
                }
            },
            {
                "name": "mcp__megamind__knowledge_optimize_retrieval",
                "description": "Optimize knowledge retrieval based on usage patterns",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "usage_window_days": {"type": "integer", "default": 7, "description": "Days of usage data to analyze"},
                        "session_id": {"type": "string", "description": "Optional session ID for tracking"}
                    },
                    "required": []
                }
            },
            {
                "name": "mcp__megamind__knowledge_get_related",
                "description": "Get chunks related to a given chunk",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "Source chunk ID"},
                        "relationship_types": {"type": "array", "items": {"type": "string"}, "description": "Optional filter for relationship types"},
                        "max_depth": {"type": "integer", "default": 2, "description": "Maximum depth to traverse"},
                        "session_id": {"type": "string", "description": "Optional session ID for tracking"}
                    },
                    "required": ["chunk_id"]
                }
            },
            # Phase 3: Session Tracking Functions
            {
                "name": "mcp__megamind__session_create_operational",
                "description": "Create a new operational tracking session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_type": {"type": "string", "default": "general", "description": "Type of session"},
                        "user_id": {"type": "string", "description": "Optional user identifier"},
                        "description": {"type": "string", "description": "Optional session description"}
                    },
                    "required": []
                }
            },
            {
                "name": "mcp__megamind__session_track_action",
                "description": "Track an action in the operational session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session identifier"},
                        "action_type": {"type": "string", "description": "Type of action"},
                        "description": {"type": "string", "description": "Human-readable description"},
                        "details": {"type": "object", "description": "Optional action details"},
                        "priority": {"type": "string", "description": "Optional priority level"}
                    },
                    "required": ["session_id", "action_type", "description"]
                }
            },
            {
                "name": "mcp__megamind__session_get_recap",
                "description": "Get a recap of the operational session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session identifier"}
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "mcp__megamind__session_prime_context",
                "description": "Prime context for session resumption",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session identifier"}
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "mcp__megamind__session_list_recent",
                "description": "List recent operational sessions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "Optional filter by user"},
                        "limit": {"type": "integer", "default": 10, "description": "Maximum number of sessions to return"}
                    },
                    "required": []
                }
            },
            {
                "name": "mcp__megamind__session_close",
                "description": "Close an operational session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session identifier"}
                    },
                    "required": ["session_id"]
                }
            },
            # Phase 4: AI Enhancement Functions
            {
                "name": "mcp__megamind__ai_improve_chunk_quality",
                "description": "Analyze chunk quality and suggest/apply improvements",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "ID of chunk to improve"},
                        "session_id": {"type": "string", "description": "Current session ID"},
                        "apply_automated": {"type": "boolean", "default": False, "description": "Whether to automatically apply improvements"}
                    },
                    "required": ["chunk_id", "session_id"]
                }
            },
            {
                "name": "mcp__megamind__ai_record_user_feedback",
                "description": "Record user feedback and trigger adaptive learning",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "feedback_type": {"type": "string", "enum": ["chunk_quality", "boundary_accuracy", "retrieval_success", "manual_correction"], "description": "Type of feedback"},
                        "target_id": {"type": "string", "description": "ID of target (chunk, document, etc.)"},
                        "rating": {"type": "number", "minimum": 0, "maximum": 1, "description": "Rating from 0.0 to 1.0"},
                        "details": {"type": "object", "description": "Additional feedback details"},
                        "user_id": {"type": "string", "description": "User providing feedback"},
                        "session_id": {"type": "string", "description": "Current session ID"}
                    },
                    "required": ["feedback_type", "target_id", "rating", "user_id", "session_id"]
                }
            },
            {
                "name": "mcp__megamind__ai_get_adaptive_strategy",
                "description": "Get current adaptive strategy based on learned patterns",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "context": {"type": "object", "description": "Context information for strategy selection"},
                        "session_id": {"type": "string", "description": "Current session ID"}
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "mcp__megamind__ai_curate_chunks",
                "description": "Run automated curation workflow on chunks",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_ids": {"type": "array", "items": {"type": "string"}, "description": "List of chunk IDs to curate"},
                        "workflow_id": {"type": "string", "default": "standard_quality", "description": "Curation workflow to use"},
                        "session_id": {"type": "string", "description": "Current session ID"},
                        "auto_apply": {"type": "boolean", "default": False, "description": "Whether to automatically apply decisions"}
                    },
                    "required": ["chunk_ids", "session_id"]
                }
            },
            {
                "name": "mcp__megamind__ai_optimize_performance",
                "description": "Optimize system performance based on usage patterns",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "optimization_type": {"type": "string", "enum": ["batch_size", "cache_strategy", "model_selection", "preprocessing"], "description": "Type of optimization"},
                        "parameters": {"type": "object", "description": "Parameters for optimization"},
                        "session_id": {"type": "string", "description": "Current session ID"},
                        "apply": {"type": "boolean", "default": False, "description": "Whether to apply the optimization"}
                    },
                    "required": ["optimization_type", "parameters", "session_id"]
                }
            },
            {
                "name": "mcp__megamind__ai_get_performance_insights",
                "description": "Get current performance insights and recommendations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Current session ID"}
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "mcp__megamind__ai_generate_enhancement_report",
                "description": "Generate comprehensive AI enhancement report",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "report_type": {"type": "string", "enum": ["quality", "learning", "curation", "optimization"], "description": "Type of report"},
                        "start_date": {"type": "string", "format": "date-time", "description": "Report period start"},
                        "end_date": {"type": "string", "format": "date-time", "description": "Report period end"},
                        "session_id": {"type": "string", "description": "Current session ID"}
                    },
                    "required": ["report_type", "start_date", "end_date", "session_id"]
                }
            },
            
            # GitHub Issue #26: Chunk Approval Functions
            {
                "name": "mcp__megamind__get_pending_chunks",
                "description": "Get all pending chunks across the system",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 20, "description": "Maximum number of chunks to return"},
                        "realm_filter": {"type": "string", "description": "Optional realm filter (e.g., 'PROJECT', 'GLOBAL')"}
                    },
                    "required": []
                }
            },
            {
                "name": "mcp__megamind__approve_chunk",
                "description": "Approve a chunk by updating its approval status",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "Chunk ID to approve"},
                        "approved_by": {"type": "string", "description": "User performing the approval"},
                        "approval_notes": {"type": "string", "description": "Optional approval notes"}
                    },
                    "required": ["chunk_id", "approved_by"]
                }
            },
            {
                "name": "mcp__megamind__reject_chunk",
                "description": "Reject a chunk by updating its approval status",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "Chunk ID to reject"},
                        "rejected_by": {"type": "string", "description": "User performing the rejection"},
                        "rejection_reason": {"type": "string", "description": "Reason for rejection"}
                    },
                    "required": ["chunk_id", "rejected_by", "rejection_reason"]
                }
            },
            {
                "name": "mcp__megamind__bulk_approve_chunks",
                "description": "Approve multiple chunks in bulk",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_ids": {"type": "array", "items": {"type": "string"}, "description": "List of chunk IDs to approve"},
                        "approved_by": {"type": "string", "description": "User performing the bulk approval"}
                    },
                    "required": ["chunk_ids", "approved_by"]
                }
            }
        ]
    
    def extract_realm_from_arguments(self, tool_args: Dict[str, Any]) -> Optional[str]:
        """Extract realm_id from tool arguments for future realm-aware operations"""
        realm_id = tool_args.get('realm_id')
        if realm_id:
            logger.debug(f"Extracted realm_id: {realm_id}")
            return realm_id
        return None
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP requests"""
        try:
            method = request.get('method', '')
            params = request.get('params', {})
            request_id = request.get('id')
            
            if method == 'initialize':
                tools_list = self.get_tools_list()
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {tool["name"]: tool for tool in tools_list}
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
                        "tools": self.get_tools_list()
                    }
                }
            
            elif method == 'tools/call':
                tool_name = params.get('name', '')
                tool_args = params.get('arguments', {})
                
                if tool_name == 'mcp__megamind__search_chunks':
                    # Extract realm_id parameter (optional, for future use)
                    realm_id = self.extract_realm_from_arguments(tool_args)
                    # For now, use existing dual-realm search (Phase 1 compatibility)
                    results = self.db_manager.search_chunks_dual_realm(
                        query=tool_args.get('query', ''),
                        limit=tool_args.get('limit', 10),
                        search_type=tool_args.get('search_type', 'hybrid')
                    )
                    # Clean any remaining Decimal objects before JSON serialization
                    clean_results = clean_decimal_objects(results)
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_results, indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__get_chunk':
                    # Extract realm_id parameter (optional, for future use)
                    realm_id = self.extract_realm_from_arguments(tool_args)
                    # For now, use existing dual-realm search (Phase 1 compatibility)
                    result = self.db_manager.get_chunk_dual_realm(
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
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder) if result else "Chunk not found"
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__get_related_chunks':
                    # Extract realm_id parameter (optional, for future use)
                    realm_id = self.extract_realm_from_arguments(tool_args)
                    # For now, use existing method (Phase 1 compatibility)
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
                                    "text": json.dumps(clean_decimal_objects(results), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__get_session_primer':
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
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__track_access':
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
                                    "text": json.dumps({"success": success}, indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__get_hot_contexts':
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
                                    "text": json.dumps(clean_decimal_objects(results), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__update_chunk':
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
                                    "text": json.dumps({"change_id": change_id}, indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__create_chunk':
                    change_id = self.db_manager.create_chunk_with_target(
                        content=tool_args.get('content', ''),
                        source_document=tool_args.get('source_document', ''),
                        section_path=tool_args.get('section_path', ''),
                        session_id=tool_args.get('session_id', ''),
                        target_realm=tool_args.get('target_realm')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps({"change_id": change_id}, indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__add_relationship':
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
                                    "text": json.dumps({"change_id": change_id}, indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__get_pending_changes':
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
                                    "text": json.dumps(clean_decimal_objects(results), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__commit_session_changes':
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
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__search_chunks_semantic':
                    # Extract realm_id parameter (optional, for future use)
                    realm_id = self.extract_realm_from_arguments(tool_args)
                    # For now, use existing semantic search (Phase 1 compatibility)
                    results = self.db_manager.search_chunks_semantic(
                        query=tool_args.get('query', ''),
                        limit=tool_args.get('limit', 10),
                        threshold=tool_args.get('threshold', None)
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(results), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__search_chunks_by_similarity':
                    # Extract realm_id parameter (optional, for future use)
                    realm_id = self.extract_realm_from_arguments(tool_args)
                    # For now, use existing similarity search (Phase 1 compatibility)
                    results = self.db_manager.search_chunks_by_similarity(
                        reference_chunk_id=tool_args.get('reference_chunk_id', ''),
                        limit=tool_args.get('limit', 10),
                        threshold=tool_args.get('threshold', None)
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(results), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__batch_generate_embeddings':
                    result = self.db_manager.batch_generate_embeddings(
                        chunk_ids=tool_args.get('chunk_ids'),
                        realm_id=tool_args.get('realm_id')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__create_promotion_request':
                    promotion_id = self.db_manager.create_promotion_request(
                        source_chunk_id=tool_args.get('source_chunk_id'),
                        target_realm_id=tool_args.get('target_realm_id'),
                        promotion_type=tool_args.get('promotion_type', 'copy'),
                        justification=tool_args.get('justification'),
                        business_impact=tool_args.get('business_impact', 'medium'),
                        requested_by=tool_args.get('requested_by'),
                        session_id=tool_args.get('session_id')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps({"promotion_id": promotion_id}, indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__get_promotion_requests':
                    requests = self.db_manager.get_promotion_requests(
                        status=tool_args.get('status'),
                        user_id=tool_args.get('user_id'),
                        limit=tool_args.get('limit', 50)
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(requests), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__approve_promotion_request':
                    success = self.db_manager.approve_promotion_request(
                        promotion_id=tool_args.get('promotion_id'),
                        reviewed_by=tool_args.get('reviewed_by'),
                        review_notes=tool_args.get('review_notes', '')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps({"success": success}, indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__reject_promotion_request':
                    success = self.db_manager.reject_promotion_request(
                        promotion_id=tool_args.get('promotion_id'),
                        reviewed_by=tool_args.get('reviewed_by'),
                        review_notes=tool_args.get('review_notes')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps({"success": success}, indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__get_promotion_impact':
                    impact = self.db_manager.get_promotion_impact(
                        promotion_id=tool_args.get('promotion_id')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(impact), indent=2, cls=MegaMindJSONEncoder) if impact else "Impact analysis not found"
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__get_promotion_queue_summary':
                    summary = self.db_manager.get_promotion_queue_summary(
                        realm_id=tool_args.get('realm_id')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(summary), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                # Phase 2: Enhanced Embedding Functions
                elif tool_name == 'mcp__megamind__content_analyze_document':
                    if not self.enhanced_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Enhanced embedding functions not available"
                            }
                        }
                    
                    result = await self.enhanced_functions.content_analyze_document(
                        content=tool_args.get('content', ''),
                        document_name=tool_args.get('document_name'),
                        session_id=tool_args.get('session_id'),
                        metadata=tool_args.get('metadata')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__content_create_chunks':
                    if not self.enhanced_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Enhanced embedding functions not available"
                            }
                        }
                    
                    result = await self.enhanced_functions.content_create_chunks(
                        content=tool_args.get('content', ''),
                        document_name=tool_args.get('document_name'),
                        session_id=tool_args.get('session_id'),
                        strategy=tool_args.get('strategy'),
                        max_tokens=tool_args.get('max_tokens'),
                        min_tokens=tool_args.get('min_tokens'),
                        target_realm=tool_args.get('target_realm')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__content_assess_quality':
                    if not self.enhanced_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Enhanced embedding functions not available"
                            }
                        }
                    
                    result = await self.enhanced_functions.content_assess_quality(
                        chunk_ids=tool_args.get('chunk_ids', []),
                        session_id=tool_args.get('session_id'),
                        include_context=tool_args.get('include_context', False)
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__content_optimize_embeddings':
                    if not self.enhanced_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Enhanced embedding functions not available"
                            }
                        }
                    
                    result = await self.enhanced_functions.content_optimize_embeddings(
                        chunk_ids=tool_args.get('chunk_ids', []),
                        session_id=tool_args.get('session_id'),
                        model=tool_args.get('model'),
                        cleaning_level=tool_args.get('cleaning_level'),
                        batch_size=tool_args.get('batch_size')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__session_create':
                    if not self.enhanced_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Enhanced embedding functions not available"
                            }
                        }
                    
                    result = await self.enhanced_functions.session_create(
                        session_type=tool_args.get('session_type', 'mixed'),
                        created_by=tool_args.get('created_by', 'mcp_user'),
                        description=tool_args.get('description'),
                        metadata=tool_args.get('metadata')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__session_get_state':
                    if not self.enhanced_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Enhanced embedding functions not available"
                            }
                        }
                    
                    result = await self.enhanced_functions.session_get_state(
                        session_id=tool_args.get('session_id', '')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__session_complete':
                    if not self.enhanced_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Enhanced embedding functions not available"
                            }
                        }
                    
                    result = await self.enhanced_functions.session_complete(
                        session_id=tool_args.get('session_id', '')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                # Phase 3: Knowledge Management Functions
                elif tool_name == 'mcp__megamind__knowledge_ingest_document':
                    if not self.phase3_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Knowledge management functions not available"
                            }
                        }
                    
                    result = await self.phase3_functions.knowledge_ingest_document(
                        document_path=tool_args.get('document_path', ''),
                        title=tool_args.get('title'),
                        knowledge_type=tool_args.get('knowledge_type'),
                        tags=tool_args.get('tags'),
                        session_id=tool_args.get('session_id')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__knowledge_discover_relationships':
                    if not self.phase3_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Knowledge management functions not available"
                            }
                        }
                    
                    result = await self.phase3_functions.knowledge_discover_relationships(
                        chunk_ids=tool_args.get('chunk_ids'),
                        similarity_threshold=tool_args.get('similarity_threshold', 0.7),
                        session_id=tool_args.get('session_id')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__knowledge_optimize_retrieval':
                    if not self.phase3_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Knowledge management functions not available"
                            }
                        }
                    
                    result = await self.phase3_functions.knowledge_optimize_retrieval(
                        usage_window_days=tool_args.get('usage_window_days', 7),
                        session_id=tool_args.get('session_id')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__knowledge_get_related':
                    if not self.phase3_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Knowledge management functions not available"
                            }
                        }
                    
                    result = await self.phase3_functions.knowledge_get_related(
                        chunk_id=tool_args.get('chunk_id', ''),
                        relationship_types=tool_args.get('relationship_types'),
                        max_depth=tool_args.get('max_depth', 2),
                        session_id=tool_args.get('session_id')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                # Phase 3: Session Tracking Functions
                elif tool_name == 'mcp__megamind__session_create_operational':
                    if not self.phase3_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Session tracking functions not available"
                            }
                        }
                    
                    result = await self.phase3_functions.session_create_operational(
                        session_type=tool_args.get('session_type', 'general'),
                        user_id=tool_args.get('user_id'),
                        description=tool_args.get('description')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__session_track_action':
                    if not self.phase3_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Session tracking functions not available"
                            }
                        }
                    
                    result = await self.phase3_functions.session_track_action(
                        session_id=tool_args.get('session_id', ''),
                        action_type=tool_args.get('action_type', ''),
                        description=tool_args.get('description', ''),
                        details=tool_args.get('details'),
                        priority=tool_args.get('priority')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__session_get_recap':
                    if not self.phase3_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Session tracking functions not available"
                            }
                        }
                    
                    result = await self.phase3_functions.session_get_recap(
                        session_id=tool_args.get('session_id', '')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__session_prime_context':
                    if not self.phase3_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Session tracking functions not available"
                            }
                        }
                    
                    result = await self.phase3_functions.session_prime_context(
                        session_id=tool_args.get('session_id', '')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__session_list_recent':
                    if not self.phase3_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Session tracking functions not available"
                            }
                        }
                    
                    result = await self.phase3_functions.session_list_recent(
                        user_id=tool_args.get('user_id'),
                        limit=tool_args.get('limit', 10)
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__session_close':
                    if not self.phase3_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "Session tracking functions not available"
                            }
                        }
                    
                    result = await self.phase3_functions.session_close(
                        session_id=tool_args.get('session_id', '')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                # Phase 4: AI Enhancement Functions
                elif tool_name == 'mcp__megamind__ai_improve_chunk_quality':
                    if not self.phase4_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "AI enhancement functions not available"
                            }
                        }
                    
                    result = await self.phase4_functions.ai_improve_chunk_quality(
                        chunk_id=tool_args.get('chunk_id', ''),
                        session_id=tool_args.get('session_id', ''),
                        apply_automated=tool_args.get('apply_automated', False)
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__ai_record_user_feedback':
                    if not self.phase4_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "AI enhancement functions not available"
                            }
                        }
                    
                    result = await self.phase4_functions.ai_record_user_feedback(
                        feedback_type=tool_args.get('feedback_type', ''),
                        target_id=tool_args.get('target_id', ''),
                        rating=tool_args.get('rating', 0.5),
                        details=tool_args.get('details', {}),
                        user_id=tool_args.get('user_id', ''),
                        session_id=tool_args.get('session_id', '')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__ai_get_adaptive_strategy':
                    if not self.phase4_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "AI enhancement functions not available"
                            }
                        }
                    
                    result = await self.phase4_functions.ai_get_adaptive_strategy(
                        context=tool_args.get('context', {}),
                        session_id=tool_args.get('session_id', '')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__ai_curate_chunks':
                    if not self.phase4_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "AI enhancement functions not available"
                            }
                        }
                    
                    result = await self.phase4_functions.ai_curate_chunks(
                        chunk_ids=tool_args.get('chunk_ids', []),
                        workflow_id=tool_args.get('workflow_id', 'standard_quality'),
                        session_id=tool_args.get('session_id', ''),
                        auto_apply=tool_args.get('auto_apply', False)
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__ai_optimize_performance':
                    if not self.phase4_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "AI enhancement functions not available"
                            }
                        }
                    
                    result = await self.phase4_functions.ai_optimize_performance(
                        optimization_type=tool_args.get('optimization_type', ''),
                        parameters=tool_args.get('parameters', {}),
                        session_id=tool_args.get('session_id', ''),
                        apply=tool_args.get('apply', False)
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__ai_get_performance_insights':
                    if not self.phase4_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "AI enhancement functions not available"
                            }
                        }
                    
                    result = await self.phase4_functions.ai_get_performance_insights(
                        session_id=tool_args.get('session_id', '')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                elif tool_name == 'mcp__megamind__ai_generate_enhancement_report':
                    if not self.phase4_functions:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "AI enhancement functions not available"
                            }
                        }
                    
                    result = await self.phase4_functions.ai_generate_enhancement_report(
                        report_type=tool_args.get('report_type', ''),
                        start_date=tool_args.get('start_date', ''),
                        end_date=tool_args.get('end_date', ''),
                        session_id=tool_args.get('session_id', '')
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                }
                            ]
                        }
                    }
                
                # GitHub Issue #26: Chunk Approval Functions
                elif tool_name == 'mcp__megamind__get_pending_chunks':
                    try:
                        result = self.db_manager.get_pending_chunks_dual_realm(
                            limit=tool_args.get('limit', 20),
                            realm_filter=tool_args.get('realm_filter')
                        )
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                    }
                                ]
                            }
                        }
                    except Exception as e:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32603,
                                "message": f"Error getting pending chunks: {str(e)}"
                            }
                        }
                
                elif tool_name == 'mcp__megamind__approve_chunk':
                    try:
                        result = self.db_manager.approve_chunk_dual_realm(
                            chunk_id=tool_args.get('chunk_id'),
                            approved_by=tool_args.get('approved_by'),
                            approval_notes=tool_args.get('approval_notes')
                        )
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                    }
                                ]
                            }
                        }
                    except Exception as e:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32603,
                                "message": f"Error approving chunk: {str(e)}"
                            }
                        }
                
                elif tool_name == 'mcp__megamind__reject_chunk':
                    try:
                        result = self.db_manager.reject_chunk_dual_realm(
                            chunk_id=tool_args.get('chunk_id'),
                            rejected_by=tool_args.get('rejected_by'),
                            rejection_reason=tool_args.get('rejection_reason')
                        )
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                    }
                                ]
                            }
                        }
                    except Exception as e:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32603,
                                "message": f"Error rejecting chunk: {str(e)}"
                            }
                        }
                
                elif tool_name == 'mcp__megamind__bulk_approve_chunks':
                    try:
                        result = self.db_manager.bulk_approve_chunks_dual_realm(
                            chunk_ids=tool_args.get('chunk_ids', []),
                            approved_by=tool_args.get('approved_by')
                        )
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                                    }
                                ]
                            }
                        }
                    except Exception as e:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32603,
                                "message": f"Error bulk approving chunks: {str(e)}"
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
                    print(json.dumps(response, cls=MegaMindJSONEncoder), flush=True)
                    
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
                    print(json.dumps(error_response, cls=MegaMindJSONEncoder), flush=True)
        
        except Exception as e:
            logger.error(f"Server error: {e}")
        
        logger.info("MCP Server shutting down")

async def wait_for_embedding_service_ready(db_manager, timeout: int = 90) -> bool:
    """Wait for embedding service to be fully ready with comprehensive testing"""
    import time
    
    start_time = time.time()
    logger.info("Waiting for embedding service to initialize...")
    
    # Phase 1: Wait for basic availability
    while not db_manager.embedding_service.is_available():
        elapsed = time.time() - start_time
        if elapsed > timeout:
            logger.error(f"Embedding service failed to initialize within {timeout} seconds")
            return False
        
        if elapsed % 10 == 0:  # Log every 10 seconds
            logger.info(f"Still waiting for embedding service... ({elapsed:.0f}s elapsed)")
        
        await asyncio.sleep(1)
    
    # Phase 2: Comprehensive readiness test
    logger.info("Embedding service available, running comprehensive readiness test...")
    try:
        readiness_result = db_manager.embedding_service.test_readiness()
        
        if not readiness_result['ready']:
            logger.error(f"Embedding service readiness test failed: {readiness_result['error']}")
            logger.error(f"Readiness details: {readiness_result}")
            return False
        
        elapsed = time.time() - start_time
        logger.info(f"Embedding service ready and functional (elapsed: {elapsed:.2f}s)")
        logger.info(f"Readiness test passed: model_loaded={readiness_result['model_loaded']}, "
                   f"test_successful={readiness_result['test_embedding_successful']}, "
                   f"dimension_correct={readiness_result['embedding_dimension_correct']}")
        return True
        
    except Exception as e:
        logger.error(f"Embedding service readiness test failed with exception: {e}")
        return False

async def main():
    """Main entry point for the MCP server with enhanced readiness checking"""
    try:
        # Setup environment paths first
        logger.info("Setting up environment paths...")
        path_config = setup_environment_paths()
        
        # Load configuration
        db_config = load_config()
        
        if not db_config['password']:
            logger.error("Database password not configured. Set MEGAMIND_DB_PASSWORD environment variable.")
            return 1
        
        logger.info("Initializing MegaMind MCP Server...")
        
        # Initialize realm-aware database
        logger.info("Initializing realm-aware database...")
        db_manager = RealmAwareMegaMindDatabase(db_config)
        
        # Wait for embedding service to be ready
        embedding_ready = await wait_for_embedding_service_ready(db_manager, timeout=90)
        if not embedding_ready:
            logger.error("Failed to initialize embedding service - server cannot start")
            return 1
        
        # Test database connection with semantic capabilities
        logger.info("Testing database connection and semantic search capabilities...")
        test_results = db_manager.search_chunks_dual_realm("test", limit=1)
        logger.info(f"Database connection successful. Found {len(test_results)} test results.")
        
        # Get embedding service statistics for diagnostic info
        embedding_stats = db_manager.embedding_service.get_embedding_stats()
        logger.info(f"Embedding service stats: model={embedding_stats['model_name']}, "
                   f"device={embedding_stats['device']}, available={embedding_stats['available']}")
        
        # Final readiness confirmation
        logger.info("All systems ready. Starting MCP server...")
        
        # Start MCP server
        mcp_server = MCPServer(db_manager)
        await mcp_server.run()
            
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))