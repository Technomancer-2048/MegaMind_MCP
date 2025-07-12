#!/usr/bin/env python3
"""
Realm-Aware MegaMind Database Implementation
Extends MegaMindDatabase with dual-realm access patterns and environment-based configuration
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any, Union

import mysql.connector
from mysql.connector import pooling

from realm_config import get_realm_config, get_realm_access_controller, RealmConfigurationManager, RealmAccessController
from inheritance_resolver import InheritanceResolver

logger = logging.getLogger(__name__)

class RealmAwareMegaMindDatabase:
    """Realm-aware database operations with dual-realm access patterns"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection_pool = None
        self.realm_config = get_realm_config()
        self.realm_access = get_realm_access_controller()
        self.inheritance_resolver = None  # Initialized after connection setup
        self._setup_connection_pool()
    
    def _setup_connection_pool(self):
        """Setup MySQL connection pool"""
        try:
            pool_config = {
                'pool_name': 'megamind_realm_pool',
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
            
            # Initialize inheritance resolver with a test connection
            test_connection = self.get_connection()
            self.inheritance_resolver = InheritanceResolver(test_connection)
            test_connection.close()
            
            logger.info(f"Realm-aware database connection pool established for project: {self.realm_config.config.project_realm}")
        except Exception as e:
            logger.error(f"Failed to setup realm-aware database connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get connection from pool"""
        try:
            connection = self.connection_pool.get_connection()
            # Set session variable for current realm
            cursor = connection.cursor()
            cursor.execute("SET @current_realm = %s", (self.realm_config.config.project_realm,))
            cursor.close()
            return connection
        except Exception as e:
            logger.error(f"Failed to get database connection: {e}")
            raise
    
    # Dual-Realm Search Operations
    
    def search_chunks_dual_realm(self, query: str, limit: int = 10, search_type: str = "hybrid") -> List[Dict[str, Any]]:
        """Enhanced dual-realm search with inheritance-aware filtering"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get effective search realms
            search_realms = self.realm_config.get_search_realms()
            
            if search_type == "semantic":
                raw_results = self._realm_semantic_search(cursor, query, limit * 2, search_realms)  # Get more for filtering
            elif search_type == "keyword":
                raw_results = self._realm_keyword_search(cursor, query, limit * 2, search_realms)
            else:  # hybrid (default)
                raw_results = self._realm_hybrid_search(cursor, query, limit * 2, search_realms)
            
            # Apply inheritance-aware filtering
            if self.inheritance_resolver:
                self.inheritance_resolver.db = connection  # Update resolver connection
                filtered_results = self.inheritance_resolver.filter_chunks_by_inheritance(
                    raw_results, self.realm_config.config.project_realm
                )
            else:
                filtered_results = raw_results
            
            # Return limited results
            return filtered_results[:limit]
                
        except Exception as e:
            logger.error(f"Dual-realm search failed: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def _realm_keyword_search(self, cursor, query: str, limit: int, search_realms: List[str]) -> List[Dict[str, Any]]:
        """Keyword search across specified realms with priority weighting"""
        realm_placeholders = ', '.join(['%s'] * len(search_realms))
        
        # Multi-word search with realm priority
        search_query = f"""
        SELECT c.chunk_id, c.content, c.source_document, c.section_path,
               c.chunk_type, c.realm_id, c.access_count, c.last_accessed,
               r.realm_name,
               CASE WHEN c.realm_id = %s THEN 'direct' ELSE 'inherited' END as access_type,
               CASE WHEN c.realm_id = %s THEN 1.0 ELSE 0.8 END as realm_priority_weight,
               (MATCH(c.content) AGAINST(%s IN BOOLEAN MODE) * 
                CASE WHEN c.realm_id = %s THEN 1.2 ELSE 1.0 END) as relevance_score
        FROM megamind_chunks c
        JOIN megamind_realms r ON c.realm_id = r.realm_id
        WHERE c.realm_id IN ({realm_placeholders})
          AND (MATCH(c.content) AGAINST(%s IN BOOLEAN MODE) > 0
               OR c.source_document LIKE %s
               OR c.section_path LIKE %s)
        ORDER BY 
            CASE WHEN c.realm_id = %s THEN 0 ELSE 1 END,
            relevance_score DESC,
            c.access_count DESC
        LIMIT %s
        """
        
        # Prepare search parameters
        like_pattern = f"%{query}%"
        boolean_query = ' '.join([f'+{word}*' for word in query.split()])
        project_realm = self.realm_config.config.project_realm
        
        params = [
            project_realm,  # For access_type calculation
            project_realm,  # For realm_priority_weight calculation
            boolean_query,  # MATCH AGAINST for relevance_score
            project_realm,  # For relevance_score realm boost
            *search_realms,  # For realm_id IN clause
            boolean_query,  # MATCH AGAINST for WHERE clause
            like_pattern,   # source_document LIKE
            like_pattern,   # section_path LIKE
            project_realm,  # For ORDER BY realm priority
            limit
        ]
        
        cursor.execute(search_query, params)
        results = cursor.fetchall()
        
        # Update access counts for retrieved chunks
        if results:
            chunk_ids = [result['chunk_id'] for result in results]
            self._update_access_counts(cursor, chunk_ids)
        
        return results
    
    def _realm_semantic_search(self, cursor, query: str, limit: int, search_realms: List[str]) -> List[Dict[str, Any]]:
        """Semantic search across realms (placeholder for future implementation)"""
        # TODO: Implement semantic search with sentence-transformers
        # For now, fall back to keyword search
        logger.info("Semantic search not yet implemented, falling back to keyword search")
        return self._realm_keyword_search(cursor, query, limit, search_realms)
    
    def _realm_hybrid_search(self, cursor, query: str, limit: int, search_realms: List[str]) -> List[Dict[str, Any]]:
        """Combine semantic and keyword results with realm prioritization"""
        # TODO: Implement hybrid search combining semantic and keyword results
        # For now, use keyword search
        return self._realm_keyword_search(cursor, query, limit, search_realms)
    
    def _update_access_counts(self, cursor, chunk_ids: List[str]):
        """Update access counts for accessed chunks"""
        if not chunk_ids:
            return
        
        placeholders = ', '.join(['%s'] * len(chunk_ids))
        update_query = f"""
        UPDATE megamind_chunks 
        SET access_count = access_count + 1,
            last_accessed = CURRENT_TIMESTAMP
        WHERE chunk_id IN ({placeholders})
        """
        cursor.execute(update_query, chunk_ids)
    
    # Realm-Aware Chunk Operations
    
    def create_chunk_with_target(self, content: str, source_document: str, section_path: str,
                                 session_id: str, target_realm: str = None) -> str:
        """Enhanced realm-aware chunk creation with embedding generation"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Determine target realm
            target = self.realm_config.get_target_realm(target_realm)
            
            # Validate write access
            can_write, message = self.realm_access.validate_realm_operation('create', target)
            if not can_write:
                raise PermissionError(f"Cannot create chunk in realm {target}: {message}")
            
            # Generate new chunk ID
            new_chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
            
            # TODO: Generate embedding for new content
            # embedding = self.embedding_service.generate_embedding(content)
            embedding = None  # Placeholder for now
            
            # Store in session changes for review workflow
            change_data = {
                "chunk_id": new_chunk_id,
                "content": content,
                "source_document": source_document,
                "section_path": section_path,
                "chunk_type": "section",  # Default type
                "line_count": len(content.split('\n')),
                "realm_id": target,
                "embedding": embedding,
                "token_count": len(content.split()),
                "content_hash": None  # TODO: Calculate content hash
            }
            
            # Calculate impact score based on content length and complexity
            impact_score = min(1.0, len(content) / 1000.0)
            
            # Insert into session changes
            change_id = f"change_{uuid.uuid4().hex[:12]}"
            insert_change_query = """
            INSERT INTO megamind_session_changes 
            (change_id, session_id, change_type, change_data, impact_score, source_realm_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_change_query, (
                change_id, session_id, 'create', json.dumps(change_data), 
                impact_score, target
            ))
            
            # Update session metadata
            self._update_session_metadata(cursor, session_id, target)
            
            connection.commit()
            logger.info(f"Chunk creation buffered for realm {target}: {new_chunk_id}")
            return new_chunk_id
            
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Failed to create chunk: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def get_chunk_dual_realm(self, chunk_id: str, include_relationships: bool = True) -> Optional[Dict[str, Any]]:
        """Get chunk from accessible realms with relationship data"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get effective search realms
            search_realms = self.realm_config.get_search_realms()
            realm_placeholders = ', '.join(['%s'] * len(search_realms))
            
            # Main chunk query with realm filtering
            chunk_query = f"""
            SELECT c.chunk_id, c.content, c.source_document, c.section_path,
                   c.chunk_type, c.realm_id, c.access_count, c.last_accessed,
                   c.created_at, c.embedding, c.token_count, c.complexity_score,
                   r.realm_name,
                   CASE WHEN c.realm_id = %s THEN 'direct' ELSE 'inherited' END as access_type
            FROM megamind_chunks c
            JOIN megamind_realms r ON c.realm_id = r.realm_id
            WHERE c.chunk_id = %s AND c.realm_id IN ({realm_placeholders})
            """
            
            params = [self.realm_config.config.project_realm, chunk_id] + search_realms
            cursor.execute(chunk_query, params)
            chunk = cursor.fetchone()
            
            if not chunk:
                return None
            
            # Update access count
            self._update_access_counts(cursor, [chunk_id])
            
            # Get relationships if requested
            if include_relationships:
                chunk['relationships'] = self._get_chunk_relationships(cursor, chunk_id, search_realms)
            
            connection.commit()
            return chunk
            
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None
        finally:
            if connection:
                connection.close()
    
    def _get_chunk_relationships(self, cursor, chunk_id: str, search_realms: List[str]) -> List[Dict[str, Any]]:
        """Get relationships for a chunk within accessible realms"""
        realm_placeholders = ', '.join(['%s'] * len(search_realms))
        
        relationship_query = f"""
        SELECT cr.relationship_type, cr.strength, cr.is_cross_realm,
               c2.chunk_id as related_chunk_id, c2.content as related_content,
               c2.source_document, c2.section_path, c2.realm_id as related_realm_id,
               r.realm_name as related_realm_name
        FROM megamind_chunk_relationships cr
        JOIN megamind_chunks c2 ON cr.related_chunk_id = c2.chunk_id
        JOIN megamind_realms r ON c2.realm_id = r.realm_id
        WHERE cr.chunk_id = %s AND c2.realm_id IN ({realm_placeholders})
        ORDER BY cr.strength DESC, cr.created_at DESC
        LIMIT 10
        """
        
        params = [chunk_id] + search_realms
        cursor.execute(relationship_query, params)
        return cursor.fetchall()
    
    # Cross-Realm Relationship Operations
    
    def get_cross_realm_relationships(self, chunk_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Get relationships that cross realm boundaries with inheritance awareness"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get accessible realms for the current project
            search_realms = self.realm_config.get_search_realms()
            realm_placeholders = ', '.join(['%s'] * len(search_realms))
            
            # Find cross-realm relationships
            cross_realm_query = f"""
            SELECT cr.relationship_id, cr.chunk_id, cr.related_chunk_id,
                   cr.relationship_type, cr.strength, cr.is_cross_realm,
                   c1.realm_id as source_realm, c2.realm_id as target_realm,
                   c1.content as source_content, c2.content as target_content,
                   c1.source_document as source_doc, c2.source_document as target_doc,
                   r1.realm_name as source_realm_name, r2.realm_name as target_realm_name
            FROM megamind_chunk_relationships cr
            JOIN megamind_chunks c1 ON cr.chunk_id = c1.chunk_id
            JOIN megamind_chunks c2 ON cr.related_chunk_id = c2.chunk_id
            JOIN megamind_realms r1 ON c1.realm_id = r1.realm_id
            JOIN megamind_realms r2 ON c2.realm_id = r2.realm_id
            WHERE (cr.chunk_id = %s OR cr.related_chunk_id = %s)
              AND (c1.realm_id IN ({realm_placeholders}) AND c2.realm_id IN ({realm_placeholders}))
              AND cr.is_cross_realm = TRUE
            ORDER BY cr.strength DESC
            LIMIT %s
            """
            
            params = [chunk_id, chunk_id] + search_realms + search_realms + [max_depth * 10]
            cursor.execute(cross_realm_query, params)
            relationships = cursor.fetchall()
            
            # Filter relationships through inheritance resolver
            if self.inheritance_resolver:
                self.inheritance_resolver.db = connection
                filtered_relationships = []
                
                for rel in relationships:
                    # Check access to source chunk
                    source_access = self.inheritance_resolver.resolve_chunk_access(
                        rel['chunk_id'], self.realm_config.config.project_realm
                    )
                    # Check access to target chunk
                    target_access = self.inheritance_resolver.resolve_chunk_access(
                        rel['related_chunk_id'], self.realm_config.config.project_realm
                    )
                    
                    if source_access.access_granted and target_access.access_granted:
                        rel['source_access_type'] = source_access.access_type
                        rel['target_access_type'] = target_access.access_type
                        rel['relationship_weight'] = self._calculate_relationship_weight(rel, source_access, target_access)
                        filtered_relationships.append(rel)
                
                # Sort by relationship weight
                filtered_relationships.sort(key=lambda x: x['relationship_weight'], reverse=True)
                return filtered_relationships
            else:
                return relationships
                
        except Exception as e:
            logger.error(f"Failed to get cross-realm relationships for {chunk_id}: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def _calculate_relationship_weight(self, relationship: Dict, source_access, target_access) -> float:
        """Calculate weighted score for cross-realm relationships"""
        base_strength = relationship['strength']
        
        # Apply access type modifiers
        source_modifier = 1.2 if source_access.access_type == 'direct' else 0.8
        target_modifier = 1.2 if target_access.access_type == 'direct' else 0.8
        
        # Apply realm priority modifiers
        project_realm = self.realm_config.config.project_realm
        source_realm_modifier = 1.1 if relationship['source_realm'] == project_realm else 1.0
        target_realm_modifier = 1.1 if relationship['target_realm'] == project_realm else 1.0
        
        return base_strength * source_modifier * target_modifier * source_realm_modifier * target_realm_modifier
    
    def discover_cross_realm_patterns(self, realm_id: str = None) -> Dict[str, Any]:
        """Discover patterns and insights from cross-realm relationships"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            realm_id = realm_id or self.realm_config.config.project_realm
            
            # Get cross-realm relationship statistics
            stats_query = """
            SELECT 
                source_realm, target_realm, relationship_type,
                COUNT(*) as relationship_count,
                AVG(strength) as avg_strength,
                MAX(strength) as max_strength,
                MIN(strength) as min_strength
            FROM cross_realm_relationships 
            WHERE source_realm = %s OR target_realm = %s
            GROUP BY source_realm, target_realm, relationship_type
            ORDER BY relationship_count DESC, avg_strength DESC
            """
            
            cursor.execute(stats_query, (realm_id, realm_id))
            relationship_stats = cursor.fetchall()
            
            # Get most connected chunks across realms
            connected_chunks_query = """
            SELECT c.chunk_id, c.content, c.source_document, c.realm_id,
                   r.realm_name, COUNT(cr.relationship_id) as connection_count,
                   AVG(cr.strength) as avg_connection_strength
            FROM megamind_chunks c
            JOIN megamind_realms r ON c.realm_id = r.realm_id
            JOIN megamind_chunk_relationships cr ON (c.chunk_id = cr.chunk_id OR c.chunk_id = cr.related_chunk_id)
            WHERE cr.is_cross_realm = TRUE
            GROUP BY c.chunk_id, c.content, c.source_document, c.realm_id, r.realm_name
            HAVING connection_count >= 2
            ORDER BY connection_count DESC, avg_connection_strength DESC
            LIMIT 10
            """
            
            cursor.execute(connected_chunks_query)
            connected_chunks = cursor.fetchall()
            
            return {
                'realm_id': realm_id,
                'relationship_statistics': relationship_stats,
                'highly_connected_chunks': connected_chunks,
                'insights': self._generate_cross_realm_insights(relationship_stats, connected_chunks)
            }
            
        except Exception as e:
            logger.error(f"Failed to discover cross-realm patterns for {realm_id}: {e}")
            return {'error': str(e)}
        finally:
            if connection:
                connection.close()
    
    def _generate_cross_realm_insights(self, stats: List[Dict], connected_chunks: List[Dict]) -> List[str]:
        """Generate insights from cross-realm relationship analysis"""
        insights = []
        
        if stats:
            # Find most common relationship types
            type_counts = {}
            for stat in stats:
                rel_type = stat['relationship_type']
                type_counts[rel_type] = type_counts.get(rel_type, 0) + stat['relationship_count']
            
            if type_counts:
                most_common = max(type_counts, key=type_counts.get)
                insights.append(f"Most common cross-realm relationship: {most_common} ({type_counts[most_common]} connections)")
        
        if connected_chunks:
            # Find realm with most cross-connections
            realm_connections = {}
            for chunk in connected_chunks:
                realm = chunk['realm_id']
                realm_connections[realm] = realm_connections.get(realm, 0) + chunk['connection_count']
            
            if realm_connections:
                most_connected_realm = max(realm_connections, key=realm_connections.get)
                insights.append(f"Most interconnected realm: {most_connected_realm} ({realm_connections[most_connected_realm]} total connections)")
        
        if len(connected_chunks) >= 3:
            avg_connections = sum(chunk['connection_count'] for chunk in connected_chunks) / len(connected_chunks)
            insights.append(f"Average cross-realm connections per highly-connected chunk: {avg_connections:.1f}")
        
        return insights
    
    # Session and Configuration Management
    
    def _update_session_metadata(self, cursor, session_id: str, realm_id: str):
        """Update or create session metadata"""
        # Check if session exists
        check_query = "SELECT session_id FROM megamind_session_metadata WHERE session_id = %s"
        cursor.execute(check_query, (session_id,))
        
        if cursor.fetchone():
            # Update existing session
            update_query = """
            UPDATE megamind_session_metadata 
            SET last_activity = CURRENT_TIMESTAMP,
                pending_changes_count = pending_changes_count + 1
            WHERE session_id = %s
            """
            cursor.execute(update_query, (session_id,))
        else:
            # Create new session
            insert_query = """
            INSERT INTO megamind_session_metadata 
            (session_id, user_context, project_context, realm_id)
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                session_id, 'mcp_user', 
                self.realm_config.config.project_name, realm_id
            ))
    
    def get_realm_info(self) -> Dict[str, Any]:
        """Get comprehensive realm configuration and status"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get basic realm configuration
            realm_info = self.realm_config.get_realm_info()
            
            # Get realm statistics
            search_realms = self.realm_config.get_search_realms()
            realm_placeholders = ', '.join(['%s'] * len(search_realms))
            
            stats_query = f"""
            SELECT 
                r.realm_id,
                r.realm_name,
                r.realm_type,
                COUNT(c.chunk_id) as chunk_count,
                AVG(c.access_count) as avg_access_count,
                MAX(c.last_accessed) as last_access
            FROM megamind_realms r
            LEFT JOIN megamind_chunks c ON r.realm_id = c.realm_id
            WHERE r.realm_id IN ({realm_placeholders})
            GROUP BY r.realm_id, r.realm_name, r.realm_type
            """
            
            cursor.execute(stats_query, search_realms)
            realm_stats = cursor.fetchall()
            
            realm_info['realm_statistics'] = realm_stats
            realm_info['database_status'] = 'connected'
            realm_info['effective_search_realms'] = self.realm_access.get_effective_realms_for_search()
            
            # Add inheritance information
            if self.inheritance_resolver:
                self.inheritance_resolver.db = connection
                realm_info['inheritance_paths'] = self.inheritance_resolver.get_inheritance_paths(
                    self.realm_config.config.project_realm
                )
                realm_info['accessibility_matrix'] = self.inheritance_resolver.get_realm_accessibility_matrix(
                    self.realm_config.config.project_realm
                )
                realm_info['inheritance_stats'] = self.inheritance_resolver.get_inheritance_stats(
                    self.realm_config.config.project_realm
                )
            
            return realm_info
            
        except Exception as e:
            logger.error(f"Failed to get realm info: {e}")
            return {
                'error': str(e),
                'database_status': 'error',
                **self.realm_config.get_realm_info()
            }
        finally:
            if connection:
                connection.close()
    
    # Hot Contexts with Realm Support
    
    def get_realm_hot_contexts(self, model_type: str = "sonnet", 
                               limit: int = 20, include_inherited: bool = True) -> List[Dict[str, Any]]:
        """Get hot contexts for current realm with inheritance support"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Adjust thresholds by model type
            threshold = 2 if model_type.lower() == "opus" else 1
            
            # Get search realms
            if include_inherited:
                search_realms = self.realm_config.get_search_realms()
            else:
                search_realms = [self.realm_config.config.project_realm]
            
            realm_placeholders = ', '.join(['%s'] * len(search_realms))
            
            hot_query = f"""
            SELECT c.chunk_id, c.content, c.source_document, c.section_path,
                   c.chunk_type, c.realm_id, c.access_count, c.last_accessed,
                   r.realm_name,
                   CASE WHEN c.realm_id = %s THEN 'direct' ELSE 'inherited' END as access_type,
                   (c.access_count * CASE WHEN c.realm_id = %s THEN 1.2 ELSE 1.0 END) as prioritized_score
            FROM megamind_chunks c
            JOIN megamind_realms r ON c.realm_id = r.realm_id
            WHERE c.realm_id IN ({realm_placeholders}) AND c.access_count >= %s
            ORDER BY 
                CASE WHEN c.realm_id = %s THEN 0 ELSE 1 END,
                prioritized_score DESC, 
                c.last_accessed DESC
            LIMIT %s
            """
            
            project_realm = self.realm_config.config.project_realm
            params = [project_realm, project_realm] + search_realms + [threshold, project_realm, limit]
            cursor.execute(hot_query, params)
            return cursor.fetchall()
            
        except Exception as e:
            logger.error(f"Realm hot contexts failed: {e}")
            return []
        finally:
            if connection:
                connection.close()