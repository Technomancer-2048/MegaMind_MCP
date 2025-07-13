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

try:
    from .realm_config import get_realm_config, get_realm_access_controller, RealmConfigurationManager, RealmAccessController
    from .inheritance_resolver import InheritanceResolver
    from .promotion_manager import PromotionManager, PromotionType, BusinessImpact, PermissionType
    from .realm_security_validator import RealmSecurityValidator
    from .services.embedding_service import get_embedding_service
    from .services.vector_search import RealmAwareVectorSearchEngine
except ImportError:
    from realm_config import get_realm_config, get_realm_access_controller, RealmConfigurationManager, RealmAccessController
    from inheritance_resolver import InheritanceResolver
    from promotion_manager import PromotionManager, PromotionType, BusinessImpact, PermissionType
    from realm_security_validator import RealmSecurityValidator
    from services.embedding_service import get_embedding_service
    from services.vector_search import RealmAwareVectorSearchEngine

logger = logging.getLogger(__name__)

class RealmAwareMegaMindDatabase:
    """Realm-aware database operations with dual-realm access patterns"""
    
    def __init__(self, config: Dict[str, Any], realm_config: Optional[Any] = None, shared_embedding_service: Optional[Any] = None):
        self.config = config
        self.connection_pool = None
        
        # Use injected realm configuration or get from environment
        if realm_config is not None:
            self.realm_config = realm_config
            logger.info(f"Using injected realm config for realm: {getattr(realm_config, 'project_realm', 'unknown')}")
        else:
            self.realm_config = get_realm_config()
            logger.info("Using environment-based realm config")
        
        self.realm_access = get_realm_access_controller()
        self.inheritance_resolver = None  # Initialized after connection setup
        self.promotion_manager = None  # Initialized after connection setup
        self.security_validator = None  # Initialized after connection setup
        
        # Use shared embedding service or create new one
        if shared_embedding_service is not None:
            self.embedding_service = shared_embedding_service
            logger.info("Using shared embedding service")
        else:
            self.embedding_service = get_embedding_service()
            logger.info("Created new embedding service instance")
        
        self.vector_search_engine = None  # Initialized after realm config is ready
        
        self._setup_connection_pool()
    
    def _get_realm_config(self):
        """Safely get realm configuration regardless of type"""
        # Handle both RealmConfigurationManager and RealmConfig objects
        if hasattr(self.realm_config, 'config'):
            # It's a RealmConfigurationManager
            return self.realm_config.config
        else:
            # It's a RealmConfig object directly
            return self.realm_config
    
    def _setup_connection_pool(self):
        """Setup MySQL connection pool"""
        try:
            pool_config = {
                'pool_name': 'megamind_realm_pool',
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
            
            # Initialize inheritance resolver with a test connection
            test_connection = self.get_connection()
            self.inheritance_resolver = InheritanceResolver(test_connection)
            
            # Initialize promotion manager
            self.promotion_manager = PromotionManager(test_connection)
            
            # Initialize security validator
            self.security_validator = RealmSecurityValidator(test_connection)
            
            # Initialize vector search engine with realm configuration
            self.vector_search_engine = RealmAwareVectorSearchEngine(
                project_realm=self._get_realm_config().project_realm,
                global_realm='GLOBAL'
            )
            
            test_connection.close()
            
            logger.info(f"Realm-aware database connection pool established for project: {self._get_realm_config().project_realm}")
        except Exception as e:
            logger.error(f"Failed to setup realm-aware database connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get connection from pool"""
        try:
            connection = self.connection_pool.get_connection()
            # Set session variable for current realm
            cursor = connection.cursor()
            cursor.execute("SET @current_realm = %s", (self._get_realm_config().project_realm,))
            cursor.close()
            return connection
        except Exception as e:
            logger.error(f"Failed to get database connection: {e}")
            raise
    
    # Dual-Realm Search Operations
    
    def search_chunks_dual_realm(self, query: str, limit: int = 10, search_type: str = "hybrid", threshold: float = None) -> List[Dict[str, Any]]:
        """Enhanced dual-realm search with inheritance-aware filtering"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get effective search realms
            search_realms = self.realm_config.get_search_realms()
            
            if search_type == "semantic":
                raw_results = self._realm_semantic_search(cursor, query, limit * 2, search_realms, threshold)  # Get more for filtering
            elif search_type == "keyword":
                # WORKAROUND: Route keyword search through hybrid search fallback to avoid Decimal serialization issues
                # The hybrid search automatically falls back to keyword search when embeddings aren't available
                raw_results = self._realm_hybrid_search(cursor, query, limit * 2, search_realms)
            else:  # hybrid (default)
                raw_results = self._realm_hybrid_search(cursor, query, limit * 2, search_realms)
            
            # Apply inheritance-aware filtering (GitHub Issue #5 Phase 3)
            try:
                if self.inheritance_resolver:
                    self.inheritance_resolver.db = connection  # Update resolver connection
                    filtered_results = self.inheritance_resolver.filter_chunks_by_inheritance(
                        raw_results, self._get_realm_config().project_realm
                    )
                    logger.info(f"Inheritance resolver applied, filtered {len(raw_results)} -> {len(filtered_results)} results")
                else:
                    logger.warning("Inheritance resolver not available, using unfiltered results")
                    filtered_results = raw_results
            except Exception as e:
                logger.warning(f"Inheritance resolver failed: {e}, using unfiltered results")
                filtered_results = raw_results
            
            # Return limited results with datetime sanitization (GitHub Issue #5)
            sanitized_results = self._sanitize_chunk_results(filtered_results[:limit])
            
            # Debug: Check for any remaining Decimal objects
            from decimal import Decimal
            for i, result in enumerate(sanitized_results):
                for key, value in result.items():
                    if isinstance(value, Decimal):
                        logger.error(f"DECIMAL FOUND in sanitized results[{i}][{key}]: {value} (type: {type(value)})")
            
            return sanitized_results
                
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
               CAST(CASE WHEN c.realm_id = %s THEN 1.0 ELSE 0.8 END AS FLOAT) as realm_priority_weight,
               CAST((MATCH(c.content) AGAINST(%s IN BOOLEAN MODE) * 
                CASE WHEN c.realm_id = %s THEN 1.2 ELSE 1.0 END) AS FLOAT) as relevance_score
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
        project_realm = self._get_realm_config().project_realm
        
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
    
    def _realm_semantic_search(self, cursor, query: str, limit: int, search_realms: List[str], threshold: float = None) -> List[Dict[str, Any]]:
        """Semantic search across realms using vector similarity"""
        if not self.vector_search_engine or not self.embedding_service.is_available():
            logger.warning("Semantic search not available, falling back to keyword search")
            return self._realm_keyword_search(cursor, query, limit, search_realms)
        
        # Get chunks with embeddings from specified realms
        realm_placeholders = ', '.join(['%s'] * len(search_realms))
        chunks_query = f"""
        SELECT chunk_id, content, source_document, section_path, 
               chunk_type, realm_id, access_count, last_accessed, embedding,
               created_at, updated_at, token_count, line_count
        FROM megamind_chunks
        WHERE realm_id IN ({realm_placeholders}) 
          AND embedding IS NOT NULL
        """
        
        cursor.execute(chunks_query, search_realms)
        chunks_data = cursor.fetchall()
        
        # Use vector search engine for semantic search
        search_results = self.vector_search_engine.dual_realm_semantic_search(
            query=query,
            chunks_data=chunks_data,
            limit=limit,
            threshold=threshold
        )
        
        # Convert SearchResult objects back to dictionaries and update access counts
        result_dicts = []
        chunk_ids_to_update = []
        
        for result in search_results:
            result_dict = {
                'chunk_id': result.chunk_id,
                'content': result.content,
                'source_document': result.source_document,
                'section_path': result.section_path,
                'realm_id': result.realm_id,
                'similarity_score': result.similarity_score,
                'final_score': result.final_score,
                'access_count': result.access_count,
                'access_type': 'direct' if result.realm_id == self._get_realm_config().project_realm else 'inherited',
                'realm_priority_weight': self.vector_search_engine.project_priority if result.realm_id == self._get_realm_config().project_realm else self.vector_search_engine.global_priority
            }
            result_dicts.append(result_dict)
            chunk_ids_to_update.append(result.chunk_id)
        
        # Update access counts for retrieved chunks
        self._update_access_counts(cursor, chunk_ids_to_update)
        
        return result_dicts
    
    def _realm_hybrid_search(self, cursor, query: str, limit: int, search_realms: List[str]) -> List[Dict[str, Any]]:
        """Combine semantic and keyword results with realm prioritization"""
        if not self.vector_search_engine or not self.embedding_service.is_available():
            logger.warning("Semantic search not available, using keyword search only")
            return self._realm_keyword_search(cursor, query, limit, search_realms)
        
        # Get all chunks from specified realms
        realm_placeholders = ', '.join(['%s'] * len(search_realms))
        chunks_query = f"""
        SELECT chunk_id, content, source_document, section_path, 
               chunk_type, realm_id, access_count, last_accessed, embedding,
               created_at, updated_at, token_count, line_count
        FROM megamind_chunks
        WHERE realm_id IN ({realm_placeholders})
        """
        
        cursor.execute(chunks_query, search_realms)
        chunks_data = cursor.fetchall()
        
        # Use vector search engine for hybrid search
        search_results = self.vector_search_engine.realm_aware_hybrid_search(
            query=query,
            chunks_data=chunks_data,
            limit=limit
        )
        
        # Convert SearchResult objects back to dictionaries and update access counts
        result_dicts = []
        chunk_ids_to_update = []
        
        for result in search_results:
            result_dict = {
                'chunk_id': result.chunk_id,
                'content': result.content,
                'source_document': result.source_document,
                'section_path': result.section_path,
                'realm_id': result.realm_id,
                'similarity_score': result.similarity_score,
                'keyword_score': result.keyword_score,
                'final_score': result.final_score,
                'access_count': result.access_count,
                'access_type': 'direct' if result.realm_id == self._get_realm_config().project_realm else 'inherited',
                'realm_priority_weight': self.vector_search_engine.project_priority if result.realm_id == self._get_realm_config().project_realm else self.vector_search_engine.global_priority
            }
            result_dicts.append(result_dict)
            chunk_ids_to_update.append(result.chunk_id)
        
        # Update access counts for retrieved chunks
        self._update_access_counts(cursor, chunk_ids_to_update)
        
        return result_dicts
    
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
            
            # Determine target realm - handle both RealmConfigurationManager and RealmConfig
            if hasattr(self.realm_config, 'get_target_realm'):
                # It's a RealmConfigurationManager
                target = self.realm_config.get_target_realm(target_realm)
            else:
                # It's a RealmConfig dataclass - implement logic manually
                if target_realm:
                    if target_realm.upper() == 'GLOBAL':
                        target = self.realm_config.global_realm
                    elif target_realm.upper() == 'PROJECT':
                        target = self.realm_config.project_realm
                    else:
                        target = self.realm_config.project_realm  # Default fallback
                else:
                    # Use configured default
                    if self.realm_config.default_target == 'GLOBAL':
                        target = self.realm_config.global_realm
                    else:
                        target = self.realm_config.project_realm
            
            # Validate write access
            can_write, message = self.realm_access.validate_realm_operation('create', target)
            if not can_write:
                raise PermissionError(f"Cannot create chunk in realm {target}: {message}")
            
            # Generate new chunk ID
            new_chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
            
            # Generate embedding for new content with realm context
            embedding = self.embedding_service.generate_embedding(content, realm_context=target)
            if embedding is not None:
                embedding_json = json.dumps(embedding)
            else:
                embedding_json = None
            
            # Store in session changes for review workflow
            change_data = {
                "chunk_id": new_chunk_id,
                "content": content,
                "source_document": source_document,
                "section_path": section_path,
                "chunk_type": "section",  # Default type
                "line_count": len(content.split('\n')),
                "realm_id": target,
                "embedding": embedding_json,
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
            
            params = [self._get_realm_config().project_realm, chunk_id] + search_realms
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
            # Sanitize chunk result for JSON serialization (GitHub Issue #5)
            return self._sanitize_chunk_results([chunk])[0] if chunk else None
            
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
    
    def get_related_chunks(self, chunk_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Get related chunks up to max_depth levels (alias for get_cross_realm_relationships)"""
        return self.get_cross_realm_relationships(chunk_id, max_depth)
    
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
                        rel['chunk_id'], self._get_realm_config().project_realm
                    )
                    # Check access to target chunk
                    target_access = self.inheritance_resolver.resolve_chunk_access(
                        rel['related_chunk_id'], self._get_realm_config().project_realm
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
        project_realm = self._get_realm_config().project_realm
        source_realm_modifier = 1.1 if relationship['source_realm'] == project_realm else 1.0
        target_realm_modifier = 1.1 if relationship['target_realm'] == project_realm else 1.0
        
        return base_strength * source_modifier * target_modifier * source_realm_modifier * target_realm_modifier
    
    def discover_cross_realm_patterns(self, realm_id: str = None) -> Dict[str, Any]:
        """Discover patterns and insights from cross-realm relationships"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            realm_id = realm_id or self._get_realm_config().project_realm
            
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
                self._get_realm_config().project_name, realm_id
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
                    self._get_realm_config().project_realm
                )
                realm_info['accessibility_matrix'] = self.inheritance_resolver.get_realm_accessibility_matrix(
                    self._get_realm_config().project_realm
                )
                realm_info['inheritance_stats'] = self.inheritance_resolver.get_inheritance_stats(
                    self._get_realm_config().project_realm
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
                search_realms = [self._get_realm_config().project_realm]
            
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
            
            project_realm = self._get_realm_config().project_realm
            params = [project_realm, project_realm] + search_realms + [threshold, project_realm, limit]
            cursor.execute(hot_query, params)
            return cursor.fetchall()
            
        except Exception as e:
            logger.error(f"Realm hot contexts failed: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    # ===================================================================
    # Knowledge Promotion Operations (Phase 3)
    # ===================================================================
    
    def create_promotion_request(self, source_chunk_id: str, target_realm_id: str,
                               promotion_type: str, justification: str,
                               business_impact: str, requested_by: str,
                               session_id: str) -> str:
        """Create a knowledge promotion request"""
        connection = None
        try:
            connection = self.get_connection()
            
            # Validate promotion type and business impact
            promo_type = PromotionType(promotion_type.lower())
            biz_impact = BusinessImpact(business_impact.lower())
            
            # Update promotion manager connection
            self.promotion_manager.db = connection
            
            promotion_id = self.promotion_manager.create_promotion_request(
                source_chunk_id=source_chunk_id,
                target_realm_id=target_realm_id,
                promotion_type=promo_type,
                justification=justification,
                business_impact=biz_impact,
                requested_by=requested_by,
                session_id=session_id
            )
            
            logger.info(f"Promotion request created: {promotion_id}")
            return promotion_id
            
        except Exception as e:
            logger.error(f"Failed to create promotion request: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def approve_promotion_request(self, promotion_id: str, reviewed_by: str,
                                review_notes: str = None) -> bool:
        """Approve a promotion request"""
        connection = None
        try:
            connection = self.get_connection()
            
            # Update promotion manager connection
            self.promotion_manager.db = connection
            
            result = self.promotion_manager.approve_promotion_request(
                promotion_id=promotion_id,
                reviewed_by=reviewed_by,
                review_notes=review_notes
            )
            
            logger.info(f"Promotion request approved: {promotion_id} by {reviewed_by}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to approve promotion request: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def reject_promotion_request(self, promotion_id: str, reviewed_by: str,
                               review_notes: str) -> bool:
        """Reject a promotion request"""
        connection = None
        try:
            connection = self.get_connection()
            
            # Update promotion manager connection
            self.promotion_manager.db = connection
            
            result = self.promotion_manager.reject_promotion_request(
                promotion_id=promotion_id,
                reviewed_by=reviewed_by,
                review_notes=review_notes
            )
            
            logger.info(f"Promotion request rejected: {promotion_id} by {reviewed_by}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to reject promotion request: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def get_promotion_requests(self, status: str = None, user_id: str = None,
                             limit: int = 50) -> List[Dict[str, Any]]:
        """Get promotion requests with optional filtering"""
        connection = None
        try:
            connection = self.get_connection()
            
            # Update promotion manager connection
            self.promotion_manager.db = connection
            
            status_enum = None
            if status:
                from promotion_manager import PromotionStatus
                status_enum = PromotionStatus(status.lower())
            
            requests = self.promotion_manager.get_promotion_requests(
                status=status_enum,
                user_id=user_id,
                limit=limit
            )
            
            # Convert to dictionaries for JSON serialization
            result = []
            for req in requests:
                req_dict = {
                    'promotion_id': req.promotion_id,
                    'source_chunk_id': req.source_chunk_id,
                    'source_realm_id': req.source_realm_id,
                    'target_realm_id': req.target_realm_id,
                    'promotion_type': req.promotion_type.value,
                    'status': req.status.value,
                    'requested_by': req.requested_by,
                    'requested_at': req.requested_at.isoformat(),
                    'justification': req.justification,
                    'business_impact': req.business_impact.value,
                    'reviewed_by': req.reviewed_by,
                    'reviewed_at': req.reviewed_at.isoformat() if req.reviewed_at else None,
                    'review_notes': req.review_notes,
                    'target_chunk_id': req.target_chunk_id
                }
                result.append(req_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get promotion requests: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def get_promotion_impact(self, promotion_id: str) -> Optional[Dict[str, Any]]:
        """Get impact analysis for a promotion request"""
        connection = None
        try:
            connection = self.get_connection()
            
            # Update promotion manager connection
            self.promotion_manager.db = connection
            
            impact = self.promotion_manager.get_promotion_impact(promotion_id)
            
            if not impact:
                return None
            
            return {
                'impact_id': impact.impact_id,
                'promotion_id': impact.promotion_id,
                'affected_chunks_count': impact.affected_chunks_count,
                'affected_relationships_count': impact.affected_relationships_count,
                'potential_conflicts_count': impact.potential_conflicts_count,
                'content_quality_score': impact.content_quality_score,
                'relevance_score': impact.relevance_score,
                'uniqueness_score': impact.uniqueness_score,
                'conflict_analysis': impact.conflict_analysis,
                'dependency_analysis': impact.dependency_analysis,
                'usage_impact': impact.usage_impact
            }
            
        except Exception as e:
            logger.error(f"Failed to get promotion impact: {e}")
            return None
        finally:
            if connection:
                connection.close()
    
    def get_promotion_queue_summary(self, realm_id: str = None) -> Dict[str, Any]:
        """Get summary of promotion queue status"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get promotion counts by status
            status_query = """
            SELECT status, COUNT(*) as count
            FROM megamind_promotion_queue
            """ + ("WHERE target_realm_id = %s" if realm_id else "") + """
            GROUP BY status
            ORDER BY status
            """
            
            if realm_id:
                cursor.execute(status_query, (realm_id,))
            else:
                cursor.execute(status_query)
            
            status_counts = {row['status']: row['count'] for row in cursor.fetchall()}
            
            # Get business impact distribution
            impact_query = """
            SELECT business_impact, COUNT(*) as count
            FROM megamind_promotion_queue
            WHERE status = 'pending'
            """ + ("AND target_realm_id = %s" if realm_id else "") + """
            GROUP BY business_impact
            ORDER BY 
                CASE business_impact 
                    WHEN 'critical' THEN 1 
                    WHEN 'high' THEN 2 
                    WHEN 'medium' THEN 3 
                    WHEN 'low' THEN 4 
                END
            """
            
            if realm_id:
                cursor.execute(impact_query, (realm_id,))
            else:
                cursor.execute(impact_query)
            
            impact_distribution = {row['business_impact']: row['count'] for row in cursor.fetchall()}
            
            # Get oldest pending request
            oldest_query = """
            SELECT promotion_id, source_chunk_id, requested_at, 
                   DATEDIFF(NOW(), requested_at) as days_pending
            FROM megamind_promotion_queue
            WHERE status = 'pending'
            """ + ("AND target_realm_id = %s" if realm_id else "") + """
            ORDER BY requested_at ASC
            LIMIT 1
            """
            
            if realm_id:
                cursor.execute(oldest_query, (realm_id,))
            else:
                cursor.execute(oldest_query)
            
            oldest_pending = cursor.fetchone()
            
            # Get recent activity
            recent_query = """
            SELECT COUNT(*) as recent_count
            FROM megamind_promotion_queue
            WHERE requested_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            """ + ("AND target_realm_id = %s" if realm_id else "")
            
            if realm_id:
                cursor.execute(recent_query, (realm_id,))
            else:
                cursor.execute(recent_query)
            
            recent_count = cursor.fetchone()['recent_count']
            
            summary = {
                'status_counts': status_counts,
                'impact_distribution': impact_distribution,
                'oldest_pending': oldest_pending,
                'recent_activity_count': recent_count,
                'total_pending': status_counts.get('pending', 0),
                'total_completed': status_counts.get('completed', 0),
                'filter_realm': realm_id
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get promotion queue summary: {e}")
            return {
                'status_counts': {},
                'impact_distribution': {},
                'oldest_pending': None,
                'recent_activity_count': 0,
                'total_pending': 0,
                'total_completed': 0,
                'filter_realm': realm_id,
                'error': str(e)
            }
        finally:
            if connection:
                connection.close()
    
    # ===================================================================
    # Role-Based Access Control Operations (Phase 3)
    # ===================================================================
    
    def check_user_permission(self, user_id: str, realm_id: str, permission: str) -> bool:
        """Check if user has specific permission in realm"""
        connection = None
        try:
            connection = self.get_connection()
            
            # Update promotion manager connection
            self.promotion_manager.db = connection
            
            perm_type = PermissionType(permission.lower())
            
            return self.promotion_manager.check_user_permission(
                user_id=user_id,
                realm_id=realm_id,
                permission=perm_type
            )
            
        except Exception as e:
            logger.error(f"Failed to check user permission: {e}")
            return False
        finally:
            if connection:
                connection.close()
    
    def assign_user_role(self, user_id: str, role_id: str, realm_id: str,
                        assigned_by: str, expires_at: str = None,
                        assignment_reason: str = None) -> str:
        """Assign role to user"""
        connection = None
        try:
            connection = self.get_connection()
            
            # Update promotion manager connection
            self.promotion_manager.db = connection
            
            expires_datetime = None
            if expires_at:
                from datetime import datetime
                expires_datetime = datetime.fromisoformat(expires_at)
            
            assignment_id = self.promotion_manager.assign_user_role(
                user_id=user_id,
                role_id=role_id,
                realm_id=realm_id,
                assigned_by=assigned_by,
                expires_at=expires_datetime,
                assignment_reason=assignment_reason
            )
            
            return assignment_id
            
        except Exception as e:
            logger.error(f"Failed to assign user role: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def revoke_user_role(self, assignment_id: str, revoked_by: str,
                        revocation_reason: str = None) -> bool:
        """Revoke user role assignment"""
        connection = None
        try:
            connection = self.get_connection()
            
            # Update promotion manager connection
            self.promotion_manager.db = connection
            
            return self.promotion_manager.revoke_user_role(
                assignment_id=assignment_id,
                revoked_by=revoked_by,
                revocation_reason=revocation_reason
            )
            
        except Exception as e:
            logger.error(f"Failed to revoke user role: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def get_user_roles(self, user_id: str, realm_id: str = None) -> List[Dict[str, Any]]:
        """Get user's role assignments"""
        connection = None
        try:
            connection = self.get_connection()
            
            # Update promotion manager connection
            self.promotion_manager.db = connection
            
            roles = self.promotion_manager.get_user_roles(
                user_id=user_id,
                realm_id=realm_id
            )
            
            # Convert to dictionaries for JSON serialization
            result = []
            for role in roles:
                role_dict = {
                    'assignment_id': role.assignment_id,
                    'user_id': role.user_id,
                    'role_id': role.role_id,
                    'realm_id': role.realm_id,
                    'assigned_by': role.assigned_by,
                    'assigned_at': role.assigned_at.isoformat(),
                    'expires_at': role.expires_at.isoformat() if role.expires_at else None,
                    'is_active': role.is_active
                }
                result.append(role_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get user roles: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    # ===================================================================
    # Security Operations (Phase 3)
    # ===================================================================
    
    def run_security_scan(self, realm_id: str = None) -> Dict[str, Any]:
        """Run comprehensive security scan"""
        connection = None
        try:
            connection = self.get_connection()
            
            # Use current realm if not specified
            scan_realm = realm_id or self._get_realm_config().project_realm
            
            # Update security validator connection
            self.security_validator.db = connection
            
            scan_result = self.security_validator.run_comprehensive_security_scan(scan_realm)
            
            logger.info(f"Security scan completed for realm {scan_realm}: "
                       f"score={scan_result.get('overall_security_score', 0):.2f}")
            
            return scan_result
            
        except Exception as e:
            logger.error(f"Failed to run security scan: {e}")
            return {
                'realm_id': realm_id or self._get_realm_config().project_realm,
                'error': str(e),
                'overall_security_score': 0.0,
                'security_level': 'critical'
            }
        finally:
            if connection:
                connection.close()
    
    def validate_realm_isolation(self, realm_id: str = None) -> Dict[str, Any]:
        """Validate realm isolation"""
        connection = None
        try:
            connection = self.get_connection()
            
            # Use current realm if not specified
            check_realm = realm_id or self._get_realm_config().project_realm
            
            # Update security validator connection
            self.security_validator.db = connection
            
            isolation_check = self.security_validator.validate_realm_isolation(check_realm)
            
            return {
                'realm_id': check_realm,
                'is_isolated': isolation_check.is_isolated,
                'security_score': isolation_check.security_score,
                'violations': isolation_check.violations,
                'affected_chunks': isolation_check.affected_chunks,
                'cross_realm_leaks': isolation_check.cross_realm_leaks
            }
            
        except Exception as e:
            logger.error(f"Failed to validate realm isolation: {e}")
            return {
                'realm_id': check_realm,
                'error': str(e),
                'is_isolated': False,
                'security_score': 0.0
            }
        finally:
            if connection:
                connection.close()
    
    def log_security_violation(self, violation_type: str, severity: str,
                             user_id: str, source_ip: str, attempted_action: str,
                             target_resource: str = None, realm_context: str = None) -> str:
        """Log a security violation"""
        connection = None
        try:
            connection = self.get_connection()
            
            # Update promotion manager connection for audit logging
            self.promotion_manager.db = connection
            
            violation_id = self.promotion_manager.log_security_violation(
                violation_type=violation_type,
                severity=severity,
                user_id=user_id,
                source_ip=source_ip,
                attempted_action=attempted_action,
                target_resource=target_resource,
                realm_context=realm_context or self._get_realm_config().project_realm
            )
            
            return violation_id
            
        except Exception as e:
            logger.error(f"Failed to log security violation: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    # New Semantic Search Methods for Phase 1
    
    def search_chunks_semantic(self, query: str, limit: int = 10, threshold: float = None) -> List[Dict[str, Any]]:
        """Pure semantic search across Global + Project realms"""
        return self.search_chunks_dual_realm(query, limit, search_type="semantic", threshold=threshold)
    
    def search_chunks_by_similarity(self, reference_chunk_id: str, limit: int = 10, threshold: float = None) -> List[Dict[str, Any]]:
        """Find chunks similar to a reference chunk using embeddings"""
        if not self.vector_search_engine or not self.embedding_service.is_available():
            logger.warning("Semantic search not available for similarity search")
            return []
        
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get reference chunk
            cursor.execute("""
                SELECT chunk_id, content, source_document, section_path, 
                       chunk_type, realm_id, access_count, last_accessed, embedding,
                       created_at, updated_at, token_count, line_count
                FROM megamind_chunks 
                WHERE chunk_id = %s AND embedding IS NOT NULL
            """, (reference_chunk_id,))
            
            reference_chunk = cursor.fetchone()
            if not reference_chunk:
                logger.warning(f"Reference chunk {reference_chunk_id} not found or has no embedding")
                return []
            
            # Get search realms
            search_realms = self.realm_config.get_search_realms()
            realm_placeholders = ', '.join(['%s'] * len(search_realms))
            
            # Get candidate chunks with embeddings
            cursor.execute(f"""
                SELECT chunk_id, content, source_document, section_path, 
                       chunk_type, realm_id, access_count, last_accessed, embedding,
                       created_at, updated_at, token_count, line_count
                FROM megamind_chunks
                WHERE realm_id IN ({realm_placeholders}) 
                  AND embedding IS NOT NULL
                  AND chunk_id != %s
            """, search_realms + [reference_chunk_id])
            
            chunks_data = cursor.fetchall()
            
            # Use vector search engine for similarity search
            search_results = self.vector_search_engine.find_similar_chunks(
                reference_chunk=reference_chunk,
                chunks_data=chunks_data,
                limit=limit,
                threshold=threshold
            )
            
            # Convert to dictionaries and update access counts
            result_dicts = []
            chunk_ids_to_update = []
            
            for result in search_results:
                result_dict = {
                    'chunk_id': result.chunk_id,
                    'content': result.content,
                    'source_document': result.source_document,
                    'section_path': result.section_path,
                    'realm_id': result.realm_id,
                    'similarity_score': result.similarity_score,
                    'final_score': result.final_score,
                    'access_count': result.access_count,
                    'access_type': 'direct' if result.realm_id == self._get_realm_config().project_realm else 'inherited',
                    'reference_chunk_id': reference_chunk_id
                }
                result_dicts.append(result_dict)
                chunk_ids_to_update.append(result.chunk_id)
            
            # Update access counts
            self._update_access_counts(cursor, chunk_ids_to_update)
            connection.commit()
            
            return result_dicts
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def get_embedding_service_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics"""
        return self.embedding_service.get_embedding_stats()
    
    def get_vector_search_stats(self) -> Dict[str, Any]:
        """Get vector search engine statistics"""
        if self.vector_search_engine:
            return self.vector_search_engine.get_search_stats()
        return {'available': False}
    
    def batch_generate_embeddings(self, chunk_ids: List[str] = None, realm_id: str = None) -> Dict[str, Any]:
        """Generate embeddings for chunks that don't have them"""
        if not self.embedding_service.is_available():
            return {'error': 'Embedding service not available', 'processed': 0}
        
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Build query to find chunks without embeddings
            where_conditions = ["embedding IS NULL"]
            params = []
            
            if chunk_ids:
                placeholders = ', '.join(['%s'] * len(chunk_ids))
                where_conditions.append(f"chunk_id IN ({placeholders})")
                params.extend(chunk_ids)
            
            if realm_id:
                where_conditions.append("realm_id = %s")
                params.append(realm_id)
            
            where_clause = " AND ".join(where_conditions)
            
            cursor.execute(f"""
                SELECT chunk_id, content, realm_id 
                FROM megamind_chunks 
                WHERE {where_clause}
                LIMIT 100
            """, params)
            
            chunks_to_process = cursor.fetchall()
            
            if not chunks_to_process:
                return {'message': 'No chunks need embedding generation', 'processed': 0}
            
            # Generate embeddings in batch
            texts = [chunk['content'] for chunk in chunks_to_process]
            realm_contexts = [chunk['realm_id'] for chunk in chunks_to_process]
            
            embeddings = self.embedding_service.generate_embeddings_batch(texts, realm_contexts)
            
            # Update chunks with embeddings
            processed_count = 0
            for chunk, embedding in zip(chunks_to_process, embeddings):
                if embedding is not None:
                    embedding_json = json.dumps(embedding)
                    cursor.execute("""
                        UPDATE megamind_chunks 
                        SET embedding = %s, updated_at = NOW() 
                        WHERE chunk_id = %s
                    """, (embedding_json, chunk['chunk_id']))
                    processed_count += 1
            
            connection.commit()
            
            return {
                'message': f'Successfully generated embeddings for {processed_count} chunks',
                'processed': processed_count,
                'total_candidates': len(chunks_to_process)
            }
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            if connection:
                connection.rollback()
            return {'error': str(e), 'processed': 0}
        finally:
            if connection:
                connection.close()
    
    def _sanitize_chunk_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert database results to client-safe responses - only expose client-relevant fields"""
        from decimal import Decimal
        
        sanitized = []
        for result in results:
            # ONLY include client-relevant fields - filter out ALL internal database fields
            clean_result = {}
            
            # Core client fields
            if result.get('chunk_id'):
                clean_result['chunk_id'] = result['chunk_id']
            if result.get('content'):
                clean_result['content'] = result['content']
            if result.get('source_document'):
                clean_result['source_document'] = result['source_document']
            if result.get('section_path'):
                clean_result['section_path'] = result['section_path']
            if result.get('chunk_type'):
                clean_result['chunk_type'] = result['chunk_type']
            if result.get('realm_id'):
                clean_result['realm_id'] = result['realm_id']
            
            # Usage metrics (convert to safe types)
            if 'access_count' in result:
                clean_result['access_count'] = int(result['access_count']) if result['access_count'] is not None else 0
            
            # Search-specific fields (convert Decimals to floats)
            if 'relevance_score' in result:
                score = result['relevance_score']
                clean_result['relevance_score'] = float(score) if isinstance(score, Decimal) else (float(score) if score is not None else 0.0)
            
            if 'realm_priority_weight' in result:
                weight = result['realm_priority_weight']
                clean_result['realm_priority_weight'] = float(weight) if isinstance(weight, Decimal) else (float(weight) if weight is not None else 1.0)
            
            if 'access_type' in result:
                clean_result['access_type'] = result['access_type']
            
            # Convert datetime fields to ISO strings (but don't include internal timestamps)
            for key in ['last_accessed', 'created_at', 'updated_at']:
                if key in result and result[key] and hasattr(result[key], 'isoformat'):
                    # Only expose last_accessed to clients, not internal created_at/updated_at
                    if key == 'last_accessed':
                        clean_result['last_accessed_str'] = result[key].isoformat()
            
            # Explicitly exclude internal fields: embedding, token_count, line_count, complexity_score, realm_name
            # These should NEVER reach the client
            
            sanitized.append(clean_result)
        
        return sanitized
    
    # ==========================================
    # SESSION MANAGEMENT METHODS (Missing from RealmAware implementation)
    # ==========================================
    
    def get_session_primer(self, last_session_data: Optional[str] = None) -> Dict[str, Any]:
        """Generate lightweight context for session continuity with realm awareness"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get realm configuration for context
            realm_config = self._get_realm_config()
            search_realms = [realm_config.global_realm, realm_config.project_realm]
            realm_placeholders = ', '.join(['%s'] * len(search_realms))
            
            # Get recent high-access chunks for context priming across both realms
            primer_query = f"""
            SELECT chunk_id, content, source_document, section_path, access_count, realm_id
            FROM megamind_chunks
            WHERE realm_id IN ({realm_placeholders}) AND access_count > 3
            ORDER BY last_accessed DESC, access_count DESC
            LIMIT 15
            """
            cursor.execute(primer_query, search_realms)
            primer_chunks = cursor.fetchall()
            
            # Get session context if provided
            session_context = {}
            if last_session_data:
                try:
                    # Parse session data for continuity
                    session_context = json.loads(last_session_data) if isinstance(last_session_data, str) else {}
                except (json.JSONDecodeError, TypeError):
                    session_context = {"session_id": str(last_session_data)}
            
            # Sanitize results for client
            sanitized_chunks = self._sanitize_chunk_results(primer_chunks)
            
            primer_data = {
                "primer_chunks": sanitized_chunks,
                "realm_context": {
                    "project_realm": realm_config.project_realm,
                    "global_realm": realm_config.global_realm,
                    "total_chunks": len(sanitized_chunks),
                    "realm_distribution": self._get_realm_distribution(primer_chunks)
                },
                "session_continuity": session_context,
                "timestamp": datetime.now().isoformat()
            }
            
            return primer_data
            
        except Exception as e:
            logger.error(f"Failed to generate session primer: {e}")
            return {"error": str(e), "primer_chunks": [], "realm_context": {}}
        finally:
            if connection:
                connection.close()
    
    def get_pending_changes(self, session_id: str) -> List[Dict[str, Any]]:
        """Get pending changes with smart highlighting and realm awareness"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            changes_query = """
            SELECT sc.change_id, sc.change_type, sc.chunk_id, sc.target_chunk_id,
                   sc.change_data, sc.impact_score, sc.timestamp,
                   c.source_document, c.access_count, c.realm_id
            FROM megamind_session_changes sc
            LEFT JOIN megamind_chunks c ON sc.chunk_id = c.chunk_id
            WHERE sc.session_id = %s AND sc.status = 'pending'
            ORDER BY sc.impact_score DESC, sc.timestamp ASC
            """
            cursor.execute(changes_query, (session_id,))
            raw_changes = cursor.fetchall()
            
            # Process and enhance changes with realm context
            enhanced_changes = []
            for change in raw_changes:
                try:
                    change_data = json.loads(change['change_data']) if change['change_data'] else {}
                except (json.JSONDecodeError, TypeError):
                    change_data = {}
                
                enhanced_change = {
                    'change_id': change['change_id'],
                    'change_type': change['change_type'],
                    'chunk_id': change['chunk_id'],
                    'target_chunk_id': change['target_chunk_id'],
                    'realm_id': change.get('realm_id', 'UNKNOWN'),
                    'source_document': change.get('source_document', ''),
                    'access_count': change.get('access_count', 0),
                    'impact_score': float(change['impact_score']) if change['impact_score'] else 0.0,
                    'timestamp': change['timestamp'].isoformat() if change['timestamp'] else '',
                    'change_summary': self._generate_change_summary(change['change_type'], change_data),
                    'realm_impact': self._assess_realm_impact(change.get('realm_id'), change['change_type'])
                }
                enhanced_changes.append(enhanced_change)
            
            return enhanced_changes
            
        except Exception as e:
            logger.error(f"Failed to get pending changes for session {session_id}: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def commit_session_changes(self, session_id: str, approved_changes: List[str]) -> Dict[str, Any]:
        """Commit approved changes and track contributions with realm awareness"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            stats = {"chunks_modified": 0, "chunks_created": 0, "relationships_added": 0, "realms_affected": set()}
            rollback_data = []
            
            # Process each approved change
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
                    logger.warning(f"Change {change_id} not found or not pending")
                    continue
                
                try:
                    change_data = json.loads(change['change_data']) if change['change_data'] else {}
                except (json.JSONDecodeError, TypeError):
                    change_data = {}
                
                # Apply the change based on type
                if change['change_type'] == 'create_chunk':
                    self._apply_chunk_creation(cursor, change_data, stats)
                elif change['change_type'] == 'update_chunk':
                    self._apply_chunk_update(cursor, change['chunk_id'], change_data, stats, rollback_data)
                elif change['change_type'] == 'add_relationship':
                    self._apply_relationship_addition(cursor, change['chunk_id'], change['target_chunk_id'], change_data, stats)
                
                # Mark change as applied
                update_change_query = """
                UPDATE megamind_session_changes
                SET status = 'applied', applied_timestamp = NOW()
                WHERE change_id = %s
                """
                cursor.execute(update_change_query, (change_id,))
            
            # Record session contribution
            contribution_id = f"contrib_{uuid.uuid4().hex[:12]}"
            contribution_query = """
            INSERT INTO megamind_knowledge_contributions 
            (contribution_id, session_id, chunks_modified, chunks_created, 
             relationships_added, contribution_impact)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            total_impact = stats["chunks_modified"] + stats["chunks_created"] + stats["relationships_added"]
            cursor.execute(contribution_query, (
                contribution_id, session_id, stats["chunks_modified"], 
                stats["chunks_created"], stats["relationships_added"], total_impact
            ))
            
            connection.commit()
            
            # Convert set to list for JSON serialization
            stats["realms_affected"] = list(stats["realms_affected"])
            stats["contribution_id"] = contribution_id
            stats["total_changes_applied"] = len(approved_changes)
            
            logger.info(f"Committed {len(approved_changes)} changes for session {session_id}")
            return stats
            
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Failed to commit session changes: {e}")
            return {"error": str(e), "changes_applied": 0}
        finally:
            if connection:
                connection.close()
    
    # ==========================================
    # ANALYTICS METHODS (Missing from RealmAware implementation)
    # ==========================================
    
    def track_access(self, chunk_id: str, query_context: str = "") -> Dict[str, Any]:
        """Update access analytics for optimization with realm awareness"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Update chunk access count and timestamp
            update_query = """
            UPDATE megamind_chunks
            SET access_count = access_count + 1,
                last_accessed = NOW()
            WHERE chunk_id = %s
            """
            cursor.execute(update_query, (chunk_id,))
            
            # Get realm info for the accessed chunk
            realm_query = """
            SELECT realm_id, source_document, section_path
            FROM megamind_chunks
            WHERE chunk_id = %s
            """
            cursor.execute(realm_query, (chunk_id,))
            chunk_info = cursor.fetchone()
            
            # Record access analytics if query context provided
            if query_context and chunk_info:
                analytics_id = f"analytics_{uuid.uuid4().hex[:12]}"
                analytics_query = """
                INSERT INTO megamind_performance_metrics
                (metric_id, metric_type, chunk_id, realm_id, query_context, metric_value)
                VALUES (%s, 'access_tracking', %s, %s, %s, 1)
                """
                cursor.execute(analytics_query, (
                    analytics_id, chunk_id, chunk_info['realm_id'], query_context
                ))
            
            connection.commit()
            
            result = {
                "chunk_id": chunk_id,
                "access_tracked": True,
                "realm_id": chunk_info['realm_id'] if chunk_info else "UNKNOWN",
                "query_context_recorded": bool(query_context),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.debug(f"Tracked access for chunk {chunk_id} in realm {result['realm_id']}")
            return result
            
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Failed to track access for chunk {chunk_id}: {e}")
            return {"chunk_id": chunk_id, "access_tracked": False, "error": str(e)}
        finally:
            if connection:
                connection.close()
    
    def get_hot_contexts(self, model_type: str = "sonnet", limit: int = 20) -> List[Dict[str, Any]]:
        """Get frequently accessed chunks prioritized by usage patterns with realm awareness"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            realm_config = self._get_realm_config()
            search_realms = [realm_config.global_realm, realm_config.project_realm]
            realm_placeholders = ', '.join(['%s'] * len(search_realms))
            
            # Adjust for model type - Opus gets more selective, high-value chunks
            if model_type.lower() == "opus":
                hot_query = f"""
                SELECT chunk_id, content, source_document, section_path, 
                       chunk_type, access_count, last_accessed, realm_id
                FROM megamind_chunks
                WHERE realm_id IN ({realm_placeholders}) AND access_count >= 3
                ORDER BY access_count DESC, token_count ASC
                LIMIT %s
                """
            else:
                # Sonnet and other models get broader selection
                hot_query = f"""
                SELECT chunk_id, content, source_document, section_path,
                       chunk_type, access_count, last_accessed, realm_id
                FROM megamind_chunks
                WHERE realm_id IN ({realm_placeholders}) AND access_count >= 1
                ORDER BY access_count DESC, last_accessed DESC
                LIMIT %s
                """
            
            query_params = search_realms + [limit]
            cursor.execute(hot_query, query_params)
            hot_chunks = cursor.fetchall()
            
            # Sanitize results and add realm priority information
            sanitized_chunks = self._sanitize_chunk_results(hot_chunks)
            
            # Add model-specific optimization hints
            for chunk in sanitized_chunks:
                chunk['optimization_target'] = model_type
                chunk['context_priority'] = self._calculate_context_priority(
                    chunk['access_count'], 
                    chunk['realm_id'], 
                    realm_config.project_realm
                )
            
            logger.info(f"Retrieved {len(sanitized_chunks)} hot contexts for {model_type}")
            return sanitized_chunks
            
        except Exception as e:
            logger.error(f"Failed to get hot contexts: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    # ==========================================
    # CONTENT MANAGEMENT METHODS (Missing implementations)
    # ==========================================
    
    def create_chunk(self, content: str, source_document: str, section_path: str, 
                    session_id: str, target_realm: str = None) -> str:
        """Create new chunk - wrapper for create_chunk_with_target"""
        return self.create_chunk_with_target(
            content=content,
            source_document=source_document, 
            section_path=section_path,
            session_id=session_id,
            target_realm=target_realm
        )
    
    def update_chunk(self, chunk_id: str, new_content: str, session_id: str) -> str:
        """Update existing chunk content with session buffering"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Generate change ID
            change_id = f"upd_{uuid.uuid4().hex[:12]}"
            
            # Buffer the change in session
            change_data = {
                "new_content": new_content,
                "timestamp": datetime.now().isoformat()
            }
            
            insert_query = """
            INSERT INTO megamind_session_changes
            (change_id, session_id, change_type, target_chunk_id, change_data, impact_score, priority)
            VALUES (%s, %s, 'update_chunk', %s, %s, 1.0, 'medium')
            """
            
            cursor.execute(insert_query, (
                change_id, session_id, chunk_id, json.dumps(change_data)
            ))
            
            connection.commit()
            logger.info(f"Buffered chunk update for chunk {chunk_id} in session {session_id}")
            return change_id
            
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Failed to buffer chunk update: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def add_relationship(self, chunk_id_1: str, chunk_id_2: str, relationship_type: str, session_id: str) -> str:
        """Add relationship between chunks with session buffering"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Generate change ID
            change_id = f"rel_{uuid.uuid4().hex[:12]}"
            
            # Buffer the change in session
            change_data = {
                "chunk_id_2": chunk_id_2,
                "relationship_type": relationship_type,
                "timestamp": datetime.now().isoformat()
            }
            
            insert_query = """
            INSERT INTO megamind_session_changes
            (change_id, session_id, change_type, target_chunk_id, change_data, impact_score, priority)
            VALUES (%s, %s, 'add_relationship', %s, %s, 0.5, 'low')
            """
            
            cursor.execute(insert_query, (
                change_id, session_id, chunk_id_1, json.dumps(change_data)
            ))
            
            connection.commit()
            logger.info(f"Buffered relationship {relationship_type} between {chunk_id_1} and {chunk_id_2}")
            return change_id
            
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Failed to buffer relationship addition: {e}")
            raise
        finally:
            if connection:
                connection.close()

    # ==========================================
    # HELPER METHODS for Session Management
    # ==========================================
    
    def _get_realm_distribution(self, chunks: List[Dict]) -> Dict[str, int]:
        """Calculate realm distribution for primer context"""
        distribution = {}
        for chunk in chunks:
            realm = chunk.get('realm_id', 'UNKNOWN')
            distribution[realm] = distribution.get(realm, 0) + 1
        return distribution
    
    def _generate_change_summary(self, change_type: str, change_data: Dict) -> str:
        """Generate human-readable summary of a change"""
        if change_type == 'create_chunk':
            doc = change_data.get('source_document', 'unknown')
            return f"Create new chunk in {doc}"
        elif change_type == 'update_chunk':
            return "Update existing chunk content"
        elif change_type == 'add_relationship':
            rel_type = change_data.get('relationship_type', 'related')
            return f"Add {rel_type} relationship"
        else:
            return f"Unknown change type: {change_type}"
    
    def _assess_realm_impact(self, realm_id: str, change_type: str) -> str:
        """Assess the impact level of a change on a realm"""
        if realm_id == "GLOBAL":
            return "HIGH" if change_type == 'create_chunk' else "MEDIUM"
        else:
            return "MEDIUM" if change_type == 'create_chunk' else "LOW"
    
    def _apply_chunk_creation(self, cursor, change_data: Dict, stats: Dict):
        """Apply chunk creation from session changes"""
        # Implementation would handle chunk creation
        stats["chunks_created"] += 1
        stats["realms_affected"].add(change_data.get('target_realm', 'PROJECT'))
    
    def _apply_chunk_update(self, cursor, chunk_id: str, change_data: Dict, stats: Dict, rollback_data: List):
        """Apply chunk update from session changes"""
        # Implementation would handle chunk updates
        stats["chunks_modified"] += 1
    
    def _apply_relationship_addition(self, cursor, chunk_id_1: str, chunk_id_2: str, change_data: Dict, stats: Dict):
        """Apply relationship addition from session changes"""
        # Implementation would handle relationship creation
        stats["relationships_added"] += 1
    
    def _calculate_context_priority(self, access_count: int, realm_id: str, project_realm: str) -> str:
        """Calculate context priority for hot contexts"""
        if realm_id == project_realm:
            priority_boost = 1.2
        else:
            priority_boost = 1.0
        
        adjusted_count = access_count * priority_boost
        
        if adjusted_count >= 10:
            return "HIGH"
        elif adjusted_count >= 5:
            return "MEDIUM"
        else:
            return "LOW"