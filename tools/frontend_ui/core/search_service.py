from typing import List, Dict, Any
from .chunk_service import ChunkService
import mysql.connector
from mysql.connector import Error
import logging

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self, chunk_service: ChunkService):
        self.chunk_service = chunk_service
    
    def simulate_agent_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Simulate agent search using FULLTEXT search on megamind_chunks"""
        if not query.strip():
            return []
        
        try:
            conn = self.chunk_service._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Use FULLTEXT search with MATCH() AGAINST() for semantic relevance
            # First try natural language mode
            search_query = """
            SELECT chunk_id, realm_id, content, complexity_score,
                   source_document, section_path, chunk_type,
                   token_count, access_count, approval_status,
                   MATCH(content, section_path) AGAINST(%s IN NATURAL LANGUAGE MODE) as relevance_score
            FROM megamind_chunks 
            WHERE MATCH(content, section_path) AGAINST(%s IN NATURAL LANGUAGE MODE)
            ORDER BY relevance_score DESC, access_count DESC
            LIMIT %s
            """
            
            cursor.execute(search_query, (query, query, limit))
            results = cursor.fetchall()
            
            # If no results with FULLTEXT, fall back to LIKE search
            if not results:
                fallback_query = """
                SELECT chunk_id, realm_id, content, complexity_score,
                       source_document, section_path, chunk_type,
                       token_count, access_count, approval_status,
                       1.0 as relevance_score
                FROM megamind_chunks 
                WHERE content LIKE %s OR section_path LIKE %s
                ORDER BY access_count DESC, created_at DESC
                LIMIT %s
                """
                
                like_pattern = f"%{query}%"
                cursor.execute(fallback_query, (like_pattern, like_pattern, limit))
                results = cursor.fetchall()
            
            # Format results with content preview
            formatted_results = []
            for result in results:
                # Create content preview (first 200 chars)
                content_preview = result['content'][:200] if result['content'] else ""
                if result['content'] and len(result['content']) > 200:
                    content_preview += "..."
                
                # Highlight search terms in preview (simple version)
                highlighted_preview = self._highlight_search_terms(content_preview, query)
                
                formatted_results.append({
                    'chunk_id': result['chunk_id'],
                    'realm_id': result['realm_id'],
                    'content_preview': highlighted_preview,
                    'relevance_score': float(result['relevance_score']),
                    'complexity_score': result['complexity_score'],
                    'source_document': result['source_document'],
                    'section_path': result['section_path'],
                    'chunk_type': result['chunk_type'],
                    'token_count': result['token_count'],
                    'access_count': result['access_count'],
                    'approval_status': result['approval_status']
                })
            
            cursor.close()
            conn.close()
            
            logger.info(f"Search for '{query}' returned {len(formatted_results)} results")
            return formatted_results
            
        except Error as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _highlight_search_terms(self, text: str, query: str) -> str:
        """Simple search term highlighting"""
        if not text or not query:
            return text
        
        # Simple case-insensitive highlighting
        import re
        words = query.split()
        highlighted = text
        
        for word in words:
            if len(word) > 2:  # Only highlight words longer than 2 chars
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                highlighted = pattern.sub(f"<mark>{word}</mark>", highlighted)
        
        return highlighted
    
    def get_chunk_relationships(self, chunk_id: str) -> List[Dict[str, Any]]:
        """Get related chunks using megamind_chunk_relationships table"""
        try:
            conn = self.chunk_service._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Get relationships where this chunk is the source
            relationship_query = """
            SELECT mcr.relationship_type, mcr.strength, mcr.discovered_by,
                   mc.chunk_id, mc.content, mc.source_document, mc.section_path,
                   mc.approval_status
            FROM megamind_chunk_relationships mcr
            JOIN megamind_chunks mc ON mcr.related_chunk_id = mc.chunk_id
            WHERE mcr.chunk_id = %s
            ORDER BY mcr.strength DESC
            LIMIT 10
            """
            
            cursor.execute(relationship_query, (chunk_id,))
            relationships = cursor.fetchall()
            
            # Format relationships with content preview
            formatted_relationships = []
            for rel in relationships:
                content_preview = rel['content'][:150] if rel['content'] else ""
                if rel['content'] and len(rel['content']) > 150:
                    content_preview += "..."
                
                formatted_relationships.append({
                    'chunk_id': rel['chunk_id'],
                    'relationship_type': rel['relationship_type'],
                    'strength': rel['strength'],
                    'discovered_by': rel['discovered_by'],
                    'content_preview': content_preview,
                    'source_document': rel['source_document'],
                    'section_path': rel['section_path'],
                    'approval_status': rel['approval_status']
                })
            
            cursor.close()
            conn.close()
            
            logger.info(f"Found {len(formatted_relationships)} relationships for chunk {chunk_id}")
            return formatted_relationships
            
        except Error as e:
            logger.error(f"Relationship query error: {e}")
            return []
    
    def search_by_realm(self, query: str, realm_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search chunks within a specific realm"""
        if not query.strip():
            return []
        
        try:
            conn = self.chunk_service._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Realm-specific search
            search_query = """
            SELECT chunk_id, realm_id, content, complexity_score,
                   source_document, section_path, chunk_type,
                   token_count, access_count, approval_status,
                   MATCH(content, section_path) AGAINST(%s IN NATURAL LANGUAGE MODE) as relevance_score
            FROM megamind_chunks 
            WHERE realm_id = %s
              AND MATCH(content, section_path) AGAINST(%s IN NATURAL LANGUAGE MODE)
            ORDER BY relevance_score DESC, access_count DESC
            LIMIT %s
            """
            
            cursor.execute(search_query, (query, realm_id, query, limit))
            results = cursor.fetchall()
            
            # Format results
            formatted_results = []
            for result in results:
                content_preview = result['content'][:200] if result['content'] else ""
                if result['content'] and len(result['content']) > 200:
                    content_preview += "..."
                
                formatted_results.append({
                    'chunk_id': result['chunk_id'],
                    'realm_id': result['realm_id'],
                    'content_preview': content_preview,
                    'relevance_score': float(result['relevance_score']),
                    'complexity_score': result['complexity_score'],
                    'source_document': result['source_document'],
                    'section_path': result['section_path'],
                    'chunk_type': result['chunk_type'],
                    'token_count': result['token_count'],
                    'access_count': result['access_count'],
                    'approval_status': result['approval_status']
                })
            
            cursor.close()
            conn.close()
            
            logger.info(f"Realm-specific search for '{query}' in '{realm_id}' returned {len(formatted_results)} results")
            return formatted_results
            
        except Error as e:
            logger.error(f"Realm search error: {e}")
            return []
    
    def get_recent_searches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently accessed chunks (simulating search history)"""
        try:
            conn = self.chunk_service._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Get recently accessed chunks
            recent_query = """
            SELECT chunk_id, realm_id, content, source_document, 
                   section_path, last_accessed, access_count,
                   approval_status
            FROM megamind_chunks 
            WHERE last_accessed IS NOT NULL
            ORDER BY last_accessed DESC
            LIMIT %s
            """
            
            cursor.execute(recent_query, (limit,))
            results = cursor.fetchall()
            
            # Format results
            formatted_results = []
            for result in results:
                content_preview = result['content'][:150] if result['content'] else ""
                if result['content'] and len(result['content']) > 150:
                    content_preview += "..."
                
                formatted_results.append({
                    'chunk_id': result['chunk_id'],
                    'realm_id': result['realm_id'],
                    'content_preview': content_preview,
                    'source_document': result['source_document'],
                    'section_path': result['section_path'],
                    'last_accessed': result['last_accessed'].isoformat() if result['last_accessed'] else None,
                    'access_count': result['access_count'],
                    'approval_status': result['approval_status']
                })
            
            cursor.close()
            conn.close()
            
            return formatted_results
            
        except Error as e:
            logger.error(f"Recent searches error: {e}")
            return []