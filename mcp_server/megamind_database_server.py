#!/usr/bin/env python3
"""
MegaMind Context Database MCP Server
Phase 2: Intelligence Layer Enhancement

Standalone MCP server providing semantic chunk retrieval, relationship traversal,
session management, and advanced context management through direct database interactions.
"""

import json
import logging
import os
import time
import re
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import asyncio

import mysql.connector
from mysql.connector import pooling
from mcp import Server, Tool
from mcp.server import NotificationOptions
from mcp.types import (
    Content,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChunkResult:
    """Result structure for chunk retrieval"""
    chunk_id: str
    content: str
    source_document: str
    section_path: str
    chunk_type: str
    line_count: int
    token_count: int
    access_count: int
    last_accessed: str
    relevance_score: float = 0.0

class DatabaseManager:
    """Manages database connections and operations"""
    
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
    
    def search_chunks(self, query: str, limit: int = 10, chunk_type: Optional[str] = None) -> List[ChunkResult]:
        """Search chunks using full-text search"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Build search query
            where_conditions = ["MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE)"]
            params = [query]
            
            if chunk_type:
                where_conditions.append("chunk_type = %s")
                params.append(chunk_type)
            
            search_query = f"""
            SELECT chunk_id, content, source_document, section_path, chunk_type,
                   line_count, token_count, access_count, last_accessed,
                   MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE) as relevance_score
            FROM megamind_chunks
            WHERE {' AND '.join(where_conditions)}
            ORDER BY relevance_score DESC, access_count DESC
            LIMIT %s
            """
            
            params.insert(0, query)  # For the SELECT relevance score calculation
            params.append(limit)
            
            cursor.execute(search_query, params)
            results = cursor.fetchall()
            
            chunks = []
            for row in results:
                chunk = ChunkResult(
                    chunk_id=row['chunk_id'],
                    content=row['content'],
                    source_document=row['source_document'],
                    section_path=row['section_path'],
                    chunk_type=row['chunk_type'],
                    line_count=row['line_count'],
                    token_count=row['token_count'] or 0,
                    access_count=row['access_count'],
                    last_accessed=row['last_accessed'].isoformat() if row['last_accessed'] else '',
                    relevance_score=float(row['relevance_score'] or 0.0)
                )
                chunks.append(chunk)
            
            logger.info(f"Found {len(chunks)} chunks matching query: {query[:50]}")
            return chunks
            
        except Exception as e:
            logger.error(f"Search query failed: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def get_chunk(self, chunk_id: str, include_relationships: bool = False) -> Optional[ChunkResult]:
        """Retrieve a specific chunk by ID"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get chunk data
            query = """
            SELECT chunk_id, content, source_document, section_path, chunk_type,
                   line_count, token_count, access_count, last_accessed
            FROM megamind_chunks
            WHERE chunk_id = %s
            """
            
            cursor.execute(query, (chunk_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            chunk = ChunkResult(
                chunk_id=row['chunk_id'],
                content=row['content'],
                source_document=row['source_document'],
                section_path=row['section_path'],
                chunk_type=row['chunk_type'],
                line_count=row['line_count'],
                token_count=row['token_count'] or 0,
                access_count=row['access_count'],
                last_accessed=row['last_accessed'].isoformat() if row['last_accessed'] else ''
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None
        finally:
            if connection:
                connection.close()
    
    def track_access(self, chunk_id: str, query_context: str = "") -> bool:
        """Track chunk access for analytics"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Update access statistics
            update_query = """
            UPDATE megamind_chunks 
            SET access_count = access_count + 1, 
                last_accessed = CURRENT_TIMESTAMP
            WHERE chunk_id = %s
            """
            
            cursor.execute(update_query, (chunk_id,))
            connection.commit()
            
            logger.debug(f"Tracked access for chunk: {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to track access for {chunk_id}: {e}")
            return False
        finally:
            if connection:
                connection.close()
    
    def get_related_chunks(self, chunk_id: str, max_depth: int = 2) -> List[ChunkResult]:
        """Get related chunks through relationship graph traversal"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get directly related chunks
            query = """
            SELECT c.chunk_id, c.content, c.source_document, c.section_path, c.chunk_type,
                   c.line_count, c.token_count, c.access_count, c.last_accessed,
                   r.relationship_type, r.strength
            FROM megamind_chunks c
            JOIN megamind_chunk_relationships r ON (
                (r.chunk_id = %s AND r.related_chunk_id = c.chunk_id) OR
                (r.related_chunk_id = %s AND r.chunk_id = c.chunk_id)
            )
            WHERE c.chunk_id != %s
            ORDER BY r.strength DESC, c.access_count DESC
            LIMIT 10
            """
            
            cursor.execute(query, (chunk_id, chunk_id, chunk_id))
            results = cursor.fetchall()
            
            chunks = []
            for row in results:
                chunk = ChunkResult(
                    chunk_id=row['chunk_id'],
                    content=row['content'],
                    source_document=row['source_document'],
                    section_path=row['section_path'],
                    chunk_type=row['chunk_type'],
                    line_count=row['line_count'],
                    token_count=row['token_count'] or 0,
                    access_count=row['access_count'],
                    last_accessed=row['last_accessed'].isoformat() if row['last_accessed'] else '',
                    relevance_score=float(row['strength'])
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to get related chunks for {chunk_id}: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def search_by_tags(self, tag_type: str, tag_value: Optional[str] = None, limit: int = 10) -> List[ChunkResult]:
        """Search chunks by semantic tags"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Build query based on parameters
            if tag_value:
                query = """
                SELECT c.chunk_id, c.content, c.source_document, c.section_path, 
                       c.chunk_type, c.line_count, c.token_count, c.access_count, c.last_accessed
                FROM megamind_chunks c
                JOIN megamind_chunk_tags t ON c.chunk_id = t.chunk_id
                WHERE t.tag_type = %s AND t.tag_value = %s
                ORDER BY c.access_count DESC, t.confidence DESC
                LIMIT %s
                """
                params = (tag_type, tag_value, limit)
            else:
                query = """
                SELECT c.chunk_id, c.content, c.source_document, c.section_path, 
                       c.chunk_type, c.line_count, c.token_count, c.access_count, c.last_accessed
                FROM megamind_chunks c
                JOIN megamind_chunk_tags t ON c.chunk_id = t.chunk_id
                WHERE t.tag_type = %s
                ORDER BY c.access_count DESC, t.confidence DESC
                LIMIT %s
                """
                params = (tag_type, limit)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            chunks = []
            for row in results:
                chunk = ChunkResult(
                    chunk_id=row['chunk_id'],
                    content=row['content'],
                    source_document=row['source_document'],
                    section_path=row['section_path'],
                    chunk_type=row['chunk_type'],
                    line_count=row['line_count'],
                    token_count=row['token_count'] or 0,
                    access_count=row['access_count'],
                    last_accessed=row['last_accessed'].isoformat() if row['last_accessed'] else ''
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to search by tags: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def get_session_primer(self, last_session_data: str = "", project_context: str = "") -> Dict[str, Any]:
        """Generate lightweight context primer for session continuity"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Parse CLAUDE.md if available to get session context
            claude_md_path = Path.cwd() / "CLAUDE.md"
            claude_context = ""
            
            if claude_md_path.exists():
                try:
                    with open(claude_md_path, 'r', encoding='utf-8') as f:
                        claude_content = f.read()
                        # Extract project overview and current status
                        if "## Project Overview" in claude_content:
                            overview_start = claude_content.find("## Project Overview")
                            overview_end = claude_content.find("\n## ", overview_start + 1)
                            if overview_end == -1:
                                overview_end = len(claude_content)
                            claude_context = claude_content[overview_start:overview_end]
                except Exception as e:
                    logger.warning(f"Could not read CLAUDE.md: {e}")
            
            # Get recent high-access chunks for context
            recent_chunks_query = """
            SELECT chunk_id, section_path, chunk_type, access_count
            FROM megamind_chunks
            WHERE access_count > 5
            ORDER BY last_accessed DESC, access_count DESC
            LIMIT 10
            """
            
            cursor.execute(recent_chunks_query)
            recent_chunks = cursor.fetchall()
            
            # Get project-relevant tags
            tags_query = """
            SELECT tag_type, tag_value, COUNT(*) as usage_count
            FROM megamind_chunk_tags
            GROUP BY tag_type, tag_value
            ORDER BY usage_count DESC
            LIMIT 20
            """
            
            cursor.execute(tags_query)
            tags = cursor.fetchall()
            
            primer = {
                "timestamp": datetime.now().isoformat(),
                "project_context": claude_context,
                "recent_activity": {
                    "hot_chunks": [
                        {
                            "chunk_id": chunk['chunk_id'],
                            "section": chunk['section_path'],
                            "type": chunk['chunk_type'],
                            "access_count": chunk['access_count']
                        }
                        for chunk in recent_chunks
                    ]
                },
                "project_tags": [
                    {
                        "type": tag['tag_type'],
                        "value": tag['tag_value'],
                        "usage": tag['usage_count']
                    }
                    for tag in tags
                ],
                "session_notes": last_session_data,
                "context_summary": f"Project has {len(recent_chunks)} active chunks with recent activity. "
                                 f"Primary focus areas: {', '.join(set(chunk['chunk_type'] for chunk in recent_chunks[:5]))}"
            }
            
            return primer
            
        except Exception as e:
            logger.error(f"Failed to generate session primer: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
        finally:
            if connection:
                connection.close()
    
    def search_by_embedding(self, query: str, limit: int = 10, similarity_threshold: float = 0.7) -> List[ChunkResult]:
        """Search chunks using embedding similarity (requires sentence transformers)"""
        connection = None
        try:
            # Try to import sentence transformers
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np
            except ImportError:
                logger.warning("Sentence transformers not available, falling back to text search")
                return self.search_chunks(query, limit)
            
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Load model (should be cached after first use)
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Generate query embedding
            query_embedding = model.encode([query])[0]
            
            # Get chunks with embeddings
            chunks_query = """
            SELECT chunk_id, content, source_document, section_path, chunk_type,
                   line_count, token_count, access_count, last_accessed, embedding
            FROM megamind_chunks
            WHERE embedding IS NOT NULL AND JSON_LENGTH(embedding) > 0
            """
            
            cursor.execute(chunks_query)
            chunk_rows = cursor.fetchall()
            
            # Calculate similarities
            similarities = []
            for row in chunk_rows:
                try:
                    # Parse embedding from JSON
                    embedding_json = row['embedding']
                    if isinstance(embedding_json, str):
                        chunk_embedding = np.array(json.loads(embedding_json))
                    else:
                        chunk_embedding = np.array(embedding_json)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, chunk_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                    )
                    
                    if similarity >= similarity_threshold:
                        similarities.append((row, float(similarity)))
                        
                except Exception as e:
                    logger.warning(f"Error processing embedding for chunk {row['chunk_id']}: {e}")
                    continue
            
            # Sort by similarity and limit results
            similarities.sort(key=lambda x: x[1], reverse=True)
            similarities = similarities[:limit]
            
            # Convert to ChunkResult objects
            chunks = []
            for row, similarity in similarities:
                chunk = ChunkResult(
                    chunk_id=row['chunk_id'],
                    content=row['content'],
                    source_document=row['source_document'],
                    section_path=row['section_path'],
                    chunk_type=row['chunk_type'],
                    line_count=row['line_count'],
                    token_count=row['token_count'] or 0,
                    access_count=row['access_count'],
                    last_accessed=row['last_accessed'].isoformat() if row['last_accessed'] else '',
                    relevance_score=similarity
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to search by embedding: {e}")
            # Fallback to text search
            return self.search_chunks(query, limit)
        finally:
            if connection:
                connection.close()

class MegaMindMCPServer:
    """MegaMind MCP Server implementation"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.server = Server("megamind-database")
        self.db_manager = DatabaseManager(db_config)
        self._setup_tools()
    
    def _setup_tools(self):
        """Setup MCP tools"""
        
        @self.server.tool()
        async def mcp__megamind_db__search_chunks(
            query: str,
            limit: int = 10,
            chunk_type: str = None,
            model_type: str = "sonnet"
        ) -> Dict[str, Any]:
            """
            Search for content chunks using semantic similarity.
            
            Args:
                query: Search query text
                limit: Maximum number of results (default: 10)
                chunk_type: Filter by chunk type (rule, function, section, example)
                model_type: Target model type for optimization (sonnet, opus)
            """
            try:
                # Adjust limit based on model type
                if model_type == "opus":
                    limit = min(limit, 5)  # More selective for Opus
                
                chunks = self.db_manager.search_chunks(query, limit, chunk_type)
                
                results = []
                for chunk in chunks:
                    # Track access
                    self.db_manager.track_access(chunk.chunk_id, f"search:{query}")
                    
                    results.append({
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "source_document": chunk.source_document,
                        "section_path": chunk.section_path,
                        "chunk_type": chunk.chunk_type,
                        "line_count": chunk.line_count,
                        "token_count": chunk.token_count,
                        "access_count": chunk.access_count,
                        "relevance_score": chunk.relevance_score,
                        "last_accessed": chunk.last_accessed
                    })
                
                return {
                    "status": "success",
                    "query": query,
                    "total_results": len(results),
                    "chunks": results
                }
                
            except Exception as e:
                logger.error(f"Search chunks failed: {e}")
                return {
                    "status": "error",
                    "message": str(e)
                }
        
        @self.server.tool()
        async def mcp__megamind_db__get_chunk(
            chunk_id: str,
            include_relationships: bool = True
        ) -> Dict[str, Any]:
            """
            Retrieve a specific chunk by ID.
            
            Args:
                chunk_id: Unique identifier for the chunk
                include_relationships: Include related chunks
            """
            try:
                chunk = self.db_manager.get_chunk(chunk_id, include_relationships)
                
                if not chunk:
                    return {
                        "status": "error",
                        "message": f"Chunk not found: {chunk_id}"
                    }
                
                # Track access
                self.db_manager.track_access(chunk_id, "direct_access")
                
                result = {
                    "status": "success",
                    "chunk": {
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "source_document": chunk.source_document,
                        "section_path": chunk.section_path,
                        "chunk_type": chunk.chunk_type,
                        "line_count": chunk.line_count,
                        "token_count": chunk.token_count,
                        "access_count": chunk.access_count,
                        "last_accessed": chunk.last_accessed
                    }
                }
                
                # Add relationships if requested
                if include_relationships:
                    related_chunks = self.db_manager.get_related_chunks(chunk_id)
                    result["related_chunks"] = [
                        {
                            "chunk_id": rel.chunk_id,
                            "section_path": rel.section_path,
                            "chunk_type": rel.chunk_type,
                            "relevance_score": rel.relevance_score
                        }
                        for rel in related_chunks
                    ]
                
                return result
                
            except Exception as e:
                logger.error(f"Get chunk failed: {e}")
                return {
                    "status": "error",
                    "message": str(e)
                }
        
        @self.server.tool()
        async def mcp__megamind_db__get_related_chunks(
            chunk_id: str,
            max_depth: int = 2
        ) -> Dict[str, Any]:
            """
            Get chunks related to the specified chunk through relationship traversal.
            
            Args:
                chunk_id: Source chunk ID
                max_depth: Maximum relationship depth to traverse
            """
            try:
                related_chunks = self.db_manager.get_related_chunks(chunk_id, max_depth)
                
                results = []
                for chunk in related_chunks:
                    results.append({
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                        "source_document": chunk.source_document,
                        "section_path": chunk.section_path,
                        "chunk_type": chunk.chunk_type,
                        "relevance_score": chunk.relevance_score
                    })
                
                return {
                    "status": "success",
                    "source_chunk_id": chunk_id,
                    "total_related": len(results),
                    "related_chunks": results
                }
                
            except Exception as e:
                logger.error(f"Get related chunks failed: {e}")
                return {
                    "status": "error",
                    "message": str(e)
                }
        
        @self.server.tool()
        async def mcp__megamind_db__track_access(
            chunk_id: str,
            query_context: str = ""
        ) -> Dict[str, Any]:
            """
            Track access to a chunk for analytics.
            
            Args:
                chunk_id: Chunk identifier
                query_context: Context of the access
            """
            try:
                success = self.db_manager.track_access(chunk_id, query_context)
                
                return {
                    "status": "success" if success else "error",
                    "chunk_id": chunk_id,
                    "tracked": success
                }
                
            except Exception as e:
                logger.error(f"Track access failed: {e}")
                return {
                    "status": "error",
                    "message": str(e)
                }
        
        @self.server.tool()
        async def mcp__megamind_db__search_by_tags(
            tag_type: str,
            tag_value: str = None,
            limit: int = 10
        ) -> Dict[str, Any]:
            """
            Search chunks by semantic tags.
            
            Args:
                tag_type: Type of tag to search (subsystem, function_type, language, etc.)
                tag_value: Specific tag value to filter by
                limit: Maximum number of results
            """
            try:
                chunks = self.db_manager.search_by_tags(tag_type, tag_value, limit)
                
                results = []
                for chunk in chunks:
                    # Track access
                    self.db_manager.track_access(chunk.chunk_id, f"tag_search:{tag_type}:{tag_value}")
                    
                    results.append({
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                        "source_document": chunk.source_document,
                        "section_path": chunk.section_path,
                        "chunk_type": chunk.chunk_type,
                        "token_count": chunk.token_count,
                        "access_count": chunk.access_count
                    })
                
                return {
                    "status": "success",
                    "tag_type": tag_type,
                    "tag_value": tag_value,
                    "total_results": len(results),
                    "chunks": results
                }
                
            except Exception as e:
                logger.error(f"Tag search failed: {e}")
                return {
                    "status": "error",
                    "message": str(e)
                }
        
        @self.server.tool()
        async def mcp__megamind_db__get_session_primer(
            last_session_data: str = "",
            project_context: str = ""
        ) -> Dict[str, Any]:
            """
            Generate lightweight context primer for session continuity.
            
            Args:
                last_session_data: Previous session information
                project_context: Current project context
            """
            try:
                primer = self.db_manager.get_session_primer(last_session_data, project_context)
                
                return {
                    "status": "success",
                    "primer": primer,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Session primer failed: {e}")
                return {
                    "status": "error",
                    "message": str(e)
                }
        
        @self.server.tool()
        async def mcp__megamind_db__search_by_embedding(
            query: str,
            limit: int = 10,
            similarity_threshold: float = 0.7
        ) -> Dict[str, Any]:
            """
            Search chunks using semantic embedding similarity.
            
            Args:
                query: Search query text
                limit: Maximum number of results
                similarity_threshold: Minimum similarity score
            """
            try:
                chunks = self.db_manager.search_by_embedding(query, limit, similarity_threshold)
                
                results = []
                for chunk in chunks:
                    # Track access
                    self.db_manager.track_access(chunk.chunk_id, f"embedding_search:{query}")
                    
                    results.append({
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "source_document": chunk.source_document,
                        "section_path": chunk.section_path,
                        "chunk_type": chunk.chunk_type,
                        "similarity_score": chunk.relevance_score,
                        "token_count": chunk.token_count,
                        "access_count": chunk.access_count
                    })
                
                return {
                    "status": "success",
                    "query": query,
                    "total_results": len(results),
                    "chunks": results
                }
                
            except Exception as e:
                logger.error(f"Embedding search failed: {e}")
                return {
                    "status": "error",
                    "message": str(e)
                }

def load_config():
    """Load configuration from environment variables"""
    return {
        'host': os.getenv('MEGAMIND_DB_HOST', 'localhost'),
        'port': os.getenv('MEGAMIND_DB_PORT', '3309'),
        'database': os.getenv('MEGAMIND_DB_NAME', 'megamind_database'),
        'user': os.getenv('MEGAMIND_DB_USER', 'megamind_user'),
        'password': os.getenv('MEGAMIND_DB_PASSWORD', ''),
        'pool_size': os.getenv('CONNECTION_POOL_SIZE', '10')
    }

async def main():
    """Main entry point for the MCP server"""
    try:
        # Load configuration
        db_config = load_config()
        
        if not db_config['password']:
            logger.error("Database password not configured. Set MEGAMIND_DB_PASSWORD environment variable.")
            return 1
        
        # Initialize MCP server
        mcp_server = MegaMindMCPServer(db_config)
        
        logger.info("MegaMind MCP Server starting...")
        
        # Run the server
        async with mcp_server.server:
            await mcp_server.server.run()
            
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))