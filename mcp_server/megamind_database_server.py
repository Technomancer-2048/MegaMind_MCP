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
                SELECT target_chunk_id, relationship_type, strength
                FROM megamind_chunk_relationships
                WHERE source_chunk_id = %s
                """
                cursor.execute(rel_query, (chunk_id,))
                chunk['relationships'] = cursor.fetchall()
            
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
                SELECT target_chunk_id, relationship_type, strength, 1 as depth
                FROM megamind_chunk_relationships
                WHERE source_chunk_id = %s
                
                UNION ALL
                
                SELECT cr.target_chunk_id, cr.relationship_type, cr.strength, depth + 1
                FROM megamind_chunk_relationships cr
                JOIN chunk_relations r ON cr.source_chunk_id = r.target_chunk_id
                WHERE depth < %s
            )
            SELECT DISTINCT c.chunk_id, c.content, c.source_document, c.section_path,
                   c.chunk_type, c.line_count, c.token_count, r.relationship_type, r.strength
            FROM chunk_relations r
            JOIN megamind_chunks c ON r.target_chunk_id = c.chunk_id
            ORDER BY r.strength DESC
            """
            
            cursor.execute(related_query, (chunk_id, max_depth))
            results = cursor.fetchall()
            
            return results
            
        except Exception as e:
            logger.error(f"Get related chunks failed: {e}")
            return []
        finally:
            if connection:
                connection.close()

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