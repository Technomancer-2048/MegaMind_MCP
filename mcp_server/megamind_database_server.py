#!/usr/bin/env python3
"""
MegaMind Context Database MCP Server
Fixed implementation for MCP v1.x API
"""

import json
import logging
import os
import asyncio
from typing import List, Dict, Optional, Any

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
        
        logger.info("MegaMind MCP Server starting...")
        
        # Test database connection
        test_results = db_manager.search_chunks("test", limit=1)
        logger.info(f"Database test successful. Found {len(test_results)} results.")
        
        # Keep server running
        logger.info("MegaMind MCP Server ready!")
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))