from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import mysql.connector
from mysql.connector import Error
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    chunk_id: str
    realm_id: str
    content: str
    complexity_score: float
    source_document: str
    section_path: Optional[str]
    chunk_type: str
    line_count: int
    token_count: int
    access_count: int
    created_at: str
    last_accessed: str

class ChunkService:
    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        self.db_config = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password,
            'autocommit': True
        }
    
    def _get_connection(self):
        """Get database connection"""
        try:
            return mysql.connector.connect(**self.db_config)
        except Error as e:
            logger.error(f"Failed to create database connection: {e}")
            logger.error(f"Connection config: {dict((k, v if k != 'password' else '***') for k, v in self.db_config.items())}")
            raise
    
    def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            conn.close()
            return True
        except Error as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_pending_approval(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chunks with approval_status = 'pending'"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Get chunks pending approval with proper status field
            query = """
            SELECT chunk_id, realm_id, content, complexity_score, 
                   source_document, section_path, chunk_type, 
                   line_count, token_count, access_count,
                   approval_status, created_at, last_accessed,
                   approved_at, approved_by, rejection_reason
            FROM megamind_chunks 
            WHERE approval_status = 'pending'
            ORDER BY created_at DESC, complexity_score ASC 
            LIMIT %s
            """
            
            cursor.execute(query, (limit,))
            chunks = cursor.fetchall()
            
            # Convert datetime objects to strings for JSON serialization
            for chunk in chunks:
                chunk['created_at'] = chunk['created_at'].isoformat() if chunk['created_at'] else None
                chunk['last_accessed'] = chunk['last_accessed'].isoformat() if chunk['last_accessed'] else None
                chunk['approved_at'] = chunk['approved_at'].isoformat() if chunk['approved_at'] else None
                
                # Truncate content for preview
                if chunk['content'] and len(chunk['content']) > 500:
                    chunk['content_preview'] = chunk['content'][:500] + "..."
                else:
                    chunk['content_preview'] = chunk['content']
            
            cursor.close()
            conn.close()
            return chunks
            
        except Error as e:
            logger.error(f"Database error in get_pending_approval: {e}")
            return []
    
    def approve_chunks(self, chunk_ids: List[str], approved_by: str = "frontend_ui") -> Dict[str, Any]:
        """Approve chunks by updating approval_status to 'approved'"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Update approval status and metadata
            placeholders = ','.join(['%s'] * len(chunk_ids))
            query = f"""
            UPDATE megamind_chunks 
            SET approval_status = 'approved',
                approved_at = CURRENT_TIMESTAMP,
                approved_by = %s,
                updated_at = CURRENT_TIMESTAMP,
                access_count = access_count + 1
            WHERE chunk_id IN ({placeholders})
              AND approval_status = 'pending'
            """
            
            cursor.execute(query, [approved_by] + chunk_ids)
            affected_rows = cursor.rowcount
            
            cursor.close()
            conn.close()
            
            logger.info(f"Approved {affected_rows} chunks by {approved_by}")
            
            return {
                'success': True,
                'approved_count': affected_rows,
                'chunk_ids': chunk_ids,
                'approved_by': approved_by
            }
            
        except Error as e:
            logger.error(f"Database error in approve_chunks: {e}")
            return {
                'success': False,
                'error': str(e),
                'approved_count': 0
            }
    
    def reject_chunks(self, chunk_ids: List[str], rejection_reason: str, rejected_by: str = "frontend_ui") -> Dict[str, Any]:
        """Reject chunks by updating approval_status to 'rejected'"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Update rejection status and reason
            placeholders = ','.join(['%s'] * len(chunk_ids))
            query = f"""
            UPDATE megamind_chunks 
            SET approval_status = 'rejected',
                rejection_reason = %s,
                rejected_by = %s,
                rejected_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE chunk_id IN ({placeholders})
              AND approval_status = 'pending'
            """
            
            cursor.execute(query, [rejection_reason, rejected_by] + chunk_ids)
            affected_rows = cursor.rowcount
            
            cursor.close()
            conn.close()
            
            logger.info(f"Rejected {affected_rows} chunks by {rejected_by}")
            
            return {
                'success': True,
                'rejected_count': affected_rows,
                'chunk_ids': chunk_ids,
                'rejection_reason': rejection_reason
            }
            
        except Error as e:
            logger.error(f"Database error in reject_chunks: {e}")
            return {
                'success': False,
                'error': str(e),
                'rejected_count': 0
            }
    
    def get_chunk_with_context(self, chunk_id: str) -> Dict[str, Any]:
        """Get chunk with surrounding context from same document"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Get the target chunk
            chunk_query = """
            SELECT * FROM megamind_chunks WHERE chunk_id = %s
            """
            cursor.execute(chunk_query, (chunk_id,))
            chunk = cursor.fetchone()
            
            if not chunk:
                return {'error': 'Chunk not found'}
            
            # Get related chunks from same document
            context_query = """
            SELECT chunk_id, section_path, chunk_type, content,
                   approval_status, complexity_score
            FROM megamind_chunks 
            WHERE source_document = %s 
              AND chunk_id != %s
            ORDER BY section_path, chunk_id
            LIMIT 10
            """
            cursor.execute(context_query, (chunk['source_document'], chunk_id))
            context_chunks = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            # Format timestamps
            if chunk['created_at']:
                chunk['created_at'] = chunk['created_at'].isoformat()
            if chunk['last_accessed']:
                chunk['last_accessed'] = chunk['last_accessed'].isoformat()
            if chunk['updated_at']:
                chunk['updated_at'] = chunk['updated_at'].isoformat()
            if chunk['approved_at']:
                chunk['approved_at'] = chunk['approved_at'].isoformat()
            if chunk['rejected_at']:
                chunk['rejected_at'] = chunk['rejected_at'].isoformat()
            
            return {
                'chunk': chunk,
                'context_chunks': context_chunks
            }
            
        except Error as e:
            logger.error(f"Database error in get_chunk_with_context: {e}")
            return {'error': str(e)}
    
    def get_approval_stats(self) -> Dict[str, Any]:
        """Get approval statistics for dashboard"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            stats_query = """
            SELECT 
                approval_status,
                COUNT(*) as count
            FROM megamind_chunks 
            GROUP BY approval_status
            """
            
            cursor.execute(stats_query)
            stats = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            # Convert to dictionary
            stats_dict = {stat['approval_status']: stat['count'] for stat in stats}
            
            return {
                'pending': stats_dict.get('pending', 0),
                'approved': stats_dict.get('approved', 0),
                'rejected': stats_dict.get('rejected', 0),
                'total': sum(stats_dict.values())
            }
            
        except Error as e:
            logger.error(f"Database error in get_approval_stats: {e}")
            return {
                'pending': 0,
                'approved': 0,
                'rejected': 0,
                'total': 0,
                'error': str(e)
            }