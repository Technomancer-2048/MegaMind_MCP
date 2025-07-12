#!/usr/bin/env python3
"""
MegaMind Change Review Interface
Phase 3: Bidirectional Flow

Web-based interface for reviewing and approving AI-generated knowledge changes
with impact assessment and selective approval capabilities.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, request, jsonify, redirect, url_for
import mysql.connector
from mysql.connector import pooling

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChangeReviewManager:
    """Manages change review operations and database interactions"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.config = db_config
        self.connection_pool = None
        self._setup_connection_pool()
    
    def _setup_connection_pool(self):
        """Setup MySQL connection pool"""
        try:
            pool_config = {
                'pool_name': 'megamind_review_pool',
                'pool_size': 5,
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
            logger.info("Review interface database connection pool established")
        except Exception as e:
            logger.error(f"Failed to setup database connection pool: {e}")
            raise
    
    def get_pending_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions with pending changes"""
        connection = None
        try:
            connection = self.connection_pool.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT sm.session_id, sm.user_context, sm.project_context, 
                       sm.start_timestamp, sm.last_activity, sm.pending_changes_count,
                       COUNT(sc.change_id) as total_changes,
                       SUM(CASE WHEN sc.impact_score > 0.7 THEN 1 ELSE 0 END) as critical_changes,
                       SUM(CASE WHEN sc.impact_score BETWEEN 0.3 AND 0.7 THEN 1 ELSE 0 END) as important_changes,
                       AVG(sc.impact_score) as avg_impact_score
                FROM megamind_session_metadata sm
                LEFT JOIN megamind_session_changes sc ON sm.session_id = sc.session_id AND sc.status = 'pending'
                WHERE sm.pending_changes_count > 0
                GROUP BY sm.session_id
                ORDER BY sm.last_activity DESC
            """)
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    'session_id': row['session_id'],
                    'user_context': row['user_context'],
                    'project_context': row['project_context'],
                    'start_timestamp': row['start_timestamp'].isoformat(),
                    'last_activity': row['last_activity'].isoformat(),
                    'pending_changes_count': row['pending_changes_count'],
                    'total_changes': row['total_changes'] or 0,
                    'critical_changes': row['critical_changes'] or 0,
                    'important_changes': row['important_changes'] or 0,
                    'avg_impact_score': float(row['avg_impact_score'] or 0.0)
                })
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get pending sessions: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def get_session_changes(self, session_id: str) -> Dict[str, Any]:
        """Get detailed changes for a specific session"""
        connection = None
        try:
            connection = self.connection_pool.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get session metadata
            cursor.execute("""
                SELECT * FROM megamind_session_metadata WHERE session_id = %s
            """, (session_id,))
            
            session = cursor.fetchone()
            if not session:
                return {"error": "Session not found"}
            
            # Get all pending changes with chunk details
            cursor.execute("""
                SELECT sc.change_id, sc.change_type, sc.chunk_id, sc.target_chunk_id,
                       sc.change_data, sc.impact_score, sc.timestamp, sc.status,
                       mc.source_document, mc.section_path, mc.access_count, mc.chunk_type,
                       mc2.source_document as target_source, mc2.section_path as target_section
                FROM megamind_session_changes sc
                LEFT JOIN megamind_chunks mc ON sc.chunk_id = mc.chunk_id
                LEFT JOIN megamind_chunks mc2 ON sc.target_chunk_id = mc2.chunk_id
                WHERE sc.session_id = %s AND sc.status = 'pending'
                ORDER BY sc.impact_score DESC, sc.timestamp ASC
            """, (session_id,))
            
            changes = []
            for row in cursor.fetchall():
                change_data = json.loads(row['change_data'])
                
                change = {
                    'change_id': row['change_id'],
                    'change_type': row['change_type'],
                    'chunk_id': row['chunk_id'],
                    'target_chunk_id': row['target_chunk_id'],
                    'impact_score': float(row['impact_score']),
                    'timestamp': row['timestamp'].isoformat(),
                    'status': row['status'],
                    'source_document': row['source_document'],
                    'section_path': row['section_path'],
                    'access_count': row['access_count'] if row['access_count'] else 0,
                    'chunk_type': row['chunk_type'],
                    'change_data': change_data,
                    'priority_level': self._get_priority_level(float(row['impact_score'])),
                    'diff_preview': self._generate_diff_preview(row['change_type'], change_data)
                }
                
                if row['target_chunk_id']:
                    change['target_source_document'] = row['target_source']
                    change['target_section_path'] = row['target_section']
                
                changes.append(change)
            
            return {
                'session': {
                    'session_id': session['session_id'],
                    'user_context': session['user_context'],
                    'project_context': session['project_context'],
                    'start_timestamp': session['start_timestamp'].isoformat(),
                    'last_activity': session['last_activity'].isoformat(),
                    'pending_changes_count': session['pending_changes_count']
                },
                'changes': changes,
                'summary': self._generate_change_summary(changes)
            }
            
        except Exception as e:
            logger.error(f"Failed to get session changes: {e}")
            return {"error": str(e)}
        finally:
            if connection:
                connection.close()
    
    def _get_priority_level(self, impact_score: float) -> str:
        """Determine priority level based on impact score"""
        if impact_score > 0.7:
            return "critical"
        elif impact_score >= 0.3:
            return "important"
        else:
            return "standard"
    
    def _generate_diff_preview(self, change_type: str, change_data: Dict[str, Any]) -> str:
        """Generate a preview of the change for display"""
        if change_type == 'update':
            original = change_data.get('original_content', '')[:100]
            new = change_data.get('new_content', '')[:100]
            return f"Modified content: {original}... → {new}..."
        elif change_type == 'create':
            content = change_data.get('content', '')[:100]
            return f"New chunk: {content}..."
        elif change_type == 'relate':
            rel_type = change_data.get('relationship_type', 'unknown')
            return f"Added {rel_type} relationship"
        else:
            return f"Unknown change type: {change_type}"
    
    def _generate_change_summary(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for changes"""
        total = len(changes)
        critical = sum(1 for c in changes if c['priority_level'] == 'critical')
        important = sum(1 for c in changes if c['priority_level'] == 'important')
        standard = sum(1 for c in changes if c['priority_level'] == 'standard')
        
        by_type = {}
        for change in changes:
            change_type = change['change_type']
            by_type[change_type] = by_type.get(change_type, 0) + 1
        
        return {
            'total': total,
            'priority_breakdown': {
                'critical': critical,
                'important': important,
                'standard': standard
            },
            'by_type': by_type,
            'avg_impact': sum(c['impact_score'] for c in changes) / total if total > 0 else 0.0
        }
    
    def approve_changes(self, session_id: str, approved_change_ids: List[str]) -> Dict[str, Any]:
        """Approve specific changes and commit them"""
        connection = None
        try:
            connection = self.connection_pool.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Start transaction
            connection.start_transaction()
            
            committed_changes = {
                "chunks_modified": 0,
                "chunks_created": 0,
                "relationships_added": 0,
                "tags_added": 0
            }
            
            rollback_data = []
            
            for change_id in approved_change_ids:
                # Get change details
                cursor.execute("""
                    SELECT * FROM megamind_session_changes 
                    WHERE change_id = %s AND session_id = %s AND status = 'pending'
                """, (change_id, session_id))
                
                change = cursor.fetchone()
                if not change:
                    continue
                
                change_data = json.loads(change['change_data'])
                
                if change['change_type'] == 'update':
                    # Update existing chunk
                    cursor.execute("""
                        SELECT * FROM megamind_chunks WHERE chunk_id = %s
                    """, (change['chunk_id'],))
                    original_chunk = cursor.fetchone()
                    
                    if original_chunk:
                        rollback_data.append({
                            "type": "update",
                            "chunk_id": change['chunk_id'],
                            "original_content": original_chunk['content']
                        })
                        
                        cursor.execute("""
                            UPDATE megamind_chunks 
                            SET content = %s, last_modified = CURRENT_TIMESTAMP 
                            WHERE chunk_id = %s
                        """, (change_data['new_content'], change['chunk_id']))
                        
                        committed_changes["chunks_modified"] += 1
                
                elif change['change_type'] == 'create':
                    # Create new chunk
                    import uuid
                    new_chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
                    
                    cursor.execute("""
                        INSERT INTO megamind_chunks 
                        (chunk_id, content, source_document, section_path, chunk_type, 
                         line_count, token_count, created_at, last_modified)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """, (
                        new_chunk_id, change_data['content'], change_data['source_document'],
                        change_data['section_path'], change_data['chunk_type'],
                        change_data['line_count'], change_data['token_count']
                    ))
                    
                    rollback_data.append({
                        "type": "create",
                        "chunk_id": new_chunk_id
                    })
                    
                    committed_changes["chunks_created"] += 1
                
                elif change['change_type'] == 'relate':
                    # Add relationship
                    cursor.execute("""
                        INSERT INTO megamind_chunk_relationships 
                        (chunk_id, related_chunk_id, relationship_type, strength, discovered_by)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        change_data['chunk_id_1'], change_data['chunk_id_2'],
                        change_data['relationship_type'], change_data['strength'],
                        change_data['discovered_by']
                    ))
                    
                    rollback_data.append({
                        "type": "relate",
                        "chunk_id_1": change_data['chunk_id_1'],
                        "chunk_id_2": change_data['chunk_id_2'],
                        "relationship_type": change_data['relationship_type']
                    })
                    
                    committed_changes["relationships_added"] += 1
                
                # Mark change as approved
                cursor.execute("""
                    UPDATE megamind_session_changes 
                    SET status = 'approved' 
                    WHERE change_id = %s
                """, (change_id,))
            
            # Create contribution record
            import uuid
            contribution_id = f"contrib_{uuid.uuid4().hex[:12]}"
            cursor.execute("""
                INSERT INTO megamind_knowledge_contributions 
                (contribution_id, session_id, chunks_modified, chunks_created, 
                 relationships_added, tags_added, rollback_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                contribution_id, session_id, committed_changes["chunks_modified"],
                committed_changes["chunks_created"], committed_changes["relationships_added"],
                committed_changes["tags_added"], json.dumps(rollback_data)
            ))
            
            # Update session metadata
            cursor.execute("""
                UPDATE megamind_session_metadata 
                SET pending_changes_count = pending_changes_count - %s 
                WHERE session_id = %s
            """, (len(approved_change_ids), session_id))
            
            connection.commit()
            
            return {
                "success": True,
                "contribution_id": contribution_id,
                "changes_committed": committed_changes,
                "total_approved": len(approved_change_ids)
            }
            
        except Exception as e:
            logger.error(f"Failed to approve changes: {e}")
            if connection:
                connection.rollback()
            return {"success": False, "error": str(e)}
        finally:
            if connection:
                connection.close()
    
    def reject_changes(self, session_id: str, rejected_change_ids: List[str]) -> Dict[str, Any]:
        """Reject specific changes"""
        connection = None
        try:
            connection = self.connection_pool.get_connection()
            cursor = connection.cursor()
            
            # Mark changes as rejected
            for change_id in rejected_change_ids:
                cursor.execute("""
                    UPDATE megamind_session_changes 
                    SET status = 'rejected' 
                    WHERE change_id = %s AND session_id = %s AND status = 'pending'
                """, (change_id, session_id))
            
            # Update session pending count
            cursor.execute("""
                UPDATE megamind_session_metadata 
                SET pending_changes_count = pending_changes_count - %s 
                WHERE session_id = %s
            """, (len(rejected_change_ids), session_id))
            
            connection.commit()
            
            return {
                "success": True,
                "total_rejected": len(rejected_change_ids)
            }
            
        except Exception as e:
            logger.error(f"Failed to reject changes: {e}")
            if connection:
                connection.rollback()
            return {"success": False, "error": str(e)}
        finally:
            if connection:
                connection.close()

# Flask Application
app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)

# Global review manager
review_manager = None

def init_review_manager():
    """Initialize the review manager with database configuration"""
    global review_manager
    
    db_config = {
        'host': os.getenv('MEGAMIND_DB_HOST', '10.255.250.22'),
        'port': os.getenv('MEGAMIND_DB_PORT', '3309'),
        'database': os.getenv('MEGAMIND_DB_NAME', 'megamind_database'),
        'user': os.getenv('MEGAMIND_DB_USER', 'megamind_user'),
        'password': os.getenv('MEGAMIND_DB_PASSWORD', ''),
    }
    
    if not db_config['password']:
        raise Exception("Database password not configured. Set MEGAMIND_DB_PASSWORD environment variable.")
    
    review_manager = ChangeReviewManager(db_config)

@app.route('/')
def index():
    """Main dashboard showing all pending sessions"""
    try:
        sessions = review_manager.get_pending_sessions()
        return render_template('review_dashboard.html', sessions=sessions)
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        return render_template('error.html', error=str(e)), 500

@app.route('/session/<session_id>')
def session_review(session_id):
    """Detailed review page for a specific session"""
    try:
        session_data = review_manager.get_session_changes(session_id)
        
        if 'error' in session_data:
            return render_template('error.html', error=session_data['error']), 404
        
        return render_template('session_review.html', 
                             session=session_data['session'],
                             changes=session_data['changes'],
                             summary=session_data['summary'])
    except Exception as e:
        logger.error(f"Error loading session review: {e}")
        return render_template('error.html', error=str(e)), 500

@app.route('/api/approve', methods=['POST'])
def approve_changes():
    """API endpoint to approve changes"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        approved_changes = data.get('approved_changes', [])
        
        result = review_manager.approve_changes(session_id, approved_changes)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error approving changes: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/reject', methods=['POST'])
def reject_changes():
    """API endpoint to reject changes"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        rejected_changes = data.get('rejected_changes', [])
        
        result = review_manager.reject_changes(session_id, rejected_changes)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error rejecting changes: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/session/<session_id>/refresh')
def refresh_session(session_id):
    """API endpoint to refresh session data"""
    try:
        session_data = review_manager.get_session_changes(session_id)
        return jsonify(session_data)
    except Exception as e:
        logger.error(f"Error refreshing session: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        init_review_manager()
        port = int(os.getenv('REVIEW_PORT', 5001))
        debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
        
        logger.info(f"Starting MegaMind Change Review Interface on port {port}")
        app.run(host='0.0.0.0', port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Failed to start review interface: {e}")
        exit(1)