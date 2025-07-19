from flask import Flask, render_template, request, jsonify
from core.chunk_service import ChunkService
from core.search_service import SearchService
from core.rejection_tracking import RejectionTracker
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize services with megamind_database
chunk_service = ChunkService(
    host=os.getenv('DB_HOST', 'localhost'),
    port=int(os.getenv('DB_PORT', 3306)),
    database=os.getenv('DB_NAME', 'megamind_database'),
    user=os.getenv('DB_USER', 'dev'),
    password=os.getenv('DB_PASSWORD', '')
)
search_service = SearchService(chunk_service)
rejection_tracker = RejectionTracker(chunk_service)

@app.route('/')
def dashboard():
    """Main dashboard with chunk review interface"""
    return render_template('chunk_review.html')

@app.route('/api/chunks/pending')
def get_pending_chunks():
    """Get chunks pending approval based on approval_status"""
    limit = request.args.get('limit', 50, type=int)
    chunks = chunk_service.get_pending_approval(limit=limit)
    
    return jsonify({
        'success': True,
        'chunks': chunks,
        'count': len(chunks)
    })

@app.route('/api/chunks/stats')
def get_chunk_stats():
    """Get approval statistics for dashboard"""
    stats = chunk_service.get_approval_stats()
    return jsonify(stats)

@app.route('/api/chunks/approve', methods=['POST'])
def approve_chunks():
    """Batch approve chunks by updating approval_status"""
    try:
        data = request.get_json()
        chunk_ids = data.get('chunk_ids', [])
        approved_by = data.get('approved_by', 'frontend_ui')
        
        if not chunk_ids:
            return jsonify({
                'success': False,
                'error': 'No chunk IDs provided'
            }), 400
        
        result = chunk_service.approve_chunks(chunk_ids, approved_by)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error approving chunks: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chunks/reject', methods=['POST'])
def reject_chunks():
    """Batch reject chunks with reason"""
    try:
        data = request.get_json()
        chunk_ids = data.get('chunk_ids', [])
        rejection_reason = data.get('rejection_reason', 'Rejected via frontend')
        rejected_by = data.get('rejected_by', 'frontend_ui')
        
        if not chunk_ids:
            return jsonify({
                'success': False,
                'error': 'No chunk IDs provided'
            }), 400
        
        if not rejection_reason.strip():
            return jsonify({
                'success': False,
                'error': 'Rejection reason is required'
            }), 400
        
        result = chunk_service.reject_chunks(chunk_ids, rejection_reason, rejected_by)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error rejecting chunks: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/search/simulate', methods=['POST'])
def simulate_search():
    """Simulate agent search using FULLTEXT search on content"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        limit = data.get('limit', 10)
        
        if not query.strip():
            return jsonify({
                'success': False,
                'error': 'Query is required'
            }), 400
        
        results = search_service.simulate_agent_search(query, limit)
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results),
            'query': query
        })
        
    except Exception as e:
        logger.error(f"Error simulating search: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/search/unified', methods=['POST'])
def unified_search():
    """Unified search endpoint supporting multiple search types"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        search_type = data.get('search_type', 'simulate')
        limit = data.get('limit', 10)
        
        if not query.strip():
            return jsonify({
                'success': False,
                'error': 'Query is required'
            }), 400
        
        # Route to appropriate search method based on type
        if search_type == 'simulate':
            results = search_service.simulate_agent_search(query, limit)
        elif search_type == 'hybrid':
            results = search_service.hybrid_search(query, limit)
        elif search_type == 'semantic':
            results = search_service.semantic_search(query, limit)
        elif search_type == 'keyword':
            results = search_service.keyword_search(query, limit)
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported search type: {search_type}'
            }), 400
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results),
            'query': query,
            'search_type': search_type
        })
        
    except Exception as e:
        logger.error(f"Error in unified search: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/search/realm', methods=['POST'])
def search_by_realm():
    """Search chunks within a specific realm"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        realm_id = data.get('realm_id', '')
        limit = data.get('limit', 10)
        
        if not query.strip():
            return jsonify({
                'success': False,
                'error': 'Query is required'
            }), 400
        
        if not realm_id.strip():
            return jsonify({
                'success': False,
                'error': 'Realm ID is required'
            }), 400
        
        results = search_service.search_by_realm(query, realm_id, limit)
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results),
            'query': query,
            'realm_id': realm_id
        })
        
    except Exception as e:
        logger.error(f"Error searching by realm: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chunks/<chunk_id>/context')
def get_chunk_context(chunk_id):
    """Get chunk with surrounding context for boundary visualization"""
    try:
        chunk_data = chunk_service.get_chunk_with_context(chunk_id)
        
        if 'error' in chunk_data:
            return jsonify({
                'success': False,
                'error': chunk_data['error']
            }), 404
        
        return jsonify({
            'success': True,
            'data': chunk_data
        })
        
    except Exception as e:
        logger.error(f"Error getting chunk context: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chunks/<chunk_id>/relationships')
def get_chunk_relationships(chunk_id):
    """Get related chunks for a specific chunk"""
    try:
        relationships = search_service.get_chunk_relationships(chunk_id)
        
        return jsonify({
            'success': True,
            'relationships': relationships,
            'count': len(relationships),
            'chunk_id': chunk_id
        })
        
    except Exception as e:
        logger.error(f"Error getting chunk relationships: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/search/recent')
def get_recent_searches():
    """Get recently accessed chunks"""
    try:
        limit = request.args.get('limit', 10, type=int)
        results = search_service.get_recent_searches(limit)
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error getting recent searches: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        db_connected = chunk_service.test_connection()
        
        if db_connected:
            # Get basic stats to ensure database is accessible
            stats = chunk_service.get_approval_stats()
            
            return jsonify({
                'status': 'healthy',
                'database': 'connected',
                'stats': stats
            })
        else:
            return jsonify({
                'status': 'unhealthy',
                'database': 'disconnected',
                'error': 'Database connection failed'
            }), 503
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

@app.route('/api/config')
def get_config():
    """Get current configuration"""
    return jsonify({
        'database': {
            'host': chunk_service.db_config['host'],
            'port': chunk_service.db_config['port'],
            'database': chunk_service.db_config['database'],
            'user': chunk_service.db_config['user']
        },
        'environment': {
            'DB_HOST': os.getenv('DB_HOST', 'localhost'),
            'DB_PORT': os.getenv('DB_PORT', '3306'),
            'DB_NAME': os.getenv('DB_NAME', 'megamind_database'),
            'ENVIRONMENT': os.getenv('ENVIRONMENT', 'development')
        }
    })

# Rejection tracking endpoints
@app.route('/api/rejection/templates')
def get_rejection_templates():
    """Get rejection reason templates"""
    try:
        templates = rejection_tracker.get_rejection_templates()
        return jsonify({
            'success': True,
            'templates': templates
        })
    except Exception as e:
        logger.error(f"Error getting rejection templates: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rejection/patterns')
def get_rejection_patterns():
    """Get rejection pattern analysis"""
    try:
        days = request.args.get('days', 30, type=int)
        patterns = rejection_tracker.analyze_rejection_patterns(days)
        
        return jsonify({
            'success': True,
            'patterns': patterns
        })
    except Exception as e:
        logger.error(f"Error getting rejection patterns: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rejection/statistics')
def get_rejection_statistics():
    """Get comprehensive rejection statistics"""
    try:
        days = request.args.get('days', 30, type=int)
        statistics = rejection_tracker.get_rejection_statistics(days)
        
        return jsonify({
            'success': True,
            'statistics': statistics
        })
    except Exception as e:
        logger.error(f"Error getting rejection statistics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rejection/track', methods=['POST'])
def track_rejection_resolution():
    """Track how a rejection was resolved"""
    try:
        data = request.get_json()
        chunk_id = data.get('chunk_id')
        resolution_action = data.get('resolution_action')
        resolution_notes = data.get('resolution_notes', '')
        
        if not chunk_id or not resolution_action:
            return jsonify({
                'success': False,
                'error': 'chunk_id and resolution_action are required'
            }), 400
        
        result = rejection_tracker.track_rejection_resolution(
            chunk_id, resolution_action, resolution_notes
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error tracking rejection resolution: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rejection/export')
def export_rejection_data():
    """Export rejection data for analysis"""
    try:
        days = request.args.get('days', 30, type=int)
        format_type = request.args.get('format', 'json')
        
        result = rejection_tracker.export_rejection_data(days, format_type)
        
        if result['success']:
            if format_type == 'csv':
                from flask import Response
                return Response(
                    result['data'],
                    mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename=rejection_data_{days}d.csv'}
                )
            else:
                return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error exporting rejection data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Chunk management endpoints for modal actions
@app.route('/api/chunks/<chunk_id>/toggle-realm', methods=['POST'])
def toggle_chunk_realm(chunk_id):
    """Toggle chunk between GLOBAL and project realm"""
    try:
        data = request.get_json()
        justification = data.get('justification', 'Realm toggled via frontend interface')
        action_by = data.get('action_by', 'frontend_user')
        
        result = chunk_service.toggle_realm_promotion(chunk_id, justification, action_by)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error toggling realm for chunk {chunk_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chunks/<chunk_id>/toggle-approval', methods=['POST'])
def toggle_chunk_approval(chunk_id):
    """Toggle chunk approval status between approved/pending"""
    try:
        data = request.get_json()
        action_by = data.get('action_by', 'frontend_user')
        reason = data.get('reason', 'Status changed via frontend')
        
        result = chunk_service.toggle_approval_status(chunk_id, action_by, reason)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error toggling approval for chunk {chunk_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chunks/<chunk_id>/delete', methods=['DELETE'])
def delete_chunk(chunk_id):
    """Delete chunk from database"""
    try:
        data = request.get_json() if request.get_json() else {}
        deleted_by = data.get('deleted_by', 'frontend_user')
        reason = data.get('reason', 'Deleted via frontend interface')
        
        result = chunk_service.delete_chunk(chunk_id, deleted_by, reason)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error deleting chunk {chunk_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configure logging level
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    # Test database connection on startup
    if chunk_service.test_connection():
        logger.info("Database connection successful")
    else:
        logger.error("Database connection failed - service may not work properly")
    
    # Run the application
    debug_mode = os.getenv('ENVIRONMENT', 'development') == 'development'
    app.run(host='0.0.0.0', port=5004, debug=debug_mode)