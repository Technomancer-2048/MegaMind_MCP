#!/usr/bin/env python3
"""
MegaMind Context Database - Analytics Dashboard
Phase 2: Intelligence Layer

Flask-based web dashboard for monitoring usage patterns, relationship visualization,
and context efficiency metrics.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import mysql.connector
from mysql.connector import pooling
import numpy as np

from flask import Flask, render_template, jsonify, request
import plotly.graph_objects as go
import plotly.utils
from plotly.subplots import make_subplots
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextAnalytics:
    """Analytics engine for MegaMind context database"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.connection_pool = None
        self._setup_connection_pool()
    
    def _setup_connection_pool(self):
        """Setup MySQL connection pool"""
        try:
            pool_config = {
                'pool_name': 'analytics_pool',
                'pool_size': 3,
                'host': self.db_config['host'],
                'port': int(self.db_config['port']),
                'database': self.db_config['database'],
                'user': self.db_config['user'],
                'password': self.db_config['password'],
                'autocommit': True,
                'charset': 'utf8mb4',
                'use_unicode': True
            }
            self.connection_pool = pooling.MySQLConnectionPool(**pool_config)
            logger.info("Analytics database connection pool established")
        except Exception as e:
            logger.error(f"Failed to setup analytics connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get a connection from the pool"""
        return self.connection_pool.get_connection()
    
    def get_usage_heatmap_data(self) -> Dict[str, Any]:
        """Get data for usage heatmap visualization"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get chunk access patterns
            query = """
            SELECT 
                chunk_id,
                source_document,
                section_path,
                chunk_type,
                access_count,
                last_accessed,
                DATEDIFF(NOW(), last_accessed) as days_since_access,
                token_count
            FROM megamind_chunks
            ORDER BY access_count DESC
            LIMIT 100
            """
            
            cursor.execute(query)
            chunks = cursor.fetchall()
            
            # Categorize chunks by access frequency
            hot_chunks = [c for c in chunks if c['access_count'] > 10]
            warm_chunks = [c for c in chunks if 3 <= c['access_count'] <= 10]
            cold_chunks = [c for c in chunks if c['access_count'] < 3]
            
            return {
                'hot_chunks': hot_chunks,
                'warm_chunks': warm_chunks,
                'cold_chunks': cold_chunks,
                'total_chunks': len(chunks),
                'avg_access_count': sum(c['access_count'] for c in chunks) / len(chunks) if chunks else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get usage heatmap data: {e}")
            return {}
        finally:
            if connection:
                connection.close()
    
    def get_relationship_network_data(self) -> Dict[str, Any]:
        """Get data for relationship network visualization"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get relationships with chunk metadata
            query = """
            SELECT 
                r.chunk_id,
                r.related_chunk_id,
                r.relationship_type,
                r.strength,
                c1.chunk_type as chunk_type_1,
                c2.chunk_type as chunk_type_2,
                c1.source_document as source_1,
                c2.source_document as source_2,
                c1.access_count as access_count_1,
                c2.access_count as access_count_2
            FROM megamind_chunk_relationships r
            JOIN megamind_chunks c1 ON r.chunk_id = c1.chunk_id
            JOIN megamind_chunks c2 ON r.related_chunk_id = c2.chunk_id
            WHERE r.strength > 0.6
            ORDER BY r.strength DESC
            LIMIT 200
            """
            
            cursor.execute(query)
            relationships = cursor.fetchall()
            
            # Build node and edge lists for network visualization
            nodes = {}
            edges = []
            
            for rel in relationships:
                # Add nodes
                for chunk_id, chunk_type, source, access_count in [
                    (rel['chunk_id'], rel['chunk_type_1'], rel['source_1'], rel['access_count_1']),
                    (rel['related_chunk_id'], rel['chunk_type_2'], rel['source_2'], rel['access_count_2'])
                ]:
                    if chunk_id not in nodes:
                        nodes[chunk_id] = {
                            'id': chunk_id,
                            'type': chunk_type,
                            'source': source,
                            'access_count': access_count,
                            'size': min(50, max(10, access_count * 2))  # Node size based on access
                        }
                
                # Add edge
                edges.append({
                    'source': rel['chunk_id'],
                    'target': rel['related_chunk_id'],
                    'type': rel['relationship_type'],
                    'strength': rel['strength']
                })
            
            return {
                'nodes': list(nodes.values()),
                'edges': edges,
                'relationship_types': list(set(rel['relationship_type'] for rel in relationships))
            }
            
        except Exception as e:
            logger.error(f"Failed to get relationship network data: {e}")
            return {'nodes': [], 'edges': [], 'relationship_types': []}
        finally:
            if connection:
                connection.close()
    
    def get_search_pattern_data(self) -> Dict[str, Any]:
        """Get search pattern analysis data"""
        # Note: This would require tracking search queries, which we'll simulate for now
        # In production, you'd want to log search queries and analyze them
        
        return {
            'popular_queries': [
                {'query': 'database triggers', 'count': 45, 'avg_results': 8},
                {'query': 'mcp functions', 'count': 32, 'avg_results': 12},
                {'query': 'authentication', 'count': 28, 'avg_results': 6},
                {'query': 'error handling', 'count': 24, 'avg_results': 9},
                {'query': 'configuration', 'count': 19, 'avg_results': 7}
            ],
            'query_types': {
                'technical': 65,
                'procedural': 25,
                'troubleshooting': 10
            }
        }
    
    def get_efficiency_metrics(self) -> Dict[str, Any]:
        """Calculate context efficiency metrics"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get chunk statistics
            stats_query = """
            SELECT 
                COUNT(*) as total_chunks,
                AVG(token_count) as avg_tokens_per_chunk,
                SUM(token_count) as total_tokens,
                AVG(access_count) as avg_access_count,
                MAX(access_count) as max_access_count,
                COUNT(DISTINCT source_document) as total_documents
            FROM megamind_chunks
            WHERE token_count > 0
            """
            
            cursor.execute(stats_query)
            stats = cursor.fetchone()
            
            # Calculate efficiency metrics
            # Traditional markdown loading would load entire documents
            # We estimate 70-80% reduction based on selective chunk retrieval
            traditional_tokens_per_query = stats['total_tokens'] / stats['total_documents'] if stats['total_documents'] > 0 else 0
            selective_tokens_per_query = stats['avg_tokens_per_chunk'] * 5  # Assume 5 chunks per query
            
            efficiency_ratio = 1 - (selective_tokens_per_query / traditional_tokens_per_query) if traditional_tokens_per_query > 0 else 0
            
            return {
                'total_chunks': stats['total_chunks'],
                'total_documents': stats['total_documents'],
                'total_tokens': stats['total_tokens'],
                'avg_tokens_per_chunk': round(stats['avg_tokens_per_chunk'], 1),
                'traditional_tokens_per_query': round(traditional_tokens_per_query, 0),
                'selective_tokens_per_query': round(selective_tokens_per_query, 0),
                'efficiency_ratio': round(efficiency_ratio * 100, 1),  # As percentage
                'avg_access_count': round(stats['avg_access_count'], 1),
                'max_access_count': stats['max_access_count']
            }
            
        except Exception as e:
            logger.error(f"Failed to get efficiency metrics: {e}")
            return {}
        finally:
            if connection:
                connection.close()
    
    def get_tag_distribution(self) -> Dict[str, Any]:
        """Get tag distribution data"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            query = """
            SELECT 
                tag_type,
                tag_value,
                COUNT(*) as chunk_count,
                AVG(confidence) as avg_confidence
            FROM megamind_chunk_tags
            GROUP BY tag_type, tag_value
            ORDER BY tag_type, chunk_count DESC
            """
            
            cursor.execute(query)
            tags = cursor.fetchall()
            
            # Group by tag type
            tag_distribution = {}
            for tag in tags:
                tag_type = tag['tag_type']
                if tag_type not in tag_distribution:
                    tag_distribution[tag_type] = []
                
                tag_distribution[tag_type].append({
                    'value': tag['tag_value'],
                    'count': tag['chunk_count'],
                    'confidence': round(tag['avg_confidence'], 2)
                })
            
            return tag_distribution
            
        except Exception as e:
            logger.error(f"Failed to get tag distribution: {e}")
            return {}
        finally:
            if connection:
                connection.close()

# Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-in-production')

# Initialize analytics engine
analytics = None

def init_analytics():
    """Initialize analytics engine with database configuration"""
    global analytics
    
    db_config = {
        'host': os.getenv('MEGAMIND_DB_HOST', '10.255.250.22'),
        'port': os.getenv('MEGAMIND_DB_PORT', '3309'),
        'database': os.getenv('MEGAMIND_DB_NAME', 'megamind_database'),
        'user': os.getenv('MEGAMIND_DB_USER', 'megamind_user'),
        'password': os.getenv('MEGAMIND_DB_PASSWORD', '')
    }
    
    if not db_config['password']:
        logger.error("Database password not configured")
        return False
    
    try:
        analytics = ContextAnalytics(db_config)
        return True
    except Exception as e:
        logger.error(f"Failed to initialize analytics: {e}")
        return False

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/usage-heatmap')
def api_usage_heatmap():
    """API endpoint for usage heatmap data"""
    if not analytics:
        return jsonify({'error': 'Analytics not initialized'}), 500
    
    data = analytics.get_usage_heatmap_data()
    return jsonify(data)

@app.route('/api/relationship-network')
def api_relationship_network():
    """API endpoint for relationship network data"""
    if not analytics:
        return jsonify({'error': 'Analytics not initialized'}), 500
    
    data = analytics.get_relationship_network_data()
    return jsonify(data)

@app.route('/api/search-patterns')
def api_search_patterns():
    """API endpoint for search pattern data"""
    if not analytics:
        return jsonify({'error': 'Analytics not initialized'}), 500
    
    data = analytics.get_search_pattern_data()
    return jsonify(data)

@app.route('/api/efficiency-metrics')
def api_efficiency_metrics():
    """API endpoint for efficiency metrics"""
    if not analytics:
        return jsonify({'error': 'Analytics not initialized'}), 500
    
    data = analytics.get_efficiency_metrics()
    return jsonify(data)

@app.route('/api/tag-distribution')
def api_tag_distribution():
    """API endpoint for tag distribution data"""
    if not analytics:
        return jsonify({'error': 'Analytics not initialized'}), 500
    
    data = analytics.get_tag_distribution()
    return jsonify(data)

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    if not analytics:
        return jsonify({'status': 'error', 'message': 'Analytics not initialized'}), 500
    
    try:
        # Test database connection
        connection = analytics.get_connection()
        connection.close()
        return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    if init_analytics():
        logger.info("Starting MegaMind Analytics Dashboard...")
        app.run(
            host='0.0.0.0',
            port=int(os.getenv('DASHBOARD_PORT', 5000)),
            debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
        )
    else:
        logger.error("Failed to initialize analytics dashboard")
        exit(1)