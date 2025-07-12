#!/usr/bin/env python3
"""
Realm Performance Dashboard System for MegaMind Context Database
Provides web-based dashboards for monitoring realm performance and health
"""

import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
try:
    from flask import Flask, render_template, jsonify, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class DashboardWidget:
    """Represents a dashboard widget configuration"""
    widget_id: str
    widget_type: str  # 'chart', 'metric', 'table', 'status'
    title: str
    data_source: str
    config: Dict[str, Any]
    position: Dict[str, int]  # x, y, width, height
    refresh_interval: int = 300  # seconds

@dataclass
class DashboardConfig:
    """Represents a dashboard configuration"""
    dashboard_id: str
    name: str
    description: str
    realm_id: Optional[str]
    widgets: List[DashboardWidget]
    layout: Dict[str, Any]
    created_by: str
    created_at: datetime
    is_public: bool = False

class RealmDashboard:
    """Dashboard system for realm performance monitoring"""
    
    def __init__(self, db_connection, config: Dict[str, Any] = None):
        self.db = db_connection
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.app = Flask(__name__)
        CORS(self.app)
        self._setup_routes()
    
    # ===================================================================
    # Data Collection for Dashboards
    # ===================================================================
    
    def get_realm_overview_data(self, realm_id: Optional[str] = None) -> Dict[str, Any]:
        """Get overview data for realm dashboard"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            if realm_id:
                # Single realm overview
                overview_query = """
                SELECT 
                    r.realm_id,
                    r.realm_name,
                    r.realm_type,
                    COUNT(c.chunk_id) as total_chunks,
                    COUNT(CASE WHEN c.created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY) THEN 1 END) as chunks_last_7_days,
                    COUNT(CASE WHEN c.last_accessed >= DATE_SUB(NOW(), INTERVAL 1 DAY) THEN 1 END) as chunks_accessed_today,
                    AVG(c.access_count) as avg_access_count,
                    MAX(c.last_accessed) as last_activity,
                    COUNT(DISTINCT s.session_id) as active_sessions,
                    COUNT(DISTINCT cr.relationship_id) as total_relationships
                FROM megamind_realms r
                LEFT JOIN megamind_chunks c ON r.realm_id = c.realm_id
                LEFT JOIN megamind_session_metadata s ON r.realm_id = s.realm_id AND s.is_active = TRUE
                LEFT JOIN megamind_chunk_relationships cr ON c.chunk_id = cr.chunk_id
                WHERE r.realm_id = %s AND r.is_active = TRUE
                GROUP BY r.realm_id, r.realm_name, r.realm_type
                """
                cursor.execute(overview_query, (realm_id,))
                result = cursor.fetchone()
                
                if result:
                    return {
                        "realm_info": result,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {"error": "Realm not found"}
            else:
                # Multi-realm overview
                overview_query = """
                SELECT 
                    r.realm_id,
                    r.realm_name,
                    r.realm_type,
                    COUNT(c.chunk_id) as total_chunks,
                    AVG(c.access_count) as avg_access_count,
                    COUNT(DISTINCT s.session_id) as active_sessions
                FROM megamind_realms r
                LEFT JOIN megamind_chunks c ON r.realm_id = c.realm_id
                LEFT JOIN megamind_session_metadata s ON r.realm_id = s.realm_id AND s.is_active = TRUE
                WHERE r.is_active = TRUE
                GROUP BY r.realm_id, r.realm_name, r.realm_type
                ORDER BY total_chunks DESC
                """
                cursor.execute(overview_query)
                realms = cursor.fetchall()
                
                return {
                    "realms": realms,
                    "total_realms": len(realms),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get realm overview data: {e}")
            return {"error": str(e)}
        finally:
            if cursor:
                cursor.close()
    
    def get_usage_trends_data(self, realm_id: str, days: int = 30) -> Dict[str, Any]:
        """Get usage trends data for charting"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Daily chunk access trends
            trends_query = """
            SELECT 
                DATE(period_start) as date,
                metric_type,
                SUM(metric_value) as value
            FROM megamind_realm_metrics
            WHERE realm_id = %s 
            AND measurement_period = 'daily'
            AND period_start >= DATE_SUB(NOW(), INTERVAL %s DAY)
            GROUP BY DATE(period_start), metric_type
            ORDER BY date DESC
            """
            
            cursor.execute(trends_query, (realm_id, days))
            trends = cursor.fetchall()
            
            # Organize data by metric type
            trend_data = {}
            for row in trends:
                metric_type = row['metric_type']
                if metric_type not in trend_data:
                    trend_data[metric_type] = []
                trend_data[metric_type].append({
                    "date": row['date'].isoformat(),
                    "value": row['value']
                })
            
            # Access pattern by hour of day
            hourly_query = """
            SELECT 
                HOUR(last_accessed) as hour,
                COUNT(*) as access_count
            FROM megamind_chunks
            WHERE realm_id = %s 
            AND last_accessed >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            GROUP BY HOUR(last_accessed)
            ORDER BY hour
            """
            
            cursor.execute(hourly_query, (realm_id,))
            hourly_access = cursor.fetchall()
            
            return {
                "realm_id": realm_id,
                "trends": trend_data,
                "hourly_access_pattern": hourly_access,
                "period_days": days,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get usage trends data: {e}")
            return {"error": str(e)}
        finally:
            if cursor:
                cursor.close()
    
    def get_subsystem_breakdown_data(self, realm_id: str) -> Dict[str, Any]:
        """Get subsystem breakdown data for pie charts"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Chunk distribution by subsystem
            subsystem_query = """
            SELECT 
                COALESCE(ct.tag_value, 'uncategorized') as subsystem,
                COUNT(c.chunk_id) as chunk_count,
                AVG(c.access_count) as avg_access,
                SUM(c.access_count) as total_access
            FROM megamind_chunks c
            LEFT JOIN megamind_chunk_tags ct ON c.chunk_id = ct.chunk_id AND ct.tag_type = 'subsystem'
            WHERE c.realm_id = %s
            GROUP BY ct.tag_value
            ORDER BY chunk_count DESC
            """
            
            cursor.execute(subsystem_query, (realm_id,))
            subsystems = cursor.fetchall()
            
            # Chunk type distribution
            type_query = """
            SELECT 
                chunk_type,
                COUNT(*) as count,
                AVG(access_count) as avg_access
            FROM megamind_chunks
            WHERE realm_id = %s
            GROUP BY chunk_type
            ORDER BY count DESC
            """
            
            cursor.execute(type_query, (realm_id,))
            chunk_types = cursor.fetchall()
            
            # Relationship type distribution
            relationship_query = """
            SELECT 
                cr.relationship_type,
                COUNT(*) as count,
                AVG(cr.strength) as avg_strength
            FROM megamind_chunk_relationships cr
            JOIN megamind_chunks c ON cr.chunk_id = c.chunk_id
            WHERE c.realm_id = %s
            GROUP BY cr.relationship_type
            ORDER BY count DESC
            """
            
            cursor.execute(relationship_query, (realm_id,))
            relationships = cursor.fetchall()
            
            return {
                "realm_id": realm_id,
                "subsystem_distribution": subsystems,
                "chunk_type_distribution": chunk_types,
                "relationship_type_distribution": relationships,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get subsystem breakdown data: {e}")
            return {"error": str(e)}
        finally:
            if cursor:
                cursor.close()
    
    def get_performance_metrics_data(self, realm_id: str) -> Dict[str, Any]:
        """Get performance metrics for gauges and indicators"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Health score from system health
            health_query = """
            SELECT health_score, status, performance_metrics
            FROM megamind_system_health
            WHERE component_type = 'realm' AND component_name = %s
            ORDER BY last_check DESC
            LIMIT 1
            """
            
            cursor.execute(health_query, (realm_id,))
            health_data = cursor.fetchone()
            
            # Search performance (mock data - would be real in production)
            search_performance = {
                "avg_search_time_ms": 150,
                "cache_hit_rate": 85.5,
                "search_accuracy": 92.3
            }
            
            # Content quality metrics
            quality_query = """
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(CASE WHEN access_count > 1 THEN 1 END) as accessed_chunks,
                COUNT(CASE WHEN access_count > 5 THEN 1 END) as popular_chunks,
                AVG(CASE WHEN complexity_score IS NOT NULL THEN complexity_score END) as avg_complexity,
                COUNT(CASE WHEN last_accessed < DATE_SUB(NOW(), INTERVAL 90 DAY) THEN 1 END) as stale_chunks
            FROM megamind_chunks
            WHERE realm_id = %s
            """
            
            cursor.execute(quality_query, (realm_id,))
            quality_data = cursor.fetchone()
            
            # Calculate derived metrics
            utilization_rate = 0
            popularity_rate = 0
            staleness_rate = 0
            
            if quality_data and quality_data['total_chunks'] > 0:
                utilization_rate = (quality_data['accessed_chunks'] / quality_data['total_chunks']) * 100
                popularity_rate = (quality_data['popular_chunks'] / quality_data['total_chunks']) * 100
                staleness_rate = (quality_data['stale_chunks'] / quality_data['total_chunks']) * 100
            
            return {
                "realm_id": realm_id,
                "health_score": health_data['health_score'] if health_data else 0,
                "health_status": health_data['status'] if health_data else 'unknown',
                "search_performance": search_performance,
                "content_quality": {
                    "total_chunks": quality_data['total_chunks'] if quality_data else 0,
                    "utilization_rate": round(utilization_rate, 1),
                    "popularity_rate": round(popularity_rate, 1),
                    "staleness_rate": round(staleness_rate, 1),
                    "avg_complexity": round(quality_data['avg_complexity'] or 0, 2)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics data: {e}")
            return {"error": str(e)}
        finally:
            if cursor:
                cursor.close()
    
    def get_system_status_data(self) -> Dict[str, Any]:
        """Get overall system status for system dashboard"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # System health components
            system_health_query = """
            SELECT 
                component_type,
                component_name,
                status,
                health_score,
                last_check
            FROM megamind_system_health
            ORDER BY component_type, component_name
            """
            
            cursor.execute(system_health_query)
            system_components = cursor.fetchall()
            
            # Recent alerts
            alerts_query = """
            SELECT 
                component_type,
                severity,
                COUNT(*) as count
            FROM megamind_monitoring_alerts
            WHERE triggered_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
            AND resolved_at IS NULL
            GROUP BY component_type, severity
            ORDER BY severity DESC
            """
            
            cursor.execute(alerts_query)
            alert_summary = cursor.fetchall()
            
            # System metrics
            metrics_query = """
            SELECT 
                SUM(CASE WHEN r.realm_type = 'global' THEN 1 ELSE 0 END) as global_realms,
                SUM(CASE WHEN r.realm_type = 'project' THEN 1 ELSE 0 END) as project_realms,
                COUNT(DISTINCT c.chunk_id) as total_chunks,
                COUNT(DISTINCT cr.relationship_id) as total_relationships,
                COUNT(DISTINCT s.session_id) as active_sessions
            FROM megamind_realms r
            LEFT JOIN megamind_chunks c ON r.realm_id = c.realm_id
            LEFT JOIN megamind_chunk_relationships cr ON c.chunk_id = cr.chunk_id
            LEFT JOIN megamind_session_metadata s ON r.realm_id = s.realm_id AND s.is_active = TRUE
            WHERE r.is_active = TRUE
            """
            
            cursor.execute(metrics_query)
            system_metrics = cursor.fetchone()
            
            # Calculate overall system health
            healthy_components = sum(1 for comp in system_components if comp['status'] == 'healthy')
            total_components = len(system_components)
            system_health_percentage = (healthy_components / total_components * 100) if total_components > 0 else 0
            
            return {
                "system_health": {
                    "overall_percentage": round(system_health_percentage, 1),
                    "healthy_components": healthy_components,
                    "total_components": total_components,
                    "components": system_components
                },
                "alert_summary": alert_summary,
                "system_metrics": system_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system status data: {e}")
            return {"error": str(e)}
        finally:
            if cursor:
                cursor.close()
    
    # ===================================================================
    # Dashboard Configuration Management
    # ===================================================================
    
    def create_dashboard(self, dashboard_config: DashboardConfig) -> str:
        """Create a new dashboard configuration"""
        try:
            cursor = self.db.cursor()
            
            # Store dashboard configuration
            dashboard_query = """
            INSERT INTO megamind_dashboards 
            (dashboard_id, name, description, realm_id, layout_config, created_by, is_public)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(dashboard_query, (
                dashboard_config.dashboard_id,
                dashboard_config.name,
                dashboard_config.description,
                dashboard_config.realm_id,
                json.dumps(asdict(dashboard_config)),
                dashboard_config.created_by,
                dashboard_config.is_public
            ))
            
            self.db.commit()
            return dashboard_config.dashboard_id
            
        except Exception as e:
            self.logger.error(f"Failed to create dashboard: {e}")
            self.db.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
    
    def get_dashboard_config(self, dashboard_id: str) -> Optional[DashboardConfig]:
        """Get dashboard configuration"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            query = """
            SELECT * FROM megamind_dashboards 
            WHERE dashboard_id = %s
            """
            
            cursor.execute(query, (dashboard_id,))
            result = cursor.fetchone()
            
            if result:
                layout_config = json.loads(result['layout_config'])
                return DashboardConfig(**layout_config)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard config: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def get_predefined_dashboards(self) -> Dict[str, DashboardConfig]:
        """Get predefined dashboard configurations"""
        dashboards = {}
        
        # Realm Overview Dashboard
        realm_overview = DashboardConfig(
            dashboard_id="realm_overview",
            name="Realm Overview",
            description="Comprehensive realm performance and health overview",
            realm_id=None,  # Can be applied to any realm
            widgets=[
                DashboardWidget(
                    widget_id="realm_stats",
                    widget_type="metric",
                    title="Realm Statistics",
                    data_source="realm_overview",
                    config={"metrics": ["total_chunks", "avg_access_count", "active_sessions"]},
                    position={"x": 0, "y": 0, "width": 4, "height": 2}
                ),
                DashboardWidget(
                    widget_id="usage_trends",
                    widget_type="chart",
                    title="Usage Trends",
                    data_source="usage_trends",
                    config={"chart_type": "line", "time_period": 30},
                    position={"x": 4, "y": 0, "width": 8, "height": 4}
                ),
                DashboardWidget(
                    widget_id="subsystem_breakdown",
                    widget_type="chart",
                    title="Subsystem Distribution",
                    data_source="subsystem_breakdown",
                    config={"chart_type": "pie"},
                    position={"x": 0, "y": 2, "width": 4, "height": 4}
                ),
                DashboardWidget(
                    widget_id="performance_metrics",
                    widget_type="metric",
                    title="Performance Metrics",
                    data_source="performance_metrics",
                    config={"metrics": ["health_score", "utilization_rate", "search_performance"]},
                    position={"x": 0, "y": 6, "width": 12, "height": 2}
                )
            ],
            layout={"columns": 12, "row_height": 100},
            created_by="system",
            created_at=datetime.now()
        )
        dashboards["realm_overview"] = realm_overview
        
        # System Status Dashboard
        system_status = DashboardConfig(
            dashboard_id="system_status",
            name="System Status",
            description="Overall system health and monitoring",
            realm_id=None,
            widgets=[
                DashboardWidget(
                    widget_id="system_health",
                    widget_type="status",
                    title="System Health",
                    data_source="system_status",
                    config={"status_type": "overall"},
                    position={"x": 0, "y": 0, "width": 6, "height": 3}
                ),
                DashboardWidget(
                    widget_id="component_status",
                    widget_type="table",
                    title="Component Status",
                    data_source="system_status",
                    config={"table_type": "components"},
                    position={"x": 6, "y": 0, "width": 6, "height": 3}
                ),
                DashboardWidget(
                    widget_id="alert_summary",
                    widget_type="chart",
                    title="Active Alerts",
                    data_source="system_status",
                    config={"chart_type": "bar", "metric": "alerts"},
                    position={"x": 0, "y": 3, "width": 12, "height": 3}
                )
            ],
            layout={"columns": 12, "row_height": 100},
            created_by="system",
            created_at=datetime.now()
        )
        dashboards["system_status"] = system_status
        
        return dashboards
    
    # ===================================================================
    # Flask Web Interface
    # ===================================================================
    
    def _setup_routes(self):
        """Setup Flask routes for dashboard API"""
        
        @self.app.route('/')
        def index():
            return """
            <h1>MegaMind Realm Dashboard</h1>
            <p>Available endpoints:</p>
            <ul>
                <li><a href="/api/dashboards">Dashboard List</a></li>
                <li><a href="/api/realms">Realms Overview</a></li>
                <li><a href="/api/system/status">System Status</a></li>
            </ul>
            """
        
        @self.app.route('/api/dashboards')
        def list_dashboards():
            """List available dashboards"""
            predefined = self.get_predefined_dashboards()
            return jsonify({
                "predefined_dashboards": [
                    {
                        "id": dash_id,
                        "name": config.name,
                        "description": config.description
                    }
                    for dash_id, config in predefined.items()
                ],
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/api/dashboard/<dashboard_id>/data')
        def get_dashboard_data(dashboard_id):
            """Get data for a specific dashboard"""
            realm_id = request.args.get('realm_id')
            
            if dashboard_id == "realm_overview":
                if not realm_id:
                    return jsonify({"error": "realm_id required for realm dashboard"}), 400
                
                data = {
                    "overview": self.get_realm_overview_data(realm_id),
                    "trends": self.get_usage_trends_data(realm_id),
                    "breakdown": self.get_subsystem_breakdown_data(realm_id),
                    "performance": self.get_performance_metrics_data(realm_id)
                }
                return jsonify(data)
            
            elif dashboard_id == "system_status":
                data = {
                    "status": self.get_system_status_data()
                }
                return jsonify(data)
            
            else:
                return jsonify({"error": "Dashboard not found"}), 404
        
        @self.app.route('/api/realms')
        def list_realms():
            """List all realms with basic info"""
            data = self.get_realm_overview_data()
            return jsonify(data)
        
        @self.app.route('/api/realm/<realm_id>/overview')
        def realm_overview(realm_id):
            """Get overview data for specific realm"""
            data = self.get_realm_overview_data(realm_id)
            return jsonify(data)
        
        @self.app.route('/api/realm/<realm_id>/trends')
        def realm_trends(realm_id):
            """Get trends data for specific realm"""
            days = int(request.args.get('days', 30))
            data = self.get_usage_trends_data(realm_id, days)
            return jsonify(data)
        
        @self.app.route('/api/realm/<realm_id>/breakdown')
        def realm_breakdown(realm_id):
            """Get breakdown data for specific realm"""
            data = self.get_subsystem_breakdown_data(realm_id)
            return jsonify(data)
        
        @self.app.route('/api/realm/<realm_id>/performance')
        def realm_performance(realm_id):
            """Get performance metrics for specific realm"""
            data = self.get_performance_metrics_data(realm_id)
            return jsonify(data)
        
        @self.app.route('/api/system/status')
        def system_status():
            """Get overall system status"""
            data = self.get_system_status_data()
            return jsonify(data)
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint"""
            try:
                cursor = self.db.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
            except Exception as e:
                return jsonify({"status": "unhealthy", "error": str(e)}), 500
    
    def run_dashboard_server(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the dashboard web server"""
        self.logger.info(f"Starting dashboard server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# ===================================================================
# CLI Interface for Dashboard Management
# ===================================================================

def main():
    """CLI interface for dashboard operations"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="MegaMind Realm Dashboard")
    parser.add_argument('--host', default='0.0.0.0', help='Dashboard server host')
    parser.add_argument('--port', type=int, default=5000, help='Dashboard server port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Database configuration
    db_config = {
        'host': os.getenv('DATABASE_HOST', 'localhost'),
        'port': int(os.getenv('DATABASE_PORT', '3306')),
        'user': os.getenv('DATABASE_USER', 'megamind_user'),
        'password': os.getenv('DATABASE_PASSWORD', ''),
        'database': os.getenv('DATABASE_NAME', 'megamind_database')
    }
    
    try:
        # Connect to database
        db_connection = mysql.connector.connect(**db_config)
        
        # Create dashboard instance
        dashboard = RealmDashboard(db_connection)
        
        # Run dashboard server
        dashboard.run_dashboard_server(args.host, args.port, args.debug)
        
    except Exception as e:
        print(f"Failed to start dashboard: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())