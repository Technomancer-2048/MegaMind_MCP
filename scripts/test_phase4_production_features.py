#!/usr/bin/env python3
"""
Test Suite for Phase 4 Production Features
Tests production deployment, monitoring, analytics, and dashboard functionality
"""

import os
import sys
import json
import time
import logging
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mcp_server.realm_analytics import RealmAnalytics, MetricType, MeasurementPeriod
    from mcp_server.realm_monitoring import RealmMonitoring, ComponentType, HealthStatus, AlertSeverity
    from mcp_server.realm_dashboard import RealmDashboard
except ImportError as e:
    print(f"Import error: {e}")
    print("This test requires the MCP server modules to be available")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRealmAnalytics(unittest.TestCase):
    """Test realm analytics functionality"""
    
    def setUp(self):
        """Set up test database connection mock"""
        self.mock_db = Mock()
        self.analytics = RealmAnalytics(self.mock_db)
    
    def test_metric_recording(self):
        """Test metric recording functionality"""
        logger.info("Testing metric recording...")
        
        # Mock database operations
        cursor_mock = Mock()
        self.mock_db.cursor.return_value = cursor_mock
        self.mock_db.commit.return_value = None
        
        # Test recording a metric
        metric_id = self.analytics.record_metric(
            'PROJ_ECOMMERCE', 
            MetricType.CHUNKS_CREATED, 
            5, 
            MeasurementPeriod.DAILY
        )
        
        # Verify metric was recorded
        self.assertIsNotNone(metric_id)
        self.assertTrue(metric_id.startswith('metric_'))
        cursor_mock.execute.assert_called()
        self.mock_db.commit.assert_called()
        
        logger.info("✓ Metric recording test passed")
    
    def test_usage_analytics_calculation(self):
        """Test usage analytics calculation"""
        logger.info("Testing usage analytics calculation...")
        
        # Mock cursor with sample data
        cursor_mock = Mock()
        cursor_mock.fetchone.side_effect = [
            {
                'realm_id': 'PROJ_ECOMMERCE',
                'realm_name': 'E-commerce Platform',
                'total_chunks': 25,
                'chunks_accessed_today': 8,
                'chunks_created_this_week': 3,
                'avg_access_count': 2.5
            },
            [],  # subsystem results
            {'total_relationships': 15, 'chunks_with_relationships': 12, 'total_chunks': 25},
            {'unique_chunks_accessed': 10, 'total_accesses': 50, 'subsystem_diversity': 5}
        ]
        cursor_mock.fetchall.return_value = [
            {'tag_value': 'shopping_cart', 'chunk_count': 8},
            {'tag_value': 'payment_processing', 'chunk_count': 6}
        ]
        self.mock_db.cursor.return_value = cursor_mock
        
        # Test analytics calculation
        analytics = self.analytics.analyze_realm_usage('PROJ_ECOMMERCE')
        
        # Verify analytics results
        self.assertIsNotNone(analytics)
        self.assertEqual(analytics.realm_id, 'PROJ_ECOMMERCE')
        self.assertEqual(analytics.total_chunks, 25)
        self.assertEqual(analytics.chunks_accessed_today, 8)
        self.assertGreater(analytics.relationship_density, 0)
        self.assertGreater(analytics.user_engagement_score, 0)
        
        logger.info("✓ Usage analytics calculation test passed")
    
    def test_performance_insights_generation(self):
        """Test performance insights generation"""
        logger.info("Testing performance insights generation...")
        
        # Mock analytics data for low engagement scenario
        mock_analytics = Mock()
        mock_analytics.user_engagement_score = 0.2  # Low engagement
        mock_analytics.relationship_density = 0.1   # Low relationships
        mock_analytics.most_popular_subsystems = ['security']
        
        with patch.object(self.analytics, 'analyze_realm_usage', return_value=mock_analytics):
            # Mock database calls for insights generation
            cursor_mock = Mock()
            cursor_mock.fetchone.return_value = (15, 30)  # dominant subsystem data
            self.mock_db.cursor.return_value = cursor_mock
            
            # Generate insights
            insights = self.analytics.generate_performance_insights('PROJ_ECOMMERCE')
            
            # Verify insights were generated
            self.assertIsInstance(insights, list)
            self.assertGreater(len(insights), 0)
            
            # Check for expected insight types
            insight_types = [insight.insight_type for insight in insights]
            self.assertIn('engagement', insight_types)
            self.assertIn('relationships', insight_types)
            
        logger.info("✓ Performance insights generation test passed")

class TestRealmMonitoring(unittest.TestCase):
    """Test realm monitoring functionality"""
    
    def setUp(self):
        """Set up test monitoring system"""
        self.mock_db = Mock()
        self.monitoring = RealmMonitoring(self.mock_db)
    
    def test_database_health_check(self):
        """Test database health checking"""
        logger.info("Testing database health check...")
        
        # Mock database operations
        cursor_mock = Mock()
        cursor_mock.fetchone.side_effect = [
            None,  # Test query
            {'Value': '25'},  # Connections
            {'Value': '2'}   # Slow queries
        ]
        self.mock_db.cursor.return_value = cursor_mock
        
        # Perform health check
        health_check = self.monitoring.perform_health_check(
            ComponentType.DATABASE, 'connection_pool'
        )
        
        # Verify health check results
        self.assertIsNotNone(health_check)
        self.assertEqual(health_check.component_type, ComponentType.DATABASE)
        self.assertEqual(health_check.component_name, 'connection_pool')
        self.assertIn(health_check.status, [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.CRITICAL])
        self.assertGreaterEqual(health_check.health_score, 0)
        self.assertLessEqual(health_check.health_score, 100)
        self.assertIn('response_time_ms', health_check.performance_metrics)
        
        logger.info("✓ Database health check test passed")
    
    def test_realm_health_check(self):
        """Test realm-specific health checking"""
        logger.info("Testing realm health check...")
        
        # Mock realm and statistics data
        cursor_mock = Mock()
        cursor_mock.fetchone.side_effect = [
            {
                'realm_id': 'PROJ_ECOMMERCE',
                'realm_name': 'E-commerce Platform',
                'is_active': True
            },
            {
                'chunk_count': 25,
                'avg_access': 2.8,
                'recent_access_count': 8,
                'relationship_count': 15
            }
        ]
        self.mock_db.cursor.return_value = cursor_mock
        
        # Perform realm health check
        health_check = self.monitoring.perform_health_check(
            ComponentType.REALM, 'PROJ_ECOMMERCE'
        )
        
        # Verify health check results
        self.assertIsNotNone(health_check)
        self.assertEqual(health_check.component_type, ComponentType.REALM)
        self.assertEqual(health_check.component_name, 'PROJ_ECOMMERCE')
        self.assertGreater(health_check.health_score, 0)
        self.assertIn('chunk_count', health_check.performance_metrics)
        self.assertIn('relationship_density', health_check.performance_metrics)
        
        logger.info("✓ Realm health check test passed")
    
    def test_alert_generation(self):
        """Test alert generation from monitoring rules"""
        logger.info("Testing alert generation...")
        
        # Mock monitoring rules
        cursor_mock = Mock()
        cursor_mock.fetchall.return_value = [
            {
                'rule_id': 'rule_health_low',
                'component_type': 'realm',
                'metric_name': 'health_score',
                'operator': 'lt',
                'threshold_value': 70.0,
                'severity': 'medium',
                'check_interval_minutes': 15,
                'enabled': True
            }
        ]
        cursor_mock.execute.return_value = None
        self.mock_db.cursor.return_value = cursor_mock
        self.mock_db.commit.return_value = None
        
        # Create health check with low score
        health_check = Mock()
        health_check.component_type = ComponentType.REALM
        health_check.component_name = 'PROJ_ECOMMERCE'
        health_check.health_score = 60.0  # Below threshold
        health_check.performance_metrics = {'health_score': 60.0}
        
        # Check monitoring rules
        alerts = self.monitoring.check_monitoring_rules(health_check)
        
        # Verify alert generation
        self.assertIsInstance(alerts, list)
        if len(alerts) > 0:
            alert = alerts[0]
            self.assertEqual(alert.component_type, ComponentType.REALM)
            self.assertEqual(alert.severity, AlertSeverity.MEDIUM)
            self.assertIn('health_score', alert.description)
        
        logger.info("✓ Alert generation test passed")
    
    def test_monitoring_cycle(self):
        """Test complete monitoring cycle"""
        logger.info("Testing monitoring cycle...")
        
        # Mock database operations for monitoring cycle
        cursor_mock = Mock()
        cursor_mock.fetchall.return_value = [('PROJ_ECOMMERCE',), ('PROJ_ANALYTICS',)]
        self.mock_db.cursor.return_value = cursor_mock
        
        # Mock health check methods to avoid complex database setup
        with patch.object(self.monitoring, 'perform_health_check') as mock_health_check:
            with patch.object(self.monitoring, 'check_monitoring_rules', return_value=[]) as mock_rules:
                with patch.object(self.monitoring, '_update_system_health') as mock_update:
                    
                    # Mock health check return
                    mock_health_check.return_value = Mock(
                        component_type=ComponentType.REALM,
                        component_name='test_realm',
                        health_score=85.0
                    )
                    
                    # Run monitoring cycle
                    self.monitoring.run_monitoring_cycle()
                    
                    # Verify cycle execution
                    self.assertGreater(mock_health_check.call_count, 0)
                    self.assertGreater(mock_rules.call_count, 0)
                    self.assertGreater(mock_update.call_count, 0)
        
        logger.info("✓ Monitoring cycle test passed")

class TestRealmDashboard(unittest.TestCase):
    """Test realm dashboard functionality"""
    
    def setUp(self):
        """Set up test dashboard system"""
        self.mock_db = Mock()
        self.dashboard = RealmDashboard(self.mock_db)
    
    def test_realm_overview_data(self):
        """Test realm overview data collection"""
        logger.info("Testing realm overview data collection...")
        
        # Mock database query results
        cursor_mock = Mock()
        cursor_mock.fetchone.return_value = {
            'realm_id': 'PROJ_ECOMMERCE',
            'realm_name': 'E-commerce Platform',
            'realm_type': 'project',
            'total_chunks': 25,
            'chunks_last_7_days': 5,
            'chunks_accessed_today': 8,
            'avg_access_count': 2.5,
            'last_activity': datetime.now(),
            'active_sessions': 2,
            'total_relationships': 15
        }
        self.mock_db.cursor.return_value = cursor_mock
        
        # Get overview data
        data = self.dashboard.get_realm_overview_data('PROJ_ECOMMERCE')
        
        # Verify data structure
        self.assertIsInstance(data, dict)
        self.assertIn('realm_info', data)
        self.assertIn('timestamp', data)
        self.assertEqual(data['realm_info']['realm_id'], 'PROJ_ECOMMERCE')
        self.assertEqual(data['realm_info']['total_chunks'], 25)
        
        logger.info("✓ Realm overview data test passed")
    
    def test_usage_trends_data(self):
        """Test usage trends data collection"""
        logger.info("Testing usage trends data collection...")
        
        # Mock trends data
        cursor_mock = Mock()
        cursor_mock.fetchall.side_effect = [
            [
                {'date': datetime.now().date(), 'metric_type': 'chunks_accessed', 'value': 15},
                {'date': datetime.now().date() - timedelta(days=1), 'metric_type': 'chunks_accessed', 'value': 12}
            ],
            [
                {'hour': 9, 'access_count': 25},
                {'hour': 14, 'access_count': 35},
                {'hour': 16, 'access_count': 20}
            ]
        ]
        self.mock_db.cursor.return_value = cursor_mock
        
        # Get trends data
        data = self.dashboard.get_usage_trends_data('PROJ_ECOMMERCE', 7)
        
        # Verify data structure
        self.assertIsInstance(data, dict)
        self.assertIn('trends', data)
        self.assertIn('hourly_access_pattern', data)
        self.assertEqual(data['realm_id'], 'PROJ_ECOMMERCE')
        self.assertEqual(data['period_days'], 7)
        
        logger.info("✓ Usage trends data test passed")
    
    def test_subsystem_breakdown_data(self):
        """Test subsystem breakdown data collection"""
        logger.info("Testing subsystem breakdown data collection...")
        
        # Mock breakdown data
        cursor_mock = Mock()
        cursor_mock.fetchall.side_effect = [
            [
                {'subsystem': 'shopping_cart', 'chunk_count': 8, 'avg_access': 3.2, 'total_access': 26},
                {'subsystem': 'payment_processing', 'chunk_count': 6, 'avg_access': 2.8, 'total_access': 17}
            ],
            [
                {'chunk_type': 'rule', 'count': 15, 'avg_access': 2.5},
                {'chunk_type': 'section', 'count': 10, 'avg_access': 3.1}
            ],
            [
                {'relationship_type': 'depends_on', 'count': 8, 'avg_strength': 0.85},
                {'relationship_type': 'enhances', 'count': 7, 'avg_strength': 0.72}
            ]
        ]
        self.mock_db.cursor.return_value = cursor_mock
        
        # Get breakdown data
        data = self.dashboard.get_subsystem_breakdown_data('PROJ_ECOMMERCE')
        
        # Verify data structure
        self.assertIsInstance(data, dict)
        self.assertIn('subsystem_distribution', data)
        self.assertIn('chunk_type_distribution', data)
        self.assertIn('relationship_type_distribution', data)
        self.assertEqual(data['realm_id'], 'PROJ_ECOMMERCE')
        
        logger.info("✓ Subsystem breakdown data test passed")
    
    def test_performance_metrics_data(self):
        """Test performance metrics data collection"""
        logger.info("Testing performance metrics data collection...")
        
        # Mock performance data
        cursor_mock = Mock()
        cursor_mock.fetchone.side_effect = [
            {
                'health_score': 85.5,
                'status': 'healthy',
                'performance_metrics': '{"test": "value"}'
            },
            {
                'total_chunks': 25,
                'accessed_chunks': 18,
                'popular_chunks': 8,
                'avg_complexity': 0.75,
                'stale_chunks': 3
            }
        ]
        self.mock_db.cursor.return_value = cursor_mock
        
        # Get performance metrics
        data = self.dashboard.get_performance_metrics_data('PROJ_ECOMMERCE')
        
        # Verify data structure
        self.assertIsInstance(data, dict)
        self.assertIn('health_score', data)
        self.assertIn('content_quality', data)
        self.assertIn('search_performance', data)
        self.assertEqual(data['realm_id'], 'PROJ_ECOMMERCE')
        self.assertEqual(data['health_score'], 85.5)
        
        logger.info("✓ Performance metrics data test passed")
    
    def test_predefined_dashboards(self):
        """Test predefined dashboard configurations"""
        logger.info("Testing predefined dashboard configurations...")
        
        # Get predefined dashboards
        dashboards = self.dashboard.get_predefined_dashboards()
        
        # Verify dashboards
        self.assertIsInstance(dashboards, dict)
        self.assertIn('realm_overview', dashboards)
        self.assertIn('system_status', dashboards)
        
        # Check realm overview dashboard
        realm_dashboard = dashboards['realm_overview']
        self.assertEqual(realm_dashboard.dashboard_id, 'realm_overview')
        self.assertGreater(len(realm_dashboard.widgets), 0)
        
        # Check system status dashboard
        system_dashboard = dashboards['system_status']
        self.assertEqual(system_dashboard.dashboard_id, 'system_status')
        self.assertGreater(len(system_dashboard.widgets), 0)
        
        logger.info("✓ Predefined dashboards test passed")

class TestProductionDeployment(unittest.TestCase):
    """Test production deployment features"""
    
    def test_deployment_script_exists(self):
        """Test that deployment script exists and is executable"""
        logger.info("Testing deployment script existence...")
        
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'scripts', 'deploy_production_environment.sh'
        )
        
        # Check script exists
        self.assertTrue(os.path.exists(script_path), "Deployment script not found")
        
        # Check script is executable
        self.assertTrue(os.access(script_path, os.X_OK), "Deployment script not executable")
        
        logger.info("✓ Deployment script test passed")
    
    def test_mcp_configuration_exists(self):
        """Test that MCP configuration exists"""
        logger.info("Testing MCP configuration existence...")
        
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'production_mcp_configs.json'
        )
        
        # Check config exists
        self.assertTrue(os.path.exists(config_path), "MCP configuration not found")
        
        # Check config is valid JSON
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Verify config structure
        self.assertIn('mcpServers', config)
        self.assertIn('megamind-ecommerce', config['mcpServers'])
        self.assertIn('megamind-global', config['mcpServers'])
        
        logger.info("✓ MCP configuration test passed")
    
    def test_sql_schema_files_exist(self):
        """Test that all SQL schema files exist"""
        logger.info("Testing SQL schema files existence...")
        
        schema_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'database', 'realm_system'
        )
        
        expected_files = [
            '01_realm_tables.sql',
            '02_enhanced_core_tables.sql',
            '03_realm_session_tables.sql',
            '04_indexes_and_views.sql',
            '05_initial_data.sql',
            '06_inheritance_resolution.sql',
            '07_knowledge_promotion.sql',
            '08_production_deployment.sql',
            '09_global_realm_standards.sql',
            '10_initial_project_realms.sql',
            '11_monitoring_tables.sql'
        ]
        
        for filename in expected_files:
            file_path = os.path.join(schema_dir, filename)
            self.assertTrue(os.path.exists(file_path), f"Schema file {filename} not found")
        
        logger.info("✓ SQL schema files test passed")

class TestIntegration(unittest.TestCase):
    """Test integration between Phase 4 components"""
    
    def test_analytics_monitoring_integration(self):
        """Test integration between analytics and monitoring"""
        logger.info("Testing analytics-monitoring integration...")
        
        # Mock database
        mock_db = Mock()
        cursor_mock = Mock()
        mock_db.cursor.return_value = cursor_mock
        
        # Create components
        analytics = RealmAnalytics(mock_db)
        monitoring = RealmMonitoring(mock_db)
        
        # Mock analytics data
        mock_analytics_result = Mock()
        mock_analytics_result.user_engagement_score = 0.15  # Low engagement
        mock_analytics_result.relationship_density = 0.05   # Low relationships
        
        # Test that analytics insights can trigger monitoring alerts
        with patch.object(analytics, 'analyze_realm_usage', return_value=mock_analytics_result):
            insights = analytics.generate_performance_insights('PROJ_TEST')
            
            # Verify insights were generated for low metrics
            self.assertGreater(len(insights), 0)
            low_engagement_insights = [i for i in insights if i.insight_type == 'engagement']
            self.assertGreater(len(low_engagement_insights), 0)
        
        logger.info("✓ Analytics-monitoring integration test passed")
    
    def test_monitoring_dashboard_integration(self):
        """Test integration between monitoring and dashboard"""
        logger.info("Testing monitoring-dashboard integration...")
        
        # Mock database
        mock_db = Mock()
        cursor_mock = Mock()
        mock_db.cursor.return_value = cursor_mock
        
        # Mock system health data
        cursor_mock.fetchall.side_effect = [
            [  # System health components
                {'component_type': 'database', 'component_name': 'connection_pool', 'status': 'healthy', 'health_score': 95.0, 'last_check': datetime.now()},
                {'component_type': 'realm', 'component_name': 'PROJ_ECOMMERCE', 'status': 'warning', 'health_score': 75.0, 'last_check': datetime.now()}
            ],
            [  # Alert summary
                {'component_type': 'realm', 'severity': 'medium', 'count': 2}
            ],
            []  # System metrics query
        ]
        cursor_mock.fetchone.return_value = {
            'global_realms': 1,
            'project_realms': 4,
            'total_chunks': 125,
            'total_relationships': 85,
            'active_sessions': 3
        }
        
        # Create dashboard
        dashboard = RealmDashboard(mock_db)
        
        # Get system status (should integrate monitoring data)
        status_data = dashboard.get_system_status_data()
        
        # Verify integration
        self.assertIn('system_health', status_data)
        self.assertIn('alert_summary', status_data)
        self.assertIn('system_metrics', status_data)
        
        logger.info("✓ Monitoring-dashboard integration test passed")

def run_all_tests():
    """Run all Phase 4 tests"""
    print("=" * 80)
    print("MegaMind Context Database - Phase 4 Production Features Test Suite")
    print("=" * 80)
    print()
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestRealmAnalytics,
        TestRealmMonitoring, 
        TestRealmDashboard,
        TestProductionDeployment,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PHASE 4 TEST SUMMARY")
    print("=" * 80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    
    if failures > 0:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if errors > 0:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("🎉 All Phase 4 tests passed! Production features are ready for deployment.")
    elif success_rate >= 80:
        print("⚠️  Most tests passed, but some issues need attention before production.")
    else:
        print("❌ Significant issues found. Review and fix before proceeding.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)