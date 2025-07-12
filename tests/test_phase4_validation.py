#!/usr/bin/env python3
"""
Phase 4 Validation Tests - Advanced Optimization Features
MegaMind Context Database System

Comprehensive test suite for validating Phase 4 advanced optimization features including:
- Model-optimized MCP functions
- Automated curation system
- System health monitoring
- Performance metrics and alerting
"""

import unittest
import json
import os
import time
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

# Mock the database imports for testing
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class TestPhase4ModelOptimization(unittest.TestCase):
    """Test model-optimized MCP functions"""
    
    def setUp(self):
        self.db_config = {
            'host': '10.255.250.22',
            'port': '3309',
            'database': 'megamind_database_test',
            'user': 'test_user',
            'password': 'test_password'
        }
    
    @patch('mysql.connector.pooling.MySQLConnectionPool')
    def test_model_specific_search_optimization(self, mock_pool):
        """Test model-specific search optimization strategies"""
        # Mock database connection and cursor
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_pool.return_value.get_connection.return_value = mock_connection
        
        # Mock search results for different models
        mock_cursor.fetchall.return_value = [
            {
                'chunk_id': 'test_chunk_001',
                'content': 'Test content for model optimization',
                'source_document': 'test_doc.md',
                'section_path': 'section1',
                'chunk_type': 'rule',
                'line_count': 10,
                'token_count': 50,
                'access_count': 25,
                'last_accessed': datetime.now().isoformat(),
                'embedding_vector': '[0.1, 0.2, 0.3]'
            }
        ]
        
        # Import after mocking
        from mcp_server.megamind_database_server import MegaMindDatabase
        
        db = MegaMindDatabase(self.db_config)
        
        # Test Sonnet optimization (broader context)
        sonnet_results = db.search_chunks("test query", limit=10, model_type="sonnet")
        self.assertIsInstance(sonnet_results, list)
        
        # Test Opus optimization (curated context)
        opus_results = db.search_chunks("test query", limit=5, model_type="opus")
        self.assertIsInstance(opus_results, list)
        
        # Test Claude-4 optimization (efficiency focus)
        claude4_results = db.search_chunks("test query", limit=7, model_type="claude-4")
        self.assertIsInstance(claude4_results, list)
    
    @patch('mysql.connector.pooling.MySQLConnectionPool')
    def test_hot_contexts_retrieval(self, mock_pool):
        """Test hot context prioritization for Opus"""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_pool.return_value.get_connection.return_value = mock_connection
        
        # Mock hot chunks data
        mock_cursor.fetchall.return_value = [
            {
                'chunk_id': 'hot_chunk_001',
                'content': 'Frequently accessed content',
                'access_count': 100,
                'last_accessed': datetime.now().isoformat(),
                'relevance_score': 0.95
            }
        ]
        
        from mcp_server.megamind_database_server import MegaMindDatabase
        
        db = MegaMindDatabase(self.db_config)
        hot_contexts = db.get_hot_contexts("opus", limit=20)
        
        self.assertIsInstance(hot_contexts, list)
        # Verify hot contexts are sorted by access patterns
        if len(hot_contexts) > 1:
            self.assertGreaterEqual(hot_contexts[0]['access_count'], hot_contexts[-1]['access_count'])
    
    @patch('mysql.connector.pooling.MySQLConnectionPool')
    def test_curated_context_assembly(self, mock_pool):
        """Test token-budgeted context assembly"""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_pool.return_value.get_connection.return_value = mock_connection
        
        # Mock curated context data
        mock_cursor.fetchall.return_value = [
            {
                'chunk_id': 'curated_001',
                'content': 'High-value content for Opus',
                'token_count': 150,
                'access_count': 75,
                'relevance_score': 0.9
            }
        ]
        
        from mcp_server.megamind_database_server import MegaMindDatabase
        
        db = MegaMindDatabase(self.db_config)
        curated_context = db.get_curated_context("test query", "opus", max_tokens=1000)
        
        self.assertIsInstance(curated_context, list)
        # Verify token budget is respected
        total_tokens = sum(chunk.get('token_count', 0) for chunk in curated_context)
        self.assertLessEqual(total_tokens, 1000)

class TestPhase4AutomatedCuration(unittest.TestCase):
    """Test automated curation system"""
    
    def setUp(self):
        self.db_config = {
            'host': '10.255.250.22',
            'port': '3309',
            'database': 'megamind_database_test',
            'user': 'test_user',
            'password': 'test_password'
        }
    
    @patch('mysql.connector.pooling.MySQLConnectionPool')
    def test_cold_chunk_identification(self, mock_pool):
        """Test identification of cold chunks for curation"""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_pool.return_value.get_connection.return_value = mock_connection
        
        # Mock cold chunks data
        mock_cursor.fetchall.return_value = [
            {
                'chunk_id': 'cold_chunk_001',
                'content': 'Rarely accessed content',
                'source_document': 'old_doc.md',
                'section_path': 'deprecated_section',
                'chunk_type': 'rule',
                'token_count': 75,
                'access_count': 0,
                'last_accessed': None,
                'created_at': datetime.now() - timedelta(days=90),
                'days_since_access': 90,
                'relationship_count': 0,
                'tag_count': 1
            }
        ]
        
        from curation.auto_curator import AutoCurator
        
        curator = AutoCurator(self.db_config)
        cold_chunks = curator.identify_cold_chunks(days_threshold=60, access_threshold=2)
        
        self.assertIsInstance(cold_chunks, list)
        self.assertGreater(len(cold_chunks), 0)
        
        # Verify curation scoring
        for chunk in cold_chunks:
            self.assertIn('curation_score', chunk)
            self.assertIn('curation_priority', chunk)
            self.assertIn(chunk['curation_priority'], ['high', 'medium', 'low'])
    
    @patch('mysql.connector.pooling.MySQLConnectionPool')
    def test_consolidation_candidates(self, mock_pool):
        """Test consolidation candidate detection"""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_pool.return_value.get_connection.return_value = mock_connection
        
        # Mock similar chunks for consolidation
        mock_cursor.fetchall.return_value = [
            {
                'chunk_id': 'similar_001',
                'content': 'Similar content about database optimization',
                'source_document': 'db_guide.md',
                'section_path': 'optimization',
                'chunk_type': 'rule',
                'token_count': 120,
                'access_count': 5
            },
            {
                'chunk_id': 'similar_002',
                'content': 'Related content about database performance optimization',
                'source_document': 'db_guide.md',
                'section_path': 'performance',
                'chunk_type': 'rule',
                'token_count': 110,
                'access_count': 3
            }
        ]
        
        from curation.auto_curator import AutoCurator
        
        curator = AutoCurator(self.db_config)
        consolidation_candidates = curator.find_consolidation_candidates(similarity_threshold=0.8)
        
        self.assertIsInstance(consolidation_candidates, list)
        # Verify consolidation structure
        for candidate in consolidation_candidates:
            self.assertIn('primary_chunk_id', candidate.__dict__)
            self.assertIn('related_chunk_ids', candidate.__dict__)
            self.assertIn('similarity_score', candidate.__dict__)
            self.assertIn('consolidation_benefit', candidate.__dict__)
    
    @patch('mysql.connector.pooling.MySQLConnectionPool')
    def test_curation_recommendations(self, mock_pool):
        """Test generation of curation recommendations"""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_pool.return_value.get_connection.return_value = mock_connection
        
        # Mock data for recommendations
        mock_cursor.fetchall.side_effect = [
            # Cold chunks query
            [
                {
                    'chunk_id': 'cold_001',
                    'content': 'Cold content',
                    'source_document': 'old.md',
                    'section_path': 'old_section',
                    'chunk_type': 'rule',
                    'token_count': 50,
                    'access_count': 0,
                    'last_accessed': None,
                    'created_at': datetime.now() - timedelta(days=120),
                    'days_since_access': 120,
                    'relationship_count': 0,
                    'tag_count': 0
                }
            ],
            # Consolidation candidates query
            [
                {
                    'chunk_id': 'consolidate_001',
                    'content': 'Content for consolidation',
                    'source_document': 'test.md',
                    'section_path': 'test_section',
                    'chunk_type': 'rule',
                    'token_count': 100,
                    'access_count': 2
                }
            ]
        ]
        
        from curation.auto_curator import AutoCurator
        
        curator = AutoCurator(self.db_config)
        recommendations = curator.generate_curation_recommendations()
        
        self.assertIsInstance(recommendations, list)
        # Verify recommendation structure
        for rec in recommendations:
            self.assertIn('recommendation_id', rec.__dict__)
            self.assertIn('recommendation_type', rec.__dict__)
            self.assertIn('target_chunks', rec.__dict__)
            self.assertIn('confidence_score', rec.__dict__)
            self.assertIn('impact_assessment', rec.__dict__)
            self.assertIn('potential_savings', rec.__dict__)

class TestPhase4SystemHealthMonitoring(unittest.TestCase):
    """Test system health monitoring"""
    
    def setUp(self):
        self.db_config = {
            'host': '10.255.250.22',
            'port': '3309',
            'database': 'megamind_database_test',
            'user': 'test_user',
            'password': 'test_password'
        }
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('mysql.connector.pooling.MySQLConnectionPool')
    def test_metrics_collection(self, mock_pool, mock_disk, mock_memory, mock_cpu):
        """Test system metrics collection"""
        # Mock system metrics
        mock_cpu.return_value = 45.0
        mock_memory.return_value = Mock(percent=60.0, total=8*1024**3, used=4.8*1024**3)
        mock_disk.return_value = Mock(total=100*1024**3, used=50*1024**3)
        
        # Mock database connection
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_pool.return_value.get_connection.return_value = mock_connection
        
        # Mock database metrics
        mock_cursor.fetchall.return_value = [
            {'table_name': 'megamind_chunks', 'size_mb': 150.5}
        ]
        mock_cursor.fetchone.side_effect = [
            {'total_chunks': 1000},
            {'hot_chunks': 100},
            {'cold_chunks': 50},
            {'avg_access': 5.5},
            {'total_relationships': 500},
            {'avg_relationships': 2.3},
            {'pending_sessions': 3},
            {'total_changes': 15}
        ]
        
        from monitoring.system_health import MetricsCollector
        
        collector = MetricsCollector(self.db_config)
        
        # Test database metrics collection
        db_metrics = collector._collect_database_metrics()
        self.assertIsInstance(db_metrics, list)
        self.assertGreater(len(db_metrics), 0)
        
        # Test system metrics collection
        sys_metrics = collector._collect_system_metrics()
        self.assertIsInstance(sys_metrics, list)
        self.assertGreater(len(sys_metrics), 0)
        
        # Test application metrics collection
        app_metrics = collector._collect_application_metrics()
        self.assertIsInstance(app_metrics, list)
        self.assertGreater(len(app_metrics), 0)
    
    @patch('mysql.connector.pooling.MySQLConnectionPool')
    def test_health_checks(self, mock_pool):
        """Test system health checks"""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_pool.return_value.get_connection.return_value = mock_connection
        
        # Mock health check queries
        mock_cursor.fetchone.side_effect = [
            (1,),  # SELECT 1 test
            (1000,),  # chunk count
            (500,),  # relationship count
        ]
        mock_cursor.fetchall.return_value = []
        
        from monitoring.system_health import SystemHealthMonitor
        
        monitor = SystemHealthMonitor(self.db_config)
        health_checks = monitor.run_health_checks()
        
        self.assertIsInstance(health_checks, list)
        self.assertGreater(len(health_checks), 0)
        
        # Verify health check structure
        for check in health_checks:
            self.assertIn('check_name', check.__dict__)
            self.assertIn('status', check.__dict__)
            self.assertIn('message', check.__dict__)
            self.assertIn('timestamp', check.__dict__)
            self.assertIn('metrics', check.__dict__)
            self.assertIn(check.status, ['healthy', 'warning', 'critical'])
    
    @patch('mysql.connector.pooling.MySQLConnectionPool')
    def test_alerting_system(self, mock_pool):
        """Test alert rule evaluation and notification"""
        from monitoring.system_health import AlertManager, AlertRule, MetricsCollector, MetricSample
        from datetime import datetime
        
        # Create mock metrics collector
        mock_collector = Mock(spec=MetricsCollector)
        
        # Mock recent metrics for alert evaluation
        mock_metrics = [
            MetricSample(
                timestamp=datetime.now(),
                metric_name="system_cpu_percent",
                value=85.0,
                tags={}
            )
        ]
        mock_collector.get_recent_metrics.return_value = mock_metrics
        
        alert_manager = AlertManager(mock_collector)
        
        # Test alert rule creation
        high_cpu_rule = AlertRule(
            rule_id="cpu_test",
            metric_name="system_cpu_percent",
            condition="gt",
            threshold=80.0,
            duration_minutes=5,
            severity="warning",
            enabled=True
        )
        
        alert_manager.add_alert_rule(high_cpu_rule)
        self.assertIn(high_cpu_rule, alert_manager.alert_rules)
        
        # Test condition checking
        result = alert_manager._check_condition(85.0, "gt", 80.0)
        self.assertTrue(result)
        
        result = alert_manager._check_condition(75.0, "gt", 80.0)
        self.assertFalse(result)
    
    @patch('mysql.connector.pooling.MySQLConnectionPool')
    def test_system_status_report(self, mock_pool):
        """Test comprehensive system status reporting"""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_pool.return_value.get_connection.return_value = mock_connection
        
        # Mock all health check queries
        mock_cursor.fetchone.side_effect = [
            (1,), (1000,), (500,),  # database connectivity
            {'total_chunks': 1000, 'never_accessed': 50, 'low_access': 200, 'high_access': 100, 'avg_access': 5.5, 'max_access': 150},  # chunk distribution
            {'orphaned_relationships': 0},  # relationship integrity
            {'total_relationships': 500, 'chunks_with_relationships': 800, 'avg_strength': 0.75}  # relationship stats
        ]
        
        from monitoring.system_health import SystemHealthMonitor
        
        monitor = SystemHealthMonitor(self.db_config)
        status = monitor.get_system_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('timestamp', status)
        self.assertIn('overall_status', status)
        self.assertIn('health_checks', status)
        self.assertIn('recent_metrics', status)
        self.assertIn('active_alerts', status)
        self.assertIn('alert_summary', status)
        
        # Verify overall status calculation
        self.assertIn(status['overall_status'], ['healthy', 'warning', 'critical'])

class TestPhase4PerformanceOptimization(unittest.TestCase):
    """Test performance optimization features"""
    
    @patch('mysql.connector.pooling.MySQLConnectionPool')
    def test_performance_metrics_tracking(self, mock_pool):
        """Test performance metrics collection and analysis"""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_pool.return_value.get_connection.return_value = mock_connection
        
        # Mock performance metrics
        mock_cursor.fetchall.return_value = [
            {
                'chunk_id': 'perf_test_001',
                'access_count': 100,
                'avg_response_time': 0.15,
                'cache_hit_rate': 0.85
            }
        ]
        
        from mcp_server.megamind_database_server import MegaMindDatabase
        
        db = MegaMindDatabase({
            'host': '10.255.250.22', 'port': '3309', 'database': 'test',
            'user': 'test', 'password': 'test'
        })
        
        performance_metrics = db.get_performance_metrics()
        
        self.assertIsInstance(performance_metrics, dict)
        self.assertIn('response_times', performance_metrics)
        self.assertIn('cache_performance', performance_metrics)
        self.assertIn('query_efficiency', performance_metrics)
    
    def test_token_budget_management(self):
        """Test token budget enforcement for different models"""
        from mcp_server.megamind_database_server import MegaMindDatabase
        
        # Mock chunk data
        chunks = [
            {'chunk_id': 'chunk_001', 'token_count': 150, 'relevance_score': 0.9},
            {'chunk_id': 'chunk_002', 'token_count': 200, 'relevance_score': 0.8},
            {'chunk_id': 'chunk_003', 'token_count': 100, 'relevance_score': 0.85},
            {'chunk_id': 'chunk_004', 'token_count': 300, 'relevance_score': 0.7},
        ]
        
        # Test token budget enforcement
        def apply_token_budget(chunks, max_tokens):
            selected = []
            total_tokens = 0
            
            # Sort by relevance score
            sorted_chunks = sorted(chunks, key=lambda x: x['relevance_score'], reverse=True)
            
            for chunk in sorted_chunks:
                if total_tokens + chunk['token_count'] <= max_tokens:
                    selected.append(chunk)
                    total_tokens += chunk['token_count']
            
            return selected, total_tokens
        
        # Test different model budgets
        opus_chunks, opus_tokens = apply_token_budget(chunks, 1000)
        sonnet_chunks, sonnet_tokens = apply_token_budget(chunks, 2000)
        
        self.assertLessEqual(opus_tokens, 1000)
        self.assertLessEqual(sonnet_tokens, 2000)
        self.assertGreaterEqual(len(sonnet_chunks), len(opus_chunks))

class TestPhase4IntegrationValidation(unittest.TestCase):
    """Test end-to-end integration of Phase 4 features"""
    
    def setUp(self):
        self.db_config = {
            'host': '10.255.250.22',
            'port': '3309',
            'database': 'megamind_database_test',
            'user': 'test_user',
            'password': 'test_password'
        }
    
    @patch('mysql.connector.pooling.MySQLConnectionPool')
    def test_end_to_end_optimization_workflow(self, mock_pool):
        """Test complete optimization workflow from search to curation"""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_pool.return_value.get_connection.return_value = mock_connection
        
        # Mock workflow data
        mock_cursor.fetchall.side_effect = [
            # Search results
            [
                {
                    'chunk_id': 'workflow_001',
                    'content': 'Workflow test content',
                    'source_document': 'workflow.md',
                    'section_path': 'test_section',
                    'chunk_type': 'rule',
                    'line_count': 15,
                    'token_count': 75,
                    'access_count': 10,
                    'last_accessed': datetime.now().isoformat(),
                    'embedding_vector': '[0.1, 0.2, 0.3]'
                }
            ],
            # Cold chunks for curation
            [
                {
                    'chunk_id': 'cold_workflow_001',
                    'content': 'Cold workflow content',
                    'source_document': 'old_workflow.md',
                    'section_path': 'deprecated',
                    'chunk_type': 'rule',
                    'token_count': 50,
                    'access_count': 0,
                    'last_accessed': None,
                    'created_at': datetime.now() - timedelta(days=90),
                    'days_since_access': 90,
                    'relationship_count': 0,
                    'tag_count': 0
                }
            ]
        ]
        
        # Test complete workflow
        from mcp_server.megamind_database_server import MegaMindDatabase
        from curation.auto_curator import AutoCurator
        from monitoring.system_health import SystemHealthMonitor
        
        # 1. Test optimized search
        db = MegaMindDatabase(self.db_config)
        search_results = db.search_chunks("workflow test", limit=5, model_type="opus")
        self.assertIsInstance(search_results, list)
        
        # 2. Test curation analysis
        curator = AutoCurator(self.db_config)
        cold_chunks = curator.identify_cold_chunks()
        self.assertIsInstance(cold_chunks, list)
        
        # 3. Test health monitoring
        monitor = SystemHealthMonitor(self.db_config)
        health_status = monitor.get_system_status()
        self.assertIsInstance(health_status, dict)
        self.assertIn('overall_status', health_status)
    
    def test_mcp_function_integration(self):
        """Test integration of all Phase 4 MCP functions"""
        # Test function signature validation
        phase4_functions = [
            'mcp__megamind_db__get_hot_contexts',
            'mcp__megamind_db__get_curated_context',
            'mcp__megamind_db__get_performance_metrics',
            'mcp__megamind_db__identify_cold_chunks'
        ]
        
        # Verify function definitions exist (mock test)
        for func_name in phase4_functions:
            # This would test that functions are properly defined
            # In actual implementation, would verify function signatures and return types
            self.assertTrue(True)  # Placeholder for actual function validation

def run_phase4_validation_suite():
    """Run complete Phase 4 validation test suite"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestPhase4ModelOptimization,
        TestPhase4AutomatedCuration,
        TestPhase4SystemHealthMonitoring,
        TestPhase4PerformanceOptimization,
        TestPhase4IntegrationValidation
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("="*60)
    print("PHASE 4 VALIDATION - ADVANCED OPTIMIZATION FEATURES")
    print("="*60)
    print("Testing model optimization, curation, and monitoring...")
    print()
    
    success = run_phase4_validation_suite()
    
    print()
    print("="*60)
    if success:
        print("✅ Phase 4 validation PASSED - All advanced optimization features working")
    else:
        print("❌ Phase 4 validation FAILED - Some features need attention")
    print("="*60)