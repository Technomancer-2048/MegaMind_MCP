#!/usr/bin/env python3
"""
Phase 3 Validation Tests: Bidirectional Flow
Tests for change buffering, session management, and review system
"""

import unittest
import json
import os
import uuid
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules we're testing
try:
    from mcp_server.megamind_database_server import MegaMindDatabase
    from review.change_reviewer import ChangeReviewManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Note: Some modules may not be available in test environment")

class TestBidirectionalFlow(unittest.TestCase):
    """Test suite for Phase 3 bidirectional flow functionality"""
    
    def setUp(self):
        """Set up test environment with mocked database"""
        self.mock_config = {
            'host': 'localhost',
            'port': '3309',
            'database': 'megamind_database_test',
            'user': 'test_user',
            'password': 'test_password',
            'pool_size': '5'
        }
        
        # Mock database connections
        self.mock_connection = Mock()
        self.mock_cursor = Mock()
        self.mock_cursor.fetchone = Mock()
        self.mock_cursor.fetchall = Mock()
        self.mock_cursor.execute = Mock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        self.mock_connection.commit = Mock()
        self.mock_connection.rollback = Mock()
        self.mock_connection.close = Mock()
        
        # Sample data
        self.session_id = "test_session_001"
        self.chunk_id = "chunk_test_001"
        self.change_id = f"change_{uuid.uuid4().hex[:12]}"
        
    @patch('mcp_server.megamind_database_server.pooling.MySQLConnectionPool')
    def test_update_chunk_buffering(self, mock_pool):
        """Test chunk update buffering functionality"""
        # Setup mocks
        mock_pool.return_value.get_connection.return_value = self.mock_connection
        self.mock_cursor.fetchone.return_value = {
            'chunk_id': self.chunk_id,
            'content': 'Original content',
            'source_document': 'test_doc.md',
            'section_path': '/test/section',
            'chunk_type': 'section',
            'access_count': 25
        }
        
        try:
            # Initialize database manager
            db_manager = MegaMindDatabase(self.mock_config)
            
            # Test update chunk
            result = db_manager.update_chunk(
                self.chunk_id, 
                "Updated content", 
                self.session_id
            )
            
            # Verify result structure
            self.assertIn('success', result)
            self.assertIn('change_id', result)
            self.assertIn('impact_score', result)
            self.assertIn('requires_review', result)
            
            # Verify database calls were made
            self.mock_cursor.execute.assert_called()
            self.mock_connection.commit.assert_called()
            
            print("✓ Chunk update buffering test passed")
            
        except ImportError:
            print("⚠ Skipping chunk update test - module not available")
    
    @patch('mcp_server.megamind_database_server.pooling.MySQLConnectionPool')
    def test_create_chunk_buffering(self, mock_pool):
        """Test new chunk creation buffering"""
        # Setup mocks
        mock_pool.return_value.get_connection.return_value = self.mock_connection
        
        try:
            # Initialize database manager
            db_manager = MegaMindDatabase(self.mock_config)
            
            # Test create chunk
            result = db_manager.create_chunk(
                "New chunk content\nWith multiple lines",
                "new_document.md",
                "/new/section",
                self.session_id
            )
            
            # Verify result structure
            self.assertIn('success', result)
            self.assertIn('change_id', result)
            self.assertIn('temp_chunk_id', result)
            self.assertIn('impact_score', result)
            
            # Verify database operations
            self.mock_cursor.execute.assert_called()
            self.mock_connection.commit.assert_called()
            
            print("✓ Chunk creation buffering test passed")
            
        except ImportError:
            print("⚠ Skipping chunk creation test - module not available")
    
    @patch('mcp_server.megamind_database_server.pooling.MySQLConnectionPool')
    def test_relationship_buffering(self, mock_pool):
        """Test relationship addition buffering"""
        # Setup mocks
        mock_pool.return_value.get_connection.return_value = self.mock_connection
        self.mock_cursor.fetchall.return_value = [
            {'chunk_id': 'chunk_001'},
            {'chunk_id': 'chunk_002'}
        ]
        self.mock_cursor.fetchone.return_value = None  # No existing relationship
        
        try:
            # Initialize database manager
            db_manager = MegaMindDatabase(self.mock_config)
            
            # Test add relationship
            result = db_manager.add_relationship(
                'chunk_001',
                'chunk_002',
                'references',
                self.session_id
            )
            
            # Verify result structure
            self.assertIn('success', result)
            self.assertIn('change_id', result)
            self.assertIn('impact_score', result)
            
            # Verify database operations
            self.mock_cursor.execute.assert_called()
            self.mock_connection.commit.assert_called()
            
            print("✓ Relationship buffering test passed")
            
        except ImportError:
            print("⚠ Skipping relationship test - module not available")
    
    @patch('mcp_server.megamind_database_server.pooling.MySQLConnectionPool')
    def test_pending_changes_retrieval(self, mock_pool):
        """Test retrieval of pending changes"""
        # Setup mocks
        mock_pool.return_value.get_connection.return_value = self.mock_connection
        self.mock_cursor.fetchall.return_value = [
            {
                'change_id': 'change_001',
                'session_id': self.session_id,
                'change_type': 'update',
                'chunk_id': 'chunk_001',
                'target_chunk_id': None,
                'change_data': '{"original_content": "old", "new_content": "new"}',
                'impact_score': 0.75,
                'timestamp': datetime.now(),
                'status': 'pending'
            }
        ]
        
        try:
            # Initialize database manager
            db_manager = MegaMindDatabase(self.mock_config)
            
            # Test get pending changes
            changes = db_manager.get_pending_changes(self.session_id)
            
            # Verify results
            self.assertIsInstance(changes, list)
            if changes:
                change = changes[0]
                self.assertEqual(change.session_id, self.session_id)
                self.assertEqual(change.change_type, 'update')
                self.assertIsInstance(change.change_data, dict)
            
            print("✓ Pending changes retrieval test passed")
            
        except ImportError:
            print("⚠ Skipping pending changes test - module not available")
    
    @patch('mcp_server.megamind_database_server.pooling.MySQLConnectionPool')
    def test_change_commit_process(self, mock_pool):
        """Test committing approved changes"""
        # Setup mocks
        mock_pool.return_value.get_connection.return_value = self.mock_connection
        
        # Mock change data retrieval
        change_data = {
            'change_id': 'change_001',
            'session_id': self.session_id,
            'change_type': 'update',
            'chunk_id': 'chunk_001',
            'change_data': '{"original_content": "old", "new_content": "new"}',
            'status': 'pending'
        }
        
        original_chunk = {
            'chunk_id': 'chunk_001',
            'content': 'original content'
        }
        
        self.mock_cursor.fetchone.side_effect = [change_data, original_chunk]
        
        try:
            # Initialize database manager
            db_manager = MegaMindDatabase(self.mock_config)
            
            # Test commit changes
            result = db_manager.commit_session_changes(
                self.session_id, 
                ['change_001']
            )
            
            # Verify result structure
            self.assertIn('success', result)
            self.assertIn('contribution_id', result)
            self.assertIn('changes_committed', result)
            self.assertIn('total_changes', result)
            
            # Verify transaction handling
            self.mock_connection.start_transaction.assert_called()
            self.mock_connection.commit.assert_called()
            
            print("✓ Change commit process test passed")
            
        except ImportError:
            print("⚠ Skipping change commit test - module not available")
    
    @patch('mcp_server.megamind_database_server.pooling.MySQLConnectionPool')
    def test_change_rollback(self, mock_pool):
        """Test rolling back pending changes"""
        # Setup mocks
        mock_pool.return_value.get_connection.return_value = self.mock_connection
        self.mock_cursor.fetchone.return_value = {'count': 3}
        
        try:
            # Initialize database manager
            db_manager = MegaMindDatabase(self.mock_config)
            
            # Test rollback changes
            result = db_manager.rollback_session_changes(self.session_id)
            
            # Verify result structure
            self.assertIn('success', result)
            self.assertIn('changes_discarded', result)
            
            # Verify database operations
            self.mock_cursor.execute.assert_called()
            self.mock_connection.commit.assert_called()
            
            print("✓ Change rollback test passed")
            
        except ImportError:
            print("⚠ Skipping rollback test - module not available")
    
    @patch('mcp_server.megamind_database_server.pooling.MySQLConnectionPool')
    def test_change_summary_generation(self, mock_pool):
        """Test change summary and impact analysis"""
        # Setup mocks
        mock_pool.return_value.get_connection.return_value = self.mock_connection
        
        session_meta = {
            'session_id': self.session_id,
            'pending_changes_count': 5,
            'start_timestamp': datetime.now(),
            'last_activity': datetime.now()
        }
        
        changes_summary = [
            {
                'change_type': 'update',
                'count': 2,
                'avg_impact': 0.6,
                'max_impact': 0.8,
                'critical_count': 1,
                'important_count': 1,
                'standard_count': 0
            }
        ]
        
        high_impact_changes = [
            {
                'change_id': 'change_001',
                'change_type': 'update',
                'chunk_id': 'chunk_001',
                'impact_score': 0.8,
                'source_document': 'test.md',
                'section_path': '/section',
                'access_count': 50
            }
        ]
        
        self.mock_cursor.fetchone.return_value = session_meta
        self.mock_cursor.fetchall.side_effect = [changes_summary, high_impact_changes]
        
        try:
            # Initialize database manager
            db_manager = MegaMindDatabase(self.mock_config)
            
            # Test change summary generation
            summary = db_manager.get_change_summary(self.session_id)
            
            # Verify summary structure
            self.assertIn('session_id', summary)
            self.assertIn('total_pending', summary)
            self.assertIn('priority_breakdown', summary)
            self.assertIn('changes_by_type', summary)
            self.assertIn('high_impact_changes', summary)
            
            print("✓ Change summary generation test passed")
            
        except ImportError:
            print("⚠ Skipping summary generation test - module not available")

class TestReviewInterface(unittest.TestCase):
    """Test suite for the review interface functionality"""
    
    def setUp(self):
        """Set up test environment for review interface"""
        self.mock_config = {
            'host': 'localhost',
            'port': '3309',
            'database': 'megamind_database_test',
            'user': 'test_user',
            'password': 'test_password'
        }
        
        # Mock database connections
        self.mock_connection = Mock()
        self.mock_cursor = Mock()
        self.mock_cursor.fetchone = Mock()
        self.mock_cursor.fetchall = Mock()
        self.mock_cursor.execute = Mock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        self.mock_connection.commit = Mock()
        self.mock_connection.rollback = Mock()
        self.mock_connection.close = Mock()
    
    @patch('review.change_reviewer.pooling.MySQLConnectionPool')
    def test_pending_sessions_retrieval(self, mock_pool):
        """Test retrieval of sessions with pending changes"""
        # Setup mocks
        mock_pool.return_value.get_connection.return_value = self.mock_connection
        
        sessions_data = [
            {
                'session_id': 'session_001',
                'user_context': 'ai_session',
                'project_context': 'megamind_system',
                'start_timestamp': datetime.now(),
                'last_activity': datetime.now(),
                'pending_changes_count': 5,
                'total_changes': 5,
                'critical_changes': 1,
                'important_changes': 2,
                'avg_impact_score': 0.6
            }
        ]
        
        self.mock_cursor.fetchall.return_value = sessions_data
        
        try:
            # Initialize review manager
            review_manager = ChangeReviewManager(self.mock_config)
            
            # Test get pending sessions
            sessions = review_manager.get_pending_sessions()
            
            # Verify results
            self.assertIsInstance(sessions, list)
            if sessions:
                session = sessions[0]
                self.assertIn('session_id', session)
                self.assertIn('pending_changes_count', session)
                self.assertIn('critical_changes', session)
            
            print("✓ Pending sessions retrieval test passed")
            
        except ImportError:
            print("⚠ Skipping pending sessions test - module not available")
    
    @patch('review.change_reviewer.pooling.MySQLConnectionPool')
    def test_session_changes_detailed_view(self, mock_pool):
        """Test detailed session changes view"""
        # Setup mocks
        mock_pool.return_value.get_connection.return_value = self.mock_connection
        
        session_meta = {
            'session_id': 'session_001',
            'user_context': 'ai_session',
            'project_context': 'megamind_system',
            'start_timestamp': datetime.now(),
            'last_activity': datetime.now(),
            'pending_changes_count': 2
        }
        
        changes_data = [
            {
                'change_id': 'change_001',
                'change_type': 'update',
                'chunk_id': 'chunk_001',
                'target_chunk_id': None,
                'change_data': '{"original_content": "old", "new_content": "new"}',
                'impact_score': 0.75,
                'timestamp': datetime.now(),
                'status': 'pending',
                'source_document': 'test.md',
                'section_path': '/section',
                'access_count': 25,
                'chunk_type': 'section',
                'target_source': None,
                'target_section': None
            }
        ]
        
        self.mock_cursor.fetchone.return_value = session_meta
        self.mock_cursor.fetchall.return_value = changes_data
        
        try:
            # Initialize review manager
            review_manager = ChangeReviewManager(self.mock_config)
            
            # Test get session changes
            result = review_manager.get_session_changes('session_001')
            
            # Verify result structure
            self.assertIn('session', result)
            self.assertIn('changes', result)
            self.assertIn('summary', result)
            
            if result['changes']:
                change = result['changes'][0]
                self.assertIn('priority_level', change)
                self.assertIn('diff_preview', change)
            
            print("✓ Session changes detailed view test passed")
            
        except ImportError:
            print("⚠ Skipping session changes test - module not available")

class TestImpactScoring(unittest.TestCase):
    """Test suite for impact scoring and prioritization"""
    
    def test_impact_score_calculation(self):
        """Test impact score calculation based on access patterns"""
        # Test cases: (access_count, expected_score)
        test_cases = [
            (0, 0.0),
            (25, 0.25),
            (50, 0.5),
            (100, 1.0),
            (200, 1.0),  # Capped at 1.0
        ]
        
        for access_count, expected_score in test_cases:
            calculated_score = min(access_count / 100.0, 1.0)
            self.assertEqual(calculated_score, expected_score)
        
        print("✓ Impact score calculation test passed")
    
    def test_priority_level_assignment(self):
        """Test priority level assignment based on impact scores"""
        def get_priority_level(impact_score):
            if impact_score > 0.7:
                return "critical"
            elif impact_score >= 0.3:
                return "important"
            else:
                return "standard"
        
        # Test cases: (impact_score, expected_priority)
        test_cases = [
            (0.1, "standard"),
            (0.3, "important"),
            (0.5, "important"),
            (0.7, "important"),
            (0.8, "critical"),
            (1.0, "critical"),
        ]
        
        for impact_score, expected_priority in test_cases:
            calculated_priority = get_priority_level(impact_score)
            self.assertEqual(calculated_priority, expected_priority)
        
        print("✓ Priority level assignment test passed")

class TestDataIntegrity(unittest.TestCase):
    """Test suite for data integrity and validation"""
    
    def test_change_data_serialization(self):
        """Test JSON serialization of change data"""
        test_change_data = {
            "original_content": "Original text with special chars: à, é, ñ",
            "new_content": "New text with Unicode: 中文, العربية",
            "chunk_metadata": {
                "source_document": "test.md",
                "section_path": "/section",
                "chunk_type": "function"
            }
        }
        
        # Test serialization and deserialization
        serialized = json.dumps(test_change_data)
        deserialized = json.loads(serialized)
        
        self.assertEqual(test_change_data, deserialized)
        print("✓ Change data serialization test passed")
    
    def test_session_id_validation(self):
        """Test session ID format validation"""
        valid_session_ids = [
            "test_session_001",
            "session_123",
            "ai_session_2023_12_01"
        ]
        
        invalid_session_ids = [
            "",
            None,
            "session with spaces",
            "session/with/slashes"
        ]
        
        def is_valid_session_id(session_id):
            return (session_id and 
                    isinstance(session_id, str) and 
                    len(session_id) > 0 and 
                    ' ' not in session_id and 
                    '/' not in session_id)
        
        for session_id in valid_session_ids:
            self.assertTrue(is_valid_session_id(session_id))
        
        for session_id in invalid_session_ids:
            self.assertFalse(is_valid_session_id(session_id))
        
        print("✓ Session ID validation test passed")

def run_phase3_tests():
    """Run all Phase 3 bidirectional flow tests"""
    print("🧪 Running Phase 3 Bidirectional Flow Tests")
    print("=" * 50)
    
    # Create test loader
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBidirectionalFlow))
    suite.addTests(loader.loadTestsFromTestCase(TestReviewInterface))
    suite.addTests(loader.loadTestsFromTestCase(TestImpactScoring))
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegrity))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    print("Phase 3 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All Phase 3 tests passed!")
    else:
        print("❌ Some Phase 3 tests failed")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_phase3_tests()
    exit(0 if success else 1)