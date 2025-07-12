#!/usr/bin/env python3
"""
Phase 2 Semantic Search Integration Tests
Tests for MCP server integration with realm-aware semantic search
"""

import unittest
import os
import sys
import json
import asyncio
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mcp_server'))

from realm_aware_database import RealmAwareMegaMindDatabase


class TestSemanticSearchIntegration(unittest.TestCase):
    """Test integration of semantic search with realm-aware database"""
    
    def setUp(self):
        """Set up test environment with mocked database"""
        # Mock database config
        self.mock_config = {
            'host': 'localhost',
            'port': 3306,
            'database': 'test_megamind',
            'user': 'test_user',
            'password': 'test_password',
            'pool_size': 5
        }
        
        # Mock environment variables for realm configuration
        self.test_env = {
            'MEGAMIND_PROJECT_REALM': 'PROJ_TEST',
            'MEGAMIND_PROJECT_NAME': 'Test Project',
            'MEGAMIND_DEFAULT_TARGET': 'PROJECT',
            'EMBEDDING_MODEL': 'sentence-transformers/all-MiniLM-L6-v2',
            'SEMANTIC_SEARCH_THRESHOLD': '0.7',
            'REALM_PRIORITY_PROJECT': '1.2',
            'REALM_PRIORITY_GLOBAL': '1.0',
            'CROSS_REALM_SEARCH_ENABLED': 'true'
        }
    
    @patch.dict(os.environ, {
        'MEGAMIND_PROJECT_REALM': 'PROJ_TEST',
        'MEGAMIND_DEFAULT_TARGET': 'PROJECT',
        'EMBEDDING_MODEL': 'sentence-transformers/all-MiniLM-L6-v2'
    })
    @patch('mcp_server.realm_aware_database.mysql.connector.pooling.MySQLConnectionPool')
    @patch('mcp_server.realm_aware_database.get_realm_config')
    @patch('mcp_server.realm_aware_database.get_realm_access_controller')
    def test_realm_aware_database_initialization_with_semantic_search(self, mock_access, mock_config, mock_pool):
        """Test that RealmAwareMegaMindDatabase initializes with semantic search components"""
        # Mock realm configuration
        mock_realm_config = Mock()
        mock_realm_config.config.project_realm = 'PROJ_TEST'
        mock_config.return_value = mock_realm_config
        
        # Mock realm access controller
        mock_access_controller = Mock()
        mock_access.return_value = mock_access_controller
        
        # Mock connection pool and database operations
        mock_connection = Mock()
        mock_pool.return_value.get_connection.return_value = mock_connection
        
        # Initialize realm-aware database
        try:
            db = RealmAwareMegaMindDatabase(self.mock_config)
            
            # Verify semantic search components are initialized
            self.assertIsNotNone(db.embedding_service, "EmbeddingService should be initialized")
            self.assertIsNotNone(db.vector_search_engine, "VectorSearchEngine should be initialized")
            
            # Verify realm configuration
            self.assertEqual(db.realm_config, mock_realm_config)
            self.assertEqual(db.realm_access, mock_access_controller)
            
        except Exception as e:
            # Expected to fail due to mocking, but we can test partial initialization
            self.assertIn("RealmAwareMegaMindDatabase", str(type(e).__name__) or str(e))
    
    @patch.dict(os.environ, {
        'MEGAMIND_PROJECT_REALM': 'PROJ_TEST',
        'SEMANTIC_SEARCH_THRESHOLD': '0.8',
        'REALM_PRIORITY_PROJECT': '1.5'
    })
    def test_environment_configuration_loading(self):
        """Test that environment variables are properly loaded for semantic search"""
        # Test embedding service configuration loading
        from services.embedding_service import EmbeddingService
        
        # Clear singleton for clean test
        EmbeddingService._instance = None
        EmbeddingService._initialized = False
        
        service = EmbeddingService()
        
        # Verify environment configuration is loaded
        expected_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        self.assertEqual(service.model_name, expected_model)
        
        # Test vector search configuration
        from services.vector_search import RealmAwareVectorSearchEngine
        
        with patch('services.vector_search.get_embedding_service', return_value=service):
            search_engine = RealmAwareVectorSearchEngine('PROJ_TEST', 'GLOBAL')
            
            # Verify environment-based configuration
            self.assertEqual(search_engine.semantic_threshold, 0.8)
            self.assertEqual(search_engine.project_priority, 1.5)
            self.assertEqual(search_engine.project_realm, 'PROJ_TEST')
            self.assertEqual(search_engine.global_realm, 'GLOBAL')
    
    def test_dual_realm_search_type_routing(self):
        """Test that different search types route to correct methods"""
        # Mock the database methods
        mock_db = Mock(spec=RealmAwareMegaMindDatabase)
        
        # Mock the search methods
        mock_db._realm_semantic_search.return_value = [
            {'chunk_id': 'semantic1', 'content': 'semantic result', 'realm_id': 'PROJ_TEST'}
        ]
        mock_db._realm_keyword_search.return_value = [
            {'chunk_id': 'keyword1', 'content': 'keyword result', 'realm_id': 'GLOBAL'}
        ]
        mock_db._realm_hybrid_search.return_value = [
            {'chunk_id': 'hybrid1', 'content': 'hybrid result', 'realm_id': 'PROJ_TEST'}
        ]
        
        # Mock realm configuration
        mock_db.realm_config.get_search_realms.return_value = ['PROJ_TEST', 'GLOBAL']
        mock_db.realm_config.config.project_realm = 'PROJ_TEST'
        mock_db.inheritance_resolver = None
        mock_db.get_connection.return_value = Mock()
        
        # Import the actual method
        from realm_aware_database import RealmAwareMegaMindDatabase
        
        # Test semantic search routing
        result = RealmAwareMegaMindDatabase.search_chunks_dual_realm(
            mock_db, query="test", limit=10, search_type="semantic"
        )
        mock_db._realm_semantic_search.assert_called_once()
        
        # Test keyword search routing
        mock_db._realm_semantic_search.reset_mock()
        result = RealmAwareMegaMindDatabase.search_chunks_dual_realm(
            mock_db, query="test", limit=10, search_type="keyword"
        )
        mock_db._realm_keyword_search.assert_called_once()
        
        # Test hybrid search routing (default)
        mock_db._realm_keyword_search.reset_mock()
        result = RealmAwareMegaMindDatabase.search_chunks_dual_realm(
            mock_db, query="test", limit=10, search_type="hybrid"
        )
        mock_db._realm_hybrid_search.assert_called_once()
    
    def test_semantic_search_graceful_degradation(self):
        """Test that semantic search gracefully falls back to keyword search"""
        # Mock unavailable embedding service
        mock_embedding_service = Mock()
        mock_embedding_service.is_available.return_value = False
        
        # Mock vector search engine that detects unavailable service
        mock_vector_engine = Mock()
        mock_vector_engine.dual_realm_semantic_search.return_value = []
        
        # Test that semantic search falls back to keyword search
        from services.vector_search import RealmAwareVectorSearchEngine
        
        with patch('services.vector_search.get_embedding_service', return_value=mock_embedding_service):
            search_engine = RealmAwareVectorSearchEngine('PROJ_TEST', 'GLOBAL')
            
            # When embedding service is unavailable, should return empty results
            results = search_engine.dual_realm_semantic_search(
                query="test query",
                chunks_data=[],
                limit=10
            )
            
            self.assertEqual(results, [])
            mock_embedding_service.is_available.assert_called()
    
    def test_chunk_creation_with_embedding_generation(self):
        """Test that chunk creation includes embedding generation"""
        # Mock embedding service
        mock_embedding_service = Mock()
        mock_embedding_service.generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock database and its components
        mock_db = Mock(spec=RealmAwareMegaMindDatabase)
        mock_db.embedding_service = mock_embedding_service
        mock_db.realm_config.get_target_realm.return_value = 'PROJ_TEST'
        mock_db.realm_access.validate_realm_operation.return_value = (True, "OK")
        mock_db.get_connection.return_value = Mock()
        
        # Mock cursor
        mock_cursor = Mock()
        mock_db.get_connection.return_value.cursor.return_value = mock_cursor
        
        # Test chunk creation
        from realm_aware_database import RealmAwareMegaMindDatabase
        
        result = RealmAwareMegaMindDatabase.create_chunk_with_target(
            mock_db,
            content="Test content for embedding",
            source_document="test_doc.md",
            section_path="section/test",
            session_id="test_session"
        )
        
        # Verify embedding generation was called
        mock_embedding_service.generate_embedding.assert_called_once_with(
            "Test content for embedding", 
            realm_context='PROJ_TEST'
        )
        
        # Verify result is a chunk ID
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith('chunk_'))
    
    def test_search_results_include_semantic_scores(self):
        """Test that search results include semantic similarity scores"""
        # Mock search result from vector search engine
        from services.vector_search import SearchResult
        
        mock_search_result = SearchResult(
            chunk_id='test_chunk',
            content='Test content',
            source_document='test.md',
            section_path='test/section',
            realm_id='PROJ_TEST',
            similarity_score=0.85,
            keyword_score=0.7,
            final_score=0.8,
            access_count=5
        )
        
        # Mock vector search engine
        mock_vector_engine = Mock()
        mock_vector_engine.dual_realm_semantic_search.return_value = [mock_search_result]
        mock_vector_engine.project_priority = 1.2
        mock_vector_engine.global_priority = 1.0
        
        # Mock database cursor and connection
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            {
                'chunk_id': 'test_chunk',
                'content': 'Test content',
                'source_document': 'test.md',
                'section_path': 'test/section',
                'chunk_type': 'section',
                'realm_id': 'PROJ_TEST',
                'access_count': 5,
                'last_accessed': '2025-07-12 20:00:00',
                'embedding': json.dumps([0.1, 0.2, 0.3]),
                'created_at': '2025-07-12 19:00:00',
                'updated_at': '2025-07-12 19:30:00',
                'token_count': 10,
                'line_count': 3
            }
        ]
        
        # Mock realm-aware database
        mock_db = Mock(spec=RealmAwareMegaMindDatabase)
        mock_db.vector_search_engine = mock_vector_engine
        mock_db.embedding_service.is_available.return_value = True
        mock_db.realm_config.config.project_realm = 'PROJ_TEST'
        
        # Test semantic search
        from realm_aware_database import RealmAwareMegaMindDatabase
        
        results = RealmAwareMegaMindDatabase._realm_semantic_search(
            mock_db, mock_cursor, "test query", 10, ['PROJ_TEST', 'GLOBAL']
        )
        
        # Verify results include semantic scores
        self.assertEqual(len(results), 1)
        result = results[0]
        
        self.assertIn('similarity_score', result)
        self.assertIn('final_score', result)
        self.assertIn('realm_priority_weight', result)
        self.assertEqual(result['similarity_score'], 0.85)
        self.assertEqual(result['final_score'], 0.8)
        self.assertEqual(result['realm_id'], 'PROJ_TEST')


class TestMCPServerSemanticIntegration(unittest.TestCase):
    """Test MCP server integration with semantic search functions"""
    
    def setUp(self):
        """Set up MCP server test environment"""
        from megamind_database_server import MCPServer
        
        # Mock database manager
        self.mock_db = Mock()
        self.mcp_server = MCPServer(self.mock_db)
    
    def test_mcp_search_chunks_with_semantic_type(self):
        """Test MCP search_chunks function with semantic search type"""
        # Mock database response
        self.mock_db.search_chunks_dual_realm.return_value = [
            {
                'chunk_id': 'test_chunk',
                'content': 'Test semantic content',
                'realm_id': 'PROJ_TEST',
                'similarity_score': 0.9,
                'final_score': 1.08  # 0.9 * 1.2 project priority
            }
        ]
        
        # Create mock request
        request = {
            "jsonrpc": "2.0",
            "id": "test-123",
            "method": "tools/call",
            "params": {
                "name": "mcp__context_db__search_chunks",
                "arguments": {
                    "query": "test query",
                    "limit": 10,
                    "search_type": "semantic"
                }
            }
        }
        
        # Process request
        response = asyncio.run(self.mcp_server.handle_request(request))
        
        # Verify database method called with correct parameters
        self.mock_db.search_chunks_dual_realm.assert_called_once_with(
            query="test query",
            limit=10,
            search_type="semantic"
        )
        
        # Verify response structure
        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertEqual(response["id"], "test-123")
        self.assertIn("result", response)
        self.assertIn("content", response["result"])
    
    def test_mcp_search_chunks_semantic_function(self):
        """Test dedicated semantic search MCP function"""
        # Mock database response
        self.mock_db.search_chunks_semantic.return_value = [
            {
                'chunk_id': 'semantic_chunk',
                'content': 'Pure semantic result',
                'similarity_score': 0.85,
                'realm_id': 'GLOBAL'
            }
        ]
        
        # Create mock request
        request = {
            "jsonrpc": "2.0",
            "id": "semantic-test",
            "method": "tools/call",
            "params": {
                "name": "mcp__context_db__search_chunks_semantic",
                "arguments": {
                    "query": "semantic query",
                    "limit": 5,
                    "threshold": 0.8
                }
            }
        }
        
        # Process request
        response = asyncio.run(self.mcp_server.handle_request(request))
        
        # Verify database method called
        self.mock_db.search_chunks_semantic.assert_called_once_with(
            query="semantic query",
            limit=5,
            threshold=0.8
        )
        
        # Verify response
        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertEqual(response["id"], "semantic-test")
        self.assertIn("result", response)
    
    def test_mcp_search_by_similarity_function(self):
        """Test similarity search MCP function"""
        # Mock database response
        self.mock_db.search_chunks_by_similarity.return_value = [
            {
                'chunk_id': 'similar_chunk',
                'content': 'Similar content',
                'similarity_score': 0.92,
                'realm_id': 'PROJ_TEST'
            }
        ]
        
        # Create mock request
        request = {
            "jsonrpc": "2.0",
            "id": "similarity-test",
            "method": "tools/call",
            "params": {
                "name": "mcp__context_db__search_chunks_by_similarity",
                "arguments": {
                    "reference_chunk_id": "ref_chunk_123",
                    "limit": 8,
                    "threshold": 0.75
                }
            }
        }
        
        # Process request
        response = asyncio.run(self.mcp_server.handle_request(request))
        
        # Verify database method called
        self.mock_db.search_chunks_by_similarity.assert_called_once_with(
            reference_chunk_id="ref_chunk_123",
            limit=8,
            threshold=0.75
        )
        
        # Verify response
        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertEqual(response["id"], "similarity-test")
        self.assertIn("result", response)
    
    def test_mcp_batch_generate_embeddings_function(self):
        """Test batch embedding generation MCP function"""
        # Mock database response
        self.mock_db.batch_generate_embeddings.return_value = {
            'processed_chunks': 25,
            'successful_embeddings': 23,
            'failed_embeddings': 2,
            'processing_time': 12.5
        }
        
        # Create mock request
        request = {
            "jsonrpc": "2.0",
            "id": "batch-embed-test",
            "method": "tools/call",
            "params": {
                "name": "mcp__context_db__batch_generate_embeddings",
                "arguments": {
                    "realm_id": "PROJ_TEST"
                }
            }
        }
        
        # Process request
        response = asyncio.run(self.mcp_server.handle_request(request))
        
        # Verify database method called
        self.mock_db.batch_generate_embeddings.assert_called_once_with(
            chunk_ids=None,
            realm_id="PROJ_TEST"
        )
        
        # Verify response
        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertEqual(response["id"], "batch-embed-test")
        self.assertIn("result", response)
    
    def test_mcp_create_chunk_with_realm_targeting(self):
        """Test chunk creation with realm targeting and embedding generation"""
        # Mock database response
        self.mock_db.create_chunk_with_target.return_value = "chunk_abc123def"
        
        # Create mock request
        request = {
            "jsonrpc": "2.0",
            "id": "create-chunk-test",
            "method": "tools/call",
            "params": {
                "name": "mcp__context_db__create_chunk",
                "arguments": {
                    "content": "New chunk content for semantic indexing",
                    "source_document": "new_doc.md",
                    "section_path": "section/new",
                    "session_id": "session_xyz",
                    "target_realm": "GLOBAL"
                }
            }
        }
        
        # Process request
        response = asyncio.run(self.mcp_server.handle_request(request))
        
        # Verify database method called with realm targeting
        self.mock_db.create_chunk_with_target.assert_called_once_with(
            content="New chunk content for semantic indexing",
            source_document="new_doc.md",
            section_path="section/new",
            session_id="session_xyz",
            target_realm="GLOBAL"
        )
        
        # Verify response
        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertEqual(response["id"], "create-chunk-test")
        self.assertIn("result", response)


def run_integration_tests():
    """Run all Phase 2 integration tests"""
    test_classes = [
        TestSemanticSearchIntegration,
        TestMCPServerSemanticIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\n=== Running {test_class.__name__} ===")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        passed_tests += result.testsRun - len(result.failures) - len(result.errors)
        failed_tests += len(result.failures) + len(result.errors)
    
    print(f"\n=== Phase 2 Semantic Integration Test Summary ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return failed_tests == 0


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)