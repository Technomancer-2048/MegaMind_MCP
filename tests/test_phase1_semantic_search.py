#!/usr/bin/env python3
"""
Test Phase 1 Semantic Search Implementation
Tests for EmbeddingService, RealmAwareVectorSearchEngine, and enhanced database operations
"""

import unittest
import os
import sys
import json
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.services.embedding_service import EmbeddingService, get_embedding_service
from mcp_server.services.vector_search import RealmAwareVectorSearchEngine, SearchResult


class TestEmbeddingService(unittest.TestCase):
    """Test EmbeddingService functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.embedding_service = EmbeddingService()
        # Clear singleton to get fresh instance for testing
        EmbeddingService._instance = None
        EmbeddingService._initialized = False
    
    def test_singleton_pattern(self):
        """Test that EmbeddingService is a singleton"""
        service1 = get_embedding_service()
        service2 = get_embedding_service()
        self.assertIs(service1, service2)
    
    def test_embedding_service_initialization(self):
        """Test embedding service initialization with environment variables"""
        with patch.dict(os.environ, {
            'EMBEDDING_MODEL': 'test-model',
            'EMBEDDING_DEVICE': 'cpu',
            'EMBEDDING_BATCH_SIZE': '25',
            'EMBEDDING_CACHE_SIZE': '500'
        }):
            service = EmbeddingService()
            self.assertEqual(service.model_name, 'test-model')
            self.assertEqual(service.device, 'cpu')
            self.assertEqual(service.batch_size, 25)
            self.assertEqual(service.cache_size, 500)
    
    @patch('mcp_server.services.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    def test_graceful_degradation_without_dependencies(self):
        """Test that service handles missing dependencies gracefully"""
        service = EmbeddingService()
        self.assertFalse(service.is_available())
        
        result = service.generate_embedding("test text")
        self.assertIsNone(result)
        
        results = service.generate_embeddings_batch(["text1", "text2"])
        self.assertEqual(results, [None, None])
    
    @patch('mcp_server.services.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    def test_embedding_generation_with_realm_context(self):
        """Test embedding generation with realm context"""
        # Mock the model directly without patching SentenceTransformer import
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        
        service = EmbeddingService()
        service.model = mock_model
        
        # Test with realm context
        result = service.generate_embedding("test content", realm_context="PROJ_TEST")
        
        self.assertIsNotNone(result)
        self.assertEqual(result, [0.1, 0.2, 0.3])
        
        # Verify the text was preprocessed with realm context
        call_args = mock_model.encode.call_args[0][0]
        self.assertIn("[PROJ_TEST]", call_args)
        self.assertIn("test content", call_args)
    
    @patch('mcp_server.services.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    def test_embedding_caching(self):
        """Test embedding caching functionality"""
        # Mock the model directly
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        
        service = EmbeddingService()
        service.model = mock_model
        
        # First call should use the model
        result1 = service.generate_embedding("test content")
        self.assertEqual(mock_model.encode.call_count, 1)
        
        # Second call with same content should use cache
        result2 = service.generate_embedding("test content")
        self.assertEqual(mock_model.encode.call_count, 1)  # No additional call
        self.assertEqual(result1, result2)
    
    def test_text_preprocessing(self):
        """Test text preprocessing functionality"""
        service = EmbeddingService()
        
        # Test basic cleaning
        result = service.preprocess_text("  test   text  \n\n  ")
        self.assertEqual(result, "test text")
        
        # Test with realm context
        result = service.preprocess_text("test content", realm_context="PROJ_TEST")
        self.assertEqual(result, "[PROJ_TEST] test content")
        
        # Test with GLOBAL realm (should not add prefix)
        result = service.preprocess_text("test content", realm_context="GLOBAL")
        self.assertEqual(result, "test content")
    
    def test_embedding_stats(self):
        """Test embedding service statistics"""
        service = EmbeddingService()
        stats = service.get_embedding_stats()
        
        self.assertIn('model_name', stats)
        self.assertIn('device', stats)
        self.assertIn('embedding_dimension', stats)
        self.assertIn('cache_size', stats)
        self.assertIn('available', stats)
        self.assertIn('dependencies', stats)


class TestRealmAwareVectorSearchEngine(unittest.TestCase):
    """Test RealmAwareVectorSearchEngine functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.project_realm = "PROJ_TEST"
        self.global_realm = "GLOBAL"
        
        # Mock embedding service
        self.mock_embedding_service = Mock()
        self.mock_embedding_service.is_available.return_value = True
        self.mock_embedding_service.generate_embedding.return_value = [0.5, 0.5, 0.5]
        
        with patch('mcp_server.services.vector_search.get_embedding_service', return_value=self.mock_embedding_service):
            self.search_engine = RealmAwareVectorSearchEngine(
                project_realm=self.project_realm,
                global_realm=self.global_realm
            )
    
    def test_search_engine_initialization(self):
        """Test search engine initialization"""
        self.assertEqual(self.search_engine.project_realm, "PROJ_TEST")
        self.assertEqual(self.search_engine.global_realm, "GLOBAL")
        self.assertEqual(self.search_engine.semantic_threshold, 0.7)
        self.assertEqual(self.search_engine.project_priority, 1.2)
        self.assertEqual(self.search_engine.global_priority, 1.0)
    
    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation"""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = self.search_engine._calculate_cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = self.search_engine._calculate_cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 0.0, places=5)
        
        vec1 = [1.0, 1.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = self.search_engine._calculate_cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 0.7071, places=3)
    
    def test_realm_priority_weighting(self):
        """Test realm-based priority weighting"""
        base_score = 0.8
        
        # Project realm should get higher priority
        project_score = self.search_engine._apply_realm_priority(base_score, self.project_realm)
        self.assertAlmostEqual(project_score, 0.8 * 1.2, places=5)
        
        # Global realm should get standard priority
        global_score = self.search_engine._apply_realm_priority(base_score, self.global_realm)
        self.assertAlmostEqual(global_score, 0.8 * 1.0, places=5)
        
        # Unknown realm should get lower priority
        unknown_score = self.search_engine._apply_realm_priority(base_score, "UNKNOWN")
        self.assertAlmostEqual(unknown_score, 0.8 * 0.8, places=5)
    
    def test_keyword_score_calculation(self):
        """Test keyword matching score calculation"""
        query_terms = ["test", "content"]
        chunk = {
            'content': "This is test content for testing",
            'source_document': "test_document.md",
            'section_path': "section/content"
        }
        
        score = self.search_engine._calculate_keyword_score(query_terms, chunk)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_semantic_search_with_mock_data(self):
        """Test semantic search with mock chunk data"""
        query = "test query"
        chunks_data = [
            {
                'chunk_id': 'chunk1',
                'content': 'This is test content',
                'source_document': 'doc1.md',
                'section_path': 'section1',
                'realm_id': self.project_realm,
                'access_count': 5,
                'embedding': json.dumps([0.9, 0.1, 0.0])  # High similarity
            },
            {
                'chunk_id': 'chunk2',
                'content': 'Different content',
                'source_document': 'doc2.md',
                'section_path': 'section2',
                'realm_id': self.global_realm,
                'access_count': 3,
                'embedding': json.dumps([0.1, 0.9, 0.0])  # Lower similarity
            }
        ]
        
        results = self.search_engine.dual_realm_semantic_search(
            query=query,
            chunks_data=chunks_data,
            limit=10,
            threshold=0.1
        )
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
        
        if results:
            # First result should be SearchResult object
            self.assertIsInstance(results[0], SearchResult)
            # Results should be sorted by final_score descending
            if len(results) > 1:
                self.assertGreaterEqual(results[0].final_score, results[1].final_score)
    
    def test_hybrid_search_with_mock_data(self):
        """Test hybrid search combining semantic and keyword scores"""
        query = "test query"
        chunks_data = [
            {
                'chunk_id': 'chunk1',
                'content': 'This is test content with query words',
                'source_document': 'doc1.md',
                'section_path': 'section1',
                'realm_id': self.project_realm,
                'access_count': 5,
                'embedding': json.dumps([0.5, 0.5, 0.0])
            }
        ]
        
        results = self.search_engine.realm_aware_hybrid_search(
            query=query,
            chunks_data=chunks_data,
            limit=10
        )
        
        self.assertIsInstance(results, list)
        if results:
            result = results[0]
            self.assertIsInstance(result, SearchResult)
            self.assertIsNotNone(result.similarity_score)
            self.assertIsNotNone(result.keyword_score)
            self.assertIsNotNone(result.final_score)
    
    def test_search_engine_graceful_degradation(self):
        """Test search engine behavior when embedding service unavailable"""
        # Mock unavailable embedding service
        mock_unavailable_service = Mock()
        mock_unavailable_service.is_available.return_value = False
        
        with patch('mcp_server.services.vector_search.get_embedding_service', return_value=mock_unavailable_service):
            search_engine = RealmAwareVectorSearchEngine(
                project_realm=self.project_realm,
                global_realm=self.global_realm
            )
            
            results = search_engine.dual_realm_semantic_search(
                query="test",
                chunks_data=[],
                limit=10
            )
            
            self.assertEqual(results, [])
    
    def test_search_stats(self):
        """Test search engine statistics"""
        stats = self.search_engine.get_search_stats()
        
        self.assertIn('project_realm', stats)
        self.assertIn('global_realm', stats)
        self.assertIn('semantic_threshold', stats)
        self.assertIn('project_priority', stats)
        self.assertIn('global_priority', stats)
        self.assertIn('cross_realm_enabled', stats)
        self.assertIn('embedding_service_available', stats)


class TestEmbeddingIntegration(unittest.TestCase):
    """Integration tests for embedding functionality"""
    
    def test_environment_configuration_loading(self):
        """Test loading configuration from environment variables"""
        test_env = {
            'EMBEDDING_MODEL': 'test-model',
            'SEMANTIC_SEARCH_THRESHOLD': '0.8',
            'REALM_PRIORITY_PROJECT': '1.5',
            'REALM_PRIORITY_GLOBAL': '0.9',
            'CROSS_REALM_SEARCH_ENABLED': 'true'
        }
        
        with patch.dict(os.environ, test_env):
            # Clear singleton for clean test
            EmbeddingService._instance = None
            EmbeddingService._initialized = False
            
            embedding_service = EmbeddingService()
            search_engine = RealmAwareVectorSearchEngine('TEST_PROJECT', 'GLOBAL')
            
            self.assertEqual(embedding_service.model_name, 'test-model')
            self.assertEqual(search_engine.semantic_threshold, 0.8)
            self.assertEqual(search_engine.project_priority, 1.5)
            self.assertEqual(search_engine.global_priority, 0.9)
            self.assertTrue(search_engine.cross_realm_enabled)
    
    @patch('mcp_server.services.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    def test_end_to_end_embedding_and_search(self):
        """Test end-to-end embedding generation and search"""
        # Mock the transformer model directly
        mock_model = Mock()
        mock_model.encode.side_effect = [
            [0.9, 0.1, 0.0],  # Query embedding
            [0.8, 0.2, 0.0],  # Similar chunk
            [0.1, 0.9, 0.0]   # Different chunk
        ]
        
        # Clear singleton for clean test
        EmbeddingService._instance = None
        EmbeddingService._initialized = False
        
        # Create embedding service and search engine
        embedding_service = EmbeddingService()
        embedding_service.model = mock_model
        
        with patch('mcp_server.services.vector_search.get_embedding_service', return_value=embedding_service):
            search_engine = RealmAwareVectorSearchEngine('PROJ_TEST', 'GLOBAL')
            
            # Prepare test data
            chunks_data = [
                {
                    'chunk_id': 'chunk1',
                    'content': 'Similar content',
                    'source_document': 'doc1.md',
                    'section_path': 'section1',
                    'realm_id': 'PROJ_TEST',
                    'access_count': 5,
                    'embedding': json.dumps([0.8, 0.2, 0.0])
                },
                {
                    'chunk_id': 'chunk2',
                    'content': 'Different content',
                    'source_document': 'doc2.md',
                    'section_path': 'section2',
                    'realm_id': 'GLOBAL',
                    'access_count': 3,
                    'embedding': json.dumps([0.1, 0.9, 0.0])
                }
            ]
            
            # Perform search
            results = search_engine.dual_realm_semantic_search(
                query="test query",
                chunks_data=chunks_data,
                limit=10,
                threshold=0.1
            )
            
            # Verify results
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)
            
            # Check that project realm chunk gets higher priority
            project_chunks = [r for r in results if r.realm_id == 'PROJ_TEST']
            global_chunks = [r for r in results if r.realm_id == 'GLOBAL']
            
            if project_chunks and global_chunks:
                # Project realm chunk should have higher final score due to priority weighting
                max_project_score = max(p.final_score for p in project_chunks)
                max_global_score = max(g.final_score for g in global_chunks)
                self.assertGreater(max_project_score, max_global_score * 0.9)  # Account for similarity differences


def run_all_tests():
    """Run all Phase 1 semantic search tests"""
    test_classes = [
        TestEmbeddingService,
        TestRealmAwareVectorSearchEngine,
        TestEmbeddingIntegration
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
    
    print(f"\n=== Phase 1 Semantic Search Test Summary ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return failed_tests == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)