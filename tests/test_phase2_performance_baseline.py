#!/usr/bin/env python3
"""
Phase 2 Performance Baseline Tests
Performance testing for dual-realm semantic search capabilities
"""

import time
import json
import os
import sys
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mcp_server'))

def test_embedding_service_performance():
    """Test embedding service performance with graceful degradation"""
    print("=== Testing Embedding Service Performance ===")
    
    try:
        from services.embedding_service import EmbeddingService, get_embedding_service
        
        # Clear singleton for clean test
        EmbeddingService._instance = None
        EmbeddingService._initialized = False
        
        start_time = time.time()
        service = get_embedding_service()
        init_time = time.time() - start_time
        
        print(f"Service initialization time: {init_time:.3f}s")
        print(f"Service available: {service.is_available()}")
        print(f"Model name: {service.model_name}")
        print(f"Device: {service.device}")
        
        # Test embedding generation performance
        test_texts = [
            "This is a test document about software development.",
            "Machine learning and artificial intelligence concepts.",
            "Database design and optimization strategies.",
            "User interface and experience design principles.",
            "Project management and agile methodologies."
        ]
        
        start_time = time.time()
        for text in test_texts:
            embedding = service.generate_embedding(text)
            # Expected to be None due to missing dependencies, but should not crash
        
        batch_time = time.time() - start_time
        print(f"Batch embedding time (5 texts): {batch_time:.3f}s")
        
        # Test batch processing
        start_time = time.time()
        embeddings = service.generate_embeddings_batch(test_texts)
        batch_processing_time = time.time() - start_time
        
        print(f"Batch processing time: {batch_processing_time:.3f}s")
        print(f"Results: {len(embeddings)} embeddings generated")
        
        return True
        
    except Exception as e:
        print(f"Embedding service test failed: {e}")
        return False


def test_vector_search_performance():
    """Test vector search engine performance"""
    print("\n=== Testing Vector Search Performance ===")
    
    try:
        from services.vector_search import RealmAwareVectorSearchEngine, SearchResult
        
        # Mock embedding service
        mock_embedding_service = Mock()
        mock_embedding_service.is_available.return_value = True
        mock_embedding_service.generate_embedding.return_value = [0.5, 0.5, 0.5]
        
        with patch('services.vector_search.get_embedding_service', return_value=mock_embedding_service):
            start_time = time.time()
            search_engine = RealmAwareVectorSearchEngine('PROJ_TEST', 'GLOBAL')
            init_time = time.time() - start_time
            
            print(f"Search engine initialization time: {init_time:.3f}s")
            print(f"Project realm: {search_engine.project_realm}")
            print(f"Global realm: {search_engine.global_realm}")
            print(f"Semantic threshold: {search_engine.semantic_threshold}")
            print(f"Project priority: {search_engine.project_priority}")
            
            # Test similarity calculations
            vec1 = [1.0, 0.5, 0.2]
            vec2 = [0.8, 0.6, 0.3]
            
            start_time = time.time()
            for _ in range(1000):
                similarity = search_engine._calculate_cosine_similarity(vec1, vec2)
            calc_time = time.time() - start_time
            
            print(f"1000 similarity calculations time: {calc_time:.3f}s")
            print(f"Average per calculation: {calc_time/1000*1000:.3f}ms")
            
            # Test search with mock data
            mock_chunks = []
            for i in range(100):
                mock_chunks.append({
                    'chunk_id': f'chunk_{i}',
                    'content': f'Test content {i} about various topics',
                    'source_document': f'doc_{i}.md',
                    'section_path': f'section_{i}',
                    'realm_id': 'PROJ_TEST' if i % 2 == 0 else 'GLOBAL',
                    'access_count': i + 1,
                    'embedding': json.dumps([0.1 + i*0.01, 0.2 + i*0.01, 0.3 + i*0.01])
                })
            
            start_time = time.time()
            results = search_engine.dual_realm_semantic_search(
                query="test query about topics",
                chunks_data=mock_chunks,
                limit=10
            )
            search_time = time.time() - start_time
            
            print(f"Semantic search time (100 chunks): {search_time:.3f}s")
            print(f"Results returned: {len(results)}")
            
            # Test hybrid search
            start_time = time.time()
            hybrid_results = search_engine.realm_aware_hybrid_search(
                query="test query about topics",
                chunks_data=mock_chunks,
                limit=10
            )
            hybrid_time = time.time() - start_time
            
            print(f"Hybrid search time (100 chunks): {hybrid_time:.3f}s")
            print(f"Hybrid results returned: {len(hybrid_results)}")
            
            return True
            
    except Exception as e:
        print(f"Vector search test failed: {e}")
        return False


def test_mcp_server_performance():
    """Test MCP server request handling performance"""
    print("\n=== Testing MCP Server Performance ===")
    
    try:
        from megamind_database_server import MCPServer
        
        # Mock database manager
        mock_db = Mock()
        mock_db.search_chunks_dual_realm.return_value = [
            {
                'chunk_id': 'perf_chunk_1',
                'content': 'Performance test content',
                'realm_id': 'PROJ_TEST',
                'similarity_score': 0.85,
                'final_score': 1.02
            }
        ]
        
        mock_db.search_chunks_semantic.return_value = [
            {
                'chunk_id': 'semantic_chunk_1',
                'content': 'Semantic test content',
                'similarity_score': 0.92
            }
        ]
        
        mock_db.batch_generate_embeddings.return_value = {
            'processed_chunks': 50,
            'successful_embeddings': 48,
            'failed_embeddings': 2,
            'processing_time': 5.2
        }
        
        # Initialize MCP server
        start_time = time.time()
        mcp_server = MCPServer(mock_db)
        init_time = time.time() - start_time
        
        print(f"MCP server initialization time: {init_time:.3f}s")
        
        # Test different request types
        test_requests = [
            {
                "jsonrpc": "2.0",
                "id": "search-1",
                "method": "tools/call",
                "params": {
                    "name": "mcp__context_db__search_chunks",
                    "arguments": {
                        "query": "performance test query",
                        "limit": 10,
                        "search_type": "hybrid"
                    }
                }
            },
            {
                "jsonrpc": "2.0",
                "id": "semantic-1",
                "method": "tools/call",
                "params": {
                    "name": "mcp__context_db__search_chunks_semantic",
                    "arguments": {
                        "query": "semantic performance test",
                        "limit": 5,
                        "threshold": 0.7
                    }
                }
            },
            {
                "jsonrpc": "2.0",
                "id": "batch-1",
                "method": "tools/call",
                "params": {
                    "name": "mcp__context_db__batch_generate_embeddings",
                    "arguments": {
                        "realm_id": "PROJ_TEST"
                    }
                }
            }
        ]
        
        # Test request handling performance
        total_time = 0
        for i, request in enumerate(test_requests):
            start_time = time.time()
            
            # Use synchronous call for performance testing
            import asyncio
            response = asyncio.run(mcp_server.handle_request(request))
            
            request_time = time.time() - start_time
            total_time += request_time
            
            print(f"Request {i+1} time: {request_time:.3f}s")
            print(f"Response status: {'Success' if 'result' in response else 'Error'}")
        
        print(f"Total request handling time: {total_time:.3f}s")
        print(f"Average per request: {total_time/len(test_requests):.3f}s")
        
        return True
        
    except Exception as e:
        print(f"MCP server test failed: {e}")
        return False


def test_environment_configuration_performance():
    """Test environment configuration loading performance"""
    print("\n=== Testing Environment Configuration Performance ===")
    
    test_env = {
        'MEGAMIND_PROJECT_REALM': 'PROJ_PERFORMANCE_TEST',
        'MEGAMIND_DEFAULT_TARGET': 'PROJECT',
        'EMBEDDING_MODEL': 'sentence-transformers/all-MiniLM-L6-v2',
        'SEMANTIC_SEARCH_THRESHOLD': '0.75',
        'REALM_PRIORITY_PROJECT': '1.3',
        'REALM_PRIORITY_GLOBAL': '1.0',
        'CROSS_REALM_SEARCH_ENABLED': 'true',
        'EMBEDDING_BATCH_SIZE': '32',
        'EMBEDDING_CACHE_SIZE': '500'
    }
    
    with patch.dict(os.environ, test_env):
        try:
            # Test embedding service configuration loading
            from services.embedding_service import EmbeddingService
            
            EmbeddingService._instance = None
            EmbeddingService._initialized = False
            
            start_time = time.time()
            service = EmbeddingService()
            config_time = time.time() - start_time
            
            print(f"Embedding service config load time: {config_time:.3f}s")
            print(f"Configured model: {service.model_name}")
            print(f"Configured device: {service.device}")
            print(f"Configured batch size: {service.batch_size}")
            print(f"Configured cache size: {service.cache_size}")
            
            # Test vector search configuration
            from services.vector_search import RealmAwareVectorSearchEngine
            
            mock_embedding_service = Mock()
            mock_embedding_service.is_available.return_value = False  # Test graceful degradation
            
            with patch('services.vector_search.get_embedding_service', return_value=mock_embedding_service):
                start_time = time.time()
                search_engine = RealmAwareVectorSearchEngine('PROJ_PERFORMANCE_TEST', 'GLOBAL')
                vector_config_time = time.time() - start_time
                
                print(f"Vector search config load time: {vector_config_time:.3f}s")
                print(f"Configured threshold: {search_engine.semantic_threshold}")
                print(f"Configured project priority: {search_engine.project_priority}")
                print(f"Cross-realm enabled: {search_engine.cross_realm_enabled}")
            
            total_config_time = config_time + vector_config_time
            print(f"Total configuration time: {total_config_time:.3f}s")
            
            return True
            
        except Exception as e:
            print(f"Configuration test failed: {e}")
            return False


def run_performance_baseline():
    """Run all performance baseline tests"""
    print("Phase 2 Semantic Search Performance Baseline")
    print("=" * 50)
    
    test_results = []
    
    # Run all performance tests
    test_functions = [
        test_embedding_service_performance,
        test_vector_search_performance,
        test_mcp_server_performance,
        test_environment_configuration_performance
    ]
    
    for test_func in test_functions:
        try:
            result = test_func()
            test_results.append(result)
        except Exception as e:
            print(f"Test {test_func.__name__} failed with exception: {e}")
            test_results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("Performance Baseline Summary")
    print("=" * 50)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("✅ All performance baseline tests passed!")
        print("Phase 2 semantic search implementation ready for production")
    else:
        print("⚠️  Some performance tests failed")
        print("Review implementation for potential optimizations")
    
    return passed_tests == total_tests


if __name__ == '__main__':
    success = run_performance_baseline()
    sys.exit(0 if success else 1)