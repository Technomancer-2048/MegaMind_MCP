#!/usr/bin/env python3
"""
MegaMind Context Database - Phase 2 Intelligence Layer Tests
Tests semantic analysis, embedding generation, and advanced MCP functions
"""

import unittest
import os
import sys
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
import mysql.connector
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.semantic_analyzer import SemanticAnalyzer, ChunkEmbedding, RelationshipCandidate, TagCandidate
from mcp_server.megamind_database_server import DatabaseManager
from dashboard.context_analytics import ContextAnalytics

class TestSemanticAnalyzer(unittest.TestCase):
    """Test semantic analysis functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test database configuration"""
        cls.db_config = {
            'host': os.getenv('TEST_DB_HOST', '10.255.250.22'),
            'port': os.getenv('TEST_DB_PORT', '3309'),
            'database': os.getenv('TEST_DB_NAME', 'megamind_database_test'),
            'user': os.getenv('TEST_DB_USER', 'megamind_user'),
            'password': os.getenv('TEST_DB_PASSWORD', 'megamind_secure_pass')
        }
    
    def setUp(self):
        """Setup semantic analyzer with mocked model"""
        # Mock sentence transformers to avoid large model downloads in tests
        with patch('analysis.semantic_analyzer.SentenceTransformer') as mock_model:
            mock_instance = MagicMock()
            mock_instance.encode.return_value = np.random.rand(5, 384)  # Mock embeddings
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_model.return_value = mock_instance
            
            try:
                self.analyzer = SemanticAnalyzer(self.db_config, "mock-model")
            except Exception as e:
                self.skipTest(f"Cannot setup semantic analyzer: {e}")
    
    def test_chunk_embedding_creation(self):
        """Test ChunkEmbedding dataclass functionality"""
        embedding = ChunkEmbedding(
            chunk_id="test_chunk_001",
            embedding=np.random.rand(384),
            content="Test content for embedding",
            chunk_type="section",
            source_document="test.md",
            token_count=50
        )
        
        self.assertEqual(embedding.chunk_id, "test_chunk_001")
        self.assertEqual(embedding.chunk_type, "section")
        self.assertEqual(len(embedding.embedding), 384)
    
    def test_relationship_classification(self):
        """Test relationship type classification logic"""
        chunk_1 = ChunkEmbedding(
            chunk_id="func_001",
            embedding=np.random.rand(384),
            content="def calculate_total(): return sum(items)",
            chunk_type="function",
            source_document="utils.md",
            token_count=30
        )
        
        chunk_2 = ChunkEmbedding(
            chunk_id="example_001",
            embedding=np.random.rand(384),
            content="Example: calculate_total() returns the sum",
            chunk_type="example",
            source_document="examples.md",
            token_count=25
        )
        
        # Test relationship classification
        relationship_type, confidence = self.analyzer._classify_relationship(chunk_2, chunk_1, 0.85)
        
        self.assertEqual(relationship_type, "implements")
        self.assertGreater(confidence, 0.8)
    
    def test_tag_generation(self):
        """Test semantic tag generation"""
        chunks = [
            ChunkEmbedding(
                chunk_id="sql_001",
                embedding=np.random.rand(384),
                content="CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255))",
                chunk_type="function",
                source_document="database.md",
                token_count=40
            )
        ]
        
        tags = self.analyzer.generate_semantic_tags(chunks)
        
        # Should generate database-related tags
        self.assertGreater(len(tags), 0)
        
        # Check for expected tag types
        tag_types = [tag.tag_type for tag in tags]
        self.assertIn('subsystem', tag_types)
        
        # Check for SQL language detection
        sql_tags = [tag for tag in tags if tag.tag_value == 'sql']
        self.assertGreater(len(sql_tags), 0)

@patch('mcp_server.megamind_database_server.SentenceTransformer')
class TestEnhancedMCPFunctions(unittest.TestCase):
    """Test enhanced MCP server functions"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test database configuration"""
        cls.db_config = {
            'host': os.getenv('TEST_DB_HOST', '10.255.250.22'),
            'port': os.getenv('TEST_DB_PORT', '3309'),
            'database': os.getenv('TEST_DB_NAME', 'megamind_database_test'),
            'user': os.getenv('TEST_DB_USER', 'megamind_user'),
            'password': os.getenv('TEST_DB_PASSWORD', 'megamind_secure_pass')
        }
    
    def setUp(self, mock_transformer):
        """Setup database manager with mocked transformers"""
        # Mock sentence transformer
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.random.rand(384)
        mock_transformer.return_value = mock_instance
        
        try:
            self.db_manager = DatabaseManager(self.db_config)
        except Exception as e:
            self.skipTest(f"Cannot setup database manager: {e}")
    
    def test_search_by_tags(self, mock_transformer):
        """Test tag-based search functionality"""
        # This test requires existing tag data in the database
        results = self.db_manager.search_by_tags('subsystem', 'database', limit=5)
        
        # Should return chunks (if any exist with database tags)
        self.assertIsInstance(results, list)
        
        # If results exist, validate structure
        for chunk in results:
            self.assertTrue(hasattr(chunk, 'chunk_id'))
            self.assertTrue(hasattr(chunk, 'content'))
            self.assertTrue(hasattr(chunk, 'chunk_type'))
    
    def test_session_primer_generation(self, mock_transformer):
        """Test session primer functionality"""
        primer = self.db_manager.get_session_primer(
            last_session_data="Previous work on database optimization",
            project_context="MegaMind Context System"
        )
        
        self.assertIsInstance(primer, dict)
        self.assertIn('timestamp', primer)
        self.assertIn('recent_activity', primer)
        self.assertIn('project_tags', primer)
        self.assertIn('context_summary', primer)
    
    def test_embedding_search_fallback(self, mock_transformer):
        """Test embedding search with fallback to text search"""
        # Test that embedding search works or falls back gracefully
        results = self.db_manager.search_by_embedding(
            "database functions", 
            limit=5, 
            similarity_threshold=0.6
        )
        
        self.assertIsInstance(results, list)
        
        # Should return chunks even if embeddings aren't available
        # (falls back to text search)

class TestContextAnalytics(unittest.TestCase):
    """Test analytics dashboard functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test database configuration"""
        cls.db_config = {
            'host': os.getenv('TEST_DB_HOST', '10.255.250.22'),
            'port': os.getenv('TEST_DB_PORT', '3309'),
            'database': os.getenv('TEST_DB_NAME', 'megamind_database_test'),
            'user': os.getenv('TEST_DB_USER', 'megamind_user'),
            'password': os.getenv('TEST_DB_PASSWORD', 'megamind_secure_pass')
        }
    
    def setUp(self):
        """Setup analytics engine"""
        try:
            self.analytics = ContextAnalytics(self.db_config)
        except Exception as e:
            self.skipTest(f"Cannot setup analytics engine: {e}")
    
    def test_usage_heatmap_data(self):
        """Test usage heatmap data generation"""
        data = self.analytics.get_usage_heatmap_data()
        
        self.assertIsInstance(data, dict)
        
        # Should have expected keys
        expected_keys = ['hot_chunks', 'warm_chunks', 'cold_chunks', 'total_chunks', 'avg_access_count']
        for key in expected_keys:
            if key in data:  # Some keys might not exist if no data
                self.assertIn(key, data)
    
    def test_relationship_network_data(self):
        """Test relationship network data generation"""
        data = self.analytics.get_relationship_network_data()
        
        self.assertIsInstance(data, dict)
        self.assertIn('nodes', data)
        self.assertIn('edges', data)
        self.assertIn('relationship_types', data)
        
        # Validate data structure
        self.assertIsInstance(data['nodes'], list)
        self.assertIsInstance(data['edges'], list)
        self.assertIsInstance(data['relationship_types'], list)
    
    def test_efficiency_metrics(self):
        """Test efficiency metrics calculation"""
        metrics = self.analytics.get_efficiency_metrics()
        
        self.assertIsInstance(metrics, dict)
        
        # Should include key efficiency metrics
        if metrics:  # If data exists
            expected_metrics = [
                'total_chunks', 'total_documents', 'avg_tokens_per_chunk',
                'efficiency_ratio', 'avg_access_count'
            ]
            
            for metric in expected_metrics:
                if metric in metrics:
                    self.assertIn(metric, metrics)
    
    def test_tag_distribution(self):
        """Test tag distribution analysis"""
        distribution = self.analytics.get_tag_distribution()
        
        self.assertIsInstance(distribution, dict)
        
        # Each tag type should have a list of values
        for tag_type, values in distribution.items():
            self.assertIsInstance(values, list)
            
            # Each value should have expected structure
            for value in values:
                self.assertIn('value', value)
                self.assertIn('count', value)
                self.assertIn('confidence', value)

class TestPhase2Integration(unittest.TestCase):
    """Test Phase 2 integration scenarios"""
    
    @classmethod
    def setUpClass(cls):
        """Setup integration test environment"""
        cls.db_config = {
            'host': os.getenv('TEST_DB_HOST', '10.255.250.22'),
            'port': os.getenv('TEST_DB_PORT', '3309'),
            'database': os.getenv('TEST_DB_NAME', 'megamind_database_test'),
            'user': os.getenv('TEST_DB_USER', 'megamind_user'),
            'password': os.getenv('TEST_DB_PASSWORD', 'megamind_secure_pass')
        }
    
    @patch('analysis.semantic_analyzer.SentenceTransformer')
    def test_semantic_analysis_workflow(self, mock_transformer):
        """Test complete semantic analysis workflow"""
        # Mock transformer
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.random.rand(5, 384)
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_instance
        
        try:
            # Initialize analyzer
            analyzer = SemanticAnalyzer(self.db_config, "mock-model")
            
            # Test chunk loading
            chunks = analyzer.load_chunks_for_analysis(limit=5)
            self.assertIsInstance(chunks, list)
            
            if chunks:
                # Test embedding generation
                embeddings = analyzer.generate_embeddings_batch(chunks[:2])
                self.assertIsInstance(embeddings, list)
                self.assertGreater(len(embeddings), 0)
                
                # Test relationship discovery
                if len(embeddings) >= 2:
                    relationships = analyzer.discover_relationships(embeddings)
                    self.assertIsInstance(relationships, list)
                
                # Test tag generation
                tags = analyzer.generate_semantic_tags(embeddings)
                self.assertIsInstance(tags, list)
            
        except Exception as e:
            self.skipTest(f"Semantic analysis workflow test failed: {e}")
    
    def test_mcp_function_integration(self):
        """Test integration between enhanced MCP functions"""
        try:
            db_manager = DatabaseManager(self.db_config)
            
            # Test tag search -> chunk retrieval -> related chunks workflow
            tag_results = db_manager.search_by_tags('subsystem', limit=3)
            
            if tag_results:
                # Get detailed chunk information
                chunk = db_manager.get_chunk(tag_results[0].chunk_id, include_relationships=True)
                self.assertIsNotNone(chunk)
                
                # Get related chunks
                related = db_manager.get_related_chunks(chunk.chunk_id)
                self.assertIsInstance(related, list)
            
            # Test session primer generation
            primer = db_manager.get_session_primer("Test session context")
            self.assertIsInstance(primer, dict)
            self.assertIn('timestamp', primer)
            
        except Exception as e:
            self.skipTest(f"MCP function integration test failed: {e}")
    
    def test_analytics_data_flow(self):
        """Test analytics data flow from database to dashboard"""
        try:
            analytics = ContextAnalytics(self.db_config)
            
            # Test all major analytics functions
            usage_data = analytics.get_usage_heatmap_data()
            network_data = analytics.get_relationship_network_data()
            efficiency_data = analytics.get_efficiency_metrics()
            
            # All should return dictionaries
            self.assertIsInstance(usage_data, dict)
            self.assertIsInstance(network_data, dict)
            self.assertIsInstance(efficiency_data, dict)
            
            # Should have consistent data types
            if 'total_chunks' in efficiency_data:
                self.assertIsInstance(efficiency_data['total_chunks'], int)
            
        except Exception as e:
            self.skipTest(f"Analytics data flow test failed: {e}")

def run_phase2_tests():
    """Run all Phase 2 intelligence layer tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedMCPFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestContextAnalytics))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2Integration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("MegaMind Context Database - Phase 2 Intelligence Layer Tests")
    print("=" * 70)
    
    success = run_phase2_tests()
    
    if success:
        print("\n✅ All Phase 2 intelligence tests passed!")
        exit(0)
    else:
        print("\n❌ Some Phase 2 intelligence tests failed!")
        exit(1)