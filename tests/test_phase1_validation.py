#!/usr/bin/env python3
"""
MegaMind Context Database - Phase 1 Validation Tests
Tests core infrastructure functionality
"""

import unittest
import os
import sys
import tempfile
import shutil
import json
from pathlib import Path
import mysql.connector
from mysql.connector import pooling

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.markdown_ingester import MarkdownIngester
from mcp_server.megamind_database_server import DatabaseManager

class TestDatabaseSchema(unittest.TestCase):
    """Test database schema and basic operations"""
    
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
        """Setup test database connection"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            self.cursor = self.connection.cursor()
        except Exception as e:
            self.skipTest(f"Cannot connect to test database: {e}")
    
    def tearDown(self):
        """Cleanup database connection"""
        if hasattr(self, 'connection') and self.connection.is_connected():
            self.cursor.close()
            self.connection.close()
    
    def test_tables_exist(self):
        """Test that all required tables exist"""
        self.cursor.execute("SHOW TABLES")
        tables = [table[0] for table in self.cursor.fetchall()]
        
        required_tables = [
            'megamind_chunks',
            'megamind_chunk_relationships',
            'megamind_chunk_tags'
        ]
        
        for table in required_tables:
            self.assertIn(table, tables, f"Required table {table} is missing")
    
    def test_chunks_table_structure(self):
        """Test megamind_chunks table structure"""
        self.cursor.execute("DESCRIBE megamind_chunks")
        columns = {row[0]: row[1] for row in self.cursor.fetchall()}
        
        required_columns = {
            'chunk_id': 'varchar(50)',
            'content': 'text',
            'source_document': 'varchar(255)',
            'section_path': 'varchar(500)',
            'chunk_type': "enum('rule','function','section','example')",
            'line_count': 'int',
            'created_at': 'timestamp',
            'last_accessed': 'timestamp',
            'access_count': 'int'
        }
        
        for col, expected_type in required_columns.items():
            self.assertIn(col, columns, f"Required column {col} is missing")
    
    def test_relationships_table_structure(self):
        """Test megamind_chunk_relationships table structure"""
        self.cursor.execute("DESCRIBE megamind_chunk_relationships")
        columns = {row[0]: row[1] for row in self.cursor.fetchall()}
        
        required_columns = [
            'relationship_id', 'chunk_id', 'related_chunk_id',
            'relationship_type', 'strength', 'discovered_by'
        ]
        
        for col in required_columns:
            self.assertIn(col, columns, f"Required column {col} is missing")
    
    def test_tags_table_structure(self):
        """Test megamind_chunk_tags table structure"""
        self.cursor.execute("DESCRIBE megamind_chunk_tags")
        columns = {row[0]: row[1] for row in self.cursor.fetchall()}
        
        required_columns = [
            'tag_id', 'chunk_id', 'tag_type', 'tag_value'
        ]
        
        for col in required_columns:
            self.assertIn(col, columns, f"Required column {col} is missing")

class TestMarkdownIngester(unittest.TestCase):
    """Test markdown ingestion functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test database and sample files"""
        cls.db_config = {
            'host': os.getenv('TEST_DB_HOST', '10.255.250.22'),
            'port': os.getenv('TEST_DB_PORT', '3309'),
            'database': os.getenv('TEST_DB_NAME', 'megamind_database_test'),
            'user': os.getenv('TEST_DB_USER', 'megamind_user'),
            'password': os.getenv('TEST_DB_PASSWORD', 'megamind_secure_pass')
        }
        
        # Create temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create sample markdown file
        cls.sample_md = Path(cls.temp_dir) / "test_document.md"
        with open(cls.sample_md, 'w') as f:
            f.write("""# Test Document

## Database Rules

This section contains important database rules.

### Rule 1: Always use transactions

When working with database operations, always use transactions.

```sql
BEGIN;
UPDATE users SET status = 'active';
COMMIT;
```

## Functions

### Calculate Total

This function calculates totals.

```python
def calculate_total(items):
    return sum(item.price for item in items)
```

## Examples

Here are some examples of usage.
""")
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup test files"""
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Setup ingester"""
        try:
            self.ingester = MarkdownIngester(self.db_config)
        except Exception as e:
            self.skipTest(f"Cannot setup ingester: {e}")
    
    def test_parse_markdown_file(self):
        """Test markdown file parsing"""
        chunks = self.ingester.parse_markdown_file(str(self.sample_md))
        
        self.assertGreater(len(chunks), 0, "Should parse at least one chunk")
        
        # Check chunk properties
        chunk = chunks[0]
        self.assertTrue(chunk.chunk_id, "Chunk should have an ID")
        self.assertTrue(chunk.content, "Chunk should have content")
        self.assertEqual(chunk.source_document, "test_document.md")
        self.assertTrue(chunk.section_path, "Chunk should have section path")
        self.assertIn(chunk.chunk_type, ['rule', 'function', 'section', 'example'])
        self.assertGreater(chunk.line_count, 0, "Chunk should have positive line count")
    
    def test_chunk_type_detection(self):
        """Test chunk type detection logic"""
        chunks = self.ingester.parse_markdown_file(str(self.sample_md))
        
        # Should detect function type for code examples
        function_chunks = [c for c in chunks if c.chunk_type == 'function']
        self.assertGreater(len(function_chunks), 0, "Should detect function chunks")
        
        # Should detect rule type for rule sections
        rule_chunks = [c for c in chunks if 'rule' in c.section_path.lower()]
        self.assertGreater(len(rule_chunks), 0, "Should detect rule-related chunks")
    
    def test_insert_chunks(self):
        """Test chunk insertion into database"""
        chunks = self.ingester.parse_markdown_file(str(self.sample_md))
        
        # Clean up any existing test data
        connection = self.ingester.get_connection()
        cursor = connection.cursor()
        cursor.execute("DELETE FROM megamind_chunks WHERE source_document = 'test_document.md'")
        connection.commit()
        connection.close()
        
        # Insert chunks
        success = self.ingester.insert_chunks(chunks)
        self.assertTrue(success, "Chunk insertion should succeed")
        
        # Verify insertion
        connection = self.ingester.get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM megamind_chunks WHERE source_document = 'test_document.md'")
        count = cursor.fetchone()[0]
        connection.close()
        
        self.assertEqual(count, len(chunks), "All chunks should be inserted")

class TestDatabaseManager(unittest.TestCase):
    """Test database manager functionality"""
    
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
        """Setup database manager"""
        try:
            self.db_manager = DatabaseManager(self.db_config)
        except Exception as e:
            self.skipTest(f"Cannot setup database manager: {e}")
    
    def test_search_chunks(self):
        """Test chunk search functionality"""
        results = self.db_manager.search_chunks("database", limit=5)
        
        # Should return results (if sample data exists)
        for result in results:
            self.assertTrue(result.chunk_id, "Result should have chunk_id")
            self.assertTrue(result.content, "Result should have content")
            self.assertGreaterEqual(result.relevance_score, 0.0, "Should have relevance score")
    
    def test_get_chunk(self):
        """Test individual chunk retrieval"""
        # First get a chunk ID from search
        search_results = self.db_manager.search_chunks("database", limit=1)
        
        if search_results:
            chunk_id = search_results[0].chunk_id
            chunk = self.db_manager.get_chunk(chunk_id)
            
            self.assertIsNotNone(chunk, "Should retrieve chunk")
            self.assertEqual(chunk.chunk_id, chunk_id, "Should return correct chunk")
    
    def test_track_access(self):
        """Test access tracking functionality"""
        # Get a chunk ID
        search_results = self.db_manager.search_chunks("database", limit=1)
        
        if search_results:
            chunk_id = search_results[0].chunk_id
            original_count = search_results[0].access_count
            
            # Track access
            success = self.db_manager.track_access(chunk_id, "test_access")
            self.assertTrue(success, "Access tracking should succeed")
            
            # Verify count increased
            chunk = self.db_manager.get_chunk(chunk_id)
            self.assertGreater(chunk.access_count, original_count, "Access count should increase")

class TestSystemIntegration(unittest.TestCase):
    """Test end-to-end system functionality"""
    
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
    
    def test_full_workflow(self):
        """Test complete ingestion to retrieval workflow"""
        try:
            # Create test document
            temp_dir = tempfile.mkdtemp()
            test_file = Path(temp_dir) / "integration_test.md"
            
            with open(test_file, 'w') as f:
                f.write("""# Integration Test

## Search Function

This is a test function for searching.

```python
def search_function(query):
    return database.search(query)
```

## Configuration Rules

Always configure the system properly.
""")
            
            # Ingest document
            ingester = MarkdownIngester(self.db_config)
            chunks = ingester.parse_markdown_file(str(test_file))
            success = ingester.insert_chunks(chunks)
            
            self.assertTrue(success, "Ingestion should succeed")
            
            # Search for content
            db_manager = DatabaseManager(self.db_config)
            results = db_manager.search_chunks("search function")
            
            self.assertGreater(len(results), 0, "Should find relevant chunks")
            
            # Verify content
            found_content = any("search_function" in result.content for result in results)
            self.assertTrue(found_content, "Should find function content")
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            self.fail(f"Integration test failed: {e}")

def run_validation_tests():
    """Run all Phase 1 validation tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseSchema))
    suite.addTests(loader.loadTestsFromTestCase(TestMarkdownIngester))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseManager))
    suite.addTests(loader.loadTestsFromTestCase(TestSystemIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("MegaMind Context Database - Phase 1 Validation Tests")
    print("=" * 60)
    
    success = run_validation_tests()
    
    if success:
        print("\n✅ All Phase 1 validation tests passed!")
        exit(0)
    else:
        print("\n❌ Some Phase 1 validation tests failed!")
        exit(1)