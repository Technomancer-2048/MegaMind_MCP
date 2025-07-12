#!/usr/bin/env python3
"""
Phase 3 Ingestion Integration Tests
End-to-end testing for realm-aware ingestion with semantic embedding generation
"""

import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mcp_server'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

from realm_aware_markdown_ingester import RealmAwareMarkdownIngester, RealmAwareChunkMetadata
from bulk_semantic_ingester import BulkSemanticIngester, BulkIngestionConfig


class TestRealmAwareMarkdownIngester(unittest.TestCase):
    """Test realm-aware markdown ingester functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_db_config = {
            'host': 'localhost',
            'port': '3306',
            'database': 'test_megamind',
            'user': 'test_user',
            'password': 'test_password'
        }
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # Test environment variables
        self.test_env = {
            'MEGAMIND_PROJECT_REALM': 'PROJ_TEST',
            'MEGAMIND_DEFAULT_TARGET': 'PROJECT',
            'EMBEDDING_MODEL': 'sentence-transformers/all-MiniLM-L6-v2'
        }
    
    @patch.dict(os.environ, {
        'MEGAMIND_PROJECT_REALM': 'PROJ_TEST',
        'MEGAMIND_DEFAULT_TARGET': 'PROJECT'
    })
    @patch('realm_aware_markdown_ingester.RealmAwareMegaMindDatabase')
    @patch('realm_aware_markdown_ingester.get_embedding_service')
    def test_ingester_initialization(self, mock_embedding_service, mock_database):
        """Test ingester initialization with realm awareness"""
        # Mock embedding service
        mock_embedding_service.return_value.is_available.return_value = True
        
        # Mock database
        mock_database.return_value = Mock()
        
        # Initialize ingester
        ingester = RealmAwareMarkdownIngester(
            db_config=self.mock_db_config,
            target_realm='PROJECT',
            session_id='test_session'
        )
        
        # Verify initialization
        self.assertEqual(ingester.target_realm, 'PROJECT')
        self.assertEqual(ingester.project_realm, 'PROJ_TEST')
        self.assertEqual(ingester.session_id, 'test_session')
        mock_database.assert_called_once_with(self.mock_db_config)
        mock_embedding_service.assert_called_once()
    
    def test_markdown_parsing(self):
        """Test markdown file parsing into chunks"""
        # Create test markdown file
        test_content = """# Introduction
This is the introduction section.

## Overview
This section provides an overview of the project.

### Implementation Details
Here are the implementation details.

```python
def example_function():
    return "Hello, World!"
```

## Conclusion
This is the conclusion.
"""
        
        test_file = Path(self.temp_dir) / "test.md"
        test_file.write_text(test_content)
        
        # Mock database and embedding service
        with patch('realm_aware_markdown_ingester.RealmAwareMegaMindDatabase') as mock_db, \
             patch('realm_aware_markdown_ingester.get_embedding_service') as mock_embedding:
            
            mock_embedding.return_value.is_available.return_value = False
            
            ingester = RealmAwareMarkdownIngester(self.mock_db_config)
            chunks = ingester.parse_markdown_file(str(test_file))
            
            # Verify chunks were created
            self.assertGreater(len(chunks), 0)
            
            # Check chunk structure
            for chunk in chunks:
                self.assertIsInstance(chunk, RealmAwareChunkMetadata)
                self.assertIsNotNone(chunk.chunk_id)
                self.assertIsNotNone(chunk.content)
                self.assertIsNotNone(chunk.section_path)
                self.assertIn(chunk.chunk_type, ['section', 'function', 'rule', 'example'])
            
            # Verify hierarchical section paths
            section_paths = [chunk.section_path for chunk in chunks]
            self.assertIn('/introduction', section_paths)
            self.assertIn('/introduction/overview', section_paths)
    
    @patch.dict(os.environ, {
        'MEGAMIND_PROJECT_REALM': 'PROJ_TEST',
        'MEGAMIND_DEFAULT_TARGET': 'PROJECT'
    })
    def test_realm_assignment_logic(self):
        """Test automatic realm assignment based on content"""
        # Mock database and embedding service
        with patch('realm_aware_markdown_ingester.RealmAwareMegaMindDatabase') as mock_db, \
             patch('realm_aware_markdown_ingester.get_embedding_service') as mock_embedding:
            
            mock_embedding.return_value.is_available.return_value = False
            
            ingester = RealmAwareMarkdownIngester(self.mock_db_config)
            
            # Test global realm content
            global_chunk = RealmAwareChunkMetadata(
                chunk_id='global_test',
                content='Company standards and organizational policies for all teams.',
                source_document='standards.md',
                section_path='/standards',
                chunk_type='rule',
                line_count=1,
                token_count=10,
                start_line=1,
                end_line=1,
                realm_id=''
            )
            
            realm_id = ingester._determine_chunk_realm(global_chunk)
            self.assertEqual(realm_id, 'GLOBAL')
            
            # Test project realm content
            project_chunk = RealmAwareChunkMetadata(
                chunk_id='project_test',
                content='Implementation details for the new feature API endpoint.',
                source_document='api.md',
                section_path='/implementation',
                chunk_type='function',
                line_count=1,
                token_count=10,
                start_line=1,
                end_line=1,
                realm_id=''
            )
            
            realm_id = ingester._determine_chunk_realm(project_chunk)
            self.assertEqual(realm_id, 'PROJ_TEST')
    
    @patch.dict(os.environ, {
        'MEGAMIND_PROJECT_REALM': 'PROJ_TEST'
    })
    @patch('realm_aware_markdown_ingester.RealmAwareMegaMindDatabase')
    @patch('realm_aware_markdown_ingester.get_embedding_service')
    def test_embedding_generation_integration(self, mock_embedding_service, mock_database):
        """Test embedding generation during chunk processing"""
        # Mock embedding service
        mock_embedding = Mock()
        mock_embedding.is_available.return_value = True
        mock_embedding.generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
        mock_embedding_service.return_value = mock_embedding
        
        # Mock database
        mock_database.return_value = Mock()
        
        ingester = RealmAwareMarkdownIngester(self.mock_db_config)
        
        # Test chunk with embedding generation
        test_chunk = RealmAwareChunkMetadata(
            chunk_id='embed_test',
            content='Test content for embedding generation',
            source_document='test.md',
            section_path='/test',
            chunk_type='section',
            line_count=1,
            token_count=6,
            start_line=1,
            end_line=1,
            realm_id=''
        )
        
        processed_chunk = ingester._process_chunk_with_realm(test_chunk)
        
        # Verify embedding was generated
        self.assertIsNotNone(processed_chunk)
        self.assertIsNotNone(processed_chunk.embedding)
        self.assertEqual(processed_chunk.embedding, [0.1, 0.2, 0.3, 0.4])
        self.assertIsNotNone(processed_chunk.embedding_hash)
        self.assertIsNotNone(processed_chunk.content_hash)
        
        # Verify embedding service was called with realm context
        mock_embedding.generate_embedding.assert_called_once_with(
            'Test content for embedding generation',
            realm_context=processed_chunk.realm_id
        )
    
    @patch('realm_aware_markdown_ingester.RealmAwareMegaMindDatabase')
    @patch('realm_aware_markdown_ingester.get_embedding_service')
    def test_file_ingestion_end_to_end(self, mock_embedding_service, mock_database):
        """Test complete file ingestion workflow"""
        # Create test file
        test_content = """# Project Documentation

## API Design
This section describes the API design for the project.

### Authentication
Authentication is handled via JWT tokens.

## Database Schema
The database schema includes the following tables.
"""
        
        test_file = Path(self.temp_dir) / "project_docs.md"
        test_file.write_text(test_content)
        
        # Mock embedding service
        mock_embedding = Mock()
        mock_embedding.is_available.return_value = True
        mock_embedding.generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_embedding_service.return_value = mock_embedding
        
        # Mock database
        mock_db = Mock()
        mock_db.create_chunk_with_target.return_value = 'chunk_123'
        mock_database.return_value = mock_db
        
        # Initialize ingester
        ingester = RealmAwareMarkdownIngester(self.mock_db_config, session_id='test_session')
        
        # Ingest file
        result = ingester.ingest_file(str(test_file))
        
        # Verify successful ingestion
        self.assertTrue(result['success'])
        self.assertEqual(result['file_path'], str(test_file))
        self.assertGreater(result['chunks_processed'], 0)
        self.assertEqual(result['session_id'], 'test_session')
        
        # Verify database calls were made
        self.assertTrue(mock_db.create_chunk_with_target.called)
        
        # Verify statistics were updated
        stats = ingester.get_ingestion_statistics()
        self.assertEqual(stats['files_processed'], 1)
        self.assertGreater(stats['chunks_created'], 0)


class TestBulkSemanticIngester(unittest.TestCase):
    """Test bulk semantic ingester functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_db_config = {
            'host': 'localhost',
            'port': '3306',
            'database': 'test_megamind',
            'user': 'test_user',
            'password': 'test_password'
        }
        
        # Create temporary directory with test files
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # Create test markdown files
        self._create_test_files()
        
        # Bulk ingestion configuration
        self.bulk_config = BulkIngestionConfig(
            batch_size=3,
            max_workers=2,
            embedding_batch_size=5,
            enable_parallel_processing=False,  # Disable for testing
            enable_embedding_cache=True,
            auto_commit_batches=False
        )
    
    def _create_test_files(self):
        """Create test markdown files"""
        test_files = {
            'project_overview.md': """# Project Overview
This document provides an overview of the project implementation.

## Features
- Feature A: Implementation details
- Feature B: API design
""",
            'global_standards.md': """# Global Standards
Company-wide standards and policies.

## Security Guidelines
All applications must follow security standards.

## Code Review Policy
All code changes require review.
""",
            'api_documentation.md': """# API Documentation
Complete API reference for the project.

## Authentication Endpoints
POST /auth/login - User authentication
GET /auth/verify - Token verification

## Data Endpoints
GET /data/users - Retrieve user data
POST /data/users - Create new user
""",
            'empty_file.md': """# Empty File
This file has minimal content.
""",
            'large_file.md': """# Large File
""" + "This is repeated content. " * 100  # Create larger content
        }
        
        for filename, content in test_files.items():
            file_path = Path(self.temp_dir) / filename
            file_path.write_text(content)
    
    @patch('bulk_semantic_ingester.RealmAwareMarkdownIngester')
    def test_bulk_ingester_initialization(self, mock_ingester_class):
        """Test bulk ingester initialization"""
        # Mock the base ingester
        mock_ingester = Mock()
        mock_ingester.session_id = 'bulk_test_session'
        mock_ingester_class.return_value = mock_ingester
        
        # Mock embedding service
        with patch('bulk_semantic_ingester.get_embedding_service') as mock_embedding:
            mock_embedding.return_value.is_available.return_value = True
            
            # Initialize bulk ingester
            bulk_ingester = BulkSemanticIngester(
                db_config=self.mock_db_config,
                config=self.bulk_config
            )
            
            # Verify initialization
            self.assertEqual(bulk_ingester.config.batch_size, 3)
            self.assertEqual(bulk_ingester.config.max_workers, 2)
            mock_ingester_class.assert_called_once()
    
    @patch('bulk_semantic_ingester.RealmAwareMarkdownIngester')
    @patch('bulk_semantic_ingester.get_embedding_service')
    def test_file_discovery_and_validation(self, mock_embedding_service, mock_ingester_class):
        """Test file discovery and validation"""
        # Mock services
        mock_embedding_service.return_value.is_available.return_value = True
        mock_ingester_class.return_value = Mock()
        
        # Initialize bulk ingester
        bulk_ingester = BulkSemanticIngester(
            db_config=self.mock_db_config,
            config=self.bulk_config
        )
        
        # Discover files
        files = bulk_ingester._discover_files(
            directory=Path(self.temp_dir),
            pattern='*.md',
            recursive=False,
            file_filters=None
        )
        
        # Verify files were discovered
        self.assertGreater(len(files), 0)
        self.assertTrue(all(f.suffix == '.md' for f in files))
        
        # Test file validation
        for file_path in files:
            self.assertTrue(bulk_ingester._validate_file(file_path))
    
    @patch('bulk_semantic_ingester.RealmAwareMarkdownIngester')
    @patch('bulk_semantic_ingester.get_embedding_service')
    def test_directory_ingestion(self, mock_embedding_service, mock_ingester_class):
        """Test bulk directory ingestion"""
        # Mock embedding service
        mock_embedding_service.return_value.is_available.return_value = True
        
        # Mock base ingester
        mock_ingester = Mock()
        mock_ingester.session_id = 'bulk_test_session'
        mock_ingester.ingest_file.return_value = {
            'success': True,
            'chunks_processed': 3,
            'processing_time': 0.5
        }
        mock_ingester.get_ingestion_statistics.return_value = {
            'files_processed': 1,
            'chunks_created': 3,
            'embeddings_generated': 3,
            'embedding_coverage': 100.0,
            'realm_assignments': {'PROJECT': 2, 'GLOBAL': 1},
            'processing_time': 0.5,
            'errors_count': 0,
            'errors': []
        }
        mock_ingester.commit_session_changes.return_value = {'success': True}
        mock_ingester_class.return_value = mock_ingester
        
        # Initialize bulk ingester
        bulk_ingester = BulkSemanticIngester(
            db_config=self.mock_db_config,
            config=self.bulk_config
        )
        
        # Perform bulk ingestion
        result = bulk_ingester.ingest_directory(
            directory_path=self.temp_dir,
            pattern='*.md',
            recursive=False
        )
        
        # Verify successful ingestion
        self.assertTrue(result['success'])
        self.assertEqual(result['source_path'], self.temp_dir)
        self.assertGreater(result['statistics']['files']['total'], 0)
        self.assertGreater(result['processing_time'], 0)
        
        # Verify statistics structure
        stats = result['statistics']
        self.assertIn('files', stats)
        self.assertIn('chunks', stats)
        self.assertIn('embeddings', stats)
        self.assertIn('performance', stats)
    
    @patch('bulk_semantic_ingester.RealmAwareMarkdownIngester')
    @patch('bulk_semantic_ingester.get_embedding_service')
    def test_batch_processing_performance(self, mock_embedding_service, mock_ingester_class):
        """Test batch processing performance optimization"""
        # Mock services
        mock_embedding_service.return_value.is_available.return_value = True
        
        # Mock base ingester with performance tracking
        mock_ingester = Mock()
        mock_ingester.session_id = 'perf_test_session'
        
        call_count = 0
        def mock_ingest_file(file_path):
            nonlocal call_count
            call_count += 1
            return {
                'success': True,
                'chunks_processed': 2,
                'processing_time': 0.1
            }
        
        mock_ingester.ingest_file.side_effect = mock_ingest_file
        mock_ingester_class.return_value = mock_ingester
        
        # Initialize bulk ingester with small batch size
        config = BulkIngestionConfig(
            batch_size=2,
            max_workers=1,
            enable_parallel_processing=False,
            performance_monitoring=True
        )
        
        bulk_ingester = BulkSemanticIngester(
            db_config=self.mock_db_config,
            config=config
        )
        
        # Process directory
        result = bulk_ingester.ingest_directory(self.temp_dir)
        
        # Verify all files were processed
        self.assertTrue(result['success'])
        self.assertGreater(call_count, 0)
        
        # Check performance metrics
        stats = result['statistics']
        self.assertIn('throughput', stats['performance'])
        self.assertGreater(stats['performance']['throughput'].get('files_per_second', 0), 0)


class TestIngestionIntegrationScenarios(unittest.TestCase):
    """Test comprehensive integration scenarios"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        self.mock_db_config = {
            'host': 'localhost',
            'port': '3306',
            'database': 'test_megamind',
            'user': 'test_user',
            'password': 'test_password'
        }
    
    @patch.dict(os.environ, {
        'MEGAMIND_PROJECT_REALM': 'PROJ_INTEGRATION',
        'MEGAMIND_DEFAULT_TARGET': 'PROJECT'
    })
    def test_cross_realm_ingestion_workflow(self):
        """Test ingestion workflow across different realms"""
        # Create mixed content files
        global_content = """# Global Standards
Company-wide security policies and governance frameworks.

## Security Requirements
All applications must implement authentication.

## Compliance Standards
Follow industry best practices for data protection.
"""
        
        project_content = """# Feature Implementation
Implementation guide for the new user management feature.

## API Endpoints
POST /users - Create new user
GET /users/{id} - Retrieve user details

## Database Schema
Users table with authentication fields.
"""
        
        # Write test files
        global_file = Path(self.temp_dir) / "global_standards.md"
        global_file.write_text(global_content)
        
        project_file = Path(self.temp_dir) / "feature_impl.md"
        project_file.write_text(project_content)
        
        # Mock database and services
        with patch('realm_aware_markdown_ingester.RealmAwareMegaMindDatabase') as mock_db, \
             patch('realm_aware_markdown_ingester.get_embedding_service') as mock_embedding:
            
            # Mock embedding service
            mock_embedding_svc = Mock()
            mock_embedding_svc.is_available.return_value = True
            mock_embedding_svc.generate_embedding.return_value = [0.1, 0.2, 0.3]
            mock_embedding.return_value = mock_embedding_svc
            
            # Mock database
            mock_database = Mock()
            mock_database.create_chunk_with_target.return_value = 'test_chunk_id'
            mock_db.return_value = mock_database
            
            # Initialize ingester
            ingester = RealmAwareMarkdownIngester(
                db_config=self.mock_db_config,
                session_id='cross_realm_test'
            )
            
            # Ingest global content
            global_result = ingester.ingest_file(str(global_file))
            
            # Ingest project content
            project_result = ingester.ingest_file(str(project_file))
            
            # Verify both files were processed successfully
            self.assertTrue(global_result['success'])
            self.assertTrue(project_result['success'])
            
            # Verify database was called for both files
            self.assertTrue(mock_database.create_chunk_with_target.called)
            
            # Check that realm assignment logic was applied
            call_args_list = mock_database.create_chunk_with_target.call_args_list
            self.assertGreater(len(call_args_list), 0)
    
    def test_error_handling_and_recovery(self):
        """Test error handling during ingestion"""
        # Create test file with problematic content
        problem_file = Path(self.temp_dir) / "problem.md"
        problem_file.write_text("# Test\nContent that might cause issues")
        
        # Mock database to simulate errors
        with patch('realm_aware_markdown_ingester.RealmAwareMegaMindDatabase') as mock_db, \
             patch('realm_aware_markdown_ingester.get_embedding_service') as mock_embedding:
            
            # Mock services
            mock_embedding.return_value.is_available.return_value = False
            
            # Mock database to raise exception
            mock_database = Mock()
            mock_database.create_chunk_with_target.side_effect = Exception("Database error")
            mock_db.return_value = mock_database
            
            # Initialize ingester
            ingester = RealmAwareMarkdownIngester(self.mock_db_config)
            
            # Attempt ingestion (should handle errors gracefully)
            result = ingester.ingest_file(str(problem_file))
            
            # Verify error was handled
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            
            # Check statistics include error tracking
            stats = ingester.get_ingestion_statistics()
            self.assertGreater(stats['errors_count'], 0)
    
    def test_session_based_change_management(self):
        """Test session-based change management workflow"""
        # Create test file
        test_file = Path(self.temp_dir) / "session_test.md"
        test_file.write_text("""# Session Test
This content will be tracked in a session.

## Changes
These changes will be buffered for review.
""")
        
        # Mock database with session support
        with patch('realm_aware_markdown_ingester.RealmAwareMegaMindDatabase') as mock_db, \
             patch('realm_aware_markdown_ingester.get_embedding_service') as mock_embedding:
            
            # Mock services
            mock_embedding.return_value.is_available.return_value = True
            mock_embedding.return_value.generate_embedding.return_value = [0.1, 0.2]
            
            # Mock database with session methods
            mock_database = Mock()
            mock_database.create_chunk_with_target.return_value = 'session_chunk_id'
            mock_database.get_pending_changes.return_value = [
                {'change_id': 'change_1', 'change_type': 'create'},
                {'change_id': 'change_2', 'change_type': 'create'}
            ]
            mock_database.commit_session_changes.return_value = {
                'success': True,
                'chunks_modified': 0,
                'chunks_created': 2,
                'relationships_added': 0
            }
            mock_db.return_value = mock_database
            
            # Initialize ingester with specific session ID
            session_id = 'test_session_123'
            ingester = RealmAwareMarkdownIngester(
                self.mock_db_config,
                session_id=session_id
            )
            
            # Ingest file
            result = ingester.ingest_file(str(test_file))
            self.assertTrue(result['success'])
            self.assertEqual(result['session_id'], session_id)
            
            # Check session changes
            pending_changes = ingester.get_session_changes()
            self.assertEqual(len(pending_changes), 2)
            
            # Commit changes
            commit_result = ingester.commit_session_changes()
            self.assertTrue(commit_result['success'])
            self.assertEqual(commit_result['chunks_created'], 2)


def run_phase3_tests():
    """Run all Phase 3 ingestion integration tests"""
    test_classes = [
        TestRealmAwareMarkdownIngester,
        TestBulkSemanticIngester,
        TestIngestionIntegrationScenarios
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
    
    print(f"\n=== Phase 3 Ingestion Integration Test Summary ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return failed_tests == 0


if __name__ == '__main__':
    success = run_phase3_tests()
    sys.exit(0 if success else 1)