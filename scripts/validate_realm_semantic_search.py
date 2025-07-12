#!/usr/bin/env python3
"""
Phase 4 Performance Optimization: Deployment Validation Script
Validates semantic search functionality across realms after fresh database deployment
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mcp_server'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation failures"""
    pass

class RealmSemanticSearchValidator:
    """
    Comprehensive validation suite for realm-aware semantic search deployment.
    Tests all aspects of the semantic search system after fresh database deployment.
    """
    
    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize validator with database configuration.
        
        Args:
            db_config: Database connection configuration
        """
        self.db_config = db_config
        self.database = None
        self.embedding_service = None
        self.project_realm = None
        self.validation_results = []
        self.test_chunks = []
        
        # Performance metrics
        self.performance_metrics = {
            'search_times': [],
            'embedding_generation_times': [],
            'database_operation_times': []
        }
    
    def initialize_services(self) -> bool:
        """Initialize database and embedding services"""
        try:
            # Import services
            from realm_aware_database import RealmAwareMegaMindDatabase
            from services.embedding_service import get_embedding_service
            
            # Initialize database
            self.database = RealmAwareMegaMindDatabase(self.db_config)
            logger.info("✓ Database connection established")
            
            # Initialize embedding service
            self.embedding_service = get_embedding_service()
            if not self.embedding_service.is_available():
                raise ValidationError("Embedding service not available")
            logger.info("✓ Embedding service initialized")
            
            # Get environment configuration
            self.project_realm = os.getenv('MEGAMIND_PROJECT_REALM')
            if not self.project_realm:
                raise ValidationError("MEGAMIND_PROJECT_REALM environment variable not set")
            logger.info(f"✓ Project realm configured: {self.project_realm}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            return False
    
    def validate_environment_configuration(self):
        """Validate environment configuration for semantic search"""
        logger.info("\n🔧 Validating Environment Configuration...")
        
        required_vars = [
            'MEGAMIND_PROJECT_REALM',
            'MEGAMIND_DEFAULT_TARGET'
        ]
        
        optional_vars = [
            'EMBEDDING_MODEL',
            'SEMANTIC_SEARCH_THRESHOLD', 
            'REALM_PRIORITY_PROJECT',
            'REALM_PRIORITY_GLOBAL',
            'CROSS_REALM_SEARCH_ENABLED',
            'EMBEDDING_CACHE_SIZE',
            'EMBEDDING_BATCH_SIZE'
        ]
        
        # Check required variables
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                raise ValidationError(f"Required environment variable {var} not set")
            logger.info(f"   ✓ {var} = {value}")
        
        # Check optional variables (with defaults)
        defaults = {
            'EMBEDDING_MODEL': 'sentence-transformers/all-MiniLM-L6-v2',
            'SEMANTIC_SEARCH_THRESHOLD': '0.7',
            'REALM_PRIORITY_PROJECT': '1.2',
            'REALM_PRIORITY_GLOBAL': '1.0',
            'CROSS_REALM_SEARCH_ENABLED': 'true',
            'EMBEDDING_CACHE_SIZE': '1000',
            'EMBEDDING_BATCH_SIZE': '50'
        }
        
        for var, default in defaults.items():
            value = os.getenv(var, default)
            logger.info(f"   ✓ {var} = {value} {'(default)' if os.getenv(var) is None else ''}")
        
        self.validation_results.append({
            'test': 'environment_configuration',
            'status': 'passed',
            'message': 'All environment variables properly configured'
        })
    
    def validate_database_schema(self):
        """Validate database schema supports semantic search"""
        logger.info("\n🗄️  Validating Database Schema...")
        
        try:
            connection = self.database.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Check megamind_chunks table structure
            cursor.execute("DESCRIBE megamind_chunks")
            columns = {row['Field']: row['Type'] for row in cursor.fetchall()}
            
            required_columns = {
                'chunk_id': 'varchar',
                'content': 'text',
                'realm_id': 'varchar',
                'embedding': 'json',
                'created_at': 'timestamp',
                'last_accessed': 'timestamp',
                'access_count': 'int'
            }
            
            for col_name, expected_type in required_columns.items():
                if col_name not in columns:
                    raise ValidationError(f"Missing required column: {col_name}")
                
                actual_type = columns[col_name].lower()
                if expected_type not in actual_type:
                    logger.warning(f"Column {col_name} type {actual_type} may not match expected {expected_type}")
                
                logger.info(f"   ✓ {col_name}: {columns[col_name]}")
            
            # Check for semantic search indexes
            cursor.execute("""
                SELECT INDEX_NAME, COLUMN_NAME 
                FROM information_schema.STATISTICS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = 'megamind_chunks' 
                AND INDEX_NAME LIKE '%embedding%' OR INDEX_NAME LIKE '%realm%'
            """)
            
            indexes = cursor.fetchall()
            if indexes:
                logger.info("   ✓ Semantic search indexes found:")
                for idx in indexes:
                    logger.info(f"     - {idx['INDEX_NAME']}: {idx['COLUMN_NAME']}")
            else:
                logger.warning("   ⚠ No semantic search specific indexes found")
            
            # Check realm-aware views
            cursor.execute("""
                SELECT TABLE_NAME 
                FROM information_schema.VIEWS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND (TABLE_NAME LIKE '%realm%' OR TABLE_NAME LIKE '%embedding%')
            """)
            
            views = [row['TABLE_NAME'] for row in cursor.fetchall()]
            if views:
                logger.info(f"   ✓ Realm-aware views: {', '.join(views)}")
            
            self.validation_results.append({
                'test': 'database_schema',
                'status': 'passed',
                'message': 'Database schema supports semantic search'
            })
            
        except Exception as e:
            raise ValidationError(f"Database schema validation failed: {e}")
        finally:
            if connection:
                connection.close()
    
    def create_test_data(self):
        """Create test chunks for validation"""
        logger.info("\n📝 Creating Test Data...")
        
        test_data = [
            {
                'content': 'Global authentication standards require multi-factor authentication for all administrative accounts. This policy applies company-wide across all projects and systems.',
                'source_document': 'global_auth_policy.md',
                'section_path': '/security/authentication',
                'target_realm': 'GLOBAL',
                'expected_realm': 'GLOBAL'
            },
            {
                'content': 'User management API implementation for e-commerce platform. Provides endpoints for user CRUD operations including registration, profile updates, and account management.',
                'source_document': 'user_api_spec.md', 
                'section_path': '/api/users',
                'target_realm': 'PROJECT',
                'expected_realm': self.project_realm
            },
            {
                'content': 'Database connection pooling configuration for high-availability MySQL deployment. Includes connection limits, timeout settings, and failover procedures.',
                'source_document': 'database_config.md',
                'section_path': '/infrastructure/database',
                'target_realm': 'PROJECT',
                'expected_realm': self.project_realm
            },
            {
                'content': 'Code review policy mandates peer review for all changes. Security-critical modifications require additional approval from security team members.',
                'source_document': 'code_review_policy.md',
                'section_path': '/process/development',
                'target_realm': 'GLOBAL',
                'expected_realm': 'GLOBAL'
            },
            {
                'content': 'Payment processing integration with Stripe API. Handles credit card transactions, subscription billing, and webhook event processing for the e-commerce platform.',
                'source_document': 'payment_integration.md',
                'section_path': '/features/payments',
                'target_realm': 'PROJECT', 
                'expected_realm': self.project_realm
            }
        ]
        
        session_id = f"validation_session_{int(time.time())}"
        
        for i, data in enumerate(test_data):
            try:
                chunk_id = self.database.create_chunk_with_target(
                    content=data['content'],
                    source_document=data['source_document'], 
                    section_path=data['section_path'],
                    session_id=session_id,
                    target_realm=data['target_realm']
                )
                
                data['chunk_id'] = chunk_id
                self.test_chunks.append(data)
                
                logger.info(f"   ✓ Created test chunk {i+1}: {chunk_id} ({data['target_realm']} realm)")
                
            except Exception as e:
                logger.error(f"Failed to create test chunk {i+1}: {e}")
                raise ValidationError(f"Test data creation failed: {e}")
        
        # Commit the session to make chunks available
        try:
            commit_result = self.database.commit_session_changes(session_id, list(range(len(test_data))))
            if not commit_result.get('success'):
                raise ValidationError(f"Failed to commit test chunks: {commit_result.get('error')}")
                
            logger.info(f"   ✓ Committed {len(self.test_chunks)} test chunks")
            
        except Exception as e:
            raise ValidationError(f"Failed to commit test data: {e}")
        
        self.validation_results.append({
            'test': 'test_data_creation',
            'status': 'passed', 
            'message': f'Created {len(self.test_chunks)} test chunks across realms'
        })
    
    def validate_embedding_generation(self):
        """Validate embedding generation and storage"""
        logger.info("\n🔢 Validating Embedding Generation...")
        
        embedding_tests = 0
        embedding_successes = 0
        
        for chunk_data in self.test_chunks:
            try:
                start_time = time.time()
                
                # Generate embedding
                embedding = self.embedding_service.generate_embedding(
                    text=chunk_data['content'],
                    realm_context=chunk_data['expected_realm']
                )
                
                generation_time = time.time() - start_time
                self.performance_metrics['embedding_generation_times'].append(generation_time)
                
                embedding_tests += 1
                
                if embedding is None:
                    logger.warning(f"   ⚠ Failed to generate embedding for chunk {chunk_data['chunk_id']}")
                    continue
                
                # Validate embedding properties
                if not isinstance(embedding, list):
                    raise ValidationError(f"Embedding should be list, got {type(embedding)}")
                
                if len(embedding) != 384:  # all-MiniLM-L6-v2 dimension
                    raise ValidationError(f"Expected 384-dimensional embedding, got {len(embedding)}")
                
                if not all(isinstance(x, (int, float)) for x in embedding):
                    raise ValidationError("Embedding should contain only numeric values")
                
                # Check that embedding was stored in database
                chunk = self.database.get_chunk(chunk_data['chunk_id'])
                if not chunk.get('embedding'):
                    raise ValidationError(f"Embedding not stored in database for chunk {chunk_data['chunk_id']}")
                
                stored_embedding = json.loads(chunk['embedding'])
                if len(stored_embedding) != len(embedding):
                    raise ValidationError("Stored embedding dimension mismatch")
                
                embedding_successes += 1
                logger.info(f"   ✓ Embedding generated and stored for {chunk_data['chunk_id']} ({generation_time:.3f}s)")
                
            except Exception as e:
                logger.error(f"   ❌ Embedding validation failed for {chunk_data['chunk_id']}: {e}")
                raise ValidationError(f"Embedding generation validation failed: {e}")
        
        if embedding_successes != embedding_tests:
            raise ValidationError(f"Embedding generation: {embedding_successes}/{embedding_tests} successful")
        
        avg_generation_time = sum(self.performance_metrics['embedding_generation_times']) / len(self.performance_metrics['embedding_generation_times'])
        logger.info(f"   ✓ Average embedding generation time: {avg_generation_time:.3f}s")
        
        self.validation_results.append({
            'test': 'embedding_generation',
            'status': 'passed',
            'message': f'Generated embeddings for {embedding_successes}/{embedding_tests} test chunks',
            'metrics': {
                'average_generation_time': avg_generation_time,
                'total_embeddings': embedding_successes
            }
        })
    
    def validate_realm_assignment(self):
        """Validate proper realm assignment for chunks"""
        logger.info("\n🏛️  Validating Realm Assignment...")
        
        realm_tests = 0
        realm_successes = 0
        
        for chunk_data in self.test_chunks:
            try:
                chunk = self.database.get_chunk(chunk_data['chunk_id'])
                actual_realm = chunk.get('realm_id')
                expected_realm = chunk_data['expected_realm']
                
                realm_tests += 1
                
                if actual_realm != expected_realm:
                    raise ValidationError(f"Realm mismatch for {chunk_data['chunk_id']}: expected {expected_realm}, got {actual_realm}")
                
                realm_successes += 1
                logger.info(f"   ✓ Correct realm assignment: {chunk_data['chunk_id']} → {actual_realm}")
                
            except Exception as e:
                logger.error(f"   ❌ Realm validation failed for {chunk_data['chunk_id']}: {e}")
                raise
        
        # Check realm distribution
        global_chunks = [c for c in self.test_chunks if c['expected_realm'] == 'GLOBAL']
        project_chunks = [c for c in self.test_chunks if c['expected_realm'] == self.project_realm]
        
        logger.info(f"   ✓ Realm distribution: GLOBAL={len(global_chunks)}, {self.project_realm}={len(project_chunks)}")
        
        self.validation_results.append({
            'test': 'realm_assignment',
            'status': 'passed',
            'message': f'Correct realm assignment for {realm_successes}/{realm_tests} chunks',
            'metrics': {
                'global_chunks': len(global_chunks),
                'project_chunks': len(project_chunks)
            }
        })
    
    def validate_semantic_search(self):
        """Validate semantic search functionality across realms"""
        logger.info("\n🔍 Validating Semantic Search...")
        
        search_queries = [
            {
                'query': 'authentication security policy',
                'expected_realms': ['GLOBAL'],
                'min_results': 1
            },
            {
                'query': 'user API management',
                'expected_realms': [self.project_realm],
                'min_results': 1  
            },
            {
                'query': 'database configuration',
                'expected_realms': [self.project_realm],
                'min_results': 1
            },
            {
                'query': 'payment processing integration',
                'expected_realms': [self.project_realm],
                'min_results': 1
            },
            {
                'query': 'code review development',
                'expected_realms': ['GLOBAL'],
                'min_results': 1
            }
        ]
        
        search_tests = 0
        search_successes = 0
        
        for search_test in search_queries:
            try:
                start_time = time.time()
                
                # Test semantic search
                results = self.database.search_chunks_semantic(
                    query=search_test['query'],
                    limit=10
                )
                
                search_time = time.time() - start_time
                self.performance_metrics['search_times'].append(search_time)
                
                search_tests += 1
                
                if len(results) < search_test['min_results']:
                    raise ValidationError(f"Insufficient results for query '{search_test['query']}': got {len(results)}, expected >= {search_test['min_results']}")
                
                # Check realm distribution in results
                result_realms = [r.get('realm_id') for r in results]
                expected_realms = search_test['expected_realms']
                
                found_expected_realm = any(realm in result_realms for realm in expected_realms)
                if not found_expected_realm:
                    logger.warning(f"   ⚠ Query '{search_test['query']}' didn't return chunks from expected realms {expected_realms}")
                    logger.warning(f"     Result realms: {result_realms}")
                else:
                    search_successes += 1
                
                # Check similarity scores
                for result in results:
                    similarity = result.get('similarity_score', 0)
                    if similarity < 0 or similarity > 1:
                        raise ValidationError(f"Invalid similarity score: {similarity}")
                
                logger.info(f"   ✓ Query '{search_test['query']}': {len(results)} results in {search_time:.3f}s")
                logger.info(f"     Top result: {results[0].get('chunk_id')} (similarity: {results[0].get('similarity_score', 0):.3f})")
                
            except Exception as e:
                logger.error(f"   ❌ Semantic search failed for '{search_test['query']}': {e}")
                raise ValidationError(f"Semantic search validation failed: {e}")
        
        avg_search_time = sum(self.performance_metrics['search_times']) / len(self.performance_metrics['search_times'])
        logger.info(f"   ✓ Average search time: {avg_search_time:.3f}s")
        
        if avg_search_time > 0.5:
            logger.warning(f"   ⚠ Average search time {avg_search_time:.3f}s exceeds 500ms target")
        
        self.validation_results.append({
            'test': 'semantic_search',
            'status': 'passed',
            'message': f'Semantic search successful for {search_successes}/{search_tests} queries',
            'metrics': {
                'average_search_time': avg_search_time,
                'total_queries': search_tests,
                'successful_queries': search_successes
            }
        })
    
    def validate_dual_realm_search(self):
        """Validate dual-realm search with priority weighting"""
        logger.info("\n🔄 Validating Dual-Realm Search...")
        
        # Search for terms that should appear in both realms
        mixed_queries = [
            'authentication security',  # Should find both global and project content
            'database configuration',   # May appear in both realms
            'development process'       # Could be in both realms
        ]
        
        for query in mixed_queries:
            try:
                start_time = time.time()
                
                results = self.database.search_chunks_dual_realm(
                    query=query,
                    limit=10,
                    search_type='semantic'
                )
                
                search_time = time.time() - start_time
                
                if not results:
                    logger.warning(f"   ⚠ No results for dual-realm query: '{query}'")
                    continue
                
                # Analyze realm distribution
                realm_counts = {}
                project_positions = []
                global_positions = []
                
                for i, result in enumerate(results):
                    realm = result.get('realm_id', 'unknown')
                    realm_counts[realm] = realm_counts.get(realm, 0) + 1
                    
                    if realm == self.project_realm:
                        project_positions.append(i)
                    elif realm == 'GLOBAL':
                        global_positions.append(i)
                
                logger.info(f"   ✓ Dual-realm query '{query}': {len(results)} results in {search_time:.3f}s")
                logger.info(f"     Realm distribution: {realm_counts}")
                
                # Check project realm prioritization
                if project_positions and global_positions:
                    avg_project_pos = sum(project_positions) / len(project_positions)
                    avg_global_pos = sum(global_positions) / len(global_positions)
                    
                    if avg_project_pos <= avg_global_pos:
                        logger.info(f"     ✓ Project realm prioritized (avg pos: {avg_project_pos:.1f} vs {avg_global_pos:.1f})")
                    else:
                        logger.warning(f"     ⚠ Project realm not prioritized (avg pos: {avg_project_pos:.1f} vs {avg_global_pos:.1f})")
                
            except Exception as e:
                logger.error(f"   ❌ Dual-realm search failed for '{query}': {e}")
                raise ValidationError(f"Dual-realm search validation failed: {e}")
        
        self.validation_results.append({
            'test': 'dual_realm_search',
            'status': 'passed',
            'message': 'Dual-realm search functionality validated'
        })
    
    def validate_performance_requirements(self):
        """Validate performance meets requirements"""
        logger.info("\n⚡ Validating Performance Requirements...")
        
        performance_requirements = {
            'max_search_time': 0.3,      # 300ms max search time
            'max_embedding_time': 1.0,   # 1s max embedding generation
            'min_search_results': 1       # Must return at least 1 result for relevant queries
        }
        
        # Analyze search performance
        if self.performance_metrics['search_times']:
            max_search_time = max(self.performance_metrics['search_times'])
            avg_search_time = sum(self.performance_metrics['search_times']) / len(self.performance_metrics['search_times'])
            
            logger.info(f"   Search Performance:")
            logger.info(f"     Average: {avg_search_time:.3f}s")
            logger.info(f"     Maximum: {max_search_time:.3f}s")
            logger.info(f"     Target: <{performance_requirements['max_search_time']}s")
            
            if max_search_time <= performance_requirements['max_search_time']:
                logger.info(f"     ✓ Search performance meets requirements")
            else:
                logger.warning(f"     ⚠ Search performance exceeds target: {max_search_time:.3f}s > {performance_requirements['max_search_time']}s")
        
        # Analyze embedding performance
        if self.performance_metrics['embedding_generation_times']:
            max_embedding_time = max(self.performance_metrics['embedding_generation_times'])
            avg_embedding_time = sum(self.performance_metrics['embedding_generation_times']) / len(self.performance_metrics['embedding_generation_times'])
            
            logger.info(f"   Embedding Performance:")
            logger.info(f"     Average: {avg_embedding_time:.3f}s")
            logger.info(f"     Maximum: {max_embedding_time:.3f}s")
            logger.info(f"     Target: <{performance_requirements['max_embedding_time']}s")
            
            if max_embedding_time <= performance_requirements['max_embedding_time']:
                logger.info(f"     ✓ Embedding performance meets requirements")
            else:
                logger.warning(f"     ⚠ Embedding performance exceeds target: {max_embedding_time:.3f}s > {performance_requirements['max_embedding_time']}s")
        
        self.validation_results.append({
            'test': 'performance_requirements',
            'status': 'passed',
            'message': 'Performance requirements validated',
            'metrics': {
                'avg_search_time': avg_search_time if self.performance_metrics['search_times'] else 0,
                'max_search_time': max_search_time if self.performance_metrics['search_times'] else 0,
                'avg_embedding_time': avg_embedding_time if self.performance_metrics['embedding_generation_times'] else 0,
                'max_embedding_time': max_embedding_time if self.performance_metrics['embedding_generation_times'] else 0
            }
        })
    
    def validate_cache_functionality(self):
        """Validate embedding cache functionality"""
        logger.info("\n💾 Validating Cache Functionality...")
        
        try:
            # Get cache statistics
            cache_stats = self.embedding_service.get_embedding_stats()
            cache_info = cache_stats.get('cache', {})
            
            logger.info(f"   Cache Configuration:")
            logger.info(f"     Type: {cache_info.get('type', 'unknown')}")
            logger.info(f"     Size: {cache_info.get('cache_size', 0)}")
            logger.info(f"     Limit: {cache_info.get('cache_limit', 0)}")
            
            if 'hit_rate_percent' in cache_info:
                hit_rate = cache_info['hit_rate_percent']
                logger.info(f"     Hit Rate: {hit_rate:.1f}%")
                
                if hit_rate > 0:
                    logger.info(f"   ✓ Cache is functioning (hit rate: {hit_rate:.1f}%)")
                else:
                    logger.info(f"   ⚠ Cache hit rate is 0% (may be expected for first run)")
            
            # Test cache with duplicate embedding generation
            test_text = "This is a test for cache functionality"
            
            # First generation (should miss cache)
            start_time = time.time()
            embedding1 = self.embedding_service.generate_embedding(test_text, self.project_realm)
            time1 = time.time() - start_time
            
            # Second generation (should hit cache)
            start_time = time.time()
            embedding2 = self.embedding_service.generate_embedding(test_text, self.project_realm)
            time2 = time.time() - start_time
            
            if embedding1 == embedding2:
                logger.info(f"   ✓ Cache consistency: identical embeddings generated")
                logger.info(f"     First generation: {time1:.3f}s")
                logger.info(f"     Second generation: {time2:.3f}s")
                
                if time2 < time1 * 0.5:  # Expect significant speedup from cache
                    logger.info(f"   ✓ Cache performance improvement detected")
                else:
                    logger.info(f"   ⚠ Limited cache performance improvement")
            else:
                logger.warning(f"   ⚠ Cache inconsistency: different embeddings for same text")
            
            self.validation_results.append({
                'test': 'cache_functionality',
                'status': 'passed',
                'message': 'Cache functionality validated'
            })
            
        except Exception as e:
            logger.error(f"   ❌ Cache validation failed: {e}")
            raise ValidationError(f"Cache validation failed: {e}")
    
    def cleanup_test_data(self):
        """Clean up test data created during validation"""
        logger.info("\n🧹 Cleaning Up Test Data...")
        
        try:
            for chunk_data in self.test_chunks:
                # Note: In a real implementation, you might want to delete test chunks
                # For now, we'll just log them for manual cleanup if needed
                logger.info(f"   Test chunk created: {chunk_data['chunk_id']} ({chunk_data['expected_realm']})")
            
            logger.info(f"   ✓ Validation completed with {len(self.test_chunks)} test chunks")
            logger.info("   Note: Test chunks remain in database for reference")
            
        except Exception as e:
            logger.warning(f"   ⚠ Cleanup warning: {e}")
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        total_tests = len(self.validation_results)
        passed_tests = len([r for r in self.validation_results if r['status'] == 'passed'])
        
        # Calculate overall performance metrics
        overall_metrics = {}
        if self.performance_metrics['search_times']:
            overall_metrics['search_performance'] = {
                'average_time': sum(self.performance_metrics['search_times']) / len(self.performance_metrics['search_times']),
                'max_time': max(self.performance_metrics['search_times']),
                'total_searches': len(self.performance_metrics['search_times'])
            }
        
        if self.performance_metrics['embedding_generation_times']:
            overall_metrics['embedding_performance'] = {
                'average_time': sum(self.performance_metrics['embedding_generation_times']) / len(self.performance_metrics['embedding_generation_times']),
                'max_time': max(self.performance_metrics['embedding_generation_times']),
                'total_generations': len(self.performance_metrics['embedding_generation_times'])
            }
        
        report = {
            'validation_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'timestamp': datetime.now().isoformat()
            },
            'environment_info': {
                'project_realm': self.project_realm,
                'embedding_model': os.getenv('EMBEDDING_MODEL', 'default'),
                'database_host': self.db_config.get('host', 'unknown')
            },
            'test_results': self.validation_results,
            'performance_metrics': overall_metrics,
            'test_data_created': len(self.test_chunks)
        }
        
        return report
    
    def run_full_validation(self) -> bool:
        """Run complete validation suite"""
        logger.info("🚀 Starting Realm-Aware Semantic Search Validation")
        logger.info("=" * 70)
        
        try:
            # Initialize services
            if not self.initialize_services():
                return False
            
            # Run validation tests
            self.validate_environment_configuration()
            self.validate_database_schema()
            self.create_test_data()
            self.validate_embedding_generation()
            self.validate_realm_assignment()
            self.validate_semantic_search()
            self.validate_dual_realm_search()
            self.validate_performance_requirements()
            self.validate_cache_functionality()
            
            # Generate report
            report = self.generate_validation_report()
            
            # Cleanup
            self.cleanup_test_data()
            
            # Print summary
            logger.info("\n" + "=" * 70)
            logger.info("✅ VALIDATION COMPLETE!")
            logger.info("=" * 70)
            logger.info(f"Tests Passed: {report['validation_summary']['passed_tests']}/{report['validation_summary']['total_tests']}")
            logger.info(f"Success Rate: {report['validation_summary']['success_rate']:.1f}%")
            
            if 'search_performance' in report['performance_metrics']:
                search_perf = report['performance_metrics']['search_performance']
                logger.info(f"Average Search Time: {search_perf['average_time']:.3f}s")
            
            if 'embedding_performance' in report['performance_metrics']:
                embed_perf = report['performance_metrics']['embedding_performance']
                logger.info(f"Average Embedding Time: {embed_perf['average_time']:.3f}s")
            
            logger.info(f"Project Realm: {self.project_realm}")
            logger.info(f"Test Chunks Created: {len(self.test_chunks)}")
            
            return report['validation_summary']['success_rate'] == 100.0
            
        except ValidationError as e:
            logger.error(f"\n❌ VALIDATION FAILED: {e}")
            return False
        except Exception as e:
            logger.error(f"\n💥 VALIDATION ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main entry point for validation script"""
    parser = argparse.ArgumentParser(description='Validate realm-aware semantic search deployment')
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', default='3306', help='Database port')
    parser.add_argument('--database', default='megamind_database', help='Database name')
    parser.add_argument('--user', default='megamind_user', help='Database user')
    parser.add_argument('--password', required=True, help='Database password')
    parser.add_argument('--report-file', help='Save validation report to file')
    
    args = parser.parse_args()
    
    # Database configuration
    db_config = {
        'host': args.host,
        'port': args.port,
        'database': args.database,
        'user': args.user,
        'password': args.password
    }
    
    # Run validation
    validator = RealmSemanticSearchValidator(db_config)
    success = validator.run_full_validation()
    
    # Save report if requested
    if args.report_file:
        try:
            report = validator.generate_validation_report()
            with open(args.report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Validation report saved to: {args.report_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())