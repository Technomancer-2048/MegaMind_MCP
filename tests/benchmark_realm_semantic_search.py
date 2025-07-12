#!/usr/bin/env python3
"""
Phase 4 Performance Optimization: Comprehensive Benchmarking Suite
Tests semantic search performance across different scenarios and configurations
"""

import os
import sys
import time
import json
import statistics
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
import argparse
import random

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mcp_server'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Individual benchmark test result"""
    test_name: str
    operation: str
    duration: float
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class BenchmarkSuite:
    """Collection of benchmark results for analysis"""
    suite_name: str
    results: List[BenchmarkResult]
    start_time: datetime
    end_time: datetime
    
    @property
    def total_duration(self) -> float:
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        successful = len([r for r in self.results if r.success])
        return (successful / len(self.results)) * 100

class RealmSemanticSearchBenchmark:
    """
    Comprehensive benchmark suite for realm-aware semantic search performance.
    Tests various scenarios including load, concurrency, and optimization effectiveness.
    """
    
    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize benchmark suite.
        
        Args:
            db_config: Database configuration
        """
        self.db_config = db_config
        self.database = None
        self.embedding_service = None
        self.project_realm = None
        
        # Benchmark results
        self.benchmark_suites: List[BenchmarkSuite] = []
        self.test_data_cache = {}
        
        # Performance baselines
        self.baselines = {
            'single_search_time': 0.3,      # 300ms baseline
            'batch_embedding_time': 2.0,     # 2s for 10 embeddings
            'concurrent_search_time': 1.0,   # 1s for concurrent searches
            'cache_hit_improvement': 0.5     # 50% improvement from cache hits
        }
    
    def initialize_services(self) -> bool:
        """Initialize database and embedding services"""
        try:
            from realm_aware_database import RealmAwareMegaMindDatabase
            from services.embedding_service import get_embedding_service
            
            self.database = RealmAwareMegaMindDatabase(self.db_config)
            self.embedding_service = get_embedding_service()
            self.project_realm = os.getenv('MEGAMIND_PROJECT_REALM', 'PROJ_TEST')
            
            if not self.embedding_service.is_available():
                raise RuntimeError("Embedding service not available")
            
            logger.info("✓ Benchmark services initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            return False
    
    def create_benchmark_data(self, num_chunks: int = 100) -> List[Dict[str, Any]]:
        """Create test data for benchmarking"""
        logger.info(f"Creating {num_chunks} benchmark chunks...")
        
        # Diverse content patterns for realistic testing
        content_templates = [
            "Authentication security policy for {domain} requires multi-factor authentication and regular password rotation procedures.",
            "Database configuration for {system} includes connection pooling with {count} connections and failover mechanisms.",
            "API endpoint documentation for {service} covers REST operations including GET, POST, PUT, and DELETE methods.",
            "User interface components for {feature} implement responsive design patterns and accessibility standards.",
            "Performance optimization strategies for {component} focus on caching, indexing, and query optimization techniques.",
            "Security vulnerability assessment for {module} identifies potential risks and mitigation strategies.",
            "Development workflow documentation covers code review processes, testing requirements, and deployment procedures.",
            "Integration testing framework for {service} validates API contracts and data consistency across systems.",
            "Monitoring and alerting configuration for {infrastructure} includes metrics collection and notification rules.",
            "Backup and recovery procedures for {data_source} ensure business continuity and data integrity."
        ]
        
        domains = ["enterprise", "platform", "application", "system", "infrastructure"]
        systems = ["MySQL", "PostgreSQL", "Redis", "Elasticsearch", "MongoDB"]
        services = ["UserService", "AuthService", "PaymentService", "NotificationService", "AnalyticsService"]
        features = ["dashboard", "profile", "settings", "reporting", "messaging"]
        components = ["frontend", "backend", "database", "cache", "queue"]
        modules = ["authentication", "authorization", "payment", "user-management", "reporting"]
        
        session_id = f"benchmark_session_{int(time.time())}"
        benchmark_chunks = []
        
        for i in range(num_chunks):
            # Select random template and fill in variables
            template = random.choice(content_templates)
            content = template.format(
                domain=random.choice(domains),
                system=random.choice(systems),
                service=random.choice(services),
                feature=random.choice(features),
                component=random.choice(components),
                module=random.choice(modules),
                count=random.randint(10, 100)
            )
            
            # Assign to realms with realistic distribution
            if i % 3 == 0:  # 1/3 to Global realm
                target_realm = 'GLOBAL'
            else:  # 2/3 to Project realm
                target_realm = 'PROJECT'
            
            chunk_data = {
                'content': content,
                'source_document': f'benchmark_doc_{i // 10}.md',
                'section_path': f'/section_{i % 10}',
                'target_realm': target_realm,
                'chunk_index': i
            }
            
            # Create chunk in database
            try:
                chunk_id = self.database.create_chunk_with_target(
                    content=content,
                    source_document=chunk_data['source_document'],
                    section_path=chunk_data['section_path'],
                    session_id=session_id,
                    target_realm=target_realm
                )
                
                chunk_data['chunk_id'] = chunk_id
                benchmark_chunks.append(chunk_data)
                
                if (i + 1) % 20 == 0:
                    logger.info(f"  Created {i + 1}/{num_chunks} chunks")
                    
            except Exception as e:
                logger.error(f"Failed to create benchmark chunk {i}: {e}")
        
        # Commit session to make chunks available
        try:
            commit_result = self.database.commit_session_changes(session_id, list(range(len(benchmark_chunks))))
            if not commit_result.get('success'):
                raise RuntimeError(f"Failed to commit benchmark chunks: {commit_result.get('error')}")
        except Exception as e:
            logger.error(f"Failed to commit benchmark data: {e}")
            raise
        
        logger.info(f"✓ Created {len(benchmark_chunks)} benchmark chunks")
        self.test_data_cache['benchmark_chunks'] = benchmark_chunks
        return benchmark_chunks
    
    def benchmark_single_search_performance(self) -> BenchmarkSuite:
        """Benchmark individual search operation performance"""
        logger.info("\n🔍 Benchmarking Single Search Performance...")
        
        start_time = datetime.now()
        results = []
        
        # Test queries with different characteristics
        test_queries = [
            "authentication security policy",
            "database configuration optimization",
            "API endpoint documentation", 
            "user interface components",
            "performance monitoring strategies",
            "security vulnerability assessment",
            "development workflow documentation",
            "integration testing framework",
            "backup recovery procedures",
            "monitoring alerting configuration"
        ]
        
        search_types = ['semantic', 'keyword', 'hybrid']
        
        for query in test_queries:
            for search_type in search_types:
                try:
                    # Warm up
                    self.database.search_chunks_dual_realm(query, limit=5, search_type=search_type)
                    
                    # Actual benchmark
                    operation_start = time.time()
                    results_data = self.database.search_chunks_dual_realm(
                        query=query, 
                        limit=10, 
                        search_type=search_type
                    )
                    duration = time.time() - operation_start
                    
                    results.append(BenchmarkResult(
                        test_name=f"single_search_{search_type}",
                        operation=f"search: '{query[:30]}...' ({search_type})",
                        duration=duration,
                        success=True,
                        metadata={
                            'query': query,
                            'search_type': search_type,
                            'result_count': len(results_data),
                            'avg_similarity': statistics.mean([r.get('similarity_score', 0) for r in results_data]) if results_data else 0
                        }
                    ))
                    
                except Exception as e:
                    results.append(BenchmarkResult(
                        test_name=f"single_search_{search_type}",
                        operation=f"search: '{query[:30]}...' ({search_type})",
                        duration=0,
                        success=False,
                        error_message=str(e)
                    ))
        
        end_time = datetime.now()
        suite = BenchmarkSuite("single_search_performance", results, start_time, end_time)
        
        # Analyze results
        successful_results = [r for r in results if r.success]
        if successful_results:
            avg_time = statistics.mean([r.duration for r in successful_results])
            max_time = max([r.duration for r in successful_results])
            min_time = min([r.duration for r in successful_results])
            
            logger.info(f"  Average search time: {avg_time:.3f}s")
            logger.info(f"  Min/Max search time: {min_time:.3f}s / {max_time:.3f}s")
            logger.info(f"  Success rate: {suite.success_rate:.1f}%")
            
            # Compare to baseline
            baseline_exceeded = len([r for r in successful_results if r.duration > self.baselines['single_search_time']])
            if baseline_exceeded > 0:
                logger.warning(f"  ⚠ {baseline_exceeded} searches exceeded {self.baselines['single_search_time']}s baseline")
            else:
                logger.info(f"  ✓ All searches within {self.baselines['single_search_time']}s baseline")
        
        self.benchmark_suites.append(suite)
        return suite
    
    def benchmark_batch_embedding_performance(self) -> BenchmarkSuite:
        """Benchmark batch embedding generation performance"""
        logger.info("\n🔢 Benchmarking Batch Embedding Performance...")
        
        start_time = datetime.now()
        results = []
        
        # Test different batch sizes
        batch_sizes = [1, 5, 10, 20, 50]
        benchmark_chunks = self.test_data_cache.get('benchmark_chunks', [])
        
        if not benchmark_chunks:
            logger.warning("No benchmark chunks available, skipping batch embedding test")
            return BenchmarkSuite("batch_embedding_performance", [], start_time, datetime.now())
        
        for batch_size in batch_sizes:
            # Select random chunks for testing
            test_chunks = random.sample(benchmark_chunks, min(batch_size, len(benchmark_chunks)))
            texts = [chunk['content'] for chunk in test_chunks]
            realm_contexts = [self.project_realm if chunk['target_realm'] == 'PROJECT' else 'GLOBAL' for chunk in test_chunks]
            
            try:
                # Warm up
                self.embedding_service.generate_embeddings_batch(texts[:2], realm_contexts[:2])
                
                # Actual benchmark
                operation_start = time.time()
                embeddings = self.embedding_service.generate_embeddings_batch(texts, realm_contexts)
                duration = time.time() - operation_start
                
                successful_embeddings = len([e for e in embeddings if e is not None])
                
                results.append(BenchmarkResult(
                    test_name=f"batch_embedding_{batch_size}",
                    operation=f"generate {batch_size} embeddings",
                    duration=duration,
                    success=True,
                    metadata={
                        'batch_size': batch_size,
                        'successful_embeddings': successful_embeddings,
                        'embeddings_per_second': batch_size / duration if duration > 0 else 0,
                        'avg_time_per_embedding': duration / batch_size if batch_size > 0 else 0
                    }
                ))
                
            except Exception as e:
                results.append(BenchmarkResult(
                    test_name=f"batch_embedding_{batch_size}",
                    operation=f"generate {batch_size} embeddings",
                    duration=0,
                    success=False,
                    error_message=str(e)
                ))
        
        end_time = datetime.now()
        suite = BenchmarkSuite("batch_embedding_performance", results, start_time, end_time)
        
        # Analyze results
        successful_results = [r for r in results if r.success]
        if successful_results:
            for result in successful_results:
                batch_size = result.metadata['batch_size']
                eps = result.metadata['embeddings_per_second']
                logger.info(f"  Batch size {batch_size}: {result.duration:.3f}s ({eps:.1f} embeddings/sec)")
            
            # Check batch efficiency
            single_result = [r for r in successful_results if r.metadata['batch_size'] == 1]
            large_batch_result = [r for r in successful_results if r.metadata['batch_size'] >= 20]
            
            if single_result and large_batch_result:
                single_eps = single_result[0].metadata['embeddings_per_second']
                batch_eps = large_batch_result[0].metadata['embeddings_per_second']
                improvement = (batch_eps / single_eps) if single_eps > 0 else 0
                
                logger.info(f"  Batch efficiency improvement: {improvement:.1f}x")
                if improvement >= 2.0:
                    logger.info("  ✓ Good batch processing efficiency")
                else:
                    logger.warning("  ⚠ Limited batch processing efficiency")
        
        self.benchmark_suites.append(suite)
        return suite
    
    def benchmark_concurrent_search_performance(self) -> BenchmarkSuite:
        """Benchmark concurrent search performance"""
        logger.info("\n🔄 Benchmarking Concurrent Search Performance...")
        
        start_time = datetime.now()
        results = []
        
        # Test different concurrency levels
        concurrency_levels = [1, 2, 5, 10]
        test_queries = [
            "authentication security",
            "database configuration", 
            "API documentation",
            "user interface",
            "performance optimization"
        ]
        
        for concurrency in concurrency_levels:
            # Prepare concurrent operations
            concurrent_queries = []
            for i in range(concurrency):
                query = test_queries[i % len(test_queries)]
                search_type = ['semantic', 'keyword', 'hybrid'][i % 3]
                concurrent_queries.append((query, search_type))
            
            try:
                # Warm up
                for query, search_type in concurrent_queries[:1]:
                    self.database.search_chunks_dual_realm(query, limit=5, search_type=search_type)
                
                # Actual concurrent benchmark
                operation_start = time.time()
                
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = []
                    for query, search_type in concurrent_queries:
                        future = executor.submit(
                            self.database.search_chunks_dual_realm,
                            query, 10, search_type
                        )
                        futures.append(future)
                    
                    # Wait for all to complete
                    concurrent_results = []
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            concurrent_results.append(result)
                        except Exception as e:
                            logger.error(f"Concurrent search failed: {e}")
                
                duration = time.time() - operation_start
                
                results.append(BenchmarkResult(
                    test_name=f"concurrent_search_{concurrency}",
                    operation=f"{concurrency} concurrent searches",
                    duration=duration,
                    success=True,
                    metadata={
                        'concurrency_level': concurrency,
                        'successful_searches': len(concurrent_results),
                        'searches_per_second': concurrency / duration if duration > 0 else 0,
                        'avg_time_per_search': duration / concurrency if concurrency > 0 else 0
                    }
                ))
                
            except Exception as e:
                results.append(BenchmarkResult(
                    test_name=f"concurrent_search_{concurrency}",
                    operation=f"{concurrency} concurrent searches",
                    duration=0,
                    success=False,
                    error_message=str(e)
                ))
        
        end_time = datetime.now()
        suite = BenchmarkSuite("concurrent_search_performance", results, start_time, end_time)
        
        # Analyze results
        successful_results = [r for r in results if r.success]
        if successful_results:
            for result in successful_results:
                concurrency = result.metadata['concurrency_level']
                sps = result.metadata['searches_per_second']
                logger.info(f"  Concurrency {concurrency}: {result.duration:.3f}s ({sps:.1f} searches/sec)")
            
            # Check scalability
            single_result = [r for r in successful_results if r.metadata['concurrency_level'] == 1]
            high_concurrency = [r for r in successful_results if r.metadata['concurrency_level'] >= 5]
            
            if single_result and high_concurrency:
                single_sps = single_result[0].metadata['searches_per_second']
                concurrent_sps = max([r.metadata['searches_per_second'] for r in high_concurrency])
                scalability = concurrent_sps / single_sps if single_sps > 0 else 0
                
                logger.info(f"  Scalability factor: {scalability:.1f}x")
                if scalability >= 2.0:
                    logger.info("  ✓ Good concurrent scalability")
                else:
                    logger.warning("  ⚠ Limited concurrent scalability")
        
        self.benchmark_suites.append(suite)
        return suite
    
    def benchmark_cache_performance(self) -> BenchmarkSuite:
        """Benchmark embedding cache performance"""
        logger.info("\n💾 Benchmarking Cache Performance...")
        
        start_time = datetime.now()
        results = []
        
        # Test cache effectiveness
        test_texts = [
            "Database connection pooling configuration for high availability",
            "Authentication security policies for enterprise applications", 
            "API documentation standards for REST endpoints",
            "User interface component library usage guidelines",
            "Performance monitoring and alerting best practices"
        ]
        
        # Clear cache to start fresh
        self.embedding_service.clear_cache()
        
        for i, text in enumerate(test_texts):
            try:
                # First access (cache miss)
                operation_start = time.time()
                embedding1 = self.embedding_service.generate_embedding(text, self.project_realm)
                miss_duration = time.time() - operation_start
                
                # Second access (cache hit)
                operation_start = time.time()
                embedding2 = self.embedding_service.generate_embedding(text, self.project_realm)
                hit_duration = time.time() - operation_start
                
                # Verify consistency
                if embedding1 == embedding2:
                    improvement = (miss_duration - hit_duration) / miss_duration if miss_duration > 0 else 0
                    
                    results.append(BenchmarkResult(
                        test_name=f"cache_test_{i}",
                        operation=f"cache test for text {i+1}",
                        duration=hit_duration,
                        success=True,
                        metadata={
                            'miss_duration': miss_duration,
                            'hit_duration': hit_duration,
                            'improvement_factor': improvement,
                            'cache_effective': hit_duration < miss_duration * 0.8
                        }
                    ))
                else:
                    results.append(BenchmarkResult(
                        test_name=f"cache_test_{i}",
                        operation=f"cache test for text {i+1}",
                        duration=hit_duration,
                        success=False,
                        error_message="Cache inconsistency: different embeddings"
                    ))
                    
            except Exception as e:
                results.append(BenchmarkResult(
                    test_name=f"cache_test_{i}",
                    operation=f"cache test for text {i+1}",
                    duration=0,
                    success=False,
                    error_message=str(e)
                ))
        
        end_time = datetime.now()
        suite = BenchmarkSuite("cache_performance", results, start_time, end_time)
        
        # Analyze cache effectiveness
        successful_results = [r for r in results if r.success]
        if successful_results:
            improvements = [r.metadata['improvement_factor'] for r in successful_results]
            avg_improvement = statistics.mean(improvements)
            effective_cache = len([r for r in successful_results if r.metadata['cache_effective']])
            
            logger.info(f"  Average cache improvement: {avg_improvement:.1%}")
            logger.info(f"  Effective cache hits: {effective_cache}/{len(successful_results)}")
            
            if avg_improvement >= self.baselines['cache_hit_improvement']:
                logger.info(f"  ✓ Cache performance meets {self.baselines['cache_hit_improvement']:.0%} baseline")
            else:
                logger.warning(f"  ⚠ Cache improvement below {self.baselines['cache_hit_improvement']:.0%} baseline")
        
        self.benchmark_suites.append(suite)
        return suite
    
    def benchmark_realm_priority_performance(self) -> BenchmarkSuite:
        """Benchmark realm priority weighting performance"""
        logger.info("\n🏛️  Benchmarking Realm Priority Performance...")
        
        start_time = datetime.now()
        results = []
        
        # Test queries that should appear in both realms
        mixed_queries = [
            "authentication security configuration",
            "database performance optimization",
            "API endpoint documentation standards",
            "user interface design patterns", 
            "monitoring and alerting procedures"
        ]
        
        for query in mixed_queries:
            try:
                operation_start = time.time()
                
                # Perform dual-realm search
                search_results = self.database.search_chunks_dual_realm(
                    query=query,
                    limit=20,
                    search_type='semantic'
                )
                
                duration = time.time() - operation_start
                
                # Analyze realm distribution and ordering
                realm_positions = {'GLOBAL': [], self.project_realm: []}
                
                for i, result in enumerate(search_results):
                    realm = result.get('realm_id', 'unknown')
                    if realm in realm_positions:
                        realm_positions[realm].append(i)
                
                # Calculate priority effectiveness
                project_avg_pos = statistics.mean(realm_positions[self.project_realm]) if realm_positions[self.project_realm] else float('inf')
                global_avg_pos = statistics.mean(realm_positions['GLOBAL']) if realm_positions['GLOBAL'] else float('inf')
                
                priority_effective = project_avg_pos < global_avg_pos if project_avg_pos != float('inf') and global_avg_pos != float('inf') else None
                
                results.append(BenchmarkResult(
                    test_name=f"realm_priority",
                    operation=f"dual-realm search: '{query[:30]}...'",
                    duration=duration,
                    success=True,
                    metadata={
                        'query': query,
                        'total_results': len(search_results),
                        'project_realm_results': len(realm_positions[self.project_realm]),
                        'global_realm_results': len(realm_positions['GLOBAL']),
                        'project_avg_position': project_avg_pos if project_avg_pos != float('inf') else None,
                        'global_avg_position': global_avg_pos if global_avg_pos != float('inf') else None,
                        'priority_effective': priority_effective
                    }
                ))
                
            except Exception as e:
                results.append(BenchmarkResult(
                    test_name=f"realm_priority",
                    operation=f"dual-realm search: '{query[:30]}...'",
                    duration=0,
                    success=False,
                    error_message=str(e)
                ))
        
        end_time = datetime.now()
        suite = BenchmarkSuite("realm_priority_performance", results, start_time, end_time)
        
        # Analyze realm prioritization effectiveness
        successful_results = [r for r in results if r.success and r.metadata.get('priority_effective') is not None]
        if successful_results:
            effective_prioritization = len([r for r in successful_results if r.metadata['priority_effective']])
            
            logger.info(f"  Effective realm prioritization: {effective_prioritization}/{len(successful_results)}")
            
            if effective_prioritization / len(successful_results) >= 0.7:
                logger.info("  ✓ Good realm prioritization performance")
            else:
                logger.warning("  ⚠ Realm prioritization may need tuning")
        
        self.benchmark_suites.append(suite)
        return suite
    
    def benchmark_memory_usage(self) -> BenchmarkSuite:
        """Benchmark memory usage during operations"""
        logger.info("\n🧠 Benchmarking Memory Usage...")
        
        start_time = datetime.now()
        results = []
        
        try:
            import psutil
            process = psutil.Process(os.getpid())
            
            # Baseline memory usage
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            operations = [
                ("embedding_service_stats", lambda: self.embedding_service.get_embedding_stats()),
                ("single_search", lambda: self.database.search_chunks_dual_realm("test query", 10, 'semantic')),
                ("batch_embedding", lambda: self.embedding_service.generate_embeddings_batch(["test text"] * 10)),
                ("concurrent_searches", lambda: self._run_concurrent_searches(5))
            ]
            
            for op_name, operation in operations:
                try:
                    operation_start = time.time()
                    pre_memory = process.memory_info().rss / 1024 / 1024
                    
                    # Run operation
                    operation()
                    
                    post_memory = process.memory_info().rss / 1024 / 1024
                    duration = time.time() - operation_start
                    
                    memory_delta = post_memory - pre_memory
                    
                    results.append(BenchmarkResult(
                        test_name=f"memory_{op_name}",
                        operation=f"memory usage for {op_name}",
                        duration=duration,
                        success=True,
                        metadata={
                            'baseline_memory_mb': baseline_memory,
                            'pre_operation_mb': pre_memory,
                            'post_operation_mb': post_memory,
                            'memory_delta_mb': memory_delta,
                            'memory_efficiency': memory_delta < 50  # Less than 50MB increase
                        }
                    ))
                    
                except Exception as e:
                    results.append(BenchmarkResult(
                        test_name=f"memory_{op_name}",
                        operation=f"memory usage for {op_name}",
                        duration=0,
                        success=False,
                        error_message=str(e)
                    ))
            
        except ImportError:
            logger.warning("psutil not available, skipping memory benchmarks")
            results.append(BenchmarkResult(
                test_name="memory_monitoring",
                operation="memory monitoring setup",
                duration=0,
                success=False,
                error_message="psutil not available"
            ))
        
        end_time = datetime.now()
        suite = BenchmarkSuite("memory_usage", results, start_time, end_time)
        
        # Analyze memory usage
        successful_results = [r for r in results if r.success]
        if successful_results:
            max_delta = max([r.metadata['memory_delta_mb'] for r in successful_results])
            avg_delta = statistics.mean([r.metadata['memory_delta_mb'] for r in successful_results])
            
            logger.info(f"  Maximum memory increase: {max_delta:.1f} MB")
            logger.info(f"  Average memory increase: {avg_delta:.1f} MB")
            
            efficient_operations = len([r for r in successful_results if r.metadata['memory_efficiency']])
            logger.info(f"  Memory efficient operations: {efficient_operations}/{len(successful_results)}")
        
        self.benchmark_suites.append(suite)
        return suite
    
    def _run_concurrent_searches(self, count: int) -> List[Any]:
        """Helper method for concurrent search testing"""
        with ThreadPoolExecutor(max_workers=count) as executor:
            futures = []
            for i in range(count):
                future = executor.submit(
                    self.database.search_chunks_dual_realm,
                    f"test query {i}", 5, 'semantic'
                )
                futures.append(future)
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception:
                    pass
            
            return results
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        all_results = []
        for suite in self.benchmark_suites:
            all_results.extend(suite.results)
        
        total_tests = len(all_results)
        successful_tests = len([r for r in all_results if r.success])
        
        # Performance summaries by test type
        performance_summary = {}
        for suite in self.benchmark_suites:
            suite_results = [r for r in suite.results if r.success]
            if suite_results:
                performance_summary[suite.suite_name] = {
                    'total_tests': len(suite.results),
                    'successful_tests': len(suite_results),
                    'success_rate': suite.success_rate,
                    'average_duration': statistics.mean([r.duration for r in suite_results]),
                    'max_duration': max([r.duration for r in suite_results]),
                    'min_duration': min([r.duration for r in suite_results])
                }
        
        # Overall baseline comparison
        baseline_compliance = {}
        
        # Single search baseline
        single_search_results = [r for r in all_results if 'single_search' in r.test_name and r.success]
        if single_search_results:
            avg_search_time = statistics.mean([r.duration for r in single_search_results])
            baseline_compliance['single_search'] = {
                'average_time': avg_search_time,
                'baseline': self.baselines['single_search_time'],
                'meets_baseline': avg_search_time <= self.baselines['single_search_time']
            }
        
        # Cache performance baseline
        cache_results = [r for r in all_results if 'cache' in r.test_name and r.success]
        if cache_results:
            avg_improvement = statistics.mean([r.metadata.get('improvement_factor', 0) for r in cache_results])
            baseline_compliance['cache_performance'] = {
                'average_improvement': avg_improvement,
                'baseline': self.baselines['cache_hit_improvement'],
                'meets_baseline': avg_improvement >= self.baselines['cache_hit_improvement']
            }
        
        report = {
            'benchmark_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'overall_success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                'total_benchmark_time': sum([suite.total_duration for suite in self.benchmark_suites]),
                'timestamp': datetime.now().isoformat()
            },
            'performance_by_suite': performance_summary,
            'baseline_compliance': baseline_compliance,
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'operation': r.operation,
                    'duration': r.duration,
                    'success': r.success,
                    'error_message': r.error_message,
                    'metadata': r.metadata
                }
                for r in all_results
            ],
            'environment_info': {
                'project_realm': self.project_realm,
                'database_host': self.db_config.get('host', 'unknown'),
                'embedding_model': os.getenv('EMBEDDING_MODEL', 'default')
            }
        }
        
        return report
    
    def run_full_benchmark_suite(self) -> bool:
        """Run complete benchmark suite"""
        logger.info("🚀 Starting Comprehensive Semantic Search Benchmarks")
        logger.info("=" * 70)
        
        try:
            # Initialize services
            if not self.initialize_services():
                return False
            
            # Create benchmark data
            self.create_benchmark_data(50)  # Create 50 test chunks
            
            # Run benchmark suites
            self.benchmark_single_search_performance()
            self.benchmark_batch_embedding_performance()
            self.benchmark_concurrent_search_performance()
            self.benchmark_cache_performance()
            self.benchmark_realm_priority_performance()
            self.benchmark_memory_usage()
            
            # Generate comprehensive report
            report = self.generate_performance_report()
            
            # Print summary
            logger.info("\n" + "=" * 70)
            logger.info("📊 BENCHMARK RESULTS SUMMARY")
            logger.info("=" * 70)
            
            summary = report['benchmark_summary']
            logger.info(f"Total Tests: {summary['total_tests']}")
            logger.info(f"Successful Tests: {summary['successful_tests']}")
            logger.info(f"Overall Success Rate: {summary['overall_success_rate']:.1f}%")
            logger.info(f"Total Benchmark Time: {summary['total_benchmark_time']:.1f}s")
            
            logger.info("\n📈 Performance by Suite:")
            for suite_name, perf in report['performance_by_suite'].items():
                logger.info(f"  {suite_name}: {perf['successful_tests']}/{perf['total_tests']} tests, "
                           f"avg: {perf['average_duration']:.3f}s")
            
            logger.info("\n🎯 Baseline Compliance:")
            for metric_name, compliance in report['baseline_compliance'].items():
                status = "✓" if compliance['meets_baseline'] else "⚠"
                logger.info(f"  {status} {metric_name}: {compliance['average_time']:.3f}s "
                           f"(baseline: {compliance['baseline']:.3f}s)")
            
            return summary['overall_success_rate'] >= 85.0  # 85% success threshold
            
        except Exception as e:
            logger.error(f"\n💥 BENCHMARK ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main entry point for benchmark script"""
    parser = argparse.ArgumentParser(description='Benchmark realm-aware semantic search performance')
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', default='3306', help='Database port')
    parser.add_argument('--database', default='megamind_database', help='Database name')
    parser.add_argument('--user', default='megamind_user', help='Database user')
    parser.add_argument('--password', required=True, help='Database password')
    parser.add_argument('--report-file', help='Save benchmark report to file')
    parser.add_argument('--test-chunks', type=int, default=50, help='Number of test chunks to create')
    
    args = parser.parse_args()
    
    # Database configuration
    db_config = {
        'host': args.host,
        'port': args.port,
        'database': args.database,
        'user': args.user,
        'password': args.password
    }
    
    # Run benchmarks
    benchmark = RealmSemanticSearchBenchmark(db_config)
    success = benchmark.run_full_benchmark_suite()
    
    # Save report if requested
    if args.report_file:
        try:
            report = benchmark.generate_performance_report()
            with open(args.report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Benchmark report saved to: {args.report_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())