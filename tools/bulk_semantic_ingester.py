#!/usr/bin/env python3
"""
MegaMind Context Database - Bulk Semantic Ingester
Phase 3: Ingestion Integration

High-performance bulk ingestion with optimized embedding generation and realm-aware processing.
Designed for large-scale document ingestion with batch processing and performance optimization.
"""

import os
import sys
import json
import logging
import time
import asyncio
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mcp_server'))

from realm_aware_markdown_ingester import RealmAwareMarkdownIngester, RealmAwareChunkMetadata
from services.embedding_service import get_embedding_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BulkIngestionConfig:
    """Configuration for bulk semantic ingestion"""
    batch_size: int = 50
    max_workers: int = 4
    embedding_batch_size: int = 25
    max_file_size_mb: int = 10
    chunk_size_limit: int = 8000  # characters
    enable_parallel_processing: bool = True
    enable_embedding_cache: bool = True
    auto_commit_batches: bool = False
    realm_distribution_strategy: str = 'auto'  # 'auto', 'project_only', 'global_only'
    performance_monitoring: bool = True

@dataclass
class BulkIngestionStats:
    """Comprehensive statistics for bulk ingestion"""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    successful_chunks: int = 0
    failed_chunks: int = 0
    embeddings_generated: int = 0
    embedding_failures: int = 0
    realm_distribution: Dict[str, int] = field(default_factory=dict)
    processing_times: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    error_summary: List[str] = field(default_factory=list)
    
    def get_duration(self) -> float:
        """Get total processing duration in seconds"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def get_throughput(self) -> Dict[str, float]:
        """Calculate throughput metrics"""
        duration = self.get_duration()
        if duration == 0:
            return {}
        
        return {
            'files_per_second': self.processed_files / duration,
            'chunks_per_second': self.successful_chunks / duration,
            'embeddings_per_second': self.embeddings_generated / duration,
            'chars_per_second': sum(self.processing_times.values()) / duration if self.processing_times else 0
        }

class BulkSemanticIngester:
    """High-performance bulk ingestion with optimized embedding generation and realm support"""
    
    def __init__(self, db_config: Dict[str, str], config: BulkIngestionConfig = None):
        """
        Initialize bulk semantic ingester
        
        Args:
            db_config: Database configuration
            config: Bulk ingestion configuration
        """
        self.db_config = db_config
        self.config = config or BulkIngestionConfig()
        self.stats = BulkIngestionStats()
        
        # Initialize base ingester
        self.base_ingester = RealmAwareMarkdownIngester(
            db_config=db_config,
            session_id=f"bulk_ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Initialize embedding service for batch processing
        self.embedding_service = get_embedding_service()
        
        # Cache for processed content hashes (deduplication)
        self.processed_hashes: Set[str] = set()
        
        # Embedding cache for performance
        self.embedding_cache: Dict[str, List[float]] = {}
        
        logger.info(f"Initialized BulkSemanticIngester:")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Max workers: {self.config.max_workers}")
        logger.info(f"  Embedding batch size: {self.config.embedding_batch_size}")
        logger.info(f"  Session ID: {self.base_ingester.session_id}")
        logger.info(f"  Embedding service available: {self.embedding_service.is_available()}")
    
    def ingest_directory(self, directory_path: str, pattern: str = "*.md", 
                        recursive: bool = True, file_filters: List[str] = None) -> Dict[str, any]:
        """
        Bulk ingest all files in a directory with optimized processing
        
        Args:
            directory_path: Path to directory containing files
            pattern: File pattern to match
            recursive: Search subdirectories recursively
            file_filters: List of filename patterns to exclude
            
        Returns:
            Comprehensive ingestion results
        """
        self.stats.start_time = datetime.now()
        
        try:
            directory = Path(directory_path)
            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            # Discover files
            files = self._discover_files(directory, pattern, recursive, file_filters)
            self.stats.total_files = len(files)
            
            if not files:
                logger.warning(f"No files found matching criteria")
                return {'success': False, 'error': 'No files found'}
            
            logger.info(f"Discovered {len(files)} files for bulk ingestion")
            
            # Process files in optimized batches
            if self.config.enable_parallel_processing and len(files) > self.config.max_workers:
                results = self._process_files_parallel(files)
            else:
                results = self._process_files_sequential(files)
            
            # Finalize statistics
            self.stats.end_time = datetime.now()
            
            # Optionally commit all changes
            commit_result = None
            if self.config.auto_commit_batches:
                commit_result = self._commit_all_changes()
            
            return self._create_bulk_results(directory_path, results, commit_result)
            
        except Exception as e:
            error_msg = f"Bulk ingestion failed: {str(e)}"
            logger.error(error_msg)
            self.stats.error_summary.append(error_msg)
            return {'success': False, 'error': error_msg}
    
    def ingest_file_list(self, file_paths: List[str]) -> Dict[str, any]:
        """
        Ingest a specific list of files
        
        Args:
            file_paths: List of file paths to ingest
            
        Returns:
            Bulk ingestion results
        """
        self.stats.start_time = datetime.now()
        self.stats.total_files = len(file_paths)
        
        try:
            # Filter and validate files
            valid_files = []
            for file_path in file_paths:
                path = Path(file_path)
                if path.exists() and path.is_file():
                    if self._validate_file(path):
                        valid_files.append(path)
                    else:
                        logger.warning(f"Skipping invalid file: {file_path}")
                        self.stats.failed_files += 1
                else:
                    logger.warning(f"File not found: {file_path}")
                    self.stats.failed_files += 1
            
            if not valid_files:
                return {'success': False, 'error': 'No valid files to process'}
            
            # Process files
            if self.config.enable_parallel_processing and len(valid_files) > self.config.max_workers:
                results = self._process_files_parallel(valid_files)
            else:
                results = self._process_files_sequential(valid_files)
            
            self.stats.end_time = datetime.now()
            
            # Optionally commit changes
            commit_result = None
            if self.config.auto_commit_batches:
                commit_result = self._commit_all_changes()
            
            return self._create_bulk_results("file_list", results, commit_result)
            
        except Exception as e:
            error_msg = f"File list ingestion failed: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _discover_files(self, directory: Path, pattern: str, recursive: bool, 
                       file_filters: List[str] = None) -> List[Path]:
        """Discover files matching criteria with filtering"""
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
        
        # Apply filters
        if file_filters:
            filtered_files = []
            for file_path in files:
                if not any(filter_pattern in str(file_path) for filter_pattern in file_filters):
                    filtered_files.append(file_path)
            files = filtered_files
        
        # Validate and filter files
        valid_files = []
        for file_path in files:
            if self._validate_file(file_path):
                valid_files.append(file_path)
            else:
                logger.debug(f"Filtered out invalid file: {file_path}")
        
        return valid_files
    
    def _validate_file(self, file_path: Path) -> bool:
        """Validate file for ingestion"""
        try:
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                logger.warning(f"File too large ({file_size_mb:.1f}MB): {file_path}")
                return False
            
            # Check if file is readable
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first few lines to validate encoding
                f.readline()
            
            return True
            
        except Exception as e:
            logger.debug(f"File validation failed for {file_path}: {e}")
            return False
    
    def _process_files_sequential(self, files: List[Path]) -> List[Dict[str, any]]:
        """Process files sequentially with batch optimization"""
        results = []
        batch = []
        
        for file_path in files:
            batch.append(file_path)
            
            if len(batch) >= self.config.batch_size:
                batch_results = self._process_file_batch(batch)
                results.extend(batch_results)
                batch = []
        
        # Process remaining files
        if batch:
            batch_results = self._process_file_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _process_files_parallel(self, files: List[Path]) -> List[Dict[str, any]]:
        """Process files in parallel using ThreadPoolExecutor"""
        results = []
        
        # Split files into batches
        batches = [files[i:i + self.config.batch_size] 
                  for i in range(0, len(files), self.config.batch_size)]
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_file_batch, batch): batch 
                for batch in batches
            }
            
            for future in future_to_batch:
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    batch = future_to_batch[future]
                    error_msg = f"Batch processing failed for {len(batch)} files: {e}"
                    logger.error(error_msg)
                    self.stats.error_summary.append(error_msg)
                    self.stats.failed_files += len(batch)
        
        return results
    
    def _process_file_batch(self, files: List[Path]) -> List[Dict[str, any]]:
        """Process a batch of files with optimized embedding generation"""
        batch_start = time.time()
        batch_results = []
        all_chunks = []
        
        # Parse all files in batch
        for file_path in files:
            try:
                result = self.base_ingester.ingest_file(str(file_path))
                batch_results.append(result)
                
                if result['success']:
                    self.stats.processed_files += 1
                else:
                    self.stats.failed_files += 1
                    self.stats.error_summary.append(f"File processing failed: {file_path}")
                
            except Exception as e:
                error_msg = f"Failed to process {file_path}: {e}"
                logger.error(error_msg)
                self.stats.failed_files += 1
                self.stats.error_summary.append(error_msg)
                batch_results.append({'success': False, 'error': error_msg, 'file_path': str(file_path)})
        
        # Calculate batch processing time
        batch_time = time.time() - batch_start
        self.stats.processing_times[f"batch_{len(self.stats.processing_times)}"] = batch_time
        
        logger.info(f"Processed batch of {len(files)} files in {batch_time:.2f}s")
        
        return batch_results
    
    def _generate_embeddings_batch(self, chunks: List[RealmAwareChunkMetadata]) -> Dict[str, List[float]]:
        """Generate embeddings for a batch of chunks with caching"""
        if not self.embedding_service.is_available():
            logger.warning("Embedding service not available for batch processing")
            return {}
        
        embeddings = {}
        texts_to_process = []
        chunk_mappings = []
        
        # Prepare texts for batch processing
        for chunk in chunks:
            cache_key = f"{chunk.realm_id}:{chunk.content_hash}" if chunk.content_hash else None
            
            # Check cache first
            if self.config.enable_embedding_cache and cache_key and cache_key in self.embedding_cache:
                embeddings[chunk.chunk_id] = self.embedding_cache[cache_key]
                continue
            
            texts_to_process.append(chunk.content)
            chunk_mappings.append((chunk.chunk_id, chunk.realm_id, cache_key))
        
        # Generate embeddings in batch
        if texts_to_process:
            try:
                start_time = time.time()
                batch_embeddings = self.embedding_service.generate_embeddings_batch(texts_to_process)
                generation_time = time.time() - start_time
                
                logger.debug(f"Generated {len(batch_embeddings)} embeddings in {generation_time:.3f}s")
                
                # Map embeddings back to chunks and update cache
                for i, (chunk_id, realm_id, cache_key) in enumerate(chunk_mappings):
                    if i < len(batch_embeddings) and batch_embeddings[i] is not None:
                        embedding = batch_embeddings[i]
                        embeddings[chunk_id] = embedding
                        self.stats.embeddings_generated += 1
                        
                        # Update cache
                        if self.config.enable_embedding_cache and cache_key:
                            self.embedding_cache[cache_key] = embedding
                    else:
                        self.stats.embedding_failures += 1
                        logger.warning(f"Failed to generate embedding for chunk {chunk_id}")
                
            except Exception as e:
                logger.error(f"Batch embedding generation failed: {e}")
                self.stats.embedding_failures += len(texts_to_process)
        
        return embeddings
    
    def _commit_all_changes(self) -> Dict[str, any]:
        """Commit all pending changes for the session"""
        try:
            logger.info("Committing all session changes...")
            return self.base_ingester.commit_session_changes()
        except Exception as e:
            error_msg = f"Failed to commit session changes: {e}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _create_bulk_results(self, source_path: str, file_results: List[Dict[str, any]], 
                           commit_result: Dict[str, any] = None) -> Dict[str, any]:
        """Create comprehensive bulk ingestion results"""
        # Calculate final statistics
        for result in file_results:
            if result.get('success'):
                chunks_processed = result.get('chunks_processed', 0)
                self.stats.total_chunks += chunks_processed
                self.stats.successful_chunks += chunks_processed
        
        throughput = self.stats.get_throughput()
        
        return {
            'success': True,
            'source_path': source_path,
            'session_id': self.base_ingester.session_id,
            'processing_time': self.stats.get_duration(),
            'statistics': {
                'files': {
                    'total': self.stats.total_files,
                    'processed': self.stats.processed_files,
                    'failed': self.stats.failed_files,
                    'success_rate': (self.stats.processed_files / max(self.stats.total_files, 1)) * 100
                },
                'chunks': {
                    'total': self.stats.total_chunks,
                    'successful': self.stats.successful_chunks,
                    'failed': self.stats.failed_chunks,
                    'success_rate': (self.stats.successful_chunks / max(self.stats.total_chunks, 1)) * 100
                },
                'embeddings': {
                    'generated': self.stats.embeddings_generated,
                    'failed': self.stats.embedding_failures,
                    'coverage': (self.stats.embeddings_generated / max(self.stats.total_chunks, 1)) * 100
                },
                'performance': {
                    'throughput': throughput,
                    'batch_processing_times': self.stats.processing_times,
                    'memory_usage': self.stats.memory_usage
                },
                'realm_distribution': self.stats.realm_distribution,
                'errors': {
                    'count': len(self.stats.error_summary),
                    'summary': self.stats.error_summary[:10]  # First 10 errors
                }
            },
            'commit_result': commit_result,
            'file_results': file_results,
            'config': {
                'batch_size': self.config.batch_size,
                'max_workers': self.config.max_workers,
                'embedding_batch_size': self.config.embedding_batch_size,
                'parallel_processing': self.config.enable_parallel_processing,
                'embedding_cache': self.config.enable_embedding_cache
            }
        }
    
    def get_session_summary(self) -> Dict[str, any]:
        """Get summary of current ingestion session"""
        return {
            'session_id': self.base_ingester.session_id,
            'statistics': self.base_ingester.get_ingestion_statistics(),
            'bulk_stats': {
                'duration': self.stats.get_duration(),
                'throughput': self.stats.get_throughput(),
                'realm_distribution': self.stats.realm_distribution,
                'error_count': len(self.stats.error_summary)
            },
            'pending_changes_count': len(self.base_ingester.get_session_changes()),
            'embedding_cache_size': len(self.embedding_cache) if hasattr(self, 'embedding_cache') else 0
        }


def load_db_config() -> Dict[str, str]:
    """Load database configuration from environment variables"""
    return {
        'host': os.getenv('MEGAMIND_DB_HOST', 'localhost'),
        'port': os.getenv('MEGAMIND_DB_PORT', '3306'),
        'database': os.getenv('MEGAMIND_DB_NAME', 'megamind_context'),
        'user': os.getenv('MEGAMIND_DB_USER', 'megamind_user'),
        'password': os.getenv('MEGAMIND_DB_PASSWORD', '')
    }


def main():
    """Command-line interface for bulk semantic ingestion"""
    parser = argparse.ArgumentParser(description='Bulk Semantic Ingester for MegaMind Context Database')
    parser.add_argument('path', help='Path to directory or file list for bulk ingestion')
    parser.add_argument('--batch-size', type=int, default=50, help='Files per batch (default: 50)')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum parallel workers (default: 4)')
    parser.add_argument('--embedding-batch-size', type=int, default=25, help='Embeddings per batch (default: 25)')
    parser.add_argument('--max-file-size', type=int, default=10, help='Maximum file size in MB (default: 10)')
    parser.add_argument('--pattern', default='*.md', help='File pattern (default: *.md)')
    parser.add_argument('--recursive', '-r', action='store_true', help='Search recursively')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    parser.add_argument('--no-cache', action='store_true', help='Disable embedding cache')
    parser.add_argument('--auto-commit', action='store_true', help='Automatically commit changes')
    parser.add_argument('--exclude', nargs='*', help='File patterns to exclude')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--performance', action='store_true', help='Enable performance monitoring')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    db_config = load_db_config()
    if not db_config['password']:
        logger.error("Database password not configured. Set MEGAMIND_DB_PASSWORD environment variable.")
        return 1
    
    # Create bulk ingestion configuration
    config = BulkIngestionConfig(
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        embedding_batch_size=args.embedding_batch_size,
        max_file_size_mb=args.max_file_size,
        enable_parallel_processing=args.parallel,
        enable_embedding_cache=not args.no_cache,
        auto_commit_batches=args.auto_commit,
        performance_monitoring=args.performance
    )
    
    try:
        # Initialize bulk ingester
        ingester = BulkSemanticIngester(db_config=db_config, config=config)
        
        # Process input
        input_path = Path(args.path)
        if input_path.is_dir():
            result = ingester.ingest_directory(
                str(input_path),
                pattern=args.pattern,
                recursive=args.recursive,
                file_filters=args.exclude
            )
        elif input_path.is_file() and str(input_path).endswith('.txt'):
            # File list mode
            with open(input_path, 'r') as f:
                file_paths = [line.strip() for line in f if line.strip()]
            result = ingester.ingest_file_list(file_paths)
        else:
            logger.error(f"Invalid input: {args.path} (expected directory or .txt file list)")
            return 1
        
        # Display results
        if result['success']:
            stats = result['statistics']
            print(f"\n✅ Bulk ingestion completed successfully!")
            print(f"   Source: {result['source_path']}")
            print(f"   Session ID: {result['session_id']}")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            
            print(f"\n📁 Files:")
            print(f"   Processed: {stats['files']['processed']}/{stats['files']['total']}")
            print(f"   Success rate: {stats['files']['success_rate']:.1f}%")
            
            print(f"\n📄 Chunks:")
            print(f"   Created: {stats['chunks']['successful']}")
            print(f"   Success rate: {stats['chunks']['success_rate']:.1f}%")
            
            print(f"\n🔍 Embeddings:")
            print(f"   Generated: {stats['embeddings']['generated']}")
            print(f"   Coverage: {stats['embeddings']['coverage']:.1f}%")
            
            print(f"\n⚡ Performance:")
            throughput = stats['performance']['throughput']
            print(f"   Files/sec: {throughput.get('files_per_second', 0):.2f}")
            print(f"   Chunks/sec: {throughput.get('chunks_per_second', 0):.2f}")
            
            if stats['errors']['count'] > 0:
                print(f"\n⚠️  Errors: {stats['errors']['count']}")
                for error in stats['errors']['summary'][:3]:
                    print(f"   - {error}")
            
            if result.get('commit_result'):
                commit = result['commit_result']
                if commit.get('success'):
                    print(f"\n✅ Changes committed successfully")
                else:
                    print(f"\n❌ Commit failed: {commit.get('error')}")
            else:
                print(f"\n💡 Use --auto-commit to automatically commit changes")
                print(f"   Session ID for manual commit: {result['session_id']}")
        else:
            print(f"❌ Bulk ingestion failed: {result['error']}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Bulk ingestion failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())