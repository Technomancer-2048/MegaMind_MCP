#!/usr/bin/env python3
"""
Phase 4 Performance Optimization: Async Embedding Processor
Background embedding generation for existing chunks with realm awareness
"""

import asyncio
import logging
import time
import json
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from enum import Enum

# Try importing database components
try:
    from ..realm_aware_database import RealmAwareMegaMindDatabase
    from .embedding_service import get_embedding_service
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logging.warning("Database components not available for async processor")

logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    """Status of async processing jobs"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class EmbeddingJob:
    """Represents an embedding generation job"""
    job_id: str
    chunk_id: str
    content: str
    realm_id: str
    priority: int = 1
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ProcessingStats:
    """Statistics for async processing"""
    total_jobs: int = 0
    pending_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    cancelled_jobs: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    embeddings_per_second: float = 0.0
    realm_distribution: Dict[str, int] = None
    
    def __post_init__(self):
        if self.realm_distribution is None:
            self.realm_distribution = {}

class AsyncEmbeddingProcessor:
    """
    Background embedding generation service with realm-aware processing.
    Manages queued embedding generation for chunks without embeddings.
    """
    
    def __init__(self, db_config: Dict[str, str], max_workers: int = 3, batch_size: int = 10):
        """
        Initialize async embedding processor.
        
        Args:
            db_config: Database configuration
            max_workers: Maximum number of worker threads
            batch_size: Number of chunks to process in each batch
        """
        self.db_config = db_config
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Job management
        self._job_queue: List[EmbeddingJob] = []
        self._completed_jobs: Dict[str, EmbeddingJob] = {}
        self._job_lock = threading.Lock()
        self._job_counter = 0
        
        # Processing state
        self._is_running = False
        self._should_stop = False
        self._executor: Optional[ThreadPoolExecutor] = None
        self._processing_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._stats = ProcessingStats()
        self._stats_lock = threading.Lock()
        
        # Services
        self.embedding_service = get_embedding_service() if DATABASE_AVAILABLE else None
        self.database: Optional[RealmAwareMegaMindDatabase] = None
        
        # Callbacks
        self.progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self.completion_callback: Optional[Callable[[str, bool, str], None]] = None
        
        logger.info(f"Initialized AsyncEmbeddingProcessor: max_workers={max_workers}, batch_size={batch_size}")
    
    def _initialize_database(self) -> bool:
        """Initialize database connection"""
        if not DATABASE_AVAILABLE:
            return False
        
        try:
            self.database = RealmAwareMegaMindDatabase(self.db_config)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    def start(self) -> bool:
        """Start the async processing service"""
        if self._is_running:
            logger.warning("Async processor already running")
            return True
        
        if not self._initialize_database():
            logger.error("Cannot start processor: database initialization failed")
            return False
        
        if not self.embedding_service or not self.embedding_service.is_available():
            logger.error("Cannot start processor: embedding service not available")
            return False
        
        self._should_stop = False
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()
        self._is_running = True
        
        logger.info("Async embedding processor started")
        return True
    
    def stop(self, timeout: float = 30.0) -> bool:
        """Stop the async processing service"""
        if not self._is_running:
            return True
        
        logger.info("Stopping async embedding processor...")
        self._should_stop = True
        
        # Wait for processing thread to finish
        if self._processing_thread:
            self._processing_thread.join(timeout=timeout)
            if self._processing_thread.is_alive():
                logger.warning("Processing thread did not stop cleanly")
        
        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True, timeout=timeout)
        
        self._is_running = False
        logger.info("Async embedding processor stopped")
        return True
    
    def add_embedding_job(self, chunk_id: str, content: str, realm_id: str, priority: int = 1) -> str:
        """
        Add a new embedding generation job to the queue.
        
        Args:
            chunk_id: ID of the chunk to process
            content: Text content to embed
            realm_id: Realm ID for context
            priority: Job priority (higher = more important)
            
        Returns:
            Job ID for tracking
        """
        with self._job_lock:
            self._job_counter += 1
            job_id = f"embed_job_{self._job_counter}_{int(time.time())}"
            
            job = EmbeddingJob(
                job_id=job_id,
                chunk_id=chunk_id,
                content=content,
                realm_id=realm_id,
                priority=priority
            )
            
            # Insert in priority order
            inserted = False
            for i, existing_job in enumerate(self._job_queue):
                if job.priority > existing_job.priority:
                    self._job_queue.insert(i, job)
                    inserted = True
                    break
            
            if not inserted:
                self._job_queue.append(job)
            
            # Update stats
            with self._stats_lock:
                self._stats.total_jobs += 1
                self._stats.pending_jobs += 1
                realm = job.realm_id
                self._stats.realm_distribution[realm] = self._stats.realm_distribution.get(realm, 0) + 1
        
        logger.debug(f"Added embedding job {job_id} for chunk {chunk_id} in realm {realm_id}")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job"""
        with self._job_lock:
            # Check active queue
            for job in self._job_queue:
                if job.job_id == job_id:
                    return {
                        'job_id': job.job_id,
                        'chunk_id': job.chunk_id,
                        'realm_id': job.realm_id,
                        'status': job.status.value,
                        'created_at': job.created_at.isoformat(),
                        'started_at': job.started_at.isoformat() if job.started_at else None,
                        'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                        'retry_count': job.retry_count,
                        'error_message': job.error_message
                    }
            
            # Check completed jobs
            if job_id in self._completed_jobs:
                job = self._completed_jobs[job_id]
                return {
                    'job_id': job.job_id,
                    'chunk_id': job.chunk_id,
                    'realm_id': job.realm_id,
                    'status': job.status.value,
                    'created_at': job.created_at.isoformat(),
                    'started_at': job.started_at.isoformat() if job.started_at else None,
                    'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                    'retry_count': job.retry_count,
                    'error_message': job.error_message
                }
        
        return None
    
    def process_missing_embeddings(self, realm_id: Optional[str] = None, limit: int = 100) -> str:
        """
        Find and queue chunks without embeddings for processing.
        
        Args:
            realm_id: Optional realm to focus on (None for all realms)
            limit: Maximum number of chunks to queue
            
        Returns:
            Batch job ID for tracking
        """
        if not self.database:
            raise RuntimeError("Database not initialized")
        
        try:
            # Query chunks without embeddings
            query_conditions = ["embedding IS NULL"]
            query_params = []
            
            if realm_id:
                query_conditions.append("realm_id = %s")
                query_params.append(realm_id)
            
            # Add order by for consistent processing
            query_conditions.append("ORDER BY access_count DESC, created_at ASC")
            query_conditions.append("LIMIT %s")
            query_params.append(limit)
            
            connection = self.database.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            query = f"""
            SELECT chunk_id, content, realm_id, access_count, token_count
            FROM megamind_chunks 
            WHERE {' AND '.join(query_conditions[:-2])}
            {' '.join(query_conditions[-2:])}
            """
            
            cursor.execute(query, query_params)
            chunks = cursor.fetchall()
            
            # Create batch job ID
            batch_id = f"batch_{int(time.time())}_{len(chunks)}"
            
            # Queue embedding jobs
            for chunk in chunks:
                # Prioritize based on access count and realm
                priority = chunk['access_count'] or 1
                if chunk['realm_id'] and chunk['realm_id'].startswith('PROJ_'):
                    priority += 10  # Boost project realm priority
                
                self.add_embedding_job(
                    chunk_id=chunk['chunk_id'],
                    content=chunk['content'],
                    realm_id=chunk['realm_id'],
                    priority=priority
                )
            
            logger.info(f"Queued {len(chunks)} chunks for embedding generation (batch: {batch_id})")
            return batch_id
            
        except Exception as e:
            logger.error(f"Failed to process missing embeddings: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def update_embeddings_batch(self, chunk_ids: List[str]) -> str:
        """
        Queue specific chunks for embedding updates.
        
        Args:
            chunk_ids: List of chunk IDs to update
            
        Returns:
            Batch job ID for tracking
        """
        if not self.database:
            raise RuntimeError("Database not initialized")
        
        try:
            connection = self.database.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get chunk details
            placeholders = ','.join(['%s'] * len(chunk_ids))
            query = f"""
            SELECT chunk_id, content, realm_id, access_count
            FROM megamind_chunks 
            WHERE chunk_id IN ({placeholders})
            """
            
            cursor.execute(query, chunk_ids)
            chunks = cursor.fetchall()
            
            batch_id = f"update_batch_{int(time.time())}_{len(chunks)}"
            
            # Queue embedding jobs with high priority
            for chunk in chunks:
                priority = 50 + (chunk['access_count'] or 1)  # High priority for updates
                self.add_embedding_job(
                    chunk_id=chunk['chunk_id'],
                    content=chunk['content'],
                    realm_id=chunk['realm_id'],
                    priority=priority
                )
            
            logger.info(f"Queued {len(chunks)} chunks for embedding updates (batch: {batch_id})")
            return batch_id
            
        except Exception as e:
            logger.error(f"Failed to queue embedding updates: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def _processing_loop(self):
        """Main processing loop running in background thread"""
        logger.info("Started async embedding processing loop")
        
        while not self._should_stop:
            try:
                # Get next batch of jobs
                jobs_to_process = []
                with self._job_lock:
                    for _ in range(min(self.batch_size, len(self._job_queue))):
                        if self._job_queue:
                            job = self._job_queue.pop(0)
                            job.status = ProcessingStatus.RUNNING
                            job.started_at = datetime.now()
                            jobs_to_process.append(job)
                
                if not jobs_to_process:
                    # No jobs, sleep and continue
                    time.sleep(1.0)
                    continue
                
                # Process jobs in parallel
                futures = {}
                for job in jobs_to_process:
                    future = self._executor.submit(self._process_single_job, job)
                    futures[future] = job
                
                # Wait for completion
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        success = future.result()
                        self._handle_job_completion(job, success)
                    except Exception as e:
                        logger.error(f"Job {job.job_id} failed with exception: {e}")
                        job.error_message = str(e)
                        self._handle_job_completion(job, False)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(5.0)  # Back off on errors
        
        logger.info("Async embedding processing loop stopped")
    
    def _process_single_job(self, job: EmbeddingJob) -> bool:
        """Process a single embedding job"""
        try:
            start_time = time.time()
            
            # Generate embedding
            embedding = self.embedding_service.generate_embedding(
                text=job.content, 
                realm_context=job.realm_id
            )
            
            if embedding is None:
                raise RuntimeError("Failed to generate embedding")
            
            # Update database
            if not self._update_chunk_embedding(job.chunk_id, embedding):
                raise RuntimeError("Failed to update chunk embedding in database")
            
            processing_time = time.time() - start_time
            
            # Update statistics
            with self._stats_lock:
                self._stats.total_processing_time += processing_time
                self._stats.completed_jobs += 1
                
                if self._stats.completed_jobs > 0:
                    self._stats.average_processing_time = (
                        self._stats.total_processing_time / self._stats.completed_jobs
                    )
                    self._stats.embeddings_per_second = 1.0 / self._stats.average_processing_time
            
            # Trigger progress callback
            if self.progress_callback:
                self.progress_callback(job.job_id, {
                    'chunk_id': job.chunk_id,
                    'realm_id': job.realm_id,
                    'processing_time': processing_time,
                    'embedding_dimension': len(embedding)
                })
            
            logger.debug(f"Successfully processed job {job.job_id} in {processing_time:.3f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process job {job.job_id}: {e}")
            job.error_message = str(e)
            return False
    
    def _update_chunk_embedding(self, chunk_id: str, embedding: List[float]) -> bool:
        """Update chunk embedding in database"""
        if not self.database:
            return False
        
        try:
            connection = self.database.get_connection()
            cursor = connection.cursor()
            
            embedding_json = json.dumps(embedding)
            query = """
            UPDATE megamind_chunks 
            SET embedding = %s, last_accessed = CURRENT_TIMESTAMP 
            WHERE chunk_id = %s
            """
            
            cursor.execute(query, (embedding_json, chunk_id))
            connection.commit()
            
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"Failed to update embedding for chunk {chunk_id}: {e}")
            return False
        finally:
            if connection:
                connection.close()
    
    def _handle_job_completion(self, job: EmbeddingJob, success: bool):
        """Handle job completion (success or failure)"""
        job.completed_at = datetime.now()
        
        if success:
            job.status = ProcessingStatus.COMPLETED
            with self._stats_lock:
                self._stats.pending_jobs = max(0, self._stats.pending_jobs - 1)
                self._stats.running_jobs = max(0, self._stats.running_jobs - 1)
        else:
            job.retry_count += 1
            
            if job.retry_count <= job.max_retries:
                # Retry job
                job.status = ProcessingStatus.PENDING
                job.started_at = None
                job.completed_at = None
                
                with self._job_lock:
                    # Add back to queue with lower priority
                    job.priority = max(1, job.priority - 1)
                    self._job_queue.append(job)
                    
                logger.info(f"Retrying job {job.job_id} (attempt {job.retry_count + 1})")
                return
            else:
                # Max retries exceeded
                job.status = ProcessingStatus.FAILED
                with self._stats_lock:
                    self._stats.pending_jobs = max(0, self._stats.pending_jobs - 1)
                    self._stats.running_jobs = max(0, self._stats.running_jobs - 1)
                    self._stats.failed_jobs += 1
        
        # Move to completed jobs
        with self._job_lock:
            self._completed_jobs[job.job_id] = job
            
        # Trigger completion callback
        if self.completion_callback:
            self.completion_callback(job.job_id, success, job.error_message or "")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        with self._stats_lock:
            stats_dict = {
                'total_jobs': self._stats.total_jobs,
                'pending_jobs': self._stats.pending_jobs,
                'running_jobs': self._stats.running_jobs,
                'completed_jobs': self._stats.completed_jobs,
                'failed_jobs': self._stats.failed_jobs,
                'cancelled_jobs': self._stats.cancelled_jobs,
                'total_processing_time': round(self._stats.total_processing_time, 3),
                'average_processing_time': round(self._stats.average_processing_time, 3),
                'embeddings_per_second': round(self._stats.embeddings_per_second, 2),
                'realm_distribution': dict(self._stats.realm_distribution),
                'is_running': self._is_running,
                'max_workers': self.max_workers,
                'batch_size': self.batch_size
            }
            
        with self._job_lock:
            stats_dict['queue_size'] = len(self._job_queue)
            stats_dict['completed_jobs_stored'] = len(self._completed_jobs)
            
        return stats_dict
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job"""
        with self._job_lock:
            for i, job in enumerate(self._job_queue):
                if job.job_id == job_id and job.status == ProcessingStatus.PENDING:
                    job.status = ProcessingStatus.CANCELLED
                    job.completed_at = datetime.now()
                    
                    # Remove from queue and add to completed
                    self._job_queue.pop(i)
                    self._completed_jobs[job_id] = job
                    
                    # Update stats
                    with self._stats_lock:
                        self._stats.pending_jobs = max(0, self._stats.pending_jobs - 1)
                        self._stats.cancelled_jobs += 1
                    
                    logger.info(f"Cancelled job {job_id}")
                    return True
        
        return False
    
    def clear_completed_jobs(self, older_than_hours: int = 24):
        """Clear completed jobs older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        with self._job_lock:
            jobs_to_remove = []
            for job_id, job in self._completed_jobs.items():
                if job.completed_at and job.completed_at < cutoff_time:
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self._completed_jobs[job_id]
            
            if jobs_to_remove:
                logger.info(f"Cleared {len(jobs_to_remove)} completed jobs older than {older_than_hours} hours")
    
    def set_progress_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Set callback for progress updates"""
        self.progress_callback = callback
    
    def set_completion_callback(self, callback: Callable[[str, bool, str], None]):
        """Set callback for job completion"""
        self.completion_callback = callback


# Global processor instance
_async_processor_instance: Optional[AsyncEmbeddingProcessor] = None

def get_async_embedding_processor(db_config: Dict[str, str], 
                                 max_workers: int = 3, 
                                 batch_size: int = 10) -> AsyncEmbeddingProcessor:
    """Get singleton async embedding processor instance"""
    global _async_processor_instance
    
    if _async_processor_instance is None:
        _async_processor_instance = AsyncEmbeddingProcessor(
            db_config=db_config,
            max_workers=max_workers,
            batch_size=batch_size
        )
    
    return _async_processor_instance